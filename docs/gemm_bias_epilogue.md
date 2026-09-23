# Design: fold Dense bias (and residual) into the GEMM with cuBLASLt

Status: **implemented and measured** (A10G, 2026-09-23). Results at the end; the
design below is kept as approved.

## Goal

Remove kernel launches. The float16 Whisper decode issues a separate elementwise
kernel after every biased Dense GEMM (`ops::apply_bias_and_activation`,
`src/ops/gemm.cc:11-24`, called at `src/ops/gemm.cc:76`). Success is measured in
`cudaLaunchKernel + cuLaunchKernel`, starting from **545,522** at `9352e203`.
Replacing that kernel with a different kernel does not count.

## Where the 79,118 launches come from (code-verified structure, estimated counts)

For Whisper (pre-norm, so `_layer_norm` is set) every Dense has a bias. Which bias
kernel runs depends on whether a residual is passed:

| Dense call | residual | kernel today | per |
|---|---|---|---|
| self-attn fused QKV `_linear[0]` | no | `add_block_broadcast` (`cuda::plus`) | decoder layer-step, encoder layer |
| cross-attn Q `_linear[0]` | no | `add_block_broadcast` | decoder layer-step |
| cross-attn memory KV `_linear[1]` | no | `add_block_broadcast` | encoder chunk × layer |
| self/cross-attn out `_linear.back()` (`attention.cc:694,882`, `flash_attention.cc:154`) | yes | `plus3` (`trinary_add`) | decoder layer-step ×2, encoder layer ×1 |
| FFN out `_ff2` (`transformer.cc:39`) | yes | `plus3` | decoder layer-step, encoder layer |
| FFN in `_ff1` | no, + GELU | `op_epilogue<plus, gelu>` | **out of scope, already fused** |

Solving `D + E = 15,600` (GELU launches = decoder layer-steps + encoder layer-runs)
with E ≈ 660 (12 layers × 5 chunks × 11 runs) gives, as an **estimate**:

- residual (`plus3`): 3D + 2E ≈ **46,100**
- no residual (`add_block_broadcast`): 2D + E + memory KV ≈ **31,200**
- Dense bias total ≈ **77,300**; the remaining ≈ 1,800 of the 79,118 are most likely
  plain `ops::Add` calls, which share the `cuda::plus` functor and are not biases.
  (Refuted by the trace, see Results: all 79,118 are Dense biases. E was also wrong:
  the workload decodes a single 30 s window, so E = 12 layers × 13 decodes = 156.
  With that, 3D + 2E = 46,644, exactly the `plus3` count.)

Unverified: these counts. The first step on the GPU box is to read the **existing**
`/opt/prof_kvk.nsys-rep` with `cuda_gpu_kern_sum` (no new profiling) and split the
79,118 by template instantiation. If the Dense share is far below ~77,000, stop and
report before writing code.

## Design

### 1. Scope of cuBLASLt: one new call site, coexisting with `cublasGemmEx`

- New `cuda::gemm_bias_lt(...)` (`src/cuda/gemm_lt.cu`) backed by `cublasLtMatmul`, used
  **only** from `Gemm::operator()` when: device is CUDA, dtype is fp16/bf16/fp32,
  `bias != nullptr`, and `_activation_type == nullptr`.
- Everything else keeps the current path untouched: no-bias GEMMs, batched GEMMs
  (attention), bias+activation (the GELU case), int8.
- HIP builds (`CT2_USE_HIP`) keep the current path. No hipBLASLt.
- Compute type and scalars mirror the existing `gemm` specializations
  (`CUBLAS_COMPUTE_16F` when `use_true_fp16_gemm()`, else `32F`; bf16 → `32F`).

### 2. Residual: fold it through `beta * C`, with C ≠ D

cuBLASLt computes `D = alpha·op(A)op(B) + beta·C + bias` and, unlike
`cublasGemmEx`, allows C and D to be different buffers. So:

- no residual → `beta = 0`, epilogue `CUBLASLT_EPILOGUE_BIAS`
- residual → `C = residual`, `beta = 1`, same epilogue

This covers both `plus3` and `add_block_broadcast`. Excluding the residual case would
leave ~46,000 of the ~77,000 in place, so it is in scope.

Bias broadcast: CTranslate2 computes row-major `C[m,n]` as column-major `C^T[n,m]`
(`primitives.cu:556`). The Lt bias vector has length equal to the column-major row
count, i.e. `n` = output features, which is exactly the Dense bias.

### 3. Out of scope, stated explicitly

- int8 / int8_float16: bias is already applied inside the dequantize kernel;
  the residual `ops::Add` there stays.
- AWQ (`awq/gemm.cc`, `awq/gemv.cc`), `dequantize_cpu.cc`, `conv1d_gpu.cu`.
- `EPILOGUE_GELU_BIAS` for `_ff1`. It would remove ~15,600 more launches, but it is a
  different lever and was excluded. Reported, not done.

### 4. Launches removed (target)

One bias kernel per converted call, provided the Lt call launches the same number
of kernels as the `cublasGemmEx` it replaces. Estimate **~77,000**, so
545,522 → **~468,000**.

Risk that could eat this: Lt's heuristic picking split-K with a separate reduction
kernel where `cublasGemmEx` did not. Checked with nsys; if it happens, it is
reported before any tuning.

### 5. Plumbing

- Link `CUDA::cublasLt` (or `${CUDA_cublasLt_LIBRARY}`). Our build uses
  `CUDA_DYNAMIC_LOADING=OFF` (default, `CMakeLists.txt:18`); the dynamic-loading
  stub `cublas_stub.cc` does not cover Lt. Proposal: with dynamic loading ON, keep
  the old path (`#ifdef`) rather than write a second stub.
- Handle: `thread_local` `cublasLtHandle_t` next to `get_cublas_handle()`.
- Workspace: one persistent per-thread buffer (4 MiB) allocated once, not per call —
  a per-call allocation would add back a `cudaMallocAsync/FreeAsync` pair per GEMM.
- Descriptors: created per call first (host-only). Cache only if measurement shows
  host cost; algorithm choice via `cublasLtMatmulAlgoGetHeuristic` relies on
  cuBLASLt's built-in heuristics cache.

## Numerics risk

Today the GEMM result is rounded to fp16, then bias and residual are added in fp16
(two more roundings). The epilogue adds them before the final rounding. Outputs will
differ in the last fp16 bit. The previous session showed small beam near-ties can
flip (flash: 1/40 differences on small/float16). **The 32/32 token match may fail
for a legitimate reason.** If it does, I report which cases and how the text
differs; I will not reorder the arithmetic to force a match without asking.

## Verification

1. Launches: nsys `cudaLaunchKernel + cuLaunchKernel`, before (rebuilt in the same
   session) vs after, plus `cuda_gpu_kern_sum` to show `plus3` /
   `add_block_broadcast` counts dropping and no new kernel appearing.
2. Tokens: `/opt/tok.py` vs `/opt/tok_upstream.json`.
3. `--gtest_filter='CUDA/*'` → 178 passed / 3 skipped / 0 failed, plus new CUDA
   tests for bias, bias+residual, fp16/bf16/fp32, odd `n`, and a NaN-poisoned
   output buffer (so the test fails if D is read instead of C).
4. Speed: `/opt/bench.py` pair comparison, warmup 2 + median of 7.

**Control axis changes.** Flash ON is *not* a control here: `flash_attention.cc:154`
passes the residual too, so both axes take the new path. The valid control is
**int8_float16**: the Dense GEMMs there are int8 and never receive a bias
(`common.cc:396`).

Local (mac) checks before starting the box: build compiles the CPU path unchanged;
the CUDA code cannot be exercised on the mac, so the temporary-`throw` check runs on
the box.

## Results

Same workload as before (`/opt/prof.py`: whisper-small float16, beam 5, first 30 s of the 150 s lecture,
10 decodes, timestamps on). The "before" launch counts are the existing
`9352e203` trace from the previous session, reused as instructed rather than
re-profiled; the same binary state was rebuilt and installed to a separate prefix in
this session as the speed reference.

### 1. Launches

The existing trace (`/opt/prof_kvk.nsys-rep`, read, not re-profiled) splits the
79,118 into two instantiations, both using `repeat_vec_block`, i.e. both are bias
broadcasts through `BiasAdd`:

| kernel | before | after |
|---|---|---|
| `plus3` (bias + residual) | 46,644 | 0 |
| `cuda::plus` + `repeat_vec_block` (bias only) | 32,474 | 0 |
| `cublasLt::epilogue::impl::globalKernel` (new) | 0 | 312 |
| `cublasLt::splitKreduce_kernel` | 61,308 | 61,308 |

| | before (`9352e203`) | after |
|---|---|---|
| `cudaLaunchKernel` | 432,396 | 353,590 |
| `cuLaunchKernel` | 113,126 | 113,126 |
| **launches** | **545,522** | **466,716 (−78,806)** |
| GPU kernel instances | 545,860 | 467,054 |
| host CUDA API total | 4.039 s | 3.568 s |
| GPU kernel time | 2.815 s | 2.695 s |
| `cudaMallocAsync` / `cudaFreeAsync` | 253,223 each | 253,223 each |

All 79,118 bias kernels are gone. For 312 calls (one shape) cuBLASLt ran its
epilogue as a separate kernel, so those 312 were replaced one for one, not removed.
The split-K reduction count is unchanged: where the GEMM already used split-K, the
bias moved into the existing reduction kernel. The split-K risk from the design did
not happen.

The estimate that ~1,800 of the 79,118 were plain `ops::Add` was wrong: `ops::Add`
does not go through `repeat_vec_block`, and every instance of both kernels came
from `BiasAdd`.

### 2. Tokens: 30/32, the two differences are float16 near-ties

Against `tok_upstream.json` and `tok_kv2.json`, identically:

- int8_float16 (16 cases, path not taken): tokens and scores identical.
- float16 (16 cases): tokens identical in 14, scores moved in the 3rd–4th decimal in
  14 cases.
- `large-v3|float16|physics|b5`: one timestamp token `<|15.66|>` → `<|15.64|>`.
- `large-v3|float16|korean|b5`: Korean audio forced to `<|en|>`, output is a
  meaningless romanization either way; its spelling changed.

This is the rounding change predicted in "Numerics risk": bias and residual are now
added before the final fp16 rounding instead of after it. The arithmetic was not
reordered to force a match.

### 3. Tests

`--gtest_filter='CUDA/*'`: 181 passed / 3 skipped / 0 failed (178 before plus the
three `GemmBiasEpilogue` dtypes; the skips are the pre-existing
`Conv1DGroupNoBiasQuantized`).

Is the new path taken: a temporary `throw` in the branch fails exactly the three
`GemmBiasEpilogue` tests (fp32, fp16, bf16). The existing `GemmBias`,
`GemmResidual` and `GemmGELU` tests keep passing because they use beta=1 or an
activation and stay on the old path, as designed. The throw only proves the first
shape of the test; that the Whisper shapes take the path is shown by the trace above.

### 4. Speed (`/opt/bench.py`, warmup 2 + median of 7, same-session reference)

| axis | n | median | range |
|---|---|---|---|
| float16 (changed path) | 32 | **−4.32%** | −9.45% to −2.31% |
| int8_float16 (control) | 32 | +0.69% | −2.89% to +5.43% |
| float16, flash off | 16 | −5.21% | −9.45% to −3.38% |
| float16, flash on | 16 | −4.32% | −9.13% to −2.31% |

Decode lengths identical in all 64 rows. whisper-small gains more (−2.3% to −9.5%)
than large-v3 (−3.4% to −4.8%): the removed launch cost is fixed per call, and
large-v3's kernels are longer. The per-call descriptor creation and heuristic query
did not cause a visible regression, so no descriptor cache was added.

### Not covered

- AWQ: the dequantized path (`common.cc:417-424`) builds an `ops::Gemm` with the
  Dense configuration, so it now takes the new path too. Not measured; AWQ has no
  test model here.
- `CUDA_DYNAMIC_LOADING=ON` builds and HIP keep the old path.
- Left as reported, not done: bias+GELU through `CUBLASLT_EPILOGUE_GELU_BIAS`
  (15,600 more launches).
