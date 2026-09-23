# Design: reorder all beam states with one kernel

Status: **implemented and measured** (A10G, 2026-09-23). Results at the end; the
design below is kept as approved.

## Goal

Remove kernel launches. After every beam search step, `Decoder::update_state`
(`src/layers/decoder.cc:39-55`) calls `ops::Gather` once per replicated state
tensor, and each call is one thrust kernel. Success is measured in
`cudaLaunchKernel + cuLaunchKernel` against the `50e1e882` profile
(`/opt/prof_v.nsys-rep`, 467,054 kernels). Replacing each gather with a different
per-tensor kernel does not count.

## Where the 31,552 gather launches come from (trace-verified)

`cuda_gpu_kern_sum` on `/opt/prof_v.nsys-rep` (whisper-small fp16, beam 5,
timestamps on, 13 decodes of one 30 s window):

| kernel | launches |
|---|---|
| thrust for_each with `ops::batch_gather_index_map` (uint4) | **31,552** |
| `fastertransformer::topk_stage_1` (one per decoding step) | 1,274 |

Split of the 31,552, derived from the code and the step count:

| call site | per | launches |
|---|---|---|
| `decoder.cc:51` beam reorder of `self_keys_i`/`self_values_i` | 24 tensors × (1,274 − 13 last steps) | **30,264** |
| `common.cc:79` token embedding lookup | one per step | 1,274 |
| unattributed | | 14 |

The `memory_*` entries are not replicated (`transformer.cc:561-564`) and with
batch 1 are never gathered. Sampler outputs and `alive_seq` live on the host, so
the `decoding.cc:683-703` gathers are CPU work and launch nothing.

## Change

Add a multi-tensor gather along axis 0 and use it in `Decoder::update_state` for
the replicated entries:

1. `ops::Gather` gains a static `batch(std::vector<StorageView*>, indices)` that
   gathers every tensor with the same indices. On CUDA it allocates the outputs
   as today, then launches **one** kernel whose parameter struct carries up to
   32 `{src, dst, row_size}` triples (chunked if there are more, so Whisper
   large-v3 with 64 tensors needs 2 launches). `grid.y` picks the tensor,
   `grid.x` strides over its uint4 elements with the same index map as today.
2. Tensors that are not CUDA, not the same dtype, or whose row is not a multiple
   of 16 bytes fall back to the current per-tensor `ops::Gather`. The CPU
   in-place path (`gather.cc:19-41`) is untouched.
3. `update_state` collects the replicated, non-empty entries and makes one call.
   The `alive_batches` branch and the embedding lookup stay as they are.

Out of scope, report only: the per-step allocations (24 `cudaMallocAsync`/`FreeAsync`
pairs stay; removing them needs persistent ping-pong buffers per state entry),
the embedding gather, `copy_2d_kernel` (31,824, runs inside each layer so it cannot
be merged across layers), and `inner_dim_offset_map` (47,112).

Correction to the earlier pitch: this does **not** remove the CUDA Graphs obstacle of
changing KV buffer addresses. The outputs are still fresh allocations each step.

## Expected numbers

- Launches: 30,264 → 1,261 (small, one launch per step), so **−29,003**,
  467,054 → about **438,000**.
- Output: a gather is a copy, so tokens and scores should be bitwise identical.
- Speed: the 30,264 kernels take 113.6 ms of GPU time and about 150 ms of launch API
  time over 13 decodes. Expect low single-digit percent on beam 5, **unverified**.

## Verification

1. Launch count: `nsys` `cuda_api_sum`, `cudaLaunchKernel + cuLaunchKernel`.
2. Tokens: `/opt/tok.py` against `/opt/tok_upstream.json`, expect no new differences.
3. CUDA tests: `--gtest_filter='CUDA/*'`, plus a new test that gathers tensors of
   different row sizes and more than 32 tensors (chunking), with repeated indices.
4. Speed: `/opt/bench.py`, paired, reference rebuilt in the same session. The control
   is **beam 1**: greedy search never calls the beam `update_state`, so it must not move.
5. Local (Mac, CPU): a temporary `throw` in the new `update_state` branch must break the
   CPU beam tests, proving the call site is reached; the CPU fallback must keep them
   passing byte for byte.

## Results (A10G, whisper-small / large-v3, 2026-09-23)

Launches (`/opt/prof.py`, small fp16 beam 5, 13 decodes), `cuda_api_sum`:

| | `50e1e882` | this change |
|---|---|---|
| `cudaLaunchKernel` | 353,590 | 324,587 |
| `cuLaunchKernel` | 113,126 | 113,126 |
| `cudaLaunchKernelExC` | 338 | 338 |
| **total** | **467,054** | **438,051** (−29,003, exactly the prediction) |
| `cudaMallocAsync` / `FreeAsync` | 253,223 each | 253,223 each (out of scope, unchanged) |

Output: tokens **and** scores identical to a `50e1e882` build in 32/32 `/opt/tok.py`
cases (small and large-v3, fp16 and int8_float16, beam 1 and 5).

CUDA tests: 182 passed, 3 skipped (the known Conv1D quantized cases), including
`GatherBatch` with 60 tensors (48 batched, so two launches).

Speed, `/opt/bench.py` paired against a `50e1e882` build rebuilt in the same session
(64 cases, warm-up 2 + median of 7, `ntok` identical in all 64):

| | median | range |
|---|---|---|
| beam 5 (changed path) | **−1.33%** | −6.20% .. +2.15% |
| beam 1 (control, never calls the beam `update_state`) | −0.19% | −1.26% .. +2.73% |
| small, beam 5 | −2.22% | −6.20% .. +1.84% |
| large-v3, beam 5 | −0.83% | −2.90% .. +2.15% |

The largest gains are on the 30 s continuous-speech window with beam 5 (more steps),
e.g. small int8_float16 354.3 → 332.4 ms. large-v3 gains less: its decoder does more
GPU work per step, so the launch overhead is a smaller share.

Measurement pitfall hit on the way: `/opt/ct2-ref-install` already existed from an
earlier session and held a `9352e203` build, so a `[ -d ] || cp` guard silently kept
the stale reference. Against it, 14 fp16 cases "differed", which were the known
bias-epilogue rounding changes. The reference is now `/opt/ct2-50e1e882-install`
(and the stale one was renamed to `/opt/ct2-9352e203-install`).

Raw data: `results/2026-09-23/bench_50e1e882_rerun.json`, `bench_beam_gather_batch.json`,
`tok_beam_gather_batch.json`.
