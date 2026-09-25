# Round 3 results: per-site allocation reuse (Stage A) + opt-in CUDA graphs (Stage B)

Measured on 2026-09-25, one A10G session, builds installed side by side and run
back to back. Raw data: [`results/2026-09-25-r3/`](../results/2026-09-25-r3/)
(box artifacts) and [`results/2026-09-26/`](../results/2026-09-26/) (the paired
batch matrix and comparison). The change plan and gate protocol are in
[perf_round3_validation.md](perf_round3_validation.md); this file records only
what the session measured.

## What shipped

**Stage A (7 commits, `a98bf0e`..`40928f1`)** — pure allocation reuse, no op,
shape or numeric change. Six per-site commits (beam-reorder gather shadow
buffers, TopK scratch slots, memoized timestamp-rule row ids, decoder-layer
activation slots, lazy alignment-head selection, opt-in KV preallocation) plus
one fix this round: the beam>1 cross-attention `split_heads`/`combine_heads`
head-transpose was re-allocating an exact-fit function-local every layer every
step and destroying the workspace slots' grown capacities. It was threaded onto
a `DecodeWorkspace::head_transpose` scratch slot. The bar for every Stage-A
commit was bit-identical tokens, and the session held it (below).

**Stage B (`c5c1804`)** — opt-in CUDA-graph replay of the decoder forward,
gated behind `CT2_CUDA_GRAPHS=1` (implies KV preallocation + padded GEMMs).
Stage B ran the full gate chain B0-B3 and **shipped only greedy**; beam5 was
held back at the Gate-B3 wall-parity check and the flag stays default-off /
experimental. Evidence for the beam5 hold is in the paired numbers below: at
beam5 the replay saving does not pay back the 448-capacity padding tax, so
graphs are a measured net loss versus eager.

## Token identity

- **Stage A (r3, default flags) vs round-2 tip (r2base = `b3fc0d1`)**: token ids
  **bit-identical across all 48 configurations** (2 models x 2 compute types x
  ts on/off x beam 1/5 x batch 1/4/8), flash off and flash on both. The
  allocation-reuse claim holds on GPU. (`SUMMARY` in
  [`results/2026-09-26/compare_full.txt`](../results/2026-09-26/compare_full.txt):
  `r3_vs_r2base: true`, `r3_vs_r2base_flash: true`.)
- **Graphs (r3 + `CT2_CUDA_GRAPHS=1`) vs r3 eager**:
  - flash **on**: bit-identical across all 48 configs (`graphs_vs_eager_flash:
    true`) — the flash path refuses graph capture and falls back to eager, so
    the flag is effectively a no-op there (wall deltas are ±0.1% noise, see
    below).
  - flash **off**: **not identical** (`graphs_vs_eager: false`). The diffs are
    the anticipated padded-GEMM re-tiling: cuBLAS picks different reduction
    tilings at the padded n/k, and the flips land on near-tied timestamp
    tokens. In the validation session's controlled dual-run these were
    single-token timestamp flips with text/WER parity (19/21 windows identical);
    in the full batch matrix the TOK-DIFF marks concentrate on int8 and
    large-v3 large-batch configs. Graphs are opt-in and the default path is
    untouched, so this does not affect shipped behavior.

## Paired numbers, this session

### Batch matrix, r3 vs r2base (eager, default flags, `bench_batch.py`)

Both builds eager; negative = r3 faster. Tokens identical everywhere.

| config (median ms) | flash off | flash on |
| --- | --- | --- |
| small fp16 ts=1 b5 n=1 | 234.0 -> 215.3 (**-8.0%**) | 258.2 -> 247.7 (-4.1%) |
| small int8 ts=1 b5 n=1 | 289.8 -> 273.6 (-5.6%) | 269.9 -> 259.2 (-3.9%) |
| small fp16 ts=0 b5 n=1 | 63.9 -> 61.1 (-4.5%) | 61.8 -> 75.2 (**+21.8%**) |
| large-v3 fp16 ts=1 b5 n=1 | 416.9 -> 405.4 (-2.8%) | 464.4 -> 442.9 (-4.6%) |
| large-v3 fp16 ts=0 b5 n=1 | 334.0 -> 339.4 (+1.6%) | 359.7 -> 427.7 (**+18.9%**) |

The gain is real but small and concentrated at the launch-bound end (small,
beam5, batch1), consistent with round 2. Flash-off is a clean win or noise
across the matrix (median ~-1 to -2%). **Caveat:** flash-**on** beam5/batch1
with timestamps **off** regresses hard on three configs (+16 to +22%: small
fp16, small int8, large-v3 fp16/int8 ts=0 b5 n=1), while the same configs with
timestamps on improve. This is a genuine regression in that corner, not noise
(IQRs are tight), and is not yet explained — flagged for round 4.

### Gate-progression single-window walls (nsys protocol, beam1 / beam5)

One 30 s window, 84 tokens, median wall. r2base tip -> Stage A -> +head-transpose
fix -> padded-eager -> graphs:

| build | beam1 | beam5 |
| --- | --- | --- |
| tip `b3fc0d1` (r2base) | 236.1 | 426.6 |
| Stage A | 164.1 | 227.1 |
| Stage A + head-transpose fix (eager default) | 163.2 | 215.2 |
| padded-eager (`CT2_CUDA_PAD_KV=1`) | 167.6 | 255.3 |
| graphs (`CT2_CUDA_GRAPHS=1`) | **151.9** | 234.3 |
| flag OFF (parity check) | 163.7 | 216.6 |

Reading, net of the ~25 ms non-decode overhead (encoder + prompt):

- **greedy batch1**: graphs 151.9 vs eager 163.2 = **-6.9% total, -8.2% on
  decode wall** — Stage B ships here.
- **beam5 batch1**: padding alone costs +40 ms (215.2 -> 255.3, +18.6%); the
  replay saves ~21 ms back (255.3 -> 234.3) but still lands at **+10.1% vs
  eager** on decode wall. Net loss → beam5 graphs held default-off.
- flag OFF matches r2base tip within noise (163.7 / 216.6 vs 163.2 / 215.2).

### Graphs vs eager in the batch matrix

Confirms the wall progression at scale: with flash **off**, graphs are a net
loss at every batch>1 and most beam5 configs (up to +56% at large-v3 b5 n=8),
the padding tax dominating; the only clear win is greedy small ts=1 b1 n=1
(-7.2%). With flash **on**, graphs are ±0.1% everywhere (fall-back no-op).

## nsys API counts (this session, small fp16 beam5 ts-on batch1, 5 decodes)

From [`results/2026-09-26/nsys_api_counts.txt`](../results/2026-09-26/nsys_api_counts.txt)
and the `cuda_api_sum` CSVs:

| API | r3 eager | r3 graphs |
| --- | --- | --- |
| `cudaLaunchKernel` | 164,197 | 18,693 |
| `cudaGraphLaunch` | 0 | 616 |
| `cudaMallocAsync` | 2,990 | 2,317 |
| `cudaFreeAsync` | 2,990 | 2,317 |
| `cudaLaunchKernelExC` | 182 | 182 |

Two mechanism checks pass. (1) Stage A collapsed `cudaMallocAsync` to **2,990**
from the round-2 tip's 54,321 in this exact config (validation Gate A), i.e. the
per-step alloc/free churn is gone. (2) Graph replay collapses eager kernel
launches **164,197 -> 18,693**, with 616 `cudaGraphLaunch` replaying nearly
every steady step across the 7 decodes.

### Gate B0 microbenchmark (`bench_graphs`, A10G sm86, 300 reps median)

From [`results/2026-09-25-r3/b0_result.txt`](../results/2026-09-25-r3/b0_result.txt):
at 130 launches, eager 330.1 us vs one graph replay 144.4 us = **185.7 us saved
per step**; per-step `SetParams`+launch (153.7 us) is barely cheaper than a plain
replay and whole-graph `ExecUpdate`+launch (231.5 us) costs more than replay —
both rejected as the primary scheme, plain replay chosen. The projected
per-step saving justified GO into B1.

## Local CPU suite (re-run this session)

macOS, Accelerate CPU backend, no CUDA compiled. `ctranslate2_test` rebuilt
incrementally and run at the branch tip `c5c1804`: **suite green except the
known pre-existing `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`**
failure (no INT8 GEMM backend with `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON`,
documented in [flash_attention_validation.md](flash_attention_validation.md)).
The Stage-A per-commit tests (gather shadows, TopK scratch, memoized upload,
layer-slot ping-pong across steps, alignment-head memo, cache-reserve
preallocation + growth fallback) all pass on CPU.

## Honest caveats

- **Beam5 graphs are a measured net loss** and ship default-off. The
  448-capacity padding tax (+18.6% at beam5) is the cause; the round-4 lever is
  a `CT2_CUDA_GRAPHS_RESERVE` knob capping the padded capacity below max_length.
- **Graphs change tokens (flash off)**: single timestamp-token flips from padded
  fp16 GEMM re-tiling, text/WER parity but not bit-identical. Acceptable only
  because graphs are opt-in; the default eager path is bit-identical to round 2.
- **Flash-on + graphs is a no-op** (capture refused, falls back to eager). The
  flag buys nothing on the flash path today.
- **Flash-on eager regression at beam5/batch1/ts-off** (+16 to +22% on four
  configs) appeared this session in the r3-vs-r2base matrix, tokens identical.
  Unexplained, not a graphs artifact, carried to round 4.
- **One box session, within-session pairs only.** Absolute numbers are not
  comparable to the 2026-09-23 session (audio substitution carried over from
  round 2: a 203 s LibriVox reading under the lost `physicsworks.wav` name).
- **CPU suite cannot exercise any GPU path.** The graph capture, padded GEMMs,
  and fault-injection fallbacks are only covered by the A10G session logs, not
  by the local build.
