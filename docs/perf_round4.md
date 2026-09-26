# Round 4 results: capping the CUDA-graph KV reserve (the beam5 flip)

Measured on 2026-09-26, one A10G session (sm86), builds run back to back within the
session. Raw data: [`results/2026-09-26-r4/`](../results/2026-09-26-r4/) (box
artifacts, JSON/CSV/logs only; the `.nsys-rep`/`.sqlite` captures stay on the box).
This file records only what the session measured; the round-3 hold it resolves is in
[perf_round3.md](perf_round3.md).

## What shipped

**`b55f686`** — alloc cleanup: `Decoder::replicate_state` now clears
`_reorder_shadows` at the start of each generation. The beam-reorder gather stages
through these shadow buffers, which only ever grow; without the reset a previous
larger decode leaves them oversized, and once a shadow exceeds the sync caching
allocator's 16 MB max cached bin the paired post-gather buffer can no longer be
recycled by a later smaller decode. Defensive hygiene, not a numeric change.

**`a6114ac`** — the round's lever: a new opt-in int env `CT2_CUDA_GRAPHS_RESERVE`
that caps how many KV steps the graph path reserves.

## The reserve knob design

- Default `0` (unset) keeps the full `options.max_length` reserve (today's 448) and
  the path stays **bit-identical** to round 3. A value `>0` reserves
  `min(max_length, knob)` steps.
- Independent of `CT2_CUDA_GRAPHS` and `CT2_CUDA_PAD_KV` — usable with either.
- `append_to_cache` rounds the reserve up to a 32-step block, so `knob=100` reserves
  128.
- The only runtime change is the reserve computation in `WhisperReplica::generate`.
  The graph fingerprint already bakes the KV capacity `dim(2)` (`transformer.cc`), so
  a smaller cap can never satisfy a stale executable; a window longer than the cap
  hits the existing host guard → disable-for-decode → eager tail with 32-block growth
  (no recapture needed).

Round-3 held beam5 graphs default-off because at reserve=448 the padding tax (+40 ms)
outweighed the ~-20 ms replay saving. The knob cuts the tax while keeping the replay.

## Did beam5 flip to a win? Yes — with the cap

Primary config: whisper-small fp16, beam5, batch1, timestamps on, first 30 s window,
84 tokens, median-of-7, **total generate wall**. Negative = faster than eager
([`results/2026-09-26-r4/g1g2.log`](../results/2026-09-26-r4/g1g2.log)).

| reserve | pad-eager (ms) | graphs (ms) | graphs vs eager |
| --- | --- | --- | --- |
| eager (ref) | — | — | 215.7 |
| 448 (r3 default) | 254.0 | 233.7 | **+8.3%** (net loss — the r3 hold) |
| 256 | 235.0 | 215.0 | break-even |
| 160 | 225.7 | 206.5 | -4.3% |
| **128** | 221.1 | **201.5** | **-6.6%** |
| 96 | 219.3 | 199.1 | **-7.7%** |

So beam5 graphs go from a **+8.3% net loss at the round-3 default** to a **-6.6% win at
reserve=128** and -7.7% at reserve=96. Decode wall at reserve=128 is ~176 ms vs eager
189.7 ms. **128 is the safe knee** (44-step margin over L=84); 96 wins a little more but
leaves only a 12-step margin before the crossing guard fires.

Greedy batch1 still wins on top of this: reserve=128 150.1 ms vs eager 165.6 ms
(**-9.4%**).

## Regression outcome: the round-3 beam5 hold is fixed, with numbers

G1 isolated the two effects and confirmed the mechanism. The padding tax scales with
the reserve; the replay saving does not:

| reserve | pad-eager tax vs eager | replay saving (graphs vs pad@same) |
| --- | --- | --- |
| 448 | +38.3 ms | -20.3 ms |
| 128 | +5.4 ms | -19.6 ms |
| 96 | +3.6 ms | -20.2 ms |

The replay saving is C-invariant at ~-20 ms; capping the reserve is purely a tax cut.
That is exactly why the round-3 net loss disappears: the ~-20 ms was always there,
round 3 just paid a +38 ms tax to get it, and reserve=128 drops that to +5 ms.

**Still deferred:** the *separate* round-3 caveat — flash-**on** *eager* regression at
beam5/batch1/timestamps-off (+16 to +22% on four configs) — was not re-measured this
session and remains unexplained. It is not a graphs artifact and is untouched by this
round; carried to round 5.

## Token identity

- **Flag-off (default) vs round-3 eager baseline**: bit-identical across the full
  48-config matrix (`configs=48 identical=48 diff=0`, `BITIDENTICAL` in
  [`results/2026-09-26-r4/g4matrix_san.log`](../results/2026-09-26-r4/g4matrix_san.log)).
  The knob does not touch the default path.
- **graphs@C vs padded-eager@C**: bit-identical (`|dscore| 0`).
- **pad@128 vs eager** on the primary fp16 window: tokens identical, `|dscore| <=
  1.86e-4`, WER 0.
- **Carried flash-off divergence:** graphs (flash off) still flip tokens on the int8 /
  large-batch corner — e.g. small int8 ts-on b5 n=1 is 84 tok eager but 75 tok under
  both graphs and reserve=128
  ([`tokcheck.log`](../results/2026-09-26-r4/tokcheck.log)). This is the round-3
  padded-GEMM re-tiling divergence, unchanged; acceptable only because graphs are
  opt-in and the default eager path is bit-identical.

## Crossing safety and fallbacks (G3/G4)

From [`g3g4.log`](../results/2026-09-26-r4/g3g4.log):

- **Forced crossing** (reserve=64 on the 84-tok window): fires exactly one
  disable-for-decode at step 64 (`cache capacity 64, position table 448, going
  eager`), runs the tail eager, and produces tokens **identical to eager**.
  compute-sanitizer memcheck: **0 errors** (`g4matrix_san.log`).
- **CHECK dual-run** (reserve=128, 3 windows of 84/77/88 tok): **0 logits mismatches**.
- **5 fault injections** (capture / instantiate / replay / fingerprint / alloc): every
  one falls back to eager and returns correct tokens (NTOK 84 each).
- **Memory front**, reserve=128 vs 448: small 986 vs 1082 MB, large-v3 3678 vs 3902 MB
  — the cap also trims the KV reservation.

## nsys API counts

whisper-small fp16 beam5 ts-on batch1, 5 decodes, via `nsys stats --report
cuda_api_sum` on the box captures. The cap does not cost replay coverage:

| API | eager (`nsys_eager`) | graphs @128 | graphs @96 |
| --- | --- | --- | --- |
| `cudaLaunchKernel` | 164,209 | 19,197 | 19,197 |
| `cudaGraphLaunch` | 0 | 616 | 616 |
| `cudaMallocAsync` | 3,158 | 2,460 | 2,460 |
| `cudaFreeAsync` | 3,158 | 2,460 | 2,460 |
| `cudaLaunchKernelExC` | 182 | 182 | 182 |

`cudaGraphLaunch` = 616 is identical to the round-3 reserve=448 count, i.e. capping the
reserve to 128 (or 96) retains the same replay coverage while cutting the padded-GEMM
tax; kernel launches still collapse 164,209 → 19,197.

## Local CPU suite (re-run this session)

macOS, Accelerate CPU backend, no CUDA compiled. `ctranslate2_test` rebuilt
incrementally at the branch tip `a6114ac` and run: **227 passed, 2 skipped, 1 failed —
the known pre-existing `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`** (no
INT8 GEMM backend with `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON`, documented in
[flash_attention_validation.md](flash_attention_validation.md)). Suite green otherwise.

## Honest caveats

- **The beam5 win requires the knob.** At the default reserve (unset = 448) beam5
  graphs remain a net loss, so `CT2_CUDA_GRAPHS` still ships default-off and beam5
  needs an explicit `CT2_CUDA_GRAPHS_RESERVE`. The knob is the opt-in that makes it a
  win, not a change to the default.
- **The cap only helps when the window fits.** A decode longer than the cap hits the
  disable-for-decode guard and runs the tail eager — safe and token-identical, but no
  speedup on that window. reserve=128 was chosen for a 44-step margin over the 84-token
  measurement window; a workload with longer windows would need a higher cap (and pay
  back some tax).
- **flash-off graphs still change tokens** on int8/large-batch (padded-GEMM
  re-tiling). Opt-in only; the default eager path is bit-identical to round 2/3.
- **Flash-on + graphs remains a no-op** (capture refused → eager fallback); not
  re-measured this round, assumed unchanged.
- **Flash-on eager beam5/ts-off regression** (round-3, +16–22%) is deferred, not fixed.
- **One box session, within-session pairs only.** Absolute numbers are not comparable
  across sessions.
- **CPU suite cannot exercise any GPU path.** The graph capture, reserve cap, crossing
  guard, and fault-injection fallbacks are covered only by the A10G session logs.
