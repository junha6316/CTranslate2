# Round 5 results: capacity tiers for the CUDA-graph decoder

Measured on 2026-09-26 (UTC), one A10G session (sm86, CUDA 12.8, cuDNN 9.10.2), every
comparison paired within the session with a fresh process per config. Builds, back to
back on the box: the reference **1a7abfb** (its install snapshotted before any change;
all tracked `src/include/python/tests/cmake` files verified identical to the commit by
md5) and the tiers tip (1a7abfb + this change). Raw data:
[`results/2026-09-27-r5/`](../results/2026-09-27-r5/) (JSON/CSV/logs and the gate
scripts under `scripts/`; the `.nsys-rep`/`.sqlite` captures, the dense-window audio and
feature arrays stay on the box). Only numbers measured in this session appear here.

## What shipped

**`6e8736e`**: an opt-in capacity ladder for the CUDA-graph whisper decoder
(`CT2_CUDA_GRAPHS_TIERS`, see [environment variables](environment_variables.md) and the
design in [kv_cache_preallocation.md](kv_cache_preallocation.md#capacity-tiers-under-cuda-graphs-round-5)).
Without it a decode that outgrows the capped reserve (`CT2_CUDA_GRAPHS_RESERVE`) runs
the rest eager; with it the crossing step grows the caches to the next tier and the
graph runner re-captures at the new shapes. Default flags are unchanged and the default
path is bit-identical (G1).

Recommended for long-form beam5: `CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128
CT2_CUDA_GRAPHS_TIERS=+64`.

## Gates

| gate | result | evidence |
| --- | --- | --- |
| G0 CPU | pass | 234 passed / 2 skipped / 1 known failure (229 at 1a7abfb + 5 new tests) |
| G1 default bit-identity | pass | 48-config matrix BITIDENTICAL (tokens + scores) flash off and on; graphs@128 matrix BITIDENTICAL; R=64 crossing reproduces round 4 exactly |
| G-short | pass | ratios 0.9997x-1.0011x on 4 configs; 36/36 rows identical; nsys counts identical |
| G-mech | pass | 1 transition, re-capture at 65/66, 4 instantiates/decode; beam5 -3.93 ms vs single cap @64; K ~ 1.9 ms |
| G-safety | pass, with a tool limit | memcheck 0 errors; use-after-free race tracking adds 0 over a load-only control; `--track-stream-ordered-races all` aborts the unmodified build's encoder, so it could not run; CHECK 0 mismatches; Warmup/CaptureSecond/Ready crossings covered; batch-2 shrink refuses the transition |
| G-fallback | pass | 4 tier faults: 1 transition, 1 fallback, tokens = no-fault run, next decode recovers; 5 existing faults fall back |
| G-tokens | pass on the tier-correctness bar; eager-parity clause not met on one dense window | tiers vs tiers+CHECK bit-identical (dscore 0) on every workload; vs eager identical on the 84-tok window; dense:3 text flips vs eager also occur with single caps @128/@256, which run no tier code |
| G-long | pass for beam5 | W_dense +12.0 ms, W_prev +9.7 ms, greedy +13.4 ms vs single cap @128; no row slower; greedy W_dense: single @256 is 0.9 ms faster than the candidate |
| G-memory | pass | small 1178 vs 1242 MB (R=448); large-v3 4158 vs 4446 MB |
| G-large | pass | x = 0.57 ms/step, b = 4.1 us/slot/step; tiers vs single cap mean +13.0 ms, worst -0.3 ms |
| G-gather | measured | gather_rows_kernel 0.735 us/slot/step |

## G0: local CPU suite

macOS, Accelerate backend, no CUDA. `ctranslate2_test` at the tip: **234 passed, 2
skipped, 1 failed** (the known `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`).
The 1a7abfb baseline is 229 / 2 / 1 (1a7abfb itself added two env tests to the 227 of
round 4). The five new tests call the production code:

- `CacheTierPolicyParse`: `models::parse_cache_tier_policy` + `CacheTierPolicy::next`
  (off/malformed/base >= top inactive; `+64` ladder 128>192, 160>224, ..., 416>448, 448>0;
  `+50` = stride 64; `1`/`on` aliases; explicit lists rounded, clamped, sorted, filtered).
- `CacheReserveMidDecodeBump`: `set_cache_reserve_steps(40)` then
  `apply_cache_reserve(128)` before step 64: `dim(2)` 64 then 128 (one change), logits
  bit-identical to an unreserved decoder at all 128 steps, the `self_length` record at
  >= 128*16 bytes with a constant buffer over steps 64..127.
- `CacheReserveLayerOnlyBumpRegrowsRecord` (negative control): bumping only the layers
  leaves the record at 96*16 bytes until step 96, then it regrows.
- `CacheReserveBumpResetAtDecodeBoundary`, `SetCacheReserveStepsRestoresBase`.

The CUDA build's own `ctranslate2_test` on the box: tip 419 passed / 5 skipped / 3
failed, reference 1a7abfb 414 / 5 / 3: the same three pre-existing
`CPU/OpDeviceFPTest.Gemm*/float32` failures of the box build, no new ones.

## G1: default path

- `CT2_CUDA_GRAPHS_TIERS` unset, all flags off, the 48-config `bench_batch` matrix
  (small/large-v3 x fp16/int8 x ts x beam 1/5 x batch 1/4/8) with scores: tip vs 1a7abfb
  **BITIDENTICAL** (48/48 tokens and scores, max |dscore| 0) with flash attention off
  and on; the same matrix under `CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128` with
  tiers unset is BITIDENTICAL to 1a7abfb graphs@128.
- RESERVE=64 without tiers, one decode: 1 "does not fit" line, 1 "falling back" line,
  0 tier lines, NTOK 84, tokens identical to round 4's `g3_cross` dump and to 1a7abfb,
  score -0.1947544664144516 (round 4: -0.19475).

## G-short: inert without a crossing

whisper-small fp16, batch1, 84-token window, RESERVE=128. Order ABC x3 (A = 1a7abfb
graphs@128, B = tip `TIERS=+64`, C = tip tiers unset), median of the three per-run
medians:

| config | ref (ms) | +64 | unset |
| --- | --- | --- | --- |
| ts-on beam5 | 201.56 | 201.54 (0.9999x) | 201.59 (1.0001x) |
| ts-on beam1 | 150.05 | 150.01 (0.9997x) | 150.10 (1.0003x) |
| ts-off beam5 | 61.95 | 62.01 (1.0010x) | 62.00 (1.0008x) |
| ts-off beam1 | 36.00 | 36.00 (1.0000x) | 36.04 (1.0011x) |

The reference reproduces round 4's 201.5 ms. Tokens and scores over the 9 standard
windows x ts on/off x beam 1/5 (36 rows): BITIDENTICAL to the reference for both tip
configs, 0 tier transitions (longest decode 100 tokens). nsys, beam5 ts-on, 7 decodes,
reference vs `+64`: `cudaGraphLaunch` 616 / 616, `cudaGraphInstantiate` 14 / 14,
`cudaMallocAsync` 2460 / 2460, `cudaLaunchKernel` 19197 / 19197.

## G-mech: forced transitions on the 84-token window

CT2_VERBOSE=2 logs, two decodes per process
([`gmech.log`](../results/2026-09-27-r5/gmech.log)):

| config | per decode |
| --- | --- |
| R=64 +64 beam5 | 1 × `tier 64 -> 128 at step 64 (from Ready, fast re-entry)`, 0 fallbacks, captures at 65 and 66, 83 replays (56 + 27), 4 instantiates (nsys 28 / 7 decodes) |
| R=64 +64 greedy | 1 transition, capture at 65, 79 replays (58 + 21), 2 instantiates (14 / 7) |
| R=32 +32 beam5 | 2 transitions (32>64, 64>96), 6 captures, 80 replays (24 + 29 + 27) |
| R=32 +32 greedy | 2 transitions, 3 captures, 77 replays |

Walls, two interleaved rounds, median of 7:

| config | beam5 (ms) | greedy (ms) |
| --- | --- | --- |
| single cap @64 | 205.05 | 155.22 |
| **R=64 +64** | **201.12** (-3.93) | **151.55** (-3.67) |
| R=64 +64, REENTRY=warmup | 201.68 | 151.88 |
| single cap @32 | 213.58 | 162.91 |
| R=32 +32 | 202.94 (-10.64) | 153.12 (-9.79) |
| single cap @128 | 201.44 | 150.89 |

Single cap @64 reproduces round 4 (205.2 / 155.4). The gain of R=64 +64 over it passes
both bars (beam5 >= 1.5 ms, model 3.5; greedy >= 1.0 ms). Per-crossing cost, from the
model `K ~ 27x - 1.06 - gain` with round 4's x = 0.256 ms/step: **K ~ 1.9 ms** (beam5).
Fast re-entry beats warmup re-entry by 0.56 ms (beam5) and 0.33 ms (greedy): it keeps
three (beam5) or two (greedy) more replays per crossing.

## G-safety

- **compute-sanitizer memcheck** (default `cuda_malloc_async`, release threshold unset):
  **ERROR SUMMARY 0** on R=32 +32 beam5 and greedy (2 crossings each), R=128 +64 on a
  dense window with 2 crossings (199 tokens), and the whole phase sweep below (12
  decodes, 36 transitions).
- **Stream-ordered race tracking.** `--track-stream-ordered-races all` cannot run a
  whisper decode on this stack: its use-before-alloc check aborts the *encoder's* cuDNN
  convolution (`nchwToNhwcKernel`, then the xmma fprop kernel) with "unspecified launch
  failure" on the unmodified 1a7abfb build in plain eager mode too
  ([`gss_ref_eager_all.log`](../results/2026-09-27-r5/gss_ref_eager_all.log)).
  Excluding every cuDNN kernel from checking (`--kernel-name-exclude regex=cudnn`) does
  not help: the same 1a7abfb eager decode then aborts on use-before-alloc reports in a
  thrust transform reading the encoder output
  ([`gsan_all.log`](../results/2026-09-27-r5/gsan_all.log)), so this check is unusable on
  the codebase independently of this change.
  With `--track-stream-ordered-races use-after-free` (the hazard a transition introduces:
  freeing buffers that in-flight replays still read), every run reports 395 errors, the
  same count as a load-only control that never decodes (host-to-device weight copies at
  model load); the tier runs add **0**.
- **CHECK dual-run** (`CT2_CUDA_GRAPHS_CHECK=1`): R=32 +32 on 3 standard windows and
  R=128 +64 on the 3 dense windows with 2 crossings, beam5 and greedy: **0 logits
  mismatches**, at least 5 replays in every post-transition segment.
- **Phase coverage**: a previous-text prompt sweep P = 20..31 at R=32 +32 (beam5)
  crosses step 32 from Ready (P = 20..26), CaptureSecond (P = 27) and Warmup
  (P = 28..31, which re-enters through the warmup); all sanitizer-clean and CHECK-clean,
  CHECK vs plain tokens BITIDENTICAL.
- **Batch-2 early EOS** (84-token window + a 61-token window, R=64 +64): the short
  sequence ends at step 63, the batch shrinks, and the step-64 crossing is refused
  (batch 5 vs the captured 10): 0 transitions, tokens identical to eager for both.

## G-fallback

R=64 +64, two decodes per process
([`gfallback.log`](../results/2026-09-27-r5/gfallback.log)). Each tier fault
(`tier_capture`, `tier_instantiate`, `tier_replay`, `tier_alloc`) logs exactly one
transition, then one "falling back" line, then nothing; tokens and score are identical
to the no-fault tiers run (-0.19494048), and the second decode transitions normally with
0 fallbacks (tier faults fire once per process). The five existing faults at R=128 and
at R=64 +64 all fall back, NTOK 84, tokens identical to eager. `fingerprint` at R=64 +64
disables before the crossing: 0 transitions, one "does not fit" line at capacity 64,
tokens and score identical to single cap @64.

## G-tokens

- **tiers vs tiers + CHECK**: identical ids and |dscore| = 0 on every workload: R=64 +64
  on the 84-token window, R=32 +32 on 3 standard windows, R=128 +64 on all 10 G-long
  workloads, beam5 and greedy (28 rows), plus the 12-prompt phase sweep. The CHECK run
  re-executes every replayed step eagerly and compares logits bit for bit, so these
  tokens are padded-eager on the identical capacity schedule.
- **vs default eager, 84-token window**: identical ids at R=64 +64 and R=32 +32, beam5
  and greedy (|dscore| <= 4.1e-4); standard windows 1 and 2 at R=32 +32 identical too.
- **vs default eager, long workloads** (R=128 +64): 16 of 20 rows identical, 2
  timestamp-only diffs (dense:2 beam5, prev:192 beam5), 2 text diffs on the same window
  (dense:3: beam5 one text-token edit, WER 0.6%; greedy WER 3.2%), |dscore| up to 6.3e-3. These are
  the padded-attention numerics, not the tier logic: the single caps, which never run tier
  code, diverge on the same rows — single cap @128 has the identical 183-token dense:3
  greedy output and the same timestamp diffs on dense:2 and prev:192; single cap @256
  flips text on dense:3 beam5 and greedy; @448 flips text on dense:4 beam5
  ([`gl_tokens.txt`](../results/2026-09-27-r5/gl_tokens.txt)). The round-3/4 caveat
  (padded GEMM re-tiling changes tokens on int8 / large batch) therefore also covers
  time-compressed dense fp16 windows at batch 1.

## G-long: the ship gate

Workloads: **W_dense**, six 30 s windows of 55-60 s of speech time-compressed with
ffmpeg atempo 1.84-2.0 (physicsworks.wav), eager beam5 169-204 tokens; **W_prev**,
window 0 with a previous-text prompt of the last k eager text tokens of windows 1-8,
k = 64/128/192/223 (P = 67/131/195/226). Median of 7, two rounds in opposite config
order, the row value is the mean of the two run medians. Deltas vs single-cap graphs@128, positive
= faster ([`glong_report.txt`](../results/2026-09-27-r5/glong_report.txt)):

| beam5 | eager | single @128 | **+64** | +64 warmup | "256,448" | single @256 | single @448 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense:0 (183 tok) | 422.8 | 396.8 | **384.5 (+12.4)** | 385.1 | 388.7 | 403.9 | 439.4 |
| dense:1 (199) | 458.5 | 430.8 | **417.4 (+13.4)** | 419.1 | 418.9 | 434.9 | 475.2 |
| dense:2 (169) | 391.4 | 363.2 | **355.4 (+7.9)** | 356.1 | 358.4 | 373.3 | 408.8 |
| dense:3 (184) | 425.0 | 396.8 | **385.5 (+11.4)** | 386.6 | 388.6 | 403.9 | 442.3 |
| dense:4 (204) | 469.0 | 442.4 | **428.3 (+14.0)** | 429.3 | 429.3 | 445.1 | 487.4 |
| dense:5 (198) | 456.1 | 428.7 | **415.4 (+13.2)** | 416.9 | 417.3 | 432.5 | 473.2 |
| prev:64 (P=67) | 213.1 | 202.7 | **198.5 (+4.2)** | 199.4 | 200.8 | 205.8 | 223.7 |
| prev:128 (P=131) | 222.5 | 219.4 | **207.8 (+11.6)** | 208.7 | 209.4 | 209.3 | 226.7 |
| prev:192 (P=195) | 228.9 | 220.6 | **209.0 (+11.6)** | 210.1 | 214.6 | 216.9 | 227.9 |
| prev:223 (P=226) | 224.7 | 220.6 | **209.1 (+11.5)** | 210.2 | 216.6 | 221.7 | 221.5 |

| greedy | eager | single @128 | **+64** | single @256 | single @448 |
| --- | --- | --- | --- | --- | --- |
| W_dense mean | 349.0 | 325.5 | **312.2** | 311.3 | 315.4 |
| W_prev mean | 170.0 | 164.5 | **155.2** | 158.3 | 155.6 |

The runner summaries show the expected ladders: dense windows cross once (128>192) or
twice (128>192>256), W_prev once from its prompt-sized tier 0 (128>192, 160>224,
224>288, 256>320), with 4 or 6 captures per beam5 decode and no disable.

Pass criteria, candidate vs single cap @128:

1. beam5 W_dense mean **+12.0 ms** (per window +7.9 to +14.0, all >= +3): pass.
2. beam5 W_prev mean **+9.7 ms** (per k +4.2 / +11.6 / +11.6 / +11.5): pass.
3. greedy W_dense mean **+13.4 ms**: pass.
4. worst row of all 20 (beam5 and greedy): **+4.0 ms**, i.e. every row faster: pass.
5. beam5 W_dense mean 397.8 ms vs single @256 415.6 and @448 454.4: pass. For greedy
   the padding tax is small, and single @256 (no crossing on these windows) edges the
   candidate by 0.9 ms (312.2 vs 311.3; @448 315.4). The recommendation is for beam5.
6. with `CT2_CUDA_POOL_RELEASE_THRESHOLD=512M` on both sides: beam5 W_dense +11.3,
   W_prev +9.1, greedy +12.4, worst row +3.6 ms: pass.
7. fast vs warmup re-entry: fast is faster by 1.0 ms (beam5) and 0.85 ms (greedy) on
   average, on every row: the default stays `fast`.

Versus eager, the candidate saves 39.4 ms (beam5, W_dense), 16.2 ms (beam5, W_prev),
36.8 ms (greedy, W_dense) per window; single @128 saves 27.4 / 6.5 / 23.4 ms.

## G-memory

Peak process GPU memory over all W_dense + W_prev workloads, beam5 then greedy (NVML,
sampled every 2 ms): whisper-small **1178 MB** with the candidate vs **1242 MB** single
cap R=448 (single @128: 1178 MB); large-v3 (W_prev 128/223 + two dense windows)
**4158 MB** vs **4446 MB**. Both are below the R=448 footprint.

## G-large

large-v3 fp16 beam5 at R=128, first measurement of x and b (3 standard windows, two
rounds): replay saving **x = 0.57 ms/step** (0.43-0.64), padding tax **b = 4.1
us/slot/step** (3.9-4.3), about 3.6x small's same-session 1.14 us (G-gather)
(eager / pad@128 / graphs@128: 406.4 / 422.7 / 406.0, 760.8 / 790.2 / 738.3,
867.6 / 899.9 / 838.2 ms). Tiers vs single cap @128: dense:1 **+36.8 ms**, dense:4
**+15.3 ms**, prev:128 -0.3 ms, prev:223 -0.0 ms (large-v3 emits 20 tokens after these
prompts and never crosses, so tiers stay inert): mean +13.0 ms, worst -0.3 ms, above the
-8 ms floor. Tiers are not harmful on large-v3; the long-form gain there is larger.

## G-gather (measure only)

nsys `cuda_gpu_kern_sum`, whisper-small beam5, 7 decodes (637 reorders each):
`gather_rows_kernel` averages **89.6 us** at pad@128 and **324.9 us** at pad@448, i.e.
**0.735 us per cache slot per step** for the full-capacity reorder.
The same-session padding tax for small beam5 (84-token window, pad@448 - pad@128 =
32.8 ms over 28,800 extra slot-steps) is **b = 1.14 us/slot/step**, so the full-capacity
reorder is about **65%, roughly 2/3, of the padding tax**. With K ~ 1.9 ms the model's
optimal stride sqrt(2K/b) is ~58 slots today, which is where `+64` sits. A
prefix-bounded reorder gather (it must zero the shadows on every (re)allocation) is a
round-6 candidate: it would cut b to ~0.4 us and move the optimal stride to ~98 slots
(about 1.7x wider), so the `+N` default should be re-tuned with it.

## Regression track correction (1a7abfb)

The regression track's 1a7abfb commit message states, from the same box session, that
the round-3 "flash-on beam5 ts-off eager regression" deferred in
[perf_round4.md](perf_round4.md) was never about flash or timestamps: the beam-reorder
shadows grew across decodes (a98bf0e) and the async pool unmapped them at the end of
every generate. Its numbers (flash-on small beam5 ts-off after a batch-8 decode, median of
11): b3fc0d1 61.84, a98bf0e 77.87, b55f686 56.80, tip 56.69 ms; b55f686 already removes
the penalty, so its "no measured win" message and perf_round4.md's "deferred,
unexplained" are wrong on this point. The remaining release/remap cost is what the
opt-in `CT2_CUDA_POOL_RELEASE_THRESHOLD` addresses (matrices in that commit message).

## Honest caveats

- **Padded attention loses WER parity on long windows (applies to every graphs mode, not
  just tiers).** Round 3 accepted pad mode, and therefore graphs, on text/WER parity. It
  checked that only on the 84-token window. On the long dense windows, padded eager itself
  flips text against default eager: dense:3 beam5 one token edit (WER 0.6%), greedy WER 3.2%,
  |dscore| up to 6.3e-3. The single caps @128 and @256 flip dense:3 as well, and @448 flips
  dense:4. So the round-3 acceptance holds for short windows only. All graph and pad flags
  remain opt-in and the default path is untouched, but anyone enabling them for long-form
  audio should validate WER on their own data.
- **Tiers are opt-in and need the capped reserve.** The recommendation is beam5
  long-form; for greedy the reserve choice matters little (single @256 is within 0.9 ms).
- **Numerics.** Replays are bit-identical to padded eager on the same capacity
  schedule, but a transition changes the schedule, so tokens can differ from eager and
  from the single caps where beams are near ties (dense:3 above), exactly as the single
  caps differ from each other.
- **Race tracking.** Only the use-after-free half of stream-ordered race tracking could
  run (it adds 0 errors over a load-only control); the use-before-alloc half aborts the
  unmodified build's encoder even with every cuDNN kernel excluded. Plain memcheck is
  clean on every tier configuration.
- **K** comes from the round-4 value of x (0.256 ms/step) inserted in the model, not
  from a direct per-crossing measurement.
- One box session, within-session pairs only; absolute numbers are not comparable across
  sessions.
- **Test coverage gaps found in review.** The CPU tests reach the tier policy parsing,
  `CacheTierPolicy::next`, `apply_cache_reserve` and `set_cache_reserve_steps`, but the test
  that bumps the reserve mid-decode picks the step itself instead of going through the guard
  in `TransformerDecoder::decode`. `DecoderGraphRunner` transitions (the `begin_tier_transition`
  refusals, Reentry -> ReentrySecond -> Ready) are covered only by the GPU gates. The
  batch-shrink refusal in G-safety was reached because one window happens to end at step 63;
  the gate script does not assert which branch refused.
- **G1 log.** The committed `g1.log` shows the compare step failing (`python: command not
  found`) and `scripts/g1.sh` was edited to `python3` afterwards. The G1 verdict
  (48/48 bit-identical, flash off and on, and under graphs@128) was recomputed from the
  committed JSONs during review and holds.

## Post-review fixes (ef365c0)

- A malformed `CT2_CUDA_POOL_RELEASE_THRESHOLD` no longer breaks the `cub_caching`
  allocator, where the variable is ignored. The warning now checks the raw string and does
  not parse it. Under `cuda_malloc_async` a malformed value still fails at the first
  allocation.
- `clear_cache()` synchronizes the device before trimming the pool, so `unload_model()`
  returns memory whose asynchronous frees were still in flight. Checked on the A10G: with
  the threshold at `max`, unloading after a b5 x n8 decode goes from 1595 MB to 283 MB,
  the same floor as the default (1243 -> 283 MB).
- Final paired bench (round-4 tip vs this branch, ABBA, full matrices) is in
  [`results/2026-09-26-r5/`](../results/2026-09-26-r5/) (`final_matrix_report.txt`,
  `final_graphs_report.txt`, nsys `final_nsys_*_cuda_api_sum.csv`), together with the
  regression bisection logs.
