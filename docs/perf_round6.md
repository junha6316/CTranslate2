# Round 6 results: prefix-bounded beam reorder of the self-attention cache

Measured on 2026-09-27 (UTC), one A10G session (g5.2xlarge, sm86, CUDA 12.8), every
comparison paired within the session with a fresh process per config and every timing
job serialized under one GPU lock. Builds, side by side on the box: the reference
**90f7243** (`~/w/base`) and the round-6 tree (`~/w/r6`, 90f7243 + this change; all 12
changed files md5-identical to `bcc294e`). Raw data:
[`results/2026-09-27-r6/`](../results/2026-09-27-r6/) (JSON/CSV/logs and the gate
scripts; the `.nsys-rep`/`.sqlite` captures stay on the box, and the five sanitizer logs
over 1 MiB are committed as `san/*.trimmed.log`). Only numbers measured in this session
appear here.

## The lever and its anchor

Every beam-search step reorders the decoder self-attention K/V cache by the parent beam
(`Gather::batch` with the round-3 shadow ping-pong). The kernel copied **full rows**, i.e.
the whole allocated capacity, although only the first `L = offset` time steps are valid:

- **Flash attention** preallocates the cache at 3 + 512 = 515 steps. Base nsys
  (whisper-small fp16 flash beam5 batch1 ts-off, 11 decodes): `gather_rows_kernel` 176
  launches, avg **370.2 us**, **5.92 ms per decode** out of ~50 ms of GPU time.
- **Preallocated MHA** (`CT2_CUDA_PREALLOC_KV`, `CT2_CUDA_PAD_KV`, the graphs reserve and
  tiers) has layout `[N, H, C, D]` with `C` = 128/448 or the current tier. Round 5 measured
  the full-capacity reorder at 0.735 us/slot/step, about **65% of the graphs padding tax**
  b = 1.14 us/slot/step (perf_round5.md, G-gather), and predicted that a prefix-bounded
  gather would bring b to ~0.4 us.

## What shipped

Two commits on `perf/flash-gather`:

- **`810a24b`**, stage 1, flash cache (`[N, C, H, D]`): the gather copies only `[0, L)` of
  each row. FA2 bounds keys by `seqlens_k = offset` and the cache grows by concatenation
  only when full, so the tail is never read (verified in G6).
- **`bcc294e`**, stage 2, preallocated MHA cache (`[N, H, C, D]`): one segment per head
  (`segments = H`, `length = L * stride(2)`, `pitch = stride(1)`) with `keep_tail = true`.

On by default for CUDA; `CT2_CUDA_GATHER_PREFIX=0` is the kill switch. The eager default
MHA cache (reserve 0) is untouched (full rows, which are already `L` long).

## Design and why

- **Kernel.** `gather_rows_kernel` / `GatherRowsArgs` take a per-tensor segment
  descriptor (`segs x seg_words` at `seg_pitch`, in 16-byte words). A full row is
  `segs = 1, seg_words = seg_pitch = row_words`, i.e. exactly the old index math, so the
  batched one-launch-per-step structure of round 3 is kept and non-cache tensors are
  unchanged. `ops::GatherRowSegments` + a 4-argument `Gather::batch`; the old overloads
  forward `segs = nullptr`.
- **Hook.** `Decoder::reorder_segments` is a virtual hook; `TransformerDecoder` returns
  descriptors only for `self_keys_*`/`self_values_*`, using the new
  `AttentionLayer::cache_time_dim()` to pick the flash (time dim 1) or MHA (time dim 2)
  layout. Everything else in the state keeps full-row gathers.
- **Why `keep_tail` for MHA.** Padded attention (graphs, `PAD_KV`) reads the spare tail
  under a mask, so the shadow bytes must stay identical to what the full-row path would
  produce, or the masked logits could change through NaN/Inf garbage. `append_to_cache`
  zeroes on growth (`grown.zero()`), appends write at `offset`, so if both ping-pong buffers
  start with zero tails they keep them by induction. `ops::GatherShadows` therefore records
  the buffers of its last swap and treats a step as **steady** only when both buffers are
  the ones this slot paired itself; every non-steady step (first reorder of a decode,
  growth / tier transition, batch shrink, re-paired buffers) copies full rows, which
  carries the zeroed tail into the shadow. The flash cache is never read past `L`, so it
  uses `keep_tail = false` (always prefix).
- **Why not skip the reorder altogether** (indirect beam attention): that changes every
  attention kernel and the graph-captured shapes. The prefix gather is local to the
  gather op, bit-identical by construction, and keeps the round-3/4/5 machinery intact.

## Gates

| gate | result | evidence |
| --- | --- | --- |
| G0 unit tests | pass | local CPU suite 241 passed / 2 skipped / 1 known failure; box CUDA `Gather*` + `TransformerDecoder*` 48/48; `TransformerDecoderPrefixTest` 6/6 on CPU and CUDA, also under `PAD_KV=1` and `GRAPHS=1`; mutation `keep_tail = false` fails 6/6 |
| G1 bit-identity | pass | 96-config matrix (48 x flash off/on), tokens and scores vs 90f7243: 0 diffs for each stage, and with `CT2_CUDA_GATHER_PREFIX=0` |
| G2 opt-in modes | pass | PREALLOC_KV, PAD_KV, graphs pad@128/@448, tiers +64 (std, dense 128>192(>256), prev:192) x beam5/greedy + batch-4 shrink: 63 rows, 0 token/score diffs |
| G3 gather time | pass | flash 5.92 -> 0.117 ms/decode (small), 42.1 -> 1.39 (large-v3); MHA pad@128 8.16 -> 3.09, pad@448 29.59 -> 3.24, tiers std 8.16 -> 3.09, tiers dense:1 22.24 -> 15.01 ms/decode |
| G4 wall | pass | flash beam5 small ts-on 248.2 -> 218.7 ms, batch8 -31.8%; graphs pad@448 beam5 -10.2..-10.5%, pad@128 -1.7..-2.1%, tiers -0.7..-2.1% |
| G5 no regression | pass | full matrix flash-off median +0.1%, no config > 3% slower (details and the drift probe below) |
| G6 sanitizer | pass | memcheck 0 errors on every cell; initcheck totals equal to base with the writers instrumented |
| final paired bench | pass (one caveat) | independent chain `final6_chain.sh`, base vs `bcc294e`, two passes in opposite order: tokens 96/96 both passes, graphs 16 cells same tokens, gather 370.0 -> 7.28 us, memcheck 0; two single-pass flash-off cells > 3% that do not repeat (see G5) |

## G0: unit tests

- Local macOS CPU (Accelerate, no CUDA) `ctranslate2_test` at `bcc294e`: **241 passed,
  2 skipped, 1 failed** (the known `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`);
  the same count at `810a24b` alone. New tests: `ops_test` `GatherBatchSegments` (full,
  prefix and keep_tail descriptors, steady and mispaired shadows) and 6
  `TransformerDecoderPrefixTest` variants that compare logits and self K/V bytes after every
  `update_state` against the full-row path (`Decoder::set_prefix_reorder(true)` runs the
  host implementation on CPU).
- Box CUDA build: `Gather*` and `TransformerDecoder*` 48/48; the prefix tests 6/6 on CUDA,
  also with `CT2_CUDA_PAD_KV=1` and `CT2_CUDA_GRAPHS=1`, with the reorder shadows compared
  byte for byte and `prefix_descriptors > 0` asserted. The full box suite shows only the
  pre-existing `CPU/OpDeviceFPTest.Gemm*/float32` failures of the box build.
- Mutation: forcing `keep_tail = false` for MHA fails 6/6 prefix tests.

## G1 / G2: tokens

- 96-config `bench_batch` matrix (small/large-v3 x fp16/int8_float16 x ts x beam 1/5 x
  batch 1/4/8, flash off and on) with scores: stage 1 and stage 2 each **0 token and 0
  score diffs** vs 90f7243, and 0 diffs with the kill switch.
- Opt-in modes (`g2/`, `g2cmp.py`): PREALLOC_KV, PAD_KV, graphs pad@128, pad@448, tiers +64
  on the standard windows std:0-3 (+ std:0 ts-off), dense:0/1/3 (tier crossings
  128>192(>256)) and prev:192, beam5 and greedy, plus a batch-4 beam5 case where a
  sequence ends early and the batch shrinks: **63 rows, 0 diffs** (tokens and scores) for
  stage 2, and 0 diffs for stage 1.

## G3: gather time (nsys `cuda_gpu_kern_sum`)

Flash (stage 1), `gather_rows_kernel`, launch counts unchanged:

| model | avg per launch | per decode |
| --- | --- | --- |
| small fp16 beam5 batch1 ts-off (176 launches, 16/decode) | 370.2 -> **7.32 us** | 5.92 -> **0.117 ms** |
| large-v3 fp16 beam5 batch1 (825 launches) | 561.1 -> **18.6 us** | 42.1 -> **1.39 ms** |

Preallocated MHA (stage 2), whisper-small fp16 beam5 batch1 ts-on, 7 decodes / 637
reorders unless noted (`gg/`):

| mode | per decode | avg per launch | target |
| --- | --- | --- | --- |
| graphs pad@128, std:0 | 8.16 -> **3.09 ms** | 89.7 -> 33.9 us | <= 4 |
| graphs pad@448, std:0 | 29.59 -> **3.24 ms** | 325.1 -> 35.6 us | <= 6 |
| tiers +64, std:0 | 8.16 -> **3.09 ms** | 89.7 -> 34.0 us | |
| tiers +64, dense:1 (199 tok, 5 decodes, 995 reorders) | 22.24 -> **15.01 ms** | 111.7 -> 75.4 us | model 22.58 -> 15.52 |

All-kernel GPU time per decode: pad@128 49.19 -> 44.10 ms, pad@448 71.29 -> 44.88 ms,
tiers dense:1 78.3 -> 71.1 ms. Stage 1 alone leaves the MHA numbers at base (pad@128
56.95 ms per 7 decodes), and stage 2 leaves the flash anchor at stage 1 (7.34 vs 7.32 us).

## G4: wall

Flash (stage 1, median of 7 per cell, 96-config matrix pass):

| config | base | stage 1 |
| --- | --- | --- |
| small fp16 beam5 batch1 ts-off | 57.6 ms | **52.2 ms** |
| small fp16 beam5 batch1 ts-on | 248.2 ms | **218.7 ms** |
| large-v3 fp16 beam5 batch1 ts-on | 441.3 ms | **387.2 ms** |
| large-v3 fp16 beam5 batch1 ts-off | 340.5 ms | **300.9 ms** |
| beam5 batch8 ts-on | | small **-31.8%**, large-v3 **-31.4%** |

Preallocated MHA (stage 2 vs a fresh base pass run right after it, median of 7, beam5):
PREALLOC_KV **-10.0..-10.3%**, PAD_KV **-9.7..-10.2%**, graphs pad@128 **-1.7..-2.1%**,
pad@448 **-10.2..-10.5%**, tiers +64 std **-1.4..-2.1%**, tiers dense **-0.7..-1.0%**,
prev:192 -0.3%. Batch-4 beam5: PREALLOC_KV -21.5%, PAD_KV -19.7%, pad@128 -4.8%, pad@448
-19.7%, tiers -3.7%. Greedy never reorders at batch 1: -0.4..+0.7%.

Final paired bench (`final6_chain.sh`, bench_long.py, mean of two rounds run base->r6 then
r6->base, median of 7 each, whisper-small fp16, tokens identical in every cell):

| mode | std:0 beam5 | dense:1 beam5 | std:0 greedy | dense:1 greedy |
| --- | --- | --- | --- | --- |
| eager | 216.4 -> 217.2 (+0.3%) | 455.6 -> 456.8 (+0.3%) | -0.7% | -0.4% |
| graphs @128 | 201.5 -> 197.8 (**-1.9%**) | 430.6 -> 426.3 (-1.0%) | +0.1% | -0.1% |
| graphs @448 | 233.6 -> 209.3 (**-10.4%**) | 475.6 -> 427.8 (**-10.0%**) | +0.1% | +0.2% |
| tiers +64 | 201.8 -> 197.8 (**-2.0%**) | 418.2 -> 412.7 (-1.3%) | -0.1% | +0.1% |

## G5: no regression on the default matrix

- Stage 2 vs base (96 configs, one pass each): flash-off median **+0.1%** (range
  -0.5..+1.3%), flash-on median **-0.9%** (range -31.9..+1.9%); vs stage 1 off -0.0%
  (-1.5..+0.4%), on -0.1% (-0.5..+1.9%). No config is more than 3% slower. Tokens 48/48 in
  every comparison.
- Stage 1, three interleaved passes: per-cell median over passes, then median over cells,
  flash-off +0.17% (range -0.37..+4.20%), flash-on -0.73% (-31.76..+1.23%). The flash-off
  outliers were small int8_float16 beam1 cells (ts1 b1 n1 +1.23/+7.18/+3.53% per pass), a
  path that never reorders. Every matrix pass ran base first. Process-interleaved probes
  (`abab*.sh`, ABBA, 6-8 rounds) put those cells at +0.5..+1.0%, **the same with
  `CT2_CUDA_GATHER_PREFIX=0`** (+0.76..+2.20% vs s1 +1.04..+1.21%) and within +0.4..+0.8%
  when the base Python package is merely rebuilt a second way (base vs base: -0.25..+0.43%).
  Unchanged host object files are byte-identical between the two builds; `.text` grew by
  0x2330 bytes. So the residual <= 1% on host-bound cells is a library-level (likely code
  layout) effect, not the reorder logic, and the large matrix deltas were run-order drift.

- Final paired bench, two full 96-config passes (`final_{base,r6}_{1,2}_{off,on}.json`; pass 1
  ran base first, pass 2 r6 first): tokens **48/48 identical** in all four comparisons.
  Flash-off median **+0.1% / +0.2%** (range -0.9..+4.3% / -0.8..+3.0%), flash-on median
  **-2.3% / -0.6%** (range -31.8..+1.3% / -31.7..+0.9%). One cell per pass exceeds 3%:
  small int8_float16 ts-off beam1 batch4 **+4.3%** in pass 1 (a beam1 path that never
  reorders) and small int8_float16 ts-on beam5 batch1 **+3.0%** in pass 2; neither repeats in
  the other pass. Read strictly, the "no config > 3%" gate fails on single passes; both cells
  are the host-bound small int8 cells already shown above to move with run order.

## G6: sanitizer

- **memcheck** (all kernels): **0 errors** for stage 1 and stage 2 on flash beam5, flash
  beam4 (batch shrink), graphs pad@128 beam5, PAD_KV beam4 (shrink), and tiers +64 on
  dense:1 (199 tokens, 128>192 crossing).
- **initcheck, flash** (gather kernel only): 5.92M -> 1.07M flagged reads (beam5), 59.2M
  -> 10.2M (beam4): the full-row reads of the never-written +512 tail are gone. FA2 tail
  check: flagged reads attributed to `flash_fwd` (gather|flash_fwd minus gather-only) 52.25M
  with stage 1 vs 53.29M base; a tail read would add ~5e8. **FA2 does not read the tail.**
- **initcheck, MHA**: with the writers instrumented too (gather, thrust, cub, copy_2d,
  fill) and the attention kernels, stage 2 totals **equal base exactly**: graphs@128 beam5
  43,404,666; PAD_KV beam4 312,067,766; tiers dense 43,404,956 (writers only, graphs@128:
  392,285,073 both). With only the gather kernel instrumented stage 2 flags more reads
  (b5 1,993,376 vs 1,474,931): a temporary debug print showed 1 non-steady and 90 steady
  reorders, and the excess is exactly 24 tensors x 45 steps x 480 words, i.e. the appended
  slot written by the uninstrumented append kernel, which the base full-row gather used to
  re-read (and "bless") on every step. A tool artifact, not an uninitialized read.
- `--track-stream-ordered-races` and a full-kernel initcheck remain unusable on this stack
  (pre-existing cuDNN encoder reports, see perf_round5.md), and filtered initcheck of the MHA
  attention GEMMs with backtraces times out at 2400 s.

- Final chain rerun (`final6_chain.log`, SAN step): memcheck **0 errors** on flash_b5 and
  b5_GRAPHS for both trees; filtered initcheck totals reproduce the numbers above exactly
  (flash_b5 gather 1,072,291 vs base 5,922,122; b5_GRAPHS gather 1,993,376 vs 1,474,931).
  nsys (small fp16 flash beam5, cold): `gather_rows_kernel` 176 launches, 370.0 -> 7.28 us
  avg, 11.8% -> 0.3% of GPU time.

## Padding tax and the `CT2_CUDA_GRAPHS_TIERS` stride (measure only)

whisper-small beam5 std:0, `b = (pad@448 - pad@128) / 28,800` extra slot-steps (90 steps x
320 slots):

| tree | pad@128 | pad@448 | b |
| --- | --- | --- | --- |
| base | 201.6 ms | 233.6 ms | **1.11 us/slot/step** |
| stage 2 | 197.8 ms | 209.3 ms | **0.40 us/slot/step** |

That is the ~0.4 us round 5 predicted. With the round-5 capture cost K ~ 1.9 ms, the
model's optimal stride sqrt(2K/b) moves from **~58 slots to ~98 slots**. The default
recommendation stays `+64` in this round; `+96` or `+128` should be measured on the
round-5 G-long workloads (W_dense, W_prev, beam5 and greedy) before it changes. Two things
temper the expected gain: the model's cost curve is flat near the optimum (a wider stride
trades one crossing, K ~ 1.9 ms, against padding that now costs 0.40 us/slot/step), and on
dense:1 the prefix gather still costs 15 ms per decode because `L` itself is long there
(75 us per launch), which no stride changes.

## Honest caveats

(The review items below were fixed after the round; see "Review items addressed after
the round".)

- **The residual <= 1% on host-bound flash-off cells** (small int8 beam1) is not explained
  beyond "library-level": it appears with the kill switch and on paths that never reorder.
  `perf` needs sudo on the box (`perf_event_paranoid = 4`) and `cuobjdump` is not
  installed, so the code-layout hypothesis is unverified.
- **Unit-test coverage gaps found in review.**
  - `TransformerDecoderPrefixTest` claims that under `CT2_CUDA_PAD_KV=1` the CUDA run takes
    the padded-attention path, but the test puts `memory_lengths` into the state and
    `TransformerDecoder::decode` enables padded KV only when it is absent, so the only reader
    of the MHA tail is never exercised by a unit test. The keep_tail mutation fails only
    because the test compares shadow bytes. Tail-zero logit parity is covered by the
    end-to-end G1/G2 runs only; no unit variant uses a flash model.
  - No test proves that the MHA keep_tail steady path actually copies segments: the
    correctly paired steady MHA slots are not poisoned before call 2 of
    `GatherBatchSegments`, and `prefix_descriptors > 0` counts descriptors returned by
    `reorder_segments`, not ones `use_row_segments` accepted. A regression to full-row
    copies would pass every test; only the G3 nsys numbers would catch it.
- **keep_tail assumes uniform row tails.** On a steady step the segmented copy leaves row i's
  tail as that buffer's own previous tail, where a full-row gather would write the tail of
  source row `indices[i]`. They agree only because every MHA tail is zero
  (`append_to_cache` zeroes on growth). `GatherShadows`' "tails agree by induction" comment
  is stronger than what holds; a future writer past `L` (padded kernel, `copy_2d_indirect`)
  would silently get different bytes. This should be written into the `GatherRowSegments` /
  `GatherShadows` contract or guarded by a debug check.
- **CPU expand-after-first-step path.** Shadows are cleared only in
  `Decoder::replicate_state`. CPU beam search (`expand_after_first_step`) skips it, so the
  previous decode's `last_src`/`last_dst` survive, and if malloc returns the same address
  with the same shape the first step counts as steady and keeps stale tail bytes. Reachable
  only through the `set_prefix_reorder(true)` test hook (CPU default is prefix off), and CPU
  never reads the tail, so logits are unaffected. CUDA always calls `replicate_state` first.
- **`CT2_CUDA_GATHER_PREFIX` parsing.** It is default-on, but `read_bool_from_env` treats
  only `1`/`true`/`TRUE` as true, so `on`, `yes`, `True` or an empty export silently turn
  the feature **off**; the value is read once per process. Consistent with the codebase's
  other flags, but inverted from what a kill switch suggests.
- The stage-2 vs stage-1 G4 comparison is not paired (the stage-1 opt-in walls are from
  earlier in the session); the paired comparison is stage 2 vs the fresh base pass.
- One box session, within-session pairs only; absolute numbers are not comparable across
  sessions.

## Review items addressed after the round

All five review items above are fixed on this branch; none changes decoding output. Local
CPU suite 242 passed / 2 skipped / 1 known failure. The CUDA half (the `segmented`
increment in the fused path, `TransformerDecoderPrefixTest` under `CT2_CUDA_PAD_KV=1` /
`CT2_CUDA_GRAPHS=1`) is not yet run on a GPU.

| item | commit | change |
| --- | --- | --- |
| kill-switch parsing | `39d8ceb` | `CT2_CUDA_GATHER_PREFIX` is off only for `0`/`false` (any case); unset, empty, `on`, `yes` keep it on |
| CPU expand-after-first-step shadows | `e5c64c5` | `Decoder::reset_reorder_shadows()` at the start of every `BeamSearch::search` |
| uniform-tail requirement | `bf11596` | stated in the `GatherRowSegments::keep_tail` / `GatherShadows` / `reorder_segments` comments (no debug check) |
| steady MHA copy unproven | `3b0095c` | `GatherShadows::segmented` counts accepted segmented entries; `GatherBatchSegments` poisons the paired steady MHA shadows and requires the poison to survive. Forcing keep_tail to full rows now fails the test |
| prefix test never padded | `33c4516` | the test state has no `memory_lengths`, so padded attention can run; the prefix run must have `segmented > 0` on every device, and on CUDA with PAD_KV/GRAPHS set both runs must have padded steps (otherwise neither may) |

Still open: no deterministic regression test for the stale-shadow case (it needs the
allocator to return the same address). Under `CT2_CUDA_GRAPHS=1` the prefix test
allocates its logits inside the step loop, so a capture is expected to abort and every
step to run eager padded attention; graph replay is not covered by this unit test (read
from the code, not run).

## Round-7 levers

1. **Re-tune the tiers stride** to `+96` / `+128` against `+64` on the round-5 G-long
   workloads (prediction: optimum ~98 slots at b = 0.40 us).
2. Fix the review items: done, see "Review items addressed after the round" above.
3. **Remaining gather cost scales with L** (dense:1 tiers still 15 ms/decode, 75 us per
   launch at L ~ 200). Removing the copy entirely needs beam-indirect attention (read K/V
   through the parent-index chain) or a paged cache, which touches every attention kernel
   and the graph-captured shapes.
4. With the reorder no longer dominant, the next measured items in graphs mode are the
   padded attention itself (b = 0.40 us/slot/step is now mostly the masked attention over
   the spare capacity) and the per-crossing capture cost K.
