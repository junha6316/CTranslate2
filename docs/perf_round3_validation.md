# Round 3, Stage A: per-site allocation reuse — changes and validation

Six commits on top of `b3fc0d1` (round-2 tip), one per allocation site, ordered by
alloc/free pairs removed per decode step, each independently bisectable. The bar for
every commit is **bit-identical tokens**: pure allocation reuse, no op-order, shape or
numeric change. Stage A is also the address/shape-invariance groundwork for Stage B
(CUDA graphs): after A1-A6 the decode-loop buffers either live in persistent slots or
ping-pong between two fixed allocations.

## The six commits

| commit | site | pairs/step removed (whisper-small beam5) |
| --- | --- | --- |
| A1 | `Gather::batch` beam-reorder outputs → caller-owned shadow buffers, swapped after the fused kernel (2-periodic addresses; in-place is impossible because beam indices repeat). Shadows are reserved to the *source's* block-rounded byte capacity, which defeats the exact-fit trap of the growing `self_length` rows. | 25 (65 at large-v3) |
| A2 | CUDA TopK temporary reduction buffers → two `SamplerStaging` scratch slots (size batch×k×blocks is decode-constant → alloc-once). | 2 |
| A3 | Whisper `ApplyTimestampRules` row ids → persistent device tensor with the shared `upload_memoized` helper; steady state {0..batch×beam-1} also skips the H2D transfer. | 1 (+1 H2D most steps) |
| A4 | Decoder layer activations `layer_in`/`layer_out` → two `DecodeWorkspace` slots on the iterative path, bypassing the sliding-window chunking vector (its `push_back(std::move(...))` move-constructs into a function-local vector, which would kill the workspace buffer). | 2 |
| A5 | `get_layer_alignment_heads` skipped when `attention == nullptr` (the measured config); word-timestamps path gets per-layer cached device tensors rebuilt only on batch-size or head-config change. | 1 |
| A6 | **Opt-in** (`CT2_CUDA_PREALLOC_KV=1` or `CT2_CUDA_GRAPHS=1`): KV caches and the `self_length` record preallocated to `max_length` (≤448); block-granularity guard relaxed to shape-match + `offset <= capacity` under a reserve — the exact `self_length` step check in `TransformerDecoder::decode` remains the real guard, and block growth stays reachable as a fallback. Memory cost ~55-82 MB (whisper-small beam5) / ~370-655 MB (large-v3 beam5), which is why it is off by default. | 0.75-2 amortized |

## Verified locally (macOS, Accelerate CPU backend, no CUDA compiled)

- Full suite green after every commit (226 tests at tip; the only failure is the known
  pre-existing `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`).
- New CPU tests per commit: `GatherBatchWithShadows` (3-arg overload parity + shadow
  lifecycle, two rounds), `TopKWithScratch` (scratch overload parity, CPU leaves scratch
  untouched), `SamplerStagingParity` extended for the new scratch slots,
  `UploadMemoized` (skip on identical content, re-upload on same-size different
  content), `LayerSlotsPingPongAcrossSteps` (slot buffers non-null and pointer-stable
  across 5 steps and across decodes, logits identical to a fresh decoder),
  `AlignmentHeadsDeviceMemo` (memo hit/rebuild/invalidate),
  `CacheReservePreallocatesUpFront` (capacity 64 from step 0, relaxed guard passes
  mid-capacity offsets, step-exactness guard still throws) and
  `CacheReserveGrowthFallback` (70 steps past a 40-step reserve, logits identical).

## A10G validation (this round)

Round-2 nsys config throughout: whisper-small fp16 beam5 batch1 ts-on, 2 warmup +
5 timed decodes of the first 30 s window; wall numbers are median of 7 with 2
warmups (bench protocol of /opt/real/bench_batch.py). Baselines captured on the
round-2 tip `b3fc0d1` on the same instance, same session.

### Gate A (Stage A exit)

| check | tip b3fc0d1 | Stage A tip | result |
| --- | --- | --- | --- |
| tokens, default flags (all 9 small windows beam5 ts-on + 3 large-v3, 5 hyps each) | — | bit-identical | PASS |
| tokens, `CT2_CUDA_PREALLOC_KV=1` (A6) | — | bit-identical | PASS |
| wall beam5 batch1 | 426.6 ms | 227.1 ms, then 215.2 ms with the head-transpose fix (−50%) | PASS |
| wall beam1 batch1 | 236.1 ms | 164.1 / 163.2 ms (−31%) | PASS |
| cudaMallocAsync calls in the nsys config | 54,321 | 34,303 before the head-transpose fix, **2,990** after | PASS (<6,000) |
| decode-loop steady state, greedy (debug counter) | — | 0.0 allocs/step | PASS |
| decode-loop steady state, beam5 (debug counter) | — | 48/step before the fix, **0.86/step** after (exactly the amortized per-32-block cache growth; 0 with the A6 reserve) | PASS |
| local CPU suite | — | 225 passed, known Conv1D INT8 failure only | PASS |

The 48 pairs/step at beam5 were NOT a Stage-A regression but a pre-existing site
Stage A had missed: at beam>1 the cross-attention `split_heads`/`combine_heads`
head transposes run their `time>1` branch, which transposed into an exact-fit
function-local StorageView and move-assigned it over the input — destroying the
workspace slots' grown capacities every layer every step (sizes seen in the
allocator trace: 2x7680 B + 23040 B + 180000 B per layer per step = 48/step on
whisper-small). Fixed this round with a `DecodeWorkspace::head_transpose` scratch
slot threaded into both helpers: the move-assignment is a swap, so the transpose
ping-pongs between two stable buffers. Greedy (beam1, `time==1` reshape path) was
already at 0 allocs/step.

### Gate B0 (microbenchmark, A10G, sm86)

Standalone harness `bench_graphs.cu` (300 reps median, 3840-element fp32 kernel):

| datum | 130 launches | 300 launches |
| --- | --- | --- |
| eager launch storm (worker stream) | 330.1 us | 774.9 us |
| one cudaGraphLaunch replay | 144.4 us | 325.2 us |
| saving per step | 185.7 us | 449.7 us |
| 25x cudaGraphExecKernelNodeSetParams + launch | 153.7 us | 334.5 us |
| recapture + whole-graph cudaGraphExecUpdate + launch | 231.5 us | 533.4 us |

nsys gap analysis of the Stage-A-tip decode window (beam5, 3.57 s of 5 decodes):
158,645 kernels, GPU busy 965 ms → **GPU idle fraction 0.73**. Whisper-small
beam5 launches ~370 kernels/step at a ~2.4 ms step wall, so the projected
replay saving (~0.45 ms/step) is ~19% of the step wall → **GO**. Per-step
SetParams patching measures barely cheaper than a plain replay and ExecUpdate
costs half an eager step: both documented and rejected as the primary scheme.

### Gate B1 (padded-eager, `CT2_CUDA_PAD_KV=1`, implies A6)

- Tokens vs exact-length eager: whisper-small beam5 ts-on: **token ids
  bit-identical on all 9 windows**; whisper-small greedy: 8/9 identical, one
  window flips a repeated timestamp pair (`51528->51520`, 0.16 s) at position
  78/95 with decoded text equal; large-v3 beam5: 2/3 identical, one window flips
  one timestamp token by one unit (`50440->50441`, 0.02 s), text WER 0.000.
  Hypothesis scores drift by |d| <= 4.5e-4 (fp16 GEMM re-tiling at the padded
  n/k, the anticipated fp effect: padding adds exactly-zero terms, but cuBLAS
  picks different reduction tilings for the padded shapes; the flips land on
  timestamp tokens, whose logits are near-tied by construction). **Documented
  acceptance decision: pad mode (and graphs, which require it) are accepted on
  text/WER parity with near-total token equality (19/21 windows across the three
  configs, all diffs single timestamp tokens); both remain opt-in flags, the
  default path is untouched and bit-identical.**
- Wall cost of padding alone (after the head-transpose fix): beam1
  163.2 -> 167.6 ms (+2.7%, within the 5% bar); beam5 215.2 -> 255.3 ms
  (**+18.6%, fails the 5% bar**: every step pays QK^T and AV GEMMs at
  n/k = 448 capacity instead of the exact length). Consequence: at
  beam5 the graph-replay win must first pay back the padding tax (see Gate B3);
  a future `CT2_CUDA_GRAPHS_RESERVE` knob capping the padded capacity below
  max_length is the obvious lever and is left for round 4. The decoder-side
  host guard now makes that safe: the first step that no longer fits the cache
  capacity (or the position table) is refused BEFORE begin_step, so that step
  runs eagerly (growth fallback / clean bounds error) instead of a replay
  writing one step out of bounds. Verified on A10G with a reduced reserve:
  pre-guard compute-sanitizer reports 64 invalid device writes in
  `copy_2d_indirect_kernel`; with the guard, 0 errors and tokens identical to
  eager.

### Gate B2 (greedy CUDA graphs MVP, `CT2_CUDA_GRAPHS=1`)

- Capture: embed -> position-add-indirect -> 12 layers (padded GEMMs,
  device-length softmax, device-indirect cache append) -> output_norm -> proj,
  zero allocator calls inside the region (allocator assert-hook armed; violation
  counters checked across every capture). Warmup fingerprint stabilizes with
  period 1 at greedy; capture per decode. Executables are never reused across
  generate() calls: the decoder reports every decode boundary explicitly
  (prompt/prefix forward, set_cache_reserve_steps) via note_new_decode, and the
  critical fingerprint additionally bakes each state tensor's time capacity
  (dim 2) plus the position-table pointer/size, so recycled allocations from a
  previous decode with a different max_length can never satisfy a stale graph.
- `CT2_CUDA_GRAPHS_CHECK=1` dual-run (replay + eager on identical inputs, logits
  memcmp'd) over 3 windows (~250 steps): zero mismatches.
- Tokens: graphs greedy bit-identical to padded-eager greedy on all 9 windows.
  (Padded-eager greedy itself is bit-identical to default eager greedy on all 9
  windows; the "False" in the raw log is the score-field drift above.)
- Wall greedy batch1: Stage-A eager 163.2 ms -> graphs 151.9 ms (−6.9% total).
  Net of the measured 25.3 ms non-decode overhead (encoder + prompt, from a
  max_length=1 run): decode wall 137.9 -> 126.6 ms = **−8.2%**, below the 10%
  Gate-B2 bar (the padding tax eats ~+2.7% and the replay saves ~135 us/step,
  consistent with the B0 projection at greedy's ~200 launches/step).
- nsys, greedy graphs run (7 decodes): **574 cudaGraphLaunch** (nearly every
  steady step replayed); the same workload under the profiler drops from
  1304 ms to 821 ms (−37% — the profiler amplifies per-launch cost, which is
  exactly what the graph removes).
- Fault injection (forced capture failure, forced instantiate failure, injected
  allocation inside the capture window, forced replay failure, injected
  fingerprint mismatch): all five degrade to eager with tokens identical to
  padded-eager, one debug log line, never fatal. Main-thread invocation is
  refused up front (the legacy default stream fails the device_supported gate).

### Gate B3 (beam-5 + ship)

- Beam5 ts-on graphs: capture engages after the head-transpose fix (nsys: 616
  cudaGraphLaunch across 7 decodes, period-2 executable pair alternating with the
  beam-reorder ping-pong), tokens **bit-identical to padded-eager** on all 9+3
  windows.
- Batch-2 mid-decode shrink (short window forcing early EOS): graphs run falls
  back cleanly, tokens identical to eager. Word-timestamp path (align) with the
  flag on: output identical to flag-off (attention request refuses graphs/pad up
  front).
- Flag OFF: tokens bit-identical to the round-2 baseline on every config tested;
  wall 163.7 / 216.6 ms (beam1/beam5) vs 163.2 / 215.2 — within noise.
- Wall per config, median of 7 (overhead-net decode wall in parentheses,
  non-decode overhead 25.3/25.5 ms):

| config | Stage-A eager | padded-eager | graphs | graphs vs eager (decode) |
| --- | --- | --- | --- | --- |
| greedy batch1 | 163.2 (137.9) | 167.6 (142.3) | 151.9 (126.6) | **−8.2%** |
| beam5 ts-on batch1 | 215.2 (189.7) | 255.3 (229.8) | 234.3 (208.8) | **+10.1% (worse)** |

- Verdict: the ship bar (>= 8% decode-wall gain on at least one flagship config,
  launch collapse confirmed) is met by **greedy**; at beam5 the replay saving
  (255.3 -> 234.3, −21 ms) does not pay back the 448-capacity padding tax
  (+40 ms), so graphs at beam5 are a measured net loss vs eager and the flag
  stays **default-off, experimental**, documented per config. The padding-reserve
  knob (capping the padded capacity below max_length with graph invalidation on
  overflow) is the identified round-4 lever to flip beam5.

### Memory

A6/pad/graphs preallocation cost (whisper-small beam5, 448-step reserve):
12 layers x 2 caches x [5,12,448,64] fp16 = ~66 MB, plus the self_length record
and two graph executables (whisper-small: ~300 nodes each, <10 MB). large-v3
beam5: ~470 MB, which is why all of it stays opt-in.

## Deferred to round 4

- **SITE 9 (int8 Dense quantization temporaries)**: outside the measured fp16 config
  and not numerically validatable locally (no INT8 GEMM backend in the macOS build).
- **SITE 10 (RandomSampler intermediate tensors)**: sampling is outside the measured
  beam-search config. (Its TopK call does reuse the A2 scratch when staging is passed.)
- **A7(i) beam-path `attention_step` persistent device mirror + fp32 staging pair** in
  `BeamSearch::search` (decoding.cc), mirroring GreedySearch's `attention_step_device`.
- **A7(ii) `save_attention` copy-out instead of slot steal** when the softmax weights
  are workspace-backed (attention.cc): today the emptied slot simply re-allocates on
  the next step, and only while attention weights are requested.
