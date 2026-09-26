# Removing the per-step KV cache copy

Design note, written before the code. Target: `src/layers/attention.cc:537-546`.

## The problem

Every decoding step, in every layer, the self-attention cache is rebuilt:

```cpp
const ops::Concat concat_op(_cache_time_dim);
tmp = std::move(*cached_keys);
concat_op({&tmp, &keys_proj}, *cached_keys);
```

`ops::Concat` allocates a new tensor and copies the whole cache into it, then copies the
one new step after it. With 12 layers and both K and V that is 24 full cache copies per
step, so the copied volume grows as O(T^2). At step 100 of a `whisper-small` beam-5
decode one step moves about 18 MB.

From the Nsight Systems trace in `gpu_bottlenecks.md` (A10G, 10 beam-5 decodes of 150 s
audio):

| | |
| --- | --- |
| `Concat`/`Split` kernels | 108,264 instances, 230.6 ms GPU |
| `cudaMallocAsync` + `cudaFreeAsync` | 281,121 each, 849 ms host, ~19% of CUDA API time |
| host CUDA API total | 4.387 s, against 2.911 s of GPU kernel time |

The decode is launch-bound and allocation-heavy, and this loop is a large contributor to
both.

## Why it is not a one-line fix

The cache is laid out `[batch, heads, time, d_head]`. Time is the *middle* dimension, so
appending a step is not a tail write: the per-`(batch, head)` blocks all shift. That is
why `Concat` is there at all.

Pre-allocating the cache means the tensor's time dimension becomes a *capacity*, not a
length, which breaks the invariant `shape == logical content` that the rest of the code
relies on:

- `dot_product_attention` reads the key length from `keys.dim(2)`.
- the sliding-window `Slide` reads `cached_keys->shape()[2]`.
- beam reordering gathers the whole tensor along dim 0.
- relative position and bias helpers take `key_length` from the keys tensor.

## The insight that makes it cheap

Both attention matmuls go through `primitives<D>::gemm_batch_strided`, which already
takes the leading dimension and the *batch stride* as separate arguments
(`src/ops/matmul.cc:66-75`). A cache of capacity `C` holding `t` valid steps is exactly a
batched GEMM with `n = t` (or `k = t`) and batch stride `C * d_head`:

| | queries x keys^T | attn x values |
| --- | --- | --- |
| b | keys `[B*H, C, D]`, `trans_b` | values `[B*H, C, D]` |
| overridden | `n = t` | `k = t` |
| `ldb` | `D` — unchanged | `D` — unchanged |
| batch stride | `C*D` instead of `t*D` | `C*D` instead of `t*D` |

The GEMM simply never reads rows `[t, C)`. **No masking is needed, and the score tensor
that comes out is still exactly `[B, H, q, t]`** — the same shape as today, so `SoftMax`,
the length mask, `save_attention` and the DTW alignment path all see unchanged inputs.

Attention never reads the tail, but other code does: beam reordering, `repeat_batch` and
`get_batch_state` gather the whole tensor, and `tests/model_test.cc:161-164` compares every
decoder state tensor between a 5x1 decode and a 3+2 chunked decode. So the grown buffer is
zeroed on allocation, and the capacity is a deterministic `round_up(length, 32)`, which
makes those two states bit-identical.

## Design

### 1. Capacity, grown in blocks

`cached_keys` / `cached_values` keep shape `[B, H, C, D]` with `C = round_up(t, BLOCK)`,
`BLOCK = 32`. Growth reallocates and copies once every 32 steps instead of every step, so
the copied volume drops from `T^2/2` to about `T^2/64`, and the allocation pairs drop by
the same factor.

Block growth is deliberately chosen over allocating `max_decoding_length` up front:
beam reordering gathers the *whole* tensor every step, so capacity that is never used is
paid for on every gather. With `BLOCK = 32` the average waste is 16 time steps.

### 2. Where the logical length comes from

`MultiHeadAttention::operator()` already receives `offset`, and
`FlashMultiHeadAttention` already treats it as the current cache length
(`flash_attention.cc:101`). Under today's code the invariant is
`cached_keys->dim(_cache_time_dim) == offset` at every append.

This was checked, not assumed: a temporary `throw` on mismatch was compiled into the
current code and the whole CPU suite run against it (198 passed / 2 skipped / 1
pre-existing `Conv1DGroupNoBiasQuantized` failure, identical to the baseline). The
suite covers greedy, beam, prefix-forced and biased decoding.

The same equality cannot be the permanent guard. Once the dimension is a capacity it is
unsatisfiable, and a reused static prompt cache (`src/models/language_model.cc:189-216`)
legitimately re-enters with `offset < capacity`. What the append can still check is that
`offset` falls inside the last block, since the capacity is always
`round_up(length, 32)`: an offset a whole block or more behind — a reset step counter, a
skipped prompt forward — throws (`ModelTest.DecoderRejectsStaleStep`). **An offset that is
wrong by only a few steps is not caught.** The cache has no way to know: the length is not
stored anywhere, and storing it in `DecoderState` would mean reading an int back from the
device every step, which is the cost this work exists to remove.

What makes that acceptable is that `offset` is not a cache-private value. The decoder
derives it from the same step counter it feeds to the position encoder
(`transformer.cc:649`) and to the rotary embeddings (`attention.cc`), so an off-by-a-few
`offset` already produces wrong positions today, before it reaches the cache.

### 3. New pieces

**`primitives<D>::copy_2d`** plus two file-static helpers in `attention.cc`
(`copy_cache_steps`, `append_to_cache`) — writes `src [B, H, n, D]` into
`cache [B, H, C, D]` at time `offset`. `B*H` rows of `n*D` contiguous elements,
destination row pitch `C*D`. CPU: a loop of `std::copy`. CUDA: one block per row. The same
helper does the copy when the cache grows (`offset = 0` into the larger buffer), so
`ops::Concat` disappears from this path entirely.

The CUDA side started as a single `cudaMemcpy2DAsync`, which looked free because it is not
a kernel launch. It is not free: the trace measured it at 7.34 us on the host against
4.77 us for `cudaLaunchKernel`, over 31,824 calls. The kernel is worth about 1% end to end
over the memcpy, measured as a paired run.

Growing always builds a new tensor. Resizing the cache in place would be wrong:
`StorageView::reserve` keeps the existing buffer whenever it is large enough
(`storage_view.cc:166`), and every row would then sit at the wrong offset for the new time
pitch.

Each growth also zeroes the new buffer before copying, for the reason above. That is one
`B*H*C*D` memset per 32 steps, which the cost estimate below has to account for.

**`ops::MatMul` overload** taking `dim_t b_rows`, which overrides `b.dim(-2)` for the
GEMM's `n` (when `trans_b`) or `k` (when not) while keeping `ldb` and the batch stride
derived from the real shape. About 20 lines in `matmul.cc`, no new kernels, works on
both devices because `gemm_batch_strided` already exposes what is needed. Three details
that are easy to get wrong:

- `b_rows` is an argument of `operator()`, not constructor state. The same `keys_matmul`
  and `values_matmul` objects are reused for the relative-position products
  (`attention.cc`, `add_relative_representations`), whose `b` is fully populated; a member
  would silently truncate those for every relative-position model.
- `b_batch_size` has to become `b.size() / (b.dim(-2) * b.dim(-1))`, otherwise the
  `a_batch_size != b_batch_size` check throws for every `t != C`.
- overriding `k_b` is what keeps the `k_a != k_b` check quiet on the second matmul;
  `ldb` resolves to `b.dim(-1)` in both branches and must stay untouched, which is also
  what keeps the ruy backend from rejecting the call outright.

**`dot_product_attention`** gains a `keys_length` parameter (0 = use `keys.dim(-2)`,
the behaviour for encoder and cross attention). It is used for the two matmuls and
everywhere `key_length` is currently read from the keys tensor, so relative positions,
relative attention bias and ALiBi stay correct.

### 4. Scope limits

| path | treatment |
| --- | --- |
| decoder self-attention, `_sliding_window == 0`, `[B, H, T, D]` layout | new path |
| `_sliding_window > 0` | unchanged — the window `Slide` assumes the cache starts at time 0 |
| `_merge_time_and_head_dims` (multi-query) | unchanged — the cache is 3-D `[B, T, D]`, a different layout |
| `MultiHeadAttention::forward_merged` | unchanged |
| `FlashMultiHeadAttention` | unchanged, it has its own pre-allocation |
| cross attention, encoder | unchanged, written once, capacity == length |

## What does not change

- `DecoderState` stays `map<string, StorageView>`; no new entries, no new gather rules.
- Beam reordering, `repeat_batch`, `resize(0, n)` and `get_batch_state` all act on dim 0
  and are indifferent to the time dimension. They do copy the spare capacity, which is why
  it is zeroed.
- The attention score and probability tensors keep their exact current shapes, so
  `save_attention`, the length masks and Whisper's DTW alignment are untouched. Decoder
  self-attention never fills `attention` at all (`transformer.cc:241, 315, 339` pass
  `nullptr`); Whisper's alignment runs the full-sequence path with no cache.
- `FlashMultiHeadAttention::_encoder_fallback` is a `MultiHeadAttention` that receives the
  same cache pointers, but it is only built when `!is_decoder`
  (`flash_attention.cc:15-16`), so it never sees a decoder self-attention cache. If that
  ever changes, the flash layout (`_cache_time_dim == 1`) and this one would write the same
  state tensor with different layouts.

## Expected gain, and how it will be attributed

Predicted, not measured: removing about 61,000 of the 281,000 allocate/free pairs
(~185 ms host) and most of the 230 ms of `Concat` GPU time, against a 4.387 s host /
2.911 s GPU budget. That is roughly **3-8% end to end**, with the host side dominating
because the decode is launch-bound.

Against that, the new costs: one `B*H*C*D` memset per growth, and one copy kernel per
layer per step in place of the `Concat` kernel. The launch count is therefore unchanged;
what goes away is the allocation pair behind every `Concat` and the whole-cache copy.

It was measured with `/opt/bench.py` against upstream v4.8.2, warmup 2 plus 7
measured runs, median, noise band +/-3%. Control axes:

- **flash attention on** — the decoder self-attention then goes through
  `FlashMultiHeadAttention`, which this change does not touch. This is the control that
  worked.
- short audio at beam 1 and beam 5 were also meant as controls, on the assumption that the
  gain is the quadratic copy. They are not: see the measurements below.

## Risks

1. **`offset` wrong on some path.** Every in-tree caller was traced and the invariant
   holds; the runtime check turns a gross error into a loud failure. An error smaller than
   one block stays silent in the cache, but not in the position encoding, which reads the
   same counter.
2. **Beam gather copies capacity.** Bounded to 32 extra time steps by the block size;
   this is the reason the design does not pre-allocate the full decoding length.
3. **`gemm_batch_strided` with a B batch stride larger than `k*n`.** Legal everywhere:
   cuBLAS documents the strides as plain element offsets with no upper bound, and
   `src/ops/conv1d_gpu.cu:104-111` already passes oversized `strideb`/`stridec` for grouped
   convolutions. The CPU backends either forward the strides to `cblas_*_batch_strided` or
   fall into a `parallel_for` over `b + i*strideb`. Untested until now on the CPU side,
   hence the dedicated unit test.

## Validation

Done locally (macOS arm64, CPU, Accelerate backend, `WITH_CUDA=OFF`):

| check | result |
| --- | --- |
| CPU suite | 202 passed / 2 skipped / 1 failed — the failure is the pre-existing `Conv1DGroupNoBiasQuantized`, identical to the baseline before this change |
| is the new path actually taken | yes: a temporary `throw` in it fails 57 of the tests, covering greedy, beam, prefix, biased and batched decoding |
| growth logic | the whole suite passes with `kv_cache_block` forced to 1, 2, 3 and 7, which makes the cache grow on almost every step and produces non-power-of-two capacities |
| 100-step and 150-step decodes | byte-identical output, scores included, against a build of the same tree without this change, at beam 1 and beam 5 |
| oversized B batch stride | new `MatMulSpareRows` test, with the spare rows filled with NaN so that reading them would poison the output; covers `trans_b` and not, and both the batched and the single-matrix GEMM paths |
| long decode locked down | new `LongDecodingKeepsCachedAttentionExact` test pins a 100-step greedy and beam-4 output |
| stale step rejected | new `ModelTest.DecoderRejectsStaleStep` |
| step-by-step vs chunked state | `ModelTest.DecoderIterativeSequence` compares every state tensor element by element, spare capacity included — this is what forces the zeroing |

Side observation, not a gate: the 150-step CPU decode in the driver above runs in 0.53-0.55 s
against 0.58-0.60 s before the change, five runs each. CPU is not the target and there is no
control axis here, so this is indicative only.

On the GPU box (`i-074f30405a215073e`, A10G, CUDA 12.8):

| gate | result |
| --- | --- |
| token equality | 32/32 exact against upstream v4.8.2 **and** against the branch tip, scores identical too |
| CUDA suite | 178 passed / 3 skipped / 0 failed; the 3 extra over the previous 175 are this change's own test |
| decoded length | identical in all 64 benchmark rows |

Performance, paired against the branch tip rebuilt and measured **in the same session**
(the older `bench_gate.json` is not a usable baseline: the control moves with it, so it
predates the timestamp-gate commit):

| axis | n | median | range |
| --- | --- | --- | --- |
| flash off — the changed path | 32 | **-5.21%** | -7.79% to -1.63% |
| flash on — control, has its own preallocation | 32 | -0.18% | -3.00% to +0.44% |

An earlier build of the same change using `cudaMemcpy2DAsync` instead of the copy kernel
measured -3.32% on the same axis, with the control at -0.26%. The two comparisons against
the tip differ by more than the direct kernel-versus-memcpy run does (-1.02%), so read the
gain as **3 to 5%**, not as a single number.

What the trace says about the mechanism, same workload as `gpu_bottlenecks.md`:

| | branch tip | this change |
| --- | --- | --- |
| `cudaMallocAsync`/`cudaFreeAsync` | 281,121 each | 253,223 each |
| host CUDA API total | 4.387 s | 4.039 s |

The saving is the allocation pair behind every `Concat`, not the copied volume. That is why
the short-audio rows gain about as much as the long ones: the allocation churn is per step,
not per cache length. The "beam 1 with a short audio" axis in the design above is therefore
**not** a control for this change; only flash-on is.

Traced by hand but not run anywhere in the C++ suite:

- static prompt cache (`language_model.cc:207`, `DecoderStateCache::save`/`get`) — reuses a
  capacity-backed state across calls; lives only in
  `python/tests/test_transformers.py:609-647`, which downloads a Hugging Face model.
- `include_prompt_in_result = false` (`language_model.cc:233`) — the only production path
  that appends more than one step at a non-zero offset.
- Whisper's `forward_prompt` (`whisper.cc:282-294`).

All three keep the invariant: `check_prompts` rejects prompts of differing length, and the
generator trims prompts to a common length, so no padding makes `step` and the cache
disagree. The Whisper token comparison on the GPU box exercises the third one.

## Capacity tiers under CUDA graphs (round 5)

The CUDA-graph decoder path (`CT2_CUDA_GRAPHS=1`, opt-in) replays captured executables
that bake the cache capacity `C` into every GEMM and softmax shape, so it preallocates
the caches for `CT2_CUDA_GRAPHS_RESERVE` steps. Round 4 capped that reserve (128 for
beam5) to cut the padding tax, at a price: a decode that outgrows the cap hits the host
guard in `TransformerDecoder::decode`, disables replays for the rest of the decode and
runs an eager tail with 32-block growth. Long windows (dense speech, a previous-text
prompt) lose the replay saving exactly where they have the most steps left.

`CT2_CUDA_GRAPHS_TIERS` (see [environment variables](environment_variables.md)) turns
that crossing into a transition: the cache grows to the next capacity tier and the
graph runner re-captures at the new shapes. Default off; with the policy inactive the
guard takes today's branch and the path is bit-identical.

### The ladder

Tier 0 is the capacity the prompt forward allocates, `C0 = round_up(max(P, R), 32)`.
At a crossing with capacity `C`, `CacheTierPolicy::next(C)` names the next tier:
`min(top, C + S)` for a relative stride `+N` (`S = max(32, round_up(N, 32))`,
`top = round_up(max_length, 32)`), or the first listed tier above `C` for an explicit
list. With `R = 128` and `+64`: `P = 2` gives 128, 192, 256, 320, 384, 448; `P = 226`
(a 223-token previous-text prompt) gives 256, 320, 384, 448. A decode transitions at
most `ceil((top - C0) / S)` times.

### One crossing, in order

At the step `s` whose write would exceed the capacity (`C == s`):

1. The host guard sees a capacity-only overflow and a next tier `T`.
2. `DecoderGraphRunner::begin_tier_transition` destroys **both** executables on the
   host, before anything is freed. Replays of steps `s-1`/`s-2` still in flight
   complete normally; the driver defers the release of a launched executable. It
   refuses (and the decode keeps today's eager tail) when the decode is Disabled, a
   capture is active, or the batch changed since the decode's first capture.
3. `apply_cache_reserve(T)` raises the per-layer reserve **and**
   `_cache_reserve_steps`. Both matter: `append_to_cache` sizes the growth from the
   first, `update_length_record` rounds the `self_length` record from the second
   (bumping only the layers would leave the record regrowing in 32-step blocks inside
   the tier, a mid-tier reallocation; `CacheReserveLayerOnlyBumpRegrowsRecord` pins it).
4. Step `s` runs eager at `C = T`: every cache is reallocated to `T`, zeroed, the `s`
   old steps copied and step `s` written; the old buffers are freed stream-ordered
   after all earlier work. The record and the capacity-dependent workspace (attention
   scores) grow to `T` in the same step, and the beam reorder re-reserves its shadow
   buffers to `T` after the sampler.
5. Step `s+1` re-captures (fast re-entry: the period is known from the decode's first
   capture), and `s+2` captures the second slot for beam search. Both captures are
   allocation-free; replays resume at `s+2`/`s+3`, checked against the critical
   fingerprint as before.

Between steps 2 and 5 no executable exists; every live executable only references
post-crossing buffers; at most one executable pair is alive. The crossing step itself is
never replayed. Any capture, instantiate, launch, replay, fingerprint or allocation
failure after a transition disables replays for the rest of the decode (the tail then
runs padded-eager at `T` and grows in 32-blocks past it). A bump lives for one decode:
the next `set_cache_reserve_steps` or the next prompt/step-0 forward restores the base
reserve.

### Memory

The steady state is the new tier's caches plus their reorder shadows, bounded by what
the uncapped reserve (`R = max_length`) allocates up front since `T <= max_length`.
During the crossing step each layer briefly holds its old and new cache (the old one is
freed right after the copy), and the old shadows live until the reorder re-reserves
them at `T`, so the transient stays below the uncapped footprint as well.

Measurements (gates, wall time, memory, token parity): [perf_round5.md](perf_round5.md).
