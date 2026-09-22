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
destination row pitch `C*D`. CPU: a loop of `std::copy`. CUDA: one `cudaMemcpy2DAsync` on
the compute stream. The same helper does the copy when the cache grows (`offset = 0` into
the larger buffer), so `ops::Concat` disappears from this path entirely.

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

Against that, the new costs: one `B*H*C*D` memset per growth, and one `cudaMemcpy2DAsync`
per layer per step in place of the `Concat` kernel. Whether a strided 2-D copy beats a
small kernel launch here is **not measured** — `cudaMemcpy2DAsync` on device-to-device
memory usually lowers to a kernel anyway, so the saving to claim is the allocation churn
and the copied volume, not the launch.

It will be measured with `/opt/bench.py` against upstream v4.8.2, warmup 2 plus 7
measured runs, median, noise band +/-3%. Control axes:

- **encoder-only** and **beam 1 with a short audio** — no long cache accumulates, so
  neither should move. If they do, the gain is not this change.
- **beam 5 with a short audio** — the cache is allocated and zeroed but barely fills, so
  this axis carries the new memset cost without the removed `Concat` cost. It separates the
  two.

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

Still to run on the GPU box (`i-074f30405a215073e`, stopped):

1. `/opt/tok.py` token dump vs upstream, compared with `/opt/diff.py`: 32/32 exact.
2. `--gtest_filter='CUDA/*'`: 175 passed, 3 skipped, 0 failed. This is the first run of
   `primitives<CUDA>::copy_2d` and of cuBLAS with an oversized B batch stride from this
   code path.
3. `/opt/bench.py` with the control axis described above.

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
