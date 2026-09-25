# Round 2: allocation churn and per-step uploads — changes and validation plan

What the two round-2 commits change, what has been verified locally, and the exact
A10G session that will produce the numbers. **No GPU number in this file is a
measurement of this round**; the trace figures below are the targets from
[gpu_bottlenecks.md](gpu_bottlenecks.md), measured before these commits existed.

## The two commits

On top of `192aa5c` (`fix/review-refinements`):

| commit | change | trace numbers it targets |
| --- | --- | --- |
| `e85c04e` | `perf(decoding): reuse decoder layer temporaries across decode steps` — `TransformerDecoder` owns a `DecodeWorkspace` threaded through the decoder layers, attention (plain and flash) and FFN as an opt-in trailing parameter. Persistent slots replace the ~140–165 per-step locals (LN output, fused QKV, Split outputs, cross-attention temporaries, FFN inner activation); after the first step every use is resize-only because `StorageView::reserve` keeps the allocation when the byte size fits. Encoders and unthreaded paths pass `nullptr` and are unchanged. | Lever 2: 281,121 `cudaMallocAsync`/`cudaFreeAsync` pairs, 849 ms of host time (~19% of CUDA API time). The decoder-layer locals are the bulk of it: ~140–165 pairs per decode step. |
| `58c584e` | `perf(decoding): stage per-step uploads in persistent device buffers` — the search loops' remaining per-step device allocations become copies into loop-scoped persistent mirrors (`to_device_staged` for the beam/greedy transfer tensors, a caller-owned `SamplerStaging` for the GPU sampler, const-ref `beam_indices` instead of a by-value deep copy). `DisableTokens` gets caller-owned buffers with a memoized upload: when the host content matches the previous step — every steady-state Whisper step — the H2D copy of the ~88-id suppress list is skipped entirely. | The rest of lever 2 (~7–9 alloc/free pairs per step from the search loop) and the H2D half of lever 3: 7,988 host-to-device transfers, most of them the tiny `DisableTokens` flat-index arrays uploaded every step. The 2,548 D2H transfers (~30 bytes each) are intentionally untouched: the sampler's blocking result copies feed host bookkeeping. |

Both commits claim no numeric op, shape or op order changes, so decoding output
should be **bit-identical** to `192aa5c` on CPU and GPU. That is a stronger claim
than earlier rounds (the bias epilogue changed rounding); the A10G session must
hold it to zero token differences, not "word-level and benign".

## Verified locally (macOS, Accelerate CPU backend, no CUDA compiled)

- Full suite at `58c584e`: **219 tests, 216 passed, 2 skipped, 1 failed** —
  `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`, the known pre-existing
  build-configuration failure documented in
  [flash_attention_validation.md](flash_attention_validation.md) (no INT8 GEMM
  backend with `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON`).
- New CPU tests in `e85c04e` pin the workspace reuse hazards: translating twice
  with one instance, and a batch-size 3 → 1 → 5 sequence against fresh instances,
  with and without `return_attention`.
- New CPU tests in `58c584e` cover the `DisableTokens` index-upload path via a
  test-only `force_index_path` flag (parity with the direct-fill path, proof the
  memo skips the upload, re-upload on batch shrink), sampler staging parity, and
  the same-device pass-through.

## Not yet measured

- **Any GPU behavior at all.** CUDA sources are not compiled here; the
  cross-device staging branches and the flash-attention workspace path are
  compile-checked at most and runtime-unreachable locally.
- The predicted collapse of the alloc/free pairs. ~150–175 pairs per step out of
  the trace's 281,121 total is an estimate read off the source, not a count from
  a new trace.
- Wall-clock effect. Lever 2 was 849 ms of host time under nsys against a 4.0 s
  decode; nsys inflates host costs, so the un-profiled gain is unknown and may be
  modest. Batch 1 (most launch-bound) should gain the largest fraction.
- The flash path has **zero runtime coverage** so far: `FlashMultiHeadAttention`
  hoists its projections into the same workspace, and the KV-cache
  shallow-copy trap it dodges is exactly the kind of bug only a GPU run can catch.
  Flash-on runs are mandatory before merge, not optional.

## A10G validation plan

Same box and layout as the 2026-09-23 session ([results/2026-09-23/real/](../results/2026-09-23/real/)):
g5.2xlarge, CUDA 12.8, models under `/opt/models/faster-whisper-{small,large-v3}`,
audio under `/opt/audio/`, scripts copied to `/opt/real/`. Three builds installed
side by side in one session and run back to back, never numbers from different
sessions multiplied together:

- `up`: upstream v4.8.2
- `prev`: `192aa5c` (previous tip, `fix/review-refinements`)
- `head`: `58c584e` (this tip)

Each build:

```sh
cmake -S . -B build -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_ARCH_LIST=8.6 \
  -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd python && pip install -e . --force-reinstall --no-deps
```

Timing protocol as in [whisper_gpu_results.md](whisper_gpu_results.md): 2 warmups,
then the median of 7. `bench_batch.py` ships with `REPS, WARMUP = (5, 1)`; set
`REPS, WARMUP = (7, 2)` for this session so the protocols match.

### 1. Throughput, per build (reinstall between the three)

```sh
python /opt/real/bench_batch.py up   /opt/real/r3/batch_up.json
python /opt/real/bench_batch.py prev /opt/real/r3/batch_prev.json
python /opt/real/bench_batch.py head /opt/real/r3/batch_head.json
```

Covers small and large-v3, float16 and int8_float16, beam 1 and 5, timestamps
on/off, batch 1/4/8, and stores the token ids of every window. Compare
`ms_per_window` at the same batch only. `head` vs `prev` isolates this round;
`prev` vs `up` should reproduce the 2026-09-23 deltas as a sanity check that the
session is healthy.

Flash is off in `bench_batch.py`. Add the flash-on axis by re-running with
`ctranslate2.models.Whisper(..., flash_attention=True)` (one-line edit, save as
`batch_*_flash.json`) — this is the only throughput coverage the flash workspace
path gets.

### 2. End to end, per build

```sh
python /opt/real/e2e.py up   /opt/real/r3/e2e_up.json
python /opt/real/e2e.py prev /opt/real/r3/e2e_prev.json
python /opt/real/e2e.py head /opt/real/r3/e2e_head.json
```

faster-whisper sequential and batched-8 on the 203 s lecture. Known caveat from
the last session: large-v3 sequential text can differ run to run (temperature
fallback), so only stable text differences count.

### 3. The trace claim, directly

```sh
nsys profile -o /opt/real/r3/p_prev_small_1 python /opt/real/prof_batch.py small 1
nsys profile -o /opt/real/r3/p_head_small_1 python /opt/real/prof_batch.py small 1
nsys stats --report cuda_api_sum,cuda_gpu_mem_time_sum /opt/real/r3/p_head_small_1.nsys-rep
```

(`prof_batch.py`: fp16, beam 5, timestamps on, 2 warm-up + 5 timed runs; run the
`prev` profile before reinstalling `head`.) Pass criteria, against the
`50e1e882` column of whisper_gpu_results.md §4 (253,223 pairs, 3.568 s host API
time):

- `cudaMallocAsync`/`cudaFreeAsync` counts drop by an order of magnitude — the
  per-step pairs should collapse to first-step-only allocations. If the count is
  still proportional to decoded tokens, a slot is being stolen and re-allocated
  every step and the shallow-copy trap notes in `e85c04e` are the first place to look.
- H2D transfer count falls well below 7,988: the steady-state `DisableTokens`
  upload is memoized away.
- D2H stays ~unchanged (intentionally untouched).

Repeat both profiles with `flash_attention=True` (edit `prof_batch.py`'s
constructor) to confirm the flash workspace path frees nothing it should keep.

### 4. Output equivalence

`bench_batch.py` already stores `tok_ids` per configuration; diff them across the
three JSONs. Expectations:

- `head` vs `prev`: **identical everywhere**, all 48 configurations, both compute
  types, flash on and off. Any difference is a bug in this round, full stop.
- `head` vs `up`: same differences as `prev` vs `up` on 2026-09-23 (the known
  bias-epilogue float16 rounding, 9 of 48 configurations, one window each) and
  nothing new.

## Filed as

Results go to `results/<date>/r3/` with the raw JSONs and nsys logs, and the
summary tables get appended to [whisper_gpu_results.md](whisper_gpu_results.md)
in the same paired-session format.
