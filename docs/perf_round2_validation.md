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

---

## Measured results (2026-09-25, A10G)

The plan above was executed on 2026-09-25. Raw data is in
[`results/2026-09-25/`](../results/2026-09-25/).

### Session

- AWS g5.2xlarge (A10G, sm86), Deep Learning Base GPU AMI, CUDA 12.8,
  cuDNN 9.10.2 (matching the 2026-09-23 session), Nsight Systems 2026.1.3.
- Builds: `up` = v4.8.2 (`d44d2d0`), `prev` = `192aa5c`, `head` = `58c584e`
  (+docs `d59f20e`), all `WITH_CUDA=ON WITH_CUDNN=ON WITH_FLASH_ATTN=ON
  WITH_MKL=OFF WITH_RUY=ON OPENMP_RUNTIME=COMP CUDA_ARCH_LIST=8.6 Release`,
  installed side by side and run back to back. `bench_batch.py` at
  `REPS, WARMUP = (7, 2)`.
- **Audio substitution**: the original `physicsworks.wav` was lost with the old
  instance. This session uses a 203 s LibriVox reading (Art of War ch. 1–2,
  16 kHz mono) under the same filename, plus the same `jfk_x5.flac` recipe.
  Within-session pairs are unaffected; comparisons against 2026-09-23 absolute
  numbers are approximate.
- One environment trap worth recording: with cuDNN 9.26 (the AMI's apt default),
  any run **under nsys** fails at the first Conv1D with
  `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` while the same build runs fine
  outside nsys. Downgrading to cuDNN 9.10.2 fixed it; all reported numbers,
  benches and profiles, use 9.10.2.

### Output equivalence: PASS

`tok_ids` across all 48 configurations (2 models x 2 compute types x ts on/off
x beam 1/5 x batch 1/4/8): `head` vs `prev` **identical in all 48**, flash off
and flash on both. The bit-identical claim holds on GPU.

`head`/`prev` vs `up` differ in 4/48 (flash off; the bias-epilogue rounding, as
in round 1) and 22/48 (flash on — expected: upstream ignores encoder flash
attention, so the fixed encoder changes results). The flash-on cases where
upstream looks "faster" are not wins: e.g. large-v3 int8 ts-on beam 5 decodes
11 tokens on `up` against 35 on `head` — upstream's broken encoder path
produces degenerate short transcripts there.

### Throughput, `head` vs `prev` (the round-2 delta)

48 paired cases, flash off: **median -2.3%**, range -16.1% .. -0.4%.

| axis | median |
| --- | --- |
| batch 1 | **-4.7%** |
| batch 4 | -2.3% |
| batch 8 | -1.4% |
| small, batch 1 | **-7.5%** |
| large-v3, batch 1 | -2.1% |

Flash on, 48 cases: median -1.3% (range -10.0% .. 0.0%). Largest single case:
small / float16 / ts on / beam 1 / batch 1 at **-16.1%**. The gain concentrates
exactly where the trace predicted: launch-bound small-model batch-1 decoding,
and it thins out as the batch grows and the GPU stays busy.

Cumulative against upstream v4.8.2: median **-11.8%** flash off, **-23.6%**
flash on (the latter includes the encoder flash fix, and 22/48 flash-on cases
decode different tokens, so treat per-case flash-on deltas as indicative only).

### End to end (faster-whisper, 203 s audio), `head` vs `prev`

| case | prev | head | delta |
| --- | --- | --- | --- |
| small fp16 sequential | 1.83 s | 1.69 s (120x rt) | **-7.8%** |
| small int8 sequential | 2.61 s | 2.46 s (83x rt) | -6.0% |
| large-v3 fp16 sequential | 14.70 s | 12.92 s (16x rt) | -12.1% |
| large-v3 int8 sequential | 12.96 s | 9.40 s (22x rt) | -27.5% |
| batched8 (all four) | — | — | -0.6% .. -1.7% |

The large-v3 sequential rows carry the known temperature-fallback run-to-run
variance (`prev` measured *slower than upstream* on the int8 row, which is that
noise, not a regression) — take the small rows as the trustworthy sequential
signal.

### The trace claim (nsys, small fp16 beam 5 ts on batch 1, 5 decodes)

| metric | prev | head | delta |
| --- | --- | --- | --- |
| `cudaMallocAsync`/`cudaFreeAsync` pairs | 129,535 | 54,321 | **-58%** |
| alloc+free host time | 387 ms | 198 ms | -49% |
| `cudaMemcpyAsync` (H2D) | 6,823 | 5,297 | -1,526 |
| `cudaLaunchKernel` | 164,832 | 164,832 | **identical** |
| wall (5 runs, under nsys) | 2,444 ms | 2,079 ms | -15.0% |

Flash on: pairs 144,655 -> 69,357 (-52%), launches identical (141,984), wall
-13.2%. Identical launch counts are the mechanism check passing: no op changed,
only the host work between launches.

The plan's "order of magnitude" pass criterion was **not met**: -58%, not -90%.
The remaining ~54k pairs over ~1,274 steps (~42/step) are the paths round 2
deliberately left alone — beam-reorder gather outputs, TopK scratch, softmax
workspaces, and the logits pipeline — plus block-boundary cache growth. That
list is the natural round-3 worklist (or it dissolves entirely under CUDA
Graphs, which these two rounds have now made addressable: buffers and cache
addresses are stable across steady-state steps).
