# Batched and end-to-end Whisper measurements

A10G, 2026-09-23. `head` is `1221aff9`, `up` is upstream v4.8.2. Flash attention off.

## Scripts

- `bench_batch.py`: `Whisper.generate` over batches of 30 s windows (physicsworks.wav
  cut into 7 windows, then jfk_x5.flac into 2). Batch N decodes the first N windows,
  so compare builds at the same N, not N against N.
- `e2e.py`: faster-whisper 1.2.1 on physicsworks.wav (203 s). Sequential
  `WhisperModel.transcribe` with library defaults, and `BatchedInferencePipeline`
  with `batch_size=8`. The batched run uses fixed 30 s `clip_timestamps` instead of VAD,
  because onnxruntime is not installed on the box, so its chunk boundaries differ from
  the library default.
- `prof_batch.py`: nsys workload, fp16, beam 5, timestamps on, 2 warm-up + 5 runs.

## Results

ms per window, `head` (`up`), float16, beam 5:

| model | timestamps | batch 1 | batch 4 | batch 8 |
|---|---|---|---|---|
| small | on | 274.5 (503.0) | 95.6 (308.6) | 71.0 (292.9) |
| small | off | 270.3 (343.5) | 90.1 (107.3) | 67.7 (73.4) |
| large-v3 | on | 1023.7 (1250.3) | 400.2 (566.2) | 301.0 (482.6) |
| large-v3 | off | 906.6 (986.9) | 365.8 (387.2) | 282.4 (297.5) |

With timestamps on, upstream barely gains from batching: the timestamp rule syncs the
host once per row, so its cost grows with the batch. With timestamps off, the gap
between the builds shrinks as the batch grows.

End to end, times realtime (`head` / `up`):

| model | ctype | sequential | batched 8 |
|---|---|---|---|
| small | float16 | 102.5 / 58.8 | 335.5 / 320.3 |
| small | int8_float16 | 82.7 / 52.2 | 294.4 / 288.5 |
| large-v3 | float16 | 13.5 / 10.7 | 91.8 / 88.2 |
| large-v3 | int8_float16 | 14.3 / 10.4 | 87.6 / 85.5 |

GPU busy fraction for `head`, from the summed kernel time × 5/7 over the wall time
of the 5 timed runs under nsys. nsys slows the host, so these are lower bounds:

| model | batch 1 | batch 4 | batch 8 |
|---|---|---|---|
| small | 42% | 58% | 71% |
| large-v3 | 68% | 82% | 90% |

For `up`, small drops from 28% to 20% as the batch grows.

## Output

- `bench_batch.py`: int8_float16 is identical in all cases. float16 differs in 9 of
  48 configurations, one window each: the known bias-epilogue rounding.
- `e2e.py`: small is identical. large-v3 sequential text differs for both compute
  types. **Unverified cause**: faster-whisper's temperature fallback samples at random,
  so one build run twice may also differ.
