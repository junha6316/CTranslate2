# Whisper GPU decoding: measured results

Results only. How each change works and why is in the linked design notes.
Raw data for everything measured on 2026-09-23 is in
[`results/2026-09-23/`](../results/2026-09-23/).

## Setup

- AWS g5.2xlarge, NVIDIA A10G (compute capability 8.6), CUDA 12.8, `CUDA_ARCH_LIST=8.6`
- Models: faster-whisper `small`, `large-v3`, `large-v3-turbo`; compute types
  `float16` and `int8_float16`
- Audio: `jfk_30s` (11 s of JFK speech, padded) and `long_150s` (a 150 s lecture),
  batch 1. **Every script decodes only the first 30 s mel window** (`[..., :3000]`),
  so `long_150s` means "30 s of continuous speech", not a 150 s transcription. The
  label is kept because the raw data uses it.
- Timing: `generate()` wall clock, 2 warmup runs, then the median of 7
- Every speed comparison below is **paired within one session**: both builds were
  installed side by side on the same machine and run back to back. Numbers from
  different sessions are not multiplied together.
- Builds compared:
  - `upstream`: CTranslate2 v4.8.2
  - `9352e203`: this branch before the bias epilogue change
  - `50e1e882`: this branch, current tip

## 1. Current branch vs upstream v4.8.2 (small, large-v3)

| axis | cases | median | range |
| --- | --- | --- | --- |
| all | 64 | **-13.4%** | -44.1% .. -0.2% |
| float16, flash off, timestamps on | 8 | **-17.0%** | -44.1% .. -6.7% |
| float16, flash off, timestamps off | 8 | **-10.6%** | -16.6% .. -5.5% |
| float16, flash on, timestamps on | 8 | **-21.8%** | -40.5% .. -11.5% |
| float16, flash on, timestamps off | 8 | **-15.4%** | -23.7% .. -8.2% |
| int8_float16, flash off, timestamps on | 8 | **-9.8%** | -37.0% .. -4.6% |
| int8_float16, flash off, timestamps off | 8 | **-4.3%** | -8.5% .. -0.2% |
| int8_float16, flash on, timestamps on | 8 | **-14.8%** | -38.0% .. -3.1% |
| int8_float16, flash on, timestamps off | 8 | **-8.7%** | -13.7% .. -4.1% |

Read the **flash off** rows as the decoding changes alone. Upstream ignores
`flash_attention=True` in the encoder, so the flash on rows also include the
encoder flash attention fix on this branch.

The largest single case is small / float16 / `long_150s` / beam 5 / timestamps on:
514.5 ms to 287.6 ms (−44.1%).

<details>
<summary>All 64 cases</summary>

| model | compute | flash | audio | beam | ts | upstream ms | 50e1e882 ms | vs upstream |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large-v3 | float16 | off | jfk_30s | 1 | off | 316.3 | 298.4 | -5.6% |
| large-v3 | float16 | off | jfk_30s | 1 | on | 334.8 | 312.4 | -6.7% |
| large-v3 | float16 | off | jfk_30s | 5 | off | 374.1 | 353.4 | -5.5% |
| large-v3 | float16 | off | jfk_30s | 5 | on | 418.7 | 370.5 | -11.5% |
| large-v3 | float16 | off | long_150s | 1 | off | 794.6 | 736.8 | -7.3% |
| large-v3 | float16 | off | long_150s | 1 | on | 919.3 | 833.3 | -9.4% |
| large-v3 | float16 | off | long_150s | 5 | off | 985.9 | 921.0 | -6.6% |
| large-v3 | float16 | off | long_150s | 5 | on | 1243.9 | 1038.3 | -16.5% |
| large-v3 | float16 | on | jfk_30s | 1 | off | 321.6 | 254.6 | -20.8% |
| large-v3 | float16 | on | jfk_30s | 1 | on | 339.6 | 269.3 | -20.7% |
| large-v3 | float16 | on | jfk_30s | 5 | off | 437.0 | 365.6 | -16.3% |
| large-v3 | float16 | on | jfk_30s | 5 | on | 479.4 | 388.0 | -19.1% |
| large-v3 | float16 | on | long_150s | 1 | off | 807.6 | 721.4 | -10.7% |
| large-v3 | float16 | on | long_150s | 1 | on | 930.7 | 823.4 | -11.5% |
| large-v3 | float16 | on | long_150s | 5 | off | 1165.7 | 1070.2 | -8.2% |
| large-v3 | float16 | on | long_150s | 5 | on | 1425.9 | 1213.0 | -14.9% |
| large-v3 | int8_float16 | off | jfk_30s | 1 | off | 334.9 | 320.9 | -4.2% |
| large-v3 | int8_float16 | off | jfk_30s | 1 | on | 361.9 | 343.6 | -5.1% |
| large-v3 | int8_float16 | off | jfk_30s | 5 | off | 364.3 | 348.2 | -4.4% |
| large-v3 | int8_float16 | off | jfk_30s | 5 | on | 421.2 | 376.6 | -10.6% |
| large-v3 | int8_float16 | off | long_150s | 1 | off | 874.9 | 815.5 | -6.8% |
| large-v3 | int8_float16 | off | long_150s | 1 | on | 999.8 | 927.2 | -7.3% |
| large-v3 | int8_float16 | off | long_150s | 5 | off | 1009.2 | 923.2 | -8.5% |
| large-v3 | int8_float16 | off | long_150s | 5 | on | 1236.6 | 1045.0 | -15.5% |
| large-v3 | int8_float16 | on | jfk_30s | 1 | off | 316.5 | 273.1 | -13.7% |
| large-v3 | int8_float16 | on | jfk_30s | 1 | on | 334.5 | 289.1 | -13.6% |
| large-v3 | int8_float16 | on | jfk_30s | 5 | off | 407.8 | 352.7 | -13.5% |
| large-v3 | int8_float16 | on | jfk_30s | 5 | on | 463.3 | 384.8 | -16.9% |
| large-v3 | int8_float16 | on | long_150s | 1 | off | 795.6 | 743.2 | -6.6% |
| large-v3 | int8_float16 | on | long_150s | 1 | on | 920.3 | 891.7 | -3.1% |
| large-v3 | int8_float16 | on | long_150s | 5 | off | 1109.5 | 1058.1 | -4.6% |
| large-v3 | int8_float16 | on | long_150s | 5 | on | 1375.7 | 1194.0 | -13.2% |
| small | float16 | off | jfk_30s | 1 | off | 90.3 | 77.8 | -13.9% |
| small | float16 | off | jfk_30s | 1 | on | 99.4 | 82.0 | -17.5% |
| small | float16 | off | jfk_30s | 5 | off | 113.6 | 97.5 | -14.2% |
| small | float16 | off | jfk_30s | 5 | on | 162.0 | 105.9 | -34.7% |
| small | float16 | off | long_150s | 1 | off | 260.8 | 217.7 | -16.6% |
| small | float16 | off | long_150s | 1 | on | 294.2 | 223.6 | -24.0% |
| small | float16 | off | long_150s | 5 | off | 338.9 | 283.4 | -16.4% |
| small | float16 | off | long_150s | 5 | on | 514.5 | 287.6 | -44.1% |
| small | float16 | on | jfk_30s | 1 | off | 83.0 | 63.4 | -23.7% |
| small | float16 | on | jfk_30s | 1 | on | 90.7 | 67.8 | -25.3% |
| small | float16 | on | jfk_30s | 5 | off | 104.7 | 88.1 | -15.8% |
| small | float16 | on | jfk_30s | 5 | on | 151.2 | 96.5 | -36.2% |
| small | float16 | on | long_150s | 1 | off | 232.2 | 197.2 | -15.0% |
| small | float16 | on | long_150s | 1 | on | 263.4 | 202.9 | -23.0% |
| small | float16 | on | long_150s | 5 | off | 302.8 | 278.0 | -8.2% |
| small | float16 | on | long_150s | 5 | on | 475.7 | 282.9 | -40.5% |
| small | int8_float16 | off | jfk_30s | 1 | off | 108.5 | 108.2 | -0.2% |
| small | int8_float16 | off | jfk_30s | 1 | on | 119.8 | 114.2 | -4.6% |
| small | int8_float16 | off | jfk_30s | 5 | off | 125.4 | 124.6 | -0.6% |
| small | int8_float16 | off | jfk_30s | 5 | on | 172.8 | 131.4 | -24.0% |
| small | int8_float16 | off | long_150s | 1 | off | 330.7 | 326.0 | -1.4% |
| small | int8_float16 | off | long_150s | 1 | on | 366.3 | 333.6 | -8.9% |
| small | int8_float16 | off | long_150s | 5 | off | 370.4 | 349.5 | -5.7% |
| small | int8_float16 | off | long_150s | 5 | on | 616.3 | 388.3 | -37.0% |
| small | int8_float16 | on | jfk_30s | 1 | off | 101.9 | 88.6 | -13.0% |
| small | int8_float16 | on | jfk_30s | 1 | on | 112.5 | 94.7 | -15.9% |
| small | int8_float16 | on | jfk_30s | 5 | off | 115.7 | 103.1 | -10.9% |
| small | int8_float16 | on | jfk_30s | 5 | on | 173.9 | 113.3 | -34.8% |
| small | int8_float16 | on | long_150s | 1 | off | 302.9 | 290.5 | -4.1% |
| small | int8_float16 | on | long_150s | 1 | on | 341.7 | 295.0 | -13.7% |
| small | int8_float16 | on | long_150s | 5 | off | 338.9 | 320.6 | -5.4% |
| small | int8_float16 | on | long_150s | 5 | on | 572.0 | 354.8 | -38.0% |

</details>

## 2. large-v3-turbo (upstream, `9352e203`, `50e1e882` in one session)

Current branch vs upstream:

| axis | cases | median | range |
| --- | --- | --- | --- |
| all | 32 | **-25.1%** | -56.2% .. -1.6% |
| float16, flash off | 8 | **-5.4%** | -45.0% .. -2.9% |
| float16, flash on | 8 | **-40.8%** | -50.2% .. -23.5% |
| int8_float16, flash off | 8 | **-4.7%** | -45.4% .. -1.6% |
| int8_float16, flash on | 8 | **-39.7%** | -56.2% .. -23.4% |

Turbo has 4 decoder layers and 32 encoder layers, so the encoder dominates. The flash
on gain is almost entirely the encoder flash attention fix. With flash off, the
large gains are limited to timestamps on with beam 5 (`long_150s`: 469.2 ms to 258.3 ms).

Bias epilogue only (`50e1e882` vs `9352e203`):

| axis | cases | median | range |
| --- | --- | --- | --- |
| all | 32 | **-1.2%** | -6.8% .. +0.4% |
| float16, flash off | 8 | **-2.7%** | -6.8% .. -2.3% |
| float16, flash on | 8 | **-3.4%** | -4.6% .. -2.8% |
| int8_float16, flash off | 8 | **+0.2%** | -0.0% .. +0.3% |
| int8_float16, flash on | 8 | **+0.0%** | -0.2% .. +0.4% |

<details>
<summary>All 32 cases</summary>

| model | compute | flash | audio | beam | ts | upstream ms | 9352e203 ms | 50e1e882 ms | vs upstream | vs upstream |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large-v3-turbo | float16 | off | jfk_30s | 1 | off | 142.4 | 142.0 | 138.3 | -0.3% | -2.9% |
| large-v3-turbo | float16 | off | jfk_30s | 1 | on | 149.7 | 144.9 | 141.2 | -3.2% | -5.7% |
| large-v3-turbo | float16 | off | jfk_30s | 5 | off | 152.4 | 151.7 | 147.8 | -0.5% | -3.0% |
| large-v3-turbo | float16 | off | jfk_30s | 5 | on | 185.0 | 155.3 | 151.1 | -16.0% | -18.3% |
| large-v3-turbo | float16 | off | long_150s | 1 | off | 222.9 | 219.1 | 212.8 | -1.7% | -4.5% |
| large-v3-turbo | float16 | off | long_150s | 1 | on | 259.0 | 228.2 | 223.0 | -11.9% | -13.9% |
| large-v3-turbo | float16 | off | long_150s | 5 | off | 260.4 | 254.8 | 247.0 | -2.1% | -5.1% |
| large-v3-turbo | float16 | off | long_150s | 5 | on | 469.2 | 277.0 | 258.3 | -41.0% | -45.0% |
| large-v3-turbo | float16 | on | jfk_30s | 1 | off | 142.8 | 85.2 | 81.3 | -40.3% | -43.1% |
| large-v3-turbo | float16 | on | jfk_30s | 1 | on | 148.8 | 88.2 | 84.4 | -40.7% | -43.3% |
| large-v3-turbo | float16 | on | jfk_30s | 5 | off | 157.6 | 100.6 | 96.8 | -36.2% | -38.6% |
| large-v3-turbo | float16 | on | jfk_30s | 5 | on | 187.9 | 104.8 | 101.0 | -44.2% | -46.3% |
| large-v3-turbo | float16 | on | long_150s | 1 | off | 223.6 | 164.0 | 159.0 | -26.7% | -28.9% |
| large-v3-turbo | float16 | on | long_150s | 1 | on | 257.1 | 174.5 | 169.6 | -32.1% | -34.0% |
| large-v3-turbo | float16 | on | long_150s | 5 | off | 276.8 | 218.6 | 211.7 | -21.0% | -23.5% |
| large-v3-turbo | float16 | on | long_150s | 5 | on | 450.9 | 231.3 | 224.3 | -48.7% | -50.2% |
| large-v3-turbo | int8_float16 | off | jfk_30s | 1 | off | 142.9 | 140.6 | 140.5 | -1.6% | -1.6% |
| large-v3-turbo | int8_float16 | off | jfk_30s | 1 | on | 149.8 | 143.5 | 143.5 | -4.3% | -4.2% |
| large-v3-turbo | int8_float16 | off | jfk_30s | 5 | off | 149.7 | 147.1 | 147.1 | -1.7% | -1.7% |
| large-v3-turbo | int8_float16 | off | jfk_30s | 5 | on | 193.2 | 150.4 | 150.6 | -22.1% | -22.0% |
| large-v3-turbo | int8_float16 | off | long_150s | 1 | off | 226.3 | 217.1 | 217.7 | -4.0% | -3.8% |
| large-v3-turbo | int8_float16 | off | long_150s | 1 | on | 264.1 | 226.2 | 226.8 | -14.4% | -14.1% |
| large-v3-turbo | int8_float16 | off | long_150s | 5 | off | 254.7 | 241.2 | 241.6 | -5.3% | -5.1% |
| large-v3-turbo | int8_float16 | off | long_150s | 5 | on | 483.1 | 263.4 | 263.8 | -45.5% | -45.4% |
| large-v3-turbo | int8_float16 | on | jfk_30s | 1 | off | 140.4 | 82.1 | 82.2 | -41.5% | -41.5% |
| large-v3-turbo | int8_float16 | on | jfk_30s | 1 | on | 146.6 | 85.0 | 85.1 | -42.0% | -42.0% |
| large-v3-turbo | int8_float16 | on | jfk_30s | 5 | off | 152.5 | 94.9 | 94.7 | -37.8% | -37.9% |
| large-v3-turbo | int8_float16 | on | jfk_30s | 5 | on | 192.9 | 98.8 | 98.6 | -48.8% | -48.9% |
| large-v3-turbo | int8_float16 | on | long_150s | 1 | off | 216.3 | 158.2 | 158.6 | -26.8% | -26.7% |
| large-v3-turbo | int8_float16 | on | long_150s | 1 | on | 252.0 | 166.9 | 167.6 | -33.8% | -33.5% |
| large-v3-turbo | int8_float16 | on | long_150s | 5 | off | 259.9 | 199.2 | 199.0 | -23.3% | -23.4% |
| large-v3-turbo | int8_float16 | on | long_150s | 5 | on | 486.4 | 213.4 | 213.0 | -56.1% | -56.2% |

</details>

## 3. Each change on its own

Each row was measured against its own same-session reference. The control is an
axis the change cannot affect.

| change | changed axis | median | control | design note |
| --- | --- | --- | --- | --- |
| timestamp rule on the GPU (`TimestampGate`, token ranges) | timestamps on, 16 cases, vs upstream | **−8.2%** (−33.6% .. −0.8%) | timestamps off: +0.1% | [timestamp_rule_cost.md](timestamp_rule_cost.md) |
| KV cache written in place (no per-step Concat) | flash off, 32 cases | **−5.2%** | flash on: −0.18% | [kv_cache_preallocation.md](kv_cache_preallocation.md) |
| Dense bias and residual in the cuBLASLt epilogue | float16, 32 cases | **−4.3%** (−9.5% .. −2.3%) | int8_float16: +0.7% | [gemm_bias_epilogue.md](gemm_bias_epilogue.md) |

The KV cache change measured −3.3% in an earlier build of the same idea and −1.0%
in a direct A/B between the two builds, so the honest range is 3 to 5%.

Bias epilogue detail (`9352e203` vs the change, small and large-v3):

| axis | cases | median | range |
| --- | --- | --- | --- |
| float16 (changed path) | 32 | **-4.3%** | -9.5% .. -2.3% |
| int8_float16 (control) | 32 | **+0.7%** | -2.9% .. +5.4% |
| float16, flash off | 16 | **-5.2%** | -9.5% .. -3.4% |
| float16, flash on | 16 | **-4.3%** | -9.1% .. -2.3% |

The gain is larger on small (−6% .. −9.5%) than on large-v3 (−3.4% .. −4.8%): the
removed launch cost is fixed per call, and large-v3's kernels run longer.

## 4. Kernel launches and host time

Nsight Systems, whisper-small float16, beam 5, the first 30 s of the lecture, 10 decodes,
timestamps on.

| | branch before KV cache (`6ccd6335`) | KV cache (`9352e203`) | bias epilogue (`50e1e882`) |
| --- | --- | --- | --- |
| kernel launches (`cudaLaunchKernel` + `cuLaunchKernel`) | 572,666 | 545,522 | **466,716** |
| `cudaMallocAsync` / `cudaFreeAsync` pairs | 281,121 | 253,223 | 253,223 |
| host CUDA API time | 4.387 s | 4.039 s | 3.568 s |
| GPU kernel time | 2.911 s | 2.815 s | 2.695 s |

- The bias epilogue change removed both bias kernels (46,644 with a residual, 32,474
  without). cuBLASLt ran a separate epilogue kernel for one shape (312 calls), so the
  net reduction is 78,806.
- Host CUDA API time is still above GPU kernel time: the decode remains
  launch-bound.

## 5. Output equivalence

Token ids dumped with timestamps on for 4 clips (JFK, the lecture, silence, a
Korean clip forced to `<|en|>`), beam 1 and 5.

| comparison | cases | identical tokens | notes |
| --- | --- | --- | --- |
| `50e1e882` vs upstream, small + large-v3 | 32 | 30 | both differences are large-v3 / float16 / beam 5 |
| `50e1e882` vs `9352e203`, small + large-v3 | 32 | 30 | same two cases; int8_float16 identical including scores |
| `9352e203` vs upstream, small + large-v3 | 32 | 32 | tokens and scores identical |
| `50e1e882` vs `9352e203`, large-v3-turbo | 16 | 15 | float16 / lecture / beam 1 |
| `9352e203` vs upstream, large-v3-turbo | 16 | 16 | |

The differences come from the bias epilogue: bias and residual are now added before
the final fp16 rounding instead of after it, which flips near-tied decisions.

- large-v3, lecture, beam 5: one timestamp `<|15.66|>` becomes `<|15.64|>`
- large-v3, Korean clip, beam 5: the (already meaningless) romanization is spelled
  differently
- turbo, lecture, beam 1: "15 times 10 is about 150" becomes "Fifteen times ten is
  about 150", and one timestamp `<|15.50|>` becomes `<|15.56|>`

Flash attention on vs off (measured 2026-09-22, small + large-v3, 40 cases each; the
control runs flash off twice):

| model / compute type | control differences | flash on vs off differences |
| --- | --- | --- |
| small / float16 | 0 | 1 |
| small / int8_float16 | 0 | 4 |
| large-v3 / float16 | 0 | 0 |
| large-v3 / int8_float16 | 0 | 3 |

All differences are word-level, none are degenerate. Flash attention accumulates
QKᵀ in fp32, tiles the softmax online and uses `exp2f` built with
`--use_fast_math`. The default path accumulates QKᵀ in fp16 and writes the scores
to memory. Neither is a ground truth. Which one is closer to a float32 decode has not
been measured.

## Hypotheses that the measurements refuted

- "Removing the duplicate LogSoftMax speeds up the GPU": no effect. The cost was the
  blocking device-to-host copies in `max` and `logsumexp`, 190 to 380 µs per call
  regardless of size.
- "`cudaMemcpy2DAsync` is free because it is not a kernel launch": 7.34 µs per call
  on the host against 4.77 µs for `cudaLaunchKernel`.
- "Short audio is a control for per-step changes": it is not. Per-step overhead
  shrinks regardless of length.
- "About 1,800 of the 79,118 bias-like kernels are plain `ops::Add`": all 79,118
  came from `BiasAdd`.

## Raw data

| file | content |
| --- | --- |
| `bench_upstream_v4.8.2.json`, `bench_50e1e882.json` | section 1 |
| `bench_9352e203.json`, `bench_bias_epilogue_uncommitted.json` | bias epilogue A/B (same code as `50e1e882` apart from comments) |
| `turbo_bench_*.json` | section 2 |
| `tok_bias_epilogue.json`, `turbo_tok_*.json` | section 5 token dumps |

Earlier sessions: `bench_upstream.json`, `bench_branch.json`, `bench_gate.json`,
`ab_results.json` in the repository root.
