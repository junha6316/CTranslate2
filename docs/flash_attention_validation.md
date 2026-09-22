# Encoder Flash Attention validation

## Scope and current status

This change preserves bidirectional encoder attention, keeps decoder attention
causal, and enables the Flash Attention option in the Whisper encoder. Encoder
calls that require unsupported features fall back to standard attention.

The implementation is based on `perf/profiled-fixes` at
`8791844a4f0b524e30133e2f62875695a515ef55`.

Local validation used macOS, AppleClang 21, and the Accelerate CPU backend:

- C++ library and test executable built successfully.
- The three new CPU regression tests passed: length masks/padding removal,
  relative position bias creation/reuse, and Q/K/V normalization.
- The full suite ran 201 tests: 198 passed, 2 skipped, and 1 failed.
- `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32` failed. A baseline run has
  now confirmed this failure is **pre-existing and unrelated to this change**: it
  reproduces identically at `8791844a` (`perf/profiled-fixes`) and at unmodified
  upstream `d44d2d06` (v4.8.2), with the same value at the same index. The cause is
  the build configuration, not the code: `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON`
  leaves no INT8 GEMM backend on CPU, and the test throws
  `No INT8 GEMM backend for CPU` before reaching any attention code.
- Suite counts across the three commits, same compiler and build options:

  | commit | tests | passed | skipped | failed |
  | --- | --- | --- | --- | --- |
  | `9585b22c` (this change) | 201 | 198 | 2 | 1 |
  | `8791844a` (baseline) | 197 | 195 | 1 | 1 |
  | `d44d2d06` (upstream v4.8.2) | 197 | 195 | 1 | 1 |

  The change adds 4 tests: 3 CPU regression tests that pass, and 1 CUDA case that
  is skipped on this machine.
- CUDA kernel execution, Whisper output equivalence, and performance have now been
  measured on an A10G (compute capability 8.6); see "GPU results" below.

Local reproduction commands:

```sh
cmake -S . -B build -DWITH_MKL=OFF -DWITH_ACCELERATE=ON \
  -DOPENMP_RUNTIME=NONE -DBUILD_CLI=OFF -DBUILD_TESTS=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target ctranslate2_test -j 6
build/tests/ctranslate2_test tests/data --gtest_filter='*FlashEncoderAttentionTest*'
build/tests/ctranslate2_test tests/data --gtest_brief=1
```

## GPU results (A10G, compute capability 8.6)

Measured on an EC2 g5.2xlarge with CUDA 12.8 and cuDNN 9.10.2, built with
`WITH_CUDA=ON WITH_CUDNN=ON WITH_FLASH_ATTN=ON CUDA_ARCH_LIST=8.6`.

### The CUDA regression tests run and pass

All 8 CUDA cases execute; none are skipped.

| case | float16 | bfloat16 |
| --- | --- | --- |
| `PreservesLengthMaskAndPaddingRemoval` | pass | pass |
| `PreservesRelativePositionBias` | pass | pass |
| `PreservesProjectionNormalization` | pass | pass |
| `UnmaskedEncoderIsBidirectionalAndDecoderIsCausal` | pass | pass |

The same build fails 34 CPU tests, all with `No SGEMM backend on CPU`. That is the
build configuration, not the code: this build sets `WITH_MKL=OFF` and Linux has no
Accelerate, so the CPU has no BLAS backend at all. Zero CUDA tests fail. A CPU BLAS
backend still needs to be added before the Linux CPU suite means anything.

### Flash Attention changes Whisper output in some configurations

Each configuration was run three times: twice with Flash Attention off (`off1`,
`off2`) and once on. `off1` vs `off2` measures run-to-run nondeterminism; `off1` vs
`on` measures the effect of Flash Attention. Language was forced so language
detection could not confound the comparison. Inputs: short English, 2.5-minute
English, English repeated past the 30-second window, 20 seconds of silence, and
Korean. Beam sizes 1 and 5.

| model / compute type | cases | control diffs | Flash Attention diffs |
| --- | --- | --- | --- |
| small / float16 | 10 | 0 | 1 |
| small / int8_float16 | 10 | 0 | 4 |
| large-v3 / float16 | 10 | 0 | 0 |
| large-v3 / int8_float16 | 10 | 0 | 3 |

**The control is clean in all 40 cases**, so the decoder is deterministic run to run
and the differences cannot be dismissed as noise. Flash Attention changes the output
in 8 of 40 cases.

This is not by itself a defect. Flash Attention tiles the attention computation and
uses an online softmax, so it sums in a different order and rounds differently in
float16. A small perturbation flips a beam decision that was nearly tied. The pattern
fits that reading: every difference is a wording change, not a breakdown, and they
concentrate in `int8_float16` (7 of 8) and at beam 5 (6 of 8), where ties are closest.
`large-v3 / float16` is identical in all 10 cases.

What this means in practice: enabling Flash Attention is not output-preserving, so it
cannot be turned on silently for a workload whose transcripts are compared against
stored ones. Which compute type is used matters more than the model size.

Differences per case are in `ab_results.json` from the measurement run.

### Performance

Both builds were compiled from the same source tree with the same compiler, CUDA
version, architecture and flags, and each got its own Python bindings so no build ran
against another build's headers. Each configuration was warmed up twice, then timed
seven times; the median is reported. Run-to-run spread was under 2% almost everywhere,
so treat anything inside roughly +/-3% as noise.

#### The perf commits do not change GPU latency

Branch against upstream v4.8.2, Flash Attention off in both, so this isolates the five
commits on `perf/profiled-fixes`:

| axis | cases | median delta | range |
| --- | --- | --- | --- |
| timestamps on | 16 | -0.3% | -2.2% .. +1.0% |
| timestamps off | 16 | +0.2% | -3.2% .. +1.9% |

Every case is inside the noise band, and the generated token count is identical in all
32, so both builds did the same work.

This **contradicts the expectation** that removing the redundant LogSoftMax would help
on GPU. That code sits in the timestamp rule, so the effect should appear on the
timestamps-on axis; that axis is if anything the quieter of the two. The reasoning that
motivated these commits came from CPU profiling and is untouched by this measurement,
but no GPU speedup should be claimed for them.

#### Flash Attention is a large win at beam 1 and a loss at beam 5 on large-v3

Flash Attention on against off, both on the branch build. 32 cases, median -12.4%,
range -21.9% .. +11.8%. The sign is not constant:

| regime | effect |
| --- | --- |
| beam 1, every model and compute type | 4.4% to 21.9% faster |
| small, beam 5 | 7.1% to 17.4% faster |
| large-v3, beam 5, 150s audio | 8.3% to 11.8% **slower** |

The encoder saving is roughly fixed, so it dominates when decoding is short. At beam 5
on large-v3 with long audio the decoder dominates and the saving is more than cancelled.

This overlaps the correctness result above: `int8_float16` at beam 5 is both where the
output changes and, on large-v3, where Flash Attention costs time. There is no reason
to enable it there.

Raw measurements are in `bench_upstream.json` and `bench_branch.json`.

## Required follow-up checks

- [x] Reproduce the quantized Conv1D failure at the baseline commit using the same
  compiler and build options. Done: it fails identically at `8791844a` and at
  upstream `d44d2d06`, so it does not block this change. It stays a known failure
  of CPU-only Accelerate builds.
- [x] Build on Linux with a supported CUDA/cuDNN toolchain and an Ampere or newer
  NVIDIA GPU. Done on an A10G with CUDA 12.8 and cuDNN 9.10.2. Note that CUDA 13.x is
  the default toolkit on current Deep Learning AMIs and the bundled CUTLASS is not
  known to build with it; 12.8 was selected explicitly.
- [x] Run `ctranslate2_test tests/data --gtest_filter='*FlashEncoderAttentionTest*'`
  from that build. Done: all 8 CUDA cases execute and pass, none skipped.
- [x] Confirm the unmasked encoder matches standard attention within the test's
  tolerance, and that changing a later token changes an earlier encoder output.
  Confirm the same perturbation does not affect earlier decoder outputs. Covered by
  `UnmaskedEncoderIsBidirectionalAndDecoderIsCausal`, passing on CUDA in both dtypes.
- [ ] Exercise the fallback cases on GPU: unequal sequence lengths, padding
  removal, relative position bias, and each Q/K/V normalization variant.
  Compare outputs with standard attention.
- [ ] Add coverage for relative position keys/values (including asymmetric
  positions), rotary embeddings, returned attention weights, and unsupported
  head sizes. These are not all covered by the current regression tests.
- [ ] Compare standard and Flash Attention with real Whisper models, including
  the production model. Test short speech, long recordings split into windows,
  silence/noise, and multilingual audio where applicable.
- [x] Cover beam sizes 1 and 5, and production compute types float16 and
  int8_float16, with timestamps on and off. Done for correctness and for latency.
  Batch sizes above 1 and word alignment are still uncovered.
- [ ] Compare encoder outputs, token IDs, transcripts, timestamps, and alignment
  results. Record tolerances and every discrepancy. Small numerical differences
  can change decoding choices; investigate token differences and evaluate WER
  or CER on a fixed labeled set rather than assuming exact equivalence.
- [ ] Run the existing GPU suite and the application's faster-whisper integration
  tests, including repeated calls and concurrent workers if used in production.

## Performance measurement

Use identical GPU, model files, audio, compute type, batch/beam settings, and
worker count for each comparison. Record the GPU, driver, CUDA/cuDNN versions,
build options, and exact commit IDs. Run in separate fresh processes so the
memory allocator state from one configuration does not affect another.

Compare these configurations separately:

1. Baseline `8791844a...` with `flash_attention=False` versus this change with
   `flash_attention=False`, to detect overhead when the option is disabled.
2. This change with Flash Attention off versus on, to measure the total effect
   of enabling Flash Attention.
3. Baseline `8791844a...` with Flash Attention on versus this change with it on,
   to measure the incremental Whisper encoder improvement. For generic
   Transformer encoders, that baseline has incorrect Flash Attention semantics;
   use standard attention as the correctness reference.

Warm up each configuration before timing. Synchronize CUDA around timed regions
or use CUDA events; host enqueue time alone does not measure GPU execution.
Perform at least 20 measured repetitions and report median and p95. Keep model
loading and preprocessing separate from inference timing, and also measure
the application's complete transcription path.

Record the following in a results table:

| Configuration | Encoder ms | End-to-end ms | Audio seconds / wall second | Peak GPU memory | Output differences |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline, Flash off | pending | pending | pending | pending | pending |
| Fixed, Flash off | pending | pending | pending | pending | pending |
| Baseline, Flash on | pending | pending | pending | pending | pending |
| Fixed, Flash on | pending | pending | pending | pending | pending |

Use a process/device-level GPU memory measurement that includes CTranslate2's
native CUDA allocations. PyTorch allocator statistics alone may miss these
allocations. State the sampling method and distinguish peak from idle memory.

Confirm with profiling that eligible Whisper encoder calls reach the Flash
Attention kernel. Fallback paths are expected to use standard attention and
must not be counted as accelerated. Report measured end-to-end speedup and
memory savings only after correctness checks pass.

## Open items

- Add a CPU BLAS backend to the Linux build so the CPU suite is meaningful there.
  34 tests currently throw `No SGEMM backend on CPU` and the suite aborts partway.
- Measure batched throughput. Everything here is batch size 1.
- Decide whether the output differences above are acceptable for the intended
  workload, or whether Flash Attention should stay off for `int8_float16`.
