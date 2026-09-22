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
- `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32` failed. Its relationship
  to the baseline has **not** been tested; do not report the full suite as green
  or label the failure pre-existing until a baseline run confirms that.
- CUDA kernel execution, Whisper output equivalence, and performance have **not**
  been validated. No measured speedup or memory reduction is claimed.

Local reproduction commands:

```sh
cmake -S . -B build -DWITH_MKL=OFF -DWITH_ACCELERATE=ON \
  -DOPENMP_RUNTIME=NONE -DBUILD_CLI=OFF -DBUILD_TESTS=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target ctranslate2_test -j 6
build/tests/ctranslate2_test tests/data --gtest_filter='*FlashEncoderAttentionTest*'
build/tests/ctranslate2_test tests/data --gtest_brief=1
```

## Required follow-up checks

- [ ] Reproduce the quantized Conv1D failure at the baseline commit using the same
  compiler and build options. If it only fails with this change, investigate
  before merging.
- [ ] Build on Linux with a supported CUDA/cuDNN toolchain and an Ampere or newer
  NVIDIA GPU. Enable `WITH_CUDA=ON`, `WITH_FLASH_ATTN=ON`, and `BUILD_TESTS=ON`
  in the project's GPU build configuration. For Whisper, include cuDNN support.
- [ ] Run `ctranslate2_test tests/data --gtest_filter='*FlashEncoderAttentionTest*'`
  from that build. Confirm that the CUDA float16 and bfloat16 cases are listed
  and actually execute; a CPU-only run or skipped GPU cases is not sufficient.
- [ ] Confirm the unmasked encoder matches standard attention within the test's
  tolerance, and that changing a later token changes an earlier encoder output.
  Confirm the same perturbation does not affect earlier decoder outputs.
- [ ] Exercise the fallback cases on GPU: unequal sequence lengths, padding
  removal, relative position bias, and each Q/K/V normalization variant.
  Compare outputs with standard attention.
- [ ] Add coverage for relative position keys/values (including asymmetric
  positions), rotary embeddings, returned attention weights, and unsupported
  head sizes. These are not all covered by the current regression tests.
- [ ] Compare standard and Flash Attention with real Whisper models, including
  the production model. Test short speech, long recordings split into windows,
  silence/noise, and multilingual audio where applicable.
- [ ] Cover batch sizes 1 and production size, beam sizes 1 and 5, and production
  compute types such as float16 and int8_float16. Cover timestamps on/off and
  word alignment if used by the application.
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
