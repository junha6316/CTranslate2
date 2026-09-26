# Environment variables

Some environment variables can be configured to customize the execution. When using Python, these variables should be set before importing the `ctranslate2` module, e.g.:

```python
import os
os.environ["CT2_VERBOSE"] = "1"

import ctranslate2
```

```{note}
Boolean environment variables can be enabled with `"1"` or `"true"`.
```

## `CT2_CUDA_ALLOCATOR`

Allocating memory on the GPU with `cudaMalloc` is costly and is best avoided in high-performance code. For this reason CTranslate2 integrates caching allocators which enable a fast reuse of previously allocated buffers. The following allocators are integrated:

* `cuda_malloc_async` (default for CUDA >= 11.2)<br/>Uses the [asynchronous allocator with memory pools](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html) introduced in CUDA 11.2.
* `cub_caching` (default for CUDA < 11.2)<br/>Uses the caching allocator from the [CUB project](https://github.com/NVIDIA/cub).

## `CT2_CUDA_ALLOW_BF16`

Allow using BF16 computation on GPU even if the device does not have efficient BF16 support.

## `CT2_CUDA_ALLOW_FP16`

Allow using FP16 computation on GPU even if the device does not have efficient FP16 support.

## `CT2_CUDA_TRUE_FP16_GEMM`

Allow using true FP16 computation in GEMM operations. When disabled, the computation or accumulation may use FP32 instead.

This flag is enabled by default, but some models may automatically disable it when they are known to work better with the increased precision.

## `CT2_CUDA_CACHING_ALLOCATOR_CONFIG`

The `cub_caching` allocator can be configured to tradeoff memory usage and speed. By default, CTranslate2 uses the following values which have been selected experimentally:

* `bin_growth = 4`
* `min_bin = 3`
* `max_bin = 12`
* `max_cached_bytes = 209715200` (200MB)

You can override these parameters with comma-separated values in the same order as the list above:

```bash
export CT2_CUDA_CACHING_ALLOCATOR_CONFIG=8,3,7,6291455
```

See the description of each parameter in the [allocator implementation](https://github.com/NVIDIA/cub/blob/main/cub/util_allocator.cuh).

## `CT2_CUDA_POOL_RELEASE_THRESHOLD`

Release threshold of the memory pool used by the `cuda_malloc_async` allocator: the number of bytes (in use plus cached) the pool keeps across synchronizations instead of returning them to the device. The value is a byte count with an optional `K`, `M` or `G` suffix (binary multiples), or `max` to never shrink the pool on synchronization.

By default (unset or `0`) the pool keeps nothing: every synchronization, including the one a worker makes when its job queue runs empty, releases all unused memory, and the next job maps it back. With large decoder caches (e.g. flash attention) or batch sizes that vary between requests, this release and remap is a measurable part of each request. Raising the threshold removes it at the cost of keeping up to that much GPU memory reserved between requests. The kept memory is returned when the model is unloaded with `unload_model()`.

```bash
export CT2_CUDA_POOL_RELEASE_THRESHOLD=8G
```

The threshold applies to the device's current memory pool, which is shared with any other library using `cudaMallocAsync` in the same process.

## `CT2_CUDA_GRAPHS_TIERS`

Capacity tiers for the opt-in CUDA-graph replay of the Whisper decoder (`CT2_CUDA_GRAPHS=1`). The graph path preallocates the self-attention caches for `CT2_CUDA_GRAPHS_RESERVE` decoding steps (or the whole decode length when the reserve is unset) and replays captured graphs at that fixed capacity. Without tiers, a decode that outgrows a capped reserve runs its remaining steps eagerly. With tiers, the step that crosses the capacity grows the caches to the next tier and the decoder re-captures its graphs at the new shapes, so the rest of the decode keeps replaying.

* unset, `0`, `off` or `false`: inactive (default).
* `+N`: at each crossing, grow the capacity by `N` steps (rounded up to a multiple of 32, at least 32), up to the decode length (`max_length`, 448 for Whisper). `1`, `on` and `true` are aliases for `+64`.
* `a,b,...`: explicit capacities, each rounded up to a multiple of 32 and capped at the decode length; values not above the reserve are ignored. Past the last listed capacity the decode continues eagerly, as without tiers.

Any other value leaves tiers inactive and logs a warning. Tiers have no effect without `CT2_CUDA_GRAPHS=1`, when the reserve already covers the whole decode (`CT2_CUDA_GRAPHS_RESERVE` unset), or with flash attention, which the graph path does not support. Tokens match the graph path at the same capacities: a decode that transitions is bit-identical to the padded eager decode on the same capacity schedule.

The recommended setting for long-form beam search is:

```bash
export CT2_CUDA_GRAPHS=1 CT2_CUDA_GRAPHS_RESERVE=128 CT2_CUDA_GRAPHS_TIERS=+64
```

The caches never grow past what an unset reserve would allocate up front. A crossing briefly holds the old and the new cache together.

`CT2_CUDA_GRAPHS_TIERS_REENTRY=warmup` is a debugging switch: each transition repeats the full capture warmup instead of re-capturing at the next step (`fast`, the default).

## `CT2_FORCE_CPU_ISA`

Force CTranslate2 to select a specific instruction set architecture (ISA). Possible values are:

* `GENERIC`
* `AVX`
* `AVX2`
* `AVX512`

```{attention}
This does not impact backend libraries (such as Intel MKL) which usually have their own environment variables to configure ISA dispatching.
```

## `CT2_PACKED_GEMM`

Enable or disable the packed GEMM API for Intel MKL. Packed GEMM is enabled by default when using the MKL backend and improves decoding performance by pre-packing weight matrices at model load time. Set to `0` to disable it. See [Intel's article](https://software.intel.com/content/www/us/en/develop/articles/introducing-the-new-packed-apis-for-gemm.html) to learn more about packed GEMM.

## `CT2_USE_MKL`

Force CTranslate2 to use (or not) Intel MKL. By default, the runtime automatically decides whether to use Intel MKL or not based on the CPU vendor.

## `CT2_VERBOSE`

Configure the default logs verbosity:

* -3 = off
* -2 = critical
* -1 = error
* 0 = warning (default)
* 1 = info
* 2 = debug
* 3 = trace

```{tip}
The log level can also be controlled by API. See for example the Python function [`ctranslate2.set_log_level`](python/ctranslate2.set_log_level.rst).
```
