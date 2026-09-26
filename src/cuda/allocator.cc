#include "ctranslate2/allocator.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cstdio>

#include "ctranslate2/utils.h"
#include "cuda/alloc_stats.h"
#include "cuda/utils.h"
#include "env.h"

#ifdef CT2_USE_HIP
#include <hip/hip_runtime.h>
#include <hipcub/util_allocator.hpp>
#define cub hipcub
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaFreeAsync hipFreeAsync
#define cudaMallocAsync hipMallocAsync
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#define cudaMemPool_t hipMemPool_t
#define cudaDeviceGetMemPool hipDeviceGetMemPool
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemPoolTrimTo hipMemPoolTrimTo
// Async allocactor has crashing issues on Windows
// https://github.com/OpenNMT/CTranslate2/issues/1072#issuecomment-3418768140
#define CT2_USE_ASYNC_ALLOC !_WIN32
#else
#include <cuda.h>
#include <cub/util_allocator.cuh>
#define CT2_USE_ASYNC_ALLOC CUDA_VERSION >= 11020
#endif
#include <spdlog/spdlog.h>

namespace ctranslate2 {
  namespace cuda {

    static std::atomic<std::uint64_t> g_allocation_count{0};
    static std::atomic<std::uint64_t> g_free_count{0};
    static std::atomic<std::uint64_t> g_capture_violations{0};
    static thread_local bool t_capture_active = false;

    // Frees requested while a capture was active on their thread: performing them
    // would record graph nodes, so they are parked here and drained on the next
    // allocator entry outside a capture.
    static std::mutex g_deferred_frees_mutex;
    static std::vector<std::pair<void*, int>> g_deferred_frees;

    std::uint64_t allocation_count() {
      return g_allocation_count.load(std::memory_order_relaxed);
    }

    std::uint64_t free_count() {
      return g_free_count.load(std::memory_order_relaxed);
    }

    void set_capture_active(bool active) {
      t_capture_active = active;
    }

    bool capture_active() {
      return t_capture_active;
    }

    std::uint64_t capture_violation_count() {
      return g_capture_violations.load(std::memory_order_relaxed);
    }

    static void count_allocation_or_abort_capture(size_t size) {
      if (t_capture_active)
        // Thrown before any CUDA call so nothing is recorded in the capture; the
        // graph runner catches this, ends the capture and reruns the step eagerly.
        throw std::runtime_error("ct2_graph_capture_allocation");
      g_allocation_count.fetch_add(1, std::memory_order_relaxed);
      // Debug: print every allocation size so a steady-state decode step's remaining
      // allocations can be attributed to their sites.
      static const bool trace = read_bool_from_env("CT2_CUDA_ALLOC_TRACE");
      if (trace)
        fprintf(stderr, "CT2_ALLOC size=%zu\n", size);
    }

    // Returns true when the caller should perform the free now. Returns false when
    // the free was deferred because a capture is active on this thread.
    template <typename FreeFn>
    static bool count_free_or_defer(void* ptr, int device_index, const FreeFn&) {
      if (t_capture_active) {
        g_capture_violations.fetch_add(1, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> lock(g_deferred_frees_mutex);
        g_deferred_frees.emplace_back(ptr, device_index);
        return false;
      }
      g_free_count.fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    template <typename FreeFn>
    static void drain_deferred_frees(const FreeFn& do_free) {
      if (t_capture_active)
        return;
      std::vector<std::pair<void*, int>> pending;
      {
        const std::lock_guard<std::mutex> lock(g_deferred_frees_mutex);
        if (g_deferred_frees.empty())
          return;
        pending.swap(g_deferred_frees);
      }
      for (const auto& [ptr, device_index] : pending) {
        g_free_count.fetch_add(1, std::memory_order_relaxed);
        do_free(ptr, device_index);
      }
    }

    // See https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html.
    class CubCachingAllocator : public Allocator {
    public:
      CubCachingAllocator() {
        unsigned int bin_growth = 4;
        unsigned int min_bin = 3;
        unsigned int max_bin = 12;
        size_t max_cached_bytes = 200 * (1 << 20);  // 200MB

        const char* config_env = std::getenv("CT2_CUDA_CACHING_ALLOCATOR_CONFIG");
        if (config_env) {
          const std::vector<std::string> values = split_string(config_env, ',');
          if (values.size() != 4)
            throw std::invalid_argument("CT2_CUDA_CACHING_ALLOCATOR_CONFIG environment variable "
                                        "should have format: "
                                        "bin_growth,min_bin,max_bin,max_cached_bytes");
          bin_growth = std::stoul(values[0]);
          min_bin = std::stoul(values[1]);
          max_bin = std::stoul(values[2]);
          max_cached_bytes = std::stoull(values[3]);
        }

        _allocator = std::make_unique<cub::CachingDeviceAllocator>(bin_growth,
                                                                   min_bin,
                                                                   max_bin,
                                                                   max_cached_bytes);
      }

      void* allocate(size_t size, int device_index) override {
        count_allocation_or_abort_capture(size);
        drain_deferred_frees([this](void* p, int device) { _allocator->DeviceFree(device, p); });
        void* ptr = nullptr;
        CUDA_CHECK(_allocator->DeviceAllocate(device_index, &ptr, size, cuda::get_cuda_stream()));
        return ptr;
      }

      void free(void* ptr, int device_index) override {
        const auto do_free = [this](void* p, int device) { _allocator->DeviceFree(device, p); };
        if (!count_free_or_defer(ptr, device_index, do_free))
          return;
        drain_deferred_frees(do_free);
        _allocator->DeviceFree(device_index, ptr);
      }

      void clear_cache() override {
        _allocator->FreeAllCached();
      }

    private:
      std::unique_ptr<cub::CachingDeviceAllocator> _allocator;
    };

    // cudaMallocAsync draws from the device's current memory pool, whose release threshold
    // defaults to 0: every stream, event or device synchronization unmaps all the memory the
    // pool holds but does not use. A replica worker synchronizes whenever its job queue runs
    // empty (ReplicaWorker::idle), i.e. after every job of a synchronous client, so each job
    // maps its working set back in, and the unmap, done under the job queue lock, delays
    // the next job. The cost follows the bytes released: it is largest with big KV caches
    // (the flash-attention cache carries 512 spare time steps) and for a job that follows a
    // larger one. A positive CT2_CUDA_POOL_RELEASE_THRESHOLD lets the pool keep that many
    // bytes (in use plus cached) across synchronizations instead; clear_cache() (e.g.
    // unload_model) hands the kept memory back. Opt-in because the pool then holds its high
    // water mark between jobs; unset or 0 leaves the pool at the driver default.
    static uint64_t pool_release_threshold() {
      static const uint64_t threshold
        = read_byte_size_from_env("CT2_CUDA_POOL_RELEASE_THRESHOLD", 0);
      return threshold;
    }

    class CudaAsyncAllocator : public Allocator {
    public:
      static void free_on_stream(void* ptr, int device_index) {
#if CT2_USE_ASYNC_ALLOC
        int prev_device_index = -1;
        if (device_index >= 0) {
          CUDA_CHECK(cudaGetDevice(&prev_device_index));
          CUDA_CHECK(cudaSetDevice(device_index));
        }
        CUDA_CHECK(cudaFreeAsync(ptr, get_cuda_stream()));
        if (prev_device_index >= 0) {
          CUDA_CHECK(cudaSetDevice(prev_device_index));
        }
#else
        (void)ptr;
        (void)device_index;
#endif
      }

      void* allocate(size_t size, int device_index) override {
#if CT2_USE_ASYNC_ALLOC
        count_allocation_or_abort_capture(size);
        drain_deferred_frees(free_on_stream);
        int prev_device_index = -1;
        if (device_index >= 0) {
          CUDA_CHECK(cudaGetDevice(&prev_device_index));
          CUDA_CHECK(cudaSetDevice(device_index));
        }

        if (_release_threshold > 0)
          configure_pool(device_index);

        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size, get_cuda_stream()));

        if (prev_device_index >= 0) {
          CUDA_CHECK(cudaSetDevice(prev_device_index));
        }

        return ptr;
#else
        (void)size;
        (void)device_index;
        throw std::runtime_error("The asynchronous CUDA allocator requires CUDA >= 11.2");
#endif
      }

      void free(void* ptr, int device_index) override {
#if CT2_USE_ASYNC_ALLOC
        if (!count_free_or_defer(ptr, device_index, free_on_stream))
          return;
        drain_deferred_frees(free_on_stream);
        free_on_stream(ptr, device_index);
#else
        (void)ptr;
        (void)device_index;
        throw std::runtime_error("The asynchronous CUDA allocator requires CUDA >= 11.2");
#endif
      }

      void clear_cache() override {
#if CT2_USE_ASYNC_ALLOC
        // A pool with a raised release threshold keeps its unused memory across
        // synchronizations: hand it back now. A pool at the default threshold already
        // released it on the last synchronization.
        const std::lock_guard<std::mutex> lock(_pools_mutex);
        for (const auto& pool : _pools)
          CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));
#endif
      }

    private:
#if CT2_USE_ASYNC_ALLOC
      // Applies the release threshold to the current device's pool, the one cudaMallocAsync
      // draws from. The attribute write is idempotent, so the per-thread memo only saves
      // the calls and a race between threads configuring the same device is benign.
      void configure_pool(int device_index) {
        static thread_local std::vector<bool> configured;
        int device = device_index;
        if (device < 0)
          CUDA_CHECK(cudaGetDevice(&device));
        if (static_cast<size_t>(device) < configured.size() && configured[device])
          return;

        cudaMemPool_t pool;
        CUDA_CHECK(cudaDeviceGetMemPool(&pool, device));
        uint64_t threshold = _release_threshold;
        CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
        {
          const std::lock_guard<std::mutex> lock(_pools_mutex);
          if (std::find(_pools.begin(), _pools.end(), pool) == _pools.end()) {
            _pools.push_back(pool);
            spdlog::info("CUDA memory pool release threshold on device {}: {} bytes",
                         device, threshold);
          }
        }

        if (configured.size() <= static_cast<size_t>(device))
          configured.resize(device + 1, false);
        configured[device] = true;
      }

      const uint64_t _release_threshold = pool_release_threshold();
      std::mutex _pools_mutex;
      std::vector<cudaMemPool_t> _pools;
#endif
    };

    static bool support_cuda_malloc_async() {
#if !CT2_USE_ASYNC_ALLOC
      return false;
#else
      for (int i = 0; i < get_gpu_count(); ++i) {
        int supported = 0;
        cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, i);
        if (!supported)
          return false;
      }
      return true;
#endif
    }

    enum class CudaAllocator {
      CubCaching,
      MallocAsync,
    };

    static CudaAllocator resolve_cuda_allocator() {
      const bool cuda_malloc_async_is_supported = support_cuda_malloc_async();
      const auto allocator_name = read_string_from_env("CT2_CUDA_ALLOCATOR",
                                                       cuda_malloc_async_is_supported
                                                       ? "cuda_malloc_async"
                                                       : "cub_caching");

      CudaAllocator allocator = CudaAllocator::MallocAsync;

      if (allocator_name == "cub_caching") {
        allocator = CudaAllocator::CubCaching;
      } else if (allocator_name == "cuda_malloc_async") {
        if (!cuda_malloc_async_is_supported)
          throw std::runtime_error("The asynchronous CUDA allocator requires CUDA >= 11.2");
        allocator = CudaAllocator::MallocAsync;
      } else {
        throw std::invalid_argument("Invalid CUDA allocator " + allocator_name);
      }

      static std::once_flag log_once_flag;
      std::call_once(log_once_flag, [&allocator_name, allocator]() {
        spdlog::info("Using CUDA allocator: {}", allocator_name);
        if (allocator != CudaAllocator::MallocAsync && pool_release_threshold() > 0)
          spdlog::warn("CT2_CUDA_POOL_RELEASE_THRESHOLD only applies to the "
                       "cuda_malloc_async allocator and is ignored");
      });

      return allocator;
    }

  }

  template<>
  Allocator& get_allocator<Device::CUDA>() {
    static const cuda::CudaAllocator cuda_allocator = cuda::resolve_cuda_allocator();

    if (cuda_allocator == cuda::CudaAllocator::CubCaching) {
      // Use 1 allocator per thread for performance.
      static thread_local cuda::CubCachingAllocator allocator;
      return allocator;
    }

    static cuda::CudaAsyncAllocator allocator;
    return allocator;
  }

}
