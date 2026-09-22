#include "ctranslate2/ops/timestamp_gate.h"

#include <cub/block/block_reduce.cuh>

#include "type_dispatch.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    constexpr int timestamp_gate_threads = 256;

    // One block per row. The block reduces the text tokens to a max and the timestamp
    // tokens to a logsumexp, compares them, and masks the text tokens if the timestamp
    // side wins. Both scalars stay in shared memory, so nothing is read back to host.
    template <typename T>
    __global__ void timestamp_gate_kernel(T* logits,
                                          const int32_t* row_ids,
                                          cuda::index_t vocabulary_size,
                                          cuda::index_t num_text_tokens,
                                          cuda::index_t num_timestamp_tokens,
                                          float disable_value) {
      typedef cub::BlockReduce<float, timestamp_gate_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp;
      __shared__ float shared_max_text;
      __shared__ float shared_max_timestamp;
      __shared__ float shared_sum_timestamp;

      T* row = logits + static_cast<cuda::index_t>(row_ids[blockIdx.x]) * vocabulary_size;
      const T* timestamps = row + num_text_tokens;

      // max over the text tokens
      float local = -INFINITY;
      for (cuda::index_t i = threadIdx.x; i < num_text_tokens; i += blockDim.x)
        local = fmaxf(local, static_cast<float>(row[i]));
      float reduced = BlockReduce(temp).Reduce(local, cub::Max());
      if (threadIdx.x == 0)
        shared_max_text = reduced;

      // max over the timestamp tokens, to shift the exponentials
      __syncthreads();
      local = -INFINITY;
      for (cuda::index_t i = threadIdx.x; i < num_timestamp_tokens; i += blockDim.x)
        local = fmaxf(local, static_cast<float>(timestamps[i]));
      reduced = BlockReduce(temp).Reduce(local, cub::Max());
      if (threadIdx.x == 0)
        shared_max_timestamp = reduced;

      // sum of the shifted exponentials
      __syncthreads();
      const float shift = shared_max_timestamp;
      local = 0.f;
      for (cuda::index_t i = threadIdx.x; i < num_timestamp_tokens; i += blockDim.x)
        local += expf(static_cast<float>(timestamps[i]) - shift);
      reduced = BlockReduce(temp).Sum(local);
      if (threadIdx.x == 0)
        shared_sum_timestamp = reduced;

      __syncthreads();
      const float timestamp_log_prob = shift + logf(shared_sum_timestamp);
      if (timestamp_log_prob <= shared_max_text)
        return;

      const T value = static_cast<T>(disable_value);
      for (cuda::index_t i = threadIdx.x; i < num_text_tokens; i += blockDim.x)
        row[i] = value;
    }

    template <Device D, typename T>
    void TimestampGate::compute(StorageView& logits,
                                const StorageView& row_ids,
                                const float disable_value) const {
      timestamp_gate_kernel<<<row_ids.size(), timestamp_gate_threads, 0,
                              cuda::get_cuda_stream()>>>(
        cuda::device_cast(logits.data<T>()),
        row_ids.data<int32_t>(),
        logits.dim(-1),
        _num_text_tokens,
        _num_timestamp_tokens,
        disable_value);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TimestampGate::compute<Device::CUDA, T>(StorageView&,               \
                                            const StorageView&,         \
                                            const float) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
