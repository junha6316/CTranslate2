#include "ctranslate2/ops/gather.h"

#include <algorithm>
#include <vector>

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/helpers.h"
#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Functor mapping output index to data index when axis == batch_dims.
    template <typename T>
    class batch_gather_index_map {
    private:
      const int32_t* _indices;
      const T _num_indices_per_batch;
      const T _batch_stride;
      const T _axis_stride;
    public:
      batch_gather_index_map(const int32_t* indices,
                             const T num_indices_per_batch,
                             const T batch_stride,
                             const T axis_stride)
        : _indices(indices)
        , _num_indices_per_batch(num_indices_per_batch)
        , _batch_stride(batch_stride)
        , _axis_stride(axis_stride)
      {
      }
      __device__
      T operator()(const T i) const {
        const T inner_index = i % _axis_stride;
        const T outer_index = i / _axis_stride;
        const T batch_index = outer_index / _num_indices_per_batch;
        return batch_index * _batch_stride + _indices[outer_index] * _axis_stride + inner_index;
      }
    };

    template <typename T, typename IndexMap>
    void run_gather(const IndexMap& index_map,
                    const T* src,
                    T* dst,
                    const dim_t dst_size) {
      auto gather_ids = thrust::make_transform_iterator(thrust::counting_iterator<cuda::index_t>(0),
                                                        index_map);
      THRUST_CALL(thrust::gather, gather_ids, gather_ids + dst_size, src, dst);
    }

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      const dim_t dst_size = output.size();
      const int32_t* indices = input.data<int32_t>();
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      if (axis == batch_dims) {
        const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
        const dim_t batch_size = data.size() / batch_stride;
        const dim_t num_indices_per_batch = input.size() / batch_size;
        const dim_t gather_size = data.stride(axis);
        const dim_t gather_bytes = gather_size * sizeof (T);

        if (gather_bytes % sizeof (uint4) == 0) {
          const dim_t dst_bytes = dst_size * sizeof (T);
          const dim_t batch_stride_bytes = batch_stride * sizeof (T);
          run_gather(batch_gather_index_map<cuda::index_t>(indices,
                                                           num_indices_per_batch,
                                                           batch_stride_bytes / sizeof (uint4),
                                                           gather_bytes / sizeof (uint4)),
                     reinterpret_cast<const uint4*>(src),
                     reinterpret_cast<uint4*>(dst),
                     dst_bytes / sizeof (uint4));
        } else {
          run_gather(batch_gather_index_map<cuda::index_t>(indices,
                                                           num_indices_per_batch,
                                                           batch_stride,
                                                           gather_size),
                     src, dst, dst_size);
        }

      } else {
        throw std::invalid_argument("Gather only supports indexing the first non batch dimension");
      }
    }

    constexpr int gather_batch_max_tensors = 32;

    // Passed by value as a kernel parameter, so the pointers need no upload.
    struct GatherRowsArgs {
      const uint4* src[gather_batch_max_tensors];
      uint4* dst[gather_batch_max_tensors];
      cuda::index_t row_size[gather_batch_max_tensors];
    };

    // blockIdx.y selects the tensor, the x dimension strides over its output words.
    __global__ void gather_rows_kernel(const GatherRowsArgs args,
                                       const int32_t* indices,
                                       const cuda::index_t num_indices) {
      const uint4* src = args.src[blockIdx.y];
      uint4* dst = args.dst[blockIdx.y];
      const cuda::index_t row_size = args.row_size[blockIdx.y];
      const cuda::index_t size = row_size * num_indices;
      for (cuda::index_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < size;
           i += gridDim.x * blockDim.x) {
        const cuda::index_t row = i / row_size;
        dst[i] = src[static_cast<cuda::index_t>(indices[row]) * row_size + i - row * row_size];
      }
    }

    void gather_rows_cuda(const std::vector<const void*>& src,
                          const std::vector<void*>& dst,
                          const std::vector<dim_t>& row_size,
                          const int32_t* indices,
                          const dim_t num_indices) {
      // gather.cc bounds the tensor size with threads * max_blocks, keep them in sync.
      constexpr dim_t threads = 256;
      constexpr dim_t max_blocks = 1024;

      for (size_t begin = 0; begin < src.size(); begin += gather_batch_max_tensors) {
        const size_t end = std::min(src.size(), begin + gather_batch_max_tensors);
        GatherRowsArgs args;
        dim_t max_size = 0;
        for (size_t t = begin; t < end; ++t) {
          args.src[t - begin] = static_cast<const uint4*>(src[t]);
          args.dst[t - begin] = static_cast<uint4*>(dst[t]);
          args.row_size[t - begin] = row_size[t];
          max_size = std::max(max_size, row_size[t] * num_indices);
        }

        const dim3 grid(std::min((max_size + threads - 1) / threads, max_blocks), end - begin);
        gather_rows_kernel<<<grid, threads, 0, cuda::get_cuda_stream()>>>(
          args, indices, num_indices);
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::CUDA, T>(const StorageView& data,           \
                                     const StorageView& input,          \
                                     const dim_t axis,                  \
                                     const dim_t batch_dims,            \
                                     StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
