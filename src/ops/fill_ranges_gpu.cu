#include "ctranslate2/ops/fill_ranges.h"

#include "type_dispatch.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    __global__ void fill_ranges_kernel(T* input,
                                       const int32_t* ranges,
                                       cuda::index_t row_size,
                                       float value) {
      const int32_t* r = ranges + blockIdx.x * 3;
      T* row = input + static_cast<cuda::index_t>(r[0]) * row_size;
      const cuda::index_t begin = r[1];
      const cuda::index_t end = r[2];

      const T v = static_cast<T>(value);
      for (cuda::index_t i = begin + threadIdx.x; i < end; i += blockDim.x)
        row[i] = v;
    }

    template <Device D, typename T>
    void FillRanges::compute(StorageView& input,
                             const StorageView& ranges,
                             const float value) const {
      fill_ranges_kernel<<<ranges.dim(0), 1024, 0, cuda::get_cuda_stream()>>>(
        cuda::device_cast(input.data<T>()),
        ranges.data<int32_t>(),
        input.dim(-1),
        value);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    FillRanges::compute<Device::CUDA, T>(StorageView&,                  \
                                         const StorageView&,            \
                                         const float) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
