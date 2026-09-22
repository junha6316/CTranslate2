#include "ctranslate2/ops/fill_ranges.h"

#include <algorithm>

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void FillRanges::compute(StorageView& input,
                             const StorageView& ranges,
                             const float value) const {
      const dim_t row_size = input.dim(-1);
      const int32_t* r = ranges.data<int32_t>();

      for (dim_t i = 0; i < ranges.dim(0); ++i) {
        T* row = input.data<T>() + r[i * 3] * row_size;
        std::fill(row + r[i * 3 + 1], row + r[i * 3 + 2], static_cast<T>(value));
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    FillRanges::compute<Device::CPU, T>(StorageView&,                   \
                                        const StorageView&,             \
                                        const float) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
