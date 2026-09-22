#include "ctranslate2/ops/fill_ranges.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void FillRanges::operator()(StorageView& input,
                                const StorageView& ranges,
                                const float value) const {
      PROFILE("FillRanges");
      if (ranges.dim(0) == 0)
        return;

      DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(),
                               (compute<D, T>(input, ranges, value)));
    }

  }
}
