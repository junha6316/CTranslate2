#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class TopK : public Op {
    public:
      TopK(dim_t k, dim_t axis = -1);
      void operator()(const StorageView& x, StorageView& values, StorageView& indices) const;

      // Same, with optional caller-owned scratch for the CUDA kernel's two temporary
      // reduction buffers, whose size is constant across decoding steps: with scratch the
      // allocation happens once instead of one allocate/free pair each per call. The CPU
      // implementation ignores the scratch.
      void operator()(const StorageView& x,
                      StorageView& values,
                      StorageView& indices,
                      StorageView* scratch_ids,
                      StorageView* scratch_vals) const;

    private:
      dim_t _k;

      template <Device D, typename DataType, typename IndexType>
      void compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices,
                   StorageView* scratch_ids,
                   StorageView* scratch_vals) const;

    };

  }
}
