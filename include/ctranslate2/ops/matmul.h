#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class MatMul : public BinaryOp {
    public:
      MatMul(bool trans_a = false, bool trans_b = false, float alpha = 1);
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const;

      // Same, but only the first "b_rows" rows of each matrix in b take part in the
      // product. The rows that follow are still allocated: they are skipped by keeping
      // the batch stride of b derived from its real shape. b_rows == 0 means "all rows".
      // This lets a KV cache be stored with spare capacity in its time dimension.
      void operator()(const StorageView& a,
                      const StorageView& b,
                      StorageView& c,
                      dim_t b_rows) const;

    private:
      bool _trans_a;
      bool _trans_b;
      float _alpha;

      template <Device D, typename T>
      void compute(const StorageView& a,
                   const StorageView& b,
                   StorageView& c,
                   dim_t b_rows) const;
    };

  }
}
