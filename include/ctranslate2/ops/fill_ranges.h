#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Writes a constant over contiguous spans of a 2D tensor's rows.
    //
    // Used to disable tokens. The alternative is one index per token, built on the
    // host and uploaded every step, which for a whole text vocabulary is 50k entries
    // and a 200 KB transfer.
    class FillRanges : public Op {
    public:
      // ranges is an int32 tensor of shape [n, 3], each row (batch_id, begin, end),
      // on the same device as input. The span is half-open.
      void operator()(StorageView& input,
                      const StorageView& ranges,
                      const float value) const;

    private:
      template <Device D, typename T>
      void compute(StorageView& input, const StorageView& ranges, const float value) const;
    };

  }
}
