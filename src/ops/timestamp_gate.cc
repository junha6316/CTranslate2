#include "ctranslate2/ops/timestamp_gate.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    TimestampGate::TimestampGate(const dim_t num_text_tokens,
                                 const dim_t num_timestamp_tokens)
      : _num_text_tokens(num_text_tokens)
      , _num_timestamp_tokens(num_timestamp_tokens)
    {
    }

    void TimestampGate::operator()(StorageView& logits,
                                   const StorageView& row_ids,
                                   const float disable_value) const {
      PROFILE("TimestampGate");
      if (row_ids.size() == 0)
        return;

      DEVICE_AND_FLOAT_DISPATCH("TimestampGate", logits.device(), logits.dtype(),
                                (compute<D, T>(logits, row_ids, disable_value)));
    }

  }
}
