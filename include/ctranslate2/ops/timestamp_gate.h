#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Decides, for selected rows of the Whisper logits, whether a timestamp token
    // should be sampled, and masks the text tokens of those rows when it should.
    //
    // The decision compares a max over the text tokens against a logsumexp over the
    // timestamp tokens. Doing it here keeps both scalars on the device: reading them
    // back per row costs a blocking device-to-host round trip per reduction, and that
    // round trip, not the arithmetic, is what the operation costs on GPU.
    class TimestampGate : public Op {
    public:
      TimestampGate(const dim_t num_text_tokens, const dim_t num_timestamp_tokens);

      // row_ids holds int32 row indices into logits, on the same device as logits.
      void operator()(StorageView& logits,
                      const StorageView& row_ids,
                      const float disable_value) const;

    private:
      template <Device D, typename T>
      void compute(StorageView& logits,
                   const StorageView& row_ids,
                   const float disable_value) const;

      const dim_t _num_text_tokens;
      const dim_t _num_timestamp_tokens;
    };

  }
}
