#include "ctranslate2/ops/timestamp_gate.h"

#include "ctranslate2/primitives.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void TimestampGate::compute(StorageView& logits,
                                const StorageView& row_ids,
                                const float disable_value) const {
      const dim_t vocabulary_size = logits.dim(-1);
      const int32_t* rows = row_ids.data<int32_t>();

      for (dim_t i = 0; i < row_ids.size(); ++i) {
        T* row = logits.data<T>() + rows[i] * vocabulary_size;

        const float max_text = primitives<D>::max(row, _num_text_tokens);
        const float timestamp = primitives<D>::logsumexp(row + _num_text_tokens,
                                                         _num_timestamp_tokens);
        if (timestamp > max_text)
          primitives<D>::fill(row, static_cast<T>(disable_value), _num_text_tokens);
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TimestampGate::compute<Device::CPU, T>(StorageView&,                \
                                           const StorageView&,          \
                                           const float) const;

    DECLARE_IMPL(float)

  }
}
