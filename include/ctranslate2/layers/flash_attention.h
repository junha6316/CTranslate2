#pragma once

#include "ctranslate2/layers/attention_layer.h"

namespace ctranslate2 {
  namespace layers {

    class RotaryEmbeddings;
    class Alibi;

    class FlashMultiHeadAttention : public AttentionLayer
    {
    public:
      FlashMultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false,
                         Alibi* alibi = nullptr);
      void operator()(const StorageView& queries,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention = nullptr,
                      const Padder* queries_padder = nullptr,
                      const Padder* values_padder = nullptr,
                      bool return_normalized_attention = true,
                      StorageView* position_bias = nullptr,
                      dim_t offset = 0) const override;

      virtual bool has_positional_embeddings() const override {
        return _encoder_fallback
          ? _encoder_fallback->has_positional_embeddings()
          : bool(_rotary_embeddings || _alibi);
      }

      // The decoder cache grows by _offset_free_space and the kernel writes in place at
      // the step offset, so its shape carries capacity, not length. A sliding window is
      // excluded like in MultiHeadAttention: the cache is then slid and truncated to the
      // window, so it no longer holds exactly one entry per decoded step. Encoder
      // instances manage no decoder cache.
      bool preallocates_cache() const override {
        return _is_decoder && _sliding_window == 0;
      }

    private:
      static void split_heads(StorageView& x,
                               dim_t num_heads,
                               const Padder* padder = nullptr,
                               dim_t beam_size = 1);

      static void combine_heads(StorageView& x,
                                dim_t num_heads,
                                const Padder* padder = nullptr,
                                dim_t beam_size = 1);

      const dim_t _cache_time_dim;
      const std::unique_ptr<AttentionLayer> _encoder_fallback;
      const bool _encoder_requires_fallback;
      static constexpr dim_t _offset_free_space{512};
    };
  }
}
