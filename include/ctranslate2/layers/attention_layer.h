#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {
    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position);

    class RotaryEmbeddings;
    class Alibi;

    // Persistent temporaries for iterative decoding. A decoder owns one instance and
    // threads it down to its layers so the projection, attention and feed-forward buffers
    // created inside every decode step keep their allocation across steps: their shapes
    // are constant within a decode (the QK^T buffer aside), so StorageView::reserve keeps
    // the buffer and every step after the first becomes resize-only instead of paying one
    // allocate/free pair per temporary per layer per step.
    struct DecodeWorkspace {
      StorageView fused_proj;
      StorageView queries_proj;
      StorageView keys_proj;
      StorageView values_proj;
      StorageView attn;
      StorageView cross_context;
      StorageView ffn_inner;
      StorageView ffn_linear;

      // Returns the slot ready to stand in for a local StorageView(dtype, device). The
      // reassignment normally runs once, before the slot's first allocation; a later
      // dtype or device change (e.g. a replica reloaded differently) drops the buffer and
      // the slot re-allocates on its next use.
      static StorageView& prepare(StorageView& s, DataType dtype, Device device) {
        if (s.dtype() != dtype || s.device() != device)
          s = StorageView(dtype, device);
        return s;
      }
    };

    class AttentionLayer : public Layer
    {
    public:
      AttentionLayer(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false,
                         Alibi* alibi = nullptr,
                         bool is_flash_attn = false);
      virtual ~AttentionLayer() {};
      DataType output_type() const override;
      dim_t output_size() const override;
      virtual void operator()(const StorageView& queries,
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
                      dim_t offset = 0,
                      DecodeWorkspace* workspace = nullptr) const = 0;

      virtual bool has_positional_embeddings() const = 0;

      bool multi_query() const {
        return _multi_query;
      }

      // Whether the decoder self-attention cache managed by this layer is stored with
      // spare time capacity and written in place at the step offset (see append_to_cache
      // in attention.cc). Such caches do not carry their exact length in their shape.
      virtual bool preallocates_cache() const {
        return false;
      }

      static StorageView prepare_length_mask(const StorageView& lengths,
                                             const dim_t num_heads,
                                             const dim_t num_queries,
                                             const bool mask_future = false,
                                             const bool multi_query = false);

    protected:
      const bool _tensor_parallel;
      const dim_t _num_heads;
      const bool _self_attention;
      const bool _is_decoder;
      const std::vector<Dense> _linear;
      const dim_t _d_model;
      const dim_t _d_head;
      const bool _pre_norm;
      const std::unique_ptr<const LayerNorm> _layer_norm;
      const std::unique_ptr<RotaryEmbeddings> _rotary_embeddings;
      Alibi* _alibi;
      const float _queries_scale;
      const bool _multi_query;
      const dim_t _num_heads_kv;
      const dim_t _sliding_window;
    };

    enum class RotaryScalingType {
      None = -1,
      Linear,
      Su,
      Llama3,
    };

    class RotaryEmbeddings {
    public:
      RotaryEmbeddings(const dim_t dim = 0,
                       const bool interleave = true,
                       const RotaryScalingType scaling_type = RotaryScalingType::None,
                       const float scaling_factor = 1,
                       const float base = 10000,
                       const dim_t num_initial_positions = 2048,
                       const StorageView* long_scaling_factor = nullptr,
                       const StorageView* short_scaling_factor = nullptr,
                       const float low_freq_factor = 1.0,
                       const float high_freq_factor = 4.0,
                       const dim_t original_max_position_embeddings = 0,
                       const dim_t max_position_embeddings = 0,
                       const bool transpose = true);

      void apply(StorageView& x, const dim_t offset = 0, bool fa2 = false);

      StorageView& get_cos_half() {
        return *_cos_half;
      }

      StorageView& get_sin_half() {
        return *_sin_half;
      }

      bool get_interleave() const {
        return _interleave;
      }

    private:
      void initialize(const dim_t num_positions,
                      const dim_t dim,
                      const Device device,
                      const DataType dtype);

      const dim_t _dim;
      const bool _interleave;
      const RotaryScalingType _scaling_type;
      const float _scaling_factor;
      const float _base;
      const dim_t _num_initial_positions;
      std::unique_ptr<StorageView> _rotary_scaling_long_factor;
      std::unique_ptr<StorageView> _rotary_scaling_short_factor;
      const float _rotary_low_freq_factor;
      const float _rotary_high_freq_factor;
      const dim_t _original_max_position_embeddings;
      const dim_t _max_position_embeddings;
      const ops::Rotary _rotary_op;
      const bool _transpose;

      StorageView _sin;
      StorageView _cos;
      std::unique_ptr<StorageView> _sin_half;
      std::unique_ptr<StorageView> _cos_half;
    };


    class Alibi {
    public:
      Alibi(const bool use_positive_positions = false, const bool scale_alibi = false, const dim_t num_initial_positions = 2048);

      void apply(StorageView& x, const float scale = 1);

    private:
      const bool _use_positive_positions;
      const dim_t _num_initial_positions;
      const bool _scale_alibi;
      const ops::AlibiAdd _alibi_op;

      StorageView _alibi;
    };
  }
}
