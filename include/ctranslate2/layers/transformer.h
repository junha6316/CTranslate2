#pragma once

#include <algorithm>
#include <vector>

#include "ctranslate2/layers/attention.h"
#include "ctranslate2/layers/flash_attention.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace cuda {
    class DecoderGraphRunner;
  }

  namespace layers {

    // Capacity ladder for the CUDA-graph decode path (CT2_CUDA_GRAPHS_TIERS, parsed by
    // models::parse_cache_tier_policy). When a decode outgrows its KV capacity C, the
    // decoder's host guard asks next(C) for the capacity to grow to, so the graph runner
    // can re-capture at the larger shapes instead of running the rest of the decode
    // eager. Either a relative stride (next = min(top, C + stride) while C < top) or an
    // explicit ascending list of block-rounded tiers (next = the first tier > C). 0 means
    // "no next tier": the decode keeps today's behaviour (eager tail, 32-block growth).
    // A default-constructed policy is inactive and next() always returns 0.
    struct CacheTierPolicy {
      dim_t stride = 0;
      std::vector<dim_t> tiers;  // Ascending, unique; only used when stride == 0.
      dim_t top = 0;

      bool active() const {
        return stride > 0 ? top > 0 : !tiers.empty();
      }

      dim_t next(dim_t capacity) const {
        if (stride > 0)
          return capacity < top ? std::min(top, capacity + stride) : 0;
        for (const dim_t tier : tiers) {
          if (tier > capacity)
            return tier;
        }
        return 0;
      }
    };

    class FeedForwardNetwork : public Layer
    {
    public:
      FeedForwardNetwork(const models::Model& model,
                         const std::string& scope,
                         const bool pre_norm = true,
                         const ops::ActivationType activation_type = ops::ActivationType::ReLU);

      void operator()(const StorageView& input,
                      StorageView& output,
                      DecodeWorkspace* workspace = nullptr) const;

      DataType output_type() const override {
        return _ff2.output_type();
      }

      dim_t output_size() const override {
        return _ff2.output_size();
      }

    private:
      const std::unique_ptr<const LayerNorm> _layer_norm;
      const bool _pre_norm;
      const ops::ActivationType _activation_type;
      const Dense _ff1;
      const std::unique_ptr<const Dense> _ff1_noact;
      const Dense _ff2;
      const bool _tensor_parallel;
    };

    class TransformerEncoderLayer : public Layer
    {
    public:
      TransformerEncoderLayer(const models::Model& model,
                              const std::string& scope,
                              const dim_t num_heads,
                              const bool pre_norm = true,
                              const ops::ActivationType activation_type = ops::ActivationType::ReLU,
                              bool use_flash_attention = false);

      void operator()(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output,
                      const Padder* padder = nullptr,
                      StorageView* position_bias = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

      const AttentionLayer& get_self_attention() const {
        return *_self_attention;
      }

    private:
      std::unique_ptr<AttentionLayer> _self_attention;
      const std::unique_ptr<const LayerNorm> _input_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_attention_layer_norm;
      const std::unique_ptr<const LayerNorm> _pre_feedforward_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_feedforward_layer_norm;
      const FeedForwardNetwork _ff;
    };

    class TransformerDecoderLayer : public Layer
    {
    public:
      TransformerDecoderLayer(const models::Model& model,
                              const std::string& scope,
                              const dim_t num_heads,
                              const bool pre_norm = true,
                              const ops::ActivationType activation_type = ops::ActivationType::ReLU,
                              const bool use_flash_attention = true,
                              Alibi* alibi = nullptr);

      void operator()(const StorageView& input,
                      const StorageView* input_lengths,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView* cached_self_attn_keys,
                      StorageView* cached_self_attn_values,
                      StorageView* cached_attn_keys,
                      StorageView* cached_attn_values,
                      StorageView& output,
                      StorageView* attention = nullptr,
                      const Padder* input_padder = nullptr,
                      const Padder* memory_padder = nullptr,
                      bool return_normalized_attention = true,
                      StorageView* position_bias = nullptr,
                      dim_t offset = 0,
                      DecodeWorkspace* workspace = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

      bool has_cross_attention() const {
        return bool(_encoder_attention) || _has_merged_encoder_attention;
      }

      const AttentionLayer& get_self_attention() const {
        return *_self_attention;
      }

    private:
      const std::unique_ptr<AttentionLayer> _self_attention;
      const std::unique_ptr<const LayerNorm> _shared_layer_norm;
      const std::unique_ptr<const LayerNorm> _input_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_attention_layer_norm;
      const std::unique_ptr<const LayerNorm> _pre_feedforward_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_feedforward_layer_norm;
      const std::unique_ptr<const AttentionLayer> _encoder_attention;
      const FeedForwardNetwork _ff;
      const std::unique_ptr<const LayerNorm> _external_pre_encoder_attention_layer_norm;
      const std::unique_ptr<const LayerNorm> _external_post_encoder_attention_layer_norm;
      const float _layer_scalar;
      const bool _has_merged_encoder_attention;
    };

    class TransformerEncoder : public Encoder
    {
    public:
      TransformerEncoder(const models::Model& model, const std::string& scope);

      void operator()(const std::vector<StorageView>& ids,
                      const StorageView* lengths,
                      StorageView& output) override;

      size_t num_input_features() const override {
        return _embeddings.num_inputs();
      }

      DataType output_type() const override {
        return _layers.back()->output_type();
      }

      dim_t output_size() const override {
        return _layers.back()->output_size();
      }

    private:
      const ParallelEmbeddings _embeddings;
      const std::unique_ptr<const StorageView> _embeddings_scale;
      const dim_t _num_heads;
      const ComputeType _compute_type;
      const std::unique_ptr<const LayerNorm> _layernorm_embedding;
      const std::unique_ptr<const LayerNorm> _output_norm;
      const bool _use_flash_attention;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const bool _tensor_parallel;
    };

    class TransformerDecoder : public Decoder
    {
    public:
      TransformerDecoder(const models::Model& model, const std::string& scope);
      ~TransformerDecoder() override;

      DecoderState initial_state(bool iterative_decoding = true) const override;
      bool replicate_state(const std::string& name) const override;

      void operator()(dim_t step,
                      const StorageView& ids,
                      DecoderState& state,
                      StorageView* logits = nullptr,
                      StorageView* attention = nullptr) override;
      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      DecoderState& state,
                      StorageView& logits,
                      StorageView* attention = nullptr) override;

      void set_alignment_heads(const dim_t layer, const dim_t num_heads_to_average);
      void set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads);

      // Opt-in: preallocate the self-attention caches (and the self_length record) for
      // this many decoding steps so their buffers and addresses stay fixed for the whole
      // decode. Costs the full cache memory up front; 0 restores block-by-block growth.
      // This is the decode's base reserve: a capacity tier (see set_cache_tier_policy)
      // raises the reserve for the rest of one decode only, and the next call here or the
      // next decode boundary (a prompt/sequence or step-0 forward) restores this value.
      // Also a decode boundary for the CUDA graph runner.
      void set_cache_reserve_steps(dim_t steps);

      // Capacity tiers for the CUDA-graph decode path (opt-in, inactive by default). When
      // a graph-eligible step outgrows the KV capacity and the policy names a next tier,
      // the crossing step runs eager at the next tier's reserve and the graph runner
      // re-captures at the new shapes instead of disabling replays for the rest of the
      // decode. Only consulted on the CUDA graph path.
      void set_cache_tier_policy(CacheTierPolicy policy);

      // Returns the device tensor selecting the alignment heads of this layer, or nullptr
      // when the layer has none. The tensor is cached: its content only depends on the
      // layer and the batch size, so it is rebuilt lazily per layer when the batch size
      // changes instead of allocating and uploading on every decode step.
      const StorageView*
      get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const;

      virtual bool return_normalized_attention() const {
        return true;
      }

    protected:
      Dense& output_layer() override {
        return _proj;
      }

      void decode(const StorageView& ids,
                  const StorageView* lengths,
                  dim_t step,
                  DecoderState& state,
                  StorageView* outputs = nullptr,
                  StorageView* attention = nullptr,
                  bool return_logits = true);

      // Sets the reserve used by the next cache growth, in BOTH places that read it: the
      // per-layer self-attention reserve (append_to_cache) and _cache_reserve_steps (the
      // self_length record rounding and the padded-KV gate). Unlike
      // set_cache_reserve_steps it neither changes the base reserve nor touches the graph
      // runner, so a capacity tier can raise the reserve mid-decode.
      void apply_cache_reserve(dim_t steps);

      const dim_t _num_heads;
      const ComputeType _compute_type;
      const Embeddings _embeddings;
      const bool _start_from_zero_embedding;
      const std::unique_ptr<const StorageView> _embeddings_scale;
      std::unique_ptr<const StorageView> _outputs_scale;
      const std::unique_ptr<const LayerNorm> _layernorm_embedding;
      const std::unique_ptr<const LayerNorm> _output_norm;
      const std::unique_ptr<const Dense> _project_in;
      const std::unique_ptr<const Dense> _project_out;
      const std::unique_ptr<Alibi> _alibi;
      const bool _use_flash_attention;
      const std::vector<std::unique_ptr<const TransformerDecoderLayer>> _layers;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const bool _with_encoder_attention;
      std::vector<std::vector<dim_t>> _alignment_heads;
      bool _average_alignment_heads;
      // Cache for get_layer_alignment_heads, keyed by the batch size (mutable for the
      // same single-thread-per-replica reason as _workspace below).
      mutable std::vector<StorageView> _alignment_heads_device;
      mutable dim_t _alignment_heads_batch = -1;
      // Current reserve (the base, or a capacity tier raised mid-decode) and the base
      // reserve set by set_cache_reserve_steps, restored at every decode boundary.
      dim_t _cache_reserve_steps = 0;
      dim_t _cache_reserve_base = 0;
      CacheTierPolicy _cache_tier_policy;
      Dense _proj;
      const dim_t _sliding_window;
      const bool _tensor_parallel;
      const float _final_logit_softcapping;
      // Layer temporaries reused across decode steps. A plain member is safe for the
      // same reason decode() and operator() are non-const: each model replica is only
      // ever run by one thread at a time (update_output_layer already relies on this).
      DecodeWorkspace _workspace;
      // CUDA graph capture/replay controller for the iterative decode path (created
      // lazily; always null in CPU builds and when CT2_CUDA_GRAPHS is not set).
      std::unique_ptr<cuda::DecoderGraphRunner> _graph_runner;
    };

  }
}
