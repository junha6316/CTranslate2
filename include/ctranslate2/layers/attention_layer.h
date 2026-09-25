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
      // The decoder's two layer activations: on the iterative decode path they ping-pong
      // between these two slots (every move-assignment between them is a swap), so both
      // buffers survive across layers and steps in a deterministic two-slot choreography.
      StorageView layer_in;
      StorageView layer_out;
      // Backs the head-transpose in split_heads/combine_heads on the beam>1 cross
      // attention path, so the transpose ping-pongs between two stable buffers
      // instead of replacing a slot's grown capacity with an exact-fit local every
      // step (which reintroduced ~48 allocate/free pairs per beam step).
      StorageView head_transpose;

      // Padded KV mode (CT2_CUDA_PAD_KV / CT2_CUDA_GRAPHS): the decoder sets padded_kv
      // for steps where the preallocated self-attention caches should be consumed at
      // their full capacity, with self_lengths (INT32, one entry per softmax row =
      // batch x beam x heads, all equal to the exact cached length) masking the spare
      // tail inside ops::SoftMax. This keeps every GEMM and softmax shape in the
      // decoder forward constant across the whole decode, a CUDA graph prerequisite.
      // Exactness: the spare cache capacity is zero-filled once at allocation, so the
      // padded K rows produce exactly-0 scores, the softmax lengths mask zeroes those
      // columns, and the padded V rows contribute 0 to the values matmul.
      bool padded_kv = false;
      StorageView self_lengths;
      // Report-back from the attention layers: set when padded_kv was requested but a
      // layer consumed the cache at its exact length instead (relative positions/bias,
      // alibi, or a self_lengths row count that does not match the cache layout). Such
      // a forward is NOT shape-constant across steps, so the decoder-level CUDA-graph
      // gate must not capture it and must disable replays for the decode (the pointer
      // fingerprint alone can miss a growing scratch reallocated within the same bin).
      bool padded_kv_fallback = false;
      // CUDA graphs: when graph_indirect is set the step-dependent offsets (position
      // encoding index, cache append offset) are read on the device from step_state
      // (INT32[2] = {step, cache_length_before_append}), so the captured kernels
      // replay correctly after the host re-seeds step_state each step.
      bool graph_indirect = false;
      StorageView step_state;
      // Logits shape recorded at capture time, restored on every replay (the replay
      // path skips the eager code that would otherwise resize the output).
      Shape graph_logits_shape;

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

      // Opt-in: when non-zero, a self-attention cache growing from empty is sized for
      // this many steps up front, so the block-by-block growth copies disappear and the
      // cache address stays fixed for the whole decode (a CUDA-graph capture
      // prerequisite). const with a mutable member because decoders only hold const layer
      // references and each model replica runs single-threaded.
      void set_cache_reserve_steps(dim_t steps) const {
        _cache_reserve = steps;
      }

      dim_t cache_reserve_steps() const {
        return _cache_reserve;
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
      mutable dim_t _cache_reserve = 0;
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
