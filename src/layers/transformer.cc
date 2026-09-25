#include "ctranslate2/layers/transformer.h"

#include <cmath>

namespace ctranslate2 {
  namespace layers {

    FeedForwardNetwork::FeedForwardNetwork(const models::Model& model,
                                           const std::string& scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _pre_norm(pre_norm)
      , _activation_type(activation_type)
      , _ff1(model, scope + "/linear_0", &_activation_type)
      , _ff1_noact(build_optional_layer<Dense>(model, scope + "/linear_0_noact"))
      , _ff2(model, scope + "/linear_1", nullptr, true)
      , _tensor_parallel(model.tensor_parallel()) {
    }

    void FeedForwardNetwork::operator()(const StorageView& input,
                                        StorageView& output,
                                        DecodeWorkspace* workspace) const {
      const StorageView* x = &input;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(input, output);
        x = &output;
      }

      const Device device = input.device();
      const DataType dtype = input.dtype();

      // The inner activations come from the decode workspace when one is passed so their
      // allocation survives across decoding steps.
      StorageView local_inner(dtype, device);
      StorageView& inner = workspace
        ? DecodeWorkspace::prepare(workspace->ffn_inner, dtype, device) : local_inner;
      _ff1(*x, inner);
      if (_ff1_noact) {
        StorageView local_linear(dtype, device);
        StorageView& linear = workspace
          ? DecodeWorkspace::prepare(workspace->ffn_linear, dtype, device) : local_linear;
        (*_ff1_noact)(*x, linear);
        ops::Mul()(linear, inner, inner);
      }

      _ff2(inner, output, _layer_norm ? &input : nullptr);

      if (_tensor_parallel) {
        Shape shape = output.shape();
        StorageView tmp(std::move(shape), output.dtype(), output.device());
        ops::ReduceAll red_op(ops::ReduceAll::RED_OP::SUM);
        red_op(output, tmp);
        output = std::move(tmp);
      }

      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_feedforward_layer_norm"))
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }


    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView* lengths,
                                             StorageView& output,
                                             const Padder* padder,
                                             StorageView* position_bias) const {
      PROFILE("TransformerEncoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      // Check if using pre_post_layer_norm pattern (T5Gemma style)
      const bool pre_post_layer_norm = _input_layer_norm && _post_attention_layer_norm
                                        && _pre_feedforward_layer_norm && _post_feedforward_layer_norm;

      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);

        (*_input_layer_norm)(input, hidden);

        if (_self_attention)
          (*_self_attention)(hidden,
                          hidden,
                          lengths,
                          context,
                          nullptr,
                          nullptr,
                          nullptr,
                          padder,
                          padder,
                          true,
                          position_bias);

        // post_self_attn_layernorm
        (*_post_attention_layer_norm)(context, output);

        // residual + hidden_states
        ops::Add()(input, output, output);

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        // mlp
        _ff(hidden, output);

        // post_feedforward_layernorm
        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);

        // residual + hidden_states
        ops::Add()(context, output, output);
        return;
      }

      // Original path for standard pre-norm/post-norm architectures
      StorageView context(dtype, device);
      if (_self_attention)
        (*_self_attention)(input,
                        input,
                        lengths,
                        context,
                        nullptr,
                        nullptr,
                        nullptr,
                        padder,
                        padder,
                        true,
                        position_bias);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     const bool use_flash_attention,
                                                     Alibi* alibi)
      : _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)))
      , _shared_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/shared_layer_norm"))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_feedforward_layer_norm"))
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm,
                                                                    /*is_decoder=*/true))
      , _ff(model, scope + "/ffn", pre_norm, activation_type)
      , _external_pre_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_pre_encoder_attention_layer_norm"))
      , _external_post_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_post_encoder_attention_layer_norm"))
      , _layer_scalar(model.get_attribute_with_default<float>(scope + "/layer_scalar", 1.f))
      , _has_merged_encoder_attention(
          !use_flash_attention
          && static_cast<MultiHeadAttention*>(_self_attention.get())->has_merged_encoder_attention())
      {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView* input_length,
                                             const StorageView* memory,
                                             const StorageView* memory_lengths,
                                             StorageView* cached_self_attn_keys,
                                             StorageView* cached_self_attn_values,
                                             StorageView* cached_attn_keys,
                                             StorageView* cached_attn_values,
                                             StorageView& output,
                                             StorageView* attention,
                                             const Padder* input_padder,
                                             const Padder* memory_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset,
                                             DecodeWorkspace* workspace) const {
      PROFILE("TransformerDecoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      const bool pre_post_layer_norm = _post_feedforward_layer_norm && _pre_feedforward_layer_norm;
      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);
        (*_input_layer_norm)(input, hidden);

        if (_has_merged_encoder_attention) {
          static_cast<MultiHeadAttention*>(_self_attention.get())->forward_merged(
            hidden,
            memory,
            memory_lengths,
            input_length,
            context,
            cached_self_attn_keys,
            cached_self_attn_values,
            cached_attn_keys,
            cached_attn_values,
            input_padder,
            memory_padder,
            offset);
        } else if (_self_attention) {
          (*_self_attention)(hidden,
                             hidden,
                             input_length,
                             context,
                             cached_self_attn_keys,
                             cached_self_attn_values,
                             nullptr,
                             input_padder,
                             input_padder,
                             true,
                             position_bias,
                             offset,
                             workspace);
        }
        (*_post_attention_layer_norm)(context, output);
        ops::Add()(output, input, output);

        if (_encoder_attention) {
            StorageView cross_attn_in = output;  // save for residual

            StorageView query_normalized(dtype, device);
            if (_external_pre_encoder_attention_layer_norm) {
                (*_external_pre_encoder_attention_layer_norm)(output, query_normalized);
            }
            else {
                query_normalized.shallow_copy(output);
            }

            (*_encoder_attention)(query_normalized,
                                  *memory,
                                  memory_lengths,
                                  context,
                                  cached_attn_keys,
                                  cached_attn_values,
                                  attention,
                                  input_padder,
                                  memory_padder,
                                  return_normalized_attention,
                                  nullptr,
                                  0,
                                  workspace);

            if (_external_post_encoder_attention_layer_norm) {
                (*_external_post_encoder_attention_layer_norm)(context, context);
            }
            ops::Add()(context, cross_attn_in, output);
        }

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        _ff(hidden, output, workspace);

        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);
        ops::Add()(output, context, output);

        // Gemma 4 layer scalar
        if (_layer_scalar != 1.f)
          ops::Mul()(output, StorageView(_layer_scalar).to(dtype), output);

        return;
      }

      const bool use_parallel_residual = _shared_layer_norm || _input_layer_norm;

      if (use_parallel_residual) {
        // The parallel residual implementation assumes there is no cross attention.
        StorageView hidden(dtype, device);

        if (_shared_layer_norm)
          (*_shared_layer_norm)(input, hidden);
        else
          (*_input_layer_norm)(input, hidden);

        StorageView attn(dtype, device);
        if (_self_attention)
          (*_self_attention)(hidden,
                        hidden,
                        input_length,
                        attn,
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        nullptr,
                        input_padder,
                        input_padder,
                        true,
                        position_bias,
                        offset,
                        workspace);

        if (_post_attention_layer_norm)
          (*_post_attention_layer_norm)(input, hidden);

        _ff(hidden, output, workspace);

        ops::Add()(output, input, output);
        ops::Add()(output, attn, output);

        return;
      }
      if (_self_attention)
        (*_self_attention)(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder,
                      true,
                      position_bias,
                      offset,
                      workspace);

      // The cross-attention output comes from the decode workspace when one is passed.
      // The move below when there is no encoder attention is a swap, so both the slot and
      // the caller's output keep a valid buffer of stable capacity.
      StorageView local_context(dtype, device);
      StorageView& context = workspace
        ? DecodeWorkspace::prepare(workspace->cross_context, dtype, device) : local_context;
      if (_encoder_attention) {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder,
                              return_normalized_attention,
                              nullptr,
                              0,
                              workspace);
      }
      else {
        context = std::move(output);
      }

      _ff(context, output, workspace);
    }


    static std::unique_ptr<PositionEncoder>
    build_position_encoder(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }

    static std::unique_ptr<const StorageView>
    build_embeddings_scale(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      const auto* scale = model.get_variable_if_exists(scope + "/scale_embeddings");

      // Backward compatibility with older models.
      if (!scale)
        scale = model.get_variable_if_exists(scope + "/embeddings/multiply_by_sqrt_depth");

      StorageView value;

      // The attribute can either be a boolean flag or the actual scale value.
      if (!scale || (scale->dtype() == DataType::INT8 && scale->as_scalar<int8_t>()))
        value = StorageView(std::sqrt(static_cast<float>(embeddings.output_size())));
      else if (scale->dtype() != DataType::INT8 && scale->as_scalar<float>() != 1.f)
        value = *scale;
      else
        return nullptr;

      return std::make_unique<StorageView>(value.to(embeddings.output_type()));
    }


    TransformerEncoder::TransformerEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings",
                    model.get_enum_value<EmbeddingsMerge>(scope + "/embeddings_merge"))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerEncoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation"),
                  _use_flash_attention))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _tensor_parallel(model.tensor_parallel())
    {
    }

    void TransformerEncoder::operator()(const std::vector<StorageView>& ids,
                                        const StorageView* lengths,
                                        StorageView& output) {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_embeddings_scale)
        ops::Mul()(input, *_embeddings_scale, input);
      if (_position_encoder)
        (*_position_encoder)(input);
      if (_layernorm_embedding)
        (*_layernorm_embedding)(input, input);

      const dim_t max_time = input.dim(1);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      std::unique_ptr<StorageView> lengths_mask;

      if (lengths) {
        if (Padder::allow_padding_removal(output.device(), _compute_type)) {
          padder = std::make_unique<Padder>(*lengths, max_time);
          padder->remove_padding(input);
        }

        int num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }
        lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(*lengths, num_heads, max_time));
      }

      StorageView position_bias(output.dtype(), output.device());

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask.get(), output, padder.get(), &position_bias);
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }


    static std::unique_ptr<Alibi> make_alibi(const models::Model& model, const std::string& scope) {
      const bool use_alibi = model.get_flag_with_default(scope + "/alibi", false);
      if (!use_alibi)
        return nullptr;

      const bool use_positive_positions = model.get_flag_with_default(
        scope + "/alibi_use_positive_positions", true);
      const bool scale_alibi = model.get_flag_with_default(
        scope + "/scale_alibi", false);

      return std::make_unique<Alibi>(use_positive_positions, scale_alibi);
    }

    // Bookkeeping entry recording the exact number of time steps written to the
    // preallocated self-attention caches (self_keys_i/self_values_i). Those tensors only
    // carry their capacity (round_up(length, kv_cache_block)) in their shape, so the true
    // length is recorded here instead. ONLY THE SHAPE of this entry is meaningful: it is
    // [batch, cached_length, 16] with dtype INT8 and its contents are never read.
    // - dim(0) tracks the batch x beam size like every other replicated state entry, so
    //   beam reorder, batch reduction, replication, tiling and per-batch slicing apply to
    //   it unchanged (they all operate on dim 0 only), and Decoder::batch_size stays
    //   correct if this entry lands at state.begin().
    // - dim(1) is the record: the number of valid steps in every self-attention cache.
    // - the trailing 16 keeps each row a multiple of 16 bytes so the entry stays on the
    //   fused one-kernel beam reorder path (ops::Gather::batch).
    // The name must not start with "memory" so replicate_state() treats it like the caches.
    static const char* kSelfCacheLengthState = "self_length";

    TransformerDecoder::TransformerDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _start_from_zero_embedding(model.get_flag_with_default(scope + "/start_from_zero_embedding",
                                                               false))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _project_in(build_optional_layer<Dense>(model, scope + "/project_in"))
      , _project_out(build_optional_layer<Dense>(model, scope + "/project_out"))
      , _alibi(make_alibi(model, scope))
      , _use_flash_attention(model.use_flash_attention())
      , _layers(build_layers_list<const TransformerDecoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation"),
                  _use_flash_attention,
                  _alibi.get()))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _proj(model, scope + "/projection")
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
      , _tensor_parallel(model.tensor_parallel())
      , _final_logit_softcapping(model.get_attribute_with_default<float>(scope + "/final_logit_softcapping", 0.f)) {

      dim_t alignment_layer = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_layer", -1));
      dim_t alignment_heads = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_heads", 1));

      if (alignment_layer < 0)
        alignment_layer = _layers.size() + alignment_layer;
      if (alignment_heads == 0)
        alignment_heads = _num_heads;

      set_alignment_heads(alignment_layer, alignment_heads);

      const auto* outputs_scale = model.get_variable_if_exists(scope + "/scale_outputs");
      if (outputs_scale) {
        const DataType dtype = get_default_float_type(_compute_type);
        _outputs_scale = std::make_unique<StorageView>(outputs_scale->to(dtype));
      }
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const {
      DecoderState state;

      if (iterative_decoding) {
        const size_t state_size = _layers.size() * (_with_encoder_attention ? 4 : 2) + 1;
        state.reserve(state_size);

        const DataType dtype = output_type();

        for (size_t i = 0; i < _layers.size(); ++i) {
          const std::string i_str = std::to_string(i);
          state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("self_values_" + i_str, StorageView(dtype, _device));
          if (_with_encoder_attention) {
            state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
            state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
          }
        }

        // The entry starts empty, which encodes a cached length of 0.
        if (!_layers.empty() && _layers.front()->get_self_attention().preallocates_cache())
          state.emplace(kSelfCacheLengthState, StorageView(DataType::INT8, _device));
      }

      return state;
    }

    bool TransformerDecoder::replicate_state(const std::string& name) const {
      // No need to replicate projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::set_alignment_heads(const dim_t layer,
                                                 const dim_t num_heads_to_average) {
      std::vector<dim_t> range(num_heads_to_average);
      std::iota(range.begin(), range.end(), dim_t(0));

      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      _alignment_heads[layer] = std::move(range);

      _average_alignment_heads = true;
      _alignment_heads_batch = -1;  // The cached device tensors describe the old heads.
    }

    void TransformerDecoder::set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads) {
      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      for (const auto& [layer, head] : alignment_heads)
        _alignment_heads[layer].push_back(head);

      _average_alignment_heads = false;
      _alignment_heads_batch = -1;  // The cached device tensors describe the old heads.
    }

    void TransformerDecoder::set_cache_reserve_steps(dim_t steps) {
      _cache_reserve_steps = steps;
      for (const auto& layer : _layers)
        layer->get_self_attention().set_cache_reserve_steps(steps);
    }

    const StorageView*
    TransformerDecoder::get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const {
      if (_alignment_heads.empty())
        return nullptr;

      const auto& heads = _alignment_heads[layer];
      const dim_t num_heads = heads.size();

      if (heads.empty())
        return nullptr;

      // batch x beam is constant within a decode, so the tensors are built once per
      // requested layer and reused by every following step instead of allocating and
      // uploading batch_size x num_heads indices per layer per step.
      if (batch_size != _alignment_heads_batch) {
        _alignment_heads_device.assign(_layers.size(), StorageView());
        _alignment_heads_batch = batch_size;
      }

      StorageView& cached = _alignment_heads_device[layer];
      if (cached.empty()) {
        std::vector<int32_t> indices;
        indices.reserve(batch_size * num_heads);
        for (dim_t i = 0; i < batch_size; ++i)
          indices.insert(indices.end(), heads.begin(), heads.end());
        cached = StorageView({batch_size, num_heads}, indices, _device);
      }

      return &cached;
    }

    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView& ids,
                                        DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      return decode(ids, nullptr, step, state, logits, attention);
    }

    void TransformerDecoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        DecoderState& state,
                                        StorageView& logits,
                                        StorageView* attention) {
      return decode(ids, &lengths, -1, state, &logits, attention);
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    dim_t step,
                                    DecoderState& state,
                                    StorageView* outputs,
                                    StorageView* attention,
                                    bool return_logits) {
      PROFILE("TransformerDecoder");
      const DataType dtype = output_type();
      const Device device = ids.device();
      const bool is_sequence = ids.rank() > 1;

      // The step must equal the number of steps already written to the preallocated
      // self-attention caches. Their shape only bounds the length to a kv_cache_block
      // window (see append_to_cache), so an off-by-a-few-steps counter would silently
      // corrupt attention: on the iterative path there is no softmax mask and correctness
      // rests on offset + steps being the exact cache length. The check runs before the
      // position encoder and rotary embeddings, which consume the same counter.
      const bool tracked_cache_length =
          (step >= 0 && _layers.front()->get_self_attention().preallocates_cache());
      if (tracked_cache_length) {
        const auto it = state.find(kSelfCacheLengthState);
        const dim_t cached_length =
            (it == state.end() || it->second.empty()) ? 0 : it->second.dim(1);
        if (step != cached_length)
          throw std::runtime_error("Decoding step " + std::to_string(step)
                                   + " does not match the number of steps already in the "
                                   "self-attention cache (" + std::to_string(cached_length)
                                   + ")");
      }

      // On the iterative path the two layer activations come from the decode workspace so
      // their buffers survive across steps; every move-assignment between them below is a
      // swap (StorageView move-assignment), keeping a deterministic two-slot ping-pong.
      // The chunking vector is bypassed on that path because
      // layer_ins.push_back(std::move(layer_in)) move-CONSTRUCTS into the vector: the
      // workspace buffer would transit through a function-local vector and die with it.
      // The !return_logits path keeps locals since *outputs = std::move(layer_in) at the
      // end would steal a slot; the sequence and sliding-window paths are unchanged.
      const bool reuse_layer_slots = step >= 0 && _sliding_window == 0 && return_logits;
      StorageView local_layer_in(dtype, device);
      StorageView local_layer_out(dtype, device);
      StorageView& layer_in = reuse_layer_slots
        ? DecodeWorkspace::prepare(_workspace.layer_in, dtype, device)
        : local_layer_in;
      StorageView& layer_out = reuse_layer_slots
        ? DecodeWorkspace::prepare(_workspace.layer_out, dtype, device)
        : local_layer_out;

      _embeddings(ids, layer_in);
      if (_start_from_zero_embedding)
        zero_first_timestep(layer_in, step);
      if (_embeddings_scale && (!_start_from_zero_embedding || step != 0))
        ops::Mul()(layer_in, *_embeddings_scale, layer_in);
      if (_project_in) {
        (*_project_in)(layer_in, layer_out);
        layer_in = std::move(layer_out);
      }
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));
      if (_layernorm_embedding)
        (*_layernorm_embedding)(layer_in, layer_in);

      const dim_t batch_size = layer_in.dim(0);
      dim_t max_time;

      if (_sliding_window > 0 && layer_in.dim(1) > _sliding_window) {
        max_time = _sliding_window;
      } else
        max_time = layer_in.dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (is_sequence && !lengths) {
        input_lengths = std::make_unique<StorageView>(Shape{ids.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }

        dim_t num_heads = _num_heads;
        if (_tensor_parallel) {
          num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
        }

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          *lengths,
          num_heads,
          max_time,
          /*mask_future=*/true,
          multi_query);


        if (step > 0)
          ops::Add()(lengths_mask, StorageView(int32_t(step)), lengths_mask);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

      StorageView* memory = nullptr;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        const auto it = state.find("memory_lengths");
        const StorageView* memory_lengths = it != state.end() ? &it->second : nullptr;

        if (step <= 0) {
          memory = &state.at("memory");

          if (memory_lengths && allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(*memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }

        if (memory_lengths) {
          dim_t num_heads = _num_heads;
          if (_tensor_parallel) {
            num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
          }
          const dim_t beam_size = batch_size / memory_lengths->dim(0);
          memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*memory_lengths,
                                                            num_heads,
                                                            beam_size > 1 ? beam_size : max_time));
        }
      }

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      StorageView position_bias(dtype, device);

      std::vector<StorageView> layer_ins;

      if (!reuse_layer_slots) {
        while (true) {
          dim_t prompt_size = layer_in.dim(1);
          if (_sliding_window == 0 || prompt_size <= _sliding_window || _use_flash_attention) {
            layer_ins.push_back(std::move(layer_in));
            break;
          }
          if (layer_in.dim(1) > _sliding_window) {
            StorageView tmp(dtype, device);
            const ops::Split split_op(1, {_sliding_window, prompt_size - _sliding_window});
            split_op(layer_in, tmp, layer_in);
            layer_ins.push_back(std::move(tmp));
          }
        }
      }

      const size_t num_chunks = reuse_layer_slots ? 1 : layer_ins.size();
      for (size_t i = 0; i < num_chunks; ++i) {
        StorageView* layer_in_chunk = reuse_layer_slots ? &layer_in : &layer_ins[i];
        for (size_t l = 0; l < _layers.size(); ++l) {
          StorageView* cached_self_attn_keys = nullptr;
          StorageView* cached_self_attn_values = nullptr;
          StorageView* cached_attn_keys = nullptr;
          StorageView* cached_attn_values = nullptr;

          if (step >= 0) {
            const std::string l_str = std::to_string(l);
            cached_self_attn_keys = &state.at("self_keys_" + l_str);
            cached_self_attn_values = &state.at("self_values_" + l_str);
            if (_with_encoder_attention) {
              cached_attn_keys = &state.at("memory_keys_" + l_str);
              cached_attn_values = &state.at("memory_values_" + l_str);
            }
          }

          // The selection is only consumed when attention weights are requested (see the
          // layer_attention gather below), so skip the lookup entirely otherwise.
          const StorageView* heads_to_select =
              attention ? get_layer_alignment_heads(l, batch_size) : nullptr;
          std::unique_ptr<StorageView> layer_attention;
          if (heads_to_select)
            layer_attention = std::make_unique<StorageView>(dtype, device);

          dim_t offset = _sliding_window * i + step;
          offset = offset < 0 ? 0 : offset;
          if (i > 0) {
            auto max_tokens = _sliding_window + layer_in_chunk->dim(1);
            StorageView tmp_lengths = StorageView(Shape{layer_in_chunk->dim(0)}, int32_t(max_tokens), device);
            int num_heads = _num_heads;
            if (_tensor_parallel) {
              num_heads = SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks());
            }
            StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
              tmp_lengths,
              num_heads,
              max_tokens,
              /*mask_future=*/true,
              multi_query);

            const ops::Slide slide_lengths_op(2, _sliding_window, layer_in_chunk->dim(1));
            // reuse tmp_lengths
            slide_lengths_op(lengths_mask, tmp_lengths);
            input_lengths_mask = std::make_unique<StorageView>(std::move(tmp_lengths));
          }

          (*_layers[l])(*layer_in_chunk,
                        input_lengths_mask.get(),
                        memory,
                        memory_lengths_mask.get(),
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        cached_attn_keys,
                        cached_attn_values,
                        layer_out,
                        layer_attention.get(),
                        input_padder.get(),
                        memory_padder.get(),
                        return_normalized_attention(),
                        &position_bias,
                        offset,
                        &_workspace);
          *layer_in_chunk = std::move(layer_out);

          if (layer_attention) {
            alignment_heads.emplace_back(dtype, device);
            ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
          }
        }
        if (!reuse_layer_slots)  // On the fast path the chunk already is layer_in.
          layer_in = std::move(*layer_in_chunk);
      }

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (tracked_cache_length) {
        auto it = state.find(kSelfCacheLengthState);
        if (it == state.end())  // State built outside initial_state: start tracking now.
          it = state.emplace(kSelfCacheLengthState, StorageView(DataType::INT8, _device)).first;
        StorageView& cache_length = it->second;
        const dim_t new_length = step + max_time;
        // Grow the buffer by blocks like the caches do, so the greedy path only touches
        // metadata on most steps (StorageView::resize keeps the buffer when the byte size
        // fits). No consumer ever reads past the current shape (gathers and tiles copy
        // shape-defined bytes only), so zeroing the spare region is hygiene, not
        // correctness; do it once per growth to leave no undefined bytes behind.
        const dim_t needed_bytes = batch_size * new_length * 16;  // INT8: 1 byte/element
        if (needed_bytes > cache_length.reserved_memory()) {
          // Mirror the caches' opt-in reserve so this entry also stops growing
          // mid-decode; dim(0) stays batch x beam and the rows stay 16 bytes, the
          // invariants that keep it on the fused one-kernel beam reorder path.
          const dim_t rounded = (std::max(new_length, _cache_reserve_steps) + 31) / 32 * 32;
          cache_length.resize({batch_size, rounded, 16});
          cache_length.zero();
        }
        cache_length.resize({batch_size, new_length, 16});
      }

      if (attention && !alignment_heads.empty()) {
        if (_average_alignment_heads) {
          ops::Mean(1)(alignment_heads[0], *attention);
          if (!is_sequence)
            attention->squeeze(1);

        } else {
          std::vector<const StorageView*> alignment_heads_ptr;
          alignment_heads_ptr.reserve(alignment_heads.size());
          for (const auto& heads : alignment_heads)
            alignment_heads_ptr.emplace_back(&heads);

          ops::Concat(1)(alignment_heads_ptr, *attention);
          if (!is_sequence)
            attention->squeeze(2);
        }
      }

      if (outputs) {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        if (_project_out) {
          (*_project_out)(layer_in, layer_out);
          layer_in = std::move(layer_out);
        }

        if (_outputs_scale)
          ops::Mul()(layer_in, *_outputs_scale, layer_in);

        if (return_logits) {
          _proj(layer_in, *outputs);
          if (_final_logit_softcapping != 0.f) {
            // logits = tanh(logits / cap) * cap  — squashes logits to (-cap, cap)
            const auto dtype = outputs->dtype();
            const auto device = outputs->device();
            ops::Mul()(*outputs, StorageView(1.f / _final_logit_softcapping).to(dtype), *outputs);
            ops::Tanh()(*outputs, *outputs);
            ops::Mul()(*outputs, StorageView(_final_logit_softcapping).to(dtype), *outputs);
          }
        } else
          *outputs = std::move(layer_in);

        if (!is_sequence)
          outputs->squeeze(1);
        else if (input_padder)
          input_padder->add_padding(*outputs);
      }
    }

  }
}
