#include "layers/decode_core.h"

#include <stdexcept>
#include <string>

#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/softmax.h"
#include "env.h"

namespace ctranslate2 {
  namespace layers {

    static std::string shape_to_string(const StorageView& x) {
      std::string s = "[";
      for (dim_t i = 0; i < x.rank(); ++i)
        s += (i > 0 ? "," : "") + std::to_string(x.dim(i));
      return s + "]";
    }

    CoreDesc make_core_desc(const StorageView& queries,
                            const StorageView& keys_cache,
                            const StorageView& values_cache,
                            const dim_t length,
                            const float queries_scale,
                            const StorageView& scores_slot,
                            const StorageView& context_slot,
                            const dim_t index) {
      if (queries.rank() != 4 || keys_cache.rank() != 4
          || keys_cache.shape() != values_cache.shape()
          || queries.dim(0) != keys_cache.dim(0) || queries.dim(1) != keys_cache.dim(1)
          || queries.dim(2) != 1 || queries.dim(3) != keys_cache.dim(3))
        throw std::runtime_error("piecewise: unexpected self-attention core shapes (queries "
                                 + shape_to_string(queries) + ", keys "
                                 + shape_to_string(keys_cache) + ", values "
                                 + shape_to_string(values_cache) + ")");
      const DataType dtype = queries.dtype();
      const Device device = queries.device();
      for (const StorageView* x : {&keys_cache, &values_cache, &scores_slot, &context_slot}) {
        if (x->dtype() != dtype || x->device() != device)
          throw std::runtime_error("piecewise: self-attention core operands disagree on "
                                   "dtype or device");
      }

      CoreDesc core;
      core.dtype = dtype;
      core.device = device;
      core.index = index;
      core.batch = keys_cache.dim(0);
      core.heads = keys_cache.dim(1);
      core.capacity = keys_cache.dim(2);
      core.depth = keys_cache.dim(3);
      core.length = length;
      core.queries_scale = queries_scale;
      if (length < 1 || length > core.capacity)
        throw std::runtime_error("piecewise: cached length " + std::to_string(length)
                                 + " is outside the cache capacity "
                                 + std::to_string(core.capacity));

      // A replay at a later step writes up to B*H*C scores into the buffer baked now, so
      // it must already hold them: the slot is reserved before the forward
      // (DecodeWorkspace::reserve_core_slots), never grown by the core itself.
      const dim_t item = queries.item_size();
      const dim_t rows = core.batch * core.heads;
      core.scores_bytes = scores_slot.reserved_memory();
      core.context_bytes = context_slot.reserved_memory();
      if (core.scores_bytes < rows * core.capacity * item
          || core.context_bytes < rows * core.depth * item)
        throw std::runtime_error("piecewise: the self-attention core slots are not reserved "
                                 "for the cache capacity (scores "
                                 + std::to_string(core.scores_bytes) + " bytes, need "
                                 + std::to_string(rows * core.capacity * item) + ")");

      core.queries = const_cast<void*>(queries.buffer());
      core.keys = const_cast<void*>(keys_cache.buffer());
      core.values = const_cast<void*>(values_cache.buffer());
      core.scores = const_cast<void*>(scores_slot.buffer());
      core.context = const_cast<void*>(context_slot.buffer());
      if (!core.queries || !core.keys || !core.values || !core.scores || !core.context)
        throw std::runtime_error("piecewise: a self-attention core operand has no buffer");
      return core;
    }

    bool replay_core(const CoreDesc& core, const dim_t length) {
      if (length < 1 || length > core.capacity
          || !core.queries || !core.keys || !core.values || !core.scores || !core.context)
        return false;

      // Non-owning views on the recorded buffers. A view's extent is its shape, and each op
      // below resizes its output to the very same shape, which keeps the buffer.
      const dim_t B = core.batch;
      const dim_t H = core.heads;
      const dim_t D = core.depth;
      StorageView queries(core.dtype, core.device);
      StorageView keys(core.dtype, core.device);
      StorageView values(core.dtype, core.device);
      StorageView scores(core.dtype, core.device);
      StorageView context(core.dtype, core.device);
      const dim_t item = queries.item_size();
      if (B * H * length * item > core.scores_bytes || B * H * D * item > core.context_bytes)
        return false;

      queries.view(core.queries, {B, H, 1, D});
      keys.view(core.keys, {B, H, core.capacity, D});
      values.view(core.values, {B, H, core.capacity, D});
      scores.view(core.scores, {B, H, 1, length});
      context.view(core.context, {B, H, 1, D});

      // Same ops and arguments as dot_product_attention on the exact-length path
      // (attention.cc): b_rows = length bounds both products to the cached steps.
      try {
        ops::MatMul(/*trans_a=*/false, /*trans_b=*/true, core.queries_scale)(
          queries, keys, scores, length);
        const StorageView* no_lengths = nullptr;  // The stock call passes values_lengths == nullptr.
        ops::SoftMax()(scores, no_lengths, scores);
        ops::MatMul()(scores, values, context, length);
      } catch (const std::exception&) {
        return false;
      }

      return scores.buffer() == core.scores && context.buffer() == core.context;
    }

    const DecodeKvEnv& decode_kv_env() {
      static const DecodeKvEnv env = [] {
        DecodeKvEnv e;
        e.graphs = read_bool_from_env("CT2_CUDA_GRAPHS");
        e.piecewise = read_bool_from_env("CT2_CUDA_GRAPHS_PIECEWISE");
        e.pad_kv = read_bool_from_env("CT2_CUDA_PAD_KV");
        return e;
      }();
      return env;
    }

    DecodeKvMode select_decode_kv_mode(const DecodeKvEnv& env,
                                       const Device device,
                                       const bool eligible) {
      if (!eligible || device != Device::CUDA)
        return DecodeKvMode::Stock;
      if (env.piecewise_graphs())
        return DecodeKvMode::ExactCore;
      if (env.pad_kv || env.graphs)
        return DecodeKvMode::Padded;
      return DecodeKvMode::Stock;
    }

  }
}
