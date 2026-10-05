#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // Piecewise CUDA graphs (CT2_CUDA_GRAPHS=1 with CT2_CUDA_GRAPHS_PIECEWISE=1): the
    // decoder step is captured as one graph segment per layer boundary, and each layer's
    // self-attention core (QK^T -> softmax -> AV at the exact cached length) runs eagerly
    // between the segments, with the same ops and arguments as the stock path. CoreDesc
    // is what a replay needs to re-issue one core: the buffers are the ones the captured
    // segments bake (fixed for the decode, see DecodeWorkspace::reserve_core_slots), only
    // the length changes from step to step.
    struct CoreDesc {
      DataType dtype = DataType::FLOAT32;
      Device device = Device::CPU;
      dim_t index = -1;         // Core order within the step (= decoder layer index).
      dim_t batch = 0;          // Cache dim 0 (batch x beam).
      dim_t heads = 0;          // Cache dim 1.
      dim_t capacity = 0;       // Cache dim 2 (C): the K/V view is [batch, heads, C, depth].
      dim_t depth = 0;          // Cache dim 3.
      dim_t length = 0;         // Exact cached length at capture time (step + 1).
      float queries_scale = 1;
      void* queries = nullptr;  // [batch, heads, 1, depth], read only.
      void* keys = nullptr;     // Self-attention cache, read only.
      void* values = nullptr;   // Self-attention cache, read only.
      void* scores = nullptr;   // QK^T output, softmax in place ([batch, heads, 1, length]).
      void* context = nullptr;  // AV output [batch, heads, 1, depth].
      dim_t scores_bytes = 0;   // reserved_memory() of the scores buffer at capture.
      dim_t context_bytes = 0;  // reserved_memory() of the context buffer at capture.
    };

    // Called by the decoder self-attention layers at every core boundary of a step that
    // runs piecewise (DecodeWorkspace::graph_hook). Both calls throw std::runtime_error on
    // failure; the decoder then discards the step's capture and reruns the step eagerly.
    class SegmentHook {
    public:
      virtual ~SegmentHook() = default;
      // After the cache append, right before the core: closes the running segment (a
      // capture hook ends the stream capture, instantiates and launches the segment so the
      // eager core reads its outputs) and records the core.
      virtual void end_segment(const CoreDesc& core) = 0;
      // Right after the core ran eagerly: opens the next segment.
      virtual void begin_segment() = 0;
    };

    // Builds the descriptor of the core about to run on the live tensors. scores_slot is the
    // buffer the QK^T product will be written to (the fused_proj slot at core entry) and
    // context_slot the one the AV product will land in (the attn slot at core entry, see the
    // swap in dot_product_attention). Throws std::runtime_error when the shapes are not
    // [B, H, 1, D] queries / [B, H, C, D] caches or when scores_slot cannot hold B*H*C
    // elements (missing reserve_core_slots): a replay at a later, longer step would overflow.
    CoreDesc make_core_desc(const StorageView& queries,
                            const StorageView& keys_cache,
                            const StorageView& values_cache,
                            dim_t length,
                            float queries_scale,
                            const StorageView& scores_slot,
                            const StorageView& context_slot,
                            dim_t index);

    // Re-issues the core at "length" (the step's exact cached length) on the recorded
    // buffers with the ops and arguments of the stock path:
    //   ops::MatMul(false, true, queries_scale)(q, K, scores, length);
    //   ops::SoftMax()(scores, nullptr, scores);
    //   ops::MatMul()(scores, V, context, length);
    // Returns false without touching memory when length is not in [1, capacity] or the
    // scores buffer is too small, and false after the fact if an op moved a buffer (a view
    // resized past its extent re-allocates silently, see StorageView::view/reserve).
    bool replay_core(const CoreDesc& core, dim_t length);

    // The opt-in KV modes of the single-token decode step on CUDA, as selected by the env:
    //   CT2_CUDA_PAD_KV=1 or CT2_CUDA_GRAPHS=1         -> Padded (full-capacity attention
    //                                                   with a device lengths row),
    //   CT2_CUDA_GRAPHS=1 + CT2_CUDA_GRAPHS_PIECEWISE=1 -> ExactCore (piecewise graphs; the
    //                                                   padded mode is not used, PAD_KV
    //                                                   included),
    //   anything else, another device or an ineligible step -> Stock.
    // The decoder (transformer.cc) and the CUDA graph runner (piecewise_enabled) both
    // derive the piecewise sub-mode from decode_kv_env(), so they cannot disagree.
    enum class DecodeKvMode { Stock, Padded, ExactCore };

    struct DecodeKvEnv {
      bool graphs = false;     // CT2_CUDA_GRAPHS
      bool piecewise = false;  // CT2_CUDA_GRAPHS_PIECEWISE
      bool pad_kv = false;     // CT2_CUDA_PAD_KV

      bool piecewise_graphs() const {
        return graphs && piecewise;
      }
    };

    // The env switches, read once per process.
    const DecodeKvEnv& decode_kv_env();

    // eligible: the step passes the decoder's KV-mode gate (single-token iterative step
    // with preallocated caches, no mask, attention weights, tensor parallelism or memory
    // lengths). The CPU test hook (TransformerDecoder::_piecewise_test_hook) is applied by
    // the caller on top of Stock.
    DecodeKvMode select_decode_kv_mode(const DecodeKvEnv& env, Device device, bool eligible);

  }
}
