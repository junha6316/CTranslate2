#pragma once

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/ops/gather.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    using DecoderState = std::unordered_map<std::string, StorageView>;

    void zero_first_timestep(StorageView& x, dim_t step);

    // Base class for decoders.
    class Decoder : public Layer {
    public:
      Decoder(Device device);

      virtual DecoderState initial_state(bool iterative_decoding = true) const = 0;

      // Forwards one step.
      virtual void operator()(dim_t step,
                              const StorageView& ids,
                              DecoderState& state,
                              StorageView* logits = nullptr,
                              StorageView* attention = nullptr) = 0;

      // Forwards a full sequence.
      virtual void operator()(const StorageView& ids,
                              const StorageView& lengths,
                              DecoderState& state,
                              StorageView& logits,
                              StorageView* attention = nullptr) = 0;

      // Update the decoder state in greedy search.
      void update_state(DecoderState& state, const StorageView& alive_batches) const;

      // Update the decoder state in beam search.
      void update_state(DecoderState& state,
                        const StorageView& beam_indices,
                        const dim_t beam_size,
                        const StorageView* alive_batches = nullptr) const;

      // Replicate the decoder state beam_size times.
      void replicate_state(DecoderState& state, const dim_t beam_size) const;

      // Returns true if the state must be replicated beam_size times.
      virtual bool replicate_state(const std::string& name) const;

      // Drops the beam-reorder shadow buffers (see _reorder_shadows). Called at the start
      // of every beam search; replicate_state also does it.
      void reset_reorder_shadows() const {
        _reorder_shadows.clear();
      }

      // Forces the prefix-bounded beam reorder (see reorder_segments) on or off for this
      // decoder, on any device. By default it is on for CUDA unless
      // CT2_CUDA_GATHER_PREFIX is 0 or false, and off on CPU; forcing it on is a test hook that runs
      // the CPU implementation of the segmented gather.
      void set_prefix_reorder(bool enable) {
        _prefix_reorder = enable ? 1 : 0;
      }

      // Restrict the output layer to a set of ids and/or resize it to a preferred size multiple.
      // Elements in restrict_ids must be unique and sorted.
      void update_output_layer(const dim_t size_multiple = 1,
                               const std::vector<size_t>& restrict_ids = {});

      bool output_layer_is_updated() const {
        return !_to_original_word_id.empty();
      }

      bool is_in_output(size_t word_id) const {
        return _to_output_word_id.find(word_id) != _to_output_word_id.end();
      }

      size_t to_output_word_id(size_t original_id) const {
        return _to_output_word_id.empty() ? original_id : _to_output_word_id.at(original_id);
      }

      size_t to_original_word_id(size_t output_id) const {
        return _to_original_word_id.empty() ? output_id : _to_original_word_id.at(output_id);
      }

      Device device() const {
        return _device;
      }

      DataType output_type() const override {
        return const_cast<Decoder&>(*this).output_layer().output_type();
      }

      dim_t output_size() const override {
        return const_cast<Decoder&>(*this).output_layer().output_size();
      }

    protected:
      // Returns the current batch size from the decoder state.
      virtual dim_t batch_size(const DecoderState& state) const;
      // Returns the output linear layer.
      virtual Dense& output_layer() = 0;
      // Beam reorder hook: when it returns true, the replicated state entry `name` (value
      // v) only needs the part of each row described by `out` copied (see
      // ops::GatherRowSegments), e.g. the valid prefix of a preallocated KV cache. The
      // default copies full rows.
      virtual bool reorder_segments(const std::string& name,
                                    const StorageView& v,
                                    const DecoderState& state,
                                    ops::GatherRowSegments& out) const {
        (void)name;
        (void)v;
        (void)state;
        (void)out;
        return false;
      }
      // The beam-reorder shadow buffers (read-only, for tests).
      const ops::GatherShadows& reorder_shadows() const {
        return _reorder_shadows;
      }

      const Device _device;

    private:
      std::vector<size_t> _to_original_word_id;
      std::unordered_map<size_t, size_t> _to_output_word_id;
      dim_t _vocabulary_size = 0;
      // Shadow buffers for the fused beam-reorder gather (ops::Gather::batch): entry i
      // pairs with the i-th replicated state entry by position in the batch call. No
      // explicit invalidation is needed: the gather resets a shadow on dtype mismatch,
      // resizes it on shape change, and tracks the buffers of its last swap, so a
      // mispairing after a state-map order change (or any foreign or re-allocated buffer)
      // only makes that step non-steady, which copies full rows — the prefix-bounded
      // reorder never relies on the tail bytes of a buffer it did not pair itself.
      // Mutable because update_state is const while the shadows are a pure allocation
      // cache.
      mutable ops::GatherShadows _reorder_shadows;
      // Reads CT2_CUDA_GATHER_PREFIX once (default true).
      static bool prefix_reorder_enabled();
      // -1: CUDA default (CT2_CUDA_GATHER_PREFIX), 0: forced off, 1: forced on.
      int _prefix_reorder = -1;
    };


    class DecoderStateCache {
    public:
      void save(std::vector<size_t> prompt, DecoderState state);
      const DecoderState* get(const std::vector<size_t>& prompt) const;

    private:
      std::map<std::vector<size_t>, DecoderState> _cache;
      mutable std::mutex _mutex;
    };

  }
}
