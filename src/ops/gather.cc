#include "ctranslate2/ops/gather.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    static inline Shape compute_output_shape(const StorageView& data,
                                             const StorageView& input,
                                             const dim_t axis) {
      Shape output_shape(input.shape());
      for (dim_t i = axis + 1; i < data.rank(); ++i)
        output_shape.push_back(data.dim(i));
      return output_shape;
    }

    static bool support_gather_batch_inplace(const StorageView& data, const StorageView& input) {
      // We can gather in place if the output is not larger than data and indices are in
      // strictly increasing order (i.e. we never need to gather from a previous index).
      const auto* input_begin = input.data<int32_t>();
      const auto* input_end = input_begin + input.size();
      return (input.device() == Device::CPU
              && input.size() <= data.dim(0)
              && std::adjacent_find(input_begin, input_end, std::greater_equal<int32_t>()) == input_end);
    }

    template <typename T>
    void gather_batch_inplace(StorageView& data, const StorageView& input) {
      const auto* indices = input.data<int32_t>();
      auto* dst = data.data<T>();
      const auto* src = dst;
      const auto copy_dim = data.stride(0);
      for (dim_t i = 0; i < input.size(); ++i) {
        const dim_t index = indices[i];
        if (index != i)
          primitives<Device::CPU>::copy(src + index * copy_dim, dst, copy_dim);
        dst += copy_dim;
      }
    }


#ifdef CT2_WITH_CUDA
    // Defined in gather_gpu.cu. Copies rows of 16-byte words: dst[t] row i is src[t] row
    // indices[i], with a row of row_size[t] words.
    void gather_rows_cuda(const std::vector<const void*>& src,
                          const std::vector<void*>& dst,
                          const std::vector<dim_t>& row_size,
                          const int32_t* indices,
                          dim_t num_indices);
    // Same, copying per row of tensor t only segs[t] segments of seg_words[t] words, the
    // k-th starting at word k * seg_pitch[t] of the row (row_words[t] words).
    void gather_rows_cuda(const std::vector<const void*>& src,
                          const std::vector<void*>& dst,
                          const std::vector<dim_t>& row_words,
                          const std::vector<dim_t>& seg_words,
                          const std::vector<dim_t>& segs,
                          const std::vector<dim_t>& seg_pitch,
                          const int32_t* indices,
                          dim_t num_indices);
    // Upper bound on the threads of one gather_rows_cuda launch (blocks x threads).
    constexpr dim_t gather_rows_max_threads = 1024 * 256;
#endif

    Gather::Gather(const dim_t axis, const dim_t batch_dims)
      : _axis(axis)
      , _batch_dims(batch_dims) {
    }

    void Gather::operator()(StorageView& data, const StorageView& input) const {
      if (_axis == 0 && _batch_dims == 0 && support_gather_batch_inplace(data, input)) {
        PROFILE("Gather");
        TYPE_DISPATCH(data.dtype(), (gather_batch_inplace<T>(data, input)));
        data.resize(compute_output_shape(data, input, _axis));
      } else {
        StorageView clone(std::move(data));
        operator()(clone, input, data);
      }
    }

    void Gather::operator()(const StorageView& data,
                            const StorageView& input,
                            StorageView& output) const {
      PROFILE("Gather");

      if (_batch_dims > 0) {
        if (data.rank() < _batch_dims)
          throw std::invalid_argument("Gather: rank of data should greater than or equal to "
                                      + std::to_string(_batch_dims));
        if (input.rank() < _batch_dims)
          throw std::invalid_argument("Gather: rank of input should greater than or equal to "
                                      + std::to_string(_batch_dims));

        const auto& data_shape = data.shape();
        const auto& input_shape = input.shape();
        if (!std::equal(data_shape.begin(),
                        data_shape.begin() + _batch_dims,
                        input_shape.begin()))
          throw std::invalid_argument("Gather: first " + std::to_string(_batch_dims)
                                      + " dimensions of data and input should match");
      }

      const dim_t axis = _axis < 0 ? data.rank() + _axis : _axis;
      output.resize(compute_output_shape(data, input, axis));
      DEVICE_AND_TYPE_DISPATCH(data.device(), data.dtype(),
                               (compute<D, T>(data, input, axis, _batch_dims, output)));
    }

    void Gather::batch(const std::vector<StorageView*>& data, const StorageView& input) {
      batch(data, input, nullptr, nullptr);
    }

    void Gather::batch(const std::vector<StorageView*>& data,
                       const StorageView& input,
                       GatherShadows* shadows) {
      batch(data, input, shadows, nullptr);
    }

    // Whether the descriptor of one tensor can replace its full row this call. Fresh
    // (non-shadow) outputs always take full rows: their bytes outside the segments would
    // be undefined.
    static bool use_row_segments(const GatherRowSegments* seg,
                                 const StorageView& value,
                                 const bool have_shadow,
                                 const bool steady) {
      if (!seg || !have_shadow || seg->length <= 0 || seg->segments < 1)
        return false;
      if (seg->keep_tail && !steady)
        return false;
      const dim_t row = value.stride(0);
      const dim_t span = seg->segments == 1 ? row : seg->pitch;
      if (seg->length >= span
          || (seg->segments - 1) * seg->pitch + seg->length > row)
        return false;
      const dim_t item = value.item_size();
      return (seg->length * item) % 16 == 0 && (seg->pitch * item) % 16 == 0;
    }

    // Prepares shadow n as the gather output of value (same rules on CPU and CUDA) and
    // returns whether this call is steady for it (see GatherShadows).
    static bool prepare_shadow(GatherShadows& shadows,
                               const size_t n,
                               const StorageView& value,
                               const StorageView& input) {
      StorageView& s = shadows.bufs[n];
      // Same reset rule as DecodeWorkspace::prepare: a dtype or device change drops
      // the buffer and the shadow re-allocates below.
      if (s.dtype() != value.dtype() || s.device() != value.device())
        s = StorageView(value.dtype(), value.device());
      const void* prev_buf = s.buffer();
      const Shape prev_shape = s.shape();
      // Reserve the source's reserved byte capacity, not the exact output bytes: the
      // source buffers are block-rounded (see append_to_cache and the self_length
      // entry), while the gathered rows of the self_length entry grow every step — an
      // exact-fit shadow would re-allocate on each of those steps.
      s.reserve((value.reserved_memory() + s.item_size() - 1) / s.item_size());
      s.resize(compute_output_shape(value, input, 0));
      return (s.buffer() == prev_buf
              && prev_shape == s.shape()
              && value.buffer() == shadows.last_dst[n]
              && s.buffer() == shadows.last_src[n]);
    }

    static void sync_shadow_slots(GatherShadows* shadows, const size_t size) {
      if (!shadows)
        return;
      if (shadows->bufs.size() != size)
        shadows->bufs.resize(size);
      shadows->last_src.resize(size, nullptr);
      shadows->last_dst.resize(size, nullptr);
    }

    void Gather::batch(const std::vector<StorageView*>& data,
                       const StorageView& input,
                       GatherShadows* shadows,
                       const std::vector<GatherRowSegments>* segs) {
      const auto segment_of = [segs](const size_t n) -> const GatherRowSegments* {
        return segs && n < segs->size() ? &(*segs)[n] : nullptr;
      };

#ifdef CT2_WITH_CUDA
      if (input.device() == Device::CUDA) {
        PROFILE("GatherBatch");
        constexpr dim_t word_size = 16;  // sizeof (uint4)
        std::vector<StorageView*> gathered;
        std::vector<StorageView*> gathered_out;
        std::vector<size_t> gathered_slot;
        std::vector<StorageView> outputs;
        std::vector<const void*> src;
        std::vector<void*> dst;
        std::vector<dim_t> row_words, seg_words, seg_count, seg_pitch;
        outputs.reserve(data.size());  // dst holds pointers into these buffers
        sync_shadow_slots(shadows, data.size());

        for (size_t n = 0; n < data.size(); ++n) {
          StorageView* value = data[n];
          // Entries taking a per-tensor fallback leave their slot untracked: only the
          // swap below makes the next call steady for a slot.
          const auto untrack = [shadows, n]() {
            if (shadows) {
              shadows->last_src[n] = nullptr;
              shadows->last_dst[n] = nullptr;
            }
          };
          if (value->device() != Device::CUDA || value->empty()) {
            untrack();
            Gather()(*value, input);
            continue;
          }
          const dim_t row_bytes = value->stride(0) * value->item_size();
          // The kernel indexes words with 32-bit integers, on both sides of the copy, and
          // its grid-stride loop must not wrap past the last word. Full-row words bound
          // the segmented walk too.
          const dim_t words = row_bytes / word_size * std::max(value->dim(0), input.size());
          if (row_bytes % word_size != 0
              || words > std::numeric_limits<uint32_t>::max() - gather_rows_max_threads) {
            untrack();
            Gather()(*value, input);
            continue;
          }

          StorageView* out;
          bool steady = false;
          if (shadows) {
            steady = prepare_shadow(*shadows, n, *value, input);
            out = &shadows->bufs[n];
          } else {
            outputs.emplace_back(compute_output_shape(*value, input, 0),
                                 value->dtype(),
                                 Device::CUDA);
            out = &outputs.back();
          }
          const GatherRowSegments* seg = segment_of(n);
          const dim_t row = row_bytes / word_size;
          gathered.push_back(value);
          gathered_out.push_back(out);
          gathered_slot.push_back(n);
          src.push_back(value->buffer());
          dst.push_back(out->buffer());
          row_words.push_back(row);
          if (use_row_segments(seg, *value, shadows != nullptr, steady)) {
            ++shadows->segmented;
            const dim_t item = value->item_size();
            seg_words.push_back(seg->length * item / word_size);
            seg_count.push_back(seg->segments);
            seg_pitch.push_back(seg->pitch * item / word_size);
          } else {
            seg_words.push_back(row);
            seg_count.push_back(1);
            seg_pitch.push_back(row);
          }
        }

        if (!gathered.empty() && input.size() > 0)
          gather_rows_cuda(src, dst, row_words, seg_words, seg_count, seg_pitch,
                           input.data<int32_t>(), input.size());
        // A swap (move-assignment already is one, storage_view.cc): the data tensor takes
        // the gather output and the old buffer lands in the output slot — with shadows
        // that buffer is exactly what the next step reuses.
        for (size_t i = 0; i < gathered.size(); ++i) {
          std::swap(*gathered[i], *gathered_out[i]);
          if (shadows) {
            shadows->last_src[gathered_slot[i]] = gathered_out[i]->buffer();
            shadows->last_dst[gathered_slot[i]] = gathered[i]->buffer();
          }
        }
        return;
      }
#endif

      if (!segs || !shadows) {
        for (StorageView* value : data)
          Gather()(*value, input);
        return;
      }

      // Host path of the segmented gather. Production decoding reaches it only when the
      // decoder forces the prefix reorder on (Decoder::set_prefix_reorder, a test hook);
      // it follows the CUDA path's shadow and steady rules with plain memcpy.
      sync_shadow_slots(shadows, data.size());
      const int32_t* indices = input.data<int32_t>();
      for (size_t n = 0; n < data.size(); ++n) {
        StorageView* value = data[n];
        if (value->device() != Device::CPU || value->empty()) {
          shadows->last_src[n] = nullptr;
          shadows->last_dst[n] = nullptr;
          Gather()(*value, input);
          continue;
        }
        const bool steady = prepare_shadow(*shadows, n, *value, input);
        StorageView& out = shadows->bufs[n];
        const GatherRowSegments* seg = segment_of(n);
        const dim_t item = value->item_size();
        const dim_t row_bytes = value->stride(0) * item;
        const auto* src = static_cast<const char*>(value->buffer());
        auto* dst = static_cast<char*>(out.buffer());
        const bool segmented = use_row_segments(seg, *value, true, steady);
        shadows->segmented += segmented;
        for (dim_t i = 0; i < input.size(); ++i) {
          const char* src_row = src + static_cast<dim_t>(indices[i]) * row_bytes;
          char* dst_row = dst + i * row_bytes;
          if (!segmented) {
            std::memcpy(dst_row, src_row, row_bytes);
            continue;
          }
          for (dim_t k = 0; k < seg->segments; ++k)
            std::memcpy(dst_row + k * seg->pitch * item,
                        src_row + k * seg->pitch * item,
                        seg->length * item);
        }
        std::swap(*value, out);
        shadows->last_src[n] = out.buffer();
        shadows->last_dst[n] = value->buffer();
      }
    }

  }
}
