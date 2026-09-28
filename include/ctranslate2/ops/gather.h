#pragma once

#include <vector>

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Which part of each row Gather::batch must copy, in elements of the tensor. The row
    // (one slice along axis 0) is read as `segments` segments of `length` elements, segment
    // k starting at element k * pitch of the row. length == 0 means the full row. Elements
    // outside the segments are left as they are in the output buffer (a shadow), which is
    // only correct when nothing reads them. keep_tail = true means the bytes outside the
    // segments must equal what a full-row gather would write: the segments are then used
    // only on a steady step, and any other step copies the full row.
    //
    // keep_tail requires uniform tails: every row of the tensor must hold the same bytes
    // outside the segments. On a steady step output row i keeps its own buffer's previous
    // tail, where a full-row gather would write the tail of source row indices[i]; the two
    // agree only when all tails are equal. The preallocated KV caches satisfy this because
    // their tails are all zero (append_to_cache zeroes on growth and appends write at the
    // offset). A writer that puts row-specific bytes past the segments breaks it silently.
    struct GatherRowSegments {
      dim_t segments = 1;
      dim_t length = 0;
      dim_t pitch = 0;
      bool keep_tail = false;
    };

    // Caller-owned output buffers for Gather::batch: bufs[i] pairs with data[i] by position.
    // last_src[i] / last_dst[i] record the buffers of the previous call's swap (the old
    // data buffer, now in bufs[i], and the new data buffer). A call is "steady" for entry i
    // when bufs[i] is still that old data buffer, data[i] still the new one, and neither
    // was re-allocated or re-shaped: the two buffers then carry the ping-pong history of
    // this one tensor. That alone does not make their tails match a full-row gather; it
    // does when the tails are uniform (see GatherRowSegments::keep_tail). segmented counts
    // the entries that copied their segments instead of full rows since the last clear().
    struct GatherShadows {
      std::vector<StorageView> bufs;
      std::vector<const void*> last_src;
      std::vector<const void*> last_dst;
      size_t segmented = 0;
      void clear() {
        bufs.clear();
        last_src.clear();
        last_dst.clear();
        segmented = 0;
      }
    };

    class Gather : public BinaryOp {
    public:
      Gather(const dim_t axis = 0, const dim_t batch_dims = 0);
      using BinaryOp::operator();

      void operator()(StorageView& data, const StorageView& input) const;
      void operator()(const StorageView& data,
                      const StorageView& input,
                      StorageView& output) const override;

      // Gathers each tensor along axis 0 with the same indices, replacing it like the
      // in-place operator() does. Used to reorder the decoder state after a beam search
      // step: on CUDA it launches one kernel per 32 tensors instead of one per tensor.
      // Tensors that do not qualify take the per-tensor path.
      static void batch(const std::vector<StorageView*>& data, const StorageView& input);

      // Same, but the fused CUDA path writes into caller-owned shadow buffers instead of
      // freshly allocated tensors: shadows->bufs[i] pairs with data[i] by position, and after
      // the gather the shadow and the data tensor swap buffers, so each pair ping-pongs
      // between two persistent allocations across steps instead of paying one
      // allocate/free pair per tensor per step. Gathering in place is impossible here
      // because beam indices repeat, so this two-buffer scheme is the minimal correct
      // form. Entries taking the per-tensor fallback skip their shadow; the CPU path
      // ignores the shadows entirely.
      static void batch(const std::vector<StorageView*>& data,
                        const StorageView& input,
                        GatherShadows* shadows);

      // Same, with an optional per-tensor segment descriptor (segs == nullptr, or a
      // default entry, copies full rows). A descriptor is honored only when it is
      // 16-byte aligned, shorter than the row and, with keep_tail, on a steady step;
      // otherwise the entry copies its full row. On CPU the shadows are used only when
      // segs is given (otherwise the per-tensor Gather() path runs, as above).
      static void batch(const std::vector<StorageView*>& data,
                        const StorageView& input,
                        GatherShadows* shadows,
                        const std::vector<GatherRowSegments>* segs);

    private:
      template <Device D, typename T>
      void compute(const StorageView& data,
                   const StorageView& input,
                   const dim_t axis,
                   const dim_t batch_dims,
                   StorageView& output) const;

      const dim_t _axis;
      const dim_t _batch_dims;
    };

  }
}
