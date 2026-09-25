#pragma once

#include <vector>

#include "op.h"

namespace ctranslate2 {
  namespace ops {

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
      // freshly allocated tensors: (*shadows)[i] pairs with data[i] by position, and after
      // the gather the shadow and the data tensor swap buffers, so each pair ping-pongs
      // between two persistent allocations across steps instead of paying one
      // allocate/free pair per tensor per step. Gathering in place is impossible here
      // because beam indices repeat, so this two-buffer scheme is the minimal correct
      // form. Entries taking the per-tensor fallback skip their shadow; the CPU path
      // ignores the shadows entirely.
      static void batch(const std::vector<StorageView*>& data,
                        const StorageView& input,
                        std::vector<StorageView>* shadows);

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
