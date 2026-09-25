#include "ctranslate2/ops/topk.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    TopK::TopK(dim_t k, dim_t axis)
      : _k(k) {
      if (axis != -1)
        throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
    }

    void TopK::operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
      operator()(x, values, indices, nullptr, nullptr);
    }

    void TopK::operator()(const StorageView& x,
                          StorageView& values,
                          StorageView& indices,
                          StorageView* scratch_ids,
                          StorageView* scratch_vals) const {
      PROFILE("TopK");
      const dim_t batch_size = x.size() / x.dim(-1);
      values.resize({batch_size, _k});
      indices.resize({batch_size, _k});

      // Fix the scratch dtypes here so the device implementation can reserve() in
      // elements of the right size; a dtype or device change drops the buffer like
      // DecodeWorkspace::prepare does.
      if (scratch_ids && scratch_vals && x.device() != Device::CPU) {
        if (scratch_ids->dtype() != DataType::INT32 || scratch_ids->device() != x.device())
          *scratch_ids = StorageView(DataType::INT32, x.device());
        if (scratch_vals->dtype() != x.dtype() || scratch_vals->device() != x.device())
          *scratch_vals = StorageView(x.dtype(), x.device());
      }

      DEVICE_AND_FLOAT_DISPATCH("TopK", x.device(), x.dtype(),
                                (compute<D, T, int32_t>(x, values, indices,
                                                        scratch_ids, scratch_vals)));
    }

  }
}
