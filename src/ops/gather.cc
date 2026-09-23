#include "ctranslate2/ops/gather.h"

#include <algorithm>
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
#ifdef CT2_WITH_CUDA
      if (input.device() == Device::CUDA) {
        PROFILE("GatherBatch");
        constexpr dim_t word_size = 16;  // sizeof (uint4)
        std::vector<StorageView*> gathered;
        std::vector<StorageView> outputs;
        std::vector<const void*> src;
        std::vector<void*> dst;
        std::vector<dim_t> row_size;
        outputs.reserve(data.size());  // dst holds pointers into these buffers

        for (StorageView* value : data) {
          if (value->device() != Device::CUDA || value->empty()) {
            Gather()(*value, input);
            continue;
          }
          const dim_t row_bytes = value->stride(0) * value->item_size();
          // The kernel indexes words with 32-bit integers, on both sides of the copy, and
          // its grid-stride loop must not wrap past the last word.
          const dim_t words = row_bytes / word_size * std::max(value->dim(0), input.size());
          if (row_bytes % word_size != 0
              || words > std::numeric_limits<uint32_t>::max() - gather_rows_max_threads) {
            Gather()(*value, input);
            continue;
          }

          outputs.emplace_back(compute_output_shape(*value, input, 0),
                               value->dtype(),
                               Device::CUDA);
          gathered.push_back(value);
          src.push_back(value->buffer());
          dst.push_back(outputs.back().buffer());
          row_size.push_back(row_bytes / word_size);
        }

        if (!gathered.empty() && input.size() > 0)
          gather_rows_cuda(src, dst, row_size, input.data<int32_t>(), input.size());
        for (size_t i = 0; i < gathered.size(); ++i)
          *gathered[i] = std::move(outputs[i]);
        return;
      }
#endif

      for (StorageView* value : data)
        Gather()(*value, input);
    }

  }
}
