#include "test_utils.h"
#include "ctranslate2/primitives.h"
#include "dispatch.h"

class PrimitiveTest : public ::testing::TestWithParam<Device> {
};

TEST_P(PrimitiveTest, IndirectVariants) {
  // The device-indirect variants (offset read from a tensor at kernel time, used
  // under CUDA graphs) must match their host-offset counterparts exactly.
  const Device device = GetParam();
  const dim_t depth = 8;
  const dim_t rows = 3;
  const dim_t capacity = 6;
  const int32_t offset = 2;
  const StorageView offset_view({1}, std::vector<int32_t>{offset}, device);

  // copy_2d_indirect against copy_2d shifted by offset * depth.
  std::vector<float> src_values(rows * depth);
  for (size_t i = 0; i < src_values.size(); ++i)
    src_values[i] = float(i) + 1;
  const StorageView src({rows, 1, depth}, src_values, device);
  StorageView dst_direct({rows, capacity, depth}, 0.f, device);
  StorageView dst_indirect({rows, capacity, depth}, 0.f, device);
  DEVICE_DISPATCH(device,
                  primitives<D>::copy_2d(src.data<float>(), depth,
                                         dst_direct.data<float>() + offset * depth,
                                         capacity * depth, depth, rows));
  DEVICE_DISPATCH(device,
                  primitives<D>::copy_2d_indirect(src.data<float>(), depth,
                                                  dst_indirect.data<float>(),
                                                  capacity * depth, depth, rows,
                                                  offset_view.data<int32_t>(), depth));
  expect_storage_eq(dst_indirect, dst_direct);

  // add_batch_broadcast_indirect against add_batch_broadcast at base + offset * depth.
  std::vector<float> base_values(5 * depth);
  for (size_t i = 0; i < base_values.size(); ++i)
    base_values[i] = float(i) * 0.5f;
  const StorageView base({5, depth}, base_values, device);
  StorageView y_direct({rows, depth}, 1.f, device);
  StorageView y_indirect({rows, depth}, 1.f, device);
  DEVICE_DISPATCH(device,
                  primitives<D>::add_batch_broadcast(base.data<float>() + offset * depth,
                                                     y_direct.data<float>(),
                                                     depth, rows * depth));
  DEVICE_DISPATCH(device,
                  primitives<D>::add_batch_broadcast_indirect(base.data<float>(),
                                                              offset_view.data<int32_t>(),
                                                              depth,
                                                              y_indirect.data<float>(),
                                                              rows * depth));
  expect_storage_eq(y_indirect, y_direct);
}

TEST_P(PrimitiveTest, StridedFill) {
  const Device device = GetParam();
  StorageView x({3, 2}, float(0), device);
  StorageView expected({3, 2}, std::vector<float>{1, 0, 1, 0, 1, 0}, device);
  DEVICE_DISPATCH(device, primitives<D>::strided_fill(x.data<float>(), 1.f, 2, 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, IndexedFill) {
  const Device device = GetParam();
  StorageView x({6}, float(0), device);
  StorageView ids({3}, std::vector<int32_t>{0, 2, 5}, device);
  StorageView expected({6}, std::vector<float>{1, 0, 1, 0, 0, 1}, device);
  DEVICE_DISPATCH(device, primitives<D>::indexed_fill(x.data<float>(), 1.f, ids.data<int32_t>(), 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, LogSumExp) {
  const Device device = GetParam();
  StorageView x({8}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2}, device);
  float result = 0;
  DEVICE_DISPATCH(device, result = primitives<D>::logsumexp(x.data<float>(), x.size()));
  EXPECT_NEAR(result, 2.1908040046691895, 1e-6);
}

TEST_P(PrimitiveTest, PenalizePreviousTokens) {
  const Device device = GetParam();
  const float penalty = 1.2f;
  StorageView scores({2, 4}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2});
  StorageView previous_ids({2, 2}, std::vector<int32_t>{2, 2, 1, 2}, device);
  StorageView previous_scores({2, 2}, std::vector<float>{-1.2, -1.2, 0.5, -1.3}, device);
  StorageView expected = scores;
  expected.at<float>({0, 2}) *= penalty;
  expected.at<float>({1, 1}) /= penalty;
  expected.at<float>({1, 2}) *= penalty;
  scores = scores.to(device);
  DEVICE_DISPATCH(device, primitives<D>::penalize_previous_tokens(scores.data<float>(),
                                                                  previous_scores.data<float>(),
                                                                  previous_ids.data<int32_t>(),
                                                                  penalty,
                                                                  scores.dim(0),
                                                                  previous_ids.dim(1),
                                                                  scores.dim(1)));
  expect_storage_eq(scores, expected);
}

INSTANTIATE_TEST_SUITE_P(CPU, PrimitiveTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PrimitiveTest, ::testing::Values(Device::CUDA));
#endif
