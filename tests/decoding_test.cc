#include <ctranslate2/decoding.h>

#include "test_utils.h"

TEST(DecodingTest, DisableTokens) {
  StorageView input({2, 5}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  StorageView expected({2, 5}, std::vector<float>{1, 0, 0, 4, 5, 6, 7, 0, 9, 0});

  DisableTokens disable_tokens(input, 0);
  disable_tokens.add(2);
  disable_tokens.add(0, 1);
  disable_tokens.add(1, 4);
  disable_tokens.apply();

  expect_storage_eq(input, expected);
}

static StorageView make_logits(dim_t batch_size, dim_t vocabulary_size, float start) {
  std::vector<float> values(batch_size * vocabulary_size);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = start + 0.01f * float(i);
  return StorageView({batch_size, vocabulary_size}, values);
}

TEST(DecodingTest, DisableTokensForcedIndexPath) {
  DisableTokens::Buffers buffers;

  // Apply the same token set through the direct-fill reference and the forced index
  // upload path: both must produce identical logits on every simulated step.
  auto run_step = [&buffers](dim_t batch_size, float start, bool add_range) {
    StorageView reference = make_logits(batch_size, 8, start);
    StorageView candidate = make_logits(batch_size, 8, start);

    DisableTokens direct(reference, 0);
    DisableTokens indexed(candidate, 0, &buffers, /*force_index_path=*/true);

    for (DisableTokens* disable : {&direct, &indexed}) {
      disable->add(2);
      disable->add(0, 1);
      if (add_range)
        disable->add_range(0, 5, 7);
      disable->apply();
    }

    expect_storage_eq(candidate, reference);
  };

  run_step(2, 1, true);
  const std::vector<int32_t> expected_indices{1, 2, 10};
  assert_vector_eq(buffers.last_indices, expected_indices);
  assert_vector_eq(buffers.last_ranges, std::vector<int32_t>{0, 5, 7});
  const void* indices_buffer = buffers.indices.buffer();

  // Identical token set on new logits: the memo hits and the resident tensor is reused.
  run_step(2, 5, true);
  assert_vector_eq(buffers.last_indices, expected_indices);
  EXPECT_EQ(buffers.indices.buffer(), indices_buffer);

  // Prove the upload was actually skipped: corrupt the resident tensor and check the
  // fill lands on the corrupted index instead of the recorded one.
  buffers.indices.at<int32_t>(0) = 3;
  StorageView poked = make_logits(2, 8, 9);
  DisableTokens indexed(poked, 0, &buffers, /*force_index_path=*/true);
  indexed.add(2);
  indexed.add(0, 1);
  indexed.apply();
  EXPECT_EQ(poked.at<float>(3), 0);
  EXPECT_NE(poked.at<float>(1), 0);

  // A batch shrink changes the flat indices: the memo misses and re-uploads, which
  // also restores the corrupted tensor.
  run_step(1, 3, false);
  assert_vector_eq(buffers.last_indices, std::vector<int32_t>{1, 2});
  EXPECT_EQ(buffers.indices.buffer(), indices_buffer);

  // Growing back to the original set re-uploads again and matches the first step.
  run_step(2, 7, false);
  assert_vector_eq(buffers.last_indices, expected_indices);
}

TEST(DecodingTest, UploadMemoized) {
  // The pattern behind the DisableTokens buffers and the Whisper timestamp row ids:
  // same content twice must skip the transfer, changed content must re-upload.
  StorageView device_tensor;
  std::vector<int32_t> last_values;

  const std::vector<int32_t> rows{0, 1, 2, 3, 4};
  EXPECT_TRUE(upload_memoized(rows, {5}, Device::CPU, device_tensor, last_values));
  expect_storage_eq(device_tensor, StorageView({5}, rows));

  EXPECT_FALSE(upload_memoized(rows, {5}, Device::CPU, device_tensor, last_values));

  // Same size, different content: the memo must compare the values, not the size.
  const std::vector<int32_t> other{0, 1, 2, 3, 5};
  EXPECT_TRUE(upload_memoized(other, {5}, Device::CPU, device_tensor, last_values));
  expect_storage_eq(device_tensor, StorageView({5}, other));

  const std::vector<int32_t> shrunk{2};
  EXPECT_TRUE(upload_memoized(shrunk, {1}, Device::CPU, device_tensor, last_values));
  expect_storage_eq(device_tensor, StorageView({1}, shrunk));
}

TEST(DecodingTest, ToDeviceStagedSameDevice) {
  const StorageView src({2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView staging(src.dtype(), src.device());
  const StorageView& result = to_device_staged(src, staging);
  EXPECT_EQ(&result, &src);
  EXPECT_TRUE(staging.empty());
}

TEST(DecodingTest, SamplerStagingParity) {
  const StorageView scores({2, 6}, std::vector<float>{
      0.1f, 2.0f, -1.0f, 0.4f, 1.5f, 0.0f,
      -0.3f, 0.2f, 3.1f, -2.0f, 0.9f, 1.1f});

  const BestSampler best;
  const RandomSampler random(/*from_topk=*/1);  // Top-1 sampling is deterministic.

  const std::vector<std::pair<const Sampler*, dim_t>> cases = {{&best, 2}, {&random, 1}};

  for (const auto& [sampler, num_samples] : cases) {
    StorageView ids(DataType::INT32);
    StorageView sampled_scores;
    (*sampler)(scores, ids, sampled_scores, num_samples);

    SamplerStaging staging;
    StorageView staged_ids(DataType::INT32);
    StorageView staged_scores;
    (*sampler)(scores, staged_ids, staged_scores, num_samples, &staging);

    expect_storage_eq(staged_ids, ids);
    expect_storage_eq(staged_scores, sampled_scores);
    // Scores live on CPU here, and the CPU TopK never touches the scratch slots.
    EXPECT_TRUE(staging.topk_tmp_ids.empty());
    EXPECT_TRUE(staging.topk_tmp_vals.empty());
  }
}
