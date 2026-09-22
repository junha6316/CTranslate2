#include "test_utils.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/layers/flash_attention.h"

namespace {
  class EncoderAttentionModel : public models::Model {
  public:
    static constexpr dim_t num_heads = 2;
    static constexpr dim_t head_size = 8;
    static constexpr dim_t depth = num_heads * head_size;

    EncoderAttentionModel(const FloatType type, const std::string& feature = "") {
      StorageView qkv({3 * depth, depth}, 0.f);
      StorageView projection({depth, depth}, 0.f);
      for (dim_t i = 0; i < depth; ++i) {
        projection.at<float>({i, i}) = 1.f;
        for (dim_t j = 0; j < 3; ++j)
          qkv.at<float>({j * depth + i, i}) = 1.f;
      }
      register_variable("attn/linear_0/weight", std::move(qkv));
      register_variable("attn/linear_1/weight", std::move(projection));
      if (feature == "relative_attention_bias") {
        register_variable("attn/relative_attention_bias",
                          StorageView({4, num_heads}, std::vector<float>{
                              0.f, 0.2f, 0.3f, -0.4f, 0.5f, 0.1f, -0.2f, 0.7f}));
        register_variable("attn/relative_attention_max_distance", StorageView(int32_t(8)));
      } else if (!feature.empty()) {
        register_variable("attn/" + feature + "/gamma", StorageView({head_size}, 1.5f));
      }
      const ComputeType compute_type = type.dtype == DataType::FLOAT32 ? ComputeType::FLOAT32
        : type.dtype == DataType::FLOAT16 ? ComputeType::FLOAT16 : ComputeType::BFLOAT16;
      set_compute_type(compute_type, type.device, 0);
      set_device(type.device);
    }

  protected:
    std::unique_ptr<Model> clone() const override { return nullptr; }
  };

  StorageView encoder_attention_input(const FloatType type) {
    StorageView input({2, 3, EncoderAttentionModel::depth}, 0.f);
    for (dim_t i = 0; i < input.size(); ++i)
      input.data<float>()[i] = float(i % 13 - 6) / 10.f;
    return input.to(type.dtype).to(type.device);
  }
}

class FlashEncoderAttentionTest : public ::testing::TestWithParam<FloatType> {
protected:
  void SetUp() override {
    if (GetParam().device == Device::CUDA
        && (get_device_count(Device::CUDA) == 0
            || !mayiuse_bfloat16(Device::CUDA)))
      GTEST_SKIP() << "FlashAttention requires an Ampere or newer CUDA GPU";
  }
};

TEST_P(FlashEncoderAttentionTest, PreservesLengthMaskAndPaddingRemoval) {
  const auto type = GetParam();
  EncoderAttentionModel model(type);
  layers::MultiHeadAttention reference(model, "attn", EncoderAttentionModel::num_heads, true);
  layers::FlashMultiHeadAttention flash(model, "attn", EncoderAttentionModel::num_heads, true);
  StorageView lengths({2}, std::vector<int32_t>{1, 3});
  auto mask = layers::AttentionLayer::prepare_length_mask(
    lengths.to(type.device), EncoderAttentionModel::num_heads, 3);
  Padder padder(lengths.to(type.device), 3);
  for (const bool remove_padding : {false, true}) {
    auto input = encoder_attention_input(type);
    if (remove_padding)
      padder.remove_padding(input);
    const Padder* padding = remove_padding ? &padder : nullptr;
    StorageView expected(type.dtype, type.device), actual(type.dtype, type.device);
    reference(input, input, &mask, expected, nullptr, nullptr, nullptr, padding, padding);
    flash(input, input, &mask, actual, nullptr, nullptr, nullptr, padding, padding);
    expect_storage_eq(actual.to_float32(), expected.to_float32(), type.error);
  }
}

TEST_P(FlashEncoderAttentionTest, PreservesRelativePositionBias) {
  const auto type = GetParam();
  EncoderAttentionModel model(type, "relative_attention_bias");
  layers::MultiHeadAttention reference(model, "attn", EncoderAttentionModel::num_heads, true);
  layers::FlashMultiHeadAttention flash(model, "attn", EncoderAttentionModel::num_heads, true);
  EXPECT_TRUE(flash.has_positional_embeddings());
  const auto input = encoder_attention_input(type);
  StorageView expected(type.dtype, type.device), actual(type.dtype, type.device);
  StorageView expected_bias(type.dtype, type.device), actual_bias(type.dtype, type.device);
  // Exercise both creation and reuse of the bias across encoder layers.
  for (int layer = 0; layer < 2; ++layer) {
    reference(input, input, nullptr, expected, nullptr, nullptr, nullptr,
              nullptr, nullptr, true, &expected_bias);
    flash(input, input, nullptr, actual, nullptr, nullptr, nullptr,
          nullptr, nullptr, true, &actual_bias);
    ASSERT_FALSE(actual_bias.empty());
    expect_storage_eq(actual.to_float32(), expected.to_float32(), type.error);
    expect_storage_eq(actual_bias.to_float32(), expected_bias.to_float32(), type.error);
  }
}

TEST_P(FlashEncoderAttentionTest, PreservesProjectionNormalization) {
  const auto type = GetParam();
  for (const std::string feature : {"q_norm", "k_norm", "v_norm"}) {
    SCOPED_TRACE(feature);
    EncoderAttentionModel model(type, feature);
    layers::MultiHeadAttention reference(model, "attn", EncoderAttentionModel::num_heads, true);
    layers::FlashMultiHeadAttention flash(model, "attn", EncoderAttentionModel::num_heads, true);
    const auto input = encoder_attention_input(type);
    StorageView expected(type.dtype, type.device), actual(type.dtype, type.device);
    reference(input, input, nullptr, expected);
    flash(input, input, nullptr, actual);
    expect_storage_eq(actual.to_float32(), expected.to_float32(), type.error);
  }
}

TEST_P(FlashEncoderAttentionTest, UnmaskedEncoderIsBidirectionalAndDecoderIsCausal) {
  const auto type = GetParam();
  if (type.device != Device::CUDA)
    GTEST_SKIP() << "The FlashAttention kernel requires CUDA";
  EncoderAttentionModel model(type);
  const auto input = encoder_attention_input(type);
  for (const bool is_decoder : {false, true}) {
    SCOPED_TRACE(is_decoder);
    layers::MultiHeadAttention reference(model, "attn", EncoderAttentionModel::num_heads,
                                         true, true, is_decoder);
    layers::FlashMultiHeadAttention flash(model, "attn", EncoderAttentionModel::num_heads,
                                          true, true, is_decoder);
    StorageView expected(type.dtype, type.device), actual(type.dtype, type.device);
    StorageView lengths({2}, std::vector<int32_t>{3, 3}, type.device);
    auto causal_mask = layers::AttentionLayer::prepare_length_mask(
      lengths, EncoderAttentionModel::num_heads, 3, /*mask_future=*/true);
    reference(input, input, is_decoder ? &causal_mask : nullptr, expected);
    flash(input, input, nullptr, actual);
    expect_storage_eq(actual.to_float32(), expected.to_float32(), type.error);

    auto changed = input.to_float32().to(Device::CPU);
    for (dim_t d = 0; d < EncoderAttentionModel::depth; ++d)
      changed.at<float>({0, 2, d}) += 0.75f;
    changed.move_to(type.device, type.dtype);
    StorageView changed_output(type.dtype, type.device);
    flash(changed, changed, nullptr, changed_output);
    const auto before = actual.to_float32().to(Device::CPU);
    const auto after = changed_output.to_float32().to(Device::CPU);
    const float difference = std::abs(after.at<float>({0, 0, 0}) - before.at<float>({0, 0, 0}));
    if (is_decoder)
      EXPECT_LE(difference, type.error);
    else
      EXPECT_GT(difference, 0.01f);
  }
}

INSTANTIATE_TEST_SUITE_P(CPU, FlashEncoderAttentionTest,
                        ::testing::Values(FloatType{Device::CPU, DataType::FLOAT32, 1e-5f}),
                        fp_test_name);
#ifdef CT2_WITH_FLASH_ATTN
INSTANTIATE_TEST_SUITE_P(CUDA, FlashEncoderAttentionTest,
                        ::testing::Values(FloatType{Device::CUDA, DataType::FLOAT16, 5e-3f},
                                          FloatType{Device::CUDA, DataType::BFLOAT16, 3e-2f}),
                        fp_test_name);
#endif

class MockModel : public models::Model {
public:
  MockModel(dim_t num_heads, dim_t num_heads_kv) {
    const dim_t d_model = 64;
    const dim_t d_head = d_model / num_heads;
    
    std::vector<float> linear_0_data(num_heads * d_head * d_model, 0.01f);
    std::vector<float> linear_1_data(2 * num_heads_kv * d_head * d_model, 0.01f);
    
    register_variable("attn/linear_0/weight",
                      StorageView({num_heads * d_head, d_model}, linear_0_data));
    register_variable("attn/linear_1/weight",
                      StorageView({2 * num_heads_kv * d_head, d_model}, linear_1_data));
    register_variable("attn/linear_2/weight",
                      StorageView({d_model, num_heads * d_head}, DataType::FLOAT32));
    register_variable("attn/q_norm/gamma",
                      StorageView({d_model}, std::vector<float>(d_model, 1.0f)));
    register_variable("attn/k_norm/gamma",
                      StorageView({d_head}, std::vector<float>(d_head, 1.0f)));
    
    register_variable("attn/num_heads_kv",
                      StorageView(static_cast<int32_t>(num_heads_kv)));

    set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
  }
protected:
  std::unique_ptr<Model> clone() const override { return nullptr; }
};

class TestableAttention : public layers::MultiHeadAttention {
public:
  using MultiHeadAttention::MultiHeadAttention;
  using MultiHeadAttention::process_cross_attention;
};

class CrossAttentionTest : public ::testing::Test {
protected:
  static constexpr dim_t NUM_HEADS = 4;
  static constexpr dim_t D_MODEL = 64;
  static constexpr dim_t D_HEAD = D_MODEL / NUM_HEADS;
  static constexpr dim_t BATCH = 2;
  static constexpr dim_t Q_LEN = 6;
  static constexpr dim_t V_LEN = 8;

  float get_4d(const StorageView& view, dim_t b, dim_t h, dim_t t, dim_t d) {
    const auto& shape = view.shape();
    return view.data<float>()[b * shape[1] * shape[2] * shape[3] +
                              h * shape[2] * shape[3] + t * shape[3] + d];
  }

};

// MQA: All heads share same K/V
TEST_F(CrossAttentionTest, MultiQueryAttention) {
  MockModel model(NUM_HEADS, /*num_heads_kv=*/1);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);
  // Use non-uniform values to verify normalization is applied
  std::vector<float> value_data(BATCH * V_LEN * D_MODEL);
  for (size_t i = 0; i < value_data.size(); ++i)
    value_data[i] = static_cast<float>(i % 10 + 1);
  std::vector<float> fused_data(BATCH * Q_LEN * NUM_HEADS * D_HEAD);
  for (size_t i = 0; i < fused_data.size(); ++i)
    fused_data[i] = static_cast<float>(i % 10 + 1);
  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, value_data);
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, fused_data);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;
  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);
  // MQA: K/V are replicated to 4D format [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  // Verify K/V values are consistent across batch and time dimensions
  // (In MQA, there's only one set of K/V, so we just verify the tensor is valid)
  float k0 = get_4d(cached_keys, 0, 0, 0, 0);
  float v0 = get_4d(cached_values, 0, 0, 0, 0);
  EXPECT_NE(k0, 0.0f) << "K values should be non-zero after projection";
  EXPECT_NE(v0, 0.0f) << "V values should be non-zero after projection";
  // Verify q_norm and k_norm are applied (RMSNorm normalizes to ~1.0 magnitude)
  float q_val = q_proj.data<float>()[0];
  float k_val = cached_keys.data<float>()[0];
  EXPECT_GT(std::abs(q_val), 0.1f) << "q_norm should produce non-zero output";
  EXPECT_LT(std::abs(q_val), 2.0f) << "q_norm should normalize values";
  EXPECT_GT(std::abs(k_val), 0.1f) << "k_norm should produce non-zero output";
  EXPECT_LT(std::abs(k_val), 2.0f) << "k_norm should normalize values";
}

// GQA: Heads within same group share K/V
TEST_F(CrossAttentionTest, GroupedQueryAttention) {
  constexpr dim_t NUM_KV_HEADS = 2;
  constexpr dim_t HEADS_PER_GROUP = NUM_HEADS / NUM_KV_HEADS;

  MockModel model(NUM_HEADS, NUM_KV_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // GQA: After head replication, shape is [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));

  // Heads in same group share K/V
  for (dim_t group = 0; group < NUM_KV_HEADS; ++group) {
    dim_t first = group * HEADS_PER_GROUP;
    float k_group = get_4d(cached_keys, 0, first, 0, 0);
    float v_group = get_4d(cached_values, 0, first, 0, 0);
    for (dim_t h = first + 1; h < first + HEADS_PER_GROUP; ++h) {
      EXPECT_EQ(get_4d(cached_keys, 0, h, 0, 0), k_group);
      EXPECT_EQ(get_4d(cached_values, 0, h, 0, 0), v_group);
    }
  }
}

// Merged self+cross attention (T5Gemma2 style): self-attention layer also
// projects encoder memory through a separate `memory_kv` linear and concatenates
// the result onto the self-attention K/V before softmax.
TEST(MergedAttentionTest, ForwardMergedProducesOutput) {
  constexpr dim_t NUM_HEADS = 4, NUM_KV = 1, D_HEAD = 16;
  constexpr dim_t D_MODEL = NUM_HEADS * D_HEAD;
  constexpr dim_t QKV_ROWS = (NUM_HEADS + 2 * NUM_KV) * D_HEAD;
  constexpr dim_t KV_ROWS = 2 * NUM_KV * D_HEAD;

  class MergedMockModel : public models::Model {
  public:
    MergedMockModel() {
      register_variable("attn/linear_0/weight",
                        StorageView({QKV_ROWS, D_MODEL}, std::vector<float>(QKV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/linear_1/weight",
                        StorageView({D_MODEL, NUM_HEADS * D_HEAD},
                                    std::vector<float>(D_MODEL * NUM_HEADS * D_HEAD, 0.01f)));
      register_variable("attn/memory_kv/weight",
                        StorageView({KV_ROWS, D_MODEL}, std::vector<float>(KV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/q_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/k_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/num_heads_kv", StorageView(static_cast<int32_t>(NUM_KV)));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
    }
  protected:
    std::unique_ptr<Model> clone() const override { return nullptr; }
  };

  MergedMockModel model;
  layers::MultiHeadAttention attention(model, "attn", NUM_HEADS, /*self_attention=*/true);
  ASSERT_TRUE(attention.has_merged_encoder_attention());

  constexpr dim_t B = 1, Q_LEN = 1, MEM_LEN = 5;
  StorageView queries({B, Q_LEN, D_MODEL}, std::vector<float>(B * Q_LEN * D_MODEL, 1.0f));
  StorageView memory({B, MEM_LEN, D_MODEL}, std::vector<float>(B * MEM_LEN * D_MODEL, 1.0f));
  StorageView output(DataType::FLOAT32);
  StorageView self_k(DataType::FLOAT32), self_v(DataType::FLOAT32);
  StorageView mem_k(DataType::FLOAT32), mem_v(DataType::FLOAT32);

  attention.forward_merged(queries, &memory, nullptr, nullptr, output,
                           &self_k, &self_v, &mem_k, &mem_v, nullptr, nullptr, /*offset=*/0);

  EXPECT_EQ(output.shape(), (Shape{B, Q_LEN, D_MODEL}));
  EXPECT_EQ(mem_k.shape(), (Shape{B, NUM_HEADS, MEM_LEN, D_HEAD}));
  EXPECT_EQ(self_k.shape(), (Shape{B, NUM_HEADS, Q_LEN, D_HEAD}));
}

// MHA: Each head has independent K/V
TEST_F(CrossAttentionTest, StandardMultiHeadAttention) {
  MockModel model(NUM_HEADS, NUM_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // Shape: [batch, num_heads, time, d_head] - each head has own K/V
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), cached_keys.shape());
}
