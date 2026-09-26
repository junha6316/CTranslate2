#include <ctranslate2/buffered_translation_wrapper.h>
#include <ctranslate2/translator.h>
#include <ctranslate2/decoding.h>
#include <ctranslate2/layers/transformer.h>
#include <ctranslate2/models/whisper.h>

#include <algorithm>
#include <unordered_set>

#include "test_utils.h"

static std::string
path_to_test_name(::testing::TestParamInfo<std::pair<std::string, DataType>> param_info) {
  std::string name = param_info.param.first;
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  return name;
}

static std::string beam_to_test_name(::testing::TestParamInfo<size_t> param_info) {
  if (param_info.param == 1)
    return "GreedySearch";
  else
    return "BeamSearch";
}

static void check_weights_dtype(const std::unordered_map<std::string, StorageView>& variables,
                                DataType expected_dtype) {
  for (const auto& variable : variables) {
    const auto& name = variable.first;
    const auto& value = variable.second;
    if (ends_with(name, "weight")) {
      EXPECT_EQ(value.dtype(), expected_dtype) << "Expected type " << dtype_name(expected_dtype)
                                               << " for weight " << name << ", got "
                                               << dtype_name(value.dtype()) << " instead";
    }
  }
}

static DataType dtype_with_fallback(DataType dtype, Device device) {
  const bool support_int8 = mayiuse_int8(device);
  const bool support_int16 = mayiuse_int16(device);
  if (dtype == DataType::INT16 && !support_int16)
    return support_int8 ? DataType::INT8 : DataType::FLOAT32;
  if (dtype == DataType::INT8 && !support_int8)
    return support_int16 ? DataType::INT16 : DataType::FLOAT32;
  return dtype;
}


// Test that we can load and translate with different versions of the same model.
class ModelVariantTest : public ::testing::TestWithParam<std::pair<std::string, DataType>> {
};

TEST_P(ModelVariantTest, Transliteration) {
  auto params = GetParam();
  const std::string model_path = get_data_dir() + "/models/" + params.first;
  const DataType model_dtype = params.second;
  const Device device = Device::CPU;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> expected = {"a", "t", "z", "m", "o", "n"};

  std::vector<std::pair<ComputeType, DataType>> type_params;
  type_params.emplace_back(ComputeType::DEFAULT, dtype_with_fallback(model_dtype, device));
  type_params.emplace_back(ComputeType::FLOAT32, DataType::FLOAT32);
  if (mayiuse_int16(device))
    type_params.emplace_back(ComputeType::INT16, DataType::INT16);
  if (mayiuse_int8(device)) {
    type_params.emplace_back(ComputeType::INT8, DataType::INT8);
    type_params.emplace_back(ComputeType::AUTO, DataType::INT8);
  } else if (mayiuse_int16(device)) {
    type_params.emplace_back(ComputeType::AUTO, DataType::INT16);
  } else {
    type_params.emplace_back(ComputeType::AUTO, DataType::FLOAT32);
  }

  for (const auto& types : type_params) {
    const ComputeType compute_type = types.first;
    const DataType expected_type = types.second;
    const auto model = models::Model::load(model_path, device, 0, compute_type);
    check_weights_dtype(model->get_variables(), expected_type);
    Translator translator(model);
    auto result = translator.translate_batch({input})[0];
    EXPECT_EQ(result.output(), expected);
  }
}

INSTANTIATE_TEST_SUITE_P(
  TranslatorTest,
  ModelVariantTest,
  ::testing::Values(
    std::make_pair("v1/aren-transliteration", DataType::FLOAT32),
    std::make_pair("v1/aren-transliteration-i16", DataType::INT16),
    std::make_pair("v2/aren-transliteration", DataType::FLOAT32),
    std::make_pair("v2/aren-transliteration-i16", DataType::INT16),
    std::make_pair("v2/aren-transliteration-i8", DataType::INT8)
    ),
  path_to_test_name);


class SearchVariantTest : public ::testing::TestWithParam<size_t> {
};

static Translator default_translator(Device device = Device::CPU) {
  return Translator(default_model_dir(), device);
}

TEST_P(SearchVariantTest, SetMaxDecodingLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_decoding_length = 3;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.output().size(), options.max_decoding_length);
}

TEST_P(SearchVariantTest, SetMinDecodingLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.min_decoding_length = 8;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.output().size(), options.min_decoding_length);
}

// A decode long enough to grow the self-attention cache several times past its
// preallocated block. The expected strings were produced before the cache stopped being
// rebuilt at every step, so this pins the output across that change.
TEST_P(SearchVariantTest, LongDecodingKeepsCachedAttentionExact) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.min_decoding_length = 100;
  options.max_decoding_length = 100;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate_batch({input}, options)[0];
  std::string output;
  for (const auto& token : result.output())
    output += token;
  const std::string expected = (GetParam() == 1
    ? "atzmonenenehehehehennehehehtehehehehehtememehounehehehehehehehumennnnehtamennnnnnehumoumehehehehehen"
    : "atzmonetenehehehennnehehehennnnnehehehehoumehehehehehehehehaumehumennehtamennnnnne'tamennneheheheheh");
  EXPECT_EQ(result.output().size(), size_t(100));
  EXPECT_EQ(output, expected);
}

// The decoder reuses its layer temporaries across decoding steps and across translate
// calls (see DecodeWorkspace). The following tests target the reuse hazards: data or
// capacities left in a buffer by a previous call must never leak into a later one.

static void expect_same_results(const std::vector<TranslationResult>& a,
                                const std::vector<TranslationResult>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].hypotheses, b[i].hypotheses);
    EXPECT_EQ(a[i].scores, b[i].scores);
    EXPECT_EQ(a[i].attention, b[i].attention);
  }
}

// Translating the same batch twice with one instance must be deterministic: the second
// call runs entirely on reused buffers, so any stale data surviving in them shows up as
// a diverging output. Attention is requested to also exercise the path where
// save_attention steals a workspace-backed buffer.
TEST_P(SearchVariantTest, StatefulDecodeBufferReuse) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.return_scores = true;
  const std::vector<std::vector<std::string>> inputs = {
    {"آ", "ت", "ز", "م", "و", "ن"},
    {"آ", "ز", "ا"}
  };
  for (const bool return_attention : {false, true}) {
    options.return_attention = return_attention;
    const auto first = translator.translate_batch(inputs, options);
    const auto second = translator.translate_batch(inputs, options);
    expect_same_results(first, second);
  }
}

// A sequence of calls whose batch size grows and shrinks, each compared against a fresh
// instance. The reused buffers keep the capacity of the largest batch seen, so a resize
// that reads stale bytes past the logical shape would diverge from the fresh instance.
TEST_P(SearchVariantTest, DecodeBufferCapacityChurn) {
  Translator reused = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.num_hypotheses = GetParam();
  options.return_scores = true;
  options.return_attention = true;
  const std::vector<std::vector<std::vector<std::string>>> batches = {
    {{"آ", "ت", "ز", "م", "و", "ن"}, {"آ", "ز", "ا"}, {"آ", "ت", "ز"}},
    {{"آ", "ز", "ا"}},
    // Mixed lengths so that some sentences finish early and the batch shrinks while
    // decoding is still running on reused buffers.
    {{"آ", "ت", "ز", "م", "و", "ن"},
     {"آ", "ز", "ا"},
     {"آ", "ت", "ز", "م", "و", "ن"},
     {"م", "و", "ن"},
     {"آ", "ت"}}
  };
  for (const auto& batch : batches) {
    Translator fresh = default_translator();
    expect_same_results(reused.translate_batch(batch, options),
                        fresh.translate_batch(batch, options));
  }
}

TEST_P(SearchVariantTest, SetMaxInputLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_input_length = 3;
  options.return_attention = true;

  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate_batch({input}, options)[0];

  EXPECT_EQ(result.hypotheses[0].size(), options.max_input_length);

  // Check that attention vectors have the size of the original input.
  ASSERT_TRUE(result.has_attention());
  for (size_t i = 0; i < result.attention[0].size(); ++i) {
    ASSERT_EQ(result.attention[0][i].size(), input.size());
    for (size_t t = options.max_input_length; t < input.size(); ++t) {
      EXPECT_EQ(result.attention[0][i][t], 0);
    }
  }
}

TEST_P(SearchVariantTest, ReturnAllHypotheses) {
  auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.num_hypotheses(), beam_size);
}

TEST_P(SearchVariantTest, ReturnAttention) {
  auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  const std::vector<std::vector<std::string>> inputs = {
    {"آ", "ز", "ا"},
    {"آ", "ت", "ز", "م", "و", "ن"}
  };
  const std::vector<std::pair<size_t, size_t>> expected_shapes = {
    {4, 3},
    {6, 6}
  };  // (target_length, source_length)
  const auto results = translator.translate_batch(inputs, options);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const TranslationResult& result = results[i];
    const auto& expected_shape = expected_shapes[i];
    ASSERT_TRUE(result.has_attention());
    const auto& attention = result.attention;
    EXPECT_EQ(attention.size(), beam_size);
    EXPECT_EQ(attention[0].size(), expected_shape.first);
    for (const auto& vector : attention[0]) {
      EXPECT_EQ(vector.size(), expected_shape.second);
    }
  }
}

TEST_P(SearchVariantTest, ReturnAttentionWithPrefix) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"<unk>", "t"};
  std::vector<std::string> expected = {"<unk>", "t", "z", "m", "o", "n" };
  auto result = translator.translate_batch({input}, {prefix}, options)[0];
  EXPECT_EQ(result.output(), expected);
  EXPECT_TRUE(result.has_attention());
  for (const auto& vector : result.attention[0]) {
    EXPECT_EQ(vector.size(), input.size());
  }
}

TEST_P(SearchVariantTest, ReturnEndToken) {
  auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.return_end_token = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> expected = {"a", "t", "z", "m", "o", "n", "</s>"};
  const auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.hypotheses[0], expected);
}

TEST_P(SearchVariantTest, TranslateWithPrefix) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"a", "t", "s"};
  std::vector<std::string> expected = {"a", "t", "s", "u", "m", "o", "n"};
  auto result = translator.translate_batch({input}, {prefix}, options)[0];
  EXPECT_EQ(result.num_hypotheses(), beam_size);
  EXPECT_EQ(result.output(), expected);
  ASSERT_TRUE(result.has_attention());
  const auto& attention = result.attention;
  EXPECT_EQ(attention.size(), options.beam_size);
  EXPECT_EQ(attention[0].size(), 7);
  EXPECT_EQ(attention[0][0].size(), 6);
}

TEST_P(SearchVariantTest, TranslateBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.return_scores = true;
  options.beam_size = GetParam();
  std::vector<std::vector<std::string>> inputs = {
    {"آ", "ز", "ا"},
    {"آ", "ت", "ز", "م", "و", "ن"}};
  std::vector<std::vector<std::string>> expected = {
    {"a", "z", "z", "a"},
    {"a", "t", "z", "m", "o", "n"}};
  auto result = translator.translate_batch(inputs, options);
  EXPECT_TRUE(result[0].has_scores());
  EXPECT_TRUE(result[1].has_scores());
  EXPECT_EQ(result[0].output(), expected[0]);
  EXPECT_EQ(result[1].output(), expected[1]);
}

TEST_P(SearchVariantTest, SuppressSequences) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.suppress_sequences = {{"o"}, {"t", "z", "m"}};
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> expected = {"a", "t", "z", "u", "m", "u", "n"};
  auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.output(), expected);
}

TEST_P(SearchVariantTest, SuppressSequenceOOV) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.suppress_sequences = {{"o"}, {"t", "oovtoken", "m"}};
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  EXPECT_THROW(translator.translate_batch({input}, options), std::invalid_argument);
}

TEST_P(SearchVariantTest, EndToken) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.end_token = "m";
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> expected = {"a", "t", "z"};
  auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.output(), expected);
}

TEST_P(SearchVariantTest, EndTokenOOV) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.end_token = "oovtoken";
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  EXPECT_THROW(translator.translate_batch({input}, options), std::invalid_argument);
}

TEST_P(SearchVariantTest, ReplaceUnknowns) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.replace_unknowns = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"<unk>", "t"};
  std::vector<std::string> expected = {"ت", "t", "z", "m", "o", "n" };
  auto result = translator.translate_batch({input}, {prefix}, options)[0];
  EXPECT_EQ(result.output(), expected);
}

TEST_P(SearchVariantTest, RepetitionPenalty) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.repetition_penalty = 100;  // Force the decoding to produce unique symbols.
  const auto result = translator.translate_batch({{"ن", "ن", "ن", "ن", "ن"}}, options)[0];
  const auto& tokens = result.output();
  const std::unordered_set<std::string> unique_tokens(tokens.begin(), tokens.end());
  EXPECT_EQ(unique_tokens.size(), tokens.size());
}

TEST_P(SearchVariantTest, NoRepeatNgram) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.no_repeat_ngram_size = 3;
  const std::vector<std::string> input(50, "ن");
  const auto result = translator.translate_batch({input}, options)[0];
  const auto output = join_string(result.output());

  std::unordered_set<std::string> ngrams;
  for (size_t i = 0; i < output.size() - options.no_repeat_ngram_size; ++i)
    ngrams.emplace(output.substr(i, options.no_repeat_ngram_size));

  EXPECT_EQ(ngrams.size(), output.size() - options.no_repeat_ngram_size);
}

static void check_normalized_score(const std::vector<std::string>& input,
                                   TranslationOptions options,
                                   bool output_has_eos = true) {
  Translator translator = default_translator();
  options.return_scores = true;
  options.length_penalty = 0;
  const auto score = translator.translate_batch({input}, options)[0].scores[0];

  options.length_penalty = 1;
  const auto normalized_result = translator.translate_batch({input}, options)[0];
  const auto normalized_score = normalized_result.scores[0];
  auto normalized_length = normalized_result.hypotheses[0].size();
  if (output_has_eos)
    normalized_length += 1;

  EXPECT_NEAR(normalized_score, score / normalized_length, 1e-6);
}

TEST_P(SearchVariantTest, NormalizeScores) {
  TranslationOptions options;
  options.beam_size = GetParam();
  check_normalized_score({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}, options);
}

TEST_P(SearchVariantTest, NormalizeScoresNoEos) {
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_decoding_length = 6;
  check_normalized_score({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}, options, false);
}


INSTANTIATE_TEST_SUITE_P(
  TranslatorTest,
  SearchVariantTest,
  ::testing::Values(1, 4),
  beam_to_test_name);

TEST(TranslatorTest, TranslateEmptyBatch) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs;
  auto results = translator.translate_batch(inputs);
  EXPECT_TRUE(results.empty());
}

static void check_empty_result(const TranslationResult& result,
                               size_t num_hypotheses = 1,
                               bool with_attention = false,
                               bool with_score = false) {
  EXPECT_TRUE(result.output().empty());
  EXPECT_EQ(result.num_hypotheses(), num_hypotheses);
  EXPECT_EQ(result.hypotheses.size(), num_hypotheses);
  EXPECT_EQ(result.has_scores(), with_score);
  if (with_score) {
    EXPECT_EQ(result.scores.size(), num_hypotheses);
    EXPECT_EQ(result.score(), 0);
    for (const auto score : result.scores) {
      EXPECT_EQ(score, 0);
    }
  }
  EXPECT_EQ(result.has_attention(), with_attention);
  if (with_attention) {
    const auto& attention = result.attention;
    EXPECT_EQ(attention.size(), num_hypotheses);
    EXPECT_TRUE(attention[0].empty());
  }
}

TEST(TranslatorTest, TranslateBatchWithEmptySource) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs = {
    {}, {"آ", "ز", "ا"}, {}, {"آ", "ت", "ز", "م", "و", "ن"}, {}};
  auto results = translator.translate_batch(inputs);
  EXPECT_EQ(results.size(), 5);
  check_empty_result(results[0]);
  EXPECT_EQ(results[1].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
  check_empty_result(results[2]);
  EXPECT_EQ(results[3].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  check_empty_result(results[4]);
}

TEST(TranslatorTest, TranslateBatchWithOnlyEmptySource) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs{{}, {}};
  auto results = translator.translate_batch(inputs);
  EXPECT_EQ(results.size(), 2);
  check_empty_result(results[0]);
  check_empty_result(results[1]);
}

TEST(TranslatorTest, TranslateEmptySourceWithoutScore) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.return_scores = false;
  EXPECT_FALSE(translator.translate_batch({{}}, options)[0].has_scores());
}

TEST(TranslatorTest, TranslateBatchWithHardPrefixAndEmpty) {
  Translator translator = default_translator();
  const TranslationOptions options;
  const std::vector<std::vector<std::string>> input = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {},
    {"آ" ,"ز" ,"ا"}};
  const std::vector<std::vector<std::string>> prefix = {
    {"a", "t", "s"},
    {},
    {"a", "t", "z", "o"},
    {},
    {}};
  const auto result = translator.translate_batch(input, prefix, options);
  EXPECT_EQ(result[0].output(), (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));
  EXPECT_EQ(result[1].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[2].output(), (std::vector<std::string>{"a", "t", "z", "o", "m", "o", "n"}));
  EXPECT_TRUE(result[3].output().empty());
  EXPECT_EQ(result[4].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
}

TEST(TranslatorTest, TranslateBatchWithStronglyBiasedPrefix) {
  // This test should produce the same results as TranslateBatchWithHardPrefixAndEmpty
  // because prefix_bias_beta is set to 0.99, which is almost equivalent to using a hard prefix.
  Translator translator = default_translator();
  TranslationOptions options;
  options.prefix_bias_beta = 0.99;
  options.beam_size = 2;
  const std::vector<std::vector<std::string>> input = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ز" ,"ا"}};
  const std::vector<std::vector<std::string>> prefix = {
    {"a", "t", "s"},
    {},
    {"a", "t", "z", "o"},
    {}};
  const auto result = translator.translate_batch(input, prefix, options);
  EXPECT_EQ(result[0].output(), (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));
  EXPECT_EQ(result[1].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[2].output(), (std::vector<std::string>{"a", "t", "z", "o", "m", "o", "n"}));
  EXPECT_EQ(result[3].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
}

TEST(TranslatorTest, TranslateBatchWithWeaklyBiasedPrefix) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.prefix_bias_beta = 0.01;
  options.beam_size = 2;
  const std::vector<std::vector<std::string>> input = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ز" ,"ا"}
  };
  const std::vector<std::vector<std::string>> prefix = {
    {"a", "t", "s", "s", "s"},   // Test divergence at divergence first 's'
    {},
    {"a", "t", "z", "o"},
    {}
  };
  const auto result = translator.translate_batch(input, prefix, options);
  EXPECT_EQ(result[0].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[1].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[2].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[3].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
}

class BiasedDecodingDeviceFPTest : public ::testing::TestWithParam<FloatType> {
};

TEST_P(BiasedDecodingDeviceFPTest, OneBatchOneBeam) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size},
                       std::vector<float>{4, 6});
    StorageView softmax;
    ops::SoftMax()(logits, softmax);
    std::vector<float> expected_log_probs_vec = {
      std::log((1-prefix_bias_beta) * softmax.at<float>(0) + prefix_bias_beta),
      std::log((1-prefix_bias_beta) * softmax.at<float>(1)),
    };
    StorageView expected_log_probs(logits.shape(), expected_log_probs_vec, device);

    StorageView log_probs(device, dtype);
    const size_t step = 0;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0}};
    ctranslate2::BiasedDecoder biased_decoder(prefix_bias_beta, prefix_ids);
    biased_decoder.decode(batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          logits.to(device).to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float32(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, TwoBatchesTwoBeams) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 2;
    const dim_t beam_size = 2;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{
        4, 6,  // batch1 beam1
        7, 3,  // batch1 beam2
        1, 9,  // batch2 beam1
        8, 2   // batch2 beam2
    });
    StorageView softmax;
    ops::SoftMax()(logits, softmax);
    const std::vector<std::vector<size_t>> prefix_ids = {
      {0},  // bias batch 1 towards token0
      {1}}; // bias batch 2 towards token1
    std::vector<float> expected_log_probs_vec = {
      //batch1 beam1
      std::log((1-prefix_bias_beta) * softmax.at<float>(0) + prefix_bias_beta),
      std::log((1-prefix_bias_beta) * softmax.at<float>(1)),
      //batch1 beam2
      std::log((1-prefix_bias_beta) * softmax.at<float>(2) + prefix_bias_beta),
      std::log((1-prefix_bias_beta) * softmax.at<float>(3)),
      //batch2 beam1
      std::log((1-prefix_bias_beta) * softmax.at<float>(4)),
      std::log((1-prefix_bias_beta) * softmax.at<float>(5) + prefix_bias_beta),
      //batch2 beam2
      std::log((1-prefix_bias_beta) * softmax.at<float>(6)),
      std::log((1-prefix_bias_beta) * softmax.at<float>(7) + prefix_bias_beta),
    };
    StorageView expected_log_probs(logits.shape(), expected_log_probs_vec, device);

    StorageView log_probs(dtype, device);
    const size_t step = 0;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false, false}, {false, false}};
    const std::vector<dim_t> batch_offset = {0, 1};
    BiasedDecoder biased_decoder(prefix_bias_beta, prefix_ids);
    biased_decoder.decode(batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          logits.to(device).to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float32(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, BeamDiverged) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6}, device);
    StorageView expected_log_probs(device);
    ops::LogSoftMax()(logits, expected_log_probs);

    StorageView log_probs(dtype, device);
    const size_t step = 0;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{true}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0}};
    BiasedDecoder biased_decoder(prefix_bias_beta, prefix_ids);
    biased_decoder.decode(batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          logits.to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float32(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, TimeStepPastPrefix) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6}, device);
    StorageView expected_log_probs(device);
    ops::LogSoftMax()(logits, expected_log_probs);

    StorageView log_probs(dtype, device);
    const size_t step = 1;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0}};
    BiasedDecoder biased_decoder(prefix_bias_beta, prefix_ids);
    biased_decoder.decode(batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          logits.to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float32(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, NonZeroTimestepBias) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6});
    StorageView softmax;
    ops::SoftMax()(logits, softmax);
    std::vector<float> expected_log_probs_vec = {
      std::log((1-prefix_bias_beta) * softmax.at<float>(0)),
      std::log((1-prefix_bias_beta) * softmax.at<float>(1) + prefix_bias_beta),
    };
    StorageView expected_log_probs(logits.shape(), expected_log_probs_vec, device);

    StorageView log_probs(dtype, device);
    const size_t step = 1;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{false}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0, 1, 0}};
    BiasedDecoder biased_decoder(prefix_bias_beta, prefix_ids);
    biased_decoder.decode(batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          logits.to(device).to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float32(), expected_log_probs, 0.01);
}

TEST_P(BiasedDecodingDeviceFPTest, NonZeroTimestepDiverge) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const dim_t vocab_size = 2;
    const dim_t batch_size = 1;
    const dim_t beam_size = 1;
    const float prefix_bias_beta = 0.35;

    StorageView logits({batch_size * beam_size, 1, vocab_size}, std::vector<float>{4, 6}, device);
    StorageView expected_log_probs(device);
    ops::LogSoftMax()(logits, expected_log_probs);

    StorageView log_probs(dtype, device);
    const size_t step = 1;
    const std::vector<std::vector<bool>> beams_diverged_from_prefix = {{true}};
    const std::vector<dim_t> batch_offset = {0};
    const std::vector<std::vector<size_t>> prefix_ids = {{0, 1, 0}};
    BiasedDecoder biased_decoder(prefix_bias_beta, prefix_ids);
    biased_decoder.decode(batch_size,
                          step,
                          batch_offset,
                          beams_diverged_from_prefix,
                          logits.to(dtype),
                          log_probs);

    expect_storage_eq(log_probs.to_float32(), expected_log_probs, 0.01);
}

INSTANTIATE_TEST_SUITE_P(CPU, BiasedDecodingDeviceFPTest,
                         ::testing::Values(FloatType{Device::CPU, DataType::FLOAT32}),
                         fp_test_name);
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, BiasedDecodingDeviceFPTest,
                         ::testing::Values(FloatType{Device::CUDA, DataType::FLOAT32},
                                           FloatType{Device::CUDA, DataType::FLOAT16}),
                         fp_test_name);
#endif

TEST(TranslatorTest, TranslatePrefixWithLargeBeam) {
  // Related to issue https://github.com/OpenNMT/CTranslate2/issues/277
  // This is an example where </s> appears in the topk of the first unconstrained decoding
  // step and produces an incorrect hypothesis that dominates others.
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = 5;
  const std::vector<std::string> input = {"أ" ,"و" ,"ل" ,"ي" ,"س" ,"س"};
  const std::vector<std::string> prefix = {"u", "l", "i", "s", "e"};
  const auto result = translator.translate_batch({input}, {prefix}, options)[0];
  EXPECT_EQ(result.output(), (std::vector<std::string>{"u", "l", "i", "s", "e", "s"}));
}

TEST(TranslatorTest, AlternativesFromPrefix) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  options.return_attention = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> prefix = {"a", "t"};
  const TranslationResult result = translator.translate_batch({input}, {prefix}, options)[0];
  ASSERT_EQ(result.num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(result.hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result.hypotheses[1], (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));

  // Tokens at the first unconstrained decoding position should be unique.
  std::vector<std::string> tokens_at_position;
  tokens_at_position.reserve(options.num_hypotheses);
  for (const std::vector<std::string>& hypothesis : result.hypotheses)
    tokens_at_position.emplace_back(hypothesis[prefix.size()]);
  EXPECT_EQ(std::unique(tokens_at_position.begin(), tokens_at_position.end()),
            tokens_at_position.end());

  EXPECT_TRUE(result.has_attention());
  EXPECT_EQ(result.attention[0].size(), 6);
}

TEST(TranslatorTest, AlternativesFromPrefixMinExpansionProb) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_scores = true;
  options.return_attention = true;
  options.return_alternatives = true;
  options.min_alternative_expansion_prob = 0.001;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> prefix = {"a", "t"};
  const size_t expected_alternatives = 6;
  const TranslationResult result = translator.translate_batch({input}, {prefix}, options)[0];
  EXPECT_EQ(result.hypotheses.size(), expected_alternatives);
  EXPECT_EQ(result.scores.size(), expected_alternatives);
  EXPECT_EQ(result.attention.size(), expected_alternatives);
}

TEST(TranslatorTest, AlternativesFromPrefixBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::vector<std::string>> input = {
    {"آ", "ز", "ا"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"}
  };
  const std::vector<std::vector<std::string>> prefix = {{"a"}, {"a", "t"}};
  const auto results = translator.translate_batch(input, prefix, options);
  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(results[0].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[0].hypotheses[0], (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(results[0].hypotheses[1], (std::vector<std::string>{"a", "s", "z", "a"}));
  ASSERT_EQ(results[1].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[1].hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(results[1].hypotheses[1], (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));
}

TEST(TranslatorTest, AlternativesFromScratch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const TranslationResult result = translator.translate_batch({input}, options)[0];
  ASSERT_EQ(result.num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(result.hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}

TEST(TranslatorTest, AlternativesFromScratchBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::vector<std::string>> inputs = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ", "ز", "ا"}
  };
  const std::vector<TranslationResult> results = translator.translate_batch(inputs, options);
  ASSERT_EQ(results.size(), inputs.size());
  ASSERT_EQ(results[0].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[0].hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(results[0].hypotheses[1], (std::vector<std::string>{"e", "t", "z", "m", "o", "n"}));
  ASSERT_EQ(results[1].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[1].hypotheses[0], (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(results[1].hypotheses[1], (std::vector<std::string>{"e", "z", "z", "a"}));
}

TEST(TranslatorTest, AlternativesFromFullTarget) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 4;
  options.return_alternatives = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> target = {"a", "t", "z", "m", "o", "n"};
  const TranslationResult result = translator.translate_batch({input}, {target}, options)[0];
  EXPECT_EQ(result.hypotheses[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n", "e"}));
}

TEST(TranslatorTest, AlternativesMaxDecodingLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 4;
  options.max_decoding_length = 2;
  options.return_alternatives = true;
  options.return_scores = true;
  options.return_attention = true;

  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::vector<std::string>> target_samples = {
    {}, {"a"}, {"a", "t"}, {"a", "t", "z"}
  };

  for (const auto& target : target_samples) {
    const auto result = translator.translate_batch({input}, {target}, options)[0];

    for (size_t i = 0; i < result.num_hypotheses(); ++i) {
      EXPECT_EQ(result.hypotheses[i].size(), options.max_decoding_length);
      EXPECT_EQ(result.attention[i].size(), options.max_decoding_length);

      for (size_t t = 0; t < std::min(target.size(), options.max_decoding_length); ++t) {
        EXPECT_EQ(result.hypotheses[i][t], target[t]);
      }

      if (target.size() < options.max_decoding_length) {
        EXPECT_NE(result.scores[i], 0);
      } else {
        EXPECT_EQ(result.scores[i], 0);
      }
    }
  }
}

TEST(TranslatorTest, InvalidNumHypotheses) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 0;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  EXPECT_THROW(translator.translate_batch({input}, options), std::invalid_argument);
}

TEST(TranslatorTest, Patience) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = 4;
  options.patience = 2;
  options.num_hypotheses = 8;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.num_hypotheses(), options.num_hypotheses);
}

TEST(TranslatorTest, IgnoreScore) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = 1;
  options.return_scores = false;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const TranslationResult result = translator.translate_batch({input}, options)[0];
  EXPECT_FALSE(result.has_scores());
  EXPECT_EQ(result.output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}

TEST(TranslatorTest, SameBeamAndGreedyScore) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.return_scores = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  options.beam_size = 1;
  const auto greedy_score = translator.translate_batch({input}, options)[0].score();
  options.beam_size = 2;
  const auto beam_score = translator.translate_batch({input}, options)[0].score();
  EXPECT_NEAR(greedy_score, beam_score, 1e-5);
}

TEST(TranslatorTest, BeamSizeLargerThanVocabSize) {
  Translator translator = default_translator();
  TranslationOptions options;
  // 22*2=44 candidates are retrieved from the model output but the vocabulary size is 42.
  options.beam_size = 22;
  options.num_hypotheses = options.beam_size;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate_batch({input}, options)[0];
  EXPECT_EQ(result.num_hypotheses(), options.num_hypotheses);
}

TEST(BufferedTranslationWrapperTest, Basic) {
  BufferedTranslationWrapper wrapper(std::make_shared<Translator>(default_model_dir()),
                                     /*max_batch_size=*/32,
                                     /*batch_timeout_in_micros=*/5000);

  auto future1 = wrapper.translate_async({"آ", "ز", "ا"});
  auto future2 = wrapper.translate_async({"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"});

  EXPECT_EQ(future1.get().hypotheses[0],
            (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(future2.get().hypotheses[0],
            (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}

TEST(TranslatorTest, Scoring) {
  const std::vector<std::vector<std::string>> source = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ش" ,"ي" ,"س" ,"و" ,"ن"},
    {"آ" ,"ر" ,"ب" ,"ا" ,"ك" ,"ه"},
    {},
    {"آ" ,"ر" ,"ث" ,"ر"},
  };
  const std::vector<std::vector<std::string>> target = {
    {"a", "t", "z", "m", "o", "n"},
    {"a", "c", "h", "i", "s", "o", "n"},
    {"a", "r", "b", "a", "k", "e"},
    {},
    {"a", "r", "t", "h", "e", "r"},
  };
  const std::vector<std::vector<float>> expected_scores = {
    {-0.106023, -0.065410, -0.056002, -0.447953, -0.230714, -0.092184, -0.063463},
    {-0.072660, -0.300309, -0.181187, -0.395671, -0.025631, -0.123466, -0.002034, -0.012639},
    {-0.103136, -0.089504, -0.063889, -0.007327, -0.452072, -0.060154, -0.016636},
    {0},
    {-0.076704, -0.036037, -0.029253, -0.030273, -0.149276, -0.002440, -0.003742},
  };
  constexpr float abs_diff = 1e-5;

  Translator translator = default_translator();
  const auto scores = translator.score_batch(source, target);

  ASSERT_EQ(scores.size(), expected_scores.size());
  for (size_t i = 0; i < scores.size(); ++i)
    expect_vector_eq(scores[i].tokens_score, expected_scores[i], abs_diff);
}

TEST(TranslatorTest, ScoringMaxInputLength) {
  const std::vector<std::string> source = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> target = {"a", "t", "z", "m", "o", "n"};

  ScoringOptions options;
  options.max_input_length = 4;
  Translator translator = default_translator();
  const auto result = translator.score_batch({source}, {target}, options)[0];

  EXPECT_EQ(result.tokens, (std::vector<std::string>{"a", "t", "z", "</s>"}));
  EXPECT_EQ(result.tokens_score.size(), options.max_input_length);
}

// TransformerDecoder::decode verifies the decoding step against the "self_length" state
// entry, which records the exact number of steps written to the preallocated
// self-attention caches. The regular translation paths never trip the guard, so these
// tests drive the decoder directly with inconsistent steps.
class TransformerDecoderTest : public ::testing::Test {
protected:
  TransformerDecoderTest()
    : _model(models::Model::load(default_model_dir()))
    , _encoder(*_model, "encoder")
    , _decoder(*_model, "decoder") {
  }

  layers::DecoderState make_state(bool iterative_decoding = true) {
    StorageView src_ids({1, 4}, std::vector<int32_t>{3, 4, 5, 6});
    StorageView src_lengths({1}, int32_t(4));
    StorageView memory(_encoder.output_type());
    _encoder({src_ids}, &src_lengths, memory);
    auto state = _decoder.initial_state(iterative_decoding);
    state.emplace("memory", std::move(memory));
    state.emplace("memory_lengths", std::move(src_lengths));
    return state;
  }

  std::shared_ptr<const models::Model> _model;
  layers::TransformerEncoder _encoder;
  layers::TransformerDecoder _decoder;
};

TEST_F(TransformerDecoderTest, CacheLengthGuardStepByStep) {
  auto state = make_state();
  StorageView tok({1}, int32_t(1));
  StorageView logits;

  _decoder(0, tok, state, &logits);
  _decoder(1, tok, state, &logits);
  EXPECT_THROW(_decoder(3, tok, state, &logits), std::runtime_error);  // Off by +1.
  EXPECT_THROW(_decoder(1, tok, state, &logits), std::runtime_error);  // Off by -1.
  EXPECT_THROW(_decoder(0, tok, state, &logits), std::runtime_error);  // Reset counter.
  // The guard threw before any cache write, so the state is still consistent.
  _decoder(2, tok, state, &logits);

  const auto& record = state.at("self_length");
  EXPECT_EQ(record.dim(0), 1);
  EXPECT_EQ(record.dim(1), 3);
  EXPECT_EQ(record.dtype(), DataType::INT8);
}

TEST_F(TransformerDecoderTest, CacheLengthGuardBulkPrompt) {
  auto state = make_state();
  StorageView prompt({1, 3}, std::vector<int32_t>{1, 3, 4});
  StorageView tok({1}, int32_t(1));
  StorageView logits;

  _decoder(0, prompt, state, &logits);
  EXPECT_THROW(_decoder(2, tok, state, &logits), std::runtime_error);  // prompt_len - 1.
  EXPECT_THROW(_decoder(4, tok, state, &logits), std::runtime_error);  // prompt_len + 1.
  _decoder(3, tok, state, &logits);
}

TEST_F(TransformerDecoderTest, CacheLengthGuardFreshStateNonZeroStep) {
  auto state = make_state();
  StorageView tok({1}, int32_t(1));
  StorageView logits;

  // The block-level check in append_to_cache also rejects step 5 on an empty cache, so
  // pin the message to make sure the step-level guard fired first.
  try {
    _decoder(5, tok, state, &logits);
    FAIL() << "expected the cache length guard to throw";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("does not match the number of steps"),
              std::string::npos)
      << "unexpected message: " << e.what();
  }
}

TEST_F(TransformerDecoderTest, CacheLengthSurvivesReplication) {
  auto state = make_state();
  StorageView tok({1}, int32_t(1));
  StorageView logits;

  _decoder(0, tok, state, &logits);
  _decoder(1, tok, state, &logits);

  // The name is qualified because TransformerDecoder::replicate_state(name) hides the
  // base overload.
  _decoder.layers::Decoder::replicate_state(state, /*beam_size=*/2);
  _decoder.update_state(state, StorageView({2}, std::vector<int32_t>{1, 0}), /*beam_size=*/2);

  StorageView tok2({2}, int32_t(1));
  EXPECT_THROW(_decoder(1, tok2, state, &logits), std::runtime_error);
  EXPECT_THROW(_decoder(3, tok2, state, &logits), std::runtime_error);
  _decoder(2, tok2, state, &logits);

  const auto& record = state.at("self_length");
  EXPECT_EQ(record.dim(0), 2);
  EXPECT_EQ(record.dim(1), 3);
}

TEST_F(TransformerDecoderTest, LayerSlotsPingPongAcrossSteps) {
  // The iterative decode path binds the two layer activations to workspace slots. After
  // the first step both slots must hold an allocation and the {layer_in, layer_out} buffer
  // pair must be stable across steps (each layer swaps them, so only the pairing order
  // may flip), while the logits stay identical to a fresh decoder instance.
  struct TestDecoder : layers::TransformerDecoder {
    using layers::TransformerDecoder::TransformerDecoder;
    const layers::DecodeWorkspace& workspace() const {
      return _workspace;
    }
  };

  TestDecoder decoder(*_model, "decoder");
  layers::TransformerDecoder reference(*_model, "decoder");

  auto run = [&](layers::TransformerDecoder& dec,
                 std::vector<StorageView>& logits_per_step,
                 const TestDecoder* watch) {
    auto state = make_state();
    StorageView tok({1}, int32_t(1));
    const void* slot_a = nullptr;
    const void* slot_b = nullptr;
    for (dim_t step = 0; step < 5; ++step) {
      StorageView logits;
      dec(step, tok, state, &logits);
      logits_per_step.emplace_back(std::move(logits));
      if (watch) {
        const void* in_buffer = watch->workspace().layer_in.buffer();
        const void* out_buffer = watch->workspace().layer_out.buffer();
        EXPECT_NE(in_buffer, nullptr) << "step " << step;
        EXPECT_NE(out_buffer, nullptr) << "step " << step;
        if (step == 0) {
          slot_a = in_buffer;
          slot_b = out_buffer;
        } else {
          EXPECT_TRUE((in_buffer == slot_a && out_buffer == slot_b)
                      || (in_buffer == slot_b && out_buffer == slot_a))
            << "step " << step << " does not reuse the two step-0 buffers";
        }
      }
    }
  };

  std::vector<StorageView> logits_first;
  std::vector<StorageView> logits_second;
  std::vector<StorageView> logits_reference;
  run(decoder, logits_first, &decoder);
  run(decoder, logits_second, &decoder);  // Reused instance: slots survive decodes.
  run(reference, logits_reference, nullptr);

  for (size_t i = 0; i < logits_reference.size(); ++i) {
    expect_storage_eq(logits_first[i], logits_reference[i]);
    expect_storage_eq(logits_second[i], logits_reference[i]);
  }
}

TEST_F(TransformerDecoderTest, AlignmentHeadsDeviceMemo) {
  // No heads configured: no tensor is built at all.
  EXPECT_EQ(_decoder.get_layer_alignment_heads(0, 2), nullptr);

  _decoder.set_alignment_heads(/*layer=*/0, /*num_heads_to_average=*/2);
  const StorageView* first = _decoder.get_layer_alignment_heads(0, 2);
  ASSERT_NE(first, nullptr);
  expect_storage_eq(*first, StorageView({2, 2}, std::vector<int32_t>{0, 1, 0, 1}));

  // Same batch size: the cached tensor (and its buffer) is returned, not a rebuild.
  const void* buffer = first->buffer();
  const StorageView* again = _decoder.get_layer_alignment_heads(0, 2);
  EXPECT_EQ(again, first);
  EXPECT_EQ(again->buffer(), buffer);

  // A batch size change rebuilds with the new content.
  const StorageView* grown = _decoder.get_layer_alignment_heads(0, 3);
  ASSERT_NE(grown, nullptr);
  expect_storage_eq(*grown, StorageView({3, 2}, std::vector<int32_t>{0, 1, 0, 1, 0, 1}));

  // Reconfiguring the heads invalidates the cache even at an unchanged batch size.
  _decoder.set_alignment_heads(/*layer=*/0, /*num_heads_to_average=*/1);
  const StorageView* reconfigured = _decoder.get_layer_alignment_heads(0, 3);
  ASSERT_NE(reconfigured, nullptr);
  expect_storage_eq(*reconfigured, StorageView({3, 1}, std::vector<int32_t>{0, 0, 0}));
}

TEST_F(TransformerDecoderTest, CacheReservePreallocatesUpFront) {
  // Capped reserve (the CT2_CUDA_GRAPHS_RESERVE case where the knob < max_length): the
  // cache must be sized once at step 0 to the block-rounded cap and stay there for every
  // step that fits under it -- it is preallocated up front, not grown block by block --
  // while the logits stay identical to an unreserved decoder at every step. dim(2) holding
  // constant is exactly the property the graph runner's capacity fingerprint depends on.
  layers::TransformerDecoder reference(*_model, "decoder");
  _decoder.set_cache_reserve_steps(40);

  auto state = make_state();
  auto ref_state = make_state();
  StorageView tok({1}, int32_t(1));

  // Step 0 allocates the whole reserve up front: round_up(40, 32) = 64. The unreserved
  // reference, in contrast, is still at one block (32) after the same first step -- it only
  // grows later, block by block. This up-front-vs-incremental gap is the whole point.
  {
    StorageView logits;
    StorageView ref_logits;
    _decoder(0, tok, state, &logits);
    reference(0, tok, ref_state, &ref_logits);
    expect_storage_eq(logits, ref_logits);
  }
  EXPECT_EQ(state.at("self_keys_0").dim(2), 64);
  EXPECT_EQ(ref_state.at("self_keys_0").dim(2), 32);

  // Every later step under the cap reuses that capacity: the reserved cache's dim(2) never
  // changes while the window (40 steps) fits, and the logits stay identical to the reference
  // at each step (the reference grows to 64 on its own around step 32 -- also correct).
  for (dim_t step = 1; step < 40; ++step) {
    StorageView logits;
    StorageView ref_logits;
    _decoder(step, tok, state, &logits);
    reference(step, tok, ref_state, &ref_logits);
    expect_storage_eq(logits, ref_logits);
    EXPECT_EQ(state.at("self_keys_0").dim(2), 64) << "cap must hold at step " << step;
  }
  // The self_length record mirrors the reserve.
  EXPECT_GE(state.at("self_length").reserved_memory(), dim_t(40 * 16));

  // The step-exactness guard is untouched by the relaxation.
  StorageView logits;
  EXPECT_THROW(_decoder(8, tok, state, &logits), std::runtime_error);
}

TEST_F(TransformerDecoderTest, CacheReserveGrowthFallback) {
  // A window longer than the cap (the CT2_CUDA_GRAPHS_RESERVE crossing case): once the
  // reserved capacity is exhausted the cache falls back to 32-block growth, one block at a
  // time, and every step -- before and after each crossing -- stays bit-identical to an
  // unreserved decoder. This is what makes the single-cap graph policy safe: past the cap
  // the tail simply grows the cache the ordinary way (under CUDA graphs the host guard then
  // turns that step ineligible and the tail runs eager).
  layers::TransformerDecoder reference(*_model, "decoder");
  _decoder.set_cache_reserve_steps(40);  // cap = round_up(40, 32) = 64

  auto state = make_state();
  auto ref_state = make_state();
  StorageView tok({1}, int32_t(1));

  dim_t last_capacity = 0;
  for (dim_t step = 0; step < 100; ++step) {
    StorageView logits;
    StorageView ref_logits;
    _decoder(step, tok, state, &logits);
    reference(step, tok, ref_state, &ref_logits);
    expect_storage_eq(logits, ref_logits);

    const dim_t capacity = state.at("self_keys_0").dim(2);
    EXPECT_EQ(capacity % 32, 0) << "growth must stay 32-block aligned at step " << step;
    if (capacity != last_capacity) {
      if (last_capacity != 0) {
        EXPECT_EQ(capacity, last_capacity + 32) << "one block at a time at step " << step;
      }
      last_capacity = capacity;
    }
  }

  // Two crossings past the cap: 64 -> 96 (capacity exceeded at step 64) -> 128 (step 96).
  EXPECT_EQ(state.at("self_keys_0").dim(2), 128);
  EXPECT_EQ(state.at("self_length").dim(1), 100);
}

TEST_F(TransformerDecoderTest, CacheCapacityIsWitnessedByDim2) {
  // The CUDA-graph fingerprint (transformer.cc, guarded by CT2_WITH_CUDA) bakes each KV
  // cache's time capacity dim(2) alongside its buffer address, because both CUDA allocators
  // deterministically recycle freed blocks: a later, smaller-cap decode can be handed the
  // same buffer back at a different capacity, and a captured GEMM/softmax leading dimension
  // would then silently disagree with the new layout. That backstop is only sound if dim(2)
  // is a faithful witness of the capacity -- the field that distinguishes two otherwise
  // identically shaped caches. This pins that invariant on the CPU build (the fingerprint
  // push itself is CUDA-only, inspected at transformer.cc and dual-run-checked by Gate G3):
  // the reserve is the only thing changed, and it moves dim(2) alone.
  _decoder.set_cache_reserve_steps(40);       // -> round_up(40, 32) = 64
  layers::TransformerDecoder big_reserve(*_model, "decoder");
  big_reserve.set_cache_reserve_steps(200);   // -> round_up(200, 32) = 224

  auto small_state = make_state();
  auto big_state = make_state();
  StorageView tok({1}, int32_t(1));
  StorageView small_logits;
  StorageView big_logits;
  _decoder(0, tok, small_state, &small_logits);
  big_reserve(0, tok, big_state, &big_logits);

  const StorageView& a = small_state.at("self_keys_0");
  const StorageView& b = big_state.at("self_keys_0");
  ASSERT_EQ(a.rank(), 4);
  ASSERT_EQ(b.rank(), 4);
  // Batch, heads and depth are identical; the capacity (dim 2) is the sole differentiator,
  // so it must be part of the fingerprint or a recycled buffer would masquerade as a match.
  EXPECT_EQ(a.dim(0), b.dim(0));
  EXPECT_EQ(a.dim(1), b.dim(1));
  EXPECT_EQ(a.dim(3), b.dim(3));
  EXPECT_NE(a.dim(2), b.dim(2));
  EXPECT_EQ(a.dim(2), 64);
  EXPECT_EQ(b.dim(2), 224);
}

TEST_F(TransformerDecoderTest, CacheReserveKnobClampAndBlockRounding) {
  // The CT2_CUDA_GRAPHS_RESERVE clamp -- reserve = knob > 0 ? min(max_length, knob)
  // : max_length -- is applied by WhisperReplica::generate through the free function
  // models::clamp_cache_reserve_steps (whisper.cc). Driving generate() directly would
  // need a WhisperModel fixture the CPU test data does not ship (the env-gated wiring is
  // additionally covered end-to-end on the A10G box by Gate G1, RESERVE=128 -> measured
  // cache capacity 128), so here we call the SAME production function -- not a copy of it
  // -- and pin the two behaviours the knob relies on: the clamp arithmetic itself, and
  // that the clamped value block-rounds through append_to_cache to dim(2) = round_up(reserve, 32).
  const dim_t max_length = 448;  // whisper-small options.max_length
  EXPECT_EQ(models::clamp_cache_reserve_steps(max_length, 0), 448);    // unset -> full max_length (bit-identical path)
  EXPECT_EQ(models::clamp_cache_reserve_steps(max_length, -1), 448);   // negative treated as unset
  EXPECT_EQ(models::clamp_cache_reserve_steps(max_length, 512), 448);  // above max_length -> clamped down
  EXPECT_EQ(models::clamp_cache_reserve_steps(max_length, 448), 448);  // exactly max_length -> unchanged
  EXPECT_EQ(models::clamp_cache_reserve_steps(max_length, 128), 128);  // below max_length -> the knob
  EXPECT_EQ(models::clamp_cache_reserve_steps(max_length, 96), 96);

  // knob -> the block-rounded capacity the first append allocates for the clamped reserve.
  const std::vector<std::pair<int, dim_t>> knob_to_dim2 = {
    {0, 448},    // unset: no cap, full max_length (already a 32 multiple)
    {128, 128},  // 4 blocks exactly
    {96, 96},    // 3 blocks exactly
    {160, 160},  // 5 blocks exactly
    {100, 128},  // block rounding: round_up(100, 32) = 128
  };
  for (const auto& knob_dim2 : knob_to_dim2) {
    const int knob = knob_dim2.first;
    const dim_t expected_dim2 = knob_dim2.second;
    const dim_t reserve = models::clamp_cache_reserve_steps(max_length, knob);
    _decoder.set_cache_reserve_steps(reserve);
    auto state = make_state();
    StorageView tok({1}, int32_t(1));
    StorageView logits;
    _decoder(0, tok, state, &logits);
    EXPECT_EQ(state.at("self_keys_0").dim(2), expected_dim2)
        << "knob=" << knob << " reserve=" << reserve;
  }
}

TEST_F(TransformerDecoderTest, CacheLengthGuardIgnoresScoring) {
  auto state = make_state(/*iterative_decoding=*/false);
  StorageView target({1, 4}, std::vector<int32_t>{1, 3, 4, 5});
  StorageView lengths({1}, int32_t(4));
  StorageView logits;

  // The full-sequence overload decodes with step -1 and no cache, so it is exempt.
  _decoder(target, lengths, state, logits);
  EXPECT_EQ(state.find("self_length"), state.end());
}
