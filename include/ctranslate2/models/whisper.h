#pragma once

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/whisper.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    struct WhisperOptions {
      // Beam size to use for beam search (set 1 to run greedy search).
      size_t beam_size = 5;

      // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
      // The decoding will continue until beam_size*patience hypotheses are finished.
      float patience = 1;

      // Exponential penalty applied to the length during beam search.
      float length_penalty = 1;

      // Penalty applied to the score of previously generated tokens, as described in
      // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
      float repetition_penalty = 1;

      // Prevent repetitions of ngrams with this size (set 0 to disable).
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 448;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // High temperatures increase randomness.
      float sampling_temperature = 1;

      // Number of hypotheses to include in the result.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Include log probs of each token in the result
      bool return_logits_vocab = false;

      // Include the probability of the no speech token in the result.
      bool return_no_speech_prob = false;

      // Maximum index of the first predicted timestamp.
      size_t max_initial_timestamp_index = 50;

      // Suppress blank outputs at the beginning of the sampling.
      bool suppress_blank = true;

      // List of token IDs to suppress.
      // -1 will suppress a default set of symbols as defined in the model config.json file.
      std::vector<int> suppress_tokens = {-1};
    };

    // Resolve how many decode steps the KV caches are reserved for under the opt-in
    // preallocation path, given the CT2_CUDA_GRAPHS_RESERVE knob. An unset knob (<= 0)
    // keeps the full decode length so the path stays bit-identical; a positive knob
    // reserves only min(max_length, knob) steps. This is the exact clamp applied by
    // WhisperReplica::generate, factored out here so it is directly unit-testable.
    dim_t clamp_cache_reserve_steps(dim_t max_length, int reserve_knob);

    // Parse the CT2_CUDA_GRAPHS_TIERS capacity ladder for one decode, given the decode
    // length and the base reserve R (clamp_cache_reserve_steps above). With
    // base = round_up(R, 32) and top = round_up(max_length, 32):
    // - "", "0", "off", "false": inactive (the single-cap policy);
    // - "1", "on", "true": the "+64" alias;
    // - "+N" (N >= 1): relative stride max(32, round_up(N, 32)), next = min(top, C + S);
    // - "a,b,...": absolute tiers, each block-rounded and clamped to top, keeping only
    //   those > base (sorted, deduplicated); top is not appended implicitly;
    // - anything else: inactive, with one warning.
    // The policy is also inactive when base >= top (e.g. an unset reserve).
    layers::CacheTierPolicy parse_cache_tier_policy(const std::string& spec,
                                                    dim_t max_length,
                                                    dim_t reserve);

    struct WhisperGenerationResult {
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;
      std::vector<std::vector<StorageView>> logits;
      float no_speech_prob = 0;

      size_t num_sequences() const {
        return sequences.size();
      }

      bool has_scores() const {
        return !scores.empty();
      }
    };

    struct WhisperAlignmentResult {
      std::vector<std::pair<dim_t, dim_t>> alignments;
      std::vector<float> text_token_probs;
    };

    class WhisperModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };

    class WhisperReplica : public ModelReplica {
    public:
      static std::unique_ptr<WhisperReplica> create_from_model(const Model& model);

      WhisperReplica(const std::shared_ptr<const WhisperModel>& model);

      bool is_multilingual() const {
        return _is_multilingual;
      }

      size_t n_mels() const {
        return _n_mels;
      }

      size_t num_languages() const {
        return _num_languages;
      }

      StorageView encode(StorageView features, const bool to_cpu);

      std::vector<WhisperGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<std::string>>& prompts,
               const WhisperOptions& options);

      std::vector<WhisperGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<size_t>>& prompts,
               const WhisperOptions& options);

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(StorageView features);

      std::vector<WhisperAlignmentResult>
      align(StorageView features,
            const std::vector<size_t>& start_sequence,
            const std::vector<std::vector<size_t>>& text_tokens,
            std::vector<size_t> num_frames,
            dim_t median_filter_width);

    private:
      const std::shared_ptr<const WhisperModel> _model;
      const std::unique_ptr<layers::WhisperEncoder> _encoder;
      const std::unique_ptr<layers::WhisperDecoder> _decoder;

      size_t _sot_id;
      size_t _eot_id;
      size_t _no_timestamps_id;
      size_t _no_speech_id;
      size_t _n_mels;
      size_t _num_languages;
      bool _is_multilingual;

      StorageView maybe_encode(StorageView features);
    };

    class Whisper : public ReplicaPool<WhisperReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      bool is_multilingual() const;
      size_t n_mels() const;
      size_t num_languages() const;

      std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

      std::vector<std::future<WhisperGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<std::string>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<WhisperGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<size_t>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<std::vector<std::pair<std::string, float>>>>
      detect_language(const StorageView& features);

      std::vector<std::future<WhisperAlignmentResult>>
      align(const StorageView& features,
            std::vector<size_t> start_sequence,
            std::vector<std::vector<size_t>> text_tokens,
            std::vector<size_t> num_frames,
            dim_t median_filter_width);

    };

  }
}
