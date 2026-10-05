// Tests of the device-agnostic half of the piecewise CUDA-graph path
// (CT2_CUDA_GRAPHS=1 with CT2_CUDA_GRAPHS_PIECEWISE=1): exact-length self-attention cores
// reported to a layers::SegmentHook at every layer boundary, re-issued from a CoreDesc by
// layers::replay_core, and the decoder's eager rerun of a step whose capture aborted.
// The layer-level tests drive MultiHeadAttention directly with a shared DecodeWorkspace;
// the decoder-level tests go through TransformerDecoder::_piecewise_test_hook, which runs
// a step exactly as a piecewise capture does minus the CUDA calls. Every comparison is
// byte for byte: the goal of the mode is bit identity with the default eager path.

#include <ctranslate2/layers/attention.h>
#include <ctranslate2/layers/transformer.h>
#include <ctranslate2/models/model.h>
#include <ctranslate2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layers/decode_core.h"
#include "test_utils.h"

namespace {

  using Bytes = std::vector<uint8_t>;

  Bytes host_bytes(const StorageView& x) {
    const StorageView c = x.to(Device::CPU);
    const auto* p = static_cast<const uint8_t*>(c.buffer());
    return Bytes(p, p + c.size() * c.item_size());
  }

  dim_t round_up_block(const dim_t steps) {
    return (steps + 31) / 32 * 32;
  }

  dim_t item_size_of(const DataType dtype) {
    return StorageView(dtype).item_size();
  }

  bool is_self_cache(const std::string& name) {
    return starts_with(name, "self_keys_") || starts_with(name, "self_values_");
  }

  // Records every core boundary of a step and checks, at each begin_segment, that
  // replay_core re-issues the core bit for bit: the live scores and context are saved,
  // both buffers are poisoned, the core is replayed from the descriptor, and the bytes
  // must come back unchanged. A replay that wrote nothing, or wrote anything else, also
  // corrupts what the next layer reads, so the decoder-level comparisons catch it too.
  // Optionally throws at one boundary (the abort of a piecewise capture) or re-enters
  // the decoder's structural guard (a protocol violation that must abort the same way).
  class CoreReplayHook : public layers::SegmentHook {
  public:
    enum class Where { End, Begin };
    enum class Reentry { None, EndTwice, BeginTwice };

    // Per-step record, reset by start_step().
    std::vector<layers::CoreDesc> cores;
    size_t ends = 0;
    size_t begins = 0;
    bool alternation_error = false;

    // Totals over the hook's lifetime.
    size_t total_ends = 0;
    size_t replay_failures = 0;
    size_t replay_mismatches = 0;

    // Replays of a descriptor captured at an earlier step (see check_replay). Off when
    // stale_period is 0; otherwise the parity period of the decode (1 greedy, 2 beam).
    dim_t stale_period = 0;
    dim_t stale_from = 2;
    size_t stale_replays = 0;
    size_t stale_recaptures = 0;
    size_t stale_failures = 0;
    size_t stale_mismatches = 0;

    // Fault injection.
    dim_t throw_step = -1;
    dim_t throw_core = -1;
    Where throw_where = Where::End;
    Reentry reentry = Reentry::None;
    const layers::DecodeWorkspace* workspace = nullptr;  // For the reentry cases.
    bool fired = false;
    dim_t fired_step = -1;
    bool guard_threw = false;

    void start_step(const dim_t step) {
      _step = step;
      cores.clear();
      ends = 0;
      begins = 0;
      alternation_error = false;
      _in_core = false;
    }

    void end_segment(const layers::CoreDesc& core) override {
      if (_in_core)
        alternation_error = true;
      _in_core = true;
      ++ends;
      ++total_ends;
      cores.push_back(core);
      if (fire(core.index, Where::End)) {
        if (reentry == Reentry::EndTwice) {
          try {
            workspace->graph_hook->end_segment(core);
          } catch (const std::runtime_error&) {
            guard_threw = true;
            throw;
          }
        }
        throw std::runtime_error("injected end_segment failure");
      }
    }

    void begin_segment() override {
      if (!_in_core || cores.empty())
        alternation_error = true;
      _in_core = false;
      ++begins;
      if (!cores.empty())
        check_replay(cores.back());
      if (!cores.empty() && fire(cores.back().index, Where::Begin)) {
        if (reentry == Reentry::BeginTwice) {
          try {
            workspace->graph_hook->begin_segment();
          } catch (const std::runtime_error&) {
            guard_threw = true;
            throw;
          }
        }
        throw std::runtime_error("injected begin_segment failure");
      }
    }

  private:
    bool fire(const dim_t core, const Where where) {
      if (fired || _step != throw_step || core != throw_core || where != throw_where)
        return false;
      fired = true;
      fired_step = _step;
      return true;
    }

    void check_replay(const layers::CoreDesc& core) {
      const dim_t item = item_size_of(core.dtype);
      const size_t scores_bytes = core.batch * core.heads * core.length * item;
      const size_t context_bytes = core.batch * core.heads * core.depth * item;
      Bytes live_scores(scores_bytes);
      Bytes live_context(context_bytes);
      std::memcpy(live_scores.data(), core.scores, scores_bytes);
      std::memcpy(live_context.data(), core.context, context_bytes);
      const auto replay_matches = [&](const layers::CoreDesc& desc, size_t& failures,
                                      size_t& mismatches) {
        std::memset(core.scores, 0xFF, scores_bytes);  // NaN for every float type.
        std::memset(core.context, 0xFF, context_bytes);
        if (!layers::replay_core(desc, core.length)) {
          ++failures;
          // Restore the live values so the rest of the forward stays meaningful.
          std::memcpy(core.scores, live_scores.data(), scores_bytes);
          std::memcpy(core.context, live_context.data(), context_bytes);
          return;
        }
        // A mismatch is left in place: the next layer then reads it, so the decoder-level
        // comparisons fail too.
        mismatches += std::memcmp(live_scores.data(), core.scores, scores_bytes) != 0
                      || std::memcmp(live_context.data(), core.context, context_bytes) != 0;
      };
      replay_matches(core, replay_failures, replay_mismatches);

      // What a CUDA replay actually issues: the descriptor captured at an earlier step for
      // this parity slot, re-run at the current step's length (replay_piecewise with
      // g.cores[k]). The first descriptor seen per (core, parity) from stale_from on plays
      // the capture; a change of buffers or capacity is a new capture, as the CUDA
      // fingerprint and tier transitions make it.
      if (stale_period <= 0 || _step < stale_from)
        return;
      const auto key = std::make_pair(core.index, _step % stale_period);
      const auto it = _captured.find(key);
      if (it == _captured.end() || it->second.queries != core.queries
          || it->second.keys != core.keys || it->second.values != core.values
          || it->second.scores != core.scores || it->second.context != core.context
          || it->second.capacity != core.capacity) {
        stale_recaptures += it != _captured.end();
        _captured[key] = core;
        return;
      }
      ++stale_replays;
      replay_matches(it->second, stale_failures, stale_mismatches);
    }

    dim_t _step = -1;
    bool _in_core = false;
    std::map<std::pair<dim_t, dim_t>, layers::CoreDesc> _captured;
  };

  // ---------------------------------------------------------------------------------
  // Layer level: two self-attention layers sharing one workspace (no cross-attention).

  std::vector<float> randn(size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n);
    for (auto& x : v)
      x = d(g);
    return v;
  }

  class SelfAttnModel : public models::Model {
  public:
    SelfAttnModel(dim_t d_model, unsigned seed) {
      std::mt19937 g(seed);
      const float s = 1.f / std::sqrt(float(d_model));
      for (int l = 0; l < 2; ++l) {
        const std::string p = "attn" + std::to_string(l);
        register_variable(p + "/linear_0/weight",
                          StorageView({3 * d_model, d_model},
                                      randn(3 * d_model * d_model, g, s)));
        register_variable(p + "/linear_0/bias",
                          StorageView({3 * d_model}, randn(3 * d_model, g, 0.1f)));
        register_variable(p + "/linear_1/weight",
                          StorageView({d_model, d_model}, randn(d_model * d_model, g, s)));
        register_variable(p + "/linear_1/bias",
                          StorageView({d_model}, randn(d_model, g, 0.1f)));
      }
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
      set_device(Device::CPU);
    }

  protected:
    std::unique_ptr<Model> clone() const override {
      return nullptr;
    }
  };

  constexpr dim_t kLayerHeads = 12;
  constexpr dim_t kLayerDepth = 64;
  constexpr dim_t kLayerModel = kLayerHeads * kLayerDepth;

  enum class LayerMode {
    Default,      // Stock iterative decoding: no reserve, direct append.
    Piecewise,    // exact_core + indirect append + reserved slots + the recording hook.
    ExactNoPin,   // exact_core + indirect append, no reservation, no hook.
  };

  struct LayerRun {
    std::vector<Bytes> outputs;                  // Per step.
    std::vector<std::vector<layers::CoreDesc>> cores;  // Per step (Piecewise only).
    std::vector<std::pair<const void*, const void*>> slots;  // {fused_proj, attn} per step.
    std::vector<bool> alternation_errors;
    std::vector<dim_t> piecewise_cores;
    bool fallback = false;
    size_t replay_failures = 0;
    size_t replay_mismatches = 0;
    size_t stale_replays = 0;
    size_t stale_recaptures = 0;
    size_t stale_failures = 0;
    size_t stale_mismatches = 0;
  };

  LayerRun run_layers(const dim_t batch, const dim_t reserve, const dim_t steps,
                      const LayerMode mode) {
    SelfAttnModel model(kLayerModel, 7);
    const layers::MultiHeadAttention a0(model, "attn0", kLayerHeads, true, true, true);
    const layers::MultiHeadAttention a1(model, "attn1", kLayerHeads, true, true, true);
    a0.set_cache_reserve_steps(mode == LayerMode::Default ? 0 : reserve);
    a1.set_cache_reserve_steps(mode == LayerMode::Default ? 0 : reserve);

    StorageView k0, v0, k1, v1;
    layers::DecodeWorkspace w;
    CoreReplayHook hook;
    hook.stale_period = 1;
    LayerRun run;
    std::mt19937 g(99);
    for (dim_t step = 0; step < steps; ++step) {
      const StorageView x({batch, 1, kLayerModel}, randn(batch * kLayerModel, g, 1.f));
      w.padded_kv = false;
      w.padded_kv_fallback = false;
      w.exact_core = mode != LayerMode::Default;
      w.graph_indirect = mode != LayerMode::Default;
      if (w.graph_indirect)
        w.step_state = StorageView({2}, std::vector<int32_t>{int32_t(step), int32_t(step)});
      if (mode == LayerMode::Piecewise) {
        // As the decoder does before the forward: the capacity this step's append ends
        // with (the cache grows one block at a time past the reserve).
        w.reserve_core_slots(DataType::FLOAT32, Device::CPU,
                             batch * kLayerHeads * round_up_block(std::max(reserve, step + 1)));
        w.graph_hook = &hook;
      }
      w.piecewise_cores = 0;
      hook.start_step(step);

      StorageView h(DataType::FLOAT32, Device::CPU);
      a0(x, x, nullptr, h, &k0, &v0, nullptr, nullptr, nullptr, true, nullptr, step, &w);
      const StorageView hin(h);
      StorageView out(DataType::FLOAT32, Device::CPU);
      a1(hin, hin, nullptr, out, &k1, &v1, nullptr, nullptr, nullptr, true, nullptr, step, &w);
      w.graph_hook = nullptr;

      run.fallback = run.fallback || w.padded_kv_fallback;
      run.outputs.emplace_back(host_bytes(out));
      run.cores.emplace_back(hook.cores);
      run.slots.emplace_back(w.fused_proj.buffer(), w.attn.buffer());
      run.alternation_errors.push_back(hook.alternation_error || hook.ends != hook.begins);
      run.piecewise_cores.push_back(w.piecewise_cores);
    }
    run.replay_failures = hook.replay_failures;
    run.replay_mismatches = hook.replay_mismatches;
    run.stale_replays = hook.stale_replays;
    run.stale_recaptures = hook.stale_recaptures;
    run.stale_failures = hook.stale_failures;
    run.stale_mismatches = hook.stale_mismatches;
    return run;
  }

  // Distinct {fused_proj, attn} buffers over the run, and the new ones from step 2 on.
  struct SlotStats {
    size_t distinct = 0;
    size_t new_after_warmup = 0;
    dim_t first_new_step = -1;
  };

  SlotStats slot_stats(const LayerRun& run) {
    SlotStats stats;
    std::set<const void*> seen;
    for (size_t step = 0; step < run.slots.size(); ++step) {
      for (const void* p : {run.slots[step].first, run.slots[step].second}) {
        const bool fresh = seen.insert(p).second;
        if (fresh && step >= 2) {
          ++stats.new_after_warmup;
          if (stats.first_new_step < 0)
            stats.first_new_step = dim_t(step);
        }
      }
    }
    stats.distinct = seen.size();
    return stats;
  }

  // ---------------------------------------------------------------------------------
  // Decoder level: the aren-transliteration test model (6 decoder layers with
  // cross-attention), through TransformerDecoder::_piecewise_test_hook.

  struct PiecewiseTestDecoder : layers::TransformerDecoder {
    using layers::TransformerDecoder::TransformerDecoder;
    void set_hook(layers::SegmentHook* hook) {
      _piecewise_test_hook = hook;
    }
    const layers::DecodeWorkspace& workspace() const {
      return _workspace;
    }
    dim_t num_layers() const {
      return dim_t(_layers.size());
    }
  };

  struct DecoderConfig {
    dim_t reserve = 128;
    dim_t steps = 100;
    dim_t batch = 1;
    dim_t beam = 1;
    int prefix = -1;  // set_prefix_reorder: -1 leaves the default (off on CPU).
  };

  struct DecoderRun {
    std::vector<Bytes> logits;                          // Per step.
    std::vector<std::map<std::string, Bytes>> caches;   // Self caches after each step.
    std::vector<bool> hooked;                           // The step reached the hook.
    std::vector<bool> expected_hooked;                  // step + 1 <= capacity before it.
    std::vector<std::vector<layers::CoreDesc>> cores;   // Per step.
    std::vector<bool> structure_errors;                 // Count/order/length/alternation.
    std::vector<std::vector<uintptr_t>> slot_sets;      // Sorted 11-slot buffer set.
    std::vector<dim_t> record_steps;                    // self_length.dim(1) after the step.
    std::vector<bool> flags_clean;                      // graph_hook == nullptr after.
    std::vector<bool> flags_off;                        // exact_core etc. all off after.
    std::map<std::string, dim_t> slot_reserved;         // After the last step.
    size_t replay_failures = 0;
    size_t replay_mismatches = 0;
    size_t stale_replays = 0;
    size_t stale_recaptures = 0;
    size_t stale_failures = 0;
    size_t stale_mismatches = 0;
    dim_t num_layers = 0;
  };

  class PiecewiseDecoderTest : public ::testing::Test {
  protected:
    PiecewiseDecoderTest()
      : _model(models::Model::load(default_model_dir()))
      , _encoder(*_model, "encoder") {
    }

    // hook == nullptr runs the decoder without the test hook. "memory_lengths" is not
    // set: the graph-eligible gate requires its absence, and the sources all have the
    // same length so the mask carries no information anyway.
    DecoderRun run(const DecoderConfig& cfg, CoreReplayHook* hook, dim_t reserve) {
      PiecewiseTestDecoder decoder(*_model, "decoder");
      decoder.set_cache_reserve_steps(reserve);
      if (cfg.prefix >= 0)
        decoder.set_prefix_reorder(cfg.prefix > 0);
      decoder.set_hook(hook);
      if (hook) {
        hook->workspace = &decoder.workspace();
        if (hook->stale_period == 0)
          hook->stale_period = cfg.beam > 1 ? 2 : 1;  // The beam reorder ping-pong.
      }

      std::vector<int32_t> src;
      for (dim_t b = 0; b < cfg.batch; ++b)
        for (int32_t v : {3, 4, 5, 6})
          src.push_back(v + int32_t(b));
      const StorageView src_ids({cfg.batch, 4}, src);
      const StorageView src_lengths({cfg.batch}, std::vector<int32_t>(cfg.batch, 4));
      StorageView memory(_encoder.output_type());
      _encoder({src_ids}, &src_lengths, memory);
      auto state = decoder.initial_state();
      state.emplace("memory", std::move(memory));
      if (cfg.beam > 1)
        decoder.layers::Decoder::replicate_state(state, cfg.beam);

      DecoderRun out;
      out.num_layers = decoder.num_layers();
      const dim_t rows = cfg.batch * cfg.beam;
      for (dim_t step = 0; step < cfg.steps; ++step) {
        const auto cache_it = state.find("self_keys_0");
        const dim_t capacity = (cache_it != state.end() && cache_it->second.rank() == 4)
          ? cache_it->second.dim(2)
          : 0;
        out.expected_hooked.push_back(hook && step + 1 <= capacity);

        std::vector<int32_t> ids(rows);
        for (dim_t r = 0; r < rows; ++r)
          ids[r] = int32_t(1 + (step * 7 + r * 3) % 6);
        const size_t ends_before = hook ? hook->total_ends : 0;
        if (hook)
          hook->start_step(step);
        StorageView logits;
        decoder(step, StorageView({rows}, ids), state, &logits);
        out.logits.emplace_back(host_bytes(logits));

        const auto& ws = decoder.workspace();
        out.flags_clean.push_back(ws.graph_hook == nullptr);
        out.flags_off.push_back(!ws.exact_core && ws.graph_hook == nullptr
                                && ws.piecewise_cores == 0 && !ws.graph_indirect);
        std::vector<uintptr_t> slots;
        for (const StorageView* slot : {&ws.fused_proj, &ws.queries_proj, &ws.keys_proj,
                                        &ws.values_proj, &ws.attn, &ws.cross_context,
                                        &ws.ffn_inner, &ws.ffn_linear, &ws.layer_in,
                                        &ws.layer_out, &ws.head_transpose})
          slots.push_back(reinterpret_cast<uintptr_t>(slot->buffer()));
        std::sort(slots.begin(), slots.end());
        out.slot_sets.emplace_back(std::move(slots));

        if (hook) {
          const bool hooked = hook->total_ends > ends_before;
          out.hooked.push_back(hooked);
          out.cores.emplace_back(hook->cores);
          bool error = hook->alternation_error;
          if (hooked && hook->fired_step != step) {
            error = error || hook->ends != size_t(out.num_layers)
                    || hook->begins != size_t(out.num_layers);
            for (size_t i = 0; i < hook->cores.size(); ++i)
              error = error || hook->cores[i].index != dim_t(i)
                      || hook->cores[i].length != step + 1;
          }
          out.structure_errors.push_back(error);
        } else {
          out.hooked.push_back(false);
          out.cores.emplace_back();
          out.structure_errors.push_back(false);
        }
        out.record_steps.push_back(state.at("self_length").dim(1));

        if (cfg.beam > 1) {
          // Beam indices with repeats and cross-beam moves, different at every step.
          std::vector<int32_t> indices(rows);
          for (dim_t b = 0; b < cfg.batch; ++b)
            for (dim_t k = 0; k < cfg.beam; ++k)
              indices[b * cfg.beam + k] =
                int32_t(b * cfg.beam + (step % 3 == 0 ? 0 : (k + step) % cfg.beam));
          decoder.update_state(state, StorageView({rows}, indices), cfg.beam);
        }
        std::map<std::string, Bytes> caches;
        for (const auto& [name, value] : state)
          if (is_self_cache(name))
            caches.emplace(name, host_bytes(value));
        out.caches.emplace_back(std::move(caches));
      }

      const auto& ws = decoder.workspace();
      out.slot_reserved = {{"fused_proj", ws.fused_proj.reserved_memory()},
                           {"queries_proj", ws.queries_proj.reserved_memory()},
                           {"attn", ws.attn.reserved_memory()},
                           {"head_transpose", ws.head_transpose.reserved_memory()}};
      if (hook) {
        out.replay_failures = hook->replay_failures;
        out.replay_mismatches = hook->replay_mismatches;
        out.stale_replays = hook->stale_replays;
        out.stale_recaptures = hook->stale_recaptures;
        out.stale_failures = hook->stale_failures;
        out.stale_mismatches = hook->stale_mismatches;
      }
      return out;
    }

    static void expect_same_logits(const DecoderRun& a, const DecoderRun& b, const char* what) {
      ASSERT_EQ(a.logits.size(), b.logits.size());
      for (size_t s = 0; s < a.logits.size(); ++s)
        ASSERT_EQ(a.logits[s], b.logits[s]) << what << ": logits differ at step " << s;
    }

    static void expect_same_caches(const DecoderRun& a, const DecoderRun& b, const char* what) {
      ASSERT_EQ(a.caches.size(), b.caches.size());
      for (size_t s = 0; s < a.caches.size(); ++s) {
        ASSERT_EQ(a.caches[s].size(), b.caches[s].size()) << what << " at step " << s;
        for (const auto& [name, bytes] : a.caches[s])
          ASSERT_EQ(bytes, b.caches[s].at(name)) << what << ": " << name
                                                 << " differs at step " << s;
      }
    }

    static void expect_hook_protocol(const DecoderRun& run) {
      EXPECT_EQ(run.replay_failures, 0u);
      EXPECT_EQ(run.replay_mismatches, 0u);
      EXPECT_EQ(run.stale_failures, 0u);
      EXPECT_EQ(run.stale_mismatches, 0u);
      for (size_t s = 0; s < run.hooked.size(); ++s) {
        EXPECT_EQ(run.hooked[s], run.expected_hooked[s]) << "hook use at step " << s;
        EXPECT_FALSE(run.structure_errors[s]) << "hook protocol at step " << s;
        EXPECT_TRUE(run.flags_clean[s]) << "graph_hook left installed after step " << s;
      }
    }

    std::shared_ptr<const models::Model> _model;
    layers::TransformerEncoder _encoder;
  };

}

// T1a: the exact-length core reported between segments, replayed from its descriptor,
// is bit-identical to the live core, and the whole path is bit-identical to the stock
// default at every step, including past a reserve that the cache outgrows (C = 128).
TEST(PiecewiseCoreBitExact, LayerCoreReplay) {
  for (const dim_t batch : {dim_t(1), dim_t(5)}) {
    const LayerRun reference = run_layers(batch, 0, 448, LayerMode::Default);
    for (const dim_t reserve : {dim_t(128), dim_t(448)}) {
      SCOPED_TRACE("B=" + std::to_string(batch) + " C=" + std::to_string(reserve));
      const LayerRun run = run_layers(batch, reserve, 448, LayerMode::Piecewise);
      EXPECT_FALSE(run.fallback);
      EXPECT_EQ(run.replay_failures, 0u);
      EXPECT_EQ(run.replay_mismatches, 0u);
      // Every hooked step from 3 on also replays the descriptor captured earlier for the
      // same buffers and capacity, at the step's own longer length.
      EXPECT_GT(run.stale_replays, 0u);
      EXPECT_EQ(run.stale_failures, 0u);
      EXPECT_EQ(run.stale_mismatches, 0u);
      if (reserve == 448)
        EXPECT_EQ(run.stale_replays, size_t(2 * (448 - 3)));  // No re-capture at all.
      size_t diff_steps = 0;
      for (size_t step = 0; step < run.outputs.size(); ++step) {
        if (run.outputs[step] != reference.outputs[step]) {
          if (diff_steps == 0)
            ADD_FAILURE() << "output differs from the default at step " << step;
          ++diff_steps;
        }
        const auto& cores = run.cores[step];
        ASSERT_EQ(cores.size(), 2u) << "step " << step;
        EXPECT_FALSE(run.alternation_errors[step]) << "step " << step;
        EXPECT_EQ(run.piecewise_cores[step], 2) << "step " << step;
        for (size_t i = 0; i < cores.size(); ++i) {
          EXPECT_EQ(cores[i].index, dim_t(i)) << "step " << step;
          EXPECT_EQ(cores[i].length, dim_t(step) + 1) << "step " << step;
          EXPECT_EQ(cores[i].batch, batch);
          EXPECT_EQ(cores[i].heads, kLayerHeads);
          EXPECT_EQ(cores[i].depth, kLayerDepth);
          EXPECT_EQ(cores[i].capacity, round_up_block(std::max(reserve, dim_t(step) + 1)));
        }
      }
      EXPECT_EQ(diff_steps, 0u);
    }
  }
}

// T2a: with the slots reserved, the core never re-allocates: {fused_proj, attn} hold
// exactly two buffers for the whole 448-step decode (the swap ping-pong), and every
// core reads and writes the same buffers at every step -- what a replay of segments
// captured once bakes.
TEST(SlotPinning, ReservedSlotsKeepTwoBuffers) {
  const dim_t batch = 5;
  const dim_t reserve = 448;
  const LayerRun run = run_layers(batch, reserve, 448, LayerMode::Piecewise);
  const SlotStats stats = slot_stats(run);
  EXPECT_EQ(stats.distinct, 2u);
  EXPECT_EQ(stats.new_after_warmup, 0u);
  EXPECT_EQ(run.stale_recaptures, 0u);

  const std::set<const void*> pair = {run.slots.back().first, run.slots.back().second};
  ASSERT_EQ(run.cores.front().size(), 2u);
  for (size_t step = 0; step < run.cores.size(); ++step) {
    ASSERT_EQ(run.cores[step].size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
      const auto& core = run.cores[step][i];
      const auto& first = run.cores.front()[i];
      EXPECT_EQ(core.scores, first.scores) << "core " << i << " step " << step;
      EXPECT_EQ(core.context, first.context) << "core " << i << " step " << step;
      EXPECT_EQ(core.queries, first.queries) << "core " << i << " step " << step;
      EXPECT_EQ(core.keys, first.keys) << "core " << i << " step " << step;
      EXPECT_EQ(core.values, first.values) << "core " << i << " step " << step;
      EXPECT_NE(core.scores, core.context);
      EXPECT_EQ(pair.count(core.scores), 1u);
      EXPECT_EQ(pair.count(core.context), 1u);
      EXPECT_GE(core.scores_bytes, batch * kLayerHeads * reserve * dim_t(sizeof(float)));
    }
    // The second layer's core runs on the buffers the first one swapped.
    EXPECT_EQ(run.cores[step][1].scores, run.cores[step][0].context);
    EXPECT_EQ(run.cores[step][1].context, run.cores[step][0].scores);
  }
}

// T2b: negative control for T2a. Same exact-length path without the reservation: the
// growing QK^T product re-allocates the slots mid-decode (past the QKV size, i.e. after
// L > 192 at H = 12), which is what a captured segment must never see. If the
// reservation in the decoder or in reserve_core_slots were lost, T2a would fail the
// same way.
TEST(SlotPinning, UnreservedSlotsReallocate) {
  for (const dim_t batch : {dim_t(1), dim_t(5)}) {
    SCOPED_TRACE("B=" + std::to_string(batch));
    const LayerRun run = run_layers(batch, 448, 448, LayerMode::ExactNoPin);
    const LayerRun reference = run_layers(batch, 0, 448, LayerMode::Default);
    const SlotStats stats = slot_stats(run);
    EXPECT_GT(stats.new_after_warmup, 0u);
    EXPECT_GT(stats.first_new_step, 191);
    EXPECT_FALSE(run.fallback);
    // Without a hook the exact-length path is the stock core.
    for (size_t step = 0; step < run.outputs.size(); ++step)
      ASSERT_EQ(run.outputs[step], reference.outputs[step]) << "step " << step;
  }
}

// T1b: the decoder's piecewise step (test hook) matches the default decoder bit for bit
// at every step, greedy and beam (with and without the prefix reorder), and its caches
// match a decoder preallocated with the same reserve byte for byte, spare tail included.
// With C = 64 the steps that outgrow the capacity (64 and 96) run eager without the
// hook, like the CUDA host guard makes them.
TEST_F(PiecewiseDecoderTest, PiecewiseCoreBitExact) {
  std::vector<DecoderConfig> configs;
  for (const dim_t reserve : {dim_t(128), dim_t(64)}) {
    DecoderConfig greedy;
    greedy.reserve = reserve;
    configs.push_back(greedy);
    for (const int prefix : {0, 1}) {
      DecoderConfig beam;
      beam.reserve = reserve;
      beam.batch = 2;
      beam.beam = 3;
      beam.prefix = prefix;
      configs.push_back(beam);
    }
  }

  for (const auto& cfg : configs) {
    SCOPED_TRACE("C=" + std::to_string(cfg.reserve) + " batch=" + std::to_string(cfg.batch)
                 + " beam=" + std::to_string(cfg.beam) + " prefix=" + std::to_string(cfg.prefix));
    const DecoderRun reference = run(cfg, nullptr, 0);
    const DecoderRun prealloc = run(cfg, nullptr, cfg.reserve);
    CoreReplayHook hook;
    const DecoderRun piecewise = run(cfg, &hook, cfg.reserve);

    expect_same_logits(prealloc, reference, "prealloc vs default");
    expect_same_logits(piecewise, reference, "piecewise vs default");
    expect_same_caches(piecewise, prealloc, "piecewise vs prealloc cache");
    expect_hook_protocol(piecewise);
    // The CPU beam reorder without the prefix bound gathers into new cache buffers at
    // every step, so only the pinned decodes (greedy, prefix on) replay a stale capture.
    if (cfg.beam == 1 || cfg.prefix > 0)
      EXPECT_GT(piecewise.stale_replays, 0u);

    // Step 0 starts from an empty cache (capacity 0), which no host guard admits.
    std::vector<dim_t> eager_steps;
    for (size_t s = 0; s < piecewise.hooked.size(); ++s)
      if (!piecewise.hooked[s])
        eager_steps.push_back(dim_t(s));
    const std::vector<dim_t> expected = cfg.reserve == 64 ? std::vector<dim_t>{0, 64, 96}
                                                          : std::vector<dim_t>{0};
    EXPECT_EQ(eager_steps, expected);
    for (size_t s = 0; s < piecewise.hooked.size(); ++s)
      if (piecewise.hooked[s])
        EXPECT_EQ(piecewise.cores[s].size(), size_t(piecewise.num_layers)) << "step " << s;
  }
}

// T2c: decoder-level pinning. From step 2 on, the set of workspace slot buffers (the
// one the CUDA fingerprint compares in piecewise mode) never changes, and each core's
// buffers repeat with the period of the decode: 1 for greedy, at most 2 with the beam
// reorder ping-pong (prefix reorder on: the CPU host path then uses the same shadow
// swap as CUDA). The scores slot holds B*H*C values.
TEST_F(PiecewiseDecoderTest, SlotPinning) {
  for (const dim_t beam : {dim_t(1), dim_t(3)}) {
    DecoderConfig cfg;
    cfg.reserve = 128;
    cfg.steps = 100;
    cfg.batch = beam > 1 ? 2 : 1;
    cfg.beam = beam;
    cfg.prefix = 1;
    SCOPED_TRACE("beam=" + std::to_string(beam));
    const size_t period = beam > 1 ? 2 : 1;
    CoreReplayHook hook;
    const DecoderRun piecewise = run(cfg, &hook, cfg.reserve);
    expect_hook_protocol(piecewise);
    // One capture per (core, parity), replayed at every later length.
    EXPECT_EQ(piecewise.stale_recaptures, 0u);
    EXPECT_EQ(piecewise.stale_replays,
              size_t(piecewise.num_layers) * (size_t(cfg.steps) - 2 - period));

    for (size_t s = 2; s < piecewise.slot_sets.size(); ++s)
      ASSERT_EQ(piecewise.slot_sets[s], piecewise.slot_sets[2]) << "slot set at step " << s;

    for (size_t s = 2 + period; s < piecewise.cores.size(); ++s) {
      const auto& now = piecewise.cores[s];
      const auto& before = piecewise.cores[s - period];
      ASSERT_EQ(now.size(), size_t(piecewise.num_layers)) << "step " << s;
      ASSERT_EQ(before.size(), now.size()) << "step " << s;
      for (size_t i = 0; i < now.size(); ++i) {
        EXPECT_EQ(now[i].queries, before[i].queries) << "core " << i << " step " << s;
        EXPECT_EQ(now[i].scores, before[i].scores) << "core " << i << " step " << s;
        EXPECT_EQ(now[i].context, before[i].context) << "core " << i << " step " << s;
        EXPECT_EQ(now[i].keys, before[i].keys) << "core " << i << " step " << s;
        EXPECT_EQ(now[i].values, before[i].values) << "core " << i << " step " << s;
        const dim_t item = item_size_of(now[i].dtype);
        EXPECT_GE(now[i].scores_bytes, now[i].batch * now[i].heads * now[i].capacity * item);
        EXPECT_EQ(now[i].capacity, cfg.reserve);
      }
    }
  }
}

// T3: a piecewise step aborted at any core boundary (the capture failing at segment k)
// and rerun eagerly ends exactly like the uninterrupted piecewise run and the default:
// same logits at every step, same caches byte for byte, same self_length record. The
// following steps take the hook again. Step 0 never reaches the hook (empty cache),
// so the injection points start at step 1.
TEST_F(PiecewiseDecoderTest, PiecewiseRerunIdempotent) {
  DecoderConfig cfg;
  cfg.reserve = 128;
  cfg.steps = 40;
  cfg.batch = 2;
  cfg.beam = 3;
  cfg.prefix = 1;
  const DecoderRun reference = run(cfg, nullptr, 0);
  CoreReplayHook clean_hook;
  const DecoderRun clean = run(cfg, &clean_hook, cfg.reserve);
  expect_same_logits(clean, reference, "piecewise vs default");
  expect_hook_protocol(clean);
  EXPECT_FALSE(clean.hooked[0]);
  const dim_t layers = clean.num_layers;
  ASSERT_GE(layers, 3);

  for (const dim_t s : {dim_t(1), dim_t(5), dim_t(37)}) {
    for (const dim_t k : {dim_t(0), layers / 2, layers - 1}) {
      for (const auto where : {CoreReplayHook::Where::End, CoreReplayHook::Where::Begin}) {
        SCOPED_TRACE("s=" + std::to_string(s) + " k=" + std::to_string(k)
                     + (where == CoreReplayHook::Where::End ? " end" : " begin"));
        CoreReplayHook hook;
        hook.throw_step = s;
        hook.throw_core = k;
        hook.throw_where = where;
        const DecoderRun aborted = run(cfg, &hook, cfg.reserve);
        EXPECT_TRUE(hook.fired);
        expect_same_logits(aborted, clean, "aborted vs uninterrupted");
        expect_same_logits(aborted, reference, "aborted vs default");
        expect_same_caches(aborted, clean, "aborted vs uninterrupted cache");
        EXPECT_EQ(aborted.record_steps[s], s + 1);
        EXPECT_EQ(aborted.replay_failures, 0u);
        EXPECT_EQ(aborted.replay_mismatches, 0u);
        EXPECT_EQ(aborted.stale_failures, 0u);
        EXPECT_EQ(aborted.stale_mismatches, 0u);
        for (size_t t = 0; t < aborted.hooked.size(); ++t) {
          EXPECT_TRUE(aborted.flags_clean[t]) << "step " << t;
          EXPECT_EQ(aborted.hooked[t], clean.hooked[t]) << "step " << t;
          if (dim_t(t) != s)
            EXPECT_FALSE(aborted.structure_errors[t]) << "step " << t;
        }
        // The abort happened at core k: cores 0..k had been reported (End or Begin).
        EXPECT_EQ(aborted.cores[s].size(), size_t(k + 1));
      }
    }
  }

  // A protocol violation caught by the decoder's structural guard aborts the step the
  // same way: a segment closed twice, and a segment opened twice.
  for (const auto reentry : {CoreReplayHook::Reentry::EndTwice,
                             CoreReplayHook::Reentry::BeginTwice}) {
    SCOPED_TRACE(reentry == CoreReplayHook::Reentry::EndTwice ? "end twice" : "begin twice");
    CoreReplayHook hook;
    hook.throw_step = 7;
    hook.throw_core = 2;
    hook.throw_where = reentry == CoreReplayHook::Reentry::EndTwice
      ? CoreReplayHook::Where::End
      : CoreReplayHook::Where::Begin;
    hook.reentry = reentry;
    const DecoderRun aborted = run(cfg, &hook, cfg.reserve);
    EXPECT_TRUE(hook.fired);
    EXPECT_TRUE(hook.guard_threw);
    expect_same_logits(aborted, clean, "guard abort vs uninterrupted");
    expect_same_caches(aborted, clean, "guard abort vs uninterrupted cache");
    EXPECT_EQ(aborted.record_steps[7], 8);
  }
}

// T4c: without the test hook (and on CPU, where the CUDA mode can never apply) nothing
// of the piecewise mode runs: the workspace flags stay off after every step, the core
// slots are not reserved, and a preallocated decode stays bit-identical to the default.
TEST_F(PiecewiseDecoderTest, PiecewiseDefaultUntouched) {
  for (const dim_t beam : {dim_t(1), dim_t(3)}) {
    DecoderConfig cfg;
    cfg.reserve = 448;
    cfg.steps = 12;
    cfg.batch = beam > 1 ? 2 : 1;
    cfg.beam = beam;
    SCOPED_TRACE("beam=" + std::to_string(beam));
    const DecoderRun reference = run(cfg, nullptr, 0);
    const DecoderRun prealloc = run(cfg, nullptr, cfg.reserve);
    CoreReplayHook hook;
    const DecoderRun piecewise = run(cfg, &hook, cfg.reserve);

    expect_same_logits(prealloc, reference, "prealloc vs default");
    expect_same_logits(piecewise, reference, "piecewise vs default");
    for (const DecoderRun* r : {&reference, &prealloc}) {
      for (size_t s = 0; s < r->flags_off.size(); ++s)
        EXPECT_TRUE(r->flags_off[s]) << "piecewise state set without the hook at step " << s;
    }

    // The hooked decode reserves each core slot for the full capacity; the hook-less
    // ones keep the stock sizes, far below it after 12 steps.
    ASSERT_FALSE(piecewise.cores.back().empty());
    const auto& core = piecewise.cores.back().front();
    const dim_t pinned = core.batch * core.heads * round_up_block(cfg.reserve)
                         * item_size_of(core.dtype);
    for (const auto& [name, bytes] : piecewise.slot_reserved)
      EXPECT_GE(bytes, pinned) << name;
    for (const DecoderRun* r : {&reference, &prealloc}) {
      EXPECT_LT(r->slot_reserved.at("attn"), pinned);
      EXPECT_LT(r->slot_reserved.at("head_transpose"), pinned);
    }
  }
}

// T4d: the env gate itself, which no CPU decode can reach (the CUDA modes need a CUDA
// device and the env is read once per process). Without CT2_CUDA_GRAPHS_PIECEWISE the
// existing modes are untouched: CT2_CUDA_GRAPHS=1 or CT2_CUDA_PAD_KV=1 still selects the
// padded path, and the piecewise sub-mode needs both switches. The CUDA graph runner's
// piecewise_enabled() reads the same decode_kv_env().
TEST(PiecewiseGate, SelectDecodeKvMode) {
  using layers::DecodeKvMode;
  struct Row {
    bool graphs;
    bool piecewise;
    bool pad_kv;
    DecodeKvMode cuda;  // Eligible step on CUDA.
  };
  const std::vector<Row> rows = {
    {false, false, false, DecodeKvMode::Stock},
    {false, false, true, DecodeKvMode::Padded},
    {true, false, false, DecodeKvMode::Padded},
    {true, false, true, DecodeKvMode::Padded},
    {false, true, false, DecodeKvMode::Stock},   // PIECEWISE alone does nothing.
    {false, true, true, DecodeKvMode::Padded},
    {true, true, false, DecodeKvMode::ExactCore},
    {true, true, true, DecodeKvMode::ExactCore},  // PAD_KV is ignored.
  };
  for (const Row& row : rows) {
    layers::DecodeKvEnv env;
    env.graphs = row.graphs;
    env.piecewise = row.piecewise;
    env.pad_kv = row.pad_kv;
    SCOPED_TRACE("graphs=" + std::to_string(row.graphs) + " piecewise="
                 + std::to_string(row.piecewise) + " pad_kv=" + std::to_string(row.pad_kv));
    EXPECT_EQ(env.piecewise_graphs(), row.graphs && row.piecewise);
    EXPECT_EQ(layers::select_decode_kv_mode(env, Device::CUDA, true), row.cuda);
    EXPECT_EQ(layers::select_decode_kv_mode(env, Device::CUDA, false), DecodeKvMode::Stock);
    EXPECT_EQ(layers::select_decode_kv_mode(env, Device::CPU, true), DecodeKvMode::Stock);
    EXPECT_EQ(layers::select_decode_kv_mode(env, Device::CPU, false), DecodeKvMode::Stock);
  }
}
