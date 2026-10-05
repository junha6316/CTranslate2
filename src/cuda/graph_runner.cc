#include "cuda/graph_runner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "cuda/alloc_stats.h"
#include "cuda/utils.h"
#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    // Serializes stream captures across replicas (and against any cuBLAS handle
    // creation that would run on another replica thread mid-capture).
    static std::mutex g_capture_mutex;

    namespace {

      enum class Phase {
        Warmup,
        CaptureSecond,  // Period-2 pair: the odd-parity capture must be the next step.
        Ready,
        Reentry,        // After a tier transition: capture at the next step.
        ReentrySecond,  // Period-2 re-entry: the second capture of the pair is next.
        Disabled,
      };

      const char* phase_name(Phase phase) {
        switch (phase) {
        case Phase::Warmup: return "Warmup";
        case Phase::CaptureSecond: return "CaptureSecond";
        case Phase::Ready: return "Ready";
        case Phase::Reentry: return "Reentry";
        case Phase::ReentrySecond: return "ReentrySecond";
        case Phase::Disabled: return "Disabled";
        }
        return "?";
      }

      // Destroys every executable of the list. An executable still in flight on the
      // stream completes normally: the driver defers its release.
      void destroy_execs(std::vector<cudaGraphExec_t>& execs) {
        for (cudaGraphExec_t exec : execs) {
          if (exec)
            cudaGraphExecDestroy(exec);
        }
        execs.clear();
      }

      struct GraphExec {
        cudaGraphExec_t exec = nullptr;            // Whole-step mode.
        std::vector<cudaGraphExec_t> segments;     // Piecewise: cores.size() + 1.
        std::vector<layers::CoreDesc> cores;       // Piecewise.
        std::vector<std::uintptr_t> fingerprint;

        bool ready() const {
          return exec || !segments.empty();
        }

        void destroy() {
          if (exec) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
          }
          destroy_execs(segments);
          cores.clear();
          fingerprint.clear();
        }
      };

      // Fault injection for the piecewise-only faults: true once per process, so the
      // next decode of the same process demonstrates the recovery. One flag is enough:
      // CT2_CUDA_GRAPHS_FAULT names a single fault.
      bool consume_segment_fault(const char* name) {
        static std::atomic<bool> fired(false);
        if (DecoderGraphRunner::injected_fault() != name)
          return false;
        bool expected = false;
        return fired.compare_exchange_strong(expected, true);
      }

    }

    struct DecoderGraphRunner::Impl {
      Phase phase = Phase::Warmup;
      dim_t expected_step = -1;
      dim_t steps_in_decode = 0;
      dim_t period = 1;
      std::vector<std::vector<std::uintptr_t>> history;  // last pre-step fingerprints
      GraphExec execs[2];  // period 1: slot 0 only; period 2: indexed by step parity.
      std::size_t num_critical = 0;

      // Live capture state.
      bool capturing = false;
      std::unique_lock<std::mutex> capture_lock;
      std::uint64_t violations_at_capture = 0;  // Piecewise: at the running segment's start.
      std::vector<std::uintptr_t> capture_fingerprint;
      dim_t current_parity = 0;
      dim_t current_step = -1;
      dim_t current_batch = 0;

      // Live piecewise capture state, from begin_capture() to the commit in
      // end_capture_and_launch() or abort_capture(). The capture lock is held for the
      // whole step, eager cores included; "capturing" is true only inside a segment.
      bool piecewise_active = false;
      std::vector<cudaGraphExec_t> pending_segments;  // Instantiated (and launched).
      std::vector<layers::CoreDesc> pending_cores;
      dim_t piecewise_segments = 0;  // Segments per step of the last commit (summary).

      // Per-decode tier state and counters (cleared by reset_decode).
      bool captured_this_decode = false;
      dim_t captured_batch = 0;
      dim_t decode_steps = 0;  // Unlike steps_in_decode, not reset by a transition.
      dim_t tier_transitions = 0;
      dim_t captures = 0;
      dim_t replays = 0;
      std::vector<dim_t> segment_replays = {0};  // Replays before/after each transition.
      std::string tier_log;                      // "from->to@step" per transition.
      std::string disable_reason;

      dim_t slot_index(dim_t parity) const {
        return period == 1 ? 0 : parity;
      }

      static bool critical_match(const std::vector<std::uintptr_t>& a,
                                 const std::vector<std::uintptr_t>& b,
                                 std::size_t n) {
        if (a.size() < n || b.size() < n)
          return false;
        return std::equal(a.begin(), a.begin() + n, b.begin());
      }

      int device_checked = -1;  // -1 unknown, 0 refused, 1 ok

      void log_summary() const {
        if (decode_steps == 0)
          return;  // Nothing ran through the runner since the last boundary.
        if (!spdlog::should_log(spdlog::level::debug))
          return;
        std::string segments;
        for (size_t i = 0; i < segment_replays.size(); ++i) {
          if (i > 0)
            segments += ",";
          segments += std::to_string(segment_replays[i]);
        }
        const std::string piecewise =
          DecoderGraphRunner::piecewise_enabled()
          ? ", piecewise=1, segments=" + std::to_string(piecewise_segments)
          : std::string();
        spdlog::debug("CUDA graphs: decode summary (steps={}, captures={}, replays={},"
                      " transitions={}, segment_replays=[{}], tiers=[{}], phase={},"
                      " disabled={}{})",
                      decode_steps, captures, replays, tier_transitions, segments,
                      tier_log, phase_name(phase),
                      disable_reason.empty() ? std::string("none") : disable_reason,
                      piecewise);
      }

      void reset_decode() {
        log_summary();
        phase = Phase::Warmup;
        steps_in_decode = 0;
        history.clear();
        execs[0].destroy();
        execs[1].destroy();
        captured_this_decode = false;
        decode_steps = 0;
        current_batch = 0;
        captured_batch = 0;
        tier_transitions = 0;
        captures = 0;
        replays = 0;
        segment_replays.assign(1, 0);
        tier_log.clear();
        disable_reason.clear();
        piecewise_segments = 0;
      }

      void disable(const char* reason) {
        if (phase != Phase::Disabled) {
          spdlog::debug("CUDA graphs: falling back to eager for this decode ({})", reason);
          disable_reason = reason;
        }
        phase = Phase::Disabled;
      }

      void release_capture_lock() {
        if (capture_lock.owns_lock()) {
          capture_lock.unlock();
          capture_lock.release();
        }
      }

      // Ends the piecewise step's state: destroys the segments instantiated so far (the
      // launched ones complete normally, see destroy_execs) and forgets the cores.
      void drop_pending() {
        destroy_execs(pending_segments);
        pending_cores.clear();
        piecewise_active = false;
      }

      [[noreturn]] void fail_segment(const char* reason) {
        disable(reason);
        throw std::runtime_error(std::string("CUDA graphs: ") + reason);
      }

      // Ends the running segment's capture, instantiates it and launches it (the
      // recorded kernels did not run yet, and the eager core that follows reads their
      // outputs). Returns nullptr on success, with the executable appended to
      // pending_segments, or the failure reason (the executable, if any, is still in
      // pending_segments for drop_pending()).
      const char* close_segment(DecoderGraphRunner& runner) {
        set_capture_active(false);
        capturing = false;

        cudaGraph_t graph = nullptr;
        const cudaError_t end_status = cudaStreamEndCapture(get_cuda_stream(), &graph);
        if (end_status != cudaSuccess || !graph) {
          spdlog::debug("CUDA graphs: cudaStreamEndCapture failed: {}",
                        cudaGetErrorString(end_status));
          cudaGetLastError();  // Clear any capture-invalidation error, as abort_capture does.
          if (graph)
            cudaGraphDestroy(graph);
          return "end capture failed";
        }

        // Zero-allocation assertion over this segment, with the semantics described in
        // end_capture_and_launch(). The eager cores between the segments are not
        // recorded, so their allocator activity does not count; a core that moved a
        // buffer a segment bakes is caught by the decoder's pointer checks instead.
        if (capture_violation_count() != violations_at_capture) {
          cudaGraphDestroy(graph);
          return "allocator activity inside the capture";
        }

        // The whole-step faults keep their meaning on segment 0.
        const std::size_t index = pending_segments.size();
        const bool instantiate_fault =
          index == 0
          ? (DecoderGraphRunner::injected_fault() == "instantiate"
             || runner.consume_tier_fault("tier_instantiate"))
          : (index == 1 && consume_segment_fault("segment_instantiate"));
        cudaGraphExec_t exec = nullptr;
        const cudaError_t inst_status =
          instantiate_fault
          ? cudaErrorUnknown
          : cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        if (inst_status != cudaSuccess) {
          spdlog::debug("CUDA graphs: cudaGraphInstantiate failed for segment {}: {}",
                        index, cudaGetErrorString(inst_status));
          return "instantiate failed";
        }
        pending_segments.push_back(exec);

        const cudaError_t launch_status = cudaGraphLaunch(exec, get_cuda_stream());
        if (launch_status != cudaSuccess) {
          spdlog::debug("CUDA graphs: post-capture launch failed for segment {}: {}",
                        index, cudaGetErrorString(launch_status));
          return "post-capture launch failed";
        }
        return nullptr;
      }

      // SegmentHook::end_segment: closes the running segment before core.index runs.
      void end_segment(DecoderGraphRunner& runner, const layers::CoreDesc& core) {
        if (!piecewise_active || !capturing)
          fail_segment("segment boundary outside a piecewise capture");
        if (core.index != dim_t(pending_cores.size()))
          fail_segment("self-attention core reported out of order");
        const char* reason = close_segment(runner);
        if (reason)
          fail_segment(reason);
        pending_cores.push_back(core);
      }

      // SegmentHook::begin_segment: opens the next segment after the eager core.
      void begin_segment() {
        if (!piecewise_active || capturing)
          fail_segment("segment boundary outside a piecewise capture");
        if (pending_segments.size() == 1 && consume_segment_fault("segment_capture"))
          fail_segment("injected segment capture failure");
        violations_at_capture = capture_violation_count();
        const cudaError_t status =
          cudaStreamBeginCapture(get_cuda_stream(), cudaStreamCaptureModeRelaxed);
        if (status != cudaSuccess) {
          spdlog::debug("CUDA graphs: cudaStreamBeginCapture failed for segment {}: {}",
                        pending_segments.size(), cudaGetErrorString(status));
          fail_segment("begin segment capture failed");
        }
        capturing = true;
        set_capture_active(true);
      }

      // The SegmentHook handed to the decoder (see segment_hook()).
      class Hook : public layers::SegmentHook {
      public:
        Hook(Impl& impl, DecoderGraphRunner& runner)
          : _state(impl)
          , _runner(runner) {
        }

        void end_segment(const layers::CoreDesc& core) override {
          _state.end_segment(_runner, core);
        }

        void begin_segment() override {
          _state.begin_segment();
        }

      private:
        Impl& _state;
        DecoderGraphRunner& _runner;
      };

      Hook hook;

      explicit Impl(DecoderGraphRunner& runner)
        : hook(*this, runner) {
      }
    };

    DecoderGraphRunner::DecoderGraphRunner()
      : _impl(new Impl(*this)) {
    }

    DecoderGraphRunner::~DecoderGraphRunner() {
      // A piecewise step cut short (never expected: the decoder commits or aborts every
      // capture) still holds pending segments and the capture lock.
      if (_impl->piecewise_active)
        abort_capture("graph runner destroyed mid-capture");
      // Logs the last decode's summary and destroys both executables.
      _impl->reset_decode();
    }

    bool DecoderGraphRunner::env_enabled() {
      static const bool enabled = read_bool_from_env("CT2_CUDA_GRAPHS");
      return enabled;
    }

    bool DecoderGraphRunner::check_enabled() {
      static const bool enabled = read_bool_from_env("CT2_CUDA_GRAPHS_CHECK");
      return enabled;
    }

    const std::string& DecoderGraphRunner::injected_fault() {
      static const std::string fault = read_string_from_env("CT2_CUDA_GRAPHS_FAULT");
      return fault;
    }

    bool DecoderGraphRunner::piecewise_enabled() {
      // The same switches the decoder selects its KV mode from (layers::select_decode_kv_mode).
      return layers::decode_kv_env().piecewise_graphs();
    }

    layers::SegmentHook& DecoderGraphRunner::segment_hook() {
      return _impl->hook;
    }

    bool DecoderGraphRunner::warmup_reentry() {
      static const bool warmup = [] {
        const std::string mode = read_string_from_env("CT2_CUDA_GRAPHS_TIERS_REENTRY");
        if (!mode.empty() && mode != "fast" && mode != "warmup")
          spdlog::warn("CT2_CUDA_GRAPHS_TIERS_REENTRY: unknown value '{}', using 'fast'",
                       mode);
        return mode == "warmup";
      }();
      return warmup;
    }

    bool DecoderGraphRunner::device_supported() {
      if (_impl->device_checked < 0) {
        int ok = 0;
        // The main thread is assigned the legacy default stream (see utils.cc), which
        // cudaStreamBeginCapture rejects.
        if (get_cuda_stream() != cudaStreamDefault) {
          int device = -1;
          int major = 0;
          if (cudaGetDevice(&device) == cudaSuccess
              && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)
                 == cudaSuccess
              && major >= 7)
            ok = 1;
        }
        _impl->device_checked = ok;
        if (!ok)
          spdlog::debug("CUDA graphs: device/stream preconditions not met, staying eager");
      }
      return _impl->device_checked == 1;
    }

    void DecoderGraphRunner::note_ineligible(dim_t step) {
      _impl->expected_step = step + 1;
      ++_impl->steps_in_decode;
      ++_impl->decode_steps;
      // An eager excursion invalidates a captured pair's parity alignment guarantees;
      // fingerprints would catch a real mismatch, but keep the MVP conservative.
      if (_impl->phase != Phase::Warmup)
        _impl->disable("ineligible step mid-decode");
    }

    void DecoderGraphRunner::note_new_decode() {
      // Both CUDA allocators deterministically recycle freed blocks, so a new decode
      // can get the previous decode's buffers back at the same addresses: the old
      // executables must never survive an explicit boundary.
      _impl->reset_decode();
      _impl->expected_step = -1;
    }

    bool DecoderGraphRunner::begin_tier_transition(dim_t step,
                                                   dim_t batch,
                                                   dim_t from,
                                                   dim_t to) {
      Impl& impl = *_impl;
      if (impl.phase == Phase::Disabled
          || impl.capturing
          || impl.piecewise_active
          || (impl.captured_this_decode && batch != impl.captured_batch))
        return false;

      // The eager crossing step that follows frees the buffers both executables bake
      // (the old caches, the self_length record, the capacity-dependent workspace), so
      // they are destroyed first, here on the host. Replays still in flight on the
      // stream complete normally: the driver defers the release of a launched
      // executable. Both slots, so a half-captured period-2 pair is covered too.
      impl.execs[0].destroy();
      impl.execs[1].destroy();
      impl.history.clear();
      // The refused step counts as this decode's step, as note_ineligible counts it,
      // and the next step must not be mistaken for a new decode.
      impl.steps_in_decode = 1;
      ++impl.decode_steps;
      impl.expected_step = step + 1;
      ++impl.tier_transitions;
      impl.segment_replays.push_back(0);
      if (!impl.tier_log.empty())
        impl.tier_log += ",";
      impl.tier_log += std::to_string(from) + "->" + std::to_string(to) + "@"
                       + std::to_string(step);

      const Phase from_phase = impl.phase;
      // Fast re-entry reuses this decode's detected period and captures at the next
      // step: the pre-step fingerprint of the crossing step is pre-growth, so a fresh
      // warmup would only delay the capture. Warmup (no capture yet, or the debug
      // CT2_CUDA_GRAPHS_TIERS_REENTRY=warmup) restarts the 12-step budget and
      // re-detects the period.
      const bool fast = impl.captured_this_decode && !warmup_reentry();
      impl.phase = fast ? Phase::Reentry : Phase::Warmup;
      spdlog::debug("CUDA graphs: tier {} -> {} at step {} (from {}, {} re-entry)",
                    from, to, step, phase_name(from_phase), fast ? "fast" : "warmup");
      return true;
    }

    bool DecoderGraphRunner::consume_tier_fault(const char* name) {
      // Once per process (not per decode), so the next decode of the same process
      // demonstrates the recovery: Disabled is only sticky until the decode boundary.
      static std::atomic<bool> fired(false);
      if (_impl->tier_transitions == 0 || injected_fault() != name)
        return false;
      bool expected = false;
      return fired.compare_exchange_strong(expected, true);
    }

    DecoderGraphRunner::Action
    DecoderGraphRunner::begin_step(dim_t step,
                                   std::vector<std::uintptr_t> fingerprint,
                                   std::size_t num_critical,
                                   dim_t batch) {
      Impl& impl = *_impl;
      impl.num_critical = num_critical;

      if (step != impl.expected_step) {
        // New decode (or a caller we lost track of): restart the warmup. The previous
        // executables are dropped: the decoder state tensors are recreated per decode,
        // so their addresses rarely survive and the capture cost amortizes within one
        // decode.
        impl.reset_decode();
      }
      impl.expected_step = step + 1;
      ++impl.steps_in_decode;
      ++impl.decode_steps;
      impl.current_parity = step & 1;
      impl.current_step = step;
      impl.current_batch = batch;

      switch (impl.phase) {
      case Phase::Disabled:
        return Action::Eager;

      case Phase::Ready: {
        static bool fingerprint_fault_pending = injected_fault() == "fingerprint";
        const GraphExec& g = impl.execs[impl.slot_index(impl.current_parity)];
        if (!g.ready()
            || fingerprint_fault_pending
            || !Impl::critical_match(g.fingerprint, fingerprint, impl.num_critical)) {
          fingerprint_fault_pending = false;
          impl.disable("pointer fingerprint changed");
          return Action::Eager;
        }
        return Action::Replay;
      }

      case Phase::CaptureSecond: {
        // The second capture of a period-2 pair must be the immediately following
        // step and must match the fingerprint observed two steps ago.
        if (impl.history.size() >= 2
            && fingerprint == impl.history[impl.history.size() - 2]) {
          impl.capture_fingerprint = fingerprint;
          impl.history.push_back(std::move(fingerprint));
          return Action::Capture;
        }
        impl.disable("period-2 fingerprint did not hold for the pair capture");
        return Action::Eager;
      }

      case Phase::Reentry:
      case Phase::ReentrySecond:
        // Re-capture after a tier transition. No repeat check: the period is known
        // from this decode's first capture, and the crossing step's fingerprint was
        // taken before the growth, so it could not match anyway. The critical check
        // before every replay remains the backstop.
        impl.capture_fingerprint = fingerprint;
        impl.history.push_back(std::move(fingerprint));
        if (impl.history.size() > 6)
          impl.history.erase(impl.history.begin());
        return Action::Capture;

      case Phase::Warmup:
      default:
        break;
      }

      // Warmup: detect a stable period from the pre-step fingerprints.
      impl.history.push_back(fingerprint);
      if (impl.history.size() > 6)
        impl.history.erase(impl.history.begin());

      const auto& h = impl.history;
      const size_t n = h.size();
      bool stable_p1 = false;
      bool stable_p2 = false;
      if (impl.steps_in_decode >= 4 && n >= 3) {
        stable_p1 = (h[n - 1] == h[n - 2] && h[n - 2] == h[n - 3]);
        if (!stable_p1 && n >= 4)
          stable_p2 = (h[n - 1] == h[n - 3] && h[n - 2] == h[n - 4]
                       && h[n - 1] != h[n - 2]);
      }

      if (stable_p1 || stable_p2) {
        impl.period = stable_p1 ? 1 : 2;
        impl.capture_fingerprint = h[n - 1];
        return Action::Capture;
      }

      if (impl.steps_in_decode > 12)
        impl.disable("pointer fingerprint never stabilized");

      return Action::Eager;
    }

    bool DecoderGraphRunner::begin_capture() {
      Impl& impl = *_impl;
      if (injected_fault() == "capture" || consume_tier_fault("tier_capture")) {
        disable_for_decode("injected capture failure");
        return false;
      }
      impl.capture_lock = std::unique_lock<std::mutex>(g_capture_mutex);
      impl.violations_at_capture = capture_violation_count();
      const cudaError_t status =
        cudaStreamBeginCapture(get_cuda_stream(), cudaStreamCaptureModeRelaxed);
      if (status != cudaSuccess) {
        spdlog::debug("CUDA graphs: cudaStreamBeginCapture failed: {}",
                      cudaGetErrorString(status));
        impl.capture_lock.unlock();
        impl.capture_lock.release();
        disable_for_decode("begin capture failed");
        return false;
      }
      impl.capturing = true;
      set_capture_active(true);
      if (piecewise_enabled()) {
        // Segment 0 is open; the segment hook closes and opens the following ones, and
        // the capture lock stays held until the commit or the abort.
        destroy_execs(impl.pending_segments);
        impl.pending_cores.clear();
        impl.piecewise_active = true;
      }
      return true;
    }

    void DecoderGraphRunner::abort_capture(const char* reason) {
      Impl& impl = *_impl;
      if (impl.piecewise_active) {
        // Also reached between two segments, when no stream capture is open: an eager
        // core, a pointer check or a segment hook call failed.
        if (impl.capturing) {
          set_capture_active(false);
          cudaGraph_t graph = nullptr;
          cudaStreamEndCapture(get_cuda_stream(), &graph);
          cudaGetLastError();  // Clear any capture-invalidation error.
          if (graph)
            cudaGraphDestroy(graph);
          impl.capturing = false;
        }
        impl.drop_pending();
        impl.release_capture_lock();
        disable_for_decode(reason);
        return;
      }
      if (!impl.capturing)
        return;
      set_capture_active(false);
      cudaGraph_t graph = nullptr;
      cudaStreamEndCapture(get_cuda_stream(), &graph);
      cudaGetLastError();  // Clear any capture-invalidation error.
      if (graph)
        cudaGraphDestroy(graph);
      impl.capturing = false;
      if (impl.capture_lock.owns_lock()) {
        impl.capture_lock.unlock();
        impl.capture_lock.release();
      }
      disable_for_decode(reason);
    }

    bool DecoderGraphRunner::end_capture_and_launch() {
      Impl& impl = *_impl;
      if (impl.piecewise_active)
        return end_piecewise_capture();
      if (!impl.capturing)
        return false;
      set_capture_active(false);
      impl.capturing = false;

      cudaGraph_t graph = nullptr;
      const cudaError_t end_status = cudaStreamEndCapture(get_cuda_stream(), &graph);
      if (impl.capture_lock.owns_lock()) {
        impl.capture_lock.unlock();
        impl.capture_lock.release();
      }

      if (end_status != cudaSuccess || !graph) {
        spdlog::debug("CUDA graphs: cudaStreamEndCapture failed: {}",
                      cudaGetErrorString(end_status));
        if (graph)
          cudaGraphDestroy(graph);
        disable_for_decode("end capture failed");
        return false;
      }

      // Zero-allocation assertion over the captured region, thread-scoped: an
      // allocation on the capturing thread throws through the allocator hook
      // (aborting the capture before this point), and a free on the capturing thread
      // is deferred and recorded as a violation, so only the violation delta is
      // checked here (captures are serialized by g_capture_mutex and the hooks are
      // keyed on the thread-local t_capture_active, so the delta is ours). Allocator
      // activity on OTHER replica threads is deliberately ignored: it targets other
      // streams and cannot invalidate a relaxed-mode capture, while a process-global
      // allocation-count check would spuriously discard nearly every capture under
      // concurrent decodes.
      if (capture_violation_count() != impl.violations_at_capture) {
        cudaGraphDestroy(graph);
        disable_for_decode("allocator activity inside the capture");
        return false;
      }

      const dim_t slot_id = impl.slot_index(impl.current_parity);
      GraphExec& slot = impl.execs[slot_id];
      slot.destroy();
      const bool instantiate_fault =
        injected_fault() == "instantiate" || consume_tier_fault("tier_instantiate");
      const cudaError_t inst_status =
        instantiate_fault
        ? cudaErrorUnknown
        : cudaGraphInstantiate(&slot.exec, graph, nullptr, nullptr, 0);
      cudaGraphDestroy(graph);
      if (inst_status != cudaSuccess) {
        spdlog::debug("CUDA graphs: cudaGraphInstantiate failed: {}",
                      cudaGetErrorString(inst_status));
        slot.exec = nullptr;
        disable_for_decode("instantiate failed");
        return false;
      }
      slot.fingerprint = impl.capture_fingerprint;

      // The captured kernels were recorded, not executed: run them now for this step.
      const cudaError_t launch_status = cudaGraphLaunch(slot.exec, get_cuda_stream());
      if (launch_status != cudaSuccess) {
        spdlog::debug("CUDA graphs: post-capture launch failed: {}",
                      cudaGetErrorString(launch_status));
        slot.destroy();
        disable_for_decode("post-capture launch failed");
        return false;
      }

      impl.captured_this_decode = true;
      impl.captured_batch = impl.current_batch;
      ++impl.captures;
      spdlog::debug("CUDA graphs: captured step {} into slot {} ({})",
                    impl.current_step, slot_id,
                    impl.tier_transitions > 0 ? "re-entry" : "initial");

      if (impl.period == 2 && impl.phase == Phase::Warmup)
        impl.phase = Phase::CaptureSecond;
      else if (impl.period == 2 && impl.phase == Phase::Reentry)
        impl.phase = Phase::ReentrySecond;
      else
        impl.phase = Phase::Ready;
      return true;
    }

    bool DecoderGraphRunner::end_piecewise_capture() {
      Impl& impl = *_impl;
      // Closes the last segment (output norm and projection), then commits the step's
      // segments and cores to the parity's slot. Segments 0..N-1 and every core already
      // ran on the device; a failure here leaves the eager rerun to overwrite them.
      const char* reason =
        impl.capturing ? impl.close_segment(*this) : "no open segment at the end of the step";
      impl.release_capture_lock();
      if (reason) {
        impl.drop_pending();
        disable_for_decode(reason);
        return false;
      }

      const dim_t slot_id = impl.slot_index(impl.current_parity);
      GraphExec& slot = impl.execs[slot_id];
      slot.destroy();
      slot.segments = std::move(impl.pending_segments);
      slot.cores = std::move(impl.pending_cores);
      slot.fingerprint = impl.capture_fingerprint;
      impl.pending_segments.clear();
      impl.pending_cores.clear();
      impl.piecewise_active = false;
      impl.piecewise_segments = dim_t(slot.segments.size());

      impl.captured_this_decode = true;
      impl.captured_batch = impl.current_batch;
      ++impl.captures;
      spdlog::debug("CUDA graphs: captured step {} into slot {} ({}, piecewise, {} segments)",
                    impl.current_step, slot_id,
                    impl.tier_transitions > 0 ? "re-entry" : "initial",
                    slot.segments.size());
      if (spdlog::should_log(spdlog::level::debug)) {
        // The buffers every replay of this slot re-issues the cores on: they must not
        // move for the rest of the decode (see layers::CoreDesc).
        for (const layers::CoreDesc& core : slot.cores)
          spdlog::debug("CUDA graphs: slot {} core {} (batch={}, heads={}, capacity={},"
                        " depth={}, length={}): queries={:#x} keys={:#x} values={:#x}"
                        " scores={:#x} context={:#x}",
                        slot_id, core.index, core.batch, core.heads, core.capacity,
                        core.depth, core.length,
                        reinterpret_cast<std::uintptr_t>(core.queries),
                        reinterpret_cast<std::uintptr_t>(core.keys),
                        reinterpret_cast<std::uintptr_t>(core.values),
                        reinterpret_cast<std::uintptr_t>(core.scores),
                        reinterpret_cast<std::uintptr_t>(core.context));
      }

      if (impl.period == 2 && impl.phase == Phase::Warmup)
        impl.phase = Phase::CaptureSecond;
      else if (impl.period == 2 && impl.phase == Phase::Reentry)
        impl.phase = Phase::ReentrySecond;
      else
        impl.phase = Phase::Ready;
      return true;
    }

    bool DecoderGraphRunner::replay_piecewise(
      const std::function<bool(const layers::CoreDesc&)>& run_core) {
      Impl& impl = *_impl;
      // The whole-step replay faults keep their meaning: nothing is launched.
      static bool replay_fault_pending = injected_fault() == "replay";
      if (replay_fault_pending || consume_tier_fault("tier_replay")) {
        replay_fault_pending = false;
        disable_for_decode("injected replay failure");
        return false;
      }
      const GraphExec& g = impl.execs[impl.slot_index(impl.current_parity)];
      // Defense in depth, as in replay(): begin_step only returns Replay for a ready slot.
      if (g.segments.empty()
          || g.segments.size() != g.cores.size() + 1
          || g.fingerprint.empty()) {
        disable_for_decode("no piecewise executable for this parity");
        return false;
      }
      for (std::size_t k = 0; k < g.segments.size(); ++k) {
        if (k == 1 && consume_segment_fault("segment_replay")) {
          disable_for_decode("injected segment replay failure");
          return false;
        }
        const cudaError_t status = cudaGraphLaunch(g.segments[k], get_cuda_stream());
        if (status != cudaSuccess) {
          spdlog::debug("CUDA graphs: cudaGraphLaunch failed for segment {}: {}",
                        k, cudaGetErrorString(status));
          disable_for_decode("replay launch failed");
          return false;
        }
        if (k == g.cores.size())
          break;  // The last segment has no core after it.
        // The core runs eagerly on the same stream, after segment k and before k + 1.
        bool core_ok = false;
        try {
          core_ok = run_core(g.cores[k]);
        } catch (const std::exception& e) {
          spdlog::debug("CUDA graphs: piecewise core {} replay failed: {}", k, e.what());
        }
        if (!core_ok) {
          disable_for_decode("piecewise core replay failed");
          return false;
        }
      }
      ++impl.replays;
      ++impl.segment_replays.back();
      return true;
    }

    bool DecoderGraphRunner::replay() {
      Impl& impl = *_impl;
      if (piecewise_enabled()) {
        // The decoder calls replay_piecewise() in this mode: the slots hold segments.
        disable_for_decode("whole-step replay in piecewise mode");
        return false;
      }
      static bool replay_fault_pending = injected_fault() == "replay";
      if (replay_fault_pending || consume_tier_fault("tier_replay")) {
        replay_fault_pending = false;
        disable_for_decode("injected replay failure");
        return false;
      }
      GraphExec& g = impl.execs[impl.slot_index(impl.current_parity)];
      // Defense in depth: begin_step only returns Replay for a slot that holds an
      // executable and its fingerprint, and a tier transition clears both slots.
      if (!g.exec || g.fingerprint.empty()) {
        disable_for_decode("no executable for this parity");
        return false;
      }
      const cudaError_t status = cudaGraphLaunch(g.exec, get_cuda_stream());
      if (status != cudaSuccess) {
        spdlog::debug("CUDA graphs: cudaGraphLaunch failed: {}", cudaGetErrorString(status));
        disable_for_decode("replay launch failed");
        return false;
      }
      ++impl.replays;
      ++impl.segment_replays.back();
      return true;
    }

    void DecoderGraphRunner::disable_for_decode(const char* reason) {
      _impl->disable(reason);
    }

  }
}
