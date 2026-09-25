#include "cuda/graph_runner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>

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
        Disabled,
      };

      struct GraphExec {
        cudaGraphExec_t exec = nullptr;
        std::vector<std::uintptr_t> fingerprint;

        void destroy() {
          if (exec) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
          }
          fingerprint.clear();
        }
      };

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
      std::uint64_t violations_at_capture = 0;
      std::vector<std::uintptr_t> capture_fingerprint;
      dim_t current_parity = 0;

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

      void reset_decode() {
        phase = Phase::Warmup;
        steps_in_decode = 0;
        history.clear();
        execs[0].destroy();
        execs[1].destroy();
      }

      void disable(const char* reason) {
        if (phase != Phase::Disabled)
          spdlog::debug("CUDA graphs: falling back to eager for this decode ({})", reason);
        phase = Phase::Disabled;
      }
    };

    DecoderGraphRunner::DecoderGraphRunner()
      : _impl(new Impl()) {
    }

    DecoderGraphRunner::~DecoderGraphRunner() {
      _impl->execs[0].destroy();
      _impl->execs[1].destroy();
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

    DecoderGraphRunner::Action
    DecoderGraphRunner::begin_step(dim_t step,
                                   std::vector<std::uintptr_t> fingerprint,
                                   std::size_t num_critical) {
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
      impl.current_parity = step & 1;

      switch (impl.phase) {
      case Phase::Disabled:
        return Action::Eager;

      case Phase::Ready: {
        static bool fingerprint_fault_pending = injected_fault() == "fingerprint";
        const GraphExec& g = impl.execs[impl.slot_index(impl.current_parity)];
        if (!g.exec
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
      if (injected_fault() == "capture") {
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
      return true;
    }

    void DecoderGraphRunner::abort_capture(const char* reason) {
      Impl& impl = *_impl;
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

      GraphExec& slot = impl.execs[impl.slot_index(impl.current_parity)];
      slot.destroy();
      const cudaError_t inst_status =
        injected_fault() == "instantiate"
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

      if (impl.period == 2 && impl.phase == Phase::Warmup) {
        impl.phase = Phase::CaptureSecond;
      } else {
        impl.phase = Phase::Ready;
      }
      return true;
    }

    bool DecoderGraphRunner::replay() {
      Impl& impl = *_impl;
      static bool replay_fault_pending = injected_fault() == "replay";
      if (replay_fault_pending) {
        replay_fault_pending = false;
        disable_for_decode("injected replay failure");
        return false;
      }
      GraphExec& g = impl.execs[impl.slot_index(impl.current_parity)];
      if (!g.exec) {
        disable_for_decode("no executable for this parity");
        return false;
      }
      const cudaError_t status = cudaGraphLaunch(g.exec, get_cuda_stream());
      if (status != cudaSuccess) {
        spdlog::debug("CUDA graphs: cudaGraphLaunch failed: {}", cudaGetErrorString(status));
        disable_for_decode("replay launch failed");
        return false;
      }
      return true;
    }

    void DecoderGraphRunner::disable_for_decode(const char* reason) {
      _impl->disable(reason);
    }

  }
}
