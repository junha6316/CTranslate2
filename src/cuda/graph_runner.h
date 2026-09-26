#pragma once

// CUDA graph capture/replay controller for the iterative decoder forward
// (CT2_CUDA_GRAPHS=1). CUDA builds only: transformer.cc provides an empty stub
// class for CPU builds, so this header must only be included under CT2_WITH_CUDA.
//
// Protocol per decode step (driven by TransformerDecoder::decode):
//   1. the decoder collects the step's pointer fingerprint and calls begin_step();
//   2. on Action::Eager it runs the ordinary forward;
//   3. on Action::Capture it wraps the ordinary forward in begin_capture() /
//      end_capture_and_launch() (any failure or allocation inside the region aborts
//      the capture and the step is rerun eagerly);
//   4. on Action::Replay it skips the forward and calls replay().
// Any CUDA error degrades to eager for the remainder of the decode, logged once,
// never fatal. Mid-decode recapture happens only at a capacity-tier transition, which
// the decoder's host guard requests through begin_tier_transition() at the first step
// that outgrows the KV capacity: both executables are destroyed there, before the
// eager crossing step frees the old buffers, and the next steps re-capture at the new
// shapes. Disabled stays sticky until the next decode boundary (reported by the
// decoder through note_new_decode(), or detected by a step discontinuity), which also
// restarts the warmup.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    class DecoderGraphRunner {
    public:
      enum class Action {
        Eager,
        Capture,
        Replay,
      };

      DecoderGraphRunner();
      ~DecoderGraphRunner();

      // CT2_CUDA_GRAPHS=1, read once.
      static bool env_enabled();
      // CT2_CUDA_GRAPHS_CHECK=1: replay AND eager each step, compare logits.
      static bool check_enabled();
      // CT2_CUDA_GRAPHS_FAULT=capture|instantiate|replay|fingerprint|alloc: fault
      // injection for the fallback tests. The tier_capture|tier_instantiate|
      // tier_replay|tier_alloc values fire once, at the first capture / instantiate /
      // replay / in-capture allocation after a tier transition (see
      // consume_tier_fault). Empty string when unset.
      static const std::string& injected_fault();
      // CT2_CUDA_GRAPHS_TIERS_REENTRY=warmup (debug/A-B only): a tier transition
      // re-enters the full warmup instead of re-capturing at the next step.
      static bool warmup_reentry();

      // One-time device preconditions for the calling thread: a non-default stream
      // (the main thread owns the legacy stream, which cudaStreamBeginCapture
      // rejects) and compute capability >= 7.0.
      bool device_supported();

      // Called at the start of every graph-eligible step. The fingerprint is the
      // ordered list of device pointers (and shape scalars) that a captured graph
      // would bake. The first num_critical entries are the buffers that eager code
      // still rotates between steps while replays are active (KV caches, ids staging,
      // logits, self_lengths, step_state, shape scalars) and are re-checked before
      // every replay; the remaining entries (workspace slots) rotate only inside the
      // eager forward, freeze once replays start, and therefore only participate in
      // the warmup stability detection. Detects decode boundaries through step
      // discontinuities. batch is the step's batch x beam size (ids.dim(0)).
      Action begin_step(dim_t step,
                        std::vector<std::uintptr_t> fingerprint,
                        std::size_t num_critical,
                        dim_t batch);

      // Called when a step is not graph-eligible (e.g. attention weights requested):
      // keeps the step counter coherent and forces eager for the rest of the decode.
      void note_ineligible(dim_t step);

      // Called by the decoder's host guard at a step whose KV capacity is exhausted
      // (capacity == step) when the tier policy names a larger capacity. Returns false
      // (the caller then takes the note_ineligible path) when the decode is Disabled, a
      // capture is active, or the batch changed since this decode's first capture.
      // Otherwise destroys BOTH executables -- the step about to run eager frees the
      // buffers they bake -- and arms a re-capture: from the next step (fast re-entry,
      // once this decode has captured) or through a fresh warmup. The crossing step
      // itself must run eager; it counts as this decode's step.
      bool begin_tier_transition(dim_t step, dim_t batch, dim_t from, dim_t to);

      // Fault injection for the tier faults: true once per process, when
      // injected_fault() == name and this decode has made a tier transition.
      bool consume_tier_fault(const char* name);

      // Called at a decode boundary (a prompt/prefix forward, a new cache reserve, a
      // step-0 call): drops the executables and restarts the warmup. The step
      // discontinuity heuristic in begin_step alone is not a decode boundary: a new
      // decode whose first eligible step equals the previous decode's expected step
      // (whisper prompt lengths vary between 1 and 224+) would otherwise go straight
      // to the fingerprint check against recycled allocations.
      void note_new_decode();

      bool begin_capture();
      // Ends the capture, instantiates the executable and launches it (the captured
      // kernels did not run during recording). Returns false when anything failed;
      // the caller must then rerun the step eagerly (abort_capture was already done).
      bool end_capture_and_launch();
      // Aborts an active capture (allocation inside the region, exception, ...).
      void abort_capture(const char* reason);
      // Launches the executable graph for the current step parity.
      bool replay();

      void disable_for_decode(const char* reason);

    private:
      struct Impl;
      const std::unique_ptr<Impl> _impl;
    };

  }
}
