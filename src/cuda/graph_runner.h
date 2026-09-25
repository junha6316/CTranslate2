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
// never fatal. There is no mid-decode recapture; a new decode (reported by the
// decoder through note_new_decode(), or detected by a step discontinuity) restarts
// the warmup.

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
      // injection for the fallback tests. Empty string when unset.
      static const std::string& injected_fault();

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
      // discontinuities.
      Action begin_step(dim_t step,
                        std::vector<std::uintptr_t> fingerprint,
                        std::size_t num_critical);

      // Called when a step is not graph-eligible (e.g. attention weights requested):
      // keeps the step counter coherent and forces eager for the rest of the decode.
      void note_ineligible(dim_t step);

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
