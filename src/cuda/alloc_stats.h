#pragma once

#include <cstdint>

namespace ctranslate2 {
  namespace cuda {

    // Debug counters over all CUDA allocator instances and threads. Cheap relaxed
    // atomics, always on; consumers are debug tools (CT2_CUDA_ALLOC_DEBUG) and the
    // graph runner's zero-allocation assertion around a stream capture.
    std::uint64_t allocation_count();
    std::uint64_t free_count();

    // Thread-local flag raised by the CUDA graph runner while a stream capture is
    // active on this thread. An allocation request in that window throws (which
    // aborts the capture cleanly instead of recording a graph memory node), and a
    // free request is deferred until after the capture and counted as a violation.
    void set_capture_active(bool active);
    bool capture_active();

    // Number of free requests that were deferred because they arrived during a
    // capture. A non-zero delta across a capture invalidates the captured graph.
    std::uint64_t capture_violation_count();

  }
}
