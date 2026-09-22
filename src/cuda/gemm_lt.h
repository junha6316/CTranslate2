#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cuda {

    // c = op(a) * op(b) + bias (+ residual), with the bias and the residual applied in the
    // cuBLASLt epilogue instead of a separate kernel. Same row-major conventions as
    // primitives<Device::CUDA>::gemm. residual may be nullptr; otherwise it has the
    // shape of c.
    //
    // Returns false without touching c when cuBLASLt has no algorithm for the problem,
    // so the caller can fall back to the regular GEMM followed by BiasAdd.
    template <typename T>
    bool gemm_bias_lt(bool transpose_a, bool transpose_b,
                      dim_t m, dim_t n, dim_t k,
                      const T* a, dim_t lda,
                      const T* b, dim_t ldb,
                      const T* bias,
                      const T* residual,
                      T* c, dim_t ldc);

  }
}
