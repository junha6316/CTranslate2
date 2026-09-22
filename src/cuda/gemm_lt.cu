#include "./gemm_lt.h"

#include <cstdint>
#include <type_traits>

#include <cublasLt.h>
#include <cuda_fp16.h>

#include "./utils.h"

namespace ctranslate2 {
  namespace cuda {

    // Enough for the split-K and stream-K algorithms the heuristic picks for the
    // small-m decoder GEMMs. Allocated once per thread: a per-call allocation would
    // bring back a cudaMallocAsync/cudaFreeAsync pair for every GEMM.
    constexpr size_t lt_workspace_size = 4 * 1024 * 1024;

    class CublasLtHandle {
    public:
      CublasLtHandle() {
        CUDA_CHECK(cudaGetDevice(&_device));
        CUBLAS_CHECK(cublasLtCreate(&_handle));
        CUDA_CHECK(cudaMalloc(&_workspace, lt_workspace_size));
      }
      ~CublasLtHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cudaFree(_workspace);
        cublasLtDestroy(_handle);
      }
      cublasLtHandle_t get() const {
        return _handle;
      }
      void* workspace() const {
        return _workspace;
      }
    private:
      int _device;
      cublasLtHandle_t _handle;
      void* _workspace = nullptr;
    };

    static const CublasLtHandle& get_cublaslt_handle() {
      static thread_local CublasLtHandle handle;
      return handle;
    }

    // The heuristic assumes 256-byte aligned pointers unless told otherwise, and
    // tensors here can be views at arbitrary element offsets.
    static uint32_t alignment_of(const void* ptr) {
      const auto address = reinterpret_cast<uintptr_t>(ptr);
      uint32_t alignment = 256;
      while (alignment > 1 && address % alignment != 0)
        alignment /= 2;
      return alignment;
    }

    // Destroys the cuBLASLt descriptors on every exit path.
    struct LtDescriptors {
      cublasLtMatmulDesc_t op = nullptr;
      cublasLtMatrixLayout_t a = nullptr;
      cublasLtMatrixLayout_t b = nullptr;
      cublasLtMatrixLayout_t c = nullptr;
      cublasLtMatmulPreference_t preference = nullptr;

      ~LtDescriptors() {
        if (preference)
          cublasLtMatmulPreferenceDestroy(preference);
        if (c)
          cublasLtMatrixLayoutDestroy(c);
        if (b)
          cublasLtMatrixLayoutDestroy(b);
        if (a)
          cublasLtMatrixLayoutDestroy(a);
        if (op)
          cublasLtMatmulDescDestroy(op);
      }
    };

    template <typename T>
    struct LtTypes;

    template<>
    struct LtTypes<float> {
      static constexpr cudaDataType_t data = CUDA_R_32F;
    };

    template<>
    struct LtTypes<float16_t> {
      static constexpr cudaDataType_t data = CUDA_R_16F;
    };

    template<>
    struct LtTypes<bfloat16_t> {
      static constexpr cudaDataType_t data = CUDA_R_16BF;
    };

    template <typename T>
    bool gemm_bias_lt(bool transpose_a, bool transpose_b,
                      dim_t m, dim_t n, dim_t k,
                      const T* a, dim_t lda,
                      const T* b, dim_t ldb,
                      const T* bias,
                      const T* residual,
                      T* c, dim_t ldc) {
      const cudaDataType_t data_type = LtTypes<T>::data;

      // Match the compute types of primitives<Device::CUDA>::gemm.
      cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
      cudaDataType_t scale_type = CUDA_R_32F;
      const float alpha = 1;
      const float beta = residual ? 1 : 0;
      const __half alpha_h = alpha;
      const __half beta_h = beta;
      const void* alpha_ptr = &alpha;
      const void* beta_ptr = &beta;
      if (std::is_same<T, float16_t>::value && use_true_fp16_gemm()) {
        compute_type = CUBLAS_COMPUTE_16F;
        scale_type = CUDA_R_16F;
        alpha_ptr = &alpha_h;
        beta_ptr = &beta_h;
      }

      // cuBLAS assumes column-major storage, so swap a and b accordingly:
      // the column-major result is c^T[n, m] = op(b)[n, k] * op(a)[k, m].
      const cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
      const cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

      LtDescriptors desc;
      CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc.op, compute_type, scale_type));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &op_b, sizeof (op_b)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &op_a, sizeof (op_a)));

      // The bias has one value per row of the column-major result, i.e. per output
      // feature, which is the Dense bias.
      const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof (epilogue)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                  &bias, sizeof (bias)));

      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.a, data_type,
                                              transpose_b ? k : n, transpose_b ? n : k, ldb));
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.b, data_type,
                                              transpose_a ? m : k, transpose_a ? k : m, lda));
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.c, data_type, n, m, ldc));

      CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&desc.preference));
      const size_t workspace_size = lt_workspace_size;
      CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                     desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_size, sizeof (workspace_size)));

      // The residual is read as the C operand (beta = 1) and the result written to D.
      // C and D may differ with cuBLASLt, which cublasGemmEx does not allow.
      const void* c_in = residual ? static_cast<const void*>(residual) : c;
      const uint32_t align_a = alignment_of(b);
      const uint32_t align_b = alignment_of(a);
      const uint32_t align_c = alignment_of(c_in);
      const uint32_t align_d = alignment_of(c);
      CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                     desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                     &align_a, sizeof (align_a)));
      CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                     desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                     &align_b, sizeof (align_b)));
      CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                     desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                     &align_c, sizeof (align_c)));
      CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                     desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                     &align_d, sizeof (align_d)));

      const CublasLtHandle& handle = get_cublaslt_handle();

      cublasLtMatmulHeuristicResult_t heuristic;
      int num_results = 0;
      const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle.get(),
                                                                   desc.op,
                                                                   desc.a,
                                                                   desc.b,
                                                                   desc.c,
                                                                   desc.c,
                                                                   desc.preference,
                                                                   1,
                                                                   &heuristic,
                                                                   &num_results);
      if (status == CUBLAS_STATUS_NOT_SUPPORTED || num_results == 0)
        return false;
      CUBLAS_CHECK(status);

      CUBLAS_CHECK(cublasLtMatmul(handle.get(),
                                  desc.op,
                                  alpha_ptr,
                                  b, desc.a,
                                  a, desc.b,
                                  beta_ptr,
                                  c_in, desc.c,
                                  c, desc.c,
                                  &heuristic.algo,
                                  handle.workspace(),
                                  lt_workspace_size,
                                  get_cuda_stream()));
      return true;
    }

#define DECLARE_IMPL(T)                                                 \
    template bool gemm_bias_lt<T>(bool, bool, dim_t, dim_t, dim_t,      \
                                  const T*, dim_t,                      \
                                  const T*, dim_t,                      \
                                  const T*, const T*,                   \
                                  T*, dim_t);

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
