#include "nemotron/dense_gemm_runner.h"

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "storage_conversion.h"

namespace nemotron {
namespace {

bool IsSupportedDenseStorage(const std::string& storage_dtype) {
  return storage_dtype == "fp32" ||
         storage_dtype == "float32" ||
         storage_dtype == "float" ||
         storage_dtype == "bf16" ||
         storage_dtype == "bfloat16";
}

bool IsSupportedDenseCompute(const std::string& compute_dtype) {
  return compute_dtype == "fp32" ||
         compute_dtype == "float32" ||
         compute_dtype == "float" ||
         compute_dtype == "bf16" ||
         compute_dtype == "bfloat16";
}

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool CheckCublas(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS;
}

bool DenseGemmDebugEnabled() {
  static const bool kEnabled = []() {
    const char* dense = std::getenv("NEMOTRON_DEBUG_DENSE_GEMM_RUNNER");
    if (dense != nullptr && !(dense[0] == '0' && dense[1] == '\0')) {
      return true;
    }
    const char* fp8 = std::getenv("NEMOTRON_DEBUG_SCALED_FP8_NATIVE");
    return fp8 != nullptr && !(fp8[0] == '0' && fp8[1] == '\0');
  }();
  return kEnabled;
}

bool Fp8FastAccumSupported() {
  const char* disable_fast_accum = std::getenv("NEMOTRON_DISABLE_FP8_FAST_ACCUM");
  if (disable_fast_accum != nullptr && !(disable_fast_accum[0] == '0' && disable_fast_accum[1] == '\0')) {
    return false;
  }
  static const bool kSupported = []() {
    int device = 0;
    cudaDeviceProp prop{};
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
           ((prop.major == 8 && prop.minor == 9) || prop.major == 9);
  }();
  return kSupported;
}

const char* CublasStatusName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN";
}

void LogDenseGemmCublasFailure(
    const char* step,
    cublasStatus_t status,
    cudaDataType_t activations_type,
    cudaDataType_t weights_type,
    float alpha_scale,
    bool fast_accum,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const CublasLtGemmPlan& plan) {
  if (!DenseGemmDebugEnabled()) {
    return;
  }
  std::cerr << "dense_gemm_runner: " << step
            << " status=" << CublasStatusName(status)
            << " code=" << static_cast<int>(status)
            << " activations_type=" << static_cast<int>(activations_type)
            << " weights_type=" << static_cast<int>(weights_type)
            << " alpha_scale=" << alpha_scale
            << " fast_accum=" << fast_accum
            << " m=" << m
            << " n=" << n
            << " k=" << k
            << " lda=" << plan.lda
            << " ldb=" << plan.ldb
            << " ldc=" << plan.ldc
            << " trans_a=" << static_cast<int>(plan.transform_a)
            << " trans_b=" << static_cast<int>(plan.transform_b)
            << " order_a=" << static_cast<int>(plan.order_a)
            << " order_b=" << static_cast<int>(plan.order_b)
            << " order_c=" << static_cast<int>(plan.order_c)
            << "\n";
}

void LogDenseGemmCudaFailure(const char* step, cudaError_t status) {
  if (!DenseGemmDebugEnabled()) {
    return;
  }
  std::cerr << "dense_gemm_runner: " << step
            << " cuda_status=" << cudaGetErrorString(status)
            << " code=" << static_cast<int>(status)
            << "\n";
}

cudaDataType_t OutputDataType(const DeviceTensorFp32&) {
  return CUDA_R_32F;
}

cudaDataType_t OutputDataType(const DeviceTensorBf16&) {
  return CUDA_R_16BF;
}

cublasOperation_t ToCublasOp(CublasLtTransform transform) {
  switch (transform) {
    case CublasLtTransform::kNone:
      return CUBLAS_OP_N;
    case CublasLtTransform::kTranspose:
      return CUBLAS_OP_T;
  }
  return CUBLAS_OP_N;
}

cublasLtOrder_t ToCublasOrder(CublasLtMatrixOrder order) {
  switch (order) {
    case CublasLtMatrixOrder::kRowMajor:
      return CUBLASLT_ORDER_ROW;
    case CublasLtMatrixOrder::kColumnMajor:
      return CUBLASLT_ORDER_COL;
  }
  return CUBLASLT_ORDER_ROW;
}

struct MatrixLayoutShape {
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  std::int64_t ld = 0;
};

MatrixLayoutShape ActivationLayoutShape(
    const CublasLtGemmPlan& plan,
    std::size_t m,
    std::size_t k) {
  switch (plan.contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.lda)};
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(k),
          static_cast<std::uint64_t>(m),
          static_cast<std::int64_t>(plan.lda)};
  }
  return {};
}

MatrixLayoutShape WeightLayoutShape(
    const CublasLtGemmPlan& plan,
    std::size_t n,
    std::size_t k) {
  switch (plan.contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(n),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.ldb)};
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(k),
          static_cast<std::uint64_t>(n),
          static_cast<std::int64_t>(plan.ldb)};
  }
  return {};
}

template <typename OutputTensorT>
MatrixLayoutShape OutputLayoutShape(
    const CublasLtGemmPlan& plan,
    const OutputTensorT& output,
    std::size_t m,
    std::size_t n) {
  (void)output;
  switch (plan.contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(n),
          static_cast<std::int64_t>(plan.ldc)};
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(n),
          static_cast<std::uint64_t>(m),
          static_cast<std::int64_t>(plan.ldc)};
  }
  return {};
}

bool BuildCachedCublasLtMatmulState(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    cudaDataType_t activations_type,
    const void* activations_scale_data,
    cudaDataType_t weights_type,
    const void* weights_scale_data,
    float alpha_scale,
    bool fast_accum,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    cudaDataType_t output_type,
    CachedCublasLtMatmulState* state) {
  if (!handle.valid() || state == nullptr) {
    return false;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  int returned_results = 0;
  bool ok = true;
  const MatrixLayoutShape a_shape = ActivationLayoutShape(plan, m, k);
  const MatrixLayoutShape b_shape = WeightLayoutShape(plan, n, k);
  const MatrixLayoutShape c_shape = OutputLayoutShape(plan, output_type, m, n);

  const auto apply_cublas = [&](const char* step, cublasStatus_t status) {
    if (!CheckCublas(status)) {
      LogDenseGemmCublasFailure(
          step,
          status,
          activations_type,
          weights_type,
          alpha_scale,
          fast_accum,
          m,
          n,
          k,
          plan);
      ok = false;
      return false;
    }
    return true;
  };

  *state = CachedCublasLtMatmulState{};
  state->activations_type = activations_type;
  state->weights_type = weights_type;
  state->output_type = output_type;
  state->activations_scale_data = activations_scale_data;
  state->weights_scale_data = weights_scale_data;
  state->alpha_scale = alpha_scale;
  state->fast_accum = fast_accum;
  state->m = m;
  state->n = n;
  state->k = k;

  apply_cublas(
      "matmul_desc_create",
      cublasLtMatmulDescCreate(&state->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = ToCublasOp(plan.transform_a);
  const cublasOperation_t trans_b = ToCublasOp(plan.transform_b);
  if (ok) {
    apply_cublas(
        "set_transa",
        cublasLtMatmulDescSetAttribute(
            state->op_desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &trans_a,
            sizeof(trans_a)));
  }
  if (ok) {
    apply_cublas(
        "set_transb",
        cublasLtMatmulDescSetAttribute(
            state->op_desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &trans_b,
            sizeof(trans_b)));
  }
  if (activations_scale_data != nullptr) {
    if (ok) {
      apply_cublas(
          "set_a_scale_ptr",
          cublasLtMatmulDescSetAttribute(
              state->op_desc,
              CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
              &activations_scale_data,
              sizeof(activations_scale_data)));
    }
  }
  if (weights_scale_data != nullptr) {
    if (ok) {
      apply_cublas(
          "set_b_scale_ptr",
          cublasLtMatmulDescSetAttribute(
              state->op_desc,
              CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
              &weights_scale_data,
              sizeof(weights_scale_data)));
    }
  }
  if (fast_accum) {
    const int fast_accum_attr = 1;
    if (ok) {
      apply_cublas(
          "set_fast_accum",
          cublasLtMatmulDescSetAttribute(
              state->op_desc,
              CUBLASLT_MATMUL_DESC_FAST_ACCUM,
              &fast_accum_attr,
              sizeof(fast_accum_attr)));
    }
  }

  if (ok) {
    apply_cublas(
        "layout_a_create",
        cublasLtMatrixLayoutCreate(
            &state->a_desc,
            activations_type,
            a_shape.rows,
            a_shape.cols,
            a_shape.ld));
  }
  if (ok) {
    apply_cublas(
        "layout_b_create",
        cublasLtMatrixLayoutCreate(
            &state->b_desc,
            weights_type,
            b_shape.rows,
            b_shape.cols,
            b_shape.ld));
  }
  if (ok) {
    apply_cublas(
        "layout_c_create",
        cublasLtMatrixLayoutCreate(
            &state->c_desc,
            output_type,
            c_shape.rows,
            c_shape.cols,
            c_shape.ld));
  }

  const cublasLtOrder_t order_a = ToCublasOrder(plan.order_a);
  const cublasLtOrder_t order_b = ToCublasOrder(plan.order_b);
  const cublasLtOrder_t order_c = ToCublasOrder(plan.order_c);
  if (ok) {
    apply_cublas(
        "layout_a_set_order",
        cublasLtMatrixLayoutSetAttribute(
            state->a_desc,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order_a,
            sizeof(order_a)));
  }
  if (ok) {
    apply_cublas(
        "layout_b_set_order",
        cublasLtMatrixLayoutSetAttribute(
            state->b_desc,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order_b,
            sizeof(order_b)));
  }
  if (ok) {
    apply_cublas(
        "layout_c_set_order",
        cublasLtMatrixLayoutSetAttribute(
            state->c_desc,
            CUBLASLT_MATRIX_LAYOUT_ORDER,
            &order_c,
            sizeof(order_c)));
  }

  if (ok) {
    apply_cublas("preference_create", cublasLtMatmulPreferenceCreate(&preference));
  }
  const std::size_t max_workspace = handle.workspace_bytes();
  if (ok) {
    apply_cublas(
        "preference_set_workspace",
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace,
            sizeof(max_workspace)));
  }

  if (ok) {
    apply_cublas(
        "algo_get_heuristic",
        cublasLtMatmulAlgoGetHeuristic(
            handle.handle(),
            state->op_desc,
            state->a_desc,
            state->b_desc,
            state->c_desc,
            state->c_desc,
            preference,
            1,
            &state->heuristic,
            &returned_results));
    if (ok && returned_results <= 0) {
      if (DenseGemmDebugEnabled()) {
        std::cerr << "dense_gemm_runner: algo_get_heuristic returned no results"
                  << " activations_type=" << static_cast<int>(activations_type)
                  << " weights_type=" << static_cast<int>(weights_type)
                  << " alpha_scale=" << alpha_scale
                  << " fast_accum=" << fast_accum
                  << " m=" << m
                  << " n=" << n
                  << " k=" << k
                  << "\n";
      }
      ok = false;
    }
  }

  if (preference != nullptr) {
    cublasLtMatmulPreferenceDestroy(preference);
  }
  if (!ok) {
    *state = CachedCublasLtMatmulState{};
    return false;
  }

  state->heuristic_count = returned_results;
  return true;
}

template <typename OutputTensorT>
std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorTypedToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const void* activations_data,
    cudaDataType_t activations_type,
    const void* activations_scale_data,
    const void* weights_data,
    cudaDataType_t weights_type,
    const void* weights_scale_data,
    float alpha_scale,
    bool fast_accum,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    OutputTensorT* output,
    const CachedCublasLtMatmulState* cached_matmul_state,
    cudaStream_t stream) {
  if (!handle.valid() ||
      activations_data == nullptr ||
      weights_data == nullptr ||
      output == nullptr ||
      !output->valid() ||
      output->shape().size() != 2 ||
      output->shape()[0] != m ||
      output->shape()[1] != n) {
    return std::nullopt;
  }

  bool ok = true;

  const auto apply_cublas = [&](const char* step, cublasStatus_t status) {
    if (!CheckCublas(status)) {
      LogDenseGemmCublasFailure(
          step,
          status,
          activations_type,
          weights_type,
          alpha_scale,
          fast_accum,
          m,
          n,
          k,
          plan);
      ok = false;
      return false;
    }
    return true;
  };
  const auto apply_cuda = [&](const char* step, cudaError_t status) {
    if (!CheckCuda(status)) {
      LogDenseGemmCudaFailure(step, status);
      ok = false;
      return false;
    }
    return true;
  };

  CachedCublasLtMatmulState local_state;
  const CachedCublasLtMatmulState* matmul_state = cached_matmul_state;
  const cudaDataType_t output_type = OutputDataType(*output);
  if (matmul_state != nullptr) {
    if (matmul_state->op_desc == nullptr ||
        matmul_state->a_desc == nullptr ||
        matmul_state->b_desc == nullptr ||
        matmul_state->c_desc == nullptr ||
        matmul_state->m != m ||
        matmul_state->n != n ||
        matmul_state->k != k ||
        matmul_state->activations_type != activations_type ||
        matmul_state->weights_type != weights_type ||
        matmul_state->output_type != output_type ||
        matmul_state->activations_scale_data != activations_scale_data ||
        matmul_state->weights_scale_data != weights_scale_data ||
        matmul_state->alpha_scale != alpha_scale ||
        matmul_state->fast_accum != fast_accum) {
      return std::nullopt;
    }
  } else if (!BuildCachedCublasLtMatmulState(
                 handle,
                 plan,
                 activations_type,
                 activations_scale_data,
                 weights_type,
                 weights_scale_data,
                 alpha_scale,
                 fast_accum,
                 m,
                 n,
                 k,
                 output_type,
                 &local_state)) {
    return std::nullopt;
  } else {
    matmul_state = &local_state;
  }

  if (ok) {
    const float alpha = alpha_scale;
    const float beta = 0.0f;
    apply_cublas(
        "matmul",
        cublasLtMatmul(
            handle.handle(),
            matmul_state->op_desc,
            &alpha,
            activations_data,
            matmul_state->a_desc,
            weights_data,
            matmul_state->b_desc,
            &beta,
            output->data(),
            matmul_state->c_desc,
            output->data(),
            matmul_state->c_desc,
            &matmul_state->heuristic.algo,
            handle.workspace(),
            handle.workspace_bytes(),
            stream));
    if (ok) {
      apply_cuda("matmul_cuda", cudaGetLastError());
    }
  }

  if (!ok) {
    return std::nullopt;
  }

  return DenseRowMajorDeviceStats{
      m,
      n,
      matmul_state->heuristic.workspaceSize,
      matmul_state->heuristic_count,
  };
}

}  // namespace

void ResetCachedCublasLtMatmulState(CachedCublasLtMatmulState* state) {
  if (state == nullptr) {
    return;
  }
  if (state->c_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(state->c_desc);
  }
  if (state->b_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(state->b_desc);
  }
  if (state->a_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(state->a_desc);
  }
  if (state->op_desc != nullptr) {
    cublasLtMatmulDescDestroy(state->op_desc);
  }
  state->op_desc = nullptr;
  state->a_desc = nullptr;
  state->b_desc = nullptr;
  state->c_desc = nullptr;
  state->heuristic = {};
  state->activations_type = CUDA_R_32F;
  state->weights_type = CUDA_R_32F;
  state->output_type = CUDA_R_32F;
  state->activations_scale_data = nullptr;
  state->weights_scale_data = nullptr;
  state->alpha_scale = 1.0f;
  state->fast_accum = false;
  state->m = 0;
  state->n = 0;
  state->k = 0;
  state->heuristic_count = 0;
}

CachedCublasLtMatmulState::~CachedCublasLtMatmulState() {
  ResetCachedCublasLtMatmulState(this);
}

CachedCublasLtMatmulState::CachedCublasLtMatmulState(
    CachedCublasLtMatmulState&& other) noexcept
    : op_desc(std::exchange(other.op_desc, nullptr)),
      a_desc(std::exchange(other.a_desc, nullptr)),
      b_desc(std::exchange(other.b_desc, nullptr)),
      c_desc(std::exchange(other.c_desc, nullptr)),
      heuristic(other.heuristic),
      activations_type(other.activations_type),
      weights_type(other.weights_type),
      output_type(other.output_type),
      activations_scale_data(other.activations_scale_data),
      weights_scale_data(other.weights_scale_data),
      alpha_scale(other.alpha_scale),
      fast_accum(other.fast_accum),
      m(other.m),
      n(other.n),
      k(other.k),
      heuristic_count(other.heuristic_count) {}

CachedCublasLtMatmulState& CachedCublasLtMatmulState::operator=(
    CachedCublasLtMatmulState&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  ResetCachedCublasLtMatmulState(this);
  op_desc = std::exchange(other.op_desc, nullptr);
  a_desc = std::exchange(other.a_desc, nullptr);
  b_desc = std::exchange(other.b_desc, nullptr);
  c_desc = std::exchange(other.c_desc, nullptr);
  heuristic = other.heuristic;
  activations_type = other.activations_type;
  weights_type = other.weights_type;
  output_type = other.output_type;
  activations_scale_data = other.activations_scale_data;
  weights_scale_data = other.weights_scale_data;
  alpha_scale = other.alpha_scale;
  fast_accum = other.fast_accum;
  m = other.m;
  n = other.n;
  k = other.k;
  heuristic_count = other.heuristic_count;
  return *this;
}

std::unique_ptr<CachedCublasLtMatmulState> CreateDenseRowMajorFp8E4M3MatmulState(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const float* weight_scale_device,
    const float* input_scale_device,
    cudaDataType_t output_type) {
  auto state = std::make_unique<CachedCublasLtMatmulState>();
  if (!BuildCachedCublasLtMatmulState(
          handle,
          plan,
          CUDA_R_8F_E4M3,
          input_scale_device,
          CUDA_R_8F_E4M3,
          weight_scale_device,
          1.0f,
          Fp8FastAccumSupported(),
          plan.execution.launch_plan.m,
          plan.execution.launch_plan.n,
          plan.execution.launch_plan.k,
          output_type,
          state.get())) {
    return nullptr;
  }
  return state;
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      !IsSupportedDenseStorage(plan.execution.launch_plan.storage_dtype) ||
      !IsSupportedDenseCompute(plan.execution.launch_plan.compute_dtype)) {
    return std::nullopt;
  }

  if (activations.shape().size() != 2 || output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.output_rows() != n ||
      weights.input_cols() != k) {
    return std::nullopt;
  }
  return RunDenseRowMajorTypedToDevice(
      handle,
      plan,
      activations.data(),
      CUDA_R_32F,
      nullptr,
      weights.data(),
      CUDA_R_32F,
      nullptr,
      1.0f,
      false,
      m,
      n,
      k,
      output,
      nullptr,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      !IsSupportedDenseStorage(plan.execution.launch_plan.storage_dtype) ||
      !IsSupportedDenseCompute(plan.execution.launch_plan.compute_dtype)) {
    return std::nullopt;
  }

  if (activations.shape().size() != 2 || output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.output_rows() != n ||
      weights.input_cols() != k) {
    return std::nullopt;
  }
  return RunDenseRowMajorTypedToDevice(
      handle,
      plan,
      activations.data(),
      CUDA_R_16BF,
      nullptr,
      weights.data(),
      CUDA_R_32F,
      nullptr,
      1.0f,
      false,
      m,
      n,
      k,
      output,
      nullptr,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      !IsSupportedDenseStorage(plan.execution.launch_plan.storage_dtype) ||
      !IsSupportedDenseCompute(plan.execution.launch_plan.compute_dtype)) {
    return std::nullopt;
  }

  if (activations.shape().size() != 2 || output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.output_rows() != n ||
      weights.input_cols() != k) {
    return std::nullopt;
  }
  return RunDenseRowMajorTypedToDevice(
      handle,
      plan,
      activations.data(),
      CUDA_R_16BF,
      nullptr,
      weights.data(),
      CUDA_R_32F,
      nullptr,
      1.0f,
      false,
      m,
      n,
      k,
      output,
      nullptr,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      !IsSupportedDenseCompute(plan.execution.launch_plan.compute_dtype) ||
      activations.shape().size() != 2 ||
      weights.shape().size() != 2 ||
      output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.shape()[0] != n ||
      weights.shape()[1] != k) {
    return std::nullopt;
  }

  auto converted_activations = DeviceTensorBf16::Create(activations.shape());
  if (!converted_activations ||
      !ConvertDeviceFp32ToBf16(
          activations.data(),
          activations.numel(),
          converted_activations->data(),
          stream)) {
    return std::nullopt;
  }

  return RunDenseRowMajorTypedToDevice(
      handle,
      plan,
      converted_activations->data(),
      CUDA_R_16BF,
      nullptr,
      weights.data(),
      CUDA_R_16BF,
      nullptr,
      1.0f,
      false,
      m,
      n,
      k,
      output,
      nullptr,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      !IsSupportedDenseCompute(plan.execution.launch_plan.compute_dtype) ||
      activations.shape().size() != 2 ||
      weights.shape().size() != 2 ||
      output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.shape()[0] != n ||
      weights.shape()[1] != k) {
    return std::nullopt;
  }

  return RunDenseRowMajorTypedToDevice(
      handle,
      plan,
      activations.data(),
      CUDA_R_16BF,
      nullptr,
      weights.data(),
      CUDA_R_16BF,
      nullptr,
      1.0f,
      false,
      m,
      n,
      k,
      output,
      nullptr,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      !IsSupportedDenseCompute(plan.execution.launch_plan.compute_dtype) ||
      activations.shape().size() != 2 ||
      weights.shape().size() != 2 ||
      output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.shape()[0] != n ||
      weights.shape()[1] != k) {
    return std::nullopt;
  }

  return RunDenseRowMajorTypedToDevice(
      handle,
      plan,
      activations.data(),
      CUDA_R_16BF,
      nullptr,
      weights.data(),
      CUDA_R_16BF,
      nullptr,
      1.0f,
      false,
      m,
      n,
      k,
      output,
      nullptr,
      stream);
}

template <typename OutputTensorT>
std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp8E4M3QuantizedToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const float* weight_scale_device,
    const DeviceTensorFp8E4M3& activations,
    const float* input_scale_device,
    OutputTensorT* output,
    const CachedCublasLtMatmulState* cached_matmul_state,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      activations.shape().size() != 2 ||
      weights.shape().size() != 2 ||
      output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.shape()[0] != n ||
      weights.shape()[1] != k) {
    return std::nullopt;
  }

  return RunDenseRowMajorTypedToDevice(
      handle,
      plan,
      activations.data(),
      CUDA_R_8F_E4M3,
      input_scale_device,
      weights.data(),
      CUDA_R_8F_E4M3,
      weight_scale_device,
      1.0f,
      Fp8FastAccumSupported(),
      m,
      n,
      k,
      output,
      cached_matmul_state,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp8E4M3ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const float* weight_scale_device,
    const DeviceTensorFp32& activations,
    float input_scale,
    const float* input_scale_device,
    DeviceTensorFp32* output,
    const CachedCublasLtMatmulState* cached_matmul_state,
    DeviceTensorFp8E4M3* activation_scratch,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      activations.shape().size() != 2 ||
      weights.shape().size() != 2 ||
      output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.shape()[0] != n ||
      weights.shape()[1] != k) {
    return std::nullopt;
  }

  std::unique_ptr<DeviceTensorFp8E4M3> owned_activation_scratch;
  DeviceTensorFp8E4M3* quantized_activations = activation_scratch;
  if (quantized_activations == nullptr) {
    owned_activation_scratch = DeviceTensorFp8E4M3::Create(activations.shape());
    quantized_activations = owned_activation_scratch.get();
  }
  if (quantized_activations == nullptr ||
      !quantized_activations->valid() ||
      quantized_activations->shape() != activations.shape() ||
      !QuantizeDeviceFp32ToFp8E4M3(
          activations.data(),
          activations.numel(),
          input_scale,
          quantized_activations->data(),
          stream)) {
    return std::nullopt;
  }

  return RunDenseRowMajorFp8E4M3ToDevice(
      handle,
      plan,
      weights,
      weight_scale_device,
      *quantized_activations,
      input_scale_device,
      output,
      cached_matmul_state,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp8E4M3ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const float* weight_scale_device,
    const DeviceTensorFp8E4M3& activations,
    const float* input_scale_device,
    DeviceTensorFp32* output,
    const CachedCublasLtMatmulState* cached_matmul_state,
    cudaStream_t stream) {
  return RunDenseRowMajorFp8E4M3QuantizedToDevice(
      handle,
      plan,
      weights,
      weight_scale_device,
      activations,
      input_scale_device,
      output,
      cached_matmul_state,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp8E4M3ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const float* weight_scale_device,
    const DeviceTensorFp8E4M3& activations,
    const float* input_scale_device,
    DeviceTensorBf16* output,
    const CachedCublasLtMatmulState* cached_matmul_state,
    cudaStream_t stream) {
  return RunDenseRowMajorFp8E4M3QuantizedToDevice(
      handle,
      plan,
      weights,
      weight_scale_device,
      activations,
      input_scale_device,
      output,
      cached_matmul_state,
      stream);
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  const auto* descriptor = plan.execution.launch_plan.descriptor;
  if (descriptor == nullptr) {
    return std::nullopt;
  }

  auto uploaded_weights = DeviceDenseWeightFp32::Upload(*descriptor);
  if (!uploaded_weights || !uploaded_weights->valid()) {
    return std::nullopt;
  }

  return RunDenseRowMajorFp32ToDevice(handle, plan, *uploaded_weights, activations, output, stream);
}

std::optional<DenseRowMajorHostResult> RunDenseRowMajorFp32(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const float* host_activations,
    std::size_t activation_rows) {
  if (host_activations == nullptr || activation_rows == 0 || plan.execution.launch_plan.k == 0) {
    return std::nullopt;
  }

  auto activations = DeviceTensorFp32::Create({activation_rows, plan.execution.launch_plan.k});
  auto output = DeviceTensorFp32::Create({activation_rows, plan.execution.launch_plan.n});
  if (!activations || !output || !activations->valid() || !output->valid()) {
    return std::nullopt;
  }
  if (!activations->CopyFromHost(host_activations, activations->numel())) {
    return std::nullopt;
  }

  const auto stats = RunDenseRowMajorFp32ToDevice(handle, plan, *activations, output.get());
  if (!stats.has_value()) {
    return std::nullopt;
  }

  std::vector<float> host_output(output->numel(), 0.0f);
  if (!output->CopyToHost(host_output.data(), output->numel())) {
    return std::nullopt;
  }

  return DenseRowMajorHostResult{
      std::move(host_output),
      stats->rows,
      stats->cols,
      stats->workspace_bytes,
      stats->heuristic_count,
  };
}

}  // namespace nemotron
