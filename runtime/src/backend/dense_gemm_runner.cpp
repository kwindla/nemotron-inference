#include "nemotron/dense_gemm_runner.h"

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

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

bool DebugEnabled() {
  return std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
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

void LogCublasFailure(
    const char* op,
    cublasStatus_t status,
    std::size_t m,
    std::size_t n,
    std::size_t k) {
  if (!DebugEnabled()) {
    return;
  }
  std::cerr << "dense_gemm_runner: " << op
            << " failed"
            << " status=" << CublasStatusName(status)
            << "(" << static_cast<int>(status) << ")"
            << " M=" << m
            << " N=" << n
            << " K=" << k
            << "\n";
}

void LogCudaFailure(
    const char* op,
    cudaError_t status,
    std::size_t m,
    std::size_t n,
    std::size_t k) {
  if (!DebugEnabled()) {
    return;
  }
  std::cerr << "dense_gemm_runner: " << op
            << " failed"
            << " status=" << cudaGetErrorName(status)
            << "(" << static_cast<int>(status) << ")"
            << " detail=" << cudaGetErrorString(status)
            << " M=" << m
            << " N=" << n
            << " K=" << k
            << "\n";
}

void LogHeuristicFailure(std::size_t m, std::size_t n, std::size_t k) {
  if (!DebugEnabled()) {
    return;
  }
  std::cerr << "dense_gemm_runner: cublasLtMatmulAlgoGetHeuristic returned 0 results"
            << " M=" << m
            << " N=" << n
            << " K=" << k
            << "\n";
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

}  // namespace

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtDense ||
      plan.execution.launch_plan.descriptor == nullptr ||
      !IsSupportedDenseStorage(plan.execution.launch_plan.descriptor->storage_dtype) ||
      !IsSupportedDenseCompute(plan.execution.launch_plan.descriptor->compute_dtype)) {
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

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  bool ok = true;
  const auto check_cublas = [&](cublasStatus_t status, const char* op) {
    if (CheckCublas(status)) {
      return true;
    }
    LogCublasFailure(op, status, m, n, k);
    return false;
  };
  const auto check_cuda = [&](cudaError_t status, const char* op) {
    if (CheckCuda(status)) {
      return true;
    }
    LogCudaFailure(op, status, m, n, k);
    return false;
  };

  ok &= output->FillZero();

  ok &= check_cublas(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  const cublasOperation_t trans_a = ToCublasOp(plan.transform_a);
  const cublasOperation_t trans_b = ToCublasOp(plan.transform_b);
  ok &= check_cublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_TRANSA,
          &trans_a,
          sizeof(trans_a)),
      "cublasLtMatmulDescSetAttribute(TRANSA)");
  ok &= check_cublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_TRANSB,
          &trans_b,
          sizeof(trans_b)),
      "cublasLtMatmulDescSetAttribute(TRANSB)");

  ok &= check_cublas(
      cublasLtMatrixLayoutCreate(
          &a_desc,
          CUDA_R_32F,
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.lda)),
      "cublasLtMatrixLayoutCreate(A)");
  ok &= check_cublas(
      cublasLtMatrixLayoutCreate(
          &b_desc,
          CUDA_R_32F,
          static_cast<std::uint64_t>(n),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.ldb)),
      "cublasLtMatrixLayoutCreate(B)");
  ok &= check_cublas(
      cublasLtMatrixLayoutCreate(
          &c_desc,
          CUDA_R_32F,
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(n),
          static_cast<std::int64_t>(plan.ldc)),
      "cublasLtMatrixLayoutCreate(C)");

  const cublasLtOrder_t order_a = ToCublasOrder(plan.order_a);
  const cublasLtOrder_t order_b = ToCublasOrder(plan.order_b);
  const cublasLtOrder_t order_c = ToCublasOrder(plan.order_c);
  ok &= check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          a_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_a,
          sizeof(order_a)),
      "cublasLtMatrixLayoutSetAttribute(A_ORDER)");
  ok &= check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          b_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_b,
          sizeof(order_b)),
      "cublasLtMatrixLayoutSetAttribute(B_ORDER)");
  ok &= check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          c_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_c,
          sizeof(order_c)),
      "cublasLtMatrixLayoutSetAttribute(C_ORDER)");

  ok &= check_cublas(
      cublasLtMatmulPreferenceCreate(&preference),
      "cublasLtMatmulPreferenceCreate");
  const std::size_t max_workspace = handle.workspace_bytes();
  ok &= check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &max_workspace,
          sizeof(max_workspace)),
      "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)");

  if (ok) {
    ok &= check_cublas(
        cublasLtMatmulAlgoGetHeuristic(
            handle.handle(),
            op_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            preference,
            1,
            &heuristic,
            &returned_results),
        "cublasLtMatmulAlgoGetHeuristic");
    ok &= returned_results > 0;
    if (ok == false && returned_results == 0) {
      LogHeuristicFailure(m, n, k);
    }
  }

  if (ok) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    ok &= check_cublas(
        cublasLtMatmul(
            handle.handle(),
            op_desc,
            &alpha,
            activations.data(),
            a_desc,
            weights.data(),
            b_desc,
            &beta,
            output->data(),
            c_desc,
            output->data(),
            c_desc,
            &heuristic.algo,
            handle.workspace(),
            handle.workspace_bytes(),
            nullptr),
        "cublasLtMatmul");
    ok &= check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }

  if (preference != nullptr) {
    cublasLtMatmulPreferenceDestroy(preference);
  }
  if (c_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(c_desc);
  }
  if (b_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(b_desc);
  }
  if (a_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(a_desc);
  }
  if (op_desc != nullptr) {
    cublasLtMatmulDescDestroy(op_desc);
  }

  if (!ok) {
    return std::nullopt;
  }

  return DenseRowMajorDeviceStats{
      m,
      n,
      heuristic.workspaceSize,
      returned_results,
  };
}

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output) {
  const auto* descriptor = plan.execution.launch_plan.descriptor;
  if (descriptor == nullptr) {
    return std::nullopt;
  }

  auto uploaded_weights = DeviceDenseWeightFp32::Upload(*descriptor);
  if (!uploaded_weights || !uploaded_weights->valid()) {
    return std::nullopt;
  }

  return RunDenseRowMajorFp32ToDevice(handle, plan, *uploaded_weights, activations, output);
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
