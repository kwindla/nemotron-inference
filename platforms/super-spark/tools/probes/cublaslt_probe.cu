#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* cublas_status_name(cublasStatus_t status) {
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
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

const char* cuda_status_name(cudaError_t status) {
  return cudaGetErrorName(status);
}

size_t packed_fp4_bytes(int64_t rows, int64_t cols) {
  return static_cast<size_t>((rows * cols + 1) / 2);
}

size_t scale_bytes_for_row_major(int64_t rows, int64_t cols) {
  const int64_t blocks = (cols + 15) / 16;
  return static_cast<size_t>(rows * blocks);
}

bool check_cuda(cudaError_t status, const char* expr) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "cuda error for " << expr << ": " << cuda_status_name(status) << "\n";
  return false;
}

bool check_cublas(cublasStatus_t status, const char* expr) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << "cublasLt error for " << expr << ": " << cublas_status_name(status) << "\n";
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const int64_t m = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : 64;
  const int64_t n = argc > 2 ? std::strtoll(argv[2], nullptr, 10) : 64;
  const int64_t k = argc > 3 ? std::strtoll(argv[3], nullptr, 10) : 64;

  int device = 0;
  cudaDeviceProp prop{};
  if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice") ||
      !check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) {
    return 1;
  }

  cublasLtHandle_t handle{};
  if (!check_cublas(cublasLtCreate(&handle), "cublasLtCreate")) {
    return 1;
  }

  cublasLtMatmulDesc_t op_desc{};
  cublasLtMatrixLayout_t a_desc{};
  cublasLtMatrixLayout_t b_desc{};
  cublasLtMatrixLayout_t c_desc{};
  cublasLtMatmulPreference_t preference{};

  void* a_dev = nullptr;
  void* b_dev = nullptr;
  void* c_dev = nullptr;
  void* a_scale_dev = nullptr;
  void* b_scale_dev = nullptr;
  void* workspace_dev = nullptr;

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  cublasStatus_t heuristic_status = CUBLAS_STATUS_SUCCESS;
  cublasStatus_t matmul_status = CUBLAS_STATUS_SUCCESS;
  cudaError_t sync_status = cudaSuccess;

  const auto a_bytes = packed_fp4_bytes(m, k);
  const auto b_bytes = packed_fp4_bytes(k, n);
  const auto c_bytes = static_cast<size_t>(m * n * sizeof(float));
  const auto a_scale_bytes = scale_bytes_for_row_major(m, k);
  const auto b_scale_bytes = scale_bytes_for_row_major(k, n);
  const size_t max_workspace = 32u * 1024u * 1024u;

  bool ok = true;
  ok &= check_cublas(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  ok &= check_cublas(
      cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, m, k, k),
      "cublasLtMatrixLayoutCreate(A)");
  ok &= check_cublas(
      cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, k, n, n),
      "cublasLtMatrixLayoutCreate(B)");
  ok &= check_cublas(
      cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, n),
      "cublasLtMatrixLayoutCreate(C)");
  ok &= check_cublas(
      cublasLtMatmulPreferenceCreate(&preference),
      "cublasLtMatmulPreferenceCreate");

  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  const cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  ok &= check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
      "cublasLtMatrixLayoutSetAttribute(A order)");
  ok &= check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
      "cublasLtMatrixLayoutSetAttribute(B order)");
  ok &= check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
      "cublasLtMatrixLayoutSetAttribute(C order)");

  ok &= check_cuda(cudaMalloc(&a_dev, a_bytes), "cudaMalloc(A)");
  ok &= check_cuda(cudaMalloc(&b_dev, b_bytes), "cudaMalloc(B)");
  ok &= check_cuda(cudaMalloc(&c_dev, c_bytes), "cudaMalloc(C)");
  ok &= check_cuda(cudaMalloc(&a_scale_dev, a_scale_bytes), "cudaMalloc(A scale)");
  ok &= check_cuda(cudaMalloc(&b_scale_dev, b_scale_bytes), "cudaMalloc(B scale)");
  ok &= check_cuda(cudaMalloc(&workspace_dev, max_workspace), "cudaMalloc(workspace)");

  ok &= check_cuda(cudaMemset(a_dev, 0, a_bytes), "cudaMemset(A)");
  ok &= check_cuda(cudaMemset(b_dev, 0, b_bytes), "cudaMemset(B)");
  ok &= check_cuda(cudaMemset(c_dev, 0, c_bytes), "cudaMemset(C)");
  ok &= check_cuda(cudaMemset(a_scale_dev, 0, a_scale_bytes), "cudaMemset(A scale)");
  ok &= check_cuda(cudaMemset(b_scale_dev, 0, b_scale_bytes), "cudaMemset(B scale)");

  ok &= check_cublas(
      cublasLtMatmulDescSetAttribute(
          op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16, sizeof(vec16)),
      "cublasLtMatmulDescSetAttribute(A scale mode)");
  ok &= check_cublas(
      cublasLtMatmulDescSetAttribute(
          op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16, sizeof(vec16)),
      "cublasLtMatmulDescSetAttribute(B scale mode)");
  ok &= check_cublas(
      cublasLtMatmulDescSetAttribute(
          op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_dev, sizeof(a_scale_dev)),
      "cublasLtMatmulDescSetAttribute(A scale ptr)");
  ok &= check_cublas(
      cublasLtMatmulDescSetAttribute(
          op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_dev, sizeof(b_scale_dev)),
      "cublasLtMatmulDescSetAttribute(B scale ptr)");
  ok &= check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &max_workspace,
          sizeof(max_workspace)),
      "cublasLtMatmulPreferenceSetAttribute(max workspace)");

  if (ok) {
    heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        preference,
        1,
        &heuristic,
        &returned_results);
    ok &= check_cublas(heuristic_status, "cublasLtMatmulAlgoGetHeuristic");
  }

  if (ok && returned_results > 0) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    matmul_status = cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        a_dev,
        a_desc,
        b_dev,
        b_desc,
        &beta,
        c_dev,
        c_desc,
        c_dev,
        c_desc,
        &heuristic.algo,
        workspace_dev,
        max_workspace,
        nullptr);
    ok &= check_cublas(matmul_status, "cublasLtMatmul");
    sync_status = cudaDeviceSynchronize();
    ok &= check_cuda(sync_status, "cudaDeviceSynchronize");
  }

  const char* matmul_status_text = returned_results > 0 ? cublas_status_name(matmul_status) : "not_run";
  const char* sync_status_text = returned_results > 0 ? cuda_status_name(sync_status) : "not_run";

  std::cout << "{\n";
  std::cout << "  \"device\": \"" << prop.name << "\",\n";
  std::cout << "  \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
  std::cout << "  \"shape\": {\"m\": " << m << ", \"n\": " << n << ", \"k\": " << k << "},\n";
  std::cout << "  \"a_bytes\": " << a_bytes << ",\n";
  std::cout << "  \"b_bytes\": " << b_bytes << ",\n";
  std::cout << "  \"c_bytes\": " << c_bytes << ",\n";
  std::cout << "  \"a_scale_bytes\": " << a_scale_bytes << ",\n";
  std::cout << "  \"b_scale_bytes\": " << b_scale_bytes << ",\n";
  std::cout << "  \"heuristic_status\": \"" << cublas_status_name(heuristic_status) << "\",\n";
  std::cout << "  \"heuristic_count\": " << returned_results << ",\n";
  std::cout << "  \"workspace_bytes\": " << (returned_results > 0 ? heuristic.workspaceSize : 0) << ",\n";
  std::cout << "  \"matmul_status\": \"" << matmul_status_text << "\",\n";
  std::cout << "  \"cuda_sync_status\": \"" << sync_status_text << "\",\n";
  std::cout << "  \"success\": " << (ok ? "true" : "false") << "\n";
  std::cout << "}\n";

  if (workspace_dev != nullptr) {
    cudaFree(workspace_dev);
  }
  if (b_scale_dev != nullptr) {
    cudaFree(b_scale_dev);
  }
  if (a_scale_dev != nullptr) {
    cudaFree(a_scale_dev);
  }
  if (c_dev != nullptr) {
    cudaFree(c_dev);
  }
  if (b_dev != nullptr) {
    cudaFree(b_dev);
  }
  if (a_dev != nullptr) {
    cudaFree(a_dev);
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
  if (handle != nullptr) {
    cublasLtDestroy(handle);
  }

  return ok ? 0 : 1;
}
