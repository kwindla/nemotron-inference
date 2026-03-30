#include <cuda_runtime.h>
#include <cublasLt.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct ShapeCase {
  std::string name;
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
};

struct ContractCase {
  std::string name;
  cublasLtOrder_t order_a = CUBLASLT_ORDER_ROW;
  cublasLtOrder_t order_b = CUBLASLT_ORDER_ROW;
  cublasLtOrder_t order_c = CUBLASLT_ORDER_ROW;
  cublasOperation_t trans_a = CUBLAS_OP_N;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  bool a_dims_mk = true;
  bool b_dims_kn = true;
};

struct EnvironmentInfo {
  std::string device_name;
  int runtime_version = 0;
  int driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
};

struct ProbeResult {
  std::string shape_name;
  std::string contract_name;
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int64_t a_rows = 0;
  std::int64_t a_cols = 0;
  std::int64_t b_rows = 0;
  std::int64_t b_cols = 0;
  std::size_t a_bytes = 0;
  std::size_t b_bytes = 0;
  std::size_t c_bytes = 0;
  std::size_t a_scale_bytes = 0;
  std::size_t b_scale_bytes = 0;
  std::size_t max_workspace_bytes = 0;
  std::string create_status = "not_run";
  std::string heuristic_status = "not_run";
  int heuristic_count = 0;
  std::size_t heuristic_workspace_bytes = 0;
  std::string matmul_status = "not_run";
  std::string sync_status = "not_run";
  double heuristic_ms = 0.0;
  double matmul_ms = 0.0;
  bool success = false;
};

const std::vector<ShapeCase>& DefaultShapes() {
  static const std::vector<ShapeCase> kShapes = {
      {"toy_64", 64, 64, 64},
      {"attention_core_m64", 64, 4096, 4096},
      {"attention_core_m256", 256, 4096, 4096},
  };
  return kShapes;
}

const std::vector<ContractCase>& DefaultContracts() {
  static const std::vector<ContractCase> kContracts = {
      {
          "naive_row_nn_mk_kn",
          CUBLASLT_ORDER_ROW,
          CUBLASLT_ORDER_ROW,
          CUBLASLT_ORDER_ROW,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          true,
          true,
      },
      {
          "row_nt_mk_nk",
          CUBLASLT_ORDER_ROW,
          CUBLASLT_ORDER_ROW,
          CUBLASLT_ORDER_ROW,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          true,
          false,
      },
      {
          "col_nn_mk_kn",
          CUBLASLT_ORDER_COL,
          CUBLASLT_ORDER_COL,
          CUBLASLT_ORDER_COL,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          true,
          true,
      },
      {
          "col_nt_mk_nk",
          CUBLASLT_ORDER_COL,
          CUBLASLT_ORDER_COL,
          CUBLASLT_ORDER_COL,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          true,
          false,
      },
      {
          "row_tn_km_kn",
          CUBLASLT_ORDER_ROW,
          CUBLASLT_ORDER_ROW,
          CUBLASLT_ORDER_ROW,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          false,
          true,
      },
      {
          "col_tn_km_kn",
          CUBLASLT_ORDER_COL,
          CUBLASLT_ORDER_COL,
          CUBLASLT_ORDER_COL,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          false,
          true,
      },
  };
  return kContracts;
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
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

const char* CudaStatusName(cudaError_t status) {
  return cudaGetErrorName(status);
}

const char* OrderName(cublasLtOrder_t order) {
  switch (order) {
    case CUBLASLT_ORDER_ROW:
      return "row";
    case CUBLASLT_ORDER_COL:
      return "col";
    case CUBLASLT_ORDER_COL32:
      return "col32";
    case CUBLASLT_ORDER_COL4_4R2_8C:
      return "col4_4r2_8c";
    case CUBLASLT_ORDER_COL32_2R_4R4:
      return "col32_2r_4r4";
  }
  return "unknown";
}

const char* TransName(cublasOperation_t trans) {
  switch (trans) {
    case CUBLAS_OP_N:
      return "N";
    case CUBLAS_OP_T:
      return "T";
    case CUBLAS_OP_C:
      return "C";
  }
  return "?";
}

std::size_t PackedFp4Bytes(std::int64_t rows, std::int64_t cols) {
  return static_cast<std::size_t>((rows * cols + 1) / 2);
}

std::size_t LeadingDim(std::int64_t rows, std::int64_t cols, cublasLtOrder_t order) {
  return order == CUBLASLT_ORDER_ROW ? static_cast<std::size_t>(cols) : static_cast<std::size_t>(rows);
}

std::size_t ScaleBytes(std::int64_t rows, std::int64_t cols, cublasLtOrder_t order) {
  if (order == CUBLASLT_ORDER_ROW) {
    const std::int64_t blocks = (cols + 15) / 16;
    return static_cast<std::size_t>(rows * blocks);
  }
  const std::int64_t blocks = (rows + 15) / 16;
  return static_cast<std::size_t>(cols * blocks);
}

bool SetMatmulAttribute(
    cublasLtMatmulDesc_t desc,
    cublasLtMatmulDescAttributes_t attr,
    const void* value,
    std::size_t bytes,
    ProbeResult* result) {
  const cublasStatus_t status = cublasLtMatmulDescSetAttribute(desc, attr, value, bytes);
  if (status != CUBLAS_STATUS_SUCCESS && result != nullptr) {
    result->create_status = CublasStatusName(status);
  }
  return status == CUBLAS_STATUS_SUCCESS;
}

bool SetLayoutAttribute(
    cublasLtMatrixLayout_t layout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void* value,
    std::size_t bytes,
    ProbeResult* result) {
  const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(layout, attr, value, bytes);
  if (status != CUBLAS_STATUS_SUCCESS && result != nullptr) {
    result->create_status = CublasStatusName(status);
  }
  return status == CUBLAS_STATUS_SUCCESS;
}

double ElapsedMilliseconds(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool ReadEnvironmentInfo(EnvironmentInfo* info) {
  if (info == nullptr) {
    return false;
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return false;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return false;
  }
  info->device_name = prop.name;
  info->compute_capability_major = prop.major;
  info->compute_capability_minor = prop.minor;
  cudaRuntimeGetVersion(&info->runtime_version);
  cudaDriverGetVersion(&info->driver_version);
  return true;
}

bool AllocateZeroed(void** ptr, std::size_t bytes) {
  return cudaMalloc(ptr, bytes) == cudaSuccess && cudaMemset(*ptr, 0, bytes) == cudaSuccess;
}

bool AllocateFilled(void** ptr, std::size_t bytes, unsigned char byte_value) {
  return cudaMalloc(ptr, bytes) == cudaSuccess && cudaMemset(*ptr, byte_value, bytes) == cudaSuccess;
}

ProbeResult RunProbe(
    cublasLtHandle_t handle,
    const ShapeCase& shape,
    const ContractCase& contract,
    std::size_t max_workspace_bytes) {
  ProbeResult result;
  result.shape_name = shape.name;
  result.contract_name = contract.name;
  result.m = shape.m;
  result.n = shape.n;
  result.k = shape.k;
  result.max_workspace_bytes = max_workspace_bytes;

  result.a_rows = contract.a_dims_mk ? shape.m : shape.k;
  result.a_cols = contract.a_dims_mk ? shape.k : shape.m;
  result.b_rows = contract.b_dims_kn ? shape.k : shape.n;
  result.b_cols = contract.b_dims_kn ? shape.n : shape.k;

  result.a_bytes = PackedFp4Bytes(result.a_rows, result.a_cols);
  result.b_bytes = PackedFp4Bytes(result.b_rows, result.b_cols);
  result.c_bytes = static_cast<std::size_t>(shape.m * shape.n * sizeof(float));
  result.a_scale_bytes = ScaleBytes(result.a_rows, result.a_cols, contract.order_a);
  result.b_scale_bytes = ScaleBytes(result.b_rows, result.b_cols, contract.order_b);

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  void* a_dev = nullptr;
  void* b_dev = nullptr;
  void* c_dev = nullptr;
  void* a_scale_dev = nullptr;
  void* b_scale_dev = nullptr;
  void* workspace_dev = nullptr;

  auto cleanup = [&]() {
    if (workspace_dev != nullptr) cudaFree(workspace_dev);
    if (b_scale_dev != nullptr) cudaFree(b_scale_dev);
    if (a_scale_dev != nullptr) cudaFree(a_scale_dev);
    if (c_dev != nullptr) cudaFree(c_dev);
    if (b_dev != nullptr) cudaFree(b_dev);
    if (a_dev != nullptr) cudaFree(a_dev);
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
  };

  const cublasStatus_t create_status =
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  result.create_status = CublasStatusName(create_status);
  if (create_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return result;
  }

  if (cublasLtMatrixLayoutCreate(
          &a_desc,
          CUDA_R_4F_E2M1,
          static_cast<std::uint64_t>(result.a_rows),
          static_cast<std::uint64_t>(result.a_cols),
          static_cast<std::int64_t>(LeadingDim(result.a_rows, result.a_cols, contract.order_a))) !=
      CUBLAS_STATUS_SUCCESS) {
    result.create_status = "layout_create_a_failed";
    cleanup();
    return result;
  }
  if (cublasLtMatrixLayoutCreate(
          &b_desc,
          CUDA_R_4F_E2M1,
          static_cast<std::uint64_t>(result.b_rows),
          static_cast<std::uint64_t>(result.b_cols),
          static_cast<std::int64_t>(LeadingDim(result.b_rows, result.b_cols, contract.order_b))) !=
      CUBLAS_STATUS_SUCCESS) {
    result.create_status = "layout_create_b_failed";
    cleanup();
    return result;
  }
  if (cublasLtMatrixLayoutCreate(
          &c_desc,
          CUDA_R_32F,
          static_cast<std::uint64_t>(shape.m),
          static_cast<std::uint64_t>(shape.n),
          static_cast<std::int64_t>(LeadingDim(shape.m, shape.n, contract.order_c))) !=
      CUBLAS_STATUS_SUCCESS) {
    result.create_status = "layout_create_c_failed";
    cleanup();
    return result;
  }
  if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
    result.create_status = "preference_create_failed";
    cleanup();
    return result;
  }

  const cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  if (!SetMatmulAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &contract.trans_a, sizeof(contract.trans_a), &result) ||
      !SetMatmulAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &contract.trans_b, sizeof(contract.trans_b), &result) ||
      !SetMatmulAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16, sizeof(vec16), &result) ||
      !SetMatmulAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16, sizeof(vec16), &result)) {
    cleanup();
    return result;
  }

  if (!SetLayoutAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &contract.order_a, sizeof(contract.order_a), &result) ||
      !SetLayoutAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &contract.order_b, sizeof(contract.order_b), &result) ||
      !SetLayoutAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &contract.order_c, sizeof(contract.order_c), &result)) {
    cleanup();
    return result;
  }

  if (!AllocateZeroed(&a_dev, result.a_bytes) ||
      !AllocateZeroed(&b_dev, result.b_bytes) ||
      !AllocateZeroed(&c_dev, result.c_bytes) ||
      !AllocateFilled(&a_scale_dev, result.a_scale_bytes, 0x38) ||
      !AllocateFilled(&b_scale_dev, result.b_scale_bytes, 0x38) ||
      !AllocateZeroed(&workspace_dev, max_workspace_bytes)) {
    result.create_status = "device_allocation_failed";
    cleanup();
    return result;
  }

  if (!SetMatmulAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
          &a_scale_dev,
          sizeof(a_scale_dev),
          &result) ||
      !SetMatmulAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
          &b_scale_dev,
          sizeof(b_scale_dev),
          &result)) {
    cleanup();
    return result;
  }

  if (cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &max_workspace_bytes,
          sizeof(max_workspace_bytes)) != CUBLAS_STATUS_SUCCESS) {
    result.create_status = "preference_workspace_attr_failed";
    cleanup();
    return result;
  }

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  const auto heuristic_begin = std::chrono::steady_clock::now();
  const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
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
  const auto heuristic_end = std::chrono::steady_clock::now();
  result.heuristic_status = CublasStatusName(heuristic_status);
  result.heuristic_ms = ElapsedMilliseconds(heuristic_begin, heuristic_end);
  result.heuristic_count = returned_results;
  result.heuristic_workspace_bytes = returned_results > 0 ? heuristic.workspaceSize : 0;
  if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned_results <= 0) {
    cleanup();
    return result;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto matmul_begin = std::chrono::steady_clock::now();
  const cublasStatus_t matmul_status = cublasLtMatmul(
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
      max_workspace_bytes,
      nullptr);
  const cudaError_t sync_status = cudaDeviceSynchronize();
  const auto matmul_end = std::chrono::steady_clock::now();

  result.matmul_status = CublasStatusName(matmul_status);
  result.sync_status = CudaStatusName(sync_status);
  result.matmul_ms = ElapsedMilliseconds(matmul_begin, matmul_end);
  result.success = matmul_status == CUBLAS_STATUS_SUCCESS && sync_status == cudaSuccess;
  cleanup();
  return result;
}

std::string EscapeJson(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

bool WriteJsonReport(
    const std::filesystem::path& path,
    const EnvironmentInfo& info,
    const std::vector<ProbeResult>& results,
    std::size_t workspace_bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output) {
    return false;
  }

  output << "{\n";
  output << "  \"benchmark\": \"gb10_nvfp4_contract_bench\",\n";
  output << "  \"device\": {\n";
  output << "    \"name\": \"" << EscapeJson(info.device_name) << "\",\n";
  output << "    \"compute_capability\": \"" << info.compute_capability_major << "." << info.compute_capability_minor
         << "\",\n";
  output << "    \"cuda_runtime_version\": " << info.runtime_version << ",\n";
  output << "    \"cuda_driver_version\": " << info.driver_version << "\n";
  output << "  },\n";
  output << "  \"max_workspace_bytes\": " << workspace_bytes << ",\n";
  output << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    output << "    {\n";
    output << "      \"shape_name\": \"" << EscapeJson(r.shape_name) << "\",\n";
    output << "      \"contract_name\": \"" << EscapeJson(r.contract_name) << "\",\n";
    output << "      \"m\": " << r.m << ",\n";
    output << "      \"n\": " << r.n << ",\n";
    output << "      \"k\": " << r.k << ",\n";
    output << "      \"a_rows\": " << r.a_rows << ",\n";
    output << "      \"a_cols\": " << r.a_cols << ",\n";
    output << "      \"b_rows\": " << r.b_rows << ",\n";
    output << "      \"b_cols\": " << r.b_cols << ",\n";
    output << "      \"a_bytes\": " << r.a_bytes << ",\n";
    output << "      \"b_bytes\": " << r.b_bytes << ",\n";
    output << "      \"c_bytes\": " << r.c_bytes << ",\n";
    output << "      \"a_scale_bytes\": " << r.a_scale_bytes << ",\n";
    output << "      \"b_scale_bytes\": " << r.b_scale_bytes << ",\n";
    output << "      \"create_status\": \"" << EscapeJson(r.create_status) << "\",\n";
    output << "      \"heuristic_status\": \"" << EscapeJson(r.heuristic_status) << "\",\n";
    output << "      \"heuristic_count\": " << r.heuristic_count << ",\n";
    output << "      \"heuristic_workspace_bytes\": " << r.heuristic_workspace_bytes << ",\n";
    output << "      \"matmul_status\": \"" << EscapeJson(r.matmul_status) << "\",\n";
    output << "      \"sync_status\": \"" << EscapeJson(r.sync_status) << "\",\n";
    output << "      \"heuristic_ms\": " << std::fixed << std::setprecision(6) << r.heuristic_ms << ",\n";
    output << "      \"matmul_ms\": " << r.matmul_ms << ",\n";
    output << "      \"success\": " << (r.success ? "true" : "false") << "\n";
    output << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  output << "  ]\n";
  output << "}\n";
  return true;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [--json-output <path>] [--workspace-bytes <value>] [--list-contracts]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<std::filesystem::path> json_output_path;
  std::size_t max_workspace_bytes = 32u * 1024u * 1024u;
  bool list_only = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--json-output" && i + 1 < argc) {
      json_output_path = std::filesystem::path(argv[++i]);
      continue;
    }
    if (arg == "--workspace-bytes" && i + 1 < argc) {
      try {
        max_workspace_bytes = std::stoull(argv[++i]);
      } catch (...) {
        PrintUsage(argv[0]);
        return 1;
      }
      continue;
    }
    if (arg == "--list-contracts") {
      list_only = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    PrintUsage(argv[0]);
    return 1;
  }

  if (list_only) {
    std::cout << "Default shapes:\n";
    for (const auto& shape : DefaultShapes()) {
      std::cout << "  " << shape.name << " m=" << shape.m << " n=" << shape.n << " k=" << shape.k << "\n";
    }
    std::cout << "Default contracts:\n";
    for (const auto& contract : DefaultContracts()) {
      std::cout << "  " << contract.name
                << " A(order=" << OrderName(contract.order_a) << ",dims=" << (contract.a_dims_mk ? "m,k" : "k,m")
                << ",trans=" << TransName(contract.trans_a) << ")"
                << " B(order=" << OrderName(contract.order_b) << ",dims=" << (contract.b_dims_kn ? "k,n" : "n,k")
                << ",trans=" << TransName(contract.trans_b) << ")"
                << " C(order=" << OrderName(contract.order_c) << ")"
                << "\n";
    }
    return 0;
  }

  EnvironmentInfo info;
  if (!ReadEnvironmentInfo(&info)) {
    std::cerr << "No CUDA device available.\n";
    return 1;
  }

  cublasLtHandle_t handle = nullptr;
  if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasLtCreate failed.\n";
    return 1;
  }

  std::vector<ProbeResult> results;
  for (const auto& shape : DefaultShapes()) {
    for (const auto& contract : DefaultContracts()) {
      const ProbeResult result = RunProbe(handle, shape, contract, max_workspace_bytes);
      std::cout << "shape=" << result.shape_name
                << " contract=" << result.contract_name
                << " heuristic=" << result.heuristic_status
                << " count=" << result.heuristic_count
                << " matmul=" << result.matmul_status
                << " success=" << (result.success ? "true" : "false")
                << "\n";
      results.push_back(result);
    }
  }

  cublasLtDestroy(handle);

  if (json_output_path.has_value() &&
      !WriteJsonReport(*json_output_path, info, results, max_workspace_bytes)) {
    std::cerr << "Failed to write JSON report.\n";
    return 1;
  }

  return 0;
}
