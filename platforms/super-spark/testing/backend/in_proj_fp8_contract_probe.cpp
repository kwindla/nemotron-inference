#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/cutlass_fp8_gemm.h"
#include "nemotron/dense_gemm_runner.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/scaled_fp8_linear.h"
#include "../../runtime/src/backend/storage_conversion.h"

#include <cublasLt.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildGemmLaunchPlan;
using nemotron::CublasLtContract;
using nemotron::CublasLtGemmPlan;
using nemotron::CublasLtHandle;
using nemotron::ConvertDeviceFp32ToBf16;
using nemotron::CutlassFp8DenseGemmAvailable;
using nemotron::CublasLtMatrixOrder;
using nemotron::CublasLtTransform;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::DeviceTensorFp8E4M3;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::PrepareGemmExecution;
using nemotron::QuantizeDeviceFp32ToFp8E4M3;
using nemotron::QuantizeBf16ToFp8E4M3;
using nemotron::RunCutlassFp8DenseGemm;
using nemotron::RunDenseRowMajorFp8E4M3ToDevice;
using nemotron::ScaledFp8LinearConfig;
using nemotron::ScaledFp8LinearOp;
using nemotron::UploadPackedFloatToDeviceBf16;
using nemotron::PackedFloatStorage;

constexpr std::size_t kTokens = 40;
constexpr std::size_t kHiddenSize = 4096;
constexpr std::size_t kIntermediateSize = 8192;
constexpr std::size_t kNumHeads = 128;
constexpr std::size_t kStateSize = 128;
constexpr std::size_t kNumGroups = 8;
constexpr std::size_t kConvDim = kIntermediateSize + (2 * kNumGroups * kStateSize);
constexpr std::size_t kProjectedSize = kIntermediateSize + kConvDim + kNumHeads;

std::filesystem::path ResolveFixtureRoot() {
  const char* override_env = std::getenv("NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT_OVERRIDE");
  if (override_env != nullptr && *override_env != '\0') {
    return std::filesystem::path(override_env);
  }
  return std::filesystem::path(NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT);
}

std::vector<float> ReadFp32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  if (nbytes % sizeof(float) != 0) {
    std::cerr << "FAIL: invalid fp32 payload size " << path << "\n";
    return {};
  }
  std::vector<float> data(nbytes / sizeof(float), 0.0f);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
  return data;
}

std::vector<std::uint8_t> ReadU8(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(nbytes, 0);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
  return data;
}

bool CopyBf16TensorToHost(const DeviceTensorBf16& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  if (!tensor.CopyToHost(host_bf16.data(), host_bf16.size())) {
    return false;
  }
  output->resize(host_bf16.size(), 0.0f);
  for (std::size_t i = 0; i < host_bf16.size(); ++i) {
    (*output)[i] = __bfloat162float(host_bf16[i]);
  }
  return true;
}

bool CopyFp32TensorToHost(const DeviceTensorFp32& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  output->resize(tensor.numel(), 0.0f);
  return tensor.CopyToHost(output->data(), output->size());
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_d = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_d = std::max(max_d, std::fabs(a[i] - b[i]));
  }
  return max_d;
}

float RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    diff_sq += diff * diff;
    ref_sq += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return ref_sq == 0.0 ? 0.0f : static_cast<float>(std::sqrt(diff_sq / ref_sq));
}

void ApplyContract(CublasLtContract contract, CublasLtGemmPlan* plan) {
  if (plan == nullptr) {
    return;
  }
  plan->contract = contract;
  switch (contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      plan->transform_a = CublasLtTransform::kNone;
      plan->transform_b = CublasLtTransform::kTranspose;
      plan->order_a = CublasLtMatrixOrder::kRowMajor;
      plan->order_b = CublasLtMatrixOrder::kRowMajor;
      plan->order_c = CublasLtMatrixOrder::kRowMajor;
      return;
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      plan->transform_a = CublasLtTransform::kTranspose;
      plan->transform_b = CublasLtTransform::kNone;
      plan->order_a = CublasLtMatrixOrder::kColumnMajor;
      plan->order_b = CublasLtMatrixOrder::kColumnMajor;
      plan->order_c = CublasLtMatrixOrder::kColumnMajor;
      return;
  }
}

std::optional<CublasLtGemmPlan> BuildPlanForContract(
    const GemmDescriptor& descriptor,
    std::size_t rows,
    CublasLtContract contract) {
  GemmHeuristicCache heuristic_cache;
  const auto launch_plan = BuildGemmLaunchPlan(descriptor, rows);
  if (!launch_plan.has_value()) {
    std::cerr << "FAIL: BuildGemmLaunchPlan failed\n";
    return std::nullopt;
  }
  const auto execution = PrepareGemmExecution(*launch_plan, &heuristic_cache);
  if (!execution.has_value()) {
    std::cerr << "FAIL: PrepareGemmExecution failed\n";
    return std::nullopt;
  }
  auto plan = BuildCublasLtGemmPlan(*execution);
  if (!plan.has_value()) {
    std::cerr << "FAIL: BuildCublasLtGemmPlan failed\n";
    return std::nullopt;
  }
  ApplyContract(contract, &*plan);
  return plan;
}

bool RunContract(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const DeviceTensorFp8E4M3& quantized_activations,
    const DeviceTensorFp32& weight_scale_device,
    const DeviceTensorFp32& input_scale_device,
    DeviceTensorBf16* output,
    std::vector<float>* host_output) {
  if (output == nullptr || host_output == nullptr) {
    return false;
  }
  if (!RunDenseRowMajorFp8E4M3ToDevice(
          handle,
          plan,
          weights,
          weight_scale_device.data(),
          quantized_activations,
          input_scale_device.data(),
          output,
          nullptr,
          nullptr)
           .has_value()) {
    return false;
  }
  return CopyBf16TensorToHost(*output, host_output);
}

bool RunTorchStyleContract(
    CublasLtHandle& handle,
    const DeviceTensorFp8E4M3& weights,
    const DeviceTensorFp8E4M3& quantized_activations,
    const DeviceTensorFp32& weight_scale_device,
    const DeviceTensorFp32& input_scale_device,
    bool explicit_scalar_scale_mode,
    DeviceTensorBf16* output,
    std::vector<float>* host_output) {
  if (!handle.valid() ||
      !weights.valid() ||
      !quantized_activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      host_output == nullptr) {
    return false;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  bool ok = true;

  const auto apply_cublas = [&](const char* step, cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "FAIL: " << step << " status=" << static_cast<int>(status) << "\n";
      ok = false;
      return false;
    }
    return true;
  };
  const auto apply_cuda = [&](const char* step, cudaError_t status) {
    if (status != cudaSuccess) {
      std::cerr << "FAIL: " << step << " cuda_status=" << cudaGetErrorString(status) << "\n";
      ok = false;
      return false;
    }
    return true;
  };

  const std::int64_t m = static_cast<std::int64_t>(quantized_activations.shape()[0]);
  const std::int64_t k = static_cast<std::int64_t>(quantized_activations.shape()[1]);
  const std::int64_t n = static_cast<std::int64_t>(weights.shape()[0]);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const cublasOperation_t trans_n = CUBLAS_OP_N;
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  const cublasLtOrder_t col_order = CUBLASLT_ORDER_COL;
  apply_cublas(
      "matmul_desc_create",
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  if (ok) {
    apply_cublas(
        "set_transa",
        cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_n, sizeof(trans_n)));
  }
  if (ok) {
    apply_cublas(
        "set_transb",
        cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_n, sizeof(trans_n)));
  }
  const void* a_scale_ptr = input_scale_device.data();
  const void* b_scale_ptr = weight_scale_device.data();
  if (ok) {
    apply_cublas(
        "set_a_scale_ptr",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale_ptr,
            sizeof(a_scale_ptr)));
  }
  if (ok) {
    apply_cublas(
        "set_b_scale_ptr",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale_ptr,
            sizeof(b_scale_ptr)));
  }
  if (ok && explicit_scalar_scale_mode) {
    const cublasLtMatmulMatrixScale_t scalar_scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    apply_cublas(
        "set_a_scale_mode",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            &scalar_scale_mode,
            sizeof(scalar_scale_mode)));
  }
  if (ok && explicit_scalar_scale_mode) {
    const cublasLtMatmulMatrixScale_t scalar_scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    apply_cublas(
        "set_b_scale_mode",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            &scalar_scale_mode,
            sizeof(scalar_scale_mode)));
  }

  if (ok) {
    apply_cublas(
        "layout_a_create",
        cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, m, k, k));
  }
  if (ok) {
    apply_cublas(
        "layout_b_create",
        cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, k, n, k));
  }
  if (ok) {
    apply_cublas(
        "layout_c_create",
        cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, n));
  }
  if (ok) {
    apply_cublas(
        "layout_a_order",
        cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  }
  if (ok) {
    apply_cublas(
        "layout_b_order",
        cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order)));
  }
  if (ok) {
    apply_cublas(
        "layout_c_order",
        cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  }

  if (ok) {
    apply_cublas("preference_create", cublasLtMatmulPreferenceCreate(&preference));
  }
  const std::size_t max_workspace = handle.workspace_bytes();
  if (ok) {
    apply_cublas(
        "preference_workspace",
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
            op_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            preference,
            1,
            &heuristic,
            &returned_results));
    if (ok && returned_results <= 0) {
      std::cerr << "FAIL: algo_get_heuristic returned no results\n";
      ok = false;
    }
  }

  if (ok) {
    apply_cublas(
        "matmul",
        cublasLtMatmul(
            handle.handle(),
            op_desc,
            &alpha,
            quantized_activations.data(),
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
            nullptr));
  }
  if (ok) {
    apply_cuda("matmul_cuda", cudaGetLastError());
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
    return false;
  }
  return CopyBf16TensorToHost(*output, host_output);
}

bool RunTorchStyleContractFp32(
    CublasLtHandle& handle,
    const DeviceTensorFp8E4M3& weights,
    const DeviceTensorFp8E4M3& quantized_activations,
    const DeviceTensorFp32& weight_scale_device,
    const DeviceTensorFp32& input_scale_device,
    bool explicit_scalar_scale_mode,
    DeviceTensorFp32* output,
    std::vector<float>* host_output) {
  if (!handle.valid() ||
      !weights.valid() ||
      !quantized_activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      host_output == nullptr) {
    return false;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  bool ok = true;

  const auto apply_cublas = [&](const char* step, cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "FAIL: " << step << " status=" << static_cast<int>(status) << "\n";
      ok = false;
      return false;
    }
    return true;
  };
  const auto apply_cuda = [&](const char* step, cudaError_t status) {
    if (status != cudaSuccess) {
      std::cerr << "FAIL: " << step << " cuda_status=" << cudaGetErrorString(status) << "\n";
      ok = false;
      return false;
    }
    return true;
  };

  const std::int64_t m = static_cast<std::int64_t>(quantized_activations.shape()[0]);
  const std::int64_t k = static_cast<std::int64_t>(quantized_activations.shape()[1]);
  const std::int64_t n = static_cast<std::int64_t>(weights.shape()[0]);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const cublasOperation_t trans_n = CUBLAS_OP_N;
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  const cublasLtOrder_t col_order = CUBLASLT_ORDER_COL;
  apply_cublas(
      "matmul_desc_create",
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  if (ok) {
    apply_cublas(
        "set_transa",
        cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_n, sizeof(trans_n)));
  }
  if (ok) {
    apply_cublas(
        "set_transb",
        cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_n, sizeof(trans_n)));
  }
  const void* a_scale_ptr = input_scale_device.data();
  const void* b_scale_ptr = weight_scale_device.data();
  if (ok) {
    apply_cublas(
        "set_a_scale_ptr",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale_ptr,
            sizeof(a_scale_ptr)));
  }
  if (ok) {
    apply_cublas(
        "set_b_scale_ptr",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale_ptr,
            sizeof(b_scale_ptr)));
  }
  if (ok && explicit_scalar_scale_mode) {
    const cublasLtMatmulMatrixScale_t scalar_scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    apply_cublas(
        "set_a_scale_mode",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            &scalar_scale_mode,
            sizeof(scalar_scale_mode)));
  }
  if (ok && explicit_scalar_scale_mode) {
    const cublasLtMatmulMatrixScale_t scalar_scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    apply_cublas(
        "set_b_scale_mode",
        cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            &scalar_scale_mode,
            sizeof(scalar_scale_mode)));
  }

  if (ok) {
    apply_cublas(
        "layout_a_create",
        cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, m, k, k));
  }
  if (ok) {
    apply_cublas(
        "layout_b_create",
        cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, k, n, k));
  }
  if (ok) {
    apply_cublas(
        "layout_c_create",
        cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, n));
  }
  if (ok) {
    apply_cublas(
        "layout_a_order",
        cublasLtMatrixLayoutSetAttribute(
            a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  }
  if (ok) {
    apply_cublas(
        "layout_b_order",
        cublasLtMatrixLayoutSetAttribute(
            b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order)));
  }
  if (ok) {
    apply_cublas(
        "layout_c_order",
        cublasLtMatrixLayoutSetAttribute(
            c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  }

  if (ok) {
    apply_cublas("preference_create", cublasLtMatmulPreferenceCreate(&preference));
  }
  const std::size_t max_workspace = handle.workspace_bytes();
  if (ok) {
    apply_cublas(
        "preference_workspace",
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
            op_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            preference,
            1,
            &heuristic,
            &returned_results));
    if (ok && returned_results <= 0) {
      std::cerr << "FAIL: algo_get_heuristic returned no results\n";
      ok = false;
    }
  }

  if (ok) {
    apply_cublas(
        "matmul",
        cublasLtMatmul(
            handle.handle(),
            op_desc,
            &alpha,
            quantized_activations.data(),
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
            nullptr));
  }
  if (ok) {
    apply_cuda("matmul_cuda", cudaGetLastError());
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
    return false;
  }
  return CopyFp32TensorToHost(*output, host_output);
}

}  // namespace

int main() {
  const char* dump_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  if (dump_dir_env == nullptr || *dump_dir_env == '\0') {
    std::cerr << "SKIP: set NEMOTRON_VLLM_DUMP_DIR to a pinned vLLM dump\n";
    return 0;
  }

  const std::filesystem::path dump_dir(dump_dir_env);
  const std::filesystem::path fixture_root = ResolveFixtureRoot();

  const auto norm_output = ReadFp32(dump_dir / "norm_output_fp32.bin");
  const auto expected_projected = ReadFp32(dump_dir / "projected_states_fp32.bin");
  const auto packed_weight = ReadU8(fixture_root / "in_proj_weight_fp8.bin");
  const auto weight_scale = ReadFp32(fixture_root / "in_proj_weight_scale_fp32.bin");
  const auto input_scale = ReadFp32(fixture_root / "in_proj_input_scale_fp32.bin");

  if (norm_output.size() != (kTokens * kHiddenSize) ||
      expected_projected.size() != (kTokens * kProjectedSize) ||
      packed_weight.size() != (kProjectedSize * kHiddenSize) ||
      weight_scale.size() != 1 ||
      input_scale.size() != 1) {
    std::cerr << "FAIL: unexpected oracle fixture sizes\n";
    return 1;
  }

  GemmDescriptor descriptor;
  descriptor.tensor_name = "layer0_in_proj_contract_probe";
  descriptor.op_class = "scaled_fp8_linear";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = kProjectedSize;
  descriptor.input_cols = kHiddenSize;
  descriptor.storage_dtype = "fp8_e4m3fn";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed_weight.data();
  descriptor.packed_nbytes = packed_weight.size();

  auto handle = CublasLtHandle::Create();
  auto activations = DeviceTensorFp32::Create({kTokens, kHiddenSize});
  auto activations_bf16 = DeviceTensorBf16::Create({kTokens, kHiddenSize});
  auto weights = DeviceTensorFp8E4M3::Create({kProjectedSize, kHiddenSize});
  auto quantized_activations = DeviceTensorFp8E4M3::Create({kTokens, kHiddenSize});
  auto quantized_activations_bf16 = DeviceTensorFp8E4M3::Create({kTokens, kHiddenSize});
  auto weight_scale_device = DeviceTensorFp32::Create({1});
  auto input_scale_device = DeviceTensorFp32::Create({1});
  auto row_major_output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  auto torch_style_output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  auto torch_style_explicit_scale_output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  auto torch_style_explicit_scale_fp32_output = DeviceTensorFp32::Create({kTokens, kProjectedSize});
  auto torch_style_bf16_quant_output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  auto torch_style_bf16_quant_explicit_scale_output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  auto runtime_native_output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  auto cutlass_tiled_fp32_output = DeviceTensorFp32::Create({kTokens, kProjectedSize});
  auto cutlass_tiled_bf16_output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  if (handle == nullptr || !handle->valid() ||
      activations == nullptr || !activations->valid() ||
      activations_bf16 == nullptr || !activations_bf16->valid() ||
      weights == nullptr || !weights->valid() ||
      quantized_activations == nullptr || !quantized_activations->valid() ||
      quantized_activations_bf16 == nullptr || !quantized_activations_bf16->valid() ||
      weight_scale_device == nullptr || !weight_scale_device->valid() ||
      input_scale_device == nullptr || !input_scale_device->valid() ||
      row_major_output == nullptr || !row_major_output->valid() ||
      torch_style_output == nullptr || !torch_style_output->valid() ||
      torch_style_explicit_scale_output == nullptr || !torch_style_explicit_scale_output->valid() ||
      torch_style_explicit_scale_fp32_output == nullptr || !torch_style_explicit_scale_fp32_output->valid() ||
      torch_style_bf16_quant_output == nullptr || !torch_style_bf16_quant_output->valid() ||
      torch_style_bf16_quant_explicit_scale_output == nullptr ||
      !torch_style_bf16_quant_explicit_scale_output->valid() ||
      runtime_native_output == nullptr || !runtime_native_output->valid() ||
      cutlass_tiled_fp32_output == nullptr || !cutlass_tiled_fp32_output->valid() ||
      cutlass_tiled_bf16_output == nullptr || !cutlass_tiled_bf16_output->valid() ||
      !activations->CopyFromHost(norm_output.data(), norm_output.size()) ||
      !UploadPackedFloatToDeviceBf16(
          norm_output.data(),
          norm_output.size(),
          PackedFloatStorage::kFp32,
          1.0f,
          activations_bf16->data()) ||
      !weights->CopyFromHost(packed_weight.data(), packed_weight.size()) ||
      !weight_scale_device->CopyFromHost(weight_scale.data(), 1) ||
      !input_scale_device->CopyFromHost(input_scale.data(), 1) ||
      !QuantizeDeviceFp32ToFp8E4M3(
          activations->data(),
          activations->numel(),
          input_scale[0],
          quantized_activations->data(),
          nullptr) ||
      !QuantizeBf16ToFp8E4M3(
          *activations_bf16,
          input_scale[0],
          quantized_activations_bf16.get(),
          nullptr)) {
    std::cerr << "FAIL: contract probe setup failed\n";
    return 1;
  }

  ScaledFp8LinearConfig runtime_config;
  runtime_config.output_rows = kProjectedSize;
  runtime_config.input_cols = kHiddenSize;
  runtime_config.packed_weight_data = packed_weight.data();
  runtime_config.packed_weight_nbytes = packed_weight.size();
  runtime_config.tensor_name = "backbone.layers.0.mixer.in_proj.weight";
  runtime_config.weight_scale = weight_scale[0];
  runtime_config.input_scale = input_scale[0];
  auto runtime_native_op = ScaledFp8LinearOp::Create(runtime_config);
  if (!runtime_native_op || !runtime_native_op->valid()) {
    std::cerr << "FAIL: could not create runtime native scaled FP8 op\n";
    return 1;
  }

  const auto row_major_plan =
      BuildPlanForContract(descriptor, kTokens, CublasLtContract::kRowMajorA_N_RowMajorB_T);
  if (!row_major_plan.has_value()) {
    return 1;
  }

  std::vector<float> row_major_host;
  std::vector<float> torch_style_host;
  std::vector<float> torch_style_explicit_scale_host;
  std::vector<float> torch_style_explicit_scale_fp32_host;
  std::vector<float> torch_style_bf16_quant_host;
  std::vector<float> torch_style_bf16_quant_explicit_scale_host;
  std::vector<float> runtime_native_host;
  std::vector<float> cutlass_tiled_host;
  GemmHeuristicCache heuristic_cache;
  bool cutlass_tiled_ok = false;
  if (CutlassFp8DenseGemmAvailable()) {
    cudaMemset(
        cutlass_tiled_fp32_output->data(),
        0,
        cutlass_tiled_fp32_output->numel() * sizeof(float));
    cutlass_tiled_ok = true;
    for (std::size_t row = 0; row < kTokens; row += 16) {
      const std::size_t tile_rows = std::min<std::size_t>(16, kTokens - row);
      if (!RunCutlassFp8DenseGemm(
              static_cast<int>(tile_rows),
              static_cast<int>(kProjectedSize),
              static_cast<int>(kHiddenSize),
              quantized_activations_bf16->data() + (row * kHiddenSize),
              weights->data(),
              input_scale[0] * weight_scale[0],
              cutlass_tiled_fp32_output->data() + (row * kProjectedSize),
              nullptr)) {
        cutlass_tiled_ok = false;
        break;
      }
    }
    if (cutlass_tiled_ok) {
      cutlass_tiled_ok = ConvertDeviceFp32ToBf16(
          cutlass_tiled_fp32_output->data(),
          cutlass_tiled_fp32_output->numel(),
          cutlass_tiled_bf16_output->data(),
          nullptr);
    }
    if (cutlass_tiled_ok) {
      cutlass_tiled_ok = CopyBf16TensorToHost(*cutlass_tiled_bf16_output, &cutlass_tiled_host);
    }
  }
  if (!RunContract(
          *handle,
          *row_major_plan,
          *weights,
          *quantized_activations,
          *weight_scale_device,
          *input_scale_device,
          row_major_output.get(),
          &row_major_host) ||
      !RunTorchStyleContract(
          *handle,
          *weights,
          *quantized_activations,
          *weight_scale_device,
          *input_scale_device,
          false,
          torch_style_output.get(),
          &torch_style_host) ||
      !RunTorchStyleContract(
          *handle,
          *weights,
          *quantized_activations,
          *weight_scale_device,
          *input_scale_device,
          true,
          torch_style_explicit_scale_output.get(),
          &torch_style_explicit_scale_host) ||
      !RunTorchStyleContractFp32(
          *handle,
          *weights,
          *quantized_activations,
          *weight_scale_device,
          *input_scale_device,
          true,
          torch_style_explicit_scale_fp32_output.get(),
          &torch_style_explicit_scale_fp32_host) ||
      !RunTorchStyleContract(
          *handle,
          *weights,
          *quantized_activations_bf16,
          *weight_scale_device,
          *input_scale_device,
          false,
          torch_style_bf16_quant_output.get(),
          &torch_style_bf16_quant_host) ||
      !RunTorchStyleContract(
          *handle,
          *weights,
          *quantized_activations_bf16,
          *weight_scale_device,
          *input_scale_device,
          true,
          torch_style_bf16_quant_explicit_scale_output.get(),
          &torch_style_bf16_quant_explicit_scale_host) ||
      !runtime_native_op->Run(
          *handle,
          &heuristic_cache,
          *activations_bf16,
          runtime_native_output.get(),
          nullptr) ||
      !CopyBf16TensorToHost(*runtime_native_output, &runtime_native_host)) {
    std::cerr << "FAIL: contract probe execution failed\n";
    return 1;
  }

  const float row_major_mad = MaxAbsDiff(row_major_host, expected_projected);
  const float row_major_rel_l2 = RelL2(row_major_host, expected_projected);
  const float torch_style_mad = MaxAbsDiff(torch_style_host, expected_projected);
  const float torch_style_rel_l2 = RelL2(torch_style_host, expected_projected);
  const float torch_style_explicit_scale_mad =
      MaxAbsDiff(torch_style_explicit_scale_host, expected_projected);
  const float torch_style_explicit_scale_rel_l2 =
      RelL2(torch_style_explicit_scale_host, expected_projected);
  const float torch_style_explicit_scale_fp32_mad =
      MaxAbsDiff(torch_style_explicit_scale_fp32_host, expected_projected);
  const float torch_style_explicit_scale_fp32_rel_l2 =
      RelL2(torch_style_explicit_scale_fp32_host, expected_projected);
  const float torch_style_bf16_quant_mad =
      MaxAbsDiff(torch_style_bf16_quant_host, expected_projected);
  const float torch_style_bf16_quant_rel_l2 =
      RelL2(torch_style_bf16_quant_host, expected_projected);
  const float torch_style_bf16_quant_explicit_scale_mad =
      MaxAbsDiff(torch_style_bf16_quant_explicit_scale_host, expected_projected);
  const float torch_style_bf16_quant_explicit_scale_rel_l2 =
      RelL2(torch_style_bf16_quant_explicit_scale_host, expected_projected);
  const float runtime_native_mad = MaxAbsDiff(runtime_native_host, expected_projected);
  const float runtime_native_rel_l2 = RelL2(runtime_native_host, expected_projected);
  const float cutlass_tiled_mad =
      cutlass_tiled_ok ? MaxAbsDiff(cutlass_tiled_host, expected_projected)
                       : std::numeric_limits<float>::infinity();
  const float cutlass_tiled_rel_l2 =
      cutlass_tiled_ok ? RelL2(cutlass_tiled_host, expected_projected)
                       : std::numeric_limits<float>::infinity();

  std::cerr << "in_proj_fp8_contract_probe: row_major mad=" << row_major_mad
            << " rel_l2=" << row_major_rel_l2 << "\n";
  std::cerr << "in_proj_fp8_contract_probe: torch_style mad=" << torch_style_mad
            << " rel_l2=" << torch_style_rel_l2 << "\n";
  std::cerr << "in_proj_fp8_contract_probe: torch_style_explicit_scalar_scale_mode mad="
            << torch_style_explicit_scale_mad
            << " rel_l2=" << torch_style_explicit_scale_rel_l2 << "\n";
  std::cerr << "in_proj_fp8_contract_probe: torch_style_explicit_scalar_scale_mode_fp32 mad="
            << torch_style_explicit_scale_fp32_mad
            << " rel_l2=" << torch_style_explicit_scale_fp32_rel_l2 << "\n";
  std::cerr << "in_proj_fp8_contract_probe: torch_style_bf16_quant mad="
            << torch_style_bf16_quant_mad
            << " rel_l2=" << torch_style_bf16_quant_rel_l2 << "\n";
  std::cerr << "in_proj_fp8_contract_probe: torch_style_bf16_quant_explicit_scalar_scale_mode mad="
            << torch_style_bf16_quant_explicit_scale_mad
            << " rel_l2=" << torch_style_bf16_quant_explicit_scale_rel_l2 << "\n";
  std::cerr << "in_proj_fp8_contract_probe: runtime_native_bf16 mad="
            << runtime_native_mad
            << " rel_l2=" << runtime_native_rel_l2 << "\n";
  if (cutlass_tiled_ok) {
    std::cerr << "in_proj_fp8_contract_probe: cutlass_tiled_bf16 mad="
              << cutlass_tiled_mad
              << " rel_l2=" << cutlass_tiled_rel_l2 << "\n";
  } else {
    std::cerr << "in_proj_fp8_contract_probe: cutlass_tiled_bf16 unavailable\n";
  }

  return 0;
}
