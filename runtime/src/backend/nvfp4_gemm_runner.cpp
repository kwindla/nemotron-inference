#include "nemotron/nvfp4_gemm_runner.h"

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>

namespace nemotron {
namespace {

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
  std::cerr << "nvfp4_gemm_runner: " << op
            << " failed"
            << " status=" << CublasStatusName(status)
            << "(" << static_cast<int>(status) << ")"
            << " M=" << m
            << " N=" << n
            << " K=" << k
            << "\n";
}

void LogHeuristicFailure(std::size_t m, std::size_t n, std::size_t k) {
  if (!DebugEnabled()) {
    return;
  }
  std::cerr << "nvfp4_gemm_runner: cublasLtMatmulAlgoGetHeuristic returned 0 results"
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

cublasLtMatmulMatrixScale_t ToCublasScaleMode(CublasLtScaleMode scale_mode) {
  switch (scale_mode) {
    case CublasLtScaleMode::kNone:
      return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    case CublasLtScaleMode::kVec16UE4M3:
      return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  }
  return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
}

std::optional<float> ReadTensorScaleHostFallback(const float* device_ptr) {
  if (device_ptr == nullptr) {
    return std::nullopt;
  }
  float host_value = 0.0f;
  if (!CheckCuda(cudaMemcpy(&host_value, device_ptr, sizeof(host_value), cudaMemcpyDeviceToHost))) {
    return std::nullopt;
  }
  return host_value;
}

struct CachedNvfp4MatmulKey {
  std::uintptr_t handle = 0;
  int device = -1;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::size_t lda = 0;
  std::size_t ldb = 0;
  std::size_t ldc = 0;
  std::size_t workspace_bytes = 0;
  CublasLtTransform transform_a = CublasLtTransform::kNone;
  CublasLtTransform transform_b = CublasLtTransform::kTranspose;
  CublasLtMatrixOrder order_a = CublasLtMatrixOrder::kRowMajor;
  CublasLtMatrixOrder order_b = CublasLtMatrixOrder::kRowMajor;
  CublasLtMatrixOrder order_c = CublasLtMatrixOrder::kRowMajor;
  CublasLtScaleMode scale_mode = CublasLtScaleMode::kNone;

  bool operator==(const CachedNvfp4MatmulKey& other) const {
    return handle == other.handle &&
           device == other.device &&
           m == other.m &&
           n == other.n &&
           k == other.k &&
           lda == other.lda &&
           ldb == other.ldb &&
           ldc == other.ldc &&
           workspace_bytes == other.workspace_bytes &&
           transform_a == other.transform_a &&
           transform_b == other.transform_b &&
           order_a == other.order_a &&
           order_b == other.order_b &&
           order_c == other.order_c &&
           scale_mode == other.scale_mode;
  }
};

struct CachedNvfp4MatmulKeyHash {
  std::size_t operator()(const CachedNvfp4MatmulKey& key) const {
    std::size_t hash = 0;
    const auto mix = [&](std::size_t value) {
      hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    };
    mix(static_cast<std::size_t>(key.handle));
    mix(static_cast<std::size_t>(key.device));
    mix(key.m);
    mix(key.n);
    mix(key.k);
    mix(key.lda);
    mix(key.ldb);
    mix(key.ldc);
    mix(key.workspace_bytes);
    mix(static_cast<std::size_t>(key.transform_a));
    mix(static_cast<std::size_t>(key.transform_b));
    mix(static_cast<std::size_t>(key.order_a));
    mix(static_cast<std::size_t>(key.order_b));
    mix(static_cast<std::size_t>(key.order_c));
    mix(static_cast<std::size_t>(key.scale_mode));
    return hash;
  }
};

struct CachedNvfp4MatmulResources {
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int heuristic_count = 0;
  bool device_pointer_mode_supported = false;
  float* device_alpha = nullptr;
  float* device_beta = nullptr;

  ~CachedNvfp4MatmulResources() {
    if (device_beta != nullptr) {
      cudaFree(device_beta);
    }
    if (device_alpha != nullptr) {
      cudaFree(device_alpha);
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
  }
};

CachedNvfp4MatmulKey MakeCachedNvfp4MatmulKey(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    std::size_t m,
    std::size_t n,
    std::size_t k) {
  int current_device = -1;
  if (!CheckCuda(cudaGetDevice(&current_device))) {
    current_device = -1;
  }
  return CachedNvfp4MatmulKey{
      reinterpret_cast<std::uintptr_t>(handle.handle()),
      current_device,
      m,
      n,
      k,
      plan.lda,
      plan.ldb,
      plan.ldc,
      handle.workspace_bytes(),
      plan.transform_a,
      plan.transform_b,
      plan.order_a,
      plan.order_b,
      plan.order_c,
      plan.scale_mode,
  };
}

bool InitializeCachedNvfp4MatmulResources(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const void* activation_block_scales,
    const void* weight_block_scales,
    CachedNvfp4MatmulResources* resources) {
  if (resources == nullptr) {
    return false;
  }

  const auto check_cublas = [&](cublasStatus_t status, const char* op) {
    if (CheckCublas(status)) {
      return true;
    }
    LogCublasFailure(op, status, m, n, k);
    return false;
  };

  const cublasOperation_t trans_a = ToCublasOp(plan.transform_a);
  const cublasOperation_t trans_b = ToCublasOp(plan.transform_b);
  const cublasLtMatmulMatrixScale_t scale_mode = ToCublasScaleMode(plan.scale_mode);
  if (!check_cublas(
          cublasLtMatmulDescCreate(&resources->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "cublasLtMatmulDescCreate")) {
    return false;
  }
  if (!check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_TRANSA,
              &trans_a,
              sizeof(trans_a)),
          "cublasLtMatmulDescSetAttribute(TRANSA)") ||
      !check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_TRANSB,
              &trans_b,
              sizeof(trans_b)),
          "cublasLtMatmulDescSetAttribute(TRANSB)") ||
      !check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
              &scale_mode,
              sizeof(scale_mode)),
          "cublasLtMatmulDescSetAttribute(A_SCALE_MODE)") ||
      !check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
              &scale_mode,
              sizeof(scale_mode)),
          "cublasLtMatmulDescSetAttribute(B_SCALE_MODE)") ||
      !check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
              &activation_block_scales,
              sizeof(activation_block_scales)),
          "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER)") ||
      !check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
              &weight_block_scales,
              sizeof(weight_block_scales)),
          "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER)")) {
    return false;
  }

  if (!check_cublas(
          cublasLtMatrixLayoutCreate(
              &resources->a_desc,
              CUDA_R_4F_E2M1,
              static_cast<std::uint64_t>(m),
              static_cast<std::uint64_t>(k),
              static_cast<std::int64_t>(plan.lda)),
          "cublasLtMatrixLayoutCreate(A)") ||
      !check_cublas(
          cublasLtMatrixLayoutCreate(
              &resources->b_desc,
              CUDA_R_4F_E2M1,
              static_cast<std::uint64_t>(n),
              static_cast<std::uint64_t>(k),
              static_cast<std::int64_t>(plan.ldb)),
          "cublasLtMatrixLayoutCreate(B)") ||
      !check_cublas(
          cublasLtMatrixLayoutCreate(
              &resources->c_desc,
              CUDA_R_32F,
              static_cast<std::uint64_t>(m),
              static_cast<std::uint64_t>(n),
              static_cast<std::int64_t>(plan.ldc)),
          "cublasLtMatrixLayoutCreate(C)")) {
    return false;
  }

  const cublasLtOrder_t order_a = ToCublasOrder(plan.order_a);
  const cublasLtOrder_t order_b = ToCublasOrder(plan.order_b);
  const cublasLtOrder_t order_c = ToCublasOrder(plan.order_c);
  if (!check_cublas(
          cublasLtMatrixLayoutSetAttribute(
              resources->a_desc,
              CUBLASLT_MATRIX_LAYOUT_ORDER,
              &order_a,
              sizeof(order_a)),
          "cublasLtMatrixLayoutSetAttribute(A_ORDER)") ||
      !check_cublas(
          cublasLtMatrixLayoutSetAttribute(
              resources->b_desc,
              CUBLASLT_MATRIX_LAYOUT_ORDER,
              &order_b,
              sizeof(order_b)),
          "cublasLtMatrixLayoutSetAttribute(B_ORDER)") ||
      !check_cublas(
          cublasLtMatrixLayoutSetAttribute(
              resources->c_desc,
              CUBLASLT_MATRIX_LAYOUT_ORDER,
              &order_c,
              sizeof(order_c)),
          "cublasLtMatrixLayoutSetAttribute(C_ORDER)")) {
    return false;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  const std::size_t max_workspace = handle.workspace_bytes();
  const bool preference_ok =
      check_cublas(
          cublasLtMatmulPreferenceCreate(&preference),
          "cublasLtMatmulPreferenceCreate") &&
      check_cublas(
          cublasLtMatmulPreferenceSetAttribute(
              preference,
              CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
              &max_workspace,
              sizeof(max_workspace)),
          "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)") &&
      check_cublas(
          cublasLtMatmulAlgoGetHeuristic(
              handle.handle(),
              resources->op_desc,
              resources->a_desc,
              resources->b_desc,
              resources->c_desc,
              resources->c_desc,
              preference,
              1,
              &resources->heuristic,
              &resources->heuristic_count),
          "cublasLtMatmulAlgoGetHeuristic") &&
      resources->heuristic_count > 0;
  if (preference != nullptr) {
    cublasLtMatmulPreferenceDestroy(preference);
  }
  if (!preference_ok) {
    if (resources->heuristic_count == 0) {
      LogHeuristicFailure(m, n, k);
    }
    return false;
  }

  std::uint32_t pointer_mode_mask = 0;
  std::size_t size_written = 0;
  if (CheckCublas(
          cublasLtMatmulAlgoCapGetAttribute(
              &resources->heuristic.algo,
              CUBLASLT_ALGO_CAP_POINTER_MODE_MASK,
              &pointer_mode_mask,
              sizeof(pointer_mode_mask),
              &size_written)) &&
      size_written == sizeof(pointer_mode_mask) &&
      (pointer_mode_mask & CUBLASLT_POINTER_MODE_MASK_DEVICE) != 0u) {
    resources->device_pointer_mode_supported = true;
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&resources->device_alpha), sizeof(float))) ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&resources->device_beta), sizeof(float))) ||
        !CheckCuda(cudaMemset(resources->device_beta, 0, sizeof(float)))) {
      return false;
    }
  }

  return true;
}

CachedNvfp4MatmulResources* GetCachedNvfp4MatmulResources(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const void* activation_block_scales,
    const void* weight_block_scales) {
  thread_local std::unordered_map<
      CachedNvfp4MatmulKey,
      std::unique_ptr<CachedNvfp4MatmulResources>,
      CachedNvfp4MatmulKeyHash>
      cache;

  const CachedNvfp4MatmulKey key = MakeCachedNvfp4MatmulKey(handle, plan, m, n, k);
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second.get();
  }

  auto resources = std::make_unique<CachedNvfp4MatmulResources>();
  if (!InitializeCachedNvfp4MatmulResources(
          handle,
          plan,
          m,
          n,
          k,
          activation_block_scales,
          weight_block_scales,
          resources.get())) {
    return nullptr;
  }
  auto* resources_ptr = resources.get();
  cache.emplace(key, std::move(resources));
  return resources_ptr;
}

struct PreparedNvfp4MatmulCall {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  CachedNvfp4MatmulResources* resources = nullptr;
  const std::uint8_t* activations_packed_data = nullptr;
  const std::uint8_t* weights_packed_data = nullptr;
  float* output_data = nullptr;
};

std::optional<PreparedNvfp4MatmulCall> PrepareNvfp4MatmulCall(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const Nvfp4PackedMatrixDeviceView& activations,
    const Nvfp4PackedMatrixDeviceView& weights,
    DeviceTensorFp32* output) {
  if (!handle.valid() ||
      !activations.valid() ||
      !weights.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtNvfp4BlockScaled ||
      plan.scale_mode != CublasLtScaleMode::kVec16UE4M3 ||
      plan.contract != CublasLtContract::kRowMajorA_N_RowMajorB_T) {
    return std::nullopt;
  }

  if (output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.rows;
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.cols != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      weights.rows != n ||
      weights.cols != k) {
    return std::nullopt;
  }

  const auto check_cublas = [&](cublasStatus_t status, const char* op) {
    if (CheckCublas(status)) {
      return true;
    }
    LogCublasFailure(op, status, m, n, k);
    return false;
  };

  const void* activation_block_scales = activations.block_scales_data;
  const void* weight_block_scales = weights.block_scales_data;
  CachedNvfp4MatmulResources* const resources = GetCachedNvfp4MatmulResources(
      handle,
      plan,
      m,
      n,
      k,
      activation_block_scales,
      weight_block_scales);
  if (resources == nullptr) {
    return std::nullopt;
  }

  // Cached resources include a mutable op descriptor and reusable device alpha
  // buffer. Wait for prior launches that may still be consuming them before we
  // retarget scale pointers or rewrite alpha for the next matmul.
  if (!CheckCuda(cudaStreamSynchronize(nullptr))) {
    return std::nullopt;
  }

  if (!check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
              &activation_block_scales,
              sizeof(activation_block_scales)),
          "cublasLtMatmulDescSetAttribute(A_SCALE_POINTER)") ||
      !check_cublas(
          cublasLtMatmulDescSetAttribute(
              resources->op_desc,
              CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
              &weight_block_scales,
              sizeof(weight_block_scales)),
          "cublasLtMatmulDescSetAttribute(B_SCALE_POINTER)")) {
    return std::nullopt;
  }

  return PreparedNvfp4MatmulCall{
      m,
      n,
      k,
      resources,
      activations.packed_data,
      weights.packed_data,
      output->data(),
  };
}

std::optional<Nvfp4RowMajorDeviceStats> ExecutePreparedNvfp4Matmul(
    CublasLtHandle& handle,
    const PreparedNvfp4MatmulCall& prepared,
    const void* alpha,
    const void* beta,
    cublasLtPointerMode_t pointer_mode) {
  const auto check_cublas = [&](cublasStatus_t status, const char* op) {
    if (CheckCublas(status)) {
      return true;
    }
    LogCublasFailure(op, status, prepared.m, prepared.n, prepared.k);
    return false;
  };

  if (!check_cublas(
          cublasLtMatmulDescSetAttribute(
              prepared.resources->op_desc,
              CUBLASLT_MATMUL_DESC_POINTER_MODE,
              &pointer_mode,
              sizeof(pointer_mode)),
          "cublasLtMatmulDescSetAttribute(POINTER_MODE)") ||
      !check_cublas(
          cublasLtMatmul(
              handle.handle(),
              prepared.resources->op_desc,
              alpha,
              prepared.activations_packed_data,
              prepared.resources->a_desc,
              prepared.weights_packed_data,
              prepared.resources->b_desc,
              beta,
              prepared.output_data,
              prepared.resources->c_desc,
              prepared.output_data,
              prepared.resources->c_desc,
              &prepared.resources->heuristic.algo,
              handle.workspace(),
              handle.workspace_bytes(),
              nullptr),
          "cublasLtMatmul")) {
    return std::nullopt;
  }

  return Nvfp4RowMajorDeviceStats{
      prepared.m,
      prepared.n,
      prepared.resources->heuristic.workspaceSize,
      prepared.resources->heuristic_count,
  };
}

}  // namespace

bool Nvfp4PackedMatrixDeviceView::valid() const {
  return packed_data != nullptr &&
         packed_nbytes > 0 &&
         block_scales_data != nullptr &&
         block_scales_nbytes > 0 &&
         tensor_scale_data != nullptr &&
         tensor_scale_nbytes >= sizeof(float) &&
         rows > 0 &&
         cols > 0;
}

Nvfp4PackedMatrixDeviceView MakeNvfp4PackedMatrixDeviceView(const DeviceNvfp4Weight& matrix) {
  return Nvfp4PackedMatrixDeviceView{
      matrix.packed_data(),
      matrix.packed_nbytes(),
      matrix.matmul_block_scales_data(),
      matrix.matmul_block_scales_nbytes(),
      reinterpret_cast<const float*>(matrix.tensor_scale_data()),
      matrix.tensor_scale_nbytes(),
      matrix.output_rows(),
      matrix.input_cols(),
  };
}

Nvfp4PackedMatrixDeviceView MakeNvfp4PackedMatrixDeviceView(const DeviceNvfp4Matrix& matrix) {
  return Nvfp4PackedMatrixDeviceView{
      matrix.packed_data(),
      matrix.packed_nbytes(),
      matrix.matmul_block_scales_data(),
      matrix.matmul_block_scales_nbytes(),
      matrix.device_tensor_scale_ptr(),
      matrix.tensor_scale_nbytes(),
      matrix.rows(),
      matrix.cols(),
  };
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const Nvfp4PackedMatrixDeviceView& activations,
    float activation_tensor_scale_host,
    const Nvfp4PackedMatrixDeviceView& weights,
    float weight_tensor_scale_host,
    DeviceTensorFp32* output) {
  if (!std::isfinite(activation_tensor_scale_host) ||
      activation_tensor_scale_host <= 0.0f ||
      !std::isfinite(weight_tensor_scale_host) ||
      weight_tensor_scale_host <= 0.0f) {
    return std::nullopt;
  }

  if (!handle.valid() ||
      !activations.valid() ||
      !weights.valid() ||
      output == nullptr ||
      !output->valid() ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtNvfp4BlockScaled ||
      plan.scale_mode != CublasLtScaleMode::kVec16UE4M3 ||
      plan.contract != CublasLtContract::kRowMajorA_N_RowMajorB_T) {
    return std::nullopt;
  }
  const auto prepared = PrepareNvfp4MatmulCall(handle, plan, activations, weights, output);
  if (!prepared.has_value()) {
    return std::nullopt;
  }

  const float alpha = activation_tensor_scale_host * weight_tensor_scale_host;
  const float beta = 0.0f;
  return ExecutePreparedNvfp4Matmul(
      handle,
      *prepared,
      &alpha,
      &beta,
      CUBLASLT_POINTER_MODE_HOST);
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const Nvfp4PackedMatrixDeviceView& activations,
    const float* activation_tensor_scale_device,
    const Nvfp4PackedMatrixDeviceView& weights,
    const float* weight_tensor_scale_device,
    DeviceTensorFp32* output,
    bool allow_tensor_scale_host_fallback) {
  const auto prepared = PrepareNvfp4MatmulCall(handle, plan, activations, weights, output);
  if (!prepared.has_value()) {
    return std::nullopt;
  }

  if (activation_tensor_scale_device != nullptr &&
      weight_tensor_scale_device != nullptr &&
      prepared->resources->device_pointer_mode_supported &&
      prepared->resources->device_alpha != nullptr &&
      prepared->resources->device_beta != nullptr &&
      MultiplyDeviceTensorScales(
          activation_tensor_scale_device,
          weight_tensor_scale_device,
          prepared->resources->device_alpha)) {
    return ExecutePreparedNvfp4Matmul(
        handle,
        *prepared,
        prepared->resources->device_alpha,
        prepared->resources->device_beta,
        CUBLASLT_POINTER_MODE_DEVICE);
  }

  if (!allow_tensor_scale_host_fallback) {
    return std::nullopt;
  }

  const auto activation_tensor_scale_host =
      ReadTensorScaleHostFallback(activation_tensor_scale_device);
  const auto weight_tensor_scale_host =
      ReadTensorScaleHostFallback(weight_tensor_scale_device);
  if (!activation_tensor_scale_host.has_value() ||
      !weight_tensor_scale_host.has_value()) {
    return std::nullopt;
  }

  const float alpha = *activation_tensor_scale_host * *weight_tensor_scale_host;
  const float beta = 0.0f;
  return ExecutePreparedNvfp4Matmul(
      handle,
      *prepared,
      &alpha,
      &beta,
      CUBLASLT_POINTER_MODE_HOST);
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const Nvfp4PackedMatrixDeviceView& activations,
    const Nvfp4PackedMatrixDeviceView& weights,
    DeviceTensorFp32* output) {
  return RunNvfp4RowMajorFp32AccumToDevice(
      handle,
      plan,
      activations,
      activations.tensor_scale_data,
      weights,
      weights.tensor_scale_data,
      output);
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const Nvfp4PackedMatrixDeviceView& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output) {
  const Nvfp4PackedMatrixDeviceView weight_view = MakeNvfp4PackedMatrixDeviceView(weights);
  return RunNvfp4RowMajorFp32AccumToDevice(
      handle,
      plan,
      activations,
      activations.tensor_scale_data,
      weight_view,
      weight_view.tensor_scale_data,
      output);
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32SourceToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    const Nvfp4PackedMatrixDeviceView& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options,
    bool allow_tensor_scale_host_fallback) {
  if (!weights.valid()) {
    return std::nullopt;
  }
  auto packed = PackDeviceRowMajorFp32ToNvfp4(activations, pack_options);
  if (!packed || !packed->valid()) {
    return std::nullopt;
  }
  const auto stats = RunNvfp4RowMajorFp32AccumToDevice(
      handle,
      plan,
      MakeNvfp4PackedMatrixDeviceView(*packed),
      packed->device_tensor_scale_ptr(),
      weights,
      weights.tensor_scale_data,
      output,
      allow_tensor_scale_host_fallback);
  if (!stats.has_value()) {
    return std::nullopt;
  }

  // The packed activation buffer is owned by this helper. Keep it alive until
  // the queued matmul completes so cuBLASLt does not read freed device memory.
  if (!CheckCuda(cudaStreamSynchronize(nullptr))) {
    return std::nullopt;
  }
  return stats;
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32SourceToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options,
    bool allow_tensor_scale_host_fallback) {
  return RunNvfp4RowMajorFp32SourceToDevice(
      handle,
      plan,
      activations,
      MakeNvfp4PackedMatrixDeviceView(weights),
      output,
      pack_options,
      allow_tensor_scale_host_fallback);
}

}  // namespace nemotron
