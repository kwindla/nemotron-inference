#include "nemotron/nvfp4_gemm_runner.h"

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <atomic>

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool CheckCublas(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS;
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

bool ReadDeviceFloat(const float* device_ptr, float* host_value) {
  return device_ptr != nullptr &&
         host_value != nullptr &&
         CheckCuda(cudaMemcpy(host_value, device_ptr, sizeof(float), cudaMemcpyDeviceToHost));
}

bool GroupedNvfp4DebugEnabled() {
  static const bool enabled = std::getenv("NEMOTRON_GROUPED_NVFP4_DEBUG") != nullptr;
  return enabled;
}

std::atomic<bool>& GroupedNvfp4PointerArrayAvailableFlag() {
  static std::atomic<bool> available{true};
  return available;
}

void ReportGroupedNvfp4Failure(
    const char* stage,
    cublasStatus_t cublas_status,
    std::size_t batch_count,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    int returned_results) {
  if (!GroupedNvfp4DebugEnabled()) {
    return;
  }
  std::cerr << "nvfp4_grouped: failure at " << (stage != nullptr ? stage : "unknown")
            << " cublas_status=" << static_cast<int>(cublas_status)
            << " batch_count=" << batch_count
            << " m=" << m
            << " n=" << n
            << " k=" << k
            << " heuristic_results=" << returned_results
            << "\n";
}

}  // namespace

bool GroupedNvfp4PointerArrayAvailable() {
  return GroupedNvfp4PointerArrayAvailableFlag().load(std::memory_order_relaxed);
}

void DisableGroupedNvfp4PointerArray() {
  GroupedNvfp4PointerArrayAvailableFlag().store(false, std::memory_order_relaxed);
}

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
      reinterpret_cast<const float*>(matrix.tensor_scale_data()),
      matrix.tensor_scale_nbytes(),
      matrix.rows(),
      matrix.cols(),
  };
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const Nvfp4PackedMatrixDeviceView& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
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
      weights.output_rows() != n ||
      weights.input_cols() != k) {
    return std::nullopt;
  }

  float activation_tensor_scale = 0.0f;
  float weight_tensor_scale = 0.0f;
  if (!ReadDeviceFloat(activations.tensor_scale_data, &activation_tensor_scale) ||
      !ReadDeviceFloat(reinterpret_cast<const float*>(weights.tensor_scale_data()), &weight_tensor_scale)) {
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

  ok &= output->FillZero(stream);

  ok &= CheckCublas(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = ToCublasOp(plan.transform_a);
  const cublasOperation_t trans_b = ToCublasOp(plan.transform_b);
  const cublasLtMatmulMatrixScale_t scale_mode = ToCublasScaleMode(plan.scale_mode);
  ok &= CheckCublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_TRANSA,
          &trans_a,
          sizeof(trans_a)));
  ok &= CheckCublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_TRANSB,
          &trans_b,
          sizeof(trans_b)));
  ok &= CheckCublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
          &scale_mode,
          sizeof(scale_mode)));
  ok &= CheckCublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
          &scale_mode,
          sizeof(scale_mode)));

  const void* activation_block_scales = activations.block_scales_data;
  const void* weight_block_scales = weights.matmul_block_scales_data();
  ok &= CheckCublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
          &activation_block_scales,
          sizeof(activation_block_scales)));
  ok &= CheckCublas(
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
          &weight_block_scales,
          sizeof(weight_block_scales)));

  ok &= CheckCublas(
      cublasLtMatrixLayoutCreate(
          &a_desc,
          CUDA_R_4F_E2M1,
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.lda)));
  ok &= CheckCublas(
      cublasLtMatrixLayoutCreate(
          &b_desc,
          CUDA_R_4F_E2M1,
          static_cast<std::uint64_t>(n),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.ldb)));
  ok &= CheckCublas(
      cublasLtMatrixLayoutCreate(
          &c_desc,
          CUDA_R_32F,
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(n),
          static_cast<std::int64_t>(plan.ldc)));

  const cublasLtOrder_t order_a = ToCublasOrder(plan.order_a);
  const cublasLtOrder_t order_b = ToCublasOrder(plan.order_b);
  const cublasLtOrder_t order_c = ToCublasOrder(plan.order_c);
  ok &= CheckCublas(
      cublasLtMatrixLayoutSetAttribute(
          a_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_a,
          sizeof(order_a)));
  ok &= CheckCublas(
      cublasLtMatrixLayoutSetAttribute(
          b_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_b,
          sizeof(order_b)));
  ok &= CheckCublas(
      cublasLtMatrixLayoutSetAttribute(
          c_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_c,
          sizeof(order_c)));

  ok &= CheckCublas(cublasLtMatmulPreferenceCreate(&preference));
  const std::size_t max_workspace = handle.workspace_bytes();
  ok &= CheckCublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &max_workspace,
          sizeof(max_workspace)));

  if (ok) {
    ok &= CheckCublas(
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
    ok &= returned_results > 0;
  }

  if (ok) {
    const float alpha = activation_tensor_scale * weight_tensor_scale;
    const float beta = 0.0f;
    ok &= CheckCublas(
        cublasLtMatmul(
            handle.handle(),
            op_desc,
            &alpha,
            activations.packed_data,
            a_desc,
            weights.packed_data(),
            b_desc,
            &beta,
            output->data(),
            c_desc,
            output->data(),
            c_desc,
            &heuristic.algo,
            handle.workspace(),
            handle.workspace_bytes(),
            stream));
    ok &= CheckCuda(cudaGetLastError());
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

  return Nvfp4RowMajorDeviceStats{
      m,
      n,
      heuristic.workspaceSize,
      returned_results,
  };
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumPointerArrayToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    std::size_t batch_count,
    const void* const* activations_packed_device_array,
    const void* const* activations_block_scales_device_array,
    const void* const* weights_packed_device_array,
    const void* const* weights_block_scales_device_array,
    void* const* output_device_array) {
  if (!handle.valid() ||
      !GroupedNvfp4PointerArrayAvailable() ||
      batch_count == 0 ||
      activations_packed_device_array == nullptr ||
      activations_block_scales_device_array == nullptr ||
      weights_packed_device_array == nullptr ||
      weights_block_scales_device_array == nullptr ||
      output_device_array == nullptr ||
      plan.execution.backend_kind != GemmBackendKind::kCublasLtNvfp4BlockScaled ||
      plan.scale_mode != CublasLtScaleMode::kVec16UE4M3 ||
      plan.contract != CublasLtContract::kRowMajorA_N_RowMajorB_T) {
    return std::nullopt;
  }

  const std::size_t m = plan.execution.launch_plan.m;
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  bool ok = true;
  cublasStatus_t last_cublas_status = CUBLAS_STATUS_SUCCESS;
  const char* failure_stage = nullptr;

  const auto apply_cublas = [&](const char* stage, cublasStatus_t status) {
    failure_stage = stage;
    last_cublas_status = status;
    ok &= CheckCublas(status);
  };

  apply_cublas("matmul_desc_create", cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = ToCublasOp(plan.transform_a);
  const cublasOperation_t trans_b = ToCublasOp(plan.transform_b);
  const cublasLtMatmulMatrixScale_t scale_mode = ToCublasScaleMode(plan.scale_mode);
  const cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  apply_cublas(
      "set_transa",
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_TRANSA,
          &trans_a,
          sizeof(trans_a)));
  apply_cublas(
      "set_transb",
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_TRANSB,
          &trans_b,
          sizeof(trans_b)));
  apply_cublas(
      "set_a_scale_mode",
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
          &scale_mode,
          sizeof(scale_mode)));
  apply_cublas(
      "set_b_scale_mode",
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
          &scale_mode,
          sizeof(scale_mode)));
  apply_cublas(
      "set_pointer_mode",
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_POINTER_MODE,
          &pointer_mode,
          sizeof(pointer_mode)));

  const void* activation_block_scales = activations_block_scales_device_array;
  const void* weight_block_scales = weights_block_scales_device_array;
  apply_cublas(
      "set_a_scale_pointer",
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
          &activation_block_scales,
          sizeof(activation_block_scales)));
  apply_cublas(
      "set_b_scale_pointer",
      cublasLtMatmulDescSetAttribute(
          op_desc,
          CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
          &weight_block_scales,
          sizeof(weight_block_scales)));

  apply_cublas(
      "create_a_layout",
      cublasLtMatrixLayoutCreate(
          &a_desc,
          CUDA_R_4F_E2M1,
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.lda)));
  apply_cublas(
      "create_b_layout",
      cublasLtMatrixLayoutCreate(
          &b_desc,
          CUDA_R_4F_E2M1,
          static_cast<std::uint64_t>(n),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.ldb)));
  apply_cublas(
      "create_c_layout",
      cublasLtMatrixLayoutCreate(
          &c_desc,
          CUDA_R_32F,
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(n),
          static_cast<std::int64_t>(plan.ldc)));

  const cublasLtOrder_t order_a = ToCublasOrder(plan.order_a);
  const cublasLtOrder_t order_b = ToCublasOrder(plan.order_b);
  const cublasLtOrder_t order_c = ToCublasOrder(plan.order_c);
  const int batch_count_int = static_cast<int>(batch_count);
  const cublasLtBatchMode_t batch_mode = CUBLASLT_BATCH_MODE_POINTER_ARRAY;
  apply_cublas(
      "set_a_order",
      cublasLtMatrixLayoutSetAttribute(
          a_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_a,
          sizeof(order_a)));
  apply_cublas(
      "set_b_order",
      cublasLtMatrixLayoutSetAttribute(
          b_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_b,
          sizeof(order_b)));
  apply_cublas(
      "set_c_order",
      cublasLtMatrixLayoutSetAttribute(
          c_desc,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order_c,
          sizeof(order_c)));
  apply_cublas(
      "set_a_batch_count",
      cublasLtMatrixLayoutSetAttribute(
          a_desc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch_count_int,
          sizeof(batch_count_int)));
  apply_cublas(
      "set_b_batch_count",
      cublasLtMatrixLayoutSetAttribute(
          b_desc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch_count_int,
          sizeof(batch_count_int)));
  apply_cublas(
      "set_c_batch_count",
      cublasLtMatrixLayoutSetAttribute(
          c_desc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch_count_int,
          sizeof(batch_count_int)));
  apply_cublas(
      "set_a_batch_mode",
      cublasLtMatrixLayoutSetAttribute(
          a_desc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,
          &batch_mode,
          sizeof(batch_mode)));
  apply_cublas(
      "set_b_batch_mode",
      cublasLtMatrixLayoutSetAttribute(
          b_desc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,
          &batch_mode,
          sizeof(batch_mode)));
  apply_cublas(
      "set_c_batch_mode",
      cublasLtMatrixLayoutSetAttribute(
          c_desc,
          CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,
          &batch_mode,
          sizeof(batch_mode)));

  apply_cublas("preference_create", cublasLtMatmulPreferenceCreate(&preference));
  const std::size_t max_workspace = handle.workspace_bytes();
  apply_cublas(
      "set_max_workspace",
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &max_workspace,
          sizeof(max_workspace)));

  if (ok) {
    apply_cublas(
        "heuristic",
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
    ok &= returned_results > 0;
    if (!ok && returned_results <= 0) {
      failure_stage = "heuristic_no_results";
    }
  }

  if (ok) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const void* activations_array_ptr = activations_packed_device_array;
    const void* weights_array_ptr = weights_packed_device_array;
    const void* output_read_array_ptr = output_device_array;
    void* output_write_array_ptr = const_cast<void*>(output_read_array_ptr);
    apply_cublas(
        "matmul",
        cublasLtMatmul(
            handle.handle(),
            op_desc,
            &alpha,
            activations_array_ptr,
            a_desc,
            weights_array_ptr,
            b_desc,
            &beta,
            output_read_array_ptr,
            c_desc,
            output_write_array_ptr,
            c_desc,
            &heuristic.algo,
            handle.workspace(),
            handle.workspace_bytes(),
            nullptr));
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
    DisableGroupedNvfp4PointerArray();
    ReportGroupedNvfp4Failure(failure_stage, last_cublas_status, batch_count, m, n, k, returned_results);
    return std::nullopt;
  }

  return Nvfp4RowMajorDeviceStats{
      m,
      n,
      heuristic.workspaceSize,
      returned_results,
  };
}

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32SourceToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options,
    cudaStream_t stream) {
  auto packed = PackDeviceRowMajorFp32ToNvfp4(activations, pack_options);
  if (!packed || !packed->valid()) {
    return std::nullopt;
  }
  return RunNvfp4RowMajorFp32AccumToDevice(
      handle,
      plan,
      MakeNvfp4PackedMatrixDeviceView(*packed),
      weights,
      output,
      stream);
}

}  // namespace nemotron
