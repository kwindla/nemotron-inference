#include "nemotron/nvfp4_gemm_runner.h"

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <optional>

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

  ok &= output->FillZero();

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
            nullptr));
    ok &= CheckCuda(cudaDeviceSynchronize());
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

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32SourceToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options) {
  auto packed = PackDeviceRowMajorFp32ToNvfp4(activations, pack_options);
  if (!packed || !packed->valid()) {
    return std::nullopt;
  }
  return RunNvfp4RowMajorFp32AccumToDevice(
      handle,
      plan,
      MakeNvfp4PackedMatrixDeviceView(*packed),
      weights,
      output);
}

}  // namespace nemotron
