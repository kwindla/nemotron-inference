#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/dense_weight.h"
#include "nemotron/device_tensor.h"

namespace nemotron {

struct DenseRowMajorHostResult {
  std::vector<float> output;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t workspace_bytes = 0;
  int heuristic_count = 0;
};

struct DenseRowMajorDeviceStats {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t workspace_bytes = 0;
  int heuristic_count = 0;
};

struct CachedCublasLtMatmulState {
  CachedCublasLtMatmulState() = default;
  ~CachedCublasLtMatmulState();

  CachedCublasLtMatmulState(CachedCublasLtMatmulState&& other) noexcept;
  CachedCublasLtMatmulState& operator=(CachedCublasLtMatmulState&& other) noexcept;

  CachedCublasLtMatmulState(const CachedCublasLtMatmulState&) = delete;
  CachedCublasLtMatmulState& operator=(const CachedCublasLtMatmulState&) = delete;

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  cudaDataType_t activations_type = CUDA_R_32F;
  cudaDataType_t weights_type = CUDA_R_32F;
  cudaDataType_t output_type = CUDA_R_32F;
  const void* activations_scale_data = nullptr;
  const void* weights_scale_data = nullptr;
  float alpha_scale = 1.0f;
  bool fast_accum = false;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  int heuristic_count = 0;
};

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp8E4M3ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const float* weight_scale_device,
    const DeviceTensorFp32& activations,
    float input_scale,
    const float* input_scale_device,
    DeviceTensorFp32* output,
    const CachedCublasLtMatmulState* cached_matmul_state = nullptr,
    DeviceTensorFp8E4M3* activation_scratch = nullptr,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp8E4M3ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const float* weight_scale_device,
    const DeviceTensorFp8E4M3& activations,
    const float* input_scale_device,
    DeviceTensorFp32* output,
    const CachedCublasLtMatmulState* cached_matmul_state = nullptr,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp8E4M3ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp8E4M3& weights,
    const float* weight_scale_device,
    const DeviceTensorFp8E4M3& activations,
    const float* input_scale_device,
    DeviceTensorBf16* output,
    const CachedCublasLtMatmulState* cached_matmul_state = nullptr,
    cudaStream_t stream = nullptr);

std::unique_ptr<CachedCublasLtMatmulState> CreateDenseRowMajorFp8E4M3MatmulState(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const float* weight_scale_device,
    const float* input_scale_device,
    cudaDataType_t output_type = CUDA_R_32F);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ReferenceToDevice(
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

std::optional<DenseRowMajorHostResult> RunDenseRowMajorFp32(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const float* host_activations,
    std::size_t activation_rows);

}  // namespace nemotron
