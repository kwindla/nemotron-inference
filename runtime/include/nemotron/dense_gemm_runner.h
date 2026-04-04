#pragma once

#include <cstddef>
#include <optional>
#include <vector>

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

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightBf16& weights,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output);

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorBf16ToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output);

std::optional<DenseRowMajorHostResult> RunDenseRowMajorFp32(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const float* host_activations,
    std::size_t activation_rows);

}  // namespace nemotron
