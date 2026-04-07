#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime.h>

#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"

namespace nemotron {

struct Nvfp4PackedMatrixDeviceView {
  const std::uint8_t* packed_data = nullptr;
  std::size_t packed_nbytes = 0;
  const std::uint8_t* block_scales_data = nullptr;
  std::size_t block_scales_nbytes = 0;
  const float* tensor_scale_data = nullptr;
  std::size_t tensor_scale_nbytes = 0;
  std::size_t rows = 0;
  std::size_t cols = 0;

  bool valid() const;
};

Nvfp4PackedMatrixDeviceView MakeNvfp4PackedMatrixDeviceView(const DeviceNvfp4Weight& matrix);
Nvfp4PackedMatrixDeviceView MakeNvfp4PackedMatrixDeviceView(const DeviceNvfp4Matrix& matrix);

bool GroupedNvfp4PointerArrayAvailable();
void DisableGroupedNvfp4PointerArray();

struct Nvfp4RowMajorDeviceStats {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t workspace_bytes = 0;
  int heuristic_count = 0;
};

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const Nvfp4PackedMatrixDeviceView& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32AccumPointerArrayToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    std::size_t batch_count,
    const void* const* activations_packed_device_array,
    const void* const* activations_block_scales_device_array,
    const void* const* weights_packed_device_array,
    const void* const* weights_block_scales_device_array,
    void* const* output_device_array);

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32SourceToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options = {},
    cudaStream_t stream = nullptr);

}  // namespace nemotron
