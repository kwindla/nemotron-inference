#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

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
    DeviceTensorFp32* output);

std::optional<Nvfp4RowMajorDeviceStats> RunNvfp4RowMajorFp32SourceToDevice(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorFp32& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options = {});

}  // namespace nemotron
