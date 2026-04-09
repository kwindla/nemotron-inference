#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "nemotron/device_tensor.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {

class DeviceMoeLaunchPlan;

class DeviceNvfp4Matrix {
 public:
  static std::unique_ptr<DeviceNvfp4Matrix> Create(
      std::size_t rows,
      std::size_t cols,
      Nvfp4ScaleLayout scale_layout = Nvfp4ScaleLayout::kSwizzled128x4);

  DeviceNvfp4Matrix(DeviceNvfp4Matrix&&) noexcept;
  DeviceNvfp4Matrix& operator=(DeviceNvfp4Matrix&&) noexcept;
  ~DeviceNvfp4Matrix();

  DeviceNvfp4Matrix(const DeviceNvfp4Matrix&) = delete;
  DeviceNvfp4Matrix& operator=(const DeviceNvfp4Matrix&) = delete;

  bool valid() const;
  std::size_t rows() const;
  std::size_t cols() const;
  std::size_t packed_nbytes() const;
  std::size_t block_scales_nbytes() const;
  std::size_t matmul_block_scales_nbytes() const;
  std::size_t tensor_scale_nbytes() const;
  float host_tensor_scale() const;
  const float* device_tensor_scale_ptr() const;
  const float* effective_device_tensor_scale_ptr(const Nvfp4PackOptions& options) const;
  const std::uint8_t* packed_data() const;
  const std::uint8_t* block_scales_data() const;
  const std::uint8_t* matmul_block_scales_data() const;
  const std::uint8_t* tensor_scale_data() const;
  const void* p5_tma_load_b_descriptors(const DeviceMoeLaunchPlan& launch_plan) const;
  Nvfp4ScaleLayout scale_layout() const;
  bool PackInto(
      const DeviceTensorFp32& source,
      const Nvfp4PackOptions& options = {});

  bool CopyPackedToHost(std::vector<std::uint8_t>* output) const;
  bool CopyBlockScalesToHost(std::vector<std::uint8_t>* output) const;
  bool CopyMatmulBlockScalesToHost(std::vector<std::uint8_t>* output) const;
  bool CopyTensorScaleToHost(float* output) const;

 private:
  struct Impl;

  friend std::unique_ptr<DeviceNvfp4Matrix> PackDeviceRowMajorFp32ToNvfp4(
      const DeviceTensorFp32& source,
      const Nvfp4PackOptions& options);

  explicit DeviceNvfp4Matrix(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<DeviceNvfp4Matrix> PackDeviceRowMajorFp32ToNvfp4(
    const DeviceTensorFp32& source,
    const Nvfp4PackOptions& options = {});

bool PackDeviceRowMajorFp32ToNvfp4PerExpert(
    const float* source,
    std::size_t rows,
    std::size_t cols,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    const float* expert_tensor_scales,
    DeviceNvfp4Matrix* output);

bool PackDeviceRowMajorBf16ToNvfp4PerExpert(
    const __nv_bfloat16* source,
    std::size_t rows,
    std::size_t cols,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    const float* expert_tensor_scales,
    float* output_dequant_scales,
    DeviceNvfp4Matrix* output);

bool GatherDeviceNvfp4Rows(
    const DeviceNvfp4Matrix& source,
    const int* source_row_indices,
    std::size_t row_count,
    DeviceNvfp4Matrix* output);

bool MultiplyDeviceTensorScales(
    const float* activation_tensor_scale_device,
    const float* weight_tensor_scale_device,
    float* alpha_device);

}  // namespace nemotron
