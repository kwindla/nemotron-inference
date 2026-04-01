#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "nemotron/device_tensor.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {

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
  const std::uint8_t* packed_data() const;
  const std::uint8_t* block_scales_data() const;
  const std::uint8_t* matmul_block_scales_data() const;
  const std::uint8_t* tensor_scale_data() const;
  Nvfp4ScaleLayout scale_layout() const;

  bool CopyPackedToHost(std::vector<std::uint8_t>* output) const;
  bool CopyBlockScalesToHost(std::vector<std::uint8_t>* output) const;
  bool CopyMatmulBlockScalesToHost(std::vector<std::uint8_t>* output) const;
  bool CopyTensorScaleToHost(float* output) const;

 private:
  struct Impl;

  explicit DeviceNvfp4Matrix(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<DeviceNvfp4Matrix> PackDeviceRowMajorFp32ToNvfp4(
    const DeviceTensorFp32& source,
    const Nvfp4PackOptions& options = {});

}  // namespace nemotron
