#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "nemotron/device_tensor.h"
#include "nemotron/nvfp4_packing.h"

namespace nemotron {

class DeviceNvfp4Matrix {
 public:
  static std::unique_ptr<DeviceNvfp4Matrix> Create(std::size_t rows, std::size_t cols);

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

// Same as PackDeviceRowMajorFp32ToNvfp4 but writes into pre-allocated buffers.
// Zero cudaMalloc on the hot path. All output pointers must be device-resident
// and pre-sized for the given source dimensions.
bool PackDeviceRowMajorFp32ToNvfp4InPlace(
    const DeviceTensorFp32& source,
    const Nvfp4PackOptions& options,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    unsigned int* global_max_bits_scratch);

}  // namespace nemotron
