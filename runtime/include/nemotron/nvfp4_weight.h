#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "nemotron/gemm_catalog.h"

namespace nemotron {

class DeviceNvfp4Weight {
 public:
  static std::unique_ptr<DeviceNvfp4Weight> Upload(const GemmDescriptor& descriptor);

  DeviceNvfp4Weight(DeviceNvfp4Weight&&) noexcept;
  DeviceNvfp4Weight& operator=(DeviceNvfp4Weight&&) noexcept;
  ~DeviceNvfp4Weight();

  DeviceNvfp4Weight(const DeviceNvfp4Weight&) = delete;
  DeviceNvfp4Weight& operator=(const DeviceNvfp4Weight&) = delete;

  bool valid() const;
  std::size_t output_rows() const;
  std::size_t input_cols() const;
  std::size_t packed_nbytes() const;
  std::size_t block_scales_nbytes() const;
  std::size_t matmul_block_scales_nbytes() const;
  std::size_t tensor_scale_nbytes() const;
  const std::uint8_t* packed_data() const;
  const std::uint8_t* block_scales_data() const;
  const std::uint8_t* matmul_block_scales_data() const;
  const std::uint8_t* tensor_scale_data() const;

 private:
  struct Impl;

  explicit DeviceNvfp4Weight(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
