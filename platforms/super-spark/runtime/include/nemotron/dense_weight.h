#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "nemotron/gemm_catalog.h"

namespace nemotron {

class DeviceDenseWeightFp32 {
 public:
  static std::unique_ptr<DeviceDenseWeightFp32> Upload(
      const GemmDescriptor& descriptor,
      float storage_scale = 1.0f);
  static std::unique_ptr<DeviceDenseWeightFp32> CreateView(
      std::size_t output_rows,
      std::size_t input_cols,
      float* data);

  DeviceDenseWeightFp32(DeviceDenseWeightFp32&&) noexcept;
  DeviceDenseWeightFp32& operator=(DeviceDenseWeightFp32&&) noexcept;
  ~DeviceDenseWeightFp32();

  DeviceDenseWeightFp32(const DeviceDenseWeightFp32&) = delete;
  DeviceDenseWeightFp32& operator=(const DeviceDenseWeightFp32&) = delete;

  bool valid() const;
  std::size_t output_rows() const;
  std::size_t input_cols() const;
  std::size_t numel() const;
  const float* data() const;

 private:
  struct Impl;

  explicit DeviceDenseWeightFp32(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

std::optional<std::vector<float>> ReadDenseWeightToHostFp32(
    const GemmDescriptor& descriptor,
    float storage_scale = 1.0f);

}  // namespace nemotron
