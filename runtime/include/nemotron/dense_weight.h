#pragma once

#include <cstddef>
#include <memory>

#include "nemotron/gemm_catalog.h"

namespace nemotron {

class DeviceDenseWeightFp32 {
 public:
  static std::unique_ptr<DeviceDenseWeightFp32> Upload(const GemmDescriptor& descriptor);

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

}  // namespace nemotron
