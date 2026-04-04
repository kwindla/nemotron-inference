#pragma once

#include <cstddef>
#include <memory>

#include <cuda_bf16.h>

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

class DeviceDenseWeightBf16 {
 public:
  static std::unique_ptr<DeviceDenseWeightBf16> Upload(const GemmDescriptor& descriptor);

  DeviceDenseWeightBf16(DeviceDenseWeightBf16&&) noexcept;
  DeviceDenseWeightBf16& operator=(DeviceDenseWeightBf16&&) noexcept;
  ~DeviceDenseWeightBf16();

  DeviceDenseWeightBf16(const DeviceDenseWeightBf16&) = delete;
  DeviceDenseWeightBf16& operator=(const DeviceDenseWeightBf16&) = delete;

  bool valid() const;
  std::size_t output_rows() const;
  std::size_t input_cols() const;
  std::size_t numel() const;
  const __nv_bfloat16* data() const;

 private:
  struct Impl;

  explicit DeviceDenseWeightBf16(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
