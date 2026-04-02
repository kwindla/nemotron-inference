#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nemotron {

class DeviceTensorFp32 {
 public:
  static std::unique_ptr<DeviceTensorFp32> Create(std::vector<std::size_t> shape);
  static std::unique_ptr<DeviceTensorFp32> CreateView(
      std::vector<std::size_t> shape,
      float* data);

  DeviceTensorFp32(DeviceTensorFp32&&) noexcept;
  DeviceTensorFp32& operator=(DeviceTensorFp32&&) noexcept;
  ~DeviceTensorFp32();

  DeviceTensorFp32(const DeviceTensorFp32&) = delete;
  DeviceTensorFp32& operator=(const DeviceTensorFp32&) = delete;

  bool valid() const;
  const std::vector<std::size_t>& shape() const;
  std::size_t numel() const;
  std::size_t bytes() const;
  float* data() const;

  bool CopyFromHost(const float* host_data, std::size_t count);
  bool CopyToHost(float* host_data, std::size_t count) const;
  bool FillZero(cudaStream_t stream = nullptr);

 private:
  struct Impl;

  explicit DeviceTensorFp32(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

class DeviceTensorBf16 {
 public:
  static std::unique_ptr<DeviceTensorBf16> Create(std::vector<std::size_t> shape);
  static std::unique_ptr<DeviceTensorBf16> CreateView(
      std::vector<std::size_t> shape,
      __nv_bfloat16* data);

  DeviceTensorBf16(DeviceTensorBf16&&) noexcept;
  DeviceTensorBf16& operator=(DeviceTensorBf16&&) noexcept;
  ~DeviceTensorBf16();

  DeviceTensorBf16(const DeviceTensorBf16&) = delete;
  DeviceTensorBf16& operator=(const DeviceTensorBf16&) = delete;

  bool valid() const;
  const std::vector<std::size_t>& shape() const;
  std::size_t numel() const;
  std::size_t bytes() const;
  __nv_bfloat16* data() const;

  bool CopyFromHost(const __nv_bfloat16* host_data, std::size_t count);
  bool CopyToHost(__nv_bfloat16* host_data, std::size_t count) const;
  bool FillZero(cudaStream_t stream = nullptr);

 private:
  struct Impl;

  explicit DeviceTensorBf16(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

class DeviceTensorFp8E4M3 {
 public:
  static std::unique_ptr<DeviceTensorFp8E4M3> Create(std::vector<std::size_t> shape);
  static std::unique_ptr<DeviceTensorFp8E4M3> CreateView(
      std::vector<std::size_t> shape,
      std::uint8_t* data);

  DeviceTensorFp8E4M3(DeviceTensorFp8E4M3&&) noexcept;
  DeviceTensorFp8E4M3& operator=(DeviceTensorFp8E4M3&&) noexcept;
  ~DeviceTensorFp8E4M3();

  DeviceTensorFp8E4M3(const DeviceTensorFp8E4M3&) = delete;
  DeviceTensorFp8E4M3& operator=(const DeviceTensorFp8E4M3&) = delete;

  bool valid() const;
  const std::vector<std::size_t>& shape() const;
  std::size_t numel() const;
  std::size_t bytes() const;
  std::uint8_t* data() const;

  bool CopyFromHost(const std::uint8_t* host_data, std::size_t count);
  bool CopyToHost(std::uint8_t* host_data, std::size_t count) const;
  bool FillZero(cudaStream_t stream = nullptr);

 private:
  struct Impl;

  explicit DeviceTensorFp8E4M3(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
