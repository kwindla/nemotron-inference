#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace nemotron {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    Reset();
    data_ = other.data_;
    count_ = other.count_;
    other.data_ = nullptr;
    other.count_ = 0;
    return *this;
  }

  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool valid() const { return count_ == 0 || data_ != nullptr; }
  T* data() const { return data_; }
  std::size_t count() const { return count_; }

  bool Resize(std::size_t count) {
    if (count == count_) {
      return true;
    }
    Reset();
    if (count == 0) {
      return true;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) != cudaSuccess) {
      data_ = nullptr;
      count_ = 0;
      return false;
    }
    count_ = count;
    return true;
  }

  bool CopyFromHost(const T* host_values, std::size_t count) {
    return count <= count_ &&
           ((count == 0) ||
            (host_values != nullptr &&
             cudaMemcpy(data_, host_values, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess));
  }

  bool CopyFromHostAsync(const T* host_values, std::size_t count, cudaStream_t stream = nullptr) {
    return count <= count_ &&
           ((count == 0) ||
            (host_values != nullptr &&
             cudaMemcpyAsync(
                 data_,
                 host_values,
                 count * sizeof(T),
                 cudaMemcpyHostToDevice,
                 stream) == cudaSuccess));
  }

  bool CopyFromHost(const std::vector<T>& host_values) {
    return CopyFromHost(host_values.data(), host_values.size());
  }

  bool CopyFromHostAsync(const std::vector<T>& host_values, cudaStream_t stream = nullptr) {
    return CopyFromHostAsync(host_values.data(), host_values.size(), stream);
  }

  bool CopyFromDevice(const T* device_values, std::size_t count) {
    return count <= count_ &&
           ((count == 0) ||
            (device_values != nullptr &&
             cudaMemcpy(data_, device_values, count * sizeof(T), cudaMemcpyDeviceToDevice) ==
                 cudaSuccess));
  }

  bool CopyFromDeviceAsync(const T* device_values, std::size_t count, cudaStream_t stream = nullptr) {
    return count <= count_ &&
           ((count == 0) ||
            (device_values != nullptr &&
             cudaMemcpyAsync(
                 data_,
                 device_values,
                 count * sizeof(T),
                 cudaMemcpyDeviceToDevice,
                 stream) == cudaSuccess));
  }

  bool CopyToHost(T* host_values, std::size_t count) const {
    return count <= count_ &&
           ((count == 0) ||
            (host_values != nullptr &&
             cudaMemcpy(host_values, data_, count * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess));
  }

  bool CopyToHostAsync(T* host_values, std::size_t count, cudaStream_t stream = nullptr) const {
    return count <= count_ &&
           ((count == 0) ||
            (host_values != nullptr &&
             cudaMemcpyAsync(
                 host_values,
                 data_,
                 count * sizeof(T),
                 cudaMemcpyDeviceToHost,
                 stream) == cudaSuccess));
  }

  bool FillZero() {
    return count_ == 0 ||
           (cudaMemset(data_, 0, count_ * sizeof(T)) == cudaSuccess);
  }

  bool FillZeroAsync(cudaStream_t stream = nullptr) {
    return count_ == 0 ||
           (cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream) == cudaSuccess);
  }

  bool FillByte(int byte_value) {
    return count_ == 0 ||
           (cudaMemset(data_, byte_value, count_ * sizeof(T)) == cudaSuccess);
  }

  bool FillByteAsync(int byte_value, cudaStream_t stream = nullptr) {
    return count_ == 0 ||
           (cudaMemsetAsync(data_, byte_value, count_ * sizeof(T), stream) == cudaSuccess);
  }

 private:
  void Reset() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace nemotron
