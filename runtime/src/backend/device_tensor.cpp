#include "nemotron/device_tensor.h"

#include <cuda_runtime.h>

#include <numeric>
#include <utility>

namespace nemotron {

struct DeviceTensorFp32::Impl {
  std::vector<std::size_t> shape;
  std::size_t numel = 0;
  std::size_t bytes = 0;
  float* data = nullptr;
  bool owns_data = false;
};

struct DeviceTensorBf16::Impl {
  std::vector<std::size_t> shape;
  std::size_t numel = 0;
  std::size_t bytes = 0;
  __nv_bfloat16* data = nullptr;
  bool owns_data = false;
};

namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool HasCudaDevice() {
  static const bool kHasCudaDevice = []() {
    int device_count = 0;
    return CheckCuda(cudaGetDeviceCount(&device_count)) && device_count > 0;
  }();
  return kHasCudaDevice;
}

std::size_t NumelFromShape(const std::vector<std::size_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(
      shape.begin(),
      shape.end(),
      std::size_t{1},
      [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
}

}  // namespace

std::unique_ptr<DeviceTensorFp32> DeviceTensorFp32::Create(std::vector<std::size_t> shape) {
  const std::size_t numel = NumelFromShape(shape);
  if (numel == 0 || !HasCudaDevice()) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->shape = std::move(shape);
  impl->numel = numel;
  impl->bytes = numel * sizeof(float);
  impl->owns_data = true;
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->data), impl->bytes))) {
    return nullptr;
  }
  return std::unique_ptr<DeviceTensorFp32>(new DeviceTensorFp32(std::move(impl)));
}

std::unique_ptr<DeviceTensorFp32> DeviceTensorFp32::CreateView(
    std::vector<std::size_t> shape,
    float* data) {
  const std::size_t numel = NumelFromShape(shape);
  if (numel == 0 || data == nullptr) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->shape = std::move(shape);
  impl->numel = numel;
  impl->bytes = numel * sizeof(float);
  impl->data = data;
  impl->owns_data = false;
  return std::unique_ptr<DeviceTensorFp32>(new DeviceTensorFp32(std::move(impl)));
}

DeviceTensorFp32::DeviceTensorFp32(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceTensorFp32::DeviceTensorFp32(DeviceTensorFp32&&) noexcept = default;

DeviceTensorFp32& DeviceTensorFp32::operator=(DeviceTensorFp32&&) noexcept = default;

DeviceTensorFp32::~DeviceTensorFp32() {
  if (impl_ && impl_->owns_data && impl_->data != nullptr) {
    cudaFree(impl_->data);
  }
}

bool DeviceTensorFp32::valid() const {
  return impl_ != nullptr && impl_->data != nullptr && impl_->numel > 0;
}

const std::vector<std::size_t>& DeviceTensorFp32::shape() const {
  static const std::vector<std::size_t> kEmpty;
  if (!impl_) {
    return kEmpty;
  }
  return impl_->shape;
}

std::size_t DeviceTensorFp32::numel() const {
  return impl_ ? impl_->numel : 0;
}

std::size_t DeviceTensorFp32::bytes() const {
  return impl_ ? impl_->bytes : 0;
}

float* DeviceTensorFp32::data() const {
  return impl_ ? impl_->data : nullptr;
}

bool DeviceTensorFp32::CopyFromHost(const float* host_data, std::size_t count) {
  return valid() &&
         host_data != nullptr &&
         count == numel() &&
         CheckCuda(cudaMemcpy(data(), host_data, bytes(), cudaMemcpyHostToDevice));
}

bool DeviceTensorFp32::CopyToHost(float* host_data, std::size_t count) const {
  return valid() &&
         host_data != nullptr &&
         count == numel() &&
         CheckCuda(cudaMemcpy(host_data, data(), bytes(), cudaMemcpyDeviceToHost));
}

bool DeviceTensorFp32::FillZero() {
  return valid() && CheckCuda(cudaMemset(data(), 0, bytes()));
}

std::unique_ptr<DeviceTensorBf16> DeviceTensorBf16::Create(std::vector<std::size_t> shape) {
  const std::size_t numel = NumelFromShape(shape);
  if (numel == 0 || !HasCudaDevice()) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->shape = std::move(shape);
  impl->numel = numel;
  impl->bytes = numel * sizeof(__nv_bfloat16);
  impl->owns_data = true;
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->data), impl->bytes))) {
    return nullptr;
  }
  return std::unique_ptr<DeviceTensorBf16>(new DeviceTensorBf16(std::move(impl)));
}

std::unique_ptr<DeviceTensorBf16> DeviceTensorBf16::CreateView(
    std::vector<std::size_t> shape,
    __nv_bfloat16* data) {
  const std::size_t numel = NumelFromShape(shape);
  if (numel == 0 || data == nullptr) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->shape = std::move(shape);
  impl->numel = numel;
  impl->bytes = numel * sizeof(__nv_bfloat16);
  impl->data = data;
  impl->owns_data = false;
  return std::unique_ptr<DeviceTensorBf16>(new DeviceTensorBf16(std::move(impl)));
}

DeviceTensorBf16::DeviceTensorBf16(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceTensorBf16::DeviceTensorBf16(DeviceTensorBf16&&) noexcept = default;

DeviceTensorBf16& DeviceTensorBf16::operator=(DeviceTensorBf16&&) noexcept = default;

DeviceTensorBf16::~DeviceTensorBf16() {
  if (impl_ && impl_->owns_data && impl_->data != nullptr) {
    cudaFree(impl_->data);
  }
}

bool DeviceTensorBf16::valid() const {
  return impl_ != nullptr && impl_->data != nullptr && impl_->numel > 0;
}

const std::vector<std::size_t>& DeviceTensorBf16::shape() const {
  static const std::vector<std::size_t> kEmpty;
  if (!impl_) {
    return kEmpty;
  }
  return impl_->shape;
}

std::size_t DeviceTensorBf16::numel() const {
  return impl_ ? impl_->numel : 0;
}

std::size_t DeviceTensorBf16::bytes() const {
  return impl_ ? impl_->bytes : 0;
}

__nv_bfloat16* DeviceTensorBf16::data() const {
  return impl_ ? impl_->data : nullptr;
}

bool DeviceTensorBf16::CopyFromHost(const __nv_bfloat16* host_data, std::size_t count) {
  return valid() &&
         host_data != nullptr &&
         count == numel() &&
         CheckCuda(cudaMemcpy(data(), host_data, bytes(), cudaMemcpyHostToDevice));
}

bool DeviceTensorBf16::CopyToHost(__nv_bfloat16* host_data, std::size_t count) const {
  return valid() &&
         host_data != nullptr &&
         count == numel() &&
         CheckCuda(cudaMemcpy(host_data, data(), bytes(), cudaMemcpyDeviceToHost));
}

bool DeviceTensorBf16::FillZero() {
  return valid() && CheckCuda(cudaMemset(data(), 0, bytes()));
}

}  // namespace nemotron
