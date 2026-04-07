#include "nemotron/cudnn_handle.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <memory>

namespace nemotron {

struct CudnnHandle::Impl {
  cudnnHandle_t handle = nullptr;
  long long version = 0;
  bool valid = false;
};

std::unique_ptr<CudnnHandle> CudnnHandle::Create() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  if (cudnnCreate(&impl->handle) != CUDNN_STATUS_SUCCESS) {
    return nullptr;
  }

  impl->version = static_cast<long long>(cudnnGetVersion());
  impl->valid = true;
  return std::unique_ptr<CudnnHandle>(new CudnnHandle(std::move(impl)));
}

CudnnHandle::CudnnHandle(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

CudnnHandle::CudnnHandle(CudnnHandle&&) noexcept = default;
CudnnHandle& CudnnHandle::operator=(CudnnHandle&&) noexcept = default;

CudnnHandle::~CudnnHandle() {
  if (impl_ && impl_->handle != nullptr) {
    cudnnDestroy(impl_->handle);
  }
}

bool CudnnHandle::valid() const {
  return impl_ != nullptr && impl_->valid && impl_->handle != nullptr;
}

cudnnContext* CudnnHandle::handle() const {
  return impl_ ? impl_->handle : nullptr;
}

long long CudnnHandle::version() const {
  return impl_ ? impl_->version : 0;
}

}  // namespace nemotron
