#include "nemotron/cublaslt_handle.h"

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <memory>

namespace nemotron {

struct CublasLtHandle::Impl {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  bool valid = false;
};

std::unique_ptr<CublasLtHandle> CublasLtHandle::Create(std::size_t workspace_bytes) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  if (cublasLtCreate(&impl->handle) != CUBLAS_STATUS_SUCCESS) {
    return nullptr;
  }

  if (workspace_bytes > 0) {
    if (cudaMalloc(&impl->workspace, workspace_bytes) != cudaSuccess) {
      cublasLtDestroy(impl->handle);
      return nullptr;
    }
  }

  impl->workspace_bytes = workspace_bytes;
  impl->valid = true;
  return std::unique_ptr<CublasLtHandle>(new CublasLtHandle(std::move(impl)));
}

CublasLtHandle::CublasLtHandle(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

CublasLtHandle::CublasLtHandle(CublasLtHandle&&) noexcept = default;
CublasLtHandle& CublasLtHandle::operator=(CublasLtHandle&&) noexcept = default;

CublasLtHandle::~CublasLtHandle() {
  if (!impl_) {
    return;
  }
  if (impl_->workspace != nullptr) {
    cudaFree(impl_->workspace);
  }
  if (impl_->handle != nullptr) {
    cublasLtDestroy(impl_->handle);
  }
}

bool CublasLtHandle::valid() const {
  return impl_ != nullptr && impl_->valid;
}

cublasLtContext* CublasLtHandle::handle() const {
  return impl_->handle;
}

void* CublasLtHandle::workspace() const {
  return impl_->workspace;
}

std::size_t CublasLtHandle::workspace_bytes() const {
  return impl_->workspace_bytes;
}

}  // namespace nemotron
