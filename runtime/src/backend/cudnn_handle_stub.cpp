#include "nemotron/cudnn_handle.h"

#include <memory>

namespace nemotron {

struct CudnnHandle::Impl {};

std::unique_ptr<CudnnHandle> CudnnHandle::Create() {
  return std::unique_ptr<CudnnHandle>(new CudnnHandle(std::make_unique<Impl>()));
}

CudnnHandle::CudnnHandle(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

CudnnHandle::CudnnHandle(CudnnHandle&&) noexcept = default;
CudnnHandle& CudnnHandle::operator=(CudnnHandle&&) noexcept = default;
CudnnHandle::~CudnnHandle() = default;

bool CudnnHandle::valid() const {
  return false;
}

cudnnContext* CudnnHandle::handle() const {
  return nullptr;
}

long long CudnnHandle::version() const {
  return 0;
}

}  // namespace nemotron
