#pragma once

#include <memory>

struct cudnnContext;

namespace nemotron {

class CudnnHandle {
 public:
  static std::unique_ptr<CudnnHandle> Create();

  CudnnHandle(CudnnHandle&&) noexcept;
  CudnnHandle& operator=(CudnnHandle&&) noexcept;
  ~CudnnHandle();

  CudnnHandle(const CudnnHandle&) = delete;
  CudnnHandle& operator=(const CudnnHandle&) = delete;

  bool valid() const;
  cudnnContext* handle() const;
  long long version() const;

 private:
  struct Impl;

  explicit CudnnHandle(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
