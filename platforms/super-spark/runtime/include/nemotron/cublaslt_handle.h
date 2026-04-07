#pragma once

#include <cstddef>
#include <memory>

struct cublasLtContext;

namespace nemotron {

class CublasLtHandle {
 public:
  static std::unique_ptr<CublasLtHandle> Create(std::size_t workspace_bytes = 4u * 1024u * 1024u);

  CublasLtHandle(CublasLtHandle&&) noexcept;
  CublasLtHandle& operator=(CublasLtHandle&&) noexcept;
  ~CublasLtHandle();

  CublasLtHandle(const CublasLtHandle&) = delete;
  CublasLtHandle& operator=(const CublasLtHandle&) = delete;

  bool valid() const;
  cublasLtContext* handle() const;
  void* workspace() const;
  std::size_t workspace_bytes() const;

 private:
  struct Impl;

  explicit CublasLtHandle(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
