#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace nemotron {

using ReusableStateId = std::uint64_t;

enum class ReusableStateKind {
  kAttentionKv,
  kMambaRecurrent,
};

struct ReusableStateHandle {
  ReusableStateId id = 0;
  ReusableStateKind kind = ReusableStateKind::kAttentionKv;
  std::size_t bytes = 0;

  bool valid() const;
};

inline bool operator==(const ReusableStateHandle& lhs, const ReusableStateHandle& rhs) {
  return lhs.id == rhs.id &&
         lhs.kind == rhs.kind &&
         lhs.bytes == rhs.bytes;
}

struct ReusableStateDescriptor {
  ReusableStateHandle kv_state;
  ReusableStateHandle mamba_state;

  bool valid() const;
  std::size_t total_bytes() const;
};

inline bool operator==(const ReusableStateDescriptor& lhs, const ReusableStateDescriptor& rhs) {
  return lhs.kv_state == rhs.kv_state &&
         lhs.mamba_state == rhs.mamba_state;
}

struct ReusableStateView {
  ReusableStateId id = 0;
  ReusableStateKind kind = ReusableStateKind::kAttentionKv;
  std::string label;
  std::size_t bytes = 0;
  std::size_t ref_count = 0;
  bool has_device_storage = false;
};

class ReusableStateArena {
 public:
  explicit ReusableStateArena(std::size_t max_bytes);
  ~ReusableStateArena();

  ReusableStateArena(const ReusableStateArena&) = delete;
  ReusableStateArena& operator=(const ReusableStateArena&) = delete;
  ReusableStateArena(ReusableStateArena&&) noexcept;
  ReusableStateArena& operator=(ReusableStateArena&&) noexcept;

  ReusableStateHandle Allocate(
      ReusableStateKind kind,
      std::size_t bytes,
      const std::string& label);

  ReusableStateDescriptor AllocateDescriptor(
      std::size_t kv_bytes,
      std::size_t mamba_bytes,
      const std::string& label);

  bool Retain(const ReusableStateHandle& handle);
  bool Retain(const ReusableStateDescriptor& descriptor);
  void Release(const ReusableStateHandle& handle);
  void Release(const ReusableStateDescriptor& descriptor);

  bool CopyFromDevice(
      const ReusableStateHandle& handle,
      std::size_t offset_bytes,
      const void* device_src,
      std::size_t bytes);
  bool CopyToDevice(
      const ReusableStateHandle& handle,
      std::size_t offset_bytes,
      void* device_dst,
      std::size_t bytes) const;

  std::optional<ReusableStateView> Describe(ReusableStateId id) const;
  std::size_t current_bytes() const;
  std::size_t max_bytes() const;
  std::size_t allocation_count() const;

 private:
  struct Impl;

  std::size_t max_bytes_ = 0;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
