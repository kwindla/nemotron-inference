#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "nemotron/paged_kv_cache.h"

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

struct ReusableStateSnapshotLayout {
  std::size_t prefix_token_count = 0;
  AttentionKvCacheConfig attention_kv_cache;
  std::size_t live_kv_pages_per_layer = 0;
  std::size_t hidden_bytes = 0;
  std::size_t residual_bytes = 0;
  std::size_t mamba_conv_bytes = 0;
  std::size_t mamba_ssm_bytes = 0;

  bool valid() const;
};

inline bool operator==(
    const ReusableStateSnapshotLayout& lhs,
    const ReusableStateSnapshotLayout& rhs) {
  return lhs.prefix_token_count == rhs.prefix_token_count &&
         lhs.attention_kv_cache.layer_count == rhs.attention_kv_cache.layer_count &&
         lhs.attention_kv_cache.kv_head_count == rhs.attention_kv_cache.kv_head_count &&
         lhs.attention_kv_cache.head_dim == rhs.attention_kv_cache.head_dim &&
         lhs.attention_kv_cache.tokens_per_page == rhs.attention_kv_cache.tokens_per_page &&
         lhs.attention_kv_cache.dtype == rhs.attention_kv_cache.dtype &&
         lhs.live_kv_pages_per_layer == rhs.live_kv_pages_per_layer &&
         lhs.hidden_bytes == rhs.hidden_bytes &&
         lhs.residual_bytes == rhs.residual_bytes &&
         lhs.mamba_conv_bytes == rhs.mamba_conv_bytes &&
         lhs.mamba_ssm_bytes == rhs.mamba_ssm_bytes;
}

struct ReusableStateDescriptor {
  ReusableStateHandle kv_state;
  ReusableStateHandle mamba_state;
  ReusableStateSnapshotLayout snapshot_layout;

  bool valid() const;
  bool has_snapshot_layout() const;
  std::size_t total_bytes() const;
};

inline bool operator==(const ReusableStateDescriptor& lhs, const ReusableStateDescriptor& rhs) {
  return lhs.kv_state == rhs.kv_state &&
         lhs.mamba_state == rhs.mamba_state &&
         lhs.snapshot_layout == rhs.snapshot_layout;
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
  // The arena owns a recycled slab of snapshot storage. RequestExecutionContext
  // keeps ownership of live KV page tables and tensors; snapshots are copied
  // into this arena and restored back into freshly allocated live pages.
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
