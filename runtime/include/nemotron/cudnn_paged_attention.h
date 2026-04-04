#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <memory>
#include <optional>
#include <unordered_map>

#include "nemotron/cudnn_handle.h"
#include "nemotron/paged_attention_plan.h"

namespace nemotron {

struct CudnnPagedAttentionConfig {
  AttentionKvCacheConfig cache_config;
  std::size_t batch_size = 0;
  std::size_t query_head_count = 0;
  std::size_t max_query_tokens = 0;
  std::size_t max_kv_tokens = 0;
  std::size_t container_page_count = 0;
  std::size_t page_table_entries = 0;
  float attn_scale = 1.0f;
  bool causal = false;
  bool generate_stats = false;

  bool valid() const;
  bool operator==(const CudnnPagedAttentionConfig& other) const;
};

struct CudnnPagedAttentionConfigHash {
  std::size_t operator()(const CudnnPagedAttentionConfig& config) const;
};

std::optional<CudnnPagedAttentionConfig> BuildCudnnPagedAttentionConfig(
    const PagedAttentionBatchPlan& kv_plan,
    std::size_t query_head_count,
    std::size_t max_query_tokens,
    std::size_t container_page_count,
    float attn_scale = 0.0f,
    bool causal = false,
    bool generate_stats = false);

struct CudnnPagedAttentionExecution {
  const void* query = nullptr;
  const void* key_cache = nullptr;
  const void* value_cache = nullptr;
  const std::int32_t* seq_len_q = nullptr;
  const std::int32_t* seq_len_kv = nullptr;
  const std::int32_t* page_table_k = nullptr;
  const std::int32_t* page_table_v = nullptr;
  void* output = nullptr;
  void* stats = nullptr;
};

class CudnnPagedAttentionPlan {
 public:
  static std::unique_ptr<CudnnPagedAttentionPlan> Create(
      const CudnnHandle& handle,
      const CudnnPagedAttentionConfig& config);

  CudnnPagedAttentionPlan(CudnnPagedAttentionPlan&&) noexcept;
  CudnnPagedAttentionPlan& operator=(CudnnPagedAttentionPlan&&) noexcept;
  ~CudnnPagedAttentionPlan();

  CudnnPagedAttentionPlan(const CudnnPagedAttentionPlan&) = delete;
  CudnnPagedAttentionPlan& operator=(const CudnnPagedAttentionPlan&) = delete;

  bool valid() const;
  const CudnnPagedAttentionConfig& config() const;
  std::size_t workspace_bytes() const;
  bool Execute(const CudnnHandle& handle, const CudnnPagedAttentionExecution& execution) const;

 private:
  struct Impl;

  explicit CudnnPagedAttentionPlan(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

struct CudnnPagedAttentionPlanCacheStats {
  std::size_t hits = 0;
  std::size_t misses = 0;
  std::size_t size = 0;
};

class CudnnPagedAttentionPlanCache {
 public:
  static CudnnPagedAttentionPlanCache& Global();

  std::shared_ptr<const CudnnPagedAttentionPlan> GetOrCreate(
      const CudnnHandle& handle,
      const CudnnPagedAttentionConfig& config);

  CudnnPagedAttentionPlanCacheStats stats() const;
  void Clear();

 private:
  mutable std::mutex mutex_;
  std::unordered_map<
      CudnnPagedAttentionConfig,
      std::shared_ptr<const CudnnPagedAttentionPlan>,
      CudnnPagedAttentionConfigHash>
      cache_;
  std::size_t hits_ = 0;
  std::size_t misses_ = 0;
};

}  // namespace nemotron
