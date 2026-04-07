#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "nemotron/paged_kv_cache.h"

namespace nemotron {

struct AttentionSequencePages {
  std::size_t token_count = 0;
  std::vector<std::size_t> page_ids;
};

struct PagedAttentionBatchPlan {
  AttentionKvCacheConfig cache_config;
  std::size_t layer_index = 0;
  std::size_t batch_size = 0;
  std::size_t max_sequence_tokens = 0;
  std::size_t max_pages_per_sequence = 0;
  std::vector<std::int32_t> page_table;
  std::vector<std::int32_t> sequence_lengths;
  std::vector<std::int32_t> pages_per_sequence;

  bool valid() const;
  std::size_t page_table_entries() const;
};

std::optional<PagedAttentionBatchPlan> BuildPagedAttentionBatchPlan(
    const AttentionKvCacheConfig& cache_config,
    std::size_t layer_index,
    const std::vector<AttentionSequencePages>& sequences,
    const PagedKvCacheArena* arena = nullptr);

}  // namespace nemotron
