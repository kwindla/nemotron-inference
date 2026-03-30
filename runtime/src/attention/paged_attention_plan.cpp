#include "nemotron/paged_attention_plan.h"

#include <algorithm>
#include <limits>

namespace nemotron {

namespace {

bool FitsInt32(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
}

}  // namespace

bool PagedAttentionBatchPlan::valid() const {
  return batch_size > 0 &&
         cache_config.layer_count > 0 &&
         cache_config.tokens_per_page > 0 &&
         sequence_lengths.size() == batch_size &&
         pages_per_sequence.size() == batch_size &&
         page_table.size() == batch_size * max_pages_per_sequence;
}

std::size_t PagedAttentionBatchPlan::page_table_entries() const {
  return page_table.size();
}

std::optional<PagedAttentionBatchPlan> BuildPagedAttentionBatchPlan(
    const AttentionKvCacheConfig& cache_config,
    std::size_t layer_index,
    const std::vector<AttentionSequencePages>& sequences,
    const PagedKvCacheArena* arena) {
  if (!BuildAttentionKvPageGeometry(cache_config).has_value() ||
      layer_index >= cache_config.layer_count ||
      sequences.empty()) {
    return std::nullopt;
  }

  PagedAttentionBatchPlan plan;
  plan.cache_config = cache_config;
  plan.layer_index = layer_index;
  plan.batch_size = sequences.size();

  for (const auto& sequence : sequences) {
    if (!FitsInt32(sequence.token_count)) {
      return std::nullopt;
    }
    const std::size_t required_pages = RequiredPagesForTokens(cache_config, sequence.token_count);
    if (sequence.page_ids.size() != required_pages) {
      return std::nullopt;
    }
    if (!FitsInt32(required_pages)) {
      return std::nullopt;
    }
    for (const std::size_t page_id : sequence.page_ids) {
      if (!FitsInt32(page_id)) {
        return std::nullopt;
      }
      if (arena != nullptr) {
        const auto handle = arena->InspectPage(page_id);
        if (!handle.has_value() || handle->layer_index != layer_index) {
          return std::nullopt;
        }
      }
    }
    plan.max_sequence_tokens = std::max(plan.max_sequence_tokens, sequence.token_count);
    plan.max_pages_per_sequence = std::max(plan.max_pages_per_sequence, required_pages);
  }

  plan.sequence_lengths.reserve(plan.batch_size);
  plan.pages_per_sequence.reserve(plan.batch_size);
  plan.page_table.assign(plan.batch_size * plan.max_pages_per_sequence, -1);

  for (std::size_t batch_index = 0; batch_index < sequences.size(); ++batch_index) {
    const auto& sequence = sequences[batch_index];
    const std::size_t required_pages = RequiredPagesForTokens(cache_config, sequence.token_count);
    plan.sequence_lengths.push_back(static_cast<std::int32_t>(sequence.token_count));
    plan.pages_per_sequence.push_back(static_cast<std::int32_t>(required_pages));
    for (std::size_t page_slot = 0; page_slot < sequence.page_ids.size(); ++page_slot) {
      plan.page_table[batch_index * plan.max_pages_per_sequence + page_slot] =
          static_cast<std::int32_t>(sequence.page_ids[page_slot]);
    }
  }

  if (!plan.valid()) {
    return std::nullopt;
  }
  return plan;
}

}  // namespace nemotron
