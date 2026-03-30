#include "nemotron/paged_attention_plan.h"

#include <iostream>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::AttentionSequencePages;
using nemotron::BuildPagedAttentionBatchPlan;
using nemotron::KvCacheDataType;
using nemotron::PagedKvCacheArena;

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

AttentionKvCacheConfig make_test_config() {
  AttentionKvCacheConfig config;
  config.layer_count = 8;
  config.kv_head_count = 2;
  config.head_dim = 128;
  config.tokens_per_page = 64;
  config.dtype = KvCacheDataType::kBf16;
  return config;
}

bool test_paged_attention_batch_plan_builds_layer_page_table() {
  auto arena = PagedKvCacheArena::Create(make_test_config(), 8);
  if (!expect(arena.has_value(), "arena should build")) {
    return false;
  }
  auto mutable_arena = std::move(*arena);

  const auto seq0_pages = mutable_arena.AllocatePages(2, 1);
  const auto seq1_pages = mutable_arena.AllocatePages(2, 2);
  if (!expect(seq0_pages.has_value() && seq1_pages.has_value(), "test pages should allocate")) {
    return false;
  }

  const std::vector<AttentionSequencePages> sequences = {
      {40, {(*seq0_pages)[0].page_id}},
      {65, {(*seq1_pages)[0].page_id, (*seq1_pages)[1].page_id}},
      {0, {}},
  };

  const auto plan = BuildPagedAttentionBatchPlan(make_test_config(), 2, sequences, &mutable_arena);
  if (!expect(plan.has_value(), "paged attention plan should build for valid sequences")) {
    return false;
  }
  if (!expect(plan->valid(), "built plan should be structurally valid")) {
    return false;
  }
  if (!expect(plan->batch_size == 3, "plan should preserve batch size")) {
    return false;
  }
  if (!expect(plan->max_sequence_tokens == 65, "plan should track the longest sequence")) {
    return false;
  }
  if (!expect(plan->max_pages_per_sequence == 2, "plan should track the widest page table row")) {
    return false;
  }
  if (!expect(plan->sequence_lengths.size() == 3 && plan->sequence_lengths[0] == 40 && plan->sequence_lengths[1] == 65 &&
                  plan->sequence_lengths[2] == 0,
              "sequence lengths should be preserved")) {
    return false;
  }
  if (!expect(plan->pages_per_sequence.size() == 3 && plan->pages_per_sequence[0] == 1 && plan->pages_per_sequence[1] == 2 &&
                  plan->pages_per_sequence[2] == 0,
              "page counts per sequence should be preserved")) {
    return false;
  }
  if (!expect(plan->page_table_entries() == 6, "page table should be batch_size * max_pages_per_sequence")) {
    return false;
  }
  if (!expect(plan->page_table[0] == static_cast<std::int32_t>((*seq0_pages)[0].page_id) &&
                  plan->page_table[1] == -1 &&
                  plan->page_table[2] == static_cast<std::int32_t>((*seq1_pages)[0].page_id) &&
                  plan->page_table[3] == static_cast<std::int32_t>((*seq1_pages)[1].page_id) &&
                  plan->page_table[4] == -1 &&
                  plan->page_table[5] == -1,
              "page table should be padded with -1 entries")) {
    return false;
  }
  return true;
}

bool test_paged_attention_plan_rejects_mismatched_pages() {
  auto arena = PagedKvCacheArena::Create(make_test_config(), 4);
  if (!expect(arena.has_value(), "arena should build")) {
    return false;
  }
  auto mutable_arena = std::move(*arena);

  const auto wrong_layer_pages = mutable_arena.AllocatePages(1, 1);
  if (!expect(wrong_layer_pages.has_value(), "test page should allocate")) {
    return false;
  }

  const std::vector<AttentionSequencePages> wrong_count = {
      {65, {(*wrong_layer_pages)[0].page_id}},
  };
  if (!expect(!BuildPagedAttentionBatchPlan(make_test_config(), 1, wrong_count, &mutable_arena).has_value(),
              "token counts requiring more pages than provided should be rejected")) {
    return false;
  }

  const std::vector<AttentionSequencePages> wrong_layer = {
      {1, {(*wrong_layer_pages)[0].page_id}},
  };
  return expect(!BuildPagedAttentionBatchPlan(make_test_config(), 0, wrong_layer, &mutable_arena).has_value(),
                "page tables should reject pages owned by another attention layer");
}

}  // namespace

int main() {
  if (!test_paged_attention_batch_plan_builds_layer_page_table() ||
      !test_paged_attention_plan_rejects_mismatched_pages()) {
    return 1;
  }
  std::cout << "paged_attention_plan_test: PASS\n";
  return 0;
}
