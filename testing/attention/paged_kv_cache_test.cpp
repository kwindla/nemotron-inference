#include "nemotron/paged_kv_cache.h"

#include <iostream>
#include <vector>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::BuildAttentionKvPageGeometry;
using nemotron::KvCacheDataType;
using nemotron::ModelKvBytesPerToken;
using nemotron::PagedKvCacheArena;
using nemotron::RequiredPagesForTokens;

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

bool test_kv_page_geometry_matches_gb10_attention_budget() {
  const AttentionKvCacheConfig config = make_test_config();
  const auto geometry = BuildAttentionKvPageGeometry(config);
  if (!expect(geometry.has_value(), "valid attention config should produce page geometry")) {
    return false;
  }
  if (!expect(geometry->bytes_per_scalar == 2, "BF16 KV cache should use 2 bytes per scalar")) {
    return false;
  }
  if (!expect(geometry->bytes_per_token == 1024, "per-layer KV bytes per token should match 2*kv_heads*head_dim*2")) {
    return false;
  }
  if (!expect(geometry->bytes_per_page == 65536, "64-token pages should be 64 KiB for this layer config")) {
    return false;
  }
  if (!expect(ModelKvBytesPerToken(config) == 8192, "8 attention layers should total 8 KiB per token")) {
    return false;
  }
  if (!expect(RequiredPagesForTokens(config, 0) == 0, "zero-token sequences should need zero pages")) {
    return false;
  }
  if (!expect(RequiredPagesForTokens(config, 1) == 1, "single-token sequences should need one page")) {
    return false;
  }
  return expect(RequiredPagesForTokens(config, 65) == 2, "65-token sequences should need two pages");
}

bool test_kv_page_arena_allocates_and_reuses_pages() {
  auto arena = PagedKvCacheArena::Create(make_test_config(), 6);
  if (!expect(arena.has_value(), "arena should build for a valid config")) {
    return false;
  }

  auto mutable_arena = std::move(*arena);
  if (!expect(mutable_arena.total_pages() == 6, "arena should track total pages")) {
    return false;
  }
  if (!expect(mutable_arena.free_pages() == 6, "fresh arena should have all pages free")) {
    return false;
  }

  const auto first = mutable_arena.AllocatePages(3, 2);
  if (!expect(first.has_value(), "first allocation should succeed")) {
    return false;
  }
  if (!expect((*first)[0].page_id == 0 && (*first)[1].page_id == 1, "pages should allocate in ascending id order")) {
    return false;
  }
  if (!expect((*first)[0].byte_offset == 0, "first page should start at byte offset zero")) {
    return false;
  }
  if (!expect((*first)[1].byte_offset == mutable_arena.geometry().bytes_per_page,
              "page offsets should advance by one page size")) {
    return false;
  }
  if (!expect(mutable_arena.allocated_pages() == 2, "allocated page count should advance")) {
    return false;
  }

  const auto inspected = mutable_arena.InspectPage(1);
  if (!expect(inspected.has_value(), "allocated page should be inspectable")) {
    return false;
  }
  if (!expect(inspected->layer_index == 3, "page metadata should retain layer ownership")) {
    return false;
  }

  if (!expect(!mutable_arena.AllocatePages(8, 1).has_value(), "invalid layer index should be rejected")) {
    return false;
  }
  if (!expect(!mutable_arena.ReleasePages({1, 1}), "duplicate page ids in a release should be rejected")) {
    return false;
  }
  if (!expect(mutable_arena.ReleasePages({0, 1}), "valid release should succeed")) {
    return false;
  }
  if (!expect(mutable_arena.free_pages() == 6, "release should return all pages to the arena")) {
    return false;
  }

  const auto second = mutable_arena.AllocatePages(2, 1);
  if (!expect(second.has_value(), "page should be reusable after release")) {
    return false;
  }
  return expect((*second)[0].page_id == 0, "released low page ids should be reused first");
}

bool test_kv_page_geometry_and_release_reject_invalid_inputs() {
  AttentionKvCacheConfig invalid_config = make_test_config();
  invalid_config.tokens_per_page = 0;
  if (!expect(
          !BuildAttentionKvPageGeometry(invalid_config).has_value(),
          "invalid page geometry should be rejected")) {
    return false;
  }
  if (!expect(
          !PagedKvCacheArena::Create(invalid_config, 4).has_value(),
          "arena creation should reject invalid page geometry")) {
    return false;
  }

  auto arena = PagedKvCacheArena::Create(make_test_config(), 4);
  if (!expect(arena.has_value(), "valid arena should build for release validation")) {
    return false;
  }
  auto mutable_arena = std::move(*arena);
  const auto pages = mutable_arena.AllocatePages(0, 2);
  if (!expect(pages.has_value(), "test pages should allocate")) {
    return false;
  }

  return expect(
             !mutable_arena.ReleasePages({4}),
             "release should reject page ids outside the arena") &&
         expect(
             !mutable_arena.ReleasePages({3}),
             "release should reject pages that were never allocated") &&
         expect(
             mutable_arena.ReleasePages({(*pages)[0].page_id, (*pages)[1].page_id}),
             "release should succeed for the allocated page set");
}

}  // namespace

int main() {
  if (!test_kv_page_geometry_matches_gb10_attention_budget() ||
      !test_kv_page_arena_allocates_and_reuses_pages() ||
      !test_kv_page_geometry_and_release_reject_invalid_inputs()) {
    return 1;
  }
  std::cout << "paged_kv_cache_test: PASS\n";
  return 0;
}
