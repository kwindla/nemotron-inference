#include "nemotron/prefix_cache.h"
#include "nemotron/request_context.h"
#include "nemotron/state_snapshot.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::CacheLookupRequest;
using nemotron::CacheMatchSource;
using nemotron::KvCacheDataType;
using nemotron::PrefixCache;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;
using nemotron::ReusableStateArena;
using nemotron::ReusableStateDescriptor;
using nemotron::ReusableStateHandle;
using nemotron::ReusableStateKind;
using nemotron::SerializedPromptIdentity;
using nemotron::TokenId;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

SerializedPromptIdentity make_identity(
    std::vector<TokenId> token_ids,
    std::string serializer_revision = "chat-v1",
    bool reasoning_mode = false) {
  SerializedPromptIdentity identity;
  identity.token_ids = std::move(token_ids);
  identity.tenant_namespace = "tenant-a";
  identity.tokenizer_revision = "tok-r1";
  identity.serializer_revision = std::move(serializer_revision);
  identity.model_revision = "model-r1";
  identity.reasoning_mode = reasoning_mode;
  return identity;
}

ReusableStateDescriptor make_state(
    std::uint64_t kv_handle,
    std::uint64_t mamba_handle,
    std::size_t kv_bytes = 1024,
    std::size_t mamba_bytes = 2048) {
  ReusableStateDescriptor state;
  state.kv_state = ReusableStateHandle{
      kv_handle,
      ReusableStateKind::kAttentionKv,
      kv_bytes,
  };
  state.mamba_state = ReusableStateHandle{
      mamba_handle,
      ReusableStateKind::kMambaRecurrent,
      mamba_bytes,
  };
  return state;
}

RequestExecutionConfig make_request_config() {
  RequestExecutionConfig config;
  config.hidden_size = 16;
  config.max_tokens = 8;
  config.scratch_tokens = 4;
  config.attention_kv_cache = AttentionKvCacheConfig{
      2,
      2,
      4,
      4,
      KvCacheDataType::kBf16,
  };
  config.attention_total_pages = 4;
  config.mamba_conv_state_bytes_fp32 = 12 * sizeof(float);
  config.mamba_state_bytes_fp32 = 20 * sizeof(float);
  return config;
}

std::uint16_t bf16_bits(__nv_bfloat16 value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool same_bf16(const std::vector<__nv_bfloat16>& lhs, const std::vector<__nv_bfloat16>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (bf16_bits(lhs[i]) != bf16_bits(rhs[i])) {
      return false;
    }
  }
  return true;
}

bool test_conversation_head_hit() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  const auto node_id = cache.PublishConversationHead(
      "conv-1",
      make_identity({1, 2, 3, 4}),
      make_state(11, 21));

  CacheLookupRequest request;
  request.identity = make_identity({1, 2, 3, 4, 5, 6});
  request.conversation_id = "conv-1";
  const auto match = cache.Lookup(request);

  return expect(node_id != 0, "conversation head publish should succeed") &&
         expect(match.hit(), "conversation head should hit") &&
         expect(match.source == CacheMatchSource::kConversationCommittedHead, "match source should be committed head") &&
         expect(match.node_id == node_id, "matched node id should equal committed head") &&
         expect(match.matched_token_count == 4, "matched token count should equal cached prefix length");
}

bool test_exact_conversation_cache_hit_reports_full_length() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  const auto node_id = cache.PublishConversationHead(
      "conv-exact",
      make_identity({42, 43, 44, 45}),
      make_state(111, 222));

  CacheLookupRequest request;
  request.identity = make_identity({42, 43, 44, 45});
  request.conversation_id = "conv-exact";
  const auto match = cache.Lookup(request);

  return expect(node_id != 0, "exact committed-head publish should succeed") &&
         expect(match.hit(), "exact committed-head lookup should hit") &&
         expect(match.node_id == node_id, "exact lookup should return the published node") &&
         expect(
             match.source == CacheMatchSource::kConversationCommittedHead,
             "exact lookup should keep the committed-head source") &&
         expect(
             match.matched_token_count == request.identity.token_ids.size(),
             "exact lookup should report the full cached token count");
}

bool test_global_root_fallback() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  const auto global_node = cache.PublishGlobalRoot(
      make_identity({100, 200, 300, 400}),
      make_state(14, 24));

  CacheLookupRequest request;
  request.identity = make_identity({100, 200, 300, 400, 500});
  const auto match = cache.Lookup(request);

  return expect(global_node != 0, "global root publish should succeed") &&
         expect(match.hit(), "global root should hit") &&
         expect(match.source == CacheMatchSource::kGlobalRoot, "match source should be global root") &&
         expect(match.node_id == global_node, "global root node id should match") &&
         expect(match.matched_token_count == 4, "global root prefix length should match");
}

bool test_conversation_fallback_to_global_root() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  const auto global_node = cache.PublishGlobalRoot(
      make_identity({5, 6, 7, 8}),
      make_state(30, 40));
  cache.PublishConversationHead(
      "conv-lookup",
      make_identity({9, 9, 9}),
      make_state(31, 41));

  CacheLookupRequest request;
  request.identity = make_identity({5, 6, 7, 8, 10});
  request.conversation_id = "conv-lookup";
  const auto match = cache.Lookup(request);

  return expect(global_node != 0, "global root publish should succeed") &&
         expect(match.hit(), "lookup should fall back to global root") &&
         expect(match.source == CacheMatchSource::kGlobalRoot, "fallback source should be global root") &&
         expect(match.node_id == global_node, "fallback should select the global root node");
}

bool test_global_root_longest_prefix() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  cache.PublishGlobalRoot(
      make_identity({50, 60}),
      make_state(32, 42));
  const auto longer_node = cache.PublishGlobalRoot(
      make_identity({50, 60, 70, 80}),
      make_state(33, 43));

  CacheLookupRequest request;
  request.identity = make_identity({50, 60, 70, 80, 90});
  const auto match = cache.Lookup(request);

  return expect(longer_node != 0, "longer global root publish should succeed") &&
         expect(match.hit(), "longest global root should hit") &&
         expect(match.source == CacheMatchSource::kGlobalRoot, "match source should be global root") &&
         expect(match.node_id == longer_node, "lookup should choose the longest matching global root") &&
         expect(match.matched_token_count == 4, "lookup should report the longest prefix length");
}

bool test_lookup_refreshes_lru_eviction_order() {
  PrefixCache cache(/*max_bytes=*/13000);
  const auto first_node = cache.PublishConversationHead(
      "conv-lru-1",
      make_identity({10, 20, 30}),
      make_state(34, 44, 3000, 3000));
  const auto second_node = cache.PublishConversationHead(
      "conv-lru-2",
      make_identity({40, 50, 60}),
      make_state(35, 45, 3000, 3000));

  CacheLookupRequest refresh_request;
  refresh_request.identity = make_identity({10, 20, 30, 99});
  refresh_request.conversation_id = "conv-lru-1";
  const auto refreshed_match = cache.Lookup(refresh_request);

  const auto third_node = cache.PublishConversationHead(
      "conv-lru-3",
      make_identity({70, 80, 90}),
      make_state(36, 46, 3000, 3000));

  return expect(first_node != 0 && second_node != 0 && third_node != 0, "LRU test publishes should succeed") &&
         expect(refreshed_match.hit(), "lookup should refresh the oldest node") &&
         expect(cache.node_count() == 2, "budget pressure should evict exactly one node") &&
         expect(cache.CommittedHeadForConversation("conv-lru-1").has_value(),
                "recently touched node should survive eviction") &&
         expect(!cache.CommittedHeadForConversation("conv-lru-2").has_value(),
                "least recently used node should be evicted") &&
         expect(cache.CommittedHeadForConversation("conv-lru-3").has_value(),
                "newest node should remain resident");
}

bool test_serializer_miss() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  cache.PublishGlobalRoot(
      make_identity({7, 8, 9}, "chat-v1"),
      make_state(15, 25));

  CacheLookupRequest request;
  request.identity = make_identity({7, 8, 9, 10}, "chat-v2");
  const auto match = cache.Lookup(request);

  return expect(!match.hit(), "serializer revision mismatch should miss cache");
}

bool test_namespace_isolation() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  auto identity = make_identity({70, 71, 72});
  identity.tenant_namespace = "tenant-a";
  cache.PublishGlobalRoot(identity, make_state(36, 46));

  CacheLookupRequest request;
  request.identity = make_identity({70, 71, 72, 73});
  request.identity.tenant_namespace = "tenant-b";
  const auto match = cache.Lookup(request);

  return expect(!match.hit(), "cross-tenant lookup should miss");
}

bool test_dedup_identity_across_roles() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  const auto identity = make_identity({42, 43, 44});
  const auto conversation_node = cache.PublishConversationHead(
      "conv-3",
      identity,
      make_state(16, 26));
  const auto global_node = cache.PublishGlobalRoot(identity, make_state(17, 27));

  const auto view = cache.Describe(conversation_node);
  return expect(conversation_node == global_node, "same identity should deduplicate to the same node") &&
         expect(view.has_value(), "deduplicated node should remain describable") &&
         expect(view->is_global_root, "node should be marked global after global publish") &&
         expect(view->committed_conversations.size() == 1 && view->committed_conversations[0] == "conv-3",
                "node should retain conversation role after dedup");
}

bool test_eviction_by_bytes_clears_mappings() {
  PrefixCache cache(/*max_bytes=*/7000);
  const auto first_node = cache.PublishConversationHead(
      "conv-4",
      make_identity({1, 1, 1}),
      make_state(18, 28, 3000, 3000));
  const auto second_node = cache.PublishConversationHead(
      "conv-5",
      make_identity({2, 2, 2}),
      make_state(19, 29, 3000, 3000));

  return expect(first_node != 0 && second_node != 0, "publishes should succeed before eviction") &&
         expect(cache.node_count() == 1, "only one node should remain after eviction to budget") &&
         expect(!cache.CommittedHeadForConversation("conv-4").has_value(),
                "evicted conversation head mapping should be removed") &&
         expect(cache.CommittedHeadForConversation("conv-5").has_value(),
                "newest conversation head should remain after eviction");
}

bool test_cache_eviction_releases_owned_state() {
  ReusableStateArena arena(/*max_bytes=*/1 << 20);
  PrefixCache cache(/*max_bytes=*/7000, &arena);

  const auto first_state = arena.AllocateDescriptor(3000, 3000, "conv-6");
  const auto first_node = cache.PublishConversationHead(
      "conv-6",
      make_identity({3, 3, 3}),
      first_state);
  arena.Release(first_state);

  const auto second_state = arena.AllocateDescriptor(3000, 3000, "conv-7");
  const auto second_node = cache.PublishConversationHead(
      "conv-7",
      make_identity({4, 4, 4}),
      second_state);
  arena.Release(second_state);

  return expect(first_node != 0 && second_node != 0, "owned-state publishes should succeed") &&
         expect(cache.node_count() == 1, "cache should evict to budget with arena-managed state") &&
         expect(arena.allocation_count() == 2, "only one descriptor pair should remain after eviction") &&
         expect(!cache.CommittedHeadForConversation("conv-6").has_value(),
                "evicted owned-state conversation mapping should be removed");
}

bool test_replacing_node_state_releases_old_owned_state() {
  ReusableStateArena arena(/*max_bytes=*/1 << 20);
  PrefixCache cache(/*max_bytes=*/1 << 20, &arena);

  const auto first_state = arena.AllocateDescriptor(1024, 2048, "conv-8-first");
  const auto node_id = cache.PublishConversationHead(
      "conv-8",
      make_identity({8, 8, 8}),
      first_state);
  arena.Release(first_state);

  const auto second_state = arena.AllocateDescriptor(2048, 4096, "conv-8-second");
  const auto updated_node_id = cache.PublishConversationHead(
      "conv-8",
      make_identity({8, 8, 8}),
      second_state);
  arena.Release(second_state);

  return expect(node_id != 0 && updated_node_id == node_id, "cache should update the existing node for same identity") &&
         expect(arena.allocation_count() == 2, "old owned state should be released after replacement") &&
         expect(arena.current_bytes() == 6144, "arena bytes should reflect only the replacement descriptor");
}

bool test_boundary_token_id_round_trip() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  constexpr TokenId kBoundaryTokenId = 12345;
  const auto node_id = cache.PublishConversationHead(
      "conv-boundary-token",
      make_identity({6, 7, 8}),
      make_state(55, 65),
      kBoundaryTokenId);
  const auto entry = cache.Describe(node_id);
  const auto cached_boundary_token_id = cache.CopyBoundaryTokenId(node_id);

  return expect(node_id != 0, "boundary-token publish should succeed") &&
         expect(
             cached_boundary_token_id.has_value() &&
                 *cached_boundary_token_id == kBoundaryTokenId,
             "boundary token id should round-trip through the cache") &&
         expect(
             entry.has_value() &&
                 entry->boundary_token_id.has_value() &&
                 *entry->boundary_token_id == kBoundaryTokenId,
             "described cache entry should expose the cached boundary token id") &&
         expect(
             !entry->has_boundary_logits && entry->boundary_logits_count == 0,
             "boundary token id should not require cached boundary logits");
}

bool test_snapshot_publish_and_restore_round_trip() {
  auto source = RequestExecutionContext::Create(make_request_config());
  auto restored = RequestExecutionContext::Create(make_request_config());
  if (!source || !restored || !source->valid() || !restored->valid()) {
    std::cout << "prefix_cache_test: SKIP snapshot round-trip (no CUDA device available)\n";
    return true;
  }

  constexpr std::size_t kTokenCount = 5;
  if (!expect(source->SetSequenceLength(kTokenCount), "source request should allocate the prefix token window")) {
    return false;
  }

  std::vector<__nv_bfloat16> key_host(source->key_cache()->numel());
  std::vector<__nv_bfloat16> value_host(source->value_cache()->numel());
  for (std::size_t i = 0; i < key_host.size(); ++i) {
    key_host[i] = __float2bfloat16(static_cast<float>((static_cast<int>(i % 29) - 14)) / 9.0f);
    value_host[i] = __float2bfloat16(static_cast<float>((static_cast<int>(i % 31) - 15)) / 7.0f);
  }
  std::vector<float> conv_host(source->mamba_conv_state()->numel(), 0.0f);
  std::vector<float> ssm_host(source->mamba_state()->numel(), 0.0f);
  for (std::size_t i = 0; i < conv_host.size(); ++i) {
    conv_host[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) / 5.0f;
  }
  for (std::size_t i = 0; i < ssm_host.size(); ++i) {
    ssm_host[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) / 3.0f;
  }

  if (!expect(
          source->key_cache()->CopyFromHost(key_host.data(), key_host.size()),
          "source key cache should upload") ||
      !expect(
          source->value_cache()->CopyFromHost(value_host.data(), value_host.size()),
          "source value cache should upload") ||
      !expect(
          source->mamba_conv_state()->CopyFromHost(conv_host.data(), conv_host.size()),
          "source conv state should upload") ||
      !expect(
          source->mamba_state()->CopyFromHost(ssm_host.data(), ssm_host.size()),
          "source ssm state should upload")) {
    return false;
  }

  ReusableStateArena arena(/*max_bytes=*/1 << 20);
  PrefixCache cache(/*max_bytes=*/1 << 20, &arena);
  const SerializedPromptIdentity identity = make_identity({7, 8, 9, 10, 11});
  const std::vector<float> boundary_logits = {0.5f, -1.25f, 3.75f, 2.0f};
  const auto node_id = cache.PublishConversationHeadSnapshot(
      "conv-snapshot",
      identity,
      *source,
      "prefix-cache-test",
      &boundary_logits);
  if (!expect(node_id != 0, "snapshot-backed conversation publish should succeed")) {
    return false;
  }
  if (!expect(arena.allocation_count() == 2, "cache-owned snapshot should leave one descriptor pair resident")) {
    return false;
  }

  CacheLookupRequest request;
  request.identity = make_identity({7, 8, 9, 10, 11, 12});
  request.conversation_id = "conv-snapshot";
  const auto match = cache.Lookup(request);
  if (!expect(match.hit(), "snapshot-backed conversation publish should be discoverable by lookup") ||
      !expect(match.node_id == node_id, "lookup should return the published snapshot node") ||
      !expect(match.matched_token_count == kTokenCount, "lookup should report the cached token count")) {
    return false;
  }
  const auto cached_boundary_logits = cache.CopyBoundaryLogits(node_id);
  if (!expect(
          cache.RestoreMatchState(match, *restored),
          "lookup match should restore the cached request state")) {
    return false;
  }

  std::vector<__nv_bfloat16> restored_key(key_host.size());
  std::vector<__nv_bfloat16> restored_value(value_host.size());
  std::vector<float> restored_conv(conv_host.size(), 0.0f);
  std::vector<float> restored_ssm(ssm_host.size(), 0.0f);
  if (!expect(
          restored->key_cache()->CopyToHost(restored_key.data(), restored_key.size()),
          "restored key cache should download") ||
      !expect(
          restored->value_cache()->CopyToHost(restored_value.data(), restored_value.size()),
          "restored value cache should download") ||
      !expect(
          restored->mamba_conv_state()->CopyToHost(restored_conv.data(), restored_conv.size()),
          "restored conv state should download") ||
      !expect(
          restored->mamba_state()->CopyToHost(restored_ssm.data(), restored_ssm.size()),
          "restored ssm state should download")) {
    return false;
  }

  cache.Clear();

  return expect(
             restored->sequence_length() == kTokenCount && restored->decode_position() == kTokenCount,
             "restored snapshot should resume at the cached token boundary") &&
         expect(
             source->allocated_kv_pages(0) == 2 && source->allocated_kv_pages(1) == 2,
             "clearing cached snapshots should not release the live request's KV pages") &&
         expect(same_bf16(restored_key, key_host), "restored key cache should match the cached snapshot exactly") &&
         expect(
             same_bf16(restored_value, value_host),
             "restored value cache should match the cached snapshot exactly") &&
         expect(
             cached_boundary_logits.has_value() && *cached_boundary_logits == boundary_logits,
             "cached boundary logits should survive snapshot-backed publish and lookup") &&
         expect(restored_conv == conv_host, "restored conv state should match the cached snapshot exactly") &&
         expect(restored_ssm == ssm_host, "restored ssm state should match the cached snapshot exactly") &&
         expect(arena.current_bytes() == 0, "clearing the cache should release the cached snapshot bytes");
}

bool test_snapshot_publish_retries_by_replacing_unique_conversation_head() {
  auto first = RequestExecutionContext::Create(make_request_config());
  auto second = RequestExecutionContext::Create(make_request_config());
  auto restored = RequestExecutionContext::Create(make_request_config());
  if (!first || !second || !restored || !first->valid() || !second->valid() || !restored->valid()) {
    std::cout << "prefix_cache_test: SKIP replacement retry (no CUDA device available)\n";
    return true;
  }

  constexpr std::size_t kFirstTokenCount = 4;
  constexpr std::size_t kSecondTokenCount = 5;
  if (!expect(first->SetSequenceLength(kFirstTokenCount), "first snapshot source should allocate") ||
      !expect(second->SetSequenceLength(kSecondTokenCount), "second snapshot source should allocate")) {
    return false;
  }

  const std::size_t second_descriptor_bytes =
      nemotron::RequiredKvSnapshotBytes(*second) +
      nemotron::RequiredMambaSnapshotBytes(*second);
  if (!expect(second_descriptor_bytes != 0, "replacement retry test should compute a non-zero descriptor size")) {
    return false;
  }

  ReusableStateArena arena(second_descriptor_bytes);
  PrefixCache cache(/*max_bytes=*/second_descriptor_bytes + 4096, &arena);

  const std::vector<float> first_boundary_logits = {1.0f, 2.0f, 3.0f};
  const auto first_node = cache.PublishConversationHeadSnapshot(
      "conv-retry",
      make_identity({1, 2, 3, 4}),
      *first,
      "prefix-cache-retry/first",
      &first_boundary_logits);
  if (!expect(first_node != 0, "first committed-head snapshot should publish")) {
    return false;
  }

  const std::vector<float> second_boundary_logits = {-2.0f, 4.0f, -8.0f, 16.0f};
  const auto second_node = cache.PublishConversationHeadSnapshot(
      "conv-retry",
      make_identity({1, 2, 3, 4, 5}),
      *second,
      "prefix-cache-retry/second",
      &second_boundary_logits);
  if (!expect(
          second_node != 0,
          "second committed-head snapshot should retry after replacing the unique old head") ||
      !expect(
          cache.CommittedHeadForConversation("conv-retry").has_value() &&
              *cache.CommittedHeadForConversation("conv-retry") == second_node,
          "conversation should point at the replacement head") ||
      !expect(!cache.Describe(first_node).has_value(), "old committed head should be removed after replacement") ||
      !expect(arena.current_bytes() == second_descriptor_bytes, "arena bytes should reflect only the replacement descriptor")) {
    return false;
  }

  CacheLookupRequest request;
  request.identity = make_identity({1, 2, 3, 4, 5, 6});
  request.conversation_id = "conv-retry";
  const auto match = cache.Lookup(request);
  if (!expect(match.hit(), "replacement committed head should still be discoverable by lookup") ||
      !expect(match.node_id == second_node, "lookup should resolve to the replacement committed head") ||
      !expect(match.matched_token_count == kSecondTokenCount, "replacement match length should equal the new prefix")) {
    return false;
  }

  const auto cached_boundary_logits = cache.CopyBoundaryLogits(second_node);
  return expect(
             cached_boundary_logits.has_value() && *cached_boundary_logits == second_boundary_logits,
             "replacement committed head should preserve its new boundary logits") &&
         expect(
             cache.RestoreMatchState(match, *restored),
             "replacement committed head should restore request state") &&
         expect(
             restored->sequence_length() == kSecondTokenCount &&
                 restored->decode_position() == kSecondTokenCount,
             "replacement restore should resume at the updated prefix boundary");
}

}  // namespace

int main() {
  const bool ok =
      test_conversation_head_hit() &&
      test_exact_conversation_cache_hit_reports_full_length() &&
      test_global_root_fallback() &&
      test_conversation_fallback_to_global_root() &&
      test_global_root_longest_prefix() &&
      test_lookup_refreshes_lru_eviction_order() &&
      test_serializer_miss() &&
      test_namespace_isolation() &&
      test_dedup_identity_across_roles() &&
      test_eviction_by_bytes_clears_mappings() &&
      test_cache_eviction_releases_owned_state() &&
      test_replacing_node_state_releases_old_owned_state() &&
      test_boundary_token_id_round_trip() &&
      test_snapshot_publish_and_restore_round_trip() &&
      test_snapshot_publish_retries_by_replacing_unique_conversation_head();

  if (!ok) {
    return 1;
  }
  std::cout << "prefix_cache_test: PASS\n";
  return 0;
}
