#include "nemotron/prefix_cache.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::CacheLookupRequest;
using nemotron::CacheMatchSource;
using nemotron::ConversationCheckpointKind;
using nemotron::PrefixCache;
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

bool test_conversation_head_hit() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  const auto node_id = cache.PublishConversationHead(
      "conv-1",
      ConversationCheckpointKind::kCommittedHead,
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

bool test_prompt_head_preferred() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  cache.PublishConversationHead(
      "conv-2",
      ConversationCheckpointKind::kCommittedHead,
      make_identity({10, 20, 30}),
      make_state(12, 22));
  const auto prompt_node = cache.PublishConversationHead(
      "conv-2",
      ConversationCheckpointKind::kPromptHead,
      make_identity({10, 20, 30, 40, 50}),
      make_state(13, 23));

  CacheLookupRequest request;
  request.identity = make_identity({10, 20, 30, 40, 50});
  request.conversation_id = "conv-2";
  request.allow_prompt_head = true;
  const auto match = cache.Lookup(request);

  return expect(prompt_node != 0, "prompt head publish should succeed") &&
         expect(match.hit(), "prompt head should hit") &&
         expect(match.source == CacheMatchSource::kConversationPromptHead, "match source should be prompt head") &&
         expect(match.node_id == prompt_node, "matched node should be prompt head");
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
      ConversationCheckpointKind::kCommittedHead,
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

bool test_prompt_head_requires_opt_in() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  const auto committed_node = cache.PublishConversationHead(
      "conv-2b",
      ConversationCheckpointKind::kCommittedHead,
      make_identity({10, 20, 30}),
      make_state(34, 44));
  cache.PublishConversationHead(
      "conv-2b",
      ConversationCheckpointKind::kPromptHead,
      make_identity({10, 20, 30, 40, 50}),
      make_state(35, 45));

  CacheLookupRequest request;
  request.identity = make_identity({10, 20, 30, 40, 50});
  request.conversation_id = "conv-2b";
  const auto match = cache.Lookup(request);

  return expect(committed_node != 0, "committed head publish should succeed") &&
         expect(match.hit(), "committed head should still hit when prompt head opt-in is off") &&
         expect(match.source == CacheMatchSource::kConversationCommittedHead,
                "prompt head should not be used unless explicitly allowed") &&
         expect(match.node_id == committed_node, "lookup should fall back to committed head without opt-in");
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
      ConversationCheckpointKind::kCommittedHead,
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
      ConversationCheckpointKind::kCommittedHead,
      make_identity({1, 1, 1}),
      make_state(18, 28, 3000, 3000));
  const auto second_node = cache.PublishConversationHead(
      "conv-5",
      ConversationCheckpointKind::kCommittedHead,
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
      ConversationCheckpointKind::kCommittedHead,
      make_identity({3, 3, 3}),
      first_state);
  arena.Release(first_state);

  const auto second_state = arena.AllocateDescriptor(3000, 3000, "conv-7");
  const auto second_node = cache.PublishConversationHead(
      "conv-7",
      ConversationCheckpointKind::kCommittedHead,
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
      ConversationCheckpointKind::kCommittedHead,
      make_identity({8, 8, 8}),
      first_state);
  arena.Release(first_state);

  const auto second_state = arena.AllocateDescriptor(2048, 4096, "conv-8-second");
  const auto updated_node_id = cache.PublishConversationHead(
      "conv-8",
      ConversationCheckpointKind::kCommittedHead,
      make_identity({8, 8, 8}),
      second_state);
  arena.Release(second_state);

  return expect(node_id != 0 && updated_node_id == node_id, "cache should update the existing node for same identity") &&
         expect(arena.allocation_count() == 2, "old owned state should be released after replacement") &&
         expect(arena.current_bytes() == 6144, "arena bytes should reflect only the replacement descriptor");
}

}  // namespace

int main() {
  const bool ok =
      test_conversation_head_hit() &&
      test_prompt_head_preferred() &&
      test_global_root_fallback() &&
      test_conversation_fallback_to_global_root() &&
      test_global_root_longest_prefix() &&
      test_prompt_head_requires_opt_in() &&
      test_serializer_miss() &&
      test_namespace_isolation() &&
      test_dedup_identity_across_roles() &&
      test_eviction_by_bytes_clears_mappings() &&
      test_cache_eviction_releases_owned_state() &&
      test_replacing_node_state_releases_old_owned_state();

  if (!ok) {
    return 1;
  }
  std::cout << "prefix_cache_test: PASS\n";
  return 0;
}
