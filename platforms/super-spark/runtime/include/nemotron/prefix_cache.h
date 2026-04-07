#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nemotron/reusable_state.h"

namespace nemotron {

using TokenId = std::int32_t;
using PrefixNodeId = std::uint64_t;

struct SerializedPromptIdentity {
  std::vector<TokenId> token_ids;
  std::string tenant_namespace;
  std::string tokenizer_revision;
  std::string serializer_revision;
  std::string model_revision;
  bool reasoning_mode = false;
};

enum class ConversationCheckpointKind {
  kCommittedHead,
  kPromptHead,
};

enum class CacheMatchSource {
  kNone,
  kConversationCommittedHead,
  kConversationPromptHead,
  kGlobalRoot,
};

struct CacheLookupRequest {
  SerializedPromptIdentity identity;
  std::optional<std::string> conversation_id;
  bool allow_prompt_head = false;
};

struct CacheMatch {
  CacheMatchSource source = CacheMatchSource::kNone;
  PrefixNodeId node_id = 0;
  std::size_t matched_token_count = 0;
  ReusableStateDescriptor state;

  bool hit() const;
};

struct CacheEntryView {
  PrefixNodeId node_id = 0;
  SerializedPromptIdentity identity;
  ReusableStateDescriptor state;
  bool is_global_root = false;
  std::vector<std::string> committed_conversations;
  std::vector<std::string> prompt_conversations;
  std::uint64_t last_access_tick = 0;
  std::size_t total_bytes = 0;
};

class PrefixCache {
 public:
  explicit PrefixCache(std::size_t max_bytes);
  PrefixCache(std::size_t max_bytes, ReusableStateArena* state_arena);
  ~PrefixCache();

  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;
  PrefixCache(PrefixCache&&) noexcept;
  PrefixCache& operator=(PrefixCache&&) noexcept;

  PrefixNodeId PublishConversationHead(
      const std::string& conversation_id,
      ConversationCheckpointKind checkpoint_kind,
      const SerializedPromptIdentity& identity,
      const ReusableStateDescriptor& state);

  PrefixNodeId PublishGlobalRoot(
      const SerializedPromptIdentity& identity,
      const ReusableStateDescriptor& state);

  CacheMatch Lookup(const CacheLookupRequest& request);
  void SetEnabled(bool enabled);
  bool enabled() const;
  void Clear();

  std::optional<PrefixNodeId> CommittedHeadForConversation(const std::string& conversation_id) const;
  std::optional<PrefixNodeId> PromptHeadForConversation(const std::string& conversation_id) const;
  std::optional<CacheEntryView> Describe(PrefixNodeId node_id) const;

  std::size_t current_bytes() const;
  std::size_t max_bytes() const;
  std::size_t node_count() const;
  std::size_t global_root_count() const;

 private:
  struct Impl;

  PrefixNodeId FindOrCreateNode(
      const SerializedPromptIdentity& identity,
      const ReusableStateDescriptor& state);
  void Touch(PrefixNodeId node_id);
  void AssignConversationHead(
      PrefixNodeId node_id,
      const std::string& conversation_id,
      ConversationCheckpointKind checkpoint_kind);
  void AssignGlobalRoot(PrefixNodeId node_id);
  void RemoveConversationRole(
      PrefixNodeId node_id,
      const std::string& conversation_id,
      ConversationCheckpointKind checkpoint_kind);
  void EvictToBudget();
  bool PrefixMatches(const SerializedPromptIdentity& cached, const SerializedPromptIdentity& request) const;
  std::size_t PrefixMatchLength(
      const std::vector<TokenId>& cached_tokens,
      const std::vector<TokenId>& request_tokens) const;

  std::size_t max_bytes_ = 0;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
