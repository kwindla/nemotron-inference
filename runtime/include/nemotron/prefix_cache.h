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
class RequestExecutionContext;

struct SerializedPromptIdentity {
  std::vector<TokenId> token_ids;
  std::string tenant_namespace;
  std::string tokenizer_revision;
  std::string serializer_revision;
  std::string model_revision;
  bool reasoning_mode = false;
};

enum class CacheMatchSource {
  kNone,
  kConversationCommittedHead,
  kGlobalRoot,
};

struct CacheLookupRequest {
  SerializedPromptIdentity identity;
  std::optional<std::string> conversation_id;
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
  std::uint64_t last_access_tick = 0;
  std::size_t total_bytes = 0;
  bool has_boundary_logits = false;
  std::size_t boundary_logits_count = 0;
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

  // Conversation heads are committed boundaries only. Uncommitted prompt-head
  // lookups were removed from the production API because the runtime does not
  // use them on the hot path.
  PrefixNodeId PublishConversationHead(
      const std::string& conversation_id,
      const SerializedPromptIdentity& identity,
      const ReusableStateDescriptor& state,
      const std::vector<float>* boundary_logits = nullptr);
  PrefixNodeId PublishConversationHeadSnapshot(
      const std::string& conversation_id,
      const SerializedPromptIdentity& identity,
      const RequestExecutionContext& request_context,
      const std::string& state_label,
      const std::vector<float>* boundary_logits = nullptr);

  PrefixNodeId PublishGlobalRoot(
      const SerializedPromptIdentity& identity,
      const ReusableStateDescriptor& state,
      const std::vector<float>* boundary_logits = nullptr);
  PrefixNodeId PublishGlobalRootSnapshot(
      const SerializedPromptIdentity& identity,
      const RequestExecutionContext& request_context,
      const std::string& state_label,
      const std::vector<float>* boundary_logits = nullptr);

  CacheMatch Lookup(const CacheLookupRequest& request);
  bool RestoreMatchState(const CacheMatch& match, RequestExecutionContext& request_context) const;
  std::optional<std::vector<float>> CopyBoundaryLogits(PrefixNodeId node_id) const;
  void SetEnabled(bool enabled);
  bool enabled() const;
  void Clear();

  std::optional<PrefixNodeId> CommittedHeadForConversation(const std::string& conversation_id) const;
  std::optional<CacheEntryView> Describe(PrefixNodeId node_id) const;

  std::size_t current_bytes() const;
  std::size_t max_bytes() const;
  std::size_t node_count() const;
  std::size_t global_root_count() const;

 private:
  struct Impl;

  PrefixNodeId FindOrCreateNode(
      const SerializedPromptIdentity& identity,
      const ReusableStateDescriptor& state,
      const std::vector<float>* boundary_logits);
  void Touch(PrefixNodeId node_id);
  void AssignConversationHead(PrefixNodeId node_id, const std::string& conversation_id);
  void AssignGlobalRoot(PrefixNodeId node_id);
  void RemoveConversationRole(PrefixNodeId node_id, const std::string& conversation_id);
  void EvictToBudget();
  bool PrefixMatches(const SerializedPromptIdentity& cached, const SerializedPromptIdentity& request) const;
  std::size_t PrefixMatchLength(
      const std::vector<TokenId>& cached_tokens,
      const std::vector<TokenId>& request_tokens) const;

  std::size_t max_bytes_ = 0;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
