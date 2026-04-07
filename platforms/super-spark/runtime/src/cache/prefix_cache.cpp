#include "nemotron/prefix_cache.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nemotron {
namespace {

std::uint64_t fnv1a_append(std::uint64_t hash, std::uint64_t value) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ull;
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}

std::uint64_t hash_string(std::uint64_t hash, const std::string& value) {
  for (unsigned char ch : value) {
    hash = fnv1a_append(hash, ch);
  }
  return hash;
}

std::uint64_t fingerprint_identity(const SerializedPromptIdentity& identity) {
  constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
  std::uint64_t hash = kFnvOffsetBasis;
  hash = hash_string(hash, identity.tenant_namespace);
  hash = hash_string(hash, identity.tokenizer_revision);
  hash = hash_string(hash, identity.serializer_revision);
  hash = hash_string(hash, identity.model_revision);
  hash = fnv1a_append(hash, identity.reasoning_mode ? 1u : 0u);
  hash = fnv1a_append(hash, static_cast<std::uint64_t>(identity.token_ids.size()));
  for (TokenId token : identity.token_ids) {
    hash = fnv1a_append(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(token)));
  }
  return hash;
}

bool same_identity(const SerializedPromptIdentity& lhs, const SerializedPromptIdentity& rhs) {
  return lhs.reasoning_mode == rhs.reasoning_mode &&
         lhs.tenant_namespace == rhs.tenant_namespace &&
         lhs.tokenizer_revision == rhs.tokenizer_revision &&
         lhs.serializer_revision == rhs.serializer_revision &&
         lhs.model_revision == rhs.model_revision &&
         lhs.token_ids == rhs.token_ids;
}

std::vector<std::string> sorted_strings(const std::unordered_set<std::string>& values) {
  std::vector<std::string> result(values.begin(), values.end());
  std::sort(result.begin(), result.end());
  return result;
}

std::size_t node_bytes(const SerializedPromptIdentity& identity, const ReusableStateDescriptor& state) {
  return identity.token_ids.size() * sizeof(TokenId) +
         identity.tenant_namespace.size() +
         identity.tokenizer_revision.size() +
         identity.serializer_revision.size() +
         identity.model_revision.size() +
         state.total_bytes();
}

}  // namespace

struct PrefixCache::Impl {
  struct CacheNode {
    PrefixNodeId id = 0;
    SerializedPromptIdentity identity;
    ReusableStateDescriptor state;
    std::uint64_t fingerprint = 0;
    bool is_global_root = false;
    std::unordered_set<std::string> committed_conversations;
    std::unordered_set<std::string> prompt_conversations;
    std::uint64_t last_access_tick = 0;
    std::size_t total_bytes = 0;
  };

  std::size_t current_bytes = 0;
  std::uint64_t tick = 1;
  PrefixNodeId next_node_id = 1;
  std::unordered_map<PrefixNodeId, CacheNode> nodes;
  std::unordered_multimap<std::uint64_t, PrefixNodeId> nodes_by_fingerprint;
  std::unordered_map<std::string, PrefixNodeId> committed_heads;
  std::unordered_map<std::string, PrefixNodeId> prompt_heads;
  ReusableStateArena* state_arena = nullptr;
  bool enabled = true;

  ~Impl() {
    Reset();
  }

  void Reset() {
    if (state_arena != nullptr) {
      for (const auto& [_, node] : nodes) {
        state_arena->Release(node.state);
      }
    }
    current_bytes = 0;
    nodes.clear();
    nodes_by_fingerprint.clear();
    committed_heads.clear();
    prompt_heads.clear();
  }

  bool NodeHasRoles(const CacheNode& node) const {
    return node.is_global_root ||
           !node.committed_conversations.empty() ||
           !node.prompt_conversations.empty();
  }

  void EraseFromFingerprintIndex(const CacheNode& node) {
    auto range = nodes_by_fingerprint.equal_range(node.fingerprint);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == node.id) {
        nodes_by_fingerprint.erase(it);
        return;
      }
    }
  }

  void RemoveNode(PrefixNodeId node_id) {
    auto node_it = nodes.find(node_id);
    if (node_it == nodes.end()) {
      return;
    }

    CacheNode& node = node_it->second;
    for (const std::string& conversation_id : node.committed_conversations) {
      auto head_it = committed_heads.find(conversation_id);
      if (head_it != committed_heads.end() && head_it->second == node_id) {
        committed_heads.erase(head_it);
      }
    }
    for (const std::string& conversation_id : node.prompt_conversations) {
      auto head_it = prompt_heads.find(conversation_id);
      if (head_it != prompt_heads.end() && head_it->second == node_id) {
        prompt_heads.erase(head_it);
      }
    }

    if (state_arena != nullptr) {
      state_arena->Release(node.state);
    }
    EraseFromFingerprintIndex(node);
    current_bytes -= node.total_bytes;
    nodes.erase(node_it);
  }

  bool RetainState(const ReusableStateDescriptor& state) {
    if (state_arena == nullptr) {
      return true;
    }
    return state_arena->Retain(state);
  }
};

bool CacheMatch::hit() const {
  return source != CacheMatchSource::kNone;
}

PrefixCache::PrefixCache(std::size_t max_bytes)
    : PrefixCache(max_bytes, nullptr) {}

PrefixCache::PrefixCache(std::size_t max_bytes, ReusableStateArena* state_arena)
    : max_bytes_(max_bytes), impl_(std::make_unique<Impl>()) {
  impl_->state_arena = state_arena;
}

PrefixCache::~PrefixCache() = default;
PrefixCache::PrefixCache(PrefixCache&&) noexcept = default;
PrefixCache& PrefixCache::operator=(PrefixCache&&) noexcept = default;

PrefixNodeId PrefixCache::PublishConversationHead(
    const std::string& conversation_id,
    ConversationCheckpointKind checkpoint_kind,
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state) {
  if (!impl_->enabled) {
    return 0;
  }
  const PrefixNodeId node_id = FindOrCreateNode(identity, state);
  if (node_id == 0) {
    return 0;
  }
  AssignConversationHead(node_id, conversation_id, checkpoint_kind);
  EvictToBudget();
  return Describe(node_id).has_value() ? node_id : 0;
}

PrefixNodeId PrefixCache::PublishGlobalRoot(
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state) {
  if (!impl_->enabled) {
    return 0;
  }
  const PrefixNodeId node_id = FindOrCreateNode(identity, state);
  if (node_id == 0) {
    return 0;
  }
  AssignGlobalRoot(node_id);
  EvictToBudget();
  return Describe(node_id).has_value() ? node_id : 0;
}

CacheMatch PrefixCache::Lookup(const CacheLookupRequest& request) {
  if (!impl_->enabled) {
    return {};
  }
  auto build_match = [&](PrefixNodeId node_id, CacheMatchSource source) -> CacheMatch {
    const Impl::CacheNode& node = impl_->nodes.at(node_id);
    Touch(node_id);
    return CacheMatch{
        source,
        node.id,
        node.identity.token_ids.size(),
        node.state,
    };
  };

  if (request.conversation_id.has_value()) {
    if (request.allow_prompt_head) {
      auto prompt_it = impl_->prompt_heads.find(*request.conversation_id);
      if (prompt_it != impl_->prompt_heads.end() &&
          PrefixMatches(impl_->nodes.at(prompt_it->second).identity, request.identity)) {
        return build_match(prompt_it->second, CacheMatchSource::kConversationPromptHead);
      }
    }

    auto committed_it = impl_->committed_heads.find(*request.conversation_id);
    if (committed_it != impl_->committed_heads.end() &&
        PrefixMatches(impl_->nodes.at(committed_it->second).identity, request.identity)) {
      return build_match(committed_it->second, CacheMatchSource::kConversationCommittedHead);
    }
  }

  PrefixNodeId best_node_id = 0;
  std::size_t best_match_length = 0;
  for (const auto& [node_id, node] : impl_->nodes) {
    if (!node.is_global_root) {
      continue;
    }
    if (!PrefixMatches(node.identity, request.identity)) {
      continue;
    }
    const std::size_t match_length = node.identity.token_ids.size();
    if (match_length > best_match_length) {
      best_match_length = match_length;
      best_node_id = node_id;
    }
  }

  if (best_node_id != 0) {
    return build_match(best_node_id, CacheMatchSource::kGlobalRoot);
  }
  return {};
}

void PrefixCache::SetEnabled(bool enabled) {
  if (impl_->enabled == enabled) {
    return;
  }
  if (!enabled) {
    impl_->Reset();
  }
  impl_->enabled = enabled;
}

bool PrefixCache::enabled() const {
  return impl_->enabled;
}

void PrefixCache::Clear() {
  impl_->Reset();
}

std::optional<PrefixNodeId> PrefixCache::CommittedHeadForConversation(const std::string& conversation_id) const {
  auto it = impl_->committed_heads.find(conversation_id);
  if (it == impl_->committed_heads.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<PrefixNodeId> PrefixCache::PromptHeadForConversation(const std::string& conversation_id) const {
  auto it = impl_->prompt_heads.find(conversation_id);
  if (it == impl_->prompt_heads.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<CacheEntryView> PrefixCache::Describe(PrefixNodeId node_id) const {
  auto it = impl_->nodes.find(node_id);
  if (it == impl_->nodes.end()) {
    return std::nullopt;
  }
  const Impl::CacheNode& node = it->second;
  return CacheEntryView{
      node.id,
      node.identity,
      node.state,
      node.is_global_root,
      sorted_strings(node.committed_conversations),
      sorted_strings(node.prompt_conversations),
      node.last_access_tick,
      node.total_bytes,
  };
}

std::size_t PrefixCache::current_bytes() const {
  return impl_->current_bytes;
}

std::size_t PrefixCache::max_bytes() const {
  return max_bytes_;
}

std::size_t PrefixCache::node_count() const {
  return impl_->nodes.size();
}

std::size_t PrefixCache::global_root_count() const {
  std::size_t count = 0;
  for (const auto& [_, node] : impl_->nodes) {
    if (node.is_global_root) {
      ++count;
    }
  }
  return count;
}

PrefixNodeId PrefixCache::FindOrCreateNode(
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state) {
  const std::uint64_t fingerprint = fingerprint_identity(identity);
  auto range = impl_->nodes_by_fingerprint.equal_range(fingerprint);
  for (auto it = range.first; it != range.second; ++it) {
    Impl::CacheNode& node = impl_->nodes.at(it->second);
    if (!same_identity(node.identity, identity)) {
      continue;
    }
    if (!(node.state == state)) {
      if (!impl_->RetainState(state)) {
        return 0;
      }
      if (impl_->state_arena != nullptr) {
        impl_->state_arena->Release(node.state);
      }
      impl_->current_bytes -= node.total_bytes;
      node.state = state;
      node.total_bytes = node_bytes(node.identity, node.state);
      impl_->current_bytes += node.total_bytes;
    }
    Touch(node.id);
    return node.id;
  }

  if (!impl_->RetainState(state)) {
    return 0;
  }
  Impl::CacheNode node;
  node.id = impl_->next_node_id++;
  node.identity = identity;
  node.state = state;
  node.fingerprint = fingerprint;
  node.total_bytes = node_bytes(identity, state);
  node.last_access_tick = impl_->tick++;
  impl_->current_bytes += node.total_bytes;

  impl_->nodes_by_fingerprint.emplace(node.fingerprint, node.id);
  impl_->nodes.emplace(node.id, std::move(node));
  return node.id;
}

void PrefixCache::Touch(PrefixNodeId node_id) {
  auto it = impl_->nodes.find(node_id);
  if (it == impl_->nodes.end()) {
    return;
  }
  it->second.last_access_tick = impl_->tick++;
}

void PrefixCache::AssignConversationHead(
    PrefixNodeId node_id,
    const std::string& conversation_id,
    ConversationCheckpointKind checkpoint_kind) {
  auto* head_map =
      checkpoint_kind == ConversationCheckpointKind::kCommittedHead ? &impl_->committed_heads : &impl_->prompt_heads;
  auto existing_it = head_map->find(conversation_id);
  if (existing_it != head_map->end() && existing_it->second != node_id) {
    RemoveConversationRole(existing_it->second, conversation_id, checkpoint_kind);
  }
  (*head_map)[conversation_id] = node_id;

  Impl::CacheNode& node = impl_->nodes.at(node_id);
  if (checkpoint_kind == ConversationCheckpointKind::kCommittedHead) {
    node.committed_conversations.insert(conversation_id);
  } else {
    node.prompt_conversations.insert(conversation_id);
  }
  Touch(node_id);
}

void PrefixCache::AssignGlobalRoot(PrefixNodeId node_id) {
  Impl::CacheNode& node = impl_->nodes.at(node_id);
  node.is_global_root = true;
  Touch(node_id);
}

void PrefixCache::RemoveConversationRole(
    PrefixNodeId node_id,
    const std::string& conversation_id,
    ConversationCheckpointKind checkpoint_kind) {
  auto node_it = impl_->nodes.find(node_id);
  if (node_it == impl_->nodes.end()) {
    return;
  }
  Impl::CacheNode& node = node_it->second;
  if (checkpoint_kind == ConversationCheckpointKind::kCommittedHead) {
    node.committed_conversations.erase(conversation_id);
  } else {
    node.prompt_conversations.erase(conversation_id);
  }
  if (!impl_->NodeHasRoles(node)) {
    impl_->RemoveNode(node_id);
  }
}

void PrefixCache::EvictToBudget() {
  while (impl_->current_bytes > max_bytes_ && !impl_->nodes.empty()) {
    auto victim_it = impl_->nodes.end();
    std::uint64_t oldest_tick = std::numeric_limits<std::uint64_t>::max();
    for (auto it = impl_->nodes.begin(); it != impl_->nodes.end(); ++it) {
      if (it->second.last_access_tick < oldest_tick) {
        oldest_tick = it->second.last_access_tick;
        victim_it = it;
      }
    }
    if (victim_it == impl_->nodes.end()) {
      break;
    }
    impl_->RemoveNode(victim_it->first);
  }
}

bool PrefixCache::PrefixMatches(
    const SerializedPromptIdentity& cached,
    const SerializedPromptIdentity& request) const {
  if (cached.reasoning_mode != request.reasoning_mode ||
      cached.tenant_namespace != request.tenant_namespace ||
      cached.tokenizer_revision != request.tokenizer_revision ||
      cached.serializer_revision != request.serializer_revision ||
      cached.model_revision != request.model_revision) {
    return false;
  }
  return PrefixMatchLength(cached.token_ids, request.token_ids) == cached.token_ids.size();
}

std::size_t PrefixCache::PrefixMatchLength(
    const std::vector<TokenId>& cached_tokens,
    const std::vector<TokenId>& request_tokens) const {
  if (cached_tokens.size() > request_tokens.size()) {
    return 0;
  }
  std::size_t match_length = 0;
  while (match_length < cached_tokens.size() &&
         cached_tokens[match_length] == request_tokens[match_length]) {
    ++match_length;
  }
  return match_length;
}

}  // namespace nemotron
