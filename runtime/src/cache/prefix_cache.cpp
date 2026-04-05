#include "nemotron/prefix_cache.h"

#include "nemotron/state_snapshot.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

struct IdentityScopeKey {
  std::string tenant_namespace;
  std::string tokenizer_revision;
  std::string serializer_revision;
  std::string model_revision;
  bool reasoning_mode = false;
};

bool operator==(const IdentityScopeKey& lhs, const IdentityScopeKey& rhs) {
  return lhs.reasoning_mode == rhs.reasoning_mode &&
         lhs.tenant_namespace == rhs.tenant_namespace &&
         lhs.tokenizer_revision == rhs.tokenizer_revision &&
         lhs.serializer_revision == rhs.serializer_revision &&
         lhs.model_revision == rhs.model_revision;
}

struct IdentityScopeKeyHash {
  std::size_t operator()(const IdentityScopeKey& key) const {
    constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
    std::uint64_t hash = kFnvOffsetBasis;
    hash = hash_string(hash, key.tenant_namespace);
    hash = hash_string(hash, key.tokenizer_revision);
    hash = hash_string(hash, key.serializer_revision);
    hash = hash_string(hash, key.model_revision);
    hash = fnv1a_append(hash, key.reasoning_mode ? 1u : 0u);
    return static_cast<std::size_t>(hash);
  }
};

IdentityScopeKey make_identity_scope_key(const SerializedPromptIdentity& identity) {
  return IdentityScopeKey{
      identity.tenant_namespace,
      identity.tokenizer_revision,
      identity.serializer_revision,
      identity.model_revision,
      identity.reasoning_mode,
  };
}

std::vector<std::string> sorted_strings(const std::unordered_set<std::string>& values) {
  std::vector<std::string> result(values.begin(), values.end());
  std::sort(result.begin(), result.end());
  return result;
}

std::size_t node_bytes(
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state,
    const std::vector<float>& boundary_logits) {
  return identity.token_ids.size() * sizeof(TokenId) +
         identity.tenant_namespace.size() +
         identity.tokenizer_revision.size() +
         identity.serializer_revision.size() +
         identity.model_revision.size() +
         state.total_bytes() +
         (boundary_logits.size() * sizeof(float));
}

bool snapshot_prefix_matches_identity(
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state) {
  return !state.has_snapshot_layout() ||
         state.snapshot_layout.prefix_token_count == identity.token_ids.size();
}

}  // namespace

struct PrefixCache::Impl {
  struct GlobalRootTrieNode {
    PrefixNodeId terminal_node_id = 0;
    std::unordered_map<TokenId, std::unique_ptr<GlobalRootTrieNode>> children;
  };

  struct CacheNode {
    PrefixNodeId id = 0;
    SerializedPromptIdentity identity;
    ReusableStateDescriptor state;
    std::uint64_t fingerprint = 0;
    bool is_global_root = false;
    std::vector<float> boundary_logits;
    std::unordered_set<std::string> committed_conversations;
    std::uint64_t last_access_tick = 0;
    std::size_t total_bytes = 0;
    std::list<PrefixNodeId>::iterator lru_it;
  };

  std::size_t current_bytes = 0;
  std::size_t global_root_count = 0;
  std::uint64_t tick = 1;
  PrefixNodeId next_node_id = 1;
  std::unordered_map<PrefixNodeId, CacheNode> nodes;
  std::unordered_multimap<std::uint64_t, PrefixNodeId> nodes_by_fingerprint;
  std::unordered_map<std::string, PrefixNodeId> committed_heads;
  std::unordered_map<IdentityScopeKey, GlobalRootTrieNode, IdentityScopeKeyHash> global_root_tries;
  std::list<PrefixNodeId> lru_nodes;
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
    global_root_count = 0;
    nodes.clear();
    nodes_by_fingerprint.clear();
    committed_heads.clear();
    global_root_tries.clear();
    lru_nodes.clear();
  }

  bool NodeHasRoles(const CacheNode& node) const {
    return node.is_global_root || !node.committed_conversations.empty();
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

  void InsertGlobalRoot(const CacheNode& node) {
    auto& trie = global_root_tries[make_identity_scope_key(node.identity)];
    GlobalRootTrieNode* current = &trie;
    for (TokenId token : node.identity.token_ids) {
      auto& child = current->children[token];
      if (child == nullptr) {
        child = std::make_unique<GlobalRootTrieNode>();
      }
      current = child.get();
    }
    current->terminal_node_id = node.id;
  }

  void RemoveGlobalRoot(const CacheNode& node) {
    auto root_it = global_root_tries.find(make_identity_scope_key(node.identity));
    if (root_it == global_root_tries.end()) {
      return;
    }

    GlobalRootTrieNode* current = &root_it->second;
    std::vector<std::pair<GlobalRootTrieNode*, TokenId>> path;
    path.reserve(node.identity.token_ids.size());
    for (TokenId token : node.identity.token_ids) {
      auto child_it = current->children.find(token);
      if (child_it == current->children.end()) {
        return;
      }
      path.emplace_back(current, token);
      current = child_it->second.get();
    }
    if (current->terminal_node_id != node.id) {
      return;
    }
    current->terminal_node_id = 0;

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      auto child_it = it->first->children.find(it->second);
      if (child_it == it->first->children.end()) {
        continue;
      }
      if (child_it->second->terminal_node_id != 0 || !child_it->second->children.empty()) {
        break;
      }
      it->first->children.erase(child_it);
    }

    if (root_it->second.terminal_node_id == 0 && root_it->second.children.empty()) {
      global_root_tries.erase(root_it);
    }
  }

  PrefixNodeId LookupGlobalRoot(const SerializedPromptIdentity& identity) const {
    auto root_it = global_root_tries.find(make_identity_scope_key(identity));
    if (root_it == global_root_tries.end()) {
      return 0;
    }

    const GlobalRootTrieNode* current = &root_it->second;
    PrefixNodeId best_node_id = current->terminal_node_id;
    for (TokenId token : identity.token_ids) {
      auto child_it = current->children.find(token);
      if (child_it == current->children.end()) {
        break;
      }
      current = child_it->second.get();
      if (current->terminal_node_id != 0) {
        best_node_id = current->terminal_node_id;
      }
    }
    return best_node_id;
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
    if (node.is_global_root) {
      RemoveGlobalRoot(node);
      --global_root_count;
    }

    if (state_arena != nullptr) {
      state_arena->Release(node.state);
    }
    EraseFromFingerprintIndex(node);
    lru_nodes.erase(node.lru_it);
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
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state,
    const std::vector<float>* boundary_logits) {
  if (!impl_->enabled) {
    return 0;
  }
  if (!snapshot_prefix_matches_identity(identity, state)) {
    std::cerr << "prefix_cache: rejecting conversation head publish: snapshot prefix length "
              << state.snapshot_layout.prefix_token_count
              << " does not match identity token count "
              << identity.token_ids.size() << "\n";
    return 0;
  }
  const PrefixNodeId node_id = FindOrCreateNode(identity, state, boundary_logits);
  if (node_id == 0) {
    return 0;
  }
  AssignConversationHead(node_id, conversation_id);
  EvictToBudget();
  return Describe(node_id).has_value() ? node_id : 0;
}

PrefixNodeId PrefixCache::PublishConversationHeadSnapshot(
    const std::string& conversation_id,
    const SerializedPromptIdentity& identity,
    const RequestExecutionContext& request_context,
    const std::string& state_label,
    const std::vector<float>* boundary_logits) {
  if (!impl_->enabled || impl_->state_arena == nullptr) {
    return 0;
  }
  auto state = SnapshotRequestState(*impl_->state_arena, request_context, state_label);
  if (!state.has_value() && DropUniqueConversationHeadForRetry(conversation_id)) {
    // Committed-head replacement only needs one resident snapshot in steady
    // state. When the prior head is unique to this conversation, retry once
    // after freeing it so very large prefixes can advance under a tight
    // reusable-state budget.
    state = SnapshotRequestState(*impl_->state_arena, request_context, state_label);
  }
  if (!state.has_value()) {
    return 0;
  }
  if (!snapshot_prefix_matches_identity(identity, *state)) {
    std::cerr << "prefix_cache: rejecting conversation snapshot publish: snapshot prefix length "
              << state->snapshot_layout.prefix_token_count
              << " does not match identity token count "
              << identity.token_ids.size() << "\n";
    impl_->state_arena->Release(*state);
    return 0;
  }
  const PrefixNodeId node_id = PublishConversationHead(
      conversation_id,
      identity,
      *state,
      boundary_logits);
  impl_->state_arena->Release(*state);
  return node_id;
}

PrefixNodeId PrefixCache::PublishGlobalRoot(
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state,
    const std::vector<float>* boundary_logits) {
  if (!impl_->enabled) {
    return 0;
  }
  if (!snapshot_prefix_matches_identity(identity, state)) {
    std::cerr << "prefix_cache: rejecting global-root publish: snapshot prefix length "
              << state.snapshot_layout.prefix_token_count
              << " does not match identity token count "
              << identity.token_ids.size() << "\n";
    return 0;
  }
  const PrefixNodeId node_id = FindOrCreateNode(identity, state, boundary_logits);
  if (node_id == 0) {
    return 0;
  }
  AssignGlobalRoot(node_id);
  EvictToBudget();
  return Describe(node_id).has_value() ? node_id : 0;
}

PrefixNodeId PrefixCache::PublishGlobalRootSnapshot(
    const SerializedPromptIdentity& identity,
    const RequestExecutionContext& request_context,
    const std::string& state_label,
    const std::vector<float>* boundary_logits) {
  if (!impl_->enabled || impl_->state_arena == nullptr) {
    return 0;
  }
  const auto state = SnapshotRequestState(*impl_->state_arena, request_context, state_label);
  if (!state.has_value()) {
    return 0;
  }
  if (!snapshot_prefix_matches_identity(identity, *state)) {
    std::cerr << "prefix_cache: rejecting global-root snapshot publish: snapshot prefix length "
              << state->snapshot_layout.prefix_token_count
              << " does not match identity token count "
              << identity.token_ids.size() << "\n";
    impl_->state_arena->Release(*state);
    return 0;
  }
  const PrefixNodeId node_id = PublishGlobalRoot(identity, *state, boundary_logits);
  impl_->state_arena->Release(*state);
  return node_id;
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
    auto committed_it = impl_->committed_heads.find(*request.conversation_id);
    if (committed_it != impl_->committed_heads.end() &&
        PrefixMatches(impl_->nodes.at(committed_it->second).identity, request.identity)) {
      return build_match(committed_it->second, CacheMatchSource::kConversationCommittedHead);
    }
  }

  const PrefixNodeId best_node_id = impl_->LookupGlobalRoot(request.identity);
  if (best_node_id != 0) {
    return build_match(best_node_id, CacheMatchSource::kGlobalRoot);
  }
  return {};
}

bool PrefixCache::RestoreMatchState(const CacheMatch& match, RequestExecutionContext& request_context) const {
  if (!impl_->enabled ||
      impl_->state_arena == nullptr ||
      !match.hit() ||
      !match.state.valid() ||
      match.matched_token_count == 0) {
    return false;
  }
  if (!match.state.has_snapshot_layout()) {
    std::cerr << "prefix_cache: restore rejected for node "
              << match.node_id
              << ": cached state does not carry snapshot layout metadata\n";
    return false;
  }
  if (match.state.snapshot_layout.prefix_token_count != match.matched_token_count) {
    std::cerr << "prefix_cache: restore rejected for node "
              << match.node_id
              << ": cached snapshot prefix length "
              << match.state.snapshot_layout.prefix_token_count
              << " does not match cache identity length "
              << match.matched_token_count << "\n";
    return false;
  }
  return RestoreRequestState(
      *impl_->state_arena,
      match.state,
      match.matched_token_count,
      request_context);
}

std::optional<std::vector<float>> PrefixCache::CopyBoundaryLogits(PrefixNodeId node_id) const {
  auto it = impl_->nodes.find(node_id);
  if (it == impl_->nodes.end() || it->second.boundary_logits.empty()) {
    return std::nullopt;
  }
  return it->second.boundary_logits;
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
      node.last_access_tick,
      node.total_bytes,
      !node.boundary_logits.empty(),
      node.boundary_logits.size(),
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
  return impl_->global_root_count;
}

PrefixNodeId PrefixCache::FindOrCreateNode(
    const SerializedPromptIdentity& identity,
    const ReusableStateDescriptor& state,
    const std::vector<float>* boundary_logits) {
  const std::uint64_t fingerprint = fingerprint_identity(identity);
  auto range = impl_->nodes_by_fingerprint.equal_range(fingerprint);
  for (auto it = range.first; it != range.second; ++it) {
    Impl::CacheNode& node = impl_->nodes.at(it->second);
    if (!same_identity(node.identity, identity)) {
      continue;
    }
    const bool replace_boundary_logits =
        boundary_logits != nullptr && node.boundary_logits != *boundary_logits;
    if (!(node.state == state)) {
      if (!impl_->RetainState(state)) {
        return 0;
      }
      if (impl_->state_arena != nullptr) {
        impl_->state_arena->Release(node.state);
      }
      impl_->current_bytes -= node.total_bytes;
      node.state = state;
      if (replace_boundary_logits) {
        node.boundary_logits = *boundary_logits;
      }
      node.total_bytes = node_bytes(node.identity, node.state, node.boundary_logits);
      impl_->current_bytes += node.total_bytes;
    } else if (replace_boundary_logits) {
      impl_->current_bytes -= node.total_bytes;
      node.boundary_logits = *boundary_logits;
      node.total_bytes = node_bytes(node.identity, node.state, node.boundary_logits);
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
  if (boundary_logits != nullptr) {
    node.boundary_logits = *boundary_logits;
  }
  node.fingerprint = fingerprint;
  node.total_bytes = node_bytes(identity, state, node.boundary_logits);
  node.last_access_tick = impl_->tick++;
  impl_->current_bytes += node.total_bytes;

  impl_->lru_nodes.push_back(node.id);
  node.lru_it = std::prev(impl_->lru_nodes.end());

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
  impl_->lru_nodes.splice(impl_->lru_nodes.end(), impl_->lru_nodes, it->second.lru_it);
}

void PrefixCache::AssignConversationHead(PrefixNodeId node_id, const std::string& conversation_id) {
  auto existing_it = impl_->committed_heads.find(conversation_id);
  if (existing_it != impl_->committed_heads.end() && existing_it->second != node_id) {
    RemoveConversationRole(existing_it->second, conversation_id);
  }
  impl_->committed_heads[conversation_id] = node_id;

  Impl::CacheNode& node = impl_->nodes.at(node_id);
  node.committed_conversations.insert(conversation_id);
  Touch(node_id);
}

void PrefixCache::AssignGlobalRoot(PrefixNodeId node_id) {
  Impl::CacheNode& node = impl_->nodes.at(node_id);
  if (!node.is_global_root) {
    node.is_global_root = true;
    impl_->InsertGlobalRoot(node);
    ++impl_->global_root_count;
  }
  Touch(node_id);
}

void PrefixCache::RemoveConversationRole(PrefixNodeId node_id, const std::string& conversation_id) {
  auto node_it = impl_->nodes.find(node_id);
  if (node_it == impl_->nodes.end()) {
    return;
  }
  Impl::CacheNode& node = node_it->second;
  node.committed_conversations.erase(conversation_id);
  if (!impl_->NodeHasRoles(node)) {
    impl_->RemoveNode(node_id);
  }
}

bool PrefixCache::DropUniqueConversationHeadForRetry(const std::string& conversation_id) {
  auto head_it = impl_->committed_heads.find(conversation_id);
  if (head_it == impl_->committed_heads.end()) {
    return false;
  }

  auto node_it = impl_->nodes.find(head_it->second);
  if (node_it == impl_->nodes.end()) {
    impl_->committed_heads.erase(head_it);
    return false;
  }

  const Impl::CacheNode& node = node_it->second;
  if (node.is_global_root ||
      node.committed_conversations.size() != 1 ||
      node.committed_conversations.count(conversation_id) == 0) {
    return false;
  }

  impl_->committed_heads.erase(head_it);
  RemoveConversationRole(node.id, conversation_id);
  return true;
}

void PrefixCache::EvictToBudget() {
  while (impl_->current_bytes > max_bytes_ && !impl_->lru_nodes.empty()) {
    impl_->RemoveNode(impl_->lru_nodes.front());
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
