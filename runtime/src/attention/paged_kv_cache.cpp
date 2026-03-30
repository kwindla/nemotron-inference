#include "nemotron/paged_kv_cache.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace nemotron {

std::size_t KvCacheDataTypeBytes(KvCacheDataType dtype) {
  switch (dtype) {
    case KvCacheDataType::kFp16:
    case KvCacheDataType::kBf16:
      return 2;
  }
  return 0;
}

std::optional<AttentionKvPageGeometry> BuildAttentionKvPageGeometry(const AttentionKvCacheConfig& config) {
  if (config.layer_count == 0 ||
      config.kv_head_count == 0 ||
      config.head_dim == 0 ||
      config.tokens_per_page == 0) {
    return std::nullopt;
  }

  AttentionKvPageGeometry geometry;
  geometry.bytes_per_scalar = KvCacheDataTypeBytes(config.dtype);
  if (geometry.bytes_per_scalar == 0) {
    return std::nullopt;
  }

  geometry.scalars_per_token = 2u * config.kv_head_count * config.head_dim;
  geometry.bytes_per_token = geometry.scalars_per_token * geometry.bytes_per_scalar;
  geometry.bytes_per_page = geometry.bytes_per_token * config.tokens_per_page;
  if (geometry.bytes_per_token == 0 || geometry.bytes_per_page == 0) {
    return std::nullopt;
  }
  return geometry;
}

std::size_t ModelKvBytesPerToken(const AttentionKvCacheConfig& config) {
  const auto geometry = BuildAttentionKvPageGeometry(config);
  if (!geometry.has_value()) {
    return 0;
  }
  return geometry->bytes_per_token * config.layer_count;
}

std::size_t RequiredPagesForTokens(const AttentionKvCacheConfig& config, std::size_t token_count) {
  if (config.tokens_per_page == 0 || token_count == 0) {
    return 0;
  }
  return (token_count + config.tokens_per_page - 1) / config.tokens_per_page;
}

std::optional<PagedKvCacheArena> PagedKvCacheArena::Create(
    const AttentionKvCacheConfig& config,
    std::size_t total_pages) {
  const auto geometry = BuildAttentionKvPageGeometry(config);
  if (!geometry.has_value() || total_pages == 0) {
    return std::nullopt;
  }

  std::vector<PageState> page_states(total_pages);
  std::vector<std::size_t> free_page_ids;
  free_page_ids.reserve(total_pages);
  for (std::size_t page_id = 0; page_id < total_pages; ++page_id) {
    page_states[page_id].byte_offset = page_id * geometry->bytes_per_page;
  }
  for (std::size_t page_id = total_pages; page_id > 0; --page_id) {
    free_page_ids.push_back(page_id - 1);
  }

  return PagedKvCacheArena(config, *geometry, std::move(page_states), std::move(free_page_ids));
}

PagedKvCacheArena::PagedKvCacheArena(
    AttentionKvCacheConfig config,
    AttentionKvPageGeometry geometry,
    std::vector<PageState> page_states,
    std::vector<std::size_t> free_page_ids)
    : config_(std::move(config)),
      geometry_(geometry),
      page_states_(std::move(page_states)),
      free_page_ids_(std::move(free_page_ids)) {}

PagedKvCacheArena::PagedKvCacheArena(PagedKvCacheArena&&) noexcept = default;
PagedKvCacheArena& PagedKvCacheArena::operator=(PagedKvCacheArena&&) noexcept = default;
PagedKvCacheArena::~PagedKvCacheArena() = default;

const AttentionKvCacheConfig& PagedKvCacheArena::config() const {
  return config_;
}

const AttentionKvPageGeometry& PagedKvCacheArena::geometry() const {
  return geometry_;
}

std::size_t PagedKvCacheArena::total_pages() const {
  return page_states_.size();
}

std::size_t PagedKvCacheArena::free_pages() const {
  return free_page_ids_.size();
}

std::size_t PagedKvCacheArena::allocated_pages() const {
  return total_pages() - free_pages();
}

std::size_t PagedKvCacheArena::total_bytes() const {
  return total_pages() * geometry_.bytes_per_page;
}

std::optional<std::vector<KvPageHandle>> PagedKvCacheArena::AllocatePages(
    std::size_t layer_index,
    std::size_t page_count) {
  if (page_count == 0 || layer_index >= config_.layer_count || free_page_ids_.size() < page_count) {
    return std::nullopt;
  }

  std::vector<KvPageHandle> handles;
  handles.reserve(page_count);
  for (std::size_t i = 0; i < page_count; ++i) {
    const std::size_t page_id = free_page_ids_.back();
    free_page_ids_.pop_back();
    PageState& page_state = page_states_[page_id];
    page_state.allocated = true;
    page_state.layer_index = layer_index;
    handles.push_back(KvPageHandle{
        page_id,
        layer_index,
        page_state.byte_offset,
        config_.tokens_per_page,
    });
  }
  return handles;
}

bool PagedKvCacheArena::ReleasePages(const std::vector<std::size_t>& page_ids) {
  if (page_ids.empty()) {
    return false;
  }

  std::unordered_set<std::size_t> unique_ids;
  unique_ids.reserve(page_ids.size());
  for (const std::size_t page_id : page_ids) {
    if (page_id >= page_states_.size() || !page_states_[page_id].allocated || !unique_ids.insert(page_id).second) {
      return false;
    }
  }

  for (const std::size_t page_id : page_ids) {
    PageState& page_state = page_states_[page_id];
    page_state.allocated = false;
    page_state.layer_index = 0;
    free_page_ids_.push_back(page_id);
  }
  std::sort(free_page_ids_.begin(), free_page_ids_.end(), std::greater<std::size_t>());
  return true;
}

std::optional<KvPageHandle> PagedKvCacheArena::InspectPage(std::size_t page_id) const {
  if (page_id >= page_states_.size()) {
    return std::nullopt;
  }

  const PageState& page_state = page_states_[page_id];
  if (!page_state.allocated) {
    return std::nullopt;
  }

  return KvPageHandle{
      page_id,
      page_state.layer_index,
      page_state.byte_offset,
      config_.tokens_per_page,
  };
}

}  // namespace nemotron
