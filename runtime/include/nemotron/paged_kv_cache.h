#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace nemotron {

enum class KvCacheDataType {
  kFp16,
  kBf16,
  kFp8E4M3,
};

std::size_t KvCacheDataTypeBytes(KvCacheDataType dtype);

struct AttentionKvCacheConfig {
  std::size_t layer_count = 0;
  std::size_t kv_head_count = 0;
  std::size_t head_dim = 0;
  std::size_t tokens_per_page = 0;
  KvCacheDataType dtype = KvCacheDataType::kBf16;
  float q_scale = 1.0f;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  float prob_scale = 1.0f;
};

struct AttentionKvPageGeometry {
  std::size_t scalars_per_token = 0;
  std::size_t bytes_per_scalar = 0;
  std::size_t bytes_per_token = 0;
  std::size_t bytes_per_page = 0;
};

std::optional<AttentionKvPageGeometry> BuildAttentionKvPageGeometry(const AttentionKvCacheConfig& config);
std::size_t ModelKvBytesPerToken(const AttentionKvCacheConfig& config);
std::size_t RequiredPagesForTokens(const AttentionKvCacheConfig& config, std::size_t token_count);

struct KvPageHandle {
  std::size_t page_id = 0;
  std::size_t layer_index = 0;
  std::size_t byte_offset = 0;
  std::size_t tokens_per_page = 0;
};

class PagedKvCacheArena {
 public:
  static std::optional<PagedKvCacheArena> Create(
      const AttentionKvCacheConfig& config,
      std::size_t total_pages);

  PagedKvCacheArena(PagedKvCacheArena&&) noexcept;
  PagedKvCacheArena& operator=(PagedKvCacheArena&&) noexcept;
  ~PagedKvCacheArena();

  PagedKvCacheArena(const PagedKvCacheArena&) = delete;
  PagedKvCacheArena& operator=(const PagedKvCacheArena&) = delete;

  const AttentionKvCacheConfig& config() const;
  const AttentionKvPageGeometry& geometry() const;
  std::size_t total_pages() const;
  std::size_t free_pages() const;
  std::size_t allocated_pages() const;
  std::size_t total_bytes() const;

  std::optional<std::vector<KvPageHandle>> AllocatePages(std::size_t layer_index, std::size_t page_count);
  bool ReleasePages(const std::vector<std::size_t>& page_ids);
  std::optional<KvPageHandle> InspectPage(std::size_t page_id) const;

 private:
  struct PageState {
    bool allocated = false;
    std::size_t layer_index = 0;
    std::size_t byte_offset = 0;
  };

  PagedKvCacheArena(
      AttentionKvCacheConfig config,
      AttentionKvPageGeometry geometry,
      std::vector<PageState> page_states,
      std::vector<std::size_t> free_page_ids);

  AttentionKvCacheConfig config_;
  AttentionKvPageGeometry geometry_;
  std::vector<PageState> page_states_;
  std::vector<std::size_t> free_page_ids_;
};

}  // namespace nemotron
