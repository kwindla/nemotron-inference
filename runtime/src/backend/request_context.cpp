#include "nemotron/request_context.h"

#include <utility>

namespace nemotron {
namespace {

bool HandlesSatisfyLayerInvariant(
    const AttentionKvCacheConfig& config,
    const std::vector<KvPageHandle>& pages,
    std::size_t layer_index) {
  const auto geometry = BuildAttentionKvPageGeometry(config);
  if (!geometry.has_value() || layer_index >= config.layer_count) {
    return false;
  }

  std::size_t previous_page_id = 0;
  bool have_previous_page = false;
  for (const KvPageHandle& handle : pages) {
    const std::size_t expected_byte_offset = handle.page_id * geometry->bytes_per_page;
    if (handle.layer_index != layer_index ||
        handle.tokens_per_page != config.tokens_per_page ||
        handle.byte_offset != expected_byte_offset ||
        (have_previous_page && handle.page_id <= previous_page_id)) {
      return false;
    }
    previous_page_id = handle.page_id;
    have_previous_page = true;
  }
  return true;
}

}  // namespace

std::unique_ptr<RequestExecutionContext> RequestExecutionContext::Create(
    const RequestExecutionConfig& config) {
  if (config.hidden_size == 0 || config.max_tokens == 0) {
    return nullptr;
  }
  const std::size_t scratch_tokens = config.scratch_tokens == 0 ? config.max_tokens : config.scratch_tokens;

  auto hidden = DeviceTensorBf16::Create({config.max_tokens, config.hidden_size});
  auto residual = DeviceTensorBf16::Create({config.max_tokens, config.hidden_size});
  auto scratch = DeviceTensorBf16::Create({scratch_tokens, config.hidden_size});
  if (!hidden || !residual || !scratch) {
    return nullptr;
  }
  if (!hidden->FillZero() || !residual->FillZero() || !scratch->FillZero()) {
    return nullptr;
  }

  std::unique_ptr<DeviceTensorFp32> mamba_state;
  std::unique_ptr<DeviceTensorFp32> mamba_conv_state;
  if (config.mamba_conv_state_bytes_fp32 != 0) {
    if (config.mamba_conv_state_bytes_fp32 % sizeof(float) != 0) {
      return nullptr;
    }
    const std::size_t state_numel = config.mamba_conv_state_bytes_fp32 / sizeof(float);
    mamba_conv_state = DeviceTensorFp32::Create({state_numel});
    if (!mamba_conv_state || !mamba_conv_state->FillZero()) {
      return nullptr;
    }
  }
  if (config.mamba_state_bytes_fp32 != 0) {
    if (config.mamba_state_bytes_fp32 % sizeof(float) != 0) {
      return nullptr;
    }
    const std::size_t state_numel = config.mamba_state_bytes_fp32 / sizeof(float);
    mamba_state = DeviceTensorFp32::Create({state_numel});
    if (!mamba_state || !mamba_state->FillZero()) {
      return nullptr;
    }
  }

  std::unique_ptr<DeviceTensorBf16> key_cache;
  std::unique_ptr<DeviceTensorBf16> value_cache;
  std::optional<PagedKvCacheArena> kv_arena;
  if (config.attention_kv_cache.layer_count != 0) {
    kv_arena = PagedKvCacheArena::Create(config.attention_kv_cache, config.attention_total_pages);
    if (!kv_arena.has_value()) {
      return nullptr;
    }
    key_cache = DeviceTensorBf16::Create({
        config.attention_total_pages,
        config.attention_kv_cache.kv_head_count,
        config.attention_kv_cache.tokens_per_page,
        config.attention_kv_cache.head_dim,
    });
    value_cache = DeviceTensorBf16::Create({
        config.attention_total_pages,
        config.attention_kv_cache.kv_head_count,
        config.attention_kv_cache.tokens_per_page,
        config.attention_kv_cache.head_dim,
    });
    if (!key_cache || !value_cache || !key_cache->FillZero() || !value_cache->FillZero()) {
      return nullptr;
    }
  }

  return std::unique_ptr<RequestExecutionContext>(new RequestExecutionContext(
      config,
      std::move(hidden),
      std::move(residual),
      std::move(scratch),
      std::move(mamba_conv_state),
      std::move(mamba_state),
      std::move(key_cache),
      std::move(value_cache),
      std::move(kv_arena)));
}

RequestExecutionContext::RequestExecutionContext(
    RequestExecutionConfig config,
    std::unique_ptr<DeviceTensorBf16> hidden,
    std::unique_ptr<DeviceTensorBf16> residual,
    std::unique_ptr<DeviceTensorBf16> scratch,
    std::unique_ptr<DeviceTensorFp32> mamba_conv_state,
    std::unique_ptr<DeviceTensorFp32> mamba_state,
    std::unique_ptr<DeviceTensorBf16> key_cache,
    std::unique_ptr<DeviceTensorBf16> value_cache,
    std::optional<PagedKvCacheArena> kv_arena)
    : config_(std::move(config)),
      hidden_(std::move(hidden)),
      residual_(std::move(residual)),
      scratch_(std::move(scratch)),
      mamba_conv_state_(std::move(mamba_conv_state)),
      mamba_state_(std::move(mamba_state)),
      key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)),
      kv_arena_(std::move(kv_arena)),
      kv_pages_by_layer_(config_.attention_kv_cache.layer_count) {}

RequestExecutionContext::RequestExecutionContext(RequestExecutionContext&&) noexcept = default;
RequestExecutionContext& RequestExecutionContext::operator=(RequestExecutionContext&&) noexcept = default;
RequestExecutionContext::~RequestExecutionContext() = default;

bool RequestExecutionContext::valid() const {
  if (!hidden_ || !hidden_->valid() || !residual_ || !residual_->valid() || !scratch_ || !scratch_->valid()) {
    return false;
  }
  if (config_.mamba_state_bytes_fp32 != 0 && (!mamba_state_ || !mamba_state_->valid())) {
    return false;
  }
  if (config_.mamba_conv_state_bytes_fp32 != 0 &&
      (!mamba_conv_state_ || !mamba_conv_state_->valid())) {
    return false;
  }
  if (config_.attention_kv_cache.layer_count != 0 && !kv_arena_.has_value()) {
    return false;
  }
  if (config_.attention_kv_cache.layer_count != 0 &&
      (!key_cache_ || !key_cache_->valid() || !value_cache_ || !value_cache_->valid())) {
    return false;
  }
  return true;
}

const RequestExecutionConfig& RequestExecutionContext::config() const {
  return config_;
}

std::size_t RequestExecutionContext::sequence_length() const {
  return sequence_length_;
}

std::size_t RequestExecutionContext::decode_position() const {
  return decode_position_;
}

DeviceTensorBf16* RequestExecutionContext::hidden() {
  return hidden_.get();
}

const DeviceTensorBf16* RequestExecutionContext::hidden() const {
  return hidden_.get();
}

DeviceTensorBf16* RequestExecutionContext::residual() {
  return residual_.get();
}

const DeviceTensorBf16* RequestExecutionContext::residual() const {
  return residual_.get();
}

DeviceTensorBf16* RequestExecutionContext::scratch() {
  return scratch_.get();
}

const DeviceTensorBf16* RequestExecutionContext::scratch() const {
  return scratch_.get();
}

DeviceTensorFp32* RequestExecutionContext::mamba_state() {
  return mamba_state_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_state() const {
  return mamba_state_.get();
}

DeviceTensorFp32* RequestExecutionContext::mamba_conv_state() {
  return mamba_conv_state_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_conv_state() const {
  return mamba_conv_state_.get();
}

DeviceTensorBf16* RequestExecutionContext::key_cache() {
  return key_cache_.get();
}

const DeviceTensorBf16* RequestExecutionContext::key_cache() const {
  return key_cache_.get();
}

DeviceTensorBf16* RequestExecutionContext::value_cache() {
  return value_cache_.get();
}

const DeviceTensorBf16* RequestExecutionContext::value_cache() const {
  return value_cache_.get();
}

bool RequestExecutionContext::EnsureAttentionTokens(std::size_t token_count) {
  if (!kv_arena_.has_value()) {
    return token_count == 0;
  }
  if (token_count > config_.max_tokens) {
    return false;
  }

  const std::size_t required_pages =
      RequiredPagesForTokens(config_.attention_kv_cache, token_count);
  for (std::size_t layer_index = 0; layer_index < kv_pages_by_layer_.size(); ++layer_index) {
    std::vector<KvPageHandle>& pages = kv_pages_by_layer_[layer_index];
    if (!HandlesSatisfyLayerInvariant(config_.attention_kv_cache, pages, layer_index)) {
      return false;
    }
    if (pages.size() >= required_pages) {
      continue;
    }
    const std::size_t additional_pages = required_pages - pages.size();
    const auto allocated = kv_arena_->AllocatePages(layer_index, additional_pages);
    if (!allocated.has_value()) {
      return false;
    }
    pages.insert(pages.end(), allocated->begin(), allocated->end());
    if (!HandlesSatisfyLayerInvariant(config_.attention_kv_cache, pages, layer_index)) {
      return false;
    }
  }
  return true;
}

std::size_t RequestExecutionContext::allocated_kv_pages() const {
  std::size_t total = 0;
  for (const auto& layer_pages : kv_pages_by_layer_) {
    total += layer_pages.size();
  }
  return total;
}

std::size_t RequestExecutionContext::allocated_kv_pages(std::size_t layer_index) const {
  if (layer_index >= kv_pages_by_layer_.size()) {
    return 0;
  }
  return kv_pages_by_layer_[layer_index].size();
}

const std::vector<KvPageHandle>* RequestExecutionContext::kv_pages(std::size_t layer_index) const {
  if (layer_index >= kv_pages_by_layer_.size()) {
    return nullptr;
  }
  return &kv_pages_by_layer_[layer_index];
}

bool RequestExecutionContext::SetSequenceLength(std::size_t sequence_length) {
  if (sequence_length > config_.max_tokens) {
    return false;
  }
  sequence_length_ = sequence_length;
  decode_position_ = sequence_length;
  return EnsureAttentionTokens(sequence_length);
}

bool RequestExecutionContext::AdvanceDecodePosition(std::size_t token_count) {
  if (token_count == 0) {
    return true;
  }
  if (decode_position_ > config_.max_tokens || token_count > (config_.max_tokens - decode_position_)) {
    return false;
  }
  decode_position_ += token_count;
  if (decode_position_ > sequence_length_) {
    sequence_length_ = decode_position_;
  }
  return EnsureAttentionTokens(sequence_length_);
}

bool RequestExecutionContext::ResetForNewRequest() {
  bool ok = hidden_->FillZero() && residual_->FillZero() && scratch_->FillZero();
  if (mamba_conv_state_) {
    ok = ok && mamba_conv_state_->FillZero();
  }
  if (mamba_state_) {
    ok = ok && mamba_state_->FillZero();
  }
  if (key_cache_) {
    ok = ok && key_cache_->FillZero();
  }
  if (value_cache_) {
    ok = ok && value_cache_->FillZero();
  }

  if (kv_arena_.has_value()) {
    std::vector<std::size_t> page_ids;
    page_ids.reserve(allocated_kv_pages());
    for (std::size_t layer_index = 0; layer_index < kv_pages_by_layer_.size(); ++layer_index) {
      const auto& layer_pages = kv_pages_by_layer_[layer_index];
      ok = ok && HandlesSatisfyLayerInvariant(config_.attention_kv_cache, layer_pages, layer_index);
      for (const KvPageHandle& handle : layer_pages) {
        page_ids.push_back(handle.page_id);
      }
    }
    if (!page_ids.empty()) {
      ok = ok && kv_arena_->ReleasePages(page_ids);
    }
  }

  for (auto& layer_pages : kv_pages_by_layer_) {
    layer_pages.clear();
  }
  sequence_length_ = 0;
  decode_position_ = 0;
  return ok;
}

}  // namespace nemotron
