#include "nemotron/request_context.h"

#include <utility>

namespace nemotron {
std::unique_ptr<RequestExecutionContext> RequestExecutionContext::Create(
    const RequestExecutionConfig& config) {
  if (config.hidden_size == 0 || config.max_tokens == 0) {
    return nullptr;
  }
  const std::size_t scratch_tokens = config.scratch_tokens == 0 ? config.max_tokens : config.scratch_tokens;

  auto hidden = DeviceTensorFp32::Create({config.max_tokens, config.hidden_size});
  auto residual = DeviceTensorFp32::Create({config.max_tokens, config.hidden_size});
  auto scratch = DeviceTensorFp32::Create({scratch_tokens, config.hidden_size});
  if (!hidden || !residual || !scratch) {
    return nullptr;
  }
  if (!hidden->FillZero() || !residual->FillZero() || !scratch->FillZero()) {
    return nullptr;
  }

  std::unique_ptr<DeviceTensorFp32> mamba_state;
  std::unique_ptr<DeviceTensorFp32> mamba_conv_state;
  std::unique_ptr<DeviceTensorFp32> mamba_normalized_decode;
  std::unique_ptr<DeviceTensorFp32> mamba_projected_decode;
  std::unique_ptr<DeviceTensorFp32> mamba_scan_output_decode;
  std::unique_ptr<DeviceTensorFp32> mamba_projected_output_decode;
  std::unique_ptr<DeviceTensorFp32> attention_normed_decode;
  std::unique_ptr<DeviceTensorFp32> attention_q_decode;
  std::unique_ptr<DeviceTensorFp32> attention_k_decode;
  std::unique_ptr<DeviceTensorFp32> attention_v_decode;
  std::unique_ptr<DeviceTensorFp32> attention_output_fp32_decode;
  std::unique_ptr<DeviceTensorFp32> attention_projected_decode;
  std::unique_ptr<DeviceTensorBf16> attention_query_bf16_decode;
  std::unique_ptr<DeviceTensorBf16> attention_output_bf16_decode;
  std::unique_ptr<DeviceTensorFp32> expert_intermediate_scratch;
  std::unique_ptr<DeviceTensorFp32> expert_aux_scratch;
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
  if (config.mamba_hidden_size != 0 &&
      config.mamba_projection_size != 0 &&
      config.mamba_intermediate_size != 0) {
    mamba_normalized_decode = DeviceTensorFp32::Create({1, config.mamba_hidden_size});
    mamba_projected_decode = DeviceTensorFp32::Create({1, config.mamba_projection_size});
    mamba_scan_output_decode = DeviceTensorFp32::Create({1, config.mamba_intermediate_size});
    mamba_projected_output_decode = DeviceTensorFp32::Create({1, config.mamba_hidden_size});
    if (!mamba_normalized_decode || !mamba_projected_decode || !mamba_scan_output_decode ||
        !mamba_projected_output_decode ||
        !mamba_normalized_decode->FillZero() || !mamba_projected_decode->FillZero() ||
        !mamba_scan_output_decode->FillZero() || !mamba_projected_output_decode->FillZero()) {
      return nullptr;
    }
  }
  if (config.expert_intermediate_scratch_numel != 0) {
    expert_intermediate_scratch =
        DeviceTensorFp32::Create({config.expert_intermediate_scratch_numel});
    if (!expert_intermediate_scratch || !expert_intermediate_scratch->FillZero()) {
      return nullptr;
    }
  }
  if (config.expert_aux_scratch_numel != 0) {
    expert_aux_scratch = DeviceTensorFp32::Create({config.expert_aux_scratch_numel});
    if (!expert_aux_scratch || !expert_aux_scratch->FillZero()) {
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
    if (config.attention_query_head_count != 0 && config.attention_head_dim != 0) {
      attention_normed_decode = DeviceTensorFp32::Create({1, config.hidden_size});
      attention_q_decode = DeviceTensorFp32::Create(
          {1, config.attention_query_head_count * config.attention_head_dim});
      attention_k_decode = DeviceTensorFp32::Create(
          {1, config.attention_kv_cache.kv_head_count * config.attention_kv_cache.head_dim});
      attention_v_decode = DeviceTensorFp32::Create(
          {1, config.attention_kv_cache.kv_head_count * config.attention_kv_cache.head_dim});
      attention_output_fp32_decode = DeviceTensorFp32::Create({1, config.hidden_size});
      attention_projected_decode = DeviceTensorFp32::Create({1, config.hidden_size});
      attention_query_bf16_decode = DeviceTensorBf16::Create(
          {1, config.attention_query_head_count, 1, config.attention_head_dim});
      attention_output_bf16_decode = DeviceTensorBf16::Create(
          {1, config.attention_query_head_count, 1, config.attention_head_dim});
      if (!attention_normed_decode || !attention_q_decode || !attention_k_decode ||
          !attention_v_decode || !attention_output_fp32_decode ||
          !attention_projected_decode || !attention_query_bf16_decode ||
          !attention_output_bf16_decode ||
          !attention_normed_decode->FillZero() || !attention_q_decode->FillZero() ||
          !attention_k_decode->FillZero() || !attention_v_decode->FillZero() ||
          !attention_output_fp32_decode->FillZero() || !attention_projected_decode->FillZero() ||
          !attention_query_bf16_decode->FillZero() || !attention_output_bf16_decode->FillZero()) {
        return nullptr;
      }
    }
  }

  auto context = std::unique_ptr<RequestExecutionContext>(new RequestExecutionContext(
      config,
      std::move(hidden),
      std::move(residual),
      std::move(scratch),
      std::move(mamba_conv_state),
      std::move(mamba_state),
      std::move(mamba_normalized_decode),
      std::move(mamba_projected_decode),
      std::move(mamba_scan_output_decode),
      std::move(mamba_projected_output_decode),
      std::move(attention_normed_decode),
      std::move(attention_q_decode),
      std::move(attention_k_decode),
      std::move(attention_v_decode),
      std::move(attention_output_fp32_decode),
      std::move(attention_projected_decode),
      std::move(attention_query_bf16_decode),
      std::move(attention_output_bf16_decode),
      std::move(expert_intermediate_scratch),
      std::move(expert_aux_scratch),
      std::move(key_cache),
      std::move(value_cache),
      std::move(kv_arena)));
  if (!context->token_ids_device_.Resize(config.max_tokens)) {
    return nullptr;
  }
  if (!context->EnsureExpertSelectionCapacity(config.expert_selection_capacity)) {
    return nullptr;
  }
  return context;
}

RequestExecutionContext::RequestExecutionContext(
    RequestExecutionConfig config,
    std::unique_ptr<DeviceTensorFp32> hidden,
    std::unique_ptr<DeviceTensorFp32> residual,
    std::unique_ptr<DeviceTensorFp32> scratch,
    std::unique_ptr<DeviceTensorFp32> mamba_conv_state,
    std::unique_ptr<DeviceTensorFp32> mamba_state,
    std::unique_ptr<DeviceTensorFp32> mamba_normalized_decode,
    std::unique_ptr<DeviceTensorFp32> mamba_projected_decode,
    std::unique_ptr<DeviceTensorFp32> mamba_scan_output_decode,
    std::unique_ptr<DeviceTensorFp32> mamba_projected_output_decode,
    std::unique_ptr<DeviceTensorFp32> attention_normed_decode,
    std::unique_ptr<DeviceTensorFp32> attention_q_decode,
    std::unique_ptr<DeviceTensorFp32> attention_k_decode,
    std::unique_ptr<DeviceTensorFp32> attention_v_decode,
    std::unique_ptr<DeviceTensorFp32> attention_output_fp32_decode,
    std::unique_ptr<DeviceTensorFp32> attention_projected_decode,
    std::unique_ptr<DeviceTensorBf16> attention_query_bf16_decode,
    std::unique_ptr<DeviceTensorBf16> attention_output_bf16_decode,
    std::unique_ptr<DeviceTensorFp32> expert_intermediate_scratch,
    std::unique_ptr<DeviceTensorFp32> expert_aux_scratch,
    std::unique_ptr<DeviceTensorBf16> key_cache,
    std::unique_ptr<DeviceTensorBf16> value_cache,
    std::optional<PagedKvCacheArena> kv_arena)
    : config_(std::move(config)),
      hidden_(std::move(hidden)),
      residual_(std::move(residual)),
      scratch_(std::move(scratch)),
      mamba_conv_state_(std::move(mamba_conv_state)),
      mamba_state_(std::move(mamba_state)),
      mamba_normalized_decode_(std::move(mamba_normalized_decode)),
      mamba_projected_decode_(std::move(mamba_projected_decode)),
      mamba_scan_output_decode_(std::move(mamba_scan_output_decode)),
      mamba_projected_output_decode_(std::move(mamba_projected_output_decode)),
      attention_normed_decode_(std::move(attention_normed_decode)),
      attention_q_decode_(std::move(attention_q_decode)),
      attention_k_decode_(std::move(attention_k_decode)),
      attention_v_decode_(std::move(attention_v_decode)),
      attention_output_fp32_decode_(std::move(attention_output_fp32_decode)),
      attention_projected_decode_(std::move(attention_projected_decode)),
      attention_query_bf16_decode_(std::move(attention_query_bf16_decode)),
      attention_output_bf16_decode_(std::move(attention_output_bf16_decode)),
      expert_intermediate_scratch_(std::move(expert_intermediate_scratch)),
      expert_aux_scratch_(std::move(expert_aux_scratch)),
      key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)),
      kv_arena_(std::move(kv_arena)),
      kv_pages_by_layer_(config_.attention_kv_cache.layer_count),
      kv_page_ids_device_by_layer_(config_.attention_kv_cache.layer_count) {}

RequestExecutionContext::RequestExecutionContext(RequestExecutionContext&&) noexcept = default;
RequestExecutionContext& RequestExecutionContext::operator=(RequestExecutionContext&&) noexcept = default;
RequestExecutionContext::~RequestExecutionContext() = default;

bool RequestExecutionContext::valid() const {
  if (!hidden_ || !hidden_->valid() || !residual_ || !residual_->valid() || !scratch_ || !scratch_->valid()) {
    return false;
  }
  if (!token_ids_device_.valid()) {
    return false;
  }
  if (config_.mamba_state_bytes_fp32 != 0 && (!mamba_state_ || !mamba_state_->valid())) {
    return false;
  }
  if (config_.mamba_conv_state_bytes_fp32 != 0 &&
      (!mamba_conv_state_ || !mamba_conv_state_->valid())) {
    return false;
  }
  if ((mamba_normalized_decode_ && !mamba_normalized_decode_->valid()) ||
      (mamba_projected_decode_ && !mamba_projected_decode_->valid()) ||
      (mamba_scan_output_decode_ && !mamba_scan_output_decode_->valid()) ||
      (mamba_projected_output_decode_ && !mamba_projected_output_decode_->valid())) {
    return false;
  }
  if (config_.expert_intermediate_scratch_numel != 0 &&
      (!expert_intermediate_scratch_ || !expert_intermediate_scratch_->valid())) {
    return false;
  }
  if (config_.expert_aux_scratch_numel != 0 &&
      (!expert_aux_scratch_ || !expert_aux_scratch_->valid())) {
    return false;
  }
  if (config_.attention_kv_cache.layer_count != 0 && !kv_arena_.has_value()) {
    return false;
  }
  if (config_.attention_kv_cache.layer_count != 0 &&
      (!key_cache_ || !key_cache_->valid() || !value_cache_ || !value_cache_->valid())) {
    return false;
  }
  if ((attention_normed_decode_ && !attention_normed_decode_->valid()) ||
      (attention_q_decode_ && !attention_q_decode_->valid()) ||
      (attention_k_decode_ && !attention_k_decode_->valid()) ||
      (attention_v_decode_ && !attention_v_decode_->valid()) ||
      (attention_output_fp32_decode_ && !attention_output_fp32_decode_->valid()) ||
      (attention_projected_decode_ && !attention_projected_decode_->valid()) ||
      (attention_query_bf16_decode_ && !attention_query_bf16_decode_->valid()) ||
      (attention_output_bf16_decode_ && !attention_output_bf16_decode_->valid())) {
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

DeviceTensorFp32* RequestExecutionContext::hidden() {
  return hidden_.get();
}

const DeviceTensorFp32* RequestExecutionContext::hidden() const {
  return hidden_.get();
}

DeviceTensorFp32* RequestExecutionContext::residual() {
  return residual_.get();
}

const DeviceTensorFp32* RequestExecutionContext::residual() const {
  return residual_.get();
}

DeviceTensorFp32* RequestExecutionContext::scratch() {
  return scratch_.get();
}

const DeviceTensorFp32* RequestExecutionContext::scratch() const {
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

DeviceTensorFp32* RequestExecutionContext::mamba_normalized_decode() {
  return mamba_normalized_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_normalized_decode() const {
  return mamba_normalized_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::mamba_projected_decode() {
  return mamba_projected_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_projected_decode() const {
  return mamba_projected_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::mamba_scan_output_decode() {
  return mamba_scan_output_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_scan_output_decode() const {
  return mamba_scan_output_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::mamba_projected_output_decode() {
  return mamba_projected_output_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_projected_output_decode() const {
  return mamba_projected_output_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::expert_intermediate_scratch() {
  return expert_intermediate_scratch_.get();
}

const DeviceTensorFp32* RequestExecutionContext::expert_intermediate_scratch() const {
  return expert_intermediate_scratch_.get();
}

DeviceTensorFp32* RequestExecutionContext::expert_aux_scratch() {
  return expert_aux_scratch_.get();
}

const DeviceTensorFp32* RequestExecutionContext::expert_aux_scratch() const {
  return expert_aux_scratch_.get();
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

DeviceBuffer<std::int32_t>* RequestExecutionContext::token_ids_device() {
  return &token_ids_device_;
}

const DeviceBuffer<std::int32_t>* RequestExecutionContext::token_ids_device() const {
  return &token_ids_device_;
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
    bool pages_changed = false;
    if (pages.size() < required_pages) {
      const std::size_t additional_pages = required_pages - pages.size();
      const auto allocated = kv_arena_->AllocatePages(layer_index, additional_pages);
      if (!allocated.has_value()) {
        return false;
      }
      pages.insert(pages.end(), allocated->begin(), allocated->end());
      pages_changed = true;
    }
    if (pages.empty()) {
      if (!kv_page_ids_device_by_layer_[layer_index].Resize(0)) {
        return false;
      }
      continue;
    }
    if (!pages_changed &&
        kv_page_ids_device_by_layer_[layer_index].count() == pages.size()) {
      continue;
    }

    std::vector<std::int32_t> page_ids;
    page_ids.reserve(pages.size());
    for (const KvPageHandle& handle : pages) {
      page_ids.push_back(static_cast<std::int32_t>(handle.page_id));
    }
    if (!kv_page_ids_device_by_layer_[layer_index].Resize(page_ids.size()) ||
        !kv_page_ids_device_by_layer_[layer_index].CopyFromHost(page_ids)) {
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

DeviceBuffer<std::int32_t>* RequestExecutionContext::kv_page_ids_device(std::size_t layer_index) {
  if (layer_index >= kv_page_ids_device_by_layer_.size()) {
    return nullptr;
  }
  return &kv_page_ids_device_by_layer_[layer_index];
}

const DeviceBuffer<std::int32_t>* RequestExecutionContext::kv_page_ids_device(
    std::size_t layer_index) const {
  if (layer_index >= kv_page_ids_device_by_layer_.size()) {
    return nullptr;
  }
  return &kv_page_ids_device_by_layer_[layer_index];
}

bool RequestExecutionContext::EnsureAttentionAuxCapacity(
    std::size_t sequence_count,
    std::size_t page_table_entries) {
  return attention_seq_len_q_device_.Resize(sequence_count) &&
         attention_seq_len_kv_device_.Resize(sequence_count) &&
         attention_page_table_device_.Resize(page_table_entries);
}

DeviceBuffer<std::int32_t>* RequestExecutionContext::attention_seq_len_q_device() {
  return &attention_seq_len_q_device_;
}

const DeviceBuffer<std::int32_t>* RequestExecutionContext::attention_seq_len_q_device() const {
  return &attention_seq_len_q_device_;
}

DeviceBuffer<std::int32_t>* RequestExecutionContext::attention_seq_len_kv_device() {
  return &attention_seq_len_kv_device_;
}

const DeviceBuffer<std::int32_t>* RequestExecutionContext::attention_seq_len_kv_device() const {
  return &attention_seq_len_kv_device_;
}

DeviceBuffer<std::int32_t>* RequestExecutionContext::attention_page_table_device() {
  return &attention_page_table_device_;
}

const DeviceBuffer<std::int32_t>* RequestExecutionContext::attention_page_table_device() const {
  return &attention_page_table_device_;
}

DeviceTensorFp32* RequestExecutionContext::attention_normed_decode() {
  return attention_normed_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::attention_normed_decode() const {
  return attention_normed_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::attention_q_decode() {
  return attention_q_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::attention_q_decode() const {
  return attention_q_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::attention_k_decode() {
  return attention_k_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::attention_k_decode() const {
  return attention_k_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::attention_v_decode() {
  return attention_v_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::attention_v_decode() const {
  return attention_v_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::attention_output_fp32_decode() {
  return attention_output_fp32_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::attention_output_fp32_decode() const {
  return attention_output_fp32_decode_.get();
}

DeviceTensorFp32* RequestExecutionContext::attention_projected_decode() {
  return attention_projected_decode_.get();
}

const DeviceTensorFp32* RequestExecutionContext::attention_projected_decode() const {
  return attention_projected_decode_.get();
}

DeviceTensorBf16* RequestExecutionContext::attention_query_bf16_decode() {
  return attention_query_bf16_decode_.get();
}

const DeviceTensorBf16* RequestExecutionContext::attention_query_bf16_decode() const {
  return attention_query_bf16_decode_.get();
}

DeviceTensorBf16* RequestExecutionContext::attention_output_bf16_decode() {
  return attention_output_bf16_decode_.get();
}

const DeviceTensorBf16* RequestExecutionContext::attention_output_bf16_decode() const {
  return attention_output_bf16_decode_.get();
}

bool RequestExecutionContext::EnsureExpertSelectionCapacity(std::size_t selection_count) {
  if (selection_count <= expert_selection_capacity_) {
    return true;
  }
  if (!expert_selection_indices_device_.Resize(selection_count) ||
      !expert_selection_weights_device_.Resize(selection_count)) {
    return false;
  }
  expert_selection_indices_host_.resize(selection_count, -1);
  expert_selection_weights_host_.resize(selection_count, 0.0f);
  expert_selection_capacity_ = selection_count;
  return true;
}

std::size_t RequestExecutionContext::expert_selection_capacity() const {
  return expert_selection_capacity_;
}

DeviceBuffer<std::int32_t>* RequestExecutionContext::expert_selection_indices_device() {
  return &expert_selection_indices_device_;
}

const DeviceBuffer<std::int32_t>* RequestExecutionContext::expert_selection_indices_device() const {
  return &expert_selection_indices_device_;
}

DeviceBuffer<float>* RequestExecutionContext::expert_selection_weights_device() {
  return &expert_selection_weights_device_;
}

const DeviceBuffer<float>* RequestExecutionContext::expert_selection_weights_device() const {
  return &expert_selection_weights_device_;
}

std::vector<std::int32_t>* RequestExecutionContext::expert_selection_indices_host() {
  return &expert_selection_indices_host_;
}

const std::vector<std::int32_t>* RequestExecutionContext::expert_selection_indices_host() const {
  return &expert_selection_indices_host_;
}

std::vector<float>* RequestExecutionContext::expert_selection_weights_host() {
  return &expert_selection_weights_host_;
}

const std::vector<float>* RequestExecutionContext::expert_selection_weights_host() const {
  return &expert_selection_weights_host_;
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
  if (mamba_normalized_decode_) {
    ok = ok && mamba_normalized_decode_->FillZero();
  }
  if (mamba_projected_decode_) {
    ok = ok && mamba_projected_decode_->FillZero();
  }
  if (mamba_scan_output_decode_) {
    ok = ok && mamba_scan_output_decode_->FillZero();
  }
  if (mamba_projected_output_decode_) {
    ok = ok && mamba_projected_output_decode_->FillZero();
  }
  if (key_cache_) {
    ok = ok && key_cache_->FillZero();
  }
  if (value_cache_) {
    ok = ok && value_cache_->FillZero();
  }
  if (attention_normed_decode_) {
    ok = ok && attention_normed_decode_->FillZero();
  }
  if (attention_q_decode_) {
    ok = ok && attention_q_decode_->FillZero();
  }
  if (attention_k_decode_) {
    ok = ok && attention_k_decode_->FillZero();
  }
  if (attention_v_decode_) {
    ok = ok && attention_v_decode_->FillZero();
  }
  if (attention_output_fp32_decode_) {
    ok = ok && attention_output_fp32_decode_->FillZero();
  }
  if (attention_projected_decode_) {
    ok = ok && attention_projected_decode_->FillZero();
  }
  if (attention_query_bf16_decode_) {
    ok = ok && attention_query_bf16_decode_->FillZero();
  }
  if (attention_output_bf16_decode_) {
    ok = ok && attention_output_bf16_decode_->FillZero();
  }

  if (kv_arena_.has_value()) {
    std::vector<std::size_t> page_ids;
    page_ids.reserve(allocated_kv_pages());
    for (const auto& layer_pages : kv_pages_by_layer_) {
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
  for (auto& page_ids_device : kv_page_ids_device_by_layer_) {
    ok = ok && page_ids_device.Resize(0);
  }
  sequence_length_ = 0;
  decode_position_ = 0;
  return ok;
}

}  // namespace nemotron
