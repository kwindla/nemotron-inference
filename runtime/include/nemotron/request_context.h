#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "nemotron/device_tensor.h"
#include "nemotron/paged_kv_cache.h"

namespace nemotron {

struct RequestExecutionConfig {
  std::size_t hidden_size = 0;
  std::size_t max_tokens = 0;
  std::size_t scratch_tokens = 0;
  AttentionKvCacheConfig attention_kv_cache;
  std::size_t attention_total_pages = 0;
  std::size_t mamba_conv_state_bytes_fp32 = 0;
  std::size_t mamba_state_bytes_fp32 = 0;
};

class RequestExecutionContext {
 public:
  static std::unique_ptr<RequestExecutionContext> Create(const RequestExecutionConfig& config);

  RequestExecutionContext(RequestExecutionContext&&) noexcept;
  RequestExecutionContext& operator=(RequestExecutionContext&&) noexcept;
  ~RequestExecutionContext();

  RequestExecutionContext(const RequestExecutionContext&) = delete;
  RequestExecutionContext& operator=(const RequestExecutionContext&) = delete;

  bool valid() const;
  const RequestExecutionConfig& config() const;
  std::size_t sequence_length() const;
  std::size_t decode_position() const;

  DeviceTensorBf16* hidden();
  const DeviceTensorBf16* hidden() const;
  DeviceTensorBf16* residual();
  const DeviceTensorBf16* residual() const;
  DeviceTensorBf16* scratch();
  const DeviceTensorBf16* scratch() const;
  DeviceTensorFp32* mamba_state();
  const DeviceTensorFp32* mamba_state() const;
  DeviceTensorFp32* mamba_conv_state();
  const DeviceTensorFp32* mamba_conv_state() const;
  DeviceTensorBf16* key_cache();
  const DeviceTensorBf16* key_cache() const;
  DeviceTensorBf16* value_cache();
  const DeviceTensorBf16* value_cache() const;

  bool EnsureAttentionTokens(std::size_t token_count);
  std::size_t allocated_kv_pages() const;
  std::size_t allocated_kv_pages(std::size_t layer_index) const;
  const std::vector<KvPageHandle>* kv_pages(std::size_t layer_index) const;

  bool SetSequenceLength(std::size_t sequence_length);
  bool AdvanceDecodePosition(std::size_t token_count);
  bool ResetForNewRequest();

 private:
  RequestExecutionContext(
      RequestExecutionConfig config,
      std::unique_ptr<DeviceTensorBf16> hidden,
      std::unique_ptr<DeviceTensorBf16> residual,
      std::unique_ptr<DeviceTensorBf16> scratch,
      std::unique_ptr<DeviceTensorFp32> mamba_conv_state,
      std::unique_ptr<DeviceTensorFp32> mamba_state,
      std::unique_ptr<DeviceTensorBf16> key_cache,
      std::unique_ptr<DeviceTensorBf16> value_cache,
      std::optional<PagedKvCacheArena> kv_arena);

  RequestExecutionConfig config_;
  std::unique_ptr<DeviceTensorBf16> hidden_;
  std::unique_ptr<DeviceTensorBf16> residual_;
  std::unique_ptr<DeviceTensorBf16> scratch_;
  std::unique_ptr<DeviceTensorFp32> mamba_conv_state_;
  std::unique_ptr<DeviceTensorFp32> mamba_state_;
  std::unique_ptr<DeviceTensorBf16> key_cache_;
  std::unique_ptr<DeviceTensorBf16> value_cache_;
  std::optional<PagedKvCacheArena> kv_arena_;
  std::vector<std::vector<KvPageHandle>> kv_pages_by_layer_;
  std::size_t sequence_length_ = 0;
  std::size_t decode_position_ = 0;
};

}  // namespace nemotron
