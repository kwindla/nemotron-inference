#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_routing_device.h"
#include "nemotron/paged_kv_cache.h"

namespace nemotron {

struct MoePrefillWorkspaceConfig {
  std::size_t hidden_size = 0;
  std::size_t num_experts = 0;
  std::size_t top_k = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
};

struct RequestExecutionConfig {
  std::size_t hidden_size = 0;
  std::size_t max_tokens = 0;
  std::size_t scratch_tokens = 0;
  AttentionKvCacheConfig attention_kv_cache;
  std::size_t attention_total_pages = 0;
  std::size_t mamba_conv_state_bytes_fp32 = 0;
  std::size_t mamba_state_bytes_fp32 = 0;
  MoePrefillWorkspaceConfig moe_prefill_workspace_config;
  std::size_t moe_prefill_capacity_tokens = 0;
};

struct MoePrefillWorkspace {
  static std::optional<std::size_t> BytesForTokenCapacity(
      std::size_t token_capacity,
      const MoePrefillWorkspaceConfig& config);
  static std::unique_ptr<MoePrefillWorkspace> Create(
      std::size_t token_capacity,
      const MoePrefillWorkspaceConfig& config);

  bool valid() const;
  std::size_t token_capacity() const;

  MoePrefillWorkspaceConfig config;
  std::size_t token_capacity_value = 0;
  std::unique_ptr<DeviceTensorBf16> normalized_bf16;
  std::unique_ptr<DeviceTensorFp32> input_fp32;
  std::unique_ptr<DeviceTensorFp32> normalized;
  std::unique_ptr<DeviceTensorFp32> router_logits;
  std::unique_ptr<DeviceTensorFp32> output_fp32;
  std::unique_ptr<DeviceTensorInt32> topk_ids;
  std::unique_ptr<DeviceTensorFp32> topk_weights;
  std::unique_ptr<DeviceExpertRouting> fused_prefill_routing;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_routed_output_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_gather_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_expert_up_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_shared_up_scratch;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_gather_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_expert_up_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_shared_up_pack;
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
  DeviceTensorInt32* greedy_token_id_scratch();
  const DeviceTensorInt32* greedy_token_id_scratch() const;
  MoePrefillWorkspace* moe_prefill_workspace();
  const MoePrefillWorkspace* moe_prefill_workspace() const;

  bool EnsureAttentionTokens(std::size_t token_count);
  bool EnsureMoePrefillWorkspace(
      std::size_t token_capacity,
      const MoePrefillWorkspaceConfig& config);
  std::size_t allocated_kv_pages() const;
  std::size_t allocated_kv_pages(std::size_t layer_index) const;
  // These page handles are the live request-owned KV allocation. Snapshots copy
  // the bytes out of these pages; they do not share or retain page ownership.
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
      std::unique_ptr<DeviceTensorInt32> greedy_token_id_scratch,
      std::unique_ptr<MoePrefillWorkspace> moe_prefill_workspace,
      std::optional<PagedKvCacheArena> kv_arena);

  RequestExecutionConfig config_;
  std::unique_ptr<DeviceTensorBf16> hidden_;
  std::unique_ptr<DeviceTensorBf16> residual_;
  std::unique_ptr<DeviceTensorBf16> scratch_;
  std::unique_ptr<DeviceTensorFp32> mamba_conv_state_;
  std::unique_ptr<DeviceTensorFp32> mamba_state_;
  std::unique_ptr<DeviceTensorBf16> key_cache_;
  std::unique_ptr<DeviceTensorBf16> value_cache_;
  std::unique_ptr<DeviceTensorInt32> greedy_token_id_scratch_;
  std::unique_ptr<MoePrefillWorkspace> moe_prefill_workspace_;
  std::optional<PagedKvCacheArena> kv_arena_;
  std::vector<std::vector<KvPageHandle>> kv_pages_by_layer_;
  std::size_t sequence_length_ = 0;
  std::size_t decode_position_ = 0;
};

}  // namespace nemotron
