#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "nemotron/device_buffer.h"
#include "nemotron/device_tensor.h"
#include "nemotron/paged_kv_cache.h"

namespace nemotron {

struct RequestExecutionConfig {
  std::size_t hidden_size = 0;
  std::size_t max_tokens = 0;
  std::size_t scratch_tokens = 0;
  AttentionKvCacheConfig attention_kv_cache;
  std::size_t attention_total_pages = 0;
  std::size_t attention_query_head_count = 0;
  std::size_t attention_head_dim = 0;
  std::size_t mamba_hidden_size = 0;
  std::size_t mamba_projection_size = 0;
  std::size_t mamba_intermediate_size = 0;
  std::size_t mamba_conv_state_bytes_fp32 = 0;
  std::size_t mamba_state_bytes_fp32 = 0;
  std::size_t expert_selection_capacity = 0;
  std::size_t expert_intermediate_scratch_numel = 0;
  std::size_t expert_aux_scratch_numel = 0;
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

  DeviceTensorFp32* hidden();
  const DeviceTensorFp32* hidden() const;
  DeviceTensorFp32* residual();
  const DeviceTensorFp32* residual() const;
  DeviceTensorFp32* scratch();
  const DeviceTensorFp32* scratch() const;
  DeviceTensorBf16* hidden_decode_bf16();
  const DeviceTensorBf16* hidden_decode_bf16() const;
  DeviceTensorBf16* residual_decode_bf16();
  const DeviceTensorBf16* residual_decode_bf16() const;
  DeviceTensorBf16* scratch_decode_bf16();
  const DeviceTensorBf16* scratch_decode_bf16() const;
  DeviceTensorFp32* mamba_state();
  const DeviceTensorFp32* mamba_state() const;
  DeviceTensorFp32* mamba_conv_state();
  const DeviceTensorFp32* mamba_conv_state() const;
  DeviceTensorFp32* mamba_normalized_decode();
  const DeviceTensorFp32* mamba_normalized_decode() const;
  DeviceTensorFp32* mamba_projected_decode();
  const DeviceTensorFp32* mamba_projected_decode() const;
  DeviceTensorFp32* mamba_scan_output_decode();
  const DeviceTensorFp32* mamba_scan_output_decode() const;
  DeviceTensorFp32* mamba_projected_output_decode();
  const DeviceTensorFp32* mamba_projected_output_decode() const;
  DeviceTensorBf16* key_cache();
  const DeviceTensorBf16* key_cache() const;
  DeviceTensorFp8E4M3* key_cache_fp8();
  const DeviceTensorFp8E4M3* key_cache_fp8() const;
  DeviceTensorBf16* value_cache();
  const DeviceTensorBf16* value_cache() const;
  DeviceTensorFp8E4M3* value_cache_fp8();
  const DeviceTensorFp8E4M3* value_cache_fp8() const;
  void* key_cache_data();
  const void* key_cache_data() const;
  void* value_cache_data();
  const void* value_cache_data() const;
  DeviceBuffer<std::int32_t>* token_ids_device();
  const DeviceBuffer<std::int32_t>* token_ids_device() const;

  bool EnsureAttentionTokens(std::size_t token_count);
  std::size_t allocated_kv_pages() const;
  std::size_t allocated_kv_pages(std::size_t layer_index) const;
  const std::vector<KvPageHandle>* kv_pages(std::size_t layer_index) const;
  DeviceBuffer<std::int32_t>* kv_page_ids_device(std::size_t layer_index);
  const DeviceBuffer<std::int32_t>* kv_page_ids_device(std::size_t layer_index) const;
  bool EnsureAttentionAuxCapacity(std::size_t sequence_count, std::size_t page_table_entries);
  DeviceBuffer<std::int32_t>* attention_seq_len_q_device();
  const DeviceBuffer<std::int32_t>* attention_seq_len_q_device() const;
  DeviceBuffer<std::int32_t>* attention_seq_len_kv_device();
  const DeviceBuffer<std::int32_t>* attention_seq_len_kv_device() const;
  DeviceBuffer<std::int32_t>* attention_page_table_device();
  const DeviceBuffer<std::int32_t>* attention_page_table_device() const;
  DeviceBuffer<std::int32_t>* attention_decode_seq_len_q_device();
  const DeviceBuffer<std::int32_t>* attention_decode_seq_len_q_device() const;
  DeviceBuffer<std::int32_t>* attention_decode_seq_len_kv_device();
  const DeviceBuffer<std::int32_t>* attention_decode_seq_len_kv_device() const;
  DeviceBuffer<std::int32_t>* attention_decode_page_table_device(std::size_t layer_index);
  const DeviceBuffer<std::int32_t>* attention_decode_page_table_device(std::size_t layer_index) const;
  DeviceTensorFp32* attention_normed_decode();
  const DeviceTensorFp32* attention_normed_decode() const;
  DeviceTensorFp32* attention_q_decode();
  const DeviceTensorFp32* attention_q_decode() const;
  DeviceTensorFp32* attention_k_decode();
  const DeviceTensorFp32* attention_k_decode() const;
  DeviceTensorFp32* attention_v_decode();
  const DeviceTensorFp32* attention_v_decode() const;
  DeviceTensorFp32* attention_output_fp32_decode();
  const DeviceTensorFp32* attention_output_fp32_decode() const;
  DeviceTensorFp32* attention_projected_decode();
  const DeviceTensorFp32* attention_projected_decode() const;
  DeviceTensorBf16* attention_query_bf16_decode();
  const DeviceTensorBf16* attention_query_bf16_decode() const;
  DeviceTensorFp8E4M3* attention_query_fp8_decode();
  const DeviceTensorFp8E4M3* attention_query_fp8_decode() const;
  DeviceTensorBf16* attention_output_bf16_decode();
  const DeviceTensorBf16* attention_output_bf16_decode() const;

  bool EnsureExpertSelectionCapacity(std::size_t selection_count);
  std::size_t expert_selection_capacity() const;
  DeviceBuffer<std::int32_t>* expert_selection_indices_device();
  const DeviceBuffer<std::int32_t>* expert_selection_indices_device() const;
  DeviceBuffer<float>* expert_selection_weights_device();
  const DeviceBuffer<float>* expert_selection_weights_device() const;
  std::vector<std::int32_t>* expert_selection_indices_host();
  const std::vector<std::int32_t>* expert_selection_indices_host() const;
  std::vector<float>* expert_selection_weights_host();
  const std::vector<float>* expert_selection_weights_host() const;
  DeviceTensorFp32* expert_intermediate_scratch();
  const DeviceTensorFp32* expert_intermediate_scratch() const;
  DeviceTensorFp32* expert_aux_scratch();
  const DeviceTensorFp32* expert_aux_scratch() const;

  bool SetSequenceLength(std::size_t sequence_length);
  bool AdvanceDecodePosition(std::size_t token_count);
  bool ResetForNewRequest();

 private:
  RequestExecutionContext(
      RequestExecutionConfig config,
      std::unique_ptr<DeviceTensorFp32> hidden,
      std::unique_ptr<DeviceTensorFp32> residual,
      std::unique_ptr<DeviceTensorFp32> scratch,
      std::unique_ptr<DeviceTensorBf16> hidden_decode_bf16,
      std::unique_ptr<DeviceTensorBf16> residual_decode_bf16,
      std::unique_ptr<DeviceTensorBf16> scratch_decode_bf16,
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
      std::unique_ptr<DeviceTensorFp8E4M3> attention_query_fp8_decode,
      std::unique_ptr<DeviceTensorBf16> attention_output_bf16_decode,
      std::unique_ptr<DeviceTensorFp32> expert_intermediate_scratch,
      std::unique_ptr<DeviceTensorFp32> expert_aux_scratch,
      std::unique_ptr<DeviceTensorBf16> key_cache,
      std::unique_ptr<DeviceTensorBf16> value_cache,
      std::unique_ptr<DeviceTensorFp8E4M3> key_cache_fp8,
      std::unique_ptr<DeviceTensorFp8E4M3> value_cache_fp8,
      std::optional<PagedKvCacheArena> kv_arena);

  bool InitializeAttentionDecodeMetadata();
  bool SetAttentionDecodeSequenceLength(std::size_t sequence_length);
  bool AdvanceAttentionDecodeSequenceLength(std::size_t token_count);

  RequestExecutionConfig config_;
  std::unique_ptr<DeviceTensorFp32> hidden_;
  std::unique_ptr<DeviceTensorFp32> residual_;
  std::unique_ptr<DeviceTensorFp32> scratch_;
  std::unique_ptr<DeviceTensorBf16> hidden_decode_bf16_;
  std::unique_ptr<DeviceTensorBf16> residual_decode_bf16_;
  std::unique_ptr<DeviceTensorBf16> scratch_decode_bf16_;
  std::unique_ptr<DeviceTensorFp32> mamba_conv_state_;
  std::unique_ptr<DeviceTensorFp32> mamba_state_;
  std::unique_ptr<DeviceTensorFp32> mamba_normalized_decode_;
  std::unique_ptr<DeviceTensorFp32> mamba_projected_decode_;
  std::unique_ptr<DeviceTensorFp32> mamba_scan_output_decode_;
  std::unique_ptr<DeviceTensorFp32> mamba_projected_output_decode_;
  std::unique_ptr<DeviceTensorFp32> attention_normed_decode_;
  std::unique_ptr<DeviceTensorFp32> attention_q_decode_;
  std::unique_ptr<DeviceTensorFp32> attention_k_decode_;
  std::unique_ptr<DeviceTensorFp32> attention_v_decode_;
  std::unique_ptr<DeviceTensorFp32> attention_output_fp32_decode_;
  std::unique_ptr<DeviceTensorFp32> attention_projected_decode_;
  std::unique_ptr<DeviceTensorBf16> attention_query_bf16_decode_;
  std::unique_ptr<DeviceTensorFp8E4M3> attention_query_fp8_decode_;
  std::unique_ptr<DeviceTensorBf16> attention_output_bf16_decode_;
  std::unique_ptr<DeviceTensorFp32> expert_intermediate_scratch_;
  std::unique_ptr<DeviceTensorFp32> expert_aux_scratch_;
  std::unique_ptr<DeviceTensorBf16> key_cache_;
  std::unique_ptr<DeviceTensorBf16> value_cache_;
  std::unique_ptr<DeviceTensorFp8E4M3> key_cache_fp8_;
  std::unique_ptr<DeviceTensorFp8E4M3> value_cache_fp8_;
  DeviceBuffer<std::int32_t> token_ids_device_;
  std::optional<PagedKvCacheArena> kv_arena_;
  std::vector<std::vector<KvPageHandle>> kv_pages_by_layer_;
  std::vector<DeviceBuffer<std::int32_t>> kv_page_ids_device_by_layer_;
  DeviceBuffer<std::int32_t> attention_seq_len_q_device_;
  DeviceBuffer<std::int32_t> attention_seq_len_kv_device_;
  DeviceBuffer<std::int32_t> attention_page_table_device_;
  DeviceBuffer<std::int32_t> attention_decode_seq_len_q_device_;
  DeviceBuffer<std::int32_t> attention_decode_seq_len_kv_device_;
  DeviceBuffer<std::int32_t> attention_decode_seq_len_values_device_;
  std::vector<DeviceBuffer<std::int32_t>> attention_decode_page_tables_by_layer_;
  std::vector<std::size_t> attention_decode_page_table_counts_by_layer_;
  DeviceBuffer<std::int32_t> expert_selection_indices_device_;
  DeviceBuffer<float> expert_selection_weights_device_;
  std::vector<std::int32_t> expert_selection_indices_host_;
  std::vector<float> expert_selection_weights_host_;
  std::size_t expert_selection_capacity_ = 0;
  std::size_t sequence_length_ = 0;
  std::size_t decode_position_ = 0;
};

}  // namespace nemotron
