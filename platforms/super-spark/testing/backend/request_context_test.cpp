#include "nemotron/request_context.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::KvCacheDataType;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = __float2bfloat16(values[i]);
  }
  return converted;
}

bool copy_bf16_tensor_to_host(
    const nemotron::DeviceTensorBf16& tensor,
    std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  if (!tensor.CopyToHost(host_bf16.data(), host_bf16.size())) {
    return false;
  }
  output->resize(host_bf16.size(), 0.0f);
  for (std::size_t i = 0; i < host_bf16.size(); ++i) {
    (*output)[i] = __bfloat162float(host_bf16[i]);
  }
  return true;
}

RequestExecutionConfig make_config() {
  RequestExecutionConfig config;
  config.hidden_size = 16;
  config.max_tokens = 12;
  config.scratch_tokens = 4;
  config.attention_kv_cache = AttentionKvCacheConfig{
      2,
      2,
      8,
      4,
      KvCacheDataType::kBf16,
  };
  config.attention_total_pages = 8;
  config.mamba_hidden_size = 16;
  config.mamba_projection_size = 48;
  config.mamba_intermediate_size = 24;
  config.mamba_conv_state_bytes = 32 * sizeof(__nv_bfloat16);
  config.mamba_state_bytes_fp32 = 64 * sizeof(float);
  config.expert_selection_capacity = 24;
  config.expert_intermediate_scratch_numel = 96;
  config.expert_aux_scratch_numel = 128;
  return config;
}

bool test_request_context_allocates_buffers_and_kv_pages() {
  auto context = RequestExecutionContext::Create(make_config());
  if (!context || !context->valid()) {
    std::cout << "request_context_test: SKIP (no CUDA device available)\n";
    return true;
  }

  if (!expect(context->hidden() != nullptr && context->hidden()->shape() == std::vector<std::size_t>({12, 16}),
              "hidden buffer should use max_tokens x hidden_size shape")) {
    return false;
  }
  if (!expect(context->residual() != nullptr && context->residual()->shape() == std::vector<std::size_t>({12, 16}),
              "residual buffer should mirror hidden shape")) {
    return false;
  }
  if (!expect(context->scratch() != nullptr && context->scratch()->shape() == std::vector<std::size_t>({4, 16}),
              "scratch buffer should use the configured scratch-token window")) {
    return false;
  }
  if (!expect(context->token_ids_device() != nullptr && context->token_ids_device()->count() == 12,
              "request context should allocate reusable device token-id storage")) {
    return false;
  }
  if (!expect(context->mamba_state() != nullptr && context->mamba_state()->numel() == 64,
              "mamba state buffer should allocate the configured fp32 state size")) {
    return false;
  }
  if (!expect(context->mamba_conv_state() != nullptr && context->mamba_conv_state()->numel() == 32,
              "mamba conv-state buffer should allocate the configured fp32 state size")) {
    return false;
  }
  if (!expect(context->mamba_normalized_decode() != nullptr &&
                  context->mamba_normalized_decode()->shape() == std::vector<std::size_t>({1, 16}),
              "request context should allocate reusable single-token mamba norm scratch")) {
    return false;
  }
  if (!expect(context->mamba_projected_decode() != nullptr &&
                  context->mamba_projected_decode()->shape() == std::vector<std::size_t>({1, 48}),
              "request context should allocate reusable single-token mamba projection scratch")) {
    return false;
  }
  if (!expect(context->mamba_scan_output_decode() != nullptr &&
                  context->mamba_scan_output_decode()->shape() == std::vector<std::size_t>({1, 24}),
              "request context should allocate reusable single-token mamba scan scratch")) {
    return false;
  }
  if (!expect(context->mamba_projected_output_decode() != nullptr &&
                  context->mamba_projected_output_decode()->shape() == std::vector<std::size_t>({1, 16}),
              "request context should allocate reusable single-token mamba output scratch")) {
    return false;
  }
  if (!expect(context->expert_intermediate_scratch() != nullptr &&
                  context->expert_intermediate_scratch()->numel() == 96,
              "request context should allocate reusable expert intermediate scratch when configured")) {
    return false;
  }
  if (!expect(context->expert_aux_scratch() != nullptr &&
                  context->expert_aux_scratch()->numel() == 128,
              "request context should allocate reusable expert aux scratch when configured")) {
    return false;
  }
  if (!expect(context->key_cache() != nullptr &&
                  context->key_cache()->shape() == std::vector<std::size_t>({8, 2, 4, 8}),
              "key cache should allocate the configured BF16 page container")) {
    return false;
  }
  if (!expect(context->value_cache() != nullptr &&
                  context->value_cache()->shape() == std::vector<std::size_t>({8, 2, 4, 8}),
              "value cache should allocate the configured BF16 page container")) {
    return false;
  }
  if (!expect(context->SetSequenceLength(5), "setting sequence length within bounds should succeed")) {
    return false;
  }
  if (!expect(context->sequence_length() == 5 && context->decode_position() == 5,
              "sequence length and decode position should track the prefill boundary")) {
    return false;
  }
  if (!expect(context->allocated_kv_pages(0) == 2 && context->allocated_kv_pages(1) == 2,
              "5 tokens at 4 tokens/page should allocate two pages per attention layer")) {
    return false;
  }
  if (!expect(context->kv_page_ids_device(0) != nullptr &&
                  context->kv_page_ids_device(0)->count() == 2 &&
                  context->kv_page_ids_device(1) != nullptr &&
                  context->kv_page_ids_device(1)->count() == 2,
              "request context should keep per-layer page ids resident on device")) {
    return false;
  }
  if (!expect(context->EnsureAttentionAuxCapacity(1, 2),
              "request context should allocate reusable attention aux buffers")) {
    return false;
  }
  if (!expect(context->attention_seq_len_q_device() != nullptr &&
                  context->attention_seq_len_q_device()->count() == 1 &&
                  context->attention_seq_len_kv_device() != nullptr &&
                  context->attention_seq_len_kv_device()->count() == 1 &&
                  context->attention_page_table_device() != nullptr &&
                  context->attention_page_table_device()->count() == 2,
              "request context should retain reusable device attention aux storage")) {
    return false;
  }
  if (!expect(context->expert_selection_capacity() == 24,
              "request context should honor the configured expert-selection scratch capacity")) {
    return false;
  }
  if (!expect(context->AdvanceDecodePosition(3), "advancing decode within bounds should succeed")) {
    return false;
  }
  if (!expect(context->sequence_length() == 8 && context->decode_position() == 8,
              "decode advance should extend both length and decode position")) {
    return false;
  }
  if (!expect(context->allocated_kv_pages() == 4,
              "8 tokens should still fit inside the previously allocated two pages per layer")) {
    return false;
  }
  if (!expect(context->AdvanceDecodePosition(1), "crossing the page boundary should still succeed")) {
    return false;
  }
  if (!expect(context->allocated_kv_pages(0) == 3 && context->allocated_kv_pages(1) == 3,
              "moving to 9 tokens should allocate a third page per attention layer")) {
    return false;
  }
  return expect(!context->AdvanceDecodePosition(4), "advancing beyond max_tokens should fail");
}

bool test_request_context_reset_releases_pages_and_clears_positions() {
  auto context = RequestExecutionContext::Create(make_config());
  if (!context || !context->valid()) {
    std::cout << "request_context_test: SKIP (no CUDA device available)\n";
    return true;
  }

  if (!expect(context->SetSequenceLength(7), "sequence setup should succeed")) {
    return false;
  }
  std::vector<float> state_values(32, 1.0f);
  const std::vector<__nv_bfloat16> state_values_bf16 = to_bf16(state_values);
  std::vector<float> projected_values(48, 1.0f);
  if (!expect(
          context->mamba_conv_state()->CopyFromHost(
              state_values_bf16.data(),
              state_values_bf16.size()),
          "mamba conv-state upload should succeed")) {
    return false;
  }
  if (!expect(
          context->mamba_projected_decode()->CopyFromHost(projected_values.data(), projected_values.size()),
          "mamba decode scratch upload should succeed")) {
    return false;
  }
  if (!expect(context->allocated_kv_pages() == 4, "two pages per layer should be allocated at 7 tokens")) {
    return false;
  }
  if (!expect(context->ResetForNewRequest(), "reset should succeed")) {
    return false;
  }
  if (!expect(context->sequence_length() == 0 && context->decode_position() == 0,
              "reset should clear sequence accounting")) {
    return false;
  }
  if (!expect(context->allocated_kv_pages() == 0, "reset should release all request-local kv pages")) {
    return false;
  }
  if (!expect(context->kv_page_ids_device(0) != nullptr &&
                  context->kv_page_ids_device(0)->count() == 0,
              "reset should clear request-local device page-id buffers")) {
    return false;
  }
  if (!expect(
          copy_bf16_tensor_to_host(*context->mamba_conv_state(), &state_values),
          "mamba conv-state download should succeed after reset")) {
    return false;
  }
  if (!expect(
          std::all_of(state_values.begin(), state_values.end(), [](float value) { return value == 0.0f; }),
          "reset should zero the request-local mamba conv-state")) {
    return false;
  }
  if (!expect(
          context->mamba_projected_decode()->CopyToHost(projected_values.data(), projected_values.size()),
          "mamba decode scratch download should succeed after reset")) {
    return false;
  }
  if (!expect(
          std::all_of(projected_values.begin(), projected_values.end(), [](float value) { return value == 0.0f; }),
          "reset should zero the request-local mamba decode scratch")) {
    return false;
  }
  return expect(context->kv_pages(0) != nullptr && context->kv_pages(0)->empty(),
                "reset should leave empty per-layer page vectors behind");
}

}  // namespace

int main() {
  if (!test_request_context_allocates_buffers_and_kv_pages() ||
      !test_request_context_reset_releases_pages_and_clears_positions()) {
    return 1;
  }
  std::cout << "request_context_test: PASS\n";
  return 0;
}
