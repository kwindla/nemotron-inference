#include "nemotron/state_snapshot.h"

#include <cstddef>
#include <iostream>
#include <optional>

namespace nemotron {
namespace {

bool HasAttentionState(const RequestExecutionContext& request_context) {
  return request_context.key_cache() != nullptr &&
         request_context.value_cache() != nullptr &&
         request_context.key_cache()->valid() &&
         request_context.value_cache()->valid();
}

bool HasMambaState(const RequestExecutionContext& request_context) {
  return request_context.mamba_conv_state() != nullptr &&
         request_context.mamba_state() != nullptr &&
         request_context.mamba_conv_state()->valid() &&
         request_context.mamba_state()->valid();
}

bool HasActivationState(const RequestExecutionContext& request_context) {
  return request_context.hidden() != nullptr &&
         request_context.residual() != nullptr &&
         request_context.hidden()->valid() &&
         request_context.residual()->valid();
}

bool LayersOwnExpectedLivePages(
    const RequestExecutionContext& request_context,
    std::size_t expected_pages_per_layer) {
  for (std::size_t layer_index = 0;
       layer_index < request_context.config().attention_kv_cache.layer_count;
       ++layer_index) {
    if (request_context.allocated_kv_pages(layer_index) != expected_pages_per_layer) {
      return false;
    }
  }
  return true;
}

std::size_t LiveKvTensorBytes(const RequestExecutionContext& request_context) {
  if (!HasAttentionState(request_context) || request_context.config().attention_total_pages == 0) {
    return 0;
  }
  const std::size_t total_pages = request_context.config().attention_total_pages;
  if (request_context.key_cache()->bytes() % total_pages != 0 ||
      request_context.value_cache()->bytes() % total_pages != 0) {
    return 0;
  }
  const std::size_t live_pages = request_context.allocated_kv_pages();
  if (live_pages > total_pages) {
    return 0;
  }
  return live_pages * (request_context.key_cache()->bytes() / total_pages);
}

std::optional<ReusableStateSnapshotLayout> BuildSnapshotLayout(
    const RequestExecutionContext& request_context,
    std::size_t prefix_token_count) {
  if (!request_context.valid() ||
      !HasAttentionState(request_context) ||
      !HasActivationState(request_context) ||
      prefix_token_count == 0) {
    return std::nullopt;
  }

  ReusableStateSnapshotLayout layout;
  layout.prefix_token_count = prefix_token_count;
  layout.attention_kv_cache = request_context.config().attention_kv_cache;
  layout.live_kv_pages_per_layer =
      RequiredPagesForTokens(request_context.config().attention_kv_cache, prefix_token_count);
  layout.hidden_bytes = request_context.hidden()->bytes();
  layout.residual_bytes = request_context.residual()->bytes();
  layout.mamba_conv_bytes =
      HasMambaState(request_context) ? request_context.mamba_conv_state()->bytes() : 0;
  layout.mamba_ssm_bytes =
      HasMambaState(request_context) ? request_context.mamba_state()->bytes() : 0;
  if (!layout.valid()) {
    return std::nullopt;
  }
  return layout;
}

bool LayoutMatchesRequest(
    const ReusableStateSnapshotLayout& descriptor_layout,
    const ReusableStateSnapshotLayout& request_layout) {
  return descriptor_layout == request_layout;
}

bool DescribeLayoutMismatch(
    const ReusableStateSnapshotLayout& descriptor_layout,
    const ReusableStateSnapshotLayout& request_layout) {
  bool matches = true;
  if (descriptor_layout.prefix_token_count != request_layout.prefix_token_count) {
    std::cerr << "state_snapshot: restore rejected: prefix length "
              << descriptor_layout.prefix_token_count
              << " does not match requested prefix length "
              << request_layout.prefix_token_count << "\n";
    matches = false;
  }
  if (descriptor_layout.attention_kv_cache.layer_count != request_layout.attention_kv_cache.layer_count ||
      descriptor_layout.attention_kv_cache.kv_head_count !=
          request_layout.attention_kv_cache.kv_head_count ||
      descriptor_layout.attention_kv_cache.head_dim != request_layout.attention_kv_cache.head_dim ||
      descriptor_layout.attention_kv_cache.tokens_per_page !=
          request_layout.attention_kv_cache.tokens_per_page ||
      descriptor_layout.attention_kv_cache.dtype != request_layout.attention_kv_cache.dtype) {
    std::cerr << "state_snapshot: restore rejected: attention page geometry mismatch\n";
    matches = false;
  }
  if (descriptor_layout.live_kv_pages_per_layer != request_layout.live_kv_pages_per_layer) {
    std::cerr << "state_snapshot: restore rejected: live KV-page count per layer "
              << descriptor_layout.live_kv_pages_per_layer
              << " does not match expected "
              << request_layout.live_kv_pages_per_layer << "\n";
    matches = false;
  }
  if (descriptor_layout.hidden_bytes != request_layout.hidden_bytes ||
      descriptor_layout.residual_bytes != request_layout.residual_bytes ||
      descriptor_layout.mamba_conv_bytes != request_layout.mamba_conv_bytes ||
      descriptor_layout.mamba_ssm_bytes != request_layout.mamba_ssm_bytes) {
    std::cerr << "state_snapshot: restore rejected: activation or Mamba layout mismatch\n";
    matches = false;
  }
  return matches;
}

}  // namespace

std::size_t RequiredKvSnapshotBytes(const RequestExecutionContext& request_context) {
  const std::size_t live_key_bytes = LiveKvTensorBytes(request_context);
  return live_key_bytes == 0 ? 0 : (2 * live_key_bytes);
}

std::size_t RequiredMambaSnapshotBytes(const RequestExecutionContext& request_context) {
  if (!HasActivationState(request_context)) {
    return 0;
  }
  std::size_t bytes = request_context.hidden()->bytes() + request_context.residual()->bytes();
  if (HasMambaState(request_context)) {
    bytes += request_context.mamba_conv_state()->bytes() + request_context.mamba_state()->bytes();
  }
  return bytes;
}

std::optional<ReusableStateDescriptor> SnapshotRequestState(
    ReusableStateArena& arena,
    const RequestExecutionContext& request_context,
    const std::string& label) {
  if (!request_context.valid()) {
    return std::nullopt;
  }

  const auto layout = BuildSnapshotLayout(request_context, request_context.sequence_length());
  const std::size_t kv_bytes = RequiredKvSnapshotBytes(request_context);
  const std::size_t mamba_bytes = RequiredMambaSnapshotBytes(request_context);
  if (!layout.has_value() || kv_bytes == 0 || mamba_bytes == 0) {
    return std::nullopt;
  }
  if (!LayersOwnExpectedLivePages(request_context, layout->live_kv_pages_per_layer)) {
    return std::nullopt;
  }

  ReusableStateDescriptor descriptor = arena.AllocateDescriptor(kv_bytes, mamba_bytes, label);
  if (!descriptor.valid()) {
    return std::nullopt;
  }
  descriptor.snapshot_layout = *layout;

  const std::size_t key_bytes = LiveKvTensorBytes(request_context);
  const std::size_t value_bytes = key_bytes;
  const std::size_t hidden_bytes = request_context.hidden()->bytes();
  const std::size_t residual_bytes = request_context.residual()->bytes();
  const std::size_t conv_bytes =
      HasMambaState(request_context) ? request_context.mamba_conv_state()->bytes() : 0;
  const std::size_t ssm_bytes =
      HasMambaState(request_context) ? request_context.mamba_state()->bytes() : 0;
  const bool copied =
      arena.CopyFromDevice(
          descriptor.kv_state,
          /*offset_bytes=*/0,
          request_context.key_cache()->data(),
          key_bytes) &&
      arena.CopyFromDevice(
          descriptor.kv_state,
          /*offset_bytes=*/key_bytes,
          request_context.value_cache()->data(),
          value_bytes) &&
      arena.CopyFromDevice(
          descriptor.mamba_state,
          /*offset_bytes=*/0,
          request_context.hidden()->data(),
          hidden_bytes) &&
      arena.CopyFromDevice(
          descriptor.mamba_state,
          /*offset_bytes=*/hidden_bytes,
          request_context.residual()->data(),
          residual_bytes) &&
      (!HasMambaState(request_context) ||
       (arena.CopyFromDevice(
            descriptor.mamba_state,
            /*offset_bytes=*/hidden_bytes + residual_bytes,
            request_context.mamba_conv_state()->data(),
            conv_bytes) &&
        arena.CopyFromDevice(
            descriptor.mamba_state,
            /*offset_bytes=*/hidden_bytes + residual_bytes + conv_bytes,
            request_context.mamba_state()->data(),
            ssm_bytes)));
  if (!copied) {
    arena.Release(descriptor);
    return std::nullopt;
  }
  return descriptor;
}

bool RestoreRequestState(
    const ReusableStateArena& arena,
    const ReusableStateDescriptor& descriptor,
    std::size_t expected_prefix_token_count,
    RequestExecutionContext& request_context) {
  if (!descriptor.valid()) {
    std::cerr << "state_snapshot: restore rejected: descriptor is invalid\n";
    return false;
  }
  if (!descriptor.has_snapshot_layout()) {
    std::cerr << "state_snapshot: restore rejected: descriptor is missing snapshot layout metadata\n";
    return false;
  }
  if (expected_prefix_token_count == 0 ||
      descriptor.snapshot_layout.prefix_token_count != expected_prefix_token_count) {
    std::cerr << "state_snapshot: restore rejected: descriptor prefix length "
              << descriptor.snapshot_layout.prefix_token_count
              << " does not match requested prefix length "
              << expected_prefix_token_count << "\n";
    return false;
  }
  if (!request_context.valid()) {
    std::cerr << "state_snapshot: restore rejected: target request context is invalid\n";
    return false;
  }
  if (!HasActivationState(request_context) || !HasAttentionState(request_context)) {
    std::cerr << "state_snapshot: restore rejected: target request is missing attention or activation state\n";
    return false;
  }

  const auto expected_layout =
      BuildSnapshotLayout(request_context, expected_prefix_token_count);
  if (!expected_layout.has_value()) {
    std::cerr << "state_snapshot: restore rejected: target request cannot represent the cached prefix layout\n";
    return false;
  }
  if (!LayoutMatchesRequest(descriptor.snapshot_layout, *expected_layout)) {
    DescribeLayoutMismatch(descriptor.snapshot_layout, *expected_layout);
    return false;
  }

  if (!request_context.ResetForNewRequest() ||
      !request_context.SetSequenceLength(expected_prefix_token_count)) {
    std::cerr << "state_snapshot: restore rejected: target request could not allocate live pages for prefix "
              << expected_prefix_token_count << "\n";
    return false;
  }
  if (!LayersOwnExpectedLivePages(
          request_context,
          descriptor.snapshot_layout.live_kv_pages_per_layer)) {
    std::cerr << "state_snapshot: restore rejected: target request allocated an unexpected KV-page layout\n";
    return false;
  }

  const std::size_t key_bytes = LiveKvTensorBytes(request_context);
  const std::size_t value_bytes = key_bytes;
  const std::size_t hidden_bytes = request_context.hidden()->bytes();
  const std::size_t residual_bytes = request_context.residual()->bytes();
  const std::size_t conv_bytes =
      HasMambaState(request_context) ? request_context.mamba_conv_state()->bytes() : 0;
  const std::size_t ssm_bytes =
      HasMambaState(request_context) ? request_context.mamba_state()->bytes() : 0;
  if (descriptor.kv_state.bytes != key_bytes + value_bytes ||
      descriptor.mamba_state.bytes != hidden_bytes + residual_bytes + conv_bytes + ssm_bytes) {
    std::cerr << "state_snapshot: restore rejected: descriptor byte count does not match target tensors\n";
    return false;
  }

  return arena.CopyToDevice(
             descriptor.kv_state,
             /*offset_bytes=*/0,
             request_context.key_cache()->data(),
             key_bytes) &&
         arena.CopyToDevice(
             descriptor.kv_state,
             /*offset_bytes=*/key_bytes,
             request_context.value_cache()->data(),
             value_bytes) &&
         arena.CopyToDevice(
             descriptor.mamba_state,
             /*offset_bytes=*/0,
             request_context.hidden()->data(),
             hidden_bytes) &&
         arena.CopyToDevice(
             descriptor.mamba_state,
             /*offset_bytes=*/hidden_bytes,
             request_context.residual()->data(),
             residual_bytes) &&
         (!HasMambaState(request_context) ||
          (arena.CopyToDevice(
               descriptor.mamba_state,
               /*offset_bytes=*/hidden_bytes + residual_bytes,
               request_context.mamba_conv_state()->data(),
               conv_bytes) &&
           arena.CopyToDevice(
               descriptor.mamba_state,
               /*offset_bytes=*/hidden_bytes + residual_bytes + conv_bytes,
               request_context.mamba_state()->data(),
               ssm_bytes)));
}

}  // namespace nemotron
