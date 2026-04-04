#include "nemotron/state_snapshot.h"

#include <cstddef>

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

  const std::size_t kv_bytes = RequiredKvSnapshotBytes(request_context);
  const std::size_t mamba_bytes = RequiredMambaSnapshotBytes(request_context);
  if (kv_bytes == 0 || mamba_bytes == 0) {
    return std::nullopt;
  }

  const ReusableStateDescriptor descriptor = arena.AllocateDescriptor(kv_bytes, mamba_bytes, label);
  if (!descriptor.valid()) {
    return std::nullopt;
  }

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
    std::size_t token_count,
    RequestExecutionContext& request_context) {
  if (!descriptor.valid() ||
      !request_context.valid() ||
      !HasActivationState(request_context) ||
      !HasAttentionState(request_context) ||
      token_count == 0) {
    return false;
  }

  if (!request_context.ResetForNewRequest() ||
      !request_context.SetSequenceLength(token_count)) {
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
