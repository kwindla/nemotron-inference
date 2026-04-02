#include "nemotron/request_context.h"
#include "nemotron/reusable_state.h"
#include "nemotron/state_snapshot.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::KvCacheDataType;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;
using nemotron::ReusableStateArena;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
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
      4,
      4,
      KvCacheDataType::kBf16,
  };
  config.attention_total_pages = 6;
  config.mamba_conv_state_bytes_fp32 = 12 * sizeof(float);
  config.mamba_state_bytes_fp32 = 20 * sizeof(float);
  return config;
}

std::uint16_t bf16_bits(__nv_bfloat16 value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool same_bf16(const std::vector<__nv_bfloat16>& lhs, const std::vector<__nv_bfloat16>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (bf16_bits(lhs[i]) != bf16_bits(rhs[i])) {
      return false;
    }
  }
  return true;
}

bool bf16_range_is_zero(const std::vector<__nv_bfloat16>& values, std::size_t start) {
  for (std::size_t i = start; i < values.size(); ++i) {
    if (bf16_bits(values[i]) != 0) {
      return false;
    }
  }
  return true;
}

bool test_snapshot_restore_round_trips_request_state() {
  auto source = RequestExecutionContext::Create(make_config());
  auto restored = RequestExecutionContext::Create(make_config());
  if (!source || !restored || !source->valid() || !restored->valid()) {
    std::cout << "state_snapshot_test: SKIP (no CUDA device available)\n";
    return true;
  }

  constexpr std::size_t kTokenCount = 5;
  if (!expect(source->SetSequenceLength(kTokenCount), "source request should allocate the prefix token window")) {
    return false;
  }

  const std::size_t total_kv_pages = source->key_cache()->shape().front();
  const std::size_t kv_elems_per_page = source->key_cache()->numel() / total_kv_pages;
  const std::size_t live_kv_elems = source->allocated_kv_pages() * kv_elems_per_page;
  const std::size_t full_kv_bytes = source->key_cache()->bytes() + source->value_cache()->bytes();
  const std::size_t live_kv_bytes = nemotron::RequiredKvSnapshotBytes(*source);

  std::vector<__nv_bfloat16> key_host(source->key_cache()->numel(), __float2bfloat16(0.0f));
  std::vector<__nv_bfloat16> value_host(source->value_cache()->numel(), __float2bfloat16(0.0f));
  for (std::size_t i = 0; i < live_kv_elems; ++i) {
    key_host[i] = __float2bfloat16(static_cast<float>((static_cast<int>(i % 29) - 14)) / 9.0f);
    value_host[i] = __float2bfloat16(static_cast<float>((static_cast<int>(i % 31) - 15)) / 7.0f);
  }
  std::vector<float> conv_host(source->mamba_conv_state()->numel(), 0.0f);
  std::vector<float> ssm_host(source->mamba_state()->numel(), 0.0f);
  for (std::size_t i = 0; i < conv_host.size(); ++i) {
    conv_host[i] = static_cast<float>((static_cast<int>(i % 17) - 8)) / 5.0f;
  }
  for (std::size_t i = 0; i < ssm_host.size(); ++i) {
    ssm_host[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) / 3.0f;
  }

  if (!expect(
          source->key_cache()->CopyFromHost(key_host.data(), key_host.size()),
          "source key cache should upload") ||
      !expect(
          source->value_cache()->CopyFromHost(value_host.data(), value_host.size()),
          "source value cache should upload") ||
      !expect(
          source->mamba_conv_state()->CopyFromHost(conv_host.data(), conv_host.size()),
          "source conv state should upload") ||
      !expect(
          source->mamba_state()->CopyFromHost(ssm_host.data(), ssm_host.size()),
          "source ssm state should upload")) {
    return false;
  }

  ReusableStateArena arena(
      nemotron::RequiredKvSnapshotBytes(*source) +
      nemotron::RequiredMambaSnapshotBytes(*source));
  const auto descriptor = nemotron::SnapshotRequestState(arena, *source, "state-snapshot-test");
  if (!expect(descriptor.has_value() && descriptor->valid(), "snapshot descriptor should allocate and copy")) {
    return false;
  }
  const auto kv_view = arena.Describe(descriptor->kv_state.id);
  const auto mamba_view = arena.Describe(descriptor->mamba_state.id);
  if (!expect(kv_view.has_value() && kv_view->has_device_storage, "KV snapshot should own device storage") ||
      !expect(
          mamba_view.has_value() && mamba_view->has_device_storage,
          "Mamba snapshot should own device storage")) {
    return false;
  }
  if (!expect(
          live_kv_bytes < full_kv_bytes,
          "KV snapshot should shrink below the full cache footprint for partial prefixes") ||
      !expect(
          kv_view->bytes == live_kv_bytes,
          "KV snapshot bytes should track only the allocated KV-page prefix")) {
    return false;
  }

  if (!expect(
          nemotron::RestoreRequestState(arena, *descriptor, kTokenCount, *restored),
          "restored request should accept the snapshotted state")) {
    return false;
  }
  if (!expect(
          restored->sequence_length() == kTokenCount && restored->decode_position() == kTokenCount,
          "restored request should resume at the cached token boundary")) {
    return false;
  }
  if (!expect(
          restored->allocated_kv_pages(0) == 2 && restored->allocated_kv_pages(1) == 2,
          "restored request should allocate the same number of KV pages for the cached prefix")) {
    return false;
  }

  std::vector<__nv_bfloat16> restored_key(key_host.size());
  std::vector<__nv_bfloat16> restored_value(value_host.size());
  std::vector<float> restored_conv(conv_host.size(), 0.0f);
  std::vector<float> restored_ssm(ssm_host.size(), 0.0f);
  if (!expect(
          restored->key_cache()->CopyToHost(restored_key.data(), restored_key.size()),
          "restored key cache should download") ||
      !expect(
          restored->value_cache()->CopyToHost(restored_value.data(), restored_value.size()),
          "restored value cache should download") ||
      !expect(
          restored->mamba_conv_state()->CopyToHost(restored_conv.data(), restored_conv.size()),
          "restored conv state should download") ||
      !expect(
          restored->mamba_state()->CopyToHost(restored_ssm.data(), restored_ssm.size()),
          "restored ssm state should download")) {
    return false;
  }

  arena.Release(*descriptor);

  return expect(
             same_bf16(
                 std::vector<__nv_bfloat16>(restored_key.begin(), restored_key.begin() + live_kv_elems),
                 std::vector<__nv_bfloat16>(key_host.begin(), key_host.begin() + live_kv_elems)),
             "restored live key-cache prefix should match the snapshot exactly") &&
         expect(
             same_bf16(
                 std::vector<__nv_bfloat16>(restored_value.begin(), restored_value.begin() + live_kv_elems),
                 std::vector<__nv_bfloat16>(value_host.begin(), value_host.begin() + live_kv_elems)),
             "restored live value-cache prefix should match the snapshot exactly") &&
         expect(
             bf16_range_is_zero(restored_key, live_kv_elems),
             "restored cold key-cache tail should remain zeroed") &&
         expect(
             bf16_range_is_zero(restored_value, live_kv_elems),
             "restored cold value-cache tail should remain zeroed") &&
         expect(restored_conv == conv_host, "restored conv state should match the snapshot exactly") &&
         expect(restored_ssm == ssm_host, "restored ssm state should match the snapshot exactly") &&
         expect(arena.current_bytes() == 0, "releasing the descriptor should free arena bytes");
}

}  // namespace

int main() {
  if (!test_snapshot_restore_round_trips_request_state()) {
    return 1;
  }
  std::cout << "state_snapshot_test: PASS\n";
  return 0;
}
