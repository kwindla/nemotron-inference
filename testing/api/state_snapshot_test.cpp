#include "nemotron/cublaslt_handle.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/request_context.h"
#include "nemotron/reusable_state.h"
#include "nemotron/state_snapshot.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorFp32;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::KernelTensorDescriptor;
using nemotron::MambaLayerBindings;
using nemotron::MambaLayerConfig;
using nemotron::MambaLayerSlice;
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

float pattern(
    std::size_t index,
    std::size_t multiplier,
    std::size_t modulus,
    float scale) {
  const std::size_t raw = ((index * multiplier) + 17) % modulus;
  return (static_cast<float>(raw) / static_cast<float>(modulus)) * scale -
         (scale * 0.5f);
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

KernelTensorDescriptor make_fp32_descriptor(
    const std::string& name,
    const std::vector<float>& values,
    std::vector<std::size_t> shape) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "state-snapshot-test";
  descriptor.logical_shape = std::move(shape);
  descriptor.packed_shape = descriptor.logical_shape;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data =
      reinterpret_cast<const std::uint8_t*>(values.data());
  descriptor.packed_nbytes = values.size() * sizeof(float);
  return descriptor;
}

GemmDescriptor make_dense_descriptor(
    const std::string& name,
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t cols) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "state-snapshot-test";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = rows;
  descriptor.input_cols = cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data =
      reinterpret_cast<const std::uint8_t*>(weights.data());
  descriptor.packed_nbytes = weights.size() * sizeof(float);
  return descriptor;
}

std::vector<float> make_input_sequence(
    std::size_t hidden_size,
    std::size_t token_count) {
  std::vector<float> values(token_count * hidden_size, 0.0f);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t i = 0; i < hidden_size; ++i) {
      values[token * hidden_size + i] =
          pattern(i + (token * hidden_size), 13, 97, 0.8f) +
          (0.015f * static_cast<float>(token)) -
          (0.002f * static_cast<float>(i % 5));
    }
  }
  return values;
}

bool test_mamba_snapshot_restore_continuation_matches_cold_execution() {
  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "state_snapshot_test: SKIP (no CUDA device available)\n";
    return true;
  }

  constexpr std::size_t kHiddenSize = 8;
  constexpr std::size_t kIntermediateSize = 4;
  constexpr std::size_t kNumHeads = 2;
  constexpr std::size_t kHeadDim = 2;
  constexpr std::size_t kStateSize = 3;
  constexpr std::size_t kNGroups = 2;
  constexpr std::size_t kConvKernelSize = 3;
  constexpr std::size_t kPrefixTokens = 3;
  constexpr std::size_t kTailTokens = 2;

  const std::size_t conv_dim =
      kIntermediateSize + (2 * kNGroups * kStateSize);
  const std::size_t conv_state_elems = conv_dim * kConvKernelSize;
  const std::size_t ssm_state_elems = kIntermediateSize * kStateSize;
  const std::size_t in_proj_rows = kIntermediateSize + conv_dim + kNumHeads;
  const std::size_t total_tokens = kPrefixTokens + kTailTokens;

  std::vector<float> input_norm_weight(kHiddenSize, 0.0f);
  std::vector<float> mixer_norm_weight(kIntermediateSize, 0.0f);
  std::vector<float> in_proj_weight(in_proj_rows * kHiddenSize, 0.0f);
  std::vector<float> conv1d_weight(conv_state_elems, 0.0f);
  std::vector<float> conv1d_bias(conv_dim, 0.0f);
  std::vector<float> A_log(kNumHeads, 0.0f);
  std::vector<float> D(kNumHeads, 0.0f);
  std::vector<float> dt_bias(kNumHeads, 0.0f);
  std::vector<float> out_proj_weight(kHiddenSize * kIntermediateSize, 0.0f);
  for (std::size_t i = 0; i < input_norm_weight.size(); ++i) {
    input_norm_weight[i] = 1.0f + pattern(i, 7, 53, 0.2f);
  }
  for (std::size_t i = 0; i < mixer_norm_weight.size(); ++i) {
    mixer_norm_weight[i] = 0.8f + pattern(i, 5, 47, 0.15f);
  }
  for (std::size_t i = 0; i < in_proj_weight.size(); ++i) {
    in_proj_weight[i] = pattern(i, 17, 131, 0.3f);
  }
  for (std::size_t i = 0; i < conv1d_weight.size(); ++i) {
    conv1d_weight[i] = pattern(i, 19, 149, 0.25f);
  }
  for (std::size_t i = 0; i < conv1d_bias.size(); ++i) {
    conv1d_bias[i] = pattern(i, 11, 59, 0.07f);
  }
  for (std::size_t i = 0; i < A_log.size(); ++i) {
    A_log[i] = -0.6f + pattern(i, 3, 29, 0.2f);
    D[i] = 0.4f + pattern(i, 5, 31, 0.15f);
    dt_bias[i] = 0.05f + pattern(i, 7, 37, 0.1f);
  }
  for (std::size_t i = 0; i < out_proj_weight.size(); ++i) {
    out_proj_weight[i] = pattern(i, 23, 137, 0.22f);
  }

  const auto input_norm_descriptor =
      make_fp32_descriptor("layer.norm.weight", input_norm_weight, {kHiddenSize});
  const auto mixer_norm_descriptor =
      make_fp32_descriptor("layer.mixer.norm.weight", mixer_norm_weight, {kIntermediateSize});
  const auto in_proj_descriptor =
      make_dense_descriptor("layer.mixer.in_proj.weight", in_proj_weight, in_proj_rows, kHiddenSize);
  const auto conv1d_weight_descriptor =
      make_fp32_descriptor("layer.mixer.conv1d.weight", conv1d_weight, {conv_dim, 1, kConvKernelSize});
  const auto conv1d_bias_descriptor =
      make_fp32_descriptor("layer.mixer.conv1d.bias", conv1d_bias, {conv_dim});
  const auto A_log_descriptor =
      make_fp32_descriptor("layer.mixer.A_log", A_log, {kNumHeads});
  const auto D_descriptor =
      make_fp32_descriptor("layer.mixer.D", D, {kNumHeads});
  const auto dt_bias_descriptor =
      make_fp32_descriptor("layer.mixer.dt_bias", dt_bias, {kNumHeads});
  const auto out_proj_descriptor =
      make_dense_descriptor("layer.mixer.out_proj.weight", out_proj_weight, kHiddenSize, kIntermediateSize);

  MambaLayerBindings bindings;
  bindings.input_norm_weight = &input_norm_descriptor;
  bindings.mixer_norm_weight = &mixer_norm_descriptor;
  bindings.in_proj_gemm_weight = &in_proj_descriptor;
  bindings.conv1d_weight = &conv1d_weight_descriptor;
  bindings.conv1d_bias = &conv1d_bias_descriptor;
  bindings.A_log = &A_log_descriptor;
  bindings.D = &D_descriptor;
  bindings.dt_bias = &dt_bias_descriptor;
  bindings.out_proj_gemm_weight = &out_proj_descriptor;

  MambaLayerConfig layer_config;
  layer_config.layer_index = 0;
  layer_config.hidden_size = kHiddenSize;
  layer_config.intermediate_size = kIntermediateSize;
  layer_config.num_heads = kNumHeads;
  layer_config.head_dim = kHeadDim;
  layer_config.state_size = kStateSize;
  layer_config.n_groups = kNGroups;
  layer_config.conv_kernel_size = kConvKernelSize;
  layer_config.input_rms_epsilon = 1.0e-5f;
  layer_config.mixer_rms_epsilon = 1.0e-5f;
  layer_config.time_step_min = 1.0e-3f;

  auto layer = MambaLayerSlice::Create(layer_config, bindings);
  if (!expect(layer != nullptr && layer->valid(), "synthetic mamba layer should build")) {
    return false;
  }

  RequestExecutionConfig config;
  config.hidden_size = kHiddenSize;
  config.max_tokens = total_tokens;
  config.scratch_tokens = total_tokens;
  config.attention_kv_cache = AttentionKvCacheConfig{
      1,
      1,
      4,
      4,
      KvCacheDataType::kBf16,
  };
  config.attention_total_pages = 2;
  config.mamba_conv_state_bytes_fp32 = conv_state_elems * sizeof(float);
  config.mamba_state_bytes_fp32 = ssm_state_elems * sizeof(float);

  auto source = RequestExecutionContext::Create(config);
  auto restored = RequestExecutionContext::Create(config);
  auto prefix_input = DeviceTensorFp32::Create({kPrefixTokens, kHiddenSize});
  auto tail_input = DeviceTensorFp32::Create({kTailTokens, kHiddenSize});
  auto prefix_output = DeviceTensorFp32::Create({kPrefixTokens, kHiddenSize});
  auto source_tail_output = DeviceTensorFp32::Create({kTailTokens, kHiddenSize});
  auto restored_tail_output = DeviceTensorFp32::Create({kTailTokens, kHiddenSize});
  if (!expect(source != nullptr && source->valid(), "source request should create") ||
      !expect(restored != nullptr && restored->valid(), "restored request should create") ||
      !expect(prefix_input != nullptr && prefix_input->valid(), "prefix input tensor should create") ||
      !expect(tail_input != nullptr && tail_input->valid(), "tail input tensor should create") ||
      !expect(prefix_output != nullptr && prefix_output->valid(), "prefix output tensor should create") ||
      !expect(source_tail_output != nullptr && source_tail_output->valid(), "source tail output should create") ||
      !expect(restored_tail_output != nullptr && restored_tail_output->valid(), "restored tail output should create")) {
    return false;
  }

  const std::vector<float> full_inputs = make_input_sequence(kHiddenSize, total_tokens);
  const std::vector<float> prefix_inputs(
      full_inputs.begin(),
      full_inputs.begin() + static_cast<std::ptrdiff_t>(kPrefixTokens * kHiddenSize));
  const std::vector<float> tail_inputs(
      full_inputs.begin() + static_cast<std::ptrdiff_t>(kPrefixTokens * kHiddenSize),
      full_inputs.end());
  std::vector<float> initial_conv_state(conv_state_elems, 0.0f);
  std::vector<float> initial_ssm_state(ssm_state_elems, 0.0f);
  for (std::size_t i = 0; i < initial_conv_state.size(); ++i) {
    initial_conv_state[i] = pattern(i, 29, 167, 0.2f);
  }
  for (std::size_t i = 0; i < initial_ssm_state.size(); ++i) {
    initial_ssm_state[i] = pattern(i, 31, 173, 0.25f);
  }

  if (!expect(source->SetSequenceLength(kPrefixTokens), "source request should allocate cached-prefix pages") ||
      !expect(prefix_input->CopyFromHost(prefix_inputs.data(), prefix_inputs.size()), "prefix inputs should upload") ||
      !expect(tail_input->CopyFromHost(tail_inputs.data(), tail_inputs.size()), "tail inputs should upload") ||
      !expect(
          source->mamba_conv_state()->CopyFromHost(
              initial_conv_state.data(),
              initial_conv_state.size()),
          "initial conv state should upload") ||
      !expect(
          source->mamba_state()->CopyFromHost(
              initial_ssm_state.data(),
              initial_ssm_state.size()),
          "initial ssm state should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          layer->Run(*cublas, &heuristic_cache, *source, *prefix_input, prefix_output.get()),
          "prefix execution should succeed")) {
    return false;
  }

  std::vector<float> prefix_conv_state(conv_state_elems, 0.0f);
  std::vector<float> prefix_ssm_state(ssm_state_elems, 0.0f);
  if (!expect(
          source->mamba_conv_state()->CopyToHost(
              prefix_conv_state.data(),
              prefix_conv_state.size()),
          "prefix conv state should download") ||
      !expect(
          source->mamba_state()->CopyToHost(
              prefix_ssm_state.data(),
              prefix_ssm_state.size()),
          "prefix ssm state should download")) {
    return false;
  }

  ReusableStateArena arena(
      nemotron::RequiredKvSnapshotBytes(*source) +
      nemotron::RequiredMambaSnapshotBytes(*source));
  const auto descriptor =
      nemotron::SnapshotRequestState(arena, *source, "mamba-prefix-boundary");
  if (!expect(
          descriptor.has_value() && descriptor->valid(),
          "synthetic prefix snapshot should allocate")) {
    return false;
  }

  if (!expect(
          layer->Run(
              *cublas,
              &heuristic_cache,
              *source,
              *tail_input,
              source_tail_output.get()),
          "cold continuation should execute")) {
    arena.Release(*descriptor);
    return false;
  }

  if (!expect(
          nemotron::RestoreRequestState(arena, *descriptor, kPrefixTokens, *restored),
          "restored request should accept the cached prefix")) {
    arena.Release(*descriptor);
    return false;
  }
  if (!expect(
          restored->sequence_length() == kPrefixTokens &&
              restored->decode_position() == kPrefixTokens,
          "restored request should resume at the cached prefix boundary")) {
    arena.Release(*descriptor);
    return false;
  }

  std::vector<float> restored_prefix_conv_state(conv_state_elems, 0.0f);
  std::vector<float> restored_prefix_ssm_state(ssm_state_elems, 0.0f);
  if (!expect(
          restored->mamba_conv_state()->CopyToHost(
              restored_prefix_conv_state.data(),
              restored_prefix_conv_state.size()),
          "restored prefix conv state should download") ||
      !expect(
          restored->mamba_state()->CopyToHost(
              restored_prefix_ssm_state.data(),
              restored_prefix_ssm_state.size()),
          "restored prefix ssm state should download")) {
    arena.Release(*descriptor);
    return false;
  }

  if (!expect(
          max_abs_diff(prefix_conv_state, restored_prefix_conv_state) <= 1.0e-6f,
          "restored prefix conv state should match the cached boundary") ||
      !expect(
          max_abs_diff(prefix_ssm_state, restored_prefix_ssm_state) <= 1.0e-6f,
          "restored prefix ssm state should match the cached boundary")) {
    arena.Release(*descriptor);
    return false;
  }

  if (!expect(
          layer->Run(
              *cublas,
              &heuristic_cache,
              *restored,
              *tail_input,
              restored_tail_output.get()),
          "restored continuation should execute")) {
    arena.Release(*descriptor);
    return false;
  }

  std::vector<float> source_tail_host(tail_inputs.size(), 0.0f);
  std::vector<float> restored_tail_host(tail_inputs.size(), 0.0f);
  std::vector<float> source_final_conv_state(conv_state_elems, 0.0f);
  std::vector<float> source_final_ssm_state(ssm_state_elems, 0.0f);
  std::vector<float> restored_final_conv_state(conv_state_elems, 0.0f);
  std::vector<float> restored_final_ssm_state(ssm_state_elems, 0.0f);
  if (!expect(
          source_tail_output->CopyToHost(
              source_tail_host.data(),
              source_tail_host.size()),
          "cold continuation output should download") ||
      !expect(
          restored_tail_output->CopyToHost(
              restored_tail_host.data(),
              restored_tail_host.size()),
          "restored continuation output should download") ||
      !expect(
          source->mamba_conv_state()->CopyToHost(
              source_final_conv_state.data(),
              source_final_conv_state.size()),
          "cold continuation conv state should download") ||
      !expect(
          source->mamba_state()->CopyToHost(
              source_final_ssm_state.data(),
              source_final_ssm_state.size()),
          "cold continuation ssm state should download") ||
      !expect(
          restored->mamba_conv_state()->CopyToHost(
              restored_final_conv_state.data(),
              restored_final_conv_state.size()),
          "restored continuation conv state should download") ||
      !expect(
          restored->mamba_state()->CopyToHost(
              restored_final_ssm_state.data(),
              restored_final_ssm_state.size()),
          "restored continuation ssm state should download")) {
    arena.Release(*descriptor);
    return false;
  }

  arena.Release(*descriptor);
  return expect(
             max_abs_diff(source_tail_host, restored_tail_host) <= 1.0e-6f,
             "restored continuation output should match cold execution") &&
         expect(
             max_abs_diff(source_final_conv_state, restored_final_conv_state) <= 1.0e-6f,
             "restored continuation conv state should match cold execution") &&
         expect(
             max_abs_diff(source_final_ssm_state, restored_final_ssm_state) <= 1.0e-6f,
             "restored continuation ssm state should match cold execution") &&
         expect(arena.current_bytes() == 0, "releasing the synthetic descriptor should free arena bytes");
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
  if (!expect(
          (*source->kv_pages(0))[0].page_id == (*restored->kv_pages(0))[0].page_id &&
              (*source->kv_pages(0))[1].page_id == (*restored->kv_pages(0))[1].page_id &&
              (*source->kv_pages(1))[0].page_id == (*restored->kv_pages(1))[0].page_id &&
              (*source->kv_pages(1))[1].page_id == (*restored->kv_pages(1))[1].page_id,
          "restore should reproduce the exact allocated KV page ids required by the snapshot ABI")) {
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
  if (!test_snapshot_restore_round_trips_request_state() ||
      !test_mamba_snapshot_restore_continuation_matches_cold_execution()) {
    return 1;
  }
  std::cout << "state_snapshot_test: PASS\n";
  return 0;
}
