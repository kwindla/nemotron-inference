#include <algorithm>
#include "nemotron/expert_layer.h"
#include "nemotron/expert_staging_counters.h"
#include "nemotron/nvfp4_packing.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorFp32;
using nemotron::ExpertLayerBindings;
using nemotron::ExpertLayerConfig;
using nemotron::ExpertLayerSlice;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::GetExpertStagingCounters;
using nemotron::HostNvfp4Matrix;
using nemotron::KernelTensorDescriptor;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::ResetExpertStagingCounters;

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
    const char* current = std::getenv(name_);
    if (current != nullptr) {
      had_original_ = true;
      original_value_ = current;
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(name_, original_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_original_ = false;
  std::string original_value_;
};

struct OwnedNvfp4Descriptor {
  HostNvfp4Matrix packed;
  GemmDescriptor descriptor;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    diff = std::max(diff, std::fabs(lhs[i] - rhs[i]));
  }
  return diff;
}

KernelTensorDescriptor make_fp32_descriptor(
    const std::string& name,
    const std::vector<float>& values,
    std::vector<std::size_t> shape) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "expert_layer_fastpath_test";
  descriptor.logical_shape = std::move(shape);
  descriptor.packed_shape = descriptor.logical_shape;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(values.data());
  descriptor.packed_nbytes = values.size() * sizeof(float);
  return descriptor;
}

GemmDescriptor make_dense_descriptor(
    const std::string& name,
    const std::vector<float>& values,
    std::size_t rows,
    std::size_t cols) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "expert_layer_fastpath_test";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = rows;
  descriptor.input_cols = cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(values.data());
  descriptor.packed_nbytes = values.size() * sizeof(float);
  return descriptor;
}

GemmDescriptor make_nvfp4_descriptor(
    const std::string& name,
    const HostNvfp4Matrix& packed) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "expert_layer_fastpath_test";
  descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = packed.rows;
  descriptor.input_cols = packed.cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed.packed_data();
  descriptor.packed_nbytes = packed.packed_nbytes();
  descriptor.block_scales_data = packed.block_scales_data();
  descriptor.block_scales_nbytes = packed.block_scales_nbytes();
  descriptor.tensor_scale_data = packed.tensor_scale_data();
  descriptor.tensor_scale_nbytes = packed.tensor_scale_nbytes();
  return descriptor;
}

std::optional<OwnedNvfp4Descriptor> make_owned_nvfp4_descriptor(
    const std::string& name,
    const std::vector<float>& values,
    std::size_t rows,
    std::size_t cols) {
  const auto packed = PackRowMajorFp32ToNvfp4(values.data(), rows, cols);
  if (!packed.has_value()) {
    return std::nullopt;
  }
  OwnedNvfp4Descriptor owned;
  owned.packed = *packed;
  owned.descriptor = make_nvfp4_descriptor(name, owned.packed);
  return owned;
}

std::vector<float> make_patterned_values(
    std::size_t rows,
    std::size_t cols,
    int seed,
    float scale) {
  std::vector<float> values(rows * cols, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const int raw = static_cast<int>(((row + 1) * (seed + 7)) + ((col + 3) * (seed + 11)));
      values[row * cols + col] =
          (static_cast<float>((raw % 31) - 15) * scale) +
          (0.0025f * static_cast<float>((row + col + static_cast<std::size_t>(seed)) % 5));
    }
  }
  return values;
}

bool all_finite(const std::vector<float>& values) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

bool test_unified_fused_prefill_is_default_and_avoids_host_routing_adapter() {
  if (!has_cuda_device()) {
    std::cout << "expert_layer_fastpath_test: SKIP (no CUDA device)\n";
    return true;
  }

  ScopedEnvVar scoped_unified("NEMOTRON_FORWARD_UNIFIED_FUSED");
  ScopedEnvVar scoped_prefill("NEMOTRON_FORWARD_FUSED_MOE_PREFILL");
  ScopedEnvVar scoped_dense_fastpath("NEMOTRON_FORWARD_LINEAR_DISABLE_DENSE_FASTPATH");
  unsetenv("NEMOTRON_FORWARD_UNIFIED_FUSED");
  unsetenv("NEMOTRON_FORWARD_FUSED_MOE_PREFILL");
  setenv("NEMOTRON_FORWARD_LINEAR_DISABLE_DENSE_FASTPATH", "1", 1);

  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "expert_layer_fastpath_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kHiddenSize = 64;
  constexpr std::size_t kIntermediateSize = 64;
  constexpr std::size_t kRoutedExperts = 8;
  constexpr std::size_t kTopK = 2;
  constexpr std::size_t kTokenCount = 4;

  const std::string layer_prefix = "backbone.layers.0";
  const std::string mixer_prefix = layer_prefix + ".mixer";

  std::vector<float> input_norm_weight(kHiddenSize, 1.0f);
  std::vector<float> gate_bias = {0.02f, -0.04f, 0.03f, 0.00f, -0.01f, 0.05f, -0.02f, 0.01f};
  std::vector<float> gate_weight = make_patterned_values(kRoutedExperts, kHiddenSize, 3, 0.0125f);
  std::vector<float> shared_up_values = make_patterned_values(kIntermediateSize, kHiddenSize, 17, 0.02f);
  std::vector<float> shared_down_values = make_patterned_values(kHiddenSize, kIntermediateSize, 29, 0.02f);

  const auto input_norm_descriptor =
      make_fp32_descriptor(layer_prefix + ".norm.weight", input_norm_weight, {kHiddenSize});
  const auto gate_bias_descriptor =
      make_fp32_descriptor(mixer_prefix + ".gate.e_score_correction_bias", gate_bias, {kRoutedExperts});
  const auto gate_weight_descriptor =
      make_dense_descriptor(mixer_prefix + ".gate.weight", gate_weight, kRoutedExperts, kHiddenSize);
  const auto shared_up_descriptor =
      make_owned_nvfp4_descriptor(
          mixer_prefix + ".shared_experts.up_proj.weight",
          shared_up_values,
          kIntermediateSize,
          kHiddenSize);
  const auto shared_down_descriptor =
      make_owned_nvfp4_descriptor(
          mixer_prefix + ".shared_experts.down_proj.weight",
          shared_down_values,
          kHiddenSize,
          kIntermediateSize);
  if (!expect(shared_up_descriptor.has_value(), "shared up NVFP4 descriptor should build") ||
      !expect(shared_down_descriptor.has_value(), "shared down NVFP4 descriptor should build")) {
    return false;
  }

  std::vector<OwnedNvfp4Descriptor> routed_up_descriptors(kRoutedExperts);
  std::vector<OwnedNvfp4Descriptor> routed_down_descriptors(kRoutedExperts);
  ExpertLayerBindings bindings;
  bindings.input_norm_weight = &input_norm_descriptor;
  bindings.gate_weight = &gate_weight_descriptor;
  bindings.gate_score_correction_bias = &gate_bias_descriptor;
  bindings.shared_up_gemm_weight = &shared_up_descriptor->descriptor;
  bindings.shared_down_gemm_weight = &shared_down_descriptor->descriptor;
  bindings.routed_experts.resize(kRoutedExperts);
  for (std::size_t expert_index = 0; expert_index < kRoutedExperts; ++expert_index) {
    const std::string expert_prefix =
        mixer_prefix + ".experts." + std::to_string(expert_index);
    const auto up_values = make_patterned_values(
        kIntermediateSize,
        kHiddenSize,
        41 + static_cast<int>(expert_index * 2),
        0.0175f);
    const auto down_values = make_patterned_values(
        kHiddenSize,
        kIntermediateSize,
        73 + static_cast<int>(expert_index * 3),
        0.0175f);
    const auto up_descriptor =
        make_owned_nvfp4_descriptor(
            expert_prefix + ".up_proj.weight",
            up_values,
            kIntermediateSize,
            kHiddenSize);
    const auto down_descriptor =
        make_owned_nvfp4_descriptor(
            expert_prefix + ".down_proj.weight",
            down_values,
            kHiddenSize,
            kIntermediateSize);
    if (!expect(up_descriptor.has_value(), "routed up NVFP4 descriptor should build") ||
        !expect(down_descriptor.has_value(), "routed down NVFP4 descriptor should build")) {
      return false;
    }
    routed_up_descriptors[expert_index] = std::move(*up_descriptor);
    routed_down_descriptors[expert_index] = std::move(*down_descriptor);
    bindings.routed_experts[expert_index].up_proj =
        &routed_up_descriptors[expert_index].descriptor;
    bindings.routed_experts[expert_index].down_proj =
        &routed_down_descriptors[expert_index].descriptor;
  }

  ExpertLayerConfig config;
  config.layer_index = 0;
  config.hidden_size = kHiddenSize;
  config.moe_latent_size = 0;
  config.routed_expert_intermediate_size = kIntermediateSize;
  config.shared_expert_intermediate_size = kIntermediateSize;
  config.n_routed_experts = kRoutedExperts;
  config.top_k = kTopK;
  config.max_token_count = kTokenCount;
  config.n_group = 1;
  config.topk_group = 1;
  config.rms_epsilon = 1.0e-5f;
  config.routed_scaling_factor = 5.0f;
  config.norm_topk_prob = true;

  auto slice = ExpertLayerSlice::Create(config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "resident expert layer slice should create")) {
    return false;
  }

  std::vector<float> input_values(kTokenCount * kHiddenSize, 0.0f);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t dim = 0; dim < kHiddenSize; ++dim) {
      input_values[token * kHiddenSize + dim] =
          (static_cast<float>(((token + 1) * 13 + (dim * 7)) % 37) - 18.0f) * 0.03125f;
    }
  }

  auto batch_input = DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto batch_output = DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  if (!expect(batch_input != nullptr && batch_output != nullptr,
              "batched tensors should allocate") ||
      !expect(batch_input->CopyFromHost(input_values.data(), input_values.size()),
              "batched input should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  ResetExpertStagingCounters();
  if (!expect(
          slice->Run(*cublas, &heuristic_cache, *batch_input, batch_output.get(), nullptr),
          "unified fused batch run should succeed") ||
      !expect(cudaDeviceSynchronize() == cudaSuccess, "batch run should synchronize")) {
    return false;
  }

  const auto& counters = GetExpertStagingCounters();
  if (!expect(
          counters.host_routing_adapter_calls.load(std::memory_order_relaxed) == 0,
          "unified fused batch run should not invoke the host routing adapter") ||
      !expect(
          counters.host_routing_tensor_copies.load(std::memory_order_relaxed) == 0,
          "unified fused batch run should not copy canonical routing tensors to host")) {
    return false;
  }
  return true;
}

bool test_unified_fused_prefill_matches_reference_batch_output() {
  if (!has_cuda_device()) {
    std::cout << "expert_layer_fastpath_test: SKIP (no CUDA device)\n";
    return true;
  }

  ScopedEnvVar scoped_unified("NEMOTRON_FORWARD_UNIFIED_FUSED");
  ScopedEnvVar scoped_prefill("NEMOTRON_FORWARD_FUSED_MOE_PREFILL");
  ScopedEnvVar scoped_full_residency("NEMOTRON_EXPERT_FULL_RESIDENCY");
  ScopedEnvVar scoped_monolithic("NEMOTRON_EXPERT_MONOLITHIC");
  ScopedEnvVar scoped_dense_fastpath("NEMOTRON_FORWARD_LINEAR_DISABLE_DENSE_FASTPATH");
  setenv("NEMOTRON_EXPERT_FULL_RESIDENCY", "1", 1);
  setenv("NEMOTRON_EXPERT_MONOLITHIC", "1", 1);
  setenv("NEMOTRON_FORWARD_LINEAR_DISABLE_DENSE_FASTPATH", "1", 1);

  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "expert_layer_fastpath_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kHiddenSize = 64;
  constexpr std::size_t kIntermediateSize = 64;
  constexpr std::size_t kRoutedExperts = 8;
  constexpr std::size_t kTopK = 2;
  constexpr std::size_t kTokenCount = 4;
  constexpr float kMaxAllowedDiff = 0.05f;

  const std::string layer_prefix = "backbone.layers.2";
  const std::string mixer_prefix = layer_prefix + ".mixer";

  std::vector<float> input_norm_weight(kHiddenSize, 1.0f);
  std::vector<float> gate_bias = {0.02f, -0.04f, 0.03f, 0.00f, -0.01f, 0.05f, -0.02f, 0.01f};
  std::vector<float> gate_weight = make_patterned_values(kRoutedExperts, kHiddenSize, 7, 0.00625f);
  std::vector<float> shared_up_values = make_patterned_values(kIntermediateSize, kHiddenSize, 23, 0.01f);
  std::vector<float> shared_down_values = make_patterned_values(kHiddenSize, kIntermediateSize, 37, 0.01f);

  const auto input_norm_descriptor =
      make_fp32_descriptor(layer_prefix + ".norm.weight", input_norm_weight, {kHiddenSize});
  const auto gate_bias_descriptor =
      make_fp32_descriptor(mixer_prefix + ".gate.e_score_correction_bias", gate_bias, {kRoutedExperts});
  const auto gate_weight_descriptor =
      make_dense_descriptor(mixer_prefix + ".gate.weight", gate_weight, kRoutedExperts, kHiddenSize);
  const auto shared_up_descriptor =
      make_owned_nvfp4_descriptor(
          mixer_prefix + ".shared_experts.up_proj.weight",
          shared_up_values,
          kIntermediateSize,
          kHiddenSize);
  const auto shared_down_descriptor =
      make_owned_nvfp4_descriptor(
          mixer_prefix + ".shared_experts.down_proj.weight",
          shared_down_values,
          kHiddenSize,
          kIntermediateSize);
  if (!expect(shared_up_descriptor.has_value(), "shared up NVFP4 descriptor should build") ||
      !expect(shared_down_descriptor.has_value(), "shared down NVFP4 descriptor should build")) {
    return false;
  }

  std::vector<OwnedNvfp4Descriptor> routed_up_descriptors(kRoutedExperts);
  std::vector<OwnedNvfp4Descriptor> routed_down_descriptors(kRoutedExperts);
  ExpertLayerBindings bindings;
  bindings.input_norm_weight = &input_norm_descriptor;
  bindings.gate_weight = &gate_weight_descriptor;
  bindings.gate_score_correction_bias = &gate_bias_descriptor;
  bindings.shared_up_gemm_weight = &shared_up_descriptor->descriptor;
  bindings.shared_down_gemm_weight = &shared_down_descriptor->descriptor;
  bindings.routed_experts.resize(kRoutedExperts);
  for (std::size_t expert_index = 0; expert_index < kRoutedExperts; ++expert_index) {
    const std::string expert_prefix =
        mixer_prefix + ".experts." + std::to_string(expert_index);
    const auto up_values = make_patterned_values(
        kIntermediateSize,
        kHiddenSize,
        61 + static_cast<int>(expert_index * 2),
        0.008f);
    const auto down_values = make_patterned_values(
        kHiddenSize,
        kIntermediateSize,
        97 + static_cast<int>(expert_index * 3),
        0.008f);
    const auto up_descriptor =
        make_owned_nvfp4_descriptor(
            expert_prefix + ".up_proj.weight",
            up_values,
            kIntermediateSize,
            kHiddenSize);
    const auto down_descriptor =
        make_owned_nvfp4_descriptor(
            expert_prefix + ".down_proj.weight",
            down_values,
            kHiddenSize,
            kIntermediateSize);
    if (!expect(up_descriptor.has_value(), "routed up NVFP4 descriptor should build") ||
        !expect(down_descriptor.has_value(), "routed down NVFP4 descriptor should build")) {
      return false;
    }
    routed_up_descriptors[expert_index] = std::move(*up_descriptor);
    routed_down_descriptors[expert_index] = std::move(*down_descriptor);
    bindings.routed_experts[expert_index].up_proj =
        &routed_up_descriptors[expert_index].descriptor;
    bindings.routed_experts[expert_index].down_proj =
        &routed_down_descriptors[expert_index].descriptor;
  }

  ExpertLayerConfig config;
  config.layer_index = 2;
  config.hidden_size = kHiddenSize;
  config.moe_latent_size = 0;
  config.routed_expert_intermediate_size = kIntermediateSize;
  config.shared_expert_intermediate_size = kIntermediateSize;
  config.n_routed_experts = kRoutedExperts;
  config.top_k = kTopK;
  config.max_token_count = kTokenCount;
  config.n_group = 1;
  config.topk_group = 1;
  config.rms_epsilon = 1.0e-5f;
  config.routed_scaling_factor = 1.0f;
  config.norm_topk_prob = true;

  setenv("NEMOTRON_FORWARD_UNIFIED_FUSED", "0", 1);
  setenv("NEMOTRON_FORWARD_FUSED_MOE_PREFILL", "0", 1);
  auto reference_slice = ExpertLayerSlice::Create(config, bindings);
  unsetenv("NEMOTRON_FORWARD_UNIFIED_FUSED");
  unsetenv("NEMOTRON_FORWARD_FUSED_MOE_PREFILL");
  auto unified_slice = ExpertLayerSlice::Create(config, bindings);
  if (!expect(reference_slice != nullptr && reference_slice->valid(),
              "reference expert layer slice should create") ||
      !expect(unified_slice != nullptr && unified_slice->valid(),
              "unified expert layer slice should create")) {
    return false;
  }

  std::vector<float> input_values(kTokenCount * kHiddenSize, 0.0f);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t dim = 0; dim < kHiddenSize; ++dim) {
      input_values[token * kHiddenSize + dim] =
          (static_cast<float>(((token + 1) * 17 + (dim * 9)) % 43) - 21.0f) * 0.015625f;
    }
  }

  auto batch_input = DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto reference_output = DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto unified_output = DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  if (!expect(batch_input != nullptr && reference_output != nullptr && unified_output != nullptr,
              "batched tensors should allocate") ||
      !expect(batch_input->CopyFromHost(input_values.data(), input_values.size()),
              "batched input should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          reference_slice->Run(*cublas, &heuristic_cache, *batch_input, reference_output.get(), nullptr),
          "reference batch run should succeed") ||
      !expect(
          unified_slice->Run(*cublas, &heuristic_cache, *batch_input, unified_output.get(), nullptr),
          "unified fused batch run should succeed") ||
      !expect(cudaDeviceSynchronize() == cudaSuccess, "batch outputs should synchronize")) {
    return false;
  }

  std::vector<float> reference_host(reference_output->numel(), 0.0f);
  std::vector<float> unified_host(unified_output->numel(), 0.0f);
  if (!expect(reference_output->CopyToHost(reference_host.data(), reference_host.size()),
              "reference batch output should copy to host") ||
      !expect(unified_output->CopyToHost(unified_host.data(), unified_host.size()),
              "unified batch output should copy to host")) {
    return false;
  }
  if (!expect(all_finite(reference_host), "reference batch output should be finite") ||
      !expect(all_finite(unified_host), "unified batch output should be finite")) {
    return false;
  }

  const float diff = max_abs_diff(reference_host, unified_host);
  if (!expect(diff <= kMaxAllowedDiff, "unified fused batch output should match the reference path")) {
    std::cerr << "reference_vs_unified_max_abs_diff=" << diff << "\n";
    return false;
  }
  return true;
}

bool test_nonresident_routed_weights_reject_fastpath_and_use_fallback() {
  if (!has_cuda_device()) {
    std::cout << "expert_layer_fastpath_test: SKIP (no CUDA device)\n";
    return true;
  }

  ScopedEnvVar scoped_unified("NEMOTRON_FORWARD_UNIFIED_FUSED");
  ScopedEnvVar scoped_prefill("NEMOTRON_FORWARD_FUSED_MOE_PREFILL");
  ScopedEnvVar scoped_full_residency("NEMOTRON_EXPERT_FULL_RESIDENCY");
  ScopedEnvVar scoped_monolithic("NEMOTRON_EXPERT_MONOLITHIC");
  ScopedEnvVar scoped_dense_fastpath("NEMOTRON_FORWARD_LINEAR_DISABLE_DENSE_FASTPATH");
  unsetenv("NEMOTRON_FORWARD_UNIFIED_FUSED");
  unsetenv("NEMOTRON_FORWARD_FUSED_MOE_PREFILL");
  setenv("NEMOTRON_EXPERT_FULL_RESIDENCY", "0", 1);
  setenv("NEMOTRON_EXPERT_MONOLITHIC", "0", 1);
  setenv("NEMOTRON_FORWARD_LINEAR_DISABLE_DENSE_FASTPATH", "1", 1);

  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "expert_layer_fastpath_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kHiddenSize = 64;
  constexpr std::size_t kIntermediateSize = 64;
  constexpr std::size_t kRoutedExperts = 8;
  constexpr std::size_t kTopK = 2;
  constexpr std::size_t kTokenCount = 4;

  const std::string layer_prefix = "backbone.layers.1";
  const std::string mixer_prefix = layer_prefix + ".mixer";

  std::vector<float> input_norm_weight(kHiddenSize, 1.0f);
  std::vector<float> gate_bias = {0.02f, -0.04f, 0.03f, 0.00f, -0.01f, 0.05f, -0.02f, 0.01f};
  std::vector<float> gate_weight = make_patterned_values(kRoutedExperts, kHiddenSize, 5, 0.0125f);
  std::vector<float> shared_up_values = make_patterned_values(kIntermediateSize, kHiddenSize, 19, 0.02f);
  std::vector<float> shared_down_values = make_patterned_values(kHiddenSize, kIntermediateSize, 31, 0.02f);

  const auto input_norm_descriptor =
      make_fp32_descriptor(layer_prefix + ".norm.weight", input_norm_weight, {kHiddenSize});
  const auto gate_bias_descriptor =
      make_fp32_descriptor(mixer_prefix + ".gate.e_score_correction_bias", gate_bias, {kRoutedExperts});
  const auto gate_weight_descriptor =
      make_dense_descriptor(mixer_prefix + ".gate.weight", gate_weight, kRoutedExperts, kHiddenSize);
  const auto shared_up_descriptor =
      make_owned_nvfp4_descriptor(
          mixer_prefix + ".shared_experts.up_proj.weight",
          shared_up_values,
          kIntermediateSize,
          kHiddenSize);
  const auto shared_down_descriptor =
      make_owned_nvfp4_descriptor(
          mixer_prefix + ".shared_experts.down_proj.weight",
          shared_down_values,
          kHiddenSize,
          kIntermediateSize);
  if (!expect(shared_up_descriptor.has_value(), "shared up NVFP4 descriptor should build") ||
      !expect(shared_down_descriptor.has_value(), "shared down NVFP4 descriptor should build")) {
    return false;
  }

  std::vector<OwnedNvfp4Descriptor> routed_up_descriptors(kRoutedExperts);
  std::vector<OwnedNvfp4Descriptor> routed_down_descriptors(kRoutedExperts);
  ExpertLayerBindings bindings;
  bindings.input_norm_weight = &input_norm_descriptor;
  bindings.gate_weight = &gate_weight_descriptor;
  bindings.gate_score_correction_bias = &gate_bias_descriptor;
  bindings.shared_up_gemm_weight = &shared_up_descriptor->descriptor;
  bindings.shared_down_gemm_weight = &shared_down_descriptor->descriptor;
  bindings.routed_experts.resize(kRoutedExperts);
  for (std::size_t expert_index = 0; expert_index < kRoutedExperts; ++expert_index) {
    const std::string expert_prefix =
        mixer_prefix + ".experts." + std::to_string(expert_index);
    const auto up_values = make_patterned_values(
        kIntermediateSize,
        kHiddenSize,
        53 + static_cast<int>(expert_index * 2),
        0.0175f);
    const auto down_values = make_patterned_values(
        kHiddenSize,
        kIntermediateSize,
        89 + static_cast<int>(expert_index * 3),
        0.0175f);
    const auto up_descriptor =
        make_owned_nvfp4_descriptor(
            expert_prefix + ".up_proj.weight",
            up_values,
            kIntermediateSize,
            kHiddenSize);
    const auto down_descriptor =
        make_owned_nvfp4_descriptor(
            expert_prefix + ".down_proj.weight",
            down_values,
            kHiddenSize,
            kIntermediateSize);
    if (!expect(up_descriptor.has_value(), "routed up NVFP4 descriptor should build") ||
        !expect(down_descriptor.has_value(), "routed down NVFP4 descriptor should build")) {
      return false;
    }
    routed_up_descriptors[expert_index] = std::move(*up_descriptor);
    routed_down_descriptors[expert_index] = std::move(*down_descriptor);
    bindings.routed_experts[expert_index].up_proj =
        &routed_up_descriptors[expert_index].descriptor;
    bindings.routed_experts[expert_index].down_proj =
        &routed_down_descriptors[expert_index].descriptor;
  }

  ExpertLayerConfig config;
  config.layer_index = 1;
  config.hidden_size = kHiddenSize;
  config.moe_latent_size = 0;
  config.routed_expert_intermediate_size = kIntermediateSize;
  config.shared_expert_intermediate_size = kIntermediateSize;
  config.n_routed_experts = kRoutedExperts;
  config.top_k = kTopK;
  config.max_token_count = kTokenCount;
  config.n_group = 1;
  config.topk_group = 1;
  config.rms_epsilon = 1.0e-5f;
  config.routed_scaling_factor = 5.0f;
  config.norm_topk_prob = true;

  auto slice = ExpertLayerSlice::Create(config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "nonresident expert layer slice should create")) {
    return false;
  }

  std::vector<float> input_values(kTokenCount * kHiddenSize, 0.0f);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t dim = 0; dim < kHiddenSize; ++dim) {
      input_values[token * kHiddenSize + dim] =
          (static_cast<float>(((token + 1) * 11 + (dim * 5)) % 41) - 20.0f) * 0.03125f;
    }
  }

  auto batch_input = DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto batch_output = DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  if (!expect(batch_input != nullptr && batch_output != nullptr,
              "batched tensors should allocate") ||
      !expect(batch_input->CopyFromHost(input_values.data(), input_values.size()),
              "batched input should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  ResetExpertStagingCounters();
  if (!expect(
          slice->Run(*cublas, &heuristic_cache, *batch_input, batch_output.get(), nullptr),
          "nonresident batch run should succeed through fallback") ||
      !expect(cudaDeviceSynchronize() == cudaSuccess, "fallback batch run should synchronize")) {
    return false;
  }

  const auto& counters = GetExpertStagingCounters();
  if (!expect(
          counters.total_staging_calls.load(std::memory_order_relaxed) == 0,
          "nonresident routed weights should not silently restage fast-path experts") ||
      !expect(
          counters.total_experts_staged.load(std::memory_order_relaxed) == 0,
          "nonresident routed weights should not stage routed experts") ||
      !expect(
          counters.total_bytes_uploaded.load(std::memory_order_relaxed) == 0,
          "nonresident routed weights should not upload routed expert bytes at run time") ||
      !expect(
          counters.host_routing_adapter_calls.load(std::memory_order_relaxed) == 0,
          "nonresident fallback should not route through the batched host adapter fast path")) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  return test_unified_fused_prefill_is_default_and_avoids_host_routing_adapter() &&
                 test_unified_fused_prefill_matches_reference_batch_output() &&
                 test_nonresident_routed_weights_reject_fastpath_and_use_fallback()
             ? 0
             : 1;
}
