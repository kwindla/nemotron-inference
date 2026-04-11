#include "nemotron/expert_layer.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/expert_staging_counters.h"
#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_decode.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/linear_op.h"
#include "nemotron/monolithic_expert_weights.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/routed_expert_runtime.h"
#include "nemotron/nvfp4_weight.h"

namespace nemotron {
namespace {

bool EnvEnabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool AllowUnsafeNativeDirectMoePrefillProfiling() {
  return EnvEnabled("NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL");
}

bool DebugComparePrefillVsLegacy() {
  return EnvEnabled("NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY");
}

std::size_t ExecutionRoutedExpertIntermediateSize(const ExpertLayerConfig& config) {
  return ResolveRoutedExpertIntermediateSizeExecution(
      config.routed_expert_intermediate_size,
      config.routed_expert_intermediate_size_padded);
}

std::size_t ExecutionRoutedExpertIntermediateSize(const MoePrefillWorkspaceConfig& config) {
  return ResolveRoutedExpertIntermediateSizeExecution(
      config.routed_expert_intermediate_size,
      config.routed_expert_intermediate_size_padded);
}

MoePrefillWorkspaceConfig BuildMoePrefillWorkspaceConfig(const ExpertLayerConfig& config) {
  MoePrefillWorkspaceConfig workspace_config;
  workspace_config.hidden_size = config.hidden_size;
  workspace_config.num_experts = config.n_routed_experts;
  workspace_config.top_k = config.top_k;
  workspace_config.routed_expert_intermediate_size =
      config.routed_expert_intermediate_size;
  workspace_config.routed_expert_intermediate_size_padded =
      config.routed_expert_intermediate_size_padded;
  workspace_config.shared_expert_intermediate_size =
      config.shared_expert_intermediate_size;
  return workspace_config;
}

bool HasValidPreparedMoeWeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         weight.matmul_block_scales_data != nullptr &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

struct PreparedResidentMoeWeights {
  bool prepared = false;
  FusedNvfp4WeightView shared_up;
  FusedNvfp4WeightView shared_down;
  std::vector<FusedNvfp4WeightView> routed_up_views;
  std::vector<FusedNvfp4WeightView> routed_down_views;
};

thread_local ExpertLayerExecutionCounters g_expert_layer_execution_counters;

void RecordExpertNativeMultiTokenExecution(std::size_t token_count) {
  if (token_count <= 1) {
    return;
  }
  ++g_expert_layer_execution_counters.native_multi_token_runs;
  g_expert_layer_execution_counters.native_multi_token_tokens += token_count;
}

void RecordExpertRowReplayExecution(std::size_t token_count) {
  if (token_count <= 1) {
    return;
  }
  ++g_expert_layer_execution_counters.row_replay_runs;
  g_expert_layer_execution_counters.row_replay_tokens += token_count;
}

bool IsFp32Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp32" || storage_dtype == "float32" || storage_dtype == "float";
}

bool IsBf16Storage(const std::string& storage_dtype) {
  return storage_dtype == "bf16" || storage_dtype == "bfloat16";
}

std::size_t NumelFromShape(const std::vector<std::size_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  std::size_t total = 1;
  for (std::size_t dim : shape) {
    total *= dim;
  }
  return total;
}

std::optional<std::vector<float>> ReadTensorToHostFp32(const KernelTensorDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  const std::size_t count = NumelFromShape(descriptor.logical_shape);
  if (count == 0) {
    return std::nullopt;
  }

  std::vector<float> values(count, 0.0f);
  if (IsFp32Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return values;
  }
  if (IsBf16Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return std::nullopt;
    }
    const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = __bfloat162float(src[i]);
    }
    return values;
  }
  return std::nullopt;
}

std::optional<float> ReadScalarTensorToHostFp32(const KernelTensorDescriptor& descriptor) {
  const auto values = ReadTensorToHostFp32(descriptor);
  if (!values.has_value() || values->size() != 1) {
    return std::nullopt;
  }
  return (*values)[0];
}

std::optional<ScaledFp8LinearConfig> BuildScaledFp8LinearConfig(
    const KernelTensorDescriptor& weight,
    const KernelTensorDescriptor& weight_scale,
    const KernelTensorDescriptor& input_scale) {
  if (weight.logical_shape.size() != 2 || weight.packed_data == nullptr) {
    return std::nullopt;
  }
  const auto weight_scale_value = ReadScalarTensorToHostFp32(weight_scale);
  const auto input_scale_value = ReadScalarTensorToHostFp32(input_scale);
  if (!weight_scale_value.has_value() || !input_scale_value.has_value()) {
    return std::nullopt;
  }

  ScaledFp8LinearConfig config;
  config.output_rows = weight.logical_shape[0];
  config.input_cols = weight.logical_shape[1];
  config.packed_weight_data = weight.packed_data;
  config.packed_weight_nbytes = weight.packed_nbytes;
  config.weight_scale = *weight_scale_value;
  config.input_scale = *input_scale_value;
  return config;
}

const KernelTensorDescriptor* FindExactKernelBinding(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const std::string& local_name) {
  const LayerTensorBinding* binding = layer.FindBinding(local_name);
  if (binding == nullptr) {
    return nullptr;
  }
  return kernel_catalog.FindTensor(binding->tensor_name);
}

const GemmDescriptor* FindExactGemmBinding(
    const LayerScheduleEntry& layer,
    const GemmCatalog& gemm_catalog,
    const std::string& local_name) {
  const LayerTensorBinding* binding = layer.FindBinding(local_name);
  if (binding == nullptr) {
    return nullptr;
  }
  return gemm_catalog.FindDescriptor(binding->tensor_name);
}

float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

float Relu2(float value) {
  return value > 0.0f ? (value * value) : 0.0f;
}

float DecodeFp4(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0F;
  return static_cast<float>(value);
}

float DecodeFp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> DequantizeNvfp4Matrix(
    const std::uint8_t* packed,
    std::size_t packed_nbytes,
    const std::uint8_t* block_scales,
    std::size_t block_scales_nbytes,
    float tensor_scale,
    std::size_t rows,
    std::size_t cols) {
  if (packed == nullptr ||
      block_scales == nullptr ||
      cols == 0 ||
      cols % 16 != 0 ||
      packed_nbytes != (rows * cols) / 2 ||
      block_scales_nbytes != rows * (cols / 16)) {
    return {};
  }

  std::vector<float> output(rows * cols, 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < cols / 16; ++block) {
      const float block_scale = DecodeFp8(block_scales[scale_index++]) * tensor_scale;
      const std::size_t col_start = block * 16;
      for (std::size_t offset = 0; offset < 16; offset += 2) {
        const std::uint8_t byte = packed[packed_index++];
        output[row * cols + col_start + offset] = DecodeFp4(byte & 0x0F) * block_scale;
        output[row * cols + col_start + offset + 1] = DecodeFp4((byte >> 4) & 0x0F) * block_scale;
      }
    }
  }
  return output;
}

std::optional<std::vector<float>> DequantizeNvfp4WeightToHostFp32(const GemmDescriptor& descriptor) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr ||
      descriptor.tensor_scale_data == nullptr ||
      descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }
  float tensor_scale = 0.0f;
  std::memcpy(&tensor_scale, descriptor.tensor_scale_data, sizeof(float));
  std::vector<float> values = DequantizeNvfp4Matrix(
      descriptor.packed_data,
      descriptor.packed_nbytes,
      descriptor.block_scales_data,
      descriptor.block_scales_nbytes,
      tensor_scale,
      descriptor.output_rows,
      descriptor.input_cols);
  if (values.empty()) {
    return std::nullopt;
  }
  return values;
}

std::vector<float> CpuMatmulRowMajor(
    const std::vector<float>& activations,
    std::size_t rows,
    const std::vector<float>& weights,
    std::size_t output_rows,
    std::size_t input_cols) {
  std::vector<float> output(rows * output_rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_rows; ++out) {
      float accum = 0.0f;
      for (std::size_t col = 0; col < input_cols; ++col) {
        accum += activations[row * input_cols + col] * weights[out * input_cols + col];
      }
      output[row * output_rows + out] = accum;
    }
  }
  return output;
}

std::optional<std::vector<float>> RunNvfp4LinearHost(
    const std::vector<float>& activations,
    std::size_t rows,
    const GemmDescriptor& descriptor) {
  if (rows == 0 ||
      activations.size() != rows * descriptor.input_cols ||
      descriptor.input_cols == 0 ||
      descriptor.output_rows == 0) {
    return std::nullopt;
  }
  const auto activation_pack = PackRowMajorFp32ToNvfp4(
      activations.data(),
      rows,
      descriptor.input_cols,
      {});
  if (!activation_pack.has_value()) {
    return std::nullopt;
  }
  const std::vector<float> activation_dequant = DequantizeNvfp4Matrix(
      activation_pack->packed_data(),
      activation_pack->packed_nbytes(),
      activation_pack->block_scales_data(),
      activation_pack->block_scales_nbytes(),
      activation_pack->tensor_scale,
      rows,
      descriptor.input_cols);
  const auto weight_dequant = DequantizeNvfp4WeightToHostFp32(descriptor);
  if (activation_dequant.empty() || !weight_dequant.has_value()) {
    return std::nullopt;
  }
  return CpuMatmulRowMajor(
      activation_dequant,
      rows,
      *weight_dequant,
      descriptor.output_rows,
      descriptor.input_cols);
}

std::vector<float> ApplyRelu2(const std::vector<float>& input) {
  std::vector<float> output = input;
  for (float& value : output) {
    value = Relu2(value);
  }
  return output;
}

std::vector<ExpertSelection> SelectTopExperts(
    const ExpertLayerConfig& config,
    const std::vector<float>& router_logits,
    const std::vector<float>& correction_bias) {
  std::vector<ExpertSelection> selections;
  if (config.n_routed_experts == 0 ||
      config.top_k == 0 ||
      router_logits.size() != config.n_routed_experts ||
      correction_bias.size() != config.n_routed_experts ||
      config.n_group == 0 ||
      config.topk_group == 0 ||
      config.n_routed_experts % config.n_group != 0) {
    return selections;
  }

  std::vector<float> scores(config.n_routed_experts, 0.0f);
  std::vector<float> scores_for_choice(config.n_routed_experts, 0.0f);
  for (std::size_t i = 0; i < config.n_routed_experts; ++i) {
    scores[i] = Sigmoid(router_logits[i]);
    scores_for_choice[i] = scores[i] + correction_bias[i];
  }

  const std::size_t group_size = config.n_routed_experts / config.n_group;
  std::vector<std::pair<float, std::size_t>> group_scores;
  group_scores.reserve(config.n_group);
  for (std::size_t group = 0; group < config.n_group; ++group) {
    float top1 = -std::numeric_limits<float>::infinity();
    float top2 = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < group_size; ++i) {
      const float value = scores_for_choice[group * group_size + i];
      if (value > top1) {
        top2 = top1;
        top1 = value;
      } else if (value > top2) {
        top2 = value;
      }
    }
    if (!std::isfinite(top2)) {
      top2 = top1;
    }
    group_scores.emplace_back(top1 + top2, group);
  }

  const std::size_t selected_group_count = std::min(config.topk_group, group_scores.size());
  std::partial_sort(
      group_scores.begin(),
      group_scores.begin() + selected_group_count,
      group_scores.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

  std::vector<bool> group_selected(config.n_group, false);
  for (std::size_t i = 0; i < selected_group_count; ++i) {
    group_selected[group_scores[i].second] = true;
  }

  std::vector<std::pair<float, std::size_t>> masked_scores;
  masked_scores.reserve(config.n_routed_experts);
  for (std::size_t group = 0; group < config.n_group; ++group) {
    for (std::size_t i = 0; i < group_size; ++i) {
      const std::size_t expert_index = group * group_size + i;
      masked_scores.emplace_back(group_selected[group] ? scores_for_choice[expert_index] : 0.0f, expert_index);
    }
  }

  const std::size_t selected_expert_count = std::min(config.top_k, masked_scores.size());
  std::partial_sort(
      masked_scores.begin(),
      masked_scores.begin() + selected_expert_count,
      masked_scores.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

  selections.reserve(selected_expert_count);
  float weight_sum = 0.0f;
  for (std::size_t i = 0; i < selected_expert_count; ++i) {
    const std::size_t expert_index = masked_scores[i].second;
    const float weight = scores[expert_index];
    selections.push_back(ExpertSelection{expert_index, weight});
    weight_sum += weight;
  }

  if (config.norm_topk_prob) {
    const float denominator = weight_sum + 1.0e-20f;
    for (ExpertSelection& selection : selections) {
      selection.weight /= denominator;
    }
  }
  for (ExpertSelection& selection : selections) {
    selection.weight *= config.routed_scaling_factor;
  }
  return selections;
}

bool BuildSelectedExpertsFromTopK(
    const ExpertLayerConfig& config,
    const std::vector<int>& topk_ids,
    const std::vector<float>& topk_weights,
    std::vector<ExpertSelection>* selected_experts) {
  if (selected_experts == nullptr ||
      topk_ids.size() != config.top_k ||
      topk_weights.size() != config.top_k) {
    return false;
  }

  selected_experts->clear();
  selected_experts->reserve(config.top_k);
  for (std::size_t slot = 0; slot < config.top_k; ++slot) {
    const int expert_index = topk_ids[slot];
    if (expert_index < 0 ||
        static_cast<std::size_t>(expert_index) >= config.n_routed_experts) {
      return false;
    }
    selected_experts->push_back(ExpertSelection{
        static_cast<std::size_t>(expert_index),
        topk_weights[slot]});
  }
  return true;
}

bool CopyToHost(const DeviceTensorFp32& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  output->assign(tensor.numel(), 0.0f);
  return tensor.CopyToHost(output->data(), output->size());
}

bool CopyToHost(const DeviceTensorBf16& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  if (!tensor.CopyToHost(host_bf16.data(), host_bf16.size())) {
    return false;
  }
  output->assign(host_bf16.size(), 0.0f);
  for (std::size_t i = 0; i < host_bf16.size(); ++i) {
    (*output)[i] = __bfloat162float(host_bf16[i]);
  }
  return true;
}

bool UploadHostVector(const std::vector<float>& input, DeviceTensorFp32* output) {
  return output != nullptr &&
         output->valid() &&
         output->numel() == input.size() &&
         output->CopyFromHost(input.data(), input.size());
}

std::unique_ptr<DeviceTensorFp32> UploadHostVectorToDevice(const std::vector<float>& values) {
  if (values.empty()) {
    return nullptr;
  }
  auto output = DeviceTensorFp32::Create({values.size()});
  if (!output || !UploadHostVector(values, output.get())) {
    return nullptr;
  }
  return output;
}

template <typename T>
class DeviceArray {
 public:
  static std::unique_ptr<DeviceArray<T>> Create(std::size_t count) {
    if (count == 0) {
      return nullptr;
    }
    T* data = nullptr;
    const std::size_t bytes = count * sizeof(T);
    if (cudaMalloc(reinterpret_cast<void**>(&data), bytes) != cudaSuccess) {
      return nullptr;
    }
    return std::unique_ptr<DeviceArray<T>>(new DeviceArray<T>(data, count));
  }

  static std::unique_ptr<DeviceArray<T>> CopyFromHost(const std::vector<T>& values) {
    auto output = Create(values.size());
    if (!output) {
      return nullptr;
    }
    const std::size_t bytes = values.size() * sizeof(T);
    if (cudaMemcpy(output->data_, values.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      return nullptr;
    }
    return output;
  }

  ~DeviceArray() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&&) = delete;
  DeviceArray& operator=(DeviceArray&&) = delete;

  bool CopyToHost(std::vector<T>* output) const {
    if (output == nullptr || data_ == nullptr) {
      return false;
    }
    output->assign(size_, T{});
    return cudaMemcpy(
               output->data(),
               data_,
               size_ * sizeof(T),
               cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  DeviceArray(T* data, std::size_t size) : data_(data), size_(size) {}

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

template <typename T>
bool CopyDeviceBufferToHost(
    const T* device_data,
    std::size_t count,
    std::vector<T>* output) {
  if (device_data == nullptr || output == nullptr) {
    return false;
  }
  output->assign(count, T{});
  return cudaMemcpy(
             output->data(),
             device_data,
             count * sizeof(T),
             cudaMemcpyDeviceToHost) == cudaSuccess;
}

void WarnRoutedPhase05DumpFailure(const std::string& message) {
  static bool warned = false;
  if (!warned) {
    warned = true;
    std::cerr << "expert_layer: routed phase05 dump failed: " << message << "\n";
  }
}

void MaybeDumpRoutedPhase05Histogram(
    const ExpertLayerConfig& config,
    std::size_t token_count,
    const DeviceExpertRouting& routing,
    const DeviceMoeLaunchPlan& launch_plan) {
  const char* dump_path = std::getenv("NEMOTRON_ROUTED_PHASE05_DUMP");
  if (dump_path == nullptr || dump_path[0] == '\0') {
    return;
  }

  static std::mutex dumped_keys_mutex;
  static std::set<std::pair<std::size_t, std::size_t>> dumped_keys;
  const std::lock_guard<std::mutex> lock(dumped_keys_mutex);
  const auto dump_key = std::make_pair(config.layer_index, token_count);
  if (dumped_keys.find(dump_key) != dumped_keys.end()) {
    return;
  }

  std::vector<int> active_expert_count_host;
  std::vector<int> active_expert_ids_host;
  std::vector<int> expert_offsets_host;
  std::vector<int> cta_count_host;
  std::vector<int> total_padded_rows_host;
  std::vector<int> expert_first_token_offsets_host;
  if (!CopyDeviceBufferToHost(routing.active_expert_count(), 1, &active_expert_count_host) ||
      !CopyDeviceBufferToHost(routing.active_expert_ids(), routing.n_experts(), &active_expert_ids_host) ||
      !CopyDeviceBufferToHost(routing.expert_offsets(), routing.n_experts() + 1, &expert_offsets_host) ||
      !CopyDeviceBufferToHost(launch_plan.cta_count(), 1, &cta_count_host) ||
      !CopyDeviceBufferToHost(launch_plan.total_padded_rows(), 1, &total_padded_rows_host) ||
      !CopyDeviceBufferToHost(
          launch_plan.expert_first_token_offsets(),
          launch_plan.n_experts() + 1,
          &expert_first_token_offsets_host)) {
    WarnRoutedPhase05DumpFailure("device-to-host copy");
    return;
  }

  const int cta_count = cta_count_host.empty() ? 0 : cta_count_host.front();
  if (cta_count < 0 || static_cast<std::size_t>(cta_count) > launch_plan.cta_capacity()) {
    WarnRoutedPhase05DumpFailure("invalid cta count");
    return;
  }

  std::vector<int> cta_expert_ids_host;
  if (!CopyDeviceBufferToHost(
          launch_plan.cta_expert_ids(),
          static_cast<std::size_t>(cta_count),
          &cta_expert_ids_host)) {
    WarnRoutedPhase05DumpFailure("cta expert ids");
    return;
  }

  struct ExpertHistogramEntry {
    int expert_id = -1;
    int selection_count = 0;
    int cta_count = 0;
    int padded_row_begin = 0;
    int padded_row_end = 0;
  };

  std::vector<int> cta_counts_by_expert(routing.n_experts(), 0);
  for (int cta_index = 0; cta_index < cta_count; ++cta_index) {
    const int expert_id = cta_expert_ids_host[static_cast<std::size_t>(cta_index)];
    if (expert_id >= 0 && static_cast<std::size_t>(expert_id) < cta_counts_by_expert.size()) {
      ++cta_counts_by_expert[static_cast<std::size_t>(expert_id)];
    }
  }

  std::vector<ExpertHistogramEntry> entries;
  entries.reserve(routing.n_experts());
  for (std::size_t expert_index = 0; expert_index < routing.n_experts(); ++expert_index) {
    const int selection_count =
        expert_offsets_host[expert_index + 1] - expert_offsets_host[expert_index];
    if (selection_count <= 0) {
      continue;
    }
    entries.push_back(ExpertHistogramEntry{
        static_cast<int>(expert_index),
        selection_count,
        cta_counts_by_expert[expert_index],
        expert_first_token_offsets_host[expert_index],
        expert_first_token_offsets_host[expert_index + 1]});
  }

  std::sort(
      entries.begin(),
      entries.end(),
      [](const ExpertHistogramEntry& lhs, const ExpertHistogramEntry& rhs) {
        if (lhs.selection_count != rhs.selection_count) {
          return lhs.selection_count > rhs.selection_count;
        }
        if (lhs.cta_count != rhs.cta_count) {
          return lhs.cta_count > rhs.cta_count;
        }
        return lhs.expert_id < rhs.expert_id;
      });

  std::ofstream output(dump_path, std::ios::app);
  if (!output.is_open()) {
    WarnRoutedPhase05DumpFailure(std::string("open failed: ") + dump_path);
    return;
  }

  output << "{";
  output << "\"layer_index\":" << config.layer_index;
  output << ",\"token_count\":" << token_count;
  output << ",\"top_k\":" << config.top_k;
  output << ",\"selection_count\":" << routing.selection_count();
  output << ",\"active_expert_count\":"
         << (active_expert_count_host.empty() ? 0 : active_expert_count_host.front());
  output << ",\"selected_token_tile\":" << launch_plan.selected_token_tile();
  output << ",\"cta_count\":" << cta_count;
  output << ",\"total_padded_rows\":"
         << (total_padded_rows_host.empty() ? 0 : total_padded_rows_host.front());
  output << ",\"active_expert_ids\":[";
  const int active_expert_count =
      active_expert_count_host.empty() ? 0 : active_expert_count_host.front();
  for (int index = 0; index < active_expert_count; ++index) {
    if (index != 0) {
      output << ",";
    }
    output << active_expert_ids_host[static_cast<std::size_t>(index)];
  }
  output << "]";
  output << ",\"experts\":[";
  for (std::size_t index = 0; index < entries.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    const auto& entry = entries[index];
    output << "{";
    output << "\"expert_id\":" << entry.expert_id;
    output << ",\"selection_count\":" << entry.selection_count;
    output << ",\"cta_count\":" << entry.cta_count;
    output << ",\"padded_row_begin\":" << entry.padded_row_begin;
    output << ",\"padded_row_end\":" << entry.padded_row_end;
    output << "}";
  }
  output << "]";
  output << "}\n";
  dumped_keys.insert(dump_key);
}

std::unique_ptr<DeviceTensorFp32> CreateWorkspaceView(
    DeviceTensorFp32* buffer,
    std::size_t rows,
    std::size_t cols) {
  if (buffer == nullptr ||
      !buffer->valid() ||
      buffer->shape().size() != 2 ||
      buffer->shape()[0] < rows ||
      buffer->shape()[1] != cols) {
    return nullptr;
  }
  return DeviceTensorFp32::CreateView({rows, cols}, buffer->data());
}

std::unique_ptr<DeviceTensorFp32> CreateTokenRowView(
    DeviceTensorFp32* buffer,
    std::size_t row_index,
    std::size_t cols) {
  if (buffer == nullptr ||
      !buffer->valid() ||
      buffer->shape().size() != 2 ||
      row_index >= buffer->shape()[0] ||
      buffer->shape()[1] != cols) {
    return nullptr;
  }
  return DeviceTensorFp32::CreateView({1, cols}, buffer->data() + (row_index * cols));
}

std::unique_ptr<DeviceTensorBf16> CreateWorkspaceView(
    DeviceTensorBf16* buffer,
    std::size_t rows,
    std::size_t cols) {
  if (buffer == nullptr ||
      !buffer->valid() ||
      buffer->shape().size() != 2 ||
      buffer->shape()[0] < rows ||
      buffer->shape()[1] != cols) {
    return nullptr;
  }
  return DeviceTensorBf16::CreateView({rows, cols}, buffer->data());
}

std::unique_ptr<DeviceTensorInt32> CreateWorkspaceView(
    DeviceTensorInt32* buffer,
    std::size_t rows,
    std::size_t cols) {
  if (buffer == nullptr ||
      !buffer->valid() ||
      buffer->shape().size() != 2 ||
      buffer->shape()[0] < rows ||
      buffer->shape()[1] != cols) {
    return nullptr;
  }
  return DeviceTensorInt32::CreateView({rows, cols}, buffer->data());
}

bool WorkspaceMatchesConfig(
    const MoePrefillWorkspace& workspace,
    const ExpertLayerConfig& config) {
  return workspace.valid() &&
         workspace.config.hidden_size == config.hidden_size &&
         workspace.config.num_experts == config.n_routed_experts &&
         workspace.config.top_k == config.top_k &&
         ExecutionRoutedExpertIntermediateSize(workspace.config) ==
             ExecutionRoutedExpertIntermediateSize(config) &&
         workspace.config.shared_expert_intermediate_size ==
             config.shared_expert_intermediate_size;
}

FusedNvfp4WeightView MakeFusedNvfp4WeightView(const DeviceNvfp4Weight& weight) {
  return FusedNvfp4WeightView{
      weight.packed_data(),
      weight.block_scales_data(),
      weight.matmul_block_scales_data(),
      reinterpret_cast<const float*>(weight.tensor_scale_data()),
      weight.output_rows(),
      weight.input_cols(),
      weight.p5_tma_load_a(),
      weight.p5_tma_load_sfa(),
  };
}

Nvfp4PackedMatrixDeviceView MakeNvfp4PackedMatrixDeviceView(const FusedNvfp4WeightView& weight) {
  return Nvfp4PackedMatrixDeviceView{
      weight.packed_data,
      PackedFp4Bytes(weight.output_rows, weight.input_cols),
      weight.matmul_block_scales_data,
      ExecutionNvfp4ScaleBytes(weight.output_rows, weight.input_cols),
      weight.tensor_scale_data,
      sizeof(float),
      weight.output_rows,
      weight.input_cols,
  };
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         weight.matmul_block_scales_data != nullptr &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

std::optional<CublasLtGemmPlan> BuildRuntimeNvfp4GemmPlan(
    const GemmDescriptor& descriptor,
    const Nvfp4PackedMatrixDeviceView& weight_view,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache) {
  GemmDescriptor execution_descriptor = descriptor;
  execution_descriptor.output_rows = weight_view.rows;
  execution_descriptor.input_cols = weight_view.cols;
  execution_descriptor.packed_nbytes = weight_view.packed_nbytes;
  execution_descriptor.block_scales_nbytes = weight_view.block_scales_nbytes;
  execution_descriptor.tensor_scale_nbytes = weight_view.tensor_scale_nbytes;
  auto runtime_launch_plan = BuildGemmLaunchPlan(execution_descriptor, rows);
  if (!runtime_launch_plan.has_value()) {
    return std::nullopt;
  }
  runtime_launch_plan->packed_bytes = ByteRangeView{
      weight_view.packed_data,
      weight_view.packed_nbytes,
  };
  runtime_launch_plan->block_scales_bytes = ByteRangeView{
      weight_view.block_scales_data,
      weight_view.block_scales_nbytes,
  };
  runtime_launch_plan->tensor_scale_bytes = ByteRangeView{
      reinterpret_cast<const std::uint8_t*>(weight_view.tensor_scale_data),
      weight_view.tensor_scale_nbytes,
  };
  const auto execution = PrepareGemmExecution(*runtime_launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    return std::nullopt;
  }
  return BuildCublasLtGemmPlan(*execution);
}

std::uint64_t TotalUploadedBytes(const DeviceNvfp4Weight& weight) {
  return static_cast<std::uint64_t>(weight.packed_nbytes()) +
         static_cast<std::uint64_t>(weight.block_scales_nbytes()) +
         static_cast<std::uint64_t>(weight.matmul_block_scales_nbytes()) +
         static_cast<std::uint64_t>(weight.tensor_scale_nbytes()) +
         static_cast<std::uint64_t>(weight.p5_tma_load_a_nbytes()) +
         static_cast<std::uint64_t>(weight.p5_tma_load_sfa_nbytes());
}

std::uint64_t EstimatedUploadedBytes(const GemmDescriptor& descriptor) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    return 0;
  }
  const std::size_t execution_output_rows =
      IsRoutedExpertUpOpClass(descriptor.op_class)
          ? DefaultRoutedExpertIntermediateSizePadded(descriptor.output_rows)
          : descriptor.output_rows;
  const std::size_t execution_input_cols =
      IsRoutedExpertDownOpClass(descriptor.op_class)
          ? DefaultRoutedExpertIntermediateSizePadded(descriptor.input_cols)
          : descriptor.input_cols;
  const std::size_t packed_nbytes = (execution_output_rows * execution_input_cols) / 2u;
  const std::size_t block_scales_nbytes =
      execution_output_rows * (execution_input_cols / kNvfp4ScaleBlockWidthRuntime);
  return static_cast<std::uint64_t>(packed_nbytes) +
         static_cast<std::uint64_t>(block_scales_nbytes) +
         static_cast<std::uint64_t>(
             ExecutionNvfp4ScaleBytes(execution_output_rows, execution_input_cols)) +
         static_cast<std::uint64_t>(descriptor.tensor_scale_nbytes);
}

int GetCurrentDeviceSmVersion() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return 0;
  }

  int major = 0;
  int minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return 0;
  }
  return major * 10 + minor;
}

Nvfp4PackOptions RuntimeMoeNvfp4PackOptions() {
  Nvfp4PackOptions options;
  options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  return options;
}

const float* RawNvfp4TensorScalePtr(const GemmDescriptor& descriptor) {
  if (descriptor.tensor_scale_data == nullptr ||
      descriptor.tensor_scale_nbytes != sizeof(float)) {
    return nullptr;
  }
  return reinterpret_cast<const float*>(descriptor.tensor_scale_data);
}

bool UploadDescriptorToMonolithic(
    MonolithicNvfp4ExpertWeights* weights,
    std::size_t expert_index,
    const GemmDescriptor& descriptor) {
  if (weights == nullptr ||
      descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr) {
    return false;
  }
  const float* tensor_scale = RawNvfp4TensorScalePtr(descriptor);
  if (tensor_scale == nullptr) {
    return false;
  }
  return weights->UploadExpert(
      expert_index,
      descriptor.packed_data,
      descriptor.packed_nbytes,
      descriptor.block_scales_data,
      descriptor.block_scales_nbytes,
      tensor_scale,
      descriptor.output_rows,
      descriptor.input_cols);
}

bool HasNonFinite(const std::vector<float>& values) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  return false;
}

}  // namespace

void ResetExpertLayerExecutionCounters() {
  g_expert_layer_execution_counters = ExpertLayerExecutionCounters{};
}

ExpertLayerExecutionCounters GetExpertLayerExecutionCounters() {
  return g_expert_layer_execution_counters;
}

struct ExpertLayerSlice::Impl {
  struct RoutedExpertRuntime {
    const GemmDescriptor* up_proj = nullptr;
    const GemmDescriptor* down_proj = nullptr;
  };

  struct DirectMoeExecutionState {
    ExpertLayerRunTrace* trace = nullptr;
    MoePrefillWorkspace* workspace = nullptr;
    bool use_decode_scratch = false;
  };

  enum class ExpertTopology {
    kLatentProjection,
    kDirectMlp,
  };

  enum class ProjectionFamily {
    kNone,
    kDense,
    kScaledFp8,
    kNvfp4,
  };

  enum class SharedDownFamily {
    kNone,
    kDense,
    kScaledFp8,
    kNvfp4,
  };

  ExpertLayerConfig config;
  ExpertTopology topology = ExpertTopology::kLatentProjection;
  std::unique_ptr<DeviceTensorFp32> input_norm_weight;
  std::vector<float> gate_score_correction_bias;
  std::unique_ptr<DeviceTensorFp32> gate_score_correction_bias_device;
  std::unique_ptr<UploadedLinearOp> gate_weight;
  ProjectionFamily fc1_latent_family = ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> fc1_latent_dense;
  std::unique_ptr<ScaledFp8LinearOp> fc1_latent_scaled_fp8;
  std::unique_ptr<UploadedLinearOp> fc2_latent;
  ProjectionFamily shared_up_family = ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_up_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_up_scaled_fp8;
  const GemmDescriptor* shared_up_nvfp4 = nullptr;
  std::unique_ptr<DeviceNvfp4Weight> shared_up_nvfp4_device;
  SharedDownFamily shared_down_family = SharedDownFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_down_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_down_scaled_fp8;
  const GemmDescriptor* shared_down_nvfp4 = nullptr;
  std::unique_ptr<DeviceNvfp4Weight> shared_down_nvfp4_device;
  std::vector<RoutedExpertRuntime> routed_experts;
  std::unique_ptr<MonolithicNvfp4ExpertWeights> monolithic_up;
  std::unique_ptr<MonolithicNvfp4ExpertWeights> monolithic_down;
  std::unique_ptr<DeviceTensorInt32> device_topk_ids;
  std::unique_ptr<DeviceTensorFp32> device_topk_weights;
  std::unique_ptr<DeviceExpertRouting> fused_prefill_routing;
  std::unique_ptr<DeviceMoeLaunchPlan> fused_prefill_launch_plan;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_routed_output_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_gather_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_expert_up_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_shared_up_scratch;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_normalized_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_gather_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_expert_up_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_shared_up_pack;
  std::unique_ptr<DeviceTensorBf16> normalized_bf16_scratch;
  std::unique_ptr<DeviceTensorFp32> normalized_scratch;
  std::unique_ptr<DeviceTensorFp32> router_logits_scratch;
  std::unique_ptr<DeviceTensorFp32> output_scratch;
  std::unique_ptr<DeviceTensorFp32> routed_output_scratch;
  std::unique_ptr<DeviceTensorFp32> routed_up_scratch;
  std::unique_ptr<DeviceTensorFp32> shared_up_scratch;
  std::unique_ptr<DeviceNvfp4Matrix> normalized_pack;
  std::unique_ptr<DeviceNvfp4Matrix> routed_activated_pack;
  std::unique_ptr<DeviceNvfp4Matrix> shared_activated_pack;
  std::uint64_t resident_shared_expert_bytes = 0;
  std::uint64_t resident_routed_expert_bytes = 0;
  bool monolithic_resident = false;
  bool fused_direct_moe_supported = false;
  PreparedResidentMoeWeights direct_moe_weights;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> direct_moe_routed_up_views_device;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> direct_moe_routed_down_views_device;
  DirectMoeExecutionState direct_moe_execution_state;

  static bool BuildResidentRoutedWeightViews(
      const Impl& impl,
      std::vector<FusedNvfp4WeightView>* routed_up_views,
      std::vector<FusedNvfp4WeightView>* routed_down_views);

  bool SupportsDirectMoeDecodePath(
      const ExpertLayerConfig& config,
      std::size_t token_count,
      int device_sm_version) const;

  bool SupportsDirectMoePrefillPath(
      const ExpertLayerConfig& config,
      std::size_t token_count,
      int device_sm_version) const;

  bool HasResidentPreparedRoutedWeights() const;

  bool SupportsResidentPreparedMoePath(
      const ExpertLayerConfig& path_config,
      std::size_t token_count) const;

  DeviceTensorInt32* ResolveTopkIdsTensor() const;

  DeviceTensorFp32* ResolveTopkWeightsTensor() const;

  DeviceExpertRouting* ResolveFusedPrefillRouting() const;

  DeviceMoeLaunchPlan* ResolveFusedPrefillLaunchPlan() const;

  DeviceTensorFp32* ResolveFusedPrefillRoutedOutputScratch() const;

  DeviceTensorFp32* ResolveFusedPrefillGatherScratch() const;

  DeviceTensorFp32* ResolveFusedPrefillExpertUpScratch() const;

  DeviceTensorFp32* ResolveFusedPrefillSharedUpScratch() const;

  DeviceNvfp4Matrix* ResolveFusedPrefillNormalizedPack() const;

  DeviceNvfp4Matrix* ResolveFusedPrefillGatherPack() const;

  DeviceNvfp4Matrix* ResolveFusedPrefillExpertUpPack() const;

  DeviceNvfp4Matrix* ResolveFusedPrefillSharedUpPack() const;

  std::size_t ResolveMultiTokenCapacity() const;

  void ResetDirectMoeExecutionState(
      ExpertLayerRunTrace* trace,
      bool use_decode_scratch,
      MoePrefillWorkspace* workspace);

  bool RunDirectMoeExpertSelection(
      const DeviceTensorFp32& router_logits,
      std::size_t token_count);

  bool RunDirectMoeDecodePath(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorFp32& input,
      const DeviceTensorFp32& normalized,
      const DeviceTensorFp32& router_logits,
      const int* topk_ids,
      const float* topk_weights,
      DeviceTensorFp32* output);

  bool RunFusedMoePrefillPath(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorFp32& input,
      const DeviceTensorFp32& normalized,
      const DeviceTensorFp32& router_logits,
      const int* topk_ids,
      const float* topk_weights,
      DeviceTensorFp32* output) const;

};

struct MoeDirectDecodeScratch {
  DeviceTensorFp32* output_accum = nullptr;
  DeviceTensorFp32* routed_up = nullptr;
  DeviceTensorFp32* shared_up = nullptr;
};

bool ExpertLayerSlice::Impl::BuildResidentRoutedWeightViews(
    const Impl& impl,
    std::vector<FusedNvfp4WeightView>* routed_up_views,
    std::vector<FusedNvfp4WeightView>* routed_down_views) {
  if (routed_up_views == nullptr || routed_down_views == nullptr) {
    return false;
  }

  routed_up_views->clear();
  routed_down_views->clear();

  if (impl.monolithic_resident) {
    if (impl.monolithic_up == nullptr ||
        impl.monolithic_down == nullptr ||
        !impl.monolithic_up->valid() ||
        !impl.monolithic_down->valid()) {
      return false;
    }
    *routed_up_views = impl.monolithic_up->BuildAllViews();
    *routed_down_views = impl.monolithic_down->BuildAllViews();
  } else {
    return false;
  }

  return routed_up_views->size() == impl.routed_experts.size() &&
         routed_down_views->size() == impl.routed_experts.size();
}

bool ExpertLayerSlice::Impl::HasResidentPreparedRoutedWeights() const {
  return direct_moe_weights.prepared &&
         direct_moe_weights.routed_up_views.size() == routed_experts.size() &&
         direct_moe_weights.routed_down_views.size() == routed_experts.size();
}

DeviceTensorInt32* ExpertLayerSlice::Impl::ResolveTopkIdsTensor() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->topk_ids != nullptr &&
      direct_moe_execution_state.workspace->topk_ids->valid()) {
    return direct_moe_execution_state.workspace->topk_ids.get();
  }
  return device_topk_ids.get();
}

DeviceTensorFp32* ExpertLayerSlice::Impl::ResolveTopkWeightsTensor() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->topk_weights != nullptr &&
      direct_moe_execution_state.workspace->topk_weights->valid()) {
    return direct_moe_execution_state.workspace->topk_weights.get();
  }
  return device_topk_weights.get();
}

DeviceExpertRouting* ExpertLayerSlice::Impl::ResolveFusedPrefillRouting() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_routing != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_routing->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_routing.get();
  }
  return fused_prefill_routing.get();
}

DeviceMoeLaunchPlan* ExpertLayerSlice::Impl::ResolveFusedPrefillLaunchPlan() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_launch_plan != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_launch_plan->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_launch_plan.get();
  }
  return fused_prefill_launch_plan.get();
}

DeviceTensorFp32* ExpertLayerSlice::Impl::ResolveFusedPrefillRoutedOutputScratch() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_routed_output_scratch != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_routed_output_scratch->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_routed_output_scratch.get();
  }
  return fused_prefill_routed_output_scratch.get();
}

DeviceTensorFp32* ExpertLayerSlice::Impl::ResolveFusedPrefillGatherScratch() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_gather_scratch != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_gather_scratch->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_gather_scratch.get();
  }
  return fused_prefill_gather_scratch.get();
}

DeviceTensorFp32* ExpertLayerSlice::Impl::ResolveFusedPrefillExpertUpScratch() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_expert_up_scratch != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_expert_up_scratch->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_expert_up_scratch.get();
  }
  return fused_prefill_expert_up_scratch.get();
}

DeviceTensorFp32* ExpertLayerSlice::Impl::ResolveFusedPrefillSharedUpScratch() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_shared_up_scratch != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_shared_up_scratch->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_shared_up_scratch.get();
  }
  return fused_prefill_shared_up_scratch.get();
}

DeviceNvfp4Matrix* ExpertLayerSlice::Impl::ResolveFusedPrefillNormalizedPack() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_normalized_pack != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_normalized_pack->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_normalized_pack.get();
  }
  return fused_prefill_normalized_pack.get();
}

DeviceNvfp4Matrix* ExpertLayerSlice::Impl::ResolveFusedPrefillGatherPack() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_gather_pack != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_gather_pack->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_gather_pack.get();
  }
  return fused_prefill_gather_pack.get();
}

DeviceNvfp4Matrix* ExpertLayerSlice::Impl::ResolveFusedPrefillExpertUpPack() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_expert_up_pack != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_expert_up_pack->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_expert_up_pack.get();
  }
  return fused_prefill_expert_up_pack.get();
}

DeviceNvfp4Matrix* ExpertLayerSlice::Impl::ResolveFusedPrefillSharedUpPack() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_shared_up_pack != nullptr &&
      direct_moe_execution_state.workspace->fused_prefill_shared_up_pack->valid()) {
    return direct_moe_execution_state.workspace->fused_prefill_shared_up_pack.get();
  }
  return fused_prefill_shared_up_pack.get();
}

std::size_t ExpertLayerSlice::Impl::ResolveMultiTokenCapacity() const {
  if (direct_moe_execution_state.workspace != nullptr &&
      direct_moe_execution_state.workspace->valid()) {
    return direct_moe_execution_state.workspace->token_capacity();
  }
  return config.max_token_count;
}

bool ExpertLayerSlice::Impl::SupportsResidentPreparedMoePath(
    const ExpertLayerConfig& path_config,
    std::size_t token_count) const {
  const DeviceTensorInt32* topk_ids_tensor = ResolveTopkIdsTensor();
  const DeviceTensorFp32* topk_weights_tensor = ResolveTopkWeightsTensor();
  return token_count >= 1 &&
         token_count <= ResolveMultiTokenCapacity() &&
         path_config.hidden_size == config.hidden_size &&
         ExecutionRoutedExpertIntermediateSize(path_config) ==
             ExecutionRoutedExpertIntermediateSize(config) &&
         path_config.shared_expert_intermediate_size ==
             config.shared_expert_intermediate_size &&
         path_config.n_routed_experts == config.n_routed_experts &&
         path_config.top_k == config.top_k &&
         path_config.top_k > 0 &&
         path_config.top_k <= path_config.n_routed_experts &&
         fused_direct_moe_supported &&
         shared_up_nvfp4 != nullptr &&
         shared_down_nvfp4 != nullptr &&
         shared_up_nvfp4_device != nullptr &&
         shared_down_nvfp4_device != nullptr &&
         shared_up_nvfp4_device->valid() &&
         shared_down_nvfp4_device->valid() &&
         topk_ids_tensor != nullptr &&
         topk_ids_tensor->valid() &&
         topk_weights_tensor != nullptr &&
         topk_weights_tensor->valid() &&
         HasResidentPreparedRoutedWeights();
}

void ExpertLayerSlice::Impl::ResetDirectMoeExecutionState(
    ExpertLayerRunTrace* trace,
    bool use_decode_scratch,
    MoePrefillWorkspace* workspace) {
  direct_moe_execution_state.trace = trace;
  direct_moe_execution_state.workspace = workspace;
  direct_moe_execution_state.use_decode_scratch = use_decode_scratch;
}

bool ExpertLayerSlice::Impl::RunDirectMoeExpertSelection(
    const DeviceTensorFp32& router_logits,
    std::size_t token_count) {
  DeviceTensorInt32* topk_ids_tensor = ResolveTopkIdsTensor();
  DeviceTensorFp32* topk_weights_tensor = ResolveTopkWeightsTensor();
  if (!router_logits.valid() ||
      gate_score_correction_bias_device == nullptr ||
      !gate_score_correction_bias_device->valid() ||
      topk_ids_tensor == nullptr ||
      !topk_ids_tensor->valid() ||
      topk_weights_tensor == nullptr ||
      !topk_weights_tensor->valid() ||
      config.top_k == 0 ||
      token_count == 0 ||
      token_count > ResolveMultiTokenCapacity() ||
      token_count > (std::numeric_limits<std::size_t>::max() / config.top_k)) {
    return false;
  }

  const std::size_t selection_count = token_count * config.top_k;
  if (topk_ids_tensor->numel() < selection_count ||
      topk_weights_tensor->numel() < selection_count) {
    return false;
  }

  return RunDeviceExpertSelection(
      router_logits,
      *gate_score_correction_bias_device,
      config.n_routed_experts,
      config.top_k,
      config.n_group,
      config.topk_group,
      config.routed_scaling_factor,
      config.norm_topk_prob,
      topk_ids_tensor->data(),
      topk_weights_tensor->data());
}

bool ExpertLayerSlice::Impl::SupportsDirectMoeDecodePath(
    const ExpertLayerConfig& path_config,
    std::size_t token_count,
    int device_sm_version) const {
  (void)device_sm_version;
  return direct_moe_execution_state.trace == nullptr &&
         token_count == 1 &&
         SupportsResidentPreparedMoePath(path_config, token_count) &&
         normalized_pack != nullptr &&
         normalized_pack->valid() &&
         routed_activated_pack != nullptr &&
         routed_activated_pack->valid() &&
         shared_activated_pack != nullptr &&
         shared_activated_pack->valid();
}

bool ExpertLayerSlice::Impl::SupportsDirectMoePrefillPath(
    const ExpertLayerConfig& path_config,
    std::size_t token_count,
    int device_sm_version) const {
  (void)device_sm_version;
  const bool has_trace = direct_moe_execution_state.trace != nullptr;
  const bool multi = token_count > 1;
  const bool resident = SupportsResidentPreparedMoePath(path_config, token_count);
  const bool supported = !has_trace && multi && resident;
  if (!supported && DebugComparePrefillVsLegacy() && token_count > 1) {
    std::cerr << "PREFILL_UNSUPPORTED layer=" << config.layer_index
              << " tokens=" << token_count
              << " trace=" << has_trace
              << " fused_supported=" << fused_direct_moe_supported
              << " capacity=" << ResolveMultiTokenCapacity()
              << " has_resident=" << HasResidentPreparedRoutedWeights()
              << " shared_up=" << (shared_up_nvfp4 != nullptr)
              << " shared_down=" << (shared_down_nvfp4 != nullptr)
              << "\n";
  }
  return supported;
}

bool ExpertLayerSlice::Impl::RunFusedMoePrefillPath(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& normalized,
    const DeviceTensorFp32& router_logits,
    const int* topk_ids,
    const float* topk_weights,
    DeviceTensorFp32* output) const {
  (void)cublas_handle;
  (void)heuristic_cache;
  (void)router_logits;

  if (!input.valid() ||
      !normalized.valid() ||
      topk_ids == nullptr ||
      topk_weights == nullptr ||
      output == nullptr ||
      !output->valid() ||
      input.shape() != normalized.shape() ||
      input.shape() != output->shape() ||
      input.shape().size() != 2 ||
      input.shape()[0] == 0 ||
      input.shape()[1] != config.hidden_size) {
    return false;
  }

  const std::size_t token_count = input.shape()[0];
  const std::size_t routed_expert_intermediate_size =
      ExecutionRoutedExpertIntermediateSize(config);
  if (token_count > ResolveMultiTokenCapacity()) {
    return false;
  }

  DeviceExpertRouting* routing = ResolveFusedPrefillRouting();
  DeviceMoeLaunchPlan* launch_plan = ResolveFusedPrefillLaunchPlan();
  DeviceTensorFp32* routed_output_scratch = ResolveFusedPrefillRoutedOutputScratch();
  DeviceTensorFp32* gather_scratch = ResolveFusedPrefillGatherScratch();
  DeviceNvfp4Matrix* normalized_pack = ResolveFusedPrefillNormalizedPack();
  DeviceNvfp4Matrix* shared_fc1_pack = normalized_pack;
  DeviceNvfp4Matrix* gather_pack = ResolveFusedPrefillGatherPack();
  DeviceNvfp4Matrix* shared_fc2_pack = ResolveFusedPrefillSharedUpPack();
  DeviceTensorFp32* fc2_activation_scales =
      direct_moe_execution_state.workspace != nullptr &&
              direct_moe_execution_state.workspace->fused_prefill_fc2_activation_scales != nullptr &&
              direct_moe_execution_state.workspace->fused_prefill_fc2_activation_scales->valid()
          ? direct_moe_execution_state.workspace->fused_prefill_fc2_activation_scales.get()
          : nullptr;
  DeviceTensorFp32* expert_up_scratch = ResolveFusedPrefillExpertUpScratch();
  DeviceTensorFp32* shared_up_scratch = ResolveFusedPrefillSharedUpScratch();
  const std::size_t selection_count = token_count * config.top_k;
  const auto padded_selection_count =
      DeviceMoeLaunchPlan::PaddedRowCapacity(config.n_routed_experts, selection_count);
  const auto routed_output_shape =
      routed_output_scratch != nullptr ? routed_output_scratch->shape() : std::vector<std::size_t>{};
  const auto gather_shape =
      gather_scratch != nullptr ? gather_scratch->shape() : std::vector<std::size_t>{};
  const auto expert_up_shape =
      expert_up_scratch != nullptr ? expert_up_scratch->shape() : std::vector<std::size_t>{};
  const auto fc2_activation_scale_shape =
      fc2_activation_scales != nullptr ? fc2_activation_scales->shape() : std::vector<std::size_t>{};
  const auto shared_up_shape =
      shared_up_scratch != nullptr ? shared_up_scratch->shape() : std::vector<std::size_t>{};
  if (routing == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !padded_selection_count.has_value() ||
      launch_plan->selection_count() < selection_count ||
      !routing->valid() ||
      routing->selection_count() < selection_count ||
      shared_fc1_pack == nullptr ||
      !shared_fc1_pack->valid() ||
      shared_fc1_pack->rows() < token_count ||
      shared_fc1_pack->cols() != config.hidden_size ||
      gather_pack == nullptr ||
      !gather_pack->valid() ||
      gather_pack->rows() < *padded_selection_count ||
      gather_pack->cols() != config.hidden_size ||
      routed_output_scratch == nullptr ||
      !routed_output_scratch->valid() ||
      routed_output_shape.size() != 2 ||
      routed_output_shape[0] < token_count ||
      routed_output_shape[1] != config.hidden_size ||
      gather_scratch == nullptr ||
      !gather_scratch->valid() ||
      gather_shape.size() != 2 ||
      gather_shape[0] < *padded_selection_count ||
      gather_shape[1] != config.hidden_size ||
      fc2_activation_scales == nullptr ||
      !fc2_activation_scales->valid() ||
      fc2_activation_scale_shape.size() != 2 ||
      fc2_activation_scale_shape[0] != config.n_routed_experts ||
      fc2_activation_scale_shape[1] != 1 ||
      expert_up_scratch == nullptr ||
      !expert_up_scratch->valid() ||
      expert_up_shape.size() != 2 ||
      expert_up_shape[0] < *padded_selection_count ||
      expert_up_shape[1] != routed_expert_intermediate_size ||
      shared_up_scratch == nullptr ||
      !shared_up_scratch->valid() ||
      shared_up_shape.size() != 2 ||
      shared_up_shape[0] < token_count ||
      shared_up_shape[1] != config.shared_expert_intermediate_size ||
      shared_fc2_pack == nullptr ||
      !shared_fc2_pack->valid() ||
      shared_fc2_pack->rows() < token_count ||
      shared_fc2_pack->cols() != config.shared_expert_intermediate_size) {
    return false;
  }

  if (!shared_fc1_pack->PackIntoPerRow(normalized, RuntimeMoeNvfp4PackOptions())) {
    return false;
  }

  if (!direct_moe_weights.prepared ||
      !HasValidPreparedMoeWeightView(direct_moe_weights.shared_up) ||
      !HasValidPreparedMoeWeightView(direct_moe_weights.shared_down) ||
      direct_moe_weights.routed_up_views.size() != config.n_routed_experts ||
      direct_moe_weights.routed_down_views.size() != config.n_routed_experts ||
      direct_moe_routed_up_views_device == nullptr ||
      direct_moe_routed_up_views_device->size() != config.n_routed_experts ||
      direct_moe_routed_down_views_device == nullptr ||
      direct_moe_routed_down_views_device->size() != config.n_routed_experts) {
    return false;
  }

  for (std::size_t expert_index = 0; expert_index < config.n_routed_experts; ++expert_index) {
    if (!HasValidPreparedMoeWeightView(direct_moe_weights.routed_up_views[expert_index]) ||
        !HasValidPreparedMoeWeightView(direct_moe_weights.routed_down_views[expert_index])) {
      return false;
    }
  }

  FusedMoePrefillParams params;
  params.token_count = token_count;
  params.hidden_size = config.hidden_size;
  params.routed_expert_intermediate_size = routed_expert_intermediate_size;
  params.shared_expert_intermediate_size = config.shared_expert_intermediate_size;
  params.n_routed_experts = config.n_routed_experts;
  params.top_k = config.top_k;
  params.shared_up = direct_moe_weights.shared_up;
  params.shared_down = direct_moe_weights.shared_down;
  params.routed_up_device = direct_moe_routed_up_views_device->data();
  params.routed_down_device = direct_moe_routed_down_views_device->data();
  params.selected_indices = topk_ids;
  params.selected_weights = topk_weights;
  params.input = input.data();
  params.normalized = normalized.data();
  params.normalized_pack = shared_fc1_pack;
  params.shared_fc1_pack = shared_fc1_pack;
  params.routing = routing;
  params.launch_plan = launch_plan;
  params.heuristic_cache = heuristic_cache;
  params.routed_gather_scratch = gather_scratch->data();
  params.fc1_grouped_pack = gather_pack;
  params.routed_up_scratch = expert_up_scratch->data();
  auto* expert_up_pack = ResolveFusedPrefillExpertUpPack();
  if (expert_up_pack != nullptr && expert_up_pack->valid()) {
    params.fc2_grouped_pack = expert_up_pack;
    params.gemm1_output = expert_up_pack;
  }
  params.shared_fc2_pack = shared_fc2_pack;
  params.fc2_expert_activation_scales = fc2_activation_scales->data();
  params.activation_output_scale = fc2_activation_scales->data();
  params.shared_up_scratch = shared_up_scratch->data();
  params.output = output->data();
  params.routed_output = routed_output_scratch->data();

  if (RunFusedMoePrefill(params)) {
    MaybeDumpRoutedPhase05Histogram(config, token_count, *routing, *launch_plan);
    return true;
  }
  return false;
}

bool RunMoeDirectDecodeViaCublaslt(
    const ExpertLayerConfig& config,
    const std::vector<const GemmDescriptor*>& routed_up_descriptors,
    const std::vector<const GemmDescriptor*>& routed_down_descriptors,
    const GemmDescriptor& shared_up_descriptor,
    const FusedNvfp4WeightView& shared_up_weight,
    const GemmDescriptor& shared_down_descriptor,
    const FusedNvfp4WeightView& shared_down_weight,
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& normalized,
    const float* selected_weights_device,
    const std::vector<FusedNvfp4WeightView>& routed_up_views,
    const std::vector<FusedNvfp4WeightView>& routed_down_views,
    DeviceNvfp4Matrix* normalized_pack,
    DeviceNvfp4Matrix* routed_activated_pack,
    DeviceNvfp4Matrix* shared_activated_pack,
    DeviceTensorFp32* output,
    MoeDirectDecodeScratch* scratch) {
  if (!cublas_handle.valid() ||
      !input.valid() ||
      !normalized.valid() ||
      normalized_pack == nullptr ||
      !normalized_pack->valid() ||
      routed_activated_pack == nullptr ||
      !routed_activated_pack->valid() ||
      shared_activated_pack == nullptr ||
      !shared_activated_pack->valid() ||
      output == nullptr ||
      !output->valid() ||
      input.shape() != normalized.shape() ||
      input.shape() != output->shape() ||
      input.shape().size() != 2 ||
      input.shape()[0] != 1 ||
      input.shape()[1] != config.hidden_size ||
      routed_up_descriptors.empty() ||
      routed_down_descriptors.empty() ||
      routed_up_descriptors.size() != routed_down_descriptors.size() ||
      routed_up_descriptors.front() == nullptr ||
      routed_down_descriptors.front() == nullptr ||
      !ValidFusedNvfp4WeightView(shared_up_weight) ||
      !ValidFusedNvfp4WeightView(shared_down_weight) ||
      selected_weights_device == nullptr ||
      routed_up_descriptors.size() != routed_up_views.size() ||
      routed_up_descriptors.size() != routed_down_views.size() ||
      (scratch != nullptr &&
       (scratch->output_accum == nullptr ||
        scratch->routed_up == nullptr ||
        scratch->shared_up == nullptr))) {
    return false;
  }

  const Nvfp4PackOptions pack_options = RuntimeMoeNvfp4PackOptions();
  std::unique_ptr<DeviceTensorFp32> routed_output_owned;
  std::unique_ptr<DeviceTensorFp32> routed_up_output_owned;
  std::unique_ptr<DeviceTensorFp32> shared_up_output_owned;
  DeviceTensorFp32* routed_output = nullptr;
  DeviceTensorFp32* routed_up_output = nullptr;
  DeviceTensorFp32* shared_up_output = nullptr;
  if (scratch != nullptr) {
    routed_output = scratch->output_accum;
    routed_up_output = scratch->routed_up;
    shared_up_output = scratch->shared_up;
  } else {
    const std::size_t routed_expert_intermediate_size =
        ExecutionRoutedExpertIntermediateSize(config);
    routed_output_owned = DeviceTensorFp32::Create({1, config.hidden_size});
    routed_up_output_owned =
        DeviceTensorFp32::Create({1, routed_expert_intermediate_size});
    shared_up_output_owned =
        DeviceTensorFp32::Create({1, config.shared_expert_intermediate_size});
    routed_output = routed_output_owned.get();
    routed_up_output = routed_up_output_owned.get();
    shared_up_output = shared_up_output_owned.get();
  }
  if (!normalized_pack->PackInto(normalized, pack_options) ||
      routed_output == nullptr ||
      routed_up_output == nullptr ||
      shared_up_output == nullptr ||
      !routed_output->valid() ||
      !routed_up_output->valid() ||
      !shared_up_output->valid() ||
      !routed_output->FillZero()) {
    return false;
  }

  const Nvfp4PackedMatrixDeviceView normalized_view =
      MakeNvfp4PackedMatrixDeviceView(*normalized_pack);
  const Nvfp4PackedMatrixDeviceView routed_up_weight_view =
      MakeNvfp4PackedMatrixDeviceView(routed_up_views.front());
  const Nvfp4PackedMatrixDeviceView routed_down_weight_view =
      MakeNvfp4PackedMatrixDeviceView(routed_down_views.front());
  const Nvfp4PackedMatrixDeviceView shared_up_weight_view =
      MakeNvfp4PackedMatrixDeviceView(shared_up_weight);
  const Nvfp4PackedMatrixDeviceView shared_down_weight_view =
      MakeNvfp4PackedMatrixDeviceView(shared_down_weight);
  if (!normalized_view.valid() ||
      !routed_up_weight_view.valid() ||
      !routed_down_weight_view.valid() ||
      !shared_up_weight_view.valid() ||
      !shared_down_weight_view.valid()) {
    return false;
  }

  const std::size_t rows = normalized.shape()[0];
  const auto routed_up_plan = BuildRuntimeNvfp4GemmPlan(
      *routed_up_descriptors.front(),
      routed_up_weight_view,
      rows,
      heuristic_cache);
  const auto routed_down_plan = BuildRuntimeNvfp4GemmPlan(
      *routed_down_descriptors.front(),
      routed_down_weight_view,
      rows,
      heuristic_cache);
  const auto shared_up_plan = BuildRuntimeNvfp4GemmPlan(
      shared_up_descriptor,
      shared_up_weight_view,
      rows,
      heuristic_cache);
  const auto shared_down_plan = BuildRuntimeNvfp4GemmPlan(
      shared_down_descriptor,
      shared_down_weight_view,
      rows,
      heuristic_cache);
  if (!routed_up_plan.has_value() ||
      !routed_down_plan.has_value() ||
      !shared_up_plan.has_value() ||
      !shared_down_plan.has_value()) {
    return false;
  }

  for (std::size_t slot = 0; slot < routed_up_descriptors.size(); ++slot) {
    const Nvfp4PackedMatrixDeviceView expert_up_view =
        MakeNvfp4PackedMatrixDeviceView(routed_up_views[slot]);
    const Nvfp4PackedMatrixDeviceView expert_down_view =
        MakeNvfp4PackedMatrixDeviceView(routed_down_views[slot]);
    if (!expert_up_view.valid() ||
        !expert_down_view.valid() ||
        !RunNvfp4RowMajorFp32AccumToDevice(
             cublas_handle,
             *routed_up_plan,
             normalized_view,
             normalized_pack->device_tensor_scale_ptr(),
             expert_up_view,
             expert_up_view.tensor_scale_data,
             routed_up_output)
             .has_value() ||
        !Relu2InPlaceFp32(routed_up_output)) {
      return false;
    }

    if (!routed_activated_pack->PackInto(*routed_up_output, pack_options) ||
        !RunNvfp4RowMajorFp32AccumToDevice(
             cublas_handle,
             *routed_down_plan,
             MakeNvfp4PackedMatrixDeviceView(*routed_activated_pack),
             routed_activated_pack->device_tensor_scale_ptr(),
             expert_down_view,
             expert_down_view.tensor_scale_data,
             output)
             .has_value() ||
        !AccumulateScaledFp32ByDeviceWeight(
             *output,
             selected_weights_device,
             slot,
             routed_output)) {
      return false;
    }
  }

  if (!RunNvfp4RowMajorFp32AccumToDevice(
           cublas_handle,
           *shared_up_plan,
           normalized_view,
           normalized_pack->device_tensor_scale_ptr(),
           shared_up_weight_view,
           shared_up_weight_view.tensor_scale_data,
           shared_up_output)
           .has_value() ||
      !Relu2InPlaceFp32(shared_up_output)) {
    return false;
  }

  if (!shared_activated_pack->PackInto(*shared_up_output, pack_options) ||
      !RunNvfp4RowMajorFp32AccumToDevice(
           cublas_handle,
           *shared_down_plan,
           MakeNvfp4PackedMatrixDeviceView(*shared_activated_pack),
           shared_activated_pack->device_tensor_scale_ptr(),
           shared_down_weight_view,
           shared_down_weight_view.tensor_scale_data,
           output)
           .has_value() ||
      !ResidualAddFp32(*routed_output, *output, output)) {
    return false;
  }

  return true;
}

bool ExpertLayerSlice::Impl::RunDirectMoeDecodePath(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& normalized,
    const DeviceTensorFp32& router_logits,
    const int* topk_ids,
    const float* topk_weights,
    DeviceTensorFp32* output) {
  (void)router_logits;
  if (topk_ids == nullptr || topk_weights == nullptr) {
    return false;
  }

  std::vector<int> selected_indices_host;
  if (!CopyDeviceBufferToHost(topk_ids, config.top_k, &selected_indices_host) ||
      selected_indices_host.size() != config.top_k) {
    return false;
  }

  std::vector<std::size_t> selected_expert_indices;
  selected_expert_indices.reserve(selected_indices_host.size());
  for (int selected_index : selected_indices_host) {
    if (selected_index < 0) {
      return false;
    }
    const std::size_t expert_index = static_cast<std::size_t>(selected_index);
    if (expert_index >= routed_experts.size()) {
      return false;
    }
    selected_expert_indices.push_back(expert_index);
  }

  std::vector<const GemmDescriptor*> routed_up_descriptors;
  std::vector<const GemmDescriptor*> routed_down_descriptors;
  std::vector<FusedNvfp4WeightView> routed_up_views;
  std::vector<FusedNvfp4WeightView> routed_down_views;
  routed_up_descriptors.reserve(selected_expert_indices.size());
  routed_down_descriptors.reserve(selected_expert_indices.size());
  routed_up_views.reserve(selected_expert_indices.size());
  routed_down_views.reserve(selected_expert_indices.size());
  if (!direct_moe_weights.prepared ||
      direct_moe_weights.routed_up_views.size() != routed_experts.size() ||
      direct_moe_weights.routed_down_views.size() != routed_experts.size()) {
    return false;
  }

  auto& staging_counters = GetExpertStagingCounters();
  if (monolithic_resident) {
    staging_counters.total_staging_calls.fetch_add(1, std::memory_order_relaxed);
    staging_counters.monolithic_layers.fetch_add(1, std::memory_order_relaxed);
  }
  for (std::size_t expert_index : selected_expert_indices) {
    if (expert_index >= routed_experts.size()) {
      return false;
    }
    const RoutedExpertRuntime& runtime_pair = routed_experts[expert_index];
    if (runtime_pair.up_proj == nullptr || runtime_pair.down_proj == nullptr) {
      return false;
    }
    routed_up_descriptors.push_back(runtime_pair.up_proj);
    routed_down_descriptors.push_back(runtime_pair.down_proj);
    routed_up_views.push_back(direct_moe_weights.routed_up_views[expert_index]);
    routed_down_views.push_back(direct_moe_weights.routed_down_views[expert_index]);
  }

  const FusedNvfp4WeightView shared_up_weight_view = direct_moe_weights.shared_up;
  const FusedNvfp4WeightView shared_down_weight_view = direct_moe_weights.shared_down;
  MoeDirectDecodeScratch moe_scratch;
  MoeDirectDecodeScratch* moe_scratch_ptr = nullptr;
  if (direct_moe_execution_state.use_decode_scratch) {
    moe_scratch.output_accum = routed_output_scratch.get();
    moe_scratch.routed_up = routed_up_scratch.get();
    moe_scratch.shared_up = shared_up_scratch.get();
    moe_scratch_ptr = &moe_scratch;
  }
  return RunMoeDirectDecodeViaCublaslt(
      config,
      routed_up_descriptors,
      routed_down_descriptors,
      *shared_up_nvfp4,
      shared_up_weight_view,
      *shared_down_nvfp4,
      shared_down_weight_view,
      cublas_handle,
      heuristic_cache,
      input,
      normalized,
      topk_weights,
      routed_up_views,
      routed_down_views,
      normalized_pack.get(),
      routed_activated_pack.get(),
      shared_activated_pack.get(),
      output,
      moe_scratch_ptr);
}

std::optional<ExpertLayerBindings> BuildExpertLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog,
    std::size_t routed_expert_count) {
  ExpertLayerBindings bindings;
  bindings.input_norm_weight = FindExactKernelBinding(layer, kernel_catalog, "norm.weight");
  bindings.gate_weight = FindExactGemmBinding(layer, gemm_catalog, "mixer.gate.weight");
  bindings.gate_score_correction_bias =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.gate.e_score_correction_bias");
  bindings.fc1_latent_gemm_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.fc1_latent_proj.weight");
  bindings.fc1_latent_kernel_weight =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.fc1_latent_proj.weight");
  bindings.fc1_latent_weight_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.fc1_latent_proj.weight_scale");
  bindings.fc1_latent_input_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.fc1_latent_proj.input_scale");
  bindings.fc2_latent_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.fc2_latent_proj.weight");
  bindings.shared_up_gemm_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.shared_experts.up_proj.weight");
  bindings.shared_up_kernel_weight =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.up_proj.weight");
  bindings.shared_up_weight_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.up_proj.weight_scale");
  bindings.shared_up_input_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.up_proj.input_scale");
  bindings.shared_down_gemm_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.shared_experts.down_proj.weight");
  bindings.shared_down_kernel_weight =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.down_proj.weight");
  bindings.shared_down_weight_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.down_proj.weight_scale");
  bindings.shared_down_input_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.down_proj.input_scale");

  const bool has_fc1_latent =
      bindings.fc1_latent_gemm_weight != nullptr || bindings.fc1_latent_kernel_weight != nullptr;
  const bool has_fc2_latent = bindings.fc2_latent_weight != nullptr;
  if (has_fc1_latent != has_fc2_latent) {
    return std::nullopt;
  }

  bindings.routed_experts.resize(routed_expert_count);
  for (std::size_t expert_index = 0; expert_index < routed_expert_count; ++expert_index) {
    const std::string prefix = "mixer.experts." + std::to_string(expert_index);
    bindings.routed_experts[expert_index].up_proj =
        FindExactGemmBinding(layer, gemm_catalog, prefix + ".up_proj.weight");
    bindings.routed_experts[expert_index].down_proj =
        FindExactGemmBinding(layer, gemm_catalog, prefix + ".down_proj.weight");
  }

  if (bindings.input_norm_weight == nullptr ||
      bindings.gate_weight == nullptr ||
      bindings.gate_score_correction_bias == nullptr ||
      (bindings.shared_up_gemm_weight == nullptr &&
       bindings.shared_up_kernel_weight == nullptr) ||
      (bindings.shared_down_gemm_weight == nullptr &&
       bindings.shared_down_kernel_weight == nullptr)) {
    return std::nullopt;
  }

  if (has_fc1_latent &&
      (bindings.fc1_latent_gemm_weight == nullptr &&
       bindings.fc1_latent_kernel_weight == nullptr)) {
    return std::nullopt;
  }

  for (const ExpertWeightPair& pair : bindings.routed_experts) {
    if (pair.up_proj == nullptr || pair.down_proj == nullptr) {
      return std::nullopt;
    }
  }
  return bindings;
}

std::unique_ptr<ExpertLayerSlice> ExpertLayerSlice::Create(
    const ExpertLayerConfig& config,
    const ExpertLayerBindings& bindings) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const auto debug_fail = [&](const char* reason) -> std::unique_ptr<ExpertLayerSlice> {
    if (debug) {
      std::cerr << "expert_layer_create: layer " << config.layer_index
                << " failed: " << reason << "\n";
    }
    return nullptr;
  };
  const bool has_fc1_latent =
      bindings.fc1_latent_gemm_weight != nullptr || bindings.fc1_latent_kernel_weight != nullptr;
  const bool has_fc2_latent = bindings.fc2_latent_weight != nullptr;
  const bool uses_latent_projection = has_fc1_latent && has_fc2_latent;
  const std::size_t routed_expert_intermediate_size =
      ExecutionRoutedExpertIntermediateSize(config);
  if (config.hidden_size == 0 ||
      config.routed_expert_intermediate_size == 0 ||
      !RoutedExpertIntermediateSizePaddingValid(
          config.routed_expert_intermediate_size,
          config.routed_expert_intermediate_size_padded) ||
      config.shared_expert_intermediate_size == 0 ||
      config.n_routed_experts == 0 ||
      config.top_k == 0 ||
      config.top_k > config.n_routed_experts ||
      config.max_token_count == 0 ||
      config.max_token_count > (std::numeric_limits<std::size_t>::max() / config.top_k) ||
      config.n_group == 0 ||
      config.topk_group == 0 ||
      config.n_routed_experts % config.n_group != 0 ||
      config.rms_epsilon <= 0.0f ||
      (uses_latent_projection && config.moe_latent_size == 0) ||
      bindings.input_norm_weight == nullptr ||
      bindings.gate_weight == nullptr ||
      bindings.gate_score_correction_bias == nullptr ||
      (bindings.shared_up_gemm_weight == nullptr &&
       bindings.shared_up_kernel_weight == nullptr) ||
      (bindings.shared_down_gemm_weight == nullptr &&
       bindings.shared_down_kernel_weight == nullptr) ||
      bindings.routed_experts.size() != config.n_routed_experts) {
    return debug_fail("invalid config or missing required bindings");
  }
  if (has_fc1_latent != has_fc2_latent) {
    return debug_fail("partial latent-projection bindings are unsupported");
  }

  auto input_norm_weight = UploadVectorWeightToDeviceFp32(*bindings.input_norm_weight);
  const auto gate_score_correction_bias = ReadVectorWeightToHostFp32(*bindings.gate_score_correction_bias);
  auto gate_score_correction_bias_device =
      gate_score_correction_bias.has_value()
          ? UploadHostVectorToDevice(*gate_score_correction_bias)
          : nullptr;
  auto gate_weight = UploadedLinearOp::Create(*bindings.gate_weight);
  std::unique_ptr<UploadedLinearOp> fc2_latent;
  Impl::ProjectionFamily fc1_latent_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> fc1_latent_dense;
  std::unique_ptr<ScaledFp8LinearOp> fc1_latent_scaled_fp8;
  if (uses_latent_projection) {
    fc2_latent = UploadedLinearOp::Create(*bindings.fc2_latent_weight);
    if (bindings.fc1_latent_kernel_weight != nullptr &&
        bindings.fc1_latent_weight_scale != nullptr &&
        bindings.fc1_latent_input_scale != nullptr) {
      const auto fc1_config = BuildScaledFp8LinearConfig(
          *bindings.fc1_latent_kernel_weight,
          *bindings.fc1_latent_weight_scale,
          *bindings.fc1_latent_input_scale);
      if (fc1_config.has_value()) {
        fc1_latent_scaled_fp8 = ScaledFp8LinearOp::Create(*fc1_config);
        if (!fc1_latent_scaled_fp8 || !fc1_latent_scaled_fp8->valid()) {
          return debug_fail("fc1_latent scaled-fp8 creation failed");
        }
        fc1_latent_family = Impl::ProjectionFamily::kScaledFp8;
      }
    }
    if (fc1_latent_family == Impl::ProjectionFamily::kNone) {
      fc1_latent_dense = UploadedLinearOp::Create(*bindings.fc1_latent_gemm_weight);
      if (!fc1_latent_dense || !fc1_latent_dense->valid()) {
        return debug_fail("fc1_latent dense creation failed");
      }
      fc1_latent_family = Impl::ProjectionFamily::kDense;
    }
  }

  Impl::ProjectionFamily shared_up_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_up_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_up_scaled_fp8;
  const GemmDescriptor* shared_up_nvfp4 = nullptr;
  if (bindings.shared_up_gemm_weight != nullptr &&
      bindings.shared_up_gemm_weight->kernel_family ==
          GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    shared_up_family = Impl::ProjectionFamily::kNvfp4;
    shared_up_nvfp4 = bindings.shared_up_gemm_weight;
  } else if (bindings.shared_up_kernel_weight != nullptr &&
             bindings.shared_up_weight_scale != nullptr &&
             bindings.shared_up_input_scale != nullptr) {
    const auto shared_up_config = BuildScaledFp8LinearConfig(
        *bindings.shared_up_kernel_weight,
        *bindings.shared_up_weight_scale,
        *bindings.shared_up_input_scale);
    if (shared_up_config.has_value()) {
      shared_up_scaled_fp8 = ScaledFp8LinearOp::Create(*shared_up_config);
      if (!shared_up_scaled_fp8 || !shared_up_scaled_fp8->valid()) {
        return debug_fail("shared_up scaled-fp8 creation failed");
      }
      shared_up_family = Impl::ProjectionFamily::kScaledFp8;
    }
  }
  if (shared_up_family == Impl::ProjectionFamily::kNone) {
    shared_up_dense = UploadedLinearOp::Create(*bindings.shared_up_gemm_weight);
    if (!shared_up_dense || !shared_up_dense->valid()) {
      return debug_fail("shared_up dense creation failed");
    }
    shared_up_family = Impl::ProjectionFamily::kDense;
  }

  Impl::SharedDownFamily shared_down_family = Impl::SharedDownFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_down_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_down_scaled_fp8;
  const GemmDescriptor* shared_down_nvfp4 = nullptr;
  if (bindings.shared_down_gemm_weight != nullptr &&
      bindings.shared_down_gemm_weight->kernel_family ==
          GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    shared_down_family = Impl::SharedDownFamily::kNvfp4;
    shared_down_nvfp4 = bindings.shared_down_gemm_weight;
  } else if (bindings.shared_down_kernel_weight != nullptr &&
             bindings.shared_down_weight_scale != nullptr &&
             bindings.shared_down_input_scale != nullptr &&
             bindings.shared_down_kernel_weight->storage_dtype == "fp8_e4m3fn") {
    const auto shared_down_config = BuildScaledFp8LinearConfig(
        *bindings.shared_down_kernel_weight,
        *bindings.shared_down_weight_scale,
        *bindings.shared_down_input_scale);
    shared_down_scaled_fp8 =
        shared_down_config.has_value() ? ScaledFp8LinearOp::Create(*shared_down_config) : nullptr;
    if (!shared_down_scaled_fp8 || !shared_down_scaled_fp8->valid()) {
      return debug_fail("shared_down scaled-fp8 creation failed");
    }
    shared_down_family = Impl::SharedDownFamily::kScaledFp8;
  } else if (bindings.shared_down_gemm_weight != nullptr) {
    shared_down_dense = UploadedLinearOp::Create(*bindings.shared_down_gemm_weight);
    if (!shared_down_dense || !shared_down_dense->valid()) {
      return debug_fail("shared_down dense creation failed");
    }
    shared_down_family = Impl::SharedDownFamily::kDense;
  } else {
    return debug_fail("shared_down family resolution failed");
  }

  if (!input_norm_weight ||
      !gate_score_correction_bias.has_value() ||
      gate_score_correction_bias->size() != config.n_routed_experts ||
      !gate_score_correction_bias_device ||
      !gate_weight || !gate_weight->valid() ||
      (uses_latent_projection && (!fc2_latent || !fc2_latent->valid())) ||
      (uses_latent_projection && fc1_latent_family == Impl::ProjectionFamily::kNone) ||
      (shared_up_family == Impl::ProjectionFamily::kNone)) {
    return debug_fail("core weight materialization failed");
  }

  const std::size_t fc1_output_rows =
      fc1_latent_family == Impl::ProjectionFamily::kScaledFp8
          ? fc1_latent_scaled_fp8->output_rows()
          : (fc1_latent_family == Impl::ProjectionFamily::kDense ? fc1_latent_dense->output_rows() : 0);
  const std::size_t fc1_input_cols =
      fc1_latent_family == Impl::ProjectionFamily::kScaledFp8
          ? fc1_latent_scaled_fp8->input_cols()
          : (fc1_latent_family == Impl::ProjectionFamily::kDense ? fc1_latent_dense->input_cols() : 0);
  const std::size_t shared_up_output_rows =
      shared_up_family == Impl::ProjectionFamily::kNvfp4
          ? shared_up_nvfp4->output_rows
          : (
      shared_up_family == Impl::ProjectionFamily::kScaledFp8
          ? shared_up_scaled_fp8->output_rows()
          : shared_up_dense->output_rows());
  const std::size_t shared_up_input_cols =
      shared_up_family == Impl::ProjectionFamily::kNvfp4
          ? shared_up_nvfp4->input_cols
          : (
      shared_up_family == Impl::ProjectionFamily::kScaledFp8
          ? shared_up_scaled_fp8->input_cols()
          : shared_up_dense->input_cols());

  if (bindings.gate_weight->output_rows != config.n_routed_experts ||
      bindings.gate_weight->input_cols != config.hidden_size ||
      shared_up_output_rows != config.shared_expert_intermediate_size ||
      shared_up_input_cols != config.hidden_size) {
    return debug_fail("core shape validation failed");
  }
  if (uses_latent_projection &&
      (fc1_output_rows != config.moe_latent_size ||
       fc1_input_cols != config.hidden_size ||
       bindings.fc2_latent_weight->output_rows != config.hidden_size ||
       bindings.fc2_latent_weight->input_cols != config.moe_latent_size)) {
    return debug_fail("latent-projection shape validation failed");
  }

  if (shared_down_family == Impl::SharedDownFamily::kNvfp4) {
    if (shared_down_nvfp4 == nullptr ||
        shared_down_nvfp4->output_rows != config.hidden_size ||
        shared_down_nvfp4->input_cols != config.shared_expert_intermediate_size) {
      return debug_fail("shared_down NVFP4 shape validation failed");
    }
  } else if (shared_down_family == Impl::SharedDownFamily::kScaledFp8) {
    if (!shared_down_scaled_fp8 ||
        shared_down_scaled_fp8->output_rows() != config.hidden_size ||
        shared_down_scaled_fp8->input_cols() != config.shared_expert_intermediate_size) {
      return debug_fail("shared_down scaled-fp8 shape validation failed");
    }
  } else if (shared_down_family == Impl::SharedDownFamily::kDense) {
    if (!shared_down_dense ||
        shared_down_dense->output_rows() != config.hidden_size ||
        shared_down_dense->input_cols() != config.shared_expert_intermediate_size) {
      return debug_fail("shared_down dense shape validation failed");
    }
  } else {
    return debug_fail("shared_down family missing after validation");
  }

  for (const ExpertWeightPair& pair : bindings.routed_experts) {
    if (pair.up_proj == nullptr || pair.down_proj == nullptr) {
      continue;
    }
    if (pair.up_proj->output_rows != config.routed_expert_intermediate_size ||
        pair.down_proj->input_cols != config.routed_expert_intermediate_size) {
      return debug_fail("routed expert shape validation failed");
    }
    if (uses_latent_projection &&
        (pair.up_proj->input_cols != config.moe_latent_size ||
         pair.down_proj->output_rows != config.moe_latent_size)) {
      return debug_fail("routed expert latent shape validation failed");
    }
    if (!uses_latent_projection &&
        (pair.up_proj->input_cols != config.hidden_size ||
         pair.down_proj->output_rows != config.hidden_size)) {
      return debug_fail("routed expert shape validation failed");
    }
  }

  std::unique_ptr<DeviceNvfp4Weight> shared_up_nvfp4_device;
  if (shared_up_family == Impl::ProjectionFamily::kNvfp4) {
    shared_up_nvfp4_device = DeviceNvfp4Weight::Upload(*shared_up_nvfp4);
    if (!shared_up_nvfp4_device || !shared_up_nvfp4_device->valid()) {
      return debug_fail("shared_up NVFP4 upload failed");
    }
  }

  std::unique_ptr<DeviceNvfp4Weight> shared_down_nvfp4_device;
  if (shared_down_family == Impl::SharedDownFamily::kNvfp4) {
    shared_down_nvfp4_device = DeviceNvfp4Weight::Upload(*shared_down_nvfp4);
    if (!shared_down_nvfp4_device || !shared_down_nvfp4_device->valid()) {
      return debug_fail("shared_down NVFP4 upload failed");
    }
  }
  const std::uint64_t resident_shared_expert_bytes =
      (shared_up_nvfp4_device ? TotalUploadedBytes(*shared_up_nvfp4_device) : 0) +
      (shared_down_nvfp4_device ? TotalUploadedBytes(*shared_down_nvfp4_device) : 0);

  std::vector<Impl::RoutedExpertRuntime> routed_experts(config.n_routed_experts);
  bool fused_direct_moe_supported =
      !uses_latent_projection &&
      shared_up_family == Impl::ProjectionFamily::kNvfp4 &&
      shared_down_family == Impl::SharedDownFamily::kNvfp4;
  for (std::size_t expert_index = 0; expert_index < bindings.routed_experts.size(); ++expert_index) {
    const ExpertWeightPair& pair = bindings.routed_experts[expert_index];
    Impl::RoutedExpertRuntime runtime_pair;
    runtime_pair.up_proj = pair.up_proj;
    runtime_pair.down_proj = pair.down_proj;
    if (!uses_latent_projection &&
        pair.up_proj != nullptr &&
        pair.down_proj != nullptr &&
        pair.up_proj->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled &&
        pair.down_proj->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    } else {
      fused_direct_moe_supported = false;
    }
    routed_experts[expert_index] = std::move(runtime_pair);
  }

  const bool direct_moe_layer = !uses_latent_projection;
  std::uint64_t resident_routed_expert_bytes = 0;
  std::unique_ptr<MonolithicNvfp4ExpertWeights> monolithic_up;
  std::unique_ptr<MonolithicNvfp4ExpertWeights> monolithic_down;
  bool monolithic_resident = false;
  std::uint64_t monolithic_up_bytes = 0;
  std::uint64_t monolithic_down_bytes = 0;
  std::size_t fully_resident_layer_count = 0;
  std::size_t nonresident_layer_count = 0;
  std::size_t uploaded_routed_experts = 0;
  std::string residency_reason = direct_moe_layer ? "monolithic" : "n/a";
  std::string monolithic_detail;
  std::uint64_t cumulative_resident_routed_expert_bytes = 0;
  std::uint64_t estimated_routed_layer_bytes = 0;
  for (const Impl::RoutedExpertRuntime& runtime_pair : routed_experts) {
    if (runtime_pair.up_proj != nullptr) {
      estimated_routed_layer_bytes += EstimatedUploadedBytes(*runtime_pair.up_proj);
    }
    if (runtime_pair.down_proj != nullptr) {
      estimated_routed_layer_bytes += EstimatedUploadedBytes(*runtime_pair.down_proj);
    }
  }
  if (direct_moe_layer) {
    if (!fused_direct_moe_supported) {
      return debug_fail("direct MoE requires resident NVFP4 weights");
    }
    try {
      monolithic_up = MonolithicNvfp4ExpertWeights::Create(
          config.n_routed_experts,
          routed_expert_intermediate_size,
          config.hidden_size,
          false);
      monolithic_down = MonolithicNvfp4ExpertWeights::Create(
          config.n_routed_experts,
          config.hidden_size,
          routed_expert_intermediate_size,
          false);
      if (!monolithic_up || !monolithic_up->valid() ||
          !monolithic_down || !monolithic_down->valid()) {
        monolithic_detail = "allocation failed";
        return debug_fail("monolithic expert residency allocation failed");
      }
      for (std::size_t expert_index = 0; expert_index < routed_experts.size(); ++expert_index) {
        const Impl::RoutedExpertRuntime& runtime_pair = routed_experts[expert_index];
        if (runtime_pair.up_proj == nullptr || runtime_pair.down_proj == nullptr ||
            !UploadDescriptorToMonolithic(
                monolithic_up.get(), expert_index, *runtime_pair.up_proj) ||
            !UploadDescriptorToMonolithic(
                monolithic_down.get(), expert_index, *runtime_pair.down_proj)) {
          monolithic_detail = "expert upload failed";
          return debug_fail("monolithic expert residency upload failed");
        }
      }

      std::vector<FusedNvfp4WeightView> monolithic_up_views = monolithic_up->BuildAllViews();
      std::vector<FusedNvfp4WeightView> monolithic_down_views = monolithic_down->BuildAllViews();
      if (monolithic_up_views.size() != routed_experts.size() ||
          monolithic_down_views.size() != routed_experts.size()) {
        monolithic_detail = "view build failed";
        return debug_fail("monolithic expert residency view build failed");
      }

      monolithic_up_bytes = static_cast<std::uint64_t>(monolithic_up->total_bytes());
      monolithic_down_bytes = static_cast<std::uint64_t>(monolithic_down->total_bytes());
      resident_routed_expert_bytes = monolithic_up_bytes + monolithic_down_bytes;
      uploaded_routed_experts = routed_experts.size();
      monolithic_resident = true;
      fully_resident_layer_count = 1;
      cumulative_resident_routed_expert_bytes = resident_routed_expert_bytes;
    } catch (const std::exception& error) {
      monolithic_detail = error.what();
      return debug_fail("monolithic expert residency threw");
    } catch (...) {
      monolithic_detail = "unknown";
      return debug_fail("monolithic expert residency threw");
    }
  }

  auto normalized_bf16_scratch = DeviceTensorBf16::Create({1, config.hidden_size});
  auto normalized_scratch = DeviceTensorFp32::Create({1, config.hidden_size});
  auto router_logits_scratch = DeviceTensorFp32::Create({1, config.n_routed_experts});
  auto output_scratch = DeviceTensorFp32::Create({1, config.hidden_size});
  auto routed_output_scratch = DeviceTensorFp32::Create({1, config.hidden_size});
  auto routed_up_scratch = DeviceTensorFp32::Create({1, routed_expert_intermediate_size});
  auto shared_up_scratch = DeviceTensorFp32::Create({1, config.shared_expert_intermediate_size});
  if (!normalized_bf16_scratch ||
      !normalized_scratch ||
      !router_logits_scratch ||
      !output_scratch ||
      !routed_output_scratch ||
      !routed_up_scratch ||
      !shared_up_scratch) {
    return debug_fail("decode scratch allocation failed");
  }

  std::unique_ptr<DeviceNvfp4Matrix> normalized_pack;
  std::unique_ptr<DeviceNvfp4Matrix> routed_activated_pack;
  std::unique_ptr<DeviceNvfp4Matrix> shared_activated_pack;
  if (fused_direct_moe_supported) {
    const Nvfp4ScaleLayout pack_scale_layout =
        ResolveActivationNvfp4ScaleLayout(1, RuntimeMoeNvfp4PackOptions().execution_scale_layout);
    normalized_pack = DeviceNvfp4Matrix::Create(1, config.hidden_size, pack_scale_layout);
    routed_activated_pack = DeviceNvfp4Matrix::Create(
        1,
        routed_expert_intermediate_size,
        pack_scale_layout);
    shared_activated_pack = DeviceNvfp4Matrix::Create(
        1,
        config.shared_expert_intermediate_size,
        pack_scale_layout);
    if (!normalized_pack ||
        !normalized_pack->valid() ||
        !routed_activated_pack ||
        !routed_activated_pack->valid() ||
        !shared_activated_pack ||
        !shared_activated_pack->valid()) {
      return debug_fail("NVFP4 activation pack buffer allocation failed");
    }
  }

  std::unique_ptr<DeviceTensorInt32> device_topk_ids;
  std::unique_ptr<DeviceTensorFp32> device_topk_weights;
  std::unique_ptr<DeviceExpertRouting> fused_prefill_routing;
  std::unique_ptr<DeviceMoeLaunchPlan> fused_prefill_launch_plan;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_routed_output_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_gather_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_expert_up_scratch;
  std::unique_ptr<DeviceTensorFp32> fused_prefill_shared_up_scratch;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_normalized_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_gather_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_expert_up_pack;
  std::unique_ptr<DeviceNvfp4Matrix> fused_prefill_shared_up_pack;
  const bool resident_fastpath_ready = monolithic_resident;
  if (fused_direct_moe_supported && resident_fastpath_ready) {
    device_topk_ids = DeviceTensorInt32::Create({config.max_token_count, config.top_k});
    device_topk_weights = DeviceTensorFp32::Create({config.max_token_count, config.top_k});
    if (!device_topk_ids || !device_topk_weights) {
      return debug_fail("device top-k scratch allocation failed");
    }
  }
  (void)fused_prefill_routing;
  (void)fused_prefill_launch_plan;
  (void)fused_prefill_routed_output_scratch;
  (void)fused_prefill_gather_scratch;
  (void)fused_prefill_expert_up_scratch;
  (void)fused_prefill_shared_up_scratch;
  (void)fused_prefill_normalized_pack;
  (void)fused_prefill_gather_pack;
  (void)fused_prefill_expert_up_pack;
  (void)fused_prefill_shared_up_pack;

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->topology =
      uses_latent_projection ? Impl::ExpertTopology::kLatentProjection : Impl::ExpertTopology::kDirectMlp;
  impl->input_norm_weight = std::move(input_norm_weight);
  impl->gate_score_correction_bias = *gate_score_correction_bias;
  impl->gate_score_correction_bias_device = std::move(gate_score_correction_bias_device);
  impl->gate_weight = std::move(gate_weight);
  impl->fc1_latent_family = fc1_latent_family;
  impl->fc1_latent_dense = std::move(fc1_latent_dense);
  impl->fc1_latent_scaled_fp8 = std::move(fc1_latent_scaled_fp8);
  impl->fc2_latent = std::move(fc2_latent);
  impl->shared_up_family = shared_up_family;
  impl->shared_up_dense = std::move(shared_up_dense);
  impl->shared_up_scaled_fp8 = std::move(shared_up_scaled_fp8);
  impl->shared_up_nvfp4 = shared_up_nvfp4;
  impl->shared_up_nvfp4_device = std::move(shared_up_nvfp4_device);
  impl->shared_down_family = shared_down_family;
  impl->shared_down_dense = std::move(shared_down_dense);
  impl->shared_down_scaled_fp8 = std::move(shared_down_scaled_fp8);
  impl->shared_down_nvfp4 = shared_down_nvfp4;
  impl->shared_down_nvfp4_device = std::move(shared_down_nvfp4_device);
  impl->routed_experts = std::move(routed_experts);
  impl->monolithic_up = std::move(monolithic_up);
  impl->monolithic_down = std::move(monolithic_down);
  impl->device_topk_ids = std::move(device_topk_ids);
  impl->device_topk_weights = std::move(device_topk_weights);
  impl->fused_prefill_routing = std::move(fused_prefill_routing);
  impl->fused_prefill_launch_plan = std::move(fused_prefill_launch_plan);
  impl->fused_prefill_routed_output_scratch = std::move(fused_prefill_routed_output_scratch);
  impl->fused_prefill_gather_scratch = std::move(fused_prefill_gather_scratch);
  impl->fused_prefill_expert_up_scratch = std::move(fused_prefill_expert_up_scratch);
  impl->fused_prefill_shared_up_scratch = std::move(fused_prefill_shared_up_scratch);
  impl->fused_prefill_normalized_pack = std::move(fused_prefill_normalized_pack);
  impl->fused_prefill_gather_pack = std::move(fused_prefill_gather_pack);
  impl->fused_prefill_expert_up_pack = std::move(fused_prefill_expert_up_pack);
  impl->fused_prefill_shared_up_pack = std::move(fused_prefill_shared_up_pack);
  impl->normalized_bf16_scratch = std::move(normalized_bf16_scratch);
  impl->normalized_scratch = std::move(normalized_scratch);
  impl->router_logits_scratch = std::move(router_logits_scratch);
  impl->output_scratch = std::move(output_scratch);
  impl->routed_output_scratch = std::move(routed_output_scratch);
  impl->routed_up_scratch = std::move(routed_up_scratch);
  impl->shared_up_scratch = std::move(shared_up_scratch);
  impl->normalized_pack = std::move(normalized_pack);
  impl->routed_activated_pack = std::move(routed_activated_pack);
  impl->shared_activated_pack = std::move(shared_activated_pack);
  impl->resident_shared_expert_bytes = resident_shared_expert_bytes;
  impl->resident_routed_expert_bytes = resident_routed_expert_bytes;
  impl->monolithic_resident = monolithic_resident;
  impl->fused_direct_moe_supported = fused_direct_moe_supported;
  std::vector<FusedNvfp4WeightView> resident_routed_up_views;
  std::vector<FusedNvfp4WeightView> resident_routed_down_views;
  if (impl->monolithic_resident &&
      !Impl::BuildResidentRoutedWeightViews(
          *impl,
          &resident_routed_up_views,
          &resident_routed_down_views)) {
    return debug_fail("resident routed weight view build failed");
  }
  FusedNvfp4WeightView shared_up_view;
  if (impl->shared_up_nvfp4_device != nullptr && impl->shared_up_nvfp4_device->valid()) {
    shared_up_view = MakeFusedNvfp4WeightView(*impl->shared_up_nvfp4_device);
  }
  FusedNvfp4WeightView shared_down_view;
  if (impl->shared_down_nvfp4_device != nullptr && impl->shared_down_nvfp4_device->valid()) {
    shared_down_view = MakeFusedNvfp4WeightView(*impl->shared_down_nvfp4_device);
  }
  if (fused_direct_moe_supported) {
    auto resident_routed_up_views_device =
        DeviceArray<FusedNvfp4WeightView>::CopyFromHost(resident_routed_up_views);
    auto resident_routed_down_views_device =
        DeviceArray<FusedNvfp4WeightView>::CopyFromHost(resident_routed_down_views);
    if (!HasValidPreparedMoeWeightView(shared_up_view) ||
        !HasValidPreparedMoeWeightView(shared_down_view) ||
        resident_routed_up_views.size() != config.n_routed_experts ||
        resident_routed_down_views.size() != config.n_routed_experts ||
        resident_routed_up_views_device == nullptr ||
        resident_routed_down_views_device == nullptr) {
      return debug_fail("resident direct MoE weight preparation failed");
    }
    for (std::size_t expert_index = 0; expert_index < config.n_routed_experts; ++expert_index) {
      if (!HasValidPreparedMoeWeightView(resident_routed_up_views[expert_index]) ||
          !HasValidPreparedMoeWeightView(resident_routed_down_views[expert_index])) {
        return debug_fail("resident direct MoE expert weight preparation failed");
      }
    }
    impl->direct_moe_weights.prepared = true;
    impl->direct_moe_weights.shared_up = shared_up_view;
    impl->direct_moe_weights.shared_down = shared_down_view;
    impl->direct_moe_weights.routed_up_views = resident_routed_up_views;
    impl->direct_moe_weights.routed_down_views = resident_routed_down_views;
    impl->direct_moe_routed_up_views_device = std::move(resident_routed_up_views_device);
    impl->direct_moe_routed_down_views_device = std::move(resident_routed_down_views_device);
  }
  if (debug) {
    const std::uint64_t resident_total_expert_bytes =
        resident_shared_expert_bytes + resident_routed_expert_bytes;
    std::cout << "expert_layer_create: layer=" << config.layer_index
              << " monolithic=" << (monolithic_resident ? "true" : "false")
              << " up_bytes=" << monolithic_up_bytes
              << " down_bytes=" << monolithic_down_bytes
              << " total_mib="
              << (static_cast<double>(monolithic_up_bytes + monolithic_down_bytes) / (1024.0 * 1024.0))
              << "\n";
    std::cout << "expert_layer_create: layer=" << config.layer_index
              << " fused_direct_moe_supported=" << fused_direct_moe_supported
              << " expert_full_residency=" << monolithic_resident
              << " expert_residency_reason=" << residency_reason
              << " uploaded_routed_experts=" << uploaded_routed_experts
              << " estimated_routed_layer_bytes=" << estimated_routed_layer_bytes
              << " cumulative_resident_routed_expert_bytes="
              << cumulative_resident_routed_expert_bytes
              << " resident_layer_count=" << fully_resident_layer_count
              << " nonresident_layer_count=" << nonresident_layer_count
              << " resident_routed_expert_bytes=" << resident_routed_expert_bytes
              << " resident_shared_expert_bytes=" << resident_shared_expert_bytes
              << " resident_total_expert_bytes=" << resident_total_expert_bytes
              << " resident_total_expert_mib="
              << (static_cast<double>(resident_total_expert_bytes) / (1024.0 * 1024.0))
              << "\n";
    if (!monolithic_detail.empty()) {
      std::cout << "expert_layer_create: layer=" << config.layer_index
                << " monolithic_detail=" << monolithic_detail << "\n";
    }
  }
  return std::unique_ptr<ExpertLayerSlice>(new ExpertLayerSlice(std::move(impl)));
}

ExpertLayerSlice::ExpertLayerSlice(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExpertLayerSlice::ExpertLayerSlice(ExpertLayerSlice&&) noexcept = default;
ExpertLayerSlice& ExpertLayerSlice::operator=(ExpertLayerSlice&&) noexcept = default;
ExpertLayerSlice::~ExpertLayerSlice() = default;

bool ExpertLayerSlice::valid() const {
  return impl_ != nullptr &&
         impl_->input_norm_weight != nullptr &&
         impl_->input_norm_weight->valid() &&
         !impl_->gate_score_correction_bias.empty() &&
         impl_->gate_score_correction_bias_device != nullptr &&
         impl_->gate_score_correction_bias_device->valid() &&
         impl_->gate_weight != nullptr &&
         impl_->gate_weight->valid() &&
         ((impl_->topology == Impl::ExpertTopology::kDirectMlp) ||
          ((impl_->fc1_latent_family == Impl::ProjectionFamily::kDense &&
            impl_->fc1_latent_dense != nullptr &&
            impl_->fc1_latent_dense->valid()) ||
           (impl_->fc1_latent_family == Impl::ProjectionFamily::kScaledFp8 &&
            impl_->fc1_latent_scaled_fp8 != nullptr &&
            impl_->fc1_latent_scaled_fp8->valid())) &&
          impl_->fc2_latent != nullptr &&
          impl_->fc2_latent->valid()) &&
         ((impl_->shared_up_family == Impl::ProjectionFamily::kDense &&
           impl_->shared_up_dense != nullptr &&
           impl_->shared_up_dense->valid()) ||
          (impl_->shared_up_family == Impl::ProjectionFamily::kScaledFp8 &&
           impl_->shared_up_scaled_fp8 != nullptr &&
           impl_->shared_up_scaled_fp8->valid()) ||
          (impl_->shared_up_family == Impl::ProjectionFamily::kNvfp4 &&
           impl_->shared_up_nvfp4 != nullptr)) &&
         ((impl_->shared_down_family == Impl::SharedDownFamily::kNvfp4 &&
           impl_->shared_down_nvfp4 != nullptr) ||
          (impl_->shared_down_family == Impl::SharedDownFamily::kScaledFp8 &&
           impl_->shared_down_scaled_fp8 != nullptr &&
           impl_->shared_down_scaled_fp8->valid()) ||
          (impl_->shared_down_family == Impl::SharedDownFamily::kDense &&
           impl_->shared_down_dense != nullptr &&
           impl_->shared_down_dense->valid())) &&
         (!impl_->monolithic_resident ||
          (impl_->monolithic_up != nullptr &&
           impl_->monolithic_up->valid() &&
           impl_->monolithic_down != nullptr &&
           impl_->monolithic_down->valid())) &&
         (!(impl_->fused_direct_moe_supported &&
            impl_->monolithic_resident) ||
          (impl_->device_topk_ids != nullptr &&
           impl_->device_topk_ids->valid() &&
           impl_->device_topk_weights != nullptr &&
           impl_->device_topk_weights->valid())) &&
         impl_->normalized_bf16_scratch != nullptr &&
         impl_->normalized_bf16_scratch->valid() &&
         impl_->normalized_scratch != nullptr &&
         impl_->normalized_scratch->valid() &&
         impl_->router_logits_scratch != nullptr &&
         impl_->router_logits_scratch->valid() &&
         impl_->output_scratch != nullptr &&
         impl_->output_scratch->valid() &&
         impl_->routed_output_scratch != nullptr &&
         impl_->routed_output_scratch->valid() &&
         impl_->routed_up_scratch != nullptr &&
         impl_->routed_up_scratch->valid() &&
         impl_->shared_up_scratch != nullptr &&
         impl_->shared_up_scratch->valid() &&
         (!impl_->fused_direct_moe_supported ||
          (impl_->normalized_pack != nullptr &&
           impl_->normalized_pack->valid() &&
           impl_->routed_activated_pack != nullptr &&
           impl_->routed_activated_pack->valid() &&
           impl_->shared_activated_pack != nullptr &&
           impl_->shared_activated_pack->valid())) &&
         impl_->routed_experts.size() == impl_->config.n_routed_experts;
}

const ExpertLayerConfig& ExpertLayerSlice::config() const {
  return impl_->config;
}

bool ExpertLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorBf16& input,
    DeviceTensorBf16* residual,
    DeviceTensorBf16* output,
    ExpertLayerRunTrace* trace,
    MoePrefillWorkspace* workspace) const {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (!valid() ||
      !cublas_handle.valid() ||
      !input.valid() ||
      input.shape().size() != 2 ||
      input.shape()[1] != impl_->config.hidden_size ||
      residual == nullptr ||
      !residual->valid() ||
      residual->shape() != input.shape() ||
      output == nullptr ||
      !output->valid() ||
      output->shape() != input.shape()) {
    if (debug) {
      std::cout << "expert_layer: invalid run inputs or state\n";
    }
    return false;
  }

  const std::size_t token_count = input.shape()[0];
  if (token_count == 0 || (trace != nullptr && token_count != 1)) {
    if (debug) {
      std::cout << "expert_layer: unsupported token count/tracing combination\n";
    }
    return false;
  }

  std::unique_ptr<DeviceTensorBf16> normalized_bf16_owned;
  std::unique_ptr<DeviceTensorBf16> normalized_bf16_workspace_view;
  std::unique_ptr<DeviceTensorFp32> input_fp32_owned;
  std::unique_ptr<DeviceTensorFp32> input_fp32_workspace_view;
  std::unique_ptr<DeviceTensorFp32> normalized_owned;
  std::unique_ptr<DeviceTensorFp32> normalized_workspace_view;
  std::unique_ptr<DeviceTensorFp32> router_logits_owned;
  std::unique_ptr<DeviceTensorFp32> router_logits_workspace_view;
  std::unique_ptr<DeviceTensorFp32> output_fp32_owned;
  std::unique_ptr<DeviceTensorFp32> output_fp32_workspace_view;
  DeviceTensorBf16* normalized_bf16 = nullptr;
  DeviceTensorFp32* input_fp32 = nullptr;
  DeviceTensorFp32* normalized = nullptr;
  DeviceTensorFp32* router_logits = nullptr;
  DeviceTensorFp32* output_fp32 = nullptr;
  const bool use_decode_scratch = token_count == 1;
  bool use_request_workspace =
      token_count > 1 &&
      workspace != nullptr &&
      WorkspaceMatchesConfig(*workspace, impl_->config) &&
      workspace->token_capacity() >= token_count;
  if (use_decode_scratch) {
    normalized_bf16 = impl_->normalized_bf16_scratch.get();
    normalized = impl_->normalized_scratch.get();
    router_logits = impl_->router_logits_scratch.get();
    output_fp32 = impl_->output_scratch.get();
  } else if (use_request_workspace) {
    normalized_bf16_workspace_view = CreateWorkspaceView(
        workspace->normalized_bf16.get(),
        token_count,
        impl_->config.hidden_size);
    input_fp32_workspace_view = CreateWorkspaceView(
        workspace->input_fp32.get(),
        token_count,
        impl_->config.hidden_size);
    normalized_workspace_view = CreateWorkspaceView(
        workspace->normalized.get(),
        token_count,
        impl_->config.hidden_size);
    router_logits_workspace_view = CreateWorkspaceView(
        workspace->router_logits.get(),
        token_count,
        impl_->config.n_routed_experts);
    output_fp32_workspace_view = CreateWorkspaceView(
        workspace->output_fp32.get(),
        token_count,
        impl_->config.hidden_size);
    if (!normalized_bf16_workspace_view ||
        !input_fp32_workspace_view ||
        !normalized_workspace_view ||
        !router_logits_workspace_view ||
        !output_fp32_workspace_view) {
      use_request_workspace = false;
    } else {
      normalized_bf16 = normalized_bf16_workspace_view.get();
      input_fp32 = input_fp32_workspace_view.get();
      normalized = normalized_workspace_view.get();
      router_logits = router_logits_workspace_view.get();
      output_fp32 = output_fp32_workspace_view.get();
    }
  }
  if (!use_decode_scratch && !use_request_workspace) {
    normalized_bf16_owned = DeviceTensorBf16::Create({token_count, impl_->config.hidden_size});
    input_fp32_owned = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    normalized_owned = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    router_logits_owned = DeviceTensorFp32::Create({token_count, impl_->config.n_routed_experts});
    output_fp32_owned = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    normalized_bf16 = normalized_bf16_owned.get();
    input_fp32 = input_fp32_owned.get();
    normalized = normalized_owned.get();
    router_logits = router_logits_owned.get();
    output_fp32 = output_fp32_owned.get();
  }
  if (input_fp32 == nullptr) {
    input_fp32_owned = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    input_fp32 = input_fp32_owned.get();
  }
  if (normalized_bf16 == nullptr ||
      input_fp32 == nullptr ||
      normalized == nullptr ||
      router_logits == nullptr ||
      output_fp32 == nullptr) {
    if (debug) {
      std::cout << "expert_layer: scratch allocation failed\n";
    }
    return false;
  }

  const bool norm_ok =
      CastTensorBf16ToFp32(input, input_fp32) &&
      FusedAddRmsNormBf16(
          input,
          residual,
          *impl_->input_norm_weight,
          impl_->config.rms_epsilon,
          normalized_bf16) &&
      CastTensorBf16ToFp32(*normalized_bf16, normalized);
  const bool gate_ok =
      norm_ok &&
      impl_->gate_weight->Run(cublas_handle, heuristic_cache, *normalized, router_logits);
  if (!norm_ok || !gate_ok) {
    if (debug) {
      std::cout << "expert_layer: norm or gate projection failed"
                << " norm_ok=" << norm_ok
                << " gate_ok=" << gate_ok << "\n";
    }
    return false;
  }

  if (impl_->topology == Impl::ExpertTopology::kDirectMlp) {
    std::unique_ptr<MoePrefillWorkspace> fallback_moe_prefill_workspace;
    MoePrefillWorkspace* direct_moe_workspace = use_request_workspace ? workspace : nullptr;
    if (token_count > 1 && direct_moe_workspace == nullptr) {
      fallback_moe_prefill_workspace = MoePrefillWorkspace::Create(
          token_count,
          BuildMoePrefillWorkspaceConfig(impl_->config));
      direct_moe_workspace = fallback_moe_prefill_workspace.get();
      if (direct_moe_workspace == nullptr || !direct_moe_workspace->valid()) {
        if (debug) {
          std::cout << "expert_layer: direct MoE fallback workspace allocation failed\n";
        }
        return false;
      }
    }
    const int device_sm_version = GetCurrentDeviceSmVersion();
    impl_->ResetDirectMoeExecutionState(
        trace,
        use_decode_scratch,
        direct_moe_workspace);
    const bool supports_path =
        token_count == 1
            ? impl_->SupportsDirectMoeDecodePath(impl_->config, token_count, device_sm_version)
            : impl_->SupportsDirectMoePrefillPath(impl_->config, token_count, device_sm_version);
    if (!supports_path) {
      if (debug) {
        std::cout << "expert_layer: no resident direct MoE path available\n";
      }
      return false;
    }
    if (!impl_->RunDirectMoeExpertSelection(*router_logits, token_count)) {
      if (debug) {
        std::cout << "expert_layer: device expert selection failed\n";
      }
      return false;
    }

    DeviceTensorInt32* topk_ids_tensor = impl_->ResolveTopkIdsTensor();
    DeviceTensorFp32* topk_weights_tensor = impl_->ResolveTopkWeightsTensor();
    const int* topk_ids = topk_ids_tensor != nullptr ? topk_ids_tensor->data() : nullptr;
    const float* topk_weights =
        topk_weights_tensor != nullptr ? topk_weights_tensor->data() : nullptr;
    const bool ok =
        token_count == 1
            ? impl_->RunDirectMoeDecodePath(
                  cublas_handle,
                  heuristic_cache,
                  *input_fp32,
                  *normalized,
                  *router_logits,
                  topk_ids,
                  topk_weights,
                  output_fp32)
            : [&]() -> bool {
                if (AllowUnsafeNativeDirectMoePrefillProfiling()) {
                  RecordExpertNativeMultiTokenExecution(token_count);
                  const bool prefill_ok = impl_->RunFusedMoePrefillPath(
                      cublas_handle,
                      heuristic_cache,
                      *input_fp32,
                      *normalized,
                      *router_logits,
                      topk_ids,
                      topk_weights,
                      output_fp32);
                  if (prefill_ok && DebugComparePrefillVsLegacy()) {
                    auto legacy_output = DeviceTensorFp32::Create(output_fp32->shape());
                    if (legacy_output != nullptr && legacy_output->valid()) {
                      bool legacy_ok = true;
                      for (std::size_t row_index = 0; row_index < token_count && legacy_ok; ++row_index) {
                        auto input_row =
                            CreateTokenRowView(input_fp32, row_index, impl_->config.hidden_size);
                        auto normalized_row =
                            CreateTokenRowView(normalized, row_index, impl_->config.hidden_size);
                        auto router_row =
                            CreateTokenRowView(router_logits, row_index, impl_->config.n_routed_experts);
                        auto output_row =
                            CreateTokenRowView(legacy_output.get(), row_index, impl_->config.hidden_size);
                        legacy_ok = input_row != nullptr &&
                            normalized_row != nullptr &&
                            router_row != nullptr &&
                            output_row != nullptr &&
                            impl_->RunDirectMoeDecodePath(
                                cublas_handle,
                                heuristic_cache,
                                *input_row,
                                *normalized_row,
                                *router_row,
                                topk_ids + (row_index * impl_->config.top_k),
                                topk_weights + (row_index * impl_->config.top_k),
                                output_row.get());
                      }
                      if (legacy_ok && cudaDeviceSynchronize() == cudaSuccess) {
                        std::vector<float> prefill_host(token_count * impl_->config.hidden_size);
                        std::vector<float> legacy_host(token_count * impl_->config.hidden_size);
                        cudaMemcpy(prefill_host.data(), output_fp32->data(),
                                   prefill_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(legacy_host.data(), legacy_output->data(),
                                   legacy_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        float global_max_diff = 0.0f;
                        for (std::size_t row = 0; row < token_count; ++row) {
                          float row_max_diff = 0.0f;
                          float row_max_abs = 0.0f;
                          for (std::size_t col = 0; col < impl_->config.hidden_size; ++col) {
                            const std::size_t idx = row * impl_->config.hidden_size + col;
                            const float diff = std::fabs(prefill_host[idx] - legacy_host[idx]);
                            const float abs_val = std::fabs(legacy_host[idx]);
                            if (diff > row_max_diff) row_max_diff = diff;
                            if (abs_val > row_max_abs) row_max_abs = abs_val;
                          }
                          if (row_max_diff > global_max_diff) global_max_diff = row_max_diff;
                          std::cerr << "PREFILL_VS_LEGACY layer="
                                    << impl_->config.layer_index
                                    << " token=" << row
                                    << " max_abs_diff=" << row_max_diff
                                    << " legacy_max_abs=" << row_max_abs
                                    << (row_max_diff > 1.0f ? " ***DIVERGENT***" : "")
                                    << "\n";
                        }
                        std::cerr << "PREFILL_VS_LEGACY layer="
                                  << impl_->config.layer_index
                                  << " global_max_abs_diff=" << global_max_diff
                                  << " tokens=" << token_count
                                  << "\n";
                      }
                    }
                  }
                  return prefill_ok;
                }
                // The fused multi-row MoE prefill path uses batch-coupled activation
                // scaling. Replaying rows through the single-row decode contract keeps
                // prefix tokens invariant until the unified expert kernel replaces it.
                RecordExpertRowReplayExecution(token_count);
                for (std::size_t row_index = 0; row_index < token_count; ++row_index) {
                  auto input_row =
                      CreateTokenRowView(input_fp32, row_index, impl_->config.hidden_size);
                  auto normalized_row =
                      CreateTokenRowView(normalized, row_index, impl_->config.hidden_size);
                  auto router_row =
                      CreateTokenRowView(router_logits, row_index, impl_->config.n_routed_experts);
                  auto output_row =
                      CreateTokenRowView(output_fp32, row_index, impl_->config.hidden_size);
                  if (input_row == nullptr ||
                      normalized_row == nullptr ||
                      router_row == nullptr ||
                      output_row == nullptr ||
                      !impl_->RunDirectMoeDecodePath(
                          cublas_handle,
                          heuristic_cache,
                          *input_row,
                          *normalized_row,
                          *router_row,
                          topk_ids + (row_index * impl_->config.top_k),
                          topk_weights + (row_index * impl_->config.top_k),
                          output_row.get())) {
                    return false;
                  }
                }
                return true;
              }();
    if (!ok) {
      if (debug) {
        std::cout << "expert_layer: direct MoE "
                  << (token_count == 1 ? "decode" : "prefill")
                  << " path failed\n";
      }
      return false;
    }
    return CastTensorFp32ToBf16(*output_fp32, output);
  }

  if (token_count > 1) {
    RecordExpertNativeMultiTokenExecution(token_count);
  }

  auto latent = DeviceTensorFp32::Create({token_count, impl_->config.moe_latent_size});
  auto routed_tensor = DeviceTensorFp32::Create({token_count, impl_->config.moe_latent_size});
  auto projected_routed = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  std::unique_ptr<DeviceTensorFp32> shared_up_owned;
  std::unique_ptr<DeviceTensorFp32> shared_up_workspace_view;
  DeviceTensorFp32* shared_up = nullptr;
  if (use_request_workspace) {
    shared_up_workspace_view = CreateWorkspaceView(
        workspace->fused_prefill_shared_up_scratch.get(),
        token_count,
        impl_->config.shared_expert_intermediate_size);
    shared_up = shared_up_workspace_view.get();
  }
  if (shared_up == nullptr) {
    shared_up_owned =
        DeviceTensorFp32::Create({token_count, impl_->config.shared_expert_intermediate_size});
    shared_up = shared_up_owned.get();
  }
  if (!latent || !routed_tensor || !projected_routed || !shared_up) {
    if (debug) {
      std::cout << "expert_layer: latent scratch allocation failed\n";
    }
    return false;
  }

  const bool fc1_ok =
      (impl_->fc1_latent_family == Impl::ProjectionFamily::kScaledFp8 &&
       impl_->fc1_latent_scaled_fp8->Run(cublas_handle, heuristic_cache, *normalized, latent.get())) ||
      (impl_->fc1_latent_family == Impl::ProjectionFamily::kDense &&
       impl_->fc1_latent_dense->Run(cublas_handle, heuristic_cache, *normalized, latent.get()));
  const bool shared_up_ok =
      (impl_->shared_up_family == Impl::ProjectionFamily::kScaledFp8 &&
       impl_->shared_up_scaled_fp8->Run(cublas_handle, heuristic_cache, *normalized, shared_up)) ||
      (impl_->shared_up_family == Impl::ProjectionFamily::kDense &&
       impl_->shared_up_dense->Run(cublas_handle, heuristic_cache, *normalized, shared_up));
  if (!fc1_ok || !shared_up_ok) {
    if (debug) {
      std::cout << "expert_layer: fc1/shared_up projection failed"
                << " fc1_ok=" << fc1_ok
                << " shared_up_ok=" << shared_up_ok << "\n";
    }
    return false;
  }

  std::vector<float> normalized_host;
  std::vector<float> router_logits_host;
  std::vector<float> latent_host;
  std::vector<float> shared_up_host;
  if (!CopyToHost(*normalized, &normalized_host) ||
      !CopyToHost(*router_logits, &router_logits_host) ||
      !CopyToHost(*latent, &latent_host) ||
      !CopyToHost(*shared_up, &shared_up_host)) {
    if (debug) {
      std::cout << "expert_layer: failed to copy intermediate tensors to host\n";
    }
    return false;
  }
  if (debug && (HasNonFinite(normalized_host) ||
                HasNonFinite(router_logits_host) ||
                HasNonFinite(latent_host) ||
                HasNonFinite(shared_up_host))) {
    std::cout << "expert_layer: non-finite values detected"
              << " normalized=" << HasNonFinite(normalized_host)
              << " router=" << HasNonFinite(router_logits_host)
              << " latent=" << HasNonFinite(latent_host)
              << " shared_up=" << HasNonFinite(shared_up_host) << "\n";
  }

  std::vector<float> routed_latent_output(token_count * impl_->config.moe_latent_size, 0.0f);
  std::vector<float> shared_output(token_count * impl_->config.hidden_size, 0.0f);
  std::vector<float> activated_shared_output(
      token_count * impl_->config.shared_expert_intermediate_size,
      0.0f);
  std::vector<ExpertSelection> trace_selections;
  std::vector<float> trace_router_logits;
  std::vector<float> trace_latent_output;
  std::vector<float> trace_routed_output;
  std::vector<std::size_t> trace_routed_expert_order;
  std::vector<float> trace_routed_activated_hidden;
  std::vector<float> trace_routed_expert_outputs;
  std::vector<float> trace_routed_weighted_contributions;
  std::vector<float> trace_shared_output;

  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float* router_row = router_logits_host.data() + (token_index * impl_->config.n_routed_experts);
    const float* latent_row = latent_host.data() + (token_index * impl_->config.moe_latent_size);
    const float* normalized_row = normalized_host.data() + (token_index * impl_->config.hidden_size);
    const float* shared_up_row =
        shared_up_host.data() + (token_index * impl_->config.shared_expert_intermediate_size);

    const std::vector<float> router_row_vec(
        router_row,
        router_row + impl_->config.n_routed_experts);
    const std::vector<ExpertSelection> selections = SelectTopExperts(
        impl_->config,
        router_row_vec,
        impl_->gate_score_correction_bias);
    if (selections.size() != impl_->config.top_k) {
      if (debug) {
        std::cout << "expert_layer: top-k selection size mismatch\n";
      }
      return false;
    }

    std::vector<float> latent_row_vec(
        latent_row,
        latent_row + impl_->config.moe_latent_size);
    float* routed_row = routed_latent_output.data() + (token_index * impl_->config.moe_latent_size);
    for (const ExpertSelection& selection : selections) {
      if (selection.expert_index >= impl_->routed_experts.size()) {
        if (debug) {
          std::cout << "expert_layer: selected expert index out of range "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      const Impl::RoutedExpertRuntime& pair = impl_->routed_experts[selection.expert_index];
      if (pair.up_proj == nullptr || pair.down_proj == nullptr) {
        if (debug) {
          std::cout << "expert_layer: missing routed expert weights for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }

      const auto up_output = RunNvfp4LinearHost(latent_row_vec, 1, *pair.up_proj);
      if (!up_output.has_value()) {
        if (debug) {
          std::cout << "expert_layer: routed up_proj failed for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      const std::vector<float> activated_up = ApplyRelu2(*up_output);
      const auto down_output = RunNvfp4LinearHost(activated_up, 1, *pair.down_proj);
      if (!down_output.has_value() || down_output->size() != impl_->config.moe_latent_size) {
        if (debug) {
          std::cout << "expert_layer: routed down_proj failed for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      if (trace != nullptr && token_index == 0) {
        trace_routed_expert_order.push_back(selection.expert_index);
        trace_routed_activated_hidden.insert(
            trace_routed_activated_hidden.end(),
            activated_up.begin(),
            activated_up.end());
        trace_routed_expert_outputs.insert(
            trace_routed_expert_outputs.end(),
            down_output->begin(),
            down_output->end());
        trace_routed_weighted_contributions.insert(
            trace_routed_weighted_contributions.end(),
            down_output->size(),
            0.0f);
        float* contribution =
            trace_routed_weighted_contributions.data() +
            (trace_routed_expert_order.size() - 1) * impl_->config.moe_latent_size;
        for (std::size_t i = 0; i < impl_->config.moe_latent_size; ++i) {
          contribution[i] = (*down_output)[i] * selection.weight;
        }
      }
      for (std::size_t i = 0; i < impl_->config.moe_latent_size; ++i) {
        routed_row[i] += (*down_output)[i] * selection.weight;
      }
    }

    std::vector<float> shared_up_row_vec(
        shared_up_row,
        shared_up_row + impl_->config.shared_expert_intermediate_size);
    const std::vector<float> activated_shared_up = ApplyRelu2(shared_up_row_vec);
    std::memcpy(
        activated_shared_output.data() +
            (token_index * impl_->config.shared_expert_intermediate_size),
        activated_shared_up.data(),
        impl_->config.shared_expert_intermediate_size * sizeof(float));

    if (trace != nullptr) {
      trace->normalized_input.assign(
          normalized_row,
          normalized_row + impl_->config.hidden_size);
      trace_router_logits = router_row_vec;
      trace_selections = selections;
      trace_latent_output.assign(latent_row, latent_row + impl_->config.moe_latent_size);
      trace_routed_output.assign(routed_row, routed_row + impl_->config.moe_latent_size);
      trace->routed_expert_order = trace_routed_expert_order;
      trace->routed_expert_activated_hidden = trace_routed_activated_hidden;
      trace->routed_expert_outputs = trace_routed_expert_outputs;
      trace->routed_expert_weighted_contributions = trace_routed_weighted_contributions;
    }
  }

  if (impl_->shared_down_family == Impl::SharedDownFamily::kNvfp4) {
    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
      const float* activated_row =
          activated_shared_output.data() +
          (token_index * impl_->config.shared_expert_intermediate_size);
      std::vector<float> activated_row_vec(
          activated_row,
          activated_row + impl_->config.shared_expert_intermediate_size);
      const auto shared_row_output =
          RunNvfp4LinearHost(activated_row_vec, 1, *impl_->shared_down_nvfp4);
      if (!shared_row_output.has_value() ||
          shared_row_output->size() != impl_->config.hidden_size) {
        if (debug) {
          std::cout << "expert_layer: shared down NVFP4 path failed\n";
        }
        return false;
      }
      std::memcpy(
          shared_output.data() + (token_index * impl_->config.hidden_size),
          shared_row_output->data(),
          impl_->config.hidden_size * sizeof(float));
      if (trace != nullptr && token_index == 0) {
        trace_shared_output = *shared_row_output;
      }
    }
  } else {
    auto activated_shared_device =
        DeviceTensorFp32::Create({token_count, impl_->config.shared_expert_intermediate_size});
    auto shared_output_device =
        DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    if (!activated_shared_device ||
        !shared_output_device ||
        !UploadHostVector(activated_shared_output, activated_shared_device.get())) {
      if (debug) {
        std::cout << "expert_layer: shared down staging failed\n";
      }
      return false;
    }

    const bool shared_down_ok =
        impl_->shared_down_family == Impl::SharedDownFamily::kScaledFp8
            ? impl_->shared_down_scaled_fp8->Run(
                  cublas_handle,
                  heuristic_cache,
                  *activated_shared_device,
                  shared_output_device.get())
            : impl_->shared_down_dense->Run(
                  cublas_handle,
                  heuristic_cache,
                  *activated_shared_device,
                  shared_output_device.get());
    if (!shared_down_ok ||
        !CopyToHost(*shared_output_device, &shared_output) ||
        shared_output.size() != token_count * impl_->config.hidden_size) {
      if (debug) {
        std::cout << "expert_layer: shared down projection failed\n";
      }
      return false;
    }
    if (trace != nullptr) {
      trace_shared_output.assign(
          shared_output.begin(),
          shared_output.begin() + impl_->config.hidden_size);
    }
  }

  if (!UploadHostVector(routed_latent_output, routed_tensor.get()) ||
      !impl_->fc2_latent->Run(cublas_handle, heuristic_cache, *routed_tensor, projected_routed.get())) {
    if (debug) {
      std::cout << "expert_layer: fc2 latent projection failed\n";
    }
    return false;
  }

  std::vector<float> projected_routed_host;
  if (!CopyToHost(*projected_routed, &projected_routed_host) ||
      projected_routed_host.size() != token_count * impl_->config.hidden_size) {
    if (debug) {
      std::cout << "expert_layer: projected routed copy failed\n";
    }
    return false;
  }

  std::vector<float> mixer_output(token_count * impl_->config.hidden_size, 0.0f);
  std::vector<float> final_output(token_count * impl_->config.hidden_size, 0.0f);
  for (std::size_t i = 0; i < final_output.size(); ++i) {
    mixer_output[i] = projected_routed_host[i] + shared_output[i];
    final_output[i] = mixer_output[i];
  }

  if (!output_fp32->CopyFromHost(final_output.data(), final_output.size()) ||
      !CastTensorFp32ToBf16(*output_fp32, output)) {
    if (debug) {
      std::cout << "expert_layer: final output upload/cast failed\n";
    }
    return false;
  }

  if (trace != nullptr) {
    trace->router_logits = std::move(trace_router_logits);
    trace->selected_experts = std::move(trace_selections);
    trace->latent_output = std::move(trace_latent_output);
    trace->routed_latent_output = std::move(trace_routed_output);
    trace->projected_routed_output.assign(
        projected_routed_host.begin(),
        projected_routed_host.begin() + impl_->config.hidden_size);
    trace->shared_output = std::move(trace_shared_output);
    trace->mixer_output.assign(
        mixer_output.begin(),
        mixer_output.begin() + impl_->config.hidden_size);
  }
  return true;
}

bool ExpertLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    ExpertLayerRunTrace* trace,
    MoePrefillWorkspace* workspace) const {
  auto input_bf16 = DeviceTensorBf16::Create(input.shape());
  auto residual_bf16 = DeviceTensorBf16::Create(input.shape());
  auto output_bf16 = DeviceTensorBf16::Create(input.shape());
  const bool ok =
      input_bf16 != nullptr &&
      residual_bf16 != nullptr &&
      output_bf16 != nullptr &&
      CastTensorFp32ToBf16(input, input_bf16.get()) &&
      residual_bf16->FillZero() &&
      Run(
          cublas_handle,
          heuristic_cache,
          *input_bf16,
          residual_bf16.get(),
          output_bf16.get(),
          trace,
          workspace) &&
      CastTensorBf16ToFp32(*output_bf16, output);
  return ok;
}

}  // namespace nemotron
