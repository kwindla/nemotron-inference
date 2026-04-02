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
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/expert_staging_counters.h"
#include "nemotron/fused_moe_decode.h"
#include "nemotron/linear_op.h"
#include "nemotron/monolithic_expert_weights.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"

namespace nemotron {
namespace {

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

bool CopyToHost(const DeviceTensorFp32& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  output->assign(tensor.numel(), 0.0f);
  return tensor.CopyToHost(output->data(), output->size());
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
  static std::unique_ptr<DeviceArray<T>> CopyFromHost(const std::vector<T>& values) {
    if (values.empty()) {
      return nullptr;
    }
    T* data = nullptr;
    const std::size_t bytes = values.size() * sizeof(T);
    if (cudaMalloc(reinterpret_cast<void**>(&data), bytes) != cudaSuccess) {
      return nullptr;
    }
    if (cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(data);
      return nullptr;
    }
    return std::unique_ptr<DeviceArray<T>>(new DeviceArray<T>(data, values.size()));
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

  const T* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  DeviceArray(T* data, std::size_t size) : data_(data), size_(size) {}

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

FusedNvfp4WeightView MakeFusedNvfp4WeightView(const DeviceNvfp4Weight& weight) {
  return FusedNvfp4WeightView{
      weight.packed_data(),
      weight.block_scales_data(),
      weight.matmul_block_scales_data(),
      reinterpret_cast<const float*>(weight.tensor_scale_data()),
      weight.output_rows(),
      weight.input_cols(),
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

std::optional<CublasLtGemmPlan> BuildRuntimeNvfp4GemmPlan(
    const GemmDescriptor& descriptor,
    const Nvfp4PackedMatrixDeviceView& weight_view,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache) {
  auto runtime_launch_plan = BuildGemmLaunchPlan(descriptor, rows);
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
         static_cast<std::uint64_t>(weight.tensor_scale_nbytes());
}

std::uint64_t EstimatedUploadedBytes(const GemmDescriptor& descriptor) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    return 0;
  }
  return static_cast<std::uint64_t>(descriptor.packed_nbytes) +
         static_cast<std::uint64_t>(descriptor.block_scales_nbytes) +
         static_cast<std::uint64_t>(
             ExecutionNvfp4ScaleBytes(descriptor.output_rows, descriptor.input_cols)) +
         static_cast<std::uint64_t>(descriptor.tensor_scale_nbytes);
}

constexpr std::uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr std::uint64_t kDefaultForwardVramReserveMiB = 512ull;
constexpr std::uint64_t kExpertRuntimeHeadroomMiB = 2048ull;
constexpr const char* kNvfp4ActivationTensorScaleEnvVar =
    "NEMOTRON_FORWARD_NVFP4_ACTIVATION_TENSOR_SCALE";

std::uint64_t ParseEnvMiB(const char* name, std::uint64_t default_value_mib) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return default_value_mib;
  }

  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed_mib = std::strtoull(value, &end, 10);
  if (end == value || errno != 0) {
    return default_value_mib;
  }
  return static_cast<std::uint64_t>(parsed_mib);
}

std::optional<float> ParsePositiveFloatEnv(const char* env_var) {
  const char* value = std::getenv(env_var);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }

  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (end == value || (end != nullptr && *end != '\0') || errno == ERANGE ||
      !std::isfinite(parsed) || parsed <= 0.0f) {
    return std::nullopt;
  }
  return parsed;
}

bool ExpertFullResidencyEnabled() {
  const char* value = std::getenv("NEMOTRON_EXPERT_FULL_RESIDENCY");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

bool ExpertMonolithicEnabled() {
  const char* value = std::getenv("NEMOTRON_EXPERT_MONOLITHIC");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

bool MoeCublasLtEnabled() {
  const char* value = std::getenv("NEMOTRON_FORWARD_MOE_CUBLASLT");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

Nvfp4PackOptions RuntimeMoeNvfp4PackOptions() {
  Nvfp4PackOptions options;
  options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  if (const auto fixed_tensor_scale = ParsePositiveFloatEnv(kNvfp4ActivationTensorScaleEnvVar);
      fixed_tensor_scale.has_value()) {
    options.fixed_tensor_scale = *fixed_tensor_scale;
  }
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
      tensor_scale);
}

std::uint64_t ExpertResidencyMinimumHeadroomBytes() {
  return (ParseEnvMiB("NEMOTRON_FORWARD_VRAM_RESERVE_MB", kDefaultForwardVramReserveMiB) +
          kExpertRuntimeHeadroomMiB) *
         kBytesPerMiB;
}

std::uint64_t ExpertResidencyBudgetBytes() {
  const char* value = std::getenv("NEMOTRON_EXPERT_RESIDENCY_BUDGET_MB");
  if (value == nullptr || *value == '\0') {
    return 0;
  }

  errno = 0;
  char* end = nullptr;
  const long long parsed_mb = std::strtoll(value, &end, 10);
  if (end == value || errno != 0 || parsed_mb <= 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(parsed_mb) * 1024ull * 1024ull;
}

std::unique_ptr<DeviceNvfp4Weight> TryUploadNvfp4Weight(
    const GemmDescriptor& descriptor,
    bool debug,
    const char* context) {
  try {
    return DeviceNvfp4Weight::Upload(descriptor);
  } catch (const std::exception& error) {
    if (debug) {
      std::cerr << "expert_layer_create: " << context
                << " upload threw for " << descriptor.tensor_name
                << ": " << error.what() << "\n";
    }
  } catch (...) {
    if (debug) {
      std::cerr << "expert_layer_create: " << context
                << " upload threw for " << descriptor.tensor_name
                << ": unknown exception\n";
    }
  }
  return nullptr;
}

struct ExpertResidencyTracker {
  std::mutex mutex;
  bool initialized = false;
  bool skip_remaining_layers_for_vram = false;
  std::size_t last_layer_index = 0;
  std::uint64_t resident_routed_expert_bytes = 0;
  std::size_t resident_layer_count = 0;
  std::size_t nonresident_layer_count = 0;
};

ExpertResidencyTracker& GetExpertResidencyTracker() {
  static ExpertResidencyTracker tracker;
  return tracker;
}

void ResetExpertResidencyTrackerIfNeeded(
    ExpertResidencyTracker* tracker,
    std::size_t layer_index) {
  if (tracker == nullptr) {
    return;
  }
  if (!tracker->initialized ||
      layer_index == 0 ||
      layer_index < tracker->last_layer_index) {
    tracker->skip_remaining_layers_for_vram = false;
    tracker->resident_routed_expert_bytes = 0;
    tracker->resident_layer_count = 0;
    tracker->nonresident_layer_count = 0;
  }
  tracker->initialized = true;
  tracker->last_layer_index = layer_index;
}

bool HasNonFinite(const std::vector<float>& values) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  return false;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

}  // namespace

struct ExpertLayerSlice::Impl {
  struct RoutedExpertRuntime {
    const GemmDescriptor* up_proj = nullptr;
    const GemmDescriptor* down_proj = nullptr;
    std::unique_ptr<DeviceNvfp4Weight> up_proj_device;
    std::unique_ptr<DeviceNvfp4Weight> down_proj_device;
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
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> monolithic_up_views_device;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> monolithic_down_views_device;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> routed_up_nvfp4_views_device;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> routed_down_nvfp4_views_device;
  std::uint64_t resident_shared_expert_bytes = 0;
  std::uint64_t resident_routed_expert_bytes = 0;
  bool full_residency_enabled = false;
  bool monolithic_resident = false;
  bool fused_direct_moe_supported = false;
};

bool RunMoeDirectDecodeViaCublaslt(
    const ExpertLayerConfig& config,
    const std::vector<const GemmDescriptor*>& routed_up_descriptors,
    const std::vector<const GemmDescriptor*>& routed_down_descriptors,
    const GemmDescriptor& shared_up_descriptor,
    const DeviceNvfp4Weight& shared_up_weight,
    const GemmDescriptor& shared_down_descriptor,
    const DeviceNvfp4Weight& shared_down_weight,
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& normalized,
    const std::vector<ExpertSelection>& selected_experts,
    const std::vector<FusedNvfp4WeightView>& routed_up_views,
    const std::vector<FusedNvfp4WeightView>& routed_down_views,
    DeviceTensorFp32* output) {
  if (!cublas_handle.valid() ||
      !input.valid() ||
      !normalized.valid() ||
      output == nullptr ||
      !output->valid() ||
      input.shape() != normalized.shape() ||
      input.shape() != output->shape() ||
      input.shape().size() != 2 ||
      input.shape()[0] != 1 ||
      input.shape()[1] != config.hidden_size ||
      routed_up_descriptors.empty() ||
      routed_down_descriptors.empty() ||
      routed_up_descriptors.size() != selected_experts.size() ||
      routed_down_descriptors.size() != selected_experts.size() ||
      routed_up_descriptors.front() == nullptr ||
      routed_down_descriptors.front() == nullptr ||
      !shared_up_weight.valid() ||
      !shared_down_weight.valid() ||
      selected_experts.empty() ||
      selected_experts.size() != routed_up_views.size() ||
      selected_experts.size() != routed_down_views.size()) {
    return false;
  }

  const Nvfp4PackOptions pack_options = RuntimeMoeNvfp4PackOptions();
  auto normalized_packed = PackDeviceRowMajorFp32ToNvfp4(normalized, pack_options);
  auto routed_output = DeviceTensorFp32::Create({1, config.hidden_size});
  auto shared_output = DeviceTensorFp32::Create({1, config.hidden_size});
  auto routed_up_output =
      DeviceTensorFp32::Create({1, config.routed_expert_intermediate_size});
  auto routed_down_output = DeviceTensorFp32::Create({1, config.hidden_size});
  auto shared_up_output =
      DeviceTensorFp32::Create({1, config.shared_expert_intermediate_size});
  auto mixer_output = DeviceTensorFp32::Create({1, config.hidden_size});
  if (!normalized_packed ||
      !normalized_packed->valid() ||
      !routed_output ||
      !shared_output ||
      !routed_up_output ||
      !routed_down_output ||
      !shared_up_output ||
      !mixer_output ||
      !routed_output->FillZero()) {
    return false;
  }

  const Nvfp4PackedMatrixDeviceView normalized_view =
      MakeNvfp4PackedMatrixDeviceView(*normalized_packed);
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

  for (std::size_t slot = 0; slot < selected_experts.size(); ++slot) {
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
             expert_up_view,
             routed_up_output.get())
             .has_value() ||
        !Relu2InPlaceFp32(routed_up_output.get())) {
      return false;
    }

    auto routed_activated_packed =
        PackDeviceRowMajorFp32ToNvfp4(*routed_up_output, pack_options);
    if (!routed_activated_packed ||
        !routed_activated_packed->valid() ||
        !RunNvfp4RowMajorFp32AccumToDevice(
             cublas_handle,
             *routed_down_plan,
             MakeNvfp4PackedMatrixDeviceView(*routed_activated_packed),
             expert_down_view,
             routed_down_output.get())
             .has_value() ||
        !AccumulateScaledFp32(
             *routed_down_output,
             selected_experts[slot].weight,
             routed_output.get())) {
      return false;
    }
  }

  if (!RunNvfp4RowMajorFp32AccumToDevice(
           cublas_handle,
           *shared_up_plan,
           normalized_view,
           shared_up_weight_view,
           shared_up_output.get())
           .has_value() ||
      !Relu2InPlaceFp32(shared_up_output.get())) {
    return false;
  }

  auto shared_activated_packed =
      PackDeviceRowMajorFp32ToNvfp4(*shared_up_output, pack_options);
  if (!shared_activated_packed ||
      !shared_activated_packed->valid() ||
      !RunNvfp4RowMajorFp32AccumToDevice(
           cublas_handle,
           *shared_down_plan,
           MakeNvfp4PackedMatrixDeviceView(*shared_activated_packed),
           shared_down_weight_view,
           shared_output.get())
           .has_value() ||
      !ResidualAddFp32(*routed_output, *shared_output, mixer_output.get()) ||
      !ResidualAddFp32(input, *mixer_output, output)) {
    return false;
  }

  return true;
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
  if (config.hidden_size == 0 ||
      config.routed_expert_intermediate_size == 0 ||
      config.shared_expert_intermediate_size == 0 ||
      config.n_routed_experts == 0 ||
      config.top_k == 0 ||
      config.top_k > config.n_routed_experts ||
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

  const bool full_residency_requested = ExpertFullResidencyEnabled();
  const bool monolithic_requested = ExpertMonolithicEnabled() && !uses_latent_projection;
  const std::uint64_t residency_budget_bytes = ExpertResidencyBudgetBytes();
  const std::uint64_t minimum_headroom_bytes = ExpertResidencyMinimumHeadroomBytes();
  std::uint64_t resident_routed_expert_bytes = 0;
  std::unique_ptr<MonolithicNvfp4ExpertWeights> monolithic_up;
  std::unique_ptr<MonolithicNvfp4ExpertWeights> monolithic_down;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> monolithic_up_views_device;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> monolithic_down_views_device;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> routed_up_nvfp4_views_device;
  std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> routed_down_nvfp4_views_device;
  bool full_residency_enabled = false;
  bool monolithic_resident = false;
  std::uint64_t monolithic_up_bytes = 0;
  std::uint64_t monolithic_down_bytes = 0;
  std::size_t fully_resident_layer_count = 0;
  std::size_t nonresident_layer_count = 0;
  std::size_t uploaded_routed_experts = 0;
  std::string residency_reason = "disabled";
  std::string residency_detail;
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
  std::size_t free_vram_bytes_raw = 0;
  std::size_t total_vram_bytes_raw = 0;
  const cudaError_t mem_info_status = cudaMemGetInfo(&free_vram_bytes_raw, &total_vram_bytes_raw);
  const bool has_mem_info = mem_info_status == cudaSuccess;
  const std::uint64_t free_vram_bytes = static_cast<std::uint64_t>(free_vram_bytes_raw);
  const auto clear_monolithic_residency_uploads = [&]() {
    monolithic_up_views_device.reset();
    monolithic_down_views_device.reset();
    monolithic_up.reset();
    monolithic_down.reset();
    routed_up_nvfp4_views_device.reset();
    routed_down_nvfp4_views_device.reset();
    resident_routed_expert_bytes = 0;
    monolithic_up_bytes = 0;
    monolithic_down_bytes = 0;
    uploaded_routed_experts = 0;
    full_residency_enabled = false;
    monolithic_resident = false;
  };
  const auto clear_routed_residency_uploads = [&]() {
    for (Impl::RoutedExpertRuntime& runtime_pair : routed_experts) {
      runtime_pair.up_proj_device.reset();
      runtime_pair.down_proj_device.reset();
    }
    routed_up_nvfp4_views_device.reset();
    routed_down_nvfp4_views_device.reset();
    resident_routed_expert_bytes = 0;
  };
  if (full_residency_requested && fused_direct_moe_supported && monolithic_requested) {
    try {
      monolithic_up = MonolithicNvfp4ExpertWeights::Create(
          config.n_routed_experts,
          config.routed_expert_intermediate_size,
          config.hidden_size);
      monolithic_down = MonolithicNvfp4ExpertWeights::Create(
          config.n_routed_experts,
          config.hidden_size,
          config.routed_expert_intermediate_size);
      if (!monolithic_up || !monolithic_up->valid() ||
          !monolithic_down || !monolithic_down->valid()) {
        monolithic_detail = "allocation failed";
        clear_monolithic_residency_uploads();
      } else {
        bool monolithic_upload_failed = false;
        for (std::size_t expert_index = 0; expert_index < routed_experts.size(); ++expert_index) {
          const Impl::RoutedExpertRuntime& runtime_pair = routed_experts[expert_index];
          if (runtime_pair.up_proj == nullptr || runtime_pair.down_proj == nullptr ||
              !UploadDescriptorToMonolithic(
                  monolithic_up.get(), expert_index, *runtime_pair.up_proj) ||
              !UploadDescriptorToMonolithic(
                  monolithic_down.get(), expert_index, *runtime_pair.down_proj)) {
            monolithic_detail = "expert upload failed";
            monolithic_upload_failed = true;
            break;
          }
        }
        if (monolithic_upload_failed) {
          clear_monolithic_residency_uploads();
        } else {
          std::vector<FusedNvfp4WeightView> monolithic_up_views = monolithic_up->BuildAllViews();
          std::vector<FusedNvfp4WeightView> monolithic_down_views = monolithic_down->BuildAllViews();
          if (monolithic_up_views.size() != routed_experts.size() ||
              monolithic_down_views.size() != routed_experts.size()) {
            monolithic_detail = "view build failed";
            clear_monolithic_residency_uploads();
          } else {
            monolithic_up_views_device =
                DeviceArray<FusedNvfp4WeightView>::CopyFromHost(monolithic_up_views);
            monolithic_down_views_device =
                DeviceArray<FusedNvfp4WeightView>::CopyFromHost(monolithic_down_views);
            routed_up_nvfp4_views_device =
                DeviceArray<FusedNvfp4WeightView>::CopyFromHost(monolithic_up_views);
            routed_down_nvfp4_views_device =
                DeviceArray<FusedNvfp4WeightView>::CopyFromHost(monolithic_down_views);
            if (!monolithic_up_views_device ||
                !monolithic_down_views_device ||
                !routed_up_nvfp4_views_device ||
                !routed_down_nvfp4_views_device) {
              monolithic_detail = "view upload failed";
              clear_monolithic_residency_uploads();
            } else {
              monolithic_up_bytes = static_cast<std::uint64_t>(monolithic_up->total_bytes());
              monolithic_down_bytes = static_cast<std::uint64_t>(monolithic_down->total_bytes());
              resident_routed_expert_bytes = monolithic_up_bytes + monolithic_down_bytes;
              uploaded_routed_experts = routed_experts.size();
              full_residency_enabled = true;
              monolithic_resident = true;
            }
          }
        }
      }
    } catch (const std::exception& error) {
      monolithic_detail = error.what();
      clear_monolithic_residency_uploads();
    } catch (...) {
      monolithic_detail = "unknown";
      clear_monolithic_residency_uploads();
    }
  }
  auto& residency_tracker = GetExpertResidencyTracker();
  {
    std::lock_guard<std::mutex> lock(residency_tracker.mutex);
    ResetExpertResidencyTrackerIfNeeded(&residency_tracker, config.layer_index);
    if (!full_residency_requested) {
      residency_reason = "disabled";
      ++residency_tracker.nonresident_layer_count;
    } else if (!fused_direct_moe_supported) {
      residency_reason = "unsupported";
      ++residency_tracker.nonresident_layer_count;
    } else if (monolithic_resident) {
      residency_reason = "monolithic";
      residency_tracker.resident_routed_expert_bytes += resident_routed_expert_bytes;
      ++residency_tracker.resident_layer_count;
    } else if (residency_tracker.skip_remaining_layers_for_vram) {
      residency_reason = "vram_skip";
      ++residency_tracker.nonresident_layer_count;
      if (has_mem_info) {
        std::cout << "expert_layer_create: skipping residency for layer " << config.layer_index
                  << ": free_vram_mib=" << (free_vram_bytes / kBytesPerMiB)
                  << " minimum_headroom_mib=" << (minimum_headroom_bytes / kBytesPerMiB)
                  << "\n";
      }
    } else if (residency_budget_bytes > 0 &&
               residency_tracker.resident_routed_expert_bytes + estimated_routed_layer_bytes >
                   residency_budget_bytes) {
      residency_reason = "budget_skip";
      ++residency_tracker.nonresident_layer_count;
    } else if (has_mem_info &&
               (free_vram_bytes <= estimated_routed_layer_bytes ||
                free_vram_bytes - estimated_routed_layer_bytes <= minimum_headroom_bytes)) {
      residency_reason = "vram_skip";
      residency_tracker.skip_remaining_layers_for_vram = true;
      ++residency_tracker.nonresident_layer_count;
      std::cout << "expert_layer_create: skipping residency for layer " << config.layer_index
                << ": free_vram_mib=" << (free_vram_bytes / kBytesPerMiB)
                << " minimum_headroom_mib=" << (minimum_headroom_bytes / kBytesPerMiB)
                << "\n";
    } else {
      try {
        std::vector<FusedNvfp4WeightView> routed_up_views;
        std::vector<FusedNvfp4WeightView> routed_down_views;
        routed_up_views.reserve(routed_experts.size());
        routed_down_views.reserve(routed_experts.size());
        for (Impl::RoutedExpertRuntime& runtime_pair : routed_experts) {
          runtime_pair.up_proj_device =
              TryUploadNvfp4Weight(*runtime_pair.up_proj, debug, "routed expert up");
          if (!runtime_pair.up_proj_device || !runtime_pair.up_proj_device->valid()) {
            residency_reason = "upload_failed";
            break;
          }
          runtime_pair.down_proj_device =
              TryUploadNvfp4Weight(*runtime_pair.down_proj, debug, "routed expert down");
          if (!runtime_pair.down_proj_device || !runtime_pair.down_proj_device->valid()) {
            residency_reason = "upload_failed";
            break;
          }
          resident_routed_expert_bytes +=
              TotalUploadedBytes(*runtime_pair.up_proj_device) +
              TotalUploadedBytes(*runtime_pair.down_proj_device);
          routed_up_views.push_back(MakeFusedNvfp4WeightView(*runtime_pair.up_proj_device));
          routed_down_views.push_back(MakeFusedNvfp4WeightView(*runtime_pair.down_proj_device));
          ++uploaded_routed_experts;
        }
        if (uploaded_routed_experts == routed_experts.size()) {
          routed_up_nvfp4_views_device = DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up_views);
          routed_down_nvfp4_views_device =
              DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_down_views);
          if (routed_up_nvfp4_views_device && routed_down_nvfp4_views_device) {
            full_residency_enabled = true;
            residency_reason = "resident";
            residency_tracker.resident_routed_expert_bytes += resident_routed_expert_bytes;
            ++residency_tracker.resident_layer_count;
          } else {
            residency_reason = "view_upload_failed";
            clear_routed_residency_uploads();
            ++residency_tracker.nonresident_layer_count;
          }
        } else {
          clear_routed_residency_uploads();
          ++residency_tracker.nonresident_layer_count;
        }
      } catch (const std::exception& error) {
        residency_reason = "exception";
        residency_detail = error.what();
        clear_routed_residency_uploads();
        ++residency_tracker.nonresident_layer_count;
      } catch (...) {
        residency_reason = "exception";
        residency_detail = "unknown";
        clear_routed_residency_uploads();
        ++residency_tracker.nonresident_layer_count;
      }
    }
    fully_resident_layer_count = residency_tracker.resident_layer_count;
    nonresident_layer_count = residency_tracker.nonresident_layer_count;
    cumulative_resident_routed_expert_bytes = residency_tracker.resident_routed_expert_bytes;
  }

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
  impl->monolithic_up_views_device = std::move(monolithic_up_views_device);
  impl->monolithic_down_views_device = std::move(monolithic_down_views_device);
  impl->routed_up_nvfp4_views_device = std::move(routed_up_nvfp4_views_device);
  impl->routed_down_nvfp4_views_device = std::move(routed_down_nvfp4_views_device);
  impl->resident_shared_expert_bytes = resident_shared_expert_bytes;
  impl->resident_routed_expert_bytes = resident_routed_expert_bytes;
  impl->full_residency_enabled = full_residency_enabled;
  impl->monolithic_resident = monolithic_resident;
  impl->fused_direct_moe_supported = fused_direct_moe_supported;
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
              << " expert_full_residency_requested=" << full_residency_requested
              << " expert_full_residency=" << full_residency_enabled
              << " expert_residency_reason=" << residency_reason
              << " uploaded_routed_experts=" << uploaded_routed_experts
              << " estimated_routed_layer_bytes=" << estimated_routed_layer_bytes
              << " cumulative_resident_routed_expert_bytes="
              << cumulative_resident_routed_expert_bytes
              << " expert_residency_budget_mb="
              << (residency_budget_bytes == 0 ? 0
                                              : static_cast<long long>(
                                                    residency_budget_bytes / (1024ull * 1024ull)))
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
    if (!residency_detail.empty()) {
      std::cout << "expert_layer_create: layer=" << config.layer_index
                << " expert_residency_detail=" << residency_detail << "\n";
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
           impl_->monolithic_down->valid() &&
           impl_->monolithic_up_views_device != nullptr &&
           impl_->monolithic_down_views_device != nullptr)) &&
         (!impl_->full_residency_enabled ||
          (impl_->routed_up_nvfp4_views_device != nullptr &&
           impl_->routed_down_nvfp4_views_device != nullptr)) &&
         impl_->routed_experts.size() == impl_->config.n_routed_experts;
}

const ExpertLayerConfig& ExpertLayerSlice::config() const {
  return impl_->config;
}

bool ExpertLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    ExpertLayerRunTrace* trace) const {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool host_selection_debug =
      std::getenv("NEMOTRON_FORWARD_FUSED_MOE_HOST_SELECTION") != nullptr;
  const char* compare_fused_active_env =
      std::getenv("NEMOTRON_FORWARD_COMPARE_FUSED_MOE_ACTIVE");
  const bool compare_fused_debug =
      std::getenv("NEMOTRON_FORWARD_COMPARE_FUSED_MOE") != nullptr &&
      (compare_fused_active_env == nullptr ||
       std::strcmp(compare_fused_active_env, "0") != 0);

  if (!valid() ||
      !cublas_handle.valid() ||
      !input.valid() ||
      input.shape().size() != 2 ||
      input.shape()[1] != impl_->config.hidden_size ||
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

  auto normalized = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  auto router_logits = DeviceTensorFp32::Create({token_count, impl_->config.n_routed_experts});
  if (!normalized || !router_logits) {
    if (debug) {
      std::cout << "expert_layer: scratch allocation failed\n";
    }
    return false;
  }

  const bool norm_ok =
      RmsNormFp32(input, *impl_->input_norm_weight, impl_->config.rms_epsilon, normalized.get());
  const bool gate_ok =
      norm_ok &&
      impl_->gate_weight->Run(cublas_handle, heuristic_cache, *normalized, router_logits.get());
  if (!norm_ok || !gate_ok) {
    if (debug) {
      std::cout << "expert_layer: norm or gate projection failed"
                << " norm_ok=" << norm_ok
                << " gate_ok=" << gate_ok << "\n";
    }
    return false;
  }

  if (impl_->topology == Impl::ExpertTopology::kDirectMlp) {
    std::vector<float> fused_debug_output_host;
    std::vector<float> fused_debug_routed_host;
    std::vector<float> fused_debug_shared_host;
    bool collected_fused_debug = false;
    bool live_fused_output_written = false;
    const bool use_moe_cublaslt =
        trace == nullptr &&
        token_count == 1 &&
        !compare_fused_debug &&
        impl_->fused_direct_moe_supported &&
        FusedMoeDecodeEnabled() &&
        MoeCublasLtEnabled();
    if (use_moe_cublaslt) {
      std::vector<float> router_logits_host;
      if (!CopyToHost(*router_logits, &router_logits_host) ||
          router_logits_host.size() != impl_->config.n_routed_experts) {
        return false;
      }

      std::vector<ExpertSelection> selected_experts = SelectTopExperts(
          impl_->config,
          router_logits_host,
          impl_->gate_score_correction_bias);
      if (selected_experts.size() != impl_->config.top_k) {
        return false;
      }

      std::vector<std::unique_ptr<DeviceNvfp4Weight>> staged_up_weights;
      std::vector<std::unique_ptr<DeviceNvfp4Weight>> staged_down_weights;
      std::vector<const GemmDescriptor*> routed_up_descriptors;
      std::vector<const GemmDescriptor*> routed_down_descriptors;
      std::vector<FusedNvfp4WeightView> routed_up_views;
      std::vector<FusedNvfp4WeightView> routed_down_views;
      routed_up_descriptors.reserve(selected_experts.size());
      routed_down_descriptors.reserve(selected_experts.size());
      routed_up_views.reserve(selected_experts.size());
      routed_down_views.reserve(selected_experts.size());

      auto& staging_counters = GetExpertStagingCounters();
      if (impl_->monolithic_resident) {
        staging_counters.total_staging_calls.fetch_add(1, std::memory_order_relaxed);
        staging_counters.monolithic_layers.fetch_add(1, std::memory_order_relaxed);
        if (impl_->monolithic_up == nullptr || impl_->monolithic_down == nullptr) {
          return false;
        }
        for (const ExpertSelection& selection : selected_experts) {
          if (selection.expert_index >= impl_->routed_experts.size()) {
            return false;
          }
          const Impl::RoutedExpertRuntime& runtime_pair =
              impl_->routed_experts[selection.expert_index];
          if (runtime_pair.up_proj == nullptr || runtime_pair.down_proj == nullptr) {
            return false;
          }
          routed_up_descriptors.push_back(runtime_pair.up_proj);
          routed_down_descriptors.push_back(runtime_pair.down_proj);
          routed_up_views.push_back(impl_->monolithic_up->GetView(selection.expert_index));
          routed_down_views.push_back(impl_->monolithic_down->GetView(selection.expert_index));
        }
      } else if (impl_->full_residency_enabled) {
        for (const ExpertSelection& selection : selected_experts) {
          if (selection.expert_index >= impl_->routed_experts.size()) {
            return false;
          }
          const Impl::RoutedExpertRuntime& runtime_pair =
              impl_->routed_experts[selection.expert_index];
          if (runtime_pair.up_proj == nullptr ||
              runtime_pair.down_proj == nullptr ||
              runtime_pair.up_proj_device == nullptr ||
              runtime_pair.down_proj_device == nullptr ||
              !runtime_pair.up_proj_device->valid() ||
              !runtime_pair.down_proj_device->valid()) {
            return false;
          }
          routed_up_descriptors.push_back(runtime_pair.up_proj);
          routed_down_descriptors.push_back(runtime_pair.down_proj);
          routed_up_views.push_back(MakeFusedNvfp4WeightView(*runtime_pair.up_proj_device));
          routed_down_views.push_back(MakeFusedNvfp4WeightView(*runtime_pair.down_proj_device));
        }
      } else {
        staging_counters.total_staging_calls.fetch_add(1, std::memory_order_relaxed);
        staged_up_weights.reserve(selected_experts.size());
        staged_down_weights.reserve(selected_experts.size());
        for (const ExpertSelection& selection : selected_experts) {
          if (selection.expert_index >= impl_->routed_experts.size()) {
            return false;
          }
          const Impl::RoutedExpertRuntime& runtime_pair =
              impl_->routed_experts[selection.expert_index];
          if (runtime_pair.up_proj == nullptr || runtime_pair.down_proj == nullptr) {
            return false;
          }
          const auto up_upload_started = std::chrono::steady_clock::now();
          auto up_weight = DeviceNvfp4Weight::Upload(*runtime_pair.up_proj);
          const std::uint64_t up_elapsed_us = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - up_upload_started)
                  .count());
          staging_counters.staging_elapsed_us.fetch_add(up_elapsed_us, std::memory_order_relaxed);
          const auto down_upload_started = std::chrono::steady_clock::now();
          auto down_weight = DeviceNvfp4Weight::Upload(*runtime_pair.down_proj);
          const std::uint64_t down_elapsed_us = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - down_upload_started)
                  .count());
          staging_counters.staging_elapsed_us.fetch_add(down_elapsed_us, std::memory_order_relaxed);
          if (!up_weight || !up_weight->valid() || !down_weight || !down_weight->valid()) {
            return false;
          }
          const std::uint64_t upload_bytes =
              TotalUploadedBytes(*up_weight) + TotalUploadedBytes(*down_weight);
          staging_counters.total_bytes_uploaded.fetch_add(upload_bytes, std::memory_order_relaxed);
          staging_counters.total_experts_staged.fetch_add(1, std::memory_order_relaxed);
          routed_up_descriptors.push_back(runtime_pair.up_proj);
          routed_down_descriptors.push_back(runtime_pair.down_proj);
          routed_up_views.push_back(MakeFusedNvfp4WeightView(*up_weight));
          routed_down_views.push_back(MakeFusedNvfp4WeightView(*down_weight));
          staged_up_weights.push_back(std::move(up_weight));
          staged_down_weights.push_back(std::move(down_weight));
        }
      }

      if (RunMoeDirectDecodeViaCublaslt(
              impl_->config,
              routed_up_descriptors,
              routed_down_descriptors,
              *impl_->shared_up_nvfp4,
              *impl_->shared_up_nvfp4_device,
              *impl_->shared_down_nvfp4,
              *impl_->shared_down_nvfp4_device,
              cublas_handle,
              heuristic_cache,
              input,
              *normalized,
              selected_experts,
              routed_up_views,
              routed_down_views,
              output)) {
        return true;
      }

      if (debug) {
        std::cout << "expert_layer: cuBLASLt direct MoE fastpath failed, falling back\n";
      }
    }

    const bool use_fused_direct_decode =
        trace == nullptr &&
        token_count == 1 &&
        impl_->fused_direct_moe_supported &&
        FusedMoeDecodeEnabled();
    if (use_fused_direct_decode) {
      const bool use_monolithic_residency = impl_->monolithic_resident;
      const bool use_full_residency =
          use_monolithic_residency ||
          (impl_->full_residency_enabled &&
           impl_->routed_up_nvfp4_views_device != nullptr &&
           impl_->routed_down_nvfp4_views_device != nullptr);
      const bool needs_host_selected_experts = host_selection_debug || !use_full_residency;
      std::vector<std::unique_ptr<DeviceNvfp4Weight>> routed_up_weights;
      std::vector<std::unique_ptr<DeviceNvfp4Weight>> routed_down_weights;
      std::vector<FusedNvfp4WeightView> routed_up_views;
      std::vector<FusedNvfp4WeightView> routed_down_views;
      std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> routed_up_views_device;
      std::unique_ptr<DeviceArray<FusedNvfp4WeightView>> routed_down_views_device;
      std::vector<ExpertSelection> selected_experts;
      const FusedNvfp4WeightView* routed_up_device_ptr = nullptr;
      const FusedNvfp4WeightView* routed_down_device_ptr = nullptr;
      std::uint64_t staging_bytes_uploaded = 0;
      std::uint64_t staging_elapsed_us = 0;
      std::uint64_t experts_staged = 0;
      auto& staging_counters = GetExpertStagingCounters();
      if (needs_host_selected_experts) {
        std::vector<float> router_logits_host;
        if (!CopyToHost(*router_logits, &router_logits_host) ||
            router_logits_host.size() != impl_->config.n_routed_experts) {
          return false;
        }
        selected_experts = SelectTopExperts(
            impl_->config,
            router_logits_host,
            impl_->gate_score_correction_bias);
        if (selected_experts.size() != impl_->config.top_k) {
          return false;
        }
      }
      if (use_monolithic_residency) {
        if (impl_->monolithic_up_views_device == nullptr ||
            impl_->monolithic_down_views_device == nullptr) {
          return false;
        }
        routed_up_device_ptr = impl_->monolithic_up_views_device->data();
        routed_down_device_ptr = impl_->monolithic_down_views_device->data();
        staging_counters.total_staging_calls.fetch_add(1, std::memory_order_relaxed);
        staging_counters.monolithic_layers.fetch_add(1, std::memory_order_relaxed);
      } else if (use_full_residency) {
        routed_up_device_ptr = impl_->routed_up_nvfp4_views_device->data();
        routed_down_device_ptr = impl_->routed_down_nvfp4_views_device->data();
      } else {
        staging_counters.total_staging_calls.fetch_add(1, std::memory_order_relaxed);
        routed_up_weights.reserve(selected_experts.size());
        routed_down_weights.reserve(selected_experts.size());
        routed_up_views.reserve(selected_experts.size());
        routed_down_views.reserve(selected_experts.size());
        // Non-resident decode uploads a compact top-k view array and passes remapped
        // selected_indices so the kernel indexes only the staged experts.
        for (const ExpertSelection& selection : selected_experts) {
          if (selection.expert_index >= impl_->routed_experts.size()) {
            return false;
          }
          const Impl::RoutedExpertRuntime& runtime_pair =
              impl_->routed_experts[selection.expert_index];
          if (runtime_pair.up_proj == nullptr || runtime_pair.down_proj == nullptr) {
            return false;
          }
          const auto up_upload_started = std::chrono::steady_clock::now();
          auto up_weight = DeviceNvfp4Weight::Upload(*runtime_pair.up_proj);
          const std::uint64_t up_elapsed_us = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - up_upload_started)
                  .count());
          staging_elapsed_us += up_elapsed_us;
          staging_counters.staging_elapsed_us.fetch_add(up_elapsed_us, std::memory_order_relaxed);
          const auto down_upload_started = std::chrono::steady_clock::now();
          auto down_weight = DeviceNvfp4Weight::Upload(*runtime_pair.down_proj);
          const std::uint64_t down_elapsed_us = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - down_upload_started)
                  .count());
          staging_elapsed_us += down_elapsed_us;
          staging_counters.staging_elapsed_us.fetch_add(down_elapsed_us, std::memory_order_relaxed);
          if (!up_weight || !up_weight->valid() || !down_weight || !down_weight->valid()) {
            return false;
          }
          const std::uint64_t upload_bytes =
              TotalUploadedBytes(*up_weight) + TotalUploadedBytes(*down_weight);
          staging_bytes_uploaded += upload_bytes;
          staging_counters.total_bytes_uploaded.fetch_add(upload_bytes, std::memory_order_relaxed);
          ++experts_staged;
          staging_counters.total_experts_staged.fetch_add(1, std::memory_order_relaxed);
          routed_up_views.push_back(MakeFusedNvfp4WeightView(*up_weight));
          routed_down_views.push_back(MakeFusedNvfp4WeightView(*down_weight));
          routed_up_weights.push_back(std::move(up_weight));
          routed_down_weights.push_back(std::move(down_weight));
        }
        routed_up_views_device = DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up_views);
        routed_down_views_device = DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_down_views);
        if (!routed_up_views_device || !routed_down_views_device) {
          return false;
        }
        routed_up_device_ptr = routed_up_views_device->data();
        routed_down_device_ptr = routed_down_views_device->data();
      }
      if (debug) {
        std::cout << "expert_layer: fused direct "
                  << (use_monolithic_residency ? "monolithic"
                                               : (use_full_residency ? "resident" : "staged"))
                  << " routed experts=" << (use_full_residency ? impl_->routed_experts.size() : experts_staged)
                  << " routed experts bytes=" << staging_bytes_uploaded
                  << " elapsed_us=" << staging_elapsed_us << "\n";
      }

      std::unique_ptr<DeviceArray<int>> selected_indices_device;
      std::unique_ptr<DeviceArray<float>> selected_weights_device;
      if (needs_host_selected_experts) {
        std::vector<int> selected_indices_host(impl_->config.top_k, -1);
        std::vector<float> selected_weights_host(impl_->config.top_k, 0.0f);
        for (std::size_t slot = 0; slot < selected_experts.size(); ++slot) {
          selected_indices_host[slot] =
              use_full_residency ? static_cast<int>(selected_experts[slot].expert_index)
                                 : static_cast<int>(slot);
          selected_weights_host[slot] = selected_experts[slot].weight;
        }
        selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
        selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
        if (!selected_indices_device || !selected_weights_device) {
          return false;
        }
      }

      FusedMoeDirectLayerParams fused_params;
      fused_params.hidden_size = impl_->config.hidden_size;
      fused_params.routed_expert_intermediate_size =
          impl_->config.routed_expert_intermediate_size;
      fused_params.shared_expert_intermediate_size =
          impl_->config.shared_expert_intermediate_size;
      fused_params.n_routed_experts = impl_->config.n_routed_experts;
      fused_params.top_k = impl_->config.top_k;
      fused_params.n_group = impl_->config.n_group;
      fused_params.topk_group = impl_->config.topk_group;
      fused_params.routed_scaling_factor = impl_->config.routed_scaling_factor;
      fused_params.norm_topk_prob = impl_->config.norm_topk_prob;
      fused_params.shared_up = MakeFusedNvfp4WeightView(*impl_->shared_up_nvfp4_device);
      fused_params.shared_down = MakeFusedNvfp4WeightView(*impl_->shared_down_nvfp4_device);
      fused_params.routed_up = routed_up_device_ptr;
      fused_params.routed_down = routed_down_device_ptr;
      fused_params.correction_bias = impl_->gate_score_correction_bias_device->data();
      fused_params.selected_indices =
          selected_indices_device ? selected_indices_device->data() : nullptr;
      fused_params.selected_weights =
          selected_weights_device ? selected_weights_device->data() : nullptr;
      if (!compare_fused_debug) {
        return RunFusedMoeDirectDecode(fused_params, input, *normalized, *router_logits, output);
      }

      auto fused_routed_device = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
      auto fused_shared_device = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
      if (!fused_routed_device ||
          !fused_shared_device) {
        return false;
      }
      fused_params.routed_output = fused_routed_device->data();
      fused_params.shared_output = fused_shared_device->data();
      if (!RunFusedMoeDirectDecode(
              fused_params,
              input,
              *normalized,
              *router_logits,
              output)) {
        return false;
      }
      if (!CopyToHost(*output, &fused_debug_output_host) ||
          !CopyToHost(*fused_routed_device, &fused_debug_routed_host) ||
          !CopyToHost(*fused_shared_device, &fused_debug_shared_host)) {
        return false;
      }
      collected_fused_debug = true;
      live_fused_output_written = true;
    }

    std::vector<float> input_host;
    std::vector<float> normalized_host;
    std::vector<float> router_logits_host;
    if (!CopyToHost(input, &input_host) ||
        !CopyToHost(*normalized, &normalized_host) ||
        !CopyToHost(*router_logits, &router_logits_host)) {
      if (debug) {
        std::cout << "expert_layer: failed to copy direct-moe tensors to host\n";
      }
      return false;
    }

    std::vector<float> shared_up_host;
    if (impl_->shared_up_family != Impl::ProjectionFamily::kNvfp4) {
      auto shared_up = DeviceTensorFp32::Create({token_count, impl_->config.shared_expert_intermediate_size});
      if (!shared_up) {
        if (debug) {
          std::cout << "expert_layer: direct shared_up allocation failed\n";
        }
        return false;
      }
      const bool shared_up_ok =
          (impl_->shared_up_family == Impl::ProjectionFamily::kScaledFp8 &&
           impl_->shared_up_scaled_fp8->Run(cublas_handle, heuristic_cache, *normalized, shared_up.get())) ||
          (impl_->shared_up_family == Impl::ProjectionFamily::kDense &&
           impl_->shared_up_dense->Run(cublas_handle, heuristic_cache, *normalized, shared_up.get()));
      if (!shared_up_ok || !CopyToHost(*shared_up, &shared_up_host)) {
        if (debug) {
          std::cout << "expert_layer: direct shared_up projection failed\n";
        }
        return false;
      }
    }

    std::vector<float> routed_hidden_output(token_count * impl_->config.hidden_size, 0.0f);
    std::vector<float> shared_output(token_count * impl_->config.hidden_size, 0.0f);
    std::vector<float> activated_shared_output(
        token_count * impl_->config.shared_expert_intermediate_size,
        0.0f);
    std::vector<ExpertSelection> trace_selections;
    std::vector<float> trace_router_logits;
    std::vector<std::size_t> trace_routed_expert_order;
    std::vector<float> trace_routed_activated_hidden;
    std::vector<float> trace_routed_expert_outputs;
    std::vector<float> trace_routed_weighted_contributions;
    std::vector<float> trace_shared_output;

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
      const float* router_row = router_logits_host.data() + (token_index * impl_->config.n_routed_experts);
      const float* normalized_row = normalized_host.data() + (token_index * impl_->config.hidden_size);
      std::vector<float> router_row_vec(
          router_row,
          router_row + impl_->config.n_routed_experts);
      const std::vector<ExpertSelection> selections = SelectTopExperts(
          impl_->config,
          router_row_vec,
          impl_->gate_score_correction_bias);
      if (selections.size() != impl_->config.top_k) {
        if (debug) {
          std::cout << "expert_layer: direct top-k selection size mismatch\n";
        }
        return false;
      }

      std::vector<float> normalized_row_vec(
          normalized_row,
          normalized_row + impl_->config.hidden_size);
      float* routed_row = routed_hidden_output.data() + (token_index * impl_->config.hidden_size);
      for (const ExpertSelection& selection : selections) {
        if (selection.expert_index >= impl_->routed_experts.size()) {
          if (debug) {
            std::cout << "expert_layer: direct selected expert index out of range "
                      << selection.expert_index << "\n";
          }
          return false;
        }
        const Impl::RoutedExpertRuntime& pair = impl_->routed_experts[selection.expert_index];
        if (pair.up_proj == nullptr || pair.down_proj == nullptr) {
          if (debug) {
            std::cout << "expert_layer: direct missing routed expert weights for expert "
                      << selection.expert_index << "\n";
          }
          return false;
        }
        const auto up_output = RunNvfp4LinearHost(normalized_row_vec, 1, *pair.up_proj);
        if (!up_output.has_value()) {
          if (debug) {
            std::cout << "expert_layer: direct routed up_proj failed for expert "
                      << selection.expert_index << "\n";
          }
          return false;
        }
        const std::vector<float> activated_up = ApplyRelu2(*up_output);
        const auto down_output = RunNvfp4LinearHost(activated_up, 1, *pair.down_proj);
        if (!down_output.has_value() || down_output->size() != impl_->config.hidden_size) {
          if (debug) {
            std::cout << "expert_layer: direct routed down_proj failed for expert "
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
              (trace_routed_expert_order.size() - 1) * impl_->config.hidden_size;
          for (std::size_t i = 0; i < impl_->config.hidden_size; ++i) {
            contribution[i] = (*down_output)[i] * selection.weight;
          }
        }
        for (std::size_t i = 0; i < impl_->config.hidden_size; ++i) {
          routed_row[i] += (*down_output)[i] * selection.weight;
        }
      }

      std::vector<float> shared_up_row_vec;
      if (impl_->shared_up_family == Impl::ProjectionFamily::kNvfp4) {
        const auto shared_up_row = RunNvfp4LinearHost(normalized_row_vec, 1, *impl_->shared_up_nvfp4);
        if (!shared_up_row.has_value() ||
            shared_up_row->size() != impl_->config.shared_expert_intermediate_size) {
          if (debug) {
            std::cout << "expert_layer: direct shared_up NVFP4 path failed\n";
          }
          return false;
        }
        shared_up_row_vec = std::move(*shared_up_row);
      } else {
        const float* shared_up_row =
            shared_up_host.data() + (token_index * impl_->config.shared_expert_intermediate_size);
        shared_up_row_vec.assign(
            shared_up_row,
            shared_up_row + impl_->config.shared_expert_intermediate_size);
      }
      const std::vector<float> activated_shared_up = ApplyRelu2(shared_up_row_vec);
      std::memcpy(
          activated_shared_output.data() +
              (token_index * impl_->config.shared_expert_intermediate_size),
          activated_shared_up.data(),
          impl_->config.shared_expert_intermediate_size * sizeof(float));

      if (trace != nullptr && token_index == 0) {
        trace->normalized_input.assign(
            normalized_row,
            normalized_row + impl_->config.hidden_size);
        trace_router_logits = router_row_vec;
        trace_selections = selections;
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
            std::cout << "expert_layer: direct shared down NVFP4 path failed\n";
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
          std::cout << "expert_layer: direct shared down staging failed\n";
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
          std::cout << "expert_layer: direct shared down projection failed\n";
        }
        return false;
      }
      if (trace != nullptr) {
        trace_shared_output.assign(
            shared_output.begin(),
            shared_output.begin() + impl_->config.hidden_size);
      }
    }

    std::vector<float> mixer_output(token_count * impl_->config.hidden_size, 0.0f);
    std::vector<float> final_output(token_count * impl_->config.hidden_size, 0.0f);
    for (std::size_t i = 0; i < final_output.size(); ++i) {
      mixer_output[i] = routed_hidden_output[i] + shared_output[i];
      final_output[i] = input_host[i] + mixer_output[i];
    }

    if (!live_fused_output_written) {
      if (!output->CopyFromHost(final_output.data(), final_output.size())) {
        if (debug) {
          std::cout << "expert_layer: direct final output upload failed\n";
        }
        return false;
      }
    }

    if (trace != nullptr) {
      trace->router_logits = std::move(trace_router_logits);
      trace->selected_experts = std::move(trace_selections);
      trace->latent_output.clear();
      trace->routed_latent_output.clear();
      trace->projected_routed_output.assign(
          routed_hidden_output.begin(),
          routed_hidden_output.begin() + impl_->config.hidden_size);
      trace->shared_output = std::move(trace_shared_output);
      trace->mixer_output.assign(
          mixer_output.begin(),
          mixer_output.begin() + impl_->config.hidden_size);
    }
    if (collected_fused_debug) {
      std::cerr << "fused_moe_compare: layer=" << impl_->config.layer_index
                << " final_max_abs_diff=" << MaxAbsDiff(fused_debug_output_host, final_output)
                << " routed_max_abs_diff=" << MaxAbsDiff(fused_debug_routed_host, routed_hidden_output)
                << " shared_max_abs_diff=" << MaxAbsDiff(fused_debug_shared_host, shared_output)
                << " fused0=" << (fused_debug_output_host.empty() ? 0.0f : fused_debug_output_host.front())
                << " host0=" << (final_output.empty() ? 0.0f : final_output.front())
                << "\n";
    }
    return true;
  }

  auto latent = DeviceTensorFp32::Create({token_count, impl_->config.moe_latent_size});
  auto routed_tensor = DeviceTensorFp32::Create({token_count, impl_->config.moe_latent_size});
  auto projected_routed = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  auto shared_up = DeviceTensorFp32::Create({token_count, impl_->config.shared_expert_intermediate_size});
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
       impl_->shared_up_scaled_fp8->Run(cublas_handle, heuristic_cache, *normalized, shared_up.get())) ||
      (impl_->shared_up_family == Impl::ProjectionFamily::kDense &&
       impl_->shared_up_dense->Run(cublas_handle, heuristic_cache, *normalized, shared_up.get()));
  if (!fc1_ok || !shared_up_ok) {
    if (debug) {
      std::cout << "expert_layer: fc1/shared_up projection failed"
                << " fc1_ok=" << fc1_ok
                << " shared_up_ok=" << shared_up_ok << "\n";
    }
    return false;
  }

  std::vector<float> input_host;
  std::vector<float> normalized_host;
  std::vector<float> router_logits_host;
  std::vector<float> latent_host;
  std::vector<float> shared_up_host;
  if (!CopyToHost(input, &input_host) ||
      !CopyToHost(*normalized, &normalized_host) ||
      !CopyToHost(*router_logits, &router_logits_host) ||
      !CopyToHost(*latent, &latent_host) ||
      !CopyToHost(*shared_up, &shared_up_host)) {
    if (debug) {
      std::cout << "expert_layer: failed to copy intermediate tensors to host\n";
    }
    return false;
  }
  if (debug && (HasNonFinite(input_host) ||
                HasNonFinite(normalized_host) ||
                HasNonFinite(router_logits_host) ||
                HasNonFinite(latent_host) ||
                HasNonFinite(shared_up_host))) {
    std::cout << "expert_layer: non-finite values detected"
              << " input=" << HasNonFinite(input_host)
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
    final_output[i] = input_host[i] + mixer_output[i];
  }

  if (!output->CopyFromHost(final_output.data(), final_output.size())) {
    if (debug) {
      std::cout << "expert_layer: final output upload failed\n";
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

}  // namespace nemotron
