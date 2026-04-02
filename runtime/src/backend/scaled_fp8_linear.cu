#include "nemotron/scaled_fp8_linear.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "nemotron/runtime_stats.h"
#include "storage_conversion.h"

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool ExperimentalFp8NativeEnabled() {
  return std::getenv("NEMOTRON_DISABLE_FP8_NATIVE") == nullptr;
}

bool ExperimentalScaledFp8DequantizedDenseEnabled() {
  return std::getenv("NEMOTRON_DISABLE_SCALED_FP8_DEQUANTIZED_DENSE") == nullptr;
}

bool ExperimentalScaledFp8NativeDebugEnabled() {
  const char* value = std::getenv("NEMOTRON_DEBUG_SCALED_FP8_NATIVE");
  return value != nullptr && !(value[0] == '0' && value[1] == '\0');
}

const char* ExperimentalScaledFp8SurfaceFamilyFilter() {
  return std::getenv("NEMOTRON_EXPERIMENTAL_SCALED_FP8_SURFACE_FAMILY");
}

const char* ExperimentalScaledFp8SurfaceTensorFilter() {
  return std::getenv("NEMOTRON_EXPERIMENTAL_SCALED_FP8_SURFACE_TENSORS");
}

std::string_view ResolveScaledFp8TensorName(const ScaledFp8LinearConfig& config) {
  return config.tensor_name.empty() ? std::string_view("scaled_fp8_linear")
                                    : std::string_view(config.tensor_name);
}

std::string ResolveDequantizedTensorName(std::string_view tensor_name) {
  std::string name(tensor_name);
  name += "_dequantized";
  return name;
}

ScaledFp8RuntimeOpFamily ClassifyScaledFp8RuntimeOpFamily(std::string_view tensor_name) {
  const auto contains = [](std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
  };
  if (contains(tensor_name, "mixer.in_proj.weight") || contains(tensor_name, "in_proj.weight")) {
    return ScaledFp8RuntimeOpFamily::kMambaInProj;
  }
  if (contains(tensor_name, "mixer.out_proj.weight") || contains(tensor_name, "out_proj.weight")) {
    return ScaledFp8RuntimeOpFamily::kMambaOutProj;
  }
  if (contains(tensor_name, "fc1_latent_proj.weight")) {
    return ScaledFp8RuntimeOpFamily::kExpertFc1Latent;
  }
  if (contains(tensor_name, "shared_experts.up_proj") ||
      contains(tensor_name, "shared_expert.up_proj")) {
    return ScaledFp8RuntimeOpFamily::kExpertSharedUp;
  }
  if (contains(tensor_name, "shared_experts.down_proj") ||
      contains(tensor_name, "shared_expert.down_proj")) {
    return ScaledFp8RuntimeOpFamily::kExpertSharedDown;
  }
  return ScaledFp8RuntimeOpFamily::kOther;
}

bool ExperimentalScaledFp8SurfaceTensorMatches(const GemmDescriptor& descriptor) {
  const char* filter = ExperimentalScaledFp8SurfaceTensorFilter();
  if (filter == nullptr || *filter == '\0') {
    return true;
  }

  std::stringstream stream(filter);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::size_t begin = token.find_first_not_of(" \t");
    if (begin == std::string::npos) {
      continue;
    }
    const std::size_t end = token.find_last_not_of(" \t");
    const std::string_view needle(token.data() + begin, end - begin + 1);
    if (!needle.empty() &&
        std::string_view(descriptor.tensor_name).find(needle) != std::string_view::npos) {
      return true;
    }
  }
  return false;
}

bool ScaledFp8FamilyFilterMatchesToken(
    ScaledFp8RuntimeOpFamily family,
    std::string_view token) {
  if (token == "all") {
    return true;
  }
  switch (family) {
    case ScaledFp8RuntimeOpFamily::kMambaInProj:
      return token == "mamba" || token == "mamba_in_proj";
    case ScaledFp8RuntimeOpFamily::kMambaOutProj:
      return token == "mamba" || token == "mamba_out_proj";
    case ScaledFp8RuntimeOpFamily::kExpertFc1Latent:
      return token == "expert" || token == "expert_fc1_latent";
    case ScaledFp8RuntimeOpFamily::kExpertSharedUp:
      return token == "expert" || token == "expert_shared_up";
    case ScaledFp8RuntimeOpFamily::kExpertSharedDown:
      return token == "expert" || token == "expert_shared_down";
    case ScaledFp8RuntimeOpFamily::kOther:
      return token == "other";
  }
  return false;
}

bool ExperimentalScaledFp8SurfaceEnabledForDescriptor(
    const GemmDescriptor& descriptor,
    ScaledFp8RuntimeOpFamily family) {
  const char* filter = ExperimentalScaledFp8SurfaceFamilyFilter();
  if (filter == nullptr || *filter == '\0') {
    return ExperimentalScaledFp8SurfaceTensorMatches(descriptor);
  }

  std::stringstream stream(filter);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::size_t begin = token.find_first_not_of(" \t");
    if (begin == std::string::npos) {
      continue;
    }
    const std::size_t end = token.find_last_not_of(" \t");
    const std::string_view family_token(token.data() + begin, end - begin + 1);
    if (ScaledFp8FamilyFilterMatchesToken(family, family_token)) {
      return ExperimentalScaledFp8SurfaceTensorMatches(descriptor);
    }
  }
  return false;
}

bool ExperimentalFp8NativeEnabledForDescriptor(
    const GemmDescriptor& descriptor,
    ScaledFp8RuntimeOpFamily family) {
  return ExperimentalFp8NativeEnabled() &&
         ExperimentalScaledFp8SurfaceEnabledForDescriptor(descriptor, family);
}

bool ExperimentalScaledFp8DequantizedDenseEnabledForDescriptor(
    const GemmDescriptor& descriptor,
    ScaledFp8RuntimeOpFamily family) {
  return ExperimentalScaledFp8DequantizedDenseEnabled() &&
         ExperimentalScaledFp8SurfaceEnabledForDescriptor(descriptor, family);
}

float ClampScale(float value) {
  constexpr float kMinScale = 1.0f / 1024.0f;
  if (!std::isfinite(value) || value < kMinScale) {
    return kMinScale;
  }
  return value;
}

float DecodeFp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

const char* ScaledFp8FamilyName(ScaledFp8RuntimeOpFamily family) {
  switch (family) {
    case ScaledFp8RuntimeOpFamily::kMambaInProj:
      return "mamba_in_proj";
    case ScaledFp8RuntimeOpFamily::kMambaOutProj:
      return "mamba_out_proj";
    case ScaledFp8RuntimeOpFamily::kExpertFc1Latent:
      return "expert_fc1_latent";
    case ScaledFp8RuntimeOpFamily::kExpertSharedUp:
      return "expert_shared_up";
    case ScaledFp8RuntimeOpFamily::kExpertSharedDown:
      return "expert_shared_down";
    case ScaledFp8RuntimeOpFamily::kOther:
      return "other";
  }
  return "unknown";
}

void LogScaledFp8NativeDiagnosticOnce(const std::string& key, const std::string& message) {
  if (!ExperimentalScaledFp8NativeDebugEnabled()) {
    return;
  }
  static std::mutex mutex;
  static std::unordered_set<std::string> seen;
  const std::lock_guard<std::mutex> lock(mutex);
  if (seen.insert(key).second) {
    std::cerr << message << "\n";
  }
}

std::optional<CublasLtGemmPlan> BuildRuntimeGemmPlan(
    const GemmDescriptor& descriptor,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache) {
  const auto launch_plan = BuildGemmLaunchPlan(descriptor, rows);
  if (!launch_plan.has_value()) {
    return std::nullopt;
  }
  const auto execution = PrepareGemmExecution(*launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    return std::nullopt;
  }
  return BuildCublasLtGemmPlan(*execution);
}

__global__ void QuantizeFp8RoundTripKernel(
    const float* input,
    std::size_t numel,
    float input_scale,
    float* output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const float scale = input_scale > 0.0f ? input_scale : (1.0f / 1024.0f);
  for (std::size_t i = index; i < numel; i += stride) {
    const float normalized = input[i] / scale;
    const std::uint8_t raw = static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3));
    __nv_fp8_e4m3 quantized;
    quantized.__x = raw;
    output[i] = static_cast<float>(quantized) * scale;
  }
}

__global__ void QuantizePackedFp8WeightKernel(
    const float* input,
    std::size_t numel,
    float weight_scale,
    std::uint8_t* output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const float scale = weight_scale > 0.0f ? weight_scale : (1.0f / 1024.0f);
  for (std::size_t i = index; i < numel; i += stride) {
    output[i] = static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(input[i] / scale, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void DequantizePackedFp8WeightKernel(
    const std::uint8_t* input,
    std::size_t numel,
    float weight_scale,
    float* output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = index; i < numel; i += stride) {
    __nv_fp8_e4m3 quantized;
    quantized.__x = input[i];
    output[i] = static_cast<float>(quantized) * weight_scale;
  }
}

bool DequantizePackedFp8WeightToDeviceFp32(
    const DeviceTensorFp8E4M3& input,
    float weight_scale,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!input.valid() || output == nullptr || !output->valid() || input.shape() != output->shape()) {
    return false;
  }
  constexpr int kBlockSize = 256;
  const std::size_t numel = input.numel();
  const int grid_size = static_cast<int>(
      (numel + static_cast<std::size_t>(kBlockSize) - 1u) / static_cast<std::size_t>(kBlockSize));
  DequantizePackedFp8WeightKernel<<<grid_size, kBlockSize, 0, stream>>>(
      input.data(),
      numel,
      weight_scale,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool QuantizeDeviceFp32ToPackedFp8Weight(
    const DeviceDenseWeightFp32& input,
    float weight_scale,
    DeviceTensorFp8E4M3* output,
    cudaStream_t stream) {
  if (!input.valid() ||
      output == nullptr ||
      !output->valid() ||
      !std::isfinite(weight_scale) ||
      weight_scale <= 0.0f ||
      output->shape() != std::vector<std::size_t>{input.output_rows(), input.input_cols()}) {
    return false;
  }
  constexpr int kBlockSize = 256;
  const std::size_t numel = input.numel();
  const int grid_size = static_cast<int>(
      (numel + static_cast<std::size_t>(kBlockSize) - 1u) / static_cast<std::size_t>(kBlockSize));
  QuantizePackedFp8WeightKernel<<<grid_size, kBlockSize, 0, stream>>>(
      input.data(),
      numel,
      weight_scale,
      output->data());
  return CheckCuda(cudaGetLastError());
}

void PopulateDequantizedDescriptor(
    const ScaledFp8LinearConfig& config,
    std::string_view tensor_name,
    const DeviceDenseWeightFp32& weight,
    GemmDescriptor* descriptor) {
  if (!weight.valid() || descriptor == nullptr) {
    return;
  }
  descriptor->tensor_name = ResolveDequantizedTensorName(tensor_name);
  descriptor->op_class = "scaled_fp8_linear_dequantized";
  descriptor->kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor->output_rows = config.output_rows;
  descriptor->input_cols = config.input_cols;
  descriptor->storage_dtype = "fp32";
  descriptor->compute_dtype = "fp32";
  descriptor->layout_tag = "row_major";
  descriptor->alignment_bytes = 16;
  descriptor->packed_data = reinterpret_cast<const std::uint8_t*>(weight.data());
  descriptor->packed_nbytes = config.output_rows * config.input_cols * sizeof(float);
}

}  // namespace

struct ScaledFp8LinearOp::Impl {
  ScaledFp8LinearConfig config;
  ScaledFp8RuntimeOpFamily family = ScaledFp8RuntimeOpFamily::kOther;
  GemmDescriptor descriptor;
  GemmDescriptor dequantized_descriptor;
  std::unique_ptr<DeviceDenseWeightFp32> weight;
  std::unique_ptr<DeviceTensorFp8E4M3> packed_weight;
  std::unique_ptr<DeviceTensorFp32> dequantized_weight_storage;
  mutable std::mutex rows1_plan_mutex;
  mutable bool rows1_plan_attempted = false;
  mutable std::optional<CublasLtGemmPlan> rows1_plan;
  mutable std::mutex rows1_dequantized_plan_mutex;
  mutable bool rows1_dequantized_plan_attempted = false;
  mutable std::optional<CublasLtGemmPlan> rows1_dequantized_plan;
  mutable std::unique_ptr<DeviceTensorFp8E4M3> fp8_activation_scratch_;
  mutable std::unique_ptr<DeviceTensorFp32> quantized_scratch_;
  mutable std::unique_ptr<DeviceTensorFp32> native_input_scale_;
  mutable std::unique_ptr<DeviceTensorFp32> native_weight_scale_;
  mutable bool native_scale_tensors_initialized = false;
};

std::optional<std::vector<float>> DequantizeScaledFp8WeightToHostFp32(
    const ScaledFp8LinearConfig& config) {
  if (config.output_rows == 0 ||
      config.input_cols == 0 ||
      config.packed_weight_data == nullptr ||
      config.packed_weight_nbytes != config.output_rows * config.input_cols ||
      !std::isfinite(config.weight_scale) ||
      config.weight_scale <= 0.0f) {
    return std::nullopt;
  }

  std::vector<float> values(config.output_rows * config.input_cols, 0.0f);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = DecodeFp8(config.packed_weight_data[i]) * config.weight_scale;
  }
  return values;
}

bool QuantizeFp32ToScaledFp8RoundTrip(
    const DeviceTensorFp32& input,
    float input_scale,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!input.valid() || output == nullptr || !output->valid() || input.shape() != output->shape()) {
    return false;
  }
  const std::size_t numel = input.numel();
  if (numel == 0) {
    return false;
  }

  const int block_size = 256;
  const int grid_size = static_cast<int>((numel + static_cast<std::size_t>(block_size) - 1u) / static_cast<std::size_t>(block_size));
  QuantizeFp8RoundTripKernel<<<grid_size, block_size, 0, stream>>>(
      input.data(),
      numel,
      ClampScale(input_scale),
      output->data());
  return CheckCuda(cudaGetLastError());
}

std::unique_ptr<ScaledFp8LinearOp> ScaledFp8LinearOp::Create(const ScaledFp8LinearConfig& config) {
  if (config.output_rows == 0 ||
      config.input_cols == 0 ||
      config.packed_weight_data == nullptr ||
      config.packed_weight_nbytes != config.output_rows * config.input_cols ||
      !std::isfinite(config.weight_scale) ||
      config.weight_scale <= 0.0f) {
    return nullptr;
  }

  GemmDescriptor descriptor;
  descriptor.tensor_name = ResolveScaledFp8TensorName(config);
  descriptor.op_class = "scaled_fp8_linear";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = config.output_rows;
  descriptor.input_cols = config.input_cols;
  descriptor.storage_dtype = "fp8_e4m3fn";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = config.packed_weight_data;
  descriptor.packed_nbytes = config.packed_weight_nbytes;

  auto packed_weight = DeviceTensorFp8E4M3::Create({config.output_rows, config.input_cols});
  if (!packed_weight ||
      !packed_weight->CopyFromHost(config.packed_weight_data, config.packed_weight_nbytes)) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->family = ClassifyScaledFp8RuntimeOpFamily(descriptor.tensor_name);
  impl->descriptor = descriptor;
  impl->packed_weight = std::move(packed_weight);
  return std::unique_ptr<ScaledFp8LinearOp>(new ScaledFp8LinearOp(std::move(impl)));
}

std::unique_ptr<ScaledFp8LinearOp> ScaledFp8LinearOp::CreateView(
    const ScaledFp8LinearConfig& config,
    std::unique_ptr<DeviceDenseWeightFp32> weight_view) {
  return CreateView(config, std::move(weight_view), nullptr);
}

std::unique_ptr<ScaledFp8LinearOp> ScaledFp8LinearOp::CreateView(
    const ScaledFp8LinearConfig& config,
    std::unique_ptr<DeviceDenseWeightFp32> weight_view,
    std::unique_ptr<DeviceTensorFp8E4M3> packed_weight_view) {
  if (config.output_rows == 0 ||
      config.input_cols == 0 ||
      !std::isfinite(config.weight_scale) ||
      config.weight_scale <= 0.0f ||
      (!weight_view && !packed_weight_view)) {
    return nullptr;
  }
  if (weight_view &&
      (!weight_view->valid() ||
       weight_view->output_rows() != config.output_rows ||
       weight_view->input_cols() != config.input_cols)) {
    return nullptr;
  }
  if (packed_weight_view &&
      (!packed_weight_view->valid() ||
       packed_weight_view->shape() !=
           std::vector<std::size_t>{config.output_rows, config.input_cols})) {
    return nullptr;
  }

  std::unique_ptr<DeviceTensorFp8E4M3> synthesized_packed_weight;
  if (weight_view && weight_view->valid() && !packed_weight_view) {
    synthesized_packed_weight = DeviceTensorFp8E4M3::Create({config.output_rows, config.input_cols});
    if (synthesized_packed_weight &&
        synthesized_packed_weight->valid() &&
        QuantizeDeviceFp32ToPackedFp8Weight(
            *weight_view,
            config.weight_scale,
            synthesized_packed_weight.get(),
            nullptr) &&
        CheckCuda(cudaStreamSynchronize(nullptr))) {
      packed_weight_view = std::move(synthesized_packed_weight);
      LogScaledFp8NativeDiagnosticOnce(
          "create_view_quantized:" + std::string(ResolveScaledFp8TensorName(config)),
          "scaled_fp8_linear: synthesized packed FP8 weight from FP32 cache tensor=" +
              std::string(ResolveScaledFp8TensorName(config)));
    } else {
      synthesized_packed_weight.reset();
      LogScaledFp8NativeDiagnosticOnce(
          "create_view_quantize_failed:" + std::string(ResolveScaledFp8TensorName(config)),
          "scaled_fp8_linear: failed to synthesize packed FP8 weight tensor=" +
              std::string(ResolveScaledFp8TensorName(config)));
    }
  }

  std::unique_ptr<DeviceTensorFp32> dequantized_weight_storage;
  if ((!weight_view || !weight_view->valid()) &&
      packed_weight_view &&
      packed_weight_view->valid()) {
    dequantized_weight_storage = DeviceTensorFp32::Create({config.output_rows, config.input_cols});
    if (dequantized_weight_storage &&
        dequantized_weight_storage->valid() &&
        DequantizePackedFp8WeightToDeviceFp32(
            *packed_weight_view,
            config.weight_scale,
            dequantized_weight_storage.get(),
            nullptr) &&
        CheckCuda(cudaStreamSynchronize(nullptr))) {
      weight_view = DeviceDenseWeightFp32::CreateView(
          config.output_rows,
          config.input_cols,
          dequantized_weight_storage->data());
      if (!weight_view || !weight_view->valid()) {
        dequantized_weight_storage.reset();
      } else {
        LogScaledFp8NativeDiagnosticOnce(
            "create_view_dequantized:" + std::string(ResolveScaledFp8TensorName(config)),
            "scaled_fp8_linear: synthesized dequantized FP32 weight from native FP8 cache tensor=" +
                std::string(ResolveScaledFp8TensorName(config)));
      }
    } else {
      dequantized_weight_storage.reset();
      LogScaledFp8NativeDiagnosticOnce(
          "create_view_dequantize_failed:" + std::string(ResolveScaledFp8TensorName(config)),
          "scaled_fp8_linear: failed to synthesize dequantized FP32 weight tensor=" +
              std::string(ResolveScaledFp8TensorName(config)));
    }
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->descriptor.tensor_name = ResolveScaledFp8TensorName(config);
  impl->descriptor.op_class = "scaled_fp8_linear";
  impl->descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  impl->descriptor.output_rows = config.output_rows;
  impl->descriptor.input_cols = config.input_cols;
  impl->descriptor.compute_dtype = "fp32";
  impl->descriptor.layout_tag = "row_major";
  impl->descriptor.alignment_bytes = 16;
  impl->family = ClassifyScaledFp8RuntimeOpFamily(impl->descriptor.tensor_name);
  impl->weight = std::move(weight_view);
  impl->packed_weight = std::move(packed_weight_view);
  impl->dequantized_weight_storage = std::move(dequantized_weight_storage);
  if (impl->packed_weight && impl->packed_weight->valid()) {
    impl->descriptor.storage_dtype = "fp8_e4m3fn";
    impl->descriptor.packed_data = impl->packed_weight->data();
    impl->descriptor.packed_nbytes = impl->packed_weight->bytes();
  } else {
    impl->descriptor.storage_dtype = "fp32";
  }
  if (impl->weight && impl->weight->valid()) {
    PopulateDequantizedDescriptor(
        config,
        impl->descriptor.tensor_name,
        *impl->weight,
        &impl->dequantized_descriptor);
  }
  return std::unique_ptr<ScaledFp8LinearOp>(new ScaledFp8LinearOp(std::move(impl)));
}

ScaledFp8LinearOp::ScaledFp8LinearOp(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ScaledFp8LinearOp::ScaledFp8LinearOp(ScaledFp8LinearOp&&) noexcept = default;
ScaledFp8LinearOp& ScaledFp8LinearOp::operator=(ScaledFp8LinearOp&&) noexcept = default;
ScaledFp8LinearOp::~ScaledFp8LinearOp() = default;

bool ScaledFp8LinearOp::valid() const {
  return impl_ != nullptr &&
         ((impl_->packed_weight != nullptr && impl_->packed_weight->valid()) ||
          (impl_->weight != nullptr && impl_->weight->valid()));
}

std::size_t ScaledFp8LinearOp::output_rows() const {
  return impl_ ? impl_->config.output_rows : 0;
}

std::size_t ScaledFp8LinearOp::input_cols() const {
  return impl_ ? impl_->config.input_cols : 0;
}

float ScaledFp8LinearOp::input_scale() const {
  return impl_ ? impl_->config.input_scale : 0.0f;
}

float ScaledFp8LinearOp::weight_scale() const {
  return impl_ ? impl_->config.weight_scale : 0.0f;
}

bool ScaledFp8LinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) const {
  (void)heuristic_cache;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    return false;
  }
  if (activations.shape().size() != 2 || activations.shape()[1] != impl_->config.input_cols) {
    return false;
  }
  if (output->shape().size() != 2 ||
      output->shape()[0] != activations.shape()[0] ||
      output->shape()[1] != impl_->config.output_rows) {
    return false;
  }

  const bool packed_weight_present = impl_->packed_weight != nullptr;
  const bool packed_weight_valid = packed_weight_present && impl_->packed_weight->valid();
  const bool native_default_enabled = ExperimentalFp8NativeEnabled();
  const bool native_surface_enabled =
      ExperimentalScaledFp8SurfaceEnabledForDescriptor(impl_->descriptor, impl_->family);
  const bool native_descriptor_enabled =
      ExperimentalFp8NativeEnabledForDescriptor(impl_->descriptor, impl_->family);
  const bool native_rollout_enabled =
      packed_weight_present && packed_weight_valid && native_descriptor_enabled;
  const bool dequantized_rollout_enabled =
      ExperimentalScaledFp8DequantizedDenseEnabledForDescriptor(impl_->descriptor, impl_->family);
  bool rollout_plan_build_failed = false;

  if (!native_rollout_enabled) {
    std::ostringstream message;
    message << "scaled_fp8_linear: native FP8 guard blocked"
            << " tensor=" << impl_->descriptor.tensor_name
            << " family=" << ScaledFp8FamilyName(impl_->family)
            << " packed_weight_present=" << packed_weight_present
            << " packed_weight_valid=" << packed_weight_valid
            << " native_default_enabled=" << native_default_enabled
            << " native_surface_enabled=" << native_surface_enabled;
    LogScaledFp8NativeDiagnosticOnce(
        "guard:" + impl_->descriptor.tensor_name,
        message.str());
  }

  std::optional<CublasLtGemmPlan> plan;
  if (native_rollout_enabled) {
    if (activations.shape()[0] == 1) {
      std::lock_guard<std::mutex> lock(impl_->rows1_plan_mutex);
      if (!impl_->rows1_plan_attempted) {
        impl_->rows1_plan =
            BuildRuntimeGemmPlan(impl_->descriptor, activations.shape()[0], heuristic_cache);
        impl_->rows1_plan_attempted = true;
      } else if (impl_->rows1_plan.has_value()) {
        RecordScaledFp8PlanCacheHit();
      }
      if (impl_->rows1_plan.has_value()) {
        plan = impl_->rows1_plan;
      } else {
        rollout_plan_build_failed = true;
        LogScaledFp8NativeDiagnosticOnce(
            "plan_build_failed:" + impl_->descriptor.tensor_name,
            "scaled_fp8_linear: native FP8 plan build failed tensor=" +
                impl_->descriptor.tensor_name +
                " family=" + std::string(ScaledFp8FamilyName(impl_->family)));
      }
    } else {
      plan = BuildRuntimeGemmPlan(impl_->descriptor, activations.shape()[0], heuristic_cache);
      if (!plan.has_value()) {
        rollout_plan_build_failed = true;
        LogScaledFp8NativeDiagnosticOnce(
            "plan_build_failed:" + impl_->descriptor.tensor_name,
            "scaled_fp8_linear: native FP8 plan build failed tensor=" +
                impl_->descriptor.tensor_name +
                " family=" + std::string(ScaledFp8FamilyName(impl_->family)));
      }
    }
  }

  if (native_rollout_enabled &&
      impl_->packed_weight &&
      impl_->packed_weight->valid()) {
    if (plan.has_value()) {
      if (!impl_->native_input_scale_ ||
          impl_->native_input_scale_->shape() != std::vector<std::size_t>{1}) {
        impl_->native_input_scale_ = DeviceTensorFp32::Create({1});
      }
      if (!impl_->native_weight_scale_ ||
          impl_->native_weight_scale_->shape() != std::vector<std::size_t>{1}) {
        impl_->native_weight_scale_ = DeviceTensorFp32::Create({1});
      }
      const float input_scale = ClampScale(impl_->config.input_scale);
      const float weight_scale = impl_->config.weight_scale;
      bool native_scales_ready =
          impl_->native_input_scale_ &&
          impl_->native_input_scale_->valid() &&
          impl_->native_weight_scale_ &&
          impl_->native_weight_scale_->valid();
      if (native_scales_ready && !impl_->native_scale_tensors_initialized) {
        native_scales_ready =
            impl_->native_input_scale_->CopyFromHost(&input_scale, 1) &&
            impl_->native_weight_scale_->CopyFromHost(&weight_scale, 1);
        if (native_scales_ready) {
          impl_->native_scale_tensors_initialized = true;
        }
      }
      if (!native_scales_ready) {
        LogScaledFp8NativeDiagnosticOnce(
            "scale_init_failed:" + impl_->descriptor.tensor_name,
            "scaled_fp8_linear: native FP8 scale initialization failed tensor=" +
                impl_->descriptor.tensor_name +
                " family=" + std::string(ScaledFp8FamilyName(impl_->family)));
      } else {
        if (!impl_->fp8_activation_scratch_ ||
            impl_->fp8_activation_scratch_->shape() != activations.shape()) {
          impl_->fp8_activation_scratch_ = DeviceTensorFp8E4M3::Create(activations.shape());
        }
        DeviceTensorFp8E4M3* const fp8_activation_scratch = impl_->fp8_activation_scratch_.get();
        if (fp8_activation_scratch == nullptr || !fp8_activation_scratch->valid()) {
          LogScaledFp8NativeDiagnosticOnce(
              "activation_scratch_failed:" + impl_->descriptor.tensor_name,
              "scaled_fp8_linear: native FP8 activation scratch allocation failed tensor=" +
                  impl_->descriptor.tensor_name +
                  " family=" + std::string(ScaledFp8FamilyName(impl_->family)));
        } else {
          const auto native_stats = RunDenseRowMajorFp8E4M3ToDevice(
              handle,
              *plan,
              *impl_->packed_weight,
              impl_->native_weight_scale_->data(),
              activations,
              input_scale,
              impl_->native_input_scale_->data(),
              output,
              fp8_activation_scratch,
              stream);
          if (native_stats.has_value()) {
            RecordScaledFp8NativeSuccess(impl_->family);
            return true;
          }
          LogScaledFp8NativeDiagnosticOnce(
              "execution_failed:" + impl_->descriptor.tensor_name,
              "scaled_fp8_linear: native FP8 execution failed tensor=" +
                  impl_->descriptor.tensor_name +
                  " family=" + std::string(ScaledFp8FamilyName(impl_->family)));
        }
      }
    }
  }

  if ((!impl_->weight || !impl_->weight->valid()) &&
      impl_->packed_weight &&
      impl_->packed_weight->valid()) {
    if (!impl_->dequantized_weight_storage ||
        impl_->dequantized_weight_storage->shape() !=
            std::vector<std::size_t>{impl_->config.output_rows, impl_->config.input_cols}) {
      impl_->dequantized_weight_storage =
          DeviceTensorFp32::Create({impl_->config.output_rows, impl_->config.input_cols});
    }
    if (!impl_->dequantized_weight_storage ||
        !impl_->dequantized_weight_storage->valid() ||
        !DequantizePackedFp8WeightToDeviceFp32(
            *impl_->packed_weight,
            impl_->config.weight_scale,
            impl_->dequantized_weight_storage.get(),
            stream)) {
      return false;
    }
    impl_->weight = DeviceDenseWeightFp32::CreateView(
        impl_->config.output_rows,
        impl_->config.input_cols,
        impl_->dequantized_weight_storage->data());
    if (!impl_->weight || !impl_->weight->valid()) {
      return false;
    }
    PopulateDequantizedDescriptor(
        impl_->config,
        impl_->descriptor.tensor_name,
        *impl_->weight,
        &impl_->dequantized_descriptor);
    impl_->rows1_dequantized_plan_attempted = false;
    impl_->rows1_dequantized_plan.reset();
  } else if ((!impl_->weight || !impl_->weight->valid()) &&
             impl_->descriptor.packed_data != nullptr &&
             impl_->descriptor.packed_nbytes != 0) {
    impl_->weight = DeviceDenseWeightFp32::Upload(impl_->descriptor, impl_->config.weight_scale);
    if (impl_->weight && impl_->weight->valid()) {
      PopulateDequantizedDescriptor(
          impl_->config,
          impl_->descriptor.tensor_name,
          *impl_->weight,
          &impl_->dequantized_descriptor);
      impl_->rows1_dequantized_plan_attempted = false;
      impl_->rows1_dequantized_plan.reset();
    }
  }

  if (!impl_->quantized_scratch_ ||
      impl_->quantized_scratch_->shape() != activations.shape()) {
    impl_->quantized_scratch_ = DeviceTensorFp32::Create(activations.shape());
  }
  DeviceTensorFp32* const quantized_activations = impl_->quantized_scratch_.get();
  if (quantized_activations == nullptr || !quantized_activations->valid()) {
    return false;
  }
  if (!QuantizeFp32ToScaledFp8RoundTrip(
          activations,
          impl_->config.input_scale,
          quantized_activations,
          stream)) {
    std::cerr << "scaled_fp8_linear: failed to quantize activations on device\n";
    return false;
  }

  std::optional<CublasLtGemmPlan> dequantized_plan;
  if (dequantized_rollout_enabled &&
      impl_->weight &&
      impl_->weight->valid() &&
      impl_->dequantized_descriptor.packed_data != nullptr) {
    if (activations.shape()[0] == 1) {
      std::lock_guard<std::mutex> lock(impl_->rows1_dequantized_plan_mutex);
      if (!impl_->rows1_dequantized_plan_attempted) {
        impl_->rows1_dequantized_plan = BuildRuntimeGemmPlan(
            impl_->dequantized_descriptor, activations.shape()[0], heuristic_cache);
        impl_->rows1_dequantized_plan_attempted = true;
      } else if (impl_->rows1_dequantized_plan.has_value()) {
        RecordScaledFp8PlanCacheHit();
      }
      if (impl_->rows1_dequantized_plan.has_value()) {
        dequantized_plan = impl_->rows1_dequantized_plan;
      } else {
        rollout_plan_build_failed = true;
      }
    } else {
      dequantized_plan = BuildRuntimeGemmPlan(
          impl_->dequantized_descriptor, activations.shape()[0], heuristic_cache);
      if (!dequantized_plan.has_value()) {
        rollout_plan_build_failed = true;
      }
    }
  }

  if (dequantized_rollout_enabled &&
      dequantized_plan.has_value() &&
      impl_->weight &&
      impl_->weight->valid() &&
      RunDenseRowMajorFp32ToDevice(
          handle,
          *dequantized_plan,
          *impl_->weight,
          *quantized_activations,
          output,
          stream)
          .has_value()) {
    RecordScaledFp8DequantizedDenseSuccess();
    return true;
  }

  // Fall back to the reference surface only when the default dequantized-dense
  // path is explicitly disabled or when plan build/execution still fails.
  if (rollout_plan_build_failed) {
    RecordScaledFp8PlanBuildFailure(impl_->family);
  }
  RecordScaledFp8ReferenceFallback(impl_->family);
  return impl_->weight &&
         impl_->weight->valid() &&
         RunDenseRowMajorFp32ReferenceToDevice(
             *impl_->weight,
             *quantized_activations,
             output)
             .has_value();
}

bool ScaledFp8LinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorBf16& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) const {
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    return false;
  }
  if (activations.shape().size() != 2 || activations.shape()[1] != impl_->config.input_cols) {
    return false;
  }
  auto activations_fp32 = DeviceTensorFp32::Create(activations.shape());
  if (!activations_fp32 ||
      !ConvertDeviceBf16ToFp32(
          activations.data(),
          activations.numel(),
          activations_fp32->data(),
          stream)) {
    return false;
  }
  return Run(handle, heuristic_cache, *activations_fp32, output, stream);
}

bool ScaledFp8LinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output,
    cudaStream_t stream) const {
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    return false;
  }
  if (output->shape().size() != 2 ||
      output->shape()[0] != activations.shape()[0] ||
      output->shape()[1] != impl_->config.output_rows) {
    return false;
  }
  auto output_fp32 = DeviceTensorFp32::Create(output->shape());
  if (!output_fp32) {
    return false;
  }
  if (!Run(handle, heuristic_cache, activations, output_fp32.get(), stream)) {
    return false;
  }
  return ConvertDeviceFp32ToBf16(
      output_fp32->data(),
      output_fp32->numel(),
      output->data(),
      stream);
}

}  // namespace nemotron
