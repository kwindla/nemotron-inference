#include "nemotron/scaled_fp8_linear.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <utility>

#include "nemotron/runtime_stats.h"
#include "storage_conversion.h"

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool ExperimentalFp8NativeEnabled() {
  static const bool kEnabled = std::getenv("NEMOTRON_ENABLE_EXPERIMENTAL_FP8_NATIVE") != nullptr;
  return kEnabled;
}

bool ExperimentalScaledFp8DequantizedDenseEnabled() {
  static const bool kEnabled =
      std::getenv("NEMOTRON_DISABLE_SCALED_FP8_DEQUANTIZED_DENSE") == nullptr;
  return kEnabled;
}

const char* ExperimentalScaledFp8SurfaceFamilyFilter() {
  static const char* const kFilter =
      std::getenv("NEMOTRON_EXPERIMENTAL_SCALED_FP8_SURFACE_FAMILY");
  return kFilter;
}

const char* ExperimentalScaledFp8SurfaceTensorFilter() {
  static const char* const kFilter =
      std::getenv("NEMOTRON_EXPERIMENTAL_SCALED_FP8_SURFACE_TENSORS");
  return kFilter;
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

}  // namespace

struct ScaledFp8LinearOp::Impl {
  ScaledFp8LinearConfig config;
  ScaledFp8RuntimeOpFamily family = ScaledFp8RuntimeOpFamily::kOther;
  GemmDescriptor descriptor;
  GemmDescriptor dequantized_descriptor;
  std::unique_ptr<DeviceDenseWeightFp32> weight;
  std::unique_ptr<DeviceTensorFp8E4M3> packed_weight;
  mutable std::mutex rows1_plan_mutex;
  mutable bool rows1_plan_attempted = false;
  mutable std::optional<CublasLtGemmPlan> rows1_plan;
  mutable std::mutex rows1_dequantized_plan_mutex;
  mutable bool rows1_dequantized_plan_attempted = false;
  mutable std::optional<CublasLtGemmPlan> rows1_dequantized_plan;
  mutable std::unique_ptr<DeviceTensorFp32> quantized_scratch_;
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
  if (config.output_rows == 0 ||
      config.input_cols == 0 ||
      !std::isfinite(config.weight_scale) ||
      config.weight_scale <= 0.0f ||
      !weight_view ||
      !weight_view->valid() ||
      weight_view->output_rows() != config.output_rows ||
      weight_view->input_cols() != config.input_cols) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->descriptor.tensor_name = ResolveScaledFp8TensorName(config);
  impl->descriptor.op_class = "scaled_fp8_linear";
  impl->descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  impl->descriptor.output_rows = config.output_rows;
  impl->descriptor.input_cols = config.input_cols;
  impl->descriptor.storage_dtype = "fp32";
  impl->descriptor.compute_dtype = "fp32";
  impl->descriptor.layout_tag = "row_major";
  impl->descriptor.alignment_bytes = 16;
  impl->family = ClassifyScaledFp8RuntimeOpFamily(impl->descriptor.tensor_name);
  impl->weight = std::move(weight_view);
  impl->dequantized_descriptor = impl->descriptor;
  impl->dequantized_descriptor.packed_data =
      reinterpret_cast<const std::uint8_t*>(impl->weight->data());
  impl->dequantized_descriptor.packed_nbytes =
      config.output_rows * config.input_cols * sizeof(float);
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

  const bool native_rollout_enabled =
      impl_->packed_weight &&
      impl_->packed_weight->valid() &&
      ExperimentalFp8NativeEnabledForDescriptor(impl_->descriptor, impl_->family);
  const bool dequantized_rollout_enabled =
      ExperimentalScaledFp8DequantizedDenseEnabledForDescriptor(impl_->descriptor, impl_->family);
  bool rollout_plan_build_failed = false;

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
      }
    } else {
      plan = BuildRuntimeGemmPlan(impl_->descriptor, activations.shape()[0], heuristic_cache);
      if (!plan.has_value()) {
        rollout_plan_build_failed = true;
      }
    }
  }

  if (native_rollout_enabled &&
      impl_->packed_weight &&
      impl_->packed_weight->valid()) {
    if (plan.has_value()) {
      const auto native_stats = RunDenseRowMajorFp8E4M3ToDevice(
          handle,
          *plan,
          *impl_->packed_weight,
          ClampScale(impl_->config.input_scale) * impl_->config.weight_scale,
          activations,
          impl_->config.input_scale,
          output,
          stream);
      if (native_stats.has_value()) {
        RecordScaledFp8NativeSuccess(impl_->family);
        return true;
      }
    }
  }

  if ((!impl_->weight || !impl_->weight->valid()) &&
      impl_->descriptor.packed_data != nullptr &&
      impl_->descriptor.packed_nbytes != 0) {
    impl_->weight = DeviceDenseWeightFp32::Upload(impl_->descriptor, impl_->config.weight_scale);
    if (impl_->weight && impl_->weight->valid()) {
      impl_->dequantized_descriptor.tensor_name =
          ResolveDequantizedTensorName(impl_->descriptor.tensor_name);
      impl_->dequantized_descriptor.op_class = "scaled_fp8_linear_dequantized";
      impl_->dequantized_descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
      impl_->dequantized_descriptor.output_rows = impl_->config.output_rows;
      impl_->dequantized_descriptor.input_cols = impl_->config.input_cols;
      impl_->dequantized_descriptor.storage_dtype = "fp32";
      impl_->dequantized_descriptor.compute_dtype = "fp32";
      impl_->dequantized_descriptor.layout_tag = "row_major";
      impl_->dequantized_descriptor.alignment_bytes = 16;
      impl_->dequantized_descriptor.packed_data =
          reinterpret_cast<const std::uint8_t*>(impl_->weight->data());
      impl_->dequantized_descriptor.packed_nbytes =
          impl_->config.output_rows * impl_->config.input_cols * sizeof(float);
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
