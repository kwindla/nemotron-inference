#include "nemotron/scaled_fp8_linear.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool LinearDeviceFastpathEnabled() {
  const char* value = std::getenv("NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH");
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "0") != 0;
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

std::vector<float> CpuMatmulRowMajor(
    const std::vector<float>& activations,
    std::size_t rows,
    const float* weights,
    std::size_t output_rows,
    std::size_t input_cols) {
  std::vector<float> output(rows * output_rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_rows; ++out) {
      // This correctness-first path intentionally uses a higher-accuracy
      // reduction than the generic float32 loop. Small fp8-linear deltas were
      // large enough to get amplified by later NVFP4 expert projections during
      // decode-oracle debugging.
      double accum = 0.0;
      for (std::size_t col = 0; col < input_cols; ++col) {
        accum += static_cast<double>(activations[row * input_cols + col]) *
                 static_cast<double>(weights[out * input_cols + col]);
      }
      output[row * output_rows + out] = static_cast<float>(accum);
    }
  }
  return output;
}

std::optional<CublasLtGemmPlan> BuildRuntimeGemmPlan(
    std::size_t output_rows,
    std::size_t input_cols,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache,
    const float* packed_weight_data) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = "scaled_fp8_linear";
  descriptor.op_class = "scaled_fp8_linear";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = output_rows;
  descriptor.input_cols = input_cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(packed_weight_data);
  descriptor.packed_nbytes = output_rows * input_cols * sizeof(float);
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
    output[i] = static_cast<float>(__nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3)) * scale;
  }
}

}  // namespace

struct ScaledFp8LinearOp::Impl {
  ScaledFp8LinearConfig config;
  std::unique_ptr<DeviceDenseWeightFp32> weight;
  float* host_weight_data = nullptr;

  ~Impl() {
    std::free(host_weight_data);
  }
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
    DeviceTensorFp32* output) {
  if (!input.valid() || output == nullptr || !output->valid() || input.shape() != output->shape()) {
    return false;
  }
  const std::size_t numel = input.numel();
  if (numel == 0) {
    return false;
  }

  const int block_size = 256;
  const int grid_size = static_cast<int>((numel + static_cast<std::size_t>(block_size) - 1u) / static_cast<std::size_t>(block_size));
  QuantizeFp8RoundTripKernel<<<grid_size, block_size>>>(
      input.data(),
      numel,
      ClampScale(input_scale),
      output->data());
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize());
}

std::unique_ptr<ScaledFp8LinearOp> ScaledFp8LinearOp::Create(const ScaledFp8LinearConfig& config) {
  const auto host_weight = DequantizeScaledFp8WeightToHostFp32(config);
  if (!host_weight.has_value()) {
    return nullptr;
  }

  GemmDescriptor descriptor;
  descriptor.tensor_name = "scaled_fp8_linear";
  descriptor.op_class = "scaled_fp8_linear";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = config.output_rows;
  descriptor.input_cols = config.input_cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(host_weight->data());
  descriptor.packed_nbytes = host_weight->size() * sizeof(float);
  auto weight = DeviceDenseWeightFp32::Upload(descriptor);
  if (!weight || !weight->valid()) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->weight = std::move(weight);
  impl->host_weight_data = nullptr;
  void* raw = nullptr;
  if (posix_memalign(&raw, 16, host_weight->size() * sizeof(float)) != 0 || raw == nullptr) {
    return nullptr;
  }
  impl->host_weight_data = reinterpret_cast<float*>(raw);
  std::memcpy(impl->host_weight_data, host_weight->data(), host_weight->size() * sizeof(float));
  return std::unique_ptr<ScaledFp8LinearOp>(new ScaledFp8LinearOp(std::move(impl)));
}

ScaledFp8LinearOp::ScaledFp8LinearOp(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ScaledFp8LinearOp::ScaledFp8LinearOp(ScaledFp8LinearOp&&) noexcept = default;
ScaledFp8LinearOp& ScaledFp8LinearOp::operator=(ScaledFp8LinearOp&&) noexcept = default;
ScaledFp8LinearOp::~ScaledFp8LinearOp() = default;

bool ScaledFp8LinearOp::valid() const {
  return impl_ != nullptr && impl_->weight != nullptr && impl_->weight->valid();
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
    DeviceTensorFp32* output) const {
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

  std::vector<float> host_activations(activations.numel(), 0.0f);
  if (!activations.CopyToHost(host_activations.data(), host_activations.size())) {
    std::cerr << "scaled_fp8_linear: failed to download activations\n";
    return false;
  }
  const float input_scale = ClampScale(impl_->config.input_scale);
  for (float& value : host_activations) {
    value = DecodeFp8(static_cast<std::uint8_t>(
                          __nv_cvt_float_to_fp8(value / input_scale, __NV_SATFINITE, __NV_E4M3))) *
            input_scale;
  }

  if (LinearDeviceFastpathEnabled() && activations.shape()[0] == 1) {
    const auto plan = BuildRuntimeGemmPlan(
        impl_->config.output_rows,
        impl_->config.input_cols,
        activations.shape()[0],
        heuristic_cache,
        impl_->host_weight_data);
    if (plan.has_value()) {
      const auto result = RunDenseRowMajorFp32(handle, *plan, host_activations.data(), activations.shape()[0]);
      if (result.has_value()) {
        if (!output->CopyFromHost(result->output.data(), result->output.size())) {
          std::cerr << "scaled_fp8_linear: failed to upload output\n";
          return false;
        }
        return true;
      }
    }
  } else if (std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr) {
    std::cerr << "scaled_fp8_linear: reference path forced\n";
  }

  const std::vector<float> fallback_output = CpuMatmulRowMajor(
      host_activations,
      activations.shape()[0],
      impl_->host_weight_data,
      impl_->config.output_rows,
      impl_->config.input_cols);
  if (!output->CopyFromHost(fallback_output.data(), fallback_output.size())) {
    std::cerr << "scaled_fp8_linear: failed to upload CPU fallback output\n";
    return false;
  }
  return true;
}

}  // namespace nemotron
