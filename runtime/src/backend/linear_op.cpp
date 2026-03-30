#include "nemotron/linear_op.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace nemotron {

struct UploadedLinearOp::Impl {
  GemmDescriptor descriptor;
  std::unique_ptr<DeviceDenseWeightFp32> dense_weight;
  std::unique_ptr<DeviceNvfp4Weight> nvfp4_weight;
};

namespace {

bool LinearDeviceFastpathEnabled() {
  const char* value = std::getenv("NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH");
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "0") != 0;
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

bool IsFp32Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp32" || storage_dtype == "float32" || storage_dtype == "float";
}

bool IsBf16Storage(const std::string& storage_dtype) {
  return storage_dtype == "bf16" || storage_dtype == "bfloat16";
}

bool IsFp8Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp8_e4m3fn" || storage_dtype == "fp8_e4m3";
}

std::optional<std::vector<float>> ReadDenseWeightToHostFp32(const GemmDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0) {
    return std::nullopt;
  }
  const std::size_t count = descriptor.output_rows * descriptor.input_cols;
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
  if (IsFp8Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(__nv_fp8_e4m3)) {
      return std::nullopt;
    }
    const auto* src = reinterpret_cast<const __nv_fp8_e4m3*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = static_cast<float>(src[i]);
    }
    return values;
  }
  return std::nullopt;
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
      // Correctness-first dense fallback: use a higher-accuracy accumulation
      // surface so offline oracle drift does not get dominated by host
      // reduction noise before it reaches later mixed-precision layers.
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

}  // namespace

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::Create(const GemmDescriptor& descriptor) {
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  switch (descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      impl->dense_weight = DeviceDenseWeightFp32::Upload(descriptor);
      if (!impl->dense_weight || !impl->dense_weight->valid()) {
        return nullptr;
      }
      break;
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      impl->nvfp4_weight = DeviceNvfp4Weight::Upload(descriptor);
      if (!impl->nvfp4_weight || !impl->nvfp4_weight->valid()) {
        return nullptr;
      }
      break;
  }
  return std::unique_ptr<UploadedLinearOp>(new UploadedLinearOp(std::move(impl)));
}

UploadedLinearOp::UploadedLinearOp(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

UploadedLinearOp::UploadedLinearOp(UploadedLinearOp&&) noexcept = default;
UploadedLinearOp& UploadedLinearOp::operator=(UploadedLinearOp&&) noexcept = default;
UploadedLinearOp::~UploadedLinearOp() = default;

bool UploadedLinearOp::valid() const {
  if (!impl_) {
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      return impl_->dense_weight && impl_->dense_weight->valid();
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      return impl_->nvfp4_weight && impl_->nvfp4_weight->valid();
  }
  return false;
}

std::size_t UploadedLinearOp::output_rows() const {
  return impl_ ? impl_->descriptor.output_rows : 0;
}

std::size_t UploadedLinearOp::input_cols() const {
  return impl_ ? impl_->descriptor.input_cols : 0;
}

GemmKernelFamily UploadedLinearOp::kernel_family() const {
  return impl_ ? impl_->descriptor.kernel_family : GemmKernelFamily::kDenseRowMajor;
}

bool UploadedLinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output) const {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    if (debug) {
      std::cerr << "linear_op: invalid run state for " << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      if (LinearDeviceFastpathEnabled()) {
        const auto plan =
            BuildRuntimeGemmPlan(impl_->descriptor, activations.shape().at(0), heuristic_cache);
        if (plan.has_value()) {
          if (const auto stats = RunDenseRowMajorFp32ToDevice(
                  handle,
                  *plan,
                  *impl_->dense_weight,
                  activations,
                  output);
              stats.has_value()) {
            return true;
          }
          if (debug) {
            std::cerr << "linear_op: dense device path failed for "
                      << impl_->descriptor.tensor_name << "\n";
          }
        } else if (debug) {
          std::cerr << "linear_op: plan build failed for " << impl_->descriptor.tensor_name
                    << ", falling back to CPU\n";
        }
      } else if (debug) {
        std::cerr << "linear_op: dense reference path forced for "
                  << impl_->descriptor.tensor_name << "\n";
      }
      break;
    }
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      {
        const auto plan = BuildRuntimeGemmPlan(impl_->descriptor, activations.shape().at(0), heuristic_cache);
        if (!plan.has_value()) {
          if (debug) {
            std::cerr << "linear_op: plan build failed for NVFP4 op "
                      << impl_->descriptor.tensor_name << "\n";
          }
          return false;
        }
        return RunNvfp4RowMajorFp32SourceToDevice(
                   handle,
                   *plan,
                   activations,
                   *impl_->nvfp4_weight,
                   output)
            .has_value();
      }
  }

  if (impl_->descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor) {
    return false;
  }

  const auto host_weights = ReadDenseWeightToHostFp32(impl_->descriptor);
  if (!host_weights.has_value()) {
    if (debug) {
      std::cerr << "linear_op: dense fallback weight decode failed for "
                << impl_->descriptor.tensor_name
                << " storage=" << impl_->descriptor.storage_dtype << "\n";
    }
    return false;
  }
  std::vector<float> host_activations(activations.numel(), 0.0f);
  if (!activations.CopyToHost(host_activations.data(), host_activations.size())) {
    if (debug) {
      std::cerr << "linear_op: dense fallback activation copy failed for "
                << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  const std::vector<float> host_output = CpuMatmulRowMajor(
      host_activations,
      activations.shape()[0],
      *host_weights,
      impl_->descriptor.output_rows,
      impl_->descriptor.input_cols);
  if (!output->CopyFromHost(host_output.data(), host_output.size())) {
    if (debug) {
      std::cerr << "linear_op: dense fallback output upload failed for "
                << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  if (std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr) {
    std::cerr << "linear_op: dense CPU fallback for " << impl_->descriptor.tensor_name << "\n";
  }
  return true;
}

std::optional<std::vector<float>> ReadVectorWeightToHostFp32(
    const KernelTensorDescriptor& descriptor) {
  if (descriptor.logical_shape.size() != 1 || descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  const std::size_t count = descriptor.logical_shape[0];
  if (count == 0) {
    return std::nullopt;
  }

  std::vector<float> values(count, 0.0f);
  if (descriptor.storage_dtype == "fp32") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return values;
  }
  if (descriptor.storage_dtype == "bf16") {
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

std::unique_ptr<DeviceTensorFp32> UploadVectorWeightToDeviceFp32(
    const KernelTensorDescriptor& descriptor) {
  const auto host_values = ReadVectorWeightToHostFp32(descriptor);
  if (!host_values.has_value()) {
    return nullptr;
  }
  auto device = DeviceTensorFp32::Create({host_values->size()});
  if (!device || !device->CopyFromHost(host_values->data(), host_values->size())) {
    return nullptr;
  }
  return device;
}

}  // namespace nemotron
