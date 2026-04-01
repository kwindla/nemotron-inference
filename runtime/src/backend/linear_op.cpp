#include "nemotron/linear_op.h"

#include <cuda_bf16.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "nemotron/runtime_stats.h"
#include "storage_conversion.h"

namespace nemotron {

struct UploadedLinearOp::Impl {
  GemmDescriptor descriptor;
  std::unique_ptr<DeviceDenseWeightFp32> dense_weight;
  std::unique_ptr<DeviceTensorBf16> dense_weight_bf16;
  std::unique_ptr<DeviceNvfp4Weight> nvfp4_weight;
  mutable std::mutex dense_rows1_plan_mutex;
  mutable bool dense_rows1_plan_attempted = false;
  mutable std::optional<CublasLtGemmPlan> dense_rows1_plan;
};

namespace {

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

bool IsBf16DenseDescriptor(const GemmDescriptor& descriptor) {
  return descriptor.kernel_family == GemmKernelFamily::kDenseRowMajor &&
         descriptor.output_rows != 0 &&
         descriptor.input_cols != 0 &&
         descriptor.layout_tag == "row_major" &&
         !descriptor.is_scaled() &&
         descriptor.packed_bytes().valid() &&
         (descriptor.storage_dtype == "bf16" || descriptor.storage_dtype == "bfloat16") &&
         descriptor.packed_nbytes ==
             descriptor.output_rows * descriptor.input_cols * sizeof(__nv_bfloat16);
}

bool ExperimentalDenseDevicePlanSurfaceEnabled() {
  static const bool disabled =
      std::getenv("NEMOTRON_DISABLE_DENSE_DEVICE_PLAN_SURFACE") != nullptr;
  return !disabled;
}

const char* ExperimentalDenseDevicePlanSurfaceFamilyFilter() {
  return std::getenv("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_FAMILY");
}

const char* ExperimentalDenseDevicePlanSurfaceTensorFilter() {
  return std::getenv("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_TENSORS");
}

DenseRuntimeOpFamily ClassifyDenseRuntimeOpFamily(const GemmDescriptor& descriptor) {
  const std::string_view op_class = descriptor.op_class;
  const std::string_view tensor_name = descriptor.tensor_name;
  const auto contains = [](std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
  };
  if (contains(op_class, "attention") ||
      contains(tensor_name, "q_proj") ||
      contains(tensor_name, "k_proj") ||
      contains(tensor_name, "v_proj") ||
      contains(tensor_name, "o_proj")) {
    return DenseRuntimeOpFamily::kAttention;
  }
  if (contains(op_class, "expert") ||
      contains(op_class, "moe") ||
      contains(tensor_name, "gate.weight") ||
      contains(tensor_name, "fc1_latent_proj.weight") ||
      contains(tensor_name, "fc2_latent_proj.weight") ||
      contains(tensor_name, "shared_expert") ||
      contains(tensor_name, ".experts.")) {
    return DenseRuntimeOpFamily::kExpert;
  }
  return DenseRuntimeOpFamily::kOther;
}

bool ExperimentalDenseDevicePlanSurfaceTensorMatches(const GemmDescriptor& descriptor) {
  const char* filter = ExperimentalDenseDevicePlanSurfaceTensorFilter();
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

bool ExperimentalDenseDevicePlanSurfaceEnabledForDescriptor(
    const GemmDescriptor& descriptor,
    DenseRuntimeOpFamily family) {
  if (!ExperimentalDenseDevicePlanSurfaceEnabled()) {
    return false;
  }
  const char* filter = ExperimentalDenseDevicePlanSurfaceFamilyFilter();
  if (filter == nullptr || *filter == '\0') {
    return true;
  }
  const std::string_view filter_view(filter);
  if (filter_view == "all") {
    return true;
  }
  switch (family) {
    case DenseRuntimeOpFamily::kAttention:
      return filter_view == "attention";
    case DenseRuntimeOpFamily::kExpert:
      return filter_view == "expert";
    case DenseRuntimeOpFamily::kOther:
      if (filter_view != "other") {
        return false;
      }
      break;
  }
  return ExperimentalDenseDevicePlanSurfaceTensorMatches(descriptor);
}

std::unique_ptr<DeviceTensorBf16> UploadDenseWeightToDeviceBf16(const GemmDescriptor& descriptor) {
  if (!IsBf16DenseDescriptor(descriptor)) {
    return nullptr;
  }
  auto weight = DeviceTensorBf16::Create({descriptor.output_rows, descriptor.input_cols});
  if (!weight || !weight->CopyFromHost(
                     reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data),
                     descriptor.output_rows * descriptor.input_cols)) {
    return nullptr;
  }
  return weight;
}

}  // namespace

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::Create(const GemmDescriptor& descriptor) {
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  switch (descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      if (IsBf16DenseDescriptor(descriptor)) {
        impl->dense_weight_bf16 = UploadDenseWeightToDeviceBf16(descriptor);
        if (!impl->dense_weight_bf16 || !impl->dense_weight_bf16->valid()) {
          return nullptr;
        }
      } else {
        impl->dense_weight = DeviceDenseWeightFp32::Upload(descriptor);
        if (!impl->dense_weight || !impl->dense_weight->valid()) {
          return nullptr;
        }
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

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::CreateDenseView(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceDenseWeightFp32> weight_view) {
  if (descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor ||
      !weight_view || !weight_view->valid()) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  impl->dense_weight = std::move(weight_view);
  return std::unique_ptr<UploadedLinearOp>(new UploadedLinearOp(std::move(impl)));
}

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::CreateNvfp4View(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceNvfp4Weight> weight_view) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      !weight_view || !weight_view->valid()) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  impl->nvfp4_weight = std::move(weight_view);
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
      return (impl_->dense_weight && impl_->dense_weight->valid()) ||
             (impl_->dense_weight_bf16 && impl_->dense_weight_bf16->valid());
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

const DeviceNvfp4Weight* UploadedLinearOp::nvfp4_weight() const {
  return impl_ ? impl_->nvfp4_weight.get() : nullptr;
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
      const DenseRuntimeOpFamily dense_family = ClassifyDenseRuntimeOpFamily(impl_->descriptor);
      GemmDescriptor runtime_descriptor = impl_->descriptor;
      if (ExperimentalDenseDevicePlanSurfaceEnabledForDescriptor(impl_->descriptor, dense_family)) {
        if (impl_->dense_weight_bf16 && impl_->dense_weight_bf16->valid()) {
          runtime_descriptor.storage_dtype = "bf16";
          runtime_descriptor.packed_data =
              reinterpret_cast<const std::uint8_t*>(impl_->dense_weight_bf16->data());
          runtime_descriptor.packed_nbytes =
              impl_->descriptor.output_rows * impl_->descriptor.input_cols * sizeof(__nv_bfloat16);
        } else if (impl_->dense_weight && impl_->dense_weight->valid()) {
          runtime_descriptor.storage_dtype = "fp32";
          runtime_descriptor.packed_data =
              reinterpret_cast<const std::uint8_t*>(impl_->dense_weight->data());
          runtime_descriptor.packed_nbytes =
              impl_->descriptor.output_rows * impl_->descriptor.input_cols * sizeof(float);
        }
      }
      std::optional<CublasLtGemmPlan> plan;
      if (activations.shape().at(0) == 1) {
        std::lock_guard<std::mutex> lock(impl_->dense_rows1_plan_mutex);
        if (!impl_->dense_rows1_plan_attempted) {
          impl_->dense_rows1_plan =
              BuildRuntimeGemmPlan(runtime_descriptor, activations.shape().at(0), heuristic_cache);
          impl_->dense_rows1_plan_attempted = true;
          if (!impl_->dense_rows1_plan.has_value()) {
            RecordDensePlanBuildFailure(dense_family);
          }
        } else if (impl_->dense_rows1_plan.has_value()) {
          RecordDensePlanCacheHit();
        }
        if (impl_->dense_rows1_plan.has_value()) {
          plan = impl_->dense_rows1_plan;
        }
      } else {
        plan = BuildRuntimeGemmPlan(runtime_descriptor, activations.shape().at(0), heuristic_cache);
        if (!plan.has_value()) {
          RecordDensePlanBuildFailure(dense_family);
        }
      }
        if (plan.has_value()) {
          if (impl_->dense_weight_bf16 && impl_->dense_weight_bf16->valid()) {
            if (const auto stats = RunDenseRowMajorBf16ToDevice(
                    handle,
                    *plan,
                    *impl_->dense_weight_bf16,
                  activations,
                  output);
              stats.has_value()) {
              RecordDenseNativeSuccess(dense_family);
              return true;
            }
            RecordDenseBf16NativeFailure();
            if (debug) {
              std::cerr << "linear_op: dense BF16 cuBLASLt path failed for "
                        << impl_->descriptor.tensor_name
                        << ", falling back to FP32 reference surface\n";
            }
        } else if (const auto stats = RunDenseRowMajorFp32ToDevice(
                       handle,
                       *plan,
                       *impl_->dense_weight,
                       activations,
                       output);
                   stats.has_value()) {
          RecordDenseNativeSuccess(dense_family);
          return true;
        } else {
          RecordDenseFp32NativeFailure();
        }
        if (debug) {
          std::cerr << "linear_op: dense cuBLASLt path failed for "
                    << impl_->descriptor.tensor_name
                    << ", falling back to device reference\n";
        }
      } else if (debug) {
        std::cerr << "linear_op: dense plan build failed for " << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      }
      if ((!impl_->dense_weight || !impl_->dense_weight->valid()) &&
          IsBf16DenseDescriptor(impl_->descriptor)) {
        impl_->dense_weight = DeviceDenseWeightFp32::Upload(impl_->descriptor);
      }
      RecordDenseReferenceFallback(dense_family);
      return impl_->dense_weight && impl_->dense_weight->valid() &&
             RunDenseRowMajorFp32ReferenceToDevice(*impl_->dense_weight, activations, output)
                 .has_value();
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
  return false;
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
  if (descriptor.logical_shape.size() != 1 || descriptor.packed_data == nullptr) {
    return nullptr;
  }
  const std::size_t count = descriptor.logical_shape[0];
  if (count == 0) {
    return nullptr;
  }

  PackedFloatStorage storage;
  if (descriptor.storage_dtype == "fp32") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return nullptr;
    }
    storage = PackedFloatStorage::kFp32;
  } else if (descriptor.storage_dtype == "bf16") {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return nullptr;
    }
    storage = PackedFloatStorage::kBf16;
  } else {
    return nullptr;
  }

  auto device = DeviceTensorFp32::Create({count});
  if (!device ||
      !UploadPackedFloatToDeviceFp32(
          descriptor.packed_data,
          count,
          storage,
          1.0f,
          device->data())) {
    return nullptr;
  }
  return device;
}

}  // namespace nemotron
