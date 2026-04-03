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
#include <type_traits>
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
  mutable std::unique_ptr<DeviceTensorFp32> bf16_activation_scratch_;
  mutable std::unique_ptr<DeviceTensorFp32> bf16_output_scratch_;
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
  static const char* const kFilter =
      std::getenv("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_FAMILY");
  return kFilter;
}

const char* ExperimentalDenseDevicePlanSurfaceTensorFilter() {
  static const char* const kFilter =
      std::getenv("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_TENSORS");
  return kFilter;
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

GemmDescriptor BuildDenseRuntimeDescriptor(
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    DenseRuntimeOpFamily dense_family) {
  GemmDescriptor runtime_descriptor = descriptor;
  if (ExperimentalDenseDevicePlanSurfaceEnabledForDescriptor(descriptor, dense_family)) {
    if (dense_weight_bf16 != nullptr && dense_weight_bf16->valid()) {
      runtime_descriptor.storage_dtype = "bf16";
      runtime_descriptor.packed_data =
          reinterpret_cast<const std::uint8_t*>(dense_weight_bf16->data());
      runtime_descriptor.packed_nbytes =
          descriptor.output_rows * descriptor.input_cols * sizeof(__nv_bfloat16);
    } else if (dense_weight != nullptr && dense_weight->valid()) {
      runtime_descriptor.storage_dtype = "fp32";
      runtime_descriptor.packed_data =
          reinterpret_cast<const std::uint8_t*>(dense_weight->data());
      runtime_descriptor.packed_nbytes =
          descriptor.output_rows * descriptor.input_cols * sizeof(float);
    }
  }
  return runtime_descriptor;
}

template <typename ActivationTensorT>
std::optional<CublasLtGemmPlan> ResolveDensePlan(
    std::mutex& dense_rows1_plan_mutex,
    bool* dense_rows1_plan_attempted,
    std::optional<CublasLtGemmPlan>* dense_rows1_plan,
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    const ActivationTensorT& activations,
    GemmHeuristicCache* heuristic_cache,
    DenseRuntimeOpFamily dense_family) {
  const GemmDescriptor runtime_descriptor =
      BuildDenseRuntimeDescriptor(descriptor, dense_weight, dense_weight_bf16, dense_family);
  std::optional<CublasLtGemmPlan> plan;
  if (activations.shape().at(0) == 1) {
    std::lock_guard<std::mutex> lock(dense_rows1_plan_mutex);
    if (!*dense_rows1_plan_attempted) {
      *dense_rows1_plan =
          BuildRuntimeGemmPlan(runtime_descriptor, activations.shape().at(0), heuristic_cache);
      *dense_rows1_plan_attempted = true;
      if (!dense_rows1_plan->has_value()) {
        RecordDensePlanBuildFailure(dense_family);
      }
    } else if (dense_rows1_plan->has_value()) {
      RecordDensePlanCacheHit();
    }
    if (dense_rows1_plan->has_value()) {
      plan = *dense_rows1_plan;
    }
  } else {
    plan = BuildRuntimeGemmPlan(runtime_descriptor, activations.shape().at(0), heuristic_cache);
    if (!plan.has_value()) {
      RecordDensePlanBuildFailure(dense_family);
    }
  }
  return plan;
}

bool EnsureDenseWeightFp32(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceDenseWeightFp32>* dense_weight) {
  if (*dense_weight && (*dense_weight)->valid()) {
    return true;
  }
  if (!IsBf16DenseDescriptor(descriptor)) {
    return false;
  }
  *dense_weight = DeviceDenseWeightFp32::Upload(descriptor);
  return *dense_weight && (*dense_weight)->valid();
}

bool EnsureDenseWeightBf16(
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    std::unique_ptr<DeviceTensorBf16>* dense_weight_bf16,
    bool* dense_rows1_plan_attempted,
    std::optional<CublasLtGemmPlan>* dense_rows1_plan) {
  if (*dense_weight_bf16 && (*dense_weight_bf16)->valid()) {
    return true;
  }
  if (dense_weight == nullptr || !dense_weight->valid()) {
    return false;
  }
  auto converted = DeviceTensorBf16::Create({descriptor.output_rows, descriptor.input_cols});
  if (!converted ||
      !ConvertDeviceFp32ToBf16(
          dense_weight->data(),
          descriptor.output_rows * descriptor.input_cols,
          converted->data())) {
    return false;
  }
  *dense_weight_bf16 = std::move(converted);
  *dense_rows1_plan_attempted = false;
  dense_rows1_plan->reset();
  return true;
}

DeviceTensorFp32* EnsureFp32Scratch(
    std::unique_ptr<DeviceTensorFp32>* scratch,
    const std::vector<std::size_t>& shape) {
  if (*scratch && (*scratch)->valid() && (*scratch)->shape() == shape) {
    return scratch->get();
  }
  *scratch = DeviceTensorFp32::Create(shape);
  if (!*scratch || !(*scratch)->valid()) {
    return nullptr;
  }
  return scratch->get();
}

template <typename ActivationTensorT, typename OutputTensorT>
std::optional<DenseRowMajorDeviceStats> RunDenseNativeDispatch(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const ActivationTensorT& activations,
    OutputTensorT* output,
    cudaStream_t stream) {
  if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorFp32> &&
                std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorBf16ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorBf16ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorBf16>) {
    return RunDenseRowMajorBf16ToDevice(handle, plan, weights, activations, output, stream);
  } else {
    return std::nullopt;
  }
}

template <typename ActivationTensorT, typename OutputTensorT>
std::optional<DenseRowMajorDeviceStats> RunDenseNativeDispatch(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const ActivationTensorT& activations,
    OutputTensorT* output,
    cudaStream_t stream) {
  if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorFp32> &&
                std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorFp32ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorFp32ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorBf16>) {
    return RunDenseRowMajorFp32ToDevice(handle, plan, weights, activations, output, stream);
  } else {
    return std::nullopt;
  }
}

template <typename ActivationTensorT, typename OutputTensorT>
bool TryRunDenseNative(
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const ActivationTensorT& activations,
    OutputTensorT* output,
    DenseRuntimeOpFamily dense_family,
    bool debug,
    cudaStream_t stream) {
  if (dense_weight_bf16 != nullptr && dense_weight_bf16->valid()) {
    if (const auto stats = RunDenseNativeDispatch(
            handle, plan, *dense_weight_bf16, activations, output, stream);
        stats.has_value()) {
      RecordDenseNativeSuccess(dense_family);
      return true;
    }
    RecordDenseBf16NativeFailure();
    if (debug) {
      std::cerr << "linear_op: dense BF16 cuBLASLt path failed for "
                << descriptor.tensor_name << "\n";
    }
    return false;
  }
  if (dense_weight != nullptr && dense_weight->valid()) {
    if (const auto stats =
            RunDenseNativeDispatch(handle, plan, *dense_weight, activations, output, stream);
        stats.has_value()) {
      RecordDenseNativeSuccess(dense_family);
      return true;
    }
    RecordDenseFp32NativeFailure();
    if (debug) {
      std::cerr << "linear_op: dense FP32-weight cuBLASLt path failed for "
                << descriptor.tensor_name << "\n";
    }
  }
  return false;
}

template <typename OutputTensorT>
bool RunDenseReferenceFallback(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceDenseWeightFp32>* dense_weight,
    const DeviceTensorFp32& activations,
    OutputTensorT* output,
    std::unique_ptr<DeviceTensorFp32>* output_fp32_scratch,
    cudaStream_t stream) {
  if (!EnsureDenseWeightFp32(descriptor, dense_weight)) {
    return false;
  }
  if constexpr (std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorFp32ReferenceToDevice(**dense_weight, activations, output).has_value();
  } else {
    DeviceTensorFp32* output_fp32 = EnsureFp32Scratch(output_fp32_scratch, output->shape());
    return output_fp32 != nullptr &&
           RunDenseRowMajorFp32ReferenceToDevice(
               **dense_weight,
               activations,
               output_fp32)
               .has_value() &&
           ConvertDeviceFp32ToBf16(
               output_fp32->data(),
               output_fp32->numel(),
               output->data(),
               stream);
  }
}

template <typename OutputTensorT>
bool RunDenseReferenceFallback(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceDenseWeightFp32>* dense_weight,
    const DeviceTensorBf16& activations,
    OutputTensorT* output,
    std::unique_ptr<DeviceTensorFp32>* activations_fp32_scratch,
    std::unique_ptr<DeviceTensorFp32>* output_fp32_scratch,
    cudaStream_t stream) {
  DeviceTensorFp32* activations_fp32 =
      EnsureFp32Scratch(activations_fp32_scratch, activations.shape());
  if (activations_fp32 == nullptr ||
      !ConvertDeviceBf16ToFp32(
          activations.data(),
          activations.numel(),
          activations_fp32->data(),
          stream)) {
    return false;
  }
  return RunDenseReferenceFallback(
      descriptor,
      dense_weight,
      *activations_fp32,
      output,
      output_fp32_scratch,
      stream);
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
    DeviceTensorFp32* output,
    cudaStream_t stream) const {
  static const bool kDebug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool debug = kDebug;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    if (debug) {
      std::cerr << "linear_op: invalid run state for " << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      const DenseRuntimeOpFamily dense_family = ClassifyDenseRuntimeOpFamily(impl_->descriptor);
      const auto plan = ResolveDensePlan(
          impl_->dense_rows1_plan_mutex,
          &impl_->dense_rows1_plan_attempted,
          &impl_->dense_rows1_plan,
          impl_->descriptor,
          impl_->dense_weight.get(),
          impl_->dense_weight_bf16.get(),
          activations,
          heuristic_cache,
          dense_family);
      if (plan.has_value() &&
          TryRunDenseNative(
              impl_->descriptor,
              impl_->dense_weight.get(),
              impl_->dense_weight_bf16.get(),
              handle,
              *plan,
              activations,
              output,
              dense_family,
              debug,
              stream)) {
        return true;
      }
      if (!plan.has_value() && debug) {
        std::cerr << "linear_op: dense plan build failed for " << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      } else if (debug) {
        std::cerr << "linear_op: dense cuBLASLt path failed for "
                  << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      }
      RecordDenseReferenceFallback(dense_family);
      return RunDenseReferenceFallback(
          impl_->descriptor,
          &impl_->dense_weight,
          activations,
          output,
          &impl_->bf16_output_scratch_,
          stream);
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
                   output,
                   {},
                   stream)
            .has_value();
      }
  }
  return false;
}

bool UploadedLinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorBf16& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) const {
  static const bool kDebug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool debug = kDebug;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    if (debug) {
      std::cerr << "linear_op: invalid BF16->FP32 run state for "
                << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      if ((!impl_->dense_weight_bf16 || !impl_->dense_weight_bf16->valid()) &&
          impl_->dense_weight != nullptr &&
          impl_->dense_weight->valid()) {
        DeviceTensorFp32* activations_fp32 =
            EnsureFp32Scratch(&impl_->bf16_activation_scratch_, activations.shape());
        if (activations_fp32 == nullptr ||
            !ConvertDeviceBf16ToFp32(
                activations.data(),
                activations.numel(),
                activations_fp32->data(),
                stream)) {
          return false;
        }
        return Run(handle, heuristic_cache, *activations_fp32, output, stream);
      }
      const DenseRuntimeOpFamily dense_family = ClassifyDenseRuntimeOpFamily(impl_->descriptor);
      const auto plan = ResolveDensePlan(
          impl_->dense_rows1_plan_mutex,
          &impl_->dense_rows1_plan_attempted,
          &impl_->dense_rows1_plan,
          impl_->descriptor,
          impl_->dense_weight.get(),
          impl_->dense_weight_bf16.get(),
          activations,
          heuristic_cache,
          dense_family);
      if (plan.has_value() &&
          TryRunDenseNative(
              impl_->descriptor,
              impl_->dense_weight.get(),
              impl_->dense_weight_bf16.get(),
              handle,
              *plan,
              activations,
              output,
              dense_family,
              debug,
              stream)) {
        return true;
      }
      if (!plan.has_value() && debug) {
        std::cerr << "linear_op: dense BF16-input plan build failed for "
                  << impl_->descriptor.tensor_name << "\n";
      } else if (debug) {
        std::cerr << "linear_op: dense BF16-input cuBLASLt path failed for "
                  << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      }
      RecordDenseReferenceFallback(dense_family);
      return RunDenseReferenceFallback(
          impl_->descriptor,
          &impl_->dense_weight,
          activations,
          output,
          &impl_->bf16_activation_scratch_,
          &impl_->bf16_output_scratch_,
          stream);
    }
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled: {
      DeviceTensorFp32* activations_fp32 =
          EnsureFp32Scratch(&impl_->bf16_activation_scratch_, activations.shape());
      if (activations_fp32 == nullptr ||
          !ConvertDeviceBf16ToFp32(
              activations.data(),
              activations.numel(),
              activations_fp32->data(),
              stream)) {
        return false;
      }
      const auto plan = BuildRuntimeGemmPlan(impl_->descriptor, activations.shape().at(0), heuristic_cache);
      if (!plan.has_value()) {
        if (debug) {
          std::cerr << "linear_op: plan build failed for BF16-input NVFP4 op "
                    << impl_->descriptor.tensor_name << "\n";
        }
        return false;
      }
      return RunNvfp4RowMajorFp32SourceToDevice(
                 handle,
                 *plan,
                 *activations_fp32,
                 *impl_->nvfp4_weight,
                 output,
                 {},
                 stream)
          .has_value();
    }
  }
  return false;
}

bool UploadedLinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output,
    cudaStream_t stream) const {
  static const bool kDebug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool debug = kDebug;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    if (debug) {
      std::cerr << "linear_op: invalid BF16->BF16 run state for "
                << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      if (!EnsureDenseWeightBf16(
              impl_->descriptor,
              impl_->dense_weight.get(),
              &impl_->dense_weight_bf16,
              &impl_->dense_rows1_plan_attempted,
              &impl_->dense_rows1_plan) &&
          debug &&
          impl_->dense_weight != nullptr &&
          impl_->dense_weight->valid()) {
        std::cerr << "linear_op: failed to materialize BF16 dense weight for "
                  << impl_->descriptor.tensor_name << "\n";
      }
      const DenseRuntimeOpFamily dense_family = ClassifyDenseRuntimeOpFamily(impl_->descriptor);
      const auto plan = ResolveDensePlan(
          impl_->dense_rows1_plan_mutex,
          &impl_->dense_rows1_plan_attempted,
          &impl_->dense_rows1_plan,
          impl_->descriptor,
          impl_->dense_weight.get(),
          impl_->dense_weight_bf16.get(),
          activations,
          heuristic_cache,
          dense_family);
      if (plan.has_value() &&
          TryRunDenseNative(
              impl_->descriptor,
              impl_->dense_weight.get(),
              impl_->dense_weight_bf16.get(),
              handle,
              *plan,
              activations,
              output,
              dense_family,
              debug,
              stream)) {
        return true;
      }
      if (!plan.has_value() && debug) {
        std::cerr << "linear_op: dense BF16 surface plan build failed for "
                  << impl_->descriptor.tensor_name << "\n";
      } else if (debug) {
        std::cerr << "linear_op: dense BF16 surface cuBLASLt path failed for "
                  << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      }
      RecordDenseReferenceFallback(dense_family);
      return RunDenseReferenceFallback(
          impl_->descriptor,
          &impl_->dense_weight,
          activations,
          output,
          &impl_->bf16_activation_scratch_,
          &impl_->bf16_output_scratch_,
          stream);
    }
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled: {
      DeviceTensorFp32* output_fp32 =
          EnsureFp32Scratch(&impl_->bf16_output_scratch_, output->shape());
      if (output_fp32 == nullptr) {
        return false;
      }
      if (!Run(handle, heuristic_cache, activations, output_fp32, stream)) {
        return false;
      }
      return ConvertDeviceFp32ToBf16(
          output_fp32->data(),
          output_fp32->numel(),
          output->data(),
          stream);
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
