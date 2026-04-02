#include "nemotron/linear_op.h"

#include <cerrno>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "nemotron/linear_op_counters.h"
#include "nemotron/linear_op_trace.h"
#include "nemotron/linear_reference_kernels.h"

namespace nemotron {

struct UploadedLinearOp::Impl {
  GemmDescriptor descriptor;
  std::unique_ptr<DeviceDenseWeightFp32> dense_weight;
  std::unique_ptr<DeviceNvfp4Weight> nvfp4_weight;
  std::unique_ptr<DeviceNvfp4Matrix> activation_pack;
};

namespace {

constexpr const char* kLinearDeviceFastpathEnvVar =
    "NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH";
constexpr const char* kNvfp4ActivationTensorScaleEnvVar =
    "NEMOTRON_FORWARD_NVFP4_ACTIVATION_TENSOR_SCALE";

bool LinearDeviceFastpathEnabled() {
  const char* value = std::getenv(kLinearDeviceFastpathEnvVar);
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "0") != 0;
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

enum class GemmPlanFailureStep {
  kNone,
  kBuildGemmLaunchPlan,
  kBuildRuntimeLaunchPlan,
  kPrepareGemmExecution,
  kBuildCublasLtGemmPlan,
};

const char* GemmPlanFailureStepName(GemmPlanFailureStep step) {
  switch (step) {
    case GemmPlanFailureStep::kNone:
      return "unknown";
    case GemmPlanFailureStep::kBuildGemmLaunchPlan:
      return "BuildGemmLaunchPlan";
    case GemmPlanFailureStep::kBuildRuntimeLaunchPlan:
      return "BuildRuntimeLaunchPlan";
    case GemmPlanFailureStep::kPrepareGemmExecution:
      return "PrepareGemmExecution";
    case GemmPlanFailureStep::kBuildCublasLtGemmPlan:
      return "BuildCublasLtGemmPlan";
  }
  return "unknown";
}

void SetGemmPlanFailureStep(
    GemmPlanFailureStep* failure_step,
    GemmPlanFailureStep value) {
  if (failure_step != nullptr) {
    *failure_step = value;
  }
}

void LogGemmPlanBuildFailure(
    const GemmDescriptor& descriptor,
    std::size_t rows,
    const char* plan_source,
    GemmPlanFailureStep failure_step,
    std::optional<Nvfp4ScaleLayout> activation_scale_layout = std::nullopt) {
  std::cerr << "linear_op: plan build failed for " << descriptor.tensor_name
            << " M=" << rows
            << " N=" << descriptor.output_rows
            << " K=" << descriptor.input_cols
            << " step=plan_build"
            << " plan_source=" << plan_source
            << " sub_step=" << GemmPlanFailureStepName(failure_step);
  if (activation_scale_layout.has_value()) {
    std::cerr << " activation_scale_layout=" << ToString(*activation_scale_layout);
  }
  std::cerr << "\n";
}

void LogGemmExecuteFailure(
    const GemmDescriptor& descriptor,
    std::size_t rows,
    const char* plan_source,
    std::optional<Nvfp4ScaleLayout> activation_scale_layout = std::nullopt) {
  std::cerr << "linear_op: execute failed for " << descriptor.tensor_name
            << " M=" << rows
            << " N=" << descriptor.output_rows
            << " K=" << descriptor.input_cols
            << " step=execute"
            << " plan_source=" << plan_source;
  if (activation_scale_layout.has_value()) {
    std::cerr << " activation_scale_layout=" << ToString(*activation_scale_layout);
  }
  std::cerr << "\n";
}

Nvfp4PackOptions RuntimeNvfp4PackOptions(
    bool debug,
    const GemmDescriptor& descriptor,
    std::size_t rows) {
  Nvfp4PackOptions options;
  // The row-major cuBLASLt NVFP4 fastpath is numerically stable with the
  // 128x4 activation scale-factor layout, but the 8x4 variant still diverges
  // from the validated reference path on real small-M kernels. Keep 8x4
  // support available in the packer utilities, but force the runtime bridge to
  // use the validated 128x4 layout until the 8x4 execute contract is fixed.
  options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  const char* raw_value = std::getenv(kNvfp4ActivationTensorScaleEnvVar);
  if (raw_value == nullptr || raw_value[0] == '\0') {
    if (debug) {
      std::cerr << "linear_op: NVFP4 activation pack options for "
                << descriptor.tensor_name
                << " M=" << rows
                << " scale_layout=" << ToString(*options.execution_scale_layout)
                << " tensor_scale=dynamic\n";
    }
    return options;
  }
  const auto fixed_tensor_scale = ParsePositiveFloatEnv(kNvfp4ActivationTensorScaleEnvVar);
  if (!fixed_tensor_scale.has_value()) {
    if (debug) {
      std::cerr << "linear_op: ignoring invalid "
                << kNvfp4ActivationTensorScaleEnvVar
                << " for " << descriptor.tensor_name << "\n";
    }
    return options;
  }
  options.fixed_tensor_scale = *fixed_tensor_scale;
  if (debug) {
    std::cerr << "linear_op: NVFP4 activation pack options for "
              << descriptor.tensor_name
              << " M=" << rows
              << " scale_layout=" << ToString(*options.execution_scale_layout)
              << " tensor_scale=fixed:"
              << *fixed_tensor_scale
              << "\n";
  }
  return options;
}

std::optional<CublasLtGemmPlan> BuildDescriptorGemmPlan(
    const GemmDescriptor& descriptor,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache,
    GemmPlanFailureStep* failure_step = nullptr) {
  SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kNone);
  const auto launch_plan = BuildGemmLaunchPlan(descriptor, rows);
  if (!launch_plan.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kBuildGemmLaunchPlan);
    return std::nullopt;
  }
  const auto execution = PrepareGemmExecution(*launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kPrepareGemmExecution);
    return std::nullopt;
  }
  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!plan.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kBuildCublasLtGemmPlan);
  }
  return plan;
}

std::optional<GemmLaunchPlan> BuildRuntimeLaunchPlan(
    const GemmDescriptor& descriptor,
    std::size_t rows,
    ByteRangeView packed_bytes,
    std::optional<ByteRangeView> block_scales_bytes = std::nullopt,
    std::optional<ByteRangeView> tensor_scale_bytes = std::nullopt,
    GemmPlanFailureStep* failure_step = nullptr) {
  const auto launch_plan = BuildGemmLaunchPlan(descriptor, rows);
  if (!launch_plan.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kBuildRuntimeLaunchPlan);
    return std::nullopt;
  }
  SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kNone);
  GemmLaunchPlan runtime_launch_plan = *launch_plan;
  runtime_launch_plan.packed_bytes = packed_bytes;
  if (block_scales_bytes.has_value()) {
    runtime_launch_plan.block_scales_bytes = *block_scales_bytes;
  }
  if (tensor_scale_bytes.has_value()) {
    runtime_launch_plan.tensor_scale_bytes = *tensor_scale_bytes;
  }
  return runtime_launch_plan;
}

std::optional<CublasLtGemmPlan> BuildRuntimeGemmPlan(
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32& weight,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache,
    GemmPlanFailureStep* failure_step = nullptr) {
  SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kNone);
  const auto runtime_launch_plan = BuildRuntimeLaunchPlan(
      descriptor,
      rows,
      ByteRangeView{
          reinterpret_cast<const std::uint8_t*>(weight.data()),
          weight.numel() * sizeof(float),
      },
      std::nullopt,
      std::nullopt,
      failure_step);
  if (!runtime_launch_plan.has_value()) {
    return std::nullopt;
  }
  const auto execution = PrepareGemmExecution(*runtime_launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kPrepareGemmExecution);
    return std::nullopt;
  }
  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!plan.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kBuildCublasLtGemmPlan);
  }
  return plan;
}

std::optional<CublasLtGemmPlan> BuildRuntimeGemmPlan(
    const GemmDescriptor& descriptor,
    const DeviceNvfp4Weight& weight,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache,
    GemmPlanFailureStep* failure_step = nullptr) {
  SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kNone);
  const auto runtime_launch_plan = BuildRuntimeLaunchPlan(
      descriptor,
      rows,
      ByteRangeView{
          weight.packed_data(),
          weight.packed_nbytes(),
      },
      ByteRangeView{
          weight.matmul_block_scales_data(),
          weight.matmul_block_scales_nbytes(),
      },
      ByteRangeView{
          weight.tensor_scale_data(),
          weight.tensor_scale_nbytes(),
      },
      failure_step);
  if (!runtime_launch_plan.has_value()) {
    return std::nullopt;
  }
  const auto execution = PrepareGemmExecution(*runtime_launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kPrepareGemmExecution);
    return std::nullopt;
  }
  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!plan.has_value()) {
    SetGemmPlanFailureStep(failure_step, GemmPlanFailureStep::kBuildCublasLtGemmPlan);
  }
  return plan;
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

void AppendLinearOpTraceEntry(const LinearOpTraceEntry& entry) {
  if (!IsLinearOpTraceEnabled()) {
    return;
  }
  auto& trace = GetLinearOpTrace();
  std::lock_guard<std::mutex> lock(trace.mutex);
  trace.entries.push_back(entry);
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

}  // namespace

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::Create(const GemmDescriptor& descriptor) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  switch (descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      impl->dense_weight = DeviceDenseWeightFp32::Upload(descriptor);
      if (!impl->dense_weight || !impl->dense_weight->valid()) {
        if (debug) {
          std::cerr << "linear_op_create: dense upload failed for " << descriptor.tensor_name
                    << " storage=" << descriptor.storage_dtype
                    << " compute=" << descriptor.compute_dtype
                    << " layout=" << descriptor.layout_tag << "\n";
        }
        return nullptr;
      }
      break;
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      impl->nvfp4_weight = DeviceNvfp4Weight::Upload(descriptor);
      if (!impl->nvfp4_weight || !impl->nvfp4_weight->valid()) {
        if (debug) {
          std::cerr << "linear_op_create: NVFP4 upload failed for " << descriptor.tensor_name
                    << " rows=" << descriptor.output_rows
                    << " cols=" << descriptor.input_cols
                    << " storage=" << descriptor.storage_dtype
                    << " compute=" << descriptor.compute_dtype
                    << " layout=" << descriptor.layout_tag
                    << " packed_nbytes=" << descriptor.packed_nbytes
                    << " block_scale_nbytes=" << descriptor.block_scales_nbytes
                    << " tensor_scale_nbytes=" << descriptor.tensor_scale_nbytes << "\n";
        }
        return nullptr;
      }
      impl->activation_pack = DeviceNvfp4Matrix::Create(
          1,
          descriptor.input_cols,
          ResolveActivationNvfp4ScaleLayout(
              1,
              RuntimeNvfp4PackOptions(debug, descriptor, 1).execution_scale_layout));
      if (!impl->activation_pack || !impl->activation_pack->valid()) {
        if (debug) {
          std::cerr << "linear_op_create: NVFP4 activation pack allocation failed for "
                    << descriptor.tensor_name
                    << " cols=" << descriptor.input_cols << "\n";
        }
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
      return impl_->nvfp4_weight &&
             impl_->nvfp4_weight->valid() &&
             impl_->activation_pack &&
             impl_->activation_pack->valid();
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
  const std::size_t rows = activations.shape().at(0);
  auto& counters = GetLinearOpCounters();
  const bool trace_enabled = IsLinearOpTraceEnabled();
  bool plan_build_ok = false;
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      if (LinearDeviceFastpathEnabled()) {
        GemmPlanFailureStep failure_step = GemmPlanFailureStep::kNone;
        const auto plan = BuildRuntimeGemmPlan(
            impl_->descriptor,
            *impl_->dense_weight,
            rows,
            heuristic_cache,
            &failure_step);
        if (plan.has_value()) {
          plan_build_ok = true;
          counters.dense_fastpath_plan_success.fetch_add(1, std::memory_order_relaxed);
          if (const auto stats = RunDenseRowMajorFp32ToDevice(
                  handle,
                  *plan,
                  *impl_->dense_weight,
                  activations,
                  output);
              stats.has_value()) {
            counters.dense_fastpath_execute.fetch_add(1, std::memory_order_relaxed);
            if (trace_enabled) {
              AppendLinearOpTraceEntry(LinearOpTraceEntry{
                  impl_->descriptor.tensor_name,
                  impl_->descriptor.kernel_family,
                  LinearOpPath::kFastpath,
                  true,
                  true,
              });
            }
            return true;
          }
          counters.dense_fastpath_execute_fail.fetch_add(1, std::memory_order_relaxed);
          if (debug) {
            LogGemmExecuteFailure(impl_->descriptor, rows, "runtime");
          }
        } else {
          counters.dense_fastpath_plan_fail.fetch_add(1, std::memory_order_relaxed);
          if (debug) {
            LogGemmPlanBuildFailure(
                impl_->descriptor,
                rows,
                "runtime",
                failure_step);
          }
        }
      } else if (debug) {
        std::cerr << "linear_op: dense reference path forced for "
                  << impl_->descriptor.tensor_name << "\n";
      }
      break;
    }
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      {
        const Nvfp4PackOptions pack_options =
            RuntimeNvfp4PackOptions(debug, impl_->descriptor, rows);
        const std::optional<Nvfp4ScaleLayout> activation_scale_layout =
            pack_options.execution_scale_layout;
        const bool use_runtime_plan = LinearDeviceFastpathEnabled();
        const char* plan_source = use_runtime_plan ? "runtime" : "descriptor";
        GemmPlanFailureStep failure_step = GemmPlanFailureStep::kNone;
        auto plan = use_runtime_plan
                        ? BuildRuntimeGemmPlan(
                              impl_->descriptor,
                              *impl_->nvfp4_weight,
                              rows,
                              heuristic_cache,
                              &failure_step)
                        : BuildDescriptorGemmPlan(
                              impl_->descriptor,
                              rows,
                              heuristic_cache,
                              &failure_step);
        if (!plan.has_value() && debug) {
          LogGemmPlanBuildFailure(
              impl_->descriptor,
              rows,
              plan_source,
              failure_step,
              activation_scale_layout);
        }
        if (plan.has_value()) {
          plan_build_ok = true;
          counters.nvfp4_fastpath_plan_success.fetch_add(1, std::memory_order_relaxed);
        } else {
          counters.nvfp4_fastpath_plan_fail.fetch_add(1, std::memory_order_relaxed);
        }
        bool execute_ok = false;
        if (plan.has_value()) {
          if (rows == 1) {
            const Nvfp4PackedMatrixDeviceView weight_view =
                MakeNvfp4PackedMatrixDeviceView(*impl_->nvfp4_weight);
            execute_ok =
                weight_view.valid() &&
                impl_->activation_pack != nullptr &&
                impl_->activation_pack->valid() &&
                impl_->activation_pack->PackInto(activations, pack_options) &&
                RunNvfp4RowMajorFp32AccumToDevice(
                    handle,
                    *plan,
                    MakeNvfp4PackedMatrixDeviceView(*impl_->activation_pack),
                    impl_->activation_pack->device_tensor_scale_ptr(),
                    weight_view,
                    weight_view.tensor_scale_data,
                    output)
                    .has_value();
          } else {
            execute_ok = RunNvfp4RowMajorFp32SourceToDevice(
                             handle,
                             *plan,
                             activations,
                             *impl_->nvfp4_weight,
                             output,
                             pack_options)
                             .has_value();
          }
        }
        if (execute_ok) {
          counters.nvfp4_fastpath_execute.fetch_add(1, std::memory_order_relaxed);
          if (trace_enabled) {
            AppendLinearOpTraceEntry(LinearOpTraceEntry{
                impl_->descriptor.tensor_name,
                impl_->descriptor.kernel_family,
                LinearOpPath::kFastpath,
                true,
                true,
            });
          }
          return true;
        }
        if (plan.has_value()) {
          counters.nvfp4_fastpath_execute_fail.fetch_add(1, std::memory_order_relaxed);
        }
        if (debug && plan.has_value()) {
          LogGemmExecuteFailure(
              impl_->descriptor,
              rows,
              plan_source,
              activation_scale_layout);
        }
        break;
      }
  }

  bool device_reference_ok = false;
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      counters.dense_reference_fallback.fetch_add(1, std::memory_order_relaxed);
      device_reference_ok =
          impl_->dense_weight != nullptr &&
          RunDenseRowMajorHighPrecisionReferenceToDevice(
              activations,
              *impl_->dense_weight,
              output);
      break;
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      counters.nvfp4_reference_fallback.fetch_add(1, std::memory_order_relaxed);
      device_reference_ok =
          impl_->nvfp4_weight != nullptr &&
          RunNvfp4RowMajorReferenceToDevice(
              activations,
              *impl_->nvfp4_weight,
              output,
              RuntimeNvfp4PackOptions(debug, impl_->descriptor, rows));
      break;
  }
  if (!device_reference_ok) {
    if (trace_enabled) {
      AppendLinearOpTraceEntry(LinearOpTraceEntry{
          impl_->descriptor.tensor_name,
          impl_->descriptor.kernel_family,
          LinearOpPath::kReference,
          plan_build_ok,
          false,
      });
    }
    if (debug) {
      std::cerr << "linear_op: device reference fallback failed for "
                << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  if (trace_enabled) {
    AppendLinearOpTraceEntry(LinearOpTraceEntry{
        impl_->descriptor.tensor_name,
        impl_->descriptor.kernel_family,
        LinearOpPath::kReference,
        plan_build_ok,
        true,
    });
  }
  if (debug) {
    std::cerr << "linear_op: device reference fallback for "
              << impl_->descriptor.tensor_name << "\n";
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
