#include "nemotron/mamba_layer.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/fused_mamba_decode.h"
#include "nemotron/linear_op.h"
#include "nemotron/mamba_conv_prefill.h"
#include "nemotron/mamba_gated_group_norm.h"
#include "nemotron/mamba_ssd_prefill.h"

namespace nemotron {
namespace {

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

#if !defined(NEMOTRON_PRODUCTION_BUILD)
bool DebugCompareMambaEnabled() {
  const char* active = std::getenv("NEMOTRON_FORWARD_COMPARE_FUSED_MAMBA_ACTIVE");
  return std::getenv("NEMOTRON_FORWARD_COMPARE_FUSED_MAMBA") != nullptr &&
         (active == nullptr || std::strcmp(active, "0") != 0);
}
#endif

bool DecodeScratchEnabled() {
  const char* value = std::getenv("NEMOTRON_FORWARD_DECODE_SCRATCH");
  return value == nullptr || (value[0] != '\0' && std::string(value) != "0");
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

bool IsFp32Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp32" || storage_dtype == "float32" || storage_dtype == "float";
}

bool IsBf16Storage(const std::string& storage_dtype) {
  return storage_dtype == "bf16" || storage_dtype == "bfloat16";
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

std::unique_ptr<DeviceTensorFp32> UploadHostVectorToDevice(const std::vector<float>& values) {
  if (values.empty()) {
    return nullptr;
  }
  auto device = DeviceTensorFp32::Create({values.size()});
  if (!device || !device->CopyFromHost(values.data(), values.size())) {
    return nullptr;
  }
  return device;
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

float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

float SiLU(float value) {
  return value * Sigmoid(value);
}

float Softplus(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return std::exp(value);
  }
  return std::log1p(std::exp(value));
}

}  // namespace

struct MambaLayerSlice::Impl {
  enum class ProjectionFamily {
    kNone,
    kDense,
    kScaledFp8,
  };

  MambaLayerConfig config;
  MambaLayerStateLayout state_layout;
  std::unique_ptr<DeviceTensorFp32> input_norm_weight;
  std::vector<float> mixer_norm_weight;
  std::unique_ptr<DeviceTensorFp32> mixer_norm_weight_device;
  std::vector<float> conv1d_weight;
  std::unique_ptr<DeviceTensorFp32> conv1d_weight_device;
  std::vector<float> conv1d_bias;
  std::unique_ptr<DeviceTensorFp32> conv1d_bias_device;
  std::vector<float> A_log;
  std::unique_ptr<DeviceTensorFp32> A_log_device;
  std::vector<float> D;
  std::unique_ptr<DeviceTensorFp32> D_device;
  std::vector<float> dt_bias;
  std::unique_ptr<DeviceTensorFp32> dt_bias_device;
  ProjectionFamily in_proj_family = ProjectionFamily::kNone;
  ProjectionFamily out_proj_family = ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> in_proj_dense;
  std::unique_ptr<UploadedLinearOp> out_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> in_proj_scaled_fp8;
  std::unique_ptr<ScaledFp8LinearOp> out_proj_scaled_fp8;
  std::unique_ptr<DeviceTensorFp32> normalized_scratch;
  std::unique_ptr<DeviceTensorFp32> projected_scratch;
  std::unique_ptr<DeviceTensorFp32> scan_output_scratch;
};

std::optional<MambaLayerStateLayout> BuildMambaLayerStateLayout(
    const MambaLayerConfig& config) {
  if (config.intermediate_size == 0 ||
      config.state_size == 0 ||
      config.n_groups == 0 ||
      config.conv_kernel_size == 0 ||
      config.num_heads == 0 ||
      config.head_dim == 0) {
    return std::nullopt;
  }

  const std::size_t conv_dim =
      config.intermediate_size + (2 * config.n_groups * config.state_size);
  const std::size_t conv_state_elems = conv_dim * config.conv_kernel_size;
  const std::size_t ssm_state_elems =
      config.num_heads * config.head_dim * config.state_size;
  if (conv_state_elems == 0 || ssm_state_elems == 0) {
    return std::nullopt;
  }

  MambaLayerStateLayout layout;
  layout.conv_state_offset_elems = config.conv_state_offset_elems;
  layout.conv_state_elems = conv_state_elems;
  layout.ssm_state_offset_elems = config.ssm_state_offset_elems;
  layout.ssm_state_elems = ssm_state_elems;
  return layout;
}

std::optional<MambaLayerBindings> BuildMambaLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog) {
  MambaLayerBindings bindings;
  bindings.input_norm_weight = FindExactKernelBinding(layer, kernel_catalog, "norm.weight");
  bindings.mixer_norm_weight = FindExactKernelBinding(layer, kernel_catalog, "mixer.norm.weight");
  bindings.in_proj_gemm_weight = FindExactGemmBinding(layer, gemm_catalog, "mixer.in_proj.weight");
  bindings.in_proj_kernel_weight = FindExactKernelBinding(layer, kernel_catalog, "mixer.in_proj.weight");
  bindings.in_proj_weight_scale = FindExactKernelBinding(layer, kernel_catalog, "mixer.in_proj.weight_scale");
  bindings.in_proj_input_scale = FindExactKernelBinding(layer, kernel_catalog, "mixer.in_proj.input_scale");
  bindings.conv1d_weight = FindExactKernelBinding(layer, kernel_catalog, "mixer.conv1d.weight");
  bindings.conv1d_bias = FindExactKernelBinding(layer, kernel_catalog, "mixer.conv1d.bias");
  bindings.A_log = FindExactKernelBinding(layer, kernel_catalog, "mixer.A_log");
  bindings.D = FindExactKernelBinding(layer, kernel_catalog, "mixer.D");
  bindings.dt_bias = FindExactKernelBinding(layer, kernel_catalog, "mixer.dt_bias");
  bindings.out_proj_gemm_weight = FindExactGemmBinding(layer, gemm_catalog, "mixer.out_proj.weight");
  bindings.out_proj_kernel_weight = FindExactKernelBinding(layer, kernel_catalog, "mixer.out_proj.weight");
  bindings.out_proj_weight_scale = FindExactKernelBinding(layer, kernel_catalog, "mixer.out_proj.weight_scale");
  bindings.out_proj_input_scale = FindExactKernelBinding(layer, kernel_catalog, "mixer.out_proj.input_scale");

  if (bindings.input_norm_weight == nullptr ||
      bindings.mixer_norm_weight == nullptr ||
      bindings.in_proj_gemm_weight == nullptr ||
      bindings.conv1d_weight == nullptr ||
      bindings.conv1d_bias == nullptr ||
      bindings.A_log == nullptr ||
      bindings.D == nullptr ||
      bindings.dt_bias == nullptr ||
      bindings.out_proj_gemm_weight == nullptr) {
    return std::nullopt;
  }
  return bindings;
}

std::unique_ptr<MambaLayerSlice> MambaLayerSlice::Create(
    const MambaLayerConfig& config,
    const MambaLayerBindings& bindings) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const auto debug_fail = [&](const char* reason) -> std::unique_ptr<MambaLayerSlice> {
    if (debug) {
      std::cerr << "mamba_layer_create: layer " << config.layer_index
                << " failed: " << reason << "\n";
    }
    return nullptr;
  };
  const std::size_t conv_dim = config.intermediate_size + (2 * config.n_groups * config.state_size);
  const auto state_layout = BuildMambaLayerStateLayout(config);
  const std::size_t conv_state_elems =
      state_layout.has_value() ? state_layout->conv_state_elems : 0;
  const std::size_t ssm_state_elems =
      state_layout.has_value() ? state_layout->ssm_state_elems : 0;
  if (config.hidden_size == 0 ||
      config.intermediate_size == 0 ||
      config.num_heads == 0 ||
      config.head_dim == 0 ||
      config.state_size == 0 ||
      config.n_groups == 0 ||
      config.conv_kernel_size == 0 ||
      config.intermediate_size != config.num_heads * config.head_dim ||
      config.intermediate_size % config.n_groups != 0 ||
      config.input_rms_epsilon <= 0.0f ||
      config.mixer_rms_epsilon <= 0.0f ||
      config.time_step_min <= 0.0f ||
      !state_layout.has_value() ||
      conv_state_elems == 0 ||
      ssm_state_elems == 0) {
    return debug_fail("invalid config");
  }

  auto input_norm_weight = UploadVectorWeightToDeviceFp32(*bindings.input_norm_weight);
  auto mixer_norm_weight = ReadTensorToHostFp32(*bindings.mixer_norm_weight);
  auto conv1d_weight = ReadTensorToHostFp32(*bindings.conv1d_weight);
  auto conv1d_bias = ReadTensorToHostFp32(*bindings.conv1d_bias);
  auto A_log = ReadTensorToHostFp32(*bindings.A_log);
  auto D = ReadTensorToHostFp32(*bindings.D);
  auto dt_bias = ReadTensorToHostFp32(*bindings.dt_bias);
  auto mixer_norm_weight_device =
      mixer_norm_weight.has_value() ? UploadHostVectorToDevice(*mixer_norm_weight) : nullptr;
  auto conv1d_weight_device =
      conv1d_weight.has_value() ? UploadHostVectorToDevice(*conv1d_weight) : nullptr;
  auto conv1d_bias_device =
      conv1d_bias.has_value() ? UploadHostVectorToDevice(*conv1d_bias) : nullptr;
  auto A_log_device = A_log.has_value() ? UploadHostVectorToDevice(*A_log) : nullptr;
  auto D_device = D.has_value() ? UploadHostVectorToDevice(*D) : nullptr;
  auto dt_bias_device = dt_bias.has_value() ? UploadHostVectorToDevice(*dt_bias) : nullptr;
  if (!input_norm_weight ||
      !mixer_norm_weight.has_value() ||
      !conv1d_weight.has_value() ||
      !conv1d_bias.has_value() ||
      !A_log.has_value() ||
      !D.has_value() ||
      !dt_bias.has_value() ||
      !mixer_norm_weight_device ||
      !conv1d_weight_device ||
      !conv1d_bias_device ||
      !A_log_device ||
      !D_device ||
      !dt_bias_device) {
    return debug_fail("host/device static tensor materialization failed");
  }

  Impl::ProjectionFamily in_proj_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> in_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> in_proj_scaled_fp8;
  if (bindings.in_proj_kernel_weight != nullptr &&
      bindings.in_proj_weight_scale != nullptr &&
      bindings.in_proj_input_scale != nullptr) {
    const auto in_proj_config = BuildScaledFp8LinearConfig(
        *bindings.in_proj_kernel_weight,
        *bindings.in_proj_weight_scale,
        *bindings.in_proj_input_scale);
    if (in_proj_config.has_value()) {
      in_proj_scaled_fp8 = ScaledFp8LinearOp::Create(*in_proj_config);
      if (!in_proj_scaled_fp8 || !in_proj_scaled_fp8->valid()) {
        return debug_fail("in_proj scaled-fp8 creation failed");
      }
      in_proj_family = Impl::ProjectionFamily::kScaledFp8;
    }
  }
  if (in_proj_family == Impl::ProjectionFamily::kNone) {
    in_proj_dense = UploadedLinearOp::Create(*bindings.in_proj_gemm_weight);
    if (!in_proj_dense || !in_proj_dense->valid()) {
      return debug_fail("in_proj dense creation failed");
    }
    in_proj_family = Impl::ProjectionFamily::kDense;
  }

  Impl::ProjectionFamily out_proj_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> out_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> out_proj_scaled_fp8;
  if (bindings.out_proj_kernel_weight != nullptr &&
      bindings.out_proj_weight_scale != nullptr &&
      bindings.out_proj_input_scale != nullptr) {
    const auto out_proj_config = BuildScaledFp8LinearConfig(
        *bindings.out_proj_kernel_weight,
        *bindings.out_proj_weight_scale,
        *bindings.out_proj_input_scale);
    if (out_proj_config.has_value()) {
      out_proj_scaled_fp8 = ScaledFp8LinearOp::Create(*out_proj_config);
      if (!out_proj_scaled_fp8 || !out_proj_scaled_fp8->valid()) {
        return debug_fail("out_proj scaled-fp8 creation failed");
      }
      out_proj_family = Impl::ProjectionFamily::kScaledFp8;
    }
  }
  if (out_proj_family == Impl::ProjectionFamily::kNone) {
    out_proj_dense = UploadedLinearOp::Create(*bindings.out_proj_gemm_weight);
    if (!out_proj_dense || !out_proj_dense->valid()) {
      return debug_fail("out_proj dense creation failed");
    }
    out_proj_family = Impl::ProjectionFamily::kDense;
  }

  if (bindings.conv1d_weight->logical_shape.size() != 3 ||
      bindings.conv1d_weight->logical_shape[0] != conv_dim ||
      bindings.conv1d_weight->logical_shape[1] != 1 ||
      bindings.conv1d_weight->logical_shape[2] != config.conv_kernel_size ||
      mixer_norm_weight->size() != config.intermediate_size ||
      conv1d_weight->size() != conv_state_elems ||
      conv1d_bias->size() != conv_dim ||
      A_log->size() != config.num_heads ||
      D->size() != config.num_heads ||
      dt_bias->size() != config.num_heads) {
    return debug_fail("static tensor shapes do not match config");
  }

  const std::size_t in_proj_output_rows =
      in_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? in_proj_scaled_fp8->output_rows()
          : in_proj_dense->output_rows();
  const std::size_t in_proj_input_cols =
      in_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? in_proj_scaled_fp8->input_cols()
          : in_proj_dense->input_cols();
  const std::size_t out_proj_output_rows =
      out_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? out_proj_scaled_fp8->output_rows()
          : out_proj_dense->output_rows();
  const std::size_t out_proj_input_cols =
      out_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? out_proj_scaled_fp8->input_cols()
          : out_proj_dense->input_cols();
  if (in_proj_output_rows != (config.intermediate_size + conv_dim + config.num_heads) ||
      in_proj_input_cols != config.hidden_size ||
      out_proj_output_rows != config.hidden_size ||
      out_proj_input_cols != config.intermediate_size) {
    return debug_fail("projection shapes do not match config");
  }

  auto normalized_scratch = DeviceTensorFp32::Create({1, config.hidden_size});
  auto projected_scratch = DeviceTensorFp32::Create({1, in_proj_output_rows});
  auto scan_output_scratch = DeviceTensorFp32::Create({1, config.intermediate_size});
  if (!normalized_scratch || !projected_scratch || !scan_output_scratch) {
    return debug_fail("decode scratch allocation failed");
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->state_layout = *state_layout;
  impl->input_norm_weight = std::move(input_norm_weight);
  impl->mixer_norm_weight = std::move(*mixer_norm_weight);
  impl->mixer_norm_weight_device = std::move(mixer_norm_weight_device);
  impl->conv1d_weight = std::move(*conv1d_weight);
  impl->conv1d_weight_device = std::move(conv1d_weight_device);
  impl->conv1d_bias = std::move(*conv1d_bias);
  impl->conv1d_bias_device = std::move(conv1d_bias_device);
  impl->A_log = std::move(*A_log);
  impl->A_log_device = std::move(A_log_device);
  impl->D = std::move(*D);
  impl->D_device = std::move(D_device);
  impl->dt_bias = std::move(*dt_bias);
  impl->dt_bias_device = std::move(dt_bias_device);
  impl->in_proj_family = in_proj_family;
  impl->out_proj_family = out_proj_family;
  impl->in_proj_dense = std::move(in_proj_dense);
  impl->out_proj_dense = std::move(out_proj_dense);
  impl->in_proj_scaled_fp8 = std::move(in_proj_scaled_fp8);
  impl->out_proj_scaled_fp8 = std::move(out_proj_scaled_fp8);
  impl->normalized_scratch = std::move(normalized_scratch);
  impl->projected_scratch = std::move(projected_scratch);
  impl->scan_output_scratch = std::move(scan_output_scratch);
  return std::unique_ptr<MambaLayerSlice>(new MambaLayerSlice(std::move(impl)));
}

MambaLayerSlice::MambaLayerSlice(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MambaLayerSlice::MambaLayerSlice(MambaLayerSlice&&) noexcept = default;
MambaLayerSlice& MambaLayerSlice::operator=(MambaLayerSlice&&) noexcept = default;
MambaLayerSlice::~MambaLayerSlice() = default;

bool MambaLayerSlice::valid() const {
  return impl_ != nullptr &&
         impl_->input_norm_weight != nullptr &&
         impl_->input_norm_weight->valid() &&
         !impl_->mixer_norm_weight.empty() &&
         impl_->mixer_norm_weight_device != nullptr &&
         impl_->mixer_norm_weight_device->valid() &&
         !impl_->conv1d_weight.empty() &&
         impl_->conv1d_weight_device != nullptr &&
         impl_->conv1d_weight_device->valid() &&
         !impl_->conv1d_bias.empty() &&
         impl_->conv1d_bias_device != nullptr &&
         impl_->conv1d_bias_device->valid() &&
         !impl_->A_log.empty() &&
         impl_->A_log_device != nullptr &&
         impl_->A_log_device->valid() &&
         !impl_->D.empty() &&
         impl_->D_device != nullptr &&
         impl_->D_device->valid() &&
         !impl_->dt_bias.empty() &&
         impl_->dt_bias_device != nullptr &&
         impl_->dt_bias_device->valid() &&
         ((impl_->in_proj_family == Impl::ProjectionFamily::kDense &&
           impl_->in_proj_dense != nullptr &&
           impl_->in_proj_dense->valid()) ||
          (impl_->in_proj_family == Impl::ProjectionFamily::kScaledFp8 &&
           impl_->in_proj_scaled_fp8 != nullptr &&
           impl_->in_proj_scaled_fp8->valid())) &&
         ((impl_->out_proj_family == Impl::ProjectionFamily::kDense &&
           impl_->out_proj_dense != nullptr &&
           impl_->out_proj_dense->valid()) ||
          (impl_->out_proj_family == Impl::ProjectionFamily::kScaledFp8 &&
           impl_->out_proj_scaled_fp8 != nullptr &&
           impl_->out_proj_scaled_fp8->valid())) &&
         impl_->normalized_scratch != nullptr &&
         impl_->normalized_scratch->valid() &&
         impl_->projected_scratch != nullptr &&
         impl_->projected_scratch->valid() &&
         impl_->scan_output_scratch != nullptr &&
         impl_->scan_output_scratch->valid();
}

const MambaLayerConfig& MambaLayerSlice::config() const {
  return impl_->config;
}

const MambaLayerStateLayout& MambaLayerSlice::state_layout() const {
  return impl_->state_layout;
}

bool MambaLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    MambaLayerRunTrace* trace) const {
  struct StateView {
    float* conv_state = nullptr;
    float* ssm_state = nullptr;
    std::size_t conv_state_elems = 0;
    std::size_t ssm_state_elems = 0;

    bool valid() const {
      return conv_state != nullptr &&
             ssm_state != nullptr &&
             conv_state_elems != 0 &&
             ssm_state_elems != 0;
    }
  };

  if (!valid() ||
      !cublas_handle.valid() ||
      !request_context.valid() ||
      request_context.mamba_conv_state() == nullptr ||
      request_context.mamba_state() == nullptr ||
      !request_context.mamba_conv_state()->valid() ||
      !request_context.mamba_state()->valid() ||
      !input.valid() ||
      input.shape().size() != 2 ||
      input.shape()[1] != impl_->config.hidden_size ||
      output == nullptr ||
      !output->valid() ||
      output->shape() != input.shape()) {
    return false;
  }

  const MambaLayerStateLayout& state_layout = impl_->state_layout;
  if (!state_layout.valid() ||
      state_layout.conv_state_offset_elems + state_layout.conv_state_elems >
          request_context.mamba_conv_state()->numel() ||
      state_layout.ssm_state_offset_elems + state_layout.ssm_state_elems >
          request_context.mamba_state()->numel()) {
    return false;
  }
  StateView state_view;
  state_view.conv_state =
      request_context.mamba_conv_state()->data() + state_layout.conv_state_offset_elems;
  state_view.ssm_state =
      request_context.mamba_state()->data() + state_layout.ssm_state_offset_elems;
  state_view.conv_state_elems = state_layout.conv_state_elems;
  state_view.ssm_state_elems = state_layout.ssm_state_elems;
  if (!state_view.valid()) {
    return false;
  }

  const std::size_t token_count = input.shape()[0];
  if (token_count == 0) {
    return false;
  }

  const std::size_t conv_dim =
      impl_->config.intermediate_size + (2 * impl_->config.n_groups * impl_->config.state_size);
  const std::size_t projection_size =
      impl_->config.intermediate_size + conv_dim + impl_->config.num_heads;

  std::unique_ptr<DeviceTensorFp32> normalized_owned;
  std::unique_ptr<DeviceTensorFp32> projected_owned;
  std::unique_ptr<DeviceTensorFp32> scan_output_owned;
  std::unique_ptr<DeviceTensorFp32> projected_output_owned;
  DeviceTensorFp32* normalized = nullptr;
  DeviceTensorFp32* projected = nullptr;
  DeviceTensorFp32* scan_output = nullptr;
  DeviceTensorFp32* projected_output = nullptr;
  const bool use_decode_scratch = token_count == 1 && DecodeScratchEnabled();
  if (use_decode_scratch) {
    normalized = impl_->normalized_scratch.get();
    projected = impl_->projected_scratch.get();
    scan_output = impl_->scan_output_scratch.get();
    projected_output = trace == nullptr ? output : nullptr;
  }
  if (projected_output == nullptr) {
    if (normalized == nullptr) {
      normalized_owned = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    }
    if (projected == nullptr) {
      projected_owned = DeviceTensorFp32::Create({token_count, projection_size});
    }
    if (scan_output == nullptr) {
      scan_output_owned = DeviceTensorFp32::Create({token_count, impl_->config.intermediate_size});
    }
    projected_output_owned = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    if (normalized == nullptr) {
      normalized = normalized_owned.get();
    }
    if (projected == nullptr) {
      projected = projected_owned.get();
    }
    if (scan_output == nullptr) {
      scan_output = scan_output_owned.get();
    }
    projected_output = projected_output_owned.get();
  }
  if (!normalized || !projected || !scan_output || !projected_output) {
    return false;
  }

  if (trace != nullptr) {
    trace->norm_output.clear();
    trace->in_proj_output.clear();
    trace->scan_output.clear();
    trace->projected_output.clear();
  }

  if (!RmsNormFp32(input, *impl_->input_norm_weight, impl_->config.input_rms_epsilon, normalized)) {
    return false;
  }

  if (trace != nullptr) {
    trace->norm_output.resize(normalized->numel(), 0.0f);
    if (!normalized->CopyToHost(trace->norm_output.data(), trace->norm_output.size())) {
      return false;
    }
  }

  const bool in_proj_ok =
      (impl_->in_proj_family == Impl::ProjectionFamily::kScaledFp8 &&
       impl_->in_proj_scaled_fp8->Run(cublas_handle, heuristic_cache, *normalized, projected)) ||
      (impl_->in_proj_family == Impl::ProjectionFamily::kDense &&
       impl_->in_proj_dense->Run(cublas_handle, heuristic_cache, *normalized, projected));
  if (!in_proj_ok) {
    return false;
  }

  const auto run_output_projection = [&]() -> bool {
    const bool out_proj_ok =
        (impl_->out_proj_family == Impl::ProjectionFamily::kScaledFp8 &&
         impl_->out_proj_scaled_fp8->Run(
             cublas_handle, heuristic_cache, *scan_output, projected_output)) ||
        (impl_->out_proj_family == Impl::ProjectionFamily::kDense &&
         impl_->out_proj_dense->Run(
             cublas_handle, heuristic_cache, *scan_output, projected_output));
    if (!out_proj_ok || !ResidualAddFp32(input, *projected_output, output)) {
      return false;
    }
    if (trace != nullptr) {
      trace->projected_output.resize(projected_output->numel(), 0.0f);
      if (!projected_output->CopyToHost(
              trace->projected_output.data(),
              trace->projected_output.size())) {
        return false;
      }
    }
    return true;
  };

  const auto run_sequential_path = [&]() -> bool {
    if (trace != nullptr) {
      trace->in_proj_output.resize(projected->numel(), 0.0f);
      if (!projected->CopyToHost(trace->in_proj_output.data(), trace->in_proj_output.size())) {
        return false;
      }
    }

    auto conv_output = DeviceTensorFp32::Create({token_count, conv_dim});
    if (!conv_output || !conv_output->valid()) {
      return false;
    }

    MambaConvPrefillParams conv_params;
    conv_params.intermediate_size = impl_->config.intermediate_size;
    conv_params.state_size = impl_->config.state_size;
    conv_params.n_groups = impl_->config.n_groups;
    conv_params.conv_kernel_size = impl_->config.conv_kernel_size;
    conv_params.conv_state_elems = state_view.conv_state_elems;
    conv_params.final_conv_state = state_view.conv_state;
    conv_params.initial_conv_state = state_view.conv_state;
    conv_params.conv1d_weight = impl_->conv1d_weight_device->data();
    conv_params.conv1d_bias = impl_->conv1d_bias_device->data();
    if (!RunMambaConvPrefill(conv_params, *projected, conv_output.get())) {
      return false;
    }

    MambaSsdPrefillParams ssd_params;
    ssd_params.intermediate_size = impl_->config.intermediate_size;
    ssd_params.num_heads = impl_->config.num_heads;
    ssd_params.head_dim = impl_->config.head_dim;
    ssd_params.state_size = impl_->config.state_size;
    ssd_params.n_groups = impl_->config.n_groups;
    ssd_params.ssm_state_elems = state_view.ssm_state_elems;
    ssd_params.final_ssm_state = state_view.ssm_state;
    ssd_params.initial_ssm_state = state_view.ssm_state;
    ssd_params.time_step_min = impl_->config.time_step_min;
    ssd_params.A_log = impl_->A_log_device->data();
    ssd_params.D = impl_->D_device->data();
    ssd_params.dt_bias = impl_->dt_bias_device->data();
    if (!RunMambaSsdPrefill(ssd_params, *projected, *conv_output, scan_output)) {
      return false;
    }

    MambaGatedGroupNormParams group_norm_params;
    group_norm_params.intermediate_size = impl_->config.intermediate_size;
    group_norm_params.n_groups = impl_->config.n_groups;
    group_norm_params.mixer_rms_epsilon = impl_->config.mixer_rms_epsilon;
    group_norm_params.mixer_norm_weight = impl_->mixer_norm_weight_device->data();
    if (!RunMambaGatedGroupNorm(group_norm_params, *projected, scan_output)) {
      return false;
    }

    if (trace != nullptr) {
      trace->scan_output.resize(scan_output->numel(), 0.0f);
      if (!scan_output->CopyToHost(trace->scan_output.data(), trace->scan_output.size())) {
        return false;
      }
    }

    return run_output_projection();
  };

#if !defined(NEMOTRON_PRODUCTION_BUILD)
  const auto run_debug_compared_decode = [&]() -> bool {
    std::vector<float> projected_host(projected->numel(), 0.0f);
    std::vector<float> conv_state_host(request_context.mamba_conv_state()->numel(), 0.0f);
    std::vector<float> ssm_state_host(request_context.mamba_state()->numel(), 0.0f);
    if (!projected->CopyToHost(projected_host.data(), projected_host.size()) ||
        !request_context.mamba_conv_state()->CopyToHost(
            conv_state_host.data(),
            conv_state_host.size()) ||
        !request_context.mamba_state()->CopyToHost(
            ssm_state_host.data(),
            ssm_state_host.size())) {
      return false;
    }

    FusedMambaLayerParams fused_params;
    fused_params.intermediate_size = impl_->config.intermediate_size;
    fused_params.num_heads = impl_->config.num_heads;
    fused_params.head_dim = impl_->config.head_dim;
    fused_params.state_size = impl_->config.state_size;
    fused_params.n_groups = impl_->config.n_groups;
    fused_params.conv_kernel_size = impl_->config.conv_kernel_size;
    fused_params.conv_state_elems = state_view.conv_state_elems;
    fused_params.ssm_state_elems = state_view.ssm_state_elems;
    fused_params.conv_state = state_view.conv_state;
    fused_params.ssm_state = state_view.ssm_state;
    fused_params.mixer_rms_epsilon = impl_->config.mixer_rms_epsilon;
    fused_params.time_step_min = impl_->config.time_step_min;
    fused_params.mixer_norm_weight = impl_->mixer_norm_weight_device->data();
    fused_params.conv1d_weight = impl_->conv1d_weight_device->data();
    fused_params.conv1d_bias = impl_->conv1d_bias_device->data();
    fused_params.A_log = impl_->A_log_device->data();
    fused_params.D = impl_->D_device->data();
    fused_params.dt_bias = impl_->dt_bias_device->data();
    if (!RunFusedMambaDecode(fused_params, *projected, scan_output)) {
      return false;
    }

    std::vector<float> fused_scan_output_host(scan_output->numel(), 0.0f);
    if (!scan_output->CopyToHost(fused_scan_output_host.data(), fused_scan_output_host.size())) {
      return false;
    }

    const std::size_t group_width = impl_->config.num_heads / impl_->config.n_groups;
    const std::size_t mixer_group_size =
        impl_->config.intermediate_size / impl_->config.n_groups;
    float* layer_conv_state =
        conv_state_host.data() + state_layout.conv_state_offset_elems;
    float* layer_ssm_state =
        ssm_state_host.data() + state_layout.ssm_state_offset_elems;
    const float* projected_row = projected_host.data();
    const float* gate = projected_row;
    const float* conv_input = gate + impl_->config.intermediate_size;
    const float* dt_pre = conv_input + conv_dim;
    std::vector<float> conv_output(conv_dim, 0.0f);
    std::vector<float> B_expanded(
        impl_->config.num_heads * impl_->config.state_size,
        0.0f);
    std::vector<float> C_expanded(
        impl_->config.num_heads * impl_->config.state_size,
        0.0f);
    std::vector<float> y(impl_->config.intermediate_size, 0.0f);
    std::vector<float> reference_scan_output(
        impl_->config.intermediate_size,
        0.0f);

    for (std::size_t channel = 0; channel < conv_dim; ++channel) {
      float* state_row = layer_conv_state + (channel * impl_->config.conv_kernel_size);
      if (impl_->config.conv_kernel_size > 1) {
        std::memmove(
            state_row,
            state_row + 1,
            (impl_->config.conv_kernel_size - 1) * sizeof(float));
      }
      state_row[impl_->config.conv_kernel_size - 1] = conv_input[channel];

      float accum = impl_->conv1d_bias[channel];
      const float* weight_row =
          impl_->conv1d_weight.data() + (channel * impl_->config.conv_kernel_size);
      for (std::size_t tap = 0; tap < impl_->config.conv_kernel_size; ++tap) {
        accum += state_row[tap] * weight_row[tap];
      }
      conv_output[channel] = SiLU(accum);
    }

    const float* hidden_after_conv = conv_output.data();
    const float* B_grouped = hidden_after_conv + impl_->config.intermediate_size;
    const float* C_grouped =
        B_grouped + (impl_->config.n_groups * impl_->config.state_size);
    for (std::size_t head = 0; head < impl_->config.num_heads; ++head) {
      const std::size_t group = head / group_width;
      std::memcpy(
          B_expanded.data() + (head * impl_->config.state_size),
          B_grouped + (group * impl_->config.state_size),
          impl_->config.state_size * sizeof(float));
      std::memcpy(
          C_expanded.data() + (head * impl_->config.state_size),
          C_grouped + (group * impl_->config.state_size),
          impl_->config.state_size * sizeof(float));
    }

    for (std::size_t head = 0; head < impl_->config.num_heads; ++head) {
      const float A = -std::exp(impl_->A_log[head]);
      const float D = impl_->D[head];
      const float dt_base = dt_pre[head] + impl_->dt_bias[head];
      const float* B_head = B_expanded.data() + (head * impl_->config.state_size);
      const float* C_head = C_expanded.data() + (head * impl_->config.state_size);
      for (std::size_t dim = 0; dim < impl_->config.head_dim; ++dim) {
        const std::size_t hidden_index = (head * impl_->config.head_dim) + dim;
        const float hidden_value = hidden_after_conv[hidden_index];
        const float dt = std::max(Softplus(dt_base), impl_->config.time_step_min);
        const float decay = std::exp(dt * A);
        float accum = 0.0f;
        float* state_row =
            layer_ssm_state + (hidden_index * impl_->config.state_size);
        for (std::size_t state = 0; state < impl_->config.state_size; ++state) {
          const float next =
              state_row[state] * decay + (dt * B_head[state] * hidden_value);
          state_row[state] = next;
          accum += next * C_head[state];
        }
        y[hidden_index] = accum + (hidden_value * D);
      }
    }

    for (std::size_t group = 0; group < impl_->config.n_groups; ++group) {
      const std::size_t begin = group * mixer_group_size;
      const std::size_t end = begin + mixer_group_size;
      float variance = 0.0f;
      for (std::size_t i = begin; i < end; ++i) {
        const float gated = y[i] * SiLU(gate[i]);
        variance += gated * gated;
        reference_scan_output[i] = gated;
      }
      variance /= static_cast<float>(mixer_group_size);
      const float rstd =
          1.0f / std::sqrt(variance + impl_->config.mixer_rms_epsilon);
      for (std::size_t i = begin; i < end; ++i) {
        reference_scan_output[i] =
            reference_scan_output[i] * rstd * impl_->mixer_norm_weight[i];
      }
    }

    float max_abs_diff = 0.0f;
    for (std::size_t i = 0; i < reference_scan_output.size(); ++i) {
      max_abs_diff = std::max(
          max_abs_diff,
          std::fabs(fused_scan_output_host[i] - reference_scan_output[i]));
    }
    std::cerr << "fused_mamba_compare: layer=" << impl_->config.layer_index
              << " scan_max_abs_diff=" << max_abs_diff
              << " fused0="
              << (fused_scan_output_host.empty() ? 0.0f : fused_scan_output_host.front())
              << " host0="
              << (reference_scan_output.empty() ? 0.0f : reference_scan_output.front())
              << "\n";

    return run_output_projection();
  };
#endif

  if (token_count > 1 || trace != nullptr) {
    return run_sequential_path();
  }

#if !defined(NEMOTRON_PRODUCTION_BUILD)
  if (DebugCompareMambaEnabled()) {
    return run_debug_compared_decode();
  }
#endif

  FusedMambaLayerParams fused_params;
  fused_params.intermediate_size = impl_->config.intermediate_size;
  fused_params.num_heads = impl_->config.num_heads;
  fused_params.head_dim = impl_->config.head_dim;
  fused_params.state_size = impl_->config.state_size;
  fused_params.n_groups = impl_->config.n_groups;
  fused_params.conv_kernel_size = impl_->config.conv_kernel_size;
  fused_params.conv_state_elems = state_view.conv_state_elems;
  fused_params.ssm_state_elems = state_view.ssm_state_elems;
  fused_params.conv_state = state_view.conv_state;
  fused_params.ssm_state = state_view.ssm_state;
  fused_params.mixer_rms_epsilon = impl_->config.mixer_rms_epsilon;
  fused_params.time_step_min = impl_->config.time_step_min;
  fused_params.mixer_norm_weight = impl_->mixer_norm_weight_device->data();
  fused_params.conv1d_weight = impl_->conv1d_weight_device->data();
  fused_params.conv1d_bias = impl_->conv1d_bias_device->data();
  fused_params.A_log = impl_->A_log_device->data();
  fused_params.D = impl_->D_device->data();
  fused_params.dt_bias = impl_->dt_bias_device->data();
  if (!RunFusedMambaDecode(fused_params, *projected, scan_output)) {
    return false;
  }
  return run_output_projection();
}

}  // namespace nemotron
