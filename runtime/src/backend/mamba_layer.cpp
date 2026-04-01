#include "nemotron/mamba_layer.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/linear_op.h"
#include "nemotron/mamba_ops.h"
#include "storage_conversion.h"

namespace nemotron {
namespace {

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
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
  config.tensor_name = weight.tensor_name;
  config.weight_scale = *weight_scale_value;
  config.input_scale = *input_scale_value;
  return config;
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

std::unique_ptr<DeviceTensorFp32> UploadFlatTensorToDeviceFp32(
    const KernelTensorDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr) {
    return nullptr;
  }
  const std::size_t count = NumelFromShape(descriptor.logical_shape);
  if (count == 0) {
    return nullptr;
  }

  PackedFloatStorage storage;
  if (IsFp32Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return nullptr;
    }
    storage = PackedFloatStorage::kFp32;
  } else if (IsBf16Storage(descriptor.storage_dtype)) {
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

}  // namespace

struct MambaLayerSlice::Impl {
  enum class ProjectionFamily {
    kNone,
    kDense,
    kScaledFp8,
  };

  MambaLayerConfig config;
  std::unique_ptr<DeviceTensorFp32> input_norm_weight;
  std::unique_ptr<DeviceTensorFp32> mixer_norm_weight;
  std::unique_ptr<DeviceTensorFp32> conv1d_weight;
  std::unique_ptr<DeviceTensorFp32> conv1d_bias;
  std::unique_ptr<DeviceTensorFp32> A_log;
  std::unique_ptr<DeviceTensorFp32> D;
  std::unique_ptr<DeviceTensorFp32> dt_bias;
  ProjectionFamily in_proj_family = ProjectionFamily::kNone;
  ProjectionFamily out_proj_family = ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> in_proj_dense;
  std::unique_ptr<UploadedLinearOp> out_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> in_proj_scaled_fp8;
  std::unique_ptr<ScaledFp8LinearOp> out_proj_scaled_fp8;
};

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
    const MambaLayerBindings& bindings,
    MambaLayerBuildTimingSink timing_sink) {
  const auto time_ms = [](const auto begin, const auto end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };
  const std::size_t conv_dim = config.intermediate_size + (2 * config.n_groups * config.state_size);
  const std::size_t conv_state_elems = conv_dim * config.conv_kernel_size;
  const std::size_t ssm_state_elems = config.num_heads * config.head_dim * config.state_size;
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
      conv_state_elems == 0 ||
      ssm_state_elems == 0) {
    return nullptr;
  }

  if (!bindings.input_norm_weight ||
      !bindings.mixer_norm_weight ||
      !bindings.conv1d_weight ||
      !bindings.conv1d_bias ||
      !bindings.A_log ||
      !bindings.D ||
      !bindings.dt_bias) {
    return nullptr;
  }
  const auto input_norm_begin = std::chrono::steady_clock::now();
  auto input_norm_weight = UploadVectorWeightToDeviceFp32(*bindings.input_norm_weight);
  const auto input_norm_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("input_norm_upload", time_ms(input_norm_begin, input_norm_end));
  }
  if (!input_norm_weight) {
    return nullptr;
  }

  Impl::ProjectionFamily in_proj_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> in_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> in_proj_scaled_fp8;
  const auto in_proj_begin = std::chrono::steady_clock::now();
  if (bindings.in_proj_kernel_weight != nullptr &&
      bindings.in_proj_weight_scale != nullptr &&
      bindings.in_proj_input_scale != nullptr) {
    const auto in_proj_config_begin = std::chrono::steady_clock::now();
    const auto in_proj_config = BuildScaledFp8LinearConfig(
        *bindings.in_proj_kernel_weight,
        *bindings.in_proj_weight_scale,
        *bindings.in_proj_input_scale);
    const auto in_proj_config_end = std::chrono::steady_clock::now();
    if (timing_sink) {
      timing_sink("in_proj_config", time_ms(in_proj_config_begin, in_proj_config_end));
    }
    if (in_proj_config.has_value()) {
      const auto in_proj_op_begin = std::chrono::steady_clock::now();
      in_proj_scaled_fp8 = ScaledFp8LinearOp::Create(*in_proj_config);
      const auto in_proj_op_end = std::chrono::steady_clock::now();
      if (timing_sink) {
        timing_sink("in_proj_op_create", time_ms(in_proj_op_begin, in_proj_op_end));
      }
      if (!in_proj_scaled_fp8 || !in_proj_scaled_fp8->valid()) {
        return nullptr;
      }
      in_proj_family = Impl::ProjectionFamily::kScaledFp8;
    }
  }
  if (in_proj_family == Impl::ProjectionFamily::kNone) {
    const auto in_proj_op_begin = std::chrono::steady_clock::now();
    in_proj_dense = UploadedLinearOp::Create(*bindings.in_proj_gemm_weight);
    const auto in_proj_op_end = std::chrono::steady_clock::now();
    if (timing_sink) {
      timing_sink("in_proj_op_create", time_ms(in_proj_op_begin, in_proj_op_end));
    }
    if (!in_proj_dense || !in_proj_dense->valid()) {
      return nullptr;
    }
    in_proj_family = Impl::ProjectionFamily::kDense;
  }
  const auto in_proj_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("in_proj_create", time_ms(in_proj_begin, in_proj_end));
  }

  Impl::ProjectionFamily out_proj_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> out_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> out_proj_scaled_fp8;
  const auto out_proj_begin = std::chrono::steady_clock::now();
  if (bindings.out_proj_kernel_weight != nullptr &&
      bindings.out_proj_weight_scale != nullptr &&
      bindings.out_proj_input_scale != nullptr) {
    const auto out_proj_config_begin = std::chrono::steady_clock::now();
    const auto out_proj_config = BuildScaledFp8LinearConfig(
        *bindings.out_proj_kernel_weight,
        *bindings.out_proj_weight_scale,
        *bindings.out_proj_input_scale);
    const auto out_proj_config_end = std::chrono::steady_clock::now();
    if (timing_sink) {
      timing_sink("out_proj_config", time_ms(out_proj_config_begin, out_proj_config_end));
    }
    if (out_proj_config.has_value()) {
      const auto out_proj_op_begin = std::chrono::steady_clock::now();
      out_proj_scaled_fp8 = ScaledFp8LinearOp::Create(*out_proj_config);
      const auto out_proj_op_end = std::chrono::steady_clock::now();
      if (timing_sink) {
        timing_sink("out_proj_op_create", time_ms(out_proj_op_begin, out_proj_op_end));
      }
      if (!out_proj_scaled_fp8 || !out_proj_scaled_fp8->valid()) {
        return nullptr;
      }
      out_proj_family = Impl::ProjectionFamily::kScaledFp8;
    }
  }
  if (out_proj_family == Impl::ProjectionFamily::kNone) {
    const auto out_proj_op_begin = std::chrono::steady_clock::now();
    out_proj_dense = UploadedLinearOp::Create(*bindings.out_proj_gemm_weight);
    const auto out_proj_op_end = std::chrono::steady_clock::now();
    if (timing_sink) {
      timing_sink("out_proj_op_create", time_ms(out_proj_op_begin, out_proj_op_end));
    }
    if (!out_proj_dense || !out_proj_dense->valid()) {
      return nullptr;
    }
    out_proj_family = Impl::ProjectionFamily::kDense;
  }
  const auto out_proj_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("out_proj_create", time_ms(out_proj_begin, out_proj_end));
  }

  if (bindings.conv1d_weight->logical_shape.size() != 3 ||
      bindings.conv1d_weight->logical_shape[0] != conv_dim ||
      bindings.conv1d_weight->logical_shape[1] != 1 ||
      bindings.conv1d_weight->logical_shape[2] != config.conv_kernel_size ||
      NumelFromShape(bindings.mixer_norm_weight->logical_shape) != config.intermediate_size ||
      NumelFromShape(bindings.conv1d_weight->logical_shape) != conv_state_elems ||
      NumelFromShape(bindings.conv1d_bias->logical_shape) != conv_dim ||
      NumelFromShape(bindings.A_log->logical_shape) != config.num_heads ||
      NumelFromShape(bindings.D->logical_shape) != config.num_heads ||
      NumelFromShape(bindings.dt_bias->logical_shape) != config.num_heads) {
    return nullptr;
  }

  const auto state_upload_begin = std::chrono::steady_clock::now();
  const auto mixer_norm_begin = std::chrono::steady_clock::now();
  auto mixer_norm_weight = UploadVectorWeightToDeviceFp32(*bindings.mixer_norm_weight);
  const auto mixer_norm_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("mixer_norm_upload", time_ms(mixer_norm_begin, mixer_norm_end));
  }
  const auto conv1d_weight_begin = std::chrono::steady_clock::now();
  auto conv1d_weight = UploadFlatTensorToDeviceFp32(*bindings.conv1d_weight);
  const auto conv1d_weight_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("conv1d_weight_upload", time_ms(conv1d_weight_begin, conv1d_weight_end));
  }
  const auto conv1d_bias_begin = std::chrono::steady_clock::now();
  auto conv1d_bias = UploadVectorWeightToDeviceFp32(*bindings.conv1d_bias);
  const auto conv1d_bias_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("conv1d_bias_upload", time_ms(conv1d_bias_begin, conv1d_bias_end));
  }
  const auto a_log_begin = std::chrono::steady_clock::now();
  auto A_log = UploadVectorWeightToDeviceFp32(*bindings.A_log);
  const auto a_log_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("a_log_upload", time_ms(a_log_begin, a_log_end));
  }
  const auto d_begin = std::chrono::steady_clock::now();
  auto D = UploadVectorWeightToDeviceFp32(*bindings.D);
  const auto d_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("D_upload", time_ms(d_begin, d_end));
  }
  const auto dt_bias_begin = std::chrono::steady_clock::now();
  auto dt_bias = UploadVectorWeightToDeviceFp32(*bindings.dt_bias);
  const auto dt_bias_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("dt_bias_upload", time_ms(dt_bias_begin, dt_bias_end));
  }
  const auto state_upload_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("state_tensor_uploads", time_ms(state_upload_begin, state_upload_end));
  }
  if (!mixer_norm_weight || !conv1d_weight || !conv1d_bias || !A_log || !D || !dt_bias) {
    return nullptr;
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
    return nullptr;
  }
  MambaLayerPreparedBindings prepared;
  prepared.input_norm_weight = std::move(input_norm_weight);
  prepared.mixer_norm_weight = std::move(mixer_norm_weight);
  prepared.conv1d_weight = std::move(conv1d_weight);
  prepared.conv1d_bias = std::move(conv1d_bias);
  prepared.A_log = std::move(A_log);
  prepared.D = std::move(D);
  prepared.dt_bias = std::move(dt_bias);
  prepared.in_proj_dense = std::move(in_proj_dense);
  prepared.out_proj_dense = std::move(out_proj_dense);
  prepared.in_proj_scaled_fp8 = std::move(in_proj_scaled_fp8);
  prepared.out_proj_scaled_fp8 = std::move(out_proj_scaled_fp8);
  return CreatePrepared(config, std::move(prepared));
}

std::unique_ptr<MambaLayerSlice> MambaLayerSlice::CreatePrepared(
    const MambaLayerConfig& config,
    MambaLayerPreparedBindings bindings) {
  const std::size_t conv_dim = config.intermediate_size + (2 * config.n_groups * config.state_size);
  const std::size_t conv_state_elems = conv_dim * config.conv_kernel_size;
  const std::size_t ssm_state_elems = config.num_heads * config.head_dim * config.state_size;
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
      conv_state_elems == 0 ||
      ssm_state_elems == 0 ||
      !bindings.input_norm_weight || !bindings.input_norm_weight->valid() ||
      !bindings.mixer_norm_weight || !bindings.mixer_norm_weight->valid() ||
      !bindings.conv1d_weight || !bindings.conv1d_weight->valid() ||
      !bindings.conv1d_bias || !bindings.conv1d_bias->valid() ||
      !bindings.A_log || !bindings.A_log->valid() ||
      !bindings.D || !bindings.D->valid() ||
      !bindings.dt_bias || !bindings.dt_bias->valid()) {
    return nullptr;
  }

  Impl::ProjectionFamily in_proj_family = Impl::ProjectionFamily::kNone;
  if (bindings.in_proj_dense && bindings.in_proj_dense->valid()) {
    in_proj_family = Impl::ProjectionFamily::kDense;
  } else if (bindings.in_proj_scaled_fp8 && bindings.in_proj_scaled_fp8->valid()) {
    in_proj_family = Impl::ProjectionFamily::kScaledFp8;
  }
  Impl::ProjectionFamily out_proj_family = Impl::ProjectionFamily::kNone;
  if (bindings.out_proj_dense && bindings.out_proj_dense->valid()) {
    out_proj_family = Impl::ProjectionFamily::kDense;
  } else if (bindings.out_proj_scaled_fp8 && bindings.out_proj_scaled_fp8->valid()) {
    out_proj_family = Impl::ProjectionFamily::kScaledFp8;
  }
  if (in_proj_family == Impl::ProjectionFamily::kNone ||
      out_proj_family == Impl::ProjectionFamily::kNone) {
    return nullptr;
  }

  const std::size_t in_proj_output_rows =
      in_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.in_proj_scaled_fp8->output_rows()
          : bindings.in_proj_dense->output_rows();
  const std::size_t in_proj_input_cols =
      in_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.in_proj_scaled_fp8->input_cols()
          : bindings.in_proj_dense->input_cols();
  const std::size_t out_proj_output_rows =
      out_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.out_proj_scaled_fp8->output_rows()
          : bindings.out_proj_dense->output_rows();
  const std::size_t out_proj_input_cols =
      out_proj_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.out_proj_scaled_fp8->input_cols()
          : bindings.out_proj_dense->input_cols();
  if (in_proj_output_rows != (config.intermediate_size + conv_dim + config.num_heads) ||
      in_proj_input_cols != config.hidden_size ||
      out_proj_output_rows != config.hidden_size ||
      out_proj_input_cols != config.intermediate_size) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->input_norm_weight = std::move(bindings.input_norm_weight);
  impl->mixer_norm_weight = std::move(bindings.mixer_norm_weight);
  impl->conv1d_weight = std::move(bindings.conv1d_weight);
  impl->conv1d_bias = std::move(bindings.conv1d_bias);
  impl->A_log = std::move(bindings.A_log);
  impl->D = std::move(bindings.D);
  impl->dt_bias = std::move(bindings.dt_bias);
  impl->in_proj_family = in_proj_family;
  impl->out_proj_family = out_proj_family;
  impl->in_proj_dense = std::move(bindings.in_proj_dense);
  impl->out_proj_dense = std::move(bindings.out_proj_dense);
  impl->in_proj_scaled_fp8 = std::move(bindings.in_proj_scaled_fp8);
  impl->out_proj_scaled_fp8 = std::move(bindings.out_proj_scaled_fp8);
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
         impl_->mixer_norm_weight != nullptr &&
         impl_->mixer_norm_weight->valid() &&
         impl_->conv1d_weight != nullptr &&
         impl_->conv1d_weight->valid() &&
         impl_->conv1d_bias != nullptr &&
         impl_->conv1d_bias->valid() &&
         impl_->A_log != nullptr &&
         impl_->A_log->valid() &&
         impl_->D != nullptr &&
         impl_->D->valid() &&
         impl_->dt_bias != nullptr &&
         impl_->dt_bias->valid() &&
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
           impl_->out_proj_scaled_fp8->valid()));
}

const MambaLayerConfig& MambaLayerSlice::config() const {
  return impl_->config;
}

bool MambaLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    MambaLayerRunTrace* trace) const {
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

  const std::size_t conv_dim = impl_->config.intermediate_size + (2 * impl_->config.n_groups * impl_->config.state_size);
  const std::size_t projection_size = impl_->config.intermediate_size + conv_dim + impl_->config.num_heads;
  const std::size_t conv_state_elems = conv_dim * impl_->config.conv_kernel_size;
  const std::size_t ssm_state_elems = impl_->config.num_heads * impl_->config.head_dim * impl_->config.state_size;
  if (impl_->config.conv_state_offset_elems + conv_state_elems > request_context.mamba_conv_state()->numel() ||
      impl_->config.ssm_state_offset_elems + ssm_state_elems > request_context.mamba_state()->numel()) {
    return false;
  }

  const std::size_t token_count = input.shape()[0];
  if (token_count == 0) {
    return false;
  }

  std::unique_ptr<DeviceTensorFp32> normalized_local;
  std::unique_ptr<DeviceTensorFp32> projected_local;
  std::unique_ptr<DeviceTensorFp32> scan_output_local;
  std::unique_ptr<DeviceTensorFp32> projected_output_local;
  std::unique_ptr<DeviceTensorFp32> conv_output;
  std::unique_ptr<DeviceTensorFp32> y_output;

  DeviceTensorFp32* normalized = nullptr;
  DeviceTensorFp32* projected = nullptr;
  DeviceTensorFp32* scan_output = nullptr;
  DeviceTensorFp32* projected_output = nullptr;

  if (token_count == 1 &&
      request_context.mamba_normalized_decode() != nullptr &&
      request_context.mamba_projected_decode() != nullptr &&
      request_context.mamba_scan_output_decode() != nullptr &&
      request_context.mamba_projected_output_decode() != nullptr) {
    normalized = request_context.mamba_normalized_decode();
    projected = request_context.mamba_projected_decode();
    scan_output = request_context.mamba_scan_output_decode();
    projected_output = request_context.mamba_projected_output_decode();
  } else {
    normalized_local = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    projected_local = DeviceTensorFp32::Create({token_count, projection_size});
    scan_output_local = DeviceTensorFp32::Create({token_count, impl_->config.intermediate_size});
    projected_output_local = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    normalized = normalized_local.get();
    projected = projected_local.get();
    scan_output = scan_output_local.get();
    projected_output = projected_output_local.get();
  }
  if (token_count != 1) {
    conv_output = DeviceTensorFp32::Create({token_count, conv_dim});
    y_output = DeviceTensorFp32::Create({token_count, impl_->config.intermediate_size});
  }
  if (normalized == nullptr || projected == nullptr || scan_output == nullptr ||
      projected_output == nullptr ||
      (token_count != 1 && (!conv_output || !y_output))) {
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

  if (trace != nullptr) {
    trace->in_proj_output.resize(projected->numel(), 0.0f);
    if (!projected->CopyToHost(trace->in_proj_output.data(), trace->in_proj_output.size())) {
      return false;
    }
  }

  if (token_count == 1) {
    if (!MambaDecodeStepFusedFp32(
            *projected,
            impl_->config.intermediate_size,
            conv_dim,
            impl_->config.num_heads,
            impl_->config.head_dim,
            impl_->config.state_size,
            impl_->config.n_groups,
            impl_->config.conv_kernel_size,
            impl_->config.time_step_min,
            impl_->config.mixer_rms_epsilon,
            impl_->config.conv_state_offset_elems,
            impl_->config.ssm_state_offset_elems,
            *impl_->conv1d_weight,
            *impl_->conv1d_bias,
            *impl_->A_log,
            *impl_->D,
            *impl_->dt_bias,
            *impl_->mixer_norm_weight,
            request_context.mamba_conv_state(),
            request_context.mamba_state(),
            scan_output)) {
      return false;
    }
  } else {
    if (!MambaConv1dSiluUpdateFp32(
            *projected,
            impl_->config.intermediate_size,
            conv_dim,
            impl_->config.conv_kernel_size,
            impl_->config.conv_state_offset_elems,
            *impl_->conv1d_weight,
            *impl_->conv1d_bias,
            request_context.mamba_conv_state(),
            conv_output.get()) ||
        !MambaSsmUpdateFp32(
            *projected,
            *conv_output,
            impl_->config.intermediate_size,
            conv_dim,
            impl_->config.num_heads,
            impl_->config.head_dim,
            impl_->config.state_size,
            impl_->config.n_groups,
            impl_->config.time_step_min,
            impl_->config.ssm_state_offset_elems,
            *impl_->A_log,
            *impl_->D,
            *impl_->dt_bias,
            request_context.mamba_state(),
            y_output.get()) ||
        !GroupedRmsNormGatedFp32(
            *y_output,
            *projected,
            *impl_->mixer_norm_weight,
            impl_->config.n_groups,
            impl_->config.mixer_rms_epsilon,
            scan_output)) {
      return false;
    }
  }

  if (trace != nullptr) {
    trace->scan_output.resize(scan_output->numel(), 0.0f);
    if (!scan_output->CopyToHost(trace->scan_output.data(), trace->scan_output.size())) {
      return false;
    }
  }

  const bool out_proj_ok =
      (impl_->out_proj_family == Impl::ProjectionFamily::kScaledFp8 &&
       impl_->out_proj_scaled_fp8->Run(cublas_handle, heuristic_cache, *scan_output, projected_output)) ||
      (impl_->out_proj_family == Impl::ProjectionFamily::kDense &&
       impl_->out_proj_dense->Run(cublas_handle, heuristic_cache, *scan_output, projected_output));
  if (!out_proj_ok || !ResidualAddFp32(input, *projected_output, output)) {
    return false;
  }

  if (trace != nullptr) {
    trace->projected_output.resize(projected_output->numel(), 0.0f);
    if (!projected_output->CopyToHost(trace->projected_output.data(), trace->projected_output.size())) {
      return false;
    }
  }

  return true;
}

}  // namespace nemotron
