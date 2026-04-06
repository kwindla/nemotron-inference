#include "nemotron/mamba_layer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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

constexpr std::size_t kMambaMultiTokenMaxChunkTokens = 8192;

thread_local MambaLayerExecutionCounters g_mamba_layer_execution_counters;

enum class MambaScratchRole : std::uint8_t {
  kNormalizedBf16,
  kNormalizedFp32,
  kProjected,
  kScanOutput,
  kProjectedOutput,
  kConvOutput,
};

struct CachedMambaScratchKey {
  int device = -1;
  MambaScratchRole role = MambaScratchRole::kNormalizedFp32;
  std::size_t cols = 0;

  bool operator==(const CachedMambaScratchKey& other) const {
    return device == other.device &&
           role == other.role &&
           cols == other.cols;
  }
};

struct CachedMambaScratchKeyHash {
  std::size_t operator()(const CachedMambaScratchKey& key) const {
    std::size_t hash = 0;
    const auto mix = [&](std::size_t value) {
      hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    };
    mix(static_cast<std::size_t>(key.device));
    mix(static_cast<std::size_t>(key.role));
    mix(key.cols);
    return hash;
  }
};

void RecordMambaNativeMultiTokenExecution(std::size_t token_count) {
  if (token_count <= 1) {
    return;
  }
  ++g_mamba_layer_execution_counters.native_multi_token_runs;
  g_mamba_layer_execution_counters.native_multi_token_tokens += token_count;
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

DeviceTensorFp32* GetCachedMambaFp32Scratch(
    MambaScratchRole role,
    std::size_t rows,
    std::size_t cols) {
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess) {
    return nullptr;
  }

  thread_local std::unordered_map<
      CachedMambaScratchKey,
      std::unique_ptr<DeviceTensorFp32>,
      CachedMambaScratchKeyHash>
      cache;

  const CachedMambaScratchKey key{current_device, role, cols};
  auto it = cache.find(key);
  if (it == cache.end() ||
      it->second == nullptr ||
      !it->second->valid() ||
      it->second->shape().size() != 2 ||
      it->second->shape()[1] != cols ||
      it->second->shape()[0] < rows) {
    auto scratch = DeviceTensorFp32::Create({rows, cols});
    if (!scratch || !scratch->valid()) {
      return nullptr;
    }
    it = cache.insert_or_assign(key, std::move(scratch)).first;
  }
  return it->second.get();
}

DeviceTensorBf16* GetCachedMambaBf16Scratch(
    MambaScratchRole role,
    std::size_t rows,
    std::size_t cols) {
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess) {
    return nullptr;
  }

  thread_local std::unordered_map<
      CachedMambaScratchKey,
      std::unique_ptr<DeviceTensorBf16>,
      CachedMambaScratchKeyHash>
      cache;

  const CachedMambaScratchKey key{current_device, role, cols};
  auto it = cache.find(key);
  if (it == cache.end() ||
      it->second == nullptr ||
      !it->second->valid() ||
      it->second->shape().size() != 2 ||
      it->second->shape()[1] != cols ||
      it->second->shape()[0] < rows) {
    auto scratch = DeviceTensorBf16::Create({rows, cols});
    if (!scratch || !scratch->valid()) {
      return nullptr;
    }
    it = cache.insert_or_assign(key, std::move(scratch)).first;
  }
  return it->second.get();
}

}  // namespace

void ResetMambaLayerExecutionCounters() {
  g_mamba_layer_execution_counters = MambaLayerExecutionCounters{};
}

MambaLayerExecutionCounters GetMambaLayerExecutionCounters() {
  return g_mamba_layer_execution_counters;
}

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
  std::unique_ptr<DeviceTensorBf16> normalized_bf16_scratch;
  std::unique_ptr<DeviceTensorFp32> normalized_scratch;
  std::unique_ptr<DeviceTensorFp32> projected_scratch;
  std::unique_ptr<DeviceTensorFp32> scan_output_scratch;
  std::unique_ptr<DeviceTensorFp32> projected_output_scratch;
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
  std::optional<float> in_proj_fixed_activation_tensor_scale;
  if (bindings.in_proj_input_scale != nullptr) {
    in_proj_fixed_activation_tensor_scale = ReadScalarTensorToHostFp32(*bindings.in_proj_input_scale);
    if (!in_proj_fixed_activation_tensor_scale.has_value()) {
      return debug_fail("in_proj input_scale read failed");
    }
  }
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
    in_proj_dense = UploadedLinearOp::Create(
        *bindings.in_proj_gemm_weight,
        in_proj_fixed_activation_tensor_scale);
    if (!in_proj_dense || !in_proj_dense->valid()) {
      return debug_fail("in_proj dense creation failed");
    }
    in_proj_family = Impl::ProjectionFamily::kDense;
  }

  Impl::ProjectionFamily out_proj_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> out_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> out_proj_scaled_fp8;
  std::optional<float> out_proj_fixed_activation_tensor_scale;
  if (bindings.out_proj_input_scale != nullptr) {
    out_proj_fixed_activation_tensor_scale = ReadScalarTensorToHostFp32(*bindings.out_proj_input_scale);
    if (!out_proj_fixed_activation_tensor_scale.has_value()) {
      return debug_fail("out_proj input_scale read failed");
    }
  }
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
    out_proj_dense = UploadedLinearOp::Create(
        *bindings.out_proj_gemm_weight,
        out_proj_fixed_activation_tensor_scale);
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

  auto normalized_bf16_scratch = DeviceTensorBf16::Create({1, config.hidden_size});
  auto normalized_scratch = DeviceTensorFp32::Create({1, config.hidden_size});
  auto projected_scratch = DeviceTensorFp32::Create({1, in_proj_output_rows});
  auto scan_output_scratch = DeviceTensorFp32::Create({1, config.intermediate_size});
  auto projected_output_scratch = DeviceTensorFp32::Create({1, config.hidden_size});
  if (!normalized_bf16_scratch ||
      !normalized_scratch ||
      !projected_scratch ||
      !scan_output_scratch ||
      !projected_output_scratch) {
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
  impl->normalized_bf16_scratch = std::move(normalized_bf16_scratch);
  impl->normalized_scratch = std::move(normalized_scratch);
  impl->projected_scratch = std::move(projected_scratch);
  impl->scan_output_scratch = std::move(scan_output_scratch);
  impl->projected_output_scratch = std::move(projected_output_scratch);
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
         impl_->normalized_bf16_scratch != nullptr &&
         impl_->normalized_bf16_scratch->valid() &&
         impl_->normalized_scratch != nullptr &&
         impl_->normalized_scratch->valid() &&
         impl_->projected_scratch != nullptr &&
         impl_->projected_scratch->valid() &&
         impl_->scan_output_scratch != nullptr &&
         impl_->scan_output_scratch->valid() &&
         impl_->projected_output_scratch != nullptr &&
         impl_->projected_output_scratch->valid();
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
    const DeviceTensorBf16& input,
    DeviceTensorBf16* residual,
    DeviceTensorBf16* output,
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
      residual == nullptr ||
      !residual->valid() ||
      residual->shape() != input.shape() ||
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

  if (trace == nullptr && token_count > kMambaMultiTokenMaxChunkTokens) {
    const std::size_t hidden_size = impl_->config.hidden_size;
    for (std::size_t chunk_start = 0; chunk_start < token_count;
         chunk_start += kMambaMultiTokenMaxChunkTokens) {
      const std::size_t chunk_tokens =
          std::min(kMambaMultiTokenMaxChunkTokens, token_count - chunk_start);
      auto input_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, hidden_size},
          input.data() + (chunk_start * hidden_size));
      auto residual_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, hidden_size},
          residual->data() + (chunk_start * hidden_size));
      auto output_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, hidden_size},
          output->data() + (chunk_start * hidden_size));
      if (input_chunk == nullptr ||
          residual_chunk == nullptr ||
          output_chunk == nullptr ||
          !Run(
              cublas_handle,
              heuristic_cache,
              request_context,
              *input_chunk,
              residual_chunk.get(),
              output_chunk.get(),
              nullptr)) {
        return false;
      }
    }
    return true;
  }

  if (token_count > 1) {
    RecordMambaNativeMultiTokenExecution(token_count);
  }

  const std::size_t conv_dim =
      impl_->config.intermediate_size + (2 * impl_->config.n_groups * impl_->config.state_size);
  const std::size_t projection_size =
      impl_->config.intermediate_size + conv_dim + impl_->config.num_heads;

  std::unique_ptr<DeviceTensorBf16> normalized_bf16_owned;
  std::unique_ptr<DeviceTensorFp32> normalized_owned;
  std::unique_ptr<DeviceTensorFp32> projected_owned;
  std::unique_ptr<DeviceTensorFp32> scan_output_owned;
  std::unique_ptr<DeviceTensorFp32> projected_output_owned;
  DeviceTensorBf16* normalized_bf16 = nullptr;
  DeviceTensorFp32* normalized = nullptr;
  DeviceTensorFp32* projected = nullptr;
  DeviceTensorFp32* scan_output = nullptr;
  DeviceTensorFp32* projected_output = nullptr;
  const bool use_decode_scratch = token_count == 1;
  if (use_decode_scratch) {
    normalized_bf16 = impl_->normalized_bf16_scratch.get();
    normalized = impl_->normalized_scratch.get();
    projected = impl_->projected_scratch.get();
    scan_output = impl_->scan_output_scratch.get();
    projected_output = impl_->projected_output_scratch.get();
  } else {
    auto* normalized_bf16_backing = GetCachedMambaBf16Scratch(
        MambaScratchRole::kNormalizedBf16,
        token_count,
        impl_->config.hidden_size);
    auto* normalized_backing = GetCachedMambaFp32Scratch(
        MambaScratchRole::kNormalizedFp32,
        token_count,
        impl_->config.hidden_size);
    auto* projected_backing = GetCachedMambaFp32Scratch(
        MambaScratchRole::kProjected,
        token_count,
        projection_size);
    auto* scan_output_backing = GetCachedMambaFp32Scratch(
        MambaScratchRole::kScanOutput,
        token_count,
        impl_->config.intermediate_size);
    auto* projected_output_backing = GetCachedMambaFp32Scratch(
        MambaScratchRole::kProjectedOutput,
        token_count,
        impl_->config.hidden_size);

    if (normalized_bf16_backing != nullptr) {
      if (normalized_bf16_backing->shape()[0] == token_count) {
        normalized_bf16 = normalized_bf16_backing;
      } else {
        normalized_bf16_owned = DeviceTensorBf16::CreateView(
            {token_count, impl_->config.hidden_size},
            normalized_bf16_backing->data());
        normalized_bf16 = normalized_bf16_owned.get();
      }
    }
    if (normalized_backing != nullptr) {
      if (normalized_backing->shape()[0] == token_count) {
        normalized = normalized_backing;
      } else {
        normalized_owned = DeviceTensorFp32::CreateView(
            {token_count, impl_->config.hidden_size},
            normalized_backing->data());
        normalized = normalized_owned.get();
      }
    }
    if (projected_backing != nullptr) {
      if (projected_backing->shape()[0] == token_count) {
        projected = projected_backing;
      } else {
        projected_owned = DeviceTensorFp32::CreateView(
            {token_count, projection_size},
            projected_backing->data());
        projected = projected_owned.get();
      }
    }
    if (scan_output_backing != nullptr) {
      if (scan_output_backing->shape()[0] == token_count) {
        scan_output = scan_output_backing;
      } else {
        scan_output_owned = DeviceTensorFp32::CreateView(
            {token_count, impl_->config.intermediate_size},
            scan_output_backing->data());
        scan_output = scan_output_owned.get();
      }
    }
    if (projected_output_backing != nullptr) {
      if (projected_output_backing->shape()[0] == token_count) {
        projected_output = projected_output_backing;
      } else {
        projected_output_owned = DeviceTensorFp32::CreateView(
            {token_count, impl_->config.hidden_size},
            projected_output_backing->data());
        projected_output = projected_output_owned.get();
      }
    }
  }
  if (projected_output == nullptr || normalized_bf16 == nullptr) {
    if (normalized_bf16 == nullptr) {
      normalized_bf16_owned = DeviceTensorBf16::Create({token_count, impl_->config.hidden_size});
    }
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
    if (normalized_bf16 == nullptr) {
      normalized_bf16 = normalized_bf16_owned.get();
    }
    if (projected == nullptr) {
      projected = projected_owned.get();
    }
    if (scan_output == nullptr) {
      scan_output = scan_output_owned.get();
    }
    projected_output = projected_output_owned.get();
  }
  if (!normalized_bf16 || !normalized || !projected || !scan_output || !projected_output) {
    return false;
  }

  if (trace != nullptr) {
    trace->norm_output.clear();
    trace->in_proj_output.clear();
    trace->scan_output.clear();
    trace->projected_output.clear();
  }

  if (!FusedAddRmsNormBf16(
          input,
          residual,
          *impl_->input_norm_weight,
          impl_->config.input_rms_epsilon,
          normalized_bf16) ||
      !CastTensorBf16ToFp32(*normalized_bf16, normalized)) {
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
    if (!out_proj_ok || !CastTensorFp32ToBf16(*projected_output, output)) {
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

    auto* conv_output_backing = GetCachedMambaFp32Scratch(
        MambaScratchRole::kConvOutput,
        token_count,
        conv_dim);
    std::unique_ptr<DeviceTensorFp32> conv_output_owned;
    DeviceTensorFp32* conv_output = nullptr;
    if (conv_output_backing != nullptr) {
      if (conv_output_backing->shape()[0] == token_count) {
        conv_output = conv_output_backing;
      } else {
        conv_output_owned = DeviceTensorFp32::CreateView(
            {token_count, conv_dim},
            conv_output_backing->data());
        conv_output = conv_output_owned.get();
      }
    }
    if (conv_output == nullptr || !conv_output->valid()) {
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
    if (!RunMambaConvPrefill(conv_params, *projected, conv_output)) {
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

  if (token_count > 1 || trace != nullptr) {
    return run_sequential_path();
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
  return run_output_projection();
}

bool MambaLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    MambaLayerRunTrace* trace) const {
  auto input_bf16 = DeviceTensorBf16::Create(input.shape());
  auto residual_bf16 = DeviceTensorBf16::Create(input.shape());
  auto output_bf16 = DeviceTensorBf16::Create(input.shape());
  const bool ok =
      input_bf16 != nullptr &&
      residual_bf16 != nullptr &&
      output_bf16 != nullptr &&
      CastTensorFp32ToBf16(input, input_bf16.get()) &&
      residual_bf16->FillZero() &&
      Run(
          cublas_handle,
          heuristic_cache,
          request_context,
          *input_bf16,
          residual_bf16.get(),
          output_bf16.get(),
          trace) &&
      CastTensorBf16ToFp32(*output_bf16, output);
  return ok;
}

}  // namespace nemotron
