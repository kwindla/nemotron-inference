#include "nemotron/expert_layer.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "nemotron/cutlass_nvfp4_grouped_gemm.h"
#include "nemotron/expert_ops.h"
#include "nemotron/flashinfer_layout.h"
#include "nemotron/flashinfer_moe_backend.h"
#include "nemotron/linear_op.h"
#include "nemotron/device_buffer.h"
#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/runtime_stats.h"
#include "storage_conversion.h"

namespace nemotron {
namespace {

bool IsFp32Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp32" || storage_dtype == "float32" || storage_dtype == "float";
}

bool IsBf16Storage(const std::string& storage_dtype) {
  return storage_dtype == "bf16" || storage_dtype == "bfloat16";
}

bool IsFp8E4M3Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp8_e4m3fn" || storage_dtype == "fp8_e4m3";
}

std::optional<float> ReadTensorScaleHost(const GemmDescriptor& descriptor) {
  if (descriptor.tensor_scale_data == nullptr || descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }
  float value = 0.0f;
  std::memcpy(&value, descriptor.tensor_scale_data, sizeof(float));
  return value;
}

std::optional<PackedFloatStorage> PackedStorageForDenseDescriptor(const GemmDescriptor& descriptor) {
  if (descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0 ||
      descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  if (IsFp32Storage(descriptor.storage_dtype)) {
    return PackedFloatStorage::kFp32;
  }
  if (IsBf16Storage(descriptor.storage_dtype)) {
    return PackedFloatStorage::kBf16;
  }
  if (IsFp8E4M3Storage(descriptor.storage_dtype)) {
    return PackedFloatStorage::kFp8E4M3;
  }
  return std::nullopt;
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
  config.weight_scale = *weight_scale_value;
  config.input_scale = *input_scale_value;
  return config;
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

float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

float Relu2(float value) {
  return value > 0.0f ? (value * value) : 0.0f;
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

std::vector<float> DequantizeNvfp4Matrix(
    const std::uint8_t* packed,
    std::size_t packed_nbytes,
    const std::uint8_t* block_scales,
    std::size_t block_scales_nbytes,
    float tensor_scale,
    std::size_t rows,
    std::size_t cols) {
  if (packed == nullptr ||
      block_scales == nullptr ||
      cols == 0 ||
      cols % 16 != 0 ||
      packed_nbytes != (rows * cols) / 2 ||
      block_scales_nbytes != rows * (cols / 16)) {
    return {};
  }

  std::vector<float> output(rows * cols, 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < cols / 16; ++block) {
      const float block_scale = DecodeFp8(block_scales[scale_index++]) * tensor_scale;
      const std::size_t col_start = block * 16;
      for (std::size_t offset = 0; offset < 16; offset += 2) {
        const std::uint8_t byte = packed[packed_index++];
        output[row * cols + col_start + offset] = DecodeFp4(byte & 0x0F) * block_scale;
        output[row * cols + col_start + offset + 1] = DecodeFp4((byte >> 4) & 0x0F) * block_scale;
      }
    }
  }
  return output;
}

std::optional<std::vector<float>> DequantizeNvfp4WeightToHostFp32(const GemmDescriptor& descriptor) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr ||
      descriptor.tensor_scale_data == nullptr ||
      descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }
  float tensor_scale = 0.0f;
  std::memcpy(&tensor_scale, descriptor.tensor_scale_data, sizeof(float));
  std::vector<float> values = DequantizeNvfp4Matrix(
      descriptor.packed_data,
      descriptor.packed_nbytes,
      descriptor.block_scales_data,
      descriptor.block_scales_nbytes,
      tensor_scale,
      descriptor.output_rows,
      descriptor.input_cols);
  if (values.empty()) {
    return std::nullopt;
  }
  return values;
}

struct Nvfp4AlignedBuffers;

bool EnsureNvfp4LookupBuffers(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers,
    bool* buffers_ready);

bool EnsureNvfp4AlignedBuffers(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers,
    bool* buffers_ready);

GemmDescriptor MakeAlignedNvfp4Descriptor(
    const GemmDescriptor& descriptor,
    const Nvfp4AlignedBuffers& buffers);

std::unique_ptr<UploadedLinearOp> MaterializeNvfp4AlignedViewOp(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers,
    bool* buffers_ready);

std::unique_ptr<UploadedLinearOp> MaterializeRoutedExpertDenseFallbackOp(
    const GemmDescriptor& descriptor) {
  const auto dequantized = DequantizeNvfp4WeightToHostFp32(descriptor);
  if (!dequantized.has_value() || dequantized->empty()) {
    return nullptr;
  }

  GemmDescriptor dense_descriptor;
  dense_descriptor.tensor_name = descriptor.tensor_name + ".lazy_dense_fp32";
  dense_descriptor.op_class = descriptor.op_class;
  dense_descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  dense_descriptor.output_rows = descriptor.output_rows;
  dense_descriptor.input_cols = descriptor.input_cols;
  dense_descriptor.storage_dtype = "fp32";
  dense_descriptor.compute_dtype = "fp32";
  dense_descriptor.layout_tag = "row_major";
  dense_descriptor.alignment_bytes = 16;
  dense_descriptor.packed_data =
      reinterpret_cast<const std::uint8_t*>(dequantized->data());
  dense_descriptor.packed_nbytes = dequantized->size() * sizeof(float);
  return UploadedLinearOp::Create(dense_descriptor);
}

std::unique_ptr<UploadedLinearOp> MaterializeRoutedExpertOp(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* nvfp4_buffers,
    bool* buffers_ready) {
  if (descriptor.kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    return MaterializeNvfp4AlignedViewOp(descriptor, nvfp4_buffers, buffers_ready);
  }
  return UploadedLinearOp::Create(descriptor);
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
      float accum = 0.0f;
      for (std::size_t col = 0; col < input_cols; ++col) {
        accum += activations[row * input_cols + col] * weights[out * input_cols + col];
      }
      output[row * output_rows + out] = accum;
    }
  }
  return output;
}

std::optional<std::vector<float>> RunNvfp4LinearHost(
    const std::vector<float>& activations,
    std::size_t rows,
    const GemmDescriptor& descriptor) {
  if (rows == 0 ||
      activations.size() != rows * descriptor.input_cols ||
      descriptor.input_cols == 0 ||
      descriptor.output_rows == 0) {
    return std::nullopt;
  }
  const auto activation_pack = PackRowMajorFp32ToNvfp4(
      activations.data(),
      rows,
      descriptor.input_cols,
      {});
  if (!activation_pack.has_value()) {
    return std::nullopt;
  }
  const std::vector<float> activation_dequant = DequantizeNvfp4Matrix(
      activation_pack->packed_data(),
      activation_pack->packed_nbytes(),
      activation_pack->block_scales_data(),
      activation_pack->block_scales_nbytes(),
      activation_pack->tensor_scale,
      rows,
      descriptor.input_cols);
  const auto weight_dequant = DequantizeNvfp4WeightToHostFp32(descriptor);
  if (activation_dequant.empty() || !weight_dequant.has_value()) {
    return std::nullopt;
  }
  return CpuMatmulRowMajor(
      activation_dequant,
      rows,
      *weight_dequant,
      descriptor.output_rows,
      descriptor.input_cols);
}

std::vector<float> ApplyRelu2(const std::vector<float>& input) {
  std::vector<float> output = input;
  for (float& value : output) {
    value = Relu2(value);
  }
  return output;
}

bool CopyToHost(const DeviceTensorFp32& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  output->assign(tensor.numel(), 0.0f);
  return tensor.CopyToHost(output->data(), output->size());
}

bool UploadHostVector(const std::vector<float>& input, DeviceTensorFp32* output) {
  return output != nullptr &&
         output->valid() &&
         output->numel() == input.size() &&
         output->CopyFromHost(input.data(), input.size());
}

bool HasNonFinite(const std::vector<float>& values) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  return false;
}

struct Nvfp4AlignedBuffers {
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> block_scales;
  DeviceBuffer<std::uint8_t> matmul_block_scales;
  DeviceBuffer<std::uint8_t> tensor_scale;

  bool lookup_valid() const {
    return packed.data() != nullptr &&
           packed.count() != 0 &&
           block_scales.data() != nullptr &&
           block_scales.count() != 0 &&
           tensor_scale.data() != nullptr &&
           tensor_scale.count() != 0;
  }

  bool valid() const {
    return lookup_valid() &&
           matmul_block_scales.data() != nullptr &&
           matmul_block_scales.count() != 0;
  }
};

bool UploadNvfp4LookupBuffers(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers) {
  if (buffers == nullptr ||
      descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr ||
      descriptor.tensor_scale_data == nullptr) {
    return false;
  }
  return buffers->packed.Resize(descriptor.packed_nbytes) &&
         buffers->block_scales.Resize(descriptor.block_scales_nbytes) &&
         buffers->tensor_scale.Resize(descriptor.tensor_scale_nbytes) &&
         buffers->packed.CopyFromHost(descriptor.packed_data, descriptor.packed_nbytes) &&
         buffers->block_scales.CopyFromHost(
             descriptor.block_scales_data,
             descriptor.block_scales_nbytes) &&
         buffers->tensor_scale.CopyFromHost(
             descriptor.tensor_scale_data,
             descriptor.tensor_scale_nbytes);
}

bool UploadNvfp4ExecutionScales(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers) {
  if (buffers == nullptr ||
      descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.block_scales_data == nullptr) {
    return false;
  }
  const std::vector<std::uint8_t> matmul_block_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      descriptor.block_scales_data,
      descriptor.output_rows,
      descriptor.input_cols);
  if (matmul_block_scales.empty()) {
    return false;
  }
  return buffers->matmul_block_scales.Resize(matmul_block_scales.size()) &&
         buffers->matmul_block_scales.CopyFromHost(matmul_block_scales);
}

bool EnsureNvfp4LookupBuffers(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers,
    bool* buffers_ready) {
  if (buffers == nullptr || buffers_ready == nullptr) {
    return false;
  }
  if (*buffers_ready) {
    return buffers->lookup_valid();
  }
  if (!UploadNvfp4LookupBuffers(descriptor, buffers)) {
    return false;
  }
  *buffers_ready = true;
  return true;
}

bool EnsureNvfp4AlignedBuffers(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers,
    bool* buffers_ready) {
  if (buffers == nullptr || buffers_ready == nullptr) {
    return false;
  }
  if (*buffers_ready) {
    return buffers->valid();
  }
  if ((!buffers->lookup_valid() && !UploadNvfp4LookupBuffers(descriptor, buffers)) ||
      !UploadNvfp4ExecutionScales(descriptor, buffers)) {
    return false;
  }
  *buffers_ready = true;
  return true;
}

GemmDescriptor MakeAlignedNvfp4Descriptor(
    const GemmDescriptor& descriptor,
    const Nvfp4AlignedBuffers& buffers) {
  GemmDescriptor aligned_descriptor = descriptor;
  aligned_descriptor.packed_data = buffers.packed.data();
  aligned_descriptor.block_scales_data = buffers.matmul_block_scales.data();
  aligned_descriptor.block_scales_nbytes = buffers.matmul_block_scales.count();
  aligned_descriptor.tensor_scale_data = buffers.tensor_scale.data();
  return aligned_descriptor;
}

std::unique_ptr<UploadedLinearOp> MaterializeNvfp4AlignedViewOp(
    const GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers,
    bool* buffers_ready) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (!EnsureNvfp4AlignedBuffers(descriptor, buffers, buffers_ready)) {
    if (debug) {
      std::cerr << "expert_layer: nvfp4 aligned buffer prep failed for "
                << descriptor.tensor_name << "\n";
    }
    return nullptr;
  }

  auto weight_view = DeviceNvfp4Weight::CreateView(
      descriptor.output_rows,
      descriptor.input_cols,
      buffers->packed.data(),
      descriptor.packed_nbytes,
      buffers->block_scales.data(),
      descriptor.block_scales_nbytes,
      buffers->matmul_block_scales.data(),
      buffers->matmul_block_scales.count(),
      buffers->tensor_scale.data(),
      descriptor.tensor_scale_nbytes);
  if (!weight_view || !weight_view->valid()) {
    if (debug) {
      std::cerr << "expert_layer: nvfp4 weight view create failed for "
                << descriptor.tensor_name
                << " packed_ptr=" << static_cast<const void*>(buffers->packed.data())
                << " packed_nbytes=" << descriptor.packed_nbytes
                << " raw_block_scale_ptr="
                << static_cast<const void*>(buffers->block_scales.data())
                << " raw_block_scale_nbytes=" << descriptor.block_scales_nbytes
                << " matmul_block_scale_ptr="
                << static_cast<const void*>(buffers->matmul_block_scales.data())
                << " matmul_block_scale_nbytes=" << buffers->matmul_block_scales.count()
                << " tensor_scale_ptr="
                << static_cast<const void*>(buffers->tensor_scale.data())
                << " tensor_scale_nbytes=" << descriptor.tensor_scale_nbytes
                << "\n";
    }
    return nullptr;
  }

  GemmDescriptor aligned_descriptor = MakeAlignedNvfp4Descriptor(descriptor, *buffers);
  auto op = UploadedLinearOp::CreateNvfp4View(aligned_descriptor, std::move(weight_view));
  if ((!op || !op->valid()) && debug) {
    std::cerr << "expert_layer: nvfp4 aligned view op create failed for "
              << descriptor.tensor_name << "\n";
  }
  return op;
}

struct SelectionScratchView {
  DeviceBuffer<std::int32_t>* indices_device = nullptr;
  DeviceBuffer<float>* weights_device = nullptr;
  std::vector<std::int32_t>* indices_host = nullptr;
  std::vector<float>* weights_host = nullptr;
};

bool PrepareSelectionScratch(
    RequestExecutionContext* request_context,
    std::size_t selection_count,
    DeviceBuffer<std::int32_t>* local_indices_device,
    DeviceBuffer<float>* local_weights_device,
    std::vector<std::int32_t>* local_indices_host,
    std::vector<float>* local_weights_host,
    SelectionScratchView* scratch) {
  if (scratch == nullptr) {
    return false;
  }
  if (request_context != nullptr) {
    if (!request_context->EnsureExpertSelectionCapacity(selection_count)) {
      return false;
    }
    scratch->indices_device = request_context->expert_selection_indices_device();
    scratch->weights_device = request_context->expert_selection_weights_device();
    scratch->indices_host = request_context->expert_selection_indices_host();
    scratch->weights_host = request_context->expert_selection_weights_host();
    return scratch->indices_device != nullptr &&
           scratch->weights_device != nullptr &&
           scratch->indices_host != nullptr &&
           scratch->weights_host != nullptr;
  }

  if (local_indices_device == nullptr || local_weights_device == nullptr ||
      local_indices_host == nullptr || local_weights_host == nullptr) {
    return false;
  }
  if (!local_indices_device->Resize(selection_count) ||
      !local_weights_device->Resize(selection_count)) {
    return false;
  }
  local_indices_host->assign(selection_count, -1);
  local_weights_host->assign(selection_count, 0.0f);
  scratch->indices_device = local_indices_device;
  scratch->weights_device = local_weights_device;
  scratch->indices_host = local_indices_host;
  scratch->weights_host = local_weights_host;
  return true;
}

}  // namespace

struct ExpertLayerSlice::Impl {
  struct RoutedExpertEntry {
    GemmDescriptor up_descriptor;
    GemmDescriptor down_descriptor;
    std::optional<float> up_tensor_scale;
    std::optional<float> down_tensor_scale;
    mutable bool up_nvfp4_lookup_ready = false;
    mutable bool down_nvfp4_lookup_ready = false;
    mutable bool up_nvfp4_buffers_ready = false;
    mutable bool down_nvfp4_buffers_ready = false;
    mutable bool grouped_lookup_ready = false;
    mutable std::mutex materialize_mutex;
    mutable std::unique_ptr<Nvfp4AlignedBuffers> up_nvfp4_buffers;
    mutable std::unique_ptr<Nvfp4AlignedBuffers> down_nvfp4_buffers;
    mutable std::unique_ptr<UploadedLinearOp> up_proj;
    mutable std::unique_ptr<UploadedLinearOp> down_proj;
  };

  enum class ProjectionFamily {
    kNone,
    kDense,
    kScaledFp8,
  };

  enum class SharedDownFamily {
    kNone,
    kDense,
    kScaledFp8,
    kNvfp4,
  };

  ExpertLayerConfig config;
  std::unique_ptr<DeviceTensorFp32> input_norm_weight;
  std::unique_ptr<DeviceTensorFp32> gate_score_correction_bias_device;
  std::unique_ptr<UploadedLinearOp> gate_weight;
  ProjectionFamily fc1_latent_family = ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> fc1_latent_dense;
  std::unique_ptr<ScaledFp8LinearOp> fc1_latent_scaled_fp8;
  std::unique_ptr<UploadedLinearOp> fc2_latent;
  ProjectionFamily shared_up_family = ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_up_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_up_scaled_fp8;
  SharedDownFamily shared_down_family = SharedDownFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_down_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_down_scaled_fp8;
  bool shared_down_nvfp4_buffers_ready = false;
  std::unique_ptr<Nvfp4AlignedBuffers> shared_down_nvfp4_buffers;
  std::unique_ptr<UploadedLinearOp> shared_down_nvfp4;
  std::vector<RoutedExpertEntry> routed_experts;
  RoutedMoEBackendKind routed_backend_kind = RoutedMoEBackendKind::kCustomFused;
  std::string routed_backend_detail = "custom fused routed nvfp4";
  std::unique_ptr<FlashInferRoutedMoEBackend> flashinfer_routed_backend;
  mutable std::atomic<bool> grouped_routed_nvfp4_enabled{true};
  std::unique_ptr<CutlassNvfp4GroupedGemmPlan> cutlass_up_plan;
  std::unique_ptr<CutlassNvfp4GroupedGemmPlan> cutlass_down_plan;
  // Pre-allocated device pointer arrays for CUTLASS dispatch (sized for top_k).
  DeviceBuffer<const void*> cutlass_a_ptrs;
  DeviceBuffer<const void*> cutlass_a_sf_ptrs;
  DeviceBuffer<const void*> cutlass_c_ptrs;
  DeviceBuffer<void*> cutlass_d_ptrs;
  mutable std::mutex routed_prefetch_mutex;
  mutable std::atomic<bool> routed_prefetch_done{false};
  mutable DeviceBuffer<const void*> routed_up_packed_lookup_device;
  mutable DeviceBuffer<const void*> routed_up_raw_scale_lookup_device;
  mutable DeviceBuffer<float> routed_up_tensor_scale_lookup_device;
  mutable DeviceBuffer<const void*> routed_down_packed_lookup_device;
  mutable DeviceBuffer<const void*> routed_down_raw_scale_lookup_device;
  mutable DeviceBuffer<float> routed_down_tensor_scale_lookup_device;
  float* dense_pool = nullptr;
  std::size_t dense_pool_numel = 0;
  std::uint8_t* expert_pool = nullptr;
  std::size_t expert_pool_bytes = 0;

  // Scratch buffers for try_grouped_routed_single_token (pre-allocated to
  // avoid per-token cudaMalloc). Sized once in Create(), reused every token.
  // -- Gather output buffers --
  mutable DeviceBuffer<const void*> scratch_up_packed_ptrs;
  mutable DeviceBuffer<const void*> scratch_up_raw_scale_ptrs;
  mutable DeviceBuffer<const void*> scratch_down_packed_ptrs;
  mutable DeviceBuffer<const void*> scratch_down_raw_scale_ptrs;
  mutable DeviceBuffer<std::uint32_t> scratch_missing_count;
  mutable DeviceBuffer<std::int32_t> scratch_missing_indices;
  mutable DeviceBuffer<float> scratch_selected_up_tensor_scales;
  mutable DeviceBuffer<float> scratch_selected_down_tensor_scales;
  // -- NVFP4 packing buffers (latent activation) --
  mutable DeviceBuffer<std::uint8_t> scratch_latent_packed_data;
  mutable DeviceBuffer<std::uint8_t> scratch_latent_block_scales;
  mutable DeviceBuffer<std::uint8_t> scratch_latent_matmul_scales;
  mutable DeviceBuffer<std::uint8_t> scratch_latent_tensor_scale;
  mutable DeviceBuffer<unsigned int> scratch_global_max_bits;
  // -- ScaleRelu2 output buffers --
  mutable DeviceBuffer<std::uint8_t> scratch_down_act_packed;
  mutable DeviceBuffer<std::uint8_t> scratch_down_act_block_scales;
  mutable DeviceBuffer<float> scratch_down_act_tensor_scales;
  // -- Pre-filled row scales (constant 1.0f, avoids per-token H→D copy) --
  mutable DeviceBuffer<float> scratch_row_scales;
  // -- CUTLASS path aligned buffers --
  mutable DeviceBuffer<std::uint8_t> scratch_aligned_act_packed;
  mutable DeviceBuffer<std::uint8_t> scratch_aligned_act_scales;

  ~Impl() {
    if (dense_pool != nullptr) {
      cudaFree(dense_pool);
    }
    if (expert_pool != nullptr) {
      cudaFree(expert_pool);
    }
  }
};

enum class FlashInferWeightSurface {
  kLegacyTrtPrepared,
  kCutlassRaw,
};

FlashInferWeightSurface ResolveFlashInferWeightSurface() {
  const char* env = std::getenv("NEMOTRON_FLASHINFER_WEIGHT_SURFACE");
  if (env != nullptr && *env != '\0') {
    return std::string(env) == "cutlass_raw"
               ? FlashInferWeightSurface::kCutlassRaw
               : FlashInferWeightSurface::kLegacyTrtPrepared;
  }
  const auto availability = GetFlashInferRoutedMoEAvailability();
  return availability.backend_kind == NEMOTRON_FLASHINFER_BACKEND_KIND_CUTLASS_FUSED
             ? FlashInferWeightSurface::kCutlassRaw
             : FlashInferWeightSurface::kLegacyTrtPrepared;
}

const char* ToString(FlashInferWeightSurface surface) {
  switch (surface) {
    case FlashInferWeightSurface::kLegacyTrtPrepared:
      return "legacy_trt_prepared";
    case FlashInferWeightSurface::kCutlassRaw:
      return "cutlass_raw";
  }
  return "unknown";
}

template <typename ImplT, typename RoutedExpertEntriesT>
bool ConfigureRoutedMoEBackend(
    ImplT* impl,
    const RoutedExpertEntriesT& routed_experts,
    bool debug,
    std::string* strict_failure_reason) {
  if (impl == nullptr) {
    if (strict_failure_reason != nullptr) {
      *strict_failure_reason = "missing expert layer impl";
    }
    return false;
  }

  impl->routed_backend_kind = RoutedMoEBackendKind::kCustomFused;
  impl->routed_backend_detail = "custom fused routed nvfp4";
  impl->flashinfer_routed_backend.reset();

  if (ResolveRequestedRoutedMoEBackend() != RoutedMoEBackendKind::kFlashInfer) {
    return true;
  }

  const auto availability = GetFlashInferRoutedMoEAvailability();
  if (!availability.available) {
    impl->routed_backend_detail = availability.detail;
    if (RoutedMoEBackendStrict()) {
      if (strict_failure_reason != nullptr) {
        *strict_failure_reason = availability.detail;
      }
      return false;
    }
    if (debug) {
      std::cerr << "expert_layer: layer " << impl->config.layer_index
                << " flashinfer backend unavailable, falling back to custom fused path: "
                << availability.detail << "\n";
    }
    return true;
  }

  const FlashInferWeightSurface weight_surface = ResolveFlashInferWeightSurface();
  std::vector<FlashInferPreparedNvfp4WeightHost> up_prepared;
  std::vector<FlashInferPreparedNvfp4WeightHost> down_prepared;
  std::vector<NemotronFlashInferNvfp4WeightView> up_views(routed_experts.size());
  std::vector<NemotronFlashInferNvfp4WeightView> down_views(routed_experts.size());
  if (weight_surface == FlashInferWeightSurface::kLegacyTrtPrepared) {
    up_prepared.resize(routed_experts.size());
    down_prepared.resize(routed_experts.size());
  }
  for (std::size_t expert_index = 0; expert_index < routed_experts.size(); ++expert_index) {
    std::optional<NemotronFlashInferNvfp4WeightView> up_view;
    std::optional<NemotronFlashInferNvfp4WeightView> down_view;
    if (weight_surface == FlashInferWeightSurface::kCutlassRaw) {
      up_view = BuildFlashInferRawNvfp4WeightView(routed_experts[expert_index].up_descriptor);
      down_view = BuildFlashInferRawNvfp4WeightView(routed_experts[expert_index].down_descriptor);
    } else {
      const auto up_prepared_view = PrepareFlashInferNvfp4WeightHost(
          routed_experts[expert_index].up_descriptor);
      const auto down_prepared_view = PrepareFlashInferNvfp4WeightHost(
          routed_experts[expert_index].down_descriptor);
      if (up_prepared_view.has_value()) {
        up_prepared[expert_index] = std::move(*up_prepared_view);
        up_view = BuildFlashInferPreparedNvfp4WeightView(up_prepared[expert_index]);
      }
      if (down_prepared_view.has_value()) {
        down_prepared[expert_index] = std::move(*down_prepared_view);
        down_view = BuildFlashInferPreparedNvfp4WeightView(down_prepared[expert_index]);
      }
    }
    if (!up_view.has_value() || !down_view.has_value()) {
      const std::string detail = weight_surface == FlashInferWeightSurface::kCutlassRaw
                                     ? "flashinfer CUTLASS raw surface requires raw packed NVFP4 routed expert descriptors"
                                     : "flashinfer legacy prepared surface requires TRT-preparable NVFP4 routed expert descriptors";
      impl->routed_backend_detail = detail;
      if (RoutedMoEBackendStrict()) {
        if (strict_failure_reason != nullptr) {
          *strict_failure_reason = detail;
        }
        return false;
      }
      if (debug) {
        std::cerr << "expert_layer: layer " << impl->config.layer_index
                  << " flashinfer backend unsupported for routed expert " << expert_index
                  << ", falling back to custom fused path\n";
      }
      return true;
    }
    up_views[expert_index] = *up_view;
    down_views[expert_index] = *down_view;
  }

  NemotronFlashInferRoutedMoECreateParams params{};
  params.layer_index = impl->config.layer_index;
  params.num_experts = impl->config.n_routed_experts;
  params.hidden_size = impl->config.hidden_size;
  params.moe_latent_size = impl->config.moe_latent_size;
  params.intermediate_size = impl->config.routed_expert_intermediate_size;
  params.top_k = impl->config.top_k;
  params.n_group = impl->config.n_group;
  params.topk_group = impl->config.topk_group;
  params.routed_scaling_factor = impl->config.routed_scaling_factor;
  params.norm_topk_prob = impl->config.norm_topk_prob ? 1 : 0;
  params.routing_method_type = 2;
  params.activation_type = 6;
  params.up_experts = up_views.data();
  params.down_experts = down_views.data();

  auto backend = FlashInferRoutedMoEBackend::Create(params);
  if (backend == nullptr || !backend->valid()) {
    const std::string detail =
        backend != nullptr ? backend->availability().detail : availability.detail;
    impl->routed_backend_detail = detail;
    if (RoutedMoEBackendStrict()) {
      if (strict_failure_reason != nullptr) {
        *strict_failure_reason = detail;
      }
      return false;
    }
    if (debug) {
      std::cerr << "expert_layer: layer " << impl->config.layer_index
                << " flashinfer backend creation failed, falling back to custom fused path: "
                << detail << "\n";
    }
    return true;
  }

  impl->routed_backend_kind = RoutedMoEBackendKind::kFlashInfer;
  impl->routed_backend_detail =
      backend->availability().detail + " via " + ToString(weight_surface);
  impl->flashinfer_routed_backend = std::move(backend);
  if (debug) {
    std::cerr << "expert_layer: layer " << impl->config.layer_index
              << " using routed backend " << ToString(impl->routed_backend_kind)
              << " (" << impl->routed_backend_detail << ")\n";
  }
  return true;
}

std::optional<ExpertLayerBindings> BuildExpertLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog,
    std::size_t routed_expert_count) {
  ExpertLayerBindings bindings;
  bindings.input_norm_weight = FindExactKernelBinding(layer, kernel_catalog, "norm.weight");
  bindings.gate_weight = FindExactGemmBinding(layer, gemm_catalog, "mixer.gate.weight");
  bindings.gate_score_correction_bias =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.gate.e_score_correction_bias");
  bindings.fc1_latent_gemm_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.fc1_latent_proj.weight");
  bindings.fc1_latent_kernel_weight =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.fc1_latent_proj.weight");
  bindings.fc1_latent_weight_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.fc1_latent_proj.weight_scale");
  bindings.fc1_latent_input_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.fc1_latent_proj.input_scale");
  bindings.fc2_latent_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.fc2_latent_proj.weight");
  bindings.shared_up_gemm_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.shared_experts.up_proj.weight");
  bindings.shared_up_kernel_weight =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.up_proj.weight");
  bindings.shared_up_weight_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.up_proj.weight_scale");
  bindings.shared_up_input_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.up_proj.input_scale");
  bindings.shared_down_gemm_weight =
      FindExactGemmBinding(layer, gemm_catalog, "mixer.shared_experts.down_proj.weight");
  bindings.shared_down_kernel_weight =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.down_proj.weight");
  bindings.shared_down_weight_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.down_proj.weight_scale");
  bindings.shared_down_input_scale =
      FindExactKernelBinding(layer, kernel_catalog, "mixer.shared_experts.down_proj.input_scale");

  bindings.routed_experts.resize(routed_expert_count);
  for (std::size_t expert_index = 0; expert_index < routed_expert_count; ++expert_index) {
    const std::string prefix = "mixer.experts." + std::to_string(expert_index);
    bindings.routed_experts[expert_index].up_proj =
        FindExactGemmBinding(layer, gemm_catalog, prefix + ".up_proj.weight");
    bindings.routed_experts[expert_index].down_proj =
        FindExactGemmBinding(layer, gemm_catalog, prefix + ".down_proj.weight");
  }

  if (bindings.input_norm_weight == nullptr ||
      bindings.gate_weight == nullptr ||
      bindings.gate_score_correction_bias == nullptr ||
      (bindings.fc1_latent_gemm_weight == nullptr &&
       bindings.fc1_latent_kernel_weight == nullptr) ||
      bindings.fc2_latent_weight == nullptr ||
      (bindings.shared_up_gemm_weight == nullptr &&
       bindings.shared_up_kernel_weight == nullptr) ||
      (bindings.shared_down_gemm_weight == nullptr &&
       bindings.shared_down_kernel_weight == nullptr)) {
    return std::nullopt;
  }

  for (const ExpertWeightPair& pair : bindings.routed_experts) {
    if (pair.up_proj == nullptr || pair.down_proj == nullptr) {
      return std::nullopt;
    }
  }
  return bindings;
}

std::unique_ptr<ExpertLayerSlice> ExpertLayerSlice::Create(
    const ExpertLayerConfig& config,
    const ExpertLayerBindings& bindings,
    ExpertLayerBuildTimingSink timing_sink) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const auto time_ms = [](const auto begin, const auto end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };
  const auto debug_fail = [&](const char* reason) -> std::unique_ptr<ExpertLayerSlice> {
    if (debug) {
      std::cerr << "expert_layer_create: layer " << config.layer_index
                << " failed: " << reason << "\n";
    }
    return nullptr;
  };
  if (config.hidden_size == 0 ||
      config.moe_latent_size == 0 ||
      config.routed_expert_intermediate_size == 0 ||
      config.shared_expert_intermediate_size == 0 ||
      config.n_routed_experts == 0 ||
      config.top_k == 0 ||
      config.top_k > config.n_routed_experts ||
      config.n_group == 0 ||
      config.topk_group == 0 ||
      config.n_routed_experts % config.n_group != 0 ||
      config.rms_epsilon <= 0.0f ||
      bindings.input_norm_weight == nullptr ||
      bindings.gate_weight == nullptr ||
      bindings.gate_score_correction_bias == nullptr ||
      (bindings.fc1_latent_gemm_weight == nullptr &&
       bindings.fc1_latent_kernel_weight == nullptr) ||
      bindings.fc2_latent_weight == nullptr ||
      (bindings.shared_up_gemm_weight == nullptr &&
       bindings.shared_up_kernel_weight == nullptr) ||
      (bindings.shared_down_gemm_weight == nullptr &&
       bindings.shared_down_kernel_weight == nullptr) ||
      bindings.routed_experts.size() != config.n_routed_experts) {
    return debug_fail("invalid config or missing required bindings");
  }

  float* dense_pool = nullptr;
  std::size_t dense_pool_numel = 0;
  std::uint8_t* expert_pool = nullptr;
  std::size_t expert_pool_bytes = 0;
  const auto debug_fail_with_cleanup = [&](const char* reason) -> std::unique_ptr<ExpertLayerSlice> {
    if (expert_pool != nullptr) {
      cudaFree(expert_pool);
      expert_pool = nullptr;
    }
    if (dense_pool != nullptr) {
      cudaFree(dense_pool);
      dense_pool = nullptr;
      dense_pool_numel = 0;
    }
    return debug_fail(reason);
  };
  const auto control_upload_begin = std::chrono::steady_clock::now();
  const auto input_norm_begin = std::chrono::steady_clock::now();
  auto input_norm_weight = UploadVectorWeightToDeviceFp32(*bindings.input_norm_weight);
  const auto input_norm_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("input_norm_upload", time_ms(input_norm_begin, input_norm_end));
  }

  const auto gate_bias_begin = std::chrono::steady_clock::now();
  auto gate_score_correction_bias_device =
      UploadVectorWeightToDeviceFp32(*bindings.gate_score_correction_bias);
  const auto gate_bias_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("gate_bias_upload", time_ms(gate_bias_begin, gate_bias_end));
  }

  const auto gate_weight_begin = std::chrono::steady_clock::now();
  auto gate_weight = UploadedLinearOp::Create(*bindings.gate_weight);
  const auto gate_weight_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("gate_weight_create", time_ms(gate_weight_begin, gate_weight_end));
  }

  const auto fc2_begin = std::chrono::steady_clock::now();
  auto fc2_latent = UploadedLinearOp::Create(*bindings.fc2_latent_weight);
  const auto fc2_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("fc2_create", time_ms(fc2_begin, fc2_end));
  }

  const auto control_upload_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("norm_gate_fc2_uploads", time_ms(control_upload_begin, control_upload_end));
  }
  Impl::ProjectionFamily fc1_latent_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> fc1_latent_dense;
  std::unique_ptr<ScaledFp8LinearOp> fc1_latent_scaled_fp8;
  const auto fc1_begin = std::chrono::steady_clock::now();
  if (bindings.fc1_latent_kernel_weight != nullptr &&
      bindings.fc1_latent_weight_scale != nullptr &&
      bindings.fc1_latent_input_scale != nullptr) {
    const auto fc1_config = BuildScaledFp8LinearConfig(
        *bindings.fc1_latent_kernel_weight,
        *bindings.fc1_latent_weight_scale,
        *bindings.fc1_latent_input_scale);
    if (fc1_config.has_value()) {
      fc1_latent_scaled_fp8 = ScaledFp8LinearOp::Create(*fc1_config);
      if (!fc1_latent_scaled_fp8 || !fc1_latent_scaled_fp8->valid()) {
        return debug_fail_with_cleanup("fc1_latent scaled-fp8 creation failed");
      }
      fc1_latent_family = Impl::ProjectionFamily::kScaledFp8;
    }
  }
  if (fc1_latent_family == Impl::ProjectionFamily::kNone) {
    fc1_latent_dense = UploadedLinearOp::Create(*bindings.fc1_latent_gemm_weight);
    if (!fc1_latent_dense || !fc1_latent_dense->valid()) {
      return debug_fail_with_cleanup("fc1_latent dense creation failed");
    }
    fc1_latent_family = Impl::ProjectionFamily::kDense;
  }
  const auto fc1_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("fc1_create", time_ms(fc1_begin, fc1_end));
  }

  Impl::ProjectionFamily shared_up_family = Impl::ProjectionFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_up_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_up_scaled_fp8;
  const auto shared_up_begin = std::chrono::steady_clock::now();
  if (bindings.shared_up_kernel_weight != nullptr &&
      bindings.shared_up_weight_scale != nullptr &&
      bindings.shared_up_input_scale != nullptr) {
    const auto shared_up_config = BuildScaledFp8LinearConfig(
        *bindings.shared_up_kernel_weight,
        *bindings.shared_up_weight_scale,
        *bindings.shared_up_input_scale);
    if (shared_up_config.has_value()) {
      shared_up_scaled_fp8 = ScaledFp8LinearOp::Create(*shared_up_config);
      if (!shared_up_scaled_fp8 || !shared_up_scaled_fp8->valid()) {
        return debug_fail_with_cleanup("shared_up scaled-fp8 creation failed");
      }
      shared_up_family = Impl::ProjectionFamily::kScaledFp8;
    }
  }
  if (shared_up_family == Impl::ProjectionFamily::kNone) {
    shared_up_dense = UploadedLinearOp::Create(*bindings.shared_up_gemm_weight);
    if (!shared_up_dense || !shared_up_dense->valid()) {
      return debug_fail_with_cleanup("shared_up dense creation failed");
    }
    shared_up_family = Impl::ProjectionFamily::kDense;
  }
  const auto shared_up_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("shared_up_create", time_ms(shared_up_begin, shared_up_end));
  }

  Impl::SharedDownFamily shared_down_family = Impl::SharedDownFamily::kNone;
  std::unique_ptr<UploadedLinearOp> shared_down_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_down_scaled_fp8;
  bool shared_down_nvfp4_buffers_ready = false;
  std::unique_ptr<Nvfp4AlignedBuffers> shared_down_nvfp4_buffers;
  std::unique_ptr<UploadedLinearOp> shared_down_nvfp4;
  const auto shared_down_begin = std::chrono::steady_clock::now();
  if (bindings.shared_down_gemm_weight != nullptr &&
      bindings.shared_down_gemm_weight->kernel_family ==
          GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    shared_down_family = Impl::SharedDownFamily::kNvfp4;
    shared_down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
    shared_down_nvfp4 = MaterializeRoutedExpertOp(
        *bindings.shared_down_gemm_weight,
        shared_down_nvfp4_buffers.get(),
        &shared_down_nvfp4_buffers_ready);
    if (!shared_down_nvfp4 || !shared_down_nvfp4->valid()) {
      return debug_fail_with_cleanup("shared_down NVFP4 creation failed");
    }
  } else if (bindings.shared_down_kernel_weight != nullptr &&
             bindings.shared_down_weight_scale != nullptr &&
             bindings.shared_down_input_scale != nullptr &&
             bindings.shared_down_kernel_weight->storage_dtype == "fp8_e4m3fn") {
    const auto shared_down_config = BuildScaledFp8LinearConfig(
        *bindings.shared_down_kernel_weight,
        *bindings.shared_down_weight_scale,
        *bindings.shared_down_input_scale);
    shared_down_scaled_fp8 =
        shared_down_config.has_value() ? ScaledFp8LinearOp::Create(*shared_down_config) : nullptr;
    if (!shared_down_scaled_fp8 || !shared_down_scaled_fp8->valid()) {
      return debug_fail_with_cleanup("shared_down scaled-fp8 creation failed");
    }
    shared_down_family = Impl::SharedDownFamily::kScaledFp8;
  } else if (bindings.shared_down_gemm_weight != nullptr) {
    shared_down_dense = UploadedLinearOp::Create(*bindings.shared_down_gemm_weight);
    if (!shared_down_dense || !shared_down_dense->valid()) {
      return debug_fail_with_cleanup("shared_down dense creation failed");
    }
    shared_down_family = Impl::SharedDownFamily::kDense;
  } else {
    return debug_fail_with_cleanup("shared_down family resolution failed");
  }
  const auto shared_down_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("shared_down_create", time_ms(shared_down_begin, shared_down_end));
  }

  if (!input_norm_weight ||
      !gate_score_correction_bias_device ||
      !gate_score_correction_bias_device->valid() ||
      gate_score_correction_bias_device->shape().size() != 1 ||
      gate_score_correction_bias_device->shape()[0] != config.n_routed_experts ||
      !gate_weight || !gate_weight->valid() ||
      !fc2_latent || !fc2_latent->valid() ||
      (fc1_latent_family == Impl::ProjectionFamily::kNone) ||
      (shared_up_family == Impl::ProjectionFamily::kNone)) {
    return debug_fail_with_cleanup("core weight materialization failed");
  }

  const std::size_t fc1_output_rows =
      fc1_latent_family == Impl::ProjectionFamily::kScaledFp8
          ? fc1_latent_scaled_fp8->output_rows()
          : fc1_latent_dense->output_rows();
  const std::size_t fc1_input_cols =
      fc1_latent_family == Impl::ProjectionFamily::kScaledFp8
          ? fc1_latent_scaled_fp8->input_cols()
          : fc1_latent_dense->input_cols();
  const std::size_t shared_up_output_rows =
      shared_up_family == Impl::ProjectionFamily::kScaledFp8
          ? shared_up_scaled_fp8->output_rows()
          : shared_up_dense->output_rows();
  const std::size_t shared_up_input_cols =
      shared_up_family == Impl::ProjectionFamily::kScaledFp8
          ? shared_up_scaled_fp8->input_cols()
          : shared_up_dense->input_cols();

  if (bindings.gate_weight->output_rows != config.n_routed_experts ||
      bindings.gate_weight->input_cols != config.hidden_size ||
      fc1_output_rows != config.moe_latent_size ||
      fc1_input_cols != config.hidden_size ||
      bindings.fc2_latent_weight->output_rows != config.hidden_size ||
      bindings.fc2_latent_weight->input_cols != config.moe_latent_size ||
      shared_up_output_rows != config.shared_expert_intermediate_size ||
      shared_up_input_cols != config.hidden_size) {
    return debug_fail_with_cleanup("core shape validation failed");
  }

  if (shared_down_family == Impl::SharedDownFamily::kNvfp4) {
    if (!shared_down_nvfp4 ||
        shared_down_nvfp4->output_rows() != config.hidden_size ||
        shared_down_nvfp4->input_cols() != config.shared_expert_intermediate_size) {
      return debug_fail_with_cleanup("shared_down NVFP4 shape validation failed");
    }
  } else if (shared_down_family == Impl::SharedDownFamily::kScaledFp8) {
    if (!shared_down_scaled_fp8 ||
        shared_down_scaled_fp8->output_rows() != config.hidden_size ||
        shared_down_scaled_fp8->input_cols() != config.shared_expert_intermediate_size) {
      return debug_fail_with_cleanup("shared_down scaled-fp8 shape validation failed");
    }
  } else if (shared_down_family == Impl::SharedDownFamily::kDense) {
    if (!shared_down_dense ||
        shared_down_dense->output_rows() != config.hidden_size ||
        shared_down_dense->input_cols() != config.shared_expert_intermediate_size) {
      return debug_fail_with_cleanup("shared_down dense shape validation failed");
    }
  } else {
    return debug_fail_with_cleanup("shared_down family missing after validation");
  }

  std::vector<Impl::RoutedExpertEntry> routed_experts(config.n_routed_experts);
  const auto routed_upload_begin = std::chrono::steady_clock::now();
  for (std::size_t expert_index = 0; expert_index < bindings.routed_experts.size(); ++expert_index) {
    const ExpertWeightPair& pair = bindings.routed_experts[expert_index];
    if (pair.up_proj == nullptr || pair.down_proj == nullptr) {
      continue;
    }
    if (pair.up_proj->output_rows != config.routed_expert_intermediate_size ||
        pair.up_proj->input_cols != config.moe_latent_size ||
        pair.down_proj->output_rows != config.moe_latent_size ||
        pair.down_proj->input_cols != config.routed_expert_intermediate_size) {
      return debug_fail_with_cleanup("routed expert shape validation failed");
    }
    routed_experts[expert_index].up_descriptor = *pair.up_proj;
    routed_experts[expert_index].down_descriptor = *pair.down_proj;
    routed_experts[expert_index].up_tensor_scale = ReadTensorScaleHost(*pair.up_proj);
    routed_experts[expert_index].down_tensor_scale = ReadTensorScaleHost(*pair.down_proj);
    if (pair.up_proj->kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
        pair.down_proj->kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
      routed_experts[expert_index].up_proj = UploadedLinearOp::Create(*pair.up_proj);
      routed_experts[expert_index].down_proj = UploadedLinearOp::Create(*pair.down_proj);
      if (!routed_experts[expert_index].up_proj || !routed_experts[expert_index].down_proj ||
          !routed_experts[expert_index].up_proj->valid() || !routed_experts[expert_index].down_proj->valid()) {
        return debug_fail_with_cleanup("routed expert upload failed (non-nvfp4)");
      }
    }
  }
  const auto routed_upload_end = std::chrono::steady_clock::now();
  if (timing_sink) {
    timing_sink("routed_expert_uploads", time_ms(routed_upload_begin, routed_upload_end));
  }
  const bool eager_routed_nvfp4_lookups =
      std::getenv("NEMOTRON_EAGER_ROUTED_NVFP4_LOOKUPS") != nullptr;
  const std::size_t routed_bias_topn = []() -> std::size_t {
    const char* env = std::getenv("NEMOTRON_ROUTED_LOOKUP_BIAS_TOPN");
    if (env == nullptr || *env == '\0') {
      return 0;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0') {
      return 0;
    }
    return static_cast<std::size_t>(parsed);
  }();
  if (eager_routed_nvfp4_lookups) {
    const auto eager_begin = std::chrono::steady_clock::now();
    for (auto& entry : routed_experts) {
      if (entry.up_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          entry.down_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          !entry.up_tensor_scale.has_value() ||
          !entry.down_tensor_scale.has_value()) {
        continue;
      }
      if (entry.up_nvfp4_buffers == nullptr) {
        entry.up_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (entry.down_nvfp4_buffers == nullptr) {
        entry.down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (!EnsureNvfp4LookupBuffers(
              entry.up_descriptor,
              entry.up_nvfp4_buffers.get(),
              &entry.up_nvfp4_lookup_ready) ||
          !EnsureNvfp4LookupBuffers(
              entry.down_descriptor,
              entry.down_nvfp4_buffers.get(),
              &entry.down_nvfp4_lookup_ready)) {
        return debug_fail_with_cleanup("eager routed NVFP4 lookup prep failed");
      }
    }
    const auto eager_end = std::chrono::steady_clock::now();
    if (timing_sink) {
      timing_sink("routed_expert_eager_nvfp4_prep", time_ms(eager_begin, eager_end));
    }
  } else if (routed_bias_topn != 0) {
    const auto bias_prefetch_begin = std::chrono::steady_clock::now();
    const auto bias_values = ReadTensorToHostFp32(*bindings.gate_score_correction_bias);
    if (!bias_values.has_value() || bias_values->size() != config.n_routed_experts) {
      return debug_fail_with_cleanup("bias-routed NVFP4 hotset read failed");
    }
    std::vector<std::size_t> ranked(config.n_routed_experts);
    std::iota(ranked.begin(), ranked.end(), std::size_t{0});
    const std::size_t topn = std::min(routed_bias_topn, ranked.size());
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + topn,
        ranked.end(),
        [&](std::size_t lhs, std::size_t rhs) {
          return (*bias_values)[lhs] > (*bias_values)[rhs];
        });
    for (std::size_t rank = 0; rank < topn; ++rank) {
      auto& entry = routed_experts[ranked[rank]];
      if (entry.up_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          entry.down_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          !entry.up_tensor_scale.has_value() ||
          !entry.down_tensor_scale.has_value()) {
        continue;
      }
      if (entry.up_nvfp4_buffers == nullptr) {
        entry.up_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (entry.down_nvfp4_buffers == nullptr) {
        entry.down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (!EnsureNvfp4LookupBuffers(
              entry.up_descriptor,
              entry.up_nvfp4_buffers.get(),
              &entry.up_nvfp4_lookup_ready) ||
          !EnsureNvfp4LookupBuffers(
              entry.down_descriptor,
              entry.down_nvfp4_buffers.get(),
              &entry.down_nvfp4_lookup_ready)) {
        return debug_fail_with_cleanup("bias-routed NVFP4 hotset prep failed");
      }
    }
    const auto bias_prefetch_end = std::chrono::steady_clock::now();
    if (timing_sink) {
      timing_sink("routed_expert_bias_hotset_prep", time_ms(bias_prefetch_begin, bias_prefetch_end));
    }
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->input_norm_weight = std::move(input_norm_weight);
  impl->gate_score_correction_bias_device = std::move(gate_score_correction_bias_device);
  impl->gate_weight = std::move(gate_weight);
  impl->fc1_latent_family = fc1_latent_family;
  impl->fc1_latent_dense = std::move(fc1_latent_dense);
  impl->fc1_latent_scaled_fp8 = std::move(fc1_latent_scaled_fp8);
  impl->fc2_latent = std::move(fc2_latent);
  impl->shared_up_family = shared_up_family;
  impl->shared_up_dense = std::move(shared_up_dense);
  impl->shared_up_scaled_fp8 = std::move(shared_up_scaled_fp8);
  impl->shared_down_family = shared_down_family;
  impl->shared_down_dense = std::move(shared_down_dense);
  impl->shared_down_scaled_fp8 = std::move(shared_down_scaled_fp8);
  impl->shared_down_nvfp4_buffers_ready = shared_down_nvfp4_buffers_ready;
  impl->shared_down_nvfp4_buffers = std::move(shared_down_nvfp4_buffers);
  impl->shared_down_nvfp4 = std::move(shared_down_nvfp4);
  impl->routed_experts = std::move(routed_experts);
  impl->dense_pool = dense_pool;
  impl->dense_pool_numel = dense_pool_numel;
  impl->expert_pool = expert_pool;
  impl->expert_pool_bytes = expert_pool_bytes;
  if (!impl->routed_up_packed_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_up_raw_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_up_tensor_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_down_packed_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_down_raw_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_down_tensor_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_up_packed_lookup_device.FillZero() ||
      !impl->routed_up_raw_scale_lookup_device.FillZero() ||
      !impl->routed_up_tensor_scale_lookup_device.FillZero() ||
      !impl->routed_down_packed_lookup_device.FillZero() ||
      !impl->routed_down_raw_scale_lookup_device.FillZero() ||
      !impl->routed_down_tensor_scale_lookup_device.FillZero()) {
    impl->grouped_routed_nvfp4_enabled.store(false, std::memory_order_relaxed);
    if (debug) {
      std::cerr << "expert_layer_create: layer " << config.layer_index
                << " disabled routed fastpath during lookup buffer init\n";
    }
  }
  if ((eager_routed_nvfp4_lookups || routed_bias_topn != 0) &&
      impl->grouped_routed_nvfp4_enabled.load(std::memory_order_relaxed)) {
    for (std::size_t expert_index = 0; expert_index < impl->routed_experts.size(); ++expert_index) {
      auto& entry = impl->routed_experts[expert_index];
      if (!entry.up_nvfp4_lookup_ready ||
          !entry.down_nvfp4_lookup_ready ||
          entry.up_nvfp4_buffers == nullptr ||
          entry.down_nvfp4_buffers == nullptr ||
          !entry.up_tensor_scale.has_value() ||
          !entry.down_tensor_scale.has_value()) {
        continue;
      }
      const void* up_packed = entry.up_nvfp4_buffers->packed.data();
      const void* up_raw_scales = entry.up_nvfp4_buffers->block_scales.data();
      const float up_tensor_scale = *entry.up_tensor_scale;
      const void* down_packed = entry.down_nvfp4_buffers->packed.data();
      const void* down_raw_scales = entry.down_nvfp4_buffers->block_scales.data();
      const float down_tensor_scale = *entry.down_tensor_scale;
      const bool copied =
          cudaMemcpy(
              impl->routed_up_packed_lookup_device.data() + expert_index,
              &up_packed,
              sizeof(up_packed),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_up_raw_scale_lookup_device.data() + expert_index,
              &up_raw_scales,
              sizeof(up_raw_scales),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_up_tensor_scale_lookup_device.data() + expert_index,
              &up_tensor_scale,
              sizeof(up_tensor_scale),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_down_packed_lookup_device.data() + expert_index,
              &down_packed,
              sizeof(down_packed),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_down_raw_scale_lookup_device.data() + expert_index,
              &down_raw_scales,
              sizeof(down_raw_scales),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_down_tensor_scale_lookup_device.data() + expert_index,
              &down_tensor_scale,
              sizeof(down_tensor_scale),
              cudaMemcpyHostToDevice) == cudaSuccess;
      if (!copied) {
        impl->grouped_routed_nvfp4_enabled.store(false, std::memory_order_relaxed);
        break;
      }
      entry.grouped_lookup_ready = true;
    }
  }
  // Pre-allocate scratch buffers for try_grouped_routed_single_token.
  // All sizes derive from config and are stable for the lifetime of this layer.
  {
    constexpr std::size_t kNvfp4BlockWidth = 16;
    constexpr std::size_t kScaleRowTile = 128;
    constexpr std::size_t kScaleBlockTile = 4;
    constexpr std::size_t kCutlassAlign = 256;
    const auto round_up = [](std::size_t value, std::size_t alignment) -> std::size_t {
      return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
    };

    const std::size_t top_k = config.top_k;
    const std::size_t latent = config.moe_latent_size;
    const std::size_t intermediate = config.routed_expert_intermediate_size;

    // -- Gather output buffers (sized for top_k selections) --
    impl->scratch_up_packed_ptrs.Resize(top_k);
    impl->scratch_up_raw_scale_ptrs.Resize(top_k);
    impl->scratch_down_packed_ptrs.Resize(top_k);
    impl->scratch_down_raw_scale_ptrs.Resize(top_k);
    impl->scratch_missing_count.Resize(1);
    impl->scratch_missing_indices.Resize(top_k);
    impl->scratch_selected_up_tensor_scales.Resize(top_k);
    impl->scratch_selected_down_tensor_scales.Resize(top_k);

    // -- NVFP4 packing buffers for latent activation (M=1, K=latent) --
    impl->scratch_latent_packed_data.Resize((1 * latent + 1u) / 2u);
    impl->scratch_latent_block_scales.Resize(latent / kNvfp4BlockWidth);
    impl->scratch_latent_matmul_scales.Resize(
        round_up(1, kScaleRowTile) * round_up(latent / kNvfp4BlockWidth, kScaleBlockTile));
    impl->scratch_latent_tensor_scale.Resize(sizeof(float));
    impl->scratch_global_max_bits.Resize(1);

    // -- ScaleRelu2 output buffers (top_k rows, intermediate cols) --
    impl->scratch_down_act_packed.Resize((top_k * intermediate + 1u) / 2u);
    impl->scratch_down_act_block_scales.Resize(top_k * (intermediate / kNvfp4BlockWidth));
    impl->scratch_down_act_tensor_scales.Resize(top_k);

    // -- Pre-filled row scales (constant 1.0f, avoids per-token H→D copy) --
    if (impl->scratch_row_scales.Resize(top_k)) {
      std::vector<float> ones(top_k, 1.0f);
      impl->scratch_row_scales.CopyFromHost(ones);
    }

    // -- CUTLASS path aligned buffers --
    const std::size_t act_packed_row_bytes = intermediate / 2;
    const std::size_t act_scale_row_bytes = intermediate / kNvfp4BlockWidth;
    const std::size_t aligned_packed_row = round_up(act_packed_row_bytes, kCutlassAlign);
    const std::size_t aligned_scale_row = round_up(act_scale_row_bytes, kCutlassAlign);
    impl->scratch_aligned_act_packed.Resize(top_k * aligned_packed_row);
    impl->scratch_aligned_act_scales.Resize(top_k * aligned_scale_row);
  }

  std::string strict_failure_reason;
  if (!ConfigureRoutedMoEBackend(
          impl.get(),
          impl->routed_experts,
          debug,
          &strict_failure_reason)) {
    return debug_fail_with_cleanup(strict_failure_reason.c_str());
  }
  return std::unique_ptr<ExpertLayerSlice>(new ExpertLayerSlice(std::move(impl)));
}

std::unique_ptr<ExpertLayerSlice> ExpertLayerSlice::CreatePrepared(
    const ExpertLayerConfig& config,
    ExpertLayerPreparedBindings bindings) {
  if (config.hidden_size == 0 ||
      config.moe_latent_size == 0 ||
      config.routed_expert_intermediate_size == 0 ||
      config.shared_expert_intermediate_size == 0 ||
      config.n_routed_experts == 0 ||
      config.top_k == 0 ||
      config.top_k > config.n_routed_experts ||
      config.n_group == 0 ||
      config.topk_group == 0 ||
      config.n_routed_experts % config.n_group != 0 ||
      config.rms_epsilon <= 0.0f ||
      !bindings.input_norm_weight ||
      !bindings.input_norm_weight->valid() ||
      !bindings.gate_score_correction_bias_device ||
      !bindings.gate_score_correction_bias_device->valid() ||
      !bindings.gate_weight ||
      !bindings.gate_weight->valid() ||
      !bindings.fc2_latent ||
      !bindings.fc2_latent->valid() ||
      bindings.routed_experts.size() != config.n_routed_experts) {
    return nullptr;
  }

  Impl::ProjectionFamily fc1_latent_family = Impl::ProjectionFamily::kNone;
  if (bindings.fc1_latent_dense && bindings.fc1_latent_dense->valid()) {
    fc1_latent_family = Impl::ProjectionFamily::kDense;
  } else if (bindings.fc1_latent_scaled_fp8 && bindings.fc1_latent_scaled_fp8->valid()) {
    fc1_latent_family = Impl::ProjectionFamily::kScaledFp8;
  }
  Impl::ProjectionFamily shared_up_family = Impl::ProjectionFamily::kNone;
  if (bindings.shared_up_dense && bindings.shared_up_dense->valid()) {
    shared_up_family = Impl::ProjectionFamily::kDense;
  } else if (bindings.shared_up_scaled_fp8 && bindings.shared_up_scaled_fp8->valid()) {
    shared_up_family = Impl::ProjectionFamily::kScaledFp8;
  }
  Impl::SharedDownFamily shared_down_family = Impl::SharedDownFamily::kNone;
  if (bindings.shared_down_dense && bindings.shared_down_dense->valid()) {
    shared_down_family = Impl::SharedDownFamily::kDense;
  } else if (bindings.shared_down_scaled_fp8 && bindings.shared_down_scaled_fp8->valid()) {
    shared_down_family = Impl::SharedDownFamily::kScaledFp8;
  } else if (bindings.shared_down_nvfp4 && bindings.shared_down_nvfp4->valid()) {
    shared_down_family = Impl::SharedDownFamily::kNvfp4;
  }
  if (fc1_latent_family == Impl::ProjectionFamily::kNone ||
      shared_up_family == Impl::ProjectionFamily::kNone ||
      shared_down_family == Impl::SharedDownFamily::kNone) {
    return nullptr;
  }

  const std::size_t fc1_output_rows =
      fc1_latent_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.fc1_latent_scaled_fp8->output_rows()
          : bindings.fc1_latent_dense->output_rows();
  const std::size_t fc1_input_cols =
      fc1_latent_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.fc1_latent_scaled_fp8->input_cols()
          : bindings.fc1_latent_dense->input_cols();
  const std::size_t shared_up_output_rows =
      shared_up_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.shared_up_scaled_fp8->output_rows()
          : bindings.shared_up_dense->output_rows();
  const std::size_t shared_up_input_cols =
      shared_up_family == Impl::ProjectionFamily::kScaledFp8
          ? bindings.shared_up_scaled_fp8->input_cols()
          : bindings.shared_up_dense->input_cols();
  if (bindings.gate_weight->output_rows() != config.n_routed_experts ||
      bindings.gate_weight->input_cols() != config.hidden_size ||
      fc1_output_rows != config.moe_latent_size ||
      fc1_input_cols != config.hidden_size ||
      bindings.fc2_latent->output_rows() != config.hidden_size ||
      bindings.fc2_latent->input_cols() != config.moe_latent_size ||
      shared_up_output_rows != config.shared_expert_intermediate_size ||
      shared_up_input_cols != config.hidden_size) {
    return nullptr;
  }

  if (shared_down_family == Impl::SharedDownFamily::kNvfp4) {
    if (bindings.shared_down_nvfp4->output_rows() != config.hidden_size ||
        bindings.shared_down_nvfp4->input_cols() != config.shared_expert_intermediate_size) {
      return nullptr;
    }
  } else if (shared_down_family == Impl::SharedDownFamily::kScaledFp8) {
    if (bindings.shared_down_scaled_fp8->output_rows() != config.hidden_size ||
        bindings.shared_down_scaled_fp8->input_cols() != config.shared_expert_intermediate_size) {
      return nullptr;
    }
  } else if (bindings.shared_down_dense->output_rows() != config.hidden_size ||
             bindings.shared_down_dense->input_cols() != config.shared_expert_intermediate_size) {
    return nullptr;
  }

  std::vector<Impl::RoutedExpertEntry> routed_experts(config.n_routed_experts);
  for (std::size_t expert_index = 0; expert_index < bindings.routed_experts.size(); ++expert_index) {
    auto& prepared = bindings.routed_experts[expert_index];
    if ((prepared.up_proj == nullptr) != (prepared.down_proj == nullptr)) {
      return nullptr;
    }
    if (prepared.up_proj != nullptr) {
      if (!prepared.up_proj->valid() ||
          !prepared.down_proj->valid() ||
          prepared.up_proj->output_rows() != config.routed_expert_intermediate_size ||
          prepared.up_proj->input_cols() != config.moe_latent_size ||
          prepared.down_proj->output_rows() != config.moe_latent_size ||
          prepared.down_proj->input_cols() != config.routed_expert_intermediate_size) {
        return nullptr;
      }
      routed_experts[expert_index].up_descriptor = prepared.up_descriptor;
      routed_experts[expert_index].down_descriptor = prepared.down_descriptor;
      routed_experts[expert_index].up_tensor_scale = prepared.up_tensor_scale;
      routed_experts[expert_index].down_tensor_scale = prepared.down_tensor_scale;
      routed_experts[expert_index].up_proj = std::move(prepared.up_proj);
      routed_experts[expert_index].down_proj = std::move(prepared.down_proj);
    }
  }
  const bool eager_routed_nvfp4_lookups =
      std::getenv("NEMOTRON_EAGER_ROUTED_NVFP4_LOOKUPS") != nullptr;
  const std::size_t routed_bias_topn = []() -> std::size_t {
    const char* env = std::getenv("NEMOTRON_ROUTED_LOOKUP_BIAS_TOPN");
    if (env == nullptr || *env == '\0') {
      return 0;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0') {
      return 0;
    }
    return static_cast<std::size_t>(parsed);
  }();
  if (eager_routed_nvfp4_lookups) {
    for (auto& entry : routed_experts) {
      if (entry.up_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          entry.down_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          !entry.up_tensor_scale.has_value() ||
          !entry.down_tensor_scale.has_value()) {
        continue;
      }
      if (entry.up_nvfp4_buffers == nullptr) {
        entry.up_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (entry.down_nvfp4_buffers == nullptr) {
        entry.down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (!EnsureNvfp4LookupBuffers(
              entry.up_descriptor,
              entry.up_nvfp4_buffers.get(),
              &entry.up_nvfp4_lookup_ready) ||
          !EnsureNvfp4LookupBuffers(
              entry.down_descriptor,
              entry.down_nvfp4_buffers.get(),
              &entry.down_nvfp4_lookup_ready)) {
        return nullptr;
      }
    }
  } else if (routed_bias_topn != 0) {
    std::vector<float> bias_values(config.n_routed_experts, 0.0f);
    if (!bindings.gate_score_correction_bias_device ||
        !bindings.gate_score_correction_bias_device->CopyToHost(
            bias_values.data(),
            bias_values.size())) {
      return nullptr;
    }
    std::vector<std::size_t> ranked(config.n_routed_experts);
    std::iota(ranked.begin(), ranked.end(), std::size_t{0});
    const std::size_t topn = std::min(routed_bias_topn, ranked.size());
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + topn,
        ranked.end(),
        [&](std::size_t lhs, std::size_t rhs) {
          return bias_values[lhs] > bias_values[rhs];
        });
    for (std::size_t rank = 0; rank < topn; ++rank) {
      auto& entry = routed_experts[ranked[rank]];
      if (entry.up_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          entry.down_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
          !entry.up_tensor_scale.has_value() ||
          !entry.down_tensor_scale.has_value()) {
        continue;
      }
      if (entry.up_nvfp4_buffers == nullptr) {
        entry.up_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (entry.down_nvfp4_buffers == nullptr) {
        entry.down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      if (!EnsureNvfp4LookupBuffers(
              entry.up_descriptor,
              entry.up_nvfp4_buffers.get(),
              &entry.up_nvfp4_lookup_ready) ||
          !EnsureNvfp4LookupBuffers(
              entry.down_descriptor,
              entry.down_nvfp4_buffers.get(),
              &entry.down_nvfp4_lookup_ready)) {
        return nullptr;
      }
    }
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->input_norm_weight = std::move(bindings.input_norm_weight);
  impl->gate_score_correction_bias_device = std::move(bindings.gate_score_correction_bias_device);
  impl->gate_weight = std::move(bindings.gate_weight);
  impl->fc1_latent_family = fc1_latent_family;
  impl->fc1_latent_dense = std::move(bindings.fc1_latent_dense);
  impl->fc1_latent_scaled_fp8 = std::move(bindings.fc1_latent_scaled_fp8);
  impl->fc2_latent = std::move(bindings.fc2_latent);
  impl->shared_up_family = shared_up_family;
  impl->shared_up_dense = std::move(bindings.shared_up_dense);
  impl->shared_up_scaled_fp8 = std::move(bindings.shared_up_scaled_fp8);
  impl->shared_down_family = shared_down_family;
  impl->shared_down_dense = std::move(bindings.shared_down_dense);
  impl->shared_down_scaled_fp8 = std::move(bindings.shared_down_scaled_fp8);
  impl->shared_down_nvfp4 = std::move(bindings.shared_down_nvfp4);
  impl->routed_experts = std::move(routed_experts);
  if (!impl->routed_up_packed_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_up_raw_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_up_tensor_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_down_packed_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_down_raw_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_down_tensor_scale_lookup_device.Resize(config.n_routed_experts) ||
      !impl->routed_up_packed_lookup_device.FillZero() ||
      !impl->routed_up_raw_scale_lookup_device.FillZero() ||
      !impl->routed_up_tensor_scale_lookup_device.FillZero() ||
      !impl->routed_down_packed_lookup_device.FillZero() ||
      !impl->routed_down_raw_scale_lookup_device.FillZero() ||
      !impl->routed_down_tensor_scale_lookup_device.FillZero()) {
    impl->grouped_routed_nvfp4_enabled.store(false, std::memory_order_relaxed);
  }
  if ((eager_routed_nvfp4_lookups || routed_bias_topn != 0) &&
      impl->grouped_routed_nvfp4_enabled.load(std::memory_order_relaxed)) {
    for (std::size_t expert_index = 0; expert_index < impl->routed_experts.size(); ++expert_index) {
      auto& entry = impl->routed_experts[expert_index];
      if (!entry.up_nvfp4_lookup_ready ||
          !entry.down_nvfp4_lookup_ready ||
          entry.up_nvfp4_buffers == nullptr ||
          entry.down_nvfp4_buffers == nullptr ||
          !entry.up_tensor_scale.has_value() ||
          !entry.down_tensor_scale.has_value()) {
        continue;
      }
      const void* up_packed = entry.up_nvfp4_buffers->packed.data();
      const void* up_raw_scales = entry.up_nvfp4_buffers->block_scales.data();
      const float up_tensor_scale = *entry.up_tensor_scale;
      const void* down_packed = entry.down_nvfp4_buffers->packed.data();
      const void* down_raw_scales = entry.down_nvfp4_buffers->block_scales.data();
      const float down_tensor_scale = *entry.down_tensor_scale;
      const bool copied =
          cudaMemcpy(
              impl->routed_up_packed_lookup_device.data() + expert_index,
              &up_packed,
              sizeof(up_packed),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_up_raw_scale_lookup_device.data() + expert_index,
              &up_raw_scales,
              sizeof(up_raw_scales),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_up_tensor_scale_lookup_device.data() + expert_index,
              &up_tensor_scale,
              sizeof(up_tensor_scale),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_down_packed_lookup_device.data() + expert_index,
              &down_packed,
              sizeof(down_packed),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_down_raw_scale_lookup_device.data() + expert_index,
              &down_raw_scales,
              sizeof(down_raw_scales),
              cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaMemcpy(
              impl->routed_down_tensor_scale_lookup_device.data() + expert_index,
              &down_tensor_scale,
              sizeof(down_tensor_scale),
              cudaMemcpyHostToDevice) == cudaSuccess;
      if (!copied) {
        impl->grouped_routed_nvfp4_enabled.store(false, std::memory_order_relaxed);
        break;
      }
      entry.grouped_lookup_ready = true;
    }
  }
  std::string strict_failure_reason;
  if (!ConfigureRoutedMoEBackend(
          impl.get(),
          impl->routed_experts,
          false,
          &strict_failure_reason)) {
    return nullptr;
  }

  // Pre-allocate CUTLASS grouped GEMM plans and device pointer arrays.
  if (CutlassNvfp4GroupedGemmAvailable()) {
    const int top_k = static_cast<int>(config.top_k);
    impl->cutlass_up_plan = CutlassNvfp4GroupedGemmPlan::Create(
        top_k, 1,
        static_cast<int>(config.routed_expert_intermediate_size),
        static_cast<int>(config.moe_latent_size));
    impl->cutlass_down_plan = CutlassNvfp4GroupedGemmPlan::Create(
        top_k, 1,
        static_cast<int>(config.moe_latent_size),
        static_cast<int>(config.routed_expert_intermediate_size));
    // Pre-allocate device pointer arrays (reused every token).
    impl->cutlass_a_ptrs.Resize(static_cast<std::size_t>(top_k));
    impl->cutlass_a_sf_ptrs.Resize(static_cast<std::size_t>(top_k));
    impl->cutlass_c_ptrs.Resize(static_cast<std::size_t>(top_k));
    impl->cutlass_d_ptrs.Resize(static_cast<std::size_t>(top_k));
  }

  return std::unique_ptr<ExpertLayerSlice>(new ExpertLayerSlice(std::move(impl)));
}

ExpertLayerSlice::ExpertLayerSlice(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExpertLayerSlice::ExpertLayerSlice(ExpertLayerSlice&&) noexcept = default;
ExpertLayerSlice& ExpertLayerSlice::operator=(ExpertLayerSlice&&) noexcept = default;
ExpertLayerSlice::~ExpertLayerSlice() = default;

bool ExpertLayerSlice::valid() const {
  if (impl_ == nullptr ||
      impl_->routed_experts.size() != impl_->config.n_routed_experts) {
    return false;
  }
  if (impl_->routed_backend_kind == RoutedMoEBackendKind::kFlashInfer &&
      (impl_->flashinfer_routed_backend == nullptr ||
       !impl_->flashinfer_routed_backend->valid())) {
    return false;
  }
  for (const auto& routed_expert : impl_->routed_experts) {
    if ((routed_expert.up_proj != nullptr && !routed_expert.up_proj->valid()) ||
        (routed_expert.down_proj != nullptr && !routed_expert.down_proj->valid())) {
      return false;
    }
  }
  return
         impl_->input_norm_weight != nullptr &&
         impl_->input_norm_weight->valid() &&
         impl_->gate_score_correction_bias_device != nullptr &&
         impl_->gate_score_correction_bias_device->valid() &&
         impl_->gate_weight != nullptr &&
         impl_->gate_weight->valid() &&
         ((impl_->fc1_latent_family == Impl::ProjectionFamily::kDense &&
           impl_->fc1_latent_dense != nullptr &&
           impl_->fc1_latent_dense->valid()) ||
          (impl_->fc1_latent_family == Impl::ProjectionFamily::kScaledFp8 &&
           impl_->fc1_latent_scaled_fp8 != nullptr &&
           impl_->fc1_latent_scaled_fp8->valid())) &&
         impl_->fc2_latent != nullptr &&
         impl_->fc2_latent->valid() &&
         ((impl_->shared_up_family == Impl::ProjectionFamily::kDense &&
           impl_->shared_up_dense != nullptr &&
           impl_->shared_up_dense->valid()) ||
          (impl_->shared_up_family == Impl::ProjectionFamily::kScaledFp8 &&
           impl_->shared_up_scaled_fp8 != nullptr &&
           impl_->shared_up_scaled_fp8->valid())) &&
         ((impl_->shared_down_family == Impl::SharedDownFamily::kNvfp4 &&
           impl_->shared_down_nvfp4 != nullptr &&
           impl_->shared_down_nvfp4->valid()) ||
          (impl_->shared_down_family == Impl::SharedDownFamily::kScaledFp8 &&
           impl_->shared_down_scaled_fp8 != nullptr &&
           impl_->shared_down_scaled_fp8->valid()) ||
          (impl_->shared_down_family == Impl::SharedDownFamily::kDense &&
           impl_->shared_down_dense != nullptr &&
           impl_->shared_down_dense->valid()));
}

const ExpertLayerConfig& ExpertLayerSlice::config() const {
  return impl_->config;
}

namespace {

template <typename ImplT>
bool RunExpertLayerImpl(
    const ImplT& impl,
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext* request_context,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    ExpertLayerRunTrace* trace) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const std::size_t routed_prefetch_topn = []() -> std::size_t {
    const char* env = std::getenv("NEMOTRON_ROUTED_LOOKUP_PREFETCH_TOPN");
    if (env == nullptr || *env == '\0') {
      return 0;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0') {
      return 0;
    }
    return static_cast<std::size_t>(parsed);
  }();

  const auto use_request_expert_aux_scratch =
      [&](std::size_t token_count,
          std::unique_ptr<DeviceTensorFp32>* normalized_out,
          std::unique_ptr<DeviceTensorFp32>* router_logits_out,
          std::unique_ptr<DeviceTensorFp32>* latent_out,
          std::unique_ptr<DeviceTensorFp32>* routed_tensor_out,
          std::unique_ptr<DeviceTensorFp32>* projected_routed_out,
          std::unique_ptr<DeviceTensorFp32>* shared_up_out,
          std::unique_ptr<DeviceTensorFp32>* shared_output_out,
          std::unique_ptr<DeviceTensorFp32>* mixer_output_out,
          std::unique_ptr<DeviceTensorFp32>* expert_down_out) -> bool {
    if (token_count != 1 ||
        request_context == nullptr ||
        request_context->expert_aux_scratch() == nullptr ||
        !request_context->expert_aux_scratch()->valid()) {
      return false;
    }
    const std::size_t required_numel =
        (4 * impl.config.hidden_size) +
        (3 * impl.config.moe_latent_size) +
        impl.config.shared_expert_intermediate_size +
        impl.config.n_routed_experts;
    if (request_context->expert_aux_scratch()->numel() < required_numel) {
      return false;
    }
    float* cursor = request_context->expert_aux_scratch()->data();
    auto make_view = [&](std::size_t cols) -> std::unique_ptr<DeviceTensorFp32> {
      auto view = DeviceTensorFp32::CreateView({1, cols}, cursor);
      cursor += cols;
      return view;
    };
    *normalized_out = make_view(impl.config.hidden_size);
    *router_logits_out = make_view(impl.config.n_routed_experts);
    *latent_out = make_view(impl.config.moe_latent_size);
    *routed_tensor_out = make_view(impl.config.moe_latent_size);
    *projected_routed_out = make_view(impl.config.hidden_size);
    *shared_up_out = make_view(impl.config.shared_expert_intermediate_size);
    *shared_output_out = make_view(impl.config.hidden_size);
    *mixer_output_out = make_view(impl.config.hidden_size);
    *expert_down_out = make_view(impl.config.moe_latent_size);
    return *normalized_out && *router_logits_out && *latent_out && *routed_tensor_out &&
           *projected_routed_out && *shared_up_out && *shared_output_out &&
           *mixer_output_out && *expert_down_out;
  };

  if ((request_context != nullptr && !request_context->valid()) ||
      !cublas_handle.valid() ||
      !input.valid() ||
      input.shape().size() != 2 ||
      input.shape()[1] != impl.config.hidden_size ||
      output == nullptr ||
      !output->valid() ||
      output->shape() != input.shape()) {
    if (debug) {
      std::cout << "expert_layer: invalid run inputs or state\n";
    }
    return false;
  }

  const std::size_t token_count = input.shape()[0];
  if (token_count == 0 || (trace != nullptr && token_count != 1)) {
    if (debug) {
      std::cout << "expert_layer: unsupported token count/tracing combination\n";
    }
    return false;
  }

  std::unique_ptr<DeviceTensorFp32> normalized;
  std::unique_ptr<DeviceTensorFp32> router_logits;
  std::unique_ptr<DeviceTensorFp32> latent;
  std::unique_ptr<DeviceTensorFp32> routed_tensor;
  std::unique_ptr<DeviceTensorFp32> projected_routed;
  std::unique_ptr<DeviceTensorFp32> shared_up;
  std::unique_ptr<DeviceTensorFp32> shared_output_device;
  std::unique_ptr<DeviceTensorFp32> mixer_output_device;
  std::unique_ptr<DeviceTensorFp32> expert_down;
  if (!use_request_expert_aux_scratch(
          token_count,
          &normalized,
          &router_logits,
          &latent,
          &routed_tensor,
          &projected_routed,
          &shared_up,
          &shared_output_device,
          &mixer_output_device,
          &expert_down)) {
    normalized = DeviceTensorFp32::Create({token_count, impl.config.hidden_size});
    router_logits = DeviceTensorFp32::Create({token_count, impl.config.n_routed_experts});
    latent = DeviceTensorFp32::Create({token_count, impl.config.moe_latent_size});
    routed_tensor = DeviceTensorFp32::Create({token_count, impl.config.moe_latent_size});
    projected_routed = DeviceTensorFp32::Create({token_count, impl.config.hidden_size});
    shared_up = DeviceTensorFp32::Create({token_count, impl.config.shared_expert_intermediate_size});
    shared_output_device = DeviceTensorFp32::Create({token_count, impl.config.hidden_size});
    mixer_output_device = DeviceTensorFp32::Create({token_count, impl.config.hidden_size});
    expert_down = DeviceTensorFp32::Create({1, impl.config.moe_latent_size});
  }
  if (!normalized || !router_logits || !latent || !routed_tensor || !projected_routed ||
      !shared_up || !shared_output_device || !mixer_output_device || !expert_down) {
    if (debug) {
      std::cout << "expert_layer: scratch allocation failed\n";
    }
    return false;
  }

  const bool norm_ok =
      RmsNormFp32(input, *impl.input_norm_weight, impl.config.rms_epsilon, normalized.get());
  const bool gate_ok =
      norm_ok &&
      impl.gate_weight->Run(cublas_handle, heuristic_cache, *normalized, router_logits.get());
  if (!norm_ok || !gate_ok) {
    if (debug) {
      std::cout << "expert_layer: norm or gate projection failed"
                << " norm_ok=" << norm_ok
                << " gate_ok=" << gate_ok << "\n";
    }
    return false;
  }

  const bool fc1_ok =
      (impl.fc1_latent_scaled_fp8 != nullptr &&
       impl.fc1_latent_scaled_fp8->Run(cublas_handle, heuristic_cache, *normalized, latent.get())) ||
      (impl.fc1_latent_dense != nullptr &&
       impl.fc1_latent_dense->Run(cublas_handle, heuristic_cache, *normalized, latent.get()));
  const bool shared_up_ok =
      (impl.shared_up_scaled_fp8 != nullptr &&
       impl.shared_up_scaled_fp8->Run(cublas_handle, heuristic_cache, *normalized, shared_up.get())) ||
      (impl.shared_up_dense != nullptr &&
       impl.shared_up_dense->Run(cublas_handle, heuristic_cache, *normalized, shared_up.get()));
  if (!fc1_ok || !shared_up_ok) {
    if (debug) {
      std::cout << "expert_layer: fc1/shared_up projection failed"
                << " fc1_ok=" << fc1_ok
                << " shared_up_ok=" << shared_up_ok << "\n";
    }
    return false;
  }

  if (!routed_tensor->FillZero()) {
    if (debug) {
      std::cout << "expert_layer: routed tensor zero fill failed\n";
    }
    return false;
  }

  std::unique_ptr<DeviceTensorFp32> expert_up;
  if (request_context != nullptr &&
      request_context->expert_intermediate_scratch() != nullptr &&
      request_context->expert_intermediate_scratch()->valid() &&
      request_context->expert_intermediate_scratch()->numel() >=
          impl.config.routed_expert_intermediate_size) {
    expert_up = DeviceTensorFp32::CreateView(
        {1, impl.config.routed_expert_intermediate_size},
        request_context->expert_intermediate_scratch()->data());
  } else {
    expert_up = DeviceTensorFp32::Create({1, impl.config.routed_expert_intermediate_size});
  }
  std::unique_ptr<DeviceTensorFp32> latent_row;
  if (token_count != 1) {
    latent_row = DeviceTensorFp32::Create({1, impl.config.moe_latent_size});
  }
  if (!expert_up ||
      (token_count != 1 && !latent_row)) {
    if (debug) {
      std::cout << "expert_layer: expert scratch allocation failed\n";
    }
    return false;
  }

  const std::size_t selection_count = token_count * impl.config.top_k;
  DeviceBuffer<std::int32_t> local_selection_indices_device;
  DeviceBuffer<float> local_selection_weights_device;
  std::vector<std::int32_t> local_selection_indices_host;
  std::vector<float> local_selection_weights_host;
  SelectionScratchView selection_scratch;
  if (!PrepareSelectionScratch(
          request_context,
          selection_count,
          &local_selection_indices_device,
          &local_selection_weights_device,
          &local_selection_indices_host,
          &local_selection_weights_host,
          &selection_scratch)) {
    if (debug) {
      std::cout << "expert_layer: selection scratch preparation failed\n";
    }
    return false;
  }

  if (!SelectTopExpertsFp32(
          *router_logits,
          *impl.gate_score_correction_bias_device,
          impl.config.n_group,
          impl.config.topk_group,
          impl.config.top_k,
          impl.config.norm_topk_prob,
          impl.config.routed_scaling_factor,
          selection_scratch.indices_device->data(),
          selection_scratch.weights_device->data())) {
    if (debug) {
      std::cout << "expert_layer: device top-k selection failed\n";
      }
    return false;
  }

  auto& selected_indices_host = *selection_scratch.indices_host;
  auto& selected_weights_host = *selection_scratch.weights_host;
  bool selection_metadata_downloaded = false;
  const auto ensure_selection_metadata_host =
      [&]() -> bool {
    if (selection_metadata_downloaded) {
      return true;
    }
    if (!selection_scratch.indices_device->CopyToHost(
            selected_indices_host.data(),
            selection_count) ||
        !selection_scratch.weights_device->CopyToHost(
            selected_weights_host.data(),
            selection_count)) {
      return false;
    }
    selection_metadata_downloaded = true;
    RecordExpertSelectionMetadataDownload();
    return true;
  };
  std::vector<ExpertSelection> trace_selections;
  std::vector<float> trace_router_logits;
  std::vector<float> trace_latent_output;
  std::vector<float> trace_routed_output;
  std::vector<std::size_t> trace_routed_expert_order;
  std::vector<float> trace_routed_activated_hidden;
  std::vector<float> trace_routed_expert_outputs;
  std::vector<float> trace_routed_weighted_contributions;
  std::vector<float> trace_shared_output;
  std::vector<float> normalized_host;
  std::vector<float> router_logits_host;
  std::vector<float> latent_host;
  if (trace != nullptr) {
    if (!CopyToHost(*normalized, &normalized_host) ||
        !CopyToHost(*router_logits, &router_logits_host) ||
        !CopyToHost(*latent, &latent_host)) {
      if (debug) {
        std::cout << "expert_layer: trace tensor download failed\n";
      }
      return false;
    }
  }

  using RoutedExpertEntry = typename std::decay_t<decltype(impl.routed_experts)>::value_type;
  const auto update_routed_nvfp4_lookup =
      [&](std::size_t expert_index, const RoutedExpertEntry& entry) -> bool {
    if (expert_index >= impl.routed_experts.size() ||
        entry.up_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
        entry.down_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
        !entry.up_tensor_scale.has_value() ||
        !entry.down_tensor_scale.has_value()) {
      return false;
    }

    const void* up_packed = nullptr;
    const void* up_raw_scales = nullptr;
    if (entry.up_proj != nullptr &&
        entry.up_proj->kernel_family() == GemmKernelFamily::kCublasLtNvfp4BlockScaled &&
        entry.up_proj->nvfp4_weight() != nullptr) {
      const DeviceNvfp4Weight* up_weight = entry.up_proj->nvfp4_weight();
      up_packed = up_weight->packed_data();
      up_raw_scales = up_weight->block_scales_data();
    } else if ((entry.up_nvfp4_lookup_ready || entry.up_nvfp4_buffers_ready) &&
               entry.up_nvfp4_buffers != nullptr &&
               entry.up_nvfp4_buffers->lookup_valid()) {
      up_packed = entry.up_nvfp4_buffers->packed.data();
      up_raw_scales = entry.up_nvfp4_buffers->block_scales.data();
    } else {
      return false;
    }

    const float up_tensor_scale = *entry.up_tensor_scale;

    const void* down_packed = nullptr;
    const void* down_raw_scales = nullptr;
    if (entry.down_proj != nullptr &&
        entry.down_proj->kernel_family() == GemmKernelFamily::kCublasLtNvfp4BlockScaled &&
        entry.down_proj->nvfp4_weight() != nullptr) {
      const DeviceNvfp4Weight* down_weight = entry.down_proj->nvfp4_weight();
      down_packed = down_weight->packed_data();
      down_raw_scales = down_weight->block_scales_data();
    } else if ((entry.down_nvfp4_lookup_ready || entry.down_nvfp4_buffers_ready) &&
               entry.down_nvfp4_buffers != nullptr &&
               entry.down_nvfp4_buffers->lookup_valid()) {
      down_packed = entry.down_nvfp4_buffers->packed.data();
      down_raw_scales = entry.down_nvfp4_buffers->block_scales.data();
    } else {
      return false;
    }

    const float down_tensor_scale = *entry.down_tensor_scale;

    return cudaMemcpy(
               impl.routed_up_packed_lookup_device.data() + expert_index,
               &up_packed,
               sizeof(up_packed),
               cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(
               impl.routed_up_raw_scale_lookup_device.data() + expert_index,
               &up_raw_scales,
               sizeof(up_raw_scales),
               cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(
               impl.routed_up_tensor_scale_lookup_device.data() + expert_index,
               &up_tensor_scale,
               sizeof(up_tensor_scale),
               cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(
               impl.routed_down_packed_lookup_device.data() + expert_index,
               &down_packed,
               sizeof(down_packed),
               cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(
               impl.routed_down_raw_scale_lookup_device.data() + expert_index,
               &down_raw_scales,
               sizeof(down_raw_scales),
               cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(
               impl.routed_down_tensor_scale_lookup_device.data() + expert_index,
               &down_tensor_scale,
               sizeof(down_tensor_scale),
               cudaMemcpyHostToDevice) == cudaSuccess;
  };
  const auto ensure_routed_expert_materialized =
      [&](std::size_t expert_index) -> const RoutedExpertEntry* {
    if (expert_index >= impl.routed_experts.size()) {
      return nullptr;
    }
    const auto& entry = impl.routed_experts[expert_index];
    if (entry.up_proj != nullptr && entry.down_proj != nullptr) {
      if (!entry.grouped_lookup_ready &&
          impl.grouped_routed_nvfp4_enabled.load(std::memory_order_relaxed) &&
          update_routed_nvfp4_lookup(expert_index, entry)) {
        entry.grouped_lookup_ready = true;
      }
      return &entry;
    }
    std::lock_guard<std::mutex> lock(entry.materialize_mutex);
    if (entry.up_proj == nullptr) {
      if (entry.up_descriptor.kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled &&
          entry.up_nvfp4_buffers == nullptr) {
        entry.up_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      entry.up_proj = MaterializeRoutedExpertOp(
          entry.up_descriptor,
          entry.up_nvfp4_buffers.get(),
          &entry.up_nvfp4_buffers_ready);
      if (entry.up_proj != nullptr && entry.up_proj->valid()) {
        RecordRoutedExpertMaterialization();
      }
    }
    if (entry.down_proj == nullptr) {
      if (entry.down_descriptor.kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled &&
          entry.down_nvfp4_buffers == nullptr) {
        entry.down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      }
      entry.down_proj = MaterializeRoutedExpertOp(
          entry.down_descriptor,
          entry.down_nvfp4_buffers.get(),
          &entry.down_nvfp4_buffers_ready);
      if (entry.down_proj != nullptr && entry.down_proj->valid()) {
        RecordRoutedExpertMaterialization();
      }
    }
    if (entry.up_proj == nullptr || entry.down_proj == nullptr ||
        !entry.up_proj->valid() || !entry.down_proj->valid()) {
      return nullptr;
    }
    if (!entry.grouped_lookup_ready &&
        impl.grouped_routed_nvfp4_enabled.load(std::memory_order_relaxed) &&
        update_routed_nvfp4_lookup(expert_index, entry)) {
      entry.grouped_lookup_ready = true;
    }
    return &entry;
  };

  const auto ensure_routed_nvfp4_lookup_ready =
      [&](std::size_t expert_index) -> bool {
    if (expert_index >= impl.routed_experts.size()) {
      return false;
    }
    const auto& entry = impl.routed_experts[expert_index];
    if (entry.grouped_lookup_ready) {
      return true;
    }
    if (entry.up_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
        entry.down_descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
        !entry.up_tensor_scale.has_value() ||
        !entry.down_tensor_scale.has_value()) {
      return false;
    }
    std::lock_guard<std::mutex> lock(entry.materialize_mutex);
    if (entry.grouped_lookup_ready) {
      return true;
    }
    if (entry.up_nvfp4_buffers == nullptr) {
      entry.up_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
    }
    if (entry.down_nvfp4_buffers == nullptr) {
      entry.down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
    }
    if (!EnsureNvfp4LookupBuffers(
            entry.up_descriptor,
            entry.up_nvfp4_buffers.get(),
            &entry.up_nvfp4_lookup_ready) ||
        !EnsureNvfp4LookupBuffers(
            entry.down_descriptor,
            entry.down_nvfp4_buffers.get(),
            &entry.down_nvfp4_lookup_ready)) {
      return false;
    }
    if (!update_routed_nvfp4_lookup(expert_index, entry)) {
      return false;
    }
    entry.grouped_lookup_ready = true;
    return true;
  };

  const auto repair_missing_routed_lookups =
      [&](DeviceBuffer<std::uint32_t>& missing_lookup_count_device,
          DeviceBuffer<std::int32_t>& missing_lookup_indices_device,
          std::uint32_t* missing_lookup_count_out) -> bool {
    if (missing_lookup_count_out == nullptr ||
        !missing_lookup_count_device.CopyToHost(missing_lookup_count_out, 1)) {
      return false;
    }
    if (*missing_lookup_count_out == 0) {
      return true;
    }
    std::vector<std::int32_t> missing_indices(*missing_lookup_count_out, -1);
    if (!missing_lookup_indices_device.CopyToHost(
            missing_indices.data(),
            missing_indices.size())) {
      return false;
    }
    std::sort(missing_indices.begin(), missing_indices.end());
    missing_indices.erase(
        std::unique(missing_indices.begin(), missing_indices.end()),
        missing_indices.end());
    missing_indices.erase(
        std::remove_if(
            missing_indices.begin(),
            missing_indices.end(),
            [&](std::int32_t expert_index) {
              return expert_index < 0 ||
                     static_cast<std::size_t>(expert_index) >= impl.routed_experts.size();
            }),
        missing_indices.end());
    const std::size_t repaired_missing_count = missing_indices.size();
    const bool expand_to_touched_groups =
        std::getenv("NEMOTRON_ROUTED_LOOKUP_REPAIR_GROUPS") != nullptr;
    if (expand_to_touched_groups &&
        !missing_indices.empty() &&
        impl.config.n_group != 0 &&
        impl.config.n_routed_experts % impl.config.n_group == 0) {
      const std::size_t group_size = impl.config.n_routed_experts / impl.config.n_group;
      std::vector<bool> touched_groups(impl.config.n_group, false);
      for (const std::int32_t expert_index : missing_indices) {
        touched_groups[static_cast<std::size_t>(expert_index) / group_size] = true;
      }
      std::vector<std::int32_t> expanded_indices = missing_indices;
      expanded_indices.reserve(missing_indices.size() + (impl.config.topk_group * group_size));
      for (std::size_t group = 0; group < touched_groups.size(); ++group) {
        if (!touched_groups[group]) {
          continue;
        }
        const std::size_t group_begin = group * group_size;
        const std::size_t group_end = group_begin + group_size;
        for (std::size_t expert_index = group_begin; expert_index < group_end; ++expert_index) {
          expanded_indices.push_back(static_cast<std::int32_t>(expert_index));
        }
      }
      std::sort(expanded_indices.begin(), expanded_indices.end());
      expanded_indices.erase(
          std::unique(expanded_indices.begin(), expanded_indices.end()),
          expanded_indices.end());
      if (expanded_indices.size() > repaired_missing_count) {
        RecordRoutedExpertPrefetchLayer();
        RecordRoutedExpertPrefetchExperts(expanded_indices.size() - repaired_missing_count);
      }
      missing_indices = std::move(expanded_indices);
    }
    RecordRoutedLookupRepairDownload();
    RecordRoutedLookupRepairExperts(repaired_missing_count);
    for (const std::int32_t expert_index : missing_indices) {
      if (!ensure_routed_nvfp4_lookup_ready(static_cast<std::size_t>(expert_index))) {
        return false;
      }
    }
    return true;
  };

  const auto maybe_prefetch_routed_lookups =
      [&](std::size_t token_index) -> bool {
    if (routed_prefetch_topn <= impl.config.top_k ||
        routed_prefetch_topn == 0 ||
        impl.routed_prefetch_done.load(std::memory_order_relaxed)) {
      return true;
    }
    std::lock_guard<std::mutex> lock(impl.routed_prefetch_mutex);
    if (impl.routed_prefetch_done.load(std::memory_order_relaxed)) {
      return true;
    }
    std::vector<float> router_host(impl.config.n_routed_experts, 0.0f);
    std::vector<float> bias_host(impl.config.n_routed_experts, 0.0f);
    if (token_index >= router_logits->shape()[0] ||
        !router_logits->CopyToHost(router_host.data(), router_host.size()) ||
        !impl.gate_score_correction_bias_device->CopyToHost(
            bias_host.data(),
            bias_host.size())) {
      return false;
    }
    std::vector<std::size_t> ranked(impl.config.n_routed_experts);
    std::iota(ranked.begin(), ranked.end(), std::size_t{0});
    const auto topn = std::min(routed_prefetch_topn, ranked.size());
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + topn,
        ranked.end(),
        [&](std::size_t lhs, std::size_t rhs) {
          return (router_host[lhs] + bias_host[lhs]) > (router_host[rhs] + bias_host[rhs]);
        });
    std::size_t prefetched = 0;
    for (std::size_t i = 0; i < topn; ++i) {
      if (ensure_routed_nvfp4_lookup_ready(ranked[i])) {
        ++prefetched;
      }
    }
    impl.routed_prefetch_done.store(true, std::memory_order_relaxed);
    RecordRoutedExpertPrefetchLayer();
    RecordRoutedExpertPrefetchExperts(prefetched);
    return prefetched == topn;
  };

  const auto retry_with_dense_fallback =
      [&](const RoutedExpertEntry& entry,
          const GemmDescriptor& descriptor,
          std::unique_ptr<UploadedLinearOp>& op_slot,
          const DeviceTensorFp32& activation,
          DeviceTensorFp32* output_tensor) -> bool {
    if (op_slot == nullptr ||
        op_slot->kernel_family() != GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
      return false;
    }
    std::lock_guard<std::mutex> lock(entry.materialize_mutex);
    if (op_slot == nullptr ||
        op_slot->kernel_family() != GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
      return false;
    }
    op_slot = MaterializeRoutedExpertDenseFallbackOp(descriptor);
    if (op_slot == nullptr || !op_slot->valid()) {
      return false;
    }
    return op_slot->Run(cublas_handle, heuristic_cache, activation, output_tensor);
  };

  enum class GroupedRoutedResult {
    kUsed,
    kFallback,
    kFatal,
  };

  const auto try_grouped_routed_single_token =
      [&](std::size_t token_index,
          const DeviceTensorFp32& latent_input_row,
          const std::int32_t* selected_indices_device,
          const float* selected_weights_device,
          std::size_t batch_count) -> GroupedRoutedResult {
    if (trace != nullptr ||
        token_count != 1 ||
        batch_count == 0 ||
        !impl.grouped_routed_nvfp4_enabled.load(std::memory_order_relaxed)) {
      if (debug && !impl.grouped_routed_nvfp4_enabled.load(std::memory_order_relaxed)) {
        std::cerr << "expert_layer: layer " << impl.config.layer_index
                  << " routed fastpath disabled before token execution\n";
      }
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertPrereqFallback();
      return GroupedRoutedResult::kFallback;
    }

    if (!GatherExpertSelectionLookupsCheckedInPlace(
            selected_indices_device,
            batch_count,
            impl.routed_experts.size(),
            impl.routed_up_packed_lookup_device,
            impl.routed_up_raw_scale_lookup_device,
            impl.routed_up_tensor_scale_lookup_device,
            impl.scratch_up_packed_ptrs,
            impl.scratch_up_raw_scale_ptrs,
            impl.scratch_selected_up_tensor_scales,
            impl.scratch_missing_count,
            impl.scratch_missing_indices)) {
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertLookupFallback();
      return GroupedRoutedResult::kFallback;
    }
    std::uint32_t missing_lookup_count = 0;
    if (!repair_missing_routed_lookups(
            impl.scratch_missing_count,
            impl.scratch_missing_indices,
            &missing_lookup_count)) {
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertLookupFallback();
      return GroupedRoutedResult::kFallback;
    }
    if (missing_lookup_count != 0) {
      if ((!maybe_prefetch_routed_lookups(token_index) && routed_prefetch_topn != 0) ||
          !GatherExpertSelectionLookupsCheckedInPlace(
              selected_indices_device,
              batch_count,
              impl.routed_experts.size(),
              impl.routed_up_packed_lookup_device,
              impl.routed_up_raw_scale_lookup_device,
              impl.routed_up_tensor_scale_lookup_device,
              impl.scratch_up_packed_ptrs,
              impl.scratch_up_raw_scale_ptrs,
              impl.scratch_selected_up_tensor_scales,
              impl.scratch_missing_count,
              impl.scratch_missing_indices) ||
          !repair_missing_routed_lookups(
              impl.scratch_missing_count,
              impl.scratch_missing_indices,
              &missing_lookup_count) ||
          missing_lookup_count != 0) {
        if (debug) {
          std::cerr << "expert_layer: layer " << impl.config.layer_index
                    << " routed fastpath missing " << missing_lookup_count
                    << " selected lookup entries after lazy materialization\n";
        }
        RecordGroupedRoutedExpertFastpathFallback();
        RecordGroupedRoutedExpertLookupFallback();
        return GroupedRoutedResult::kFallback;
      }
    }

    if (!PackDeviceRowMajorFp32ToNvfp4InPlace(
            latent_input_row, {},
            impl.scratch_latent_packed_data.data(),
            impl.scratch_latent_block_scales.data(),
            impl.scratch_latent_matmul_scales.data(),
            reinterpret_cast<float*>(impl.scratch_latent_tensor_scale.data()),
            impl.scratch_global_max_bits.data())) {
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertPackFallback();
      return GroupedRoutedResult::kFallback;
    }

    std::unique_ptr<DeviceTensorFp32> grouped_up;
    if (request_context != nullptr &&
        request_context->expert_intermediate_scratch() != nullptr &&
        request_context->expert_intermediate_scratch()->valid() &&
        request_context->expert_intermediate_scratch()->numel() >=
            batch_count * impl.config.routed_expert_intermediate_size) {
      grouped_up = DeviceTensorFp32::CreateView(
          {batch_count, impl.config.routed_expert_intermediate_size},
          request_context->expert_intermediate_scratch()->data());
    } else {
      grouped_up = DeviceTensorFp32::Create({
          batch_count,
          impl.config.routed_expert_intermediate_size,
      });
    }
    if (!grouped_up || !grouped_up->valid()) {
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertPackFallback();
      return GroupedRoutedResult::kFallback;
    }
    // Try CUTLASS grouped GEMM + post-scale for up_proj.
    // Fully device-resident: uses pre-allocated plan and device pointer arrays.
    // Zero host involvement, zero per-call allocation, zero D→H copies.
    bool cutlass_up_ok = false;
    if (impl.cutlass_up_plan != nullptr && impl.cutlass_up_plan->valid() &&
        batch_count == static_cast<std::size_t>(impl.cutlass_up_plan->group_count()) &&
        std::getenv("NEMOTRON_CUTLASS_MOE") != nullptr) {
      // A pointers: all groups share the same packed activation (device fill kernel).
      FillDevicePointerArray(
          const_cast<void**>(reinterpret_cast<const void* const*>(impl.cutlass_a_ptrs.data())),
          const_cast<void*>(static_cast<const void*>(impl.scratch_latent_packed_data.data())),
          batch_count);
      FillDevicePointerArray(
          const_cast<void**>(reinterpret_cast<const void* const*>(impl.cutlass_a_sf_ptrs.data())),
          const_cast<void*>(static_cast<const void*>(impl.scratch_latent_block_scales.data())),
          batch_count);
      // D pointers: each group writes to a strided output row (device strided-fill kernel).
      BuildStridedDevicePointerArray(
          impl.cutlass_d_ptrs.data(),
          grouped_up->data(),
          impl.config.routed_expert_intermediate_size * sizeof(float),
          batch_count);
      // C = D (beta=0, so C is unused, but CUTLASS needs valid pointers).
      BuildStridedDevicePointerArray(
          const_cast<void**>(reinterpret_cast<const void* const*>(impl.cutlass_c_ptrs.data())),
          grouped_up->data(),
          impl.config.routed_expert_intermediate_size * sizeof(float),
          batch_count);
      // B pointers: already on device from GatherExpertSelectionLookupsCheckedInPlace.
      cutlass_up_ok = impl.cutlass_up_plan->Run(
          impl.cutlass_a_ptrs.data(),
          impl.cutlass_a_sf_ptrs.data(),
          impl.scratch_up_packed_ptrs.data(),
          impl.scratch_up_raw_scale_ptrs.data(),
          impl.cutlass_c_ptrs.data(),
          impl.cutlass_d_ptrs.data(),
          1.0f, 0.0f, nullptr);
      if (cutlass_up_ok) {
        cutlass_up_ok = ScaleRowsByTensorScaleFp32(
            grouped_up.get(),
            reinterpret_cast<const float*>(impl.scratch_latent_tensor_scale.data()),
            impl.scratch_selected_up_tensor_scales.data(),
            batch_count);
      }
      if (cutlass_up_ok && debug) {
        std::cerr << "expert_layer: layer " << impl.config.layer_index
                  << " CUTLASS device up_proj succeeded\n";
      }
    }
    if (!cutlass_up_ok && !FusedRoutedUpProjPackedNvfp4SingleToken(
            impl.scratch_latent_packed_data.data(),
            impl.scratch_latent_block_scales.data(),
            reinterpret_cast<const float*>(impl.scratch_latent_tensor_scale.data()),
            impl.config.moe_latent_size,
            impl.scratch_up_packed_ptrs,
            impl.scratch_up_raw_scale_ptrs,
            impl.scratch_selected_up_tensor_scales,
            grouped_up.get())) {
      if (debug) {
        std::cerr << "expert_layer: layer " << impl.config.layer_index
                  << " routed fastpath up kernel launch failed\n";
      }
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertMatmulFallback();
      return GroupedRoutedResult::kFallback;
    }

    if (!ScaleRelu2PackRowsToNvfp4InPlace(
            *grouped_up,
            impl.scratch_row_scales.data(),
            impl.scratch_down_act_packed,
            impl.scratch_down_act_block_scales,
            impl.scratch_down_act_tensor_scales)) {
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertMergeFallback();
      return GroupedRoutedResult::kFallback;
    }

    if (!GatherExpertSelectionLookupsCheckedInPlace(
            selected_indices_device,
            batch_count,
            impl.routed_experts.size(),
            impl.routed_down_packed_lookup_device,
            impl.routed_down_raw_scale_lookup_device,
            impl.routed_down_tensor_scale_lookup_device,
            impl.scratch_down_packed_ptrs,
            impl.scratch_down_raw_scale_ptrs,
            impl.scratch_selected_down_tensor_scales,
            impl.scratch_missing_count,
            impl.scratch_missing_indices)) {
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertPackFallback();
      return GroupedRoutedResult::kFallback;
    }
    if (!repair_missing_routed_lookups(
            impl.scratch_missing_count,
            impl.scratch_missing_indices,
            &missing_lookup_count)) {
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertLookupFallback();
      return GroupedRoutedResult::kFallback;
    }
    if (missing_lookup_count != 0) {
      if ((!maybe_prefetch_routed_lookups(token_index) && routed_prefetch_topn != 0) ||
          !GatherExpertSelectionLookupsCheckedInPlace(
              selected_indices_device,
              batch_count,
              impl.routed_experts.size(),
              impl.routed_down_packed_lookup_device,
              impl.routed_down_raw_scale_lookup_device,
              impl.routed_down_tensor_scale_lookup_device,
              impl.scratch_down_packed_ptrs,
              impl.scratch_down_raw_scale_ptrs,
              impl.scratch_selected_down_tensor_scales,
              impl.scratch_missing_count,
              impl.scratch_missing_indices) ||
          !repair_missing_routed_lookups(
              impl.scratch_missing_count,
              impl.scratch_missing_indices,
              &missing_lookup_count) ||
          missing_lookup_count != 0) {
        if (debug) {
          std::cerr << "expert_layer: layer " << impl.config.layer_index
                    << " routed fastpath missing " << missing_lookup_count
                    << " down lookup entries after lazy materialization\n";
        }
        RecordGroupedRoutedExpertFastpathFallback();
        RecordGroupedRoutedExpertLookupFallback();
        return GroupedRoutedResult::kFallback;
      }
    }
    // CUTLASS down_proj: fully device-resident dispatch.
    bool cutlass_down_ok = false;
    if (impl.cutlass_down_plan != nullptr && impl.cutlass_down_plan->valid() &&
        batch_count == static_cast<std::size_t>(impl.cutlass_down_plan->group_count()) &&
        std::getenv("NEMOTRON_CUTLASS_MOE") != nullptr) {
      constexpr std::size_t kCutlassAlign = 256;
      const std::size_t act_packed_row_bytes = impl.config.routed_expert_intermediate_size / 2;
      const std::size_t act_scale_row_bytes = impl.config.routed_expert_intermediate_size / 16;
      const std::size_t aligned_packed_row = ((act_packed_row_bytes + kCutlassAlign - 1) / kCutlassAlign) * kCutlassAlign;
      const std::size_t aligned_scale_row = ((act_scale_row_bytes + kCutlassAlign - 1) / kCutlassAlign) * kCutlassAlign;

      auto down_output = DeviceTensorFp32::Create({batch_count, impl.config.moe_latent_size});

      if (impl.scratch_aligned_act_packed.valid() &&
          impl.scratch_aligned_act_packed.count() >= batch_count * aligned_packed_row &&
          impl.scratch_aligned_act_scales.valid() &&
          impl.scratch_aligned_act_scales.count() >= batch_count * aligned_scale_row &&
          down_output && down_output->valid() && down_output->FillZero()) {
        // Copy activation rows to aligned offsets (D→D, no host).
        cudaMemset(impl.scratch_aligned_act_packed.data(), 0, batch_count * aligned_packed_row);
        cudaMemset(impl.scratch_aligned_act_scales.data(), 0, batch_count * aligned_scale_row);
        for (std::size_t i = 0; i < batch_count; ++i) {
          cudaMemcpy(
              impl.scratch_aligned_act_packed.data() + i * aligned_packed_row,
              impl.scratch_down_act_packed.data() + i * act_packed_row_bytes,
              act_packed_row_bytes, cudaMemcpyDeviceToDevice);
          cudaMemcpy(
              impl.scratch_aligned_act_scales.data() + i * aligned_scale_row,
              impl.scratch_down_act_block_scales.data() + i * act_scale_row_bytes,
              act_scale_row_bytes, cudaMemcpyDeviceToDevice);
        }

        // Build all pointer arrays on device — zero host involvement.
        BuildStridedDevicePointerArray(
            const_cast<void**>(reinterpret_cast<const void* const*>(impl.cutlass_a_ptrs.data())),
            impl.scratch_aligned_act_packed.data(), aligned_packed_row, batch_count);
        BuildStridedDevicePointerArray(
            const_cast<void**>(reinterpret_cast<const void* const*>(impl.cutlass_a_sf_ptrs.data())),
            impl.scratch_aligned_act_scales.data(), aligned_scale_row, batch_count);
        BuildStridedDevicePointerArray(
            impl.cutlass_d_ptrs.data(),
            down_output->data(),
            impl.config.moe_latent_size * sizeof(float), batch_count);
        BuildStridedDevicePointerArray(
            const_cast<void**>(reinterpret_cast<const void* const*>(impl.cutlass_c_ptrs.data())),
            down_output->data(),
            impl.config.moe_latent_size * sizeof(float), batch_count);

        // B pointers already on device from GatherExpertSelectionLookupsCheckedInPlace.
        cutlass_down_ok = impl.cutlass_down_plan->Run(
            impl.cutlass_a_ptrs.data(),
            impl.cutlass_a_sf_ptrs.data(),
            impl.scratch_down_packed_ptrs.data(),
            impl.scratch_down_raw_scale_ptrs.data(),
            impl.cutlass_c_ptrs.data(),
            impl.cutlass_d_ptrs.data(),
            1.0f, 0.0f, nullptr);

        if (cutlass_down_ok) {
          cutlass_down_ok = ScaleWeightedAccumulateRowsFp32(
              *down_output,
              impl.scratch_down_act_tensor_scales.data(),
              impl.scratch_selected_down_tensor_scales.data(),
              selected_weights_device,
              batch_count,
              routed_tensor.get());
        }
      }
      if (cutlass_down_ok && debug) {
        std::cerr << "expert_layer: layer " << impl.config.layer_index
                  << " CUTLASS device down_proj succeeded\n";
      }
    }
    if (!cutlass_down_ok && !FusedRoutedDownProjWeightedPackedNvfp4SingleToken(
            impl.scratch_down_act_packed.data(),
            impl.scratch_down_act_block_scales.data(),
            impl.scratch_down_act_tensor_scales,
            selected_weights_device,
            impl.config.routed_expert_intermediate_size,
            impl.scratch_down_packed_ptrs,
            impl.scratch_down_raw_scale_ptrs,
            impl.scratch_selected_down_tensor_scales,
            routed_tensor.get())) {
      if (debug) {
        std::cerr << "expert_layer: layer " << impl.config.layer_index
                  << " routed fastpath down kernel launch failed\n";
      }
      RecordGroupedRoutedExpertFastpathFallback();
      RecordGroupedRoutedExpertMatmulFallback();
      return GroupedRoutedResult::kFallback;
    }

    RecordGroupedRoutedExpertFastpathUse();
    return GroupedRoutedResult::kUsed;
  };

  const auto try_routed_fastpath_single_token =
      [&](std::size_t token_index,
          const DeviceTensorFp32& latent_input_row,
          const std::int32_t* selected_indices_device,
          const float* selected_weights_device,
          std::size_t batch_count) -> GroupedRoutedResult {
    if (impl.routed_backend_kind == RoutedMoEBackendKind::kFlashInfer) {
      if (trace != nullptr ||
          token_count != 1 ||
          batch_count == 0 ||
          impl.flashinfer_routed_backend == nullptr ||
          !impl.flashinfer_routed_backend->valid()) {
        RecordFlashInferRoutedExpertFallback();
        return GroupedRoutedResult::kFallback;
      }
      if (impl.flashinfer_routed_backend->Run(
              latent_input_row,
              1,
              selected_indices_device,
              selected_weights_device,
              routed_tensor.get())) {
        RecordFlashInferRoutedExpertUse();
        return GroupedRoutedResult::kUsed;
      }
      if (debug) {
        std::cerr << "expert_layer: layer " << impl.config.layer_index
                  << " flashinfer routed backend failed, falling back to custom fused path\n";
      }
      RecordFlashInferRoutedExpertFallback();
    }
    return try_grouped_routed_single_token(
        token_index,
        latent_input_row,
        selected_indices_device,
        selected_weights_device,
        batch_count);
  };

  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float* normalized_row =
        trace != nullptr ? (normalized_host.data() + (token_index * impl.config.hidden_size)) : nullptr;
    const float* router_row =
        trace != nullptr ? (router_logits_host.data() + (token_index * impl.config.n_routed_experts)) : nullptr;
    const float* latent_row_host =
        trace != nullptr ? (latent_host.data() + (token_index * impl.config.moe_latent_size)) : nullptr;

    const DeviceTensorFp32* latent_input = latent.get();
    if (token_count != 1) {
      if (!CopyRowFp32(*latent, token_index, latent_row.get())) {
        if (debug) {
          std::cout << "expert_layer: latent row copy failed\n";
        }
        return false;
      }
      latent_input = latent_row.get();
    }

    const GroupedRoutedResult grouped_result =
        try_routed_fastpath_single_token(
            token_index,
            *latent_input,
            selection_scratch.indices_device->data() + (token_index * impl.config.top_k),
            selection_scratch.weights_device->data() + (token_index * impl.config.top_k),
            impl.config.top_k);
    if (grouped_result == GroupedRoutedResult::kUsed) {
      continue;
    }
    if (grouped_result == GroupedRoutedResult::kFatal) {
      if (debug) {
        std::cout << "expert_layer: grouped routed expert execution failed fatally\n";
      }
      return false;
    }

    if (!ensure_selection_metadata_host()) {
      if (debug) {
        std::cout << "expert_layer: selection metadata download failed\n";
      }
      return false;
    }

    std::vector<ExpertSelection> selections;
    selections.reserve(impl.config.top_k);
    for (std::size_t selection_index = 0; selection_index < impl.config.top_k; ++selection_index) {
      const std::size_t metadata_index = token_index * impl.config.top_k + selection_index;
      const std::int32_t expert_index = selected_indices_host[metadata_index];
      if (expert_index < 0) {
        if (debug) {
          std::cout << "expert_layer: invalid selected expert index\n";
        }
        return false;
      }
      selections.push_back(
          ExpertSelection{static_cast<std::size_t>(expert_index), selected_weights_host[metadata_index]});
    }
    if (selections.size() != impl.config.top_k) {
      if (debug) {
        std::cout << "expert_layer: top-k selection size mismatch\n";
      }
      return false;
    }

    for (const ExpertSelection& selection : selections) {
      if (selection.expert_index >= impl.routed_experts.size()) {
        if (debug) {
          std::cout << "expert_layer: selected expert index out of range "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      const auto* pair = ensure_routed_expert_materialized(selection.expert_index);
      if (pair == nullptr) {
        if (debug) {
          std::cout << "expert_layer: missing routed expert weights for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }

      bool up_ok = pair->up_proj->Run(cublas_handle, heuristic_cache, *latent_input, expert_up.get());
      if (!up_ok) {
        up_ok = retry_with_dense_fallback(
            *pair,
            pair->up_descriptor,
            pair->up_proj,
            *latent_input,
            expert_up.get());
      }
      if (!up_ok) {
        if (debug) {
          std::cout << "expert_layer: routed up_proj failed for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      if (!Relu2InPlaceFp32(expert_up.get())) {
        if (debug) {
          std::cout << "expert_layer: routed relu2 failed for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      bool down_ok = pair->down_proj->Run(cublas_handle, heuristic_cache, *expert_up, expert_down.get());
      if (!down_ok) {
        down_ok = retry_with_dense_fallback(
            *pair,
            pair->down_descriptor,
            pair->down_proj,
            *expert_up,
            expert_down.get());
      }
      if (!down_ok) {
        if (debug) {
          std::cout << "expert_layer: routed down_proj failed for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      const bool add_ok = token_count == 1
          ? AddScaledFp32(*expert_down, selection.weight, routed_tensor.get())
          : AddScaledRowFp32(*expert_down, selection.weight, token_index, routed_tensor.get());
      if (!add_ok) {
        if (debug) {
          std::cout << "expert_layer: routed weighted accumulation failed for expert "
                    << selection.expert_index << "\n";
        }
        return false;
      }
      if (trace != nullptr && token_index == 0) {
        std::vector<float> activated_up_host(impl.config.routed_expert_intermediate_size, 0.0f);
        std::vector<float> expert_down_host(impl.config.moe_latent_size, 0.0f);
        if (!expert_up->CopyToHost(activated_up_host.data(), activated_up_host.size()) ||
            !expert_down->CopyToHost(expert_down_host.data(), expert_down_host.size())) {
          if (debug) {
            std::cout << "expert_layer: routed trace download failed\n";
          }
          return false;
        }
        trace_routed_expert_order.push_back(selection.expert_index);
        trace_routed_activated_hidden.insert(
            trace_routed_activated_hidden.end(),
            activated_up_host.begin(),
            activated_up_host.end());
        trace_routed_expert_outputs.insert(
            trace_routed_expert_outputs.end(),
            expert_down_host.begin(),
            expert_down_host.end());
        trace_routed_weighted_contributions.insert(
            trace_routed_weighted_contributions.end(),
            expert_down_host.size(),
            0.0f);
        float* contribution =
            trace_routed_weighted_contributions.data() +
            (trace_routed_expert_order.size() - 1) * impl.config.moe_latent_size;
        for (std::size_t i = 0; i < impl.config.moe_latent_size; ++i) {
          contribution[i] = expert_down_host[i] * selection.weight;
        }
      }
    }

    if (trace != nullptr) {
      trace->normalized_input.assign(
          normalized_row,
          normalized_row + impl.config.hidden_size);
      trace_router_logits.assign(
          router_row,
          router_row + impl.config.n_routed_experts);
      trace_selections = selections;
      trace_latent_output.assign(
          latent_row_host,
          latent_row_host + impl.config.moe_latent_size);
      trace->routed_expert_order = trace_routed_expert_order;
      trace->routed_expert_activated_hidden = trace_routed_activated_hidden;
      trace->routed_expert_outputs = trace_routed_expert_outputs;
      trace->routed_expert_weighted_contributions = trace_routed_weighted_contributions;
    }
  }

  if (!Relu2InPlaceFp32(shared_up.get())) {
    if (debug) {
      std::cout << "expert_layer: shared relu2 failed\n";
    }
    return false;
  }

  bool shared_down_ok = false;
  if (impl.shared_down_nvfp4 != nullptr) {
    shared_down_ok =
        impl.shared_down_nvfp4->Run(cublas_handle, heuristic_cache, *shared_up, shared_output_device.get());
  } else if (impl.shared_down_scaled_fp8 != nullptr) {
    shared_down_ok =
        impl.shared_down_scaled_fp8->Run(cublas_handle, heuristic_cache, *shared_up, shared_output_device.get());
  } else if (impl.shared_down_dense != nullptr) {
    shared_down_ok =
        impl.shared_down_dense->Run(cublas_handle, heuristic_cache, *shared_up, shared_output_device.get());
  }
  if (!shared_down_ok) {
    if (debug) {
      std::cout << "expert_layer: shared down projection failed\n";
    }
    return false;
  }

  if (!impl.fc2_latent->Run(cublas_handle, heuristic_cache, *routed_tensor, projected_routed.get())) {
    if (debug) {
      std::cout << "expert_layer: fc2 latent projection failed\n";
    }
    return false;
  }
  if (!ResidualAddFp32(*projected_routed, *shared_output_device, mixer_output_device.get()) ||
      !ResidualAddFp32(input, *mixer_output_device, output)) {
    if (debug) {
      std::cout << "expert_layer: residual merge failed\n";
    }
    return false;
  }

  if (trace != nullptr) {
    if (!CopyToHost(*routed_tensor, &trace_routed_output) ||
        !CopyToHost(*projected_routed, &trace->projected_routed_output) ||
        !CopyToHost(*shared_output_device, &trace_shared_output) ||
        !CopyToHost(*mixer_output_device, &trace->mixer_output)) {
      if (debug) {
        std::cout << "expert_layer: trace output download failed\n";
      }
      return false;
    }
    trace->router_logits = std::move(trace_router_logits);
    trace->selected_experts = std::move(trace_selections);
    trace->latent_output = std::move(trace_latent_output);
    trace->routed_latent_output = std::move(trace_routed_output);
    trace->shared_output = std::move(trace_shared_output);
  }
  return true;
}

}  // namespace

bool ExpertLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    ExpertLayerRunTrace* trace) const {
  return valid() &&
         RunExpertLayerImpl(
             *impl_,
             cublas_handle,
             heuristic_cache,
             nullptr,
             input,
             output,
             trace);
}

bool ExpertLayerSlice::RunWithRequestContext(
    CublasLtHandle& cublas_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output,
    ExpertLayerRunTrace* trace) const {
  return valid() &&
         RunExpertLayerImpl(
             *impl_,
             cublas_handle,
             heuristic_cache,
             &request_context,
             input,
             output,
             trace);
}

}  // namespace nemotron
