#include "nemotron/device_tensor.h"
#include "nemotron/device_buffer.h"
#include "nemotron/expert_layer.h"
#include "nemotron/model_schedule.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/nvfp4_weight.h"
#include "nemotron/request_context.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/runtime_stats.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct Options {
  enum class InputDtype {
    kFp32,
    kBf16,
  };

  std::filesystem::path manifest_path;
  std::filesystem::path input_hidden_path;
  std::filesystem::path shared_down_input_path;
  std::filesystem::path routed_up_input_path;
  std::filesystem::path expected_output_path;
  std::filesystem::path dump_root;
  std::size_t layer_index = 1;
  std::size_t routed_expert_index = 0;
  bool dump_row_traces = false;
  InputDtype input_dtype = InputDtype::kFp32;
  std::optional<float> nvfp4_fixed_input_scale;
};

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::vector<std::uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void WriteFloatFile(const std::filesystem::path& path, const std::vector<float>& values) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

void WriteInt32File(const std::filesystem::path& path, const std::vector<std::int32_t>& values) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
}

void WriteJsonFile(const std::filesystem::path& path, const std::string& json) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  output << json;
}

std::optional<std::vector<float>> LoadFloatMatrix(
    const std::filesystem::path& path,
    std::size_t hidden_size,
    std::size_t* row_count_out) {
  if (row_count_out == nullptr || hidden_size == 0) {
    return std::nullopt;
  }
  const std::vector<std::uint8_t> bytes = ReadFileBytes(path);
  if (bytes.size() % sizeof(float) != 0) {
    return std::nullopt;
  }
  const std::size_t value_count = bytes.size() / sizeof(float);
  if (value_count == 0 || value_count % hidden_size != 0) {
    return std::nullopt;
  }
  std::vector<float> values(value_count, 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  *row_count_out = value_count / hidden_size;
  return values;
}

std::vector<__nv_bfloat16> ConvertHostFp32ToBf16(const std::vector<float>& input) {
  std::vector<__nv_bfloat16> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = __float2bfloat16(input[i]);
  }
  return output;
}

std::vector<float> ConvertHostBf16ToFp32(const std::vector<__nv_bfloat16>& input) {
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = __bfloat162float(input[i]);
  }
  return output;
}

const char* InputDtypeName(Options::InputDtype dtype) {
  switch (dtype) {
    case Options::InputDtype::kFp32:
      return "fp32";
    case Options::InputDtype::kBf16:
      return "bf16";
  }
  return "unknown";
}

bool DirectSharedDownMode(const Options& options) {
  return !options.shared_down_input_path.empty();
}

bool DirectRoutedUpMode(const Options& options) {
  return !options.routed_up_input_path.empty();
}

bool ParseArgs(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--manifest" && i + 1 < argc) {
      options->manifest_path = argv[++i];
      continue;
    }
    if (arg == "--input-hidden-bin" && i + 1 < argc) {
      options->input_hidden_path = argv[++i];
      continue;
    }
    if (arg == "--shared-down-input-bin" && i + 1 < argc) {
      options->shared_down_input_path = argv[++i];
      continue;
    }
    if (arg == "--routed-up-input-bin" && i + 1 < argc) {
      options->routed_up_input_path = argv[++i];
      continue;
    }
    if (arg == "--expected-output-bin" && i + 1 < argc) {
      options->expected_output_path = argv[++i];
      continue;
    }
    if (arg == "--dump-root" && i + 1 < argc) {
      options->dump_root = argv[++i];
      continue;
    }
    if (arg == "--layer-index" && i + 1 < argc) {
      options->layer_index = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--routed-expert-index" && i + 1 < argc) {
      options->routed_expert_index = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--dump-row-traces") {
      options->dump_row_traces = true;
      continue;
    }
    if (arg == "--input-dtype" && i + 1 < argc) {
      const std::string value(argv[++i]);
      if (value == "fp32") {
        options->input_dtype = Options::InputDtype::kFp32;
      } else if (value == "bf16") {
        options->input_dtype = Options::InputDtype::kBf16;
      } else {
        std::cerr << "invalid --input-dtype: " << value << "\n";
        return false;
      }
      continue;
    }
    if (arg == "--nvfp4-fixed-input-scale" && i + 1 < argc) {
      options->nvfp4_fixed_input_scale = std::stof(argv[++i]);
      continue;
    }
    std::cerr << "unknown argument: " << arg << "\n";
    return false;
  }
  const bool has_standard_input = !options->input_hidden_path.empty();
  const bool has_direct_input = !options->shared_down_input_path.empty();
  const bool has_direct_routed_up = !options->routed_up_input_path.empty();
  const int mode_count = static_cast<int>(has_standard_input) +
                         static_cast<int>(has_direct_input) +
                         static_cast<int>(has_direct_routed_up);
  if (mode_count != 1) {
    std::cerr
        << "provide exactly one of --input-hidden-bin, --shared-down-input-bin, or --routed-up-input-bin\n";
    return false;
  }
  if (has_direct_routed_up && options->dump_row_traces) {
    std::cerr << "--dump-row-traces is not supported with --routed-up-input-bin\n";
    return false;
  }
  return !options->manifest_path.empty() && !options->dump_root.empty();
}

nemotron::RuntimeBootstrapOptions MakeOptions(std::size_t token_count) {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = token_count;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

std::string SelectionsJson(const std::vector<nemotron::ExpertSelection>& selections) {
  std::string json = "[";
  for (std::size_t i = 0; i < selections.size(); ++i) {
    if (i != 0) {
      json += ",";
    }
    json += "{\"expert_index\":" + std::to_string(selections[i].expert_index) +
            ",\"weight\":" + std::to_string(selections[i].weight) + "}";
  }
  json += "]";
  return json;
}

std::string SummaryJson(
    std::size_t layer_index,
    std::size_t token_count,
    std::size_t hidden_size,
    const char* input_dtype,
    const nemotron::ExpertLayerRunTrace& trace,
    const nemotron::RuntimeExecutionStatsSnapshot& runtime_stats) {
  return "{\n"
         "  \"layer_index\": " + std::to_string(layer_index) + ",\n" +
         "  \"token_count\": " + std::to_string(token_count) + ",\n" +
         "  \"hidden_size\": " + std::to_string(hidden_size) + ",\n" +
         "  \"input_dtype\": \"" + std::string(input_dtype) + "\",\n" +
         "  \"selected_experts\": " + SelectionsJson(trace.selected_experts) + ",\n" +
         "  \"runtime_stats\": {\n"
         "    \"scaled_fp8_plan_build_failures\": " +
         std::to_string(runtime_stats.scaled_fp8_plan_build_failures) + ",\n" +
         "    \"scaled_fp8_plan_build_failures_expert_fc1_latent\": " +
         std::to_string(runtime_stats.scaled_fp8_plan_build_failures_expert_fc1_latent) + ",\n" +
         "    \"scaled_fp8_plan_build_failures_expert_shared_up\": " +
         std::to_string(runtime_stats.scaled_fp8_plan_build_failures_expert_shared_up) + ",\n" +
         "    \"scaled_fp8_native_success\": " +
         std::to_string(runtime_stats.scaled_fp8_native_success) + ",\n" +
         "    \"scaled_fp8_native_success_expert_fc1_latent\": " +
         std::to_string(runtime_stats.scaled_fp8_native_success_expert_fc1_latent) + ",\n" +
         "    \"scaled_fp8_native_success_expert_shared_up\": " +
         std::to_string(runtime_stats.scaled_fp8_native_success_expert_shared_up) + ",\n" +
         "    \"scaled_fp8_dequantized_dense_success\": " +
         std::to_string(runtime_stats.scaled_fp8_dequantized_dense_success) + ",\n" +
         "    \"scaled_fp8_reference_fallbacks\": " +
         std::to_string(runtime_stats.scaled_fp8_reference_fallbacks) + ",\n" +
         "    \"scaled_fp8_reference_fallbacks_expert_fc1_latent\": " +
         std::to_string(runtime_stats.scaled_fp8_reference_fallbacks_expert_fc1_latent) + ",\n" +
         "    \"scaled_fp8_reference_fallbacks_expert_shared_up\": " +
         std::to_string(runtime_stats.scaled_fp8_reference_fallbacks_expert_shared_up) + ",\n" +
         "    \"routed_expert_materializations\": " +
         std::to_string(runtime_stats.routed_expert_materializations) + ",\n" +
         "    \"grouped_routed_expert_fastpath_uses\": " +
         std::to_string(runtime_stats.grouped_routed_expert_fastpath_uses) + ",\n" +
         "    \"grouped_routed_expert_fastpath_fallbacks\": " +
         std::to_string(runtime_stats.grouped_routed_expert_fastpath_fallbacks) + "\n"
         "  }\n"
         "}\n";
}

std::string SummaryJson(
    std::size_t layer_index,
    std::size_t token_count,
    std::size_t hidden_size,
    const char* input_dtype,
    bool dump_row_traces,
    std::optional<std::size_t> top_k,
    const nemotron::RuntimeExecutionStatsSnapshot& runtime_stats) {
  return "{\n"
         "  \"layer_index\": " + std::to_string(layer_index) + ",\n" +
         "  \"token_count\": " + std::to_string(token_count) + ",\n" +
         "  \"hidden_size\": " + std::to_string(hidden_size) + ",\n" +
         "  \"input_dtype\": \"" + std::string(input_dtype) + "\",\n" +
         "  \"dump_row_traces\": " + std::string(dump_row_traces ? "true" : "false") +
         (top_k.has_value() ? ",\n  \"top_k\": " + std::to_string(*top_k) : "") + ",\n" +
         "  \"runtime_stats\": {\n"
         "    \"scaled_fp8_plan_build_failures\": " +
         std::to_string(runtime_stats.scaled_fp8_plan_build_failures) + ",\n" +
         "    \"scaled_fp8_plan_build_failures_expert_fc1_latent\": " +
         std::to_string(runtime_stats.scaled_fp8_plan_build_failures_expert_fc1_latent) + ",\n" +
         "    \"scaled_fp8_plan_build_failures_expert_shared_up\": " +
         std::to_string(runtime_stats.scaled_fp8_plan_build_failures_expert_shared_up) + ",\n" +
         "    \"scaled_fp8_native_success\": " +
         std::to_string(runtime_stats.scaled_fp8_native_success) + ",\n" +
         "    \"scaled_fp8_native_success_expert_fc1_latent\": " +
         std::to_string(runtime_stats.scaled_fp8_native_success_expert_fc1_latent) + ",\n" +
         "    \"scaled_fp8_native_success_expert_shared_up\": " +
         std::to_string(runtime_stats.scaled_fp8_native_success_expert_shared_up) + ",\n" +
         "    \"scaled_fp8_dequantized_dense_success\": " +
         std::to_string(runtime_stats.scaled_fp8_dequantized_dense_success) + ",\n" +
         "    \"scaled_fp8_reference_fallbacks\": " +
         std::to_string(runtime_stats.scaled_fp8_reference_fallbacks) + ",\n" +
         "    \"scaled_fp8_reference_fallbacks_expert_fc1_latent\": " +
         std::to_string(runtime_stats.scaled_fp8_reference_fallbacks_expert_fc1_latent) + ",\n" +
         "    \"scaled_fp8_reference_fallbacks_expert_shared_up\": " +
         std::to_string(runtime_stats.scaled_fp8_reference_fallbacks_expert_shared_up) + ",\n" +
         "    \"routed_expert_materializations\": " +
         std::to_string(runtime_stats.routed_expert_materializations) + ",\n" +
         "    \"grouped_routed_expert_fastpath_uses\": " +
         std::to_string(runtime_stats.grouped_routed_expert_fastpath_uses) + ",\n" +
         "    \"grouped_routed_expert_fastpath_fallbacks\": " +
         std::to_string(runtime_stats.grouped_routed_expert_fastpath_fallbacks) + "\n"
         "  }\n"
         "}\n";
}

struct DiffStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rel_l2 = 0.0;
};

std::optional<DiffStats> ComputeDiffStats(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  if (actual.size() != expected.size() || actual.empty()) {
    return std::nullopt;
  }
  double max_abs = 0.0;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
    const double abs_diff = std::abs(diff);
    max_abs = std::max(max_abs, abs_diff);
    sum_abs += abs_diff;
    sum_sq += diff * diff;
    ref_sq += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
  }
  DiffStats stats;
  stats.max_abs = max_abs;
  stats.mean_abs = sum_abs / static_cast<double>(actual.size());
  stats.rel_l2 = ref_sq > 0.0 ? std::sqrt(sum_sq / ref_sq) : 0.0;
  return stats;
}

struct Nvfp4AlignedBuffers {
  nemotron::DeviceBuffer<std::uint8_t> packed;
  nemotron::DeviceBuffer<std::uint8_t> raw_block_scales;
  nemotron::DeviceBuffer<std::uint8_t> matmul_block_scales;
  nemotron::DeviceBuffer<std::uint8_t> tensor_scale;
};

std::optional<float> ReadTensorScaleHost(const nemotron::GemmDescriptor& descriptor) {
  if (descriptor.tensor_scale_data == nullptr || descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }
  float value = 0.0f;
  std::memcpy(&value, descriptor.tensor_scale_data, sizeof(float));
  return value;
}

std::unique_ptr<nemotron::UploadedLinearOp> CreateAlignedNvfp4LinearOp(
    const nemotron::GemmDescriptor& descriptor,
    Nvfp4AlignedBuffers* buffers) {
  if (buffers == nullptr ||
      descriptor.kernel_family != nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr) {
    return nullptr;
  }

  const auto tensor_scale = ReadTensorScaleHost(descriptor);
  if (!tensor_scale.has_value() || !std::isfinite(*tensor_scale) || *tensor_scale <= 0.0f) {
    return nullptr;
  }
  const auto swizzled_scales = nemotron::SwizzleRowMajorNvfp4ScalesForExecution(
      descriptor.block_scales_data,
      descriptor.output_rows,
      descriptor.input_cols);
  if (swizzled_scales.empty()) {
    return nullptr;
  }

  if (!buffers->packed.Resize(descriptor.packed_nbytes) ||
      !buffers->raw_block_scales.Resize(descriptor.block_scales_nbytes) ||
      !buffers->matmul_block_scales.Resize(swizzled_scales.size()) ||
      !buffers->tensor_scale.Resize(sizeof(float)) ||
      !buffers->packed.CopyFromHost(descriptor.packed_data, descriptor.packed_nbytes) ||
      !buffers->raw_block_scales.CopyFromHost(
          descriptor.block_scales_data,
          descriptor.block_scales_nbytes) ||
      !buffers->matmul_block_scales.CopyFromHost(swizzled_scales) ||
      !buffers->tensor_scale.CopyFromHost(
          reinterpret_cast<const std::uint8_t*>(&(*tensor_scale)),
          sizeof(float))) {
    return nullptr;
  }

  auto weight_view = nemotron::DeviceNvfp4Weight::CreateView(
      descriptor.output_rows,
      descriptor.input_cols,
      buffers->packed.data(),
      descriptor.packed_nbytes,
      buffers->raw_block_scales.data(),
      descriptor.block_scales_nbytes,
      buffers->matmul_block_scales.data(),
      buffers->matmul_block_scales.count(),
      buffers->tensor_scale.data(),
      buffers->tensor_scale.count());
  if (!weight_view || !weight_view->valid()) {
    return nullptr;
  }

  nemotron::GemmDescriptor aligned_descriptor = descriptor;
  aligned_descriptor.packed_data = buffers->packed.data();
  aligned_descriptor.block_scales_data = buffers->matmul_block_scales.data();
  aligned_descriptor.block_scales_nbytes = buffers->matmul_block_scales.count();
  aligned_descriptor.tensor_scale_data = buffers->tensor_scale.data();
  aligned_descriptor.tensor_scale_nbytes = buffers->tensor_scale.count();
  return nemotron::UploadedLinearOp::CreateNvfp4View(aligned_descriptor, std::move(weight_view));
}

std::optional<float> ReadScalarTensorToHostFp32(const nemotron::KernelTensorDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr ||
      descriptor.logical_shape.size() != 1 ||
      descriptor.logical_shape[0] != 1) {
    return std::nullopt;
  }
  if (descriptor.storage_dtype == "fp32" && descriptor.packed_nbytes == sizeof(float)) {
    float value = 0.0f;
    std::memcpy(&value, descriptor.packed_data, sizeof(float));
    return value;
  }
  if (descriptor.storage_dtype == "bf16" && descriptor.packed_nbytes == sizeof(__nv_bfloat16)) {
    __nv_bfloat16 value;
    std::memcpy(&value, descriptor.packed_data, sizeof(value));
    return __bfloat162float(value);
  }
  return std::nullopt;
}

std::optional<nemotron::ScaledFp8LinearConfig> BuildScaledFp8LinearConfigLocal(
    const nemotron::KernelTensorDescriptor& weight,
    const nemotron::KernelTensorDescriptor& weight_scale,
    const nemotron::KernelTensorDescriptor& input_scale) {
  if (weight.logical_shape.size() != 2 || weight.packed_data == nullptr) {
    return std::nullopt;
  }
  const auto weight_scale_value = ReadScalarTensorToHostFp32(weight_scale);
  const auto input_scale_value = ReadScalarTensorToHostFp32(input_scale);
  if (!weight_scale_value.has_value() || !input_scale_value.has_value()) {
    return std::nullopt;
  }
  nemotron::ScaledFp8LinearConfig config;
  config.output_rows = weight.logical_shape[0];
  config.input_cols = weight.logical_shape[1];
  config.packed_weight_data = weight.packed_data;
  config.packed_weight_nbytes = weight.packed_nbytes;
  config.tensor_name = weight.tensor_name;
  config.weight_scale = *weight_scale_value;
  config.input_scale = *input_scale_value;
  return config;
}

std::string DirectSharedDownSummaryJson(
    std::size_t layer_index,
    std::size_t token_count,
    std::size_t input_cols,
    std::size_t output_rows,
    const char* input_dtype,
    const std::string& family,
    std::optional<float> nvfp4_fixed_input_scale,
    const std::optional<DiffStats>& diff_stats,
    const nemotron::RuntimeExecutionStatsSnapshot& runtime_stats) {
  std::string json = "{\n"
                     "  \"mode\": \"direct_shared_down\",\n"
                     "  \"layer_index\": " + std::to_string(layer_index) + ",\n" +
                     "  \"token_count\": " + std::to_string(token_count) + ",\n" +
                     "  \"input_cols\": " + std::to_string(input_cols) + ",\n" +
                     "  \"output_rows\": " + std::to_string(output_rows) + ",\n" +
                     "  \"input_dtype\": \"" + std::string(input_dtype) + "\",\n" +
                     "  \"family\": \"" + family + "\"";
  if (nvfp4_fixed_input_scale.has_value()) {
    json += ",\n"
            "  \"nvfp4_fixed_input_scale\": " + std::to_string(*nvfp4_fixed_input_scale);
  }
  if (diff_stats.has_value()) {
    json += ",\n"
            "  \"diff\": {\n"
            "    \"max_abs_diff\": " + std::to_string(diff_stats->max_abs) + ",\n" +
            "    \"mean_abs_diff\": " + std::to_string(diff_stats->mean_abs) + ",\n" +
            "    \"rel_l2\": " + std::to_string(diff_stats->rel_l2) + "\n"
            "  }";
  }
  json += ",\n"
          "  \"runtime_stats\": {\n"
          "    \"dense_native_success\": " + std::to_string(runtime_stats.dense_native_success) + ",\n" +
          "    \"dense_reference_fallbacks\": " + std::to_string(runtime_stats.dense_reference_fallbacks) + ",\n" +
          "    \"scaled_fp8_native_success\": " + std::to_string(runtime_stats.scaled_fp8_native_success) + ",\n" +
          "    \"scaled_fp8_reference_fallbacks\": " + std::to_string(runtime_stats.scaled_fp8_reference_fallbacks) + "\n"
          "  }\n"
          "}\n";
  return json;
}

std::string DirectRoutedUpSummaryJson(
    std::size_t layer_index,
    std::size_t expert_index,
    std::size_t token_count,
    std::size_t input_cols,
    std::size_t output_rows,
    const char* input_dtype,
    const std::string& family,
    std::optional<float> checkpoint_input_scale,
    std::optional<float> effective_input_scale,
    const std::optional<DiffStats>& diff_stats,
    const nemotron::RuntimeExecutionStatsSnapshot& runtime_stats) {
  std::string json = "{\n"
                     "  \"mode\": \"direct_routed_up\",\n"
                     "  \"layer_index\": " + std::to_string(layer_index) + ",\n" +
                     "  \"expert_index\": " + std::to_string(expert_index) + ",\n" +
                     "  \"token_count\": " + std::to_string(token_count) + ",\n" +
                     "  \"input_cols\": " + std::to_string(input_cols) + ",\n" +
                     "  \"output_rows\": " + std::to_string(output_rows) + ",\n" +
                     "  \"input_dtype\": \"" + std::string(input_dtype) + "\",\n" +
                     "  \"family\": \"" + family + "\"";
  if (checkpoint_input_scale.has_value()) {
    json += ",\n"
            "  \"checkpoint_input_scale\": " + std::to_string(*checkpoint_input_scale);
  }
  if (effective_input_scale.has_value()) {
    json += ",\n"
            "  \"effective_input_scale\": " + std::to_string(*effective_input_scale);
  }
  if (diff_stats.has_value()) {
    json += ",\n"
            "  \"diff\": {\n"
            "    \"max_abs_diff\": " + std::to_string(diff_stats->max_abs) + ",\n" +
            "    \"mean_abs_diff\": " + std::to_string(diff_stats->mean_abs) + ",\n" +
            "    \"rel_l2\": " + std::to_string(diff_stats->rel_l2) + "\n"
            "  }";
  }
  json += ",\n"
          "  \"runtime_stats\": {\n"
          "    \"dense_native_success\": " + std::to_string(runtime_stats.dense_native_success) + ",\n" +
          "    \"dense_reference_fallbacks\": " + std::to_string(runtime_stats.dense_reference_fallbacks) + ",\n" +
          "    \"scaled_fp8_native_success\": " + std::to_string(runtime_stats.scaled_fp8_native_success) + ",\n" +
          "    \"scaled_fp8_reference_fallbacks\": " + std::to_string(runtime_stats.scaled_fp8_reference_fallbacks) + "\n"
          "  }\n"
          "}\n";
  return json;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    std::cerr << "usage: expert_layer_trace_dump"
              << " --manifest <path>"
              << " (--input-hidden-bin <path> | --shared-down-input-bin <path> | --routed-up-input-bin <path>)"
              << " --dump-root <path>"
              << " [--layer-index <index>]"
              << " [--routed-expert-index <index>]"
              << " [--expected-output-bin <path>]"
              << " [--nvfp4-fixed-input-scale <float>]"
              << " [--input-dtype <fp32|bf16>]\n";
    return 2;
  }
  if (!HasCudaDevice()) {
    std::cerr << "expert_layer_trace_dump: CUDA device unavailable\n";
    return 1;
  }

  nemotron::SingleTokenForwardConfig config = nemotron::KnownNemotron3Super120BA12BConfig();
  std::size_t token_count = 0;
  const auto input_width = DirectSharedDownMode(options)
      ? config.shared_expert_intermediate_size
      : (DirectRoutedUpMode(options) ? config.moe_latent_size
                                     : config.hidden_size);
  const auto input_matrix_host = LoadFloatMatrix(
      DirectSharedDownMode(options)
          ? options.shared_down_input_path
          : (DirectRoutedUpMode(options) ? options.routed_up_input_path
                                         : options.input_hidden_path),
      input_width,
      &token_count);
  if (!input_matrix_host.has_value()) {
    std::cerr << "expert_layer_trace_dump: failed to load input matrix\n";
    return 1;
  }
  config.max_tokens = token_count;

  const auto environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(
      options.manifest_path,
      MakeOptions(token_count));
  if (!environment ||
      !environment->has_model_schedule() ||
      !environment->has_kernel_catalog() ||
      !environment->has_gemm_catalog() ||
      !environment->has_gemm_heuristic_cache()) {
    std::cerr << "expert_layer_trace_dump: runtime environment unavailable\n";
    return 1;
  }

  const nemotron::ModelSchedule& schedule = *environment->model_schedule();
  const auto plan = nemotron::BuildSingleTokenForwardPlan(schedule, config);
  if (!plan.has_value()) {
    std::cerr << "expert_layer_trace_dump: failed to build forward plan\n";
    return 1;
  }

  const nemotron::LayerScheduleEntry* layer = schedule.FindLayer(options.layer_index);
  if (layer == nullptr) {
    std::cerr << "expert_layer_trace_dump: missing layer " << options.layer_index << "\n";
    return 1;
  }

  const auto bindings = nemotron::BuildExpertLayerBindings(
      *layer,
      *environment->kernel_catalog(),
      *environment->gemm_catalog(),
      config.n_routed_experts);
  if (!bindings.has_value()) {
    std::cerr << "expert_layer_trace_dump: failed to build expert bindings\n";
    return 1;
  }

  nemotron::ExpertLayerConfig expert_config;
  expert_config.layer_index = options.layer_index;
  expert_config.hidden_size = config.hidden_size;
  expert_config.moe_latent_size = config.moe_latent_size;
  expert_config.routed_expert_intermediate_size = config.routed_expert_intermediate_size;
  expert_config.shared_expert_intermediate_size = config.shared_expert_intermediate_size;
  expert_config.n_routed_experts = config.n_routed_experts;
  expert_config.top_k = config.experts_per_token;
  expert_config.n_group = config.expert_n_group;
  expert_config.topk_group = config.expert_topk_group;
  expert_config.rms_epsilon = config.layer_norm_epsilon;
  expert_config.routed_scaling_factor = config.routed_scaling_factor;
  expert_config.norm_topk_prob = config.norm_topk_prob;

  if (DirectSharedDownMode(options)) {
    std::optional<std::vector<float>> expected_output_host;
    if (!options.expected_output_path.empty()) {
      std::size_t expected_rows = 0;
      expected_output_host =
          LoadFloatMatrix(options.expected_output_path, config.hidden_size, &expected_rows);
      if (!expected_output_host.has_value() || expected_rows != token_count) {
        std::cerr << "expert_layer_trace_dump: failed to load expected shared_down output\n";
        return 1;
      }
    }

    std::unique_ptr<nemotron::UploadedLinearOp> shared_down_uploaded;
    std::unique_ptr<nemotron::ScaledFp8LinearOp> shared_down_scaled_fp8;
    std::unique_ptr<Nvfp4AlignedBuffers> shared_down_nvfp4_buffers;
    std::string shared_down_family;
    if (bindings->shared_down_gemm_weight != nullptr) {
      if (bindings->shared_down_gemm_weight->kernel_family ==
          nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
        shared_down_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
        shared_down_uploaded = CreateAlignedNvfp4LinearOp(
            *bindings->shared_down_gemm_weight,
            shared_down_nvfp4_buffers.get());
      } else {
        shared_down_uploaded = nemotron::UploadedLinearOp::Create(*bindings->shared_down_gemm_weight);
      }
      if (!shared_down_uploaded || !shared_down_uploaded->valid()) {
        std::cerr << "expert_layer_trace_dump: failed to create shared_down uploaded op\n";
        return 1;
      }
      shared_down_family =
          bindings->shared_down_gemm_weight->kernel_family ==
                  nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled
              ? "nvfp4"
              : "dense";
    } else if (bindings->shared_down_kernel_weight != nullptr &&
               bindings->shared_down_weight_scale != nullptr &&
               bindings->shared_down_input_scale != nullptr) {
      const auto shared_down_config = BuildScaledFp8LinearConfigLocal(
          *bindings->shared_down_kernel_weight,
          *bindings->shared_down_weight_scale,
          *bindings->shared_down_input_scale);
      if (!shared_down_config.has_value()) {
        std::cerr << "expert_layer_trace_dump: failed to build shared_down scaled-fp8 config\n";
        return 1;
      }
      shared_down_scaled_fp8 = nemotron::ScaledFp8LinearOp::Create(*shared_down_config);
      if (!shared_down_scaled_fp8 || !shared_down_scaled_fp8->valid()) {
        std::cerr << "expert_layer_trace_dump: failed to create shared_down scaled-fp8 op\n";
        return 1;
      }
      shared_down_family = "scaled_fp8";
    } else {
      std::cerr << "expert_layer_trace_dump: shared_down binding resolution failed\n";
      return 1;
    }

    auto cublas = nemotron::CublasLtHandle::Create();
    if (!cublas || !cublas->valid()) {
      std::cerr << "expert_layer_trace_dump: cublas handle unavailable\n";
      return 1;
    }
    std::vector<float> output_host;
    nemotron::ResetRuntimeExecutionStats();
    if (options.input_dtype == Options::InputDtype::kFp32) {
      auto input = nemotron::DeviceTensorFp32::Create({token_count, config.shared_expert_intermediate_size});
      auto output = nemotron::DeviceTensorFp32::Create({token_count, config.hidden_size});
      if (!input || !input->valid() || !output || !output->valid() ||
          !input->CopyFromHost(input_matrix_host->data(), input_matrix_host->size())) {
        std::cerr << "expert_layer_trace_dump: shared_down fp32 setup failed\n";
        return 1;
      }
      const bool ok = shared_down_uploaded
          ? (options.nvfp4_fixed_input_scale.has_value() && shared_down_family == "nvfp4"
                 ? shared_down_uploaded->Run(
                       *cublas,
                       environment->gemm_heuristic_cache(),
                       *input,
                       output.get(),
                       nemotron::Nvfp4PackOptions{options.nvfp4_fixed_input_scale})
                 : shared_down_uploaded->Run(
                       *cublas,
                       environment->gemm_heuristic_cache(),
                       *input,
                       output.get()))
          : shared_down_scaled_fp8->Run(
                *cublas,
                environment->gemm_heuristic_cache(),
                *input,
                output.get());
      if (!ok) {
        std::cerr << "expert_layer_trace_dump: shared_down fp32 execution failed\n";
        return 1;
      }
      output_host.resize(output->numel(), 0.0f);
      if (!output->CopyToHost(output_host.data(), output_host.size())) {
        std::cerr << "expert_layer_trace_dump: shared_down fp32 output download failed\n";
        return 1;
      }
    } else {
      auto input = nemotron::DeviceTensorBf16::Create({token_count, config.shared_expert_intermediate_size});
      auto output = nemotron::DeviceTensorBf16::Create({token_count, config.hidden_size});
      const auto input_bf16 = ConvertHostFp32ToBf16(*input_matrix_host);
      if (!input || !input->valid() || !output || !output->valid() ||
          !input->CopyFromHost(input_bf16.data(), input_bf16.size())) {
        std::cerr << "expert_layer_trace_dump: shared_down bf16 setup failed\n";
        return 1;
      }
      const bool ok = shared_down_uploaded
          ? (options.nvfp4_fixed_input_scale.has_value() && shared_down_family == "nvfp4"
                 ? shared_down_uploaded->Run(
                       *cublas,
                       environment->gemm_heuristic_cache(),
                       *input,
                       output.get(),
                       nemotron::Nvfp4PackOptions{options.nvfp4_fixed_input_scale})
                 : shared_down_uploaded->Run(
                       *cublas,
                       environment->gemm_heuristic_cache(),
                       *input,
                       output.get()))
          : shared_down_scaled_fp8->Run(
                *cublas,
                environment->gemm_heuristic_cache(),
                *input,
                output.get());
      if (!ok) {
        std::cerr << "expert_layer_trace_dump: shared_down bf16 execution failed\n";
        return 1;
      }
      std::vector<__nv_bfloat16> output_bf16(output->numel());
      if (!output->CopyToHost(output_bf16.data(), output_bf16.size())) {
        std::cerr << "expert_layer_trace_dump: shared_down bf16 output download failed\n";
        return 1;
      }
      output_host = ConvertHostBf16ToFp32(output_bf16);
    }
    const auto runtime_stats = nemotron::GetRuntimeExecutionStatsSnapshot();
    const auto diff_stats = expected_output_host.has_value()
        ? ComputeDiffStats(output_host, *expected_output_host)
        : std::nullopt;
    std::filesystem::create_directories(options.dump_root);
    WriteFloatFile(options.dump_root / "trace_shared_output_fp32.bin", output_host);
    WriteJsonFile(
        options.dump_root / "summary.json",
        DirectSharedDownSummaryJson(
            options.layer_index,
            token_count,
            config.shared_expert_intermediate_size,
            config.hidden_size,
            InputDtypeName(options.input_dtype),
            shared_down_family,
            options.nvfp4_fixed_input_scale,
            diff_stats,
            runtime_stats));
    std::cout << "expert_layer_trace_dump: mode=direct_shared_down"
              << " layer_index=" << options.layer_index
              << " token_count=" << token_count
              << " input_dtype=" << InputDtypeName(options.input_dtype)
              << " family=" << shared_down_family << "\n";
    return 0;
  }

  if (DirectRoutedUpMode(options)) {
    if (options.routed_expert_index >= bindings->routed_experts.size()) {
      std::cerr << "expert_layer_trace_dump: routed expert index out of range: "
                << options.routed_expert_index << "\n";
      return 1;
    }

    const auto& pair = bindings->routed_experts[options.routed_expert_index];
    if (pair.up_proj == nullptr) {
      std::cerr << "expert_layer_trace_dump: routed up binding missing for expert "
                << options.routed_expert_index << "\n";
      return 1;
    }

    std::optional<std::vector<float>> expected_output_host;
    if (!options.expected_output_path.empty()) {
      std::size_t expected_rows = 0;
      expected_output_host =
          LoadFloatMatrix(options.expected_output_path,
                          config.routed_expert_intermediate_size,
                          &expected_rows);
      if (!expected_output_host.has_value() || expected_rows != token_count) {
        std::cerr << "expert_layer_trace_dump: failed to load expected routed_up output\n";
        return 1;
      }
    }

    std::unique_ptr<nemotron::UploadedLinearOp> routed_up_uploaded;
    std::unique_ptr<Nvfp4AlignedBuffers> routed_up_nvfp4_buffers;
    std::string routed_up_family;
    if (pair.up_proj->kernel_family == nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
      routed_up_nvfp4_buffers = std::make_unique<Nvfp4AlignedBuffers>();
      routed_up_uploaded = CreateAlignedNvfp4LinearOp(*pair.up_proj, routed_up_nvfp4_buffers.get());
      routed_up_family = "nvfp4";
    } else {
      routed_up_uploaded = nemotron::UploadedLinearOp::Create(*pair.up_proj);
      routed_up_family = "dense";
    }
    if (!routed_up_uploaded || !routed_up_uploaded->valid()) {
      std::cerr << "expert_layer_trace_dump: failed to create routed_up uploaded op\n";
      return 1;
    }

    const std::optional<float> checkpoint_input_scale =
        pair.up_input_scale == nullptr ? std::nullopt : ReadScalarTensorToHostFp32(*pair.up_input_scale);
    const std::optional<float> effective_input_scale =
        options.nvfp4_fixed_input_scale.has_value() ? options.nvfp4_fixed_input_scale
                                                    : checkpoint_input_scale;

    auto cublas = nemotron::CublasLtHandle::Create();
    if (!cublas || !cublas->valid()) {
      std::cerr << "expert_layer_trace_dump: cublas handle unavailable\n";
      return 1;
    }

    std::vector<float> output_host;
    nemotron::ResetRuntimeExecutionStats();
    if (options.input_dtype == Options::InputDtype::kFp32) {
      auto input = nemotron::DeviceTensorFp32::Create({token_count, config.moe_latent_size});
      auto output = nemotron::DeviceTensorFp32::Create(
          {token_count, config.routed_expert_intermediate_size});
      if (!input || !input->valid() || !output || !output->valid() ||
          !input->CopyFromHost(input_matrix_host->data(), input_matrix_host->size())) {
        std::cerr << "expert_layer_trace_dump: routed_up fp32 setup failed\n";
        return 1;
      }
      const bool ok =
          routed_up_family == "nvfp4" && effective_input_scale.has_value()
              ? routed_up_uploaded->Run(
                    *cublas,
                    environment->gemm_heuristic_cache(),
                    *input,
                    output.get(),
                    nemotron::Nvfp4PackOptions{effective_input_scale})
              : routed_up_uploaded->Run(
                    *cublas,
                    environment->gemm_heuristic_cache(),
                    *input,
                    output.get());
      if (!ok) {
        std::cerr << "expert_layer_trace_dump: routed_up fp32 execution failed\n";
        return 1;
      }
      output_host.resize(output->numel(), 0.0f);
      if (!output->CopyToHost(output_host.data(), output_host.size())) {
        std::cerr << "expert_layer_trace_dump: routed_up fp32 output download failed\n";
        return 1;
      }
    } else {
      auto input = nemotron::DeviceTensorBf16::Create({token_count, config.moe_latent_size});
      auto output = nemotron::DeviceTensorBf16::Create(
          {token_count, config.routed_expert_intermediate_size});
      const auto input_bf16 = ConvertHostFp32ToBf16(*input_matrix_host);
      if (!input || !input->valid() || !output || !output->valid() ||
          !input->CopyFromHost(input_bf16.data(), input_bf16.size())) {
        std::cerr << "expert_layer_trace_dump: routed_up bf16 setup failed\n";
        return 1;
      }
      const bool ok =
          routed_up_family == "nvfp4" && effective_input_scale.has_value()
              ? routed_up_uploaded->Run(
                    *cublas,
                    environment->gemm_heuristic_cache(),
                    *input,
                    output.get(),
                    nemotron::Nvfp4PackOptions{effective_input_scale})
              : routed_up_uploaded->Run(
                    *cublas,
                    environment->gemm_heuristic_cache(),
                    *input,
                    output.get());
      if (!ok) {
        std::cerr << "expert_layer_trace_dump: routed_up bf16 execution failed\n";
        return 1;
      }
      std::vector<__nv_bfloat16> output_bf16(output->numel());
      if (!output->CopyToHost(output_bf16.data(), output_bf16.size())) {
        std::cerr << "expert_layer_trace_dump: routed_up bf16 output download failed\n";
        return 1;
      }
      output_host = ConvertHostBf16ToFp32(output_bf16);
    }

    const auto runtime_stats = nemotron::GetRuntimeExecutionStatsSnapshot();
    const auto diff_stats = expected_output_host.has_value()
        ? ComputeDiffStats(output_host, *expected_output_host)
        : std::nullopt;
    std::filesystem::create_directories(options.dump_root);
    WriteFloatFile(options.dump_root / "trace_routed_pre_activation_hidden_fp32.bin", output_host);
    WriteJsonFile(
        options.dump_root / "summary.json",
        DirectRoutedUpSummaryJson(
            options.layer_index,
            options.routed_expert_index,
            token_count,
            config.moe_latent_size,
            config.routed_expert_intermediate_size,
            InputDtypeName(options.input_dtype),
            routed_up_family,
            checkpoint_input_scale,
            effective_input_scale,
            diff_stats,
            runtime_stats));
    std::cout << "expert_layer_trace_dump: mode=direct_routed_up"
              << " layer_index=" << options.layer_index
              << " expert_index=" << options.routed_expert_index
              << " token_count=" << token_count
              << " input_dtype=" << InputDtypeName(options.input_dtype)
              << " family=" << routed_up_family << "\n";
    return 0;
  }

  auto slice = nemotron::ExpertLayerSlice::Create(expert_config, *bindings);
  if (!slice || !slice->valid()) {
    std::cerr << "expert_layer_trace_dump: failed to create expert slice\n";
    return 1;
  }
  if (options.dump_row_traces && options.input_dtype == Options::InputDtype::kBf16) {
    std::cerr << "expert_layer_trace_dump: --dump-row-traces is not implemented for bf16 input mode\n";
    return 2;
  }

  auto request_context = nemotron::RequestExecutionContext::Create(plan->request_config);
  auto cublas = nemotron::CublasLtHandle::Create();
  std::vector<float> output_host;
  nemotron::ExpertLayerRunTrace trace;
  nemotron::ResetRuntimeExecutionStats();
  if (options.input_dtype == Options::InputDtype::kFp32) {
    auto input = nemotron::DeviceTensorFp32::Create({token_count, config.hidden_size});
    auto output = nemotron::DeviceTensorFp32::Create({token_count, config.hidden_size});
    if (!request_context || !request_context->valid() ||
        !cublas || !cublas->valid() ||
        !input || !input->valid() ||
        !output || !output->valid() ||
        !input->CopyFromHost(input_matrix_host->data(), input_matrix_host->size())) {
      std::cerr << "expert_layer_trace_dump: execution setup failed\n";
      return 1;
    }
    if (!slice->RunWithRequestContext(
            *cublas,
            environment->gemm_heuristic_cache(),
            *request_context,
            *input,
            output.get(),
            &trace)) {
      std::cerr << "expert_layer_trace_dump: expert execution failed\n";
      return 1;
    }
    output_host.resize(output->numel(), 0.0f);
    if (!output->CopyToHost(output_host.data(), output_host.size())) {
      std::cerr << "expert_layer_trace_dump: output download failed\n";
      return 1;
    }
  } else {
    auto input = nemotron::DeviceTensorBf16::Create({token_count, config.hidden_size});
    auto output = nemotron::DeviceTensorBf16::Create({token_count, config.hidden_size});
    const auto input_hidden_bf16 = ConvertHostFp32ToBf16(*input_matrix_host);
    if (!request_context || !request_context->valid() ||
        !cublas || !cublas->valid() ||
        !input || !input->valid() ||
        !output || !output->valid() ||
        !input->CopyFromHost(input_hidden_bf16.data(), input_hidden_bf16.size())) {
      std::cerr << "expert_layer_trace_dump: execution setup failed\n";
      return 1;
    }
    if (!slice->RunWithRequestContext(
            *cublas,
            environment->gemm_heuristic_cache(),
            *request_context,
            *input,
            output.get(),
            &trace)) {
      std::cerr << "expert_layer_trace_dump: expert execution failed\n";
      return 1;
    }
    std::vector<__nv_bfloat16> output_bf16(output->numel());
    if (!output->CopyToHost(output_bf16.data(), output_bf16.size())) {
      std::cerr << "expert_layer_trace_dump: output download failed\n";
      return 1;
    }
    output_host = ConvertHostBf16ToFp32(output_bf16);
  }
  const auto runtime_stats = nemotron::GetRuntimeExecutionStatsSnapshot();

  std::filesystem::create_directories(options.dump_root);
  WriteFloatFile(options.dump_root / "final_output_fp32.bin", output_host);
  if (token_count == 1) {
    if (!trace.normalized_input.empty()) {
      WriteFloatFile(options.dump_root / "trace_normalized_input_fp32.bin", trace.normalized_input);
    }
    if (!trace.router_logits.empty()) {
      WriteFloatFile(options.dump_root / "trace_router_logits_fp32.bin", trace.router_logits);
    }
    if (!trace.latent_output.empty()) {
      WriteFloatFile(options.dump_root / "trace_latent_output_fp32.bin", trace.latent_output);
    }
    if (!trace.routed_latent_output.empty()) {
      WriteFloatFile(options.dump_root / "trace_routed_latent_output_fp32.bin", trace.routed_latent_output);
    }
    if (!trace.projected_routed_output.empty()) {
      WriteFloatFile(options.dump_root / "trace_projected_routed_output_fp32.bin", trace.projected_routed_output);
    }
    if (!trace.routed_expert_activated_hidden.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_routed_expert_activated_hidden_fp32.bin",
          trace.routed_expert_activated_hidden);
    }
    if (!trace.routed_expert_outputs.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_routed_expert_outputs_fp32.bin",
          trace.routed_expert_outputs);
    }
    if (!trace.routed_expert_weighted_contributions.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_routed_expert_weighted_contributions_fp32.bin",
          trace.routed_expert_weighted_contributions);
    }
    if (!trace.shared_output.empty()) {
      WriteFloatFile(options.dump_root / "trace_shared_output_fp32.bin", trace.shared_output);
    }
    if (!trace.mixer_output.empty()) {
      WriteFloatFile(options.dump_root / "trace_mixer_output_fp32.bin", trace.mixer_output);
    }
    if (!trace.routed_expert_order.empty()) {
      WriteJsonFile(
          options.dump_root / "summary.json",
          SummaryJson(
              options.layer_index,
              token_count,
              config.hidden_size,
              InputDtypeName(options.input_dtype),
              trace,
              runtime_stats));
    } else {
      WriteJsonFile(
          options.dump_root / "summary.json",
          SummaryJson(
              options.layer_index,
              token_count,
              config.hidden_size,
              InputDtypeName(options.input_dtype),
              false,
              std::nullopt,
              runtime_stats));
    }
  } else {
    if (!trace.routed_latent_output.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_routed_latent_output_fp32.bin",
          trace.routed_latent_output);
    }
    if (!trace.projected_routed_output.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_projected_routed_output_fp32.bin",
          trace.projected_routed_output);
    }
    if (!trace.shared_output.empty()) {
      WriteFloatFile(options.dump_root / "trace_shared_output_fp32.bin", trace.shared_output);
    }
    if (!trace.mixer_output.empty()) {
      WriteFloatFile(options.dump_root / "trace_mixer_output_fp32.bin", trace.mixer_output);
    }
    if (!trace.normalized_input.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_normalized_input_fp32.bin",
          trace.normalized_input);
    }
    if (!trace.router_logits.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_router_logits_fp32.bin",
          trace.router_logits);
    }
    if (!trace.latent_output.empty()) {
      WriteFloatFile(
          options.dump_root / "trace_latent_output_fp32.bin",
          trace.latent_output);
    }
    if (!trace.selected_experts.empty()) {
      std::vector<std::int32_t> selected_ids;
      std::vector<float> selected_weights;
      selected_ids.reserve(trace.selected_experts.size());
      selected_weights.reserve(trace.selected_experts.size());
      for (const auto& selection : trace.selected_experts) {
        selected_ids.push_back(static_cast<std::int32_t>(selection.expert_index));
        selected_weights.push_back(selection.weight);
      }
      WriteInt32File(
          options.dump_root / "trace_selected_expert_ids_i32.bin",
          selected_ids);
      WriteFloatFile(
          options.dump_root / "trace_selected_expert_weights_fp32.bin",
          selected_weights);
    }

    std::string summary_json = SummaryJson(
        options.layer_index,
        token_count,
        config.hidden_size,
        InputDtypeName(options.input_dtype),
        options.dump_row_traces,
        std::nullopt,
        runtime_stats);
    if (options.dump_row_traces) {
      auto row_input = nemotron::DeviceTensorFp32::Create({1, config.hidden_size});
      auto row_output = nemotron::DeviceTensorFp32::Create({1, config.hidden_size});
      if (!row_input || !row_input->valid() || !row_output || !row_output->valid()) {
        std::cerr << "expert_layer_trace_dump: row trace tensor allocation failed\n";
        return 1;
      }
      std::vector<float> row_input_host(config.hidden_size, 0.0f);
      std::vector<float> row_output_host(config.hidden_size, 0.0f);
      std::vector<float> trace_normalized;
      std::vector<float> trace_router_logits;
      std::vector<float> trace_latent_output;
      std::vector<float> trace_routed_latent_output;
      std::vector<float> trace_projected_routed_output;
      std::vector<float> trace_shared_output;
      std::vector<float> trace_mixer_output;
      std::vector<std::int32_t> trace_selected_expert_ids;
      std::vector<float> trace_selected_expert_weights;
      trace_selected_expert_ids.reserve(token_count * config.experts_per_token);
      trace_selected_expert_weights.reserve(token_count * config.experts_per_token);

      for (std::size_t row = 0; row < token_count; ++row) {
        std::copy_n(
            input_matrix_host->data() + (row * config.hidden_size),
            config.hidden_size,
            row_input_host.data());
        if (!row_input->CopyFromHost(row_input_host.data(), row_input_host.size())) {
          std::cerr << "expert_layer_trace_dump: row input upload failed at row " << row << "\n";
          return 1;
        }

        nemotron::ExpertLayerRunTrace row_trace;
        if (!slice->RunWithRequestContext(
                *cublas,
                environment->gemm_heuristic_cache(),
                *request_context,
                *row_input,
                row_output.get(),
                &row_trace)) {
          std::cerr << "expert_layer_trace_dump: row trace run failed at row " << row << "\n";
          return 1;
        }
        if (!row_output->CopyToHost(row_output_host.data(), row_output_host.size())) {
          std::cerr << "expert_layer_trace_dump: row output download failed at row " << row << "\n";
          return 1;
        }
        trace_normalized.insert(
            trace_normalized.end(),
            row_trace.normalized_input.begin(),
            row_trace.normalized_input.end());
        trace_router_logits.insert(
            trace_router_logits.end(),
            row_trace.router_logits.begin(),
            row_trace.router_logits.end());
        trace_latent_output.insert(
            trace_latent_output.end(),
            row_trace.latent_output.begin(),
            row_trace.latent_output.end());
        trace_routed_latent_output.insert(
            trace_routed_latent_output.end(),
            row_trace.routed_latent_output.begin(),
            row_trace.routed_latent_output.end());
        trace_projected_routed_output.insert(
            trace_projected_routed_output.end(),
            row_trace.projected_routed_output.begin(),
            row_trace.projected_routed_output.end());
        trace_shared_output.insert(
            trace_shared_output.end(),
            row_trace.shared_output.begin(),
            row_trace.shared_output.end());
        trace_mixer_output.insert(
            trace_mixer_output.end(),
            row_trace.mixer_output.begin(),
            row_trace.mixer_output.end());
        for (const auto& selection : row_trace.selected_experts) {
          trace_selected_expert_ids.push_back(static_cast<std::int32_t>(selection.expert_index));
          trace_selected_expert_weights.push_back(selection.weight);
        }
      }

      if (!trace_normalized.empty()) {
        WriteFloatFile(options.dump_root / "trace_normalized_input_fp32.bin", trace_normalized);
      }
      if (!trace_router_logits.empty()) {
        WriteFloatFile(options.dump_root / "trace_router_logits_fp32.bin", trace_router_logits);
      }
      if (!trace_latent_output.empty()) {
        WriteFloatFile(options.dump_root / "trace_latent_output_fp32.bin", trace_latent_output);
      }
      if (!trace_routed_latent_output.empty()) {
        WriteFloatFile(
            options.dump_root / "trace_routed_latent_output_fp32.bin",
            trace_routed_latent_output);
      }
      if (!trace_projected_routed_output.empty()) {
        WriteFloatFile(
            options.dump_root / "trace_projected_routed_output_fp32.bin",
            trace_projected_routed_output);
      }
      if (!trace_shared_output.empty()) {
        WriteFloatFile(options.dump_root / "trace_shared_output_fp32.bin", trace_shared_output);
      }
      if (!trace_mixer_output.empty()) {
        WriteFloatFile(options.dump_root / "trace_mixer_output_fp32.bin", trace_mixer_output);
      }
      if (!trace_selected_expert_ids.empty()) {
        WriteInt32File(
            options.dump_root / "trace_selected_expert_ids_i32.bin",
            trace_selected_expert_ids);
      }
      if (!trace_selected_expert_weights.empty()) {
        WriteFloatFile(
            options.dump_root / "trace_selected_expert_weights_fp32.bin",
            trace_selected_expert_weights);
      }
      summary_json = SummaryJson(
          options.layer_index,
          token_count,
          config.hidden_size,
          InputDtypeName(options.input_dtype),
          true,
          config.experts_per_token,
          runtime_stats);
    }
    WriteJsonFile(options.dump_root / "summary.json", summary_json);
  }

  std::cout << "expert_layer_trace_dump: layer_index=" << options.layer_index
            << " token_count=" << token_count
            << " output_rows=" << token_count
            << "\n";
  return 0;
}
