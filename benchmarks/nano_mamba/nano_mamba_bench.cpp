#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/reusable_state.h"
#include "nemotron/request_context.h"
#include "nemotron/state_snapshot.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorFp32;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::KernelTensorDescriptor;
using nemotron::KvCacheDataType;
using nemotron::MambaLayerBindings;
using nemotron::MambaLayerConfig;
using nemotron::MambaLayerSlice;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;
using nemotron::ReusableStateArena;

struct FixtureMetadata {
  std::size_t layer_index = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;
  std::size_t state_size = 0;
  std::size_t n_groups = 0;
  std::size_t conv_kernel_size = 0;
  float input_rms_epsilon = 0.0f;
  float mixer_rms_epsilon = 0.0f;
  float time_step_min = 0.0f;
};

struct BenchmarkOptions {
  std::filesystem::path fixture_root = NEMOTRON_MAMBA_LAYER_BENCH_FIXTURE_ROOT;
  std::size_t cached_prefix_tokens = 48;
  std::size_t tail_prefill_tokens = 16;
  std::size_t warmup_iterations = 3;
  std::size_t benchmark_iterations = 10;
  std::optional<std::filesystem::path> json_output_path;
};

struct BenchmarkResult {
  BenchmarkOptions options;
  FixtureMetadata metadata;
  double cold_prefill_mean_ms = 0.0;
  double restored_prefix_prefill_mean_ms = 0.0;
  double single_token_decode_mean_ms = 0.0;
  bool restored_prefix_excludes_restore_ms = true;
};

bool CheckCuda(cudaError_t status, const char* where) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "nano_mamba_bench: " << where << ": "
            << cudaGetErrorString(status) << "\n";
  return false;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::vector<std::uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

std::vector<float> ReadFloatFile(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = ReadFileBytes(path);
  if (bytes.size() % sizeof(float) != 0) {
    return {};
  }
  std::vector<float> values(bytes.size() / sizeof(float), 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

bool FixturePayloadPresent(
    const std::filesystem::path& root,
    const std::vector<std::string>& file_names) {
  for (const std::string& file_name : file_names) {
    if (!std::filesystem::exists(root / file_name)) {
      return false;
    }
  }
  return true;
}

std::optional<std::size_t> ParseJsonUintField(
    const std::string& json,
    const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(match[1].str()));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<float> ParseJsonFloatField(
    const std::string& json,
    const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  try {
    return std::stof(match[1].str());
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<FixtureMetadata> LoadMetadata(const std::filesystem::path& root) {
  const std::string json = ReadTextFile(root / "metadata.json");
  FixtureMetadata metadata;
  const auto layer_index = ParseJsonUintField(json, "layer_index");
  const auto hidden_size = ParseJsonUintField(json, "hidden_size");
  const auto intermediate_size = ParseJsonUintField(json, "intermediate_size");
  const auto num_heads = ParseJsonUintField(json, "num_heads");
  const auto head_dim = ParseJsonUintField(json, "head_dim");
  const auto state_size = ParseJsonUintField(json, "state_size");
  const auto n_groups = ParseJsonUintField(json, "n_groups");
  const auto conv_kernel_size = ParseJsonUintField(json, "conv_kernel_size");
  const auto input_rms_epsilon = ParseJsonFloatField(json, "input_rms_epsilon");
  const auto mixer_rms_epsilon = ParseJsonFloatField(json, "mixer_rms_epsilon");
  const auto time_step_min = ParseJsonFloatField(json, "time_step_min");
  if (!layer_index || !hidden_size || !intermediate_size || !num_heads ||
      !head_dim || !state_size || !n_groups || !conv_kernel_size ||
      !input_rms_epsilon || !mixer_rms_epsilon || !time_step_min) {
    return std::nullopt;
  }
  metadata.layer_index = *layer_index;
  metadata.hidden_size = *hidden_size;
  metadata.intermediate_size = *intermediate_size;
  metadata.num_heads = *num_heads;
  metadata.head_dim = *head_dim;
  metadata.state_size = *state_size;
  metadata.n_groups = *n_groups;
  metadata.conv_kernel_size = *conv_kernel_size;
  metadata.input_rms_epsilon = *input_rms_epsilon;
  metadata.mixer_rms_epsilon = *mixer_rms_epsilon;
  metadata.time_step_min = *time_step_min;
  return metadata;
}

KernelTensorDescriptor MakeFp32Descriptor(
    const std::string& name,
    const std::vector<float>& values,
    std::vector<std::size_t> shape) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "nano-mamba-bench";
  descriptor.logical_shape = std::move(shape);
  descriptor.packed_shape = descriptor.logical_shape;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data =
      reinterpret_cast<const std::uint8_t*>(values.data());
  descriptor.packed_nbytes = values.size() * sizeof(float);
  return descriptor;
}

KernelTensorDescriptor MakeFp8Descriptor(
    const std::string& name,
    const std::vector<std::uint8_t>& bytes,
    std::vector<std::size_t> shape) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "nano-mamba-bench";
  descriptor.logical_shape = std::move(shape);
  descriptor.packed_shape = descriptor.logical_shape;
  descriptor.storage_dtype = "fp8e4m3";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = bytes.data();
  descriptor.packed_nbytes = bytes.size();
  return descriptor;
}

GemmDescriptor MakeDenseDescriptor(
    const std::string& name,
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t cols) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "nano-mamba-bench";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = rows;
  descriptor.input_cols = cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data =
      reinterpret_cast<const std::uint8_t*>(weights.data());
  descriptor.packed_nbytes = weights.size() * sizeof(float);
  return descriptor;
}

std::vector<float> MakeSequenceInputs(
    const std::vector<float>& base_input,
    std::size_t hidden_size,
    std::size_t token_count) {
  std::vector<float> sequence(token_count * hidden_size, 0.0f);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t i = 0; i < hidden_size; ++i) {
      sequence[token * hidden_size + i] =
          base_input[i] +
          (0.00125f * static_cast<float>(token)) -
          (0.00007f * static_cast<float>((i + (token * 3)) % 17));
    }
  }
  return sequence;
}

std::optional<std::uint64_t> ParseUint64Arg(const char* text) {
  if (text == nullptr || *text == '\0') {
    return std::nullopt;
  }
  try {
    return static_cast<std::uint64_t>(std::stoull(text));
  } catch (...) {
    return std::nullopt;
  }
}

bool ParseArgs(int argc, char** argv, BenchmarkOptions* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto require_value = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "nano_mamba_bench: missing value for " << flag << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--fixture-root") {
      const char* value = require_value("--fixture-root");
      if (value == nullptr) {
        return false;
      }
      options->fixture_root = value;
    } else if (arg == "--cached-prefix-tokens") {
      const char* value = require_value("--cached-prefix-tokens");
      const auto parsed = ParseUint64Arg(value);
      if (!parsed || *parsed == 0) {
        return false;
      }
      options->cached_prefix_tokens = static_cast<std::size_t>(*parsed);
    } else if (arg == "--tail-prefill-tokens") {
      const char* value = require_value("--tail-prefill-tokens");
      const auto parsed = ParseUint64Arg(value);
      if (!parsed || *parsed == 0) {
        return false;
      }
      options->tail_prefill_tokens = static_cast<std::size_t>(*parsed);
    } else if (arg == "--warmup") {
      const char* value = require_value("--warmup");
      const auto parsed = ParseUint64Arg(value);
      if (!parsed) {
        return false;
      }
      options->warmup_iterations = static_cast<std::size_t>(*parsed);
    } else if (arg == "--iterations") {
      const char* value = require_value("--iterations");
      const auto parsed = ParseUint64Arg(value);
      if (!parsed || *parsed == 0) {
        return false;
      }
      options->benchmark_iterations = static_cast<std::size_t>(*parsed);
    } else if (arg == "--json-output") {
      const char* value = require_value("--json-output");
      if (value == nullptr) {
        return false;
      }
      options->json_output_path = value;
    } else if (arg == "--help") {
      std::cout
          << "Usage: nano_mamba_bench [options]\n"
          << "  --fixture-root <path>\n"
          << "  --cached-prefix-tokens <n>\n"
          << "  --tail-prefill-tokens <n>\n"
          << "  --warmup <n>\n"
          << "  --iterations <n>\n"
          << "  --json-output <path>\n";
      return false;
    } else {
      std::cerr << "nano_mamba_bench: unknown argument: " << arg << "\n";
      return false;
    }
  }
  return true;
}

bool UploadInitialState(
    RequestExecutionContext& request,
    const std::vector<float>& initial_conv_state,
    const std::vector<float>& initial_ssm_state) {
  return request.mamba_conv_state()->CopyFromHost(
             initial_conv_state.data(),
             initial_conv_state.size()) &&
         request.mamba_state()->CopyFromHost(
             initial_ssm_state.data(),
             initial_ssm_state.size());
}

double MeasureMeanMs(
    std::size_t warmup_iterations,
    std::size_t benchmark_iterations,
    const std::function<bool()>& setup,
    const std::function<bool()>& run) {
  auto run_once = [&](double* elapsed_ms) -> bool {
    if (!setup()) {
      return false;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!run() || !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
      return false;
    }
    const auto end = std::chrono::steady_clock::now();
    *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return true;
  };

  double sink_ms = 0.0;
  for (std::size_t i = 0; i < warmup_iterations; ++i) {
    if (!run_once(&sink_ms)) {
      return -1.0;
    }
  }

  double total_ms = 0.0;
  for (std::size_t i = 0; i < benchmark_iterations; ++i) {
    double elapsed_ms = 0.0;
    if (!run_once(&elapsed_ms)) {
      return -1.0;
    }
    total_ms += elapsed_ms;
  }
  return total_ms / static_cast<double>(benchmark_iterations);
}

std::string JsonEscape(const std::string& input) {
  std::ostringstream out;
  for (char c : input) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

std::string SerializeResult(const BenchmarkResult& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"benchmark\": \"nano_mamba_bench\",\n";
  out << "  \"fixture_root\": \"" << JsonEscape(result.options.fixture_root.string()) << "\",\n";
  out << "  \"cached_prefix_tokens\": " << result.options.cached_prefix_tokens << ",\n";
  out << "  \"tail_prefill_tokens\": " << result.options.tail_prefill_tokens << ",\n";
  out << "  \"warmup_iterations\": " << result.options.warmup_iterations << ",\n";
  out << "  \"benchmark_iterations\": " << result.options.benchmark_iterations << ",\n";
  out << "  \"metadata\": {\n";
  out << "    \"layer_index\": " << result.metadata.layer_index << ",\n";
  out << "    \"hidden_size\": " << result.metadata.hidden_size << ",\n";
  out << "    \"intermediate_size\": " << result.metadata.intermediate_size << ",\n";
  out << "    \"num_heads\": " << result.metadata.num_heads << ",\n";
  out << "    \"head_dim\": " << result.metadata.head_dim << ",\n";
  out << "    \"state_size\": " << result.metadata.state_size << ",\n";
  out << "    \"n_groups\": " << result.metadata.n_groups << ",\n";
  out << "    \"conv_kernel_size\": " << result.metadata.conv_kernel_size << "\n";
  out << "  },\n";
  out << "  \"cold_prefill_mean_ms\": " << result.cold_prefill_mean_ms << ",\n";
  out << "  \"restored_prefix_prefill_mean_ms\": " << result.restored_prefix_prefill_mean_ms << ",\n";
  out << "  \"single_token_decode_mean_ms\": " << result.single_token_decode_mean_ms << ",\n";
  out << "  \"restored_prefix_excludes_restore_ms\": "
      << (result.restored_prefix_excludes_restore_ms ? "true" : "false") << "\n";
  out << "}\n";
  return out.str();
}

int RunBenchmark(const BenchmarkOptions& options) {
  if (!HasCudaDevice()) {
    std::cout << "nano_mamba_bench: SKIP (no CUDA device available)\n";
    return 0;
  }

  const auto metadata = LoadMetadata(options.fixture_root);
  if (!metadata.has_value()) {
    std::cerr << "nano_mamba_bench: failed to load fixture metadata from "
              << options.fixture_root << "\n";
    return 1;
  }
  if (!FixturePayloadPresent(
          options.fixture_root,
          {
              "input_hidden_fp32.bin",
              "initial_conv_state_fp32.bin",
              "initial_ssm_state_fp32.bin",
              "input_norm_weight_fp32.bin",
              "mixer_norm_weight_fp32.bin",
              "conv1d_weight_fp32.bin",
              "conv1d_bias_fp32.bin",
              "A_log_fp32.bin",
              "D_fp32.bin",
              "dt_bias_fp32.bin",
          })) {
    std::cerr << "nano_mamba_bench: fixture payload missing under "
              << options.fixture_root
              << " (metadata exists, binary tensors do not)\n";
    return 1;
  }

  const std::vector<float> input_hidden =
      ReadFloatFile(options.fixture_root / "input_hidden_fp32.bin");
  const std::vector<float> initial_conv_state =
      ReadFloatFile(options.fixture_root / "initial_conv_state_fp32.bin");
  const std::vector<float> initial_ssm_state =
      ReadFloatFile(options.fixture_root / "initial_ssm_state_fp32.bin");
  const std::vector<float> input_norm_weight =
      ReadFloatFile(options.fixture_root / "input_norm_weight_fp32.bin");
  const std::vector<float> mixer_norm_weight =
      ReadFloatFile(options.fixture_root / "mixer_norm_weight_fp32.bin");
  const bool has_in_proj_fp8 =
      std::filesystem::exists(options.fixture_root / "in_proj_weight_fp8.bin");
  const std::vector<std::uint8_t> in_proj_weight_fp8 =
      has_in_proj_fp8
          ? ReadFileBytes(options.fixture_root / "in_proj_weight_fp8.bin")
          : std::vector<std::uint8_t>{};
  const std::vector<float> in_proj_weight_fp32 =
      !has_in_proj_fp8
          ? ReadFloatFile(options.fixture_root / "in_proj_weight_fp32.bin")
          : std::vector<float>{};
  const std::vector<float> in_proj_weight_scale =
      has_in_proj_fp8
          ? ReadFloatFile(options.fixture_root / "in_proj_weight_scale_fp32.bin")
          : std::vector<float>{};
  const std::vector<float> in_proj_input_scale =
      has_in_proj_fp8
          ? ReadFloatFile(options.fixture_root / "in_proj_input_scale_fp32.bin")
          : std::vector<float>{};
  const std::vector<float> conv1d_weight =
      ReadFloatFile(options.fixture_root / "conv1d_weight_fp32.bin");
  const std::vector<float> conv1d_bias =
      ReadFloatFile(options.fixture_root / "conv1d_bias_fp32.bin");
  const std::vector<float> A_log =
      ReadFloatFile(options.fixture_root / "A_log_fp32.bin");
  const std::vector<float> D =
      ReadFloatFile(options.fixture_root / "D_fp32.bin");
  const std::vector<float> dt_bias =
      ReadFloatFile(options.fixture_root / "dt_bias_fp32.bin");
  const bool has_out_proj_fp8 =
      std::filesystem::exists(options.fixture_root / "out_proj_weight_fp8.bin");
  const std::vector<std::uint8_t> out_proj_weight_fp8 =
      has_out_proj_fp8
          ? ReadFileBytes(options.fixture_root / "out_proj_weight_fp8.bin")
          : std::vector<std::uint8_t>{};
  const std::vector<float> out_proj_weight_fp32 =
      !has_out_proj_fp8
          ? ReadFloatFile(options.fixture_root / "out_proj_weight_fp32.bin")
          : std::vector<float>{};
  const std::vector<float> out_proj_weight_scale =
      has_out_proj_fp8
          ? ReadFloatFile(options.fixture_root / "out_proj_weight_scale_fp32.bin")
          : std::vector<float>{};
  const std::vector<float> out_proj_input_scale =
      has_out_proj_fp8
          ? ReadFloatFile(options.fixture_root / "out_proj_input_scale_fp32.bin")
          : std::vector<float>{};

  const std::size_t conv_dim =
      metadata->intermediate_size + (2 * metadata->n_groups * metadata->state_size);
  const std::size_t conv_state_elems = conv_dim * metadata->conv_kernel_size;
  const std::size_t ssm_state_elems =
      metadata->num_heads * metadata->head_dim * metadata->state_size;
  const std::size_t in_proj_rows =
      metadata->intermediate_size + conv_dim + metadata->num_heads;
  if (input_hidden.size() != metadata->hidden_size ||
      initial_conv_state.size() != conv_state_elems ||
      initial_ssm_state.size() != ssm_state_elems) {
    std::cerr << "nano_mamba_bench: fixture tensors do not match metadata\n";
    return 1;
  }

  const std::string layer_prefix =
      "backbone.layers." + std::to_string(metadata->layer_index);
  const std::string mixer_prefix = layer_prefix + ".mixer";
  const auto input_norm_descriptor =
      MakeFp32Descriptor(layer_prefix + ".norm.weight", input_norm_weight, {metadata->hidden_size});
  const auto mixer_norm_descriptor =
      MakeFp32Descriptor(mixer_prefix + ".norm.weight", mixer_norm_weight, {metadata->intermediate_size});
  const auto in_proj_fp8_descriptor =
      has_in_proj_fp8
          ? std::make_optional(
                MakeFp8Descriptor(mixer_prefix + ".in_proj.weight", in_proj_weight_fp8, {in_proj_rows, metadata->hidden_size}))
          : std::nullopt;
  const auto in_proj_dense_descriptor =
      !has_in_proj_fp8
          ? std::make_optional(
                MakeDenseDescriptor(mixer_prefix + ".in_proj.weight", in_proj_weight_fp32, in_proj_rows, metadata->hidden_size))
          : std::nullopt;
  const auto in_proj_weight_scale_descriptor =
      has_in_proj_fp8
          ? std::make_optional(
                MakeFp32Descriptor(mixer_prefix + ".in_proj.weight_scale", in_proj_weight_scale, {1}))
          : std::nullopt;
  const auto in_proj_input_scale_descriptor =
      has_in_proj_fp8
          ? std::make_optional(
                MakeFp32Descriptor(mixer_prefix + ".in_proj.input_scale", in_proj_input_scale, {1}))
          : std::nullopt;
  const auto conv1d_weight_descriptor =
      MakeFp32Descriptor(mixer_prefix + ".conv1d.weight", conv1d_weight, {conv_dim, 1, metadata->conv_kernel_size});
  const auto conv1d_bias_descriptor =
      MakeFp32Descriptor(mixer_prefix + ".conv1d.bias", conv1d_bias, {conv_dim});
  const auto A_log_descriptor =
      MakeFp32Descriptor(mixer_prefix + ".A_log", A_log, {metadata->num_heads});
  const auto D_descriptor =
      MakeFp32Descriptor(mixer_prefix + ".D", D, {metadata->num_heads});
  const auto dt_bias_descriptor =
      MakeFp32Descriptor(mixer_prefix + ".dt_bias", dt_bias, {metadata->num_heads});
  const auto out_proj_fp8_descriptor =
      has_out_proj_fp8
          ? std::make_optional(
                MakeFp8Descriptor(mixer_prefix + ".out_proj.weight", out_proj_weight_fp8, {metadata->hidden_size, metadata->intermediate_size}))
          : std::nullopt;
  const auto out_proj_dense_descriptor =
      !has_out_proj_fp8
          ? std::make_optional(
                MakeDenseDescriptor(mixer_prefix + ".out_proj.weight", out_proj_weight_fp32, metadata->hidden_size, metadata->intermediate_size))
          : std::nullopt;
  const auto out_proj_weight_scale_descriptor =
      has_out_proj_fp8
          ? std::make_optional(
                MakeFp32Descriptor(mixer_prefix + ".out_proj.weight_scale", out_proj_weight_scale, {1}))
          : std::nullopt;
  const auto out_proj_input_scale_descriptor =
      has_out_proj_fp8
          ? std::make_optional(
                MakeFp32Descriptor(mixer_prefix + ".out_proj.input_scale", out_proj_input_scale, {1}))
          : std::nullopt;

  MambaLayerBindings bindings;
  bindings.input_norm_weight = &input_norm_descriptor;
  bindings.mixer_norm_weight = &mixer_norm_descriptor;
  bindings.in_proj_gemm_weight =
      in_proj_dense_descriptor.has_value() ? &*in_proj_dense_descriptor : nullptr;
  bindings.in_proj_kernel_weight =
      in_proj_fp8_descriptor.has_value() ? &*in_proj_fp8_descriptor : nullptr;
  bindings.in_proj_weight_scale =
      in_proj_weight_scale_descriptor.has_value() ? &*in_proj_weight_scale_descriptor : nullptr;
  bindings.in_proj_input_scale =
      in_proj_input_scale_descriptor.has_value() ? &*in_proj_input_scale_descriptor : nullptr;
  bindings.conv1d_weight = &conv1d_weight_descriptor;
  bindings.conv1d_bias = &conv1d_bias_descriptor;
  bindings.A_log = &A_log_descriptor;
  bindings.D = &D_descriptor;
  bindings.dt_bias = &dt_bias_descriptor;
  bindings.out_proj_gemm_weight =
      out_proj_dense_descriptor.has_value() ? &*out_proj_dense_descriptor : nullptr;
  bindings.out_proj_kernel_weight =
      out_proj_fp8_descriptor.has_value() ? &*out_proj_fp8_descriptor : nullptr;
  bindings.out_proj_weight_scale =
      out_proj_weight_scale_descriptor.has_value() ? &*out_proj_weight_scale_descriptor : nullptr;
  bindings.out_proj_input_scale =
      out_proj_input_scale_descriptor.has_value() ? &*out_proj_input_scale_descriptor : nullptr;

  MambaLayerConfig layer_config;
  layer_config.layer_index = metadata->layer_index;
  layer_config.hidden_size = metadata->hidden_size;
  layer_config.intermediate_size = metadata->intermediate_size;
  layer_config.num_heads = metadata->num_heads;
  layer_config.head_dim = metadata->head_dim;
  layer_config.state_size = metadata->state_size;
  layer_config.n_groups = metadata->n_groups;
  layer_config.conv_kernel_size = metadata->conv_kernel_size;
  layer_config.input_rms_epsilon = metadata->input_rms_epsilon;
  layer_config.mixer_rms_epsilon = metadata->mixer_rms_epsilon;
  layer_config.time_step_min = metadata->time_step_min;

  auto layer = MambaLayerSlice::Create(layer_config, bindings);
  auto cublas = CublasLtHandle::Create();
  if (!layer || !layer->valid() || !cublas || !cublas->valid()) {
    std::cerr << "nano_mamba_bench: failed to build layer or cuBLASLt handle\n";
    return 1;
  }

  const std::size_t full_prefill_tokens =
      options.cached_prefix_tokens + options.tail_prefill_tokens;
  const std::vector<float> sequence_inputs =
      MakeSequenceInputs(input_hidden, metadata->hidden_size, full_prefill_tokens + 1);
  const std::vector<float> full_prefill_inputs(
      sequence_inputs.begin(),
      sequence_inputs.begin() +
          static_cast<std::ptrdiff_t>(full_prefill_tokens * metadata->hidden_size));
  const std::vector<float> prefix_inputs(
      sequence_inputs.begin(),
      sequence_inputs.begin() +
          static_cast<std::ptrdiff_t>(options.cached_prefix_tokens * metadata->hidden_size));
  const std::vector<float> tail_inputs(
      sequence_inputs.begin() +
          static_cast<std::ptrdiff_t>(options.cached_prefix_tokens * metadata->hidden_size),
      sequence_inputs.begin() +
          static_cast<std::ptrdiff_t>(full_prefill_tokens * metadata->hidden_size));
  const std::vector<float> decode_input(
      sequence_inputs.begin() +
          static_cast<std::ptrdiff_t>(full_prefill_tokens * metadata->hidden_size),
      sequence_inputs.end());

  RequestExecutionConfig request_config;
  request_config.hidden_size = metadata->hidden_size;
  request_config.max_tokens = full_prefill_tokens + 1;
  request_config.scratch_tokens = full_prefill_tokens + 1;
  request_config.mamba_conv_state_bytes_fp32 = conv_state_elems * sizeof(float);
  request_config.mamba_state_bytes_fp32 = ssm_state_elems * sizeof(float);
  request_config.attention_kv_cache.layer_count = 1;
  request_config.attention_kv_cache.kv_head_count = 1;
  request_config.attention_kv_cache.head_dim = 4;
  request_config.attention_kv_cache.tokens_per_page = 4;
  request_config.attention_kv_cache.dtype = KvCacheDataType::kBf16;
  request_config.attention_total_pages =
      (full_prefill_tokens + 1 + request_config.attention_kv_cache.tokens_per_page - 1) /
      request_config.attention_kv_cache.tokens_per_page;

  auto cold_request = RequestExecutionContext::Create(request_config);
  auto restored_request = RequestExecutionContext::Create(request_config);
  auto decode_request = RequestExecutionContext::Create(request_config);
  auto snapshot_source = RequestExecutionContext::Create(request_config);
  auto decode_snapshot_source = RequestExecutionContext::Create(request_config);
  auto full_prefill_tensor =
      DeviceTensorFp32::Create({full_prefill_tokens, metadata->hidden_size});
  auto prefix_tensor =
      DeviceTensorFp32::Create({options.cached_prefix_tokens, metadata->hidden_size});
  auto tail_tensor =
      DeviceTensorFp32::Create({options.tail_prefill_tokens, metadata->hidden_size});
  auto decode_tensor = DeviceTensorFp32::Create({1, metadata->hidden_size});
  auto full_prefill_output =
      DeviceTensorFp32::Create({full_prefill_tokens, metadata->hidden_size});
  auto tail_output =
      DeviceTensorFp32::Create({options.tail_prefill_tokens, metadata->hidden_size});
  auto decode_output = DeviceTensorFp32::Create({1, metadata->hidden_size});
  if (!cold_request || !restored_request || !decode_request || !snapshot_source ||
      !decode_snapshot_source || !full_prefill_tensor || !prefix_tensor ||
      !tail_tensor || !decode_tensor || !full_prefill_output || !tail_output ||
      !decode_output) {
    std::cerr << "nano_mamba_bench: request/tensor allocation failed\n";
    return 1;
  }
  if (!full_prefill_tensor->CopyFromHost(
          full_prefill_inputs.data(),
          full_prefill_inputs.size()) ||
      !prefix_tensor->CopyFromHost(prefix_inputs.data(), prefix_inputs.size()) ||
      !tail_tensor->CopyFromHost(tail_inputs.data(), tail_inputs.size()) ||
      !decode_tensor->CopyFromHost(decode_input.data(), decode_input.size())) {
    std::cerr << "nano_mamba_bench: input upload failed\n";
    return 1;
  }

  GemmHeuristicCache heuristic_cache;

  if (!snapshot_source->SetSequenceLength(options.cached_prefix_tokens) ||
      !UploadInitialState(*snapshot_source, initial_conv_state, initial_ssm_state) ||
      !layer->Run(
          *cublas,
          &heuristic_cache,
          *snapshot_source,
          *prefix_tensor,
          full_prefill_output.get())) {
    std::cerr << "nano_mamba_bench: failed to build restored-prefix snapshot\n";
    return 1;
  }
  ReusableStateArena restore_arena(
      nemotron::RequiredKvSnapshotBytes(*snapshot_source) +
      nemotron::RequiredMambaSnapshotBytes(*snapshot_source));
  const auto restore_descriptor =
      nemotron::SnapshotRequestState(restore_arena, *snapshot_source, "nano-mamba-restored-prefix");
  if (!restore_descriptor.has_value() || !restore_descriptor->valid()) {
    std::cerr << "nano_mamba_bench: failed to snapshot restored-prefix state\n";
    return 1;
  }

  if (!decode_snapshot_source->SetSequenceLength(full_prefill_tokens) ||
      !UploadInitialState(*decode_snapshot_source, initial_conv_state, initial_ssm_state) ||
      !layer->Run(
          *cublas,
          &heuristic_cache,
          *decode_snapshot_source,
          *full_prefill_tensor,
          full_prefill_output.get())) {
    std::cerr << "nano_mamba_bench: failed to build decode snapshot\n";
    restore_arena.Release(*restore_descriptor);
    return 1;
  }
  ReusableStateArena decode_arena(
      nemotron::RequiredKvSnapshotBytes(*decode_snapshot_source) +
      nemotron::RequiredMambaSnapshotBytes(*decode_snapshot_source));
  const auto decode_descriptor =
      nemotron::SnapshotRequestState(decode_arena, *decode_snapshot_source, "nano-mamba-decode");
  if (!decode_descriptor.has_value() || !decode_descriptor->valid()) {
    std::cerr << "nano_mamba_bench: failed to snapshot decode state\n";
    restore_arena.Release(*restore_descriptor);
    return 1;
  }

  const double cold_prefill_mean_ms = MeasureMeanMs(
      options.warmup_iterations,
      options.benchmark_iterations,
      [&]() {
        return cold_request->ResetForNewRequest() &&
               UploadInitialState(*cold_request, initial_conv_state, initial_ssm_state);
      },
      [&]() {
        return layer->Run(
            *cublas,
            &heuristic_cache,
            *cold_request,
            *full_prefill_tensor,
            full_prefill_output.get());
      });

  const double restored_prefix_prefill_mean_ms = MeasureMeanMs(
      options.warmup_iterations,
      options.benchmark_iterations,
      [&]() {
        return nemotron::RestoreRequestState(
            restore_arena,
            *restore_descriptor,
            options.cached_prefix_tokens,
            *restored_request);
      },
      [&]() {
        return layer->Run(
            *cublas,
            &heuristic_cache,
            *restored_request,
            *tail_tensor,
            tail_output.get());
      });

  const double single_token_decode_mean_ms = MeasureMeanMs(
      options.warmup_iterations,
      options.benchmark_iterations,
      [&]() {
        return nemotron::RestoreRequestState(
            decode_arena,
            *decode_descriptor,
            full_prefill_tokens,
            *decode_request);
      },
      [&]() {
        return layer->Run(
            *cublas,
            &heuristic_cache,
            *decode_request,
            *decode_tensor,
            decode_output.get());
      });

  restore_arena.Release(*restore_descriptor);
  decode_arena.Release(*decode_descriptor);
  if (cold_prefill_mean_ms < 0.0 || restored_prefix_prefill_mean_ms < 0.0 ||
      single_token_decode_mean_ms < 0.0) {
    return 1;
  }

  BenchmarkResult result;
  result.options = options;
  result.metadata = *metadata;
  result.cold_prefill_mean_ms = cold_prefill_mean_ms;
  result.restored_prefix_prefill_mean_ms = restored_prefix_prefill_mean_ms;
  result.single_token_decode_mean_ms = single_token_decode_mean_ms;

  const std::string serialized = SerializeResult(result);
  std::cout << serialized;
  if (options.json_output_path.has_value()) {
    std::ofstream output(*options.json_output_path);
    output << serialized;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    return (argc > 1 && std::string(argv[1]) == "--help") ? 0 : 1;
  }
  return RunBenchmark(options);
}
