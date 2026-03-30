#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildGemmLaunchPlan;
using nemotron::CublasLtHandle;
using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceNvfp4Weight;
using nemotron::DeviceTensorFp32;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::HostNvfp4Matrix;
using nemotron::MakeNvfp4PackedMatrixDeviceView;
using nemotron::Nvfp4PackedMatrixDeviceView;
using nemotron::PackDeviceRowMajorFp32ToNvfp4;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::PrepareGemmExecution;
using nemotron::PackedFp4Bytes;
using nemotron::RowMajorNvfp4ScaleBytes;
using nemotron::RunNvfp4RowMajorFp32AccumToDevice;

constexpr std::size_t MiB(std::size_t value) {
  return value * 1024ull * 1024ull;
}

struct CheckpointShapeFamily {
  std::string name;
  std::string source_tensor;
  std::string op_class;
  std::size_t n = 0;
  std::size_t k = 0;
};

enum class ActivationStagingMode {
  kHost,
  kDevice,
};

const char* ToString(ActivationStagingMode mode) {
  switch (mode) {
    case ActivationStagingMode::kHost:
      return "host";
    case ActivationStagingMode::kDevice:
      return "device";
  }
  return "unknown";
}

struct BenchmarkOptions {
  std::vector<std::string> selected_case_names;
  std::vector<std::size_t> activation_rows = {1, 32, 256};
  std::vector<std::size_t> workspace_bytes = {0, MiB(4), MiB(16)};
  ActivationStagingMode activation_staging = ActivationStagingMode::kDevice;
  std::size_t warmup_iterations = 2;
  std::size_t hot_iterations = 8;
  std::optional<std::filesystem::path> json_output_path;
  bool list_cases_only = false;
};

struct EnvironmentInfo {
  std::string device_name;
  int device_index = 0;
  int runtime_version = 0;
  int driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::size_t total_global_mem_bytes = 0;
};

struct BenchmarkResult {
  std::string case_name;
  std::string source_tensor;
  std::string op_class;
  std::string activation_staging;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::size_t workspace_bytes = 0;
  std::size_t warmup_iterations = 0;
  std::size_t hot_iterations = 0;
  double weight_upload_ms = 0.0;
  double activation_pack_ms = 0.0;
  double activation_upload_ms = 0.0;
  double prepare_cache_miss_ms = 0.0;
  double prepare_cache_hit_ms = 0.0;
  bool prepare_first_from_cache = false;
  bool prepare_second_from_cache = false;
  std::size_t heuristic_cache_size = 0;
  double cold_first_call_ms = 0.0;
  double hot_mean_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  double hot_tflops = 0.0;
  int heuristic_count = 0;
  std::size_t workspace_used_bytes = 0;
};

const std::vector<CheckpointShapeFamily>& DefaultCases() {
  static const std::vector<CheckpointShapeFamily> kCases = {
      {
          "attention_core_proj",
          "backbone.layers.36.mixer.o_proj.weight",
          "attention_projection",
          4096,
          4096,
      },
      {
          "shared_expert_up",
          "backbone.layers.59.mixer.shared_experts.up_proj.weight",
          "shared_expert",
          5376,
          4096,
      },
      {
          "shared_expert_down",
          "backbone.layers.59.mixer.shared_experts.down_proj.weight",
          "shared_expert",
          4096,
          5376,
      },
      {
          "mtp_expert_up",
          "mtp.layers.1.mixer.experts.100.up_proj.weight",
          "mtp",
          2688,
          1024,
      },
      {
          "mtp_expert_down",
          "mtp.layers.1.mixer.experts.100.down_proj.weight",
          "mtp",
          1024,
          2688,
      },
  };
  return kCases;
}

class AlignedByteBuffer {
 public:
  explicit AlignedByteBuffer(std::size_t nbytes, std::size_t alignment_bytes = 64)
      : nbytes_(nbytes) {
    const std::size_t padded_bytes =
        ((nbytes_ + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment_bytes, padded_bytes) != 0) {
      return;
    }
    data_.reset(reinterpret_cast<std::uint8_t*>(ptr));
  }

  bool valid() const {
    return data_ != nullptr && nbytes_ > 0;
  }

  std::uint8_t* data() const {
    return data_.get();
  }

  std::size_t nbytes() const {
    return nbytes_;
  }

 private:
  struct FreeDeleter {
    void operator()(std::uint8_t* ptr) const {
      std::free(ptr);
    }
  };

  std::size_t nbytes_ = 0;
  std::unique_ptr<std::uint8_t, FreeDeleter> data_;
};

class AlignedFloatBuffer {
 public:
  explicit AlignedFloatBuffer(std::size_t numel, std::size_t alignment_bytes = 64)
      : numel_(numel) {
    const std::size_t bytes = numel_ * sizeof(float);
    const std::size_t padded_bytes =
        ((bytes + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment_bytes, padded_bytes) != 0) {
      return;
    }
    data_.reset(reinterpret_cast<float*>(ptr));
  }

  bool valid() const {
    return data_ != nullptr && numel_ > 0;
  }

  float* data() const {
    return data_.get();
  }

  std::size_t numel() const {
    return numel_;
  }

 private:
  struct FreeDeleter {
    void operator()(float* ptr) const {
      std::free(ptr);
    }
  };

  std::size_t numel_ = 0;
  std::unique_ptr<float, FreeDeleter> data_;
};

std::string EscapeJson(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

bool ParseCsvUnsigned(std::string_view text, std::vector<std::size_t>* values, bool allow_zero = false) {
  if (values == nullptr || text.empty()) {
    return false;
  }

  std::vector<std::size_t> parsed;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    const std::string token(text.substr(begin, end - begin));
    if (token.empty()) {
      return false;
    }
    std::size_t value = 0;
    try {
      value = std::stoull(token);
    } catch (...) {
      return false;
    }
    if (value == 0 && !allow_zero) {
      return false;
    }
    parsed.push_back(value);
    begin = end + 1;
  }

  if (parsed.empty()) {
    return false;
  }
  *values = std::move(parsed);
  return true;
}

const CheckpointShapeFamily* FindCaseByName(std::string_view name) {
  for (const auto& shape : DefaultCases()) {
    if (shape.name == name) {
      return &shape;
    }
  }
  return nullptr;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --list-cases\n"
      << "  --case <name>                Repeat to select specific checkpoint-derived cases\n"
      << "  --rows <csv>                 Activation row counts, default: 1,32,256\n"
      << "  --workspace-bytes <csv>      Workspace sweep, default: 0,4194304,16777216\n"
      << "  --activation-staging <mode>  host or device, default: device\n"
      << "  --warmup <count>             Warmup iterations, default: 2\n"
      << "  --iterations <count>         Hot timed iterations, default: 8\n"
      << "  --json-output <path>         Write JSON results to a file\n";
}

bool ParseArgs(int argc, char** argv, BenchmarkOptions* options) {
  if (options == nullptr) {
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--list-cases") {
      options->list_cases_only = true;
      continue;
    }
    if (arg == "--case" && i + 1 < argc) {
      options->selected_case_names.push_back(argv[++i]);
      continue;
    }
    if (arg == "--rows" && i + 1 < argc) {
      if (!ParseCsvUnsigned(argv[++i], &options->activation_rows)) {
        return false;
      }
      continue;
    }
    if (arg == "--workspace-bytes" && i + 1 < argc) {
      if (!ParseCsvUnsigned(argv[++i], &options->workspace_bytes, true)) {
        return false;
      }
      continue;
    }
    if (arg == "--activation-staging" && i + 1 < argc) {
      const std::string mode = argv[++i];
      if (mode == "host") {
        options->activation_staging = ActivationStagingMode::kHost;
      } else if (mode == "device") {
        options->activation_staging = ActivationStagingMode::kDevice;
      } else {
        return false;
      }
      continue;
    }
    if (arg == "--warmup" && i + 1 < argc) {
      try {
        options->warmup_iterations = std::stoull(argv[++i]);
      } catch (...) {
        return false;
      }
      continue;
    }
    if (arg == "--iterations" && i + 1 < argc) {
      try {
        options->hot_iterations = std::stoull(argv[++i]);
      } catch (...) {
        return false;
      }
      if (options->hot_iterations == 0) {
        return false;
      }
      continue;
    }
    if (arg == "--json-output" && i + 1 < argc) {
      options->json_output_path = std::filesystem::path(argv[++i]);
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    return false;
  }

  return true;
}

std::vector<const CheckpointShapeFamily*> ResolveSelectedCases(const BenchmarkOptions& options) {
  std::vector<const CheckpointShapeFamily*> selected;
  if (options.selected_case_names.empty()) {
    for (const auto& shape : DefaultCases()) {
      selected.push_back(&shape);
    }
    return selected;
  }

  for (const std::string& name : options.selected_case_names) {
    if (const auto* shape = FindCaseByName(name)) {
      selected.push_back(shape);
    }
  }
  return selected;
}

void FillDeterministicBytes(std::uint8_t* data, std::size_t nbytes, std::uint8_t seed) {
  for (std::size_t i = 0; i < nbytes; ++i) {
    data[i] = static_cast<std::uint8_t>((seed + (i * 17u)) & 0xffu);
  }
}

void FillDeterministicFloats(float* data, std::size_t numel, std::uint32_t seed) {
  for (std::size_t i = 0; i < numel; ++i) {
    const std::uint32_t value =
        static_cast<std::uint32_t>((seed + (i * 37u) + ((i % 7u) * 19u)) % 257u);
    const float centered = static_cast<float>(static_cast<int>(value) - 128);
    data[i] = centered / 32.0f;
  }
}

void WriteFloat(std::uint8_t* dst, float value) {
  std::memcpy(dst, &value, sizeof(value));
}

GemmDescriptor MakeNvfp4Descriptor(
    const std::string& tensor_name,
    const std::string& op_class,
    std::size_t rows,
    std::size_t cols,
    const AlignedByteBuffer& packed,
    const AlignedByteBuffer& block_scales,
    const AlignedByteBuffer& tensor_scale) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = tensor_name;
  descriptor.op_class = op_class;
  descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = rows;
  descriptor.input_cols = cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed.data();
  descriptor.packed_nbytes = packed.nbytes();
  descriptor.block_scales_data = block_scales.data();
  descriptor.block_scales_nbytes = block_scales.nbytes();
  descriptor.tensor_scale_data = tensor_scale.data();
  descriptor.tensor_scale_nbytes = tensor_scale.nbytes();
  return descriptor;
}

GemmDescriptor MakeNvfp4Descriptor(
    const std::string& tensor_name,
    const std::string& op_class,
    const HostNvfp4Matrix& packed_matrix) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = tensor_name;
  descriptor.op_class = op_class;
  descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = packed_matrix.rows;
  descriptor.input_cols = packed_matrix.cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed_matrix.packed_data();
  descriptor.packed_nbytes = packed_matrix.packed_nbytes();
  descriptor.block_scales_data = packed_matrix.block_scales_data();
  descriptor.block_scales_nbytes = packed_matrix.block_scales_nbytes();
  descriptor.tensor_scale_data = packed_matrix.tensor_scale_data();
  descriptor.tensor_scale_nbytes = packed_matrix.tensor_scale_nbytes();
  return descriptor;
}

bool ReadEnvironmentInfo(EnvironmentInfo* info) {
  if (info == nullptr) {
    return false;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return false;
  }

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    return false;
  }
  info->device_name = props.name;
  info->device_index = 0;
  info->compute_capability_major = props.major;
  info->compute_capability_minor = props.minor;
  info->total_global_mem_bytes = props.totalGlobalMem;
  cudaRuntimeGetVersion(&info->runtime_version);
  cudaDriverGetVersion(&info->driver_version);
  return true;
}

double ElapsedMilliseconds(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::optional<BenchmarkResult> RunBenchmarkCase(
    const CheckpointShapeFamily& shape,
    std::size_t activation_rows,
    std::size_t workspace_bytes,
    const BenchmarkOptions& options) {
  AlignedByteBuffer weight_packed(PackedFp4Bytes(shape.n, shape.k));
  AlignedByteBuffer weight_block_scales(RowMajorNvfp4ScaleBytes(shape.n, shape.k));
  AlignedByteBuffer weight_tensor_scale(sizeof(float));
  AlignedFloatBuffer host_activations(activation_rows * shape.k);
  if (!weight_packed.valid() ||
      !weight_block_scales.valid() ||
      !weight_tensor_scale.valid() ||
      !host_activations.valid()) {
    return std::nullopt;
  }

  FillDeterministicBytes(weight_packed.data(), weight_packed.nbytes(), 0x11u);
  FillDeterministicBytes(weight_block_scales.data(), weight_block_scales.nbytes(), 0x38u);
  WriteFloat(weight_tensor_scale.data(), 1.0f);
  FillDeterministicFloats(host_activations.data(), host_activations.numel(), 0x22u);

  const GemmDescriptor weight_descriptor = MakeNvfp4Descriptor(
      shape.source_tensor,
      shape.op_class,
      shape.n,
      shape.k,
      weight_packed,
      weight_block_scales,
      weight_tensor_scale);
  const auto launch_plan = BuildGemmLaunchPlan(weight_descriptor, activation_rows);
  if (!launch_plan.has_value()) {
    return std::nullopt;
  }

  GemmHeuristicCache heuristic_cache;
  const auto prepare_miss_begin = std::chrono::steady_clock::now();
  const auto execution_miss = PrepareGemmExecution(*launch_plan, &heuristic_cache);
  const auto prepare_miss_end = std::chrono::steady_clock::now();
  if (!execution_miss.has_value()) {
    return std::nullopt;
  }

  const auto prepare_hit_begin = std::chrono::steady_clock::now();
  const auto execution_hit = PrepareGemmExecution(*launch_plan, &heuristic_cache);
  const auto prepare_hit_end = std::chrono::steady_clock::now();
  if (!execution_hit.has_value()) {
    return std::nullopt;
  }

  const auto cublas_plan = BuildCublasLtGemmPlan(*execution_hit);
  if (!cublas_plan.has_value()) {
    return std::nullopt;
  }

  const auto handle = CublasLtHandle::Create(workspace_bytes);
  if (!handle || !handle->valid()) {
    return std::nullopt;
  }

  const auto weight_upload_begin = std::chrono::steady_clock::now();
  auto device_weight = DeviceNvfp4Weight::Upload(weight_descriptor);
  const auto weight_upload_end = std::chrono::steady_clock::now();
  if (!device_weight || !device_weight->valid()) {
    return std::nullopt;
  }

  std::optional<HostNvfp4Matrix> packed_activation_host;
  std::unique_ptr<DeviceNvfp4Weight> device_activation_host;
  std::unique_ptr<DeviceTensorFp32> device_activation_source;
  std::unique_ptr<DeviceNvfp4Matrix> device_activation_packed;
  double activation_pack_ms = 0.0;
  double activation_upload_ms = 0.0;
  Nvfp4PackedMatrixDeviceView activation_view;

  if (options.activation_staging == ActivationStagingMode::kHost) {
    const auto activation_pack_begin = std::chrono::steady_clock::now();
    packed_activation_host =
        PackRowMajorFp32ToNvfp4(host_activations.data(), activation_rows, shape.k);
    const auto activation_pack_end = std::chrono::steady_clock::now();
    if (!packed_activation_host.has_value() || !packed_activation_host->valid()) {
      return std::nullopt;
    }

    const GemmDescriptor activation_descriptor = MakeNvfp4Descriptor(
        "activation." + shape.name,
        "activation",
        *packed_activation_host);

    const auto activation_upload_begin = std::chrono::steady_clock::now();
    device_activation_host = DeviceNvfp4Weight::Upload(activation_descriptor);
    const auto activation_upload_end = std::chrono::steady_clock::now();
    if (!device_activation_host || !device_activation_host->valid()) {
      return std::nullopt;
    }

    activation_pack_ms = ElapsedMilliseconds(activation_pack_begin, activation_pack_end);
    activation_upload_ms = ElapsedMilliseconds(activation_upload_begin, activation_upload_end);
    activation_view = MakeNvfp4PackedMatrixDeviceView(*device_activation_host);
  } else {
    device_activation_source = DeviceTensorFp32::Create({activation_rows, shape.k});
    if (!device_activation_source || !device_activation_source->valid()) {
      return std::nullopt;
    }
    if (!device_activation_source->CopyFromHost(host_activations.data(), host_activations.numel())) {
      return std::nullopt;
    }

    const auto activation_pack_begin = std::chrono::steady_clock::now();
    device_activation_packed = PackDeviceRowMajorFp32ToNvfp4(*device_activation_source);
    const auto activation_pack_end = std::chrono::steady_clock::now();
    if (!device_activation_packed || !device_activation_packed->valid()) {
      return std::nullopt;
    }

    activation_pack_ms = ElapsedMilliseconds(activation_pack_begin, activation_pack_end);
    activation_upload_ms = 0.0;
    activation_view = MakeNvfp4PackedMatrixDeviceView(*device_activation_packed);
  }

  auto output = DeviceTensorFp32::Create({activation_rows, shape.n});
  if (!output || !output->valid()) {
    return std::nullopt;
  }

  for (std::size_t i = 0; i < options.warmup_iterations; ++i) {
    const auto warmup = RunNvfp4RowMajorFp32AccumToDevice(
        *handle,
        *cublas_plan,
        activation_view,
        *device_weight,
        output.get());
    if (!warmup.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_begin = std::chrono::steady_clock::now();
  const auto cold_stats = RunNvfp4RowMajorFp32AccumToDevice(
      *handle,
      *cublas_plan,
      activation_view,
      *device_weight,
      output.get());
  const auto cold_end = std::chrono::steady_clock::now();
  if (!cold_stats.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = std::numeric_limits<double>::max();
  double hot_max_ms = 0.0;
  for (std::size_t i = 0; i < options.hot_iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    const auto stats = RunNvfp4RowMajorFp32AccumToDevice(
        *handle,
        *cublas_plan,
        activation_view,
        *device_weight,
        output.get());
    const auto end = std::chrono::steady_clock::now();
    if (!stats.has_value()) {
      return std::nullopt;
    }
    const double ms = ElapsedMilliseconds(begin, end);
    hot_sum_ms += ms;
    hot_min_ms = std::min(hot_min_ms, ms);
    hot_max_ms = std::max(hot_max_ms, ms);
  }

  const double hot_mean_ms = hot_sum_ms / static_cast<double>(options.hot_iterations);
  const double flop_count =
      2.0 * static_cast<double>(activation_rows) * static_cast<double>(shape.n) * static_cast<double>(shape.k);
  const double hot_tflops = hot_mean_ms > 0.0 ? flop_count / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;

  BenchmarkResult result;
  result.case_name = shape.name;
  result.source_tensor = shape.source_tensor;
  result.op_class = shape.op_class;
  result.activation_staging = ToString(options.activation_staging);
  result.m = activation_rows;
  result.n = shape.n;
  result.k = shape.k;
  result.workspace_bytes = workspace_bytes;
  result.warmup_iterations = options.warmup_iterations;
  result.hot_iterations = options.hot_iterations;
  result.weight_upload_ms = ElapsedMilliseconds(weight_upload_begin, weight_upload_end);
  result.activation_pack_ms = activation_pack_ms;
  result.activation_upload_ms = activation_upload_ms;
  result.prepare_cache_miss_ms = ElapsedMilliseconds(prepare_miss_begin, prepare_miss_end);
  result.prepare_cache_hit_ms = ElapsedMilliseconds(prepare_hit_begin, prepare_hit_end);
  result.prepare_first_from_cache = execution_miss->algorithm_from_cache;
  result.prepare_second_from_cache = execution_hit->algorithm_from_cache;
  result.heuristic_cache_size = heuristic_cache.size();
  result.cold_first_call_ms = ElapsedMilliseconds(cold_begin, cold_end);
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops = hot_tflops;
  result.heuristic_count = cold_stats->heuristic_count;
  result.workspace_used_bytes = cold_stats->workspace_bytes;
  return result;
}

void PrintCases() {
  std::cout << "Checkpoint-derived default NVFP4 GEMM cases:\n";
  for (const auto& shape : DefaultCases()) {
    std::cout << "  " << shape.name
              << "  n=" << shape.n
              << " k=" << shape.k
              << "  source=" << shape.source_tensor
              << "\n";
  }
}

void PrintEnvironmentSummary(const EnvironmentInfo& info, const BenchmarkOptions& options) {
  std::cout << "GB10 NVFP4 GEMM benchmark\n";
  std::cout << "device=" << info.device_name
            << " cc=" << info.compute_capability_major << "." << info.compute_capability_minor
            << " runtime=" << info.runtime_version
            << " driver=" << info.driver_version
            << "\n";
  std::cout << "rows=";
  for (std::size_t i = 0; i < options.activation_rows.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << options.activation_rows[i];
  }
  std::cout << " workspace_bytes=";
  for (std::size_t i = 0; i < options.workspace_bytes.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << options.workspace_bytes[i];
  }
  std::cout << " warmup=" << options.warmup_iterations
            << " activation_staging=" << ToString(options.activation_staging)
            << " iterations=" << options.hot_iterations
            << "\n";
}

bool WriteJsonReport(
    const std::filesystem::path& path,
    const EnvironmentInfo& info,
    const BenchmarkOptions& options,
    const std::vector<BenchmarkResult>& results) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output) {
    return false;
  }

  output << "{\n";
  output << "  \"benchmark\": \"gb10_nvfp4_gemm_bench\",\n";
  output << "  \"device\": {\n";
  output << "    \"name\": \"" << EscapeJson(info.device_name) << "\",\n";
  output << "    \"index\": " << info.device_index << ",\n";
  output << "    \"compute_capability\": \"" << info.compute_capability_major << "." << info.compute_capability_minor
         << "\",\n";
  output << "    \"cuda_runtime_version\": " << info.runtime_version << ",\n";
  output << "    \"cuda_driver_version\": " << info.driver_version << ",\n";
  output << "    \"total_global_mem_bytes\": " << info.total_global_mem_bytes << "\n";
  output << "  },\n";
  output << "  \"options\": {\n";
  output << "    \"warmup_iterations\": " << options.warmup_iterations << ",\n";
  output << "    \"hot_iterations\": " << options.hot_iterations << ",\n";
  output << "    \"activation_staging\": \"" << ToString(options.activation_staging) << "\",\n";
  output << "    \"activation_rows\": [";
  for (std::size_t i = 0; i < options.activation_rows.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << options.activation_rows[i];
  }
  output << "],\n";
  output << "    \"workspace_bytes\": [";
  for (std::size_t i = 0; i < options.workspace_bytes.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << options.workspace_bytes[i];
  }
  output << "]\n";
  output << "  },\n";
  output << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    output << "    {\n";
    output << "      \"case_name\": \"" << EscapeJson(result.case_name) << "\",\n";
    output << "      \"source_tensor\": \"" << EscapeJson(result.source_tensor) << "\",\n";
    output << "      \"op_class\": \"" << EscapeJson(result.op_class) << "\",\n";
    output << "      \"activation_staging\": \"" << EscapeJson(result.activation_staging) << "\",\n";
    output << "      \"m\": " << result.m << ",\n";
    output << "      \"n\": " << result.n << ",\n";
    output << "      \"k\": " << result.k << ",\n";
    output << "      \"workspace_bytes\": " << result.workspace_bytes << ",\n";
    output << "      \"weight_upload_ms\": " << std::fixed << std::setprecision(6) << result.weight_upload_ms << ",\n";
    output << "      \"activation_pack_ms\": " << result.activation_pack_ms << ",\n";
    output << "      \"activation_upload_ms\": " << result.activation_upload_ms << ",\n";
    output << "      \"prepare_cache_miss_ms\": " << result.prepare_cache_miss_ms << ",\n";
    output << "      \"prepare_cache_hit_ms\": " << result.prepare_cache_hit_ms << ",\n";
    output << "      \"prepare_first_from_cache\": " << (result.prepare_first_from_cache ? "true" : "false") << ",\n";
    output << "      \"prepare_second_from_cache\": " << (result.prepare_second_from_cache ? "true" : "false") << ",\n";
    output << "      \"heuristic_cache_size\": " << result.heuristic_cache_size << ",\n";
    output << "      \"cold_first_call_ms\": " << result.cold_first_call_ms << ",\n";
    output << "      \"hot_mean_ms\": " << result.hot_mean_ms << ",\n";
    output << "      \"hot_min_ms\": " << result.hot_min_ms << ",\n";
    output << "      \"hot_max_ms\": " << result.hot_max_ms << ",\n";
    output << "      \"hot_tflops\": " << result.hot_tflops << ",\n";
    output << "      \"heuristic_count\": " << result.heuristic_count << ",\n";
    output << "      \"workspace_used_bytes\": " << result.workspace_used_bytes << "\n";
    output << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  output << "  ]\n";
  output << "}\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (options.list_cases_only) {
    PrintCases();
    return 0;
  }

  const auto selected_cases = ResolveSelectedCases(options);
  if (selected_cases.empty()) {
    std::cerr << "No benchmark cases selected.\n";
    return 1;
  }

  EnvironmentInfo info;
  if (!ReadEnvironmentInfo(&info)) {
    std::cerr << "No CUDA device available for NVFP4 benchmark.\n";
    return 1;
  }

  PrintEnvironmentSummary(info, options);

  std::vector<BenchmarkResult> results;
  for (const auto* shape : selected_cases) {
    for (std::size_t rows : options.activation_rows) {
      for (std::size_t workspace_bytes : options.workspace_bytes) {
        const auto result = RunBenchmarkCase(*shape, rows, workspace_bytes, options);
        if (!result.has_value()) {
          std::cerr << "Failed to benchmark case=" << shape->name
                    << " rows=" << rows
                    << " workspace=" << workspace_bytes
                    << "\n";
          return 1;
        }

        std::cout << "case=" << result->case_name
                  << " m=" << result->m
                  << " n=" << result->n
                  << " k=" << result->k
                  << " activation_staging=" << result->activation_staging
                  << " workspace=" << result->workspace_bytes
                  << " weight_upload_ms=" << std::fixed << std::setprecision(3) << result->weight_upload_ms
                  << " activation_pack_ms=" << result->activation_pack_ms
                  << " activation_upload_ms=" << result->activation_upload_ms
                  << " cold_ms=" << result->cold_first_call_ms
                  << " hot_mean_ms=" << result->hot_mean_ms
                  << " hot_tflops=" << result->hot_tflops
                  << " prepare_cache_hit=" << (result->prepare_second_from_cache ? "true" : "false")
                  << "\n";

        results.push_back(*result);
      }
    }
  }

  if (options.json_output_path.has_value()) {
    if (!WriteJsonReport(*options.json_output_path, info, options, results)) {
      std::cerr << "Failed to write JSON report.\n";
      return 1;
    }
  }

  return 0;
}
