#include "nemotron/cudnn_handle.h"
#include "nemotron/cudnn_paged_attention.h"
#include "nemotron/paged_attention_plan.h"
#include "nemotron/paged_kv_cache.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::AttentionSequencePages;
using nemotron::BuildCudnnPagedAttentionConfig;
using nemotron::BuildPagedAttentionBatchPlan;
using nemotron::CudnnHandle;
using nemotron::CudnnPagedAttentionExecution;
using nemotron::CudnnPagedAttentionPlan;
using nemotron::KvCacheDataType;
using nemotron::PagedKvCacheArena;

struct BenchmarkCase {
  std::string name;
  std::string phase;
  std::size_t batch_size = 0;
  std::size_t query_tokens = 0;
  std::size_t kv_tokens = 0;
  bool causal = false;
};

struct BenchmarkOptions {
  std::vector<std::string> selected_case_names;
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
  long long cudnn_version = 0;
  std::size_t total_global_mem_bytes = 0;
};

struct BenchmarkResult {
  std::string case_name;
  std::string phase;
  std::size_t batch_size = 0;
  std::size_t query_tokens = 0;
  std::size_t kv_tokens = 0;
  bool causal = false;
  std::size_t query_head_count = 0;
  std::size_t kv_head_count = 0;
  std::size_t head_dim = 0;
  std::size_t tokens_per_page = 0;
  std::size_t page_table_entries = 0;
  std::size_t container_page_count = 0;
  std::size_t plan_workspace_bytes = 0;
  std::size_t warmup_iterations = 0;
  std::size_t hot_iterations = 0;
  double plan_build_ms = 0.0;
  double input_upload_ms = 0.0;
  double cold_first_call_ms = 0.0;
  double hot_mean_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  double batch_query_tokens_per_second = 0.0;
};

constexpr std::size_t kModelLayerCount = 8;
constexpr std::size_t kQueryHeadCount = 32;
constexpr std::size_t kKvHeadCount = 2;
constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kTokensPerPage = 64;

const std::vector<BenchmarkCase>& DefaultCases() {
  static const std::vector<BenchmarkCase> kCases = {
      {"decode_b1_kv64k", "decode", 1, 1, 64 * 1024, true},
      {"decode_b8_kv64k", "decode", 8, 1, 64 * 1024, true},
      {"prefill_b1_q256_kv8k", "prefill", 1, 256, 8 * 1024, true},
      {"prefill_b1_q512_kv32k", "prefill", 1, 512, 32 * 1024, true},
  };
  return kCases;
}

template <typename T>
class DeviceBuffer {
 public:
  static std::optional<DeviceBuffer> Create(std::size_t count) {
    if (count == 0) {
      return std::nullopt;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      return std::nullopt;
    }

    DeviceBuffer buffer;
    buffer.count_ = count;
    if (cudaMalloc(reinterpret_cast<void**>(&buffer.data_), count * sizeof(T)) != cudaSuccess) {
      return std::nullopt;
    }
    return buffer;
  }

  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = other.data_;
    count_ = other.count_;
    other.data_ = nullptr;
    other.count_ = 0;
    return *this;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const {
    return data_;
  }

  std::size_t count() const {
    return count_;
  }

  bool CopyFromHost(const std::vector<T>& host_values) {
    if (host_values.size() != count_) {
      return false;
    }
    return cudaMemcpy(
               data_,
               host_values.data(),
               count_ * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
  }

 private:
  DeviceBuffer() = default;

  T* data_ = nullptr;
  std::size_t count_ = 0;
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

const BenchmarkCase* FindCaseByName(std::string_view name) {
  for (const auto& benchmark_case : DefaultCases()) {
    if (benchmark_case.name == name) {
      return &benchmark_case;
    }
  }
  return nullptr;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --list-cases\n"
      << "  --case <name>                Repeat to select specific benchmark cases\n"
      << "  --warmup <count>             Warmup iterations, default: 2\n"
      << "  --iterations <count>         Hot timed iterations, default: 8\n"
      << "  --json-output <path>         Write JSON results to a file\n";
}

bool ParseArgs(int argc, char** argv, BenchmarkOptions* options) {
  if (options == nullptr) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--list-cases") {
      options->list_cases_only = true;
    } else if (arg == "--case") {
      if (i + 1 >= argc) {
        return false;
      }
      options->selected_case_names.emplace_back(argv[++i]);
    } else if (arg == "--warmup") {
      if (i + 1 >= argc) {
        return false;
      }
      try {
        options->warmup_iterations = std::stoull(argv[++i]);
      } catch (...) {
        return false;
      }
    } else if (arg == "--iterations") {
      if (i + 1 >= argc) {
        return false;
      }
      try {
        options->hot_iterations = std::stoull(argv[++i]);
      } catch (...) {
        return false;
      }
    } else if (arg == "--json-output") {
      if (i + 1 >= argc) {
        return false;
      }
      options->json_output_path = std::filesystem::path(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  return options->hot_iterations > 0;
}

EnvironmentInfo CollectEnvironmentInfo(const CudnnHandle& handle) {
  EnvironmentInfo info;
  int device = 0;
  cudaGetDevice(&device);
  info.device_index = device;

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    info.device_name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_global_mem_bytes = prop.totalGlobalMem;
  }

  cudaRuntimeGetVersion(&info.runtime_version);
  cudaDriverGetVersion(&info.driver_version);
  info.cudnn_version = handle.version();
  return info;
}

void PrintCaseList() {
  std::cout << "Available benchmark cases:\n";
  for (const auto& benchmark_case : DefaultCases()) {
    std::cout << "  " << benchmark_case.name
              << "  phase=" << benchmark_case.phase
              << " batch=" << benchmark_case.batch_size
              << " q=" << benchmark_case.query_tokens
              << " kv=" << benchmark_case.kv_tokens
              << " causal=" << (benchmark_case.causal ? "true" : "false")
              << "\n";
  }
}

std::vector<const BenchmarkCase*> SelectCases(const BenchmarkOptions& options) {
  std::vector<const BenchmarkCase*> selected;
  if (options.selected_case_names.empty()) {
    for (const auto& benchmark_case : DefaultCases()) {
      selected.push_back(&benchmark_case);
    }
    return selected;
  }

  for (const auto& name : options.selected_case_names) {
    const BenchmarkCase* found = FindCaseByName(name);
    if (found != nullptr) {
      selected.push_back(found);
    }
  }
  return selected;
}

std::vector<float> MakeDeterministicFloatData(std::size_t count, float scale, float freq) {
  std::vector<float> values(count, 0.0f);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = scale * std::sin(static_cast<float>(i + 1) * freq);
  }
  return values;
}

std::vector<__nv_bfloat16> ToBf16(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = __float2bfloat16(values[i]);
  }
  return converted;
}

std::optional<BenchmarkResult> RunCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    const CudnnHandle& handle) {
  AttentionKvCacheConfig cache_config;
  cache_config.layer_count = kModelLayerCount;
  cache_config.kv_head_count = kKvHeadCount;
  cache_config.head_dim = kHeadDim;
  cache_config.tokens_per_page = kTokensPerPage;
  cache_config.dtype = KvCacheDataType::kBf16;

  const std::size_t pages_per_sequence =
      (benchmark_case.kv_tokens + cache_config.tokens_per_page - 1) / cache_config.tokens_per_page;
  const std::size_t total_pages = benchmark_case.batch_size * pages_per_sequence;

  auto arena = PagedKvCacheArena::Create(cache_config, total_pages);
  if (!arena.has_value()) {
    return std::nullopt;
  }

  std::vector<AttentionSequencePages> sequences;
  sequences.reserve(benchmark_case.batch_size);
  for (std::size_t batch = 0; batch < benchmark_case.batch_size; ++batch) {
    auto pages = arena->AllocatePages(0, pages_per_sequence);
    if (!pages.has_value()) {
      return std::nullopt;
    }
    AttentionSequencePages sequence;
    sequence.token_count = benchmark_case.kv_tokens;
    sequence.page_ids.reserve(pages_per_sequence);
    for (const auto& page : *pages) {
      sequence.page_ids.push_back(page.page_id);
    }
    sequences.push_back(std::move(sequence));
  }

  const auto kv_plan = BuildPagedAttentionBatchPlan(cache_config, 0, sequences, &*arena);
  if (!kv_plan.has_value()) {
    return std::nullopt;
  }

  const auto config = BuildCudnnPagedAttentionConfig(
      *kv_plan,
      kQueryHeadCount,
      benchmark_case.query_tokens,
      arena->total_pages(),
      0.0f,
      benchmark_case.causal,
      false);
  if (!config.has_value()) {
    return std::nullopt;
  }

  const auto build_start = std::chrono::high_resolution_clock::now();
  auto plan = CudnnPagedAttentionPlan::Create(handle, *config);
  const auto build_end = std::chrono::high_resolution_clock::now();
  if (!plan || !plan->valid()) {
    return std::nullopt;
  }

  const std::size_t query_count =
      config->batch_size * config->query_head_count * config->max_query_tokens * cache_config.head_dim;
  const std::size_t kv_count =
      config->container_page_count * cache_config.kv_head_count * cache_config.tokens_per_page * cache_config.head_dim;
  const std::size_t page_table_count = kv_plan->page_table.size();

  auto query_dev = DeviceBuffer<__nv_bfloat16>::Create(query_count);
  auto key_dev = DeviceBuffer<__nv_bfloat16>::Create(kv_count);
  auto value_dev = DeviceBuffer<__nv_bfloat16>::Create(kv_count);
  auto output_dev = DeviceBuffer<__nv_bfloat16>::Create(query_count);
  auto seq_len_q_dev = DeviceBuffer<std::int32_t>::Create(config->batch_size);
  auto seq_len_kv_dev = DeviceBuffer<std::int32_t>::Create(config->batch_size);
  auto page_table_k_dev = DeviceBuffer<std::int32_t>::Create(page_table_count);
  auto page_table_v_dev = DeviceBuffer<std::int32_t>::Create(page_table_count);
  if (!query_dev.has_value() ||
      !key_dev.has_value() ||
      !value_dev.has_value() ||
      !output_dev.has_value() ||
      !seq_len_q_dev.has_value() ||
      !seq_len_kv_dev.has_value() ||
      !page_table_k_dev.has_value() ||
      !page_table_v_dev.has_value()) {
    return std::nullopt;
  }

  const auto upload_start = std::chrono::high_resolution_clock::now();
  const bool uploaded =
      query_dev->CopyFromHost(ToBf16(MakeDeterministicFloatData(query_count, 0.25f, 0.013f))) &&
      key_dev->CopyFromHost(ToBf16(MakeDeterministicFloatData(kv_count, 0.20f, 0.017f))) &&
      value_dev->CopyFromHost(ToBf16(MakeDeterministicFloatData(kv_count, 0.15f, 0.019f))) &&
      output_dev->CopyFromHost(std::vector<__nv_bfloat16>(query_count, __float2bfloat16(0.0f))) &&
      seq_len_q_dev->CopyFromHost(std::vector<std::int32_t>(config->batch_size, static_cast<std::int32_t>(benchmark_case.query_tokens))) &&
      seq_len_kv_dev->CopyFromHost(kv_plan->sequence_lengths) &&
      page_table_k_dev->CopyFromHost(kv_plan->page_table) &&
      page_table_v_dev->CopyFromHost(kv_plan->page_table);
  const auto upload_end = std::chrono::high_resolution_clock::now();
  if (!uploaded) {
    return std::nullopt;
  }

  const CudnnPagedAttentionExecution execution = {
      query_dev->data(),
      key_dev->data(),
      value_dev->data(),
      seq_len_q_dev->data(),
      seq_len_kv_dev->data(),
      page_table_k_dev->data(),
      page_table_v_dev->data(),
      output_dev->data(),
      nullptr,
  };

  for (std::size_t i = 0; i < options.warmup_iterations; ++i) {
    if (!plan->Execute(handle, execution)) {
      return std::nullopt;
    }
  }

  const auto cold_start = std::chrono::high_resolution_clock::now();
  if (!plan->Execute(handle, execution)) {
    return std::nullopt;
  }
  const auto cold_end = std::chrono::high_resolution_clock::now();

  std::vector<double> hot_ms;
  hot_ms.reserve(options.hot_iterations);
  for (std::size_t i = 0; i < options.hot_iterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    if (!plan->Execute(handle, execution)) {
      return std::nullopt;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    hot_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  const double build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start).count();
  const double upload_ms =
      std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
  const double cold_ms =
      std::chrono::duration<double, std::milli>(cold_end - cold_start).count();
  const auto [hot_min_it, hot_max_it] = std::minmax_element(hot_ms.begin(), hot_ms.end());
  const double hot_sum = std::accumulate(hot_ms.begin(), hot_ms.end(), 0.0);
  const double hot_mean = hot_sum / static_cast<double>(hot_ms.size());

  BenchmarkResult result;
  result.case_name = benchmark_case.name;
  result.phase = benchmark_case.phase;
  result.batch_size = benchmark_case.batch_size;
  result.query_tokens = benchmark_case.query_tokens;
  result.kv_tokens = benchmark_case.kv_tokens;
  result.causal = benchmark_case.causal;
  result.query_head_count = kQueryHeadCount;
  result.kv_head_count = kKvHeadCount;
  result.head_dim = kHeadDim;
  result.tokens_per_page = kTokensPerPage;
  result.page_table_entries = config->page_table_entries;
  result.container_page_count = config->container_page_count;
  result.plan_workspace_bytes = plan->workspace_bytes();
  result.warmup_iterations = options.warmup_iterations;
  result.hot_iterations = options.hot_iterations;
  result.plan_build_ms = build_ms;
  result.input_upload_ms = upload_ms;
  result.cold_first_call_ms = cold_ms;
  result.hot_mean_ms = hot_mean;
  result.hot_min_ms = *hot_min_it;
  result.hot_max_ms = *hot_max_it;
  result.batch_query_tokens_per_second =
      hot_mean > 0.0
          ? static_cast<double>(benchmark_case.batch_size * benchmark_case.query_tokens) * 1000.0 / hot_mean
          : 0.0;
  return result;
}

std::string ResultsToJson(
    const EnvironmentInfo& environment,
    const std::vector<BenchmarkResult>& results) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"environment\": {\n";
  oss << "    \"device_name\": \"" << EscapeJson(environment.device_name) << "\",\n";
  oss << "    \"device_index\": " << environment.device_index << ",\n";
  oss << "    \"runtime_version\": " << environment.runtime_version << ",\n";
  oss << "    \"driver_version\": " << environment.driver_version << ",\n";
  oss << "    \"cudnn_version\": " << environment.cudnn_version << ",\n";
  oss << "    \"compute_capability\": \"" << environment.compute_capability_major << "."
      << environment.compute_capability_minor << "\",\n";
  oss << "    \"total_global_mem_bytes\": " << environment.total_global_mem_bytes << "\n";
  oss << "  },\n";
  oss << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    oss << "    {\n";
    oss << "      \"case_name\": \"" << EscapeJson(result.case_name) << "\",\n";
    oss << "      \"phase\": \"" << EscapeJson(result.phase) << "\",\n";
    oss << "      \"batch_size\": " << result.batch_size << ",\n";
    oss << "      \"query_tokens\": " << result.query_tokens << ",\n";
    oss << "      \"kv_tokens\": " << result.kv_tokens << ",\n";
    oss << "      \"causal\": " << (result.causal ? "true" : "false") << ",\n";
    oss << "      \"query_head_count\": " << result.query_head_count << ",\n";
    oss << "      \"kv_head_count\": " << result.kv_head_count << ",\n";
    oss << "      \"head_dim\": " << result.head_dim << ",\n";
    oss << "      \"tokens_per_page\": " << result.tokens_per_page << ",\n";
    oss << "      \"page_table_entries\": " << result.page_table_entries << ",\n";
    oss << "      \"container_page_count\": " << result.container_page_count << ",\n";
    oss << "      \"plan_workspace_bytes\": " << result.plan_workspace_bytes << ",\n";
    oss << "      \"warmup_iterations\": " << result.warmup_iterations << ",\n";
    oss << "      \"hot_iterations\": " << result.hot_iterations << ",\n";
    oss << std::fixed << std::setprecision(6);
    oss << "      \"plan_build_ms\": " << result.plan_build_ms << ",\n";
    oss << "      \"input_upload_ms\": " << result.input_upload_ms << ",\n";
    oss << "      \"cold_first_call_ms\": " << result.cold_first_call_ms << ",\n";
    oss << "      \"hot_mean_ms\": " << result.hot_mean_ms << ",\n";
    oss << "      \"hot_min_ms\": " << result.hot_min_ms << ",\n";
    oss << "      \"hot_max_ms\": " << result.hot_max_ms << ",\n";
    oss << "      \"batch_query_tokens_per_second\": " << result.batch_query_tokens_per_second << "\n";
    oss << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

bool WriteTextFile(const std::filesystem::path& path, std::string_view contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return false;
  }
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  return output.good();
}

void PrintResultSummary(const BenchmarkResult& result) {
  std::cout << result.case_name
            << " phase=" << result.phase
            << " batch=" << result.batch_size
            << " q=" << result.query_tokens
            << " kv=" << result.kv_tokens
            << " causal=" << (result.causal ? "true" : "false")
            << " build_ms=" << std::fixed << std::setprecision(3) << result.plan_build_ms
            << " cold_ms=" << result.cold_first_call_ms
            << " hot_mean_ms=" << result.hot_mean_ms
            << " qtok/s=" << result.batch_query_tokens_per_second
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  if (options.list_cases_only) {
    PrintCaseList();
    return 0;
  }

  const auto handle = CudnnHandle::Create();
  if (!handle || !handle->valid()) {
    std::cerr << "error: no CUDA device or cuDNN handle unavailable\n";
    return 1;
  }
  if (handle->version() < 90500) {
    std::cerr << "error: cuDNN 9.5 or newer is required for paged attention\n";
    return 1;
  }

  const auto selected_cases = SelectCases(options);
  if (selected_cases.empty()) {
    std::cerr << "error: no valid benchmark cases selected\n";
    return 1;
  }

  const EnvironmentInfo environment = CollectEnvironmentInfo(*handle);
  std::vector<BenchmarkResult> results;
  results.reserve(selected_cases.size());

  for (const BenchmarkCase* benchmark_case : selected_cases) {
    const auto result = RunCase(*benchmark_case, options, *handle);
    if (!result.has_value()) {
      std::cerr << "error: benchmark case failed: " << benchmark_case->name << "\n";
      return 1;
    }
    PrintResultSummary(*result);
    results.push_back(*result);
  }

  const std::string json = ResultsToJson(environment, results);
  if (options.json_output_path.has_value() && !WriteTextFile(*options.json_output_path, json)) {
    std::cerr << "error: failed to write JSON output to " << options.json_output_path->string() << "\n";
    return 1;
  }

  std::cout << "\n";
  std::cout << json;
  return 0;
}
