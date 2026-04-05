#include "attention_harness_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nemotron::attention_harness::CaseSpec;
using nemotron::attention_harness::Runner;

struct BenchmarkOptions {
  std::vector<std::string> selected_case_names;
  std::size_t warmup_iterations = 1;
  std::size_t hot_iterations = 5;
  std::optional<std::filesystem::path> json_output_path;
  bool list_cases_only = false;
  bool enable_nvtx = true;
};

struct EnvironmentInfo {
  std::string device_name;
  int runtime_version = 0;
  int driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
};

struct BenchmarkResult {
  std::string case_name;
  std::string phase;
  std::string backend;
  std::size_t prefix_tokens = 0;
  std::size_t query_tokens = 0;
  std::size_t total_sequence_tokens = 0;
  std::size_t total_pages = 0;
  std::size_t warmup_iterations = 0;
  std::size_t hot_iterations = 0;
  double cold_total_ms = 0.0;
  double hot_mean_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  double query_tokens_per_second = 0.0;
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

const std::vector<CaseSpec>& DefaultCases() {
  static const std::vector<CaseSpec> kCases = {
      {"prefill_q256", "cold_prefill", 0, 256, 256, true},
      {"prefill_q1024", "cold_prefill", 0, 1024, 1024, true},
      {"prefill_q4096", "cold_prefill", 0, 4096, 4096, true},
      {"extend_q128_live8k", "restored_extend", 8192 - 128, 128, 8192, true},
      {"extend_q256_live32k", "restored_extend", 32768 - 256, 256, 32768, true},
      {"extend_q1024_live64k", "restored_extend", 65536 - 1024, 1024, 65536, true},
  };
  return kCases;
}

const CaseSpec* FindCaseByName(std::string_view name) {
  for (const auto& spec : DefaultCases()) {
    if (spec.name == name) {
      return &spec;
    }
  }
  return nullptr;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --list-cases\n"
      << "  --case <name>                Repeat to select specific benchmark cases\n"
      << "  --warmup <count>             Warmup iterations, default: 1\n"
      << "  --iterations <count>         Hot timed iterations, default: 5\n"
      << "  --json-output <path>         Write JSON results to a file\n"
      << "  --disable-nvtx              Disable NVTX ranges for profiler traces\n";
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
    } else if (arg == "--disable-nvtx") {
      options->enable_nvtx = false;
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

EnvironmentInfo CollectEnvironmentInfo() {
  EnvironmentInfo info;
  cudaRuntimeGetVersion(&info.runtime_version);
  cudaDriverGetVersion(&info.driver_version);

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return info;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    info.device_name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
  }
  return info;
}

std::vector<const CaseSpec*> SelectCases(const BenchmarkOptions& options) {
  std::vector<const CaseSpec*> selected;
  if (options.selected_case_names.empty()) {
    for (const auto& spec : DefaultCases()) {
      selected.push_back(&spec);
    }
    return selected;
  }

  for (const auto& name : options.selected_case_names) {
    const CaseSpec* spec = FindCaseByName(name);
    if (spec != nullptr) {
      selected.push_back(spec);
    }
  }
  return selected;
}

std::optional<BenchmarkResult> RunCase(
    const CaseSpec& spec,
    const BenchmarkOptions& options) {
  auto runner = Runner::Create(spec);
  if (!runner) {
    return std::nullopt;
  }

  for (std::size_t i = 0; i < options.warmup_iterations; ++i) {
    if (!runner->RunOnce(options.enable_nvtx)) {
      return std::nullopt;
    }
  }

  double cold_ms = 0.0;
  if (!runner->MeasureOnce(options.enable_nvtx, &cold_ms)) {
    return std::nullopt;
  }

  std::vector<double> hot_ms;
  hot_ms.reserve(options.hot_iterations);
  for (std::size_t i = 0; i < options.hot_iterations; ++i) {
    double measured = 0.0;
    if (!runner->MeasureOnce(options.enable_nvtx, &measured)) {
      return std::nullopt;
    }
    hot_ms.push_back(measured);
  }

  const auto [hot_min_it, hot_max_it] = std::minmax_element(hot_ms.begin(), hot_ms.end());
  const double hot_sum = std::accumulate(hot_ms.begin(), hot_ms.end(), 0.0);
  const double hot_mean = hot_sum / static_cast<double>(hot_ms.size());

  BenchmarkResult result;
  result.case_name = spec.name;
  result.phase = spec.phase;
  result.backend = "nano_multi_token";
  result.prefix_tokens = spec.prefix_tokens;
  result.query_tokens = spec.query_tokens;
  result.total_sequence_tokens = spec.total_sequence_tokens;
  result.total_pages = runner->total_pages();
  result.warmup_iterations = options.warmup_iterations;
  result.hot_iterations = options.hot_iterations;
  result.cold_total_ms = cold_ms;
  result.hot_mean_ms = hot_mean;
  result.hot_min_ms = *hot_min_it;
  result.hot_max_ms = *hot_max_it;
  result.query_tokens_per_second =
      hot_mean > 0.0
          ? static_cast<double>(spec.query_tokens) * 1000.0 / hot_mean
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
  oss << "    \"runtime_version\": " << environment.runtime_version << ",\n";
  oss << "    \"driver_version\": " << environment.driver_version << ",\n";
  oss << "    \"compute_capability_major\": " << environment.compute_capability_major << ",\n";
  oss << "    \"compute_capability_minor\": " << environment.compute_capability_minor << "\n";
  oss << "  },\n";
  oss << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    oss << "    {\n";
    oss << "      \"case_name\": \"" << EscapeJson(result.case_name) << "\",\n";
    oss << "      \"phase\": \"" << EscapeJson(result.phase) << "\",\n";
    oss << "      \"backend\": \"" << EscapeJson(result.backend) << "\",\n";
    oss << "      \"prefix_tokens\": " << result.prefix_tokens << ",\n";
    oss << "      \"query_tokens\": " << result.query_tokens << ",\n";
    oss << "      \"total_sequence_tokens\": " << result.total_sequence_tokens << ",\n";
    oss << "      \"total_pages\": " << result.total_pages << ",\n";
    oss << "      \"warmup_iterations\": " << result.warmup_iterations << ",\n";
    oss << "      \"hot_iterations\": " << result.hot_iterations << ",\n";
    oss << "      \"cold_total_ms\": " << std::fixed << std::setprecision(4) << result.cold_total_ms << ",\n";
    oss << "      \"hot_mean_ms\": " << std::fixed << std::setprecision(4) << result.hot_mean_ms << ",\n";
    oss << "      \"hot_min_ms\": " << std::fixed << std::setprecision(4) << result.hot_min_ms << ",\n";
    oss << "      \"hot_max_ms\": " << std::fixed << std::setprecision(4) << result.hot_max_ms << ",\n";
    oss << "      \"query_tokens_per_second\": " << std::fixed << std::setprecision(2)
        << result.query_tokens_per_second << "\n";
    oss << "    }";
    if (i + 1 != results.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

void PrintCaseList() {
  std::cout << "Available benchmark cases:\n";
  for (const auto& spec : DefaultCases()) {
    std::cout << "  " << spec.name
              << " phase=" << spec.phase
              << " prefix=" << spec.prefix_tokens
              << " query=" << spec.query_tokens
              << " total=" << spec.total_sequence_tokens
              << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (!nemotron::attention_harness::HasCudaDevice()) {
    std::cout << "nano_paged_attention_bench: SKIP (no CUDA device available)\n";
    return 0;
  }

  if (options.list_cases_only) {
    PrintCaseList();
    return 0;
  }

  const auto selected_cases = SelectCases(options);
  if (selected_cases.empty()) {
    std::cerr << "nano_paged_attention_bench: no matching benchmark cases\n";
    return 1;
  }

  const EnvironmentInfo environment = CollectEnvironmentInfo();
  std::vector<BenchmarkResult> results;
  for (const CaseSpec* spec : selected_cases) {
    std::cout << "nano_paged_attention_bench: running case=" << spec->name
              << " prefix=" << spec->prefix_tokens
              << " query=" << spec->query_tokens
              << " total=" << spec->total_sequence_tokens << "\n";
    const auto result = RunCase(*spec, options);
    if (!result.has_value()) {
      std::cerr << "nano_paged_attention_bench: case failed: " << spec->name << "\n";
      return 1;
    }
    results.push_back(*result);
    std::cout << "nano_paged_attention_bench: case=" << result->case_name
              << " backend=" << result->backend
              << " cold_ms=" << std::fixed << std::setprecision(4) << result->cold_total_ms
              << " hot_mean_ms=" << std::fixed << std::setprecision(4) << result->hot_mean_ms
              << " tokens_per_s=" << std::fixed << std::setprecision(2)
              << result->query_tokens_per_second << "\n";
  }

  if (options.json_output_path.has_value()) {
    std::ofstream output(*options.json_output_path);
    if (!output) {
      std::cerr << "nano_paged_attention_bench: failed to open JSON output path\n";
      return 1;
    }
    output << ResultsToJson(environment, results);
  }

  return 0;
}
