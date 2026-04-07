#include "nemotron/artifact_loader.h"
#include "nemotron/manifest.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using nemotron::ArtifactLoadMode;
using nemotron::ArtifactLoader;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;

struct BenchmarkOptions {
  std::filesystem::path manifest_path;
  int repeats = 3;
  std::optional<std::filesystem::path> json_output;
};

struct ModeResult {
  ArtifactLoadMode mode = ArtifactLoadMode::kMmap;
  bool success = false;
  int repeats = 0;
  std::size_t tensor_count = 0;
  std::size_t loaded_file_count = 0;
  std::size_t total_loaded_bytes = 0;
  double mean_open_ms = 0.0;
  double mean_lookup_ms = 0.0;
};

std::string JsonEscape(const std::string& value) {
  std::ostringstream oss;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --manifest /path/to/manifest.json [--repeats N] [--json-output out.json]\n";
}

std::optional<BenchmarkOptions> ParseArgs(int argc, char** argv) {
  BenchmarkOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--manifest" && i + 1 < argc) {
      options.manifest_path = argv[++i];
    } else if (arg == "--repeats" && i + 1 < argc) {
      options.repeats = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--json-output" && i + 1 < argc) {
      options.json_output = std::filesystem::path(argv[++i]);
    } else {
      PrintUsage(argv[0]);
      return std::nullopt;
    }
  }

  if (options.manifest_path.empty()) {
    PrintUsage(argv[0]);
    return std::nullopt;
  }
  return options;
}

double ToMilliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

ModeResult RunMode(
    const nemotron::PackedModelManifest& manifest,
    const std::filesystem::path& manifest_path,
    ArtifactLoadMode mode,
    int repeats) {
  ModeResult result;
  result.mode = mode;
  result.repeats = repeats;
  result.tensor_count = manifest.tensors.size();

  double open_ms_total = 0.0;
  double lookup_ms_total = 0.0;
  for (int i = 0; i < repeats; ++i) {
    const auto open_start = std::chrono::steady_clock::now();
    auto loader = ArtifactLoader::OpenVerifiedWithMode(manifest, manifest_path, mode);
    const auto open_end = std::chrono::steady_clock::now();
    if (!loader) {
      return result;
    }

    const auto lookup_start = std::chrono::steady_clock::now();
    bool found_all = true;
    for (const auto& tensor : manifest.tensors) {
      found_all &= loader->FindTensor(tensor.name).has_value();
    }
    const auto lookup_end = std::chrono::steady_clock::now();
    if (!found_all) {
      return result;
    }

    result.loaded_file_count = loader->loaded_file_count();
    result.total_loaded_bytes = loader->total_loaded_bytes();
    open_ms_total += ToMilliseconds(open_end - open_start);
    lookup_ms_total += ToMilliseconds(lookup_end - lookup_start);
  }

  result.success = true;
  result.mean_open_ms = open_ms_total / static_cast<double>(repeats);
  result.mean_lookup_ms = lookup_ms_total / static_cast<double>(repeats);
  return result;
}

std::string RenderJson(
    const BenchmarkOptions& options,
    const ManifestLoadResult& load_result,
    const std::vector<ModeResult>& results) {
  std::ostringstream json;
  json << "{\n"
       << "  \"manifest_path\": \"" << JsonEscape(options.manifest_path.string()) << "\",\n"
       << "  \"repeats\": " << options.repeats << ",\n"
       << "  \"manifest_ok\": " << (load_result.ok ? "true" : "false") << ",\n"
       << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const ModeResult& result = results[i];
    json << "    {\n"
         << "      \"mode\": \"" << nemotron::ToString(result.mode) << "\",\n"
         << "      \"success\": " << (result.success ? "true" : "false") << ",\n"
         << "      \"tensor_count\": " << result.tensor_count << ",\n"
         << "      \"loaded_file_count\": " << result.loaded_file_count << ",\n"
         << "      \"total_loaded_bytes\": " << result.total_loaded_bytes << ",\n"
         << "      \"mean_open_ms\": " << std::fixed << std::setprecision(6) << result.mean_open_ms << ",\n"
         << "      \"mean_lookup_ms\": " << std::fixed << std::setprecision(6) << result.mean_lookup_ms << "\n"
         << "    }";
    if (i + 1 != results.size()) {
      json << ",";
    }
    json << "\n";
  }
  json << "  ]\n"
       << "}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = ParseArgs(argc, argv);
  if (!options.has_value()) {
    return 2;
  }

  const ManifestLoadResult load_result = LoadVerifiedManifestFromJsonFile(options->manifest_path);
  if (!load_result.ok) {
    std::cerr << "failed to load verified manifest: " << options->manifest_path << "\n";
    return 1;
  }

  std::vector<ModeResult> results;
  results.push_back(RunMode(load_result.manifest, options->manifest_path, ArtifactLoadMode::kMmap, options->repeats));
  results.push_back(RunMode(load_result.manifest, options->manifest_path, ArtifactLoadMode::kReadAll, options->repeats));

  for (const ModeResult& result : results) {
    std::cout << nemotron::ToString(result.mode)
              << ": success=" << (result.success ? "true" : "false")
              << " files=" << result.loaded_file_count
              << " bytes=" << result.total_loaded_bytes
              << " open_ms=" << std::fixed << std::setprecision(6) << result.mean_open_ms
              << " lookup_ms=" << std::fixed << std::setprecision(6) << result.mean_lookup_ms
              << "\n";
  }

  if (options->json_output.has_value()) {
    if (!options->json_output->parent_path().empty()) {
      std::filesystem::create_directories(options->json_output->parent_path());
    }
    std::ofstream output(*options->json_output);
    output << RenderJson(*options, load_result, results);
  }

  for (const ModeResult& result : results) {
    if (!result.success) {
      return 1;
    }
  }
  return 0;
}
