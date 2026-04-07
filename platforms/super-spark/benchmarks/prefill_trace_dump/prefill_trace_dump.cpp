#include "nemotron/device_tensor.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct Options {
  std::filesystem::path manifest_path;
  std::filesystem::path tokens_json_path;
  std::filesystem::path dump_root;
  std::filesystem::path model_cache_path;
  std::filesystem::path json_output;
  std::optional<std::size_t> stop_layer;
};

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool CopyBf16TensorToHost(
    const nemotron::DeviceTensorBf16& tensor,
    std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  if (!tensor.CopyToHost(host_bf16.data(), host_bf16.size())) {
    return false;
  }
  output->resize(host_bf16.size(), 0.0f);
  for (std::size_t i = 0; i < host_bf16.size(); ++i) {
    (*output)[i] = __bfloat162float(host_bf16[i]);
  }
  return true;
}

std::vector<std::uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void WriteFloatFile(const std::filesystem::path& path, const std::vector<float>& values) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::string LayerFileStem(std::size_t layer_index) {
  return std::to_string(layer_index / 100 % 10) +
         std::to_string(layer_index / 10 % 10) +
         std::to_string(layer_index % 10);
}

std::optional<std::vector<std::int32_t>> ParseJsonInt32Array(
    const std::string& json,
    const std::vector<std::string>& keys) {
  for (const std::string& key : keys) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(json, match, pattern) || match.size() != 2) {
      continue;
    }
    std::vector<std::int32_t> values;
    const std::regex number_pattern("(-?[0-9]+)");
    const std::string payload = match[1].str();
    for (auto it = std::sregex_iterator(payload.begin(), payload.end(), number_pattern);
         it != std::sregex_iterator();
         ++it) {
      try {
        const long long parsed = std::stoll((*it)[1].str());
        if (parsed < static_cast<long long>(std::numeric_limits<std::int32_t>::min()) ||
            parsed > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
          return std::nullopt;
        }
        values.push_back(static_cast<std::int32_t>(parsed));
      } catch (...) {
        return std::nullopt;
      }
    }
    return values;
  }
  return std::nullopt;
}

std::optional<std::vector<std::int32_t>> LoadPromptTokenIds(const std::filesystem::path& path) {
  const std::string json = ReadTextFile(path);
  return ParseJsonInt32Array(json, {"input_token_ids", "runtime_token_ids"});
}

const char* LayerKindName(nemotron::ForwardLayerKind kind) {
  switch (kind) {
    case nemotron::ForwardLayerKind::kAttention:
      return "attention";
    case nemotron::ForwardLayerKind::kMamba:
      return "mamba";
    case nemotron::ForwardLayerKind::kExpert:
      return "expert";
  }
  return "unknown";
}

std::size_t MambaConvStateElemsPerLayer(const nemotron::SingleTokenForwardConfig& config) {
  const std::size_t conv_dim =
      config.mamba_intermediate_size + (2 * config.mamba_n_groups * config.mamba_state_size);
  return conv_dim * config.mamba_conv_kernel_size;
}

std::size_t MambaSsmStateElemsPerLayer(const nemotron::SingleTokenForwardConfig& config) {
  return config.mamba_num_heads * config.mamba_head_dim * config.mamba_state_size;
}

void DumpTraceOutputs(
    const nemotron::SingleTokenForwardTrace& trace,
    const std::filesystem::path& dump_root) {
  std::filesystem::create_directories(dump_root);
  if (!trace.embedding_output.empty()) {
    WriteFloatFile(dump_root / "embedding_output_fp32.bin", trace.embedding_output);
  }
  for (const auto& captured : trace.captured_layers) {
    const std::string name = "layer_" + LayerFileStem(captured.layer_index) + "_output_fp32.bin";
    WriteFloatFile(dump_root / name, captured.hidden);
  }
  if (!trace.final_hidden.empty()) {
    WriteFloatFile(dump_root / "final_hidden_fp32.bin", trace.final_hidden);
  }
  if (!trace.final_hidden_normed.empty()) {
    WriteFloatFile(dump_root / "final_hidden_normed_fp32.bin", trace.final_hidden_normed);
  }
  if (!trace.logits.empty()) {
    WriteFloatFile(dump_root / "logits_fp32.bin", trace.logits);
  }
}

void DumpRequestState(
    const nemotron::RequestExecutionContext& request_context,
    const nemotron::SingleTokenForwardPlan& plan,
    const nemotron::SingleTokenForwardConfig& config,
    const std::filesystem::path& dump_root) {
  if (const auto* full_conv = request_context.mamba_conv_state();
      full_conv != nullptr && full_conv->valid()) {
    std::vector<float> host(full_conv->numel(), 0.0f);
    if (CopyBf16TensorToHost(*full_conv, &host)) {
      WriteFloatFile(dump_root / "mamba_conv_state_full_fp32.bin", host);
      const std::size_t per_layer = MambaConvStateElemsPerLayer(config);
      for (const auto& layer : plan.layers) {
        if (layer.kind != nemotron::ForwardLayerKind::kMamba) {
          continue;
        }
        const std::size_t start = layer.mamba_conv_state_offset_elems;
        const std::size_t end = start + per_layer;
        if (end > host.size()) {
          continue;
        }
        WriteFloatFile(
            dump_root / ("mamba_layer_" + LayerFileStem(layer.layer_index) + "_conv_state_fp32.bin"),
            std::vector<float>(
                host.begin() + static_cast<std::ptrdiff_t>(start),
                host.begin() + static_cast<std::ptrdiff_t>(end)));
      }
    }
  }
  if (const auto* full_ssm = request_context.mamba_state();
      full_ssm != nullptr && full_ssm->valid()) {
    std::vector<float> host(full_ssm->numel(), 0.0f);
    if (full_ssm->CopyToHost(host.data(), host.size())) {
      WriteFloatFile(dump_root / "mamba_ssm_state_full_fp32.bin", host);
      const std::size_t per_layer = MambaSsmStateElemsPerLayer(config);
      for (const auto& layer : plan.layers) {
        if (layer.kind != nemotron::ForwardLayerKind::kMamba) {
          continue;
        }
        const std::size_t start = layer.mamba_state_offset_elems;
        const std::size_t end = start + per_layer;
        if (end > host.size()) {
          continue;
        }
        WriteFloatFile(
            dump_root / ("mamba_layer_" + LayerFileStem(layer.layer_index) + "_ssm_state_fp32.bin"),
            std::vector<float>(
                host.begin() + static_cast<std::ptrdiff_t>(start),
                host.begin() + static_cast<std::ptrdiff_t>(end)));
      }
    }
  }
}

std::int32_t ArgmaxLastLogitRow(
    const std::vector<float>& logits,
    std::size_t token_count,
    std::size_t vocab_size) {
  if (token_count == 0 || vocab_size == 0 || logits.size() != token_count * vocab_size) {
    return -1;
  }
  const float* row = logits.data() + ((token_count - 1) * vocab_size);
  std::size_t best_index = 0;
  float best_value = row[0];
  for (std::size_t i = 1; i < vocab_size; ++i) {
    if (row[i] > best_value) {
      best_value = row[i];
      best_index = i;
    }
  }
  if (best_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return -1;
  }
  return static_cast<std::int32_t>(best_index);
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
    if (arg == "--tokens-json" && i + 1 < argc) {
      options->tokens_json_path = argv[++i];
      continue;
    }
    if (arg == "--dump-root" && i + 1 < argc) {
      options->dump_root = argv[++i];
      continue;
    }
    if (arg == "--model-cache" && i + 1 < argc) {
      options->model_cache_path = argv[++i];
      continue;
    }
    if (arg == "--json-output" && i + 1 < argc) {
      options->json_output = argv[++i];
      continue;
    }
    if (arg == "--stop-layer" && i + 1 < argc) {
      options->stop_layer = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    std::cerr << "unknown argument: " << arg << "\n";
    return false;
  }
  return !options->manifest_path.empty() &&
         !options->tokens_json_path.empty() &&
         !options->dump_root.empty();
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

bool WriteSummaryJson(
    const Options& options,
    const nemotron::SingleTokenForwardModel& model,
    std::size_t token_count,
    std::int32_t predicted_token_id,
    const std::vector<std::int32_t>& token_ids) {
  if (options.json_output.empty()) {
    return true;
  }
  if (!options.json_output.parent_path().empty()) {
    std::filesystem::create_directories(options.json_output.parent_path());
  }
  std::ofstream output(options.json_output);
  if (!output) {
    return false;
  }
  output << "{\n";
  output << "  \"manifest_path\": \"" << options.manifest_path.string() << "\",\n";
  output << "  \"tokens_json_path\": \"" << options.tokens_json_path.string() << "\",\n";
  output << "  \"dump_root\": \"" << options.dump_root.string() << "\",\n";
  output << "  \"model_cache_path\": \"" << options.model_cache_path.string() << "\",\n";
  output << "  \"token_count\": " << token_count << ",\n";
  output << "  \"predicted_token_id\": " << predicted_token_id << ",\n";
  output << "  \"stop_layer\": ";
  if (options.stop_layer.has_value()) {
    output << *options.stop_layer;
  } else {
    output << "null";
  }
  output << ",\n";
  output << "  \"input_token_ids\": [";
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << token_ids[i];
  }
  output << "],\n";
  output << "  \"plan_layers\": [\n";
  for (std::size_t i = 0; i < model.plan().layers.size(); ++i) {
    const auto& layer = model.plan().layers[i];
    output << "    {\"layer_index\": " << layer.layer_index
           << ", \"kind\": \"" << LayerKindName(layer.kind) << "\"}";
    output << (i + 1 == model.plan().layers.size() ? "\n" : ",\n");
  }
  output << "  ]\n";
  output << "}\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    std::cerr
        << "usage: prefill_trace_dump --manifest /path/to/manifest.json "
        << "--tokens-json /path/to/tokens.json "
        << "--dump-root /path/to/dump "
        << "[--model-cache /path/to/model.cache] "
        << "[--stop-layer N] "
        << "[--json-output /path/to/summary.json]\n";
    return 2;
  }

  if (!HasCudaDevice()) {
    std::cerr << "prefill_trace_dump: no CUDA device available\n";
    return 1;
  }
  if (!std::filesystem::exists(options.manifest_path)) {
    std::cerr << "prefill_trace_dump: manifest does not exist: "
              << options.manifest_path << "\n";
    return 1;
  }
  if (!options.model_cache_path.empty() && !std::filesystem::exists(options.model_cache_path)) {
    std::cerr << "prefill_trace_dump: model cache does not exist: "
              << options.model_cache_path << "\n";
    return 1;
  }

  const auto token_ids = LoadPromptTokenIds(options.tokens_json_path);
  if (!token_ids.has_value() || token_ids->empty()) {
    std::cerr << "prefill_trace_dump: failed to load prompt token IDs from "
              << options.tokens_json_path << "\n";
    return 1;
  }

  const auto environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(
      options.manifest_path,
      MakeOptions(token_ids->size()));
  if (!environment) {
    std::cerr << "prefill_trace_dump: failed to build runtime environment\n";
    return 1;
  }

  nemotron::SingleTokenForwardConfig config =
      nemotron::KnownNemotron3Super120BA12BConfig();
  config.max_tokens = token_ids->size();

  std::unique_ptr<nemotron::SingleTokenForwardModel> model;
  if (options.model_cache_path.empty()) {
    model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  } else {
    model = nemotron::SingleTokenForwardModel::CreateFromCache(
        *environment,
        config,
        options.model_cache_path);
  }
  if (model == nullptr || !model->valid()) {
    std::cerr << "prefill_trace_dump: failed to build forward model\n";
    return 1;
  }

  auto request_context = model->CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "prefill_trace_dump: failed to create request context\n";
    return 1;
  }

  auto logits = nemotron::DeviceTensorFp32::Create({token_ids->size(), config.vocab_size});
  if (logits == nullptr || !logits->valid()) {
    std::cerr << "prefill_trace_dump: failed to create logits tensor\n";
    return 1;
  }

  std::vector<std::size_t> capture_layers;
  capture_layers.reserve(model->plan().layers.size());
  for (const auto& layer : model->plan().layers) {
    capture_layers.push_back(layer.layer_index);
  }

  nemotron::SingleTokenForwardTrace trace;
  const bool ok = model->RunPrefill(
      token_ids->data(),
      token_ids->size(),
      *request_context,
      logits.get(),
      capture_layers,
      &trace,
      options.stop_layer);
  if (!ok) {
    std::cerr << "prefill_trace_dump: RunPrefill failed\n";
    return 1;
  }

  DumpTraceOutputs(trace, options.dump_root);
  DumpRequestState(*request_context, model->plan(), config, options.dump_root);
  const std::int32_t predicted_token_id =
      ArgmaxLastLogitRow(trace.logits, token_ids->size(), config.vocab_size);
  if (!WriteSummaryJson(options, *model, token_ids->size(), predicted_token_id, *token_ids)) {
    std::cerr << "prefill_trace_dump: failed to write summary JSON\n";
    return 1;
  }

  std::cout << "prefill_trace_dump: token_count=" << token_ids->size()
            << " predicted_token_id=" << predicted_token_id
            << " cache=" << (options.model_cache_path.empty() ? "<direct>" : options.model_cache_path.string())
            << "\n";
  return 0;
}
