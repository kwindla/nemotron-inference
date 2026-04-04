#include "nemotron/device_tensor.h"
#include "nemotron/expert_layer.h"
#include "nemotron/model_schedule.h"
#include "nemotron/request_context.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct Options {
  std::filesystem::path manifest_path;
  std::filesystem::path input_hidden_path;
  std::filesystem::path dump_root;
  std::size_t layer_index = 1;
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
    if (arg == "--dump-root" && i + 1 < argc) {
      options->dump_root = argv[++i];
      continue;
    }
    if (arg == "--layer-index" && i + 1 < argc) {
      options->layer_index = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    std::cerr << "unknown argument: " << arg << "\n";
    return false;
  }
  return !options->manifest_path.empty() &&
         !options->input_hidden_path.empty() &&
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
    const nemotron::ExpertLayerRunTrace& trace) {
  return "{\n"
         "  \"layer_index\": " + std::to_string(layer_index) + ",\n" +
         "  \"token_count\": " + std::to_string(token_count) + ",\n" +
         "  \"hidden_size\": " + std::to_string(hidden_size) + ",\n" +
         "  \"selected_experts\": " + SelectionsJson(trace.selected_experts) + "\n" +
         "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    std::cerr << "usage: expert_layer_trace_dump"
              << " --manifest <path>"
              << " --input-hidden-bin <path>"
              << " --dump-root <path>"
              << " [--layer-index <index>]\n";
    return 2;
  }
  if (!HasCudaDevice()) {
    std::cerr << "expert_layer_trace_dump: CUDA device unavailable\n";
    return 1;
  }

  nemotron::SingleTokenForwardConfig config = nemotron::KnownNemotron3Super120BA12BConfig();
  std::size_t token_count = 0;
  const auto input_hidden_host = LoadFloatMatrix(options.input_hidden_path, config.hidden_size, &token_count);
  if (!input_hidden_host.has_value()) {
    std::cerr << "expert_layer_trace_dump: failed to load input hidden matrix\n";
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

  auto slice = nemotron::ExpertLayerSlice::Create(expert_config, *bindings);
  if (!slice || !slice->valid()) {
    std::cerr << "expert_layer_trace_dump: failed to create expert slice\n";
    return 1;
  }

  auto request_context = nemotron::RequestExecutionContext::Create(plan->request_config);
  auto cublas = nemotron::CublasLtHandle::Create();
  auto input = nemotron::DeviceTensorFp32::Create({token_count, config.hidden_size});
  auto output = nemotron::DeviceTensorFp32::Create({token_count, config.hidden_size});
  if (!request_context || !request_context->valid() ||
      !cublas || !cublas->valid() ||
      !input || !input->valid() ||
      !output || !output->valid() ||
      !input->CopyFromHost(input_hidden_host->data(), input_hidden_host->size())) {
    std::cerr << "expert_layer_trace_dump: execution setup failed\n";
    return 1;
  }

  nemotron::ExpertLayerRunTrace trace;
  if (!slice->RunWithRequestContext(
          *cublas,
          environment->gemm_heuristic_cache(),
          *request_context,
          *input,
          output.get(),
          token_count == 1 ? &trace : nullptr)) {
    std::cerr << "expert_layer_trace_dump: expert execution failed\n";
    return 1;
  }

  std::vector<float> output_host(output->numel(), 0.0f);
  if (!output->CopyToHost(output_host.data(), output_host.size())) {
    std::cerr << "expert_layer_trace_dump: output download failed\n";
    return 1;
  }

  std::filesystem::create_directories(options.dump_root);
  WriteFloatFile(options.dump_root / "final_output_fp32.bin", output_host);
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
        SummaryJson(options.layer_index, token_count, config.hidden_size, trace));
  } else {
    WriteJsonFile(
        options.dump_root / "summary.json",
        "{\n"
        "  \"layer_index\": " + std::to_string(options.layer_index) + ",\n" +
        "  \"token_count\": " + std::to_string(token_count) + ",\n" +
        "  \"hidden_size\": " + std::to_string(config.hidden_size) + "\n"
        "}\n");
  }

  std::cout << "expert_layer_trace_dump: layer_index=" << options.layer_index
            << " token_count=" << token_count
            << " output_rows=" << token_count
            << "\n";
  return 0;
}
