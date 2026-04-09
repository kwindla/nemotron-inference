#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr std::size_t kStableMoeWindowTokens = 23;
constexpr std::size_t kMaxNewTokens = 16;
constexpr std::size_t kRequestMaxTokens = 64;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

const std::vector<std::int32_t>& Prompt23TokenIds() {
  static const std::vector<std::int32_t> kPrompt = {
      10, 25708, 1010, 11, 1010, 10, 3263, 1010,
      7493, 1395, 1032, 1050, 1043, 1050, 1063, 11,
      1010, 10, 1503, 19464, 1010, 12, 1010,
  };
  return kPrompt;
}

const std::vector<std::int32_t>& Prompt24TokenIds() {
  static const std::vector<std::int32_t> kPrompt = {
      10, 25708, 1010, 11, 1010, 10, 3263, 1010,
      76786, 5675, 1395, 1032, 1050, 1043, 1050, 1063,
      11, 1010, 10, 1503, 19464, 1010, 12, 1010,
  };
  return kPrompt;
}

const std::vector<std::int32_t>& ExpectedPrompt23GeneratedTokenIds() {
  static const std::vector<std::int32_t> kGenerated = {
      1784, 3330, 25747, 1261, 6165, 4098, 1058, 1429,
      7493, 1395, 1032, 1050, 1043, 1050, 10555, 1531,
  };
  return kGenerated;
}

const std::vector<std::int32_t>& ExpectedPrompt24GeneratedTokenIds() {
  static const std::vector<std::int32_t> kGenerated = {
      1784, 3330, 25747, 1429, 76786, 5675, 1395, 1032,
      1050, 1043, 1050, 10555, 2157, 1681, 1261, 6165,
  };
  return kGenerated;
}

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::filesystem::path resolve_manifest_path() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env != nullptr && manifest_env[0] != '\0') {
    return std::filesystem::path(manifest_env);
  }
  return std::filesystem::path(NEMOTRON_SOURCE_ROOT) / "artifacts" / "manifests" /
         "forward_runtime_manifest_nano_rtx5090_unverified.json";
}

nemotron::RuntimeBootstrapOptions make_options() {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = kRequestMaxTokens;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

bool run_prompt_case(
    const std::string& label,
    const std::vector<std::int32_t>& prompt_token_ids,
    const std::vector<std::int32_t>& expected_generated_token_ids,
    const nemotron::SingleTokenForwardModel& model) {
  auto context = model.CreateRequestContext();
  if (!expect(context != nullptr && context->valid(), label + " request context should allocate")) {
    return false;
  }

  nemotron::GreedyDecodeConfig decode_config;
  decode_config.max_new_tokens = kMaxNewTokens;
  nemotron::GreedyDecodeResult result;
  if (!expect(
          model.RunGreedyDecode(
              prompt_token_ids.data(),
              prompt_token_ids.size(),
              decode_config,
              *context,
              &result),
          label + " greedy decode should succeed")) {
    return false;
  }

  if (!expect(
          result.generated_token_ids == expected_generated_token_ids,
          label + " generated token ids should match the focused regression oracle")) {
    std::cerr << label << " got:";
    for (std::int32_t token_id : result.generated_token_ids) {
      std::cerr << " " << token_id;
    }
    std::cerr << "\n" << label << " expected:";
    for (std::int32_t token_id : expected_generated_token_ids) {
      std::cerr << " " << token_id;
    }
    std::cerr << "\n";
    return false;
  }

  if (!expect(
          result.generated_token_ids.size() == kMaxNewTokens,
          label + " should generate the expected token count") ||
      !expect(!result.hit_capacity_limit, label + " should not hit the context limit") ||
      !expect(!result.hit_eos, label + " should not stop on EOS inside the oracle window")) {
    return false;
  }

  std::cout << label << " generated:";
  for (std::int32_t token_id : result.generated_token_ids) {
    std::cout << " " << token_id;
  }
  std::cout << "\n";
  return true;
}

bool run_regression_test() {
  if (!has_cuda_device()) {
    std::cout << "nano_24_token_prefill_regression_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const auto manifest_path = resolve_manifest_path();
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "nano_24_token_prefill_regression_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "manifest JSON should parse")) {
    return false;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "nano_24_token_prefill_regression_test: SKIP (manifest model is not Nano)\n";
    return true;
  }

  auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options());
  if (!expect(static_cast<bool>(environment), "runtime environment should build")) {
    return false;
  }

  nemotron::SingleTokenForwardConfig config = nemotron::KnownNemotron3Nano30BA3BConfig();
  config.max_tokens = std::max(config.max_tokens, kRequestMaxTokens);
  if (!expect(
          config.moe_prefill_window_tokens == kStableMoeWindowTokens,
          "Nano config should clamp routed MoE prefill to the stable 23-token window")) {
    return false;
  }

  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build")) {
    return false;
  }

  return run_prompt_case(
             "nano_24_token_prefill_regression_test/prompt23",
             Prompt23TokenIds(),
             ExpectedPrompt23GeneratedTokenIds(),
             *model) &&
         run_prompt_case(
             "nano_24_token_prefill_regression_test/prompt24",
             Prompt24TokenIds(),
             ExpectedPrompt24GeneratedTokenIds(),
             *model);
}

}  // namespace

int main() {
  if (!run_regression_test()) {
    return 1;
  }
  std::cout << "nano_24_token_prefill_regression_test: PASS\n";
  return 0;
}
