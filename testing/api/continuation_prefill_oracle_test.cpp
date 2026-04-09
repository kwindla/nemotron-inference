#include "nemotron/device_tensor.h"
#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::filesystem::path source_root() {
  return std::filesystem::path(NEMOTRON_SOURCE_ROOT);
}

std::filesystem::path resolve_manifest_path() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env != nullptr && manifest_env[0] != '\0') {
    return std::filesystem::path(manifest_env);
  }
  return source_root() / "artifacts" / "manifests" /
         "forward_runtime_manifest_nano_rtx5090_unverified.json";
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::optional<std::vector<std::int32_t>> parse_json_int32_array(const std::string& json) {
  std::vector<std::int32_t> values;
  const std::regex number_pattern("(-?[0-9]+)");
  for (auto it = std::sregex_iterator(json.begin(), json.end(), number_pattern);
       it != std::sregex_iterator();
       ++it) {
    try {
      const long long value = std::stoll((*it)[1].str());
      if (value < std::numeric_limits<std::int32_t>::min() ||
          value > std::numeric_limits<std::int32_t>::max()) {
        return std::nullopt;
      }
      values.push_back(static_cast<std::int32_t>(value));
    } catch (...) {
      return std::nullopt;
    }
  }
  if (values.empty()) {
    return std::nullopt;
  }
  return values;
}

std::optional<std::vector<std::int32_t>> load_short_chat_prompt_token_ids() {
  const std::filesystem::path prompt_path =
      source_root() / "testing" / "oracle" / "full_model_single_token_short_chat_cuda_v3" /
      "prompt_token_ids.json";
  if (!std::filesystem::exists(prompt_path)) {
    return std::nullopt;
  }
  return parse_json_int32_array(read_text_file(prompt_path));
}

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

nemotron::RuntimeBootstrapOptions make_options(std::size_t target_context_tokens) {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = target_context_tokens;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

std::vector<float> copy_tensor_to_host(const nemotron::DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::vector<float> slice_row(
    const std::vector<float>& values,
    std::size_t row_index,
    std::size_t row_width) {
  if (row_width == 0) {
    return {};
  }
  const std::size_t row_offset = row_index * row_width;
  if (row_offset + row_width > values.size()) {
    return {};
  }
  return std::vector<float>(
      values.begin() + static_cast<std::ptrdiff_t>(row_offset),
      values.begin() + static_cast<std::ptrdiff_t>(row_offset + row_width));
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::optional<std::int32_t> argmax_token_id(const std::vector<float>& logits_row) {
  if (logits_row.empty()) {
    return std::nullopt;
  }
  std::size_t best_index = 0;
  for (std::size_t i = 1; i < logits_row.size(); ++i) {
    if (logits_row[i] > logits_row[best_index]) {
      best_index = i;
    }
  }
  return static_cast<std::int32_t>(best_index);
}

}  // namespace

int main() {
  const auto manifest_path = resolve_manifest_path();
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "continuation_prefill_oracle_test: SKIP (manifest path does not exist)\n";
    return 0;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "manifest JSON should parse")) {
    return 1;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "continuation_prefill_oracle_test: SKIP (manifest model is not Nano)\n";
    return 0;
  }

  const auto prompt_token_ids = load_short_chat_prompt_token_ids();
  if (!expect(prompt_token_ids.has_value(), "short_chat prompt token ids should load") ||
      !expect(prompt_token_ids->size() >= 4, "short_chat prompt token ids should contain at least four tokens")) {
    return 1;
  }
  const std::vector<std::int32_t> prompt(
      prompt_token_ids->begin(),
      prompt_token_ids->begin() + 4);
  const std::size_t prefix_token_count = 2;
  const std::size_t suffix_token_count = prompt.size() - prefix_token_count;

  auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options(prompt.size()));
  if (!expect(static_cast<bool>(environment), "runtime environment should build from the manifest")) {
    return 1;
  }

  nemotron::SingleTokenForwardConfig config = nemotron::KnownNemotron3Nano30BA3BConfig();
  config.max_tokens = std::max<std::size_t>(config.max_tokens, prompt.size());
  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build from the runtime environment")) {
    return 1;
  }

  auto full_context = model->CreateRequestContext();
  auto split_prefill_context = model->CreateRequestContext();
  auto split_single_context = model->CreateRequestContext();
  auto split_all_single_context = model->CreateRequestContext();
  auto full_logits = nemotron::DeviceTensorFp32::Create({prompt.size(), config.vocab_size});
  auto split_prefix_logits = nemotron::DeviceTensorFp32::Create({prefix_token_count, config.vocab_size});
  auto split_tail_logits = nemotron::DeviceTensorFp32::Create({suffix_token_count, config.vocab_size});
  auto single_step_logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
  auto all_single_step_logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
  if (!expect(full_context != nullptr && full_context->valid(), "full context should allocate") ||
      !expect(split_prefill_context != nullptr && split_prefill_context->valid(), "split-prefill context should allocate") ||
      !expect(split_single_context != nullptr && split_single_context->valid(), "split-single-token context should allocate") ||
      !expect(split_all_single_context != nullptr && split_all_single_context->valid(), "split-all-single-token context should allocate") ||
      !expect(full_logits != nullptr && full_logits->valid(), "full logits should allocate") ||
      !expect(split_prefix_logits != nullptr && split_prefix_logits->valid(), "split prefix logits should allocate") ||
      !expect(split_tail_logits != nullptr && split_tail_logits->valid(), "split tail logits should allocate") ||
      !expect(single_step_logits != nullptr && single_step_logits->valid(), "single-step logits should allocate") ||
      !expect(all_single_step_logits != nullptr && all_single_step_logits->valid(), "all-single-step logits should allocate")) {
    return 1;
  }

  if (!expect(
          model->RunPrefill(prompt.data(), prompt.size(), *full_context, full_logits.get()),
          "full prefill should succeed") ||
      !expect(
          model->RunPrefill(prompt.data(), prefix_token_count, *split_prefill_context, split_prefix_logits.get()),
          "split prefix prefill should succeed") ||
      !expect(
          model->ContinuePrefill(
              prompt.data() + prefix_token_count,
              suffix_token_count,
              *split_prefill_context,
              split_tail_logits.get()),
          "split tail prefill should succeed") ||
      !expect(
          model->RunPrefill(prompt.data(), prefix_token_count, *split_single_context, split_prefix_logits.get()),
          "single-token prefix prefill should succeed")) {
    return 1;
  }

  for (std::size_t suffix_index = 0; suffix_index < suffix_token_count; ++suffix_index) {
    if (!expect(
            model->ContinueSingleToken(
                prompt[prefix_token_count + suffix_index],
                *split_single_context,
                single_step_logits.get()),
            "single-token tail continuation should succeed")) {
      return 1;
    }
  }

  for (std::size_t token_index = 0; token_index < prompt.size(); ++token_index) {
    const bool ok =
        token_index == 0
            ? model->RunSingleToken(prompt[token_index], *split_all_single_context, all_single_step_logits.get())
            : model->ContinueSingleToken(prompt[token_index], *split_all_single_context, all_single_step_logits.get());
    if (!expect(ok, "all-single-token replay should succeed")) {
      return 1;
    }
  }

  const auto full_host = copy_tensor_to_host(*full_logits);
  const auto split_tail_host = copy_tensor_to_host(*split_tail_logits);
  const auto single_host = copy_tensor_to_host(*single_step_logits);
  const auto all_single_host = copy_tensor_to_host(*all_single_step_logits);
  if (!expect(!full_host.empty() && !split_tail_host.empty() && !single_host.empty() && !all_single_host.empty(),
              "all logits buffers should copy to host")) {
    return 1;
  }

  const auto full_last_row = slice_row(full_host, prompt.size() - 1, config.vocab_size);
  const auto split_tail_last_row = slice_row(split_tail_host, suffix_token_count - 1, config.vocab_size);
  const auto full_argmax = argmax_token_id(full_last_row);
  const auto split_argmax = argmax_token_id(split_tail_last_row);
  const auto single_argmax = argmax_token_id(single_host);
  const auto all_single_argmax = argmax_token_id(all_single_host);
  const float split_diff = max_abs_diff(full_last_row, split_tail_last_row);
  const float single_diff = max_abs_diff(full_last_row, single_host);
  const float all_single_diff = max_abs_diff(full_last_row, all_single_host);

  std::cout << "continuation_prefill_oracle_test: split_prefill_max_abs_diff=" << split_diff
            << " single_token_tail_max_abs_diff=" << single_diff
            << " all_single_token_max_abs_diff=" << all_single_diff
            << " full_argmax=" << (full_argmax.has_value() ? std::to_string(*full_argmax) : "none")
            << " split_argmax=" << (split_argmax.has_value() ? std::to_string(*split_argmax) : "none")
            << " single_argmax=" << (single_argmax.has_value() ? std::to_string(*single_argmax) : "none")
            << " all_single_argmax="
            << (all_single_argmax.has_value() ? std::to_string(*all_single_argmax) : "none")
            << "\n";

  const bool split_single_matches =
      single_argmax.has_value() && full_argmax.has_value() && *single_argmax == *full_argmax;
  const bool split_prefill_matches =
      split_argmax.has_value() && full_argmax.has_value() && *split_argmax == *full_argmax;
  std::cout << "continuation_prefill_oracle_test: split_prefill_argmax_match="
            << (split_prefill_matches ? 1 : 0)
            << " split_single_tail_argmax_match=" << (split_single_matches ? 1 : 0)
            << "\n";

  if (!expect(
          split_prefill_matches,
          "split-prefill continuation should preserve the full-prefill next-token argmax") ||
      !expect(
          split_single_matches,
          "single-token continuation from a prefetched prefix should preserve the full-prefill next-token argmax") ||
      !expect(
          all_single_argmax.has_value() && full_argmax.has_value() && *all_single_argmax == *full_argmax,
          "all-single-token replay should preserve the full-prefill next-token argmax")) {
    return 1;
  }

  std::cout << "continuation_prefill_oracle_test: PASS\n";
  return 0;
}
