#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nemotron/attention_layer.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/cudnn_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/embedding_table.h"
#include "nemotron/expert_layer.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/linear_op.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/model_schedule.h"
#include "nemotron/paged_kv_cache.h"
#include "nemotron/primitive_ops.h"
#include "nemotron/runtime_environment.h"

namespace nemotron {
namespace {

constexpr std::size_t kBytesPerMiB = 1024ull * 1024ull;
constexpr std::size_t kDefaultForwardVramReserveMiB = 512ull;

struct CudaMemInfo {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

bool EnvEnabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool DecodeConsistentPrefillEnabled() {
  const char* explicit_value = std::getenv("NEMOTRON_FORWARD_DECODE_CONSISTENT_PREFILL");
  if (explicit_value != nullptr) {
    return explicit_value[0] != '\0' && std::string(explicit_value) != "0";
  }
  return EnvEnabled("NEMOTRON_FORWARD_FUSED_MAMBA_DECODE") ||
         EnvEnabled("NEMOTRON_FORWARD_FUSED_MOE_DECODE");
}

std::size_t ParseEnvMiB(const char* env_var, std::size_t default_value_mib) {
  const char* value = std::getenv(env_var);
  if (value == nullptr || value[0] == '\0') {
    return default_value_mib;
  }
  char* parse_end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &parse_end, 10);
  if (errno != 0 || parse_end == value || (parse_end != nullptr && *parse_end != '\0')) {
    return default_value_mib;
  }
  return static_cast<std::size_t>(parsed);
}

std::size_t ForwardVramReserveBytes() {
  return ParseEnvMiB("NEMOTRON_FORWARD_VRAM_RESERVE_MB", kDefaultForwardVramReserveMiB) * kBytesPerMiB;
}

std::optional<CudaMemInfo> QueryCudaMemInfo(cudaError_t* status_out = nullptr) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status_out != nullptr) {
    *status_out = status;
  }
  if (status != cudaSuccess) {
    return std::nullopt;
  }
  return CudaMemInfo{free_bytes, total_bytes};
}

void LogCudaMemInfo(const char* label, const CudaMemInfo& info, std::size_t reserve_bytes) {
  std::cerr << "single_token_forward_model: " << label
            << " free_vram_mib=" << (info.free_bytes / kBytesPerMiB)
            << " total_vram_mib=" << (info.total_bytes / kBytesPerMiB)
            << " reserve_mib=" << (reserve_bytes / kBytesPerMiB)
            << "\n";
}

std::size_t EffectiveScratchTokens(const RequestExecutionConfig& config, std::size_t max_tokens) {
  if (config.scratch_tokens == 0) {
    return max_tokens;
  }
  return std::min(config.scratch_tokens, max_tokens);
}

std::size_t RequestActivationBytes(const RequestExecutionConfig& config, std::size_t max_tokens) {
  const std::size_t scratch_tokens = EffectiveScratchTokens(config, max_tokens);
  return ((2 * max_tokens) + scratch_tokens) * config.hidden_size * sizeof(float);
}

std::size_t KvPagesForTokens(const RequestExecutionConfig& config, std::size_t max_tokens) {
  if (config.attention_kv_cache.layer_count == 0 || config.attention_total_pages == 0 || max_tokens == 0) {
    return 0;
  }
  const std::size_t requested_pages =
      config.attention_kv_cache.layer_count *
      RequiredPagesForTokens(config.attention_kv_cache, max_tokens);
  return std::min(config.attention_total_pages, requested_pages);
}

std::optional<std::size_t> KvCacheBytesForTokens(
    const RequestExecutionConfig& config,
    std::size_t max_tokens) {
  if (config.attention_kv_cache.layer_count == 0 || max_tokens == 0) {
    return std::size_t{0};
  }
  const auto geometry = BuildAttentionKvPageGeometry(config.attention_kv_cache);
  if (!geometry.has_value()) {
    return std::nullopt;
  }
  return KvPagesForTokens(config, max_tokens) * geometry->bytes_per_page;
}

std::optional<std::size_t> RequestContextBytesForTokens(
    const RequestExecutionConfig& config,
    std::size_t max_tokens) {
  const auto kv_bytes = KvCacheBytesForTokens(config, max_tokens);
  if (!kv_bytes.has_value()) {
    return std::nullopt;
  }
  return RequestActivationBytes(config, max_tokens) +
         config.mamba_conv_state_bytes_fp32 +
         config.mamba_state_bytes_fp32 +
         *kv_bytes;
}

std::optional<RequestExecutionConfig> BuildMeasuredBudgetRequestConfig(
    const RequestExecutionConfig& base_config,
    std::size_t measured_free_vram_bytes,
    std::size_t reserve_bytes,
    bool debug) {
  if (measured_free_vram_bytes <= reserve_bytes) {
    if (debug) {
      std::cerr << "single_token_forward_model: measured free VRAM is fully consumed by reserve"
                << " free_vram_mib=" << (measured_free_vram_bytes / kBytesPerMiB)
                << " reserve_mib=" << (reserve_bytes / kBytesPerMiB)
                << "\n";
    }
    return std::nullopt;
  }

  const std::size_t available_bytes = measured_free_vram_bytes - reserve_bytes;
  const auto requested_bytes = RequestContextBytesForTokens(base_config, base_config.max_tokens);
  if (!requested_bytes.has_value()) {
    return std::nullopt;
  }
  if (*requested_bytes <= available_bytes) {
    if (debug) {
      std::cerr << "single_token_forward_model: request context fits measured VRAM budget"
                << " available_mib=" << (available_bytes / kBytesPerMiB)
                << " requested_mib=" << (*requested_bytes / kBytesPerMiB)
                << " max_tokens=" << base_config.max_tokens
                << " attention_total_pages=" << base_config.attention_total_pages
                << "\n";
    }
    return base_config;
  }

  std::size_t low = 0;
  std::size_t high = base_config.max_tokens;
  while (low < high) {
    const std::size_t mid = low + ((high - low + 1) / 2);
    const auto candidate_bytes = RequestContextBytesForTokens(base_config, mid);
    if (candidate_bytes.has_value() && *candidate_bytes <= available_bytes) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  if (low == 0) {
    if (debug) {
      const auto minimum_bytes = RequestContextBytesForTokens(base_config, 1);
      std::cerr << "single_token_forward_model: request context does not fit measured VRAM budget"
                << " available_mib=" << (available_bytes / kBytesPerMiB)
                << " minimum_one_token_mib="
                << ((minimum_bytes.has_value() ? *minimum_bytes : 0) / kBytesPerMiB)
                << "\n";
    }
    return std::nullopt;
  }

  RequestExecutionConfig capped = base_config;
  capped.max_tokens = low;
  capped.scratch_tokens = base_config.scratch_tokens == 0
                              ? 0
                              : std::min(base_config.scratch_tokens, capped.max_tokens);
  capped.attention_total_pages = KvPagesForTokens(base_config, capped.max_tokens);

  if (debug) {
    const auto capped_bytes = RequestContextBytesForTokens(capped, capped.max_tokens);
    std::cerr << "single_token_forward_model: capped request context to measured VRAM budget"
              << " available_mib=" << (available_bytes / kBytesPerMiB)
              << " requested_mib=" << (*requested_bytes / kBytesPerMiB)
              << " capped_mib=" << ((capped_bytes.has_value() ? *capped_bytes : 0) / kBytesPerMiB)
              << " requested_max_tokens=" << base_config.max_tokens
              << " capped_max_tokens=" << capped.max_tokens
              << " requested_attention_total_pages=" << base_config.attention_total_pages
              << " capped_attention_total_pages=" << capped.attention_total_pages
              << "\n";
  }

  return capped;
}

bool CopyDeviceLogitsRow(
    const DeviceTensorFp32& source_row,
    std::size_t row_index,
    DeviceTensorFp32* destination) {
  if (!source_row.valid() ||
      destination == nullptr ||
      !destination->valid() ||
      source_row.shape().size() != 2 ||
      destination->shape().size() != 2 ||
      source_row.shape()[0] != 1 ||
      source_row.shape()[1] != destination->shape()[1] ||
      row_index >= destination->shape()[0]) {
    return false;
  }
  const std::size_t row_width = source_row.shape()[1];
  return cudaMemcpy(
             destination->data() + (row_index * row_width),
             source_row.data(),
             row_width * sizeof(float),
             cudaMemcpyDeviceToDevice) == cudaSuccess;
}

void AppendHostRow(const std::vector<float>& row, std::vector<float>* output) {
  if (output == nullptr) {
    return;
  }
  output->insert(output->end(), row.begin(), row.end());
}

void AppendCapturedLayers(
    const std::vector<CapturedLayerOutput>& step_layers,
    std::vector<CapturedLayerOutput>* output_layers) {
  if (output_layers == nullptr) {
    return;
  }
  for (const CapturedLayerOutput& step_layer : step_layers) {
    auto existing = std::find_if(
        output_layers->begin(),
        output_layers->end(),
        [&](const CapturedLayerOutput& candidate) {
          return candidate.layer_index == step_layer.layer_index;
        });
    if (existing == output_layers->end()) {
      CapturedLayerOutput appended;
      appended.layer_index = step_layer.layer_index;
      appended.hidden = step_layer.hidden;
      output_layers->push_back(std::move(appended));
      continue;
    }
    existing->hidden.insert(
        existing->hidden.end(),
        step_layer.hidden.begin(),
        step_layer.hidden.end());
  }
}

bool ContainsName(
    const std::vector<GlobalTensorBinding>& bindings,
    ModelGlobalRole role,
    const std::string& tensor_name) {
  for (const GlobalTensorBinding& binding : bindings) {
    if (binding.role == role && binding.tensor_name == tensor_name) {
      return true;
    }
  }
  return false;
}

template <typename DescriptorT, typename LookupFn>
const DescriptorT* FindGlobalDescriptorByCandidates(
    const std::vector<GlobalTensorBinding>& bindings,
    ModelGlobalRole role,
    LookupFn lookup,
    const std::vector<std::string>& candidates) {
  for (const GlobalTensorBinding& binding : bindings) {
    if (binding.role != role) {
      continue;
    }
    if (const DescriptorT* descriptor = lookup(binding.tensor_name); descriptor != nullptr) {
      return descriptor;
    }
  }
  for (const std::string& candidate : candidates) {
    if (const DescriptorT* descriptor = lookup(candidate); descriptor != nullptr) {
      return descriptor;
    }
  }
  return nullptr;
}

std::size_t MaxLayerIndex(const std::vector<LayerScheduleEntry>& layers) {
  std::size_t max_index = 0;
  for (const LayerScheduleEntry& layer : layers) {
    max_index = std::max(max_index, layer.layer_index);
  }
  return max_index;
}

std::size_t RequiredMambaConvStateElems(const SingleTokenForwardConfig& config) {
  const std::size_t conv_dim =
      config.mamba_intermediate_size + (2 * config.mamba_n_groups * config.mamba_state_size);
  return conv_dim * config.mamba_conv_kernel_size;
}

std::size_t RequiredMambaStateElems(const SingleTokenForwardConfig& config) {
  return config.mamba_num_heads * config.mamba_head_dim * config.mamba_state_size;
}

std::vector<float> CopyTensorToHost(const DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

bool AllFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool ContainsTokenId(const std::vector<std::int32_t>& token_ids, std::int32_t token_id) {
  return std::find(token_ids.begin(), token_ids.end(), token_id) != token_ids.end();
}

std::optional<std::int32_t> ArgMaxTokenId(
    const std::vector<float>& logits,
    std::size_t row_index,
    std::size_t vocab_size) {
  const std::size_t row_offset = row_index * vocab_size;
  if (vocab_size == 0 || row_offset + vocab_size > logits.size()) {
    return std::nullopt;
  }
  const auto row_begin = logits.begin() + static_cast<std::ptrdiff_t>(row_offset);
  const auto row_end = row_begin + static_cast<std::ptrdiff_t>(vocab_size);
  const auto max_it = std::max_element(row_begin, row_end);
  if (max_it == row_end) {
    return std::nullopt;
  }
  return static_cast<std::int32_t>(std::distance(row_begin, max_it));
}

}  // namespace

SingleTokenForwardConfig KnownNemotron3Super120BA12BConfig() {
  SingleTokenForwardConfig config;
  config.hidden_size = 4096;
  config.total_layer_count = 88;
  config.vocab_size = 131072;
  config.max_tokens = 1;

  config.attention_head_count = 32;
  config.attention_kv_head_count = 2;
  config.attention_head_dim = 128;
  config.attention_tokens_per_page = 16;

  config.mamba_intermediate_size = 8192;
  config.mamba_num_heads = 128;
  config.mamba_head_dim = 64;
  config.mamba_state_size = 128;
  config.mamba_n_groups = 8;
  config.mamba_conv_kernel_size = 4;

  config.moe_latent_size = 1024;
  config.routed_expert_intermediate_size = 2688;
  config.shared_expert_intermediate_size = 5376;
  config.n_routed_experts = 512;
  config.experts_per_token = 22;
  config.expert_n_group = 1;
  config.expert_topk_group = 1;

  config.layer_norm_epsilon = 1.0e-5f;
  config.mamba_time_step_min = 1.0e-3f;
  config.routed_scaling_factor = 5.0f;
  config.norm_topk_prob = true;
  return config;
}

SingleTokenForwardConfig KnownNemotron3Nano30BA3BConfig() {
  SingleTokenForwardConfig config;
  config.hidden_size = 2688;
  config.total_layer_count = 52;
  config.vocab_size = 131072;
  config.max_tokens = 1;

  config.attention_head_count = 32;
  config.attention_kv_head_count = 2;
  config.attention_head_dim = 128;
  config.attention_tokens_per_page = 16;

  // The runtime uses the Mamba inner width for state layout and execution planning.
  config.mamba_intermediate_size = 64 * 64;
  config.mamba_num_heads = 64;
  config.mamba_head_dim = 64;
  config.mamba_state_size = 128;
  config.mamba_n_groups = 8;
  config.mamba_conv_kernel_size = 4;

  // Nano's MoE path is a direct MLP, so routed outputs stay in hidden-size space.
  config.moe_latent_size = 2688;
  config.routed_expert_intermediate_size = 1856;
  config.shared_expert_intermediate_size = 3712;
  config.n_routed_experts = 128;
  config.experts_per_token = 6;
  config.expert_n_group = 1;
  config.expert_topk_group = 1;

  config.layer_norm_epsilon = 1.0e-5f;
  config.mamba_time_step_min = 1.0e-3f;
  config.routed_scaling_factor = 2.5f;
  config.norm_topk_prob = true;
  return config;
}

std::optional<SingleTokenForwardPlan> BuildSingleTokenForwardPlan(
    const ModelSchedule& schedule,
    const SingleTokenForwardConfig& config) {
  if (!schedule.valid() ||
      config.hidden_size == 0 ||
      config.total_layer_count == 0 ||
      config.vocab_size == 0 ||
      config.max_tokens == 0 ||
      config.attention_head_count == 0 ||
      config.attention_kv_head_count == 0 ||
      config.attention_head_dim == 0 ||
      config.attention_tokens_per_page == 0 ||
      config.mamba_intermediate_size == 0 ||
      config.mamba_num_heads == 0 ||
      config.mamba_head_dim == 0 ||
      config.mamba_state_size == 0 ||
      config.mamba_n_groups == 0 ||
      config.mamba_conv_kernel_size == 0 ||
      config.moe_latent_size == 0 ||
      config.routed_expert_intermediate_size == 0 ||
      config.shared_expert_intermediate_size == 0 ||
      config.n_routed_experts == 0 ||
      config.experts_per_token == 0 ||
      config.expert_n_group == 0 ||
      config.expert_topk_group == 0 ||
      config.layer_norm_epsilon <= 0.0f ||
      config.mamba_time_step_min <= 0.0f) {
    return std::nullopt;
  }

  const auto& layers = schedule.ordered_layers();
  if (layers.empty()) {
    return std::nullopt;
  }
  const std::size_t max_layer_index = MaxLayerIndex(layers);
  if (config.total_layer_count <= max_layer_index) {
    return std::nullopt;
  }

  SingleTokenForwardPlan plan;
  std::size_t mamba_conv_offset = 0;
  std::size_t mamba_state_offset = 0;
  const std::size_t per_layer_conv_state = RequiredMambaConvStateElems(config);
  const std::size_t per_layer_ssm_state = RequiredMambaStateElems(config);

  for (const LayerScheduleEntry& layer : layers) {
    const bool is_attention = layer.has_attention;
    const bool is_mamba = layer.has_mamba;
    const bool is_expert = layer.has_router || layer.has_routed_experts || layer.has_shared_experts;
    const std::size_t family_count =
        static_cast<std::size_t>(is_attention) +
        static_cast<std::size_t>(is_mamba) +
        static_cast<std::size_t>(is_expert);
    if (family_count != 1) {
      return std::nullopt;
    }

    ForwardLayerPlanEntry entry;
    entry.layer_index = layer.layer_index;
    if (is_attention) {
      entry.kind = ForwardLayerKind::kAttention;
      ++plan.attention_layer_count;
    } else if (is_mamba) {
      entry.kind = ForwardLayerKind::kMamba;
      entry.mamba_conv_state_offset_elems = mamba_conv_offset;
      entry.mamba_state_offset_elems = mamba_state_offset;
      mamba_conv_offset += per_layer_conv_state;
      mamba_state_offset += per_layer_ssm_state;
      ++plan.mamba_layer_count;
    } else {
      entry.kind = ForwardLayerKind::kExpert;
      ++plan.expert_layer_count;
    }
    plan.layers.push_back(entry);
  }

  plan.request_config.hidden_size = config.hidden_size;
  plan.request_config.max_tokens = config.max_tokens;
  plan.request_config.scratch_tokens = config.max_tokens;
  plan.request_config.mamba_conv_state_bytes_fp32 = mamba_conv_offset * sizeof(float);
  plan.request_config.mamba_state_bytes_fp32 = mamba_state_offset * sizeof(float);
  if (plan.attention_layer_count != 0) {
    plan.request_config.attention_kv_cache.layer_count = max_layer_index + 1;
    plan.request_config.attention_kv_cache.kv_head_count = config.attention_kv_head_count;
    plan.request_config.attention_kv_cache.head_dim = config.attention_head_dim;
    plan.request_config.attention_kv_cache.tokens_per_page = config.attention_tokens_per_page;
    plan.request_config.attention_kv_cache.dtype = KvCacheDataType::kBf16;
    plan.request_config.attention_total_pages =
        plan.request_config.attention_kv_cache.layer_count *
        RequiredPagesForTokens(plan.request_config.attention_kv_cache, config.max_tokens);
  }

  plan.valid = !plan.layers.empty();
  if (!plan.valid) {
    return std::nullopt;
  }
  return plan;
}

struct SingleTokenForwardModel::Impl {
  struct LayerEntry {
    ForwardLayerPlanEntry plan;
    std::unique_ptr<AttentionLayerSlice> attention_slice;
    std::unique_ptr<MambaLayerSlice> mamba_slice;
    std::unique_ptr<ExpertLayerSlice> expert_slice;
  };

  SingleTokenForwardConfig config;
  SingleTokenForwardPlan plan;
  const EmbeddingDescriptor* embedding = nullptr;
  const KernelTensorDescriptor* final_norm = nullptr;
  const GemmDescriptor* lm_head = nullptr;
  std::unique_ptr<CublasLtHandle> cublas;
  std::unique_ptr<CudnnHandle> cudnn;
  std::unique_ptr<GemmHeuristicCache> heuristic_cache;
  PrefixCache* prefix_cache = nullptr;
  std::unique_ptr<DeviceEmbeddingTableFp32> embedding_table;
  std::unique_ptr<UploadedLinearOp> lm_head_op;
  std::unique_ptr<DeviceTensorFp32> final_norm_weight;
  std::optional<std::size_t> post_weight_free_vram_bytes;
  std::size_t vram_reserve_bytes = 0;
  std::vector<LayerEntry> layers;
};

std::unique_ptr<SingleTokenForwardModel> SingleTokenForwardModel::Create(
    RuntimeEnvironment& environment,
    const SingleTokenForwardConfig& config) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const std::size_t vram_reserve_bytes = ForwardVramReserveBytes();
  const auto debug_fail = [&](const std::string& message) -> std::unique_ptr<SingleTokenForwardModel> {
    if (debug) {
      std::cerr << "single_token_forward_model: Create failed: " << message << "\n";
    }
    return nullptr;
  };
  {
    cudaError_t mem_info_status = cudaSuccess;
    const auto mem_info = QueryCudaMemInfo(&mem_info_status);
    if (mem_info.has_value()) {
      if (debug) {
        LogCudaMemInfo("initial_vram", *mem_info, vram_reserve_bytes);
      }
    } else if (debug) {
      std::cerr << "single_token_forward_model: initial cudaMemGetInfo failed: "
                << cudaGetErrorString(mem_info_status) << "\n";
    }
  }
  if (!environment.has_model_schedule() ||
      !environment.has_kernel_catalog() ||
      !environment.has_gemm_catalog() ||
      !environment.has_embedding_catalog()) {
    return debug_fail("runtime environment missing required catalogs");
  }
  const ModelSchedule& schedule = *environment.model_schedule();
  const auto plan = BuildSingleTokenForwardPlan(schedule, config);
  if (!plan.has_value()) {
    return debug_fail("forward plan construction failed");
  }

  const KernelCatalog& kernel_catalog = *environment.kernel_catalog();
  const GemmCatalog& gemm_catalog = *environment.gemm_catalog();
  const EmbeddingCatalog& embedding_catalog = *environment.embedding_catalog();

  const EmbeddingDescriptor* embedding = FindGlobalDescriptorByCandidates<EmbeddingDescriptor>(
      schedule.global_bindings(),
      ModelGlobalRole::kEmbedding,
      [&](const std::string& name) { return embedding_catalog.FindDescriptor(name); },
      {"backbone.embeddings.weight", "embeddings.weight"});
  const GemmDescriptor* lm_head = FindGlobalDescriptorByCandidates<GemmDescriptor>(
      schedule.global_bindings(),
      ModelGlobalRole::kLogits,
      [&](const std::string& name) { return gemm_catalog.FindDescriptor(name); },
      {"lm_head.weight", "logits.weight", "output.weight"});
  const KernelTensorDescriptor* final_norm = FindGlobalDescriptorByCandidates<KernelTensorDescriptor>(
      schedule.global_bindings(),
      ModelGlobalRole::kFinalNorm,
      [&](const std::string& name) { return kernel_catalog.FindTensor(name); },
      {"backbone.norm_f.weight",
       "norm_f.weight",
       "backbone.final_norm.weight",
       "final_norm.weight"});
  if (embedding == nullptr || lm_head == nullptr) {
    return debug_fail("embedding or lm_head descriptor missing");
  }
  if (embedding->embedding_dim != config.hidden_size ||
      embedding->vocab_size != config.vocab_size ||
      lm_head->input_cols != config.hidden_size ||
      lm_head->output_rows != config.vocab_size) {
    return debug_fail("embedding or lm_head shape mismatch");
  }
  if (final_norm != nullptr &&
      (final_norm->logical_shape.size() != 1 || final_norm->logical_shape.front() != config.hidden_size)) {
    return debug_fail("final norm shape mismatch");
  }

  auto cublas = CublasLtHandle::Create();
  auto heuristic_cache = std::make_unique<GemmHeuristicCache>();
  if (!cublas || !cublas->valid()) {
    return debug_fail("cuBLASLt handle creation failed");
  }

  std::unique_ptr<CudnnHandle> cudnn;
  if (plan->attention_layer_count != 0) {
    cudnn = CudnnHandle::Create();
    if (!cudnn) {
      return debug_fail("attention backend handle creation failed");
    }
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->plan = *plan;
  impl->embedding = embedding;
  impl->final_norm = final_norm;
  impl->lm_head = lm_head;
  impl->cublas = std::move(cublas);
  impl->cudnn = std::move(cudnn);
  impl->heuristic_cache = std::move(heuristic_cache);
  impl->prefix_cache = &environment.prefix_cache();
  impl->vram_reserve_bytes = vram_reserve_bytes;
  impl->embedding_table = DeviceEmbeddingTableFp32::Upload(*embedding);
  impl->lm_head_op = UploadedLinearOp::Create(*lm_head);
  if (!impl->embedding_table || !impl->embedding_table->valid() ||
      !impl->lm_head_op || !impl->lm_head_op->valid()) {
    return debug_fail("embedding table or lm_head materialization failed");
  }
  if (final_norm != nullptr) {
    impl->final_norm_weight = UploadVectorWeightToDeviceFp32(*final_norm);
    if (!impl->final_norm_weight || !impl->final_norm_weight->valid()) {
      return debug_fail("final norm upload failed");
    }
  }
  impl->layers.reserve(plan->layers.size());

  for (const ForwardLayerPlanEntry& plan_entry : plan->layers) {
    const LayerScheduleEntry* layer = schedule.FindLayer(plan_entry.layer_index);
    if (layer == nullptr) {
      return debug_fail("missing schedule layer " + std::to_string(plan_entry.layer_index));
    }

    Impl::LayerEntry layer_entry;
    layer_entry.plan = plan_entry;
    switch (plan_entry.kind) {
      case ForwardLayerKind::kAttention: {
        const auto bindings = BuildAttentionLayerBindings(*layer, kernel_catalog, gemm_catalog);
        if (!bindings.has_value()) {
          return debug_fail("attention bindings failed for layer " + std::to_string(plan_entry.layer_index));
        }
        AttentionLayerConfig attention_config;
        attention_config.layer_index = plan_entry.layer_index;
        attention_config.hidden_size = config.hidden_size;
        attention_config.query_head_count = config.attention_head_count;
        attention_config.kv_head_count = config.attention_kv_head_count;
        attention_config.head_dim = config.attention_head_dim;
        attention_config.rms_epsilon = config.layer_norm_epsilon;
        layer_entry.attention_slice = AttentionLayerSlice::Create(attention_config, *bindings);
        if (!layer_entry.attention_slice || !layer_entry.attention_slice->valid()) {
          return debug_fail("attention slice creation failed for layer " + std::to_string(plan_entry.layer_index));
        }
        break;
      }
      case ForwardLayerKind::kMamba: {
        const auto bindings = BuildMambaLayerBindings(*layer, kernel_catalog, gemm_catalog);
        if (!bindings.has_value()) {
          return debug_fail("mamba bindings failed for layer " + std::to_string(plan_entry.layer_index));
        }
        MambaLayerConfig mamba_config;
        mamba_config.layer_index = plan_entry.layer_index;
        mamba_config.hidden_size = config.hidden_size;
        mamba_config.intermediate_size = config.mamba_intermediate_size;
        mamba_config.num_heads = config.mamba_num_heads;
        mamba_config.head_dim = config.mamba_head_dim;
        mamba_config.state_size = config.mamba_state_size;
        mamba_config.n_groups = config.mamba_n_groups;
        mamba_config.conv_kernel_size = config.mamba_conv_kernel_size;
        mamba_config.conv_state_offset_elems = plan_entry.mamba_conv_state_offset_elems;
        mamba_config.ssm_state_offset_elems = plan_entry.mamba_state_offset_elems;
        mamba_config.input_rms_epsilon = config.layer_norm_epsilon;
        mamba_config.mixer_rms_epsilon = config.layer_norm_epsilon;
        mamba_config.time_step_min = config.mamba_time_step_min;
        layer_entry.mamba_slice = MambaLayerSlice::Create(mamba_config, *bindings);
        if (!layer_entry.mamba_slice || !layer_entry.mamba_slice->valid()) {
          return debug_fail("mamba slice creation failed for layer " + std::to_string(plan_entry.layer_index));
        }
        break;
      }
      case ForwardLayerKind::kExpert: {
        const auto bindings = BuildExpertLayerBindings(
            *layer,
            kernel_catalog,
            gemm_catalog,
            config.n_routed_experts);
        if (!bindings.has_value()) {
          return debug_fail("expert bindings failed for layer " + std::to_string(plan_entry.layer_index));
        }
        ExpertLayerConfig expert_config;
        expert_config.layer_index = plan_entry.layer_index;
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
        layer_entry.expert_slice = ExpertLayerSlice::Create(expert_config, *bindings);
        if (!layer_entry.expert_slice || !layer_entry.expert_slice->valid()) {
          return debug_fail("expert slice creation failed for layer " + std::to_string(plan_entry.layer_index));
        }
        break;
      }
    }

    impl->layers.push_back(std::move(layer_entry));
  }

  {
    cudaError_t mem_info_status = cudaSuccess;
    const auto mem_info = QueryCudaMemInfo(&mem_info_status);
    if (mem_info.has_value()) {
      impl->post_weight_free_vram_bytes = mem_info->free_bytes;
      if (debug) {
        LogCudaMemInfo("post_weight_vram", *mem_info, impl->vram_reserve_bytes);
      }
    } else if (debug) {
      std::cerr << "single_token_forward_model: post-weight cudaMemGetInfo failed: "
                << cudaGetErrorString(mem_info_status) << "\n";
    }
  }

  return std::unique_ptr<SingleTokenForwardModel>(new SingleTokenForwardModel(std::move(impl)));
}

SingleTokenForwardModel::SingleTokenForwardModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SingleTokenForwardModel::SingleTokenForwardModel(SingleTokenForwardModel&&) noexcept = default;
SingleTokenForwardModel& SingleTokenForwardModel::operator=(SingleTokenForwardModel&&) noexcept = default;
SingleTokenForwardModel::~SingleTokenForwardModel() = default;

bool SingleTokenForwardModel::valid() const {
  return impl_ != nullptr &&
         impl_->embedding != nullptr &&
         impl_->lm_head != nullptr &&
         impl_->cublas != nullptr &&
         impl_->cublas->valid() &&
         impl_->heuristic_cache != nullptr &&
         impl_->plan.valid &&
         impl_->layers.size() == impl_->plan.layers.size();
}

const SingleTokenForwardConfig& SingleTokenForwardModel::config() const {
  return impl_->config;
}

const SingleTokenForwardPlan& SingleTokenForwardModel::plan() const {
  return impl_->plan;
}

std::unique_ptr<RequestExecutionContext> SingleTokenForwardModel::CreateRequestContext() const {
  if (!valid()) {
    return nullptr;
  }
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  RequestExecutionConfig request_config = impl_->plan.request_config;
  if (impl_->post_weight_free_vram_bytes.has_value()) {
    const auto capped_config = BuildMeasuredBudgetRequestConfig(
        request_config,
        *impl_->post_weight_free_vram_bytes,
        impl_->vram_reserve_bytes,
        debug);
    if (!capped_config.has_value()) {
      if (debug) {
        std::cerr << "single_token_forward_model: request context sizing failed against measured VRAM"
                  << " free_vram_mib=" << (*impl_->post_weight_free_vram_bytes / kBytesPerMiB)
                  << " reserve_mib=" << (impl_->vram_reserve_bytes / kBytesPerMiB)
                  << "\n";
      }
      return nullptr;
    }
    request_config = *capped_config;
  }
  return RequestExecutionContext::Create(request_config);
}

bool SingleTokenForwardModel::RunSingleToken(
    std::int32_t token_id,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index) const {
  return RunTokens(
      &token_id,
      1,
      request_context,
      logits,
      capture_layer_indices,
      trace,
      stop_layer_index,
      true);
}

bool SingleTokenForwardModel::RunPrefill(
    const std::int32_t* token_ids,
    std::size_t token_count,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index) const {
  return RunTokens(
      token_ids,
      token_count,
      request_context,
      logits,
      capture_layer_indices,
      trace,
      stop_layer_index,
      true);
}

bool SingleTokenForwardModel::ContinueSingleToken(
    std::int32_t token_id,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index) const {
  return RunTokens(
      &token_id,
      1,
      request_context,
      logits,
      capture_layer_indices,
      trace,
      stop_layer_index,
      false);
}

bool SingleTokenForwardModel::ContinuePrefill(
    const std::int32_t* token_ids,
    std::size_t token_count,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index) const {
  return RunTokens(
      token_ids,
      token_count,
      request_context,
      logits,
      capture_layer_indices,
      trace,
      stop_layer_index,
      false);
}

bool SingleTokenForwardModel::RunGreedyDecode(
    const std::int32_t* prompt_token_ids,
    std::size_t prompt_token_count,
    const GreedyDecodeConfig& decode_config,
    RequestExecutionContext& request_context,
    GreedyDecodeResult* result) const {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (result == nullptr) {
    if (debug) {
      std::cerr << "single_token_forward_model: greedy decode result output is required\n";
    }
    return false;
  }
  result->generated_token_ids.clear();
  result->hit_eos = false;
  result->hit_capacity_limit = false;

  if (!valid() ||
      prompt_token_ids == nullptr ||
      prompt_token_count == 0 ||
      !request_context.valid() ||
      request_context.config().max_tokens < prompt_token_count) {
    if (debug) {
      std::cerr << "single_token_forward_model: invalid greedy decode inputs"
                << " model_valid=" << valid()
                << " have_prompt=" << (prompt_token_ids != nullptr)
                << " prompt_token_count=" << prompt_token_count
                << " request_valid=" << request_context.valid()
                << " request_max_tokens=" << request_context.config().max_tokens
                << "\n";
    }
    return false;
  }

  auto prompt_logits = DeviceTensorFp32::Create({prompt_token_count, impl_->config.vocab_size});
  if (prompt_logits == nullptr || !prompt_logits->valid()) {
    std::cerr << "single_token_forward_model: prompt logits buffer allocation failed\n";
    return false;
  }
  if (!RunPrefill(prompt_token_ids, prompt_token_count, request_context, prompt_logits.get())) {
    std::cerr << "single_token_forward_model: greedy decode prefill failed\n";
    return false;
  }

  std::vector<float> logits_host = CopyTensorToHost(*prompt_logits);
  if (logits_host.empty() || !AllFinite(logits_host)) {
    std::cerr << "single_token_forward_model: greedy decode prompt logits invalid\n";
    return false;
  }
  auto next_token = ArgMaxTokenId(logits_host, prompt_token_count - 1, impl_->config.vocab_size);
  if (!next_token.has_value()) {
    std::cerr << "single_token_forward_model: failed to select prompt decode token\n";
    return false;
  }

  return ContinueGreedyDecode(*next_token, decode_config, request_context, result);
}

bool SingleTokenForwardModel::RunGreedyConversationTurn(
    const SerializedPromptIdentity& identity,
    const std::string& conversation_id,
    const GreedyDecodeConfig& decode_config,
    RequestExecutionContext& request_context,
    GreedyDecodeResult* result,
    std::size_t* matched_token_count) const {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (matched_token_count != nullptr) {
    *matched_token_count = 0;
  }
  if (result == nullptr) {
    if (debug) {
      std::cerr << "single_token_forward_model: conversation-turn decode result output is required\n";
    }
    return false;
  }
  if (!valid() ||
      identity.token_ids.empty() ||
      conversation_id.empty() ||
      !request_context.valid() ||
      request_context.config().max_tokens < identity.token_ids.size()) {
    if (debug) {
      std::cerr << "single_token_forward_model: invalid conversation-turn inputs"
                << " model_valid=" << valid()
                << " prompt_token_count=" << identity.token_ids.size()
                << " conversation_id_empty=" << conversation_id.empty()
                << " request_valid=" << request_context.valid()
                << " request_max_tokens=" << request_context.config().max_tokens
                << "\n";
    }
    return false;
  }

  std::size_t matched_prefix_tokens = 0;
  const std::int32_t* prefill_token_ids = identity.token_ids.data();
  std::size_t prefill_token_count = identity.token_ids.size();
  bool resumed_from_cache = false;
  bool exact_cache_hit = false;
  std::vector<float> logits_host;

  if (impl_->prefix_cache != nullptr && impl_->prefix_cache->enabled()) {
    CacheLookupRequest lookup_request;
    lookup_request.identity = identity;
    lookup_request.conversation_id = conversation_id;
    const CacheMatch match = impl_->prefix_cache->Lookup(lookup_request);
    if (match.hit() &&
        match.matched_token_count > 0 &&
        impl_->prefix_cache->RestoreMatchState(match, request_context)) {
      if (match.matched_token_count < identity.token_ids.size()) {
        resumed_from_cache = true;
        matched_prefix_tokens = match.matched_token_count;
        prefill_token_ids += matched_prefix_tokens;
        prefill_token_count -= matched_prefix_tokens;
      } else if (match.matched_token_count == identity.token_ids.size()) {
        const auto cached_boundary_logits = impl_->prefix_cache->CopyBoundaryLogits(match.node_id);
        if (cached_boundary_logits.has_value() &&
            cached_boundary_logits->size() == impl_->config.vocab_size &&
            AllFinite(*cached_boundary_logits)) {
          exact_cache_hit = true;
          matched_prefix_tokens = match.matched_token_count;
          logits_host = std::move(*cached_boundary_logits);
        }
      }
    }
  }
  if (matched_token_count != nullptr) {
    *matched_token_count = matched_prefix_tokens;
  }

  if (!exact_cache_hit) {
    auto prompt_logits = DeviceTensorFp32::Create({prefill_token_count, impl_->config.vocab_size});
    if (prompt_logits == nullptr || !prompt_logits->valid()) {
      std::cerr << "single_token_forward_model: conversation-turn prompt logits buffer allocation failed\n";
      return false;
    }

    const bool prefill_ok =
        resumed_from_cache
            ? ContinuePrefill(prefill_token_ids, prefill_token_count, request_context, prompt_logits.get())
            : RunPrefill(identity.token_ids.data(), identity.token_ids.size(), request_context, prompt_logits.get());
    if (!prefill_ok) {
      std::cerr << "single_token_forward_model: conversation-turn prefill failed\n";
      return false;
    }

    logits_host = CopyTensorToHost(*prompt_logits);
    if (logits_host.empty() || !AllFinite(logits_host)) {
      std::cerr << "single_token_forward_model: conversation-turn prompt logits invalid\n";
      return false;
    }
  }

  if (impl_->prefix_cache != nullptr && impl_->prefix_cache->enabled()) {
    impl_->prefix_cache->PublishConversationHeadSnapshot(
        conversation_id,
        ConversationCheckpointKind::kPromptHead,
        identity,
        request_context,
        conversation_id + "/prompt",
        &logits_host);
  }
  const std::size_t logits_row_index = exact_cache_hit ? 0 : (prefill_token_count - 1);
  auto next_token = ArgMaxTokenId(logits_host, logits_row_index, impl_->config.vocab_size);
  if (!next_token.has_value()) {
    std::cerr << "single_token_forward_model: failed to select conversation-turn decode token\n";
    return false;
  }
  std::vector<float> committed_boundary_logits;
  if (!ContinueGreedyDecode(*next_token, decode_config, request_context, result, &committed_boundary_logits)) {
    return false;
  }
  if (committed_boundary_logits.empty()) {
    committed_boundary_logits = logits_host;
  }

  if (impl_->prefix_cache != nullptr && impl_->prefix_cache->enabled()) {
    SerializedPromptIdentity committed_identity = identity;
    committed_identity.token_ids.insert(
        committed_identity.token_ids.end(),
        result->generated_token_ids.begin(),
        result->generated_token_ids.end());
    impl_->prefix_cache->PublishConversationHeadSnapshot(
        conversation_id,
        ConversationCheckpointKind::kCommittedHead,
        committed_identity,
        request_context,
        conversation_id + "/committed",
        &committed_boundary_logits);
  }

  return true;
}

bool SingleTokenForwardModel::ContinueGreedyDecode(
    std::int32_t first_token_id,
    const GreedyDecodeConfig& decode_config,
    RequestExecutionContext& request_context,
    GreedyDecodeResult* result,
    std::vector<float>* final_boundary_logits) const {
  result->generated_token_ids.clear();
  result->hit_eos = false;
  result->hit_capacity_limit = false;
  if (final_boundary_logits != nullptr) {
    final_boundary_logits->clear();
  }
  if (decode_config.max_new_tokens == 0) {
    return true;
  }

  auto step_logits = DeviceTensorFp32::Create({1, impl_->config.vocab_size});
  if (step_logits == nullptr || !step_logits->valid()) {
    std::cerr << "single_token_forward_model: decode-step logits buffer allocation failed\n";
    return false;
  }

  std::int32_t token_id = first_token_id;
  for (std::size_t step = 0; step < decode_config.max_new_tokens; ++step) {
    if (request_context.sequence_length() >= request_context.config().max_tokens) {
      result->hit_capacity_limit = true;
      break;
    }

    result->generated_token_ids.push_back(token_id);
    if (!ContinueSingleToken(token_id, request_context, step_logits.get())) {
      std::cerr << "single_token_forward_model: greedy decode continuation failed at step "
                << step << "\n";
      return false;
    }

    std::vector<float> logits_host = CopyTensorToHost(*step_logits);
    if (logits_host.empty() || !AllFinite(logits_host)) {
      std::cerr << "single_token_forward_model: greedy decode step logits invalid at step "
                << step << "\n";
      return false;
    }
    if (final_boundary_logits != nullptr) {
      *final_boundary_logits = logits_host;
    }
    if (ContainsTokenId(decode_config.eos_token_ids, token_id)) {
      result->hit_eos = true;
      break;
    }
    const auto next_token = ArgMaxTokenId(logits_host, 0, impl_->config.vocab_size);
    if (!next_token.has_value()) {
      std::cerr << "single_token_forward_model: failed to select decode token at step "
                << step << "\n";
      return false;
    }
    token_id = *next_token;
  }

  if (!result->hit_eos &&
      result->generated_token_ids.size() < decode_config.max_new_tokens &&
      request_context.sequence_length() >= request_context.config().max_tokens) {
    result->hit_capacity_limit = true;
  }
  return true;
}

bool SingleTokenForwardModel::RunTokens(
    const std::int32_t* token_ids,
    std::size_t token_count,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index,
    bool reset_request_state) const {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;

  const bool model_valid = valid();
  const bool have_tokens = token_ids != nullptr;
  const bool nonzero_token_count = token_count != 0;
  const bool request_valid = request_context.valid();
  const bool hidden_size_match = request_context.config().hidden_size == impl_->config.hidden_size;
  const std::size_t prior_sequence_length =
      reset_request_state ? 0 : request_context.sequence_length();
  const std::size_t prior_decode_position =
      reset_request_state ? 0 : request_context.decode_position();
  const bool continuation_position_ok =
      reset_request_state || prior_sequence_length == prior_decode_position;
  const std::size_t total_sequence_length =
      continuation_position_ok ? (prior_sequence_length + token_count) : token_count;
  const bool token_capacity_ok =
      continuation_position_ok && request_context.config().max_tokens >= total_sequence_length;
  const bool have_logits = logits != nullptr;
  const bool logits_valid = have_logits && logits->valid();
  const std::vector<std::size_t> expected_logits_shape = {token_count, impl_->config.vocab_size};
  const bool logits_shape_ok = logits_valid && logits->shape() == expected_logits_shape;

  if (!model_valid ||
      !have_tokens ||
      !nonzero_token_count ||
      !request_valid ||
      !hidden_size_match ||
      !continuation_position_ok ||
      !token_capacity_ok ||
      !have_logits ||
      !logits_valid ||
      !logits_shape_ok) {
    std::cerr << "single_token_forward_model: invalid RunPrefill arguments or state"
              << " model_valid=" << model_valid
              << " have_tokens=" << have_tokens
              << " nonzero_token_count=" << nonzero_token_count
              << " request_valid=" << request_valid
              << " hidden_size_match=" << hidden_size_match
              << " continuation_position_ok=" << continuation_position_ok
              << " token_capacity_ok=" << token_capacity_ok
              << " have_logits=" << have_logits
              << " logits_valid=" << logits_valid
              << " logits_shape_ok=" << logits_shape_ok
              << " request_hidden_size=" << request_context.config().hidden_size
              << " model_hidden_size=" << impl_->config.hidden_size
              << " request_max_tokens=" << request_context.config().max_tokens
              << " prior_sequence_length=" << prior_sequence_length
              << " prior_decode_position=" << prior_decode_position
              << " total_sequence_length=" << total_sequence_length
              << " token_count=" << token_count;
    if (have_logits && logits_valid) {
      std::cerr << " logits_shape=[";
      for (std::size_t i = 0; i < logits->shape().size(); ++i) {
        if (i != 0) {
          std::cerr << ",";
        }
        std::cerr << logits->shape()[i];
      }
      std::cerr << "]";
    }
    std::cerr << " expected_logits_shape=[" << expected_logits_shape[0]
              << "," << expected_logits_shape[1] << "]\n";
    return false;
  }

  if (token_count > 1 && DecodeConsistentPrefillEnabled()) {
    if (trace != nullptr) {
      trace->embedding_output.clear();
      trace->captured_layers.clear();
      trace->final_hidden.clear();
      trace->final_hidden_normed.clear();
      trace->logits.clear();
    }
    auto step_logits = DeviceTensorFp32::Create({1, impl_->config.vocab_size});
    if (step_logits == nullptr || !step_logits->valid()) {
      std::cerr << "single_token_forward_model: sequential prefill logits buffer allocation failed\n";
      return false;
    }

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
      SingleTokenForwardTrace step_trace;
      SingleTokenForwardTrace* step_trace_ptr = trace != nullptr ? &step_trace : nullptr;
      const bool step_reset = reset_request_state && token_index == 0;
      if (!RunTokens(
              token_ids + token_index,
              1,
              request_context,
              step_logits.get(),
              capture_layer_indices,
              step_trace_ptr,
              stop_layer_index,
              step_reset)) {
        std::cerr << "single_token_forward_model: sequential prefill step failed at token "
                  << token_index << "\n";
        return false;
      }
      if (!CopyDeviceLogitsRow(*step_logits, token_index, logits)) {
        std::cerr << "single_token_forward_model: sequential prefill logits row copy failed at token "
                  << token_index << "\n";
        return false;
      }
      if (trace != nullptr) {
        AppendHostRow(step_trace.embedding_output, &trace->embedding_output);
        AppendCapturedLayers(step_trace.captured_layers, &trace->captured_layers);
        AppendHostRow(step_trace.final_hidden, &trace->final_hidden);
        AppendHostRow(step_trace.final_hidden_normed, &trace->final_hidden_normed);
        AppendHostRow(step_trace.logits, &trace->logits);
      }
    }
    return true;
  }
  if (impl_->plan.attention_layer_count != 0 && impl_->cudnn == nullptr) {
    std::cerr << "single_token_forward_model: attention backend handle unavailable\n";
    return false;
  }
  if (reset_request_state) {
    if (!request_context.ResetForNewRequest()) {
      std::cerr << "single_token_forward_model: request context reset failed\n";
      return false;
    }
  }
  if (impl_->plan.attention_layer_count != 0 &&
      !request_context.EnsureAttentionTokens(total_sequence_length)) {
    std::cerr << "single_token_forward_model: attention page reservation failed\n";
    return false;
  }

  auto hidden = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  auto residual = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  auto scratch = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  DeviceTensorFp32* current = hidden.get();
  DeviceTensorFp32* next = residual.get();
  if (current == nullptr || next == nullptr || scratch == nullptr) {
    std::cerr << "single_token_forward_model: per-run buffers unavailable\n";
    return false;
  }

  if (!impl_->embedding_table || !impl_->embedding_table->valid() ||
      !impl_->lm_head_op || !impl_->lm_head_op->valid()) {
    std::cerr << "single_token_forward_model: cached embeddings or lm_head unavailable\n";
    return false;
  }
  if (impl_->final_norm != nullptr &&
      (!impl_->final_norm_weight || !impl_->final_norm_weight->valid())) {
    std::cerr << "single_token_forward_model: cached final norm weight unavailable\n";
    return false;
  }

  if (!LookupEmbeddingRowsFp32(*impl_->embedding_table, token_ids, token_count, current).has_value()) {
    std::cerr << "single_token_forward_model: embedding lookup failed\n";
    return false;
  }
  if (debug) {
    std::cout << "single_token_forward_model: embedding lookup ok for "
              << token_count << " token(s)\n";
  }
  if (trace != nullptr) {
    trace->embedding_output = CopyTensorToHost(*current);
    if (trace->embedding_output.empty()) {
      return false;
    }
    trace->captured_layers.clear();
    trace->final_hidden.clear();
    trace->final_hidden_normed.clear();
    trace->logits.clear();
  }

  const std::unordered_set<std::size_t> capture_set(
      capture_layer_indices.begin(),
      capture_layer_indices.end());

  for (const Impl::LayerEntry& layer : impl_->layers) {
    if (debug) {
      std::cout << "single_token_forward_model: running layer "
                << layer.plan.layer_index
                << " kind=" << static_cast<int>(layer.plan.kind) << "\n";
    }
    bool ok = false;
    switch (layer.plan.kind) {
      case ForwardLayerKind::kAttention: {
        const bool created = layer.attention_slice != nullptr;
        const bool valid_slice = created && layer.attention_slice->valid();
        const bool ran =
            valid_slice &&
            layer.attention_slice->Run(
                *impl_->cublas,
                *impl_->cudnn,
                impl_->heuristic_cache.get(),
                request_context,
                prior_sequence_length,
                total_sequence_length,
                *current,
                next);
        if (debug && (!created || !valid_slice || !ran)) {
          std::cout << "single_token_forward_model: attention failure"
                    << " created=" << created
                    << " valid=" << valid_slice
                    << " ran=" << ran << "\n";
        }
        ok = created && valid_slice && ran;
        break;
      }
      case ForwardLayerKind::kMamba: {
        const bool created = layer.mamba_slice != nullptr;
        const bool valid_slice = created && layer.mamba_slice->valid();
        const bool ran =
            valid_slice &&
            layer.mamba_slice->Run(
                *impl_->cublas,
                impl_->heuristic_cache.get(),
                request_context,
                *current,
                next);
        if (debug && (!created || !valid_slice || !ran)) {
          std::cout << "single_token_forward_model: mamba failure"
                    << " created=" << created
                    << " valid=" << valid_slice
                    << " ran=" << ran << "\n";
        }
        ok = created && valid_slice && ran;
        break;
      }
      case ForwardLayerKind::kExpert: {
        const bool created = layer.expert_slice != nullptr;
        const bool valid_slice = created && layer.expert_slice->valid();
        const bool ran =
            valid_slice &&
            layer.expert_slice->Run(
                *impl_->cublas,
                impl_->heuristic_cache.get(),
                *current,
                next,
                nullptr);
        if (debug && (!created || !valid_slice || !ran)) {
          std::cout << "single_token_forward_model: expert failure"
                    << " created=" << created
                    << " valid=" << valid_slice
                    << " ran=" << ran << "\n";
        }
        ok = created && valid_slice && ran;
        break;
      }
    }
    if (!ok) {
      std::cerr << "single_token_forward_model: layer "
                << layer.plan.layer_index
                << " kind=" << static_cast<int>(layer.plan.kind)
                << " execution failed\n";
      return false;
    }
    std::swap(current, next);
    if (debug) {
      std::cout << "single_token_forward_model: layer "
                << layer.plan.layer_index << " ok\n";
    }

    if (trace != nullptr && capture_set.find(layer.plan.layer_index) != capture_set.end()) {
      CapturedLayerOutput captured;
      captured.layer_index = layer.plan.layer_index;
      captured.hidden = CopyTensorToHost(*current);
      if (captured.hidden.empty()) {
        std::cerr << "single_token_forward_model: failed to capture layer " << layer.plan.layer_index << "\n";
        return false;
      }
      trace->captured_layers.push_back(std::move(captured));
    }

    if (stop_layer_index.has_value() && layer.plan.layer_index >= *stop_layer_index) {
      break;
    }
  }

  const bool state_commit_ok =
      reset_request_state
          ? request_context.SetSequenceLength(token_count)
          : request_context.AdvanceDecodePosition(token_count);
  if (!state_commit_ok) {
    std::cerr << "single_token_forward_model: request context state commit failed\n";
    return false;
  }

  if (trace != nullptr) {
    trace->final_hidden = CopyTensorToHost(*current);
    if (trace->final_hidden.empty()) {
      std::cerr << "single_token_forward_model: failed to capture final hidden state\n";
      return false;
    }
  }

  const DeviceTensorFp32* logits_input = current;
  if (impl_->final_norm_weight != nullptr) {
    if (!RmsNormFp32(
            *current,
            *impl_->final_norm_weight,
            impl_->config.layer_norm_epsilon,
            scratch.get())) {
      std::cerr << "single_token_forward_model: final RMSNorm failed\n";
      return false;
    }
    logits_input = scratch.get();
  }

  if (trace != nullptr) {
    trace->final_hidden_normed = CopyTensorToHost(*logits_input);
    if (trace->final_hidden_normed.empty()) {
      std::cerr << "single_token_forward_model: failed to capture final normalized hidden state\n";
      return false;
    }
  }

  if (!impl_->lm_head_op->Run(
          *impl_->cublas,
          impl_->heuristic_cache.get(),
          *logits_input,
          logits)) {
    std::cerr << "single_token_forward_model: lm_head projection failed\n";
    return false;
  }
  if (debug) {
    std::cout << "single_token_forward_model: lm_head ok\n";
  }

  if (trace != nullptr) {
    trace->logits = CopyTensorToHost(*logits);
    if (trace->logits.empty()) {
      std::cerr << "single_token_forward_model: failed to capture logits\n";
      return false;
    }
  }
  return true;
}

}  // namespace nemotron
