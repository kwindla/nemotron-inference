#include "nemotron/single_token_forward_model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <atomic>
#include <string>
#include <thread>
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
#include "nemotron/model_cache.h"
#include "nemotron/model_schedule.h"
#include "nemotron/paged_kv_cache.h"
#include "nemotron/primitive_ops.h"
#include "nemotron/runtime_environment.h"
#include "../backend/storage_conversion.h"

namespace nemotron {
namespace {

std::optional<float> ReadTensorScaleHost(const GemmDescriptor& descriptor) {
  if (descriptor.tensor_scale_data == nullptr || descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }
  float value = 0.0f;
  std::memcpy(&value, descriptor.tensor_scale_data, sizeof(float));
  return value;
}

std::optional<float> ReadScalarTensorToHostFp32(const KernelTensorDescriptor& descriptor) {
  if (descriptor.storage_dtype != "fp32" ||
      descriptor.packed_data == nullptr ||
      descriptor.packed_nbytes != sizeof(float)) {
    return std::nullopt;
  }
  float value = 0.0f;
  std::memcpy(&value, descriptor.packed_data, sizeof(float));
  return value;
}

std::optional<float> ReadOptionalScalarTensorToHostFp32(const KernelTensorDescriptor* descriptor) {
  return descriptor == nullptr ? std::nullopt : ReadScalarTensorToHostFp32(*descriptor);
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

std::size_t RequiredMambaConvDim(const SingleTokenForwardConfig& config) {
  return config.mamba_intermediate_size + (2 * config.mamba_n_groups * config.mamba_state_size);
}

std::size_t RequiredMambaConvStateElems(const SingleTokenForwardConfig& config) {
  return RequiredMambaConvDim(config) * config.mamba_conv_kernel_size;
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

std::vector<float> CopyTensorToHost(const DeviceTensorBf16& tensor) {
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  if (!tensor.CopyToHost(host_bf16.data(), host_bf16.size())) {
    return {};
  }
  std::vector<float> host(host_bf16.size(), 0.0f);
  for (std::size_t i = 0; i < host_bf16.size(); ++i) {
    host[i] = __bfloat162float(host_bf16[i]);
  }
  return host;
}

const char* LayerKindName(ForwardLayerKind kind) {
  switch (kind) {
    case ForwardLayerKind::kAttention:
      return "attention";
    case ForwardLayerKind::kMamba:
      return "mamba";
    case ForwardLayerKind::kExpert:
      return "expert";
  }
  return "unknown";
}

double ToMilliseconds(const std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

bool ForwardDebugEnabled() {
  static const bool kDebug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  return kDebug;
}

bool ForwardProfileEnabled() {
  static const bool kProfile = std::getenv("NEMOTRON_FORWARD_PROFILE") != nullptr;
  return kProfile;
}

std::size_t BuildWorkerCount() {
  if (const char* override_value = std::getenv("NEMOTRON_FORWARD_BUILD_THREADS");
      override_value != nullptr) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(override_value, &end, 10);
    if (end != override_value && parsed > 0) {
      return static_cast<std::size_t>(parsed);
    }
  }
  return std::min<std::size_t>(
      4,
      std::max<std::size_t>(1, std::thread::hardware_concurrency()));
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
  plan.request_config.mamba_hidden_size = config.hidden_size;
  plan.request_config.mamba_projection_size =
      config.mamba_intermediate_size + RequiredMambaConvDim(config) + config.mamba_num_heads;
  plan.request_config.mamba_intermediate_size = config.mamba_intermediate_size;
  plan.request_config.mamba_conv_state_bytes_fp32 = mamba_conv_offset * sizeof(float);
  plan.request_config.mamba_state_bytes_fp32 = mamba_state_offset * sizeof(float);
  plan.request_config.expert_selection_capacity = config.max_tokens * config.experts_per_token;
  plan.request_config.expert_intermediate_scratch_numel =
      config.experts_per_token * config.routed_expert_intermediate_size;
  plan.request_config.expert_aux_scratch_numel =
      (4 * config.hidden_size) +
      (3 * config.moe_latent_size) +
      config.shared_expert_intermediate_size +
      config.n_routed_experts;
  if (plan.attention_layer_count != 0) {
    plan.request_config.attention_kv_cache.layer_count = max_layer_index + 1;
    plan.request_config.attention_kv_cache.kv_head_count = config.attention_kv_head_count;
    plan.request_config.attention_kv_cache.head_dim = config.attention_head_dim;
    plan.request_config.attention_kv_cache.tokens_per_page = config.attention_tokens_per_page;
    plan.request_config.attention_kv_cache.dtype = KvCacheDataType::kBf16;
    plan.request_config.attention_total_pages =
        plan.request_config.attention_kv_cache.layer_count *
        RequiredPagesForTokens(plan.request_config.attention_kv_cache, config.max_tokens);
    plan.request_config.attention_query_head_count = config.attention_head_count;
    plan.request_config.attention_head_dim = config.attention_head_dim;
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
  std::unique_ptr<DeviceEmbeddingTableFp32> embedding_table;
  std::unique_ptr<UploadedLinearOp> lm_head_op;
  std::unique_ptr<DeviceTensorFp32> final_norm_weight;
  std::unique_ptr<LoadedModelCache> model_cache;
  std::vector<LayerEntry> layers;
  SingleTokenForwardBuildReport build_report;
};

std::unique_ptr<SingleTokenForwardModel> SingleTokenForwardModel::Create(
    const RuntimeEnvironment& environment,
    const SingleTokenForwardConfig& config) {
  const bool build_debug = std::getenv("NEMOTRON_FORWARD_BUILD_DEBUG") != nullptr;
  const auto log_phase = [&](const std::string& name, double ms) {
    if (build_debug) {
      std::cerr << std::fixed << std::setprecision(3)
                << "forward_build_phase[" << name << "]=" << ms << " ms\n"
                << std::flush;
    }
  };
  const auto log_layer = [&](const SingleTokenForwardLayerBuildTiming& timing) {
    if (build_debug) {
      std::cerr << std::fixed << std::setprecision(3)
                << "forward_build_layer[" << timing.layer_index
                << "][" << LayerKindName(timing.kind) << "]"
                << " bindings_ms=" << timing.bindings_ms
                << " slice_create_ms=" << timing.slice_create_ms
                << " total_ms=" << timing.total_ms << "\n"
                << std::flush;
    }
  };
  SingleTokenForwardBuildReport build_report;
  const auto add_phase =
      [&](const std::string& name, const std::chrono::steady_clock::time_point& begin,
          const std::chrono::steady_clock::time_point& end) {
        const double ms = ToMilliseconds(end - begin);
        build_report.phases.push_back(SingleTokenForwardBuildPhaseTiming{name, ms});
        log_phase(name, ms);
      };

  if (!environment.has_model_schedule() ||
      !environment.has_kernel_catalog() ||
      !environment.has_gemm_catalog() ||
      !environment.has_embedding_catalog()) {
    return nullptr;
  }
  const ModelSchedule& schedule = *environment.model_schedule();
  const auto plan_begin = std::chrono::steady_clock::now();
  const auto plan = BuildSingleTokenForwardPlan(schedule, config);
  const auto plan_end = std::chrono::steady_clock::now();
  add_phase("plan_build", plan_begin, plan_end);
  if (!plan.has_value()) {
    return nullptr;
  }

  const auto descriptor_begin = std::chrono::steady_clock::now();
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
    return nullptr;
  }
  if (embedding->embedding_dim != config.hidden_size ||
      embedding->vocab_size != config.vocab_size ||
      lm_head->input_cols != config.hidden_size ||
      lm_head->output_rows != config.vocab_size) {
    return nullptr;
  }
  if (final_norm != nullptr &&
      (final_norm->logical_shape.size() != 1 || final_norm->logical_shape.front() != config.hidden_size)) {
    return nullptr;
  }
  const auto descriptor_end = std::chrono::steady_clock::now();
  add_phase("global_descriptor_lookup", descriptor_begin, descriptor_end);

  const auto handles_begin = std::chrono::steady_clock::now();
  auto cublas = CublasLtHandle::Create();
  auto heuristic_cache = std::make_unique<GemmHeuristicCache>();
  if (!cublas || !cublas->valid()) {
    return nullptr;
  }

  std::unique_ptr<CudnnHandle> cudnn;
  if (plan->attention_layer_count != 0) {
    cudnn = CudnnHandle::Create();
    if (!cudnn || !cudnn->valid()) {
      return nullptr;
    }
  }
  const auto handles_end = std::chrono::steady_clock::now();
  add_phase("handle_init", handles_begin, handles_end);

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->plan = *plan;
  impl->embedding = embedding;
  impl->final_norm = final_norm;
  impl->lm_head = lm_head;
  impl->cublas = std::move(cublas);
  impl->cudnn = std::move(cudnn);
  impl->heuristic_cache = std::move(heuristic_cache);

  const auto embedding_begin = std::chrono::steady_clock::now();
  impl->embedding_table = DeviceEmbeddingTableFp32::Upload(*embedding);
  const auto embedding_end = std::chrono::steady_clock::now();
  add_phase("embedding_upload", embedding_begin, embedding_end);

  const auto lm_head_begin = std::chrono::steady_clock::now();
  impl->lm_head_op = UploadedLinearOp::Create(*lm_head);
  const auto lm_head_end = std::chrono::steady_clock::now();
  add_phase("lm_head_upload", lm_head_begin, lm_head_end);
  if (!impl->embedding_table || !impl->embedding_table->valid() ||
      !impl->lm_head_op || !impl->lm_head_op->valid()) {
    return nullptr;
  }
  if (final_norm != nullptr) {
    const auto final_norm_begin = std::chrono::steady_clock::now();
    impl->final_norm_weight = UploadVectorWeightToDeviceFp32(*final_norm);
    const auto final_norm_end = std::chrono::steady_clock::now();
    add_phase("final_norm_upload", final_norm_begin, final_norm_end);
    if (!impl->final_norm_weight || !impl->final_norm_weight->valid()) {
      return nullptr;
    }
  }
  const auto build_layer = [&](const ForwardLayerPlanEntry& plan_entry,
                               Impl::LayerEntry* layer_entry_out,
                               SingleTokenForwardLayerBuildTiming* timing_out) -> bool {
    const auto layer_begin = std::chrono::steady_clock::now();
    const LayerScheduleEntry* layer = schedule.FindLayer(plan_entry.layer_index);
    if (layer == nullptr || layer_entry_out == nullptr || timing_out == nullptr) {
      return false;
    }

    Impl::LayerEntry layer_entry;
    layer_entry.plan = plan_entry;
    std::vector<SingleTokenForwardBuildPhaseTiming> detail_timings;
    const auto add_detail = [&](const char* name, double milliseconds) {
      detail_timings.push_back(SingleTokenForwardBuildPhaseTiming{name, milliseconds});
    };
    switch (plan_entry.kind) {
      case ForwardLayerKind::kAttention: {
        const auto bindings_begin = std::chrono::steady_clock::now();
        const auto bindings = BuildAttentionLayerBindings(*layer, kernel_catalog, gemm_catalog);
        const auto bindings_end = std::chrono::steady_clock::now();
        if (!bindings.has_value()) {
          return false;
        }
        AttentionLayerConfig attention_config;
        attention_config.layer_index = plan_entry.layer_index;
        attention_config.hidden_size = config.hidden_size;
        attention_config.query_head_count = config.attention_head_count;
        attention_config.kv_head_count = config.attention_kv_head_count;
        attention_config.head_dim = config.attention_head_dim;
        attention_config.rms_epsilon = config.layer_norm_epsilon;
        const auto slice_begin = std::chrono::steady_clock::now();
        layer_entry.attention_slice = AttentionLayerSlice::Create(attention_config, *bindings);
        const auto slice_end = std::chrono::steady_clock::now();
        *timing_out = SingleTokenForwardLayerBuildTiming{
            plan_entry.layer_index,
            plan_entry.kind,
            ToMilliseconds(bindings_end - bindings_begin),
            ToMilliseconds(slice_end - slice_begin),
            ToMilliseconds(slice_end - layer_begin),
            std::move(detail_timings),
        };
        if (!layer_entry.attention_slice || !layer_entry.attention_slice->valid()) {
          return false;
        }
        break;
      }
      case ForwardLayerKind::kMamba: {
        const auto bindings_begin = std::chrono::steady_clock::now();
        const auto bindings = BuildMambaLayerBindings(*layer, kernel_catalog, gemm_catalog);
        const auto bindings_end = std::chrono::steady_clock::now();
        if (!bindings.has_value()) {
          return false;
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
        const auto slice_begin = std::chrono::steady_clock::now();
        layer_entry.mamba_slice = MambaLayerSlice::Create(
            mamba_config,
            *bindings,
            [&](const char* name, double milliseconds) {
              add_detail(name, milliseconds);
            });
        const auto slice_end = std::chrono::steady_clock::now();
        *timing_out = SingleTokenForwardLayerBuildTiming{
            plan_entry.layer_index,
            plan_entry.kind,
            ToMilliseconds(bindings_end - bindings_begin),
            ToMilliseconds(slice_end - slice_begin),
            ToMilliseconds(slice_end - layer_begin),
            std::move(detail_timings),
        };
        if (!layer_entry.mamba_slice || !layer_entry.mamba_slice->valid()) {
          return false;
        }
        break;
      }
      case ForwardLayerKind::kExpert: {
        const auto bindings_begin = std::chrono::steady_clock::now();
        const auto bindings = BuildExpertLayerBindings(
            *layer,
            kernel_catalog,
            gemm_catalog,
            config.n_routed_experts);
        const auto bindings_end = std::chrono::steady_clock::now();
        if (!bindings.has_value()) {
          return false;
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
        const auto slice_begin = std::chrono::steady_clock::now();
        layer_entry.expert_slice = ExpertLayerSlice::Create(
            expert_config,
            *bindings,
            [&](const char* name, double milliseconds) {
              add_detail(name, milliseconds);
            });
        const auto slice_end = std::chrono::steady_clock::now();
        *timing_out = SingleTokenForwardLayerBuildTiming{
            plan_entry.layer_index,
            plan_entry.kind,
            ToMilliseconds(bindings_end - bindings_begin),
            ToMilliseconds(slice_end - slice_begin),
            ToMilliseconds(slice_end - layer_begin),
            std::move(detail_timings),
        };
        if (!layer_entry.expert_slice || !layer_entry.expert_slice->valid()) {
          return false;
        }
        break;
      }
    }

    *layer_entry_out = std::move(layer_entry);
    return true;
  };

  impl->layers.resize(plan->layers.size());
  build_report.layers.resize(plan->layers.size());

  const auto layer_loop_begin = std::chrono::steady_clock::now();
  const std::size_t worker_count =
      std::min<std::size_t>(BuildWorkerCount(), plan->layers.size());
  if (worker_count <= 1 || plan->layers.size() <= 1) {
    for (std::size_t layer_idx = 0; layer_idx < plan->layers.size(); ++layer_idx) {
      if (!build_layer(
              plan->layers[layer_idx],
              &impl->layers[layer_idx],
              &build_report.layers[layer_idx])) {
        return nullptr;
      }
      log_layer(build_report.layers[layer_idx]);
    }
  } else {
    std::atomic<bool> build_failed = false;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
      workers.emplace_back([&, worker_index]() {
        for (std::size_t layer_idx = worker_index;
             layer_idx < plan->layers.size();
             layer_idx += worker_count) {
          if (build_failed.load(std::memory_order_relaxed)) {
            return;
          }
          if (!build_layer(
                  plan->layers[layer_idx],
                  &impl->layers[layer_idx],
                  &build_report.layers[layer_idx])) {
            build_failed.store(true, std::memory_order_relaxed);
            return;
          }
          log_layer(build_report.layers[layer_idx]);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    if (build_failed.load(std::memory_order_relaxed)) {
      return nullptr;
    }
  }
  const auto layer_loop_end = std::chrono::steady_clock::now();
  add_phase("layer_loop_total", layer_loop_begin, layer_loop_end);
  impl->build_report = std::move(build_report);

  return std::unique_ptr<SingleTokenForwardModel>(new SingleTokenForwardModel(std::move(impl)));
}

std::unique_ptr<SingleTokenForwardModel> SingleTokenForwardModel::CreateFromCache(
    const RuntimeEnvironment& environment,
    const SingleTokenForwardConfig& config,
    const std::filesystem::path& cache_path) {
  SingleTokenForwardBuildReport build_report;
  const auto add_phase =
      [&](const std::string& name, const std::chrono::steady_clock::time_point& begin,
          const std::chrono::steady_clock::time_point& end) {
        build_report.phases.push_back(
            SingleTokenForwardBuildPhaseTiming{name, ToMilliseconds(end - begin)});
      };

  if (!environment.has_model_schedule() ||
      !environment.has_kernel_catalog() ||
      !environment.has_gemm_catalog() ||
      !environment.has_embedding_catalog()) {
    return nullptr;
  }

  const ModelSchedule& schedule = *environment.model_schedule();
  const auto plan_begin = std::chrono::steady_clock::now();
  const auto plan = BuildSingleTokenForwardPlan(schedule, config);
  const auto plan_end = std::chrono::steady_clock::now();
  add_phase("plan_build", plan_begin, plan_end);
  if (!plan.has_value()) {
    return nullptr;
  }

  const auto descriptor_begin = std::chrono::steady_clock::now();
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
    return nullptr;
  }
  const auto descriptor_end = std::chrono::steady_clock::now();
  add_phase("global_descriptor_lookup", descriptor_begin, descriptor_end);

  const auto handles_begin = std::chrono::steady_clock::now();
  auto cublas = CublasLtHandle::Create();
  auto heuristic_cache = std::make_unique<GemmHeuristicCache>();
  if (!cublas || !cublas->valid()) {
    return nullptr;
  }
  std::unique_ptr<CudnnHandle> cudnn;
  if (plan->attention_layer_count != 0) {
    cudnn = CudnnHandle::Create();
    if (!cudnn || !cudnn->valid()) {
      return nullptr;
    }
  }
  const auto handles_end = std::chrono::steady_clock::now();
  add_phase("handle_init", handles_begin, handles_end);

  const auto cache_begin = std::chrono::steady_clock::now();
  auto cache = LoadedModelCache::Load(cache_path);
  const auto cache_end = std::chrono::steady_clock::now();
  add_phase("cache_load", cache_begin, cache_end);
  if (!cache || !cache->valid()) {
    return nullptr;
  }
  const SingleTokenForwardConfig& cached_config = cache->header().config;
  if (cached_config.hidden_size != config.hidden_size ||
      cached_config.total_layer_count != config.total_layer_count ||
      cached_config.vocab_size != config.vocab_size ||
      cached_config.n_routed_experts != config.n_routed_experts ||
      cached_config.experts_per_token != config.experts_per_token) {
    return nullptr;
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
  impl->model_cache = std::move(cache);

  const auto embedding_begin = std::chrono::steady_clock::now();
  impl->embedding_table = impl->model_cache->CreateEmbeddingView(embedding->tensor_name);
  const auto embedding_end = std::chrono::steady_clock::now();
  add_phase("embedding_view", embedding_begin, embedding_end);

  const auto lm_head_begin = std::chrono::steady_clock::now();
  impl->lm_head_op = impl->model_cache->CreateDenseLinearView(*lm_head);
  const auto lm_head_end = std::chrono::steady_clock::now();
  add_phase("lm_head_view", lm_head_begin, lm_head_end);
  if (!impl->embedding_table || !impl->embedding_table->valid() ||
      !impl->lm_head_op || !impl->lm_head_op->valid()) {
    return nullptr;
  }
  if (final_norm != nullptr) {
    const auto final_norm_begin = std::chrono::steady_clock::now();
    impl->final_norm_weight = impl->model_cache->CreateTensorView(final_norm->tensor_name);
    const auto final_norm_end = std::chrono::steady_clock::now();
    add_phase("final_norm_view", final_norm_begin, final_norm_end);
    if (!impl->final_norm_weight || !impl->final_norm_weight->valid()) {
      return nullptr;
    }
  }

  impl->layers.resize(plan->layers.size());
  build_report.layers.resize(plan->layers.size());
  const auto layer_loop_begin = std::chrono::steady_clock::now();
  for (std::size_t layer_idx = 0; layer_idx < plan->layers.size(); ++layer_idx) {
    const auto layer_begin = std::chrono::steady_clock::now();
    const ForwardLayerPlanEntry& plan_entry = plan->layers[layer_idx];
    const LayerScheduleEntry* layer = schedule.FindLayer(plan_entry.layer_index);
    if (layer == nullptr) {
      return nullptr;
    }
    Impl::LayerEntry layer_entry;
    layer_entry.plan = plan_entry;
    const auto bindings_begin = std::chrono::steady_clock::now();
    switch (plan_entry.kind) {
      case ForwardLayerKind::kAttention: {
        const auto bindings = BuildAttentionLayerBindings(*layer, kernel_catalog, gemm_catalog);
        const auto bindings_end = std::chrono::steady_clock::now();
        if (!bindings.has_value()) {
          return nullptr;
        }
        AttentionLayerPreparedBindings prepared;
        prepared.norm_weight = impl->model_cache->CreateTensorView(bindings->norm_weight->tensor_name);
        prepared.q_proj = impl->model_cache->CreateDenseLinearView(*bindings->q_proj);
        prepared.k_proj = impl->model_cache->CreateDenseLinearView(*bindings->k_proj);
        prepared.v_proj = impl->model_cache->CreateDenseLinearView(*bindings->v_proj);
        prepared.o_proj = impl->model_cache->CreateDenseLinearView(*bindings->o_proj);
        const auto slice_begin = std::chrono::steady_clock::now();
        AttentionLayerConfig attention_config;
        attention_config.layer_index = plan_entry.layer_index;
        attention_config.hidden_size = config.hidden_size;
        attention_config.query_head_count = config.attention_head_count;
        attention_config.kv_head_count = config.attention_kv_head_count;
        attention_config.head_dim = config.attention_head_dim;
        attention_config.rms_epsilon = config.layer_norm_epsilon;
        layer_entry.attention_slice = AttentionLayerSlice::CreatePrepared(
            attention_config,
            std::move(prepared));
        const auto slice_end = std::chrono::steady_clock::now();
        build_report.layers[layer_idx] = {
            plan_entry.layer_index,
            plan_entry.kind,
            ToMilliseconds(bindings_end - bindings_begin),
            ToMilliseconds(slice_end - slice_begin),
            ToMilliseconds(slice_end - layer_begin),
            {}};
        if (!layer_entry.attention_slice || !layer_entry.attention_slice->valid()) {
          return nullptr;
        }
        break;
      }
      case ForwardLayerKind::kMamba: {
        const auto bindings = BuildMambaLayerBindings(*layer, kernel_catalog, gemm_catalog);
        const auto bindings_end = std::chrono::steady_clock::now();
        if (!bindings.has_value()) {
          return nullptr;
        }
        MambaLayerPreparedBindings prepared;
        prepared.input_norm_weight = impl->model_cache->CreateTensorView(bindings->input_norm_weight->tensor_name);
        prepared.mixer_norm_weight = impl->model_cache->CreateTensorView(bindings->mixer_norm_weight->tensor_name);
        prepared.conv1d_weight = impl->model_cache->CreateTensorView(bindings->conv1d_weight->tensor_name);
        prepared.conv1d_bias = impl->model_cache->CreateTensorView(bindings->conv1d_bias->tensor_name);
        prepared.A_log = impl->model_cache->CreateTensorView(bindings->A_log->tensor_name);
        prepared.D = impl->model_cache->CreateTensorView(bindings->D->tensor_name);
        prepared.dt_bias = impl->model_cache->CreateTensorView(bindings->dt_bias->tensor_name);
        if (bindings->in_proj_kernel_weight != nullptr &&
            bindings->in_proj_weight_scale != nullptr &&
            bindings->in_proj_input_scale != nullptr) {
          prepared.in_proj_scaled_fp8 = impl->model_cache->CreateScaledFp8LinearView(
              bindings->in_proj_gemm_weight->tensor_name,
              bindings->in_proj_gemm_weight->output_rows,
              bindings->in_proj_gemm_weight->input_cols);
        } else {
          prepared.in_proj_dense = impl->model_cache->CreateDenseLinearView(*bindings->in_proj_gemm_weight);
        }
        if (bindings->out_proj_kernel_weight != nullptr &&
            bindings->out_proj_weight_scale != nullptr &&
            bindings->out_proj_input_scale != nullptr) {
          prepared.out_proj_scaled_fp8 = impl->model_cache->CreateScaledFp8LinearView(
              bindings->out_proj_gemm_weight->tensor_name,
              bindings->out_proj_gemm_weight->output_rows,
              bindings->out_proj_gemm_weight->input_cols);
        } else {
          prepared.out_proj_dense = impl->model_cache->CreateDenseLinearView(*bindings->out_proj_gemm_weight);
        }
        const auto slice_begin = std::chrono::steady_clock::now();
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
        layer_entry.mamba_slice = MambaLayerSlice::CreatePrepared(mamba_config, std::move(prepared));
        const auto slice_end = std::chrono::steady_clock::now();
        build_report.layers[layer_idx] = {
            plan_entry.layer_index,
            plan_entry.kind,
            ToMilliseconds(bindings_end - bindings_begin),
            ToMilliseconds(slice_end - slice_begin),
            ToMilliseconds(slice_end - layer_begin),
            {}};
        if (!layer_entry.mamba_slice || !layer_entry.mamba_slice->valid()) {
          return nullptr;
        }
        break;
      }
      case ForwardLayerKind::kExpert: {
        const auto bindings = BuildExpertLayerBindings(*layer, kernel_catalog, gemm_catalog, config.n_routed_experts);
        const auto bindings_end = std::chrono::steady_clock::now();
        if (!bindings.has_value()) {
          return nullptr;
        }
        std::vector<SingleTokenForwardBuildPhaseTiming> detail_timings;
        const auto add_detail = [&](const char* name, double milliseconds) {
          detail_timings.push_back({name, milliseconds});
        };
        ExpertLayerPreparedBindings prepared;
        prepared.input_norm_weight = impl->model_cache->CreateTensorView(bindings->input_norm_weight->tensor_name);
        prepared.gate_score_correction_bias_device =
            impl->model_cache->CreateTensorView(bindings->gate_score_correction_bias->tensor_name);
        prepared.gate_weight = impl->model_cache->CreateDenseLinearView(*bindings->gate_weight);
        prepared.fc2_latent = impl->model_cache->CreateDenseLinearView(*bindings->fc2_latent_weight);
        if (bindings->fc1_latent_kernel_weight != nullptr &&
            bindings->fc1_latent_weight_scale != nullptr &&
            bindings->fc1_latent_input_scale != nullptr) {
          prepared.fc1_latent_scaled_fp8 = impl->model_cache->CreateScaledFp8LinearView(
              bindings->fc1_latent_gemm_weight->tensor_name,
              bindings->fc1_latent_gemm_weight->output_rows,
              bindings->fc1_latent_gemm_weight->input_cols);
        } else {
          prepared.fc1_latent_dense = impl->model_cache->CreateDenseLinearView(*bindings->fc1_latent_gemm_weight);
        }
        if (bindings->shared_up_kernel_weight != nullptr &&
            bindings->shared_up_weight_scale != nullptr &&
            bindings->shared_up_input_scale != nullptr) {
          prepared.shared_up_scaled_fp8 = impl->model_cache->CreateScaledFp8LinearView(
              bindings->shared_up_gemm_weight->tensor_name,
              bindings->shared_up_gemm_weight->output_rows,
              bindings->shared_up_gemm_weight->input_cols);
        } else {
          prepared.shared_up_dense = impl->model_cache->CreateDenseLinearView(*bindings->shared_up_gemm_weight);
        }
        if (bindings->shared_down_gemm_weight->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
          prepared.shared_down_nvfp4 = impl->model_cache->CreateNvfp4LinearView(*bindings->shared_down_gemm_weight);
        } else if (bindings->shared_down_kernel_weight != nullptr &&
                   bindings->shared_down_weight_scale != nullptr &&
                   bindings->shared_down_input_scale != nullptr &&
                   bindings->shared_down_kernel_weight->storage_dtype == "fp8_e4m3fn") {
          prepared.shared_down_scaled_fp8 = impl->model_cache->CreateScaledFp8LinearView(
              bindings->shared_down_gemm_weight->tensor_name,
              bindings->shared_down_gemm_weight->output_rows,
              bindings->shared_down_gemm_weight->input_cols);
        } else {
          prepared.shared_down_dense = impl->model_cache->CreateDenseLinearView(*bindings->shared_down_gemm_weight);
        }
        prepared.routed_experts.resize(bindings->routed_experts.size());
        for (std::size_t expert_index = 0; expert_index < bindings->routed_experts.size(); ++expert_index) {
          const ExpertWeightPair& pair = bindings->routed_experts[expert_index];
          prepared.routed_experts[expert_index].up_descriptor = *pair.up_proj;
          prepared.routed_experts[expert_index].down_descriptor = *pair.down_proj;
          prepared.routed_experts[expert_index].up_tensor_scale = ReadTensorScaleHost(*pair.up_proj);
          prepared.routed_experts[expert_index].down_tensor_scale = ReadTensorScaleHost(*pair.down_proj);
          prepared.routed_experts[expert_index].up_input_scale =
              ReadOptionalScalarTensorToHostFp32(pair.up_input_scale);
          prepared.routed_experts[expert_index].down_input_scale =
              ReadOptionalScalarTensorToHostFp32(pair.down_input_scale);
          if (pair.up_proj->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
            auto up_proj = impl->model_cache->CreateNvfp4LinearView(*pair.up_proj);
            auto down_proj = impl->model_cache->CreateNvfp4LinearView(*pair.down_proj);
            if ((up_proj == nullptr) != (down_proj == nullptr)) {
              return nullptr;
            }
            prepared.routed_experts[expert_index].up_proj = std::move(up_proj);
            prepared.routed_experts[expert_index].down_proj = std::move(down_proj);
          } else {
            prepared.routed_experts[expert_index].up_proj =
                impl->model_cache->CreateDenseLinearView(*pair.up_proj);
            prepared.routed_experts[expert_index].down_proj =
                impl->model_cache->CreateDenseLinearView(*pair.down_proj);
          }
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
        const auto slice_begin = std::chrono::steady_clock::now();
        layer_entry.expert_slice = ExpertLayerSlice::CreatePrepared(
            expert_config,
            std::move(prepared));
        const auto slice_end = std::chrono::steady_clock::now();
        build_report.layers[layer_idx] = {
            plan_entry.layer_index,
            plan_entry.kind,
            ToMilliseconds(bindings_end - bindings_begin),
            ToMilliseconds(slice_end - slice_begin),
            ToMilliseconds(slice_end - layer_begin),
            std::move(detail_timings)};
        if (!layer_entry.expert_slice || !layer_entry.expert_slice->valid()) {
          return nullptr;
        }
        if (layer_entry.expert_slice->routed_experts_contiguous()) {
          for (const ExpertWeightPair& pair : bindings->routed_experts) {
            if (pair.up_proj == nullptr || pair.down_proj == nullptr) {
              return nullptr;
            }
            impl->model_cache->ReleaseEntry(pair.up_proj->tensor_name);
            impl->model_cache->ReleaseEntry(pair.down_proj->tensor_name);
          }
        }
        break;
      }
    }
    impl->layers[layer_idx] = std::move(layer_entry);
  }
  const auto layer_loop_end = std::chrono::steady_clock::now();
  add_phase("layer_loop_total", layer_loop_begin, layer_loop_end);
  impl->model_cache->ReleaseFilePages();
  impl->build_report = std::move(build_report);
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

const SingleTokenForwardBuildReport& SingleTokenForwardModel::build_report() const {
  return impl_->build_report;
}

std::unique_ptr<RequestExecutionContext> SingleTokenForwardModel::CreateRequestContext() const {
  if (!valid()) {
    return nullptr;
  }
  return RequestExecutionContext::Create(impl_->plan.request_config);
}

bool SingleTokenForwardModel::RunSingleToken(
    std::int32_t token_id,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index) const {
  return RunPrefill(
      &token_id,
      1,
      request_context,
      logits,
      capture_layer_indices,
      trace,
      stop_layer_index);
}

bool SingleTokenForwardModel::RunDecodeStep(
    std::int32_t token_id,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index) const {
  const bool debug = ForwardDebugEnabled();
  const bool model_valid = valid();
  const bool request_valid = request_context.valid();
  const bool have_logits = logits != nullptr;
  const bool logits_valid = have_logits && logits->valid();
  const std::vector<std::size_t> expected_logits_shape = {1, impl_->config.vocab_size};
  const bool logits_shape_ok = logits_valid && logits->shape() == expected_logits_shape;
  if (!model_valid || !request_valid || !have_logits || !logits_valid || !logits_shape_ok) {
    if (debug) {
      std::cout << "single_token_forward_model: invalid RunDecodeStep arguments or state\n";
    }
    return false;
  }
  if (!request_context.AdvanceDecodePosition(1)) {
    if (debug) {
      std::cout << "single_token_forward_model: failed to advance decode position\n";
    }
    return false;
  }
  return RunPrefill(
      &token_id,
      1,
      request_context,
      logits,
      capture_layer_indices,
      trace,
      stop_layer_index);
}

bool SingleTokenForwardModel::RunPrefill(
    const std::int32_t* token_ids,
    std::size_t token_count,
    RequestExecutionContext& request_context,
    DeviceTensorFp32* logits,
    const std::vector<std::size_t>& capture_layer_indices,
    SingleTokenForwardTrace* trace,
    std::optional<std::size_t> stop_layer_index) const {
  const bool debug = ForwardDebugEnabled();

  const bool model_valid = valid();
  const bool have_tokens = token_ids != nullptr;
  const bool nonzero_token_count = token_count != 0;
  const bool request_valid = request_context.valid();
  const bool hidden_size_match = request_context.config().hidden_size == impl_->config.hidden_size;
  const bool token_capacity_ok = request_context.config().max_tokens >= token_count;
  const bool have_logits = logits != nullptr;
  const bool logits_valid = have_logits && logits->valid();
  const std::vector<std::size_t> expected_logits_shape = {token_count, impl_->config.vocab_size};
  const bool logits_shape_ok = logits_valid && logits->shape() == expected_logits_shape;

  if (!model_valid ||
      !have_tokens ||
      !nonzero_token_count ||
      !request_valid ||
      !hidden_size_match ||
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
              << " token_capacity_ok=" << token_capacity_ok
              << " have_logits=" << have_logits
              << " logits_valid=" << logits_valid
              << " logits_shape_ok=" << logits_shape_ok
              << " request_hidden_size=" << request_context.config().hidden_size
              << " model_hidden_size=" << impl_->config.hidden_size
              << " request_max_tokens=" << request_context.config().max_tokens
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
  if (impl_->plan.attention_layer_count != 0 &&
      (impl_->cudnn == nullptr || !impl_->cudnn->valid())) {
    std::cerr << "single_token_forward_model: cuDNN handle unavailable\n";
    return false;
  }
  const bool continuing_decode =
      token_count == 1 &&
      request_context.sequence_length() != 0 &&
      request_context.sequence_length() == request_context.decode_position();
  const bool use_bf16_decode_storage =
      token_count == 1 &&
      continuing_decode &&
      request_context.sequence_length() > 1;
  if (!continuing_decode && !request_context.ResetForNewRequest()) {
    std::cerr << "single_token_forward_model: request context reset failed\n";
    return false;
  }

  DeviceTensorFp32* hidden_storage = request_context.hidden();
  DeviceTensorFp32* residual_storage = request_context.residual();
  DeviceTensorFp32* scratch_storage = request_context.scratch();
  DeviceBuffer<std::int32_t>* token_ids_device = request_context.token_ids_device();
  if (hidden_storage == nullptr || residual_storage == nullptr || scratch_storage == nullptr) {
    std::cerr << "single_token_forward_model: request context buffers unavailable\n";
    return false;
  }
  if (token_ids_device == nullptr || token_ids_device->count() < token_count) {
    std::cerr << "single_token_forward_model: request token-id buffer unavailable\n";
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

  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    if (token_ids[token_index] < 0 ||
        static_cast<std::size_t>(token_ids[token_index]) >= impl_->config.vocab_size) {
      std::cerr << "single_token_forward_model: token id out of range at index "
                << token_index << " token_id=" << token_ids[token_index] << "\n";
      return false;
    }
  }

  const std::unordered_set<std::size_t> capture_set(
      capture_layer_indices.begin(),
      capture_layer_indices.end());

  const bool profile = ForwardProfileEnabled();
  cudaEvent_t profile_start = nullptr;
  cudaEvent_t profile_end = nullptr;
  if (profile) {
    cudaEventCreate(&profile_start);
    cudaEventCreate(&profile_end);
  }
  const auto destroy_profile_events = [&]() {
    if (!profile) {
      return;
    }
    if (profile_start != nullptr) {
      cudaEventDestroy(profile_start);
      profile_start = nullptr;
    }
    if (profile_end != nullptr) {
      cudaEventDestroy(profile_end);
      profile_end = nullptr;
    }
  };

  if (use_bf16_decode_storage) {
    DeviceTensorBf16* hidden_decode_storage = request_context.hidden_decode_bf16();
    DeviceTensorBf16* residual_decode_storage = request_context.residual_decode_bf16();
    if (hidden_decode_storage == nullptr || residual_decode_storage == nullptr) {
      std::cerr << "single_token_forward_model: BF16 decode storage unavailable\n";
      destroy_profile_events();
      return false;
    }

    auto current_view_bf16 = DeviceTensorBf16::CreateView(
        {token_count, impl_->config.hidden_size},
        hidden_decode_storage->data());
    auto next_view_bf16 = DeviceTensorBf16::CreateView(
        {token_count, impl_->config.hidden_size},
        residual_decode_storage->data());
    auto current_view_fp32 = DeviceTensorFp32::CreateView(
        {token_count, impl_->config.hidden_size},
        hidden_storage->data());
    auto scratch_view_fp32 = DeviceTensorFp32::CreateView(
        {token_count, impl_->config.hidden_size},
        scratch_storage->data());
    if (!current_view_bf16 || !next_view_bf16 || !current_view_fp32 || !scratch_view_fp32) {
      std::cerr << "single_token_forward_model: BF16 decode views unavailable\n";
      destroy_profile_events();
      return false;
    }
    DeviceTensorBf16* current = current_view_bf16.get();
    DeviceTensorBf16* next = next_view_bf16.get();
    DeviceTensorFp32* current_fp32 = current_view_fp32.get();
    DeviceTensorFp32* scratch_fp32 = scratch_view_fp32.get();

    if (!token_ids_device->CopyFromHostAsync(token_ids, token_count) ||
        !LookupEmbeddingRowsDeviceIdsFp32(
             *impl_->embedding_table,
             token_ids_device->data(),
             token_count,
             current_fp32)
             .has_value() ||
        !ConvertDeviceFp32ToBf16(
             current_fp32->data(),
             current_fp32->numel(),
             current->data())) {
      std::cerr << "single_token_forward_model: embedding lookup failed\n";
      destroy_profile_events();
      return false;
    }
    if (debug) {
      std::cout << "single_token_forward_model: embedding lookup ok for "
                << token_count << " token(s)\n";
    }
    if (trace != nullptr) {
      trace->embedding_output = CopyTensorToHost(*current);
      if (trace->embedding_output.empty()) {
        destroy_profile_events();
        return false;
      }
      trace->captured_layers.clear();
      trace->final_hidden.clear();
      trace->final_hidden_normed.clear();
      trace->logits.clear();
    }

    for (const Impl::LayerEntry& layer : impl_->layers) {
      if (debug) {
        std::cout << "single_token_forward_model: running layer "
                  << layer.plan.layer_index
                  << " kind=" << static_cast<int>(layer.plan.kind) << "\n";
      }
      if (profile) {
        cudaEventRecord(profile_start);
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
              layer.expert_slice->RunWithRequestContext(
                  *impl_->cublas,
                  impl_->heuristic_cache.get(),
                  request_context,
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
      if (profile) {
        cudaEventRecord(profile_end);
        cudaEventSynchronize(profile_end);
        float layer_ms = 0.0f;
        cudaEventElapsedTime(&layer_ms, profile_start, profile_end);
        const char* kind_name =
            layer.plan.kind == ForwardLayerKind::kAttention ? "attention" :
            layer.plan.kind == ForwardLayerKind::kMamba ? "mamba" :
            layer.plan.kind == ForwardLayerKind::kExpert ? "expert" : "unknown";
        std::cerr << "profile layer=" << layer.plan.layer_index
                  << " kind=" << kind_name
                  << " ms=" << std::fixed << std::setprecision(3) << layer_ms << "\n";
      }
      if (!ok) {
        std::cerr << "single_token_forward_model: layer "
                  << layer.plan.layer_index
                  << " kind=" << static_cast<int>(layer.plan.kind)
                  << " execution failed\n";
        destroy_profile_events();
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
          std::cerr << "single_token_forward_model: failed to capture layer "
                    << layer.plan.layer_index << "\n";
          destroy_profile_events();
          return false;
        }
        trace->captured_layers.push_back(std::move(captured));
      }

      if (stop_layer_index.has_value() && layer.plan.layer_index >= *stop_layer_index) {
        break;
      }
    }

    if (trace != nullptr) {
      trace->final_hidden = CopyTensorToHost(*current);
      if (trace->final_hidden.empty()) {
        std::cerr << "single_token_forward_model: failed to capture final hidden state\n";
        destroy_profile_events();
        return false;
      }
    }

    if (!ConvertDeviceBf16ToFp32(current->data(), current->numel(), current_fp32->data())) {
      std::cerr << "single_token_forward_model: failed to upcast final hidden state\n";
      destroy_profile_events();
      return false;
    }

    const DeviceTensorFp32* logits_input = current_fp32;
    if (impl_->final_norm_weight != nullptr) {
      if (!RmsNormFp32(
              *current_fp32,
              *impl_->final_norm_weight,
              impl_->config.layer_norm_epsilon,
              scratch_fp32)) {
        std::cerr << "single_token_forward_model: final RMSNorm failed\n";
        destroy_profile_events();
        return false;
      }
      logits_input = scratch_fp32;
    }

    if (trace != nullptr) {
      trace->final_hidden_normed = CopyTensorToHost(*logits_input);
      if (trace->final_hidden_normed.empty()) {
        std::cerr << "single_token_forward_model: failed to capture final normalized hidden state\n";
        destroy_profile_events();
        return false;
      }
    }

    if (!impl_->lm_head_op->Run(
            *impl_->cublas,
            impl_->heuristic_cache.get(),
            *logits_input,
            logits)) {
      std::cerr << "single_token_forward_model: lm_head projection failed\n";
      destroy_profile_events();
      return false;
    }
    if (debug) {
      std::cout << "single_token_forward_model: lm_head ok\n";
    }

    if (trace != nullptr) {
      trace->logits = CopyTensorToHost(*logits);
      if (trace->logits.empty()) {
        std::cerr << "single_token_forward_model: failed to capture logits\n";
        destroy_profile_events();
        return false;
      }
    }
    destroy_profile_events();
    return true;
  }

  auto current_view = DeviceTensorFp32::CreateView(
      {token_count, impl_->config.hidden_size},
      hidden_storage->data());
  auto next_view = DeviceTensorFp32::CreateView(
      {token_count, impl_->config.hidden_size},
      residual_storage->data());
  auto scratch_view = DeviceTensorFp32::CreateView(
      {token_count, impl_->config.hidden_size},
      scratch_storage->data());
  if (!current_view || !next_view || !scratch_view) {
    std::cerr << "single_token_forward_model: request context token views unavailable\n";
    destroy_profile_events();
    return false;
  }
  DeviceTensorFp32* current = current_view.get();
  DeviceTensorFp32* next = next_view.get();
  DeviceTensorFp32* scratch = scratch_view.get();

  if (!token_ids_device->CopyFromHostAsync(token_ids, token_count) ||
      !LookupEmbeddingRowsDeviceIdsFp32(
           *impl_->embedding_table,
           token_ids_device->data(),
           token_count,
           current)
           .has_value()) {
    std::cerr << "single_token_forward_model: embedding lookup failed\n";
    destroy_profile_events();
    return false;
  }
  if (debug) {
    std::cout << "single_token_forward_model: embedding lookup ok for "
              << token_count << " token(s)\n";
  }
  if (trace != nullptr) {
    trace->embedding_output = CopyTensorToHost(*current);
    if (trace->embedding_output.empty()) {
      destroy_profile_events();
      return false;
    }
    trace->captured_layers.clear();
    trace->final_hidden.clear();
    trace->final_hidden_normed.clear();
    trace->logits.clear();
  }

  for (const Impl::LayerEntry& layer : impl_->layers) {
    if (debug) {
      std::cout << "single_token_forward_model: running layer "
                << layer.plan.layer_index
                << " kind=" << static_cast<int>(layer.plan.kind) << "\n";
    }
    if (profile) {
      cudaEventRecord(profile_start);
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
            layer.expert_slice->RunWithRequestContext(
                *impl_->cublas,
                impl_->heuristic_cache.get(),
                request_context,
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
    if (profile) {
      cudaEventRecord(profile_end);
      cudaEventSynchronize(profile_end);
      float layer_ms = 0.0f;
      cudaEventElapsedTime(&layer_ms, profile_start, profile_end);
      const char* kind_name =
          layer.plan.kind == ForwardLayerKind::kAttention ? "attention" :
          layer.plan.kind == ForwardLayerKind::kMamba ? "mamba" :
          layer.plan.kind == ForwardLayerKind::kExpert ? "expert" : "unknown";
      std::cerr << "profile layer=" << layer.plan.layer_index
                << " kind=" << kind_name
                << " ms=" << std::fixed << std::setprecision(3) << layer_ms << "\n";
    }
    if (!ok) {
      std::cerr << "single_token_forward_model: layer "
                << layer.plan.layer_index
                << " kind=" << static_cast<int>(layer.plan.kind)
                << " execution failed\n";
      destroy_profile_events();
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
        destroy_profile_events();
        return false;
      }
      trace->captured_layers.push_back(std::move(captured));
    }

    if (stop_layer_index.has_value() && layer.plan.layer_index >= *stop_layer_index) {
      break;
    }
  }

  if (trace != nullptr) {
    trace->final_hidden = CopyTensorToHost(*current);
    if (trace->final_hidden.empty()) {
      std::cerr << "single_token_forward_model: failed to capture final hidden state\n";
      destroy_profile_events();
      return false;
    }
  }

  const DeviceTensorFp32* logits_input = current;
  if (impl_->final_norm_weight != nullptr) {
    if (!RmsNormFp32(
            *current,
            *impl_->final_norm_weight,
            impl_->config.layer_norm_epsilon,
            scratch)) {
      std::cerr << "single_token_forward_model: final RMSNorm failed\n";
      destroy_profile_events();
      return false;
    }
    logits_input = scratch;
  }

  if (trace != nullptr) {
    trace->final_hidden_normed = CopyTensorToHost(*logits_input);
    if (trace->final_hidden_normed.empty()) {
      std::cerr << "single_token_forward_model: failed to capture final normalized hidden state\n";
      destroy_profile_events();
      return false;
    }
  }

  if (!impl_->lm_head_op->Run(
          *impl_->cublas,
          impl_->heuristic_cache.get(),
          *logits_input,
          logits)) {
    std::cerr << "single_token_forward_model: lm_head projection failed\n";
    destroy_profile_events();
    return false;
  }
  if (debug) {
    std::cout << "single_token_forward_model: lm_head ok\n";
  }

  if (trace != nullptr) {
    trace->logits = CopyTensorToHost(*logits);
    if (trace->logits.empty()) {
      std::cerr << "single_token_forward_model: failed to capture logits\n";
      destroy_profile_events();
      return false;
    }
  }
  destroy_profile_events();
  return true;
}

}  // namespace nemotron
