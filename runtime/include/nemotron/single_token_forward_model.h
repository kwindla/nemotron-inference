#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "nemotron/request_context.h"

namespace nemotron {

class DeviceTensorFp32;
class ModelSchedule;
class RuntimeEnvironment;

enum class ForwardLayerKind {
  kAttention,
  kMamba,
  kExpert,
};

struct SingleTokenForwardConfig {
  std::size_t hidden_size = 0;
  std::size_t total_layer_count = 0;
  std::size_t vocab_size = 0;
  std::size_t max_tokens = 1;

  std::size_t attention_head_count = 0;
  std::size_t attention_kv_head_count = 0;
  std::size_t attention_head_dim = 0;
  std::size_t attention_tokens_per_page = 16;

  std::size_t mamba_intermediate_size = 0;
  std::size_t mamba_num_heads = 0;
  std::size_t mamba_head_dim = 0;
  std::size_t mamba_state_size = 0;
  std::size_t mamba_n_groups = 0;
  std::size_t mamba_conv_kernel_size = 0;

  std::size_t moe_latent_size = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t n_routed_experts = 0;
  std::size_t experts_per_token = 0;
  std::size_t expert_n_group = 1;
  std::size_t expert_topk_group = 1;

  float layer_norm_epsilon = 1.0e-5f;
  float mamba_time_step_min = 1.0e-3f;
  float routed_scaling_factor = 1.0f;
  bool norm_topk_prob = true;
};

struct ForwardLayerPlanEntry {
  std::size_t layer_index = 0;
  ForwardLayerKind kind = ForwardLayerKind::kMamba;
  std::size_t mamba_conv_state_offset_elems = 0;
  std::size_t mamba_state_offset_elems = 0;
};

struct SingleTokenForwardPlan {
  bool valid = false;
  std::vector<ForwardLayerPlanEntry> layers;
  RequestExecutionConfig request_config;
  std::size_t attention_layer_count = 0;
  std::size_t mamba_layer_count = 0;
  std::size_t expert_layer_count = 0;
};

struct CapturedLayerOutput {
  std::size_t layer_index = 0;
  std::vector<float> hidden;
};

struct SingleTokenForwardTrace {
  std::vector<float> embedding_output;
  std::vector<CapturedLayerOutput> captured_layers;
  std::vector<float> final_hidden;
  std::vector<float> final_hidden_normed;
  std::vector<float> logits;
};

SingleTokenForwardConfig KnownNemotron3Super120BA12BConfig();
std::optional<SingleTokenForwardPlan> BuildSingleTokenForwardPlan(
    const ModelSchedule& schedule,
    const SingleTokenForwardConfig& config);

class SingleTokenForwardModel {
 public:
  static std::unique_ptr<SingleTokenForwardModel> Create(
      const RuntimeEnvironment& environment,
      const SingleTokenForwardConfig& config);

  SingleTokenForwardModel(SingleTokenForwardModel&&) noexcept;
  SingleTokenForwardModel& operator=(SingleTokenForwardModel&&) noexcept;
  ~SingleTokenForwardModel();

  SingleTokenForwardModel(const SingleTokenForwardModel&) = delete;
  SingleTokenForwardModel& operator=(const SingleTokenForwardModel&) = delete;

  bool valid() const;
  const SingleTokenForwardConfig& config() const;
  const SingleTokenForwardPlan& plan() const;
  std::unique_ptr<RequestExecutionContext> CreateRequestContext() const;

  bool RunPrefill(
      const std::int32_t* token_ids,
      std::size_t token_count,
      RequestExecutionContext& request_context,
      DeviceTensorFp32* logits,
      const std::vector<std::size_t>& capture_layer_indices = {},
      SingleTokenForwardTrace* trace = nullptr,
      std::optional<std::size_t> stop_layer_index = std::nullopt) const;

  bool RunSingleToken(
      std::int32_t token_id,
      RequestExecutionContext& request_context,
      DeviceTensorFp32* logits,
      const std::vector<std::size_t>& capture_layer_indices = {},
      SingleTokenForwardTrace* trace = nullptr,
      std::optional<std::size_t> stop_layer_index = std::nullopt) const;

 private:
  struct Impl;

  explicit SingleTokenForwardModel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
