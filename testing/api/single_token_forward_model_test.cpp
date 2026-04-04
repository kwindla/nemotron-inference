#include "nemotron/model_schedule.h"
#include "nemotron/single_token_forward_model.h"

#include <iostream>
#include <string>

namespace {

using nemotron::BuildModelSchedule;
using nemotron::BuildSingleTokenForwardPlan;
using nemotron::PackedModelManifest;
using nemotron::SingleTokenForwardConfig;
using nemotron::TensorManifestEntry;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

TensorManifestEntry make_tensor(const std::string& name, const std::string& op_class) {
  TensorManifestEntry tensor;
  tensor.name = name;
  tensor.op_class = op_class;
  tensor.logical_shape = {4, 4};
  tensor.packed_shape = {4, 4};
  tensor.storage_dtype = "fp32";
  tensor.compute_dtype = "fp32";
  tensor.layout_tag = "row_major";
  tensor.alignment_bytes = 16;
  tensor.packed_file = "weights.bin";
  tensor.nbytes = 64;
  tensor.source_tensor_name = name;
  tensor.checksum = "ok";
  return tensor;
}

PackedModelManifest make_manifest() {
  PackedModelManifest manifest;
  manifest.schema_version = 1;
  manifest.runtime.model_id = "single-token-plan-test";
  manifest.runtime.source_revision = "rev";
  manifest.runtime.tokenizer_revision = "tok";
  manifest.runtime.packer_version = "packer";
  manifest.runtime.gpu_family = "GB10";
  manifest.runtime.compute_capability = "12.1";
  manifest.runtime.kv_bytes_per_token = 1024;
  manifest.runtime.mamba_state_bytes_fp16 = 128;
  manifest.runtime.mamba_state_bytes_fp32 = 256;

  manifest.tensors.push_back(make_tensor("backbone.embeddings.weight", "embedding"));

  manifest.tensors.push_back(make_tensor("backbone.layers.0.norm.weight", "norm"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.A_log", "mamba_param"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.conv1d.weight", "mamba_param"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.dt_bias", "mamba_param"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.in_proj.weight", "mamba_linear"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.out_proj.weight", "mamba_linear"));

  manifest.tensors.push_back(make_tensor("backbone.layers.1.norm.weight", "norm"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.gate.weight", "router"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.experts.0.up_proj", "routed_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.experts.0.down_proj", "routed_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.shared_experts.up_proj", "shared_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.shared_experts.down_proj", "shared_expert"));

  manifest.tensors.push_back(make_tensor("backbone.layers.7.norm.weight", "norm"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.q_proj.weight", "attention"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.k_proj.weight", "attention"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.v_proj.weight", "attention"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.o_proj.weight", "attention"));

  manifest.tensors.push_back(make_tensor("backbone.norm_f.weight", "final_norm"));
  manifest.tensors.push_back(make_tensor("lm_head.weight", "logits"));
  return manifest;
}

bool test_single_token_forward_plan_tracks_layer_order_and_state_layout() {
  const nemotron::ModelSchedule schedule = BuildModelSchedule(make_manifest());
  SingleTokenForwardConfig config;
  config.hidden_size = 64;
  config.total_layer_count = 8;
  config.vocab_size = 256;
  config.max_tokens = 4;
  config.attention_head_count = 4;
  config.attention_kv_head_count = 1;
  config.attention_head_dim = 16;
  config.attention_tokens_per_page = 8;
  config.mamba_intermediate_size = 32;
  config.mamba_num_heads = 4;
  config.mamba_head_dim = 8;
  config.mamba_state_size = 16;
  config.mamba_n_groups = 2;
  config.mamba_conv_kernel_size = 4;
  config.moe_latent_size = 12;
  config.routed_expert_intermediate_size = 24;
  config.shared_expert_intermediate_size = 48;
  config.n_routed_experts = 8;
  config.experts_per_token = 2;
  config.expert_n_group = 1;
  config.expert_topk_group = 1;
  config.layer_norm_epsilon = 1.0e-5f;
  config.mamba_time_step_min = 1.0e-3f;
  config.routed_scaling_factor = 5.0f;
  config.norm_topk_prob = true;

  const auto plan = BuildSingleTokenForwardPlan(schedule, config);
  if (!expect(plan.has_value(), "forward plan should build for the synthetic schedule")) {
    return false;
  }

  const std::size_t conv_dim =
      config.mamba_intermediate_size + (2 * config.mamba_n_groups * config.mamba_state_size);
  const std::size_t expected_conv_state_elems = conv_dim * config.mamba_conv_kernel_size;
  const std::size_t expected_ssm_state_elems =
      config.mamba_num_heads * config.mamba_head_dim * config.mamba_state_size;

  return expect(plan->valid, "plan should be marked valid") &&
         expect(plan->layers.size() == 3, "plan should contain three ordered layers") &&
         expect(plan->layers[0].layer_index == 0, "first plan entry should be layer 0") &&
         expect(plan->layers[0].kind == nemotron::ForwardLayerKind::kMamba,
                "layer 0 should be classified as Mamba") &&
         expect(plan->layers[0].mamba_conv_state_offset_elems == 0,
                "first Mamba layer should start at conv offset 0") &&
         expect(plan->layers[0].mamba_state_offset_elems == 0,
                "first Mamba layer should start at ssm offset 0") &&
         expect(plan->layers[1].layer_index == 1, "second plan entry should be layer 1") &&
         expect(plan->layers[1].kind == nemotron::ForwardLayerKind::kExpert,
                "layer 1 should be classified as Expert") &&
         expect(plan->layers[2].layer_index == 7, "third plan entry should be layer 7") &&
         expect(plan->layers[2].kind == nemotron::ForwardLayerKind::kAttention,
                "layer 7 should be classified as Attention") &&
         expect(plan->attention_layer_count == 1, "plan should count one attention layer") &&
         expect(plan->mamba_layer_count == 1, "plan should count one mamba layer") &&
         expect(plan->expert_layer_count == 1, "plan should count one expert layer") &&
         expect(plan->request_config.hidden_size == config.hidden_size,
                "request config should preserve hidden size") &&
         expect(plan->request_config.max_tokens == config.max_tokens,
                "forward plan should preserve configured token capacity") &&
         expect(plan->request_config.scratch_tokens == config.max_tokens,
                "forward plan should size scratch tokens to the configured token capacity") &&
         expect(plan->request_config.attention_kv_cache.layer_count == 8,
                "KV layer count should span the maximum layer index") &&
         expect(plan->request_config.attention_total_pages == 8,
                "KV page capacity should reserve one page per indexed layer slot when attention is present") &&
         expect(plan->request_config.mamba_conv_state_bytes_fp32 ==
                    expected_conv_state_elems * sizeof(float),
                "conv-state bytes should match the single Mamba layer footprint") &&
         expect(plan->request_config.mamba_state_bytes_fp32 ==
                    expected_ssm_state_elems * sizeof(float),
                "ssm-state bytes should match the single Mamba layer footprint");
}

bool test_known_nano_config_matches_runtime_expectations() {
  const SingleTokenForwardConfig config = nemotron::KnownNemotron3Nano30BA3BConfig();
  return expect(config.hidden_size == 2688, "nano config should preserve hidden size") &&
         expect(config.total_layer_count == 52, "nano config should preserve layer count") &&
         expect(config.vocab_size == 131072, "nano config should preserve vocab size") &&
         expect(config.max_tokens == 1, "nano config should stay single-token for Phase 1") &&
         expect(config.attention_head_count == 32, "nano config should preserve attention head count") &&
         expect(config.attention_kv_head_count == 2, "nano config should preserve KV head count") &&
         expect(config.attention_head_dim == 128, "nano config should preserve attention head dim") &&
         expect(config.mamba_num_heads == 64, "nano config should preserve Mamba head count") &&
         expect(config.mamba_head_dim == 64, "nano config should preserve Mamba head dim") &&
         expect(config.mamba_intermediate_size == config.mamba_num_heads * config.mamba_head_dim,
                "nano config should use runtime Mamba inner width for intermediate size") &&
         expect(config.mamba_state_size == 128, "nano config should preserve Mamba state size") &&
         expect(config.mamba_n_groups == 8, "nano config should preserve Mamba group count") &&
         expect(config.mamba_conv_kernel_size == 4, "nano config should preserve Mamba conv kernel size") &&
         expect(config.moe_latent_size == 2688,
                "nano config should use hidden-size routed outputs for direct MoE") &&
         expect(config.routed_expert_intermediate_size == 1856,
                "nano config should preserve routed expert width") &&
         expect(config.shared_expert_intermediate_size == 3712,
                "nano config should preserve shared expert width") &&
         expect(config.n_routed_experts == 128, "nano config should preserve routed expert count") &&
         expect(config.experts_per_token == 6, "nano config should preserve experts per token") &&
         expect(config.expert_n_group == 1, "nano config should preserve router group count") &&
         expect(config.expert_topk_group == 1, "nano config should preserve top-k group count") &&
         expect(config.layer_norm_epsilon == 1.0e-5f, "nano config should preserve layer norm epsilon") &&
         expect(config.mamba_time_step_min == 1.0e-3f, "nano config should preserve time step minimum") &&
         expect(config.routed_scaling_factor == 2.5f,
                "nano config should preserve routed scaling factor") &&
         expect(config.norm_topk_prob, "nano config should preserve normalized top-k routing");
}

}  // namespace

int main() {
  const bool ok = test_single_token_forward_plan_tracks_layer_order_and_state_layout() &&
                  test_known_nano_config_matches_runtime_expectations();
  if (!ok) {
    return 1;
  }
  std::cout << "single_token_forward_model_test: PASS\n";
  return 0;
}
