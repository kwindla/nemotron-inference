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
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.A_log", "mamba_state"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.conv1d.weight", "conv1d"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.dt_bias", "mamba_state"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.in_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.mixer.out_proj.weight", "dense"));

  manifest.tensors.push_back(make_tensor("backbone.layers.1.norm.weight", "norm"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.gate.weight", "router"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.experts.0.up_proj", "routed_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.experts.0.down_proj", "routed_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.shared_experts.up_proj", "shared_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.shared_experts.down_proj", "shared_expert"));

  manifest.tensors.push_back(make_tensor("backbone.layers.7.norm.weight", "norm"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.q_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.k_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.v_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.7.self_attn.o_proj.weight", "dense"));

  manifest.tensors.push_back(make_tensor("backbone.norm_f.weight", "norm"));
  manifest.tensors.push_back(make_tensor("lm_head.weight", "dense"));
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
  const std::size_t expected_mamba_projection_size =
      config.mamba_intermediate_size + conv_dim + config.mamba_num_heads;

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
         expect(plan->request_config.mamba_hidden_size == config.hidden_size,
                "request config should preserve the mamba decode hidden width") &&
         expect(plan->request_config.mamba_projection_size == expected_mamba_projection_size,
                "request config should size reusable mamba projection scratch") &&
         expect(plan->request_config.mamba_intermediate_size == config.mamba_intermediate_size,
                "request config should preserve the mamba decode intermediate width") &&
         expect(plan->request_config.attention_kv_cache.layer_count == 8,
                "KV layer count should span the maximum layer index") &&
         expect(plan->request_config.attention_total_pages == 8,
                "KV page capacity should reserve one page per indexed layer slot when attention is present") &&
         expect(plan->request_config.mamba_conv_state_bytes_fp32 ==
                    expected_conv_state_elems * sizeof(float),
                "conv-state bytes should match the single Mamba layer footprint") &&
         expect(plan->request_config.mamba_state_bytes_fp32 ==
                    expected_ssm_state_elems * sizeof(float),
                "ssm-state bytes should match the single Mamba layer footprint") &&
         expect(plan->request_config.expert_selection_capacity ==
                    config.max_tokens * config.experts_per_token,
                "expert scratch capacity should scale with token capacity and top-k routing") &&
         expect(plan->request_config.expert_intermediate_scratch_numel ==
                    config.experts_per_token * config.routed_expert_intermediate_size,
                "forward plan should size expert intermediate scratch for one routed decode step") &&
         expect(plan->request_config.expert_aux_scratch_numel ==
                    (4 * config.hidden_size) +
                    (3 * config.moe_latent_size) +
                    config.shared_expert_intermediate_size +
                    config.n_routed_experts,
                "forward plan should size expert aux scratch for decode-local expert temporaries");
}

}  // namespace

int main() {
  const bool ok = test_single_token_forward_plan_tracks_layer_order_and_state_layout();
  if (!ok) {
    return 1;
  }
  std::cout << "single_token_forward_model_test: PASS\n";
  return 0;
}
