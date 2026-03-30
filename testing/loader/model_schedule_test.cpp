#include "nemotron/model_schedule.h"

#include <iostream>
#include <string>

namespace {

using nemotron::BuildModelSchedule;
using nemotron::ModelGlobalRole;
using nemotron::PackedModelManifest;
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
  manifest.runtime.model_id = "test-model";
  manifest.runtime.source_revision = "rev";
  manifest.runtime.tokenizer_revision = "tok";
  manifest.runtime.packer_version = "packer";
  manifest.runtime.gpu_family = "GB10";
  manifest.runtime.compute_capability = "12.1";
  manifest.runtime.kv_bytes_per_token = 4096;
  manifest.runtime.mamba_state_bytes_fp16 = 128;
  manifest.runtime.mamba_state_bytes_fp32 = 256;

  manifest.tensors.push_back(make_tensor("backbone.embeddings.weight", "embedding"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.input_norm.weight", "norm"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.self_attn.q_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.self_attn.k_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.self_attn.v_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.0.self_attn.o_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mamba.in_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mamba.conv.weight", "conv1d"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mamba.A_log", "mamba_state"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mamba.dt_bias", "mamba_state"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mamba.out_proj.weight", "dense"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.router.weight", "router"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.experts.0.up_proj", "routed_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.experts.0.down_proj", "routed_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.shared_experts.up_proj", "shared_expert"));
  manifest.tensors.push_back(make_tensor("backbone.layers.1.mixer.shared_experts.down_proj", "shared_expert"));
  manifest.tensors.push_back(make_tensor("backbone.norm_f.weight", "norm"));
  manifest.tensors.push_back(make_tensor("lm_head.weight", "dense"));
  return manifest;
}

bool test_model_schedule_classifies_layers_and_globals() {
  const nemotron::ModelSchedule schedule = BuildModelSchedule(make_manifest());
  const nemotron::LayerScheduleEntry* layer0 = schedule.FindLayer(0);
  const nemotron::LayerScheduleEntry* layer1 = schedule.FindLayer(1);

  bool saw_embedding = false;
  bool saw_final_norm = false;
  bool saw_logits = false;
  for (const auto& binding : schedule.global_bindings()) {
    if (binding.role == ModelGlobalRole::kEmbedding) {
      saw_embedding = true;
    } else if (binding.role == ModelGlobalRole::kFinalNorm) {
      saw_final_norm = true;
    } else if (binding.role == ModelGlobalRole::kLogits) {
      saw_logits = true;
    }
  }

  return expect(schedule.valid(), "model schedule should build for a valid manifest") &&
         expect(schedule.issues().empty(), "schedule should not report issues for the test manifest") &&
         expect(schedule.ordered_layers().size() == 2, "schedule should contain two ordered layers") &&
         expect(schedule.attention_layer_count() == 1, "schedule should count one attention layer") &&
         expect(schedule.mamba_layer_count() == 1, "schedule should count one mamba layer") &&
         expect(schedule.routed_expert_layer_count() == 1, "schedule should count one routed-expert layer") &&
         expect(schedule.shared_expert_layer_count() == 1, "schedule should count one shared-expert layer") &&
         expect(layer0 != nullptr, "layer 0 should be present") &&
         expect(layer1 != nullptr, "layer 1 should be present") &&
         expect(layer0->has_attention, "layer 0 should be classified as attention") &&
         expect(!layer0->has_mamba, "layer 0 should not be classified as mamba") &&
         expect(layer0->FindBinding("self_attn.q_proj.weight") != nullptr,
                "layer 0 should expose its local q-projection binding") &&
         expect(layer1->has_mamba, "layer 1 should be classified as mamba") &&
         expect(layer1->has_router, "layer 1 should be classified as having a router") &&
         expect(layer1->has_routed_experts, "layer 1 should be classified as having routed experts") &&
         expect(layer1->has_shared_experts, "layer 1 should be classified as having shared experts") &&
         expect(layer1->FindBinding("mixer.experts.0.up_proj") != nullptr,
                "layer 1 should expose routed-expert local bindings") &&
         expect(saw_embedding, "global bindings should include the embedding tensor") &&
         expect(saw_final_norm, "global bindings should include the final norm tensor") &&
         expect(saw_logits, "global bindings should include the logits tensor");
}

}  // namespace

int main() {
  const bool ok = test_model_schedule_classifies_layers_and_globals();
  if (!ok) {
    return 1;
  }
  std::cout << "model_schedule_test: PASS\n";
  return 0;
}
