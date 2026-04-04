#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/manifest.h"

namespace nemotron {

enum class ModelGlobalRole {
  kOther,
  kEmbedding,
  kFinalNorm,
  kLogits,
};

enum class ModelLayerRole {
  kUnknown,
  kAttention,
  kMamba,
  kExpert,
};

struct GlobalTensorBinding {
  std::string tensor_name;
  std::string op_class;
  ModelGlobalRole role = ModelGlobalRole::kOther;
};

struct LayerTensorBinding {
  std::string tensor_name;
  std::string op_class;
  std::string local_name;
};

struct LayerScheduleEntry {
  std::size_t layer_index = 0;
  std::vector<LayerTensorBinding> bindings;
  std::unordered_map<std::string, std::vector<std::size_t>> binding_indices_by_local_name;
  ModelLayerRole role = ModelLayerRole::kUnknown;
  bool has_attention = false;
  bool has_mamba = false;
  bool has_routed_experts = false;
  bool has_shared_experts = false;
  bool has_router = false;

  const LayerTensorBinding* FindBinding(const std::string& local_name) const;
};

class ModelSchedule {
 public:
  bool valid() const;
  const RuntimeManifestMetadata& runtime_metadata() const;
  const std::vector<LayerScheduleEntry>& ordered_layers() const;
  const std::vector<GlobalTensorBinding>& global_bindings() const;
  const std::vector<ManifestValidationIssue>& issues() const;
  const LayerScheduleEntry* FindLayer(std::size_t layer_index) const;
  std::size_t attention_layer_count() const;
  std::size_t mamba_layer_count() const;
  std::size_t routed_expert_layer_count() const;
  std::size_t shared_expert_layer_count() const;

 private:
  friend ModelSchedule BuildModelSchedule(const PackedModelManifest& manifest);

  bool valid_ = false;
  RuntimeManifestMetadata runtime_metadata_;
  std::vector<LayerScheduleEntry> ordered_layers_;
  std::unordered_map<std::size_t, std::size_t> ordered_layer_indices_;
  std::vector<GlobalTensorBinding> global_bindings_;
  std::vector<ManifestValidationIssue> issues_;
  std::size_t attention_layer_count_ = 0;
  std::size_t mamba_layer_count_ = 0;
  std::size_t routed_expert_layer_count_ = 0;
  std::size_t shared_expert_layer_count_ = 0;
};

ModelSchedule BuildModelSchedule(const PackedModelManifest& manifest);

}  // namespace nemotron
