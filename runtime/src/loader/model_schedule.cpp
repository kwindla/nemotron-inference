#include "nemotron/model_schedule.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <utility>

namespace nemotron {
namespace {

void AddIssue(
    std::vector<ManifestValidationIssue>* issues,
    ManifestIssueSeverity severity,
    const std::string& tensor_name,
    const std::string& message) {
  issues->push_back(ManifestValidationIssue{
      severity,
      tensor_name,
      message,
  });
}

void AddError(
    std::vector<ManifestValidationIssue>* issues,
    const std::string& tensor_name,
    const std::string& message) {
  AddIssue(issues, ManifestIssueSeverity::kError, tensor_name, message);
}

struct LayerPath {
  std::size_t layer_index = 0;
  std::string local_name;
};

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

std::optional<LayerPath> ParseLayerPath(const std::string& tensor_name) {
  constexpr const char* kPrefixes[] = {
      "backbone.layers.",
      "layers.",
  };

  for (const char* prefix : kPrefixes) {
    const std::string prefix_string(prefix);
    if (!starts_with(tensor_name, prefix_string)) {
      continue;
    }

    std::size_t cursor = prefix_string.size();
    if (cursor >= tensor_name.size() ||
        !std::isdigit(static_cast<unsigned char>(tensor_name[cursor]))) {
      return std::nullopt;
    }

    std::size_t layer_index = 0;
    while (cursor < tensor_name.size() &&
           std::isdigit(static_cast<unsigned char>(tensor_name[cursor]))) {
      layer_index = (layer_index * 10) + static_cast<std::size_t>(tensor_name[cursor] - '0');
      ++cursor;
    }

    if (cursor >= tensor_name.size() || tensor_name[cursor] != '.') {
      return std::nullopt;
    }
    ++cursor;
    if (cursor >= tensor_name.size()) {
      return std::nullopt;
    }

    return LayerPath{
        layer_index,
        tensor_name.substr(cursor),
    };
  }

  return std::nullopt;
}

ModelLayerRole ClassifyLayerRoleFromOpClass(const std::string& op_class) {
  if (op_class == "attention") {
    return ModelLayerRole::kAttention;
  }
  if (op_class == "mamba_linear" ||
      op_class == "mamba_param" ||
      op_class == "mamba_state" ||
      op_class == "conv1d") {
    return ModelLayerRole::kMamba;
  }
  if (op_class == "router" ||
      op_class == "router_bias" ||
      op_class == "routed_expert" ||
      op_class == "routed_expert_up" ||
      op_class == "routed_expert_down" ||
      op_class == "shared_expert" ||
      op_class == "shared_expert_up" ||
      op_class == "shared_expert_down") {
    return ModelLayerRole::kExpert;
  }
  return ModelLayerRole::kUnknown;
}

bool IsRouterOpClass(const std::string& op_class) {
  return op_class == "router" || op_class == "router_bias";
}

bool IsRoutedExpertOpClass(const std::string& op_class) {
  return op_class == "routed_expert" ||
         op_class == "routed_expert_up" ||
         op_class == "routed_expert_down";
}

bool IsSharedExpertOpClass(const std::string& op_class) {
  return op_class == "shared_expert" ||
         op_class == "shared_expert_up" ||
         op_class == "shared_expert_down";
}

const char* ModelLayerRoleName(ModelLayerRole role) {
  switch (role) {
    case ModelLayerRole::kAttention:
      return "attention";
    case ModelLayerRole::kMamba:
      return "mamba";
    case ModelLayerRole::kExpert:
      return "expert";
    case ModelLayerRole::kUnknown:
    default:
      return "unknown";
  }
}

ModelGlobalRole ClassifyGlobalRole(const TensorManifestEntry& tensor) {
  if (tensor.op_class == "embedding" ||
      tensor.name == "backbone.embeddings.weight" ||
      tensor.name == "embeddings.weight") {
    return ModelGlobalRole::kEmbedding;
  }
  if (tensor.op_class == "final_norm" ||
      tensor.name == "backbone.final_norm.weight" ||
      tensor.name == "final_norm.weight" ||
      tensor.name == "backbone.norm_f.weight" ||
      tensor.name == "norm_f.weight") {
    return ModelGlobalRole::kFinalNorm;
  }
  if (tensor.op_class == "logits" ||
      tensor.name == "lm_head.weight" ||
      tensor.name == "output.weight" ||
      tensor.name == "logits.weight") {
    return ModelGlobalRole::kLogits;
  }
  return ModelGlobalRole::kOther;
}

}  // namespace

const LayerTensorBinding* LayerScheduleEntry::FindBinding(const std::string& local_name) const {
  auto it = binding_indices_by_local_name.find(local_name);
  if (it == binding_indices_by_local_name.end() || it->second.empty()) {
    return nullptr;
  }
  return &bindings[it->second.front()];
}

bool ModelSchedule::valid() const {
  return valid_;
}

const RuntimeManifestMetadata& ModelSchedule::runtime_metadata() const {
  return runtime_metadata_;
}

const std::vector<LayerScheduleEntry>& ModelSchedule::ordered_layers() const {
  return ordered_layers_;
}

const std::vector<GlobalTensorBinding>& ModelSchedule::global_bindings() const {
  return global_bindings_;
}

const std::vector<ManifestValidationIssue>& ModelSchedule::issues() const {
  return issues_;
}

const LayerScheduleEntry* ModelSchedule::FindLayer(std::size_t layer_index) const {
  auto it = ordered_layer_indices_.find(layer_index);
  if (it == ordered_layer_indices_.end()) {
    return nullptr;
  }
  return &ordered_layers_[it->second];
}

std::size_t ModelSchedule::attention_layer_count() const {
  return attention_layer_count_;
}

std::size_t ModelSchedule::mamba_layer_count() const {
  return mamba_layer_count_;
}

std::size_t ModelSchedule::routed_expert_layer_count() const {
  return routed_expert_layer_count_;
}

std::size_t ModelSchedule::shared_expert_layer_count() const {
  return shared_expert_layer_count_;
}

ModelSchedule BuildModelSchedule(const PackedModelManifest& manifest) {
  ModelSchedule schedule;
  schedule.runtime_metadata_ = manifest.runtime;
  schedule.issues_ = ValidateManifest(manifest);
  if (HasManifestErrors(schedule.issues_)) {
    return schedule;
  }

  std::unordered_map<std::size_t, std::size_t> layer_indices;

  for (const TensorManifestEntry& tensor : manifest.tensors) {
    const std::optional<LayerPath> layer_path = ParseLayerPath(tensor.name);
    if (!layer_path.has_value()) {
      schedule.global_bindings_.push_back(GlobalTensorBinding{
          tensor.name,
          tensor.op_class,
          ClassifyGlobalRole(tensor),
      });
      continue;
    }

    const std::size_t layer_index = layer_path->layer_index;
    std::size_t ordered_index = 0;
    auto layer_it = layer_indices.find(layer_index);
    if (layer_it == layer_indices.end()) {
      ordered_index = schedule.ordered_layers_.size();
      layer_indices.emplace(layer_index, ordered_index);
      schedule.ordered_layers_.push_back(LayerScheduleEntry{
          layer_index,
      });
    } else {
      ordered_index = layer_it->second;
    }

    LayerScheduleEntry& entry = schedule.ordered_layers_[ordered_index];
    const std::size_t binding_index = entry.bindings.size();
    entry.bindings.push_back(LayerTensorBinding{
        tensor.name,
        tensor.op_class,
        layer_path->local_name,
    });
    entry.binding_indices_by_local_name[layer_path->local_name].push_back(binding_index);

    if (entry.binding_indices_by_local_name[layer_path->local_name].size() > 1) {
      AddError(
          &schedule.issues_,
          tensor.name,
          "duplicate layer-local tensor binding '" + layer_path->local_name + "'");
    }

    const ModelLayerRole binding_role = ClassifyLayerRoleFromOpClass(tensor.op_class);
    if (binding_role != ModelLayerRole::kUnknown) {
      if (entry.role == ModelLayerRole::kUnknown) {
        entry.role = binding_role;
      } else if (entry.role != binding_role) {
        AddError(
            &schedule.issues_,
            tensor.name,
            "layer mixes explicit manifest roles '" +
                std::string(ModelLayerRoleName(entry.role)) +
                "' and '" + std::string(ModelLayerRoleName(binding_role)) + "'");
      }
    }

    entry.has_attention =
        entry.has_attention || binding_role == ModelLayerRole::kAttention;
    entry.has_mamba =
        entry.has_mamba || binding_role == ModelLayerRole::kMamba;
    entry.has_router = entry.has_router || IsRouterOpClass(tensor.op_class);
    entry.has_routed_experts =
        entry.has_routed_experts || IsRoutedExpertOpClass(tensor.op_class);
    entry.has_shared_experts =
        entry.has_shared_experts || IsSharedExpertOpClass(tensor.op_class);
  }

  std::sort(
      schedule.ordered_layers_.begin(),
      schedule.ordered_layers_.end(),
      [](const LayerScheduleEntry& lhs, const LayerScheduleEntry& rhs) {
        return lhs.layer_index < rhs.layer_index;
      });

  for (std::size_t ordered_index = 0; ordered_index < schedule.ordered_layers_.size(); ++ordered_index) {
    const LayerScheduleEntry& entry = schedule.ordered_layers_[ordered_index];
    schedule.ordered_layer_indices_[entry.layer_index] = ordered_index;
    if (entry.role == ModelLayerRole::kUnknown) {
      AddError(
          &schedule.issues_,
          "layer:" + std::to_string(entry.layer_index),
          "layer is missing an explicit manifest role op_class");
      continue;
    }
    if (entry.role == ModelLayerRole::kAttention) {
      ++schedule.attention_layer_count_;
    }
    if (entry.role == ModelLayerRole::kMamba) {
      ++schedule.mamba_layer_count_;
    }
    if (entry.role == ModelLayerRole::kExpert &&
        (entry.has_routed_experts || entry.has_router)) {
      ++schedule.routed_expert_layer_count_;
    }
    if (entry.role == ModelLayerRole::kExpert && entry.has_shared_experts) {
      ++schedule.shared_expert_layer_count_;
    }
  }

  schedule.valid_ = !HasManifestErrors(schedule.issues_);
  return schedule;
}

}  // namespace nemotron
