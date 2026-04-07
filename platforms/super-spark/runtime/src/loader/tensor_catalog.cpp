#include "nemotron/tensor_catalog.h"

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

TensorAuxiliaryRole ClassifyAuxiliaryRole(const std::string& name) {
  if (name == "block_scales") {
    return TensorAuxiliaryRole::kBlockScales;
  }
  if (name == "tensor_scale") {
    return TensorAuxiliaryRole::kTensorScale;
  }
  return TensorAuxiliaryRole::kUnspecified;
}

bool IsScaledTensor(const TensorManifestEntry& tensor) {
  return !tensor.block_scale_mode.empty() ||
         !tensor.block_scale_dtype.empty() ||
         !tensor.tensor_scale_dtype.empty();
}

}  // namespace

const TensorAuxiliaryDescriptor* TensorRuntimeDescriptor::FindAuxiliary(
    const std::string& name) const {
  for (const TensorAuxiliaryDescriptor& auxiliary : auxiliaries) {
    if (auxiliary.manifest_entry.name == name) {
      return &auxiliary;
    }
  }
  return nullptr;
}

const TensorAuxiliaryDescriptor* TensorRuntimeDescriptor::FindAuxiliary(
    TensorAuxiliaryRole role) const {
  for (const TensorAuxiliaryDescriptor& auxiliary : auxiliaries) {
    if (auxiliary.role == role) {
      return &auxiliary;
    }
  }
  return nullptr;
}

bool TensorRuntimeDescriptor::has_block_scales() const {
  return FindAuxiliary(TensorAuxiliaryRole::kBlockScales) != nullptr;
}

bool TensorRuntimeDescriptor::has_tensor_scale() const {
  return FindAuxiliary(TensorAuxiliaryRole::kTensorScale) != nullptr;
}

bool TensorCatalog::valid() const {
  return valid_;
}

const RuntimeManifestMetadata& TensorCatalog::runtime_metadata() const {
  return runtime_metadata_;
}

const std::vector<TensorRuntimeDescriptor>& TensorCatalog::tensors() const {
  return tensors_;
}

const std::vector<ManifestValidationIssue>& TensorCatalog::issues() const {
  return issues_;
}

const std::unordered_map<std::string, std::size_t>& TensorCatalog::indices_by_name() const {
  return indices_by_name_;
}

const std::unordered_map<std::string, std::vector<std::size_t>>& TensorCatalog::indices_by_op_class() const {
  return indices_by_op_class_;
}

const TensorRuntimeDescriptor* TensorCatalog::FindTensor(const std::string& tensor_name) const {
  auto it = indices_by_name_.find(tensor_name);
  if (it == indices_by_name_.end()) {
    return nullptr;
  }
  return &tensors_[it->second];
}

TensorCatalog BuildTensorCatalog(
    const PackedModelManifest& manifest,
    const ArtifactLoader& artifact_loader) {
  TensorCatalog catalog;
  catalog.runtime_metadata_ = manifest.runtime;
  catalog.issues_ = ValidateManifest(manifest);
  if (HasManifestErrors(catalog.issues_)) {
    return catalog;
  }

  catalog.tensors_.reserve(manifest.tensors.size());
  for (const TensorManifestEntry& tensor : manifest.tensors) {
    const std::optional<TensorArtifactView> artifact_view = artifact_loader.FindTensor(tensor.name);
    if (!artifact_view.has_value()) {
      AddError(&catalog.issues_, tensor.name, "artifact loader could not resolve tensor bytes");
      continue;
    }

    TensorRuntimeDescriptor descriptor;
    descriptor.manifest_entry = tensor;
    descriptor.packed_bytes = artifact_view->packed_bytes;
    descriptor.auxiliaries.reserve(artifact_view->auxiliaries.size());

    bool saw_block_scales = false;
    bool saw_tensor_scale = false;
    for (std::size_t auxiliary_index = 0; auxiliary_index < tensor.auxiliaries.size(); ++auxiliary_index) {
      const TensorAuxiliaryManifest& auxiliary_manifest = tensor.auxiliaries[auxiliary_index];
      const AuxiliaryArtifactView& auxiliary_view = artifact_view->auxiliaries[auxiliary_index];
      const TensorAuxiliaryRole role = ClassifyAuxiliaryRole(auxiliary_manifest.name);
      if (role == TensorAuxiliaryRole::kBlockScales) {
        if (saw_block_scales) {
          AddError(&catalog.issues_, tensor.name, "duplicate block_scales auxiliary descriptor");
        }
        saw_block_scales = true;
      } else if (role == TensorAuxiliaryRole::kTensorScale) {
        if (saw_tensor_scale) {
          AddError(&catalog.issues_, tensor.name, "duplicate tensor_scale auxiliary descriptor");
        }
        saw_tensor_scale = true;
      }

      descriptor.auxiliaries.push_back(TensorAuxiliaryDescriptor{
          role,
          auxiliary_manifest,
          auxiliary_view.bytes,
      });
    }

    if (IsScaledTensor(tensor)) {
      if (!saw_block_scales) {
        AddError(
            &catalog.issues_,
            tensor.name,
            "scaled tensor is missing required block_scales auxiliary descriptor");
      }
      if (!saw_tensor_scale) {
        AddError(
            &catalog.issues_,
            tensor.name,
            "scaled tensor is missing required tensor_scale auxiliary descriptor");
      }
    }

    const std::size_t tensor_index = catalog.tensors_.size();
    catalog.indices_by_name_.emplace(tensor.name, tensor_index);
    catalog.indices_by_op_class_[tensor.op_class].push_back(tensor_index);
    catalog.tensors_.push_back(std::move(descriptor));
  }

  catalog.valid_ = !HasManifestErrors(catalog.issues_);
  return catalog;
}

}  // namespace nemotron
