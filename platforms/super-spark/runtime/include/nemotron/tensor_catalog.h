#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/artifact_loader.h"
#include "nemotron/manifest.h"

namespace nemotron {

enum class TensorAuxiliaryRole {
  kUnspecified,
  kBlockScales,
  kTensorScale,
};

struct TensorAuxiliaryDescriptor {
  TensorAuxiliaryRole role = TensorAuxiliaryRole::kUnspecified;
  TensorAuxiliaryManifest manifest_entry;
  ByteRangeView bytes;
};

struct TensorRuntimeDescriptor {
  TensorManifestEntry manifest_entry;
  ByteRangeView packed_bytes;
  std::vector<TensorAuxiliaryDescriptor> auxiliaries;

  const TensorAuxiliaryDescriptor* FindAuxiliary(const std::string& name) const;
  const TensorAuxiliaryDescriptor* FindAuxiliary(TensorAuxiliaryRole role) const;
  bool has_block_scales() const;
  bool has_tensor_scale() const;
};

class TensorCatalog {
 public:
  bool valid() const;
  const RuntimeManifestMetadata& runtime_metadata() const;
  const std::vector<TensorRuntimeDescriptor>& tensors() const;
  const std::vector<ManifestValidationIssue>& issues() const;
  const std::unordered_map<std::string, std::size_t>& indices_by_name() const;
  const std::unordered_map<std::string, std::vector<std::size_t>>& indices_by_op_class() const;
  const TensorRuntimeDescriptor* FindTensor(const std::string& tensor_name) const;

 private:
  friend TensorCatalog BuildTensorCatalog(
      const PackedModelManifest& manifest,
      const ArtifactLoader& artifact_loader);

  bool valid_ = false;
  RuntimeManifestMetadata runtime_metadata_;
  std::vector<TensorRuntimeDescriptor> tensors_;
  std::unordered_map<std::string, std::size_t> indices_by_name_;
  std::unordered_map<std::string, std::vector<std::size_t>> indices_by_op_class_;
  std::vector<ManifestValidationIssue> issues_;
};

TensorCatalog BuildTensorCatalog(
    const PackedModelManifest& manifest,
    const ArtifactLoader& artifact_loader);

}  // namespace nemotron
