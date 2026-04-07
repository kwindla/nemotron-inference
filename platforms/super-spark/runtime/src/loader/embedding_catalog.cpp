#include "nemotron/embedding_catalog.h"

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

bool LooksLikeEmbeddingWeight(const KernelTensorDescriptor& tensor) {
  return tensor.op_class == "embedding";
}

}  // namespace

ByteRangeView EmbeddingDescriptor::packed_bytes() const {
  return ByteRangeView{packed_data, packed_nbytes};
}

bool EmbeddingCatalog::valid() const {
  return valid_;
}

const std::vector<EmbeddingDescriptor>& EmbeddingCatalog::descriptors() const {
  return descriptors_;
}

const std::vector<ManifestValidationIssue>& EmbeddingCatalog::issues() const {
  return issues_;
}

const std::unordered_map<std::string, std::size_t>& EmbeddingCatalog::indices_by_name() const {
  return indices_by_name_;
}

const std::unordered_map<std::string, std::vector<std::size_t>>& EmbeddingCatalog::indices_by_op_class() const {
  return indices_by_op_class_;
}

const EmbeddingDescriptor* EmbeddingCatalog::FindDescriptor(const std::string& tensor_name) const {
  auto it = indices_by_name_.find(tensor_name);
  if (it == indices_by_name_.end()) {
    return nullptr;
  }
  return &descriptors_[it->second];
}

EmbeddingCatalog BuildEmbeddingCatalog(const KernelCatalog& kernel_catalog) {
  EmbeddingCatalog catalog;
  catalog.issues_ = kernel_catalog.issues();
  if (!kernel_catalog.valid()) {
    return catalog;
  }

  for (const KernelTensorDescriptor& tensor : kernel_catalog.tensors()) {
    if (!LooksLikeEmbeddingWeight(tensor)) {
      continue;
    }
    if (tensor.logical_shape.size() != 2) {
      AddError(&catalog.issues_, tensor.tensor_name, "embedding descriptor requires a 2D logical shape");
      continue;
    }
    if (tensor.layout_tag != "row_major") {
      AddError(
          &catalog.issues_,
          tensor.tensor_name,
          "unsupported embedding layout_tag for first embedding descriptor layer: " + tensor.layout_tag);
      continue;
    }
    if (tensor.is_scaled()) {
      AddError(&catalog.issues_, tensor.tensor_name, "first embedding descriptor layer does not support auxiliary scale metadata");
      continue;
    }
    if (!tensor.packed_bytes().valid()) {
      AddError(&catalog.issues_, tensor.tensor_name, "embedding descriptor requires valid packed bytes");
      continue;
    }

    EmbeddingDescriptor descriptor;
    descriptor.tensor_name = tensor.tensor_name;
    descriptor.op_class = tensor.op_class;
    descriptor.vocab_size = tensor.logical_shape[0];
    descriptor.embedding_dim = tensor.logical_shape[1];
    descriptor.storage_dtype = tensor.storage_dtype;
    descriptor.compute_dtype = tensor.compute_dtype;
    descriptor.layout_tag = tensor.layout_tag;
    descriptor.alignment_bytes = tensor.alignment_bytes;
    descriptor.packed_data = tensor.packed_data;
    descriptor.packed_nbytes = tensor.packed_nbytes;

    const std::size_t index = catalog.descriptors_.size();
    catalog.indices_by_name_.emplace(descriptor.tensor_name, index);
    catalog.indices_by_op_class_[descriptor.op_class].push_back(index);
    catalog.descriptors_.push_back(std::move(descriptor));
  }

  catalog.valid_ = !HasManifestErrors(catalog.issues_);
  return catalog;
}

}  // namespace nemotron
