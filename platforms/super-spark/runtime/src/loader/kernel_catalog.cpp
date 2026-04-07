#include "nemotron/kernel_catalog.h"

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

}  // namespace

bool KernelTensorDescriptor::is_scaled() const {
  return block_scales_data != nullptr || tensor_scale_data != nullptr;
}

ByteRangeView KernelTensorDescriptor::packed_bytes() const {
  return ByteRangeView{packed_data, packed_nbytes};
}

ByteRangeView KernelTensorDescriptor::block_scales_bytes() const {
  return ByteRangeView{block_scales_data, block_scales_nbytes};
}

ByteRangeView KernelTensorDescriptor::tensor_scale_bytes() const {
  return ByteRangeView{tensor_scale_data, tensor_scale_nbytes};
}

bool KernelCatalog::valid() const {
  return valid_;
}

const std::vector<KernelTensorDescriptor>& KernelCatalog::tensors() const {
  return tensors_;
}

const std::vector<ManifestValidationIssue>& KernelCatalog::issues() const {
  return issues_;
}

const std::unordered_map<std::string, std::size_t>& KernelCatalog::indices_by_name() const {
  return indices_by_name_;
}

const std::unordered_map<std::string, std::vector<std::size_t>>& KernelCatalog::indices_by_op_class() const {
  return indices_by_op_class_;
}

const KernelTensorDescriptor* KernelCatalog::FindTensor(const std::string& tensor_name) const {
  auto it = indices_by_name_.find(tensor_name);
  if (it == indices_by_name_.end()) {
    return nullptr;
  }
  return &tensors_[it->second];
}

KernelCatalog BuildKernelCatalog(
    const TensorCatalog& tensor_catalog,
    const WeightArena& weight_arena) {
  KernelCatalog catalog;
  catalog.issues_ = tensor_catalog.issues();
  if (!tensor_catalog.valid()) {
    return catalog;
  }
  if (!weight_arena.valid()) {
    AddError(&catalog.issues_, "", "weight arena must be valid before building kernel descriptors");
    return catalog;
  }

  catalog.tensors_.reserve(tensor_catalog.tensors().size());
  for (const TensorRuntimeDescriptor& tensor : tensor_catalog.tensors()) {
    const std::optional<WeightArenaTensorView> arena_view = weight_arena.FindTensor(tensor.manifest_entry.name);
    if (!arena_view.has_value()) {
      AddError(&catalog.issues_, tensor.manifest_entry.name, "weight arena could not resolve tensor bytes");
      continue;
    }

    KernelTensorDescriptor descriptor;
    descriptor.tensor_name = tensor.manifest_entry.name;
    descriptor.op_class = tensor.manifest_entry.op_class;
    descriptor.logical_shape = tensor.manifest_entry.logical_shape;
    descriptor.packed_shape = tensor.manifest_entry.packed_shape;
    descriptor.storage_dtype = tensor.manifest_entry.storage_dtype;
    descriptor.compute_dtype = tensor.manifest_entry.compute_dtype;
    descriptor.layout_tag = tensor.manifest_entry.layout_tag;
    descriptor.alignment_bytes = tensor.manifest_entry.alignment_bytes;
    descriptor.packed_data = arena_view->packed_bytes.data;
    descriptor.packed_nbytes = arena_view->packed_bytes.size;

    for (const WeightArenaBufferView& auxiliary : arena_view->auxiliary_buffers) {
      if (auxiliary.placement == nullptr) {
        continue;
      }
      if (auxiliary.placement->role == TensorAuxiliaryRole::kBlockScales) {
        descriptor.block_scales_data = auxiliary.bytes.data;
        descriptor.block_scales_nbytes = auxiliary.bytes.size;
      } else if (auxiliary.placement->role == TensorAuxiliaryRole::kTensorScale) {
        descriptor.tensor_scale_data = auxiliary.bytes.data;
        descriptor.tensor_scale_nbytes = auxiliary.bytes.size;
      }
    }

    if (tensor.has_block_scales() && descriptor.block_scales_data == nullptr) {
      AddError(&catalog.issues_, tensor.manifest_entry.name, "kernel descriptor is missing block_scales bytes");
    }
    if (tensor.has_tensor_scale() && descriptor.tensor_scale_data == nullptr) {
      AddError(&catalog.issues_, tensor.manifest_entry.name, "kernel descriptor is missing tensor_scale bytes");
    }

    const std::size_t index = catalog.tensors_.size();
    catalog.indices_by_name_.emplace(descriptor.tensor_name, index);
    catalog.indices_by_op_class_[descriptor.op_class].push_back(index);
    catalog.tensors_.push_back(std::move(descriptor));
  }

  catalog.valid_ = !HasManifestErrors(catalog.issues_);
  return catalog;
}

KernelCatalog BuildKernelCatalog(const TensorCatalog& tensor_catalog) {
  KernelCatalog catalog;
  catalog.issues_ = tensor_catalog.issues();
  if (!tensor_catalog.valid()) {
    return catalog;
  }

  catalog.tensors_.reserve(tensor_catalog.tensors().size());
  for (const TensorRuntimeDescriptor& tensor : tensor_catalog.tensors()) {
    if (!tensor.packed_bytes.valid()) {
      AddError(
          &catalog.issues_,
          tensor.manifest_entry.name,
          "tensor catalog did not retain valid packed bytes");
      continue;
    }

    KernelTensorDescriptor descriptor;
    descriptor.tensor_name = tensor.manifest_entry.name;
    descriptor.op_class = tensor.manifest_entry.op_class;
    descriptor.logical_shape = tensor.manifest_entry.logical_shape;
    descriptor.packed_shape = tensor.manifest_entry.packed_shape;
    descriptor.storage_dtype = tensor.manifest_entry.storage_dtype;
    descriptor.compute_dtype = tensor.manifest_entry.compute_dtype;
    descriptor.layout_tag = tensor.manifest_entry.layout_tag;
    descriptor.alignment_bytes = tensor.manifest_entry.alignment_bytes;
    descriptor.packed_data = tensor.packed_bytes.data;
    descriptor.packed_nbytes = tensor.packed_bytes.size;

    for (const TensorAuxiliaryDescriptor& auxiliary : tensor.auxiliaries) {
      if (!auxiliary.bytes.valid()) {
        AddError(
            &catalog.issues_,
            tensor.manifest_entry.name,
            "tensor catalog retained an invalid auxiliary byte range");
        continue;
      }
      if (auxiliary.role == TensorAuxiliaryRole::kBlockScales) {
        descriptor.block_scales_data = auxiliary.bytes.data;
        descriptor.block_scales_nbytes = auxiliary.bytes.size;
      } else if (auxiliary.role == TensorAuxiliaryRole::kTensorScale) {
        descriptor.tensor_scale_data = auxiliary.bytes.data;
        descriptor.tensor_scale_nbytes = auxiliary.bytes.size;
      }
    }

    if (tensor.has_block_scales() && descriptor.block_scales_data == nullptr) {
      AddError(&catalog.issues_, tensor.manifest_entry.name, "kernel descriptor is missing block_scales bytes");
    }
    if (tensor.has_tensor_scale() && descriptor.tensor_scale_data == nullptr) {
      AddError(&catalog.issues_, tensor.manifest_entry.name, "kernel descriptor is missing tensor_scale bytes");
    }

    const std::size_t index = catalog.tensors_.size();
    catalog.indices_by_name_.emplace(descriptor.tensor_name, index);
    catalog.indices_by_op_class_[descriptor.op_class].push_back(index);
    catalog.tensors_.push_back(std::move(descriptor));
  }

  catalog.valid_ = !HasManifestErrors(catalog.issues_);
  return catalog;
}

}  // namespace nemotron
