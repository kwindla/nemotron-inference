#include "nemotron/weight_arena_plan.h"

#include <algorithm>

namespace nemotron {
namespace {

constexpr std::size_t kDefaultAuxiliaryAlignmentBytes = 16;

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

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + (alignment - remainder);
}

}  // namespace

bool WeightArenaPlan::valid() const {
  return valid_;
}

std::size_t WeightArenaPlan::total_arena_bytes() const {
  return total_arena_bytes_;
}

std::size_t WeightArenaPlan::packed_tensor_bytes() const {
  return packed_tensor_bytes_;
}

std::size_t WeightArenaPlan::auxiliary_bytes() const {
  return auxiliary_bytes_;
}

const std::vector<WeightArenaTensorPlacement>& WeightArenaPlan::tensors() const {
  return tensors_;
}

const std::vector<ManifestValidationIssue>& WeightArenaPlan::issues() const {
  return issues_;
}

const std::unordered_map<std::string, std::size_t>& WeightArenaPlan::indices_by_name() const {
  return indices_by_name_;
}

const WeightArenaTensorPlacement* WeightArenaPlan::FindTensor(const std::string& tensor_name) const {
  auto it = indices_by_name_.find(tensor_name);
  if (it == indices_by_name_.end()) {
    return nullptr;
  }
  return &tensors_[it->second];
}

WeightArenaPlan BuildWeightArenaPlan(const TensorCatalog& catalog) {
  WeightArenaPlan plan;
  plan.issues_ = catalog.issues();
  if (!catalog.valid()) {
    return plan;
  }

  std::size_t cursor = 0;
  plan.tensors_.reserve(catalog.tensors().size());
  for (const TensorRuntimeDescriptor& tensor : catalog.tensors()) {
    if (!tensor.packed_bytes.valid()) {
      AddError(&plan.issues_, tensor.manifest_entry.name, "tensor catalog entry is missing packed bytes");
      continue;
    }

    WeightArenaTensorPlacement placement;
    placement.tensor_name = tensor.manifest_entry.name;
    placement.op_class = tensor.manifest_entry.op_class;
    placement.layout_tag = tensor.manifest_entry.layout_tag;
    placement.storage_dtype = tensor.manifest_entry.storage_dtype;
    placement.tensor_alignment_bytes = tensor.manifest_entry.alignment_bytes;

    cursor = AlignUp(cursor, tensor.manifest_entry.alignment_bytes);
    placement.packed_buffer = WeightArenaBufferPlacement{
        tensor.manifest_entry.name,
        TensorAuxiliaryRole::kUnspecified,
        cursor,
        tensor.manifest_entry.nbytes,
        tensor.manifest_entry.alignment_bytes,
        tensor.packed_bytes,
    };
    cursor += tensor.manifest_entry.nbytes;
    plan.packed_tensor_bytes_ += tensor.manifest_entry.nbytes;

    placement.auxiliary_buffers.reserve(tensor.auxiliaries.size());
    for (const TensorAuxiliaryDescriptor& auxiliary : tensor.auxiliaries) {
      cursor = AlignUp(cursor, kDefaultAuxiliaryAlignmentBytes);
      placement.auxiliary_buffers.push_back(WeightArenaBufferPlacement{
          auxiliary.manifest_entry.name,
          auxiliary.role,
          cursor,
          auxiliary.manifest_entry.nbytes,
          kDefaultAuxiliaryAlignmentBytes,
          auxiliary.bytes,
      });
      cursor += auxiliary.manifest_entry.nbytes;
      plan.auxiliary_bytes_ += auxiliary.manifest_entry.nbytes;
    }

    const std::size_t index = plan.tensors_.size();
    plan.indices_by_name_.emplace(placement.tensor_name, index);
    plan.tensors_.push_back(std::move(placement));
  }

  plan.total_arena_bytes_ = cursor;
  plan.valid_ = !HasManifestErrors(plan.issues_);
  return plan;
}

}  // namespace nemotron
