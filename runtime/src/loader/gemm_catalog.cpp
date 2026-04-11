#include "nemotron/gemm_catalog.h"

#include <sstream>

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

bool LooksLikeGemmWeight(const KernelTensorDescriptor& tensor) {
  return tensor.logical_shape.size() == 2 && tensor.op_class != "embedding";
}

}  // namespace

const char* ToString(GemmKernelFamily family) {
  switch (family) {
    case GemmKernelFamily::kDenseRowMajor:
      return "dense_row_major";
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      return "cublaslt_nvfp4_block_scaled";
    case GemmKernelFamily::kSm120ContiguousSharedNvfp4:
      return "sm120_contiguous_shared_nvfp4";
  }
  return "unknown";
}

bool GemmDescriptor::is_scaled() const {
  return block_scales_data != nullptr || tensor_scale_data != nullptr;
}

ByteRangeView GemmDescriptor::packed_bytes() const {
  return ByteRangeView{packed_data, packed_nbytes};
}

ByteRangeView GemmDescriptor::block_scales_bytes() const {
  return ByteRangeView{block_scales_data, block_scales_nbytes};
}

ByteRangeView GemmDescriptor::tensor_scale_bytes() const {
  return ByteRangeView{tensor_scale_data, tensor_scale_nbytes};
}

std::string GemmDescriptor::heuristic_key_prefix() const {
  std::ostringstream oss;
  oss << ToString(kernel_family)
      << '|'
      << layout_tag
      << '|'
      << storage_dtype
      << '|'
      << compute_dtype
      << '|'
      << output_rows
      << 'x'
      << input_cols;
  return oss.str();
}

bool GemmCatalog::valid() const {
  return valid_;
}

const std::vector<GemmDescriptor>& GemmCatalog::descriptors() const {
  return descriptors_;
}

const std::vector<ManifestValidationIssue>& GemmCatalog::issues() const {
  return issues_;
}

const std::unordered_map<std::string, std::size_t>& GemmCatalog::indices_by_name() const {
  return indices_by_name_;
}

const std::unordered_map<std::string, std::vector<std::size_t>>& GemmCatalog::indices_by_op_class() const {
  return indices_by_op_class_;
}

const std::unordered_map<int, std::vector<std::size_t>>& GemmCatalog::indices_by_family() const {
  return indices_by_family_;
}

const GemmDescriptor* GemmCatalog::FindDescriptor(const std::string& tensor_name) const {
  auto it = indices_by_name_.find(tensor_name);
  if (it == indices_by_name_.end()) {
    return nullptr;
  }
  return &descriptors_[it->second];
}

GemmCatalog BuildGemmCatalog(const KernelCatalog& kernel_catalog) {
  GemmCatalog catalog;
  catalog.issues_ = kernel_catalog.issues();
  if (!kernel_catalog.valid()) {
    return catalog;
  }

  for (const KernelTensorDescriptor& tensor : kernel_catalog.tensors()) {
    if (!LooksLikeGemmWeight(tensor)) {
      continue;
    }

    GemmDescriptor descriptor;
    descriptor.tensor_name = tensor.tensor_name;
    descriptor.op_class = tensor.op_class;
    descriptor.output_rows = tensor.logical_shape[0];
    descriptor.input_cols = tensor.logical_shape[1];
    descriptor.storage_dtype = tensor.storage_dtype;
    descriptor.compute_dtype = tensor.compute_dtype;
    descriptor.layout_tag = tensor.layout_tag;
    descriptor.alignment_bytes = tensor.alignment_bytes;
    descriptor.packed_data = tensor.packed_data;
    descriptor.packed_nbytes = tensor.packed_nbytes;

    if (tensor.layout_tag == "cublaslt_fp4_tn_v1") {
      if (tensor.storage_dtype != "nvfp4_e2m1") {
        AddError(&catalog.issues_, tensor.tensor_name, "NVFP4 GEMM descriptor requires nvfp4_e2m1 storage");
        continue;
      }
      if (tensor.compute_dtype != "fp32_accum") {
        AddError(&catalog.issues_, tensor.tensor_name, "NVFP4 GEMM descriptor requires fp32_accum compute dtype");
        continue;
      }
      if (!tensor.block_scales_bytes().valid() || !tensor.tensor_scale_bytes().valid()) {
        AddError(&catalog.issues_, tensor.tensor_name, "NVFP4 GEMM descriptor requires block and tensor scale bytes");
        continue;
      }
      descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
      descriptor.block_scales_data = tensor.block_scales_data;
      descriptor.block_scales_nbytes = tensor.block_scales_nbytes;
      descriptor.tensor_scale_data = tensor.tensor_scale_data;
      descriptor.tensor_scale_nbytes = tensor.tensor_scale_nbytes;
    } else if (tensor.layout_tag == "row_major") {
      descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
    } else {
      AddError(
          &catalog.issues_,
          tensor.tensor_name,
          "unsupported GEMM layout_tag for first operator descriptor layer: " + tensor.layout_tag);
      continue;
    }

    const std::size_t index = catalog.descriptors_.size();
    catalog.indices_by_name_.emplace(descriptor.tensor_name, index);
    catalog.indices_by_op_class_[descriptor.op_class].push_back(index);
    catalog.indices_by_family_[static_cast<int>(descriptor.kernel_family)].push_back(index);
    catalog.descriptors_.push_back(std::move(descriptor));
  }

  catalog.valid_ = !HasManifestErrors(catalog.issues_);
  return catalog;
}

}  // namespace nemotron
