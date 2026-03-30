#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"

namespace nemotron {

struct KernelTensorDescriptor {
  std::string tensor_name;
  std::string op_class;
  std::vector<std::size_t> logical_shape;
  std::vector<std::size_t> packed_shape;
  std::string storage_dtype;
  std::string compute_dtype;
  std::string layout_tag;
  std::size_t alignment_bytes = 0;
  const std::uint8_t* packed_data = nullptr;
  std::size_t packed_nbytes = 0;
  const std::uint8_t* block_scales_data = nullptr;
  std::size_t block_scales_nbytes = 0;
  const std::uint8_t* tensor_scale_data = nullptr;
  std::size_t tensor_scale_nbytes = 0;

  bool is_scaled() const;
  ByteRangeView packed_bytes() const;
  ByteRangeView block_scales_bytes() const;
  ByteRangeView tensor_scale_bytes() const;
};

class KernelCatalog {
 public:
  bool valid() const;
  const std::vector<KernelTensorDescriptor>& tensors() const;
  const std::vector<ManifestValidationIssue>& issues() const;
  const std::unordered_map<std::string, std::size_t>& indices_by_name() const;
  const std::unordered_map<std::string, std::vector<std::size_t>>& indices_by_op_class() const;
  const KernelTensorDescriptor* FindTensor(const std::string& tensor_name) const;

 private:
  friend KernelCatalog BuildKernelCatalog(
      const TensorCatalog& tensor_catalog,
      const WeightArena& weight_arena);
  friend KernelCatalog BuildKernelCatalog(
      const TensorCatalog& tensor_catalog);

  bool valid_ = false;
  std::vector<KernelTensorDescriptor> tensors_;
  std::unordered_map<std::string, std::size_t> indices_by_name_;
  std::unordered_map<std::string, std::vector<std::size_t>> indices_by_op_class_;
  std::vector<ManifestValidationIssue> issues_;
};

KernelCatalog BuildKernelCatalog(
    const TensorCatalog& tensor_catalog,
    const WeightArena& weight_arena);
KernelCatalog BuildKernelCatalog(
    const TensorCatalog& tensor_catalog);

}  // namespace nemotron
