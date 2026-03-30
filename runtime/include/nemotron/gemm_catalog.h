#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/kernel_catalog.h"

namespace nemotron {

enum class GemmKernelFamily {
  kDenseRowMajor,
  kCublasLtNvfp4BlockScaled,
};

const char* ToString(GemmKernelFamily family);

struct GemmDescriptor {
  std::string tensor_name;
  std::string op_class;
  GemmKernelFamily kernel_family = GemmKernelFamily::kDenseRowMajor;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
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
  std::string heuristic_key_prefix() const;
};

class GemmCatalog {
 public:
  bool valid() const;
  const std::vector<GemmDescriptor>& descriptors() const;
  const std::vector<ManifestValidationIssue>& issues() const;
  const std::unordered_map<std::string, std::size_t>& indices_by_name() const;
  const std::unordered_map<std::string, std::vector<std::size_t>>& indices_by_op_class() const;
  const std::unordered_map<int, std::vector<std::size_t>>& indices_by_family() const;
  const GemmDescriptor* FindDescriptor(const std::string& tensor_name) const;

 private:
  friend GemmCatalog BuildGemmCatalog(const KernelCatalog& kernel_catalog);

  bool valid_ = false;
  std::vector<GemmDescriptor> descriptors_;
  std::unordered_map<std::string, std::size_t> indices_by_name_;
  std::unordered_map<std::string, std::vector<std::size_t>> indices_by_op_class_;
  std::unordered_map<int, std::vector<std::size_t>> indices_by_family_;
  std::vector<ManifestValidationIssue> issues_;
};

GemmCatalog BuildGemmCatalog(const KernelCatalog& kernel_catalog);

}  // namespace nemotron
