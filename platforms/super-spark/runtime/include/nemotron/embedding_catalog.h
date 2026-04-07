#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/kernel_catalog.h"

namespace nemotron {

struct EmbeddingDescriptor {
  std::string tensor_name;
  std::string op_class;
  std::size_t vocab_size = 0;
  std::size_t embedding_dim = 0;
  std::string storage_dtype;
  std::string compute_dtype;
  std::string layout_tag;
  std::size_t alignment_bytes = 0;
  const std::uint8_t* packed_data = nullptr;
  std::size_t packed_nbytes = 0;

  ByteRangeView packed_bytes() const;
};

class EmbeddingCatalog {
 public:
  bool valid() const;
  const std::vector<EmbeddingDescriptor>& descriptors() const;
  const std::vector<ManifestValidationIssue>& issues() const;
  const std::unordered_map<std::string, std::size_t>& indices_by_name() const;
  const std::unordered_map<std::string, std::vector<std::size_t>>& indices_by_op_class() const;
  const EmbeddingDescriptor* FindDescriptor(const std::string& tensor_name) const;

 private:
  friend EmbeddingCatalog BuildEmbeddingCatalog(const KernelCatalog& kernel_catalog);

  bool valid_ = false;
  std::vector<EmbeddingDescriptor> descriptors_;
  std::unordered_map<std::string, std::size_t> indices_by_name_;
  std::unordered_map<std::string, std::vector<std::size_t>> indices_by_op_class_;
  std::vector<ManifestValidationIssue> issues_;
};

EmbeddingCatalog BuildEmbeddingCatalog(const KernelCatalog& kernel_catalog);

}  // namespace nemotron
