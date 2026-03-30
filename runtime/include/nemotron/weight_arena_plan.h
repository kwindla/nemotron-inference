#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/tensor_catalog.h"

namespace nemotron {

struct WeightArenaBufferPlacement {
  std::string name;
  TensorAuxiliaryRole role = TensorAuxiliaryRole::kUnspecified;
  std::size_t arena_offset_bytes = 0;
  std::size_t nbytes = 0;
  std::size_t alignment_bytes = 0;
  ByteRangeView host_bytes;
};

struct WeightArenaTensorPlacement {
  std::string tensor_name;
  std::string op_class;
  std::string layout_tag;
  std::string storage_dtype;
  std::size_t tensor_alignment_bytes = 0;
  WeightArenaBufferPlacement packed_buffer;
  std::vector<WeightArenaBufferPlacement> auxiliary_buffers;
};

class WeightArenaPlan {
 public:
  bool valid() const;
  std::size_t total_arena_bytes() const;
  std::size_t packed_tensor_bytes() const;
  std::size_t auxiliary_bytes() const;
  const std::vector<WeightArenaTensorPlacement>& tensors() const;
  const std::vector<ManifestValidationIssue>& issues() const;
  const std::unordered_map<std::string, std::size_t>& indices_by_name() const;
  const WeightArenaTensorPlacement* FindTensor(const std::string& tensor_name) const;

 private:
  friend WeightArenaPlan BuildWeightArenaPlan(const TensorCatalog& catalog);

  bool valid_ = false;
  std::size_t total_arena_bytes_ = 0;
  std::size_t packed_tensor_bytes_ = 0;
  std::size_t auxiliary_bytes_ = 0;
  std::vector<WeightArenaTensorPlacement> tensors_;
  std::unordered_map<std::string, std::size_t> indices_by_name_;
  std::vector<ManifestValidationIssue> issues_;
};

WeightArenaPlan BuildWeightArenaPlan(const TensorCatalog& catalog);

}  // namespace nemotron
