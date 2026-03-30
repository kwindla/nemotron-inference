#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/manifest.h"
#include "nemotron/memory_budget.h"

namespace nemotron {

struct LoaderPlan {
  bool valid = false;
  RuntimeMemoryProfile memory_profile;
  MemoryBudgetSummary memory_budget;
  std::size_t tensor_count = 0;
  std::size_t total_packed_bytes = 0;
  std::size_t total_auxiliary_bytes = 0;
  std::unordered_map<std::string, std::size_t> bytes_by_op_class;
  std::unordered_map<std::string, std::size_t> bytes_by_file;
  std::vector<ManifestValidationIssue> issues;
};

LoaderPlan BuildLoaderPlan(
    const PackedModelManifest& manifest,
    const ServiceMemoryTarget& target,
    bool use_fp16_mamba_state,
    std::size_t reusable_node_metadata_bytes);

}  // namespace nemotron
