#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "nemotron/gemm_catalog.h"

namespace nemotron {

struct GemmLaunchPlan {
  const GemmDescriptor* descriptor = nullptr;
  GemmKernelFamily kernel_family = GemmKernelFamily::kDenseRowMajor;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::string tensor_name;
  std::string storage_dtype;
  std::string compute_dtype;
  bool uses_block_scales = false;
  ByteRangeView packed_bytes;
  ByteRangeView block_scales_bytes;
  ByteRangeView tensor_scale_bytes;
  std::string heuristic_key;
};

std::optional<GemmLaunchPlan> BuildGemmLaunchPlan(
    const GemmDescriptor& descriptor,
    std::size_t activation_rows);

class GemmHeuristicCache {
 public:
  bool Store(const GemmLaunchPlan& launch_plan, std::uint64_t algorithm_id);
  std::optional<std::uint64_t> Lookup(const GemmLaunchPlan& launch_plan) const;
  std::size_t size() const;

 private:
  std::unordered_map<std::string, std::uint64_t> algorithm_ids_by_key_;
};

}  // namespace nemotron
