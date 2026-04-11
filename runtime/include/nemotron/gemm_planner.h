#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "nemotron/gemm_catalog.h"

namespace nemotron {

enum class SharedNvfp4ContiguousTileShape {
  k128x128x64,
  k128x128x128,
  k256x128x64,
};

const char* ToString(SharedNvfp4ContiguousTileShape tile_shape);

struct SharedNvfp4ContiguousProfile {
  std::string profile_name;
  SharedNvfp4ContiguousTileShape tile_shape = SharedNvfp4ContiguousTileShape::k128x128x64;
  std::size_t token_bucket_lower = 0;
  std::size_t token_bucket_upper = 0;
  std::size_t threads_per_block = 0;
  bool uses_programmatic_launch = false;
};

struct SharedNvfp4ContiguousPlanRequest {
  std::string tensor_name;
  std::size_t activation_rows = 0;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  ByteRangeView packed_bytes;
  ByteRangeView block_scales_bytes;
  ByteRangeView tensor_scale_bytes;
};

struct GemmLaunchPlan {
  std::string tensor_name;
  const GemmDescriptor* descriptor = nullptr;
  GemmKernelFamily kernel_family = GemmKernelFamily::kDenseRowMajor;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  bool uses_block_scales = false;
  ByteRangeView packed_bytes;
  ByteRangeView block_scales_bytes;
  ByteRangeView tensor_scale_bytes;
  std::string heuristic_key;
  std::string descriptor_cache_key;
  std::string profile_name;
  std::size_t token_bucket_lower = 0;
  std::size_t token_bucket_upper = 0;
  std::size_t tile_m = 0;
  std::size_t tile_n = 0;
  std::size_t tile_k = 0;
  std::size_t cta_m_count = 0;
  std::size_t cta_n_count = 0;
  std::size_t threads_per_block = 0;
  bool uses_programmatic_launch = false;
};

std::optional<GemmLaunchPlan> BuildGemmLaunchPlan(
    const GemmDescriptor& descriptor,
    std::size_t activation_rows);

std::optional<SharedNvfp4ContiguousProfile> SelectSharedNvfp4ContiguousProfile(
    std::size_t activation_rows,
    std::size_t output_rows,
    std::size_t input_cols);

std::optional<GemmLaunchPlan> BuildSharedNvfp4ContiguousLaunchPlan(
    const SharedNvfp4ContiguousPlanRequest& request);

class GemmHeuristicCache {
 public:
  bool Store(const GemmLaunchPlan& launch_plan, std::uint64_t algorithm_id);
  std::optional<std::uint64_t> Lookup(const GemmLaunchPlan& launch_plan) const;
  std::size_t size() const;

 private:
  std::unordered_map<std::string, std::uint64_t> algorithm_ids_by_key_;
};

}  // namespace nemotron
