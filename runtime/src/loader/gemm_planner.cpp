#include "nemotron/gemm_planner.h"

#include <limits>
#include <sstream>

namespace nemotron {
namespace {

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return denominator == 0 ? 0 : (numerator + denominator - 1u) / denominator;
}

void TileShapeDims(
    SharedNvfp4ContiguousTileShape tile_shape,
    std::size_t* tile_m,
    std::size_t* tile_n,
    std::size_t* tile_k) {
  switch (tile_shape) {
    case SharedNvfp4ContiguousTileShape::k128x128x64:
      *tile_m = 128;
      *tile_n = 128;
      *tile_k = 64;
      return;
    case SharedNvfp4ContiguousTileShape::k128x128x128:
      *tile_m = 128;
      *tile_n = 128;
      *tile_k = 128;
      return;
    case SharedNvfp4ContiguousTileShape::k256x128x64:
      *tile_m = 256;
      *tile_n = 128;
      *tile_k = 64;
      return;
  }
  *tile_m = 0;
  *tile_n = 0;
  *tile_k = 0;
}

std::string BucketUpperLabel(std::size_t upper) {
  if (upper == std::numeric_limits<std::size_t>::max()) {
    return "inf";
  }
  return std::to_string(upper);
}

std::string SharedContiguousHeuristicKey(
    const GemmLaunchPlan& plan,
    bool include_exact_rows) {
  std::ostringstream oss;
  oss << ToString(plan.kernel_family)
      << '|'
      << plan.tensor_name
      << "|profile="
      << plan.profile_name
      << "|tile="
      << plan.tile_m
      << 'x'
      << plan.tile_n
      << 'x'
      << plan.tile_k
      << "|bucket="
      << plan.token_bucket_lower
      << '-'
      << BucketUpperLabel(plan.token_bucket_upper)
      << "|n="
      << plan.n
      << "|k="
      << plan.k;
  if (include_exact_rows) {
    oss << "|m=" << plan.m
        << "|cta_m=" << plan.cta_m_count
        << "|cta_n=" << plan.cta_n_count;
  }
  return oss.str();
}

}  // namespace

const char* ToString(SharedNvfp4ContiguousTileShape tile_shape) {
  switch (tile_shape) {
    case SharedNvfp4ContiguousTileShape::k128x128x64:
      return "128x128x64";
    case SharedNvfp4ContiguousTileShape::k128x128x128:
      return "128x128x128";
    case SharedNvfp4ContiguousTileShape::k256x128x64:
      return "256x128x64";
  }
  return "unknown";
}

std::optional<GemmLaunchPlan> BuildGemmLaunchPlan(
    const GemmDescriptor& descriptor,
    std::size_t activation_rows) {
  if (activation_rows == 0 || !descriptor.packed_bytes().valid()) {
    return std::nullopt;
  }

  GemmLaunchPlan plan;
  plan.tensor_name = descriptor.tensor_name;
  plan.descriptor = &descriptor;
  plan.kernel_family = descriptor.kernel_family;
  plan.m = activation_rows;
  plan.n = descriptor.output_rows;
  plan.k = descriptor.input_cols;
  plan.uses_block_scales = descriptor.is_scaled();
  plan.packed_bytes = descriptor.packed_bytes();
  plan.block_scales_bytes = descriptor.block_scales_bytes();
  plan.tensor_scale_bytes = descriptor.tensor_scale_bytes();
  plan.heuristic_key =
      descriptor.heuristic_key_prefix() + "|m=" + std::to_string(plan.m);
  plan.descriptor_cache_key = plan.heuristic_key;
  return plan;
}

std::optional<SharedNvfp4ContiguousProfile> SelectSharedNvfp4ContiguousProfile(
    std::size_t activation_rows,
    std::size_t output_rows,
    std::size_t input_cols) {
  if (activation_rows == 0 || output_rows == 0 || input_cols == 0 || (input_cols % 128u) != 0) {
    return std::nullopt;
  }

  struct ProfileCandidate {
    std::size_t token_bucket_lower;
    std::size_t token_bucket_upper;
    const char* profile_name;
    SharedNvfp4ContiguousTileShape tile_shape;
    std::size_t threads_per_block;
    bool uses_programmatic_launch;
  };

  static constexpr ProfileCandidate kCandidates[] = {
      {1u, 8u, "sm120_shared_p5_bucket_1_8", SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
      {9u, 16u, "sm120_shared_p5_bucket_9_16", SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
      {17u, 32u, "sm120_shared_p5_bucket_17_32", SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
      {33u, 64u, "sm120_shared_p5_bucket_33_64", SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
      {65u, 128u, "sm120_shared_p5_bucket_65_128", SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
      {129u, 256u, "sm120_shared_p5_bucket_129_256", SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
      {257u, 512u, "sm120_shared_p5_bucket_257_512", SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
      {513u, std::numeric_limits<std::size_t>::max(), "sm120_shared_p5_bucket_513_inf",
       SharedNvfp4ContiguousTileShape::k128x128x64, 128u, true},
  };

  for (const auto& candidate : kCandidates) {
    if (activation_rows >= candidate.token_bucket_lower &&
        activation_rows <= candidate.token_bucket_upper) {
      SharedNvfp4ContiguousProfile profile;
      profile.profile_name = candidate.profile_name;
      profile.tile_shape = candidate.tile_shape;
      profile.token_bucket_lower = candidate.token_bucket_lower;
      profile.token_bucket_upper = candidate.token_bucket_upper;
      profile.threads_per_block = candidate.threads_per_block;
      profile.uses_programmatic_launch = candidate.uses_programmatic_launch;
      return profile;
    }
  }

  return std::nullopt;
}

std::optional<GemmLaunchPlan> BuildSharedNvfp4ContiguousLaunchPlan(
    const SharedNvfp4ContiguousPlanRequest& request) {
  if (request.activation_rows == 0 ||
      request.output_rows == 0 ||
      request.input_cols == 0 ||
      !request.packed_bytes.valid() ||
      !request.block_scales_bytes.valid() ||
      !request.tensor_scale_bytes.valid()) {
    return std::nullopt;
  }

  const auto profile =
      SelectSharedNvfp4ContiguousProfile(
          request.activation_rows,
          request.output_rows,
          request.input_cols);
  if (!profile.has_value()) {
    return std::nullopt;
  }

  GemmLaunchPlan plan;
  plan.tensor_name =
      request.tensor_name.empty() ? "shared_contiguous_nvfp4" : request.tensor_name;
  plan.kernel_family = GemmKernelFamily::kSm120ContiguousSharedNvfp4;
  plan.m = request.activation_rows;
  plan.n = request.output_rows;
  plan.k = request.input_cols;
  plan.uses_block_scales = true;
  plan.packed_bytes = request.packed_bytes;
  plan.block_scales_bytes = request.block_scales_bytes;
  plan.tensor_scale_bytes = request.tensor_scale_bytes;
  plan.profile_name = profile->profile_name;
  plan.token_bucket_lower = profile->token_bucket_lower;
  plan.token_bucket_upper = profile->token_bucket_upper;
  plan.threads_per_block = profile->threads_per_block;
  plan.uses_programmatic_launch = profile->uses_programmatic_launch;
  TileShapeDims(
      profile->tile_shape,
      &plan.tile_m,
      &plan.tile_n,
      &plan.tile_k);
  if (plan.tile_m == 0 || plan.tile_n == 0 || plan.tile_k == 0 || (plan.k % plan.tile_k) != 0) {
    return std::nullopt;
  }
  plan.cta_m_count = CeilDiv(plan.m, plan.tile_m);
  plan.cta_n_count = CeilDiv(plan.n, plan.tile_n);
  if (plan.cta_m_count == 0 || plan.cta_n_count == 0) {
    return std::nullopt;
  }
  plan.heuristic_key = SharedContiguousHeuristicKey(plan, false);
  plan.descriptor_cache_key = SharedContiguousHeuristicKey(plan, true);
  return plan;
}

bool GemmHeuristicCache::Store(const GemmLaunchPlan& launch_plan, std::uint64_t algorithm_id) {
  if (launch_plan.heuristic_key.empty()) {
    return false;
  }
  algorithm_ids_by_key_[launch_plan.heuristic_key] = algorithm_id;
  return true;
}

std::optional<std::uint64_t> GemmHeuristicCache::Lookup(const GemmLaunchPlan& launch_plan) const {
  auto it = algorithm_ids_by_key_.find(launch_plan.heuristic_key);
  if (it == algorithm_ids_by_key_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t GemmHeuristicCache::size() const {
  return algorithm_ids_by_key_.size();
}

}  // namespace nemotron
