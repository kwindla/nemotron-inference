#include "nemotron/gemm_execution.h"

#include <string>

namespace nemotron {
namespace {

std::uint64_t Fnv1a64(std::string_view text) {
  constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
  constexpr std::uint64_t kFnvPrime = 1099511628211ull;
  std::uint64_t hash = kFnvOffsetBasis;
  for (unsigned char ch : text) {
    hash ^= ch;
    hash *= kFnvPrime;
  }
  return hash;
}

}  // namespace

const char* ToString(GemmBackendKind backend_kind) {
  switch (backend_kind) {
    case GemmBackendKind::kCublasLtDense:
      return "cublaslt_dense";
    case GemmBackendKind::kCublasLtNvfp4BlockScaled:
      return "cublaslt_nvfp4_block_scaled";
    case GemmBackendKind::kSm120ContiguousSharedNvfp4:
      return "sm120_contiguous_shared_nvfp4";
  }
  return "unknown";
}

std::optional<PreparedGemmExecution> PrepareGemmExecution(
    const GemmLaunchPlan& launch_plan,
    GemmHeuristicCache* heuristic_cache) {
  if (launch_plan.m == 0 || launch_plan.n == 0 || launch_plan.k == 0 ||
      !launch_plan.packed_bytes.valid() || launch_plan.heuristic_key.empty()) {
    return std::nullopt;
  }

  PreparedGemmExecution execution;
  execution.launch_plan = launch_plan;

  switch (launch_plan.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      execution.backend_kind = GemmBackendKind::kCublasLtDense;
      execution.workspace_bytes = 0;
      break;
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      if (!launch_plan.block_scales_bytes.valid() || !launch_plan.tensor_scale_bytes.valid()) {
        return std::nullopt;
      }
      execution.backend_kind = GemmBackendKind::kCublasLtNvfp4BlockScaled;
      execution.workspace_bytes = 0;
      execution.requires_block_scales = true;
      execution.requires_tensor_scale = true;
      break;
    case GemmKernelFamily::kSm120ContiguousSharedNvfp4:
      if (!launch_plan.block_scales_bytes.valid() ||
          !launch_plan.tensor_scale_bytes.valid() ||
          launch_plan.tile_m == 0 ||
          launch_plan.tile_n == 0 ||
          launch_plan.tile_k == 0 ||
          launch_plan.cta_m_count == 0 ||
          launch_plan.cta_n_count == 0 ||
          launch_plan.profile_name.empty()) {
        return std::nullopt;
      }
      execution.backend_kind = GemmBackendKind::kSm120ContiguousSharedNvfp4;
      execution.workspace_bytes = 0;
      execution.requires_block_scales = true;
      execution.requires_tensor_scale = true;
      break;
  }

  if (heuristic_cache != nullptr) {
    const std::optional<std::uint64_t> cached_algorithm_id =
        heuristic_cache->Lookup(launch_plan);
    if (cached_algorithm_id.has_value()) {
      execution.algorithm_id = *cached_algorithm_id;
      execution.algorithm_from_cache = true;
      return execution;
    }
  }

  execution.algorithm_id = Fnv1a64(launch_plan.heuristic_key);
  execution.algorithm_from_cache = false;
  if (heuristic_cache != nullptr) {
    heuristic_cache->Store(launch_plan, execution.algorithm_id);
  }
  return execution;
}

}  // namespace nemotron
