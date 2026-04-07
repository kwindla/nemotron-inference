#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "nemotron/gemm_planner.h"

namespace nemotron {

enum class GemmBackendKind {
  kCublasLtDense,
  kCublasLtNvfp4BlockScaled,
};

const char* ToString(GemmBackendKind backend_kind);

struct PreparedGemmExecution {
  GemmLaunchPlan launch_plan;
  GemmBackendKind backend_kind = GemmBackendKind::kCublasLtDense;
  std::uint64_t algorithm_id = 0;
  bool algorithm_from_cache = false;
  std::size_t workspace_bytes = 0;
  bool requires_block_scales = false;
  bool requires_tensor_scale = false;
};

std::optional<PreparedGemmExecution> PrepareGemmExecution(
    const GemmLaunchPlan& launch_plan,
    GemmHeuristicCache* heuristic_cache);

}  // namespace nemotron
