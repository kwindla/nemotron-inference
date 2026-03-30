#include "nemotron/memory_budget.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace nemotron {
namespace {

constexpr std::size_t KiB(std::size_t value) {
  return value * 1024ull;
}

std::size_t SaturatingAdd(std::size_t lhs, std::size_t rhs) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    return std::numeric_limits<std::size_t>::max();
  }
  return lhs + rhs;
}

bool ParseMeminfoLine(
    const std::string& line,
    const std::string& key,
    std::size_t* value_bytes) {
  if (value_bytes == nullptr) {
    return false;
  }
  if (line.rfind(key, 0) != 0) {
    return false;
  }

  std::istringstream input(line.substr(key.size()));
  std::size_t value_kib = 0;
  std::string unit;
  if (!(input >> value_kib >> unit) || unit != "kB") {
    return false;
  }
  *value_bytes = KiB(value_kib);
  return true;
}

std::size_t ResolvePlanningTotalMemoryBytes(const ServiceMemoryTarget& target, bool* used_host_snapshot) {
  if (used_host_snapshot != nullptr) {
    *used_host_snapshot = false;
  }

  std::size_t planning_total = target.total_memory_bytes;
  if (!target.prefer_host_memory_snapshot || !target.host_memory_snapshot.valid) {
    return planning_total;
  }

  const std::size_t host_budget = SaturatingAdd(
      target.host_memory_snapshot.mem_available_bytes,
      target.host_memory_snapshot.swap_free_bytes);
  if (host_budget == 0) {
    return planning_total;
  }

  if (used_host_snapshot != nullptr) {
    *used_host_snapshot = true;
  }

  if (planning_total == 0) {
    return host_budget;
  }
  return std::min(planning_total, host_budget);
}

}  // namespace

HostMemorySnapshot ReadHostMemorySnapshotFromProcMeminfo(
    const std::filesystem::path& proc_meminfo_path) {
  std::ifstream input(proc_meminfo_path);
  if (!input) {
    return HostMemorySnapshot{};
  }

  HostMemorySnapshot snapshot;
  std::string line;
  while (std::getline(input, line)) {
    ParseMeminfoLine(line, "MemAvailable:", &snapshot.mem_available_bytes);
    ParseMeminfoLine(line, "SwapFree:", &snapshot.swap_free_bytes);
  }
  snapshot.valid = snapshot.mem_available_bytes != 0 || snapshot.swap_free_bytes != 0;
  return snapshot;
}

std::size_t BytesForExactPrefixNode(
    const RuntimeMemoryProfile& profile,
    std::size_t prefix_tokens) {
  return profile.kv_bytes_per_token * prefix_tokens +
         profile.current_mamba_state_bytes +
         profile.reusable_node_metadata_bytes;
}

MemoryBudgetSummary BuildMemoryBudgetSummary(
    const RuntimeMemoryProfile& profile,
    const ServiceMemoryTarget& target) {
  MemoryBudgetSummary summary;
  summary.host_mem_available_bytes = target.host_memory_snapshot.mem_available_bytes;
  summary.host_swap_free_bytes = target.host_memory_snapshot.swap_free_bytes;
  summary.planning_total_memory_bytes =
      ResolvePlanningTotalMemoryBytes(target, &summary.used_host_memory_snapshot);
  summary.active_kv_bytes =
      profile.kv_bytes_per_token * target.target_context_tokens * target.target_active_requests;
  summary.active_mamba_bytes =
      profile.current_mamba_state_bytes * target.target_active_requests;
  summary.reserved_non_cache_bytes =
      target.weights_bytes +
      target.workspace_bytes +
      target.graph_bytes +
      target.safety_headroom_bytes +
      summary.active_kv_bytes +
      summary.active_mamba_bytes;
  summary.fits = summary.reserved_non_cache_bytes <= summary.planning_total_memory_bytes;
  summary.shared_cache_budget_bytes =
      summary.fits ? summary.planning_total_memory_bytes - summary.reserved_non_cache_bytes : 0;
  summary.bytes_per_full_context_node =
      BytesForExactPrefixNode(profile, target.target_context_tokens);
  if (summary.bytes_per_full_context_node != 0) {
    summary.max_full_context_nodes =
        summary.shared_cache_budget_bytes / summary.bytes_per_full_context_node;
  }
  return summary;
}

}  // namespace nemotron
