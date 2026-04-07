#pragma once

#include <cstddef>
#include <filesystem>

namespace nemotron {

struct RuntimeMemoryProfile {
  std::size_t kv_bytes_per_token = 0;
  std::size_t current_mamba_state_bytes = 0;
  std::size_t reusable_node_metadata_bytes = 0;
};

struct HostMemorySnapshot {
  std::size_t mem_available_bytes = 0;
  std::size_t swap_free_bytes = 0;
  bool valid = false;
};

struct ServiceMemoryTarget {
  std::size_t total_memory_bytes = 0;
  std::size_t weights_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t graph_bytes = 0;
  std::size_t safety_headroom_bytes = 0;
  std::size_t target_active_requests = 0;
  std::size_t target_context_tokens = 0;
  HostMemorySnapshot host_memory_snapshot;
  bool prefer_host_memory_snapshot = false;
};

struct MemoryBudgetSummary {
  std::size_t planning_total_memory_bytes = 0;
  std::size_t host_mem_available_bytes = 0;
  std::size_t host_swap_free_bytes = 0;
  std::size_t active_kv_bytes = 0;
  std::size_t active_mamba_bytes = 0;
  std::size_t reserved_non_cache_bytes = 0;
  std::size_t shared_cache_budget_bytes = 0;
  std::size_t bytes_per_full_context_node = 0;
  std::size_t max_full_context_nodes = 0;
  bool used_host_memory_snapshot = false;
  bool fits = false;
};

HostMemorySnapshot ReadHostMemorySnapshotFromProcMeminfo(
    const std::filesystem::path& proc_meminfo_path);

std::size_t BytesForExactPrefixNode(
    const RuntimeMemoryProfile& profile,
    std::size_t prefix_tokens);

MemoryBudgetSummary BuildMemoryBudgetSummary(
    const RuntimeMemoryProfile& profile,
    const ServiceMemoryTarget& target);

}  // namespace nemotron
