#include "nemotron/memory_budget.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using nemotron::BuildMemoryBudgetSummary;
using nemotron::BytesForExactPrefixNode;
using nemotron::HostMemorySnapshot;
using nemotron::MemoryBudgetSummary;
using nemotron::ReadHostMemorySnapshotFromProcMeminfo;
using nemotron::RuntimeMemoryProfile;
using nemotron::ServiceMemoryTarget;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

class TempFile {
 public:
  TempFile() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_meminfo_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".txt");
  }

  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

RuntimeMemoryProfile nemotron_fp16_profile() {
  RuntimeMemoryProfile profile;
  profile.kv_bytes_per_token = 4096;
  profile.current_mamba_state_bytes = 87162880;
  profile.reusable_node_metadata_bytes = 4096;
  return profile;
}

bool test_exact_prefix_node_bytes() {
  const auto profile = nemotron_fp16_profile();
  const std::size_t node_bytes = BytesForExactPrefixNode(profile, 65536);
  return expect(node_bytes == 355602432, "64k exact-prefix node bytes should match preflight math");
}

bool test_active_state_math_for_service_target() {
  const auto profile = nemotron_fp16_profile();
  ServiceMemoryTarget target;
  target.target_active_requests = 8;
  target.target_context_tokens = 65536;

  const MemoryBudgetSummary summary = BuildMemoryBudgetSummary(profile, target);
  return expect(summary.active_kv_bytes == 2147483648ull, "8x64k active KV bytes should be 2 GiB") &&
         expect(summary.active_mamba_bytes == 697303040ull, "8 active Mamba states should match FP16 preflight math") &&
         expect(summary.bytes_per_full_context_node == 355602432ull, "full-context node bytes should match helper");
}

bool test_budget_summary_reports_shared_cache_capacity() {
  const auto profile = nemotron_fp16_profile();
  ServiceMemoryTarget target;
  target.total_memory_bytes = GiB(128);
  target.weights_bytes = GiB(100);
  target.workspace_bytes = GiB(8);
  target.graph_bytes = GiB(4);
  target.safety_headroom_bytes = GiB(4);
  target.target_active_requests = 8;
  target.target_context_tokens = 65536;

  const MemoryBudgetSummary summary = BuildMemoryBudgetSummary(profile, target);
  return expect(summary.fits, "example DGX Spark budget should fit the reserved working set") &&
         expect(summary.shared_cache_budget_bytes == 10040115200ull,
                "remaining shared-cache bytes should match the budget calculation") &&
         expect(summary.max_full_context_nodes == 28ull,
                "example budget should hold 28 full 64k exact-prefix nodes");
}

bool test_overcommitted_budget_has_no_shared_cache() {
  const auto profile = nemotron_fp16_profile();
  ServiceMemoryTarget target;
  target.total_memory_bytes = GiB(4);
  target.weights_bytes = GiB(3);
  target.workspace_bytes = GiB(1);
  target.graph_bytes = GiB(1);
  target.safety_headroom_bytes = GiB(1);
  target.target_active_requests = 4;
  target.target_context_tokens = 8192;

  const MemoryBudgetSummary summary = BuildMemoryBudgetSummary(profile, target);
  return expect(!summary.fits, "overcommitted example should not fit") &&
         expect(summary.shared_cache_budget_bytes == 0, "overcommitted example should leave no shared cache budget");
}

bool test_reads_host_memory_snapshot_from_proc_meminfo() {
  TempFile file;
  std::ofstream output(file.path());
  output
      << "MemTotal:       131900028 kB\n"
      << "MemAvailable:    98304000 kB\n"
      << "SwapFree:        20971520 kB\n";
  output.close();

  const HostMemorySnapshot snapshot = ReadHostMemorySnapshotFromProcMeminfo(file.path());
  return expect(snapshot.valid, "snapshot should parse from test meminfo") &&
         expect(snapshot.mem_available_bytes == 100663296000ull,
                "MemAvailable should parse in bytes") &&
         expect(snapshot.swap_free_bytes == 21474836480ull,
                "SwapFree should parse in bytes");
}

bool test_budget_summary_prefers_host_memory_snapshot_when_requested() {
  const auto profile = nemotron_fp16_profile();
  ServiceMemoryTarget target;
  target.total_memory_bytes = GiB(128);
  target.weights_bytes = GiB(100);
  target.workspace_bytes = GiB(8);
  target.graph_bytes = GiB(4);
  target.safety_headroom_bytes = GiB(4);
  target.target_active_requests = 8;
  target.target_context_tokens = 65536;
  target.prefer_host_memory_snapshot = true;
  target.host_memory_snapshot = HostMemorySnapshot{
      /*mem_available_bytes=*/GiB(20),
      /*swap_free_bytes=*/GiB(10),
      /*valid=*/true,
  };

  const MemoryBudgetSummary summary = BuildMemoryBudgetSummary(profile, target);
  return expect(summary.used_host_memory_snapshot, "summary should report using the host snapshot") &&
         expect(summary.planning_total_memory_bytes == GiB(30),
                "planning total should clamp to MemAvailable + SwapFree") &&
         expect(!summary.fits, "host-memory-constrained budget should not fit this service target") &&
         expect(summary.shared_cache_budget_bytes == 0, "clamped host budget should leave no shared cache");
}

}  // namespace

int main() {
  const bool ok =
      test_exact_prefix_node_bytes() &&
      test_active_state_math_for_service_target() &&
      test_budget_summary_reports_shared_cache_capacity() &&
      test_overcommitted_budget_has_no_shared_cache() &&
      test_reads_host_memory_snapshot_from_proc_meminfo() &&
      test_budget_summary_prefers_host_memory_snapshot_when_requested();

  if (!ok) {
    return 1;
  }
  std::cout << "memory_budget_test: PASS\n";
  return 0;
}
