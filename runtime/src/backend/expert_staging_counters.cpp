#include "nemotron/expert_staging_counters.h"

#include <ostream>

namespace nemotron {
namespace {

std::uint64_t LoadCounter(const std::atomic<std::uint64_t>& counter) {
  return counter.load(std::memory_order_relaxed);
}

void ResetCounter(std::atomic<std::uint64_t>* counter) {
  counter->store(0, std::memory_order_relaxed);
}

}  // namespace

ExpertStagingCounters& GetExpertStagingCounters() {
  static ExpertStagingCounters counters;
  return counters;
}

void ResetExpertStagingCounters() {
  auto& counters = GetExpertStagingCounters();
  ResetCounter(&counters.total_bytes_uploaded);
  ResetCounter(&counters.total_experts_staged);
  ResetCounter(&counters.total_staging_calls);
  ResetCounter(&counters.monolithic_layers);
  ResetCounter(&counters.staging_elapsed_us);
}

void PrintExpertStagingCounterSummary(std::ostream& stream) {
  const auto& counters = GetExpertStagingCounters();
  stream << "expert_staging_counters"
         << " total_bytes_uploaded="
         << LoadCounter(counters.total_bytes_uploaded)
         << " total_experts_staged="
         << LoadCounter(counters.total_experts_staged)
         << " total_staging_calls="
         << LoadCounter(counters.total_staging_calls)
         << " monolithic_layers="
         << LoadCounter(counters.monolithic_layers)
         << " staging_elapsed_us="
         << LoadCounter(counters.staging_elapsed_us)
         << "\n";
}

}  // namespace nemotron
