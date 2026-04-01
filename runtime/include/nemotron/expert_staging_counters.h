#pragma once

#include <atomic>
#include <cstdint>
#include <iosfwd>

namespace nemotron {

struct ExpertStagingCounters {
  std::atomic<std::uint64_t> total_bytes_uploaded{0};
  std::atomic<std::uint64_t> total_experts_staged{0};
  std::atomic<std::uint64_t> total_staging_calls{0};
  std::atomic<std::uint64_t> staging_elapsed_us{0};
};

ExpertStagingCounters& GetExpertStagingCounters();
void ResetExpertStagingCounters();
void PrintExpertStagingCounterSummary(std::ostream& stream);

}  // namespace nemotron
