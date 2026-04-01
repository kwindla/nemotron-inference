#pragma once

#include <atomic>
#include <cstdint>
#include <iosfwd>

namespace nemotron {

struct LinearOpCounters {
  std::atomic<std::uint64_t> dense_fastpath_plan_success{0};
  std::atomic<std::uint64_t> dense_fastpath_plan_fail{0};
  std::atomic<std::uint64_t> dense_fastpath_execute{0};
  std::atomic<std::uint64_t> dense_fastpath_execute_fail{0};
  std::atomic<std::uint64_t> dense_reference_fallback{0};

  std::atomic<std::uint64_t> nvfp4_fastpath_plan_success{0};
  std::atomic<std::uint64_t> nvfp4_fastpath_plan_fail{0};
  std::atomic<std::uint64_t> nvfp4_fastpath_execute{0};
  std::atomic<std::uint64_t> nvfp4_fastpath_execute_fail{0};
  std::atomic<std::uint64_t> nvfp4_reference_fallback{0};

  std::atomic<std::uint64_t> scaled_fp8_fastpath_execute{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallback{0};
};

LinearOpCounters& GetLinearOpCounters();
void ResetLinearOpCounters();
void PrintLinearOpCounterSummary(std::ostream& stream);

}  // namespace nemotron
