#include "nemotron/linear_op_counters.h"

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

LinearOpCounters& GetLinearOpCounters() {
  static LinearOpCounters counters;
  return counters;
}

void ResetLinearOpCounters() {
  auto& counters = GetLinearOpCounters();
  ResetCounter(&counters.dense_fastpath_plan_success);
  ResetCounter(&counters.dense_fastpath_plan_fail);
  ResetCounter(&counters.dense_fastpath_execute);
  ResetCounter(&counters.dense_fastpath_execute_fail);
  ResetCounter(&counters.dense_reference_fallback);

  ResetCounter(&counters.nvfp4_fastpath_plan_success);
  ResetCounter(&counters.nvfp4_fastpath_plan_fail);
  ResetCounter(&counters.nvfp4_fastpath_execute);
  ResetCounter(&counters.nvfp4_fastpath_execute_fail);
  ResetCounter(&counters.nvfp4_reference_fallback);

  ResetCounter(&counters.scaled_fp8_fastpath_execute);
  ResetCounter(&counters.scaled_fp8_reference_fallback);
}

void PrintLinearOpCounterSummary(std::ostream& stream) {
  const auto& counters = GetLinearOpCounters();
  stream << "linear_op_counters"
         << " dense_fastpath_plan_success="
         << LoadCounter(counters.dense_fastpath_plan_success)
         << " dense_fastpath_plan_fail="
         << LoadCounter(counters.dense_fastpath_plan_fail)
         << " dense_fastpath_execute="
         << LoadCounter(counters.dense_fastpath_execute)
         << " dense_fastpath_execute_fail="
         << LoadCounter(counters.dense_fastpath_execute_fail)
         << " dense_reference_fallback="
         << LoadCounter(counters.dense_reference_fallback)
         << " nvfp4_fastpath_plan_success="
         << LoadCounter(counters.nvfp4_fastpath_plan_success)
         << " nvfp4_fastpath_plan_fail="
         << LoadCounter(counters.nvfp4_fastpath_plan_fail)
         << " nvfp4_fastpath_execute="
         << LoadCounter(counters.nvfp4_fastpath_execute)
         << " nvfp4_fastpath_execute_fail="
         << LoadCounter(counters.nvfp4_fastpath_execute_fail)
         << " nvfp4_reference_fallback="
         << LoadCounter(counters.nvfp4_reference_fallback)
         << " scaled_fp8_fastpath_execute="
         << LoadCounter(counters.scaled_fp8_fastpath_execute)
         << " scaled_fp8_reference_fallback="
         << LoadCounter(counters.scaled_fp8_reference_fallback)
         << "\n";
}

}  // namespace nemotron
