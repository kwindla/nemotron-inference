#pragma once

#include <iosfwd>
#include <mutex>
#include <string>
#include <vector>

#include "nemotron/gemm_catalog.h"

namespace nemotron {

enum class LinearOpPath {
  kFastpath,
  kReference,
};

struct LinearOpTraceEntry {
  std::string tensor_name;
  GemmKernelFamily kernel_family = GemmKernelFamily::kDenseRowMajor;
  LinearOpPath path_taken = LinearOpPath::kReference;
  bool plan_build_ok = false;
  bool execute_ok = false;
};

struct LinearOpTrace {
  std::mutex mutex;
  std::vector<LinearOpTraceEntry> entries;
};

LinearOpTrace& GetLinearOpTrace();
void ResetLinearOpTrace();
bool IsLinearOpTraceEnabled();
void PrintLinearOpTraceSummary(std::ostream& stream);

}  // namespace nemotron
