#include "nemotron/linear_op_trace.h"

#include <cstdlib>
#include <cstring>

#include <cstdint>
#include <map>
#include <ostream>
#include <utility>
#include <vector>

namespace nemotron {
namespace {

constexpr const char* kLinearOpTraceEnvVar = "NEMOTRON_FORWARD_LINEAR_TRACE";

struct LinearOpTraceSummaryKey {
  std::string tensor_name;
  GemmKernelFamily kernel_family = GemmKernelFamily::kDenseRowMajor;

  bool operator<(const LinearOpTraceSummaryKey& other) const {
    if (tensor_name != other.tensor_name) {
      return tensor_name < other.tensor_name;
    }
    return static_cast<int>(kernel_family) < static_cast<int>(other.kernel_family);
  }
};

struct LinearOpTraceSummaryCounts {
  std::uint64_t fastpath = 0;
  std::uint64_t reference = 0;
};

const char* SummaryKernelFamilyName(GemmKernelFamily family) {
  switch (family) {
    case GemmKernelFamily::kDenseRowMajor:
      return "dense";
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      return "nvfp4";
    case GemmKernelFamily::kSm120ContiguousSharedNvfp4:
      return "shared_nvfp4";
  }
  return "unknown";
}

}  // namespace

LinearOpTrace& GetLinearOpTrace() {
  static LinearOpTrace trace;
  return trace;
}

void ResetLinearOpTrace() {
  auto& trace = GetLinearOpTrace();
  std::lock_guard<std::mutex> lock(trace.mutex);
  trace.entries.clear();
}

bool IsLinearOpTraceEnabled() {
  const char* value = std::getenv(kLinearOpTraceEnvVar);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

void PrintLinearOpTraceSummary(std::ostream& stream) {
  std::vector<LinearOpTraceEntry> entries;
  {
    auto& trace = GetLinearOpTrace();
    std::lock_guard<std::mutex> lock(trace.mutex);
    entries = trace.entries;
  }

  std::map<LinearOpTraceSummaryKey, LinearOpTraceSummaryCounts> summaries;
  for (const auto& entry : entries) {
    LinearOpTraceSummaryKey key;
    key.tensor_name = entry.tensor_name;
    key.kernel_family = entry.kernel_family;
    auto& counts = summaries[key];
    if (entry.path_taken == LinearOpPath::kFastpath) {
      ++counts.fastpath;
    } else {
      ++counts.reference;
    }
  }

  for (const auto& summary : summaries) {
    stream << "linear_op_trace: tensor=" << summary.first.tensor_name
           << " family=" << SummaryKernelFamilyName(summary.first.kernel_family)
           << " fastpath=" << summary.second.fastpath
           << " reference=" << summary.second.reference
           << "\n";
  }
}

}  // namespace nemotron
