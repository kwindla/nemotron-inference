#include "attention_harness_common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {

using nemotron::attention_harness::CaseSpec;
using nemotron::attention_harness::MaxAbsDiff;
using nemotron::attention_harness::Runner;

constexpr float kMaxAbsTolerance = 2.0e-2f;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool VerifyCase(const CaseSpec& spec) {
  auto runner = Runner::Create(spec);
  if (!expect(runner != nullptr, "runner should initialize for " + spec.name)) {
    return false;
  }
  if (!expect(runner->RunOnce(false), "runner should execute for " + spec.name)) {
    return false;
  }

  std::vector<float> actual;
  if (!expect(
          runner->DownloadOutputRowMajor(&actual),
          "runner should download output for " + spec.name)) {
    return false;
  }

  const std::vector<float> reference = runner->ComputeReferenceRowMajor();
  const float max_abs_diff = MaxAbsDiff(actual, reference);
  if (!expect(
          actual.size() == reference.size(),
          "reference shape should match actual for " + spec.name)) {
    return false;
  }
  if (!expect(
          max_abs_diff <= kMaxAbsTolerance,
          spec.name + " should match CPU reference; max_abs_diff=" +
              std::to_string(max_abs_diff))) {
    return false;
  }
  return true;
}

bool RunAllCases() {
  const std::vector<CaseSpec> cases = {
      {"prefill_q16", "cold_prefill", 0, 16, 16, true},
      {"prefill_q32", "cold_prefill", 0, 32, 32, true},
      {"extend_prefix32_q8", "restored_extend", 32, 8, 40, true},
      {"extend_prefix48_q16", "restored_extend", 48, 16, 64, true},
  };

  for (const auto& spec : cases) {
    if (!VerifyCase(spec)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!nemotron::attention_harness::HasCudaDevice()) {
    std::cout << "nano_paged_attention_test: SKIP (no CUDA device available)\n";
    return 0;
  }
  if (!RunAllCases()) {
    return 1;
  }
  std::cout << "nano_paged_attention_test: PASS\n";
  return 0;
}
