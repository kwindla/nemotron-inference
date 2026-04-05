#include "nemotron/gemm_catalog.h"
#include "nemotron/nvfp4_scale_helpers.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

namespace {

using nemotron::GemmDescriptor;
using nemotron::GemmKernelFamily;
using nemotron::ResolveRoutedNvfp4RuntimeTensorScale;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool expect_value(
    const std::optional<float>& actual,
    float expected,
    const std::string& message) {
  if (!actual.has_value()) {
    std::cerr << "FAIL: " << message << " (missing value)\n";
    return false;
  }
  if (*actual != expected) {
    std::cerr << "FAIL: " << message << " (expected " << expected << ", got " << *actual << ")\n";
    return false;
  }
  return true;
}

GemmDescriptor MakeDescriptor(
    GemmKernelFamily kernel_family,
    const float* tensor_scale_value) {
  GemmDescriptor descriptor;
  descriptor.kernel_family = kernel_family;
  descriptor.tensor_scale_data =
      tensor_scale_value == nullptr
          ? nullptr
          : reinterpret_cast<const std::uint8_t*>(tensor_scale_value);
  descriptor.tensor_scale_nbytes = tensor_scale_value == nullptr ? 0 : sizeof(float);
  return descriptor;
}

bool test_nvfp4_helper_returns_raw_weight_scale_for_routed_nvfp4_family() {
  const float weight_scale_2 = 3.0f;
  const auto descriptor = MakeDescriptor(
      GemmKernelFamily::kCublasLtNvfp4BlockScaled,
      &weight_scale_2);
  return expect_value(
      ResolveRoutedNvfp4RuntimeTensorScale(descriptor, 2.0f),
      3.0f,
      "routed NVFP4 helper should return raw weight_scale_2");
}

bool test_nvfp4_helper_returns_raw_weight_scale_for_non_nvfp4_family() {
  const float weight_scale_2 = 3.0f;
  const auto descriptor = MakeDescriptor(
      GemmKernelFamily::kDenseRowMajor,
      &weight_scale_2);
  const bool ok_with_value = expect_value(
      ResolveRoutedNvfp4RuntimeTensorScale(descriptor, 17.0f),
      weight_scale_2,
      "non-NVFP4 helper should return raw weight_scale_2 when input_scale is present");
  const bool ok_without_value = expect_value(
      ResolveRoutedNvfp4RuntimeTensorScale(descriptor, std::nullopt),
      weight_scale_2,
      "non-NVFP4 helper should return raw weight_scale_2 when input_scale is absent");
  return ok_with_value && ok_without_value;
}

bool test_nvfp4_helper_rejects_missing_input_scale_for_nvfp4_family() {
  const float weight_scale_2 = 3.0f;
  const auto descriptor = MakeDescriptor(
      GemmKernelFamily::kCublasLtNvfp4BlockScaled,
      &weight_scale_2);
  return expect(
      !ResolveRoutedNvfp4RuntimeTensorScale(descriptor, std::nullopt).has_value(),
      "routed NVFP4 helper should reject missing input_scale");
}

bool test_nvfp4_helper_rejects_missing_tensor_scale_data() {
  const auto descriptor = MakeDescriptor(
      GemmKernelFamily::kCublasLtNvfp4BlockScaled,
      nullptr);
  return expect(
      !ResolveRoutedNvfp4RuntimeTensorScale(descriptor, 2.0f).has_value(),
      "helper should reject missing tensor_scale_data");
}

bool test_nvfp4_helper_rejects_invalid_raw_weight_scale() {
  const auto nvfp4 = GemmKernelFamily::kCublasLtNvfp4BlockScaled;

  const float valid_weight_scale = 3.0f;
  const auto valid_descriptor = MakeDescriptor(nvfp4, &valid_weight_scale);
  const bool accepts_nan_input = expect_value(
      ResolveRoutedNvfp4RuntimeTensorScale(
          valid_descriptor,
          std::numeric_limits<float>::quiet_NaN()),
      valid_weight_scale,
      "routed NVFP4 helper should ignore invalid input_scale values once presence is satisfied");

  const float nan_weight_scale = std::numeric_limits<float>::quiet_NaN();
  const auto nan_descriptor = MakeDescriptor(nvfp4, &nan_weight_scale);
  const bool rejects_nan_weight = expect(
      !ResolveRoutedNvfp4RuntimeTensorScale(nan_descriptor, 2.0f).has_value(),
      "routed NVFP4 helper should reject NaN weight_scale_2");

  const float negative_weight_scale = -3.0f;
  const auto negative_descriptor = MakeDescriptor(nvfp4, &negative_weight_scale);
  const bool rejects_negative = expect(
      !ResolveRoutedNvfp4RuntimeTensorScale(negative_descriptor, 2.0f).has_value(),
      "routed NVFP4 helper should reject negative effective scales");

  const float zero_weight_scale = 0.0f;
  const auto zero_descriptor = MakeDescriptor(nvfp4, &zero_weight_scale);
  const bool rejects_zero = expect(
      !ResolveRoutedNvfp4RuntimeTensorScale(zero_descriptor, 2.0f).has_value(),
      "routed NVFP4 helper should reject zero effective scales");

  return accepts_nan_input && rejects_nan_weight && rejects_negative && rejects_zero;
}

bool test_nvfp4_helper_accepts_very_small_positive_scales() {
  const float input_scale = 1.0e-20f;
  const float weight_scale_2 = 2.0e-10f;
  const auto descriptor = MakeDescriptor(
      GemmKernelFamily::kCublasLtNvfp4BlockScaled,
      &weight_scale_2);
  const auto actual = ResolveRoutedNvfp4RuntimeTensorScale(descriptor, input_scale);
  return expect(input_scale > 0.0f, "tiny positive input scale should remain positive") &&
         expect_value(
             actual,
             weight_scale_2,
             "routed NVFP4 helper should preserve very small positive raw weight_scale_2 values");
}

}  // namespace

int main() {
  const bool ok =
      test_nvfp4_helper_returns_raw_weight_scale_for_routed_nvfp4_family() &&
      test_nvfp4_helper_returns_raw_weight_scale_for_non_nvfp4_family() &&
      test_nvfp4_helper_rejects_missing_input_scale_for_nvfp4_family() &&
      test_nvfp4_helper_rejects_missing_tensor_scale_data() &&
      test_nvfp4_helper_rejects_invalid_raw_weight_scale() &&
      test_nvfp4_helper_accepts_very_small_positive_scales();

  if (!ok) {
    return 1;
  }
  std::cout << "nvfp4_scale_helper_test: PASS\n";
  return 0;
}
