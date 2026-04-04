#pragma once

#include <cmath>
#include <cstring>
#include <optional>

#include "nemotron/gemm_catalog.h"

namespace nemotron {

inline std::optional<float> ResolveRoutedNvfp4RuntimeTensorScale(
    const GemmDescriptor& descriptor,
    const std::optional<float>& input_scale) {
  if (descriptor.tensor_scale_data == nullptr || descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }

  float weight_scale_2 = 0.0f;
  std::memcpy(&weight_scale_2, descriptor.tensor_scale_data, sizeof(weight_scale_2));

  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
    return weight_scale_2;
  }
  if (!input_scale.has_value()) {
    return std::nullopt;
  }

  const float effective_tensor_scale = (*input_scale) * weight_scale_2;
  if (!std::isfinite(effective_tensor_scale) || effective_tensor_scale <= 0.0f) {
    return std::nullopt;
  }
  return effective_tensor_scale;
}

}  // namespace nemotron
