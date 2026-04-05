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

  // The kernel alpha path consumes: activation_tensor_scale * weight_scale_2
  // where activation_tensor_scale comes from NVFP4 activation packing
  // (for routed-up, one packed activation row per selected expert). The
  // checkpoint input_scale is NOT fused into this value here.
  // See vLLM reference: g1_alphas = a13_scale * w13_scale_2
  // TARGET CONTRACT (Steps 2-3): this helper continues to expose raw
  // weight_scale_2 only. Routed checkpoint input_scale must instead be
  // consumed during activation packing, so the packer-emitted activation
  // tensor scale is what participates in:
  //   activation_tensor_scale * weight_scale_2
  // input_scale must not be fused into the value returned here.
  (void)input_scale;
  if (!std::isfinite(weight_scale_2) || weight_scale_2 <= 0.0f) {
    return std::nullopt;
  }
  return weight_scale_2;
}

}  // namespace nemotron
