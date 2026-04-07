#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {

constexpr float kFp4MaxFinite = 6.0f;
constexpr float kMinScale = 1.0f / 1024.0f;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool expect_near(float actual, float expected, float tolerance, const std::string& message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " (expected " << expected << ", got " << actual
              << ", tol " << tolerance << ")\n";
    return false;
  }
  return true;
}

float ClampScale(float value) {
  if (!std::isfinite(value) || value < kMinScale) {
    return kMinScale;
  }
  return value;
}

struct PackScales {
  float tensor_scale;
  float stored_block_scale;
  float effective_dequant_scale;
};

PackScales ComputeRuntimePackScales(
    float checkpoint_input_scale,
    float block_max_abs,
    bool tensor_scale_is_inverse) {
  const float tensor_scale_candidate =
      tensor_scale_is_inverse ? (1.0f / checkpoint_input_scale) : checkpoint_input_scale;
  const float tensor_scale =
      (std::isfinite(tensor_scale_candidate) && tensor_scale_candidate > 0.0f)
          ? tensor_scale_candidate
          : kMinScale;
  const float stored_block_scale =
      ClampScale(block_max_abs / (kFp4MaxFinite * tensor_scale));
  return {
      tensor_scale,
      stored_block_scale,
      tensor_scale * stored_block_scale,
  };
}

struct VllmReferenceScales {
  float activation_global_scale_inverse;
  float stored_block_scale;
  float effective_dequant_scale;
  float fused_alpha;
};

VllmReferenceScales ComputeVllmReferenceScales(
    float checkpoint_input_scale,
    float block_max_abs,
    float weight_scale_2) {
  const float activation_global_scale_inverse = 1.0f / checkpoint_input_scale;
  const float stored_block_scale =
      activation_global_scale_inverse * (block_max_abs / kFp4MaxFinite);
  return {
      activation_global_scale_inverse,
      stored_block_scale,
      stored_block_scale / activation_global_scale_inverse,
      checkpoint_input_scale * weight_scale_2,
  };
}

bool test_raw_input_scale_matches_vllm_stored_blockscale() {
  const float checkpoint_input_scale = 0.002f;
  const float block_max_abs = 0.1f;
  const auto runtime = ComputeRuntimePackScales(
      checkpoint_input_scale, block_max_abs, /*tensor_scale_is_inverse=*/false);
  const auto vllm =
      ComputeVllmReferenceScales(checkpoint_input_scale, block_max_abs, /*weight_scale_2=*/3.0f);
  return expect_near(
      runtime.stored_block_scale,
      vllm.stored_block_scale,
      1.0e-6f,
      "runtime stored blockscale should match vLLM when tensor_scale carries raw input_scale");
}

bool test_inverse_mapping_flips_stored_blockscale_direction() {
  const float checkpoint_input_scale = 0.002f;
  const float block_max_abs = 0.1f;
  const auto runtime = ComputeRuntimePackScales(
      checkpoint_input_scale, block_max_abs, /*tensor_scale_is_inverse=*/true);
  const auto vllm =
      ComputeVllmReferenceScales(checkpoint_input_scale, block_max_abs, /*weight_scale_2=*/3.0f);
  return expect(
             runtime.stored_block_scale < 0.001f,
             "inverse tensor_scale mapping should collapse stored blockscale into the tiny-scale regime") &&
         expect(
             std::fabs(runtime.stored_block_scale - vllm.stored_block_scale) > 1.0f,
             "inverse tensor_scale mapping should strongly disagree with vLLM stored blockscale");
}

bool test_inverse_mapping_breaks_effective_dequant_scale_once_min_scale_applies() {
  const float checkpoint_input_scale = 0.002f;
  const float block_max_abs = 0.1f;
  const auto runtime_raw = ComputeRuntimePackScales(
      checkpoint_input_scale, block_max_abs, /*tensor_scale_is_inverse=*/false);
  const auto runtime_inverse = ComputeRuntimePackScales(
      checkpoint_input_scale, block_max_abs, /*tensor_scale_is_inverse=*/true);
  const auto vllm =
      ComputeVllmReferenceScales(checkpoint_input_scale, block_max_abs, /*weight_scale_2=*/3.0f);
  return expect_near(
             runtime_raw.effective_dequant_scale,
             vllm.effective_dequant_scale,
             1.0e-6f,
             "raw tensor_scale mapping should preserve effective dequant scale") &&
         expect(
             runtime_inverse.effective_dequant_scale > (10.0f * vllm.effective_dequant_scale),
             "inverse tensor_scale mapping should blow up effective dequant scale once kMinScale clamps the blockscale") &&
         expect(
             std::fabs(runtime_raw.stored_block_scale - runtime_inverse.stored_block_scale) > 1.0f,
             "stored FP8 blockscale should still differ sharply between the two mappings");
}

bool test_raw_input_scale_preserves_fused_alpha_semantics() {
  const float checkpoint_input_scale = 0.002f;
  const float weight_scale_2 = 3.0f;
  const auto runtime_raw = ComputeRuntimePackScales(
      checkpoint_input_scale, /*block_max_abs=*/0.1f, /*tensor_scale_is_inverse=*/false);
  const auto runtime_inverse = ComputeRuntimePackScales(
      checkpoint_input_scale, /*block_max_abs=*/0.1f, /*tensor_scale_is_inverse=*/true);
  const auto vllm =
      ComputeVllmReferenceScales(checkpoint_input_scale, /*block_max_abs=*/0.1f, weight_scale_2);
  const float runtime_raw_fused_alpha = runtime_raw.tensor_scale * weight_scale_2;
  const float runtime_inverse_fused_alpha = runtime_inverse.tensor_scale * weight_scale_2;
  return expect_near(
             runtime_raw_fused_alpha,
             vllm.fused_alpha,
             1.0e-9f,
             "raw tensor_scale mapping should preserve vLLM fused alpha semantics") &&
         expect(
             std::fabs(runtime_inverse_fused_alpha - vllm.fused_alpha) > 1.0f,
             "inverse tensor_scale mapping should break fused alpha semantics");
}

bool test_raw_input_scale_preserves_tiny_checkpoint_scales() {
  const float checkpoint_input_scale = 1.0f / 1880.0f;
  const float block_max_abs = 0.1f;
  const auto runtime_raw = ComputeRuntimePackScales(
      checkpoint_input_scale, block_max_abs, /*tensor_scale_is_inverse=*/false);
  const auto vllm =
      ComputeVllmReferenceScales(checkpoint_input_scale, block_max_abs, /*weight_scale_2=*/3.0f);
  return expect_near(
             runtime_raw.tensor_scale,
             checkpoint_input_scale,
             1.0e-9f,
             "raw tensor_scale path should preserve tiny positive checkpoint scales") &&
         expect_near(
             runtime_raw.stored_block_scale,
             vllm.stored_block_scale,
             1.0e-4f,
             "raw tensor_scale path should preserve tiny-scale stored blockscale semantics");
}

}  // namespace

int main() {
  const bool ok =
      test_raw_input_scale_matches_vllm_stored_blockscale() &&
      test_inverse_mapping_flips_stored_blockscale_direction() &&
      test_inverse_mapping_breaks_effective_dequant_scale_once_min_scale_applies() &&
      test_raw_input_scale_preserves_fused_alpha_semantics() &&
      test_raw_input_scale_preserves_tiny_checkpoint_scales();

  if (!ok) {
    return 1;
  }
  std::cout << "nvfp4_pack_scale_semantics_test: PASS\n";
  return 0;
}
