#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/dense_weight.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/linear_op.h"
#include "nemotron/linear_reference_kernels.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildGemmLaunchPlan;
using nemotron::CublasLtHandle;
using nemotron::DeviceDenseWeightFp32;
using nemotron::DeviceNvfp4Weight;
using nemotron::DeviceTensorFp32;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::HostNvfp4Matrix;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::PrepareGemmExecution;
using nemotron::RunNvfp4RowMajorReferenceToDevice;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

GemmDescriptor make_dense_descriptor(
    std::size_t output_rows,
    std::size_t input_cols,
    const std::uint8_t* packed_data,
    std::size_t packed_nbytes) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = "linear_op_dense";
  descriptor.op_class = "linear_op_dense";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = output_rows;
  descriptor.input_cols = input_cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed_data;
  descriptor.packed_nbytes = packed_nbytes;
  return descriptor;
}

GemmDescriptor make_nvfp4_descriptor(
    std::size_t output_rows,
    std::size_t input_cols,
    const std::uint8_t* packed_data,
    std::size_t packed_nbytes,
    const std::uint8_t* block_scales_data,
    std::size_t block_scales_nbytes,
    const std::uint8_t* tensor_scale_data,
    std::size_t tensor_scale_nbytes) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = "linear_op_nvfp4";
  descriptor.op_class = "linear_op_nvfp4";
  descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = output_rows;
  descriptor.input_cols = input_cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed_data;
  descriptor.packed_nbytes = packed_nbytes;
  descriptor.block_scales_data = block_scales_data;
  descriptor.block_scales_nbytes = block_scales_nbytes;
  descriptor.tensor_scale_data = tensor_scale_data;
  descriptor.tensor_scale_nbytes = tensor_scale_nbytes;
  return descriptor;
}

bool test_dense_runtime_plan_uses_uploaded_weight_alignment() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "linear_op_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kRows = 8;
  constexpr std::size_t kInputCols = 64;
  constexpr std::size_t kOutputRows = 64;

  std::vector<float> weights(kOutputRows * kInputCols, 0.0f);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = (static_cast<float>((i * 11) % 37) - 18.0f) * 0.03125f;
  }

  std::vector<std::uint8_t> misaligned_bytes(weights.size() * sizeof(float) + 1u, 0u);
  std::memcpy(misaligned_bytes.data() + 1u, weights.data(), weights.size() * sizeof(float));
  const auto descriptor = make_dense_descriptor(
      kOutputRows,
      kInputCols,
      misaligned_bytes.data() + 1u,
      weights.size() * sizeof(float));

  auto uploaded = DeviceDenseWeightFp32::Upload(descriptor);
  if (!expect(uploaded != nullptr && uploaded->valid(), "dense upload should succeed")) {
    return false;
  }

  const auto host_launch_plan = BuildGemmLaunchPlan(descriptor, kRows);
  if (!expect(host_launch_plan.has_value(), "host launch plan should build")) {
    return false;
  }
  GemmHeuristicCache cache;
  const auto host_execution = PrepareGemmExecution(*host_launch_plan, &cache);
  if (!expect(host_execution.has_value(), "host execution should prepare")) {
    return false;
  }
  if (!expect(
          !BuildCublasLtGemmPlan(*host_execution).has_value(),
          "host-backed dense plan should reject the deliberately misaligned source pointer")) {
    return false;
  }

  auto runtime_launch_plan = *host_launch_plan;
  runtime_launch_plan.packed_bytes = nemotron::ByteRangeView{
      reinterpret_cast<const std::uint8_t*>(uploaded->data()),
      uploaded->numel() * sizeof(float),
  };
  const auto runtime_execution = PrepareGemmExecution(runtime_launch_plan, &cache);
  if (!expect(runtime_execution.has_value(), "runtime execution should prepare")) {
    return false;
  }
  return expect(
      BuildCublasLtGemmPlan(*runtime_execution).has_value(),
      "dense runtime plan should accept the uploaded device pointer");
}

bool test_nvfp4_runtime_plan_uses_uploaded_weight_alignment() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "linear_op_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kRows = 8;
  constexpr std::size_t kInputCols = 64;
  constexpr std::size_t kOutputRows = 64;

  std::vector<float> weights(kOutputRows * kInputCols, 0.0f);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = (static_cast<float>((i * 5) % 19) - 9.0f) * 0.125f;
  }
  const auto packed = PackRowMajorFp32ToNvfp4(weights.data(), kOutputRows, kInputCols);
  if (!expect(packed.has_value() && packed->valid(), "NVFP4 pack should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> misaligned_packed(packed->packed_nbytes() + 1u, 0u);
  std::vector<std::uint8_t> misaligned_scales(packed->block_scales_nbytes() + 1u, 0u);
  std::vector<std::uint8_t> misaligned_tensor_scale(packed->tensor_scale_nbytes() + 1u, 0u);
  std::memcpy(misaligned_packed.data() + 1u, packed->packed_data(), packed->packed_nbytes());
  std::memcpy(misaligned_scales.data() + 1u, packed->block_scales_data(), packed->block_scales_nbytes());
  std::memcpy(
      misaligned_tensor_scale.data() + 1u,
      packed->tensor_scale_data(),
      packed->tensor_scale_nbytes());

  const auto descriptor = make_nvfp4_descriptor(
      kOutputRows,
      kInputCols,
      misaligned_packed.data() + 1u,
      packed->packed_nbytes(),
      misaligned_scales.data() + 1u,
      packed->block_scales_nbytes(),
      misaligned_tensor_scale.data() + 1u,
      packed->tensor_scale_nbytes());

  auto uploaded = DeviceNvfp4Weight::Upload(descriptor);
  if (!expect(uploaded != nullptr && uploaded->valid(), "NVFP4 upload should succeed")) {
    return false;
  }

  const auto host_launch_plan = BuildGemmLaunchPlan(descriptor, kRows);
  if (!expect(host_launch_plan.has_value(), "NVFP4 host launch plan should build")) {
    return false;
  }
  GemmHeuristicCache cache;
  const auto host_execution = PrepareGemmExecution(*host_launch_plan, &cache);
  if (!expect(host_execution.has_value(), "NVFP4 host execution should prepare")) {
    return false;
  }
  if (!expect(
          !BuildCublasLtGemmPlan(*host_execution).has_value(),
          "host-backed NVFP4 plan should reject deliberately misaligned source pointers")) {
    return false;
  }

  auto runtime_launch_plan = *host_launch_plan;
  runtime_launch_plan.packed_bytes = nemotron::ByteRangeView{
      uploaded->packed_data(),
      uploaded->packed_nbytes(),
  };
  runtime_launch_plan.block_scales_bytes = nemotron::ByteRangeView{
      uploaded->matmul_block_scales_data(),
      uploaded->matmul_block_scales_nbytes(),
  };
  runtime_launch_plan.tensor_scale_bytes = nemotron::ByteRangeView{
      uploaded->tensor_scale_data(),
      uploaded->tensor_scale_nbytes(),
  };
  const auto runtime_execution = PrepareGemmExecution(runtime_launch_plan, &cache);
  if (!expect(runtime_execution.has_value(), "NVFP4 runtime execution should prepare")) {
    return false;
  }
  return expect(
      BuildCublasLtGemmPlan(*runtime_execution).has_value(),
      "NVFP4 runtime plan should accept the uploaded device pointers");
}

bool test_uploaded_linear_op_matches_reference_for_small_m_nvfp4() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "linear_op_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kRows = 1;
  constexpr std::size_t kInputCols = 5376;
  constexpr std::size_t kOutputRows = 4096;

  std::vector<float> weights(kOutputRows * kInputCols, 0.0f);
  for (std::size_t row = 0; row < kOutputRows; ++row) {
    for (std::size_t col = 0; col < kInputCols; ++col) {
      const int pattern = static_cast<int>((row * 17 + col * 13) % 29) - 14;
      weights[row * kInputCols + col] = static_cast<float>(pattern) * 0.0078125f;
    }
  }
  const auto packed = PackRowMajorFp32ToNvfp4(weights.data(), kOutputRows, kInputCols);
  if (!expect(packed.has_value() && packed->valid(), "small-M NVFP4 weight pack should succeed")) {
    return false;
  }

  const auto descriptor = make_nvfp4_descriptor(
      kOutputRows,
      kInputCols,
      packed->packed_data(),
      packed->packed_nbytes(),
      packed->block_scales_data(),
      packed->block_scales_nbytes(),
      packed->tensor_scale_data(),
      packed->tensor_scale_nbytes());

  auto op = nemotron::UploadedLinearOp::Create(descriptor);
  auto reference_weight = DeviceNvfp4Weight::Upload(descriptor);
  auto activations = DeviceTensorFp32::Create({kRows, kInputCols});
  auto fastpath_output = DeviceTensorFp32::Create({kRows, kOutputRows});
  auto reference_output = DeviceTensorFp32::Create({kRows, kOutputRows});
  if (!expect(op != nullptr && op->valid(), "uploaded NVFP4 linear op should build")) {
    return false;
  }
  if (!expect(
          reference_weight != nullptr && reference_weight->valid() &&
              activations != nullptr && activations->valid() &&
              fastpath_output != nullptr && fastpath_output->valid() &&
              reference_output != nullptr && reference_output->valid(),
          "small-M NVFP4 linear test tensors should allocate")) {
    return false;
  }

  std::vector<float> activation_values(kRows * kInputCols, 0.0f);
  for (std::size_t col = 0; col < kInputCols; ++col) {
    const int pattern = static_cast<int>((col * 7) % 23) - 11;
    activation_values[col] = static_cast<float>(pattern) * 0.015625f;
  }
  if (!expect(
          activations->CopyFromHost(activation_values.data(), activation_values.size()),
          "small-M NVFP4 activations should upload")) {
    return false;
  }

  unsetenv("NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH");
  GemmHeuristicCache cache;
  if (!expect(
          op->Run(*handle, &cache, *activations, fastpath_output.get()),
          "uploaded NVFP4 linear fastpath should execute by default")) {
    return false;
  }

  if (!expect(
          RunNvfp4RowMajorReferenceToDevice(
              *activations,
              *reference_weight,
              reference_output.get(),
              nemotron::Nvfp4PackOptions{}),
          "uploaded NVFP4 linear reference path should execute")) {
    return false;
  }

  std::vector<float> fastpath_host(fastpath_output->numel(), 0.0f);
  std::vector<float> reference_host(reference_output->numel(), 0.0f);
  if (!expect(
          fastpath_output->CopyToHost(fastpath_host.data(), fastpath_host.size()),
          "uploaded NVFP4 fastpath output should download") ||
      !expect(
          reference_output->CopyToHost(reference_host.data(), reference_host.size()),
          "uploaded NVFP4 reference output should download")) {
    return false;
  }

  for (float value : fastpath_host) {
    if (!std::isfinite(value)) {
      return expect(false, "uploaded NVFP4 small-M fastpath output should stay finite");
    }
  }

  return expect(
      max_abs_diff(fastpath_host, reference_host) <= 5.0e-2f,
      "uploaded NVFP4 small-M fastpath output should stay close to reference");
}

}  // namespace

int main() {
  const bool ok =
      test_dense_runtime_plan_uses_uploaded_weight_alignment() &&
      test_nvfp4_runtime_plan_uses_uploaded_weight_alignment() &&
      test_uploaded_linear_op_matches_reference_for_small_m_nvfp4();
  if (!ok) {
    return 1;
  }
  std::cout << "linear_op_test: PASS\n";
  return 0;
}
