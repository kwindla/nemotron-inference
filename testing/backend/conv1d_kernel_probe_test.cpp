#include "nemotron/device_tensor.h"
#include "nemotron/mamba_ops.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::MambaConv1dSiluUpdateBf16;

constexpr std::size_t kT = 40;
constexpr std::size_t kH = 128;
constexpr std::size_t kP = 64;
constexpr std::size_t kN = 128;
constexpr std::size_t kG = 8;
constexpr std::size_t kI = kH * kP;
constexpr std::size_t kConvDim = kI + (2 * kG * kN);
constexpr std::size_t kProjSize = kI + kConvDim + kH;
constexpr std::size_t kConvKernelSize = 4;

std::vector<float> ReadFp32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<float> data(static_cast<std::size_t>(nbytes) / sizeof(float), 0.0f);
  in.read(reinterpret_cast<char*>(data.data()), nbytes);
  return data;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return INFINITY;
  }
  float max_d = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_d = std::max(max_d, std::fabs(a[i] - b[i]));
  }
  return max_d;
}

float RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return INFINITY;
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    diff_sq += diff * diff;
    ref_sq += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return ref_sq == 0.0 ? 0.0f : static_cast<float>(std::sqrt(diff_sq / ref_sq));
}

std::filesystem::path ResolveFixtureRoot() {
  const char* override_env = std::getenv("NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT_OVERRIDE");
  if (override_env != nullptr && *override_env != '\0') {
    return std::filesystem::path(override_env);
  }
  return std::filesystem::path(NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT);
}

bool CopyBf16TensorToHost(const DeviceTensorBf16& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  if (!tensor.CopyToHost(host_bf16.data(), host_bf16.size())) {
    return false;
  }
  output->resize(host_bf16.size(), 0.0f);
  for (std::size_t i = 0; i < host_bf16.size(); ++i) {
    (*output)[i] = __bfloat162float(host_bf16[i]);
  }
  return true;
}

}  // namespace

int main() {
  const char* dump_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  if (dump_dir_env == nullptr || *dump_dir_env == '\0') {
    std::cerr << "SKIP: set NEMOTRON_VLLM_DUMP_DIR to a pinned vLLM chunked-scan dump\n";
    return 0;
  }

  const std::filesystem::path dump_dir(dump_dir_env);
  const std::filesystem::path fixture_root = ResolveFixtureRoot();

  const auto projected_fp32 = ReadFp32(dump_dir / "projected_states_fp32.bin");
  const auto x_fp32 = ReadFp32(dump_dir / "x_fp32.bin");
  const auto b_fp32 = ReadFp32(dump_dir / "B_fp32.bin");
  const auto c_fp32 = ReadFp32(dump_dir / "C_fp32.bin");
  const auto conv1d_weight = ReadFp32(fixture_root / "conv1d_weight_fp32.bin");
  const auto conv1d_bias = ReadFp32(fixture_root / "conv1d_bias_fp32.bin");

  if (projected_fp32.size() != (kT * kProjSize) ||
      x_fp32.size() != (kT * kI) ||
      b_fp32.size() != (kT * kG * kN) ||
      c_fp32.size() != (kT * kG * kN) ||
      conv1d_weight.size() != (kConvDim * kConvKernelSize) ||
      conv1d_bias.size() != kConvDim) {
    std::cerr << "FAIL: unexpected fixture sizes\n";
    return 1;
  }

  std::vector<float> expected_conv_output(kT * kConvDim, 0.0f);
  for (std::size_t t = 0; t < kT; ++t) {
    std::memcpy(&expected_conv_output[t * kConvDim], &x_fp32[t * kI], kI * sizeof(float));
    std::memcpy(
        &expected_conv_output[t * kConvDim + kI],
        &b_fp32[t * kG * kN],
        kG * kN * sizeof(float));
    std::memcpy(
        &expected_conv_output[t * kConvDim + kI + (kG * kN)],
        &c_fp32[t * kG * kN],
        kG * kN * sizeof(float));
  }

  std::vector<__nv_bfloat16> projected_bf16(projected_fp32.size());
  for (std::size_t i = 0; i < projected_fp32.size(); ++i) {
    projected_bf16[i] = __float2bfloat16(projected_fp32[i]);
  }

  auto projected = DeviceTensorBf16::Create({kT, kProjSize});
  auto conv_weight = DeviceTensorFp32::Create({kConvDim * kConvKernelSize});
  auto conv_bias = DeviceTensorFp32::Create({kConvDim});
  auto conv_state = DeviceTensorBf16::Create({kConvDim * kConvKernelSize});
  auto conv_output = DeviceTensorBf16::Create({kT, kConvDim});
  if (!projected || !projected->valid() ||
      !conv_weight || !conv_weight->valid() ||
      !conv_bias || !conv_bias->valid() ||
      !conv_state || !conv_state->valid() ||
      !conv_output || !conv_output->valid() ||
      !projected->CopyFromHost(projected_bf16.data(), projected_bf16.size()) ||
      !conv_weight->CopyFromHost(conv1d_weight.data(), conv1d_weight.size()) ||
      !conv_bias->CopyFromHost(conv1d_bias.data(), conv1d_bias.size()) ||
      !conv_state->FillZero()) {
    std::cerr << "FAIL: device tensor setup failed\n";
    return 1;
  }

  if (!MambaConv1dSiluUpdateBf16(
          *projected,
          kI,
          kConvDim,
          kConvKernelSize,
          0,
          *conv_weight,
          *conv_bias,
          conv_state.get(),
          conv_output.get())) {
    std::cerr << "FAIL: MambaConv1dSiluUpdateBf16 failed\n";
    return 1;
  }

  std::vector<float> actual_conv_output;
  if (!CopyBf16TensorToHost(*conv_output, &actual_conv_output)) {
    std::cerr << "FAIL: conv output download failed\n";
    return 1;
  }

  const float max_abs_diff = MaxAbsDiff(actual_conv_output, expected_conv_output);
  const float rel_l2 = RelL2(actual_conv_output, expected_conv_output);
  std::cerr << "conv1d_kernel_probe_test: conv_output mad=" << max_abs_diff
            << " rel_l2=" << rel_l2 << "\n";

  if (max_abs_diff > 1.0e-3f) {
    std::cerr << "FAIL: exact-input conv1d output still differs from pinned vLLM\n";
    return 1;
  }

  return 0;
}
