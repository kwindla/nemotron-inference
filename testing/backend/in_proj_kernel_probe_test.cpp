#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/scaled_fp8_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorBf16;
using nemotron::GemmHeuristicCache;
using nemotron::ScaledFp8LinearConfig;
using nemotron::ScaledFp8LinearOp;

constexpr std::size_t kTokens = 40;
constexpr std::size_t kHiddenSize = 4096;
constexpr std::size_t kIntermediateSize = 8192;
constexpr std::size_t kNumHeads = 128;
constexpr std::size_t kStateSize = 128;
constexpr std::size_t kNumGroups = 8;
constexpr std::size_t kConvDim = kIntermediateSize + (2 * kNumGroups * kStateSize);
constexpr std::size_t kProjectedSize = kIntermediateSize + kConvDim + kNumHeads;

std::filesystem::path ResolveFixtureRoot() {
  const char* override_env = std::getenv("NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT_OVERRIDE");
  if (override_env != nullptr && *override_env != '\0') {
    return std::filesystem::path(override_env);
  }
  return std::filesystem::path(NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT);
}

std::vector<float> ReadFp32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  if (nbytes % sizeof(float) != 0) {
    std::cerr << "FAIL: invalid fp32 payload size " << path << "\n";
    return {};
  }
  std::vector<float> data(nbytes / sizeof(float), 0.0f);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
  return data;
}

std::vector<std::uint8_t> ReadU8(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(nbytes, 0);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
  return data;
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

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_d = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_d = std::max(max_d, std::fabs(a[i] - b[i]));
  }
  return max_d;
}

float RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
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

}  // namespace

int main() {
  const char* dump_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  if (dump_dir_env == nullptr || *dump_dir_env == '\0') {
    std::cerr << "SKIP: set NEMOTRON_VLLM_DUMP_DIR to a pinned vLLM dump\n";
    return 0;
  }

  const std::filesystem::path dump_dir(dump_dir_env);
  const std::filesystem::path fixture_root = ResolveFixtureRoot();

  const auto norm_output = ReadFp32(dump_dir / "norm_output_fp32.bin");
  const auto expected_projected = ReadFp32(dump_dir / "projected_states_fp32.bin");
  const auto packed_weight = ReadU8(fixture_root / "in_proj_weight_fp8.bin");
  const auto weight_scale = ReadFp32(fixture_root / "in_proj_weight_scale_fp32.bin");
  const auto input_scale = ReadFp32(fixture_root / "in_proj_input_scale_fp32.bin");

  if (norm_output.size() != (kTokens * kHiddenSize) ||
      expected_projected.size() != (kTokens * kProjectedSize) ||
      packed_weight.size() != (kProjectedSize * kHiddenSize) ||
      weight_scale.size() != 1 ||
      input_scale.size() != 1) {
    std::cerr << "FAIL: unexpected oracle fixture sizes\n";
    return 1;
  }

  std::vector<__nv_bfloat16> norm_output_bf16(norm_output.size());
  for (std::size_t i = 0; i < norm_output.size(); ++i) {
    norm_output_bf16[i] = __float2bfloat16(norm_output[i]);
  }

  ScaledFp8LinearConfig config;
  config.output_rows = kProjectedSize;
  config.input_cols = kHiddenSize;
  config.packed_weight_data = packed_weight.data();
  config.packed_weight_nbytes = packed_weight.size();
  config.tensor_name = "layer0_in_proj_kernel_probe";
  config.weight_scale = weight_scale[0];
  config.input_scale = input_scale[0];

  auto op = ScaledFp8LinearOp::Create(config);
  auto handle = CublasLtHandle::Create();
  auto input = DeviceTensorBf16::Create({kTokens, kHiddenSize});
  auto output = DeviceTensorBf16::Create({kTokens, kProjectedSize});
  if (op == nullptr || !op->valid() ||
      handle == nullptr || !handle->valid() ||
      input == nullptr || !input->valid() ||
      output == nullptr || !output->valid() ||
      !input->CopyFromHost(norm_output_bf16.data(), norm_output_bf16.size())) {
    std::cerr << "FAIL: scaled-fp8 probe setup failed\n";
    return 1;
  }

  GemmHeuristicCache heuristic_cache;
  if (!op->Run(*handle, &heuristic_cache, *input, output.get())) {
    std::cerr << "FAIL: in_proj ScaledFp8LinearOp::Run failed\n";
    return 1;
  }

  std::vector<float> actual_projected;
  if (!CopyBf16TensorToHost(*output, &actual_projected)) {
    std::cerr << "FAIL: in_proj output download failed\n";
    return 1;
  }

  const float max_abs_diff = MaxAbsDiff(actual_projected, expected_projected);
  const float rel_l2 = RelL2(actual_projected, expected_projected);
  std::cerr << "in_proj_kernel_probe_test: projected mad=" << max_abs_diff
            << " rel_l2=" << rel_l2 << "\n";

  if (max_abs_diff > 1.0e-3f) {
    std::cerr << "FAIL: exact-input in_proj output still differs from pinned vLLM\n";
    return 1;
  }

  return 0;
}
