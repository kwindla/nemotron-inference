// Stage-5 probe: feed exact saved chunk-scan inputs to the runtime ChunkScan
// kernel and compare y_output directly against the vLLM oracle.

#include "nemotron/device_tensor.h"
#include "nemotron/mamba_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::MambaChunkScanOnlyBf16;

constexpr std::size_t kH = 128;
constexpr std::size_t kP = 64;
constexpr std::size_t kN = 128;
constexpr std::size_t kG = 8;
constexpr std::size_t kQ = 128;
constexpr std::size_t kI = kH * kP;
constexpr std::size_t kConvDim = kI + 2 * kG * kN;

std::vector<float> read_fp32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<float> data(static_cast<std::size_t>(nbytes) / sizeof(float));
  in.read(reinterpret_cast<char*>(data.data()), nbytes);
  return data;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return INFINITY;
  }
  float max_d = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_d = std::max(max_d, std::fabs(a[i] - b[i]));
  }
  return max_d;
}

float rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return INFINITY;
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    diff_sq += d * d;
    ref_sq += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return ref_sq > 0.0 ? static_cast<float>(std::sqrt(diff_sq / ref_sq))
                      : (diff_sq == 0.0 ? 0.0f : INFINITY);
}

std::vector<float> download_bf16_as_fp32(const DeviceTensorBf16& tensor) {
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  cudaMemcpy(host_bf16.data(), tensor.data(),
             host_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  std::vector<float> host_fp32(tensor.numel());
  for (std::size_t i = 0; i < host_fp32.size(); ++i) {
    host_fp32[i] = __bfloat162float(host_bf16[i]);
  }
  return host_fp32;
}

bool run_probe() {
  const char* vllm_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  if (!vllm_dir_env) {
    std::cout << "chunk_scan_only_kernel_probe_test: SKIP (NEMOTRON_VLLM_DUMP_DIR not set)\n";
    return true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "chunk_scan_only_kernel_probe_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path vllm_dir(vllm_dir_env);
  std::cout << "chunk_scan_only_kernel_probe_test: loading vLLM inputs from " << vllm_dir << "\n";

  const auto vllm_x = read_fp32(vllm_dir / "x_fp32.bin");
  const auto vllm_C = read_fp32(vllm_dir / "C_fp32.bin");
  const auto vllm_D = read_fp32(vllm_dir / "D_fp32.bin");
  const auto vllm_dt_chunk = read_fp32(vllm_dir / "dt_chunk_fp32.bin");
  const auto vllm_dA_cumsum = read_fp32(vllm_dir / "dA_cumsum_fp32.bin");
  const auto vllm_states = read_fp32(vllm_dir / "boundary_state_fp32.bin");
  const auto vllm_cb = read_fp32(vllm_dir / "cb_chunk_fp32.bin");
  const auto vllm_y = read_fp32(vllm_dir / "y_output_fp32.bin");
  const auto vllm_initial =
      std::filesystem::exists(vllm_dir / "initial_states_fp32.bin")
          ? read_fp32(vllm_dir / "initial_states_fp32.bin")
          : std::vector<float>{};

  if (vllm_x.size() % (kH * kP) != 0) {
    std::cerr << "FAIL: x dump size is not divisible by H*P\n";
    return false;
  }
  const std::size_t kT = vllm_x.size() / (kH * kP);
  const std::size_t kC = (kT + kQ - 1) / kQ;
  if (vllm_C.size() != kT * kG * kN ||
      vllm_D.size() != kH ||
      vllm_dt_chunk.size() != kH * kC * kQ ||
      vllm_dA_cumsum.size() != kH * kC * kQ ||
      vllm_states.size() != kC * kH * kP * kN ||
      vllm_cb.size() != kC * kG * kQ * kQ ||
      vllm_y.size() != kT * kI) {
    std::cerr << "FAIL: unexpected vLLM dump sizes\n";
    return false;
  }

  std::vector<float> conv_output_fp32(kT * kConvDim, 0.0f);
  for (std::size_t t = 0; t < kT; ++t) {
    std::memcpy(&conv_output_fp32[t * kConvDim],
                &vllm_x[t * kH * kP], kI * sizeof(float));
    std::memcpy(&conv_output_fp32[t * kConvDim + kI + kG * kN],
                &vllm_C[t * kG * kN], kG * kN * sizeof(float));
  }

  std::vector<float> dt_chunk_fp32(kC * kH * kQ);
  std::vector<float> dA_cumsum_fp32(kC * kH * kQ);
  for (std::size_t h = 0; h < kH; ++h) {
    for (std::size_t c = 0; c < kC; ++c) {
      for (std::size_t q = 0; q < kQ; ++q) {
        const std::size_t rt_idx = c * kH * kQ + h * kQ + q;
        const std::size_t vl_idx = h * kC * kQ + c * kQ + q;
        dt_chunk_fp32[rt_idx] = vllm_dt_chunk[vl_idx];
        dA_cumsum_fp32[rt_idx] = vllm_dA_cumsum[vl_idx];
      }
    }
  }

  std::vector<float> entering_state_fp32(kC * kH * kP * kN, 0.0f);
  if (!vllm_initial.empty()) {
    std::memcpy(entering_state_fp32.data(), vllm_initial.data(), kH * kP * kN * sizeof(float));
  }
  for (std::size_t c = 1; c < kC; ++c) {
    std::memcpy(&entering_state_fp32[c * kH * kP * kN],
                &vllm_states[(c - 1) * kH * kP * kN],
                kH * kP * kN * sizeof(float));
  }

  std::vector<__nv_bfloat16> conv_output_bf16(conv_output_fp32.size());
  for (std::size_t i = 0; i < conv_output_fp32.size(); ++i) {
    conv_output_bf16[i] = __float2bfloat16(conv_output_fp32[i]);
  }

  auto dev_conv_output = DeviceTensorBf16::Create({kT, kConvDim});
  auto dev_d = DeviceTensorFp32::Create({kH});
  auto dev_dt_chunk = DeviceTensorFp32::Create({kC, kH, kQ});
  auto dev_dA_cumsum = DeviceTensorFp32::Create({kC, kH, kQ});
  auto dev_boundary_state = DeviceTensorFp32::Create({kC, kH, kP, kN});
  auto dev_cb_chunk = DeviceTensorFp32::Create({kC, kG, kQ, kQ});
  auto dev_y_output = DeviceTensorBf16::Create({kT, kI});
  if (!dev_conv_output || !dev_d || !dev_dt_chunk || !dev_dA_cumsum ||
      !dev_boundary_state || !dev_cb_chunk || !dev_y_output) {
    std::cerr << "FAIL: device tensor allocation\n";
    return false;
  }

  cudaMemcpy(dev_conv_output->data(), conv_output_bf16.data(),
             conv_output_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_d->data(), vllm_D.data(), vllm_D.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dt_chunk->data(), dt_chunk_fp32.data(),
             dt_chunk_fp32.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dA_cumsum->data(), dA_cumsum_fp32.data(),
             dA_cumsum_fp32.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_boundary_state->data(), entering_state_fp32.data(),
             entering_state_fp32.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_cb_chunk->data(), vllm_cb.data(),
             vllm_cb.size() * sizeof(float), cudaMemcpyHostToDevice);

  const bool ok = MambaChunkScanOnlyBf16(
      *dev_conv_output,
      kI,
      kConvDim,
      kH,
      kP,
      kN,
      kG,
      kQ,
      *dev_d,
      *dev_dt_chunk,
      *dev_dA_cumsum,
      *dev_boundary_state,
      *dev_cb_chunk,
      dev_y_output.get());
  if (!ok) {
    std::cerr << "FAIL: MambaChunkScanOnlyBf16 returned false\n";
    return false;
  }
  cudaDeviceSynchronize();

  const auto our_y = download_bf16_as_fp32(*dev_y_output);
  const float mad = max_abs_diff(our_y, vllm_y);
  const float rl2 = rel_l2(our_y, vllm_y);
  std::cout << "\n=== Runtime Stage-5 vs vLLM Reference ===\n";
  std::cout << "  y_output: mad=" << mad << " rl2=" << rl2
            << " (n=" << our_y.size() << ")\n";
  return mad <= 1.0e-3f;
}

}  // namespace

int main() {
  return run_probe() ? 0 : 1;
}
