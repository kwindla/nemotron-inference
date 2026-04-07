// Probe Stages 2-5 using exact saved vLLM Stage-1 outputs.

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
using nemotron::MambaChunkScanWorkspace;
using nemotron::MambaChunkedScanFromStage1Bf16;

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
  if (a.size() != b.size()) return INFINITY;
  float max_d = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_d = std::max(max_d, std::fabs(a[i] - b[i]));
  }
  return max_d;
}

float rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
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

std::vector<float> download_fp32(const DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel());
  cudaMemcpy(host.data(), tensor.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost);
  return host;
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
    std::cout << "chunked_scan_from_stage1_probe_test: SKIP (NEMOTRON_VLLM_DUMP_DIR not set)\n";
    return true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "chunked_scan_from_stage1_probe_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path vllm_dir(vllm_dir_env);
  std::cout << "chunked_scan_from_stage1_probe_test: loading vLLM inputs from " << vllm_dir << "\n";

  const auto vllm_x = read_fp32(vllm_dir / "x_fp32.bin");
  const auto vllm_B = read_fp32(vllm_dir / "B_fp32.bin");
  const auto vllm_C = read_fp32(vllm_dir / "C_fp32.bin");
  const auto vllm_D = read_fp32(vllm_dir / "D_fp32.bin");
  const auto vllm_dt_chunk = read_fp32(vllm_dir / "dt_chunk_fp32.bin");
  const auto vllm_dA_cumsum = read_fp32(vllm_dir / "dA_cumsum_fp32.bin");
  const auto vllm_y = read_fp32(vllm_dir / "y_output_fp32.bin");
  const auto vllm_state_out = read_fp32(vllm_dir / "ssm_state_out_fp32.bin");

  if (vllm_x.size() % (kH * kP) != 0) {
    std::cerr << "FAIL: x dump size is not divisible by H*P\n";
    return false;
  }
  const std::size_t kT = vllm_x.size() / (kH * kP);
  const std::size_t kC = (kT + kQ - 1) / kQ;
  if (vllm_B.size() != kT * kG * kN ||
      vllm_C.size() != kT * kG * kN ||
      vllm_D.size() != kH ||
      vllm_dt_chunk.size() != kH * kC * kQ ||
      vllm_dA_cumsum.size() != kH * kC * kQ ||
      vllm_y.size() != kT * kI ||
      vllm_state_out.size() != kH * kP * kN) {
    std::cerr << "FAIL: unexpected vLLM dump sizes\n";
    return false;
  }

  std::vector<float> conv_output_fp32(kT * kConvDim, 0.0f);
  for (std::size_t t = 0; t < kT; ++t) {
    std::memcpy(&conv_output_fp32[t * kConvDim],
                &vllm_x[t * kH * kP], kI * sizeof(float));
    std::memcpy(&conv_output_fp32[t * kConvDim + kI],
                &vllm_B[t * kG * kN], kG * kN * sizeof(float));
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

  std::vector<__nv_bfloat16> conv_output_bf16(conv_output_fp32.size());
  for (std::size_t i = 0; i < conv_output_fp32.size(); ++i) {
    conv_output_bf16[i] = __float2bfloat16(conv_output_fp32[i]);
  }

  auto dev_conv_output = DeviceTensorBf16::Create({kT, kConvDim});
  auto dev_d = DeviceTensorFp32::Create({kH});
  auto dev_dt_chunk = DeviceTensorFp32::Create({kC, kH, kQ});
  auto dev_dA_cumsum = DeviceTensorFp32::Create({kC, kH, kQ});
  auto dev_ssm_state = DeviceTensorFp32::Create({kH * kP * kN});
  auto dev_y_output = DeviceTensorBf16::Create({kT, kI});
  if (!dev_conv_output || !dev_d || !dev_dt_chunk || !dev_dA_cumsum ||
      !dev_ssm_state || !dev_y_output) {
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
  cudaMemset(dev_ssm_state->data(), 0, dev_ssm_state->numel() * sizeof(float));

  MambaChunkScanWorkspace workspace;
  workspace.dt_chunk = DeviceTensorFp32::Create({kC, kH, kQ});
  workspace.dA_cumsum = DeviceTensorFp32::Create({kC, kH, kQ});
  workspace.state_scratch = DeviceTensorFp32::Create({kC, kH, kP, kN});
  workspace.cb_chunk = DeviceTensorFp32::Create({kC, kG, kQ, kQ});
  workspace.capacity_tokens = kT;
  workspace.chunk_size = kQ;
  if (!workspace.dt_chunk || !workspace.dA_cumsum ||
      !workspace.state_scratch || !workspace.cb_chunk) {
    std::cerr << "FAIL: workspace allocation\n";
    return false;
  }

  const bool ok = MambaChunkedScanFromStage1Bf16(
      *dev_conv_output,
      kI,
      kConvDim,
      kH,
      kP,
      kN,
      kG,
      kQ,
      0,
      *dev_d,
      *dev_dt_chunk,
      *dev_dA_cumsum,
      dev_ssm_state.get(),
      dev_y_output.get(),
      &workspace);
  if (!ok) {
    std::cerr << "FAIL: MambaChunkedScanFromStage1Bf16 returned false\n";
    return false;
  }
  cudaDeviceSynchronize();

  const auto our_y = download_bf16_as_fp32(*dev_y_output);
  const auto our_state_out = download_fp32(*dev_ssm_state);
  const float y_mad = max_abs_diff(our_y, vllm_y);
  const float y_rl2 = rel_l2(our_y, vllm_y);
  const float state_mad = max_abs_diff(our_state_out, vllm_state_out);
  const float state_rl2 = rel_l2(our_state_out, vllm_state_out);
  std::cout << "\n=== Runtime Stages 2-5 with Exact vLLM Stage-1 Inputs ===\n";
  std::cout << "  y_output: mad=" << y_mad << " rl2=" << y_rl2 << " (n=" << our_y.size() << ")\n";
  std::cout << "  ssm_state_out: mad=" << state_mad << " rl2=" << state_rl2
            << " (n=" << our_state_out.size() << ")\n";
  return y_mad <= 1.0e-3f && state_mad <= 1.0e-3f;
}

}  // namespace

int main() {
  return run_probe() ? 0 : 1;
}
