// Kernel verification test: feed vLLM's exact Stage-1 inputs to our CUDA
// chunk-cumsum kernel and compare dt_chunk / dA_cumsum directly.
//
// Usage:
//   NEMOTRON_VLLM_DUMP_DIR=<path-to-vllm-dumps> \
//   NEMOTRON_DUMP_CHUNK_CUMSUM_ROOT=<optional-output-path> \
//   build/testing/chunk_cumsum_kernel_probe_test

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
using nemotron::MambaChunkCumsumBf16;

constexpr std::size_t kH = 128;
constexpr std::size_t kP = 64;
constexpr std::size_t kN = 128;
constexpr std::size_t kG = 8;
constexpr std::size_t kQ = 128;
constexpr std::size_t kI = kH * kP;
constexpr std::size_t kConvDim = kI + 2 * kG * kN;
constexpr std::size_t kProjSize = kI + kConvDim + kH;

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

bool write_fp32(const std::filesystem::path& path, const std::vector<float>& data) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
  return out.good();
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

std::vector<float> download_fp32(const DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel());
  cudaMemcpy(host.data(), tensor.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost);
  return host;
}

bool run_probe() {
  const char* vllm_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  if (!vllm_dir_env) {
    std::cout << "chunk_cumsum_kernel_probe_test: SKIP (NEMOTRON_VLLM_DUMP_DIR not set)\n";
    return true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "chunk_cumsum_kernel_probe_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path vllm_dir(vllm_dir_env);
  std::cout << "chunk_cumsum_kernel_probe_test: loading vLLM inputs from " << vllm_dir << "\n";

  const auto vllm_dt_pre = read_fp32(vllm_dir / "dt_pre_fp32.bin");
  const auto vllm_A = read_fp32(vllm_dir / "A_fp32.bin");
  const auto vllm_dt_bias = read_fp32(vllm_dir / "dt_bias_fp32.bin");
  const auto vllm_dt_chunk = read_fp32(vllm_dir / "dt_chunk_fp32.bin");
  const auto vllm_dA_cumsum = read_fp32(vllm_dir / "dA_cumsum_fp32.bin");

  if (vllm_dt_pre.size() % kH != 0) {
    std::cerr << "FAIL: dt_pre dump size is not divisible by H\n";
    return false;
  }
  const std::size_t kT = vllm_dt_pre.size() / kH;
  const std::size_t kC = (kT + kQ - 1) / kQ;
  if (vllm_A.size() != kH ||
      vllm_dt_bias.size() != kH ||
      vllm_dt_chunk.size() != kH * kC * kQ ||
      vllm_dA_cumsum.size() != kH * kC * kQ) {
    std::cerr << "FAIL: unexpected vLLM dump sizes\n";
    return false;
  }

  std::vector<float> projected_fp32(kT * kProjSize, 0.0f);
  for (std::size_t t = 0; t < kT; ++t) {
    std::memcpy(&projected_fp32[t * kProjSize + kI + kConvDim],
                &vllm_dt_pre[t * kH], kH * sizeof(float));
  }

  std::vector<__nv_bfloat16> projected_bf16(projected_fp32.size());
  for (std::size_t i = 0; i < projected_fp32.size(); ++i) {
    projected_bf16[i] = __float2bfloat16(projected_fp32[i]);
  }

  auto dev_projected = DeviceTensorBf16::Create({kT, kProjSize});
  auto dev_a = DeviceTensorFp32::Create({kH});
  auto dev_dt_bias = DeviceTensorFp32::Create({kH});
  auto dev_dt_chunk = DeviceTensorFp32::Create({kC, kH, kQ});
  auto dev_dA_cumsum = DeviceTensorFp32::Create({kC, kH, kQ});
  if (!dev_projected || !dev_a || !dev_dt_bias || !dev_dt_chunk || !dev_dA_cumsum) {
    std::cerr << "FAIL: device tensor allocation\n";
    return false;
  }

  cudaMemcpy(dev_projected->data(), projected_bf16.data(),
             projected_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_a->data(), vllm_A.data(), vllm_A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dt_bias->data(), vllm_dt_bias.data(),
             vllm_dt_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

  const bool ok = MambaChunkCumsumBf16(
      *dev_projected,
      kI,
      kConvDim,
      kH,
      kQ,
      *dev_a,
      *dev_dt_bias,
      dev_dt_chunk.get(),
      dev_dA_cumsum.get());
  if (!ok) {
    std::cerr << "FAIL: MambaChunkCumsumBf16 returned false\n";
    return false;
  }
  cudaDeviceSynchronize();

  auto our_dt_chunk = download_fp32(*dev_dt_chunk);
  auto our_dA_cumsum = download_fp32(*dev_dA_cumsum);

  std::vector<float> vllm_dt_transposed(vllm_dt_chunk.size());
  std::vector<float> vllm_dA_transposed(vllm_dA_cumsum.size());
  for (std::size_t h = 0; h < kH; ++h) {
    for (std::size_t c = 0; c < kC; ++c) {
      for (std::size_t q = 0; q < kQ; ++q) {
        const std::size_t rt_idx = c * kH * kQ + h * kQ + q;
        const std::size_t vl_idx = h * kC * kQ + c * kQ + q;
        vllm_dt_transposed[rt_idx] = vllm_dt_chunk[vl_idx];
        vllm_dA_transposed[rt_idx] = vllm_dA_cumsum[vl_idx];
      }
    }
  }

  auto report = [](const char* name, const std::vector<float>& ours, const std::vector<float>& ref) {
    const float mad = max_abs_diff(ours, ref);
    const float rl2 = rel_l2(ours, ref);
    std::cout << "  " << name << ": mad=" << mad << " rl2=" << rl2 << " (n=" << ours.size() << ")\n";
    return mad;
  };

  std::cout << "\n=== Runtime Stage-1 vs vLLM Reference ===\n";
  const float dt_mad = report("dt_chunk", our_dt_chunk, vllm_dt_transposed);
  const float dA_mad = report("dA_cumsum", our_dA_cumsum, vllm_dA_transposed);

  if (const char* dump_root_env = std::getenv("NEMOTRON_DUMP_CHUNK_CUMSUM_ROOT")) {
    const std::filesystem::path root(dump_root_env);
    write_fp32(root / "runtime" / "dt_chunk_fp32.bin", our_dt_chunk);
    write_fp32(root / "runtime" / "dA_cumsum_fp32.bin", our_dA_cumsum);
    write_fp32(root / "reference" / "dt_chunk_fp32.bin", vllm_dt_transposed);
    write_fp32(root / "reference" / "dA_cumsum_fp32.bin", vllm_dA_transposed);
    std::cout << "dumped stage-1 probe outputs under " << root << "\n";
  }

  return dt_mad <= 1.0e-4f && dA_mad <= 1.0e-3f;
}

}  // namespace

int main() {
  return run_probe() ? 0 : 1;
}
