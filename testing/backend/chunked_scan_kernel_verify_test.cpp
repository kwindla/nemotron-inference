// Kernel verification test: feed vLLM's exact inputs to our CUDA chunked-scan
// kernels and compare stage-by-stage outputs against vLLM's intermediates.
//
// Usage:
//   NEMOTRON_VLLM_DUMP_DIR=<path-to-vllm-dumps> \
//   NEMOTRON_DUMP_CHUNKED_SCAN_ROOT=<output-path> \
//   NEMOTRON_DUMP_CHUNKED_SCAN_LAYER=0 \
//   build/testing/chunked_scan_kernel_verify_test

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
using nemotron::MambaChunkedScanPrefillBf16;

constexpr std::size_t kH = 128;
constexpr std::size_t kP = 64;
constexpr std::size_t kN = 128;
constexpr std::size_t kG = 8;
constexpr std::size_t kQ = 128;
constexpr std::size_t kI = kH * kP;            // 8192
constexpr std::size_t kConvDim = kI + 2 * kG * kN;  // 10240
constexpr std::size_t kProjSize = kI + kConvDim + kH;  // 18560

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
  if (a.size() != b.size()) return INFINITY;
  float max_d = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_d = std::max(max_d, std::fabs(a[i] - b[i]));
  }
  return max_d;
}

float rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  double diff_sq = 0.0, ref_sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    diff_sq += d * d;
    ref_sq += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return ref_sq > 0 ? static_cast<float>(std::sqrt(diff_sq / ref_sq)) : (diff_sq == 0 ? 0.0f : INFINITY);
}

std::vector<float> download_fp32(const DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel());
  cudaMemcpy(host.data(), tensor.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost);
  return host;
}

std::vector<float> download_bf16_as_fp32(const DeviceTensorBf16& tensor) {
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  cudaMemcpy(host_bf16.data(), tensor.data(), host_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  std::vector<float> host_fp32(tensor.numel());
  for (std::size_t i = 0; i < host_fp32.size(); ++i) {
    host_fp32[i] = __bfloat162float(host_bf16[i]);
  }
  return host_fp32;
}

bool run_verify() {
  const char* vllm_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  if (!vllm_dir_env) {
    std::cout << "chunked_scan_kernel_verify_test: SKIP (NEMOTRON_VLLM_DUMP_DIR not set)\n";
    return true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "chunked_scan_kernel_verify_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path vllm_dir(vllm_dir_env);
  std::cout << "chunked_scan_kernel_verify_test: loading vLLM inputs from " << vllm_dir << "\n";

  // Load vLLM inputs
  const auto vllm_x = read_fp32(vllm_dir / "x_fp32.bin");           // [T, H, P] = 327680
  const auto vllm_B = read_fp32(vllm_dir / "B_fp32.bin");           // [T, G, N] = 40960
  const auto vllm_C = read_fp32(vllm_dir / "C_fp32.bin");           // [T, G, N] = 40960
  const auto vllm_dt_pre = read_fp32(vllm_dir / "dt_pre_fp32.bin"); // [T, H] = 5120
  const auto vllm_A = read_fp32(vllm_dir / "A_fp32.bin");           // [H] = 128 (already -exp(A_log))
  const auto vllm_dt_bias = read_fp32(vllm_dir / "dt_bias_fp32.bin"); // [H] = 128
  const auto vllm_D = read_fp32(vllm_dir / "D_fp32.bin");           // [H] = 128

  if (vllm_x.size() % (kH * kP) != 0) {
    std::cerr << "FAIL: x dump size is not divisible by H*P\n";
    return false;
  }
  const std::size_t kT = vllm_x.size() / (kH * kP);
  if (vllm_B.size() != kT * kG * kN ||
      vllm_C.size() != kT * kG * kN ||
      vllm_dt_pre.size() != kT * kH ||
      vllm_A.size() != kH ||
      vllm_dt_bias.size() != kH ||
      vllm_D.size() != kH) {
    std::cerr << "FAIL: unexpected vLLM dump sizes\n";
    return false;
  }

  // Pack vLLM inputs into runtime format:
  // conv_output[t, :] = [X[t, :I], B[t, :G*N], C[t, :G*N]]  shape [T, kConvDim]
  // projected[t, :] = [gate[t, :I], conv_part[t, :kConvDim], dt_pre[t, :H]]  shape [T, kProjSize]
  //   (gate and conv_part are not read by the chunked scan, only dt_pre at the end)
  std::vector<float> conv_output_fp32(kT * kConvDim, 0.0f);
  std::vector<float> projected_fp32(kT * kProjSize, 0.0f);

  for (std::size_t t = 0; t < kT; ++t) {
    // conv_output: [X | B | C]
    std::memcpy(&conv_output_fp32[t * kConvDim],
                &vllm_x[t * kH * kP], kI * sizeof(float));
    std::memcpy(&conv_output_fp32[t * kConvDim + kI],
                &vllm_B[t * kG * kN], kG * kN * sizeof(float));
    std::memcpy(&conv_output_fp32[t * kConvDim + kI + kG * kN],
                &vllm_C[t * kG * kN], kG * kN * sizeof(float));

    // projected: [gate(zeros) | conv_part(zeros) | dt_pre]
    // dt_pre starts at offset kI + kConvDim
    std::memcpy(&projected_fp32[t * kProjSize + kI + kConvDim],
                &vllm_dt_pre[t * kH], kH * sizeof(float));
  }

  // Convert to BF16 for upload
  std::vector<__nv_bfloat16> conv_output_bf16(kT * kConvDim);
  std::vector<__nv_bfloat16> projected_bf16(kT * kProjSize);
  for (std::size_t i = 0; i < conv_output_fp32.size(); ++i) {
    conv_output_bf16[i] = __float2bfloat16(conv_output_fp32[i]);
  }
  for (std::size_t i = 0; i < projected_fp32.size(); ++i) {
    projected_bf16[i] = __float2bfloat16(projected_fp32[i]);
  }

  // Upload to device
  auto dev_conv_output = DeviceTensorBf16::Create({kT, kConvDim});
  auto dev_projected = DeviceTensorBf16::Create({kT, kProjSize});
  auto dev_a = DeviceTensorFp32::Create({kH});
  auto dev_dt_bias = DeviceTensorFp32::Create({kH});
  auto dev_d = DeviceTensorFp32::Create({kH});
  auto dev_ssm_state = DeviceTensorFp32::Create({kH * kP * kN});
  auto dev_y_output = DeviceTensorBf16::Create({kT, kI});

  if (!dev_conv_output || !dev_projected || !dev_a || !dev_dt_bias ||
      !dev_d || !dev_ssm_state || !dev_y_output) {
    std::cerr << "FAIL: device tensor allocation\n";
    return false;
  }

  cudaMemcpy(dev_conv_output->data(), conv_output_bf16.data(),
             conv_output_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_projected->data(), projected_bf16.data(),
             projected_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_a->data(), vllm_A.data(), vllm_A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dt_bias->data(), vllm_dt_bias.data(), vllm_dt_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_d->data(), vllm_D.data(), vllm_D.size() * sizeof(float), cudaMemcpyHostToDevice);
  // Zero initial state
  cudaMemset(dev_ssm_state->data(), 0, dev_ssm_state->numel() * sizeof(float));

  // Allocate workspace
  MambaChunkScanWorkspace workspace;
  const std::size_t C = (kT + kQ - 1) / kQ;
  workspace.dt_chunk = DeviceTensorFp32::Create({C, kH, kQ});
  workspace.dA_cumsum = DeviceTensorFp32::Create({C, kH, kQ});
  workspace.state_scratch = DeviceTensorFp32::Create({C, kH, kP, kN});
  workspace.cb_chunk = DeviceTensorFp32::Create({C, kG, kQ, kQ});
  workspace.capacity_tokens = kT;
  workspace.chunk_size = kQ;

  if (!workspace.dt_chunk || !workspace.dA_cumsum ||
      !workspace.state_scratch || !workspace.cb_chunk) {
    std::cerr << "FAIL: workspace allocation\n";
    return false;
  }

  std::cout << "Running MambaChunkedScanPrefillBf16 with vLLM inputs...\n";

  // The dump env vars are already set (NEMOTRON_DUMP_CHUNKED_SCAN_LAYER=0, _ROOT=...)
  // so the function will dump intermediates automatically.
  const bool ok = MambaChunkedScanPrefillBf16(
      *dev_projected, *dev_conv_output,
      kI, kConvDim, kH, kP, kN, kG, kQ,
      0,  // ssm_state_offset_elems
      *dev_a, *dev_d, *dev_dt_bias,
      dev_ssm_state.get(), dev_y_output.get(),
      &workspace, nullptr);

  if (!ok) {
    std::cerr << "FAIL: MambaChunkedScanPrefillBf16 returned false\n";
    return false;
  }
  cudaDeviceSynchronize();

  // Download outputs and compare against vLLM
  std::cout << "\n=== Kernel Output vs vLLM Reference ===\n";

  // Load vLLM reference outputs
  const auto vllm_dt_chunk = read_fp32(vllm_dir / "dt_chunk_fp32.bin");
  const auto vllm_dA_cumsum = read_fp32(vllm_dir / "dA_cumsum_fp32.bin");
  const auto vllm_chunk_delta = read_fp32(vllm_dir / "chunk_delta_fp32.bin");
  const auto vllm_cb = read_fp32(vllm_dir / "cb_chunk_fp32.bin");
  const auto vllm_y = read_fp32(vllm_dir / "y_output_fp32.bin");
  const auto vllm_state_out = read_fp32(vllm_dir / "ssm_state_out_fp32.bin");

  // Download our outputs
  auto our_dt_chunk = download_fp32(*workspace.dt_chunk);
  auto our_dA_cumsum = download_fp32(*workspace.dA_cumsum);
  // chunk_delta was overwritten by state_passing, but we can get ssm_state_out
  auto our_cb = download_fp32(*workspace.cb_chunk);
  auto our_y = download_bf16_as_fp32(*dev_y_output);
  auto our_state_out = download_fp32(*dev_ssm_state);

  // vLLM dt_chunk/dA_cumsum shape is [H, C, Q], ours is [C, H, Q]
  // Transpose vLLM to [C, H, Q]
  std::vector<float> vllm_dt_transposed(vllm_dt_chunk.size());
  std::vector<float> vllm_dA_transposed(vllm_dA_cumsum.size());
  for (std::size_t h = 0; h < kH; ++h) {
    for (std::size_t c = 0; c < C; ++c) {
      for (std::size_t q = 0; q < kQ; ++q) {
        vllm_dt_transposed[c * kH * kQ + h * kQ + q] = vllm_dt_chunk[h * C * kQ + c * kQ + q];
        vllm_dA_transposed[c * kH * kQ + h * kQ + q] = vllm_dA_cumsum[h * C * kQ + c * kQ + q];
      }
    }
  }

  // Only compare valid positions (T=40 out of Q=128)
  // For dt_chunk and dA_cumsum, positions 40-127 should be zero-padded
  auto trim = [](const std::vector<float>& v, std::size_t count) {
    return std::vector<float>(v.begin(), v.begin() + std::min(count, v.size()));
  };

  struct Result {
    const char* name;
    float mad;
    float rl2;
  };
  std::vector<Result> results;

  auto report = [&](const char* name, const std::vector<float>& ours, const std::vector<float>& vllm) {
    const std::size_t n = std::min(ours.size(), vllm.size());
    auto a = trim(ours, n);
    auto b = trim(vllm, n);
    float d = max_abs_diff(a, b);
    float r = rel_l2(a, b);
    const char* status = d <= 1e-5 ? "EXACT" : (d <= 1e-3 ? "CLOSE" : "DIFFER");
    std::cout << "  " << status << "  " << name
              << ": mad=" << d << " rl2=" << r
              << " (n=" << n << ")\n";
    results.push_back({name, d, r});
  };

  report("dt_chunk", our_dt_chunk, vllm_dt_transposed);
  report("dA_cumsum", our_dA_cumsum, vllm_dA_transposed);
  report("CB", our_cb, vllm_cb);
  // y_output: ours is [T, I], vllm is [T, H, P] — same flattened
  report("y_output", our_y, vllm_y);
  report("ssm_state_out", our_state_out, vllm_state_out);

  std::cout << "\n";
  bool all_close = true;
  for (const auto& r : results) {
    if (r.mad > 1e-3) {
      std::cout << "KERNEL MISMATCH: " << r.name << " mad=" << r.mad << "\n";
      all_close = false;
    }
  }
  if (all_close) {
    std::cout << "ALL KERNELS MATCH vLLM (mad <= 1e-3 on identical inputs)\n";
  }

  return all_close;
}

}  // namespace

int main() {
  return run_verify() ? 0 : 1;
}
