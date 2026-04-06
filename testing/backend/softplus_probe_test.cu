// Compare a few CUDA softplus formulations against the saved vLLM dt oracle.
//
// Usage:
//   NEMOTRON_VLLM_DUMP_DIR=<path-to-vllm-dumps> \
//   build/testing/softplus_probe_test

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kH = 128;
constexpr std::size_t kQ = 128;
constexpr int kVariantCount = 7;

enum Variant : int {
  kStd = 0,
  kLog1pExp = 1,
  kStable = 2,
  kFastFast = 3,
  kFastExp = 4,
  kFastLog = 5,
  kExp2Log2 = 6,
};

__device__ float softplus_variant(float x, int variant) {
  if (x > 20.0f) {
    return x;
  }
  switch (variant) {
    case kStd:
      return logf(expf(x) + 1.0f);
    case kLog1pExp:
      return log1pf(expf(x));
    case kStable:
      return x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
    case kFastFast:
      return __logf(__expf(x) + 1.0f);
    case kFastExp:
      return logf(__expf(x) + 1.0f);
    case kFastLog:
      return __logf(expf(x) + 1.0f);
    case kExp2Log2: {
      constexpr float kLog2E = 1.4426950408889634f;
      constexpr float kLn2 = 0.6931471805599453f;
      return log2f(exp2f(x * kLog2E) + 1.0f) * kLn2;
    }
    default:
      return logf(expf(x) + 1.0f);
  }
}

__global__ void SoftplusVariantsKernel(const float* input, float* output, std::size_t n) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float x = input[idx];
  for (int variant = 0; variant < kVariantCount; ++variant) {
    output[static_cast<std::size_t>(variant) * n + idx] = softplus_variant(x, variant);
  }
}

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

const char* variant_name(int variant) {
  switch (variant) {
    case kStd:
      return "logf(expf(x)+1)";
    case kLog1pExp:
      return "log1pf(expf(x))";
    case kStable:
      return "stable-log1p";
    case kFastFast:
      return "__logf(__expf(x)+1)";
    case kFastExp:
      return "logf(__expf(x)+1)";
    case kFastLog:
      return "__logf(expf(x)+1)";
    case kExp2Log2:
      return "log2f(exp2f())*ln2";
    default:
      return "unknown";
  }
}

bool run_probe() {
  const char* vllm_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  if (!vllm_dir_env) {
    std::cout << "softplus_probe_test: SKIP (NEMOTRON_VLLM_DUMP_DIR not set)\n";
    return true;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "softplus_probe_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path vllm_dir(vllm_dir_env);
  const auto dt_pre = read_fp32(vllm_dir / "dt_pre_fp32.bin");
  const auto dt_bias = read_fp32(vllm_dir / "dt_bias_fp32.bin");
  const auto dt_chunk = read_fp32(vllm_dir / "dt_chunk_fp32.bin");
  if (dt_pre.empty() || dt_bias.size() != kH || dt_chunk.empty() || dt_pre.size() % kH != 0) {
    std::cerr << "FAIL: unexpected input sizes\n";
    return false;
  }

  const std::size_t token_count = dt_pre.size() / kH;
  const std::size_t count = token_count * kH;
  const std::size_t chunk_count = (token_count + kQ - 1) / kQ;
  if (dt_chunk.size() != kH * chunk_count * kQ) {
    std::cerr << "FAIL: unexpected dt_chunk size\n";
    return false;
  }

  std::vector<float> input(count);
  std::vector<float> oracle(count);
  for (std::size_t token = 0; token < token_count; ++token) {
    const std::size_t chunk = token / kQ;
    const std::size_t q = token % kQ;
    for (std::size_t head = 0; head < kH; ++head) {
      const std::size_t idx = token * kH + head;
      input[idx] = dt_pre[idx] + dt_bias[head];
      oracle[idx] = dt_chunk[head * chunk_count * kQ + chunk * kQ + q];
    }
  }

  float* dev_input = nullptr;
  float* dev_output = nullptr;
  cudaMalloc(&dev_input, count * sizeof(float));
  cudaMalloc(&dev_output, static_cast<std::size_t>(kVariantCount) * count * sizeof(float));
  cudaMemcpy(dev_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice);

  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  SoftplusVariantsKernel<<<grid, block>>>(dev_input, dev_output, count);
  cudaDeviceSynchronize();

  std::vector<float> host_output(static_cast<std::size_t>(kVariantCount) * count);
  cudaMemcpy(host_output.data(), dev_output,
             host_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dev_input);
  cudaFree(dev_output);

  std::cout << "softplus_probe_test: token_count=" << token_count << " variants=" << kVariantCount
            << "\n";
  float best_mad = INFINITY;
  int best_variant = -1;
  for (int variant = 0; variant < kVariantCount; ++variant) {
    const float* actual = host_output.data() + static_cast<std::size_t>(variant) * count;
    float mad = 0.0f;
    std::size_t mad_index = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const float d = std::fabs(actual[i] - oracle[i]);
      if (d > mad) {
        mad = d;
        mad_index = i;
      }
    }
    const std::size_t token = mad_index / kH;
    const std::size_t head = mad_index % kH;
    std::cout << "  " << variant_name(variant) << ": mad=" << mad << " token=" << token
              << " head=" << head << " actual=" << actual[mad_index]
              << " oracle=" << oracle[mad_index] << " input=" << input[mad_index] << "\n";
    if (mad < best_mad) {
      best_mad = mad;
      best_variant = variant;
    }
  }

  std::cout << "  best=" << variant_name(best_variant) << " mad=" << best_mad << "\n";
  return true;
}

}  // namespace

int main() {
  return run_probe() ? 0 : 1;
}
