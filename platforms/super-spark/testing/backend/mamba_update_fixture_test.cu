#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBatchSize = 1;
constexpr std::size_t kNumHeads = 128;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kStateSize = 128;

struct DeviceBuffer {
  void* data = nullptr;

  ~DeviceBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

bool CheckCuda(cudaError_t status, const char* where) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

std::vector<float> LoadFloatTensor(const std::filesystem::path& path, std::size_t expected_count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::vector<float> values(expected_count, 0.0f);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));
  if (!input || input.gcount() != static_cast<std::streamsize>(expected_count * sizeof(float))) {
    throw std::runtime_error("unexpected size for " + path.string());
  }
  return values;
}

bool Upload(const std::vector<float>& host, DeviceBuffer* device) {
  if (device == nullptr) {
    return false;
  }
  if (!CheckCuda(cudaMalloc(&device->data, host.size() * sizeof(float)), "cudaMalloc")) {
    return false;
  }
  return CheckCuda(
      cudaMemcpy(device->data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
      "cudaMemcpy host to device");
}

__global__ void MambaFixtureStateUpdateKernel(
    const float* ssm_state,
    const float* hidden,
    const float* dt,
    const float* A,
    const float* B,
    float* next_state,
    std::size_t total_state_count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < total_state_count; i += stride) {
    const std::size_t state_index = i % kStateSize;
    const std::size_t hd_index = i / kStateSize;
    const std::size_t hidden_index = hd_index;
    const std::size_t b_index = (hd_index / kHeadDim) * kStateSize + state_index;
    next_state[i] = ssm_state[i] * expf(dt[hidden_index] * A[i]) + dt[hidden_index] * B[b_index] * hidden[hidden_index];
  }
}

__global__ void MambaFixtureOutputKernel(
    const float* next_state,
    const float* hidden,
    const float* C,
    const float* D,
    float* output,
    std::size_t total_hidden_count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < total_hidden_count; i += stride) {
    const std::size_t head_index = i / kHeadDim;
    float accum = 0.0f;
    const std::size_t state_base = i * kStateSize;
    const std::size_t c_base = head_index * kStateSize;
    for (std::size_t s = 0; s < kStateSize; ++s) {
      accum += next_state[state_base + s] * C[c_base + s];
    }
    output[i] = accum + hidden[i] * D[i];
  }
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(lhs[i] - rhs[i]));
  }
  return max_abs;
}

int RunFixtureTest() {
  const std::filesystem::path root(NEMOTRON_MAMBA_ORACLE_FIXTURE_ROOT);
  const std::size_t hidden_count = kBatchSize * kNumHeads * kHeadDim;
  const std::size_t state_count = hidden_count * kStateSize;
  const std::size_t bc_count = kBatchSize * kNumHeads * kStateSize;

  const auto ssm_state = LoadFloatTensor(root / "ssm_state_fp32.bin", state_count);
  const auto hidden = LoadFloatTensor(root / "hidden_fp32.bin", hidden_count);
  const auto dt = LoadFloatTensor(root / "dt_fp32.bin", hidden_count);
  const auto A = LoadFloatTensor(root / "A_fp32.bin", state_count);
  const auto B = LoadFloatTensor(root / "B_fp32.bin", bc_count);
  const auto C = LoadFloatTensor(root / "C_fp32.bin", bc_count);
  const auto D = LoadFloatTensor(root / "D_fp32.bin", hidden_count);
  const auto expected_next_state = LoadFloatTensor(root / "expected_next_state_fp32.bin", state_count);
  const auto expected_output = LoadFloatTensor(root / "expected_output_fp32.bin", hidden_count);

  DeviceBuffer ssm_state_d;
  DeviceBuffer hidden_d;
  DeviceBuffer dt_d;
  DeviceBuffer A_d;
  DeviceBuffer B_d;
  DeviceBuffer C_d;
  DeviceBuffer D_d;
  DeviceBuffer next_state_d;
  DeviceBuffer output_d;

  if (!Upload(ssm_state, &ssm_state_d) || !Upload(hidden, &hidden_d) || !Upload(dt, &dt_d) ||
      !Upload(A, &A_d) || !Upload(B, &B_d) || !Upload(C, &C_d) || !Upload(D, &D_d)) {
    return 1;
  }
  if (!CheckCuda(cudaMalloc(&next_state_d.data, state_count * sizeof(float)), "cudaMalloc next_state") ||
      !CheckCuda(cudaMalloc(&output_d.data, hidden_count * sizeof(float)), "cudaMalloc output")) {
    return 1;
  }

  const int block_size = 256;
  const int state_grid = static_cast<int>((state_count + block_size - 1) / block_size);
  const int hidden_grid = static_cast<int>((hidden_count + block_size - 1) / block_size);
  MambaFixtureStateUpdateKernel<<<state_grid, block_size>>>(
      reinterpret_cast<const float*>(ssm_state_d.data),
      reinterpret_cast<const float*>(hidden_d.data),
      reinterpret_cast<const float*>(dt_d.data),
      reinterpret_cast<const float*>(A_d.data),
      reinterpret_cast<const float*>(B_d.data),
      reinterpret_cast<float*>(next_state_d.data),
      state_count);
  if (!CheckCuda(cudaGetLastError(), "MambaFixtureStateUpdateKernel launch") ||
      !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize state")) {
    return 1;
  }

  MambaFixtureOutputKernel<<<hidden_grid, block_size>>>(
      reinterpret_cast<const float*>(next_state_d.data),
      reinterpret_cast<const float*>(hidden_d.data),
      reinterpret_cast<const float*>(C_d.data),
      reinterpret_cast<const float*>(D_d.data),
      reinterpret_cast<float*>(output_d.data),
      hidden_count);
  if (!CheckCuda(cudaGetLastError(), "MambaFixtureOutputKernel launch") ||
      !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize output")) {
    return 1;
  }

  std::vector<float> next_state(state_count, 0.0f);
  std::vector<float> output(hidden_count, 0.0f);
  if (!CheckCuda(
          cudaMemcpy(next_state.data(), next_state_d.data, state_count * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy next_state to host") ||
      !CheckCuda(
          cudaMemcpy(output.data(), output_d.data, hidden_count * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy output to host")) {
    return 1;
  }

  const float next_state_diff = MaxAbsDiff(next_state, expected_next_state);
  const float output_diff = MaxAbsDiff(output, expected_output);
  if (next_state_diff > 1.0e-5f || output_diff > 1.0e-5f) {
    std::cerr << "mamba fixture mismatch: next_state_diff=" << next_state_diff
              << " output_diff=" << output_diff << "\n";
    return 1;
  }

  std::cout << "mamba fixture test passed: next_state_diff=" << next_state_diff
            << " output_diff=" << output_diff << "\n";
  return 0;
}

}  // namespace

int main() {
  try {
    return RunFixtureTest();
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
