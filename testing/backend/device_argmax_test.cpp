#include "nemotron/device_argmax.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using nemotron::DeviceArgmax;
using nemotron::DeviceTensorFp32;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool CheckCuda(cudaError_t status, const char* where) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

struct DeviceTokenBuffer {
  ~DeviceTokenBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

  std::int32_t* data = nullptr;
};

bool RunArgmaxCase(
    const std::vector<float>& logits,
    std::int32_t expected_token_id,
    const std::string& case_name) {
  auto logits_row = DeviceTensorFp32::Create({1, logits.size()});
  if (!expect(logits_row != nullptr && logits_row->valid(), case_name + ": logits tensor should create")) {
    return false;
  }
  if (!expect(logits_row->CopyFromHost(logits.data(), logits.size()), case_name + ": logits upload should succeed")) {
    return false;
  }

  DeviceTokenBuffer token_buffer;
  if (!expect(CheckCuda(cudaMalloc(reinterpret_cast<void**>(&token_buffer.data), sizeof(std::int32_t)), "cudaMalloc token"),
              case_name + ": token buffer allocation should succeed")) {
    return false;
  }

  if (!expect(DeviceArgmax(*logits_row, token_buffer.data), case_name + ": DeviceArgmax should succeed")) {
    return false;
  }

  std::int32_t actual_token_id = -1;
  if (!expect(CheckCuda(
                  cudaMemcpy(&actual_token_id, token_buffer.data, sizeof(actual_token_id), cudaMemcpyDeviceToHost),
                  "cudaMemcpy token"),
              case_name + ": token download should succeed")) {
    return false;
  }

  return expect(
      actual_token_id == expected_token_id,
      case_name + ": expected token " + std::to_string(expected_token_id) + ", got " + std::to_string(actual_token_id));
}

bool test_device_argmax_cases() {
  auto probe = DeviceTensorFp32::Create({1, 1});
  if (!probe || !probe->valid()) {
    std::cout << "device_argmax_test: SKIP (no CUDA device available)\n";
    return true;
  }

  if (!RunArgmaxCase(std::vector<float>(32, 1.25f), 0, "uniform")) {
    return false;
  }

  if (!RunArgmaxCase({-4.0f, -3.0f, 2.0f, 5.0f, 9.0f, 1.0f, 8.0f}, 4, "single_max")) {
    return false;
  }

  if (!RunArgmaxCase({-8.0f, -7.0f, -6.0f, -5.0f, -4.0f, 10.0f}, 5, "last_position")) {
    return false;
  }

  constexpr std::size_t kLargeVocabSize = 131072;
  constexpr std::int32_t kLargeExpectedIndex = 123456;
  std::vector<float> large_logits(kLargeVocabSize, -1000.0f);
  for (std::size_t i = 0; i < large_logits.size(); ++i) {
    large_logits[i] += static_cast<float>(i % 31);
  }
  large_logits[static_cast<std::size_t>(kLargeExpectedIndex)] = 42.0f;
  if (!RunArgmaxCase(large_logits, kLargeExpectedIndex, "large_vocab")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!test_device_argmax_cases()) {
    return 1;
  }
  std::cout << "device_argmax_test: PASS\n";
  return 0;
}
