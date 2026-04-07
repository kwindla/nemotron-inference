#include "nemotron/device_tensor.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool nearly_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, float tol) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = lhs[i] - rhs[i];
    if (diff > tol || diff < -tol) {
      return false;
    }
  }
  return true;
}

bool test_device_tensor_round_trip_and_zero_fill() {
  auto tensor = nemotron::DeviceTensorFp32::Create({2, 3});
  if (!tensor || !tensor->valid()) {
    std::cout << "device_tensor_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<float> host = {1.0f, -2.0f, 3.5f, 4.0f, 0.25f, -6.0f};
  if (!expect(tensor->shape().size() == 2, "shape rank should be preserved")) {
    return false;
  }
  if (!expect(tensor->shape()[0] == 2 && tensor->shape()[1] == 3, "shape dimensions should be preserved")) {
    return false;
  }
  if (!expect(tensor->numel() == 6, "numel should match the shape")) {
    return false;
  }
  if (!expect(tensor->bytes() == host.size() * sizeof(float), "byte size should match fp32 storage")) {
    return false;
  }
  if (!expect(tensor->CopyFromHost(host.data(), host.size()), "copy from host should succeed")) {
    return false;
  }

  std::vector<float> round_trip(host.size(), 0.0f);
  if (!expect(tensor->CopyToHost(round_trip.data(), round_trip.size()), "copy to host should succeed")) {
    return false;
  }
  if (!expect(nearly_equal(host, round_trip, 1e-6f), "round-tripped tensor should match source values")) {
    return false;
  }

  if (!expect(tensor->FillZero(), "zero fill should succeed")) {
    return false;
  }
  std::vector<float> zeros(host.size(), 1.0f);
  if (!expect(tensor->CopyToHost(zeros.data(), zeros.size()), "copy after zero fill should succeed")) {
    return false;
  }
  return expect(nearly_equal(zeros, std::vector<float>(host.size(), 0.0f), 1e-6f),
                "zero-filled tensor should read back as zeros");
}

bool test_device_tensor_view_aliases_existing_storage() {
  auto owner = nemotron::DeviceTensorFp32::Create({2, 3});
  if (!owner || !owner->valid()) {
    std::cout << "device_tensor_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<float> host = {0.5f, -1.0f, 1.5f, 2.0f, -2.5f, 3.0f};
  if (!expect(owner->CopyFromHost(host.data(), host.size()), "owner tensor upload should succeed")) {
    return false;
  }

  auto view = nemotron::DeviceTensorFp32::CreateView({2, 3}, owner->data());
  if (!expect(view != nullptr && view->valid(), "tensor view should be valid")) {
    return false;
  }

  std::vector<float> round_trip(host.size(), 0.0f);
  if (!expect(view->CopyToHost(round_trip.data(), round_trip.size()), "view copy to host should succeed")) {
    return false;
  }
  return expect(nearly_equal(host, round_trip, 1e-6f),
                "tensor view should alias the original device storage");
}

bool test_device_tensor_fp8_round_trip_and_view() {
  auto owner = nemotron::DeviceTensorFp8E4M3::Create({2, 4});
  if (!owner || !owner->valid()) {
    std::cout << "device_tensor_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<std::uint8_t> host = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  if (!expect(owner->CopyFromHost(host.data(), host.size()), "fp8 owner upload should succeed")) {
    return false;
  }

  auto view = nemotron::DeviceTensorFp8E4M3::CreateView({2, 4}, owner->data());
  if (!expect(view != nullptr && view->valid(), "fp8 tensor view should be valid")) {
    return false;
  }

  std::vector<std::uint8_t> round_trip(host.size(), 0);
  if (!expect(view->CopyToHost(round_trip.data(), round_trip.size()), "fp8 view copy should succeed")) {
    return false;
  }
  if (!expect(round_trip == host, "fp8 tensor view should alias original storage")) {
    return false;
  }

  if (!expect(owner->FillZero(), "fp8 zero fill should succeed")) {
    return false;
  }
  std::vector<std::uint8_t> zeros(host.size(), 0xFF);
  if (!expect(owner->CopyToHost(zeros.data(), zeros.size()), "fp8 copy after zero fill should succeed")) {
    return false;
  }
  return expect(zeros == std::vector<std::uint8_t>(host.size(), 0),
                "fp8 zero-filled tensor should read back as zeros");
}

}  // namespace

int main() {
  if (!test_device_tensor_round_trip_and_zero_fill() ||
      !test_device_tensor_view_aliases_existing_storage() ||
      !test_device_tensor_fp8_round_trip_and_view()) {
    return 1;
  }
  std::cout << "device_tensor_test: PASS\n";
  return 0;
}
