#include "nemotron/expert_routing_device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using nemotron::DeviceExpertRouting;
using nemotron::RunDeviceExpertRouting;
using nemotron::kMaxDeviceExpertRoutingExperts;

template <typename T>
class DeviceBuffer {
 public:
  static std::unique_ptr<DeviceBuffer> Create(std::size_t count) {
    if (!HasCudaDevice() || count == 0) {
      return nullptr;
    }

    T* data = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)) !=
        cudaSuccess) {
      return nullptr;
    }
    return std::unique_ptr<DeviceBuffer>(new DeviceBuffer(data, count));
  }

  DeviceBuffer(DeviceBuffer&&) noexcept = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const {
    return data_;
  }

  std::size_t count() const {
    return count_;
  }

  bool CopyFromHost(const std::vector<T>& values) const {
    return values.size() == count_ &&
           cudaMemcpy(
               data_,
               values.data(),
               count_ * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess;
  }

 private:
  DeviceBuffer(T* data, std::size_t count) : data_(data), count_(count) {}

  static bool HasCudaDevice() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

template <typename T>
bool CopyDeviceValues(
    const T* device_data,
    std::size_t count,
    std::vector<T>* host_values) {
  if (device_data == nullptr || host_values == nullptr) {
    return false;
  }
  host_values->assign(count, T{});
  return cudaMemcpy(
             host_values->data(),
             device_data,
             count * sizeof(T),
             cudaMemcpyDeviceToHost) == cudaSuccess;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

bool ExpectEqualVector(
    const std::vector<int>& actual,
    const std::vector<int>& expected,
    const std::string& label) {
  if (actual.size() != expected.size()) {
    std::cerr << "FAIL: " << label << " size mismatch: actual="
              << actual.size() << " expected=" << expected.size() << "\n";
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "FAIL: " << label << " mismatch at " << i
                << ": actual=" << actual[i]
                << " expected=" << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

bool ExpectEqualPrefix(
    const std::vector<int>& actual,
    const std::vector<int>& expected,
    const std::string& label) {
  if (actual.size() < expected.size()) {
    std::cerr << "FAIL: " << label << " size mismatch: actual="
              << actual.size() << " expected>=" << expected.size() << "\n";
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "FAIL: " << label << " mismatch at " << i
                << ": actual=" << actual[i]
                << " expected=" << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

void FillSelectionPattern(
    std::size_t n_experts,
    std::size_t token_count,
    std::size_t top_k,
    int variant,
    std::vector<int>* selected_indices,
    std::vector<float>* selected_weights) {
  const std::size_t selection_count = token_count * top_k;
  selected_indices->assign(selection_count, 0);
  selected_weights->assign(selection_count, 0.0f);

  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    for (std::size_t slot = 0; slot < top_k; ++slot) {
      const std::size_t selection_index = token_index * top_k + slot;
      std::size_t expert_index =
          ((token_index * static_cast<std::size_t>(17 + variant)) +
           (slot * static_cast<std::size_t>(29 + (variant * 3))) +
           (static_cast<std::size_t>(variant) * 7)) %
          n_experts;
      if (((token_index + static_cast<std::size_t>(variant)) % 5) == 0) {
        expert_index = (slot * 3 + static_cast<std::size_t>(variant)) % n_experts;
      }
      if (((slot + static_cast<std::size_t>(variant)) % 4) == 0) {
        expert_index = (token_index * 7 + 1) % n_experts;
      }
      if (((token_index + slot + static_cast<std::size_t>(variant)) % 9) == 0) {
        expert_index = 0;
      }

      (*selected_indices)[selection_index] = static_cast<int>(expert_index);

      const std::size_t raw =
          ((selection_index + 1) * static_cast<std::size_t>(13 + variant * 5) +
           7) %
          113;
      (*selected_weights)[selection_index] =
          0.05f + static_cast<float>(raw) / 127.0f;
    }
  }
}

void BuildCpuReference(
    std::size_t n_experts,
    std::size_t token_count,
    std::size_t top_k,
    const std::vector<int>& selected_indices,
    const std::vector<float>& selected_weights,
    std::vector<int>* expert_offsets,
    std::vector<int>* sorted_token_indices,
    std::vector<float>* sorted_token_weights,
    std::vector<int>* active_expert_ids) {
  const std::size_t selection_count = token_count * top_k;
  std::vector<int> expert_counts(n_experts, 0);
  for (int expert_index : selected_indices) {
    ++expert_counts[static_cast<std::size_t>(expert_index)];
  }

  expert_offsets->assign(n_experts + 1, 0);
  for (std::size_t expert_index = 0; expert_index < n_experts; ++expert_index) {
    (*expert_offsets)[expert_index + 1] =
        (*expert_offsets)[expert_index] + expert_counts[expert_index];
  }

  sorted_token_indices->assign(selection_count, 0);
  sorted_token_weights->assign(selection_count, 0.0f);
  std::vector<int> write_offsets = *expert_offsets;
  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    for (std::size_t slot = 0; slot < top_k; ++slot) {
      const std::size_t selection_index = token_index * top_k + slot;
      const int expert_index = selected_indices[selection_index];
      const int write_index = write_offsets[static_cast<std::size_t>(expert_index)]++;
      (*sorted_token_indices)[static_cast<std::size_t>(write_index)] =
          static_cast<int>(token_index);
      (*sorted_token_weights)[static_cast<std::size_t>(write_index)] =
          selected_weights[selection_index];
    }
  }

  active_expert_ids->clear();
  for (std::size_t expert_index = 0; expert_index < n_experts; ++expert_index) {
    if (expert_counts[expert_index] > 0) {
      active_expert_ids->push_back(static_cast<int>(expert_index));
    }
  }
}

bool RunParityCase(
    std::size_t n_experts,
    std::size_t token_count,
    std::size_t top_k,
    const std::vector<int>& variants) {
  const std::size_t selection_count = token_count * top_k;
  auto selected_indices_device = DeviceBuffer<int>::Create(selection_count);
  auto selected_weights_device = DeviceBuffer<float>::Create(selection_count);
  auto routing = DeviceExpertRouting::Create(n_experts, selection_count);

  if (!Expect(
          selected_indices_device != nullptr &&
              selected_weights_device != nullptr &&
              routing != nullptr &&
              routing->valid(),
          "device allocations should succeed")) {
    return false;
  }

  for (int variant : variants) {
    std::vector<int> selected_indices_host;
    std::vector<float> selected_weights_host;
    FillSelectionPattern(
        n_experts,
        token_count,
        top_k,
        variant,
        &selected_indices_host,
        &selected_weights_host);

    if (!Expect(
            selected_indices_device->CopyFromHost(selected_indices_host),
            "selected indices upload should succeed") ||
        !Expect(
            selected_weights_device->CopyFromHost(selected_weights_host),
            "selected weights upload should succeed") ||
        !Expect(
            RunDeviceExpertRouting(
                selected_indices_device->data(),
                selected_weights_device->data(),
                token_count,
                top_k,
                routing.get()),
            "RunDeviceExpertRouting should succeed") ||
        !Expect(
            cudaDeviceSynchronize() == cudaSuccess,
            "routing kernels should complete")) {
      return false;
    }

    std::vector<int> actual_expert_offsets;
    std::vector<int> actual_sorted_token_indices;
    std::vector<float> actual_sorted_token_weights;
    std::vector<int> actual_active_expert_count;
    std::vector<int> actual_active_expert_ids;
    if (!Expect(
            CopyDeviceValues(
                routing->expert_offsets(),
                n_experts + 1,
                &actual_expert_offsets),
            "expert offsets download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->sorted_token_indices(),
                selection_count,
                &actual_sorted_token_indices),
            "sorted token indices download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->sorted_token_weights(),
                selection_count,
                &actual_sorted_token_weights),
            "sorted token weights download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->active_expert_count(),
                1,
                &actual_active_expert_count),
            "active expert count download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->active_expert_ids(),
                n_experts,
                &actual_active_expert_ids),
            "active expert ids download should succeed")) {
      return false;
    }

    std::vector<int> expected_expert_offsets;
    std::vector<int> expected_sorted_token_indices;
    std::vector<float> expected_sorted_token_weights;
    std::vector<int> expected_active_expert_ids;
    BuildCpuReference(
        n_experts,
        token_count,
        top_k,
        selected_indices_host,
        selected_weights_host,
        &expected_expert_offsets,
        &expected_sorted_token_indices,
        &expected_sorted_token_weights,
        &expected_active_expert_ids);

    if (!ExpectEqualVector(
            actual_expert_offsets,
            expected_expert_offsets,
            "expert_offsets") ||
        !ExpectEqualVector(
            actual_sorted_token_indices,
            expected_sorted_token_indices,
            "sorted_token_indices") ||
        !Expect(
            MaxAbsDiff(actual_sorted_token_weights, expected_sorted_token_weights) ==
                0.0f,
            "sorted_token_weights should match exactly") ||
        !Expect(
            actual_active_expert_count.size() == 1,
            "active expert count should be scalar") ||
        !Expect(
            actual_active_expert_count[0] ==
                static_cast<int>(expected_active_expert_ids.size()),
            "active expert count should match reference") ||
        !ExpectEqualPrefix(
            actual_active_expert_ids,
            expected_active_expert_ids,
            "active_expert_ids")) {
      return false;
    }
  }

  return true;
}

bool RunCreateValidationCase() {
  return Expect(
      DeviceExpertRouting::Create(
          kMaxDeviceExpertRoutingExperts + 1,
          8) == nullptr,
      "routing buffers should reject unsupported expert counts");
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::cout << "expert_routing_device_test: SKIP (no CUDA device)\n";
    return 0;
  }

  if (!RunCreateValidationCase() ||
      !RunParityCase(128, 32, 6, {0, 1}) ||
      !RunParityCase(17, 5, 3, {2})) {
    return 1;
  }

  std::cout << "expert_routing_device_test: PASS\n";
  return 0;
}
