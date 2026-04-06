#include "nemotron/expert_routing_device.h"
#include "nemotron/moe_launch_plan_device.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using nemotron::BuildDeviceMoeLaunchPlan;
using nemotron::DeviceExpertRouting;
using nemotron::DeviceMoeLaunchPlan;

template <typename T>
class DeviceBuffer {
 public:
  static std::unique_ptr<DeviceBuffer> Create(std::size_t count) {
    if (!HasCudaDevice() || count == 0) {
      return nullptr;
    }

    T* data = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)) != cudaSuccess) {
      return nullptr;
    }
    return std::unique_ptr<DeviceBuffer>(new DeviceBuffer(data, count));
  }

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
bool CopyDeviceValues(const T* device_data, std::size_t count, std::vector<T>* host_values) {
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

bool ExpectEqualVector(
    const std::vector<int>& actual,
    const std::vector<int>& expected,
    const std::string& label) {
  if (actual.size() != expected.size()) {
    std::cerr << "FAIL: " << label << " size mismatch\n";
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

bool ExpectVectorPrefix(
    const std::vector<int>& actual,
    const std::vector<int>& expected_prefix,
    const std::string& label) {
  if (actual.size() < expected_prefix.size()) {
    std::cerr << "FAIL: " << label << " prefix size mismatch\n";
    return false;
  }
  for (std::size_t i = 0; i < expected_prefix.size(); ++i) {
    if (actual[i] != expected_prefix[i]) {
      std::cerr << "FAIL: " << label << " mismatch at " << i
                << ": actual=" << actual[i]
                << " expected=" << expected_prefix[i] << "\n";
      return false;
    }
  }
  return true;
}

bool RunLaunchPlanCase(
    std::size_t n_experts,
    std::size_t token_count,
    std::size_t top_k,
    const std::vector<int>& selected_indices_host,
    const std::vector<float>& selected_weights_host,
    const std::vector<int>& expected_expert_ids,
    const std::vector<int>& expected_row_starts,
    const std::vector<int>& expected_valid_rows,
    const std::vector<int>& expected_m_limits,
    const std::vector<int>& expected_permuted_token_indices,
    const std::vector<int>& expected_sorted_to_permuted_indices) {
  const std::size_t selection_count = token_count * top_k;
  auto selected_indices_device = DeviceBuffer<int>::Create(selection_count);
  auto selected_weights_device = DeviceBuffer<float>::Create(selection_count);
  auto routing = DeviceExpertRouting::Create(n_experts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(n_experts, selection_count);
  if (!Expect(
          selected_indices_device != nullptr &&
              selected_weights_device != nullptr &&
              routing != nullptr &&
              routing->valid() &&
              launch_plan != nullptr &&
              launch_plan->valid(),
          "launch plan device allocations should succeed") ||
      !Expect(
          selected_indices_device->CopyFromHost(selected_indices_host),
          "selected indices upload should succeed") ||
      !Expect(
          selected_weights_device->CopyFromHost(selected_weights_host),
          "selected weights upload should succeed") ||
      !Expect(
          nemotron::RunDeviceExpertRouting(
              selected_indices_device->data(),
              selected_weights_device->data(),
              token_count,
              top_k,
              routing.get()),
          "RunDeviceExpertRouting should succeed") ||
      !Expect(
          BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()),
          "BuildDeviceMoeLaunchPlan should succeed") ||
      !Expect(
          cudaDeviceSynchronize() == cudaSuccess,
          "launch plan kernels should synchronize")) {
    return false;
  }

  std::vector<int> cta_count;
  std::vector<int> total_padded_rows;
  std::vector<int> cta_expert_ids;
  std::vector<int> cta_row_starts;
  std::vector<int> cta_valid_rows;
  std::vector<int> cta_m_limits;
  std::vector<int> permuted_token_indices;
  std::vector<int> sorted_to_permuted_indices;
  if (!Expect(
          CopyDeviceValues(launch_plan->cta_count(), 1, &cta_count),
          "cta_count should download") ||
      !Expect(
          CopyDeviceValues(launch_plan->total_padded_rows(), 1, &total_padded_rows),
          "total_padded_rows should download") ||
      !Expect(
          CopyDeviceValues(
              launch_plan->cta_expert_ids(),
              launch_plan->cta_capacity(),
              &cta_expert_ids),
          "cta_expert_ids should download") ||
      !Expect(
          CopyDeviceValues(
              launch_plan->cta_row_starts(),
              launch_plan->cta_capacity(),
              &cta_row_starts),
          "cta_row_starts should download") ||
      !Expect(
          CopyDeviceValues(
              launch_plan->cta_valid_rows(),
              launch_plan->cta_capacity(),
              &cta_valid_rows),
          "cta_valid_rows should download") ||
      !Expect(
          CopyDeviceValues(
              launch_plan->cta_m_limits(),
              launch_plan->cta_capacity(),
              &cta_m_limits),
          "cta_m_limits should download") ||
      !Expect(
          CopyDeviceValues(
              launch_plan->permuted_token_indices(),
              launch_plan->cta_capacity() * nemotron::kMoeLaunchPlanTokenTile,
              &permuted_token_indices),
          "permuted_token_indices should download") ||
      !Expect(
          CopyDeviceValues(
              launch_plan->sorted_to_permuted_indices(),
              selection_count,
              &sorted_to_permuted_indices),
          "sorted_to_permuted_indices should download")) {
    return false;
  }

  return Expect(
             cta_count.size() == 1 &&
                 cta_count[0] == static_cast<int>(expected_expert_ids.size()),
             "cta_count should match expected") &&
         Expect(
             total_padded_rows.size() == 1 &&
                 total_padded_rows[0] ==
                     static_cast<int>(expected_expert_ids.size() * nemotron::kMoeLaunchPlanTokenTile),
             "total_padded_rows should match expected") &&
         ExpectVectorPrefix(
             cta_expert_ids,
             expected_expert_ids,
             "cta_expert_ids") &&
         ExpectVectorPrefix(
             cta_row_starts,
             expected_row_starts,
             "cta_row_starts") &&
         ExpectVectorPrefix(
             cta_valid_rows,
             expected_valid_rows,
             "cta_valid_rows") &&
         ExpectVectorPrefix(
             cta_m_limits,
             expected_m_limits,
             "cta_m_limits") &&
         ExpectVectorPrefix(
             permuted_token_indices,
             expected_permuted_token_indices,
             "permuted_token_indices") &&
         ExpectVectorPrefix(
             sorted_to_permuted_indices,
             expected_sorted_to_permuted_indices,
             "sorted_to_permuted_indices");
}

bool RunLaunchPlanValidationCase() {
  return Expect(
             DeviceMoeLaunchPlan::Create(0, 8) == nullptr,
             "launch plan should reject zero experts") &&
         Expect(
             DeviceMoeLaunchPlan::Create(129, 8) == nullptr,
             "launch plan should reject unsupported expert counts") &&
         Expect(
             DeviceMoeLaunchPlan::Create(8, 0) == nullptr,
             "launch plan should reject zero selections");
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::cout << "moe_launch_plan_device_test: SKIP (no CUDA device)\n";
    return 0;
  }

  const std::vector<int> small_selected_indices = {
      3, 1,
      3, 1,
      3, 4,
      4, 1,
  };
  const std::vector<float> small_selected_weights(small_selected_indices.size(), 1.0f);
  const std::vector<int> small_expected_expert_ids = {1, 3, 4};
  const std::vector<int> small_expected_row_starts = {0, 3, 6};
  const std::vector<int> small_expected_valid_rows = {3, 3, 2};
  const std::vector<int> small_expected_m_limits = {3, 11, 18};
  const std::vector<int> small_expected_permuted_token_indices = {
      0, 1, 3, -1, -1, -1, -1, -1,
      0, 1, 2, -1, -1, -1, -1, -1,
      2, 3, -1, -1, -1, -1, -1, -1,
  };
  const std::vector<int> small_expected_sorted_to_permuted_indices = {
      0, 1, 2, 8, 9, 10, 16, 17,
  };

  std::vector<int> multi_tile_selected_indices;
  std::vector<float> multi_tile_selected_weights;
  multi_tile_selected_indices.reserve(20);
  multi_tile_selected_weights.reserve(20);
  for (int index = 0; index < 9; ++index) {
    multi_tile_selected_indices.push_back(0);
    multi_tile_selected_weights.push_back(1.0f);
  }
  for (int index = 0; index < 10; ++index) {
    multi_tile_selected_indices.push_back(2);
    multi_tile_selected_weights.push_back(1.0f);
  }
  multi_tile_selected_indices.push_back(1);
  multi_tile_selected_weights.push_back(1.0f);
  const std::vector<int> multi_tile_expected_expert_ids = {0, 0, 1, 2, 2};
  const std::vector<int> multi_tile_expected_row_starts = {0, 8, 9, 10, 18};
  const std::vector<int> multi_tile_expected_valid_rows = {8, 1, 1, 8, 2};
  const std::vector<int> multi_tile_expected_m_limits = {8, 9, 17, 32, 34};
  const std::vector<int> multi_tile_expected_permuted_token_indices = {
      0, 0, 0, 0, 1, 1, 1, 1,
      2, -1, -1, -1, -1, -1, -1, -1,
      4, -1, -1, -1, -1, -1, -1, -1,
      2, 2, 2, 3, 3, 3, 3, 4,
      4, 4, -1, -1, -1, -1, -1, -1,
  };
  const std::vector<int> multi_tile_expected_sorted_to_permuted_indices = {
      0, 1, 2, 3, 4, 5, 6, 7,
      8, 16, 24, 25, 26, 27, 28, 29,
      30, 31, 32, 33,
  };

  if (!RunLaunchPlanValidationCase() ||
      !RunLaunchPlanCase(
          5,
          4,
          2,
          small_selected_indices,
          small_selected_weights,
          small_expected_expert_ids,
          small_expected_row_starts,
          small_expected_valid_rows,
          small_expected_m_limits,
          small_expected_permuted_token_indices,
          small_expected_sorted_to_permuted_indices) ||
      !RunLaunchPlanCase(
          4,
          5,
          4,
          multi_tile_selected_indices,
          multi_tile_selected_weights,
          multi_tile_expected_expert_ids,
          multi_tile_expected_row_starts,
          multi_tile_expected_valid_rows,
          multi_tile_expected_m_limits,
          multi_tile_expected_permuted_token_indices,
          multi_tile_expected_sorted_to_permuted_indices)) {
    return 1;
  }

  std::cout << "moe_launch_plan_device_test: PASS\n";
  return 0;
}
