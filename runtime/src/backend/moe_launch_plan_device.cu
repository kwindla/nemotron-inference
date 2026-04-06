#include "nemotron/moe_launch_plan_device.h"

#include <cuda_runtime.h>

#include <limits>
#include <memory>
#include <utility>

namespace nemotron {

struct DeviceMoeLaunchPlan::Impl {
  int* cta_count = nullptr;
  int* total_padded_rows = nullptr;
  int* cta_expert_ids = nullptr;
  int* cta_row_starts = nullptr;
  int* cta_valid_rows = nullptr;
  std::size_t n_experts = 0;
  std::size_t selection_count = 0;
  std::size_t cta_capacity = 0;

  ~Impl() {
    if (cta_valid_rows != nullptr) {
      cudaFree(cta_valid_rows);
    }
    if (cta_row_starts != nullptr) {
      cudaFree(cta_row_starts);
    }
    if (cta_expert_ids != nullptr) {
      cudaFree(cta_expert_ids);
    }
    if (total_padded_rows != nullptr) {
      cudaFree(total_padded_rows);
    }
    if (cta_count != nullptr) {
      cudaFree(cta_count);
    }
  }
};

namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool HasCudaDevice() {
  static const bool kHasCudaDevice = []() {
    int device_count = 0;
    return CheckCuda(cudaGetDeviceCount(&device_count)) && device_count > 0;
  }();
  return kHasCudaDevice;
}

bool AllocateDeviceBuffer(void** data, std::size_t bytes) {
  return data != nullptr &&
         bytes > 0 &&
         CheckCuda(cudaMalloc(data, bytes));
}

std::optional<std::size_t> CheckedMul(std::size_t lhs, std::size_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return std::size_t{0};
  }
  if (lhs > (std::numeric_limits<std::size_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

std::optional<std::size_t> CheckedAdd(std::size_t lhs, std::size_t rhs) {
  if (lhs > (std::numeric_limits<std::size_t>::max() - rhs)) {
    return std::nullopt;
  }
  return lhs + rhs;
}

__global__ void BuildLaunchPlanKernel(
    const int* expert_offsets,
    const int* active_expert_count,
    const int* active_expert_ids,
    int cta_capacity,
    int* cta_count,
    int* total_padded_rows,
    int* cta_expert_ids,
    int* cta_row_starts,
    int* cta_valid_rows) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const int active_count = *active_expert_count;
  int write_index = 0;
  int padded_rows = 0;
  for (int active_index = 0; active_index < active_count; ++active_index) {
    const int expert_index = active_expert_ids[active_index];
    if (expert_index < 0) {
      continue;
    }
    const int begin = expert_offsets[expert_index];
    const int end = expert_offsets[expert_index + 1];
    for (int row_start = begin; row_start < end;
         row_start += static_cast<int>(kMoeLaunchPlanTokenTile)) {
      if (write_index >= cta_capacity) {
        *cta_count = cta_capacity + 1;
        *total_padded_rows = padded_rows;
        return;
      }
      const int remaining_rows = end - row_start;
      cta_expert_ids[write_index] = expert_index;
      cta_row_starts[write_index] = row_start;
      cta_valid_rows[write_index] =
          remaining_rows < static_cast<int>(kMoeLaunchPlanTokenTile)
              ? remaining_rows
              : static_cast<int>(kMoeLaunchPlanTokenTile);
      ++write_index;
      padded_rows += static_cast<int>(kMoeLaunchPlanTokenTile);
    }
  }

  for (int index = write_index; index < cta_capacity; ++index) {
    cta_expert_ids[index] = -1;
    cta_row_starts[index] = 0;
    cta_valid_rows[index] = 0;
  }
  *cta_count = write_index;
  *total_padded_rows = padded_rows;
}

}  // namespace

std::optional<std::size_t> DeviceMoeLaunchPlan::CtaCapacity(
    std::size_t n_experts,
    std::size_t selection_count) {
  if (n_experts == 0 ||
      n_experts > kMaxDeviceExpertRoutingExperts ||
      selection_count == 0) {
    return std::nullopt;
  }
  const std::size_t initial_filled = selection_count < n_experts ? selection_count : n_experts;
  const std::size_t remaining = selection_count - initial_filled;
  return CheckedAdd(initial_filled, remaining / kMoeLaunchPlanTokenTile);
}

std::optional<std::size_t> DeviceMoeLaunchPlan::Bytes(
    std::size_t n_experts,
    std::size_t selection_count) {
  const auto cta_capacity = CtaCapacity(n_experts, selection_count);
  if (!cta_capacity.has_value()) {
    return std::nullopt;
  }

  std::size_t total = 0;
  const auto add_bytes = [&](std::optional<std::size_t> bytes) -> bool {
    if (!bytes.has_value()) {
      return false;
    }
    const auto next_total = CheckedAdd(total, *bytes);
    if (!next_total.has_value()) {
      return false;
    }
    total = *next_total;
    return true;
  };

  return add_bytes(sizeof(int)) &&
                 add_bytes(sizeof(int)) &&
                 add_bytes(CheckedMul(*cta_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*cta_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*cta_capacity, sizeof(int)))
             ? std::optional<std::size_t>(total)
             : std::nullopt;
}

std::unique_ptr<DeviceMoeLaunchPlan> DeviceMoeLaunchPlan::Create(
    std::size_t n_experts,
    std::size_t selection_count) {
  const auto cta_capacity = CtaCapacity(n_experts, selection_count);
  if (!HasCudaDevice() ||
      !cta_capacity.has_value() ||
      n_experts > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      selection_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      *cta_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->n_experts = n_experts;
  impl->selection_count = selection_count;
  impl->cta_capacity = *cta_capacity;

  if (!AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_count),
          sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->total_padded_rows),
          sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_expert_ids),
          *cta_capacity * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_row_starts),
          *cta_capacity * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_valid_rows),
          *cta_capacity * sizeof(int))) {
    return nullptr;
  }

  return std::unique_ptr<DeviceMoeLaunchPlan>(
      new DeviceMoeLaunchPlan(std::move(impl)));
}

DeviceMoeLaunchPlan::DeviceMoeLaunchPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

DeviceMoeLaunchPlan::DeviceMoeLaunchPlan(DeviceMoeLaunchPlan&&) noexcept = default;

DeviceMoeLaunchPlan& DeviceMoeLaunchPlan::operator=(DeviceMoeLaunchPlan&&) noexcept =
    default;

DeviceMoeLaunchPlan::~DeviceMoeLaunchPlan() = default;

bool DeviceMoeLaunchPlan::valid() const {
  return impl_ != nullptr &&
         impl_->cta_count != nullptr &&
         impl_->total_padded_rows != nullptr &&
         impl_->cta_expert_ids != nullptr &&
         impl_->cta_row_starts != nullptr &&
         impl_->cta_valid_rows != nullptr &&
         impl_->n_experts > 0 &&
         impl_->selection_count > 0 &&
         impl_->cta_capacity > 0;
}

std::size_t DeviceMoeLaunchPlan::n_experts() const {
  return impl_ != nullptr ? impl_->n_experts : 0;
}

std::size_t DeviceMoeLaunchPlan::selection_count() const {
  return impl_ != nullptr ? impl_->selection_count : 0;
}

std::size_t DeviceMoeLaunchPlan::cta_capacity() const {
  return impl_ != nullptr ? impl_->cta_capacity : 0;
}

int* DeviceMoeLaunchPlan::cta_count() const {
  return impl_ != nullptr ? impl_->cta_count : nullptr;
}

int* DeviceMoeLaunchPlan::total_padded_rows() const {
  return impl_ != nullptr ? impl_->total_padded_rows : nullptr;
}

int* DeviceMoeLaunchPlan::cta_expert_ids() const {
  return impl_ != nullptr ? impl_->cta_expert_ids : nullptr;
}

int* DeviceMoeLaunchPlan::cta_row_starts() const {
  return impl_ != nullptr ? impl_->cta_row_starts : nullptr;
}

int* DeviceMoeLaunchPlan::cta_valid_rows() const {
  return impl_ != nullptr ? impl_->cta_valid_rows : nullptr;
}

bool BuildDeviceMoeLaunchPlan(
    const DeviceExpertRouting& routing,
    DeviceMoeLaunchPlan* plan) {
  if (!routing.valid() ||
      plan == nullptr ||
      !plan->valid() ||
      plan->n_experts() != routing.n_experts() ||
      plan->selection_count() < routing.selection_count() ||
      plan->cta_count() == nullptr ||
      plan->total_padded_rows() == nullptr ||
      plan->cta_expert_ids() == nullptr ||
      plan->cta_row_starts() == nullptr ||
      plan->cta_valid_rows() == nullptr ||
      routing.expert_offsets() == nullptr ||
      routing.active_expert_count() == nullptr ||
      routing.active_expert_ids() == nullptr) {
    return false;
  }

  BuildLaunchPlanKernel<<<1, 1>>>(
      routing.expert_offsets(),
      routing.active_expert_count(),
      routing.active_expert_ids(),
      static_cast<int>(plan->cta_capacity()),
      plan->cta_count(),
      plan->total_padded_rows(),
      plan->cta_expert_ids(),
      plan->cta_row_starts(),
      plan->cta_valid_rows());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
