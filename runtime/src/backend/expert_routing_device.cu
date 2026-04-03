#include "nemotron/expert_routing_device.h"

#include <cuda_runtime.h>

#include <limits>
#include <memory>
#include <utility>

namespace nemotron {

struct DeviceExpertRouting::Impl {
  int* expert_counts = nullptr;
  int* expert_offsets = nullptr;
  int* sorted_token_indices = nullptr;
  float* sorted_token_weights = nullptr;
  int* active_expert_count = nullptr;
  int* active_expert_ids = nullptr;
  std::size_t n_experts = 0;
  std::size_t selection_count = 0;

  ~Impl() {
    if (expert_counts != nullptr) {
      cudaFree(expert_counts);
    }
    if (expert_offsets != nullptr) {
      cudaFree(expert_offsets);
    }
    if (sorted_token_indices != nullptr) {
      cudaFree(sorted_token_indices);
    }
    if (sorted_token_weights != nullptr) {
      cudaFree(sorted_token_weights);
    }
    if (active_expert_count != nullptr) {
      cudaFree(active_expert_count);
    }
    if (active_expert_ids != nullptr) {
      cudaFree(active_expert_ids);
    }
  }
};

namespace {

constexpr int kRoutingThreadsPerBlock = 128;
constexpr int kScanWarpThreads = 32;
constexpr int kScanItemsPerLane = 4;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

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

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

__global__ void ExpertHistogramKernel(
    const int* selected_indices,
    std::size_t selection_count,
    int n_experts,
    int* expert_counts) {
  const int expert_index =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (expert_index >= n_experts) {
    return;
  }

  int count = 0;
  for (std::size_t selection_index = 0;
       selection_index < selection_count;
       ++selection_index) {
    count += selected_indices[selection_index] == expert_index ? 1 : 0;
  }
  expert_counts[expert_index] = count;
}

__global__ void ExpertPrefixSumKernel(
    const int* expert_counts,
    int n_experts,
    int* expert_offsets) {
  const int lane = static_cast<int>(threadIdx.x);
  if (blockIdx.x != 0 || lane >= kScanWarpThreads) {
    return;
  }

  int local_offsets[kScanItemsPerLane];
  int local_total = 0;
  for (int item = 0; item < kScanItemsPerLane; ++item) {
    const int expert_index = lane * kScanItemsPerLane + item;
    local_offsets[item] = local_total;
    if (expert_index < n_experts) {
      local_total += expert_counts[expert_index];
    }
  }

  int inclusive_lane_total = local_total;
  for (int offset = 1; offset < kScanWarpThreads; offset <<= 1) {
    const int other =
        __shfl_up_sync(kFullWarpMask, inclusive_lane_total, offset);
    if (lane >= offset) {
      inclusive_lane_total += other;
    }
  }
  const int lane_base = inclusive_lane_total - local_total;

  for (int item = 0; item < kScanItemsPerLane; ++item) {
    const int expert_index = lane * kScanItemsPerLane + item;
    if (expert_index < n_experts) {
      expert_offsets[expert_index] = lane_base + local_offsets[item];
    }
  }

  if (lane == (kScanWarpThreads - 1)) {
    expert_offsets[n_experts] = inclusive_lane_total;
  }
}

__global__ void ExpertScatterKernel(
    const int* selected_indices,
    const float* selected_weights,
    std::size_t selection_count,
    int top_k,
    int n_experts,
    const int* expert_offsets,
    int* sorted_token_indices,
    float* sorted_token_weights) {
  const int expert_index =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (expert_index >= n_experts) {
    return;
  }

  int write_index = expert_offsets[expert_index];
  for (std::size_t selection_index = 0;
       selection_index < selection_count;
       ++selection_index) {
    if (selected_indices[selection_index] != expert_index) {
      continue;
    }
    sorted_token_indices[write_index] =
        static_cast<int>(selection_index / static_cast<std::size_t>(top_k));
    sorted_token_weights[write_index] = selected_weights[selection_index];
    ++write_index;
  }
}

__global__ void ExpertCompactKernel(
    const int* expert_counts,
    int n_experts,
    int* active_expert_count,
    int* active_expert_ids) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int count = 0;
  for (int expert_index = 0; expert_index < n_experts; ++expert_index) {
    if (expert_counts[expert_index] > 0) {
      active_expert_ids[count] = expert_index;
      ++count;
    }
  }
  *active_expert_count = count;
}

}  // namespace

std::unique_ptr<DeviceExpertRouting> DeviceExpertRouting::Create(
    std::size_t n_experts,
    std::size_t selection_count) {
  if (!HasCudaDevice() ||
      n_experts == 0 ||
      n_experts > kMaxDeviceExpertRoutingExperts ||
      selection_count == 0 ||
      n_experts > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      selection_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->n_experts = n_experts;
  impl->selection_count = selection_count;

  if (!AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->expert_counts),
          n_experts * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->expert_offsets),
          (n_experts + 1) * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->sorted_token_indices),
          selection_count * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->sorted_token_weights),
          selection_count * sizeof(float)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->active_expert_count),
          sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->active_expert_ids),
          n_experts * sizeof(int))) {
    return nullptr;
  }

  return std::unique_ptr<DeviceExpertRouting>(
      new DeviceExpertRouting(std::move(impl)));
}

DeviceExpertRouting::DeviceExpertRouting(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

DeviceExpertRouting::DeviceExpertRouting(DeviceExpertRouting&&) noexcept = default;

DeviceExpertRouting& DeviceExpertRouting::operator=(DeviceExpertRouting&&) noexcept =
    default;

DeviceExpertRouting::~DeviceExpertRouting() = default;

bool DeviceExpertRouting::valid() const {
  return impl_ != nullptr &&
         impl_->expert_counts != nullptr &&
         impl_->expert_offsets != nullptr &&
         impl_->sorted_token_indices != nullptr &&
         impl_->sorted_token_weights != nullptr &&
         impl_->active_expert_count != nullptr &&
         impl_->active_expert_ids != nullptr &&
         impl_->n_experts > 0 &&
         impl_->selection_count > 0;
}

std::size_t DeviceExpertRouting::n_experts() const {
  return impl_ != nullptr ? impl_->n_experts : 0;
}

std::size_t DeviceExpertRouting::selection_count() const {
  return impl_ != nullptr ? impl_->selection_count : 0;
}

int* DeviceExpertRouting::expert_offsets() const {
  return impl_ != nullptr ? impl_->expert_offsets : nullptr;
}

int* DeviceExpertRouting::sorted_token_indices() const {
  return impl_ != nullptr ? impl_->sorted_token_indices : nullptr;
}

float* DeviceExpertRouting::sorted_token_weights() const {
  return impl_ != nullptr ? impl_->sorted_token_weights : nullptr;
}

int* DeviceExpertRouting::active_expert_count() const {
  return impl_ != nullptr ? impl_->active_expert_count : nullptr;
}

int* DeviceExpertRouting::active_expert_ids() const {
  return impl_ != nullptr ? impl_->active_expert_ids : nullptr;
}

bool RunDeviceExpertRouting(
    const int* selected_indices,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    DeviceExpertRouting* routing) {
  if (selected_indices == nullptr ||
      selected_weights == nullptr ||
      token_count == 0 ||
      top_k == 0 ||
      token_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      routing == nullptr ||
      !routing->valid()) {
    return false;
  }

  if (token_count > (std::numeric_limits<std::size_t>::max() / top_k)) {
    return false;
  }

  const std::size_t selection_count = token_count * top_k;
  if (selection_count > routing->selection_count() ||
      selection_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      routing->n_experts() > kMaxDeviceExpertRoutingExperts) {
    return false;
  }

  const int n_experts = static_cast<int>(routing->n_experts());
  const int top_k_int = static_cast<int>(top_k);
  const dim3 block(kRoutingThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(CeilDiv(routing->n_experts(), block.x)));

  if (!CheckCuda(cudaMemset(
          routing->impl_->expert_counts,
          0,
          routing->n_experts() * sizeof(int))) ||
      !CheckCuda(cudaMemset(routing->impl_->active_expert_count, 0, sizeof(int)))) {
    return false;
  }

  ExpertHistogramKernel<<<grid, block>>>(
      selected_indices,
      selection_count,
      n_experts,
      routing->impl_->expert_counts);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  ExpertPrefixSumKernel<<<1, kScanWarpThreads>>>(
      routing->impl_->expert_counts,
      n_experts,
      routing->impl_->expert_offsets);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  ExpertScatterKernel<<<grid, block>>>(
      selected_indices,
      selected_weights,
      selection_count,
      top_k_int,
      n_experts,
      routing->impl_->expert_offsets,
      routing->impl_->sorted_token_indices,
      routing->impl_->sorted_token_weights);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  ExpertCompactKernel<<<1, 1>>>(
      routing->impl_->expert_counts,
      n_experts,
      routing->impl_->active_expert_count,
      routing->impl_->active_expert_ids);
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
