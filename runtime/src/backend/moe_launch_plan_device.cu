#include "nemotron/moe_launch_plan_device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

namespace nemotron {

struct DeviceMoeLaunchPlan::Impl {
  int* cta_count = nullptr;
  int* total_padded_rows = nullptr;
  int* expert_first_token_offsets = nullptr;
  int* cta_expert_ids = nullptr;
  int* cta_row_starts = nullptr;
  int* cta_valid_rows = nullptr;
  int* cta_m_limits = nullptr;
  int* permuted_token_indices = nullptr;
  int* sorted_to_permuted_indices = nullptr;
  int* task_count = nullptr;
  int* task_expert_ids = nullptr;
  int* task_row_starts = nullptr;
  int* task_valid_rows = nullptr;
  int* task_output_row_bases = nullptr;
  std::size_t n_experts = 0;
  std::size_t selection_count = 0;
  std::size_t cta_capacity = 0;
  std::size_t padded_row_capacity = 0;
  std::size_t selected_token_tile = kMoeLaunchPlanMaxTokenTile;
  std::size_t max_output_rows_per_expert = 0;
  std::size_t task_capacity = 0;

  ~Impl() {
    if (task_output_row_bases != nullptr) {
      cudaFree(task_output_row_bases);
    }
    if (task_valid_rows != nullptr) {
      cudaFree(task_valid_rows);
    }
    if (task_row_starts != nullptr) {
      cudaFree(task_row_starts);
    }
    if (task_expert_ids != nullptr) {
      cudaFree(task_expert_ids);
    }
    if (task_count != nullptr) {
      cudaFree(task_count);
    }
    if (sorted_to_permuted_indices != nullptr) {
      cudaFree(sorted_to_permuted_indices);
    }
    if (permuted_token_indices != nullptr) {
      cudaFree(permuted_token_indices);
    }
    if (cta_m_limits != nullptr) {
      cudaFree(cta_m_limits);
    }
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
    if (expert_first_token_offsets != nullptr) {
      cudaFree(expert_first_token_offsets);
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

std::size_t NextPowerOfTwo(std::size_t value) {
  if (value <= 1) {
    return 1;
  }
  std::size_t next = 1;
  while (next < value) {
    next <<= 1u;
  }
  return next;
}

std::optional<std::size_t> CtaCapacityForTile(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t token_tile) {
  if (n_experts == 0 ||
      n_experts > kMaxDeviceExpertRoutingExperts ||
      selection_count == 0 ||
      token_tile == 0) {
    return std::nullopt;
  }
  const std::size_t initial_filled = selection_count < n_experts ? selection_count : n_experts;
  const std::size_t remaining = selection_count - initial_filled;
  return CheckedAdd(initial_filled, remaining / token_tile);
}

std::optional<std::size_t> PaddedRowCapacityForAlignment(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t expert_row_alignment) {
  if (n_experts == 0 || selection_count == 0 || expert_row_alignment == 0) {
    return std::nullopt;
  }
  const std::size_t max_active_experts =
      selection_count < n_experts ? selection_count : n_experts;
  const auto padding_slack =
      CheckedMul(max_active_experts, expert_row_alignment - 1u);
  if (!padding_slack.has_value()) {
    return std::nullopt;
  }
  return CheckedAdd(selection_count, *padding_slack);
}

__global__ void BuildLaunchPlanKernel(
    const int* expert_offsets,
    const int* sorted_token_indices,
    int n_experts,
    int token_tile_dim,
    int current_cta_capacity,
    int current_padded_row_capacity,
    int* cta_count,
    int* total_padded_rows,
    int* expert_first_token_offsets,
    int* cta_expert_ids,
    int* cta_row_starts,
    int* cta_valid_rows,
    int* cta_m_limits,
    int* permuted_token_indices,
    int* sorted_to_permuted_indices,
    int active_selection_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const auto align_rows = [token_tile_dim](int value) {
    const int alignment = token_tile_dim;
    return value <= 0 ? 0 : ((value + alignment - 1) / alignment) * alignment;
  };

  for (int sorted_index = 0; sorted_index < active_selection_count; ++sorted_index) {
    sorted_to_permuted_indices[sorted_index] = -1;
  }
  for (int padded_index = 0; padded_index < current_padded_row_capacity; ++padded_index) {
    permuted_token_indices[padded_index] = -1;
  }
  int write_index = 0;
  int padded_rows = 0;
  for (int expert_index = 0; expert_index < n_experts; ++expert_index) {
    expert_first_token_offsets[expert_index] = padded_rows;
    const int begin = expert_offsets[expert_index];
    const int end = expert_offsets[expert_index + 1];
    const int rows_for_expert = end - begin;
    int local_row_start = 0;
    for (int row_start = begin; row_start < end;
         row_start += token_tile_dim,
         local_row_start += token_tile_dim) {
      if (write_index >= current_cta_capacity) {
        *cta_count = current_cta_capacity + 1;
        *total_padded_rows = padded_rows;
        expert_first_token_offsets[n_experts] = padded_rows;
        return;
      }
      const int remaining_rows = end - row_start;
      const int valid_rows = remaining_rows < token_tile_dim ? remaining_rows : token_tile_dim;
      cta_expert_ids[write_index] = expert_index;
      cta_row_starts[write_index] = padded_rows + local_row_start;
      cta_valid_rows[write_index] = valid_rows;
      const int cta_row_end = local_row_start + token_tile_dim;
      cta_m_limits[write_index] =
          padded_rows + (cta_row_end < rows_for_expert ? cta_row_end : rows_for_expert);
      const int padded_base = padded_rows + local_row_start;
      if ((padded_base + token_tile_dim) > current_padded_row_capacity) {
        *cta_count = current_cta_capacity + 1;
        *total_padded_rows = padded_rows;
        expert_first_token_offsets[n_experts] = padded_rows;
        return;
      }
      for (int token_offset = 0; token_offset < token_tile_dim; ++token_offset) {
        if (token_offset < valid_rows) {
          const int sorted_index = row_start + token_offset;
          permuted_token_indices[padded_base + token_offset] = sorted_token_indices[sorted_index];
          sorted_to_permuted_indices[sorted_index] = padded_base + token_offset;
        }
      }
      ++write_index;
    }
    padded_rows += align_rows(rows_for_expert);
  }
  expert_first_token_offsets[n_experts] = padded_rows;
  *cta_count = write_index;
  *total_padded_rows = padded_rows;
}

__global__ void BuildExactTaskMapKernel(
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    int current_task_capacity,
    int output_row_tile_count,
    int output_row_tile_size,
    int* task_count,
    int* task_expert_ids,
    int* task_row_starts,
    int* task_valid_rows,
    int* task_output_row_bases) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  const int exact_cta_count = cta_count[0];
  int write_index = 0;
  for (int cta_index = 0; cta_index < exact_cta_count; ++cta_index) {
    for (int output_row_tile = 0; output_row_tile < output_row_tile_count; ++output_row_tile) {
      if (write_index >= current_task_capacity) {
        task_count[0] = current_task_capacity + 1;
        return;
      }
      task_expert_ids[write_index] = cta_expert_ids[cta_index];
      task_row_starts[write_index] = cta_row_starts[cta_index];
      task_valid_rows[write_index] = cta_valid_rows[cta_index];
      task_output_row_bases[write_index] = output_row_tile * output_row_tile_size;
      ++write_index;
    }
  }
  task_count[0] = write_index;
}

}  // namespace

std::size_t SelectMoeLaunchPlanTokenTile(
    std::size_t n_experts,
    std::size_t active_selection_count) {
  if (n_experts == 0 || active_selection_count == 0) {
    return kMoeLaunchPlanMaxTokenTile;
  }
  const float avg_tokens_per_expert =
      static_cast<float>(active_selection_count) / static_cast<float>(n_experts);
  const std::size_t rounded = NextPowerOfTwo(static_cast<std::size_t>(avg_tokens_per_expert));
  return std::clamp(
      rounded,
      kMoeLaunchPlanMinTokenTile,
      kMoeLaunchPlanMaxTokenTile);
}

std::optional<std::size_t> DeviceMoeLaunchPlan::CtaCapacity(
    std::size_t n_experts,
    std::size_t selection_count) {
  return CtaCapacityForTile(
      n_experts, selection_count, kMoeLaunchPlanMinTokenTile);
}

std::optional<std::size_t> DeviceMoeLaunchPlan::CtaCapacity(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t token_tile) {
  return CtaCapacityForTile(n_experts, selection_count, token_tile);
}

std::optional<std::size_t> DeviceMoeLaunchPlan::TaskCapacity(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t max_output_rows_per_expert) {
  const auto cta_capacity = CtaCapacity(n_experts, selection_count);
  if (!cta_capacity.has_value() || max_output_rows_per_expert == 0) {
    return std::nullopt;
  }
  const std::size_t output_row_tile_count =
      (max_output_rows_per_expert + kMoeLaunchPlanOutputTile - 1u) /
      kMoeLaunchPlanOutputTile;
  if (output_row_tile_count == 0) {
    return std::nullopt;
  }
  return CheckedMul(*cta_capacity, output_row_tile_count);
}

std::optional<std::size_t> DeviceMoeLaunchPlan::Bytes(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t max_output_rows_per_expert) {
  const auto cta_capacity = CtaCapacity(n_experts, selection_count);
  const auto padded_row_capacity = PaddedRowCapacity(n_experts, selection_count);
  const auto task_capacity = max_output_rows_per_expert > 0
      ? TaskCapacity(n_experts, selection_count, max_output_rows_per_expert)
      : std::optional<std::size_t>(std::size_t{0});
  if (!cta_capacity.has_value()) {
    return std::nullopt;
  }
  if (!padded_row_capacity.has_value()) {
    return std::nullopt;
  }
  if (!task_capacity.has_value()) {
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
                 add_bytes(CheckedMul(n_experts + 1u, sizeof(int))) &&
                 add_bytes(CheckedMul(*cta_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*cta_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*cta_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*cta_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*padded_row_capacity, sizeof(int)))
                 &&
                 add_bytes(CheckedMul(selection_count, sizeof(int))) &&
                 add_bytes(std::optional<std::size_t>(
                     max_output_rows_per_expert > 0 ? sizeof(int) : std::size_t{0})) &&
                 add_bytes(CheckedMul(*task_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*task_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*task_capacity, sizeof(int))) &&
                 add_bytes(CheckedMul(*task_capacity, sizeof(int)))
             ? std::optional<std::size_t>(total)
             : std::nullopt;
}

std::optional<std::size_t> DeviceMoeLaunchPlan::PaddedRowCapacity(
    std::size_t n_experts,
    std::size_t selection_count) {
  return PaddedRowCapacityForAlignment(
      n_experts, selection_count, kMoeLaunchPlanExpertRowAlignment);
}

std::optional<std::size_t> DeviceMoeLaunchPlan::PaddedRowCapacity(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t expert_row_alignment) {
  return PaddedRowCapacityForAlignment(
      n_experts, selection_count, expert_row_alignment);
}

std::unique_ptr<DeviceMoeLaunchPlan> DeviceMoeLaunchPlan::Create(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t max_output_rows_per_expert) {
  const auto cta_capacity = CtaCapacity(n_experts, selection_count);
  const auto padded_row_capacity = PaddedRowCapacity(n_experts, selection_count);
  const auto task_capacity = max_output_rows_per_expert > 0
      ? TaskCapacity(n_experts, selection_count, max_output_rows_per_expert)
      : std::optional<std::size_t>(std::size_t{0});
  if (!HasCudaDevice() ||
      !cta_capacity.has_value() ||
      !padded_row_capacity.has_value() ||
      !task_capacity.has_value() ||
      n_experts > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      selection_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      *cta_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      *padded_row_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      *task_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->n_experts = n_experts;
  impl->selection_count = selection_count;
  impl->cta_capacity = *cta_capacity;
  impl->padded_row_capacity = *padded_row_capacity;
  impl->max_output_rows_per_expert = max_output_rows_per_expert;
  impl->task_capacity = *task_capacity;

  if (!AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_count),
          sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->total_padded_rows),
          sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->expert_first_token_offsets),
          (n_experts + 1u) * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_expert_ids),
          *cta_capacity * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_row_starts),
          *cta_capacity * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_valid_rows),
          *cta_capacity * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->cta_m_limits),
          *cta_capacity * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->permuted_token_indices),
          *padded_row_capacity * sizeof(int)) ||
      !AllocateDeviceBuffer(
          reinterpret_cast<void**>(&impl->sorted_to_permuted_indices),
          selection_count * sizeof(int)) ||
      (max_output_rows_per_expert > 0 &&
       (!AllocateDeviceBuffer(
            reinterpret_cast<void**>(&impl->task_count),
            sizeof(int)) ||
        !AllocateDeviceBuffer(
            reinterpret_cast<void**>(&impl->task_expert_ids),
            *task_capacity * sizeof(int)) ||
        !AllocateDeviceBuffer(
            reinterpret_cast<void**>(&impl->task_row_starts),
            *task_capacity * sizeof(int)) ||
        !AllocateDeviceBuffer(
            reinterpret_cast<void**>(&impl->task_valid_rows),
            *task_capacity * sizeof(int)) ||
        !AllocateDeviceBuffer(
            reinterpret_cast<void**>(&impl->task_output_row_bases),
            *task_capacity * sizeof(int))))) {
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
         impl_->expert_first_token_offsets != nullptr &&
         impl_->cta_expert_ids != nullptr &&
         impl_->cta_row_starts != nullptr &&
         impl_->cta_valid_rows != nullptr &&
         impl_->cta_m_limits != nullptr &&
         impl_->permuted_token_indices != nullptr &&
         impl_->sorted_to_permuted_indices != nullptr &&
         impl_->n_experts > 0 &&
         impl_->selection_count > 0 &&
         impl_->cta_capacity > 0 &&
         impl_->padded_row_capacity > 0 &&
         ((impl_->max_output_rows_per_expert == 0 &&
           impl_->task_capacity == 0 &&
           impl_->task_count == nullptr &&
           impl_->task_expert_ids == nullptr &&
           impl_->task_row_starts == nullptr &&
           impl_->task_valid_rows == nullptr &&
           impl_->task_output_row_bases == nullptr) ||
          (impl_->max_output_rows_per_expert > 0 &&
           impl_->task_capacity > 0 &&
           impl_->task_count != nullptr &&
           impl_->task_expert_ids != nullptr &&
           impl_->task_row_starts != nullptr &&
           impl_->task_valid_rows != nullptr &&
           impl_->task_output_row_bases != nullptr));
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

std::size_t DeviceMoeLaunchPlan::padded_row_capacity() const {
  return impl_ != nullptr ? impl_->padded_row_capacity : 0;
}

std::size_t DeviceMoeLaunchPlan::selected_token_tile() const {
  return impl_ != nullptr ? impl_->selected_token_tile : kMoeLaunchPlanMaxTokenTile;
}

std::size_t DeviceMoeLaunchPlan::max_output_rows_per_expert() const {
  return impl_ != nullptr ? impl_->max_output_rows_per_expert : 0;
}

std::size_t DeviceMoeLaunchPlan::task_capacity() const {
  return impl_ != nullptr ? impl_->task_capacity : 0;
}

int* DeviceMoeLaunchPlan::cta_count() const {
  return impl_ != nullptr ? impl_->cta_count : nullptr;
}

int* DeviceMoeLaunchPlan::total_padded_rows() const {
  return impl_ != nullptr ? impl_->total_padded_rows : nullptr;
}

int* DeviceMoeLaunchPlan::expert_first_token_offsets() const {
  return impl_ != nullptr ? impl_->expert_first_token_offsets : nullptr;
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

int* DeviceMoeLaunchPlan::cta_m_limits() const {
  return impl_ != nullptr ? impl_->cta_m_limits : nullptr;
}

int* DeviceMoeLaunchPlan::permuted_token_indices() const {
  return impl_ != nullptr ? impl_->permuted_token_indices : nullptr;
}

int* DeviceMoeLaunchPlan::sorted_to_permuted_indices() const {
  return impl_ != nullptr ? impl_->sorted_to_permuted_indices : nullptr;
}

int* DeviceMoeLaunchPlan::num_non_exiting_ctas() const {
  return cta_count();
}

int* DeviceMoeLaunchPlan::total_num_padded_tokens() const {
  return total_padded_rows();
}

int* DeviceMoeLaunchPlan::cta_idx_xy_to_batch_idx() const {
  return cta_expert_ids();
}

int* DeviceMoeLaunchPlan::cta_idx_xy_to_mn_limit() const {
  return cta_m_limits();
}

int* DeviceMoeLaunchPlan::permuted_idx_to_token_idx() const {
  return permuted_token_indices();
}

int* DeviceMoeLaunchPlan::expanded_idx_to_permuted_idx() const {
  return sorted_to_permuted_indices();
}

int* DeviceMoeLaunchPlan::task_count() const {
  return impl_ != nullptr ? impl_->task_count : nullptr;
}

int* DeviceMoeLaunchPlan::task_expert_ids() const {
  return impl_ != nullptr ? impl_->task_expert_ids : nullptr;
}

int* DeviceMoeLaunchPlan::task_row_starts() const {
  return impl_ != nullptr ? impl_->task_row_starts : nullptr;
}

int* DeviceMoeLaunchPlan::task_valid_rows() const {
  return impl_ != nullptr ? impl_->task_valid_rows : nullptr;
}

int* DeviceMoeLaunchPlan::task_output_row_bases() const {
  return impl_ != nullptr ? impl_->task_output_row_bases : nullptr;
}

bool BuildDeviceMoeLaunchPlan(
    const DeviceExpertRouting& routing,
    std::size_t active_selection_count,
    DeviceMoeLaunchPlan* plan) {
  if (!routing.valid() ||
      plan == nullptr ||
      !plan->valid() ||
      active_selection_count == 0 ||
      plan->n_experts() != routing.n_experts() ||
      active_selection_count > routing.selection_count() ||
      plan->selection_count() < active_selection_count ||
      plan->cta_count() == nullptr ||
      plan->total_padded_rows() == nullptr ||
      plan->expert_first_token_offsets() == nullptr ||
      plan->cta_expert_ids() == nullptr ||
      plan->cta_row_starts() == nullptr ||
      plan->cta_valid_rows() == nullptr ||
      plan->cta_m_limits() == nullptr ||
      plan->permuted_token_indices() == nullptr ||
      plan->sorted_to_permuted_indices() == nullptr ||
      routing.expert_offsets() == nullptr ||
      routing.sorted_token_indices() == nullptr) {
    return false;
  }

  const std::size_t token_tile_dim =
      SelectMoeLaunchPlanTokenTile(plan->n_experts(), active_selection_count);
  const auto current_cta_capacity =
      CtaCapacityForTile(plan->n_experts(), active_selection_count, token_tile_dim);
  const auto current_padded_row_capacity =
      PaddedRowCapacityForAlignment(plan->n_experts(), active_selection_count, token_tile_dim);
  if (!current_cta_capacity.has_value() ||
      !current_padded_row_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > plan->cta_capacity() ||
      *current_padded_row_capacity == 0 ||
      *current_padded_row_capacity > plan->padded_row_capacity() ||
      *current_cta_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  plan->impl_->selected_token_tile = token_tile_dim;

  BuildLaunchPlanKernel<<<1, 1>>>(
      routing.expert_offsets(),
      routing.sorted_token_indices(),
      static_cast<int>(plan->n_experts()),
      static_cast<int>(token_tile_dim),
      static_cast<int>(*current_cta_capacity),
      static_cast<int>(*current_padded_row_capacity),
      plan->cta_count(),
      plan->total_padded_rows(),
      plan->expert_first_token_offsets(),
      plan->cta_expert_ids(),
      plan->cta_row_starts(),
      plan->cta_valid_rows(),
      plan->cta_m_limits(),
      plan->permuted_token_indices(),
      plan->sorted_to_permuted_indices(),
      static_cast<int>(active_selection_count));
  return CheckCuda(cudaGetLastError());
}

bool BuildDeviceMoeExactTaskMap(
    std::size_t output_rows_per_expert,
    std::size_t output_row_tile_size,
    DeviceMoeLaunchPlan* plan) {
  if (plan == nullptr ||
      !plan->valid() ||
      output_rows_per_expert == 0 ||
      output_row_tile_size == 0 ||
      plan->task_count() == nullptr ||
      plan->task_expert_ids() == nullptr ||
      plan->task_row_starts() == nullptr ||
      plan->task_valid_rows() == nullptr ||
      plan->task_output_row_bases() == nullptr ||
      plan->cta_count() == nullptr ||
      plan->cta_expert_ids() == nullptr ||
      plan->cta_row_starts() == nullptr ||
      plan->cta_valid_rows() == nullptr ||
      plan->max_output_rows_per_expert() == 0 ||
      output_rows_per_expert > plan->max_output_rows_per_expert()) {
    return false;
  }

  const std::size_t output_row_tile_count =
      (output_rows_per_expert + output_row_tile_size - 1u) /
      output_row_tile_size;
  if (output_row_tile_count == 0 ||
      output_row_tile_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  const auto exact_task_capacity =
      CheckedMul(plan->cta_capacity(), output_row_tile_count);
  if (!exact_task_capacity.has_value() ||
      *exact_task_capacity == 0 ||
      *exact_task_capacity > plan->task_capacity() ||
      *exact_task_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  BuildExactTaskMapKernel<<<1, 1>>>(
      plan->cta_count(),
      plan->cta_expert_ids(),
      plan->cta_row_starts(),
      plan->cta_valid_rows(),
      static_cast<int>(*exact_task_capacity),
      static_cast<int>(output_row_tile_count),
      static_cast<int>(output_row_tile_size),
      plan->task_count(),
      plan->task_expert_ids(),
      plan->task_row_starts(),
      plan->task_valid_rows(),
      plan->task_output_row_bases());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
