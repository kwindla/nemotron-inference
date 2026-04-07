#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "nemotron/expert_routing_device.h"

namespace nemotron {

constexpr std::size_t kMoeLaunchPlanTokenTile = 8;
constexpr std::size_t kMoeLaunchPlanOutputTile = 8;
constexpr std::size_t kMoeLaunchPlanExpertRowAlignment = 128;

class DeviceMoeLaunchPlan {
 public:
  static std::optional<std::size_t> CtaCapacity(
      std::size_t n_experts,
      std::size_t selection_count);
  static std::optional<std::size_t> TaskCapacity(
      std::size_t n_experts,
      std::size_t selection_count,
      std::size_t max_output_rows_per_expert);
  static std::optional<std::size_t> PaddedRowCapacity(
      std::size_t n_experts,
      std::size_t selection_count);
  static std::optional<std::size_t> Bytes(
      std::size_t n_experts,
      std::size_t selection_count,
      std::size_t max_output_rows_per_expert = 0);
  static std::unique_ptr<DeviceMoeLaunchPlan> Create(
      std::size_t n_experts,
      std::size_t selection_count,
      std::size_t max_output_rows_per_expert = 0);

  DeviceMoeLaunchPlan(DeviceMoeLaunchPlan&&) noexcept;
  DeviceMoeLaunchPlan& operator=(DeviceMoeLaunchPlan&&) noexcept;
  ~DeviceMoeLaunchPlan();

  DeviceMoeLaunchPlan(const DeviceMoeLaunchPlan&) = delete;
  DeviceMoeLaunchPlan& operator=(const DeviceMoeLaunchPlan&) = delete;

  bool valid() const;
  std::size_t n_experts() const;
  std::size_t selection_count() const;
  std::size_t cta_capacity() const;
  std::size_t padded_row_capacity() const;
  std::size_t max_output_rows_per_expert() const;
  std::size_t task_capacity() const;

  int* cta_count() const;
  int* total_padded_rows() const;
  int* expert_first_token_offsets() const;
  int* cta_expert_ids() const;
  int* cta_row_starts() const;
  int* cta_valid_rows() const;
  int* cta_m_limits() const;
  int* permuted_token_indices() const;
  int* sorted_to_permuted_indices() const;
  int* task_count() const;
  int* task_expert_ids() const;
  int* task_row_starts() const;
  int* task_valid_rows() const;
  int* task_output_row_bases() const;

 private:
  struct Impl;

  explicit DeviceMoeLaunchPlan(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

bool BuildDeviceMoeLaunchPlan(
    const DeviceExpertRouting& routing,
    std::size_t active_selection_count,
    DeviceMoeLaunchPlan* plan);
bool BuildDeviceMoeExactTaskMap(
    std::size_t output_rows_per_expert,
    DeviceMoeLaunchPlan* plan);

}  // namespace nemotron
