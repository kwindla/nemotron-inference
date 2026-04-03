#pragma once

#include <cstddef>
#include <memory>

namespace nemotron {

constexpr std::size_t kMaxDeviceExpertRoutingExperts = 128;

class DeviceExpertRouting {
 public:
  static std::unique_ptr<DeviceExpertRouting> Create(
      std::size_t n_experts,
      std::size_t selection_count);

  DeviceExpertRouting(DeviceExpertRouting&&) noexcept;
  DeviceExpertRouting& operator=(DeviceExpertRouting&&) noexcept;
  ~DeviceExpertRouting();

  DeviceExpertRouting(const DeviceExpertRouting&) = delete;
  DeviceExpertRouting& operator=(const DeviceExpertRouting&) = delete;

  bool valid() const;
  std::size_t n_experts() const;
  std::size_t selection_count() const;

  int* expert_offsets() const;
  int* sorted_token_indices() const;
  float* sorted_token_weights() const;
  int* active_expert_count() const;
  int* active_expert_ids() const;

 private:
  friend bool RunDeviceExpertRouting(
      const int* selected_indices,
      const float* selected_weights,
      std::size_t token_count,
      std::size_t top_k,
      DeviceExpertRouting* routing);

  struct Impl;

  explicit DeviceExpertRouting(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

bool RunDeviceExpertRouting(
    const int* selected_indices,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    DeviceExpertRouting* routing);

}  // namespace nemotron
