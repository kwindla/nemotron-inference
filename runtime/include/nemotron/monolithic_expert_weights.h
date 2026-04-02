#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "nemotron/fused_moe_decode.h"

namespace nemotron {

class MonolithicNvfp4ExpertWeights {
 public:
  static std::unique_ptr<MonolithicNvfp4ExpertWeights> Create(
      std::size_t num_experts,
      std::size_t output_rows,
      std::size_t input_cols);

  MonolithicNvfp4ExpertWeights(MonolithicNvfp4ExpertWeights&&) noexcept;
  MonolithicNvfp4ExpertWeights& operator=(MonolithicNvfp4ExpertWeights&&) noexcept;
  ~MonolithicNvfp4ExpertWeights();

  MonolithicNvfp4ExpertWeights(const MonolithicNvfp4ExpertWeights&) = delete;
  MonolithicNvfp4ExpertWeights& operator=(const MonolithicNvfp4ExpertWeights&) = delete;

  bool UploadExpert(
      std::size_t expert_index,
      const std::uint8_t* host_packed,
      std::size_t packed_nbytes,
      const std::uint8_t* host_block_scales,
      std::size_t block_scales_nbytes,
      const float* host_tensor_scale);
  float host_tensor_scale(std::size_t expert_index) const;
  FusedNvfp4WeightView GetView(std::size_t expert_index) const;
  std::vector<FusedNvfp4WeightView> BuildAllViews() const;
  std::size_t total_bytes() const;
  std::size_t num_experts() const;
  bool valid() const;

 private:
  struct Impl;

  explicit MonolithicNvfp4ExpertWeights(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
