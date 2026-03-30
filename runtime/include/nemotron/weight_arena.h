#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nemotron/weight_arena_plan.h"

namespace nemotron {

struct WeightArenaBufferView {
  const WeightArenaBufferPlacement* placement = nullptr;
  ByteRangeView bytes;
};

struct WeightArenaTensorView {
  const WeightArenaTensorPlacement* placement = nullptr;
  ByteRangeView packed_bytes;
  std::vector<WeightArenaBufferView> auxiliary_buffers;
};

class WeightArena {
 public:
  static std::unique_ptr<WeightArena> CreateFromPlan(const WeightArenaPlan& plan);

  WeightArena(WeightArena&&) noexcept;
  WeightArena& operator=(WeightArena&&) noexcept;
  ~WeightArena();

  WeightArena(const WeightArena&) = delete;
  WeightArena& operator=(const WeightArena&) = delete;

  bool valid() const;
  std::size_t size_bytes() const;
  std::size_t base_alignment_bytes() const;
  const WeightArenaPlan& plan() const;
  std::optional<WeightArenaTensorView> FindTensor(const std::string& tensor_name) const;

 private:
  struct Impl;

  explicit WeightArena(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
