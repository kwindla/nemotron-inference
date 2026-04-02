#include "nemotron/monolithic_expert_weights.h"

#include <cuda_runtime.h>

#include <limits>
#include <utility>

namespace nemotron {
namespace {

constexpr std::size_t kNvfp4BlockWidth = 16;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

template <typename T>
void ReleaseBuffer(T** data) {
  if (data != nullptr && *data != nullptr) {
    cudaFree(*data);
    *data = nullptr;
  }
}

bool TryMultiply(std::size_t lhs, std::size_t rhs, std::size_t* result) {
  if (result == nullptr) {
    return false;
  }
  if (lhs != 0 && rhs > (std::numeric_limits<std::size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool ComputeLayoutSizes(
    std::size_t num_experts,
    std::size_t output_rows,
    std::size_t input_cols,
    std::size_t* expert_packed_nbytes,
    std::size_t* expert_block_scales_nbytes,
    std::size_t* total_packed_nbytes,
    std::size_t* total_block_scales_nbytes,
    std::size_t* total_tensor_scales_nbytes) {
  if (num_experts == 0 || output_rows == 0 || input_cols == 0 || input_cols % kNvfp4BlockWidth != 0) {
    return false;
  }

  const std::size_t packed_cols = input_cols / 2u;
  const std::size_t block_cols = input_cols / kNvfp4BlockWidth;
  return TryMultiply(output_rows, packed_cols, expert_packed_nbytes) &&
         TryMultiply(output_rows, block_cols, expert_block_scales_nbytes) &&
         TryMultiply(num_experts, *expert_packed_nbytes, total_packed_nbytes) &&
         TryMultiply(num_experts, *expert_block_scales_nbytes, total_block_scales_nbytes) &&
         TryMultiply(num_experts, sizeof(float), total_tensor_scales_nbytes);
}

}  // namespace

struct MonolithicNvfp4ExpertWeights::Impl {
  std::size_t num_experts = 0;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  std::size_t expert_packed_nbytes = 0;
  std::size_t expert_block_scales_nbytes = 0;
  std::size_t total_packed_nbytes = 0;
  std::size_t total_block_scales_nbytes = 0;
  std::size_t total_tensor_scales_nbytes = 0;
  std::uint8_t* packed_data = nullptr;
  std::uint8_t* block_scales_data = nullptr;
  float* tensor_scales_data = nullptr;
};

std::unique_ptr<MonolithicNvfp4ExpertWeights> MonolithicNvfp4ExpertWeights::Create(
    std::size_t num_experts,
    std::size_t output_rows,
    std::size_t input_cols) {
  auto impl = std::make_unique<Impl>();
  if (!ComputeLayoutSizes(
          num_experts,
          output_rows,
          input_cols,
          &impl->expert_packed_nbytes,
          &impl->expert_block_scales_nbytes,
          &impl->total_packed_nbytes,
          &impl->total_block_scales_nbytes,
          &impl->total_tensor_scales_nbytes)) {
    return nullptr;
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return nullptr;
  }

  impl->num_experts = num_experts;
  impl->output_rows = output_rows;
  impl->input_cols = input_cols;

  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->packed_data), impl->total_packed_nbytes)) ||
      !CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&impl->block_scales_data), impl->total_block_scales_nbytes)) ||
      !CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&impl->tensor_scales_data), impl->total_tensor_scales_nbytes))) {
    ReleaseBuffer(&impl->tensor_scales_data);
    ReleaseBuffer(&impl->block_scales_data);
    ReleaseBuffer(&impl->packed_data);
    return nullptr;
  }

  return std::unique_ptr<MonolithicNvfp4ExpertWeights>(
      new MonolithicNvfp4ExpertWeights(std::move(impl)));
}

MonolithicNvfp4ExpertWeights::MonolithicNvfp4ExpertWeights(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MonolithicNvfp4ExpertWeights::MonolithicNvfp4ExpertWeights(
    MonolithicNvfp4ExpertWeights&&) noexcept = default;

MonolithicNvfp4ExpertWeights& MonolithicNvfp4ExpertWeights::operator=(
    MonolithicNvfp4ExpertWeights&&) noexcept = default;

MonolithicNvfp4ExpertWeights::~MonolithicNvfp4ExpertWeights() {
  if (!impl_) {
    return;
  }
  ReleaseBuffer(&impl_->tensor_scales_data);
  ReleaseBuffer(&impl_->block_scales_data);
  ReleaseBuffer(&impl_->packed_data);
}

bool MonolithicNvfp4ExpertWeights::UploadExpert(
    std::size_t expert_index,
    const std::uint8_t* host_packed,
    std::size_t packed_nbytes,
    const std::uint8_t* host_block_scales,
    std::size_t block_scales_nbytes,
    const float* host_tensor_scale) {
  if (!valid() ||
      expert_index >= impl_->num_experts ||
      host_packed == nullptr ||
      host_block_scales == nullptr ||
      host_tensor_scale == nullptr ||
      packed_nbytes != impl_->expert_packed_nbytes ||
      block_scales_nbytes != impl_->expert_block_scales_nbytes) {
    return false;
  }

  std::uint8_t* expert_packed_data = impl_->packed_data + (expert_index * impl_->expert_packed_nbytes);
  std::uint8_t* expert_block_scales =
      impl_->block_scales_data + (expert_index * impl_->expert_block_scales_nbytes);
  float* expert_tensor_scale = impl_->tensor_scales_data + expert_index;

  if (!CheckCuda(cudaMemcpy(expert_packed_data, host_packed, packed_nbytes, cudaMemcpyHostToDevice))) {
    return false;
  }
  if (!CheckCuda(
          cudaMemcpy(
              expert_block_scales,
              host_block_scales,
              block_scales_nbytes,
              cudaMemcpyHostToDevice))) {
    return false;
  }
  return CheckCuda(
      cudaMemcpy(expert_tensor_scale, host_tensor_scale, sizeof(float), cudaMemcpyHostToDevice));
}

FusedNvfp4WeightView MonolithicNvfp4ExpertWeights::GetView(std::size_t expert_index) const {
  FusedNvfp4WeightView view;
  if (!valid() || expert_index >= impl_->num_experts) {
    return view;
  }

  view.packed_data = impl_->packed_data + (expert_index * impl_->expert_packed_nbytes);
  view.block_scales_data = impl_->block_scales_data + (expert_index * impl_->expert_block_scales_nbytes);
  view.tensor_scale_data = impl_->tensor_scales_data + expert_index;
  view.output_rows = impl_->output_rows;
  view.input_cols = impl_->input_cols;
  return view;
}

std::vector<FusedNvfp4WeightView> MonolithicNvfp4ExpertWeights::BuildAllViews() const {
  std::vector<FusedNvfp4WeightView> views;
  if (!valid()) {
    return views;
  }

  views.reserve(impl_->num_experts);
  for (std::size_t expert_index = 0; expert_index < impl_->num_experts; ++expert_index) {
    views.push_back(GetView(expert_index));
  }
  return views;
}

std::size_t MonolithicNvfp4ExpertWeights::total_bytes() const {
  return impl_ ? (impl_->total_packed_nbytes + impl_->total_block_scales_nbytes +
                  impl_->total_tensor_scales_nbytes)
               : 0;
}

std::size_t MonolithicNvfp4ExpertWeights::num_experts() const {
  return impl_ ? impl_->num_experts : 0;
}

bool MonolithicNvfp4ExpertWeights::valid() const {
  return impl_ != nullptr &&
         impl_->num_experts > 0 &&
         impl_->output_rows > 0 &&
         impl_->input_cols > 0 &&
         impl_->expert_packed_nbytes > 0 &&
         impl_->expert_block_scales_nbytes > 0 &&
         impl_->total_packed_nbytes > 0 &&
         impl_->total_block_scales_nbytes > 0 &&
         impl_->total_tensor_scales_nbytes > 0 &&
         impl_->packed_data != nullptr &&
         impl_->block_scales_data != nullptr &&
         impl_->tensor_scales_data != nullptr;
}

}  // namespace nemotron
