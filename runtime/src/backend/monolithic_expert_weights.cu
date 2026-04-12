#include "nemotron/monolithic_expert_weights.h"

#include <cuda_runtime.h>

#include <cstring>
#include <limits>
#include <utility>

#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/routed_expert_runtime.h"
#include "routed_p5_tma_descriptor.cuh"

namespace nemotron {
namespace {

constexpr std::size_t kNvfp4BlockWidth = 16;
constexpr std::size_t kRequiredAlignmentBytes = 16;

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

std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const std::size_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

bool ComputeLayoutSizes(
    std::size_t num_experts,
    std::size_t output_rows,
    std::size_t input_cols,
    bool store_row_major_block_scales,
    std::size_t* expert_packed_nbytes,
    std::size_t* expert_block_scales_nbytes,
    std::size_t* expert_matmul_block_scales_nbytes,
    std::size_t* expert_tensor_scale_stride_nbytes,
    std::size_t* total_packed_nbytes,
    std::size_t* total_block_scales_nbytes,
    std::size_t* total_matmul_block_scales_nbytes,
    std::size_t* total_tensor_scales_nbytes) {
  if (num_experts == 0 || output_rows == 0 || input_cols == 0 || input_cols % kNvfp4BlockWidth != 0) {
    return false;
  }

  const std::size_t packed_cols = input_cols / 2u;
  const std::size_t block_cols = input_cols / kNvfp4BlockWidth;
  const std::size_t tensor_scale_stride_nbytes =
      RoundUp(sizeof(float), kRequiredAlignmentBytes);
  if (expert_matmul_block_scales_nbytes == nullptr ||
      expert_tensor_scale_stride_nbytes == nullptr) {
    return false;
  }
  *expert_matmul_block_scales_nbytes = ExecutionNvfp4ScaleBytes(output_rows, input_cols);
  *expert_tensor_scale_stride_nbytes = tensor_scale_stride_nbytes;
  if (expert_block_scales_nbytes == nullptr || total_block_scales_nbytes == nullptr) {
    return false;
  }
  *expert_block_scales_nbytes = store_row_major_block_scales ? (output_rows * block_cols) : 0;
  return TryMultiply(output_rows, packed_cols, expert_packed_nbytes) &&
         TryMultiply(num_experts, *expert_matmul_block_scales_nbytes, total_matmul_block_scales_nbytes) &&
         TryMultiply(num_experts, *expert_packed_nbytes, total_packed_nbytes) &&
         TryMultiply(num_experts, *expert_block_scales_nbytes, total_block_scales_nbytes) &&
         TryMultiply(num_experts, *expert_tensor_scale_stride_nbytes, total_tensor_scales_nbytes);
}

}  // namespace

struct MonolithicNvfp4ExpertWeights::Impl {
  std::size_t num_experts = 0;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  std::size_t expert_packed_nbytes = 0;
  std::size_t expert_block_scales_nbytes = 0;
  std::size_t expert_matmul_block_scales_nbytes = 0;
  std::size_t expert_tensor_scale_stride_nbytes = 0;
  std::size_t total_packed_nbytes = 0;
  std::size_t total_block_scales_nbytes = 0;
  std::size_t total_matmul_block_scales_nbytes = 0;
  std::size_t total_tensor_scales_nbytes = 0;
  std::size_t total_p5_tma_load_a_descriptor_nbytes = 0;
  std::size_t total_p5_tma_load_sfa_descriptor_nbytes = 0;
  std::size_t total_p5_tma_load_b_descriptor_nbytes = 0;
  std::size_t total_p5_tma_load_sfb_descriptor_nbytes = 0;
  std::uint8_t* packed_data = nullptr;
  std::uint8_t* block_scales_data = nullptr;
  std::uint8_t* matmul_block_scales_data = nullptr;
  std::uint8_t* tensor_scales_data = nullptr;
  void* p5_tma_load_a_descriptors = nullptr;
  void* p5_tma_load_sfa_descriptors = nullptr;
  void* p5_tma_load_b_descriptors = nullptr;
  void* p5_tma_load_sfb_descriptors = nullptr;
  std::vector<float> host_tensor_scales;
  bool store_row_major_block_scales = true;
};

std::unique_ptr<MonolithicNvfp4ExpertWeights> MonolithicNvfp4ExpertWeights::Create(
    std::size_t num_experts,
    std::size_t output_rows,
    std::size_t input_cols,
    bool store_row_major_block_scales) {
  auto impl = std::make_unique<Impl>();
  if (!ComputeLayoutSizes(
          num_experts,
          output_rows,
          input_cols,
          store_row_major_block_scales,
          &impl->expert_packed_nbytes,
          &impl->expert_block_scales_nbytes,
          &impl->expert_matmul_block_scales_nbytes,
          &impl->expert_tensor_scale_stride_nbytes,
          &impl->total_packed_nbytes,
          &impl->total_block_scales_nbytes,
          &impl->total_matmul_block_scales_nbytes,
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
  impl->host_tensor_scales.assign(num_experts, 0.0f);
  impl->store_row_major_block_scales = store_row_major_block_scales;

  const bool allocate_block_scales =
      impl->total_block_scales_nbytes != 0 && impl->store_row_major_block_scales;
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->packed_data), impl->total_packed_nbytes)) ||
      (allocate_block_scales &&
       !CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&impl->block_scales_data), impl->total_block_scales_nbytes))) ||
      !CheckCuda(
          cudaMalloc(
              reinterpret_cast<void**>(&impl->matmul_block_scales_data),
              impl->total_matmul_block_scales_nbytes)) ||
      !CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&impl->tensor_scales_data), impl->total_tensor_scales_nbytes))) {
    ReleaseBuffer(&impl->tensor_scales_data);
    ReleaseBuffer(&impl->matmul_block_scales_data);
    ReleaseBuffer(&impl->block_scales_data);
    ReleaseBuffer(&impl->packed_data);
    return nullptr;
  }
  impl->p5_tma_load_a_descriptors = routed_p5_tma::CreateDeviceP5TmaLoadAArray(
      impl->packed_data,
      impl->num_experts,
      impl->expert_packed_nbytes,
      impl->output_rows,
      impl->input_cols);
  if (impl->p5_tma_load_a_descriptors != nullptr) {
    impl->total_p5_tma_load_a_descriptor_nbytes =
        num_experts * sizeof(routed_p5_tma::P5TmaLoadA);
  }
  impl->p5_tma_load_sfa_descriptors = routed_p5_tma::CreateDeviceP5TmaLoadSFAArray(
      impl->matmul_block_scales_data,
      impl->num_experts,
      impl->expert_matmul_block_scales_nbytes,
      impl->output_rows,
      impl->input_cols);
  if (impl->p5_tma_load_sfa_descriptors != nullptr) {
    impl->total_p5_tma_load_sfa_descriptor_nbytes =
        num_experts * sizeof(routed_p5_tma::P5TmaLoadSFA);
  }
  impl->p5_tma_load_b_descriptors = routed_p5_tma::CreateDeviceP5TmaLoadBArray(
      impl->packed_data,
      impl->num_experts,
      impl->expert_packed_nbytes,
      impl->output_rows,
      impl->input_cols);
  if (impl->p5_tma_load_b_descriptors != nullptr) {
    impl->total_p5_tma_load_b_descriptor_nbytes =
        num_experts * sizeof(routed_p5_tma::P5TmaLoadB);
  }
  impl->p5_tma_load_sfb_descriptors = routed_p5_tma::CreateDeviceP5TmaLoadSFBArray(
      impl->matmul_block_scales_data,
      impl->num_experts,
      impl->expert_matmul_block_scales_nbytes,
      impl->output_rows,
      impl->input_cols);
  if (impl->p5_tma_load_sfb_descriptors != nullptr) {
    impl->total_p5_tma_load_sfb_descriptor_nbytes =
        num_experts * sizeof(routed_p5_tma::P5TmaLoadSFB);
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
  routed_p5_tma::DestroyDeviceP5TmaLoadAArray(&impl_->p5_tma_load_a_descriptors);
  routed_p5_tma::DestroyDeviceP5TmaLoadSFAArray(&impl_->p5_tma_load_sfa_descriptors);
  routed_p5_tma::DestroyDeviceP5TmaLoadBArray(&impl_->p5_tma_load_b_descriptors);
  routed_p5_tma::DestroyDeviceP5TmaLoadSFBArray(&impl_->p5_tma_load_sfb_descriptors);
  ReleaseBuffer(&impl_->tensor_scales_data);
  ReleaseBuffer(&impl_->matmul_block_scales_data);
  ReleaseBuffer(&impl_->block_scales_data);
  ReleaseBuffer(&impl_->packed_data);
}

bool MonolithicNvfp4ExpertWeights::UploadExpert(
    std::size_t expert_index,
    const std::uint8_t* host_packed,
    std::size_t packed_nbytes,
    const std::uint8_t* host_block_scales,
    std::size_t block_scales_nbytes,
    const float* host_tensor_scale,
    std::size_t host_output_rows,
    std::size_t host_input_cols) {
  const std::size_t source_output_rows = host_output_rows == 0 ? impl_->output_rows : host_output_rows;
  const std::size_t source_input_cols = host_input_cols == 0 ? impl_->input_cols : host_input_cols;
  const std::size_t source_packed_nbytes = (source_output_rows * source_input_cols) / 2u;
  const std::size_t source_block_scales_nbytes =
      source_output_rows * (source_input_cols / kNvfp4BlockWidth);
  if (!valid() ||
      expert_index >= impl_->num_experts ||
      host_packed == nullptr ||
      host_block_scales == nullptr ||
      host_tensor_scale == nullptr ||
      source_output_rows == 0 ||
      source_output_rows > impl_->output_rows ||
      source_input_cols == 0 ||
      source_input_cols > impl_->input_cols ||
      (source_input_cols % kNvfp4BlockWidth) != 0 ||
      packed_nbytes != source_packed_nbytes ||
      block_scales_nbytes != source_block_scales_nbytes) {
    return false;
  }

  std::uint8_t* expert_packed_data = impl_->packed_data + (expert_index * impl_->expert_packed_nbytes);
  std::uint8_t* expert_block_scales =
      (impl_->store_row_major_block_scales && impl_->block_scales_data != nullptr)
          ? (impl_->block_scales_data + (expert_index * impl_->expert_block_scales_nbytes))
          : nullptr;
  std::uint8_t* expert_matmul_block_scales =
      impl_->matmul_block_scales_data + (expert_index * impl_->expert_matmul_block_scales_nbytes);
  std::uint8_t* expert_tensor_scale =
      impl_->tensor_scales_data + (expert_index * impl_->expert_tensor_scale_stride_nbytes);

  const std::size_t execution_block_scales_nbytes =
      impl_->output_rows * (impl_->input_cols / kNvfp4BlockWidth);
  std::vector<std::uint8_t> execution_packed(impl_->expert_packed_nbytes, 0u);
  std::vector<std::uint8_t> execution_block_scales(execution_block_scales_nbytes, 0u);
  const std::size_t source_packed_row_bytes = source_input_cols / 2u;
  const std::size_t source_blocks_per_row = source_input_cols / kNvfp4BlockWidth;
  const std::size_t destination_packed_row_bytes = impl_->input_cols / 2u;
  const std::size_t destination_blocks_per_row = impl_->input_cols / kNvfp4BlockWidth;
  for (std::size_t row = 0; row < source_output_rows; ++row) {
    std::memcpy(
        execution_packed.data() + (row * destination_packed_row_bytes),
        host_packed + (row * source_packed_row_bytes),
        source_packed_row_bytes);
    std::memcpy(
        execution_block_scales.data() + (row * destination_blocks_per_row),
        host_block_scales + (row * source_blocks_per_row),
        source_blocks_per_row);
  }

  const std::vector<std::uint8_t> matmul_block_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      execution_block_scales.data(),
      impl_->output_rows,
      impl_->input_cols);
  if (matmul_block_scales.size() != impl_->expert_matmul_block_scales_nbytes) {
    return false;
  }

  if (!CheckCuda(cudaMemcpy(
          expert_packed_data,
          execution_packed.data(),
          execution_packed.size(),
          cudaMemcpyHostToDevice))) {
    return false;
  }
  if (expert_block_scales != nullptr &&
      !CheckCuda(
          cudaMemcpy(
              expert_block_scales,
              execution_block_scales.data(),
              execution_block_scales.size(),
              cudaMemcpyHostToDevice))) {
    return false;
  }
  if (!CheckCuda(
          cudaMemcpy(
              expert_matmul_block_scales,
              matmul_block_scales.data(),
              matmul_block_scales.size(),
              cudaMemcpyHostToDevice))) {
    return false;
  }
  if (!CheckCuda(
          cudaMemcpy(expert_tensor_scale, host_tensor_scale, sizeof(float), cudaMemcpyHostToDevice))) {
    return false;
  }
  impl_->host_tensor_scales[expert_index] = *host_tensor_scale;
  return true;
}

float MonolithicNvfp4ExpertWeights::host_tensor_scale(std::size_t expert_index) const {
  if (!valid() ||
      expert_index >= impl_->num_experts ||
      expert_index >= impl_->host_tensor_scales.size()) {
    return 0.0f;
  }
  return impl_->host_tensor_scales[expert_index];
}

FusedNvfp4WeightView MonolithicNvfp4ExpertWeights::GetView(std::size_t expert_index) const {
  FusedNvfp4WeightView view;
  if (!valid() || expert_index >= impl_->num_experts) {
    return view;
  }

  view.packed_data = impl_->packed_data + (expert_index * impl_->expert_packed_nbytes);
  view.block_scales_data =
      (impl_->store_row_major_block_scales && impl_->block_scales_data != nullptr)
          ? (impl_->block_scales_data + (expert_index * impl_->expert_block_scales_nbytes))
          : nullptr;
  view.matmul_block_scales_data =
      impl_->matmul_block_scales_data + (expert_index * impl_->expert_matmul_block_scales_nbytes);
  view.tensor_scale_data = reinterpret_cast<const float*>(
      impl_->tensor_scales_data + (expert_index * impl_->expert_tensor_scale_stride_nbytes));
  view.output_rows = impl_->output_rows;
  view.input_cols = impl_->input_cols;
  if (impl_->p5_tma_load_a_descriptors != nullptr) {
    const auto* p5_tma_load_a =
        static_cast<const routed_p5_tma::P5TmaLoadA*>(impl_->p5_tma_load_a_descriptors);
    view.p5_tma_load_a = p5_tma_load_a + expert_index;
  }
  if (impl_->p5_tma_load_sfa_descriptors != nullptr) {
    const auto* p5_tma_load_sfa =
        static_cast<const routed_p5_tma::P5TmaLoadSFA*>(impl_->p5_tma_load_sfa_descriptors);
    view.p5_tma_load_sfa = p5_tma_load_sfa + expert_index;
  }
  if (impl_->p5_tma_load_b_descriptors != nullptr) {
    const auto* p5_tma_load_b =
        static_cast<const routed_p5_tma::P5TmaLoadB*>(impl_->p5_tma_load_b_descriptors);
    view.p5_tma_load_b = p5_tma_load_b + expert_index;
  }
  if (impl_->p5_tma_load_sfb_descriptors != nullptr) {
    const auto* p5_tma_load_sfb =
        static_cast<const routed_p5_tma::P5TmaLoadSFB*>(impl_->p5_tma_load_sfb_descriptors);
    view.p5_tma_load_sfb = p5_tma_load_sfb + expert_index;
  }
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
                  impl_->total_matmul_block_scales_nbytes +
                  impl_->total_tensor_scales_nbytes +
                  impl_->total_p5_tma_load_a_descriptor_nbytes +
                  impl_->total_p5_tma_load_sfa_descriptor_nbytes +
                  impl_->total_p5_tma_load_b_descriptor_nbytes +
                  impl_->total_p5_tma_load_sfb_descriptor_nbytes)
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
         impl_->expert_matmul_block_scales_nbytes > 0 &&
         impl_->expert_tensor_scale_stride_nbytes >= sizeof(float) &&
         impl_->total_packed_nbytes > 0 &&
         impl_->total_matmul_block_scales_nbytes > 0 &&
         impl_->total_tensor_scales_nbytes > 0 &&
         impl_->host_tensor_scales.size() == impl_->num_experts &&
         impl_->packed_data != nullptr &&
         (!impl_->store_row_major_block_scales ||
          (impl_->expert_block_scales_nbytes > 0 &&
           impl_->total_block_scales_nbytes > 0 &&
           impl_->block_scales_data != nullptr)) &&
         impl_->matmul_block_scales_data != nullptr &&
         impl_->tensor_scales_data != nullptr;
}

}  // namespace nemotron
