#include "nemotron/nvfp4_weight.h"

#include <cuda_runtime.h>

#include <vector>

#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {

struct DeviceNvfp4Weight::Impl {
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  std::uint8_t* packed_data = nullptr;
  std::size_t packed_nbytes = 0;
  std::uint8_t* block_scales_data = nullptr;
  std::size_t block_scales_nbytes = 0;
  std::uint8_t* matmul_block_scales_data = nullptr;
  std::size_t matmul_block_scales_nbytes = 0;
  std::uint8_t* tensor_scale_data = nullptr;
  std::size_t tensor_scale_nbytes = 0;
  bool owns_memory = true;
};

namespace {

constexpr std::size_t kBlockWidth = 16;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool AllocateAndCopy(const std::uint8_t* src, std::size_t nbytes, std::uint8_t** dst) {
  if (dst == nullptr || src == nullptr || nbytes == 0) {
    return false;
  }
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(dst), nbytes))) {
    return false;
  }
  if (!CheckCuda(cudaMemcpy(*dst, src, nbytes, cudaMemcpyHostToDevice))) {
    cudaFree(*dst);
    *dst = nullptr;
    return false;
  }
  return true;
}

void ReleaseBuffer(std::uint8_t** data) {
  if (data != nullptr && *data != nullptr) {
    cudaFree(*data);
    *data = nullptr;
  }
}

}  // namespace

std::unique_ptr<DeviceNvfp4Weight> DeviceNvfp4Weight::Upload(const GemmDescriptor& descriptor) {
  const std::size_t expected_block_scale_nbytes =
      descriptor.input_cols % kBlockWidth == 0 ? descriptor.output_rows * (descriptor.input_cols / kBlockWidth) : 0;
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0 ||
      descriptor.storage_dtype != "nvfp4_e2m1" ||
      descriptor.compute_dtype != "fp32_accum" ||
      descriptor.layout_tag != "cublaslt_fp4_tn_v1" ||
      !descriptor.packed_bytes().valid() ||
      descriptor.block_scales_nbytes != expected_block_scale_nbytes ||
      !descriptor.block_scales_bytes().valid() ||
      !descriptor.tensor_scale_bytes().valid()) {
    return nullptr;
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->output_rows = descriptor.output_rows;
  impl->input_cols = descriptor.input_cols;
  impl->packed_nbytes = descriptor.packed_nbytes;
  impl->block_scales_nbytes = descriptor.block_scales_nbytes;
  impl->tensor_scale_nbytes = descriptor.tensor_scale_nbytes;
  const std::vector<std::uint8_t> matmul_block_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      descriptor.block_scales_data,
      descriptor.output_rows,
      descriptor.input_cols);
  if (matmul_block_scales.empty()) {
    return nullptr;
  }
  impl->matmul_block_scales_nbytes = matmul_block_scales.size();

  if (!AllocateAndCopy(descriptor.packed_data, descriptor.packed_nbytes, &impl->packed_data) ||
      !AllocateAndCopy(
          descriptor.block_scales_data,
          descriptor.block_scales_nbytes,
          &impl->block_scales_data) ||
      !AllocateAndCopy(
          matmul_block_scales.data(),
          matmul_block_scales.size(),
          &impl->matmul_block_scales_data) ||
      !AllocateAndCopy(
          descriptor.tensor_scale_data,
          descriptor.tensor_scale_nbytes,
          &impl->tensor_scale_data)) {
    ReleaseBuffer(&impl->tensor_scale_data);
    ReleaseBuffer(&impl->matmul_block_scales_data);
    ReleaseBuffer(&impl->block_scales_data);
    ReleaseBuffer(&impl->packed_data);
    return nullptr;
  }

  return std::unique_ptr<DeviceNvfp4Weight>(new DeviceNvfp4Weight(std::move(impl)));
}

std::unique_ptr<DeviceNvfp4Weight> DeviceNvfp4Weight::CreateView(
    std::size_t output_rows,
    std::size_t input_cols,
    std::uint8_t* packed_data,
    std::size_t packed_nbytes,
    std::uint8_t* block_scales_data,
    std::size_t block_scales_nbytes,
    std::uint8_t* matmul_block_scales_data,
    std::size_t matmul_block_scales_nbytes,
    std::uint8_t* tensor_scale_data,
    std::size_t tensor_scale_nbytes) {
  const bool has_raw_block_scales =
      (block_scales_data != nullptr && block_scales_nbytes > 0) ||
      (block_scales_data == nullptr && block_scales_nbytes == 0);
  if (output_rows == 0 || input_cols == 0 ||
      packed_data == nullptr || packed_nbytes == 0 ||
      !has_raw_block_scales ||
      matmul_block_scales_data == nullptr || matmul_block_scales_nbytes == 0 ||
      tensor_scale_data == nullptr || tensor_scale_nbytes == 0) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->output_rows = output_rows;
  impl->input_cols = input_cols;
  impl->packed_data = packed_data;
  impl->packed_nbytes = packed_nbytes;
  impl->block_scales_data = block_scales_data;
  impl->block_scales_nbytes = block_scales_nbytes;
  impl->matmul_block_scales_data = matmul_block_scales_data;
  impl->matmul_block_scales_nbytes = matmul_block_scales_nbytes;
  impl->tensor_scale_data = tensor_scale_data;
  impl->tensor_scale_nbytes = tensor_scale_nbytes;
  impl->owns_memory = false;
  return std::unique_ptr<DeviceNvfp4Weight>(new DeviceNvfp4Weight(std::move(impl)));
}

DeviceNvfp4Weight::DeviceNvfp4Weight(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceNvfp4Weight::DeviceNvfp4Weight(DeviceNvfp4Weight&&) noexcept = default;

DeviceNvfp4Weight& DeviceNvfp4Weight::operator=(DeviceNvfp4Weight&&) noexcept = default;

DeviceNvfp4Weight::~DeviceNvfp4Weight() {
  if (!impl_ || !impl_->owns_memory) {
    return;
  }
  ReleaseBuffer(&impl_->tensor_scale_data);
  ReleaseBuffer(&impl_->matmul_block_scales_data);
  ReleaseBuffer(&impl_->block_scales_data);
  ReleaseBuffer(&impl_->packed_data);
}

bool DeviceNvfp4Weight::valid() const {
  if (impl_ == nullptr) {
    return false;
  }
  const bool has_optional_raw_block_scales =
      (impl_->block_scales_data != nullptr && impl_->block_scales_nbytes > 0) ||
      (impl_->block_scales_data == nullptr && impl_->block_scales_nbytes == 0);
  return impl_->output_rows > 0 &&
         impl_->input_cols > 0 &&
         impl_->packed_data != nullptr &&
         has_optional_raw_block_scales &&
         impl_->matmul_block_scales_data != nullptr &&
         impl_->tensor_scale_data != nullptr &&
         impl_->packed_nbytes > 0 &&
         impl_->matmul_block_scales_nbytes > 0 &&
         impl_->tensor_scale_nbytes > 0;
}

std::size_t DeviceNvfp4Weight::output_rows() const {
  return impl_ ? impl_->output_rows : 0;
}

std::size_t DeviceNvfp4Weight::input_cols() const {
  return impl_ ? impl_->input_cols : 0;
}

std::size_t DeviceNvfp4Weight::packed_nbytes() const {
  return impl_ ? impl_->packed_nbytes : 0;
}

std::size_t DeviceNvfp4Weight::block_scales_nbytes() const {
  return impl_ ? impl_->block_scales_nbytes : 0;
}

std::size_t DeviceNvfp4Weight::matmul_block_scales_nbytes() const {
  return impl_ ? impl_->matmul_block_scales_nbytes : 0;
}

std::size_t DeviceNvfp4Weight::tensor_scale_nbytes() const {
  return impl_ ? impl_->tensor_scale_nbytes : 0;
}

const std::uint8_t* DeviceNvfp4Weight::packed_data() const {
  return impl_ ? impl_->packed_data : nullptr;
}

const std::uint8_t* DeviceNvfp4Weight::block_scales_data() const {
  return impl_ ? impl_->block_scales_data : nullptr;
}

const std::uint8_t* DeviceNvfp4Weight::matmul_block_scales_data() const {
  return impl_ ? impl_->matmul_block_scales_data : nullptr;
}

const std::uint8_t* DeviceNvfp4Weight::tensor_scale_data() const {
  return impl_ ? impl_->tensor_scale_data : nullptr;
}

}  // namespace nemotron
