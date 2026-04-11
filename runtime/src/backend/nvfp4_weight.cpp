#include "nemotron/nvfp4_weight.h"

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "nvfp4_routed_padding.h"
#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron::routed_p5_tma {

void* CreateDeviceP5TmaLoadAArray(
    const std::uint8_t* packed_data,
    std::size_t expert_count,
    std::size_t expert_packed_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols);

void* CreateDeviceP5TmaLoadSFAArray(
    const std::uint8_t* scale_data,
    std::size_t expert_count,
    std::size_t expert_scale_stride_bytes,
    std::size_t output_rows,
    std::size_t input_cols);

void DestroyDeviceP5TmaLoadAArray(void** descriptor_array);
void DestroyDeviceP5TmaLoadSFAArray(void** descriptor_array);

std::size_t P5TmaLoadABytes();
std::size_t P5TmaLoadSFABytes();

}  // namespace nemotron::routed_p5_tma

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
  void* p5_tma_load_a = nullptr;
  std::size_t p5_tma_load_a_nbytes = 0;
  void* p5_tma_load_sfa = nullptr;
  std::size_t p5_tma_load_sfa_nbytes = 0;
  float host_tensor_scale = 0.0f;
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
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const auto debug_fail = [&](const char* reason) -> std::unique_ptr<DeviceNvfp4Weight> {
    if (debug) {
      std::cerr << "nvfp4_weight: upload failed for " << descriptor.tensor_name
                << ": " << reason << "\n";
    }
    return nullptr;
  };
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
      descriptor.tensor_scale_nbytes < sizeof(float) ||
      !descriptor.tensor_scale_bytes().valid()) {
    return debug_fail("descriptor validation failed");
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return debug_fail("no CUDA device available");
  }

  auto impl = std::make_unique<Impl>();
  PreparedNvfp4ExecutionWeightHostData prepared;
  if (!PrepareNvfp4WeightForExecution(
          descriptor.op_class,
          descriptor.output_rows,
          descriptor.input_cols,
          descriptor.packed_data,
          descriptor.packed_nbytes,
          descriptor.block_scales_data,
          descriptor.block_scales_nbytes,
          &prepared)) {
    return debug_fail("execution padding preparation failed");
  }

  const std::uint8_t* packed_src =
      prepared.padded ? prepared.packed.data() : descriptor.packed_data;
  const std::uint8_t* block_scales_src =
      prepared.padded ? prepared.block_scales.data() : descriptor.block_scales_data;
  const std::size_t packed_nbytes =
      prepared.padded ? prepared.packed.size() : descriptor.packed_nbytes;
  const std::size_t block_scales_nbytes =
      prepared.padded ? prepared.block_scales.size() : descriptor.block_scales_nbytes;

  impl->output_rows = prepared.output_rows;
  impl->input_cols = prepared.input_cols;
  impl->packed_nbytes = packed_nbytes;
  impl->block_scales_nbytes = block_scales_nbytes;
  impl->tensor_scale_nbytes = descriptor.tensor_scale_nbytes;
  std::memcpy(&impl->host_tensor_scale, descriptor.tensor_scale_data, sizeof(float));
  const std::vector<std::uint8_t> matmul_block_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      block_scales_src,
      impl->output_rows,
      impl->input_cols);
  if (matmul_block_scales.empty()) {
    return debug_fail("scale swizzle failed");
  }
  impl->matmul_block_scales_nbytes = matmul_block_scales.size();

  if (!AllocateAndCopy(packed_src, packed_nbytes, &impl->packed_data) ||
      !AllocateAndCopy(
          block_scales_src,
          block_scales_nbytes,
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
    return debug_fail("device allocation/copy failed");
  }

  impl->p5_tma_load_a = routed_p5_tma::CreateDeviceP5TmaLoadAArray(
      impl->packed_data,
      1,
      impl->packed_nbytes,
      impl->output_rows,
      impl->input_cols);
  if (impl->p5_tma_load_a != nullptr) {
    impl->p5_tma_load_a_nbytes = routed_p5_tma::P5TmaLoadABytes();
  }
  impl->p5_tma_load_sfa = routed_p5_tma::CreateDeviceP5TmaLoadSFAArray(
      impl->matmul_block_scales_data,
      1,
      impl->matmul_block_scales_nbytes,
      impl->output_rows,
      impl->input_cols);
  if (impl->p5_tma_load_sfa != nullptr) {
    impl->p5_tma_load_sfa_nbytes = routed_p5_tma::P5TmaLoadSFABytes();
  }

  return std::unique_ptr<DeviceNvfp4Weight>(new DeviceNvfp4Weight(std::move(impl)));
}

DeviceNvfp4Weight::DeviceNvfp4Weight(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceNvfp4Weight::DeviceNvfp4Weight(DeviceNvfp4Weight&&) noexcept = default;

DeviceNvfp4Weight& DeviceNvfp4Weight::operator=(DeviceNvfp4Weight&&) noexcept = default;

DeviceNvfp4Weight::~DeviceNvfp4Weight() {
  if (!impl_) {
    return;
  }
  routed_p5_tma::DestroyDeviceP5TmaLoadAArray(&impl_->p5_tma_load_a);
  routed_p5_tma::DestroyDeviceP5TmaLoadSFAArray(&impl_->p5_tma_load_sfa);
  ReleaseBuffer(&impl_->tensor_scale_data);
  ReleaseBuffer(&impl_->matmul_block_scales_data);
  ReleaseBuffer(&impl_->block_scales_data);
  ReleaseBuffer(&impl_->packed_data);
}

bool DeviceNvfp4Weight::valid() const {
  return impl_ != nullptr &&
         impl_->output_rows > 0 &&
         impl_->input_cols > 0 &&
         impl_->packed_data != nullptr &&
         impl_->block_scales_data != nullptr &&
         impl_->matmul_block_scales_data != nullptr &&
         impl_->tensor_scale_data != nullptr &&
         impl_->packed_nbytes > 0 &&
         impl_->block_scales_nbytes > 0 &&
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

std::size_t DeviceNvfp4Weight::p5_tma_load_a_nbytes() const {
  return impl_ ? impl_->p5_tma_load_a_nbytes : 0;
}

std::size_t DeviceNvfp4Weight::p5_tma_load_sfa_nbytes() const {
  return impl_ ? impl_->p5_tma_load_sfa_nbytes : 0;
}

float DeviceNvfp4Weight::host_tensor_scale() const {
  return impl_ ? impl_->host_tensor_scale : 0.0f;
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

const void* DeviceNvfp4Weight::p5_tma_load_a() const {
  return impl_ ? impl_->p5_tma_load_a : nullptr;
}

const void* DeviceNvfp4Weight::p5_tma_load_sfa() const {
  return impl_ ? impl_->p5_tma_load_sfa : nullptr;
}

}  // namespace nemotron
