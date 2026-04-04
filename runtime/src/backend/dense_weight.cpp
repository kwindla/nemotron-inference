#include "nemotron/dense_weight.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace nemotron {

struct DeviceDenseWeightFp32::Impl {
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  float* data = nullptr;
};

struct DeviceDenseWeightBf16::Impl {
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  __nv_bfloat16* data = nullptr;
};

namespace {

bool IsFp32Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp32" || storage_dtype == "float32" || storage_dtype == "float";
}

bool IsBf16Storage(const std::string& storage_dtype) {
  return storage_dtype == "bf16" || storage_dtype == "bfloat16";
}

bool IsFp8Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp8_e4m3fn" || storage_dtype == "fp8_e4m3";
}

bool IsSupportedDenseCompute(const std::string& compute_dtype) {
  return compute_dtype == "fp32" ||
         compute_dtype == "float32" ||
         compute_dtype == "float" ||
         compute_dtype == "bf16" ||
         compute_dtype == "bfloat16";
}

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

}  // namespace

std::unique_ptr<DeviceDenseWeightFp32> DeviceDenseWeightFp32::Upload(
    const GemmDescriptor& descriptor) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const auto debug_fail = [&](const char* message) -> std::unique_ptr<DeviceDenseWeightFp32> {
    if (debug) {
      std::cerr << "dense_weight: upload failed for " << descriptor.tensor_name
                << ": " << message << "\n";
    }
    return nullptr;
  };
  if (descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0 ||
      descriptor.layout_tag != "row_major" ||
      descriptor.is_scaled() ||
      !descriptor.packed_bytes().valid()) {
    return debug_fail("descriptor is not a supported dense row-major weight");
  }

  const bool is_fp32 = IsFp32Storage(descriptor.storage_dtype);
  const bool is_bf16 = IsBf16Storage(descriptor.storage_dtype);
  const bool is_fp8 = IsFp8Storage(descriptor.storage_dtype);
  if (!is_fp32 && !is_bf16 && !is_fp8) {
    return debug_fail("storage dtype is unsupported");
  }
  if (!IsSupportedDenseCompute(descriptor.compute_dtype)) {
    return debug_fail("compute dtype is unsupported");
  }

  const std::size_t count = descriptor.output_rows * descriptor.input_cols;
  const std::size_t expected_bytes =
      is_fp32 ? (count * sizeof(float))
              : (is_bf16 ? (count * sizeof(__nv_bfloat16)) : count * sizeof(__nv_fp8_e4m3));
  if (descriptor.packed_nbytes != expected_bytes) {
    return debug_fail("packed byte count does not match logical shape");
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return debug_fail("no CUDA device available");
  }

  std::vector<float> bf16_converted;
  std::vector<float> fp8_converted;
  const void* upload_source = descriptor.packed_data;
  if (is_bf16) {
    bf16_converted.assign(count, 0.0f);
    const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      bf16_converted[i] = __bfloat162float(src[i]);
    }
    upload_source = bf16_converted.data();
  } else if (is_fp8) {
    fp8_converted.assign(count, 0.0f);
    const auto* src = reinterpret_cast<const __nv_fp8_e4m3*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      fp8_converted[i] = static_cast<float>(src[i]);
    }
    upload_source = fp8_converted.data();
  }

  auto impl = std::make_unique<Impl>();
  impl->output_rows = descriptor.output_rows;
  impl->input_cols = descriptor.input_cols;
  const std::size_t upload_bytes = count * sizeof(float);
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->data), upload_bytes))) {
    return debug_fail("cudaMalloc failed");
  }
  if (!CheckCuda(cudaMemcpy(
          impl->data,
          upload_source,
          upload_bytes,
          cudaMemcpyHostToDevice))) {
    cudaFree(impl->data);
    return debug_fail("cudaMemcpy host-to-device failed");
  }

  return std::unique_ptr<DeviceDenseWeightFp32>(new DeviceDenseWeightFp32(std::move(impl)));
}

DeviceDenseWeightFp32::DeviceDenseWeightFp32(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceDenseWeightFp32::DeviceDenseWeightFp32(DeviceDenseWeightFp32&&) noexcept = default;

DeviceDenseWeightFp32& DeviceDenseWeightFp32::operator=(DeviceDenseWeightFp32&&) noexcept = default;

DeviceDenseWeightFp32::~DeviceDenseWeightFp32() {
  if (impl_ && impl_->data != nullptr) {
    cudaFree(impl_->data);
  }
}

bool DeviceDenseWeightFp32::valid() const {
  return impl_ != nullptr && impl_->data != nullptr && impl_->output_rows > 0 && impl_->input_cols > 0;
}

std::size_t DeviceDenseWeightFp32::output_rows() const {
  return impl_ ? impl_->output_rows : 0;
}

std::size_t DeviceDenseWeightFp32::input_cols() const {
  return impl_ ? impl_->input_cols : 0;
}

std::size_t DeviceDenseWeightFp32::numel() const {
  return output_rows() * input_cols();
}

const float* DeviceDenseWeightFp32::data() const {
  return impl_ ? impl_->data : nullptr;
}

std::unique_ptr<DeviceDenseWeightBf16> DeviceDenseWeightBf16::Upload(
    const GemmDescriptor& descriptor) {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const auto debug_fail = [&](const char* message) -> std::unique_ptr<DeviceDenseWeightBf16> {
    if (debug) {
      std::cerr << "dense_weight: upload failed for " << descriptor.tensor_name
                << ": " << message << "\n";
    }
    return nullptr;
  };
  if (descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0 ||
      descriptor.layout_tag != "row_major" ||
      descriptor.is_scaled() ||
      !descriptor.packed_bytes().valid()) {
    return debug_fail("descriptor is not a supported dense row-major weight");
  }

  if (!IsBf16Storage(descriptor.storage_dtype)) {
    return debug_fail("storage dtype is unsupported for bf16 upload");
  }
  if (!IsSupportedDenseCompute(descriptor.compute_dtype)) {
    return debug_fail("compute dtype is unsupported");
  }

  const std::size_t count = descriptor.output_rows * descriptor.input_cols;
  const std::size_t expected_bytes = count * sizeof(__nv_bfloat16);
  if (descriptor.packed_nbytes != expected_bytes) {
    return debug_fail("packed byte count does not match logical shape");
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return debug_fail("no CUDA device available");
  }

  auto impl = std::make_unique<Impl>();
  impl->output_rows = descriptor.output_rows;
  impl->input_cols = descriptor.input_cols;
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->data), expected_bytes))) {
    return debug_fail("cudaMalloc failed");
  }
  if (!CheckCuda(cudaMemcpy(
          impl->data,
          descriptor.packed_data,
          expected_bytes,
          cudaMemcpyHostToDevice))) {
    cudaFree(impl->data);
    return debug_fail("cudaMemcpy host-to-device failed");
  }

  return std::unique_ptr<DeviceDenseWeightBf16>(new DeviceDenseWeightBf16(std::move(impl)));
}

DeviceDenseWeightBf16::DeviceDenseWeightBf16(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceDenseWeightBf16::DeviceDenseWeightBf16(DeviceDenseWeightBf16&&) noexcept = default;

DeviceDenseWeightBf16& DeviceDenseWeightBf16::operator=(DeviceDenseWeightBf16&&) noexcept = default;

DeviceDenseWeightBf16::~DeviceDenseWeightBf16() {
  if (impl_ && impl_->data != nullptr) {
    cudaFree(impl_->data);
  }
}

bool DeviceDenseWeightBf16::valid() const {
  return impl_ != nullptr && impl_->data != nullptr && impl_->output_rows > 0 && impl_->input_cols > 0;
}

std::size_t DeviceDenseWeightBf16::output_rows() const {
  return impl_ ? impl_->output_rows : 0;
}

std::size_t DeviceDenseWeightBf16::input_cols() const {
  return impl_ ? impl_->input_cols : 0;
}

std::size_t DeviceDenseWeightBf16::numel() const {
  return output_rows() * input_cols();
}

const __nv_bfloat16* DeviceDenseWeightBf16::data() const {
  return impl_ ? impl_->data : nullptr;
}

}  // namespace nemotron
