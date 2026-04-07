#include "nemotron/dense_weight.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstring>

#include "storage_conversion.h"

namespace nemotron {

struct DeviceDenseWeightFp32::Impl {
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  float* data = nullptr;
  bool owns_memory = true;
};

namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

std::optional<PackedFloatStorage> PackedStorageForDenseDescriptor(const GemmDescriptor& descriptor) {
  if (descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0 ||
      descriptor.layout_tag != "row_major" ||
      descriptor.is_scaled() ||
      !descriptor.packed_bytes().valid()) {
    return std::nullopt;
  }
  if (descriptor.storage_dtype == "fp32") {
    return PackedFloatStorage::kFp32;
  }
  if (descriptor.storage_dtype == "bf16" || descriptor.storage_dtype == "bfloat16") {
    return PackedFloatStorage::kBf16;
  }
  if (descriptor.storage_dtype == "fp8_e4m3fn" || descriptor.storage_dtype == "fp8_e4m3") {
    return PackedFloatStorage::kFp8E4M3;
  }
  return std::nullopt;
}

}  // namespace

std::optional<std::vector<float>> ReadDenseWeightToHostFp32(
    const GemmDescriptor& descriptor,
    float storage_scale) {
  const auto storage = PackedStorageForDenseDescriptor(descriptor);
  if (!storage.has_value()) {
    return std::nullopt;
  }
  const std::size_t count = descriptor.output_rows * descriptor.input_cols;
  std::vector<float> output(count, 0.0f);
  switch (*storage) {
    case PackedFloatStorage::kFp32:
      if (descriptor.packed_nbytes != count * sizeof(float)) {
        return std::nullopt;
      }
      std::memcpy(output.data(), descriptor.packed_data, descriptor.packed_nbytes);
      break;
    case PackedFloatStorage::kBf16: {
      if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
        return std::nullopt;
      }
      const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
      for (std::size_t i = 0; i < count; ++i) {
        output[i] = __bfloat162float(src[i]) * storage_scale;
      }
      break;
    }
    case PackedFloatStorage::kFp8E4M3: {
      if (descriptor.packed_nbytes != count * sizeof(__nv_fp8_e4m3)) {
        return std::nullopt;
      }
      const auto* src = reinterpret_cast<const __nv_fp8_e4m3*>(descriptor.packed_data);
      for (std::size_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(src[i]) * storage_scale;
      }
      break;
    }
  }
  return output;
}

std::unique_ptr<DeviceDenseWeightFp32> DeviceDenseWeightFp32::Upload(
    const GemmDescriptor& descriptor,
    float storage_scale) {
  const auto storage = PackedStorageForDenseDescriptor(descriptor);
  if (!storage.has_value()) {
    return nullptr;
  }
  if (descriptor.compute_dtype != "fp32" &&
      descriptor.compute_dtype != "float32" &&
      descriptor.compute_dtype != "float" &&
      descriptor.compute_dtype != "bf16" &&
      descriptor.compute_dtype != "bfloat16") {
    return nullptr;
  }

  const std::size_t count = descriptor.output_rows * descriptor.input_cols;
  const std::size_t expected_bytes =
      *storage == PackedFloatStorage::kFp32 ? (count * sizeof(float))
      : (*storage == PackedFloatStorage::kBf16 ? (count * sizeof(__nv_bfloat16))
                                               : count * sizeof(__nv_fp8_e4m3));
  if (descriptor.packed_nbytes != expected_bytes) {
    return nullptr;
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->output_rows = descriptor.output_rows;
  impl->input_cols = descriptor.input_cols;
  const std::size_t upload_bytes = count * sizeof(float);
  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->data), upload_bytes))) {
    return nullptr;
  }
  if (!UploadPackedFloatToDeviceFp32(
          descriptor.packed_data,
          count,
          *storage,
          storage_scale,
          impl->data)) {
    cudaFree(impl->data);
    return nullptr;
  }

  return std::unique_ptr<DeviceDenseWeightFp32>(new DeviceDenseWeightFp32(std::move(impl)));
}

std::unique_ptr<DeviceDenseWeightFp32> DeviceDenseWeightFp32::CreateView(
    std::size_t output_rows,
    std::size_t input_cols,
    float* data) {
  if (output_rows == 0 || input_cols == 0 || data == nullptr) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->output_rows = output_rows;
  impl->input_cols = input_cols;
  impl->data = data;
  impl->owns_memory = false;
  return std::unique_ptr<DeviceDenseWeightFp32>(new DeviceDenseWeightFp32(std::move(impl)));
}

DeviceDenseWeightFp32::DeviceDenseWeightFp32(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceDenseWeightFp32::DeviceDenseWeightFp32(DeviceDenseWeightFp32&&) noexcept = default;

DeviceDenseWeightFp32& DeviceDenseWeightFp32::operator=(DeviceDenseWeightFp32&&) noexcept = default;

DeviceDenseWeightFp32::~DeviceDenseWeightFp32() {
  if (impl_ && impl_->owns_memory && impl_->data != nullptr) {
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

}  // namespace nemotron
