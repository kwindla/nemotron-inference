#include "nemotron/embedding_table.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

namespace nemotron {

struct DeviceEmbeddingTableFp32::Impl {
  std::size_t vocab_size = 0;
  std::size_t embedding_dim = 0;
  float* data = nullptr;
};

namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

const char* CudaErrorName(cudaError_t status) {
  return cudaGetErrorString(status);
}

__global__ void EmbeddingLookupKernel(
    const float* table,
    const std::int32_t* token_ids,
    float* output,
    std::size_t token_count,
    std::size_t embedding_dim) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * embedding_dim;
  if (index >= total) {
    return;
  }
  const std::size_t token_index = index / embedding_dim;
  const std::size_t dim_index = index % embedding_dim;
  const std::int32_t token_id = token_ids[token_index];
  output[index] = table[static_cast<std::size_t>(token_id) * embedding_dim + dim_index];
}

}  // namespace

std::unique_ptr<DeviceEmbeddingTableFp32> DeviceEmbeddingTableFp32::Upload(
    const EmbeddingDescriptor& descriptor) {
  if (descriptor.vocab_size == 0 ||
      descriptor.embedding_dim == 0 ||
      descriptor.layout_tag != "row_major" ||
      !descriptor.packed_bytes().valid()) {
    std::cerr << "embedding_table: invalid embedding descriptor\n";
    return nullptr;
  }

  const bool is_fp32 = descriptor.storage_dtype == "fp32";
  const bool is_bf16 =
      descriptor.storage_dtype == "bf16" || descriptor.storage_dtype == "bfloat16";
  if (!is_fp32 && !is_bf16) {
    std::cerr << "embedding_table: unsupported storage dtype " << descriptor.storage_dtype << "\n";
    return nullptr;
  }
  if (descriptor.compute_dtype != "fp32" &&
      descriptor.compute_dtype != "float32" &&
      descriptor.compute_dtype != "float" &&
      descriptor.compute_dtype != "bf16" &&
      descriptor.compute_dtype != "bfloat16") {
    std::cerr << "embedding_table: unsupported compute dtype " << descriptor.compute_dtype << "\n";
    return nullptr;
  }

  const std::size_t element_count = descriptor.vocab_size * descriptor.embedding_dim;
  const std::size_t expected_bytes =
      is_fp32 ? (element_count * sizeof(float)) : (element_count * sizeof(__nv_bfloat16));
  if (descriptor.packed_nbytes != expected_bytes) {
    std::cerr << "embedding_table: packed byte size mismatch expected=" << expected_bytes
              << " actual=" << descriptor.packed_nbytes << "\n";
    return nullptr;
  }

  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (!CheckCuda(device_status) || device_count <= 0) {
    std::cerr << "embedding_table: no CUDA device available ("
              << CudaErrorName(device_status) << ")\n";
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->vocab_size = descriptor.vocab_size;
  impl->embedding_dim = descriptor.embedding_dim;
  const std::size_t upload_bytes = element_count * sizeof(float);
  const cudaError_t alloc_status = cudaMalloc(reinterpret_cast<void**>(&impl->data), upload_bytes);
  if (!CheckCuda(alloc_status)) {
    std::cerr << "embedding_table: cudaMalloc failed for " << upload_bytes
              << " bytes (" << CudaErrorName(alloc_status) << ")\n";
    return nullptr;
  }

  std::vector<float> bf16_converted;
  const void* upload_source = descriptor.packed_data;
  if (is_bf16) {
    bf16_converted.assign(element_count, 0.0f);
    const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < element_count; ++i) {
      bf16_converted[i] = __bfloat162float(src[i]);
    }
    upload_source = bf16_converted.data();
  }

  const cudaError_t memcpy_status = cudaMemcpy(
      impl->data,
      upload_source,
      upload_bytes,
      cudaMemcpyHostToDevice);
  if (!CheckCuda(memcpy_status)) {
    std::cerr << "embedding_table: cudaMemcpy upload failed ("
              << CudaErrorName(memcpy_status) << ")\n";
    cudaFree(impl->data);
    return nullptr;
  }

  return std::unique_ptr<DeviceEmbeddingTableFp32>(new DeviceEmbeddingTableFp32(std::move(impl)));
}

DeviceEmbeddingTableFp32::DeviceEmbeddingTableFp32(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceEmbeddingTableFp32::DeviceEmbeddingTableFp32(DeviceEmbeddingTableFp32&&) noexcept = default;

DeviceEmbeddingTableFp32& DeviceEmbeddingTableFp32::operator=(DeviceEmbeddingTableFp32&&) noexcept = default;

DeviceEmbeddingTableFp32::~DeviceEmbeddingTableFp32() {
  if (impl_ && impl_->data != nullptr) {
    cudaFree(impl_->data);
  }
}

bool DeviceEmbeddingTableFp32::valid() const {
  return impl_ != nullptr && impl_->data != nullptr && impl_->vocab_size > 0 && impl_->embedding_dim > 0;
}

std::size_t DeviceEmbeddingTableFp32::vocab_size() const {
  return impl_ ? impl_->vocab_size : 0;
}

std::size_t DeviceEmbeddingTableFp32::embedding_dim() const {
  return impl_ ? impl_->embedding_dim : 0;
}

float* DeviceEmbeddingTableFp32::data() const {
  return impl_ ? impl_->data : nullptr;
}

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsFp32(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    DeviceTensorFp32* output) {
  if (!table.valid() ||
      host_token_ids == nullptr ||
      token_count == 0 ||
      output == nullptr ||
      !output->valid() ||
      output->shape().size() != 2 ||
      output->shape()[0] != token_count ||
      output->shape()[1] != table.embedding_dim()) {
    std::cerr << "embedding_table: invalid lookup request\n";
    return std::nullopt;
  }

  for (std::size_t i = 0; i < token_count; ++i) {
    if (host_token_ids[i] < 0 || static_cast<std::size_t>(host_token_ids[i]) >= table.vocab_size()) {
      std::cerr << "embedding_table: token id out of range at index " << i
                << " token_id=" << host_token_ids[i]
                << " vocab_size=" << table.vocab_size() << "\n";
      return std::nullopt;
    }
  }

  std::int32_t* token_ids_dev = nullptr;
  bool ok = true;
  const cudaError_t token_alloc_status =
      cudaMalloc(reinterpret_cast<void**>(&token_ids_dev), token_count * sizeof(std::int32_t));
  ok &= CheckCuda(token_alloc_status);
  if (!ok) {
    std::cerr << "embedding_table: token-id cudaMalloc failed ("
              << CudaErrorName(token_alloc_status) << ")\n";
  }
  const cudaError_t token_copy_status = ok
      ? cudaMemcpy(
            token_ids_dev,
            host_token_ids,
            token_count * sizeof(std::int32_t),
            cudaMemcpyHostToDevice)
      : cudaSuccess;
  ok &= CheckCuda(token_copy_status);
  if (!CheckCuda(token_copy_status)) {
    std::cerr << "embedding_table: token-id cudaMemcpy failed ("
              << CudaErrorName(token_copy_status) << ")\n";
  }

  if (ok) {
    const std::size_t total = token_count * table.embedding_dim();
    const int block_size = 256;
    const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    EmbeddingLookupKernel<<<grid_size, block_size>>>(
        table.data(),
        token_ids_dev,
        output->data(),
        token_count,
        table.embedding_dim());
    const cudaError_t launch_status = cudaPeekAtLastError();
    ok &= CheckCuda(launch_status);
    if (!CheckCuda(launch_status)) {
      std::cerr << "embedding_table: kernel launch failed ("
                << CudaErrorName(launch_status) << ")\n";
    }
  }

  if (token_ids_dev != nullptr) {
    cudaFree(token_ids_dev);
  }

  if (!ok) {
    return std::nullopt;
  }

  return EmbeddingLookupStats{
      token_count,
      table.embedding_dim(),
  };
}

}  // namespace nemotron
