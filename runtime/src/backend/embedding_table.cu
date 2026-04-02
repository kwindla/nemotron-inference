#include "nemotron/embedding_table.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

#include "storage_conversion.h"

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

template <typename OutputT>
__device__ OutputT ConvertEmbeddingValue(float value);

template <>
__device__ float ConvertEmbeddingValue<float>(float value) {
  return value;
}

template <>
__device__ __nv_bfloat16 ConvertEmbeddingValue<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename OutputT>
__global__ void EmbeddingLookupKernel(
    const float* table,
    const std::int32_t* token_ids,
    OutputT* output,
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
  output[index] = ConvertEmbeddingValue<OutputT>(
      table[static_cast<std::size_t>(token_id) * embedding_dim + dim_index]);
}

template <typename TensorT>
bool ValidateLookupRequest(
    const DeviceEmbeddingTableFp32& table,
    const void* token_ids,
    std::size_t token_count,
    const TensorT* output) {
  const bool table_valid = table.valid();
  const bool have_token_ids = token_ids != nullptr;
  const bool nonzero_token_count = token_count != 0;
  const bool have_output = output != nullptr;
  const bool output_valid = have_output && output->valid();
  const bool output_rank_ok = output_valid && output->shape().size() == 2;
  const bool output_rows_ok = output_rank_ok && output->shape()[0] == token_count;
  const bool output_cols_ok = output_rank_ok && output->shape()[1] == table.embedding_dim();
  if (table_valid &&
      have_token_ids &&
      nonzero_token_count &&
      have_output &&
      output_valid &&
      output_rank_ok &&
      output_rows_ok &&
      output_cols_ok) {
    return true;
  }

  std::cerr << "embedding_table: invalid lookup request"
            << " table_valid=" << table_valid
            << " have_token_ids=" << have_token_ids
            << " nonzero_token_count=" << nonzero_token_count
            << " have_output=" << have_output
            << " output_valid=" << output_valid
            << " output_rank_ok=" << output_rank_ok
            << " output_rows_ok=" << output_rows_ok
            << " output_cols_ok=" << output_cols_ok;
  if (output_valid) {
    std::cerr << " output_shape=[";
    for (std::size_t i = 0; i < output->shape().size(); ++i) {
      if (i != 0) {
        std::cerr << ",";
      }
      std::cerr << output->shape()[i];
    }
    std::cerr << "]";
  }
  std::cerr << " token_count=" << token_count
            << " embedding_dim=" << table.embedding_dim()
            << "\n";
  return false;
}

template <typename TensorT>
std::optional<EmbeddingLookupStats> LaunchEmbeddingLookup(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* device_token_ids,
    std::size_t token_count,
    TensorT* output,
    bool synchronize) {
  if (!ValidateLookupRequest(table, device_token_ids, token_count, output)) {
    return std::nullopt;
  }
  const std::size_t total = token_count * table.embedding_dim();
  const int block_size = 256;
  const int grid_size = static_cast<int>((total + block_size - 1) / block_size);
  EmbeddingLookupKernel<<<grid_size, block_size>>>(
      table.data(),
      device_token_ids,
      output->data(),
      token_count,
      table.embedding_dim());
  const cudaError_t launch_status = cudaPeekAtLastError();
  if (!CheckCuda(launch_status)) {
    std::cerr << "embedding_table: kernel launch failed ("
              << CudaErrorName(launch_status) << ")\n";
    return std::nullopt;
  }
  if (synchronize) {
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (!CheckCuda(sync_status)) {
      std::cerr << "embedding_table: kernel sync failed ("
                << CudaErrorName(sync_status) << ")\n";
      return std::nullopt;
    }
  }
  return EmbeddingLookupStats{token_count, table.embedding_dim()};
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

  const PackedFloatStorage storage =
      is_bf16 ? PackedFloatStorage::kBf16 : PackedFloatStorage::kFp32;
  if (!UploadPackedFloatToDeviceFp32(
          descriptor.packed_data,
          element_count,
          storage,
          1.0f,
          impl->data)) {
    std::cerr << "embedding_table: device conversion upload failed ("
              << CudaErrorName(cudaGetLastError()) << ")\n";
    cudaFree(impl->data);
    return nullptr;
  }

  return std::unique_ptr<DeviceEmbeddingTableFp32>(new DeviceEmbeddingTableFp32(std::move(impl)));
}

std::unique_ptr<DeviceEmbeddingTableFp32> DeviceEmbeddingTableFp32::CreateView(
    std::size_t vocab_size,
    std::size_t embedding_dim,
    float* data) {
  if (vocab_size == 0 || embedding_dim == 0 || data == nullptr) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->vocab_size = vocab_size;
  impl->embedding_dim = embedding_dim;
  impl->data = data;
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
  if (!ValidateLookupRequest(table, host_token_ids, token_count, output)) {
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

  std::optional<EmbeddingLookupStats> stats;
  if (ok) {
    stats = LaunchEmbeddingLookup(table, token_ids_dev, token_count, output, true);
    ok &= stats.has_value();
  }

  if (token_ids_dev != nullptr) {
    cudaFree(token_ids_dev);
  }

  if (!ok) {
    return std::nullopt;
  }

  return stats;
}

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsBf16(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    DeviceTensorBf16* output) {
  if (!ValidateLookupRequest(table, host_token_ids, token_count, output)) {
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

  std::optional<EmbeddingLookupStats> stats;
  if (ok) {
    stats = LaunchEmbeddingLookup(table, token_ids_dev, token_count, output, true);
    ok &= stats.has_value();
  }

  if (token_ids_dev != nullptr) {
    cudaFree(token_ids_dev);
  }

  if (!ok) {
    return std::nullopt;
  }

  return stats;
}

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsDeviceIdsFp32(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* device_token_ids,
    std::size_t token_count,
    DeviceTensorFp32* output) {
  return LaunchEmbeddingLookup(table, device_token_ids, token_count, output, false);
}

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsDeviceIdsBf16(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* device_token_ids,
    std::size_t token_count,
    DeviceTensorBf16* output) {
  return LaunchEmbeddingLookup(table, device_token_ids, token_count, output, false);
}

}  // namespace nemotron
