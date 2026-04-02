#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "nemotron/device_tensor.h"
#include "nemotron/embedding_catalog.h"

namespace nemotron {

class DeviceEmbeddingTableFp32 {
 public:
  static std::unique_ptr<DeviceEmbeddingTableFp32> Upload(const EmbeddingDescriptor& descriptor);
  static std::unique_ptr<DeviceEmbeddingTableFp32> CreateView(
      std::size_t vocab_size,
      std::size_t embedding_dim,
      float* data);

  DeviceEmbeddingTableFp32(DeviceEmbeddingTableFp32&&) noexcept;
  DeviceEmbeddingTableFp32& operator=(DeviceEmbeddingTableFp32&&) noexcept;
  ~DeviceEmbeddingTableFp32();

  DeviceEmbeddingTableFp32(const DeviceEmbeddingTableFp32&) = delete;
  DeviceEmbeddingTableFp32& operator=(const DeviceEmbeddingTableFp32&) = delete;

  bool valid() const;
  std::size_t vocab_size() const;
  std::size_t embedding_dim() const;
  float* data() const;

 private:
  struct Impl;

  explicit DeviceEmbeddingTableFp32(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

struct EmbeddingLookupStats {
  std::size_t token_count = 0;
  std::size_t embedding_dim = 0;
};

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsFp32(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    DeviceTensorFp32* output);

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsBf16(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    DeviceTensorBf16* output);

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsDeviceIdsFp32(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* device_token_ids,
    std::size_t token_count,
    DeviceTensorFp32* output);

std::optional<EmbeddingLookupStats> LookupEmbeddingRowsDeviceIdsBf16(
    const DeviceEmbeddingTableFp32& table,
    const std::int32_t* device_token_ids,
    std::size_t token_count,
    DeviceTensorBf16* output);

}  // namespace nemotron
