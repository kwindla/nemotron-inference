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

}  // namespace nemotron
