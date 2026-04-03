#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "nemotron/device_buffer.h"
#include "nemotron/dense_weight.h"
#include "nemotron/embedding_table.h"
#include "nemotron/linear_op.h"
#include "nemotron/nvfp4_weight.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/scaled_fp8_linear.h"
#include "nemotron/single_token_forward_model.h"

namespace nemotron {

enum class ModelCacheEntryKind : std::uint32_t {
  kTensorFp32 = 1,
  kEmbeddingFp32 = 2,
  kDenseWeightFp32 = 3,
  kScaledFp8WeightFp32 = 4,
  kNvfp4Aligned = 5,
  kScaledFp8WeightNative = 6,
  kDenseWeightBf16 = 7,
};

struct ModelCacheEntry {
  std::string tensor_name;
  ModelCacheEntryKind kind = ModelCacheEntryKind::kTensorFp32;
  std::vector<std::size_t> shape;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  std::size_t payload_offset = 0;
  std::size_t payload_nbytes = 0;
  std::size_t aux0_offset = 0;
  std::size_t aux0_nbytes = 0;
  std::size_t aux1_offset = 0;
  std::size_t aux1_nbytes = 0;
  std::size_t aux2_offset = 0;
  std::size_t aux2_nbytes = 0;
  float input_scale = 0.0f;
  float weight_scale = 0.0f;
};

struct ModelCacheHeader {
  std::uint32_t format_version = 3;
  std::string model_id;
  std::string source_revision;
  SingleTokenForwardConfig config;
  std::vector<ModelCacheEntry> entries;
  std::size_t payload_nbytes = 0;
};

struct ModelCacheWriteReport {
  std::size_t entry_count = 0;
  std::size_t payload_nbytes = 0;
};

bool WriteDeterministicModelCache(
    const RuntimeEnvironment& environment,
    const SingleTokenForwardConfig& config,
    const std::filesystem::path& cache_path,
    ModelCacheWriteReport* report = nullptr);

class LoadedModelCache {
 public:
  static std::unique_ptr<LoadedModelCache> Load(const std::filesystem::path& cache_path);

  LoadedModelCache(LoadedModelCache&&) noexcept;
  LoadedModelCache& operator=(LoadedModelCache&&) noexcept;
  ~LoadedModelCache();

  LoadedModelCache(const LoadedModelCache&) = delete;
  LoadedModelCache& operator=(const LoadedModelCache&) = delete;

  bool valid() const;
  const ModelCacheHeader& header() const;
  const ModelCacheEntry* FindEntry(const std::string& tensor_name) const;

  std::unique_ptr<DeviceTensorFp32> CreateTensorView(const std::string& tensor_name) const;
  std::unique_ptr<DeviceEmbeddingTableFp32> CreateEmbeddingView(const std::string& tensor_name) const;
  std::unique_ptr<UploadedLinearOp> CreateDenseLinearView(const GemmDescriptor& descriptor) const;
  std::unique_ptr<ScaledFp8LinearOp> CreateScaledFp8LinearView(
      const std::string& tensor_name,
      std::size_t output_rows,
      std::size_t input_cols) const;
  std::unique_ptr<UploadedLinearOp> CreateNvfp4LinearView(const GemmDescriptor& descriptor) const;
  bool ReleaseEntry(const std::string& tensor_name);
  void ReleaseFilePages() const;

 private:
  struct DeviceEntryStorage {
    DeviceBuffer<std::uint8_t> payload;
    DeviceBuffer<std::uint8_t> aux0;
    DeviceBuffer<std::uint8_t> aux1;
    DeviceBuffer<std::uint8_t> aux2;
    bool resident = false;
  };

  LoadedModelCache() = default;

  bool EnsureEntryResident(const ModelCacheEntry* entry) const;
  std::size_t EntryIndex(const ModelCacheEntry* entry) const;
  std::uint8_t* PayloadPtr(const ModelCacheEntry* entry) const;
  std::uint8_t* Aux0Ptr(const ModelCacheEntry* entry) const;
  std::uint8_t* Aux1Ptr(const ModelCacheEntry* entry) const;
  std::uint8_t* Aux2Ptr(const ModelCacheEntry* entry) const;

  ModelCacheHeader header_;
  std::filesystem::path cache_path_;
  std::streamoff payload_base_offset_ = 0;
  std::unordered_map<std::string, std::size_t> indices_by_name_;
  mutable std::mutex resident_mutex_;
  mutable std::vector<DeviceEntryStorage> device_entries_;
};

}  // namespace nemotron
