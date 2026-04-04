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

// v5 cache format: NVFP4 aux2 stores raw weight_scale_2 for all entries
// (both routed and shared-down). The runtime kernel alpha is computed as
// dynamic_activation_scale * weight_scale_2 at serving time.
constexpr std::uint32_t kModelCacheFormatVersion = 5;

// Each entry kind defines both its on-disk section layout (`payload`, `aux0`,
// `aux1`, `aux2`) and the serving-time scalar metadata that cache-backed view
// factories consume.
enum class ModelCacheEntryKind : std::uint32_t {
  // Generic FP32 tensor entry.
  // payload: contiguous FP32 values for `shape`; `CreateTensorView()` exposes
  //     them as one flat device view.
  // aux0/aux1/aux2: unused and must be empty.
  // Serving-time scalar contract: none. Cache-backed serving consumes only the
  // payload; `input_scale` and `weight_scale` are ignored.
  kTensorFp32 = 1,

  // FP32 embedding-table entry.
  // payload: row-major `[vocab_size, embedding_dim]` FP32 table.
  // aux0/aux1/aux2: unused and must be empty.
  // Serving-time scalar contract: none. Cache-backed serving consumes only the
  // payload; `input_scale` and `weight_scale` are ignored.
  kEmbeddingFp32 = 2,

  // Dense row-major FP32 GEMM weight entry.
  // payload: row-major `[output_rows, input_cols]` FP32 weight matrix.
  // aux0/aux1/aux2: unused and must be empty.
  // Serving-time scalar contract: none. `CreateDenseLinearView()` serves the
  // payload directly as the runtime weight; scalar metadata is ignored.
  kDenseWeightFp32 = 3,

  // Legacy scaled-FP8 compatibility entry.
  // payload: dequantized row-major `[output_rows, input_cols]` FP32 weight
  //     matrix.
  // aux0/aux1/aux2: unused and must be empty.
  // Serving-time scalar contract: `input_scale` and `weight_scale` remain the
  // authoritative activation/weight scalars for cache-backed FP8 serving. The
  // payload is a compatibility surface that `CreateScaledFp8LinearView()` can
  // use to re-materialize native packed FP8 and fallback dequantized execution.
  kScaledFp8WeightFp32 = 4,

  // NVFP4 aligned weight entry.
  // payload: aligned packed NVFP4 weight bytes for
  //     `[output_rows, input_cols]`.
  // aux0: raw checkpoint block scales in the source row-major layout.
  // aux1: execution-layout block scales swizzled for serving-time matmul;
  //     `CreateNvfp4LinearView()` exposes this buffer as
  //     `descriptor.block_scales_data`.
  // aux2: one FP32 serving-time tensor-scale scalar exposed as
  //     `descriptor.tensor_scale_data`.
  // Serving-time scalar contract: cache-backed NVFP4 serving treats aux2 as
  // authoritative and consumes payload + aux1 + aux2 as the runtime surface.
  // aux0 is retained as the raw block-scale side buffer, not as the matmul
  // block-scale surface.
  // NVFP4 sub-contracts:
  //   routed expert NVFP4: aux2 = raw weight_scale_2 from checkpoint.
  //       The kernel alpha is formed at serving time as:
  //       dynamic_activation_scale * weight_scale_2
  //       where dynamic_activation_scale comes from NVFP4 packing of the
  //       latent activation. Checkpoint input_scale is NOT fused here;
  //       vLLM uses it during activation quantization (a_gscale = 1/input_scale)
  //       which this runtime handles differently (purely dynamic).
  //   shared-down NVFP4: aux2 = raw weight_scale_2 (same contract)
  kNvfp4Aligned = 5,

  // Native scaled-FP8 entry.
  // payload: packed FP8 E4M3 weight bytes for `[output_rows, input_cols]`.
  // aux0/aux1/aux2: unused and must be empty.
  // Serving-time scalar contract: `input_scale` and `weight_scale` are the
  // authoritative activation/weight scalars consumed by cache-backed FP8
  // serving. The runtime can use the payload plus those scalars for both
  // native packed FP8 execution and fallback dequantized execution.
  kScaledFp8WeightNative = 6,

  // Dense row-major BF16 GEMM weight entry.
  // payload: row-major `[output_rows, input_cols]` BF16 weight matrix bytes.
  // aux0/aux1/aux2: unused and must be empty.
  // Serving-time scalar contract: none. `CreateDenseLinearView()` serves the
  // payload directly as the runtime weight; scalar metadata is ignored.
  kDenseWeightBf16 = 7,
};

struct ModelCacheEntry {
  std::string tensor_name;
  ModelCacheEntryKind kind = ModelCacheEntryKind::kTensorFp32;
  std::vector<std::size_t> shape;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  // Section meanings are kind-specific; see `ModelCacheEntryKind` above.
  std::size_t payload_offset = 0;
  std::size_t payload_nbytes = 0;
  std::size_t aux0_offset = 0;
  std::size_t aux0_nbytes = 0;
  std::size_t aux1_offset = 0;
  std::size_t aux1_nbytes = 0;
  std::size_t aux2_offset = 0;
  std::size_t aux2_nbytes = 0;
  // Only scaled-FP8 entry kinds currently consume these scalar metadata
  // fields at serving time. Other entry kinds leave them unused/ignored.
  float input_scale = 0.0f;
  float weight_scale = 0.0f;
};

struct ModelCacheHeader {
  std::uint32_t format_version = kModelCacheFormatVersion;
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
  // NVFP4 cache aux2 is the authoritative serving-time tensor scale for
  // cache-backed views. `tensor_scale_override` is only for routed-expert
  // compatibility callers that still recompute fused
  // `input_scale * weight_scale_2` from source descriptors; it must match the
  // cached aux2 scalar and is not a general replacement hook.
  std::unique_ptr<UploadedLinearOp> CreateNvfp4LinearView(
      const GemmDescriptor& descriptor,
      std::optional<float> tensor_scale_override = std::nullopt) const;
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
