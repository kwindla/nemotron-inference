#include "nemotron/model_cache.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nemotron/attention_layer.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/expert_layer.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/model_schedule.h"
#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {
namespace {

constexpr char kMagic[] = "NEMO_MODEL_CACHE_V1";
constexpr std::size_t kAlignmentBytes = 256;

template <typename T>
bool WritePod(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
  return static_cast<bool>(output);
}

template <typename T>
bool ReadPod(std::ifstream& input, T* value) {
  input.read(reinterpret_cast<char*>(value), sizeof(T));
  return static_cast<bool>(input);
}

bool WriteString(std::ofstream& output, const std::string& value) {
  const std::uint64_t size = value.size();
  return WritePod(output, size) &&
         (size == 0 ||
          static_cast<bool>(output.write(value.data(), static_cast<std::streamsize>(size))));
}

bool ReadString(std::ifstream& input, std::string* value) {
  std::uint64_t size = 0;
  if (!ReadPod(input, &size)) {
    return false;
  }
  value->assign(static_cast<std::size_t>(size), '\0');
  if (size == 0) {
    return true;
  }
  input.read(value->data(), static_cast<std::streamsize>(size));
  return static_cast<bool>(input);
}

std::size_t AlignTo(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const std::size_t remainder = value % alignment;
  return remainder == 0 ? value : (value + alignment - remainder);
}

template <typename DescriptorT, typename LookupFn>
const DescriptorT* FindGlobalDescriptorByCandidates(
    const std::vector<GlobalTensorBinding>& bindings,
    ModelGlobalRole role,
    LookupFn lookup,
    const std::vector<std::string>& candidates) {
  for (const GlobalTensorBinding& binding : bindings) {
    if (binding.role != role) {
      continue;
    }
    if (const DescriptorT* descriptor = lookup(binding.tensor_name); descriptor != nullptr) {
      return descriptor;
    }
  }
  for (const std::string& candidate : candidates) {
    if (const DescriptorT* descriptor = lookup(candidate); descriptor != nullptr) {
      return descriptor;
    }
  }
  return nullptr;
}

bool IsFp32Storage(const std::string& storage_dtype) {
  return storage_dtype == "fp32" || storage_dtype == "float32" || storage_dtype == "float";
}

bool IsBf16Storage(const std::string& storage_dtype) {
  return storage_dtype == "bf16" || storage_dtype == "bfloat16";
}

std::optional<std::vector<float>> ReadTensorToHostFlatFp32(const KernelTensorDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  std::size_t count = 1;
  for (std::size_t dim : descriptor.logical_shape) {
    count *= dim;
  }
  if (count == 0) {
    return std::nullopt;
  }
  std::vector<float> values(count, 0.0f);
  if (IsFp32Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return values;
  }
  if (IsBf16Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return std::nullopt;
    }
    const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = __bfloat162float(src[i]);
    }
    return values;
  }
  return std::nullopt;
}

std::optional<std::vector<float>> ReadEmbeddingToHostFp32(const EmbeddingDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr ||
      descriptor.vocab_size == 0 ||
      descriptor.embedding_dim == 0) {
    return std::nullopt;
  }
  const std::size_t count = descriptor.vocab_size * descriptor.embedding_dim;
  std::vector<float> values(count, 0.0f);
  if (IsFp32Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return values;
  }
  if (IsBf16Storage(descriptor.storage_dtype)) {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return std::nullopt;
    }
    const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = __bfloat162float(src[i]);
    }
    return values;
  }
  return std::nullopt;
}

std::optional<float> ReadScalarFp32(const KernelTensorDescriptor& descriptor) {
  const auto values = ReadTensorToHostFlatFp32(descriptor);
  if (!values.has_value() || values->size() != 1) {
    return std::nullopt;
  }
  return (*values)[0];
}

std::vector<std::uint8_t> CopyFloatBytes(const std::vector<float>& values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(float), 0u);
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

void FinalizeOffsets(
    std::vector<ModelCacheEntry>* entries,
    std::size_t* payload_nbytes) {
  std::size_t offset = 0;
  for (ModelCacheEntry& entry : *entries) {
    if (entry.payload_nbytes != 0) {
      offset = AlignTo(offset, kAlignmentBytes);
      entry.payload_offset = offset;
      offset += entry.payload_nbytes;
    }
    if (entry.aux0_nbytes != 0) {
      offset = AlignTo(offset, kAlignmentBytes);
      entry.aux0_offset = offset;
      offset += entry.aux0_nbytes;
    }
    if (entry.aux1_nbytes != 0) {
      offset = AlignTo(offset, kAlignmentBytes);
      entry.aux1_offset = offset;
      offset += entry.aux1_nbytes;
    }
    if (entry.aux2_nbytes != 0) {
      offset = AlignTo(offset, kAlignmentBytes);
      entry.aux2_offset = offset;
      offset += entry.aux2_nbytes;
    }
  }
  *payload_nbytes = offset;
}

bool SerializeEntries(
    std::ofstream& output,
    const ModelCacheHeader& header) {
  if (!WritePod(output, header.format_version)) {
    return false;
  }
  if (!WriteString(output, header.model_id) ||
      !WriteString(output, header.source_revision)) {
    return false;
  }
  const SingleTokenForwardConfig& config = header.config;
  return
      WritePod(output, config.hidden_size) &&
      WritePod(output, config.total_layer_count) &&
      WritePod(output, config.vocab_size) &&
      WritePod(output, config.max_tokens) &&
      WritePod(output, config.attention_head_count) &&
      WritePod(output, config.attention_kv_head_count) &&
      WritePod(output, config.attention_head_dim) &&
      WritePod(output, config.attention_tokens_per_page) &&
      WritePod(output, config.mamba_intermediate_size) &&
      WritePod(output, config.mamba_num_heads) &&
      WritePod(output, config.mamba_head_dim) &&
      WritePod(output, config.mamba_state_size) &&
      WritePod(output, config.mamba_n_groups) &&
      WritePod(output, config.mamba_conv_kernel_size) &&
      WritePod(output, config.moe_latent_size) &&
      WritePod(output, config.routed_expert_intermediate_size) &&
      WritePod(output, config.shared_expert_intermediate_size) &&
      WritePod(output, config.n_routed_experts) &&
      WritePod(output, config.experts_per_token) &&
      WritePod(output, config.expert_n_group) &&
      WritePod(output, config.expert_topk_group) &&
      WritePod(output, config.layer_norm_epsilon) &&
      WritePod(output, config.mamba_time_step_min) &&
      WritePod(output, config.routed_scaling_factor) &&
      WritePod(output, config.norm_topk_prob);
}

bool DeserializeConfig(std::ifstream& input, SingleTokenForwardConfig* config) {
  return
      ReadPod(input, &config->hidden_size) &&
      ReadPod(input, &config->total_layer_count) &&
      ReadPod(input, &config->vocab_size) &&
      ReadPod(input, &config->max_tokens) &&
      ReadPod(input, &config->attention_head_count) &&
      ReadPod(input, &config->attention_kv_head_count) &&
      ReadPod(input, &config->attention_head_dim) &&
      ReadPod(input, &config->attention_tokens_per_page) &&
      ReadPod(input, &config->mamba_intermediate_size) &&
      ReadPod(input, &config->mamba_num_heads) &&
      ReadPod(input, &config->mamba_head_dim) &&
      ReadPod(input, &config->mamba_state_size) &&
      ReadPod(input, &config->mamba_n_groups) &&
      ReadPod(input, &config->mamba_conv_kernel_size) &&
      ReadPod(input, &config->moe_latent_size) &&
      ReadPod(input, &config->routed_expert_intermediate_size) &&
      ReadPod(input, &config->shared_expert_intermediate_size) &&
      ReadPod(input, &config->n_routed_experts) &&
      ReadPod(input, &config->experts_per_token) &&
      ReadPod(input, &config->expert_n_group) &&
      ReadPod(input, &config->expert_topk_group) &&
      ReadPod(input, &config->layer_norm_epsilon) &&
      ReadPod(input, &config->mamba_time_step_min) &&
      ReadPod(input, &config->routed_scaling_factor) &&
      ReadPod(input, &config->norm_topk_prob);
}

bool WriteHeader(std::ofstream& output, const ModelCacheHeader& header) {
  const std::uint64_t magic_size = sizeof(kMagic);
  const std::uint32_t entry_count = static_cast<std::uint32_t>(header.entries.size());
  return WritePod(output, magic_size) &&
         static_cast<bool>(output.write(kMagic, static_cast<std::streamsize>(magic_size))) &&
         SerializeEntries(output, header) &&
         WritePod(output, entry_count) &&
         WritePod(output, static_cast<std::uint64_t>(header.payload_nbytes));
}

bool ReadHeader(std::ifstream& input, ModelCacheHeader* header) {
  std::uint64_t magic_size = 0;
  if (!ReadPod(input, &magic_size)) {
    return false;
  }
  std::string magic(static_cast<std::size_t>(magic_size), '\0');
  input.read(magic.data(), static_cast<std::streamsize>(magic_size));
  if (!input || magic != std::string(kMagic, sizeof(kMagic))) {
    return false;
  }
  if (!ReadPod(input, &header->format_version) ||
      !ReadString(input, &header->model_id) ||
      !ReadString(input, &header->source_revision) ||
      !DeserializeConfig(input, &header->config)) {
    return false;
  }
  std::uint32_t entry_count = 0;
  std::uint64_t payload_nbytes = 0;
  if (!ReadPod(input, &entry_count) || !ReadPod(input, &payload_nbytes)) {
    return false;
  }
  header->entries.clear();
  header->entries.reserve(entry_count);
  header->payload_nbytes = static_cast<std::size_t>(payload_nbytes);
  for (std::uint32_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    ModelCacheEntry entry;
    std::uint32_t kind = 0;
    std::uint64_t shape_count = 0;
    if (!ReadString(input, &entry.tensor_name) ||
        !ReadPod(input, &kind) ||
        !ReadPod(input, &shape_count) ||
        !ReadPod(input, &entry.output_rows) ||
        !ReadPod(input, &entry.input_cols) ||
        !ReadPod(input, &entry.payload_offset) ||
        !ReadPod(input, &entry.payload_nbytes) ||
        !ReadPod(input, &entry.aux0_offset) ||
        !ReadPod(input, &entry.aux0_nbytes) ||
        !ReadPod(input, &entry.aux1_offset) ||
        !ReadPod(input, &entry.aux1_nbytes) ||
        !ReadPod(input, &entry.aux2_offset) ||
        !ReadPod(input, &entry.aux2_nbytes) ||
        !ReadPod(input, &entry.input_scale) ||
        !ReadPod(input, &entry.weight_scale)) {
      return false;
    }
    entry.kind = static_cast<ModelCacheEntryKind>(kind);
    entry.shape.resize(static_cast<std::size_t>(shape_count), 0);
    for (std::size_t dim_index = 0; dim_index < entry.shape.size(); ++dim_index) {
      if (!ReadPod(input, &entry.shape[dim_index])) {
        return false;
      }
    }
    header->entries.push_back(std::move(entry));
  }
  return true;
}

bool WriteEntryMetadata(std::ofstream& output, const ModelCacheEntry& entry) {
  const std::uint32_t kind = static_cast<std::uint32_t>(entry.kind);
  const std::uint64_t shape_count = entry.shape.size();
  if (!WriteString(output, entry.tensor_name) ||
      !WritePod(output, kind) ||
      !WritePod(output, shape_count) ||
      !WritePod(output, entry.output_rows) ||
      !WritePod(output, entry.input_cols) ||
      !WritePod(output, entry.payload_offset) ||
      !WritePod(output, entry.payload_nbytes) ||
      !WritePod(output, entry.aux0_offset) ||
      !WritePod(output, entry.aux0_nbytes) ||
      !WritePod(output, entry.aux1_offset) ||
      !WritePod(output, entry.aux1_nbytes) ||
      !WritePod(output, entry.aux2_offset) ||
      !WritePod(output, entry.aux2_nbytes) ||
      !WritePod(output, entry.input_scale) ||
      !WritePod(output, entry.weight_scale)) {
    return false;
  }
  for (std::size_t dim : entry.shape) {
    if (!WritePod(output, dim)) {
      return false;
    }
  }
  return true;
}

bool WriteZeros(std::ofstream& output, std::size_t count) {
  if (count == 0) {
    return true;
  }
  static constexpr std::size_t kZeroChunkBytes = 1u << 20u;
  static const std::vector<char> kZeros(kZeroChunkBytes, '\0');
  while (count > 0) {
    const std::size_t chunk = std::min(count, kZeroChunkBytes);
    output.write(kZeros.data(), static_cast<std::streamsize>(chunk));
    if (!output) {
      return false;
    }
    count -= chunk;
  }
  return true;
}

bool PadOutputToOffset(
    std::ofstream& output,
    std::size_t* current_offset,
    std::size_t target_offset) {
  if (current_offset == nullptr || target_offset < *current_offset) {
    return false;
  }
  const std::size_t padding = target_offset - *current_offset;
  if (!WriteZeros(output, padding)) {
    return false;
  }
  *current_offset = target_offset;
  return true;
}

bool WriteByteSpan(
    std::ofstream& output,
    const std::uint8_t* data,
    std::size_t nbytes,
    std::size_t* current_offset) {
  if (nbytes == 0) {
    return true;
  }
  if (data == nullptr || current_offset == nullptr) {
    return false;
  }
  output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
  if (!output) {
    return false;
  }
  *current_offset += nbytes;
  return true;
}

bool AddTensorFp32Entry(
    const KernelTensorDescriptor& descriptor,
    std::unordered_set<std::string>* seen,
    std::vector<ModelCacheEntry>* entries) {
  if (!seen->insert(descriptor.tensor_name).second) {
    return true;
  }
  std::size_t count = 1;
  for (std::size_t dim : descriptor.logical_shape) {
    count *= dim;
  }
  if (descriptor.packed_data == nullptr || count == 0) {
    return false;
  }
  ModelCacheEntry entry;
  entry.tensor_name = descriptor.tensor_name;
  entry.kind = ModelCacheEntryKind::kTensorFp32;
  entry.shape = descriptor.logical_shape;
  entry.payload_nbytes = count * sizeof(float);
  entries->push_back(std::move(entry));
  return true;
}

bool AddEmbeddingEntry(
    const EmbeddingDescriptor& descriptor,
    std::unordered_set<std::string>* seen,
    std::vector<ModelCacheEntry>* entries) {
  if (!seen->insert(descriptor.tensor_name).second) {
    return true;
  }
  if (descriptor.packed_data == nullptr ||
      descriptor.vocab_size == 0 ||
      descriptor.embedding_dim == 0) {
    return false;
  }
  ModelCacheEntry entry;
  entry.tensor_name = descriptor.tensor_name;
  entry.kind = ModelCacheEntryKind::kEmbeddingFp32;
  entry.shape = {descriptor.vocab_size, descriptor.embedding_dim};
  entry.output_rows = descriptor.vocab_size;
  entry.input_cols = descriptor.embedding_dim;
  entry.payload_nbytes = descriptor.vocab_size * descriptor.embedding_dim * sizeof(float);
  entries->push_back(std::move(entry));
  return true;
}

bool AddDenseEntry(
    const GemmDescriptor& descriptor,
    float storage_scale,
    ModelCacheEntryKind kind,
    float input_scale,
    std::unordered_set<std::string>* seen,
    std::vector<ModelCacheEntry>* entries) {
  if (!seen->insert(descriptor.tensor_name).second) {
    return true;
  }
  if (descriptor.packed_data == nullptr ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0) {
    return false;
  }
  ModelCacheEntry entry;
  entry.tensor_name = descriptor.tensor_name;
  entry.kind = kind;
  entry.shape = {descriptor.output_rows, descriptor.input_cols};
  entry.output_rows = descriptor.output_rows;
  entry.input_cols = descriptor.input_cols;
  entry.input_scale = input_scale;
  entry.weight_scale = storage_scale;
  entry.payload_nbytes = descriptor.output_rows * descriptor.input_cols * sizeof(float);
  entries->push_back(std::move(entry));
  return true;
}

bool AddNvfp4Entry(
    const GemmDescriptor& descriptor,
    std::unordered_set<std::string>* seen,
    std::vector<ModelCacheEntry>* entries) {
  if (!seen->insert(descriptor.tensor_name).second) {
    return true;
  }
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr ||
      descriptor.tensor_scale_data == nullptr) {
    return false;
  }
  const std::size_t execution_scale_nbytes =
      ExecutionNvfp4ScaleBytes(descriptor.output_rows, descriptor.input_cols);
  if (execution_scale_nbytes == 0) {
    return false;
  }
  ModelCacheEntry entry;
  entry.tensor_name = descriptor.tensor_name;
  entry.kind = ModelCacheEntryKind::kNvfp4Aligned;
  entry.shape = {descriptor.output_rows, descriptor.input_cols};
  entry.output_rows = descriptor.output_rows;
  entry.input_cols = descriptor.input_cols;
  entry.payload_nbytes = descriptor.packed_nbytes;
  entry.aux0_nbytes = descriptor.block_scales_nbytes;
  entry.aux1_nbytes = execution_scale_nbytes;
  entry.aux2_nbytes = descriptor.tensor_scale_nbytes;
  entries->push_back(std::move(entry));
  return true;
}

}  // namespace

bool WriteDeterministicModelCache(
    const RuntimeEnvironment& environment,
    const SingleTokenForwardConfig& config,
    const std::filesystem::path& cache_path,
    ModelCacheWriteReport* report) {
  if (!environment.has_model_schedule() ||
      !environment.has_kernel_catalog() ||
      !environment.has_gemm_catalog() ||
      !environment.has_embedding_catalog()) {
    return false;
  }
  const ModelSchedule& schedule = *environment.model_schedule();
  const KernelCatalog& kernel_catalog = *environment.kernel_catalog();
  const GemmCatalog& gemm_catalog = *environment.gemm_catalog();
  const EmbeddingCatalog& embedding_catalog = *environment.embedding_catalog();
  const auto plan = BuildSingleTokenForwardPlan(schedule, config);
  if (!plan.has_value()) {
    return false;
  }

  std::unordered_set<std::string> seen;
  std::vector<ModelCacheEntry> entries;

  const EmbeddingDescriptor* embedding = FindGlobalDescriptorByCandidates<EmbeddingDescriptor>(
      schedule.global_bindings(),
      ModelGlobalRole::kEmbedding,
      [&](const std::string& name) { return embedding_catalog.FindDescriptor(name); },
      {"backbone.embeddings.weight", "embeddings.weight"});
  const GemmDescriptor* lm_head = FindGlobalDescriptorByCandidates<GemmDescriptor>(
      schedule.global_bindings(),
      ModelGlobalRole::kLogits,
      [&](const std::string& name) { return gemm_catalog.FindDescriptor(name); },
      {"lm_head.weight", "logits.weight", "output.weight"});
  const KernelTensorDescriptor* final_norm = FindGlobalDescriptorByCandidates<KernelTensorDescriptor>(
      schedule.global_bindings(),
      ModelGlobalRole::kFinalNorm,
      [&](const std::string& name) { return kernel_catalog.FindTensor(name); },
      {"backbone.norm_f.weight", "norm_f.weight", "backbone.final_norm.weight", "final_norm.weight"});
  if (embedding == nullptr || lm_head == nullptr) {
    return false;
  }
  if (!AddEmbeddingEntry(*embedding, &seen, &entries) ||
      !AddDenseEntry(*lm_head, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries) ||
      (final_norm != nullptr && !AddTensorFp32Entry(*final_norm, &seen, &entries))) {
    return false;
  }

  for (const ForwardLayerPlanEntry& plan_entry : plan->layers) {
    const LayerScheduleEntry* layer = schedule.FindLayer(plan_entry.layer_index);
    if (layer == nullptr) {
      return false;
    }
    switch (plan_entry.kind) {
      case ForwardLayerKind::kAttention: {
        const auto bindings = BuildAttentionLayerBindings(*layer, kernel_catalog, gemm_catalog);
        if (!bindings.has_value() ||
            !AddTensorFp32Entry(*bindings->norm_weight, &seen, &entries) ||
            !AddDenseEntry(*bindings->q_proj, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries) ||
            !AddDenseEntry(*bindings->k_proj, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries) ||
            !AddDenseEntry(*bindings->v_proj, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries) ||
            !AddDenseEntry(*bindings->o_proj, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries)) {
          return false;
        }
        break;
      }
      case ForwardLayerKind::kMamba: {
        const auto bindings = BuildMambaLayerBindings(*layer, kernel_catalog, gemm_catalog);
        if (!bindings.has_value()) {
          return false;
        }
        if (!AddTensorFp32Entry(*bindings->input_norm_weight, &seen, &entries) ||
            !AddTensorFp32Entry(*bindings->mixer_norm_weight, &seen, &entries) ||
            !AddTensorFp32Entry(*bindings->conv1d_weight, &seen, &entries) ||
            !AddTensorFp32Entry(*bindings->conv1d_bias, &seen, &entries) ||
            !AddTensorFp32Entry(*bindings->A_log, &seen, &entries) ||
            !AddTensorFp32Entry(*bindings->D, &seen, &entries) ||
            !AddTensorFp32Entry(*bindings->dt_bias, &seen, &entries)) {
          return false;
        }
        if (bindings->in_proj_kernel_weight != nullptr &&
            bindings->in_proj_weight_scale != nullptr &&
            bindings->in_proj_input_scale != nullptr) {
          const auto weight_scale = ReadScalarFp32(*bindings->in_proj_weight_scale);
          const auto input_scale = ReadScalarFp32(*bindings->in_proj_input_scale);
          if (!weight_scale.has_value() || !input_scale.has_value() ||
              !AddDenseEntry(
                  *bindings->in_proj_gemm_weight,
                  *weight_scale,
                  ModelCacheEntryKind::kScaledFp8WeightFp32,
                  *input_scale,
                  &seen,
                  &entries)) {
            return false;
          }
        } else if (!AddDenseEntry(
                       *bindings->in_proj_gemm_weight,
                       1.0f,
                       ModelCacheEntryKind::kDenseWeightFp32,
                       0.0f,
                       &seen,
                       &entries)) {
          return false;
        }
        if (bindings->out_proj_kernel_weight != nullptr &&
            bindings->out_proj_weight_scale != nullptr &&
            bindings->out_proj_input_scale != nullptr) {
          const auto weight_scale = ReadScalarFp32(*bindings->out_proj_weight_scale);
          const auto input_scale = ReadScalarFp32(*bindings->out_proj_input_scale);
          if (!weight_scale.has_value() || !input_scale.has_value() ||
              !AddDenseEntry(
                  *bindings->out_proj_gemm_weight,
                  *weight_scale,
                  ModelCacheEntryKind::kScaledFp8WeightFp32,
                  *input_scale,
                  &seen,
                  &entries)) {
            return false;
          }
        } else if (!AddDenseEntry(
                       *bindings->out_proj_gemm_weight,
                       1.0f,
                       ModelCacheEntryKind::kDenseWeightFp32,
                       0.0f,
                       &seen,
                       &entries)) {
          return false;
        }
        break;
      }
      case ForwardLayerKind::kExpert: {
        const auto bindings = BuildExpertLayerBindings(*layer, kernel_catalog, gemm_catalog, config.n_routed_experts);
        if (!bindings.has_value() ||
            !AddTensorFp32Entry(*bindings->input_norm_weight, &seen, &entries) ||
            !AddTensorFp32Entry(*bindings->gate_score_correction_bias, &seen, &entries) ||
            !AddDenseEntry(*bindings->gate_weight, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries) ||
            !AddDenseEntry(*bindings->fc2_latent_weight, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries)) {
          return false;
        }
        if (bindings->fc1_latent_kernel_weight != nullptr &&
            bindings->fc1_latent_weight_scale != nullptr &&
            bindings->fc1_latent_input_scale != nullptr) {
          const auto weight_scale = ReadScalarFp32(*bindings->fc1_latent_weight_scale);
          const auto input_scale = ReadScalarFp32(*bindings->fc1_latent_input_scale);
          if (!weight_scale.has_value() || !input_scale.has_value() ||
              !AddDenseEntry(
                  *bindings->fc1_latent_gemm_weight,
                  *weight_scale,
                  ModelCacheEntryKind::kScaledFp8WeightFp32,
                  *input_scale,
                  &seen,
                  &entries)) {
            return false;
          }
        } else if (!AddDenseEntry(*bindings->fc1_latent_gemm_weight, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries)) {
          return false;
        }
        if (bindings->shared_up_kernel_weight != nullptr &&
            bindings->shared_up_weight_scale != nullptr &&
            bindings->shared_up_input_scale != nullptr) {
          const auto weight_scale = ReadScalarFp32(*bindings->shared_up_weight_scale);
          const auto input_scale = ReadScalarFp32(*bindings->shared_up_input_scale);
          if (!weight_scale.has_value() || !input_scale.has_value() ||
              !AddDenseEntry(
                  *bindings->shared_up_gemm_weight,
                  *weight_scale,
                  ModelCacheEntryKind::kScaledFp8WeightFp32,
                  *input_scale,
                  &seen,
                  &entries)) {
            return false;
          }
        } else if (!AddDenseEntry(*bindings->shared_up_gemm_weight, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries)) {
          return false;
        }
        if (bindings->shared_down_gemm_weight->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
          if (!AddNvfp4Entry(*bindings->shared_down_gemm_weight, &seen, &entries)) {
            return false;
          }
        } else if (bindings->shared_down_kernel_weight != nullptr &&
                   bindings->shared_down_weight_scale != nullptr &&
                   bindings->shared_down_input_scale != nullptr &&
                   bindings->shared_down_kernel_weight->storage_dtype == "fp8_e4m3fn") {
          const auto weight_scale = ReadScalarFp32(*bindings->shared_down_weight_scale);
          const auto input_scale = ReadScalarFp32(*bindings->shared_down_input_scale);
          if (!weight_scale.has_value() || !input_scale.has_value() ||
              !AddDenseEntry(
                  *bindings->shared_down_gemm_weight,
                  *weight_scale,
                  ModelCacheEntryKind::kScaledFp8WeightFp32,
                  *input_scale,
                  &seen,
                  &entries)) {
            return false;
          }
        } else if (!AddDenseEntry(*bindings->shared_down_gemm_weight, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries)) {
          return false;
        }
        for (const ExpertWeightPair& pair : bindings->routed_experts) {
          if (pair.up_proj->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled) {
            if (!AddNvfp4Entry(*pair.up_proj, &seen, &entries) ||
                !AddNvfp4Entry(*pair.down_proj, &seen, &entries)) {
              return false;
            }
          } else if (!AddDenseEntry(*pair.up_proj, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries) ||
                     !AddDenseEntry(*pair.down_proj, 1.0f, ModelCacheEntryKind::kDenseWeightFp32, 0.0f, &seen, &entries)) {
            return false;
          }
        }
        break;
      }
    }
  }

  ModelCacheHeader header;
  header.config = config;
  header.entries = std::move(entries);
  FinalizeOffsets(&header.entries, &header.payload_nbytes);

  if (!cache_path.parent_path().empty()) {
    std::filesystem::create_directories(cache_path.parent_path());
  }
  std::ofstream output(cache_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  if (!WriteHeader(output, header)) {
    return false;
  }
  for (const ModelCacheEntry& entry : header.entries) {
    if (!WriteEntryMetadata(output, entry)) {
      return false;
    }
  }
  std::size_t payload_offset = 0;
  for (const ModelCacheEntry& entry : header.entries) {
    if (!PadOutputToOffset(output, &payload_offset, entry.payload_offset)) {
      return false;
    }
    switch (entry.kind) {
      case ModelCacheEntryKind::kTensorFp32: {
        const KernelTensorDescriptor* descriptor = kernel_catalog.FindTensor(entry.tensor_name);
        const auto values = descriptor == nullptr ? std::nullopt : ReadTensorToHostFlatFp32(*descriptor);
        if (!values.has_value() ||
            !WriteByteSpan(
                output,
                reinterpret_cast<const std::uint8_t*>(values->data()),
                values->size() * sizeof(float),
                &payload_offset)) {
          return false;
        }
        break;
      }
      case ModelCacheEntryKind::kEmbeddingFp32: {
        const EmbeddingDescriptor* descriptor = embedding_catalog.FindDescriptor(entry.tensor_name);
        const auto values = descriptor == nullptr ? std::nullopt : ReadEmbeddingToHostFp32(*descriptor);
        if (!values.has_value() ||
            !WriteByteSpan(
                output,
                reinterpret_cast<const std::uint8_t*>(values->data()),
                values->size() * sizeof(float),
                &payload_offset)) {
          return false;
        }
        break;
      }
      case ModelCacheEntryKind::kDenseWeightFp32:
      case ModelCacheEntryKind::kScaledFp8WeightFp32: {
        const GemmDescriptor* descriptor = gemm_catalog.FindDescriptor(entry.tensor_name);
        const auto values =
            descriptor == nullptr ? std::nullopt : ReadDenseWeightToHostFp32(*descriptor, entry.weight_scale);
        if (!values.has_value() ||
            !WriteByteSpan(
                output,
                reinterpret_cast<const std::uint8_t*>(values->data()),
                values->size() * sizeof(float),
                &payload_offset)) {
          return false;
        }
        break;
      }
      case ModelCacheEntryKind::kNvfp4Aligned: {
        const GemmDescriptor* descriptor = gemm_catalog.FindDescriptor(entry.tensor_name);
        if (descriptor == nullptr ||
            descriptor->packed_data == nullptr ||
            descriptor->block_scales_data == nullptr ||
            descriptor->tensor_scale_data == nullptr ||
            !WriteByteSpan(output, descriptor->packed_data, descriptor->packed_nbytes, &payload_offset) ||
            !PadOutputToOffset(output, &payload_offset, entry.aux0_offset) ||
            !WriteByteSpan(output, descriptor->block_scales_data, descriptor->block_scales_nbytes, &payload_offset)) {
          return false;
        }
        std::vector<std::uint8_t> swizzled(entry.aux1_nbytes, 0u);
        if (!SwizzleRowMajorNvfp4ScalesForExecutionInto(
                descriptor->block_scales_data,
                descriptor->output_rows,
                descriptor->input_cols,
                swizzled.data(),
                swizzled.size()) ||
            !PadOutputToOffset(output, &payload_offset, entry.aux1_offset) ||
            !WriteByteSpan(output, swizzled.data(), swizzled.size(), &payload_offset) ||
            !PadOutputToOffset(output, &payload_offset, entry.aux2_offset) ||
            !WriteByteSpan(output, descriptor->tensor_scale_data, descriptor->tensor_scale_nbytes, &payload_offset)) {
          return false;
        }
        break;
      }
    }
  }
  if (!PadOutputToOffset(output, &payload_offset, header.payload_nbytes)) {
    return false;
  }
  if (report != nullptr) {
    report->entry_count = header.entries.size();
    report->payload_nbytes = header.payload_nbytes;
  }
  return true;
}

std::unique_ptr<LoadedModelCache> LoadedModelCache::Load(const std::filesystem::path& cache_path) {
  std::ifstream input(cache_path, std::ios::binary);
  if (!input) {
    return nullptr;
  }
  auto cache = std::unique_ptr<LoadedModelCache>(new LoadedModelCache());
  if (!ReadHeader(input, &cache->header_)) {
    return nullptr;
  }
  for (const ModelCacheEntry& entry : cache->header_.entries) {
    cache->indices_by_name_.emplace(entry.tensor_name, cache->indices_by_name_.size());
  }
  if (cache->header_.payload_nbytes != 0) {
    if (!cache->payload_.Resize(cache->header_.payload_nbytes)) {
      return nullptr;
    }
    static constexpr std::size_t kChunkBytes = 64u << 20u;
    std::vector<std::uint8_t> host_chunk(std::min(cache->header_.payload_nbytes, kChunkBytes), 0u);
    std::size_t copied = 0;
    while (copied < cache->header_.payload_nbytes) {
      const std::size_t chunk = std::min(cache->header_.payload_nbytes - copied, host_chunk.size());
      input.read(reinterpret_cast<char*>(host_chunk.data()), static_cast<std::streamsize>(chunk));
      if (!input) {
        return nullptr;
      }
      if (cudaMemcpy(
              cache->payload_.data() + copied,
              host_chunk.data(),
              chunk,
              cudaMemcpyHostToDevice) != cudaSuccess) {
        return nullptr;
      }
      copied += chunk;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      return nullptr;
    }
  }
  return cache;
}

LoadedModelCache::LoadedModelCache(LoadedModelCache&&) noexcept = default;
LoadedModelCache& LoadedModelCache::operator=(LoadedModelCache&&) noexcept = default;
LoadedModelCache::~LoadedModelCache() = default;

bool LoadedModelCache::valid() const {
  return !header_.entries.empty() && payload_.valid();
}

const ModelCacheHeader& LoadedModelCache::header() const {
  return header_;
}

const ModelCacheEntry* LoadedModelCache::FindEntry(const std::string& tensor_name) const {
  const auto it = indices_by_name_.find(tensor_name);
  if (it == indices_by_name_.end()) {
    return nullptr;
  }
  return &header_.entries[it->second];
}

std::uint8_t* LoadedModelCache::PayloadPtr(std::size_t offset) const {
  if (!payload_.valid() || offset >= payload_.count()) {
    return nullptr;
  }
  return payload_.data() + offset;
}

std::unique_ptr<DeviceTensorFp32> LoadedModelCache::CreateTensorView(
    const std::string& tensor_name) const {
  const ModelCacheEntry* entry = FindEntry(tensor_name);
  if (entry == nullptr || entry->kind != ModelCacheEntryKind::kTensorFp32) {
    return nullptr;
  }
  const std::size_t numel = entry->payload_nbytes / sizeof(float);
  if (numel == 0) {
    return nullptr;
  }
  return DeviceTensorFp32::CreateView(
      {numel},
      reinterpret_cast<float*>(PayloadPtr(entry->payload_offset)));
}

std::unique_ptr<DeviceEmbeddingTableFp32> LoadedModelCache::CreateEmbeddingView(
    const std::string& tensor_name) const {
  const ModelCacheEntry* entry = FindEntry(tensor_name);
  if (entry == nullptr || entry->kind != ModelCacheEntryKind::kEmbeddingFp32 || entry->shape.size() != 2) {
    return nullptr;
  }
  return DeviceEmbeddingTableFp32::CreateView(
      entry->shape[0],
      entry->shape[1],
      reinterpret_cast<float*>(PayloadPtr(entry->payload_offset)));
}

std::unique_ptr<UploadedLinearOp> LoadedModelCache::CreateDenseLinearView(
    const GemmDescriptor& descriptor) const {
  const ModelCacheEntry* entry = FindEntry(descriptor.tensor_name);
  if (entry == nullptr || entry->kind != ModelCacheEntryKind::kDenseWeightFp32) {
    return nullptr;
  }
  auto weight = DeviceDenseWeightFp32::CreateView(
      entry->output_rows,
      entry->input_cols,
      reinterpret_cast<float*>(PayloadPtr(entry->payload_offset)));
  if (!weight || !weight->valid()) {
    return nullptr;
  }
  GemmDescriptor cached_descriptor = descriptor;
  cached_descriptor.storage_dtype = "fp32";
  cached_descriptor.compute_dtype = "fp32";
  cached_descriptor.layout_tag = "row_major";
  cached_descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  cached_descriptor.packed_data = nullptr;
  cached_descriptor.packed_nbytes = 0;
  return UploadedLinearOp::CreateDenseView(cached_descriptor, std::move(weight));
}

std::unique_ptr<ScaledFp8LinearOp> LoadedModelCache::CreateScaledFp8LinearView(
    const std::string& tensor_name,
    std::size_t output_rows,
    std::size_t input_cols) const {
  const ModelCacheEntry* entry = FindEntry(tensor_name);
  if (entry == nullptr || entry->kind != ModelCacheEntryKind::kScaledFp8WeightFp32) {
    return nullptr;
  }
  auto weight = DeviceDenseWeightFp32::CreateView(
      entry->output_rows,
      entry->input_cols,
      reinterpret_cast<float*>(PayloadPtr(entry->payload_offset)));
  if (!weight || !weight->valid()) {
    return nullptr;
  }
  ScaledFp8LinearConfig config;
  config.output_rows = output_rows;
  config.input_cols = input_cols;
  config.tensor_name = tensor_name;
  config.input_scale = entry->input_scale;
  config.weight_scale = entry->weight_scale;
  return ScaledFp8LinearOp::CreateView(config, std::move(weight));
}

std::unique_ptr<UploadedLinearOp> LoadedModelCache::CreateNvfp4LinearView(
    const GemmDescriptor& descriptor) const {
  const ModelCacheEntry* entry = FindEntry(descriptor.tensor_name);
  if (entry == nullptr || entry->kind != ModelCacheEntryKind::kNvfp4Aligned) {
    return nullptr;
  }
  auto weight = DeviceNvfp4Weight::CreateView(
      entry->output_rows,
      entry->input_cols,
      PayloadPtr(entry->payload_offset),
      entry->payload_nbytes,
      PayloadPtr(entry->aux0_offset),
      entry->aux0_nbytes,
      PayloadPtr(entry->aux1_offset),
      entry->aux1_nbytes,
      PayloadPtr(entry->aux2_offset),
      entry->aux2_nbytes);
  if (!weight || !weight->valid()) {
    return nullptr;
  }
  GemmDescriptor cached_descriptor = descriptor;
  cached_descriptor.packed_data = PayloadPtr(entry->payload_offset);
  cached_descriptor.packed_nbytes = entry->payload_nbytes;
  cached_descriptor.block_scales_data = PayloadPtr(entry->aux1_offset);
  cached_descriptor.block_scales_nbytes = entry->aux1_nbytes;
  cached_descriptor.tensor_scale_data = PayloadPtr(entry->aux2_offset);
  cached_descriptor.tensor_scale_nbytes = entry->aux2_nbytes;
  return UploadedLinearOp::CreateNvfp4View(cached_descriptor, std::move(weight));
}

}  // namespace nemotron
