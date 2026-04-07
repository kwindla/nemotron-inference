#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace nemotron {

struct TensorAuxiliaryManifest {
  std::string name;
  std::string file;
  std::size_t offset_bytes = 0;
  std::size_t nbytes = 0;
};

struct TensorManifestEntry {
  std::string name;
  std::string op_class;
  std::vector<std::size_t> logical_shape;
  std::vector<std::size_t> packed_shape;
  std::string storage_dtype;
  std::string compute_dtype;
  std::string layout_tag;
  std::size_t alignment_bytes = 0;
  std::string block_scale_mode;
  std::string block_scale_dtype;
  std::string tensor_scale_dtype;
  std::string packed_file;
  std::size_t offset_bytes = 0;
  std::size_t nbytes = 0;
  std::vector<TensorAuxiliaryManifest> auxiliaries;
  std::string source_tensor_name;
  std::string checksum;
};

struct RuntimeManifestMetadata {
  std::string model_id;
  std::string source_revision;
  std::string tokenizer_revision;
  std::string packer_version;
  std::string gpu_family;
  std::string compute_capability;
  std::size_t kv_bytes_per_token = 0;
  std::size_t mamba_state_bytes_fp16 = 0;
  std::size_t mamba_state_bytes_fp32 = 0;
};

struct PackedModelManifest {
  std::uint32_t schema_version = 0;
  RuntimeManifestMetadata runtime;
  std::vector<TensorManifestEntry> tensors;
};

enum class ManifestIssueSeverity {
  kWarning,
  kError,
};

struct ManifestValidationIssue {
  ManifestIssueSeverity severity = ManifestIssueSeverity::kError;
  std::string tensor_name;
  std::string message;
};

struct ManifestLoadResult {
  bool ok = false;
  PackedModelManifest manifest;
  std::vector<ManifestValidationIssue> issues;
};

std::vector<ManifestValidationIssue> ValidateManifest(const PackedModelManifest& manifest);
bool HasManifestErrors(const std::vector<ManifestValidationIssue>& issues);
ManifestLoadResult LoadManifestFromJsonFile(const std::filesystem::path& manifest_path);
std::vector<ManifestValidationIssue> VerifyManifestFiles(
    const PackedModelManifest& manifest,
    const std::filesystem::path& manifest_path);
ManifestLoadResult LoadVerifiedManifestFromJsonFile(const std::filesystem::path& manifest_path);

}  // namespace nemotron
