#include "nemotron/artifact_loader.h"
#include "nemotron/manifest.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using nemotron::ArtifactLoader;
using nemotron::ArtifactLoadMode;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::TensorArtifactView;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_artifact_loader_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::string fnv1a64_hex(const std::string& bytes) {
  constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
  constexpr std::uint64_t kFnvPrime = 1099511628211ull;
  std::uint64_t hash = kFnvOffsetBasis;
  for (unsigned char ch : bytes) {
    hash ^= ch;
    hash *= kFnvPrime;
  }
  std::ostringstream oss;
  oss << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << hash;
  return oss.str();
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output << contents;
}

std::string view_as_string(nemotron::ByteRangeView view) {
  return std::string(
      reinterpret_cast<const char*>(view.data),
      reinterpret_cast<const char*>(view.data + view.size));
}

std::string make_manifest_json(
    const std::string& packed_file,
    const std::string& checksum,
    const std::string& auxiliary_file) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4\",\n"
      << "  \"source_revision\": \"b1ffe499\",\n"
      << "  \"tokenizer_revision\": \"b1ffe499\",\n"
      << "  \"packer_version\": \"test-packer\",\n"
      << "  \"target_platform\": {\n"
      << "    \"gpu_family\": \"GB10\",\n"
      << "    \"compute_capability\": \"12.1\"\n"
      << "  },\n"
      << "  \"runtime_profile\": {\n"
      << "    \"kv_bytes_per_token\": 4096,\n"
      << "    \"mamba_state_bytes_fp16\": 87162880,\n"
      << "    \"mamba_state_bytes_fp32\": 174325760\n"
      << "  },\n"
      << "  \"tensors\": [\n"
      << "    {\n"
      << "      \"name\": \"layers.12.experts.117.w1\",\n"
      << "      \"op_class\": \"routed_expert\",\n"
      << "      \"logical_shape\": [2688, 1024],\n"
      << "      \"packed_shape\": [2688, 1024],\n"
      << "      \"storage_dtype\": \"nvfp4_e2m1\",\n"
      << "      \"compute_dtype\": \"fp32_accum\",\n"
      << "      \"layout_tag\": \"cublaslt_fp4_tn_v1\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"block_scale_mode\": \"vec16_e4m3\",\n"
      << "      \"block_scale_dtype\": \"e4m3\",\n"
      << "      \"tensor_scale_dtype\": \"fp32\",\n"
      << "      \"packed_file\": \"" << packed_file << "\",\n"
      << "      \"offset_bytes\": 2,\n"
      << "      \"nbytes\": 5,\n"
      << "      \"auxiliaries\": [\n"
      << "        {\n"
      << "          \"name\": \"block_scales\",\n"
      << "          \"file\": \"" << auxiliary_file << "\",\n"
      << "          \"offset_bytes\": 0,\n"
      << "          \"nbytes\": 4\n"
      << "        },\n"
      << "        {\n"
      << "          \"name\": \"tensor_scale\",\n"
      << "          \"file\": \"" << auxiliary_file << "\",\n"
      << "          \"offset_bytes\": 4,\n"
      << "          \"nbytes\": 4\n"
      << "        }\n"
      << "      ],\n"
      << "      \"source_tensor_name\": \"layers.12.experts.117.w1\",\n"
      << "      \"checksum\": \"fnv1a64:" << checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

bool test_artifact_loader_maps_tensor_and_auxiliary_ranges(ArtifactLoadMode mode) {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  write_file(packed_path, packed_bytes);
  write_file(auxiliary_path, "scale123");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json("weights/experts.bin", fnv1a64_hex(packed_bytes.substr(2, 5)), "weights/experts_scales.bin"));

  const ManifestLoadResult result = LoadVerifiedManifestFromJsonFile(manifest_path);
  const auto loader = ArtifactLoader::OpenVerifiedWithMode(result.manifest, manifest_path, mode);
  if (!expect(result.ok, "manifest should verify before artifact mapping")) {
    return false;
  }
  if (!expect(static_cast<bool>(loader), "artifact loader should open a verified manifest")) {
    return false;
  }

  const std::optional<TensorArtifactView> tensor = loader->FindTensor("layers.12.experts.117.w1");
  return expect(tensor.has_value(), "mapped tensor should be discoverable by name") &&
         expect(tensor->packed_bytes.valid(), "packed tensor bytes should be valid") &&
         expect(view_as_string(tensor->packed_bytes) == "cdefg", "packed tensor bytes should match declared slice") &&
         expect(tensor->auxiliaries.size() == 2, "tensor should expose both auxiliary ranges") &&
         expect(tensor->auxiliaries[0].name == "block_scales", "first auxiliary name should round-trip") &&
         expect(view_as_string(tensor->auxiliaries[0].bytes) == "scal", "first auxiliary slice should match") &&
         expect(tensor->auxiliaries[1].name == "tensor_scale", "second auxiliary name should round-trip") &&
         expect(view_as_string(tensor->auxiliaries[1].bytes) == "e123", "second auxiliary slice should match") &&
         expect(loader->load_mode() == mode, "loader should report the selected load mode") &&
         expect(loader->loaded_file_count() == 2, "loader should count each unique backing file once") &&
         expect(loader->total_loaded_bytes() == 18, "loader should report total loaded backing-file bytes") &&
         expect(loader->mapped_file_count() == 2, "loader should map each unique backing file once") &&
         expect(loader->total_mapped_bytes() == 18, "loader should report total mapped backing-file bytes");
}

bool test_artifact_loader_reports_missing_tensor(ArtifactLoadMode mode) {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  write_file(packed_path, packed_bytes);
  write_file(auxiliary_path, "scale123");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json("weights/experts.bin", fnv1a64_hex(packed_bytes.substr(2, 5)), "weights/experts_scales.bin"));

  const ManifestLoadResult result = LoadVerifiedManifestFromJsonFile(manifest_path);
  const auto loader = ArtifactLoader::OpenVerifiedWithMode(result.manifest, manifest_path, mode);
  return expect(result.ok, "manifest should verify for missing-tensor test") &&
         expect(static_cast<bool>(loader), "artifact loader should open for missing-tensor test") &&
         expect(!loader->FindTensor("missing.tensor").has_value(), "unknown tensors should return no view");
}

}  // namespace

int main() {
  const bool ok =
      test_artifact_loader_maps_tensor_and_auxiliary_ranges(ArtifactLoadMode::kMmap) &&
      test_artifact_loader_maps_tensor_and_auxiliary_ranges(ArtifactLoadMode::kReadAll) &&
      test_artifact_loader_reports_missing_tensor(ArtifactLoadMode::kMmap) &&
      test_artifact_loader_reports_missing_tensor(ArtifactLoadMode::kReadAll);

  if (!ok) {
    return 1;
  }
  std::cout << "artifact_loader_test: PASS\n";
  return 0;
}
