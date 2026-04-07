#include "nemotron/artifact_loader.h"
#include "nemotron/manifest.h"
#include "nemotron/tensor_catalog.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using nemotron::ArtifactLoader;
using nemotron::BuildTensorCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::TensorAuxiliaryRole;
using nemotron::TensorCatalog;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_tensor_catalog_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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
    const std::string& auxiliary_file,
    const std::string& first_auxiliary_name,
    const std::string& second_auxiliary_name) {
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
      << "          \"name\": \"" << first_auxiliary_name << "\",\n"
      << "          \"file\": \"" << auxiliary_file << "\",\n"
      << "          \"offset_bytes\": 0,\n"
      << "          \"nbytes\": 4\n"
      << "        },\n"
      << "        {\n"
      << "          \"name\": \"" << second_auxiliary_name << "\",\n"
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

struct LoadedCatalog {
  ManifestLoadResult manifest_result;
  std::unique_ptr<ArtifactLoader> loader;
  TensorCatalog catalog;
};

LoadedCatalog load_catalog(const std::filesystem::path& manifest_path) {
  LoadedCatalog loaded;
  loaded.manifest_result = LoadVerifiedManifestFromJsonFile(manifest_path);
  if (!loaded.manifest_result.ok) {
    return loaded;
  }
  loaded.loader = ArtifactLoader::OpenVerified(loaded.manifest_result.manifest, manifest_path);
  if (!loaded.loader) {
    return loaded;
  }
  loaded.catalog = BuildTensorCatalog(loaded.manifest_result.manifest, *loaded.loader);
  return loaded;
}

bool test_tensor_catalog_resolves_scaled_tensor_roles() {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  write_file(packed_path, packed_bytes);
  write_file(auxiliary_path, "scale123");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/experts.bin",
          fnv1a64_hex(packed_bytes.substr(2, 5)),
          "weights/experts_scales.bin",
          "block_scales",
          "tensor_scale"));

  const LoadedCatalog loaded = load_catalog(manifest_path);
  const auto* tensor = loaded.catalog.FindTensor("layers.12.experts.117.w1");
  return expect(loaded.manifest_result.ok, "manifest should verify for tensor catalog build") &&
         expect(static_cast<bool>(loaded.loader), "artifact loader should stay alive while catalog views are used") &&
         expect(loaded.catalog.valid(), "tensor catalog should build from a verified manifest-backed loader") &&
         expect(tensor != nullptr, "tensor catalog should index the routed expert tensor") &&
         expect(tensor->has_block_scales(), "scaled tensor should expose block scales by role") &&
         expect(tensor->has_tensor_scale(), "scaled tensor should expose tensor scale by role") &&
         expect(view_as_string(tensor->packed_bytes) == "cdefg", "tensor catalog should preserve packed byte view") &&
         expect(
             view_as_string(tensor->FindAuxiliary(TensorAuxiliaryRole::kBlockScales)->bytes) == "scal",
             "block_scales role should resolve the first auxiliary slice") &&
         expect(
             view_as_string(tensor->FindAuxiliary(TensorAuxiliaryRole::kTensorScale)->bytes) == "e123",
             "tensor_scale role should resolve the second auxiliary slice") &&
         expect(loaded.catalog.indices_by_op_class().count("routed_expert") == 1,
                "tensor catalog should index tensors by op class") &&
         expect(loaded.catalog.indices_by_op_class().at("routed_expert").size() == 1,
                "op-class index should contain the routed expert tensor");
}

bool test_tensor_catalog_rejects_scaled_tensor_without_named_scale_roles() {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  write_file(packed_path, packed_bytes);
  write_file(auxiliary_path, "scale123");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/experts.bin",
          fnv1a64_hex(packed_bytes.substr(2, 5)),
          "weights/experts_scales.bin",
          "scale_blob_a",
          "scale_blob_b"));

  const LoadedCatalog loaded = load_catalog(manifest_path);
  return expect(loaded.manifest_result.ok, "manifest should still verify before typed catalog checks") &&
         expect(static_cast<bool>(loaded.loader), "artifact loader should open before typed catalog checks") &&
         expect(!loaded.catalog.valid(), "scaled tensor without named scale roles should invalidate the tensor catalog") &&
         expect(loaded.catalog.issues().size() >= 2, "catalog should report missing block and tensor scale roles");
}

}  // namespace

int main() {
  const bool ok =
      test_tensor_catalog_resolves_scaled_tensor_roles() &&
      test_tensor_catalog_rejects_scaled_tensor_without_named_scale_roles();

  if (!ok) {
    return 1;
  }
  std::cout << "tensor_catalog_test: PASS\n";
  return 0;
}
