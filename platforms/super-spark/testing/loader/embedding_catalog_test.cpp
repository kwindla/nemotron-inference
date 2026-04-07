#include "nemotron/artifact_loader.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

using nemotron::ArtifactLoader;
using nemotron::BuildEmbeddingCatalog;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::EmbeddingCatalog;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_embedding_catalog_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct LoadedKernelCatalog {
  ManifestLoadResult manifest_result;
  std::unique_ptr<ArtifactLoader> loader;
  TensorCatalog tensor_catalog;
  std::unique_ptr<WeightArena> arena;
  KernelCatalog kernel_catalog;
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

std::string make_manifest_json(
    const std::string& embedding_file,
    const std::string& embedding_checksum,
    const std::string& projection_file,
    const std::string& projection_checksum,
    const std::string& embedding_layout) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"embedding-catalog-test\",\n"
      << "  \"source_revision\": \"test\",\n"
      << "  \"tokenizer_revision\": \"test\",\n"
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
      << "      \"name\": \"backbone.embeddings.weight\",\n"
      << "      \"op_class\": \"embedding\",\n"
      << "      \"logical_shape\": [4, 2],\n"
      << "      \"packed_shape\": [4, 2],\n"
      << "      \"storage_dtype\": \"bf16\",\n"
      << "      \"compute_dtype\": \"bf16\",\n"
      << "      \"layout_tag\": \"" << embedding_layout << "\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << embedding_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 16,\n"
      << "      \"source_tensor_name\": \"backbone.embeddings.weight\",\n"
      << "      \"checksum\": \"fnv1a64:" << embedding_checksum << "\"\n"
      << "    },\n"
      << "    {\n"
      << "      \"name\": \"mlp.up_proj\",\n"
      << "      \"op_class\": \"dense_linear\",\n"
      << "      \"logical_shape\": [2, 2],\n"
      << "      \"packed_shape\": [2, 2],\n"
      << "      \"storage_dtype\": \"bf16\",\n"
      << "      \"compute_dtype\": \"bf16\",\n"
      << "      \"layout_tag\": \"row_major\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << projection_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 8,\n"
      << "      \"source_tensor_name\": \"mlp.up_proj\",\n"
      << "      \"checksum\": \"fnv1a64:" << projection_checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

LoadedKernelCatalog load_kernel_catalog(const std::filesystem::path& manifest_path) {
  LoadedKernelCatalog loaded;
  loaded.manifest_result = LoadVerifiedManifestFromJsonFile(manifest_path);
  if (!loaded.manifest_result.ok) {
    return loaded;
  }
  loaded.loader = ArtifactLoader::OpenVerified(loaded.manifest_result.manifest, manifest_path);
  if (!loaded.loader) {
    return loaded;
  }
  loaded.tensor_catalog = BuildTensorCatalog(loaded.manifest_result.manifest, *loaded.loader);
  if (!loaded.tensor_catalog.valid()) {
    return loaded;
  }
  const auto plan = BuildWeightArenaPlan(loaded.tensor_catalog);
  if (!plan.valid()) {
    return loaded;
  }
  loaded.arena = WeightArena::CreateFromPlan(plan);
  if (!loaded.arena) {
    return loaded;
  }
  loaded.kernel_catalog = BuildKernelCatalog(loaded.tensor_catalog, *loaded.arena);
  return loaded;
}

bool test_embedding_catalog_indexes_row_major_embeddings() {
  TempDir temp_dir;
  const std::string embedding_bytes(16, 'e');
  const std::string projection_bytes(8, 'p');
  write_file(temp_dir.path() / "weights" / "embedding.bin", embedding_bytes);
  write_file(temp_dir.path() / "weights" / "projection.bin", projection_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/projection.bin",
          fnv1a64_hex(projection_bytes),
          "row_major"));

  const LoadedKernelCatalog loaded = load_kernel_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.kernel_catalog.valid(), "kernel catalog should build before embedding classification")) {
    return false;
  }

  const EmbeddingCatalog catalog = BuildEmbeddingCatalog(loaded.kernel_catalog);
  const auto* descriptor = catalog.FindDescriptor("backbone.embeddings.weight");
  return expect(catalog.valid(), "embedding catalog should accept row-major embeddings") &&
         expect(catalog.descriptors().size() == 1, "embedding catalog should index only embedding tensors") &&
         expect(descriptor != nullptr, "embedding descriptor should be discoverable by name") &&
         expect(descriptor->vocab_size == 4 && descriptor->embedding_dim == 2,
                "embedding descriptor should preserve logical shape") &&
         expect(descriptor->storage_dtype == "bf16" && descriptor->compute_dtype == "bf16",
                "embedding descriptor should preserve dtype metadata") &&
         expect(catalog.FindDescriptor("mlp.up_proj") == nullptr,
                "embedding catalog should not expose non-embedding tensors");
}

bool test_embedding_catalog_rejects_unsupported_layout() {
  TempDir temp_dir;
  const std::string embedding_bytes(16, 'e');
  const std::string projection_bytes(8, 'p');
  write_file(temp_dir.path() / "weights" / "embedding.bin", embedding_bytes);
  write_file(temp_dir.path() / "weights" / "projection.bin", projection_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/projection.bin",
          fnv1a64_hex(projection_bytes),
          "unknown_layout_v0"));

  const LoadedKernelCatalog loaded = load_kernel_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.kernel_catalog.valid(), "kernel catalog should build before invalid embedding classification")) {
    return false;
  }

  const EmbeddingCatalog catalog = BuildEmbeddingCatalog(loaded.kernel_catalog);
  return expect(!catalog.valid(), "embedding catalog should reject unsupported embedding layouts") &&
         expect(!catalog.issues().empty(), "invalid embedding layout should report issues");
}

}  // namespace

int main() {
  const bool ok =
      test_embedding_catalog_indexes_row_major_embeddings() &&
      test_embedding_catalog_rejects_unsupported_layout();

  if (!ok) {
    return 1;
  }
  std::cout << "embedding_catalog_test: PASS\n";
  return 0;
}
