#include "nemotron/artifact_loader.h"
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
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_kernel_catalog_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct LoadedCatalog {
  ManifestLoadResult manifest_result;
  std::unique_ptr<ArtifactLoader> loader;
  TensorCatalog tensor_catalog;
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

LoadedCatalog load_tensor_catalog(const std::filesystem::path& manifest_path) {
  LoadedCatalog loaded;
  loaded.manifest_result = LoadVerifiedManifestFromJsonFile(manifest_path);
  if (!loaded.manifest_result.ok) {
    return loaded;
  }
  loaded.loader = ArtifactLoader::OpenVerified(loaded.manifest_result.manifest, manifest_path);
  if (!loaded.loader) {
    return loaded;
  }
  loaded.tensor_catalog = BuildTensorCatalog(loaded.manifest_result.manifest, *loaded.loader);
  return loaded;
}

bool test_kernel_catalog_exposes_arena_resident_descriptors() {
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

  const LoadedCatalog loaded = load_tensor_catalog(manifest_path);
  const auto plan = BuildWeightArenaPlan(loaded.tensor_catalog);
  const auto arena = WeightArena::CreateFromPlan(plan);
  const KernelCatalog kernel_catalog = BuildKernelCatalog(loaded.tensor_catalog, *arena);
  const auto* tensor = kernel_catalog.FindTensor("layers.12.experts.117.w1");
  return expect(loaded.manifest_result.ok, "manifest should verify for kernel catalog build") &&
         expect(loaded.tensor_catalog.valid(), "tensor catalog should be valid for kernel catalog build") &&
         expect(plan.valid(), "weight arena plan should be valid for kernel catalog build") &&
         expect(static_cast<bool>(arena), "weight arena should materialize before kernel descriptor build") &&
         expect(kernel_catalog.valid(), "kernel catalog should build from valid catalog and arena") &&
         expect(tensor != nullptr, "kernel catalog should index the routed expert tensor") &&
         expect(tensor->is_scaled(), "scaled routed expert should expose scale pointers") &&
         expect(tensor->packed_nbytes == 5, "kernel descriptor should preserve packed byte count") &&
         expect(tensor->block_scales_nbytes == 4, "kernel descriptor should preserve block scale byte count") &&
         expect(tensor->tensor_scale_nbytes == 4, "kernel descriptor should preserve tensor scale byte count") &&
         expect(tensor->packed_bytes().valid(), "kernel descriptor should expose packed bytes") &&
         expect(tensor->block_scales_bytes().valid(), "kernel descriptor should expose block scale bytes") &&
         expect(tensor->tensor_scale_bytes().valid(), "kernel descriptor should expose tensor scale bytes") &&
         expect(kernel_catalog.indices_by_op_class().count("routed_expert") == 1,
                "kernel catalog should index descriptors by op class");
}

}  // namespace

int main() {
  const bool ok =
      test_kernel_catalog_exposes_arena_resident_descriptors();

  if (!ok) {
    return 1;
  }
  std::cout << "kernel_catalog_test: PASS\n";
  return 0;
}
