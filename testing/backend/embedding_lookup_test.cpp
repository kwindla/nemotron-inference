#include "nemotron/artifact_loader.h"
#include "nemotron/device_tensor.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/embedding_table.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using nemotron::ArtifactLoader;
using nemotron::BuildEmbeddingCatalog;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::DeviceEmbeddingTableBf16;
using nemotron::DeviceEmbeddingTableFp32;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::EmbeddingCatalog;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::LookupEmbeddingRowsBf16;
using nemotron::LookupEmbeddingRowsFp32;
using nemotron::ManifestLoadResult;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_embedding_lookup_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct LoadedEmbeddingCatalog {
  ManifestLoadResult manifest_result;
  std::unique_ptr<ArtifactLoader> loader;
  TensorCatalog tensor_catalog;
  std::unique_ptr<WeightArena> arena;
  KernelCatalog kernel_catalog;
  EmbeddingCatalog embedding_catalog;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
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

template <typename T>
std::string bytes_from_vector(const std::vector<T>& values) {
  return std::string(
      reinterpret_cast<const char*>(values.data()),
      values.size() * sizeof(T));
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

std::string make_manifest_json(
    const std::string& embedding_file,
    const std::string& embedding_checksum,
    const std::string& storage_dtype = "fp32",
    const std::string& compute_dtype = "fp32",
    std::size_t nbytes = 48) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"embedding-lookup-test\",\n"
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
      << "      \"logical_shape\": [4, 3],\n"
      << "      \"packed_shape\": [4, 3],\n"
      << "      \"storage_dtype\": \"" << storage_dtype << "\",\n"
      << "      \"compute_dtype\": \"" << compute_dtype << "\",\n"
      << "      \"layout_tag\": \"row_major\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << embedding_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": " << nbytes << ",\n"
      << "      \"source_tensor_name\": \"backbone.embeddings.weight\",\n"
      << "      \"checksum\": \"fnv1a64:" << embedding_checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

LoadedEmbeddingCatalog load_embedding_catalog(const std::filesystem::path& manifest_path) {
  LoadedEmbeddingCatalog loaded;
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
  if (!loaded.kernel_catalog.valid()) {
    return loaded;
  }
  loaded.embedding_catalog = BuildEmbeddingCatalog(loaded.kernel_catalog);
  return loaded;
}

std::vector<float> cpu_reference(
    const std::vector<float>& table,
    const std::vector<std::int32_t>& token_ids,
    std::size_t embedding_dim) {
  std::vector<float> output(token_ids.size() * embedding_dim, 0.0f);
  for (std::size_t token_index = 0; token_index < token_ids.size(); ++token_index) {
    const std::size_t token_id = static_cast<std::size_t>(token_ids[token_index]);
    for (std::size_t dim = 0; dim < embedding_dim; ++dim) {
      output[token_index * embedding_dim + dim] = table[token_id * embedding_dim + dim];
    }
  }
  return output;
}

bool nearly_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, float tol) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = lhs[i] - rhs[i];
    if (diff > tol || diff < -tol) {
      return false;
    }
  }
  return true;
}

std::vector<__nv_bfloat16> to_bf16_vector(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = __float2bfloat16(values[i]);
  }
  return converted;
}

std::vector<float> from_bf16_vector(const std::vector<__nv_bfloat16>& values) {
  std::vector<float> converted(values.size(), 0.0f);
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = __bfloat162float(values[i]);
  }
  return converted;
}

bool test_embedding_lookup_matches_cpu_reference() {
  if (!has_cuda_device()) {
    std::cout << "embedding_lookup_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> embedding_weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes = bytes_from_vector(embedding_weights);
  write_file(temp_dir.path() / "weights" / "embedding.bin", embedding_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json("weights/embedding.bin", fnv1a64_hex(embedding_bytes)));

  const LoadedEmbeddingCatalog loaded = load_embedding_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.embedding_catalog.valid(), "embedding catalog should load before lookup")) {
    return false;
  }
  const auto* descriptor = loaded.embedding_catalog.FindDescriptor("backbone.embeddings.weight");
  if (!expect(descriptor != nullptr, "embedding descriptor should exist before lookup")) {
    return false;
  }

  auto table = DeviceEmbeddingTableFp32::Upload(*descriptor);
  if (!expect(static_cast<bool>(table), "embedding table upload should succeed for fp32 row-major weights")) {
    return false;
  }

  const std::vector<std::int32_t> token_ids = {2, 0};
  auto output = DeviceTensorFp32::Create({token_ids.size(), descriptor->embedding_dim});
  if (!expect(output && output->valid(), "device output tensor should allocate")) {
    return false;
  }

  const auto stats = LookupEmbeddingRowsFp32(*table, token_ids.data(), token_ids.size(), output.get());
  if (!expect(stats.has_value(), "embedding lookup should execute successfully")) {
    return false;
  }

  std::vector<float> host_output(output->numel(), 0.0f);
  if (!expect(output->CopyToHost(host_output.data(), host_output.size()), "embedding lookup output should download")) {
    return false;
  }

  const auto reference = cpu_reference(embedding_weights, token_ids, descriptor->embedding_dim);
  return expect(stats->token_count == token_ids.size() && stats->embedding_dim == descriptor->embedding_dim,
                "embedding lookup should report the expected shape") &&
         expect(nearly_equal(host_output, reference, 1e-6f),
                "embedding lookup output should match the CPU reference");
}

bool test_embedding_lookup_rejects_out_of_range_token() {
  if (!has_cuda_device()) {
    std::cout << "embedding_lookup_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> embedding_weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes = bytes_from_vector(embedding_weights);
  write_file(temp_dir.path() / "weights" / "embedding.bin", embedding_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json("weights/embedding.bin", fnv1a64_hex(embedding_bytes)));

  const LoadedEmbeddingCatalog loaded = load_embedding_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.embedding_catalog.valid(), "embedding catalog should load before invalid token test")) {
    return false;
  }
  const auto* descriptor = loaded.embedding_catalog.FindDescriptor("backbone.embeddings.weight");
  if (!expect(descriptor != nullptr, "embedding descriptor should exist before invalid token test")) {
    return false;
  }

  auto table = DeviceEmbeddingTableFp32::Upload(*descriptor);
  if (!expect(static_cast<bool>(table), "embedding table upload should succeed before invalid token test")) {
    return false;
  }

  const std::vector<std::int32_t> token_ids = {5};
  auto output = DeviceTensorFp32::Create({1, descriptor->embedding_dim});
  if (!expect(output && output->valid(), "device output tensor should allocate before invalid token test")) {
    return false;
  }

  return expect(!LookupEmbeddingRowsFp32(*table, token_ids.data(), token_ids.size(), output.get()).has_value(),
                "embedding lookup should reject out-of-range tokens");
}

bool test_embedding_lookup_accepts_bf16_weights() {
  if (!has_cuda_device()) {
    std::cout << "embedding_lookup_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> embedding_weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  std::vector<__nv_bfloat16> embedding_weights_bf16(embedding_weights.size());
  for (std::size_t i = 0; i < embedding_weights.size(); ++i) {
    embedding_weights_bf16[i] = __float2bfloat16(embedding_weights[i]);
  }
  const std::string embedding_bytes = bytes_from_vector(embedding_weights_bf16);
  write_file(temp_dir.path() / "weights" / "embedding.bin", embedding_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "bf16",
          "fp32",
          embedding_bytes.size()));

  const LoadedEmbeddingCatalog loaded = load_embedding_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.embedding_catalog.valid(), "bf16 embedding catalog should load before lookup")) {
    return false;
  }
  const auto* descriptor = loaded.embedding_catalog.FindDescriptor("backbone.embeddings.weight");
  if (!expect(descriptor != nullptr, "bf16 embedding descriptor should exist before lookup")) {
    return false;
  }

  auto table = DeviceEmbeddingTableFp32::Upload(*descriptor);
  if (!expect(static_cast<bool>(table), "embedding table upload should succeed for bf16 row-major weights")) {
    return false;
  }

  const std::vector<std::int32_t> token_ids = {3, 1};
  auto output = DeviceTensorFp32::Create({token_ids.size(), descriptor->embedding_dim});
  if (!expect(output && output->valid(), "device output tensor should allocate for bf16 lookup")) {
    return false;
  }

  const auto stats = LookupEmbeddingRowsFp32(*table, token_ids.data(), token_ids.size(), output.get());
  if (!expect(stats.has_value(), "bf16 embedding lookup should execute successfully")) {
    return false;
  }

  std::vector<float> host_output(output->numel(), 0.0f);
  if (!expect(output->CopyToHost(host_output.data(), host_output.size()), "bf16 embedding lookup output should download")) {
    return false;
  }

  const auto reference = cpu_reference(embedding_weights, token_ids, descriptor->embedding_dim);
  return expect(nearly_equal(host_output, reference, 1e-3f),
                "bf16 embedding lookup output should match the CPU reference");
}

bool test_embedding_lookup_bf16_path_matches_fp32_lookup() {
  if (!has_cuda_device()) {
    std::cout << "embedding_lookup_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> embedding_weights = {
      1.125f, -2.03125f, 3.0625f,
      4.1875f, -5.28125f, 6.34375f,
      7.40625f, -8.53125f, 9.6875f,
      10.8125f, -11.9375f, 12.0625f,
  };
  const auto embedding_weights_bf16 = to_bf16_vector(embedding_weights);
  const std::string embedding_bytes_bf16 = bytes_from_vector(embedding_weights_bf16);
  write_file(temp_dir.path() / "weights" / "embedding_bf16.bin", embedding_bytes_bf16);
  write_file(
      temp_dir.path() / "manifest_bf16.json",
      make_manifest_json(
          "weights/embedding_bf16.bin",
          fnv1a64_hex(embedding_bytes_bf16),
          "bf16",
          "bf16",
          embedding_bytes_bf16.size()));

  const std::string embedding_bytes_fp32 = bytes_from_vector(embedding_weights);
  write_file(temp_dir.path() / "weights" / "embedding_fp32.bin", embedding_bytes_fp32);
  write_file(
      temp_dir.path() / "manifest_fp32.json",
      make_manifest_json(
          "weights/embedding_fp32.bin",
          fnv1a64_hex(embedding_bytes_fp32),
          "fp32",
          "fp32",
          embedding_bytes_fp32.size()));

  const LoadedEmbeddingCatalog loaded_bf16 = load_embedding_catalog(temp_dir.path() / "manifest_bf16.json");
  const LoadedEmbeddingCatalog loaded_fp32 = load_embedding_catalog(temp_dir.path() / "manifest_fp32.json");
  if (!expect(loaded_bf16.embedding_catalog.valid(), "bf16 embedding catalog should load for bf16 path test") ||
      !expect(loaded_fp32.embedding_catalog.valid(), "fp32 embedding catalog should load for bf16 path test")) {
    return false;
  }

  const auto* descriptor_bf16 = loaded_bf16.embedding_catalog.FindDescriptor("backbone.embeddings.weight");
  const auto* descriptor_fp32 = loaded_fp32.embedding_catalog.FindDescriptor("backbone.embeddings.weight");
  if (!expect(descriptor_bf16 != nullptr, "bf16 embedding descriptor should exist for bf16 path test") ||
      !expect(descriptor_fp32 != nullptr, "fp32 embedding descriptor should exist for bf16 path test")) {
    return false;
  }

  auto table_bf16 = DeviceEmbeddingTableBf16::Upload(*descriptor_bf16);
  auto table_fp32 = DeviceEmbeddingTableFp32::Upload(*descriptor_fp32);
  if (!expect(static_cast<bool>(table_bf16), "bf16 embedding upload should succeed for bf16 path test") ||
      !expect(static_cast<bool>(table_fp32), "fp32 embedding upload should succeed for bf16 path test")) {
    return false;
  }

  std::vector<__nv_bfloat16> round_trip_weights(embedding_weights_bf16.size(), __float2bfloat16(0.0f));
  if (!expect(
          cudaMemcpy(
              round_trip_weights.data(),
              table_bf16->data(),
              round_trip_weights.size() * sizeof(__nv_bfloat16),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "bf16 embedding table should stay resident as bf16")) {
    return false;
  }

  const std::vector<std::int32_t> token_ids = {3, 1, 2};
  auto output_bf16 = DeviceTensorBf16::Create({token_ids.size(), descriptor_bf16->embedding_dim});
  auto output_fp32 = DeviceTensorFp32::Create({token_ids.size(), descriptor_fp32->embedding_dim});
  if (!expect(output_bf16 && output_bf16->valid(), "bf16 embedding output tensor should allocate") ||
      !expect(output_fp32 && output_fp32->valid(), "fp32 embedding output tensor should allocate")) {
    return false;
  }

  const auto bf16_stats = LookupEmbeddingRowsBf16(*table_bf16, token_ids.data(), token_ids.size(), output_bf16.get());
  const auto fp32_stats = LookupEmbeddingRowsFp32(*table_fp32, token_ids.data(), token_ids.size(), output_fp32.get());
  if (!expect(bf16_stats.has_value(), "bf16 embedding lookup should succeed") ||
      !expect(fp32_stats.has_value(), "fp32 embedding lookup should succeed in bf16 path test")) {
    return false;
  }

  std::vector<__nv_bfloat16> host_output_bf16(output_bf16->numel(), __float2bfloat16(0.0f));
  std::vector<float> host_output_fp32(output_fp32->numel(), 0.0f);
  if (!expect(output_bf16->CopyToHost(host_output_bf16.data(), host_output_bf16.size()),
              "bf16 embedding output should download") ||
      !expect(output_fp32->CopyToHost(host_output_fp32.data(), host_output_fp32.size()),
              "fp32 embedding output should download")) {
    return false;
  }

  const std::vector<float> bf16_output_fp32 = from_bf16_vector(host_output_bf16);
  const std::vector<float> fp32_output_bf16_round_trip = from_bf16_vector(to_bf16_vector(host_output_fp32));
  const auto reference = cpu_reference(from_bf16_vector(round_trip_weights), token_ids, descriptor_bf16->embedding_dim);
  return expect(
             nearly_equal(from_bf16_vector(round_trip_weights), from_bf16_vector(embedding_weights_bf16), 1.0e-6f),
             "bf16 embedding upload should preserve the original bf16 bytes") &&
         expect(
             nearly_equal(bf16_output_fp32, fp32_output_bf16_round_trip, 1.0e-6f),
             "bf16 embedding lookup should match fp32 lookup after bf16 round-trip") &&
         expect(
             nearly_equal(bf16_output_fp32, reference, 1.0e-6f),
             "bf16 embedding lookup should match the bf16-rounded reference");
}

}  // namespace

int main() {
  const bool ok =
      test_embedding_lookup_matches_cpu_reference() &&
      test_embedding_lookup_rejects_out_of_range_token() &&
      test_embedding_lookup_accepts_bf16_weights() &&
      test_embedding_lookup_bf16_path_matches_fp32_lookup();

  if (!ok) {
    return 1;
  }
  std::cout << "embedding_lookup_test: PASS\n";
  return 0;
}
