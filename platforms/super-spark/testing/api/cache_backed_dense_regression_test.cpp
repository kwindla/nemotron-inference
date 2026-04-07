#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/model_cache.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/runtime_stats.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

constexpr std::size_t kHiddenSize = 4;
constexpr std::size_t kVocabSize = 8;
constexpr std::size_t kMambaIntermediateSize = 4;
constexpr std::size_t kMambaNumHeads = 1;
constexpr std::size_t kMambaHeadDim = 4;
constexpr std::size_t kMambaStateSize = 1;
constexpr std::size_t kMambaGroups = 1;
constexpr std::size_t kMambaConvKernelSize = 2;
constexpr std::size_t kMambaConvDim =
    kMambaIntermediateSize + (2 * kMambaGroups * kMambaStateSize);
constexpr std::size_t kMambaInProjRows =
    kMambaIntermediateSize + kMambaConvDim + kMambaNumHeads;
constexpr float kDenseAbsTol = 1.0e-4f;
constexpr float kDenseRelL2Tol = 1.0e-4f;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
    const char* current = std::getenv(name_);
    if (current != nullptr) {
      had_original_ = true;
      original_value_ = current;
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(name_, original_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

  void Unset() const {
    unsetenv(name_);
  }

 private:
  const char* name_;
  bool had_original_ = false;
  std::string original_value_;
};

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_cache_backed_dense_regression_test_" +
             std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct TensorFileSpec {
  std::string name;
  std::string op_class;
  std::vector<std::size_t> shape;
  std::string relative_path;
  std::vector<float> values;
};

struct SyntheticFixture {
  TempDir temp_dir;
  std::filesystem::path manifest_path;
  std::filesystem::path cache_path;
  nemotron::SingleTokenForwardConfig config;
  std::vector<float> in_proj_weight;
};

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

nemotron::RuntimeBootstrapOptions MakeOptions() {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(32);
  options.service_target.weights_bytes = GiB(1);
  options.service_target.workspace_bytes = GiB(1);
  options.service_target.graph_bytes = GiB(1);
  options.service_target.safety_headroom_bytes = GiB(1);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = 1;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

std::vector<float> MakeSequence(std::size_t count, float start, float step) {
  std::vector<float> values(count, 0.0f);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = start + step * static_cast<float>(i);
  }
  return values;
}

std::string ShapeJson(const std::vector<std::size_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

void WriteTextFile(const std::filesystem::path& path, const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

void WriteFloatFile(const std::filesystem::path& path, const std::vector<float>& values) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::string MakeManifestJson(const std::vector<TensorFileSpec>& tensors) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"cache-backed-dense-regression\",\n"
      << "  \"source_revision\": \"test\",\n"
      << "  \"tokenizer_revision\": \"test\",\n"
      << "  \"packer_version\": \"test-packer\",\n"
      << "  \"target_platform\": {\n"
      << "    \"gpu_family\": \"GB10\",\n"
      << "    \"compute_capability\": \"12.1\"\n"
      << "  },\n"
      << "  \"runtime_profile\": {\n"
      << "    \"kv_bytes_per_token\": 1024,\n"
      << "    \"mamba_state_bytes_fp16\": 1024,\n"
      << "    \"mamba_state_bytes_fp32\": 2048\n"
      << "  },\n"
      << "  \"tensors\": [\n";
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const TensorFileSpec& tensor = tensors[i];
    oss
        << "    {\n"
        << "      \"name\": \"" << tensor.name << "\",\n"
        << "      \"op_class\": \"" << tensor.op_class << "\",\n"
        << "      \"logical_shape\": " << ShapeJson(tensor.shape) << ",\n"
        << "      \"packed_shape\": " << ShapeJson(tensor.shape) << ",\n"
        << "      \"storage_dtype\": \"fp32\",\n"
        << "      \"compute_dtype\": \"fp32\",\n"
        << "      \"layout_tag\": \"row_major\",\n"
        << "      \"alignment_bytes\": 16,\n"
        << "      \"packed_file\": \"" << tensor.relative_path << "\",\n"
        << "      \"offset_bytes\": 0,\n"
        << "      \"nbytes\": " << tensor.values.size() * sizeof(float) << ",\n"
        << "      \"source_tensor_name\": \"" << tensor.name << "\",\n"
        << "      \"checksum\": \"ok\"\n"
        << "    }";
    oss << (i + 1 == tensors.size() ? "\n" : ",\n");
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

float RelativeL2Diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    return INFINITY;
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const double diff = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
    diff_sq += diff * diff;
    ref_sq += static_cast<double>(rhs[i]) * static_cast<double>(rhs[i]);
  }
  if (ref_sq == 0.0) {
    return diff_sq == 0.0 ? 0.0f : INFINITY;
  }
  return static_cast<float>(std::sqrt(diff_sq / ref_sq));
}

std::vector<float> CpuReferenceDense(
    const std::vector<float>& activations,
    std::size_t rows,
    const std::vector<float>& weight,
    std::size_t output_rows,
    std::size_t input_cols) {
  std::vector<float> output(rows * output_rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_rows; ++out) {
      float acc = 0.0f;
      for (std::size_t col = 0; col < input_cols; ++col) {
        acc += activations[row * input_cols + col] * weight[out * input_cols + col];
      }
      output[row * output_rows + out] = acc;
    }
  }
  return output;
}

std::int32_t Argmax(const std::vector<float>& values) {
  if (values.empty()) {
    return -1;
  }
  return static_cast<std::int32_t>(
      std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

bool InitializeFixture(SyntheticFixture* fixture) {
  if (fixture == nullptr) {
    return false;
  }

  fixture->config.hidden_size = kHiddenSize;
  fixture->config.total_layer_count = 1;
  fixture->config.vocab_size = kVocabSize;
  fixture->config.max_tokens = 1;
  fixture->config.attention_head_count = 1;
  fixture->config.attention_kv_head_count = 1;
  fixture->config.attention_head_dim = kHiddenSize;
  fixture->config.attention_tokens_per_page = 4;
  fixture->config.mamba_intermediate_size = kMambaIntermediateSize;
  fixture->config.mamba_num_heads = kMambaNumHeads;
  fixture->config.mamba_head_dim = kMambaHeadDim;
  fixture->config.mamba_state_size = kMambaStateSize;
  fixture->config.mamba_n_groups = kMambaGroups;
  fixture->config.mamba_conv_kernel_size = kMambaConvKernelSize;
  fixture->config.moe_latent_size = 1;
  fixture->config.routed_expert_intermediate_size = 1;
  fixture->config.shared_expert_intermediate_size = 1;
  fixture->config.n_routed_experts = 1;
  fixture->config.experts_per_token = 1;
  fixture->config.expert_n_group = 1;
  fixture->config.expert_topk_group = 1;
  fixture->config.layer_norm_epsilon = 1.0e-5f;
  fixture->config.mamba_time_step_min = 1.0e-3f;
  fixture->config.routed_scaling_factor = 1.0f;
  fixture->config.norm_topk_prob = true;

  std::vector<TensorFileSpec> tensors;
  tensors.push_back(TensorFileSpec{
      "backbone.embeddings.weight",
      "embedding",
      {kVocabSize, kHiddenSize},
      "weights/backbone_embeddings.bin",
      MakeSequence(kVocabSize * kHiddenSize, 0.02f, 0.01f),
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.norm.weight",
      "norm",
      {kHiddenSize},
      "weights/layer0_norm.bin",
      {1.00f, 0.90f, 1.10f, 0.95f},
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.norm.weight",
      "norm",
      {kMambaIntermediateSize},
      "weights/layer0_mixer_norm.bin",
      {0.98f, 1.02f, 1.05f, 0.92f},
  });
  fixture->in_proj_weight = MakeSequence(kMambaInProjRows * kHiddenSize, -0.20f, 0.015f);
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.in_proj.weight",
      "dense",
      {kMambaInProjRows, kHiddenSize},
      "weights/layer0_mixer_in_proj.bin",
      fixture->in_proj_weight,
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.conv1d.weight",
      "conv1d",
      {kMambaConvDim, 1, kMambaConvKernelSize},
      "weights/layer0_mixer_conv1d_weight.bin",
      MakeSequence(kMambaConvDim * kMambaConvKernelSize, -0.05f, 0.01f),
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.conv1d.bias",
      "bias",
      {kMambaConvDim},
      "weights/layer0_mixer_conv1d_bias.bin",
      {0.01f, -0.02f, 0.03f, -0.01f, 0.02f, -0.03f},
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.A_log",
      "mamba_state",
      {kMambaNumHeads},
      "weights/layer0_mixer_A_log.bin",
      {0.10f},
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.D",
      "mamba_state",
      {kMambaNumHeads},
      "weights/layer0_mixer_D.bin",
      {0.20f},
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.dt_bias",
      "mamba_state",
      {kMambaNumHeads},
      "weights/layer0_mixer_dt_bias.bin",
      {0.05f},
  });
  tensors.push_back(TensorFileSpec{
      "backbone.layers.0.mixer.out_proj.weight",
      "dense",
      {kHiddenSize, kMambaIntermediateSize},
      "weights/layer0_mixer_out_proj.bin",
      MakeSequence(kHiddenSize * kMambaIntermediateSize, 0.12f, -0.01f),
  });
  tensors.push_back(TensorFileSpec{
      "backbone.norm_f.weight",
      "norm",
      {kHiddenSize},
      "weights/backbone_norm_f.bin",
      {1.00f, 1.00f, 1.00f, 1.00f},
  });
  tensors.push_back(TensorFileSpec{
      "lm_head.weight",
      "dense",
      {kVocabSize, kHiddenSize},
      "weights/lm_head.bin",
      MakeSequence(kVocabSize * kHiddenSize, -0.08f, 0.02f),
  });

  for (const TensorFileSpec& tensor : tensors) {
    WriteFloatFile(fixture->temp_dir.path() / tensor.relative_path, tensor.values);
  }

  fixture->manifest_path = fixture->temp_dir.path() / "manifest.json";
  fixture->cache_path = fixture->temp_dir.path() / "model_cache" / "synthetic.cache";
  WriteTextFile(fixture->manifest_path, MakeManifestJson(tensors));
  return true;
}

bool TestCreateDenseLinearViewBuildsNativePlan(
    const SyntheticFixture& fixture,
    const nemotron::RuntimeEnvironment& environment,
    nemotron::CublasLtHandle& handle) {
  auto loaded_cache = nemotron::LoadedModelCache::Load(fixture.cache_path);
  if (!expect(loaded_cache != nullptr && loaded_cache->valid(), "synthetic model cache should load")) {
    return false;
  }

  const auto* descriptor =
      environment.gemm_catalog()->FindDescriptor("backbone.layers.0.mixer.in_proj.weight");
  if (!expect(descriptor != nullptr, "in-proj descriptor should exist in the synthetic GEMM catalog")) {
    return false;
  }
  const auto* entry = loaded_cache->FindEntry(descriptor->tensor_name);
  if (!expect(entry != nullptr, "cache should contain an entry for the in-proj tensor") ||
      !expect(entry->kind == nemotron::ModelCacheEntryKind::kDenseWeightFp32,
              "cache should store the in-proj tensor as a dense weight view")) {
    return false;
  }

  auto cache_backed_view = loaded_cache->CreateDenseLinearView(*descriptor);
  if (!expect(cache_backed_view != nullptr && cache_backed_view->valid(),
              "cache-backed dense linear view should create successfully")) {
    return false;
  }

  auto activations = nemotron::DeviceTensorFp32::Create({1, descriptor->input_cols});
  auto output = nemotron::DeviceTensorFp32::Create({1, descriptor->output_rows});
  if (!expect(activations != nullptr && activations->valid(), "input activation tensor should create") ||
      !expect(output != nullptr && output->valid(), "output tensor should create")) {
    return false;
  }

  const std::vector<float> activation_host = {0.25f, -0.50f, 0.75f, 1.00f};
  if (!expect(activations->CopyFromHost(activation_host.data(), activation_host.size()),
              "input activations should upload")) {
    return false;
  }

  nemotron::GemmHeuristicCache heuristic_cache;
  nemotron::ResetRuntimeExecutionStats();
  if (!expect(
          cache_backed_view->Run(handle, &heuristic_cache, *activations, output.get()),
          "cache-backed dense view should execute successfully")) {
    return false;
  }

  std::vector<float> output_host(output->numel(), 0.0f);
  if (!expect(output->CopyToHost(output_host.data(), output_host.size()),
              "cache-backed dense output should download")) {
    return false;
  }
  const std::vector<float> expected =
      CpuReferenceDense(activation_host, 1, fixture.in_proj_weight, descriptor->output_rows, descriptor->input_cols);
  const float max_abs_diff = MaxAbsDiff(output_host, expected);
  const float rel_l2 = RelativeL2Diff(output_host, expected);
  if (!expect(
          max_abs_diff <= kDenseAbsTol || rel_l2 <= kDenseRelL2Tol,
          "cache-backed dense output should match the CPU reference")) {
    return false;
  }

  const auto stats = nemotron::GetRuntimeExecutionStatsSnapshot();
  return expect(stats.dense_plan_build_failures == 0,
                "cache-backed dense view should build a native plan without dense plan failures") &&
         expect(stats.dense_reference_fallbacks == 0,
                "cache-backed dense view should not fall back to the dense reference surface") &&
         expect(stats.dense_native_success > 0,
                "cache-backed dense view should report native dense execution success");
}

bool TestCacheBackedSingleTokenDecodeKeepsDensePathNative(
    const SyntheticFixture& fixture,
    const nemotron::RuntimeEnvironment& environment) {
  auto direct_model = nemotron::SingleTokenForwardModel::Create(environment, fixture.config);
  if (!expect(direct_model != nullptr && direct_model->valid(),
              "direct synthetic single-token model should build")) {
    return false;
  }
  auto cache_model =
      nemotron::SingleTokenForwardModel::CreateFromCache(environment, fixture.config, fixture.cache_path);
  if (!expect(cache_model != nullptr && cache_model->valid(),
              "cache-backed synthetic single-token model should build")) {
    return false;
  }

  auto direct_request = direct_model->CreateRequestContext();
  auto cache_request = cache_model->CreateRequestContext();
  auto direct_logits = nemotron::DeviceTensorFp32::Create({1, fixture.config.vocab_size});
  auto cache_logits = nemotron::DeviceTensorFp32::Create({1, fixture.config.vocab_size});
  if (!expect(direct_request != nullptr && direct_request->valid(), "direct request context should create") ||
      !expect(cache_request != nullptr && cache_request->valid(), "cache-backed request context should create") ||
      !expect(direct_logits != nullptr && direct_logits->valid(), "direct logits tensor should create") ||
      !expect(cache_logits != nullptr && cache_logits->valid(), "cache-backed logits tensor should create")) {
    return false;
  }

  constexpr std::int32_t kTokenId = 3;
  if (!expect(
          direct_model->RunSingleToken(kTokenId, *direct_request, direct_logits.get()),
          "direct synthetic single-token decode should run")) {
    return false;
  }
  std::vector<float> direct_logits_host(direct_logits->numel(), 0.0f);
  if (!expect(direct_logits->CopyToHost(direct_logits_host.data(), direct_logits_host.size()),
              "direct logits should download")) {
    return false;
  }

  nemotron::ResetRuntimeExecutionStats();
  if (!expect(
          cache_model->RunSingleToken(kTokenId, *cache_request, cache_logits.get()),
          "cache-backed synthetic single-token decode should run")) {
    return false;
  }
  const auto stats = nemotron::GetRuntimeExecutionStatsSnapshot();

  std::vector<float> cache_logits_host(cache_logits->numel(), 0.0f);
  if (!expect(cache_logits->CopyToHost(cache_logits_host.data(), cache_logits_host.size()),
              "cache-backed logits should download")) {
    return false;
  }

  const float max_abs_diff = MaxAbsDiff(cache_logits_host, direct_logits_host);
  const float rel_l2 = RelativeL2Diff(cache_logits_host, direct_logits_host);
  return expect(stats.dense_plan_build_failures == 0,
                "cache-backed single-token decode should keep dense plan build failures at zero") &&
         expect(stats.dense_reference_fallbacks == 0,
                "cache-backed single-token decode should keep dense reference fallbacks at zero") &&
         expect(stats.dense_native_success > 0,
                "cache-backed single-token decode should record native dense successes") &&
         expect(Argmax(cache_logits_host) == Argmax(direct_logits_host),
                "cache-backed single-token decode should preserve the predicted token") &&
         expect(max_abs_diff <= kDenseAbsTol || rel_l2 <= kDenseRelL2Tol,
                "cache-backed single-token decode logits should match the direct model path");
}

}  // namespace

int main() {
  ScopedEnvVar disable_dense_surface("NEMOTRON_DISABLE_DENSE_DEVICE_PLAN_SURFACE");
  ScopedEnvVar dense_family_filter("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_FAMILY");
  ScopedEnvVar dense_tensor_filter("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_TENSORS");
  disable_dense_surface.Unset();
  dense_family_filter.Unset();
  dense_tensor_filter.Unset();

  if (!HasCudaDevice()) {
    std::cout << "cache_backed_dense_regression_test: SKIP (no CUDA device available)\n";
    return 0;
  }

  auto handle = nemotron::CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "cache_backed_dense_regression_test: SKIP (cublasLt unavailable)\n";
    return 0;
  }

  SyntheticFixture fixture;
  if (!expect(InitializeFixture(&fixture), "synthetic cache-backed dense fixture should initialize")) {
    return 1;
  }

  auto environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(
      fixture.manifest_path,
      MakeOptions());
  if (!expect(environment != nullptr, "synthetic runtime environment should build")) {
    return 1;
  }

  nemotron::ModelCacheWriteReport cache_report;
  if (!expect(
          nemotron::WriteDeterministicModelCache(
              *environment,
              fixture.config,
              fixture.cache_path,
              &cache_report),
          "synthetic model cache should write successfully") ||
      !expect(cache_report.entry_count > 0, "synthetic model cache should contain entries")) {
    return 1;
  }

  const bool ok =
      TestCreateDenseLinearViewBuildsNativePlan(fixture, *environment, *handle) &&
      TestCacheBackedSingleTokenDecodeKeepsDensePathNative(fixture, *environment);
  if (!ok) {
    return 1;
  }

  std::cout << "cache_backed_dense_regression_test: PASS\n";
  return 0;
}
