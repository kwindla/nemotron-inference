#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/scaled_fp8_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::GemmHeuristicCache;
using nemotron::ScaledFp8LinearConfig;
using nemotron::ScaledFp8LinearOp;

constexpr std::size_t kTokens = 40;
constexpr std::size_t kHiddenSize = 4096;
constexpr std::size_t kIntermediateSize = 8192;
constexpr std::size_t kNumHeads = 128;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kProjectedSize = 18560;
constexpr std::size_t kGroups = 8;
constexpr std::size_t kGroupSize = kIntermediateSize / kGroups;

std::filesystem::path ResolveFixtureRoot() {
  const char* override_env = std::getenv("NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT_OVERRIDE");
  if (override_env != nullptr && *override_env != '\0') {
    return std::filesystem::path(override_env);
  }
  return std::filesystem::path(NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT);
}

std::vector<float> ReadFp32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  if (nbytes % sizeof(float) != 0) {
    std::cerr << "FAIL: invalid fp32 payload size " << path << "\n";
    return {};
  }
  std::vector<float> data(nbytes / sizeof(float), 0.0f);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
  return data;
}

std::vector<std::uint8_t> ReadU8(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "FAIL: cannot read " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(nbytes, 0u);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
  return data;
}

std::optional<float> ReadJsonFloatField(
    const std::filesystem::path& path,
    const std::string& field_name) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }
  const std::string json{
      std::istreambuf_iterator<char>(in),
      std::istreambuf_iterator<char>()};
  const std::string needle = "\"" + field_name + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon_pos = json.find(':', key_pos + needle.size());
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }
  const char* begin = json.c_str() + colon_pos + 1;
  char* end = nullptr;
  const float value = std::strtof(begin, &end);
  if (begin == end) {
    return std::nullopt;
  }
  return value;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_d = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_d = std::max(max_d, std::fabs(a[i] - b[i]));
  }
  return max_d;
}

float RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::infinity();
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    diff_sq += diff * diff;
    ref_sq += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return ref_sq == 0.0 ? 0.0f : static_cast<float>(std::sqrt(diff_sq / ref_sq));
}

float SiLUHost(float value) {
  return value / (1.0f + std::exp(-value));
}

std::vector<__nv_bfloat16> ToBf16(const std::vector<float>& input) {
  std::vector<__nv_bfloat16> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = __float2bfloat16(input[i]);
  }
  return output;
}

std::vector<float> FromBf16(const std::vector<__nv_bfloat16>& input) {
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = __bfloat162float(input[i]);
  }
  return output;
}

std::uint8_t EncodeFp8(float value) {
  return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

float DecodeFp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> CpuReferenceOutProj(
    const std::vector<float>& activations,
    std::size_t rows,
    std::size_t input_cols,
    const std::vector<std::uint8_t>& weight_fp8,
    std::size_t output_rows,
    float weight_scale,
    float input_scale) {
  std::vector<float> output(rows * output_rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_rows; ++out) {
      float accum = 0.0f;
      for (std::size_t col = 0; col < input_cols; ++col) {
        const float quantized_input =
            DecodeFp8(EncodeFp8(activations[row * input_cols + col] / input_scale)) * input_scale;
        const float weight =
            DecodeFp8(weight_fp8[out * input_cols + col]) * weight_scale;
        accum += quantized_input * weight;
      }
      output[row * output_rows + out] = accum;
    }
  }
  return output;
}

bool CopyFp32TensorToHost(const DeviceTensorFp32& tensor, std::vector<float>* output) {
  if (!tensor.valid() || output == nullptr) {
    return false;
  }
  output->resize(tensor.numel(), 0.0f);
  return tensor.CopyToHost(output->data(), output->size());
}

}  // namespace

int main() {
  const char* dump_dir_env = std::getenv("NEMOTRON_VLLM_DUMP_DIR");
  const char* ref_env = std::getenv("NEMOTRON_MAMBA_OUT_PROJ_REFERENCE_FP32");
  if (dump_dir_env == nullptr || *dump_dir_env == '\0') {
    std::cerr << "SKIP: set NEMOTRON_VLLM_DUMP_DIR to a pinned vLLM chunked-scan dump\n";
    return 0;
  }
  if (ref_env == nullptr || *ref_env == '\0') {
    std::cerr << "SKIP: set NEMOTRON_MAMBA_OUT_PROJ_REFERENCE_FP32 to a vLLM out-proj reference\n";
    return 0;
  }

  const std::filesystem::path dump_dir(dump_dir_env);
  const std::filesystem::path fixture_root = ResolveFixtureRoot();
  const std::filesystem::path ref_path(ref_env);

  const auto y_output = ReadFp32(dump_dir / "y_output_fp32.bin");
  const auto projected = ReadFp32(dump_dir / "projected_states_fp32.bin");
  const auto mixer_norm_weight = ReadFp32(fixture_root / "mixer_norm_weight_fp32.bin");
  const auto out_proj_weight = ReadU8(fixture_root / "out_proj_weight_fp8.bin");
  const auto out_proj_weight_scale = ReadFp32(fixture_root / "out_proj_weight_scale_fp32.bin");
  const auto out_proj_input_scale = ReadFp32(fixture_root / "out_proj_input_scale_fp32.bin");
  const auto expected_out_proj = ReadFp32(ref_path);
  const auto mixer_rms_epsilon =
      ReadJsonFloatField(fixture_root / "metadata.json", "mixer_rms_epsilon");

  if (y_output.size() != (kTokens * kIntermediateSize) ||
      projected.size() != (kTokens * kProjectedSize) ||
      mixer_norm_weight.size() != kIntermediateSize ||
      out_proj_weight.size() != (kHiddenSize * kIntermediateSize) ||
      out_proj_weight_scale.size() != 1 ||
      out_proj_input_scale.size() != 1 ||
      expected_out_proj.size() != (kTokens * kHiddenSize) ||
      !mixer_rms_epsilon.has_value()) {
    std::cerr << "FAIL: unexpected fixture sizes or missing mixer_rms_epsilon\n";
    return 1;
  }

  std::vector<float> grouped(kTokens * kIntermediateSize, 0.0f);
  for (std::size_t row = 0; row < kTokens; ++row) {
    for (std::size_t group = 0; group < kGroups; ++group) {
      const std::size_t begin = group * kGroupSize;
      const std::size_t end = begin + kGroupSize;
      float sum_sq = 0.0f;
      for (std::size_t col = begin; col < end; ++col) {
        const std::size_t y_index = row * kIntermediateSize + col;
        const float gate = SiLUHost(projected[row * kProjectedSize + col]);
        const float gated = y_output[y_index] * gate;
        grouped[y_index] = gated;
        sum_sq += gated * gated;
      }
      const float inv_rms =
          1.0f / std::sqrt((sum_sq / static_cast<float>(kGroupSize)) + *mixer_rms_epsilon);
      for (std::size_t col = begin; col < end; ++col) {
        const std::size_t y_index = row * kIntermediateSize + col;
        grouped[y_index] *= inv_rms * mixer_norm_weight[col];
      }
    }
  }

  const auto grouped_bf16 = ToBf16(grouped);
  const auto grouped_bf16_fp32 = FromBf16(grouped_bf16);
  const auto cpu_reference = CpuReferenceOutProj(
      grouped_bf16_fp32,
      kTokens,
      kIntermediateSize,
      out_proj_weight,
      kHiddenSize,
      out_proj_weight_scale[0],
      out_proj_input_scale[0]);

  auto activations = DeviceTensorBf16::Create({kTokens, kIntermediateSize});
  auto output = DeviceTensorFp32::Create({kTokens, kHiddenSize});
  if (!activations || !activations->valid() ||
      !output || !output->valid() ||
      !activations->CopyFromHost(grouped_bf16.data(), grouped_bf16.size())) {
    std::cerr << "FAIL: device tensor setup failed\n";
    return 1;
  }

  ScaledFp8LinearConfig config;
  config.output_rows = kHiddenSize;
  config.input_cols = kIntermediateSize;
  config.packed_weight_data = out_proj_weight.data();
  config.packed_weight_nbytes = out_proj_weight.size();
  config.tensor_name = "backbone.layers.0.mixer.out_proj.weight";
  config.weight_scale = out_proj_weight_scale[0];
  config.input_scale = out_proj_input_scale[0];
  auto op = ScaledFp8LinearOp::Create(config);
  auto handle = CublasLtHandle::Create();
  if (!op || !op->valid() || !handle || !handle->valid()) {
    std::cerr << "FAIL: out-proj op setup failed\n";
    return 1;
  }

  GemmHeuristicCache heuristic_cache;
  if (!op->Run(*handle, &heuristic_cache, *activations, output.get())) {
    std::cerr << "FAIL: runtime out-proj execution failed\n";
    return 1;
  }

  std::vector<float> runtime_output;
  if (!CopyFp32TensorToHost(*output, &runtime_output)) {
    std::cerr << "FAIL: runtime output download failed\n";
    return 1;
  }

  const float cpu_vs_vllm_max = MaxAbsDiff(cpu_reference, expected_out_proj);
  const float cpu_vs_vllm_rel = RelL2(cpu_reference, expected_out_proj);
  const float runtime_vs_vllm_max = MaxAbsDiff(runtime_output, expected_out_proj);
  const float runtime_vs_vllm_rel = RelL2(runtime_output, expected_out_proj);
  const float runtime_vs_cpu_max = MaxAbsDiff(runtime_output, cpu_reference);
  const float runtime_vs_cpu_rel = RelL2(runtime_output, cpu_reference);

  std::cerr << "mamba_out_proj_vllm_probe"
            << ": cpu_vs_vllm mad=" << cpu_vs_vllm_max
            << " rel_l2=" << cpu_vs_vllm_rel
            << " runtime_vs_vllm mad=" << runtime_vs_vllm_max
            << " rel_l2=" << runtime_vs_vllm_rel
            << " runtime_vs_cpu mad=" << runtime_vs_cpu_max
            << " rel_l2=" << runtime_vs_cpu_rel
            << "\n";

  return 0;
}
