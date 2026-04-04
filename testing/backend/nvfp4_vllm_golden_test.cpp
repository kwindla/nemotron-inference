#include "nemotron/gemm_catalog.h"
#include "nemotron/monolithic_expert_weights.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using nemotron::ExecutionNvfp4ScaleBytes;
using nemotron::GemmDescriptor;
using nemotron::GemmKernelFamily;
using nemotron::HostNvfp4Matrix;
using nemotron::MonolithicNvfp4ExpertWeights;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::PackedFp4Bytes;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

constexpr const char* kSourceRoot = NEMOTRON_SOURCE_ROOT;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_nvfp4_vllm_golden_test_" +
             std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

struct OwnedNvfp4Descriptor {
  HostNvfp4Matrix packed;
  GemmDescriptor descriptor;
};

struct MoeGoldenCase {
  std::string label;
  std::size_t num_experts = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
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

std::optional<std::filesystem::path> find_vllm_python() {
  if (const char* env_python = std::getenv("NEMOTRON_VLLM_PYTHON");
      env_python != nullptr && env_python[0] != '\0') {
    return std::filesystem::path(env_python);
  }

  const std::filesystem::path repo_python =
      std::filesystem::path(kSourceRoot) / "vllm-env-cu128" / "bin" / "python";
  if (std::filesystem::exists(repo_python)) {
    return repo_python;
  }
  return std::nullopt;
}

std::string shell_quote(const std::filesystem::path& path) {
  const std::string raw = path.string();
  std::string quoted = "'";
  for (char ch : raw) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

bool run_command(const std::string& command) {
  return std::system(command.c_str()) == 0;
}

void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
}

void write_floats(const std::filesystem::path& path, const std::vector<float>& values) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

std::vector<float> read_floats(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_file(path);
  if (bytes.size() % sizeof(float) != 0) {
    return {};
  }
  std::vector<float> values(bytes.size() / sizeof(float), 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

bool copy_device_bytes(
    const std::uint8_t* device_data,
    std::size_t nbytes,
    std::vector<std::uint8_t>* host_bytes) {
  if (device_data == nullptr || host_bytes == nullptr) {
    return false;
  }
  host_bytes->assign(nbytes, 0u);
  return cudaMemcpy(
             host_bytes->data(),
             device_data,
             nbytes,
             cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool copy_device_float(
    const float* device_data,
    float* host_value) {
  if (device_data == nullptr || host_value == nullptr) {
    return false;
  }
  return cudaMemcpy(
             host_value,
             device_data,
             sizeof(float),
             cudaMemcpyDeviceToHost) == cudaSuccess;
}

std::vector<float> make_patterned_values(
    std::size_t rows,
    std::size_t cols,
    int seed,
    float scale) {
  std::vector<float> values(rows * cols, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const int raw =
          static_cast<int>(((row + 1) * (seed + 5)) + ((col + 3) * (seed + 11)));
      values[row * cols + col] =
          (static_cast<float>((raw % 29) - 14) * scale) +
          (0.003125f * static_cast<float>((row + col + static_cast<std::size_t>(seed)) % 7));
    }
  }
  return values;
}

std::vector<std::uint8_t> make_scale_bytes(std::size_t rows, std::size_t cols, int seed) {
  const std::size_t count = rows * (cols / 16u);
  std::vector<std::uint8_t> bytes(count, 0u);
  for (std::size_t i = 0; i < count; ++i) {
    bytes[i] = static_cast<std::uint8_t>(0x30u + ((i + static_cast<std::size_t>(seed * 3)) % 32u));
  }
  return bytes;
}

std::optional<OwnedNvfp4Descriptor> make_owned_nvfp4_descriptor(
    const std::string& name,
    const std::vector<float>& values,
    std::size_t rows,
    std::size_t cols) {
  const auto packed = PackRowMajorFp32ToNvfp4(values.data(), rows, cols);
  if (!packed.has_value()) {
    return std::nullopt;
  }
  OwnedNvfp4Descriptor owned;
  owned.packed = *packed;
  owned.descriptor.tensor_name = name;
  owned.descriptor.op_class = "nvfp4_vllm_golden_test";
  owned.descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  owned.descriptor.output_rows = rows;
  owned.descriptor.input_cols = cols;
  owned.descriptor.storage_dtype = "nvfp4_e2m1";
  owned.descriptor.compute_dtype = "fp32_accum";
  owned.descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  owned.descriptor.alignment_bytes = 16;
  owned.descriptor.packed_data = owned.packed.packed_data();
  owned.descriptor.packed_nbytes = owned.packed.packed_nbytes();
  owned.descriptor.block_scales_data = owned.packed.block_scales_data();
  owned.descriptor.block_scales_nbytes = owned.packed.block_scales_nbytes();
  owned.descriptor.tensor_scale_data = owned.packed.tensor_scale_data();
  owned.descriptor.tensor_scale_nbytes = owned.packed.tensor_scale_nbytes();
  return owned;
}

bool run_vllm_swizzle_reference(
    const std::filesystem::path& python,
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    std::size_t rows,
    std::size_t cols) {
  const std::filesystem::path helper =
      std::filesystem::path(kSourceRoot) / "testing" / "backend" / "nvfp4_vllm_reference.py";
  std::ostringstream command;
  command << shell_quote(python)
          << " " << shell_quote(helper)
          << " swizzle"
          << " --rows " << rows
          << " --cols " << cols
          << " --input " << shell_quote(input_path)
          << " --output " << shell_quote(output_path);
  return run_command(command.str());
}

bool run_vllm_prepare_reference(
    const std::filesystem::path& python,
    const std::filesystem::path& workdir,
    const MoeGoldenCase& test_case) {
  const std::filesystem::path helper =
      std::filesystem::path(kSourceRoot) / "testing" / "backend" / "nvfp4_vllm_reference.py";
  std::ostringstream command;
  command << shell_quote(python)
          << " " << shell_quote(helper)
          << " prepare_moe"
          << " --num-experts " << test_case.num_experts
          << " --up-rows " << test_case.intermediate_size
          << " --up-cols " << test_case.hidden_size
          << " --down-rows " << test_case.hidden_size
          << " --down-cols " << test_case.intermediate_size
          << " --up-packed " << shell_quote(workdir / "up_packed.bin")
          << " --up-scales " << shell_quote(workdir / "up_scales.bin")
          << " --up-tensor-scales " << shell_quote(workdir / "up_tensor_scales.bin")
          << " --down-packed " << shell_quote(workdir / "down_packed.bin")
          << " --down-scales " << shell_quote(workdir / "down_scales.bin")
          << " --down-tensor-scales " << shell_quote(workdir / "down_tensor_scales.bin")
          << " --output-dir " << shell_quote(workdir / "reference");
  return run_command(command.str());
}

bool test_scale_swizzle_matches_vllm() {
  const auto python = find_vllm_python();
  if (!python.has_value()) {
    std::cout << "nvfp4_vllm_golden_test: SKIP (vLLM python not found)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "nvfp4_vllm_golden_test: SKIP (no CUDA device available)\n";
    return true;
  }

  struct ShapeCase {
    const char* label;
    std::size_t rows;
    std::size_t cols;
    int seed;
  };
  const std::vector<ShapeCase> cases = {
      {"tiny_exact", 32, 16, 1},
      {"row_padding", 129, 64, 3},
      {"column_padding", 96, 80, 7},
      {"nano_up", 1856, 2688, 11},
      {"nano_down", 2688, 1856, 13},
  };

  TempDir temp_dir;
  for (const ShapeCase& shape_case : cases) {
    const std::vector<std::uint8_t> row_major_scales =
        make_scale_bytes(shape_case.rows, shape_case.cols, shape_case.seed);
    const std::vector<std::uint8_t> local =
        SwizzleRowMajorNvfp4ScalesForExecution(
            row_major_scales.data(), shape_case.rows, shape_case.cols);
    if (!expect(
            !local.empty(),
            std::string("local scale swizzle should succeed for ") + shape_case.label)) {
      return false;
    }

    const std::filesystem::path input_path =
        temp_dir.path() / (std::string(shape_case.label) + "_row_major.bin");
    const std::filesystem::path output_path =
        temp_dir.path() / (std::string(shape_case.label) + "_reference.bin");
    write_file(input_path, row_major_scales);
    if (!expect(
            run_vllm_swizzle_reference(
                *python,
                input_path,
                output_path,
                shape_case.rows,
                shape_case.cols),
            std::string("vLLM scale swizzle should run for ") + shape_case.label)) {
      return false;
    }

    const std::vector<std::uint8_t> reference = read_file(output_path);
    if (!expect(
            reference == local,
            std::string("scale swizzle should match vLLM for ") + shape_case.label)) {
      return false;
    }
  }
  return true;
}

bool test_monolithic_prepared_weights_match_vllm_on_aligned_cases() {
  const auto python = find_vllm_python();
  if (!python.has_value()) {
    std::cout << "nvfp4_vllm_golden_test: SKIP (vLLM python not found)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "nvfp4_vllm_golden_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<MoeGoldenCase> cases = {
      {"aligned_128x64", 2, 64, 128},
      {"aligned_128x48", 2, 48, 128},
  };

  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
    const MoeGoldenCase& test_case = cases[case_index];
    TempDir temp_dir;
    std::vector<OwnedNvfp4Descriptor> up_descriptors;
    std::vector<OwnedNvfp4Descriptor> down_descriptors;
    up_descriptors.reserve(test_case.num_experts);
    down_descriptors.reserve(test_case.num_experts);

    std::vector<std::uint8_t> up_packed_concat;
    std::vector<std::uint8_t> up_scales_concat;
    std::vector<float> up_tensor_scales;
    std::vector<std::uint8_t> down_packed_concat;
    std::vector<std::uint8_t> down_scales_concat;
    std::vector<float> down_tensor_scales;

    for (std::size_t expert_index = 0; expert_index < test_case.num_experts; ++expert_index) {
      const auto up_values = make_patterned_values(
          test_case.intermediate_size,
          test_case.hidden_size,
          31 + static_cast<int>(case_index * 10 + expert_index * 2),
          0.015625f);
      const auto down_values = make_patterned_values(
          test_case.hidden_size,
          test_case.intermediate_size,
          47 + static_cast<int>(case_index * 10 + expert_index * 3),
          0.015625f);
      const auto up_descriptor = make_owned_nvfp4_descriptor(
          test_case.label + ".expert" + std::to_string(expert_index) + ".up",
          up_values,
          test_case.intermediate_size,
          test_case.hidden_size);
      const auto down_descriptor = make_owned_nvfp4_descriptor(
          test_case.label + ".expert" + std::to_string(expert_index) + ".down",
          down_values,
          test_case.hidden_size,
          test_case.intermediate_size);
      if (!expect(up_descriptor.has_value(), "up descriptor should pack") ||
          !expect(down_descriptor.has_value(), "down descriptor should pack")) {
        return false;
      }
      up_packed_concat.insert(
          up_packed_concat.end(),
          up_descriptor->packed.packed.begin(),
          up_descriptor->packed.packed.end());
      up_scales_concat.insert(
          up_scales_concat.end(),
          up_descriptor->packed.block_scales.begin(),
          up_descriptor->packed.block_scales.end());
      up_tensor_scales.push_back(up_descriptor->packed.tensor_scale);
      down_packed_concat.insert(
          down_packed_concat.end(),
          down_descriptor->packed.packed.begin(),
          down_descriptor->packed.packed.end());
      down_scales_concat.insert(
          down_scales_concat.end(),
          down_descriptor->packed.block_scales.begin(),
          down_descriptor->packed.block_scales.end());
      down_tensor_scales.push_back(down_descriptor->packed.tensor_scale);
      up_descriptors.push_back(std::move(*up_descriptor));
      down_descriptors.push_back(std::move(*down_descriptor));
    }

    write_file(temp_dir.path() / "up_packed.bin", up_packed_concat);
    write_file(temp_dir.path() / "up_scales.bin", up_scales_concat);
    write_floats(temp_dir.path() / "up_tensor_scales.bin", up_tensor_scales);
    write_file(temp_dir.path() / "down_packed.bin", down_packed_concat);
    write_file(temp_dir.path() / "down_scales.bin", down_scales_concat);
    write_floats(temp_dir.path() / "down_tensor_scales.bin", down_tensor_scales);

    if (!expect(
            run_vllm_prepare_reference(*python, temp_dir.path(), test_case),
            "vLLM prepared MoE reference should run")) {
      return false;
    }

    auto monolithic_up = MonolithicNvfp4ExpertWeights::Create(
        test_case.num_experts,
        test_case.intermediate_size,
        test_case.hidden_size);
    auto monolithic_down = MonolithicNvfp4ExpertWeights::Create(
        test_case.num_experts,
        test_case.hidden_size,
        test_case.intermediate_size);
    if (!expect(
            monolithic_up != nullptr && monolithic_up->valid() &&
                monolithic_down != nullptr && monolithic_down->valid(),
            "monolithic expert weights should allocate")) {
      return false;
    }

    for (std::size_t expert_index = 0; expert_index < test_case.num_experts; ++expert_index) {
      if (!expect(
              monolithic_up->UploadExpert(
                  expert_index,
                  up_descriptors[expert_index].packed.packed_data(),
                  up_descriptors[expert_index].packed.packed_nbytes(),
                  up_descriptors[expert_index].packed.block_scales_data(),
                  up_descriptors[expert_index].packed.block_scales_nbytes(),
                  reinterpret_cast<const float*>(
                      up_descriptors[expert_index].packed.tensor_scale_data())),
              "monolithic up expert upload should succeed") ||
          !expect(
              monolithic_down->UploadExpert(
                  expert_index,
                  down_descriptors[expert_index].packed.packed_data(),
                  down_descriptors[expert_index].packed.packed_nbytes(),
                  down_descriptors[expert_index].packed.block_scales_data(),
                  down_descriptors[expert_index].packed.block_scales_nbytes(),
                  reinterpret_cast<const float*>(
                      down_descriptors[expert_index].packed.tensor_scale_data())),
              "monolithic down expert upload should succeed")) {
        return false;
      }
    }

    std::vector<std::uint8_t> local_up_packed;
    std::vector<std::uint8_t> local_up_scales;
    std::vector<float> local_up_tensor_scales;
    std::vector<std::uint8_t> local_down_packed;
    std::vector<std::uint8_t> local_down_scales;
    std::vector<float> local_down_tensor_scales;

    for (std::size_t expert_index = 0; expert_index < test_case.num_experts; ++expert_index) {
      const auto up_view = monolithic_up->GetView(expert_index);
      const auto down_view = monolithic_down->GetView(expert_index);
      std::vector<std::uint8_t> bytes;
      if (!copy_device_bytes(
              up_view.packed_data,
              PackedFp4Bytes(up_view.output_rows, up_view.input_cols),
              &bytes)) {
        return false;
      }
      local_up_packed.insert(local_up_packed.end(), bytes.begin(), bytes.end());
      if (!copy_device_bytes(
              up_view.matmul_block_scales_data,
              ExecutionNvfp4ScaleBytes(up_view.output_rows, up_view.input_cols),
              &bytes)) {
        return false;
      }
      local_up_scales.insert(local_up_scales.end(), bytes.begin(), bytes.end());
      float tensor_scale = 0.0f;
      if (!copy_device_float(up_view.tensor_scale_data, &tensor_scale)) {
        return false;
      }
      local_up_tensor_scales.push_back(tensor_scale);

      if (!copy_device_bytes(
              down_view.packed_data,
              PackedFp4Bytes(down_view.output_rows, down_view.input_cols),
              &bytes)) {
        return false;
      }
      local_down_packed.insert(local_down_packed.end(), bytes.begin(), bytes.end());
      if (!copy_device_bytes(
              down_view.matmul_block_scales_data,
              ExecutionNvfp4ScaleBytes(down_view.output_rows, down_view.input_cols),
              &bytes)) {
        return false;
      }
      local_down_scales.insert(local_down_scales.end(), bytes.begin(), bytes.end());
      if (!copy_device_float(down_view.tensor_scale_data, &tensor_scale)) {
        return false;
      }
      local_down_tensor_scales.push_back(tensor_scale);
    }

    const auto ref_up_packed = read_file(temp_dir.path() / "reference" / "ref_w13_packed.bin");
    const auto ref_up_scales = read_file(temp_dir.path() / "reference" / "ref_w13_scales.bin");
    const auto ref_up_tensor_scales =
        read_floats(temp_dir.path() / "reference" / "ref_w13_tensor_scales.bin");
    const auto ref_down_packed = read_file(temp_dir.path() / "reference" / "ref_w2_packed.bin");
    const auto ref_down_scales = read_file(temp_dir.path() / "reference" / "ref_w2_scales.bin");
    const auto ref_down_tensor_scales =
        read_floats(temp_dir.path() / "reference" / "ref_w2_tensor_scales.bin");

    if (!expect(
            local_up_packed == ref_up_packed,
            test_case.label + ": prepared up packed bytes should match vLLM") ||
        !expect(
            local_up_scales == ref_up_scales,
            test_case.label + ": prepared up execution scales should match vLLM") ||
        !expect(
            local_down_packed == ref_down_packed,
            test_case.label + ": prepared down packed bytes should match vLLM") ||
        !expect(
            local_down_scales == ref_down_scales,
            test_case.label + ": prepared down execution scales should match vLLM")) {
      return false;
    }

    if (!expect(
            local_up_tensor_scales.size() == ref_up_tensor_scales.size(),
            test_case.label + ": prepared up tensor scale count should match vLLM") ||
        !expect(
            local_down_tensor_scales.size() == ref_down_tensor_scales.size(),
            test_case.label + ": prepared down tensor scale count should match vLLM")) {
      return false;
    }
    for (std::size_t i = 0; i < local_up_tensor_scales.size(); ++i) {
      if (!expect(
              std::fabs(local_up_tensor_scales[i] - ref_up_tensor_scales[i]) <= 1.0e-6f,
              test_case.label + ": prepared up tensor scales should match vLLM")) {
        return false;
      }
    }
    for (std::size_t i = 0; i < local_down_tensor_scales.size(); ++i) {
      if (!expect(
              std::fabs(local_down_tensor_scales[i] - ref_down_tensor_scales[i]) <= 1.0e-6f,
              test_case.label + ": prepared down tensor scales should match vLLM")) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace

int main() {
  const bool ok =
      test_scale_swizzle_matches_vllm() &&
      test_monolithic_prepared_weights_match_vllm_on_aligned_cases();
  if (!ok) {
    return 1;
  }
  std::cout << "nvfp4_vllm_golden_test: PASS\n";
  return 0;
}
