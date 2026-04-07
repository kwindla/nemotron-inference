#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/monolithic_expert_weights.h"
#include "nemotron/nvfp4_packing.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::DeviceTensorFp32;
using nemotron::DeviceTensorInt32;
using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceExpertRouting;
using nemotron::DeviceMoeLaunchPlan;
using nemotron::FusedMoePrefillParams;
using nemotron::FusedNvfp4WeightView;
using nemotron::HostNvfp4Matrix;
using nemotron::MonolithicNvfp4ExpertWeights;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::RunFusedMoePrefill;

constexpr float kMaxAbsDiffTolerance = 5.0e-4f;
constexpr float kNanoPackedContractTolerance = 2.0f;

bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

float DecodeFp4(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0F;
  return static_cast<float>(value);
}

float DecodeFp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> MakePatternedValues(
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
          (0.0015f * static_cast<float>((row + col + static_cast<std::size_t>(seed)) % 7));
    }
  }
  return values;
}

std::vector<float> DequantizeNvfp4Matrix(const HostNvfp4Matrix& packed) {
  std::vector<float> output(packed.rows * packed.cols, 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < packed.rows; ++row) {
    for (std::size_t block = 0; block < packed.cols / 16; ++block) {
      const float block_scale =
          DecodeFp8(packed.block_scales[scale_index++]) * packed.tensor_scale;
      const std::size_t col_start = block * 16;
      for (std::size_t offset = 0; offset < 16; offset += 2) {
        const std::uint8_t byte = packed.packed[packed_index++];
        output[row * packed.cols + col_start + offset] =
            DecodeFp4(byte & 0x0F) * block_scale;
        output[row * packed.cols + col_start + offset + 1] =
            DecodeFp4((byte >> 4) & 0x0F) * block_scale;
      }
    }
  }
  return output;
}

std::optional<std::vector<float>> QuantizeDequantizeRow(
    const std::vector<float>& row) {
  const auto packed = PackRowMajorFp32ToNvfp4(row.data(), 1, row.size());
  if (!packed.has_value()) {
    return std::nullopt;
  }
  return DequantizeNvfp4Matrix(*packed);
}

std::optional<std::vector<float>> QuantizeDequantizeMatrixRows(
    const std::vector<float>& values,
    std::size_t rows,
    std::size_t cols) {
  const auto packed = PackRowMajorFp32ToNvfp4(values.data(), rows, cols);
  if (!packed.has_value()) {
    return std::nullopt;
  }
  return DequantizeNvfp4Matrix(*packed);
}

template <typename T>
class DeviceArray {
 public:
  static std::unique_ptr<DeviceArray> CopyFromHost(const std::vector<T>& values) {
    if (values.empty()) {
      return nullptr;
    }
    T* data = nullptr;
    const std::size_t bytes = values.size() * sizeof(T);
    if (cudaMalloc(reinterpret_cast<void**>(&data), bytes) != cudaSuccess) {
      return nullptr;
    }
    if (cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(data);
      return nullptr;
    }
    return std::unique_ptr<DeviceArray>(new DeviceArray(data, values.size()));
  }

  ~DeviceArray() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  const T* data() const {
    return data_;
  }

 private:
  DeviceArray(T* data, std::size_t size) : data_(data), size_(size) {}

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

struct UploadedWeights {
  std::unique_ptr<MonolithicNvfp4ExpertWeights> storage;
  std::vector<FusedNvfp4WeightView> views;
  std::vector<std::vector<float>> dequantized;
};

struct RepeatedUploadedWeights {
  std::unique_ptr<MonolithicNvfp4ExpertWeights> storage;
  std::vector<FusedNvfp4WeightView> views;
  std::vector<float> dequantized;
};

std::optional<UploadedWeights> UploadWeights(
    const std::vector<std::vector<float>>& matrices,
    std::size_t rows,
    std::size_t cols) {
  auto storage = MonolithicNvfp4ExpertWeights::Create(matrices.size(), rows, cols);
  if (!storage || !storage->valid()) {
    return std::nullopt;
  }

  UploadedWeights uploaded;
  uploaded.storage = std::move(storage);
  uploaded.dequantized.reserve(matrices.size());
  for (std::size_t expert_index = 0; expert_index < matrices.size(); ++expert_index) {
    const auto packed =
        PackRowMajorFp32ToNvfp4(matrices[expert_index].data(), rows, cols);
    if (!packed.has_value() ||
        !uploaded.storage->UploadExpert(
            expert_index,
            packed->packed_data(),
            packed->packed_nbytes(),
            packed->block_scales_data(),
            packed->block_scales_nbytes(),
            reinterpret_cast<const float*>(packed->tensor_scale_data()))) {
      return std::nullopt;
    }
    uploaded.dequantized.push_back(DequantizeNvfp4Matrix(*packed));
  }
  uploaded.views = uploaded.storage->BuildAllViews();
  return uploaded;
}

std::optional<RepeatedUploadedWeights> UploadRepeatedWeights(
    const std::vector<float>& matrix,
    std::size_t repeat_count,
    std::size_t rows,
    std::size_t cols) {
  auto storage = MonolithicNvfp4ExpertWeights::Create(repeat_count, rows, cols);
  if (!storage || !storage->valid()) {
    return std::nullopt;
  }

  const auto packed = PackRowMajorFp32ToNvfp4(matrix.data(), rows, cols);
  if (!packed.has_value()) {
    return std::nullopt;
  }

  for (std::size_t expert_index = 0; expert_index < repeat_count; ++expert_index) {
    if (!storage->UploadExpert(
            expert_index,
            packed->packed_data(),
            packed->packed_nbytes(),
            packed->block_scales_data(),
            packed->block_scales_nbytes(),
            reinterpret_cast<const float*>(packed->tensor_scale_data()))) {
      return std::nullopt;
    }
  }

  RepeatedUploadedWeights uploaded;
  uploaded.storage = std::move(storage);
  uploaded.views = uploaded.storage->BuildAllViews();
  uploaded.dequantized = DequantizeNvfp4Matrix(*packed);
  return uploaded;
}

std::vector<float> RowMajorMatVec(
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t cols,
    const std::vector<float>& activations) {
  std::vector<float> output(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    float accum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
      accum += weights[row * cols + col] * activations[col];
    }
    output[row] = accum;
  }
  return output;
}

void Relu2InPlace(std::vector<float>* values) {
  if (values == nullptr) {
    return;
  }
  for (float& value : *values) {
    value = value > 0.0f ? value * value : 0.0f;
  }
}

struct PrefillReferenceCase {
  std::size_t token_count = 2;
  std::size_t hidden_size = 16;
  std::size_t routed_expert_intermediate_size = 16;
  std::size_t shared_expert_intermediate_size = 16;
  std::size_t n_routed_experts = 2;
  std::size_t top_k = 2;
  std::vector<float> input;
  std::vector<float> normalized;
  std::vector<int> topk_ids;
  std::vector<float> topk_weights;
  std::vector<std::vector<float>> routed_up;
  std::vector<std::vector<float>> routed_down;
  std::vector<float> shared_up;
  std::vector<float> shared_down;
};

// Synthetic contract case: intentionally tiny and ragged so the active kernel
// path is forced through tail handling, optional outputs, and small-shape
// contract validation. These tests stay small on purpose.
PrefillReferenceCase BuildReferenceCase() {
  PrefillReferenceCase test_case;
  test_case.input = MakePatternedValues(
      test_case.token_count,
      test_case.hidden_size,
      3,
      0.03125f);
  test_case.normalized = MakePatternedValues(
      test_case.token_count,
      test_case.hidden_size,
      11,
      0.0234375f);
  test_case.topk_ids = {
      0, 1,
      1, 0,
  };
  test_case.topk_weights = {
      0.65f, 0.35f,
      0.20f, 0.80f,
  };
  test_case.routed_up = {
      MakePatternedValues(
          test_case.routed_expert_intermediate_size,
          test_case.hidden_size,
          17,
          0.015625f),
      MakePatternedValues(
          test_case.routed_expert_intermediate_size,
          test_case.hidden_size,
          19,
          0.013671875f),
  };
  test_case.routed_down = {
      MakePatternedValues(
          test_case.hidden_size,
          test_case.routed_expert_intermediate_size,
          23,
          0.017578125f),
      MakePatternedValues(
          test_case.hidden_size,
          test_case.routed_expert_intermediate_size,
          29,
          0.0146484375f),
  };
  test_case.shared_up = MakePatternedValues(
      test_case.shared_expert_intermediate_size,
      test_case.hidden_size,
      31,
      0.0126953125f);
  test_case.shared_down = MakePatternedValues(
      test_case.hidden_size,
      test_case.shared_expert_intermediate_size,
      37,
      0.0166015625f);
  return test_case;
}

// Deployment-shape case: exact Nemotron Nano NVFP4 dimensions. All routed
// experts share one weight matrix so the reference remains tractable while the
// runtime still exercises the real expert count, top-k, and tensor shapes.
PrefillReferenceCase BuildNanoDeploymentCase() {
  PrefillReferenceCase test_case;
  test_case.token_count = 2;
  test_case.hidden_size = 2688;
  test_case.routed_expert_intermediate_size = 1856;
  test_case.shared_expert_intermediate_size = 3712;
  test_case.n_routed_experts = 128;
  test_case.top_k = 6;
  test_case.input = MakePatternedValues(
      test_case.token_count,
      test_case.hidden_size,
      41,
      0.015625f);
  test_case.normalized = MakePatternedValues(
      test_case.token_count,
      test_case.hidden_size,
      43,
      0.01171875f);
  test_case.topk_ids = {
      0, 7, 13, 31, 63, 127,
      5, 6, 62, 64, 65, 126,
  };
  test_case.topk_weights = {
      0.21f, 0.17f, 0.13f, 0.19f, 0.11f, 0.19f,
      0.18f, 0.15f, 0.14f, 0.16f, 0.17f, 0.20f,
  };
  test_case.routed_up = {
      MakePatternedValues(
          test_case.routed_expert_intermediate_size,
          test_case.hidden_size,
          47,
          0.0078125f),
  };
  test_case.routed_down = {
      MakePatternedValues(
          test_case.hidden_size,
          test_case.routed_expert_intermediate_size,
          53,
          0.0078125f),
  };
  test_case.shared_up = MakePatternedValues(
      test_case.shared_expert_intermediate_size,
      test_case.hidden_size,
      59,
      0.0068359375f);
  test_case.shared_down = MakePatternedValues(
      test_case.hidden_size,
      test_case.shared_expert_intermediate_size,
      61,
      0.0068359375f);
  return test_case;
}

bool BuildReferenceOutputs(
    const PrefillReferenceCase& test_case,
    const UploadedWeights& routed_up,
    const UploadedWeights& routed_down,
    const UploadedWeights& shared_up,
    const UploadedWeights& shared_down,
    std::vector<float>* expected_output,
    std::vector<float>* expected_routed_output,
    std::vector<float>* expected_shared_output) {
  if (expected_output == nullptr ||
      expected_routed_output == nullptr ||
      expected_shared_output == nullptr) {
    return false;
  }

  expected_output->assign(
      test_case.token_count * test_case.hidden_size,
      0.0f);
  expected_routed_output->assign(
      test_case.token_count * test_case.hidden_size,
      0.0f);
  expected_shared_output->assign(
      test_case.token_count * test_case.hidden_size,
      0.0f);

  const auto routed_quantized_inputs =
      QuantizeDequantizeMatrixRows(test_case.normalized, test_case.token_count, test_case.hidden_size);
  if (!routed_quantized_inputs.has_value()) {
    return false;
  }

  for (std::size_t token_index = 0; token_index < test_case.token_count; ++token_index) {
    const float* normalized_row =
        test_case.normalized.data() + token_index * test_case.hidden_size;
    std::vector<float> normalized_vec(
        normalized_row,
        normalized_row + test_case.hidden_size);
    const auto shared_quantized_input = QuantizeDequantizeRow(normalized_vec);
    if (!shared_quantized_input.has_value()) {
      return false;
    }
    std::vector<float> routed_quantized_input(
        routed_quantized_inputs->begin() + static_cast<std::ptrdiff_t>(token_index * test_case.hidden_size),
        routed_quantized_inputs->begin() +
            static_cast<std::ptrdiff_t>((token_index + 1) * test_case.hidden_size));

    float* output_row = expected_output->data() + token_index * test_case.hidden_size;
    float* routed_output_row =
        expected_routed_output->data() + token_index * test_case.hidden_size;
    float* shared_output_row =
        expected_shared_output->data() + token_index * test_case.hidden_size;

    for (std::size_t slot = 0; slot < test_case.top_k; ++slot) {
      const std::size_t selection_index = token_index * test_case.top_k + slot;
      const int expert_index = test_case.topk_ids[selection_index];
      if (expert_index < 0) {
        continue;
      }

      auto expert_up = RowMajorMatVec(
          routed_up.dequantized[static_cast<std::size_t>(expert_index)],
          test_case.routed_expert_intermediate_size,
          test_case.hidden_size,
          routed_quantized_input);
      Relu2InPlace(&expert_up);
      const auto quantized_expert_up = QuantizeDequantizeRow(expert_up);
      if (!quantized_expert_up.has_value()) {
        return false;
      }
      const auto expert_down = RowMajorMatVec(
          routed_down.dequantized[static_cast<std::size_t>(expert_index)],
          test_case.hidden_size,
          test_case.routed_expert_intermediate_size,
          *quantized_expert_up);
      for (std::size_t dim = 0; dim < test_case.hidden_size; ++dim) {
        const float weighted = test_case.topk_weights[selection_index] * expert_down[dim];
        output_row[dim] += weighted;
        routed_output_row[dim] += weighted;
      }
    }

    auto shared_up_output = RowMajorMatVec(
        shared_up.dequantized.front(),
        test_case.shared_expert_intermediate_size,
        test_case.hidden_size,
        *shared_quantized_input);
    Relu2InPlace(&shared_up_output);
    const auto quantized_shared_up = QuantizeDequantizeRow(shared_up_output);
    if (!quantized_shared_up.has_value()) {
      return false;
    }
    const auto shared_down_output = RowMajorMatVec(
        shared_down.dequantized.front(),
        test_case.hidden_size,
        test_case.shared_expert_intermediate_size,
        *quantized_shared_up);
    for (std::size_t dim = 0; dim < test_case.hidden_size; ++dim) {
      output_row[dim] += shared_down_output[dim];
      shared_output_row[dim] = shared_down_output[dim];
    }
  }
  return true;
}

bool BuildNanoReferenceOutputs(
    const PrefillReferenceCase& test_case,
    const RepeatedUploadedWeights& routed_up,
    const RepeatedUploadedWeights& routed_down,
    const UploadedWeights& shared_up,
    const UploadedWeights& shared_down,
    std::vector<float>* expected_output,
    std::vector<float>* expected_routed_output,
    std::vector<float>* expected_shared_output) {
  if (expected_output == nullptr ||
      expected_routed_output == nullptr ||
      expected_shared_output == nullptr) {
    return false;
  }

  expected_output->assign(
      test_case.token_count * test_case.hidden_size,
      0.0f);
  expected_routed_output->assign(
      test_case.token_count * test_case.hidden_size,
      0.0f);
  expected_shared_output->assign(
      test_case.token_count * test_case.hidden_size,
      0.0f);

  const auto routed_quantized_inputs =
      QuantizeDequantizeMatrixRows(test_case.normalized, test_case.token_count, test_case.hidden_size);
  if (!routed_quantized_inputs.has_value()) {
    return false;
  }

  for (std::size_t token_index = 0; token_index < test_case.token_count; ++token_index) {
    const float* normalized_row =
        test_case.normalized.data() + token_index * test_case.hidden_size;
    std::vector<float> normalized_vec(
        normalized_row,
        normalized_row + test_case.hidden_size);
    const auto shared_quantized_input = QuantizeDequantizeRow(normalized_vec);
    if (!shared_quantized_input.has_value()) {
      return false;
    }
    std::vector<float> routed_quantized_input(
        routed_quantized_inputs->begin() + static_cast<std::ptrdiff_t>(token_index * test_case.hidden_size),
        routed_quantized_inputs->begin() +
            static_cast<std::ptrdiff_t>((token_index + 1) * test_case.hidden_size));

    auto expert_up = RowMajorMatVec(
        routed_up.dequantized,
        test_case.routed_expert_intermediate_size,
        test_case.hidden_size,
        routed_quantized_input);
    Relu2InPlace(&expert_up);
    const auto quantized_expert_up = QuantizeDequantizeRow(expert_up);
    if (!quantized_expert_up.has_value()) {
      return false;
    }
    const auto expert_down = RowMajorMatVec(
        routed_down.dequantized,
        test_case.hidden_size,
        test_case.routed_expert_intermediate_size,
        *quantized_expert_up);

    float routed_scale = 0.0f;
    for (std::size_t slot = 0; slot < test_case.top_k; ++slot) {
      routed_scale += test_case.topk_weights[token_index * test_case.top_k + slot];
    }

    float* output_row = expected_output->data() + token_index * test_case.hidden_size;
    float* routed_output_row =
        expected_routed_output->data() + token_index * test_case.hidden_size;
    float* shared_output_row =
        expected_shared_output->data() + token_index * test_case.hidden_size;

    for (std::size_t dim = 0; dim < test_case.hidden_size; ++dim) {
      const float weighted = routed_scale * expert_down[dim];
      output_row[dim] += weighted;
      routed_output_row[dim] = weighted;
    }

    auto shared_up_output = RowMajorMatVec(
        shared_up.dequantized.front(),
        test_case.shared_expert_intermediate_size,
        test_case.hidden_size,
        *shared_quantized_input);
    Relu2InPlace(&shared_up_output);
    const auto quantized_shared_up = QuantizeDequantizeRow(shared_up_output);
    if (!quantized_shared_up.has_value()) {
      return false;
    }
    const auto shared_down_output = RowMajorMatVec(
        shared_down.dequantized.front(),
        test_case.hidden_size,
        test_case.shared_expert_intermediate_size,
        *quantized_shared_up);
    for (std::size_t dim = 0; dim < test_case.hidden_size; ++dim) {
      output_row[dim] += shared_down_output[dim];
      shared_output_row[dim] = shared_down_output[dim];
    }
  }

  return true;
}

bool TestFusedMoePrefillRejectsMissingSelectionContract() {
  if (!HasCudaDevice()) {
    std::cout << "fused_moe_prefill_test: SKIP (no CUDA device)\n";
    return true;
  }

  constexpr std::size_t kHiddenSize = 16;
  constexpr std::size_t kIntermediateSize = 16;

  auto routed_up =
      UploadWeights({MakePatternedValues(kIntermediateSize, kHiddenSize, 5, 0.02f)},
                    kIntermediateSize,
                    kHiddenSize);
  auto routed_down =
      UploadWeights({MakePatternedValues(kHiddenSize, kIntermediateSize, 7, 0.02f)},
                    kHiddenSize,
                    kIntermediateSize);
  auto shared_up =
      UploadWeights({MakePatternedValues(kIntermediateSize, kHiddenSize, 11, 0.015f)},
                    kIntermediateSize,
                    kHiddenSize);
  auto shared_down =
      UploadWeights({MakePatternedValues(kHiddenSize, kIntermediateSize, 13, 0.015f)},
                    kHiddenSize,
                    kIntermediateSize);
  auto input = DeviceTensorFp32::Create({1, kHiddenSize});
  auto normalized = DeviceTensorFp32::Create({1, kHiddenSize});
  auto output = DeviceTensorFp32::Create({1, kHiddenSize});
  auto topk_ids = DeviceTensorInt32::Create({1, 1});
  auto topk_weights = DeviceTensorFp32::Create({1, 1});
  auto routing = DeviceExpertRouting::Create(1, 1);
  auto routed_gather_scratch = DeviceTensorFp32::Create({1, kHiddenSize});
  auto routed_up_scratch = DeviceTensorFp32::Create({1, kIntermediateSize});
  auto shared_up_scratch = DeviceTensorFp32::Create({1, kIntermediateSize});
  const std::vector<float> input_host = MakePatternedValues(1, kHiddenSize, 17, 0.03125f);
  const std::vector<float> normalized_host = MakePatternedValues(1, kHiddenSize, 19, 0.0234375f);
  const std::vector<int> topk_ids_host = {0};
  const std::vector<float> topk_weights_host = {1.0f};
  if (!Expect(
          routed_up.has_value() &&
              routed_down.has_value() &&
              shared_up.has_value() &&
              shared_down.has_value(),
          "test weights should upload") ||
      !Expect(
          input != nullptr &&
              normalized != nullptr &&
              output != nullptr &&
              topk_ids != nullptr &&
              topk_weights != nullptr &&
              routing != nullptr &&
              routing->valid() &&
              routed_gather_scratch != nullptr &&
              routed_up_scratch != nullptr &&
              shared_up_scratch != nullptr,
          "test tensors should allocate") ||
      !Expect(input->CopyFromHost(input_host.data(), input_host.size()),
              "input should upload") ||
      !Expect(normalized->CopyFromHost(normalized_host.data(), normalized_host.size()),
              "normalized should upload") ||
      !Expect(topk_ids->CopyFromHost(topk_ids_host.data(), topk_ids_host.size()),
              "topk ids should upload") ||
      !Expect(topk_weights->CopyFromHost(topk_weights_host.data(), topk_weights_host.size()),
              "topk weights should upload")) {
    return false;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  auto routed_down_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_down->views);
  if (!Expect(
          routed_up_views_device != nullptr &&
              routed_down_views_device != nullptr,
          "device weight-view tables should upload")) {
    return false;
  }

  FusedMoePrefillParams params;
  params.token_count = 1;
  params.hidden_size = kHiddenSize;
  params.routed_expert_intermediate_size = kIntermediateSize;
  params.shared_expert_intermediate_size = kIntermediateSize;
  params.n_routed_experts = 1;
  params.top_k = 1;
  params.shared_up = shared_up->views.front();
  params.shared_down = shared_down->views.front();
  params.routed_up_device = routed_up_views_device->data();
  params.routed_down_device = routed_down_views_device->data();
  params.selected_weights = topk_weights->data();
  params.input = input->data();
  params.normalized = normalized->data();
  params.routing = routing.get();
  params.routed_gather_scratch = routed_gather_scratch->data();
  params.routed_up_scratch = routed_up_scratch->data();
  params.shared_up_scratch = shared_up_scratch->data();
  params.output = output->data();
  if (!Expect(
          !RunFusedMoePrefill(params),
          "prefill should reject a missing selected_indices contract")) {
    return false;
  }

  params.selected_indices = topk_ids->data();
  params.selected_weights = nullptr;
  return Expect(
      !RunFusedMoePrefill(params),
      "prefill should reject a missing selected_weights contract");
}

bool TestFusedMoePrefillMatchesReferenceAndOptionalOutputs() {
  if (!HasCudaDevice()) {
    std::cout << "fused_moe_prefill_test: SKIP (no CUDA device)\n";
    return true;
  }

  const PrefillReferenceCase test_case = BuildReferenceCase();
  auto routed_up = UploadWeights(
      test_case.routed_up,
      test_case.routed_expert_intermediate_size,
      test_case.hidden_size);
  auto routed_down = UploadWeights(
      test_case.routed_down,
      test_case.hidden_size,
      test_case.routed_expert_intermediate_size);
  auto shared_up = UploadWeights(
      {test_case.shared_up},
      test_case.shared_expert_intermediate_size,
      test_case.hidden_size);
  auto shared_down = UploadWeights(
      {test_case.shared_down},
      test_case.hidden_size,
      test_case.shared_expert_intermediate_size);
  if (!Expect(
          routed_up.has_value() &&
              routed_down.has_value() &&
              shared_up.has_value() &&
              shared_down.has_value(),
          "reference weights should upload")) {
    return false;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  auto routed_down_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_down->views);
  if (!Expect(
          routed_up_views_device != nullptr &&
              routed_down_views_device != nullptr,
          "device weight-view tables should upload")) {
    return false;
  }

  auto input =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto normalized =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto output =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto routed_output =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto shared_output =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto topk_ids =
      DeviceTensorInt32::Create({test_case.token_count, test_case.top_k});
  auto topk_weights =
      DeviceTensorFp32::Create({test_case.token_count, test_case.top_k});
  auto routing =
      DeviceExpertRouting::Create(
          test_case.n_routed_experts,
          test_case.token_count * test_case.top_k);
  auto launch_plan =
      DeviceMoeLaunchPlan::Create(
          test_case.n_routed_experts,
          test_case.token_count * test_case.top_k);
  const auto padded_selection_count = DeviceMoeLaunchPlan::PaddedRowCapacity(
      test_case.n_routed_experts,
      test_case.token_count * test_case.top_k);
  auto routed_gather_scratch =
      DeviceTensorFp32::Create(
          {padded_selection_count.value_or(0), test_case.hidden_size});
  auto fc1_expert_activation_scales =
      DeviceTensorFp32::Create({test_case.n_routed_experts, 1});
  auto fc1_grouped_pack = DeviceNvfp4Matrix::Create(
      padded_selection_count.value_or(0),
      test_case.hidden_size,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  auto routed_up_scratch = DeviceTensorFp32::Create(
      {padded_selection_count.value_or(0),
       test_case.routed_expert_intermediate_size});
  auto fc2_expert_activation_scales =
      DeviceTensorFp32::Create({test_case.n_routed_experts, 1});
  auto fc2_grouped_pack = DeviceNvfp4Matrix::Create(
      padded_selection_count.value_or(0),
      test_case.routed_expert_intermediate_size,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  auto shared_up_scratch = DeviceTensorFp32::Create(
      {test_case.token_count, test_case.shared_expert_intermediate_size});
  if (!Expect(
          input != nullptr &&
              normalized != nullptr &&
              output != nullptr &&
              routed_output != nullptr &&
              shared_output != nullptr &&
              topk_ids != nullptr &&
              topk_weights != nullptr &&
              routing != nullptr &&
              routing->valid() &&
              launch_plan != nullptr &&
              launch_plan->valid() &&
              padded_selection_count.has_value() &&
              routed_gather_scratch != nullptr &&
              fc1_expert_activation_scales != nullptr &&
              fc1_grouped_pack != nullptr &&
              routed_up_scratch != nullptr &&
              fc2_expert_activation_scales != nullptr &&
              fc2_grouped_pack != nullptr &&
              shared_up_scratch != nullptr,
          "prefill tensors should allocate") ||
      !Expect(input->CopyFromHost(test_case.input.data(), test_case.input.size()),
              "input should upload") ||
      !Expect(
          normalized->CopyFromHost(
              test_case.normalized.data(),
              test_case.normalized.size()),
          "normalized should upload") ||
      !Expect(topk_ids->CopyFromHost(test_case.topk_ids.data(), test_case.topk_ids.size()),
              "topk ids should upload") ||
      !Expect(
          topk_weights->CopyFromHost(
              test_case.topk_weights.data(),
              test_case.topk_weights.size()),
          "topk weights should upload")) {
    return false;
  }

  FusedMoePrefillParams params;
  params.token_count = test_case.token_count;
  params.hidden_size = test_case.hidden_size;
  params.routed_expert_intermediate_size =
      test_case.routed_expert_intermediate_size;
  params.shared_expert_intermediate_size =
      test_case.shared_expert_intermediate_size;
  params.n_routed_experts = test_case.n_routed_experts;
  params.top_k = test_case.top_k;
  params.shared_up = shared_up->views.front();
  params.shared_down = shared_down->views.front();
  params.routed_up_device = routed_up_views_device->data();
  params.routed_down_device = routed_down_views_device->data();
  params.selected_indices = topk_ids->data();
  params.selected_weights = topk_weights->data();
  params.input = input->data();
  params.normalized = normalized->data();
  params.routing = routing.get();
  params.launch_plan = launch_plan.get();
  params.routed_gather_scratch = routed_gather_scratch->data();
  params.fc1_expert_activation_scales = fc1_expert_activation_scales->data();
  params.fc1_grouped_pack = fc1_grouped_pack.get();
  params.routed_up_scratch = routed_up_scratch->data();
  params.fc2_expert_activation_scales = fc2_expert_activation_scales->data();
  params.fc2_grouped_pack = fc2_grouped_pack.get();
  params.shared_up_scratch = shared_up_scratch->data();
  params.output = output->data();
  params.routed_output = routed_output->data();
  params.shared_output = shared_output->data();
  if (!Expect(RunFusedMoePrefill(params), "prefill kernel launch should succeed") ||
      !Expect(cudaDeviceSynchronize() == cudaSuccess, "prefill kernel should synchronize")) {
    return false;
  }

  std::vector<float> actual_output(output->numel(), 0.0f);
  std::vector<float> actual_routed_output(routed_output->numel(), 0.0f);
  std::vector<float> actual_shared_output(shared_output->numel(), 0.0f);
  if (!Expect(output->CopyToHost(actual_output.data(), actual_output.size()),
              "output should copy to host") ||
      !Expect(
          routed_output->CopyToHost(
              actual_routed_output.data(),
              actual_routed_output.size()),
          "routed output should copy to host") ||
      !Expect(
          shared_output->CopyToHost(
              actual_shared_output.data(),
              actual_shared_output.size()),
          "shared output should copy to host")) {
    return false;
  }

  std::vector<float> expected_output;
  std::vector<float> expected_routed_output;
  std::vector<float> expected_shared_output;
  if (!Expect(
          BuildReferenceOutputs(
              test_case,
              *routed_up,
              *routed_down,
              *shared_up,
              *shared_down,
              &expected_output,
              &expected_routed_output,
              &expected_shared_output),
          "CPU reference should build")) {
    return false;
  }

  const float output_diff = MaxAbsDiff(actual_output, expected_output);
  const float routed_diff =
      MaxAbsDiff(actual_routed_output, expected_routed_output);
  const float shared_diff =
      MaxAbsDiff(actual_shared_output, expected_shared_output);
  if (!Expect(
          output_diff <= kMaxAbsDiffTolerance,
          "prefill output should match reference") ||
      !Expect(
          routed_diff <= kMaxAbsDiffTolerance,
          "routed output should match reference") ||
      !Expect(
          shared_diff <= kMaxAbsDiffTolerance,
          "shared output should match reference")) {
    std::cerr << "prefill_output_max_abs_diff=" << output_diff << "\n";
    std::cerr << "prefill_routed_max_abs_diff=" << routed_diff << "\n";
    std::cerr << "prefill_shared_max_abs_diff=" << shared_diff << "\n";
    return false;
  }

  std::vector<float> recomposed_output(actual_routed_output.size(), 0.0f);
  for (std::size_t index = 0; index < recomposed_output.size(); ++index) {
    recomposed_output[index] =
        actual_routed_output[index] + actual_shared_output[index];
  }
  return Expect(
      MaxAbsDiff(actual_output, recomposed_output) <= kMaxAbsDiffTolerance,
      "prefill output should equal routed + shared contributions");
}

bool TestFusedMoePrefillNanoDeploymentShapeMatchesReference() {
  if (!HasCudaDevice()) {
    std::cout << "fused_moe_prefill_test: SKIP (no CUDA device)\n";
    return true;
  }

  const PrefillReferenceCase test_case = BuildNanoDeploymentCase();
  auto routed_up = UploadRepeatedWeights(
      test_case.routed_up.front(),
      test_case.n_routed_experts,
      test_case.routed_expert_intermediate_size,
      test_case.hidden_size);
  auto routed_down = UploadRepeatedWeights(
      test_case.routed_down.front(),
      test_case.n_routed_experts,
      test_case.hidden_size,
      test_case.routed_expert_intermediate_size);
  auto shared_up = UploadWeights(
      {test_case.shared_up},
      test_case.shared_expert_intermediate_size,
      test_case.hidden_size);
  auto shared_down = UploadWeights(
      {test_case.shared_down},
      test_case.hidden_size,
      test_case.shared_expert_intermediate_size);
  if (!Expect(
          routed_up.has_value() &&
              routed_down.has_value() &&
              shared_up.has_value() &&
              shared_down.has_value(),
          "Nano deployment-shape weights should upload")) {
    return false;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  auto routed_down_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_down->views);
  if (!Expect(
          routed_up_views_device != nullptr &&
              routed_down_views_device != nullptr,
          "Nano deployment-shape weight-view tables should upload")) {
    return false;
  }

  auto input =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto normalized =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto output =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto routed_output =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto shared_output =
      DeviceTensorFp32::Create({test_case.token_count, test_case.hidden_size});
  auto topk_ids =
      DeviceTensorInt32::Create({test_case.token_count, test_case.top_k});
  auto topk_weights =
      DeviceTensorFp32::Create({test_case.token_count, test_case.top_k});
  auto routing =
      DeviceExpertRouting::Create(
          test_case.n_routed_experts,
          test_case.token_count * test_case.top_k);
  auto launch_plan =
      DeviceMoeLaunchPlan::Create(
          test_case.n_routed_experts,
          test_case.token_count * test_case.top_k);
  const auto padded_selection_count = DeviceMoeLaunchPlan::PaddedRowCapacity(
      test_case.n_routed_experts,
      test_case.token_count * test_case.top_k);
  auto routed_gather_scratch =
      DeviceTensorFp32::Create(
          {padded_selection_count.value_or(0), test_case.hidden_size});
  auto fc1_expert_activation_scales =
      DeviceTensorFp32::Create({test_case.n_routed_experts, 1});
  auto fc1_grouped_pack = DeviceNvfp4Matrix::Create(
      padded_selection_count.value_or(0),
      test_case.hidden_size,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  auto routed_up_scratch = DeviceTensorFp32::Create(
      {padded_selection_count.value_or(0),
       test_case.routed_expert_intermediate_size});
  auto fc2_expert_activation_scales =
      DeviceTensorFp32::Create({test_case.n_routed_experts, 1});
  auto fc2_grouped_pack = DeviceNvfp4Matrix::Create(
      padded_selection_count.value_or(0),
      test_case.routed_expert_intermediate_size,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  auto shared_up_scratch = DeviceTensorFp32::Create(
      {test_case.token_count, test_case.shared_expert_intermediate_size});
  if (!Expect(
          input != nullptr &&
              normalized != nullptr &&
              output != nullptr &&
              routed_output != nullptr &&
              shared_output != nullptr &&
              topk_ids != nullptr &&
              topk_weights != nullptr &&
              routing != nullptr &&
              routing->valid() &&
              launch_plan != nullptr &&
              launch_plan->valid() &&
              padded_selection_count.has_value() &&
              routed_gather_scratch != nullptr &&
              fc1_expert_activation_scales != nullptr &&
              fc1_grouped_pack != nullptr &&
              routed_up_scratch != nullptr &&
              fc2_expert_activation_scales != nullptr &&
              fc2_grouped_pack != nullptr &&
              shared_up_scratch != nullptr,
          "Nano deployment-shape tensors should allocate") ||
      !Expect(input->CopyFromHost(test_case.input.data(), test_case.input.size()),
              "Nano deployment-shape input should upload") ||
      !Expect(
          normalized->CopyFromHost(
              test_case.normalized.data(),
              test_case.normalized.size()),
          "Nano deployment-shape normalized should upload") ||
      !Expect(
          topk_ids->CopyFromHost(test_case.topk_ids.data(), test_case.topk_ids.size()),
          "Nano deployment-shape topk ids should upload") ||
      !Expect(
          topk_weights->CopyFromHost(
              test_case.topk_weights.data(),
              test_case.topk_weights.size()),
          "Nano deployment-shape topk weights should upload")) {
    return false;
  }

  FusedMoePrefillParams params;
  params.token_count = test_case.token_count;
  params.hidden_size = test_case.hidden_size;
  params.routed_expert_intermediate_size =
      test_case.routed_expert_intermediate_size;
  params.shared_expert_intermediate_size =
      test_case.shared_expert_intermediate_size;
  params.n_routed_experts = test_case.n_routed_experts;
  params.top_k = test_case.top_k;
  params.shared_up = shared_up->views.front();
  params.shared_down = shared_down->views.front();
  params.routed_up_device = routed_up_views_device->data();
  params.routed_down_device = routed_down_views_device->data();
  params.selected_indices = topk_ids->data();
  params.selected_weights = topk_weights->data();
  params.input = input->data();
  params.normalized = normalized->data();
  params.routing = routing.get();
  params.launch_plan = launch_plan.get();
  params.routed_gather_scratch = routed_gather_scratch->data();
  params.fc1_expert_activation_scales = fc1_expert_activation_scales->data();
  params.fc1_grouped_pack = fc1_grouped_pack.get();
  params.routed_up_scratch = routed_up_scratch->data();
  params.fc2_expert_activation_scales = fc2_expert_activation_scales->data();
  params.fc2_grouped_pack = fc2_grouped_pack.get();
  params.shared_up_scratch = shared_up_scratch->data();
  params.output = output->data();
  params.routed_output = routed_output->data();
  params.shared_output = shared_output->data();
  if (!Expect(RunFusedMoePrefill(params), "Nano deployment-shape prefill should launch") ||
      !Expect(cudaDeviceSynchronize() == cudaSuccess,
              "Nano deployment-shape prefill should synchronize")) {
    return false;
  }

  std::vector<float> actual_output(output->numel(), 0.0f);
  std::vector<float> actual_routed_output(routed_output->numel(), 0.0f);
  std::vector<float> actual_shared_output(shared_output->numel(), 0.0f);
  if (!Expect(output->CopyToHost(actual_output.data(), actual_output.size()),
              "Nano deployment-shape output should copy to host") ||
      !Expect(
          routed_output->CopyToHost(
              actual_routed_output.data(),
              actual_routed_output.size()),
          "Nano deployment-shape routed output should copy to host") ||
      !Expect(
          shared_output->CopyToHost(
              actual_shared_output.data(),
              actual_shared_output.size()),
          "Nano deployment-shape shared output should copy to host")) {
    return false;
  }

  std::vector<float> expected_output;
  std::vector<float> expected_routed_output;
  std::vector<float> expected_shared_output;
  if (!Expect(
          BuildNanoReferenceOutputs(
              test_case,
              *routed_up,
              *routed_down,
              *shared_up,
              *shared_down,
              &expected_output,
              &expected_routed_output,
              &expected_shared_output),
          "Nano deployment-shape CPU reference should build")) {
    return false;
  }

  const float output_diff = MaxAbsDiff(actual_output, expected_output);
  const float routed_diff =
      MaxAbsDiff(actual_routed_output, expected_routed_output);
  const float shared_diff =
      MaxAbsDiff(actual_shared_output, expected_shared_output);
  if (!Expect(
          output_diff <= kNanoPackedContractTolerance,
          "Nano deployment-shape output should stay within the packed-contract budget") ||
      !Expect(
          routed_diff <= kNanoPackedContractTolerance,
          "Nano deployment-shape routed output should stay within the packed-contract budget") ||
      !Expect(
          shared_diff <= kMaxAbsDiffTolerance,
          "Nano deployment-shape shared output should match reference")) {
    std::cerr << "nano_prefill_output_max_abs_diff=" << output_diff << "\n";
    std::cerr << "nano_prefill_routed_max_abs_diff=" << routed_diff << "\n";
    std::cerr << "nano_prefill_shared_max_abs_diff=" << shared_diff << "\n";
    return false;
  }

  std::vector<float> recomposed_output(actual_output.size(), 0.0f);
  for (std::size_t index = 0; index < recomposed_output.size(); ++index) {
    recomposed_output[index] =
        actual_routed_output[index] + actual_shared_output[index];
  }
  return Expect(
      MaxAbsDiff(actual_output, recomposed_output) <= kMaxAbsDiffTolerance,
      "Nano deployment-shape output should equal routed + shared contributions");
}

}  // namespace

int main() {
  const bool ok =
      TestFusedMoePrefillRejectsMissingSelectionContract() &&
      TestFusedMoePrefillMatchesReferenceAndOptionalOutputs() &&
      TestFusedMoePrefillNanoDeploymentShapeMatchesReference();
  if (!ok) {
    return 1;
  }
  std::cout << "fused_moe_prefill_test: PASS\n";
  return 0;
}
