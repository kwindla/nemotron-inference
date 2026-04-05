#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_grouped.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceExpertRouting;
using nemotron::DeviceNvfp4Weight;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::FusedGroupedMoeConfig;
using nemotron::FusedGroupedMoeParams;
using nemotron::FusedMoePrefillParams;
using nemotron::FusedNvfp4WeightView;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::HostNvfp4Matrix;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::RunDeviceExpertRouting;
using nemotron::RunFusedGroupedMoe;
using nemotron::RunFusedMoePrefill;

template <typename T>
class DeviceBuffer {
 public:
  static std::unique_ptr<DeviceBuffer> Create(std::size_t count) {
    if (!HasCudaDevice() || count == 0) {
      return nullptr;
    }

    T* data = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)) !=
        cudaSuccess) {
      return nullptr;
    }
    return std::unique_ptr<DeviceBuffer>(new DeviceBuffer(data, count));
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&&) noexcept = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  T* data() const {
    return data_;
  }

  bool CopyFromHost(const std::vector<T>& values) const {
    return values.size() == count_ &&
           cudaMemcpy(
               data_,
               values.data(),
               count_ * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess;
  }

 private:
  DeviceBuffer(T* data, std::size_t count) : data_(data), count_(count) {}

  static bool HasCudaDevice() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

struct OwnedNvfp4Descriptor {
  HostNvfp4Matrix packed;
  GemmDescriptor descriptor;
};

void RebindOwnedNvfp4Descriptor(OwnedNvfp4Descriptor* owned) {
  if (owned == nullptr) {
    return;
  }
  owned->descriptor.packed_data = owned->packed.packed_data();
  owned->descriptor.packed_nbytes = owned->packed.packed_nbytes();
  owned->descriptor.block_scales_data = owned->packed.block_scales_data();
  owned->descriptor.block_scales_nbytes = owned->packed.block_scales_nbytes();
  owned->descriptor.tensor_scale_data = owned->packed.tensor_scale_data();
  owned->descriptor.tensor_scale_nbytes = owned->packed.tensor_scale_nbytes();
}

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

GemmDescriptor MakeNvfp4Descriptor(
    const std::string& name,
    const HostNvfp4Matrix& packed) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "fused_moe_grouped_test";
  descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = packed.rows;
  descriptor.input_cols = packed.cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed.packed_data();
  descriptor.packed_nbytes = packed.packed_nbytes();
  descriptor.block_scales_data = packed.block_scales_data();
  descriptor.block_scales_nbytes = packed.block_scales_nbytes();
  descriptor.tensor_scale_data = packed.tensor_scale_data();
  descriptor.tensor_scale_nbytes = packed.tensor_scale_nbytes();
  return descriptor;
}

std::optional<OwnedNvfp4Descriptor> MakeOwnedNvfp4Descriptor(
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
  owned.descriptor = MakeNvfp4Descriptor(name, owned.packed);
  return owned;
}

FusedNvfp4WeightView MakeFusedNvfp4WeightView(const DeviceNvfp4Weight& weight) {
  return FusedNvfp4WeightView{
      weight.packed_data(),
      weight.block_scales_data(),
      weight.matmul_block_scales_data(),
      reinterpret_cast<const float*>(weight.tensor_scale_data()),
      weight.output_rows(),
      weight.input_cols(),
  };
}

std::vector<__nv_bfloat16> ToBf16Vector(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = __float2bfloat16(values[i]);
  }
  return converted;
}

std::vector<float> MakeRandomVector(
    std::size_t count,
    std::mt19937* rng,
    float scale) {
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> values(count, 0.0f);
  for (float& value : values) {
    value = dist(*rng);
  }
  return values;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    diff = std::max(diff, std::fabs(lhs[i] - rhs[i]));
  }
  return diff;
}

bool TestGroupedMoeMatchesPrefillReference() {
  if (!HasCudaDevice()) {
    std::cout << "fused_moe_grouped_test: SKIP (no CUDA device)\n";
    return true;
  }

  auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "fused_moe_grouped_test: SKIP (cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kTokenCount = 32;
  constexpr std::size_t kHiddenSize = 128;
  constexpr std::size_t kRoutedIntermediateSize = 96;
  constexpr std::size_t kSharedIntermediateSize = 64;
  constexpr std::size_t kRoutedExperts = 8;
  constexpr std::size_t kTopK = 2;
  constexpr std::size_t kSelectionCount = kTokenCount * kTopK;
  constexpr float kTolerance = 1.0e-1f;

  std::mt19937 rng(7);
  const std::vector<float> normalized_host =
      MakeRandomVector(kTokenCount * kHiddenSize, &rng, 0.1f);
  const std::vector<__nv_bfloat16> normalized_bf16_host =
      ToBf16Vector(normalized_host);

  auto normalized_fp32 =
      DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto normalized_bf16 =
      DeviceTensorBf16::Create({kTokenCount, kHiddenSize});
  auto prefill_output =
      DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto prefill_routed_output =
      DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto grouped_output =
      DeviceTensorFp32::Create({kTokenCount, kHiddenSize});
  auto gather_scratch =
      DeviceTensorFp32::Create({kSelectionCount, kHiddenSize});
  auto expert_up_scratch =
      DeviceTensorFp32::Create({kSelectionCount, kRoutedIntermediateSize});
  auto shared_up_scratch =
      DeviceTensorFp32::Create({kTokenCount, kSharedIntermediateSize});
  if (!Expect(
          normalized_fp32 && normalized_bf16 && prefill_output &&
              prefill_routed_output && grouped_output && gather_scratch &&
              expert_up_scratch && shared_up_scratch,
          "device tensors should allocate") ||
      !Expect(
          normalized_fp32->CopyFromHost(
              normalized_host.data(),
              normalized_host.size()),
          "FP32 normalized input should upload") ||
      !Expect(
          normalized_bf16->CopyFromHost(
              normalized_bf16_host.data(),
              normalized_bf16_host.size()),
          "BF16 normalized input should upload")) {
    return false;
  }

  std::vector<int> selected_indices_host(kSelectionCount, 0);
  std::vector<float> selected_weights_host(kSelectionCount, 0.0f);
  std::vector<int> expert_ids(kRoutedExperts, 0);
  std::iota(expert_ids.begin(), expert_ids.end(), 0);
  std::uniform_real_distribution<float> weight_dist(0.1f, 1.0f);
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    std::shuffle(expert_ids.begin(), expert_ids.end(), rng);
    float weight_sum = 0.0f;
    for (std::size_t slot = 0; slot < kTopK; ++slot) {
      selected_indices_host[token * kTopK + slot] = expert_ids[slot];
      selected_weights_host[token * kTopK + slot] = weight_dist(rng);
      weight_sum += selected_weights_host[token * kTopK + slot];
    }
    for (std::size_t slot = 0; slot < kTopK; ++slot) {
      selected_weights_host[token * kTopK + slot] /= weight_sum;
    }
  }

  auto selected_indices_device = DeviceBuffer<int>::Create(kSelectionCount);
  auto selected_weights_device = DeviceBuffer<float>::Create(kSelectionCount);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, kSelectionCount);
  if (!Expect(
          selected_indices_device && selected_weights_device && routing &&
              routing->valid(),
          "routing buffers should allocate") ||
      !Expect(
          selected_indices_device->CopyFromHost(selected_indices_host),
          "selected indices should upload") ||
      !Expect(
          selected_weights_device->CopyFromHost(selected_weights_host),
          "selected weights should upload") ||
      !Expect(
          RunDeviceExpertRouting(
              selected_indices_device->data(),
              selected_weights_device->data(),
              kTokenCount,
              kTopK,
              routing.get()),
          "device expert routing should succeed")) {
    return false;
  }

  std::vector<OwnedNvfp4Descriptor> routed_up_owned;
  std::vector<OwnedNvfp4Descriptor> routed_down_owned;
  routed_up_owned.reserve(kRoutedExperts);
  routed_down_owned.reserve(kRoutedExperts);
  std::vector<std::unique_ptr<DeviceNvfp4Weight>> routed_up_device;
  std::vector<std::unique_ptr<DeviceNvfp4Weight>> routed_down_device;
  routed_up_device.reserve(kRoutedExperts);
  routed_down_device.reserve(kRoutedExperts);
  std::vector<FusedNvfp4WeightView> routed_up_views;
  std::vector<FusedNvfp4WeightView> routed_down_views;
  routed_up_views.reserve(kRoutedExperts);
  routed_down_views.reserve(kRoutedExperts);

  for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
    auto up_owned = MakeOwnedNvfp4Descriptor(
        "expert_up_" + std::to_string(expert),
        MakeRandomVector(kRoutedIntermediateSize * kHiddenSize, &rng, 0.05f),
        kRoutedIntermediateSize,
        kHiddenSize);
    auto down_owned = MakeOwnedNvfp4Descriptor(
        "expert_down_" + std::to_string(expert),
        MakeRandomVector(kHiddenSize * kRoutedIntermediateSize, &rng, 0.05f),
        kHiddenSize,
        kRoutedIntermediateSize);
    if (!Expect(up_owned.has_value(), "routed up descriptor should pack") ||
        !Expect(down_owned.has_value(), "routed down descriptor should pack")) {
      return false;
    }

    routed_up_owned.push_back(std::move(*up_owned));
    routed_down_owned.push_back(std::move(*down_owned));
    RebindOwnedNvfp4Descriptor(&routed_up_owned.back());
    RebindOwnedNvfp4Descriptor(&routed_down_owned.back());

    auto up_device = DeviceNvfp4Weight::Upload(routed_up_owned.back().descriptor);
    auto down_device = DeviceNvfp4Weight::Upload(routed_down_owned.back().descriptor);
    if (!Expect(
            up_device && up_device->valid(),
            "routed up weight should upload") ||
        !Expect(
            down_device && down_device->valid(),
            "routed down weight should upload")) {
      return false;
    }
    routed_up_views.push_back(MakeFusedNvfp4WeightView(*up_device));
    routed_down_views.push_back(MakeFusedNvfp4WeightView(*down_device));
    routed_up_device.push_back(std::move(up_device));
    routed_down_device.push_back(std::move(down_device));
  }

  const std::vector<float> shared_up_zero(
      kSharedIntermediateSize * kHiddenSize,
      0.0f);
  const std::vector<float> shared_down_zero(
      kHiddenSize * kSharedIntermediateSize,
      0.0f);
  auto shared_up_owned = MakeOwnedNvfp4Descriptor(
      "shared_up",
      shared_up_zero,
      kSharedIntermediateSize,
      kHiddenSize);
  auto shared_down_owned = MakeOwnedNvfp4Descriptor(
      "shared_down",
      shared_down_zero,
      kHiddenSize,
      kSharedIntermediateSize);
  if (!Expect(shared_up_owned.has_value(), "shared up descriptor should pack") ||
      !Expect(shared_down_owned.has_value(), "shared down descriptor should pack")) {
    return false;
  }
  RebindOwnedNvfp4Descriptor(&*shared_up_owned);
  RebindOwnedNvfp4Descriptor(&*shared_down_owned);

  auto shared_up_device = DeviceNvfp4Weight::Upload(shared_up_owned->descriptor);
  auto shared_down_device = DeviceNvfp4Weight::Upload(shared_down_owned->descriptor);
  if (!Expect(
          shared_up_device && shared_up_device->valid(),
          "shared up weight should upload") ||
      !Expect(
          shared_down_device && shared_down_device->valid(),
          "shared down weight should upload")) {
    return false;
  }

  auto routed_up_views_device =
      DeviceBuffer<FusedNvfp4WeightView>::Create(kRoutedExperts);
  auto routed_down_views_device =
      DeviceBuffer<FusedNvfp4WeightView>::Create(kRoutedExperts);
  if (!Expect(
          routed_up_views_device && routed_down_views_device,
          "device weight-view arrays should allocate") ||
      !Expect(
          routed_up_views_device->CopyFromHost(routed_up_views),
          "routed up views should upload") ||
      !Expect(
          routed_down_views_device->CopyFromHost(routed_down_views),
          "routed down views should upload")) {
    return false;
  }

  std::vector<const GemmDescriptor*> routed_up_descriptor_ptrs(kRoutedExperts, nullptr);
  std::vector<const GemmDescriptor*> routed_down_descriptor_ptrs(kRoutedExperts, nullptr);
  for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
    routed_up_descriptor_ptrs[expert] = &routed_up_owned[expert].descriptor;
    routed_down_descriptor_ptrs[expert] = &routed_down_owned[expert].descriptor;
  }

  GemmHeuristicCache heuristic_cache;
  FusedMoePrefillParams prefill_params;
  prefill_params.token_count = kTokenCount;
  prefill_params.selection_count = kSelectionCount;
  prefill_params.hidden_size = kHiddenSize;
  prefill_params.routed_expert_intermediate_size = kRoutedIntermediateSize;
  prefill_params.shared_expert_intermediate_size = kSharedIntermediateSize;
  prefill_params.n_routed_experts = kRoutedExperts;
  prefill_params.top_k = kTopK;
  prefill_params.cublas_handle = cublas.get();
  prefill_params.heuristic_cache = &heuristic_cache;
  prefill_params.shared_up_descriptor = &shared_up_owned->descriptor;
  prefill_params.shared_down_descriptor = &shared_down_owned->descriptor;
  prefill_params.routed_up_descriptors = routed_up_descriptor_ptrs.data();
  prefill_params.routed_down_descriptors = routed_down_descriptor_ptrs.data();
  prefill_params.shared_up = MakeFusedNvfp4WeightView(*shared_up_device);
  prefill_params.shared_down = MakeFusedNvfp4WeightView(*shared_down_device);
  prefill_params.routed_up = routed_up_views_device->data();
  prefill_params.routed_down = routed_down_views_device->data();
  prefill_params.input = normalized_fp32->data();
  prefill_params.normalized = normalized_fp32->data();
  prefill_params.output = prefill_output->data();
  prefill_params.routed_output = prefill_routed_output->data();
  prefill_params.gather_scratch = gather_scratch->data();
  prefill_params.expert_up_scratch = expert_up_scratch->data();
  prefill_params.shared_up_scratch = shared_up_scratch->data();
  prefill_params.expert_offsets = routing->expert_offsets();
  prefill_params.sorted_token_indices = routing->sorted_token_indices();
  prefill_params.sorted_token_weights = routing->sorted_token_weights();
  prefill_params.active_expert_count = routing->active_expert_count();
  prefill_params.active_expert_ids = routing->active_expert_ids();

  FusedGroupedMoeConfig grouped_config;
  grouped_config.hidden_size = kHiddenSize;
  grouped_config.routed_expert_intermediate_size = kRoutedIntermediateSize;
  grouped_config.n_routed_experts = kRoutedExperts;
  grouped_config.top_k = kTopK;
  grouped_config.tile_size = 1;

  FusedGroupedMoeParams grouped_params;
  grouped_params.token_count = kTokenCount;
  grouped_params.selection_count = kSelectionCount;
  grouped_params.normalized = normalized_bf16->data();
  grouped_params.expert_offsets = routing->expert_offsets();
  grouped_params.sorted_token_indices = routing->sorted_token_indices();
  grouped_params.sorted_token_weights = routing->sorted_token_weights();
  grouped_params.active_expert_count = routing->active_expert_count();
  grouped_params.active_expert_ids = routing->active_expert_ids();
  grouped_params.routed_up = routed_up_views_device->data();
  grouped_params.routed_down = routed_down_views_device->data();
  grouped_params.output = grouped_output->data();

  if (!Expect(
          RunFusedMoePrefill(prefill_params),
          "reference fused prefill should succeed") ||
      !Expect(
          cudaDeviceSynchronize() == cudaSuccess,
          "reference fused prefill should synchronize") ||
      !Expect(
          RunFusedGroupedMoe(grouped_params, grouped_config),
          "grouped fused MoE should succeed") ||
      !Expect(
          cudaDeviceSynchronize() == cudaSuccess,
          "grouped fused MoE should synchronize")) {
    return false;
  }

  std::vector<float> prefill_output_host(kTokenCount * kHiddenSize, 0.0f);
  std::vector<float> prefill_routed_output_host(kTokenCount * kHiddenSize, 0.0f);
  std::vector<float> grouped_output_host(kTokenCount * kHiddenSize, 0.0f);
  if (!Expect(
          prefill_output->CopyToHost(
              prefill_output_host.data(),
              prefill_output_host.size()),
          "prefill output should download") ||
      !Expect(
          prefill_routed_output->CopyToHost(
              prefill_routed_output_host.data(),
              prefill_routed_output_host.size()),
          "prefill routed output should download") ||
      !Expect(
          grouped_output->CopyToHost(
              grouped_output_host.data(),
              grouped_output_host.size()),
          "grouped output should download")) {
    return false;
  }

  const float shared_diff =
      MaxAbsDiff(prefill_output_host, prefill_routed_output_host);
  if (!Expect(
          shared_diff <= 1.0e-6f,
          "shared path should stay zero in the reference run")) {
    return false;
  }

  const float grouped_diff =
      MaxAbsDiff(prefill_routed_output_host, grouped_output_host);
  return Expect(
      grouped_diff <= kTolerance,
      "grouped routed output max abs diff should stay within tolerance");
}

}  // namespace

int main() {
  if (!TestGroupedMoeMatchesPrefillReference()) {
    return EXIT_FAILURE;
  }
  std::cout << "fused_moe_grouped_test: PASS\n";
  return EXIT_SUCCESS;
}
