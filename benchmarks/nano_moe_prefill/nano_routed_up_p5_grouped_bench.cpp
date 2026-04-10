#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/monolithic_expert_weights.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/routed_expert_runtime.h"
#include "nemotron/moe_launch_plan_device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <vector>

namespace {

using nemotron::BuildDeviceMoeLaunchPlan;
using nemotron::BuildDeviceMoeLaunchPlanWithTokenTile;
using nemotron::DefaultRoutedExpertIntermediateSizePadded;
using nemotron::DeviceExpertRouting;
using nemotron::DeviceMoeLaunchPlan;
using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::DeviceTensorInt32;
using nemotron::FusedNvfp4WeightView;
using nemotron::MonolithicNvfp4ExpertWeights;
using nemotron::Nvfp4PackOptions;
using nemotron::Nvfp4ScaleLayout;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::RunDeviceExpertRouting;
using nemotron::RunLaunchPlannedPackedNvfp4ExpertMatVecBf16;

constexpr std::size_t kHiddenSize = 2688;
constexpr std::size_t kRoutedIntermediateLogical = 1856;
constexpr std::size_t kRoutedIntermediateExecution =
    DefaultRoutedExpertIntermediateSizePadded(kRoutedIntermediateLogical);
constexpr std::size_t kRoutedExperts = 128;
constexpr std::size_t kTopK = 6;
constexpr double kNvfp4BytesPerWeight = 0.5625;

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
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
    return std::unique_ptr<DeviceArray>(new DeviceArray(data));
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
  explicit DeviceArray(T* data) : data_(data) {}

  T* data_ = nullptr;
};

class ScopedCudaEventTimer {
 public:
  ScopedCudaEventTimer() {
    valid_ =
        cudaEventCreate(&start_) == cudaSuccess &&
        cudaEventCreate(&stop_) == cudaSuccess;
  }

  ~ScopedCudaEventTimer() {
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
  }

  bool valid() const {
    return valid_;
  }

  template <typename Callable>
  std::optional<double> Measure(Callable&& callable) {
    if (!valid_) {
      return std::nullopt;
    }
    if (cudaEventRecord(start_) != cudaSuccess) {
      return std::nullopt;
    }
    if (!callable()) {
      return std::nullopt;
    }
    if (cudaEventRecord(stop_) != cudaSuccess ||
        cudaEventSynchronize(stop_) != cudaSuccess) {
      return std::nullopt;
    }
    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) != cudaSuccess) {
      return std::nullopt;
    }
    return static_cast<double>(elapsed_ms);
  }

 private:
  bool valid_ = false;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

struct BenchmarkOptions {
  std::size_t prefix_tokens = 128;
  std::size_t warmup_iterations = 1;
  std::size_t hot_iterations = 5;
  bool profile_debug = false;
  std::size_t force_token_tile = 0;
};

struct Bf16SampleStats {
  float sum = std::numeric_limits<float>::quiet_NaN();
  std::size_t finite_count = 0;
  std::size_t nonfinite_count = 0;
  float min = std::numeric_limits<float>::quiet_NaN();
  float max = std::numeric_limits<float>::quiet_NaN();
};

struct UploadedWeights {
  std::unique_ptr<MonolithicNvfp4ExpertWeights> storage;
  std::vector<FusedNvfp4WeightView> views;
};

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

std::vector<int> BuildSelectedIndices(std::size_t token_count) {
  std::vector<int> selected_indices(token_count * kTopK, 0);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t slot = 0; slot < kTopK; ++slot) {
      const std::size_t linear = token * kTopK + slot;
      selected_indices[linear] = static_cast<int>(linear % kRoutedExperts);
    }
  }
  return selected_indices;
}

std::size_t CountActiveExperts(const std::vector<int>& selected_indices) {
  std::vector<bool> seen(kRoutedExperts, false);
  for (int expert : selected_indices) {
    if (expert >= 0 && static_cast<std::size_t>(expert) < kRoutedExperts) {
      seen[static_cast<std::size_t>(expert)] = true;
    }
  }
  return static_cast<std::size_t>(
      std::count(seen.begin(), seen.end(), true));
}

std::optional<UploadedWeights> UploadRepeatedRoutedUpWeights() {
  auto storage = MonolithicNvfp4ExpertWeights::Create(
      kRoutedExperts,
      kRoutedIntermediateExecution,
      kHiddenSize,
      false);
  if (!storage || !storage->valid()) {
    return std::nullopt;
  }

  const auto host_matrix = MakePatternedValues(
      kRoutedIntermediateLogical,
      kHiddenSize,
      47,
      0.0078125f);
  const auto packed =
      PackRowMajorFp32ToNvfp4(host_matrix.data(), kRoutedIntermediateLogical, kHiddenSize);
  if (!packed.has_value()) {
    return std::nullopt;
  }

  for (std::size_t expert_index = 0; expert_index < kRoutedExperts; ++expert_index) {
    if (!storage->UploadExpert(
            expert_index,
            packed->packed_data(),
            packed->packed_nbytes(),
            packed->block_scales_data(),
            packed->block_scales_nbytes(),
            reinterpret_cast<const float*>(packed->tensor_scale_data()),
            kRoutedIntermediateLogical,
            kHiddenSize)) {
      return std::nullopt;
    }
  }

  UploadedWeights uploaded;
  uploaded.storage = std::move(storage);
  uploaded.views = uploaded.storage->BuildAllViews();
  if (uploaded.views.size() != kRoutedExperts) {
    return std::nullopt;
  }
  for (const auto& view : uploaded.views) {
    if (view.packed_data == nullptr ||
        view.matmul_block_scales_data == nullptr ||
        view.tensor_scale_data == nullptr ||
        view.output_rows != kRoutedIntermediateExecution ||
        view.input_cols != kHiddenSize ||
        view.p5_tma_load_a == nullptr ||
        view.p5_tma_load_sfa == nullptr) {
      return std::nullopt;
    }
  }
  return uploaded;
}

BenchmarkOptions ParseOptions(int argc, char** argv) {
  BenchmarkOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--prefix-tokens" && i + 1 < argc) {
      options.prefix_tokens =
          static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup_iterations =
          static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.hot_iterations =
          static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--profile-debug") {
      options.profile_debug = true;
    } else if (arg == "--force-token-tile" && i + 1 < argc) {
      options.force_token_tile =
          static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    }
  }
  return options;
}

Bf16SampleStats InspectBf16Samples(const DeviceTensorBf16& tensor, std::size_t count) {
  Bf16SampleStats stats;
  const std::size_t sample_count = std::min<std::size_t>(count, tensor.numel());
  std::vector<__nv_bfloat16> host(tensor.numel());
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return stats;
  }
  float sum = 0.0f;
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < sample_count; ++index) {
    const __nv_bfloat16 value = host[index];
    const float converted = static_cast<float>(value);
    if (std::isfinite(converted)) {
      ++stats.finite_count;
      sum += converted;
      min_value = std::min(min_value, converted);
      max_value = std::max(max_value, converted);
    } else {
      ++stats.nonfinite_count;
    }
  }
  if (stats.finite_count != 0) {
    stats.sum = sum;
    stats.min = min_value;
    stats.max = max_value;
  }
  return stats;
}

bool LaunchPlanSupportsP5SfbTma(const DeviceMoeLaunchPlan& launch_plan) {
  constexpr int kScaleTileRows = 128;
  if (launch_plan.exact_cta_count_host() <= 0 ||
      launch_plan.cta_row_starts_host() == nullptr ||
      launch_plan.cta_valid_rows_host() == nullptr) {
    return false;
  }
  for (int cta_index = 0; cta_index < launch_plan.exact_cta_count_host(); ++cta_index) {
    const int row_start = launch_plan.cta_row_starts_host()[cta_index];
    const int valid_rows = launch_plan.cta_valid_rows_host()[cta_index];
    const int row_offset = row_start & (kScaleTileRows - 1);
    if (row_start < 0 || valid_rows <= 0 || row_offset + valid_rows > kScaleTileRows) {
      return false;
    }
  }
  return true;
}

struct P5SfbPathStats {
  int global_fragment_ctas = 0;
};

P5SfbPathStats SummarizeP5SfbPaths(const DeviceMoeLaunchPlan& launch_plan) {
  P5SfbPathStats stats;
  stats.global_fragment_ctas = std::max(0, launch_plan.exact_cta_count_host());
  return stats;
}

std::string_view DetermineP5SfbLiveMode(
    const DeviceNvfp4Matrix& grouped_pack,
    bool sfb_tma_launch_compatible) {
  if (grouped_pack.matmul_block_scales_data() != nullptr &&
      grouped_pack.scale_layout() == Nvfp4ScaleLayout::kSwizzled128x4) {
    return "direct_gmem_sparse";
  }
  if (sfb_tma_launch_compatible) {
    return "tma";
  }
  return "thread_smem";
}

}  // namespace

int main(int argc, char** argv) {
  const BenchmarkOptions options = ParseOptions(argc, argv);
  if (options.profile_debug) {
    setenv("NEMOTRON_ROUTED_PROFILE_DEBUG", "1", 1);
  }

  if (!HasCudaDevice()) {
    std::cout << "nano_routed_up_p5_grouped_bench: SKIP (no CUDA device)\n";
    return 0;
  }
  if (options.prefix_tokens == 0) {
    std::cerr << "nano_routed_up_p5_grouped_bench: prefix_tokens must be > 0\n";
    return 1;
  }

  const std::size_t selection_count = options.prefix_tokens * kTopK;
  auto selected_indices_host = BuildSelectedIndices(options.prefix_tokens);
  std::vector<float> selected_weights_host(selection_count, 1.0f);
  const std::size_t active_experts = CountActiveExperts(selected_indices_host);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan =
      options.force_token_tile == 0
          ? DeviceMoeLaunchPlan::Create(
                kRoutedExperts,
                selection_count,
                std::max(kHiddenSize, kRoutedIntermediateExecution))
          : DeviceMoeLaunchPlan::Create(
                kRoutedExperts,
                selection_count,
                std::max(kHiddenSize, kRoutedIntermediateExecution),
                options.force_token_tile,
                options.force_token_tile);
  auto weights = UploadRepeatedRoutedUpWeights();
  auto weight_views_device =
      weights.has_value() ? DeviceArray<FusedNvfp4WeightView>::CopyFromHost(weights->views) : nullptr;
  auto normalized = DeviceTensorFp32::Create({options.prefix_tokens, kHiddenSize});
  auto normalized_pack = DeviceNvfp4Matrix::Create(
      options.prefix_tokens,
      kHiddenSize,
      Nvfp4ScaleLayout::kSwizzled128x4);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !weights.has_value() ||
      weight_views_device == nullptr ||
      normalized == nullptr ||
      normalized_pack == nullptr) {
    std::cerr << "nano_routed_up_p5_grouped_bench: setup allocation failed\n";
    return 1;
  }

  const std::size_t padded_selection_count = launch_plan->padded_row_capacity();
  auto grouped_pack = DeviceNvfp4Matrix::Create(
      padded_selection_count,
      kHiddenSize,
      Nvfp4ScaleLayout::kSwizzled128x4);
  auto output = DeviceTensorBf16::Create(
      {padded_selection_count, kRoutedIntermediateExecution});
  if (grouped_pack == nullptr || output == nullptr) {
    std::cerr << "nano_routed_up_p5_grouped_bench: setup allocation failed\n";
    return 1;
  }

  const auto normalized_host = MakePatternedValues(
      options.prefix_tokens,
      kHiddenSize,
      43,
      0.01171875f);
  Nvfp4PackOptions pack_options;
  pack_options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  if (!normalized->CopyFromHost(normalized_host.data(), normalized_host.size()) ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          options.prefix_tokens,
          kTopK,
          routing.get()) ||
      !(options.force_token_tile == 0
            ? BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get())
            : BuildDeviceMoeLaunchPlanWithTokenTile(
                  *routing,
                  selection_count,
                  options.force_token_tile,
                  options.force_token_tile,
                  launch_plan.get())) ||
      !normalized_pack->PackInto(*normalized, pack_options) ||
      !GatherDeviceNvfp4Rows(
          *normalized_pack,
          launch_plan->permuted_idx_to_token_idx(),
          padded_selection_count,
          grouped_pack.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "nano_routed_up_p5_grouped_bench: setup kernels failed\n";
    return 1;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    std::cerr << "nano_routed_up_p5_grouped_bench: event timer init failed\n";
    return 1;
  }

  auto run_once = [&]() {
    return RunLaunchPlannedPackedNvfp4ExpertMatVecBf16(
               *grouped_pack,
               launch_plan.get(),
               options.prefix_tokens,
               selection_count,
               weight_views_device->data(),
               kRoutedIntermediateExecution,
               output->data());
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    if (!output->FillZero()) {
      std::cerr << "nano_routed_up_p5_grouped_bench: warmup zero failed\n";
      return 1;
    }
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      std::cerr << "nano_routed_up_p5_grouped_bench: warmup failed\n";
      return 1;
    }
  }

  if (!output->FillZero()) {
    std::cerr << "nano_routed_up_p5_grouped_bench: cold zero failed\n";
    return 1;
  }
  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    std::cerr << "nano_routed_up_p5_grouped_bench: cold run failed\n";
    return 1;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  std::vector<double> hot_samples_ms;
  hot_samples_ms.reserve(options.hot_iterations);
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    if (!output->FillZero()) {
      std::cerr << "nano_routed_up_p5_grouped_bench: hot zero failed\n";
      return 1;
    }
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      std::cerr << "nano_routed_up_p5_grouped_bench: hot run failed\n";
      return 1;
    }
    hot_samples_ms.push_back(*elapsed);
    hot_sum_ms += *elapsed;
    if (iteration == 0 || *elapsed < hot_min_ms) {
      hot_min_ms = *elapsed;
    }
    if (iteration == 0 || *elapsed > hot_max_ms) {
      hot_max_ms = *elapsed;
    }
  }
  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  std::sort(hot_samples_ms.begin(), hot_samples_ms.end());
  const std::size_t median_index = hot_samples_ms.empty() ? 0 : hot_samples_ms.size() / 2;
  const std::size_t p90_index =
      hot_samples_ms.empty() ? 0 : ((hot_samples_ms.size() - 1) * 9) / 10;
  const double hot_median_ms =
      hot_samples_ms.empty() ? 0.0 : hot_samples_ms[median_index];
  const double hot_p90_ms =
      hot_samples_ms.empty() ? 0.0 : hot_samples_ms[p90_index];

  const double flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateExecution);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateExecution) *
      kNvfp4BytesPerWeight;
  const Bf16SampleStats output_samples =
      InspectBf16Samples(*output, std::min<std::size_t>(256, output->numel()));

  const void* b_tma_descriptors = grouped_pack->p5_tma_load_b_descriptors(*launch_plan);
  const void* sfb_tma_descriptors = grouped_pack->p5_tma_load_sfb_descriptors(*launch_plan);
  const bool sfb_tma_launch_compatible = LaunchPlanSupportsP5SfbTma(*launch_plan);
  const P5SfbPathStats sfb_path_stats = SummarizeP5SfbPaths(*launch_plan);
  const std::string_view sfb_live_mode =
      DetermineP5SfbLiveMode(*grouped_pack, sfb_tma_launch_compatible);

            std::cout << "nano_routed_up_p5_grouped_bench: prefix_tokens=" << options.prefix_tokens
            << " top_k=" << kTopK
            << " selection_count=" << selection_count
            << " padded_selection_count=" << padded_selection_count
            << " active_experts=" << active_experts
            << " routed_intermediate_logical=" << kRoutedIntermediateLogical
            << " routed_intermediate_execution=" << kRoutedIntermediateExecution << "\n";
  std::cout << "  exact_cta_count=" << launch_plan->exact_cta_count_host()
            << " selected_token_tile=" << launch_plan->selected_token_tile()
            << " b_tma_cached=" << (b_tma_descriptors != nullptr ? "yes" : "no")
            << " sfb_tma_cached=" << (sfb_tma_descriptors != nullptr ? "yes" : "no")
            << " sfb_tma_launch_compatible="
            << (sfb_tma_launch_compatible ? "yes" : "no")
            << " sfb_live_mode=" << sfb_live_mode
            << " sfb_global_fragment_ctas=" << sfb_path_stats.global_fragment_ctas
            << "\n";
  if (launch_plan->exact_cta_count_host() > 0 &&
      launch_plan->cta_row_starts_host() != nullptr &&
      launch_plan->cta_valid_rows_host() != nullptr) {
    std::cout << "  first_cta_row_start=" << launch_plan->cta_row_starts_host()[0]
              << " first_cta_valid_rows=" << launch_plan->cta_valid_rows_host()[0] << "\n";
  }
  std::cout << "  cold_ms=" << std::fixed << std::setprecision(3) << *cold_ms
            << " hot_mean_ms=" << hot_mean_ms
            << " hot_median_ms=" << hot_median_ms
            << " hot_p90_ms=" << hot_p90_ms
            << " hot_min_ms=" << hot_min_ms
            << " hot_max_ms=" << hot_max_ms << "\n";
  std::cout << "  hot_tflops=" << std::setprecision(3)
            << (hot_mean_ms > 0.0 ? flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0)
            << " hot_weight_gib_per_s="
            << (hot_mean_ms > 0.0
                    ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0)
                    : 0.0)
            << " output_sample_sum=" << output_samples.sum
            << " output_sample_finite=" << output_samples.finite_count
            << " output_sample_nonfinite=" << output_samples.nonfinite_count
            << " output_sample_min=" << output_samples.min
            << " output_sample_max=" << output_samples.max << "\n";
  return 0;
}
