#include "nemotron/device_tensor.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/monolithic_expert_weights.h"
#include "nemotron/nvfp4_packing.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nemotron::DeviceTensorFp32;
using nemotron::FusedNvfp4WeightView;
using nemotron::MonolithicNvfp4ExpertWeights;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::RunGroupedNvfp4ExpertMatVec;

constexpr std::size_t kHiddenSize = 2688;
constexpr std::size_t kRoutedIntermediateSize = 1856;
constexpr std::size_t kRoutedExperts = 128;
constexpr std::size_t kTopK = 6;
constexpr double kNvfp4BytesPerWeight = 0.5625;

struct BenchmarkCase {
  std::string name;
  std::size_t prefix_tokens = 0;
};

struct BenchmarkOptions {
  std::vector<std::string> selected_case_names;
  std::size_t warmup_iterations = 1;
  std::size_t hot_iterations = 5;
};

struct BenchmarkResult {
  std::string case_name;
  std::size_t prefix_tokens = 0;
  std::size_t selection_count = 0;
  std::size_t active_experts = 0;
  double cold_ms = 0.0;
  double hot_mean_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  double hot_tflops = 0.0;
  double hot_weight_gib_per_s = 0.0;
};

struct UploadedWeights {
  std::unique_ptr<MonolithicNvfp4ExpertWeights> storage;
  std::vector<FusedNvfp4WeightView> views;
};

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

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
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
          static_cast<int>(((row + 1) * (seed + 7)) + ((col + 5) * (seed + 13)));
      values[row * cols + col] =
          static_cast<float>((raw % 31) - 15) * scale +
          0.001f * static_cast<float>((row + col + static_cast<std::size_t>(seed)) % 5);
    }
  }
  return values;
}

const std::vector<BenchmarkCase>& DefaultCases() {
  static const std::vector<BenchmarkCase> kCases = {
      {"prefix4", 4},
      {"prefix128", 128},
      {"prefix4096", 4096},
  };
  return kCases;
}

bool ShouldRunCase(
    const BenchmarkOptions& options,
    const std::string& case_name) {
  return options.selected_case_names.empty() ||
         std::find(
             options.selected_case_names.begin(),
             options.selected_case_names.end(),
             case_name) != options.selected_case_names.end();
}

std::vector<int> BuildExpertOffsets(std::size_t selection_count) {
  std::vector<int> offsets(kRoutedExperts + 1, 0);
  const std::size_t base = selection_count / kRoutedExperts;
  const std::size_t remainder = selection_count % kRoutedExperts;
  int running = 0;
  for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
    offsets[expert] = running;
    running += static_cast<int>(base + (expert < remainder ? 1 : 0));
  }
  offsets[kRoutedExperts] = running;
  return offsets;
}

std::size_t CountActiveExperts(const std::vector<int>& offsets) {
  std::size_t active = 0;
  for (std::size_t expert = 0; expert + 1 < offsets.size(); ++expert) {
    if (offsets[expert + 1] > offsets[expert]) {
      ++active;
    }
  }
  return active;
}

std::optional<UploadedWeights> BuildUploadedRoutedUpViews() {
  auto storage =
      MonolithicNvfp4ExpertWeights::Create(kRoutedExperts, kRoutedIntermediateSize, kHiddenSize);
  if (!storage || !storage->valid()) {
    return std::nullopt;
  }

  const auto source =
      MakePatternedValues(kRoutedIntermediateSize, kHiddenSize, 17, 0.015625f);
  const auto packed =
      PackRowMajorFp32ToNvfp4(source.data(), kRoutedIntermediateSize, kHiddenSize);
  if (!packed.has_value()) {
    return std::nullopt;
  }

  for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
    if (!storage->UploadExpert(
            expert,
            packed->packed_data(),
            packed->packed_nbytes(),
            packed->block_scales_data(),
            packed->block_scales_nbytes(),
            reinterpret_cast<const float*>(packed->tensor_scale_data()))) {
      return std::nullopt;
    }
  }

  UploadedWeights uploaded;
  uploaded.views = storage->BuildAllViews();
  uploaded.storage = std::move(storage);
  return uploaded;
}

std::optional<BenchmarkResult> RunCase(const BenchmarkCase& benchmark_case, const BenchmarkOptions& options) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  auto expert_offsets_device = DeviceArray<int>::CopyFromHost(expert_offsets);
  if (expert_offsets_device == nullptr) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  auto input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || output == nullptr) {
    return std::nullopt;
  }

  const auto input_host = MakePatternedValues(selection_count, kHiddenSize, 23, 0.03125f);
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  auto run_once = [&]() {
    return RunGroupedNvfp4ExpertMatVec(
        input->data(),
        expert_offsets_device->data(),
        kRoutedExperts,
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  return result;
}

BenchmarkOptions ParseOptions(int argc, char** argv) {
  BenchmarkOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--warmup" && i + 1 < argc) {
      options.warmup_iterations = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.hot_iterations = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--case" && i + 1 < argc) {
      options.selected_case_names.emplace_back(argv[++i]);
    }
  }
  return options;
}

void PrintResult(const BenchmarkResult& result) {
  std::cout << "Case: " << result.case_name
            << " prefix_tokens=" << result.prefix_tokens
            << " selection_count=" << result.selection_count
            << " active_experts=" << result.active_experts << "\n";
  std::cout << "  cold_ms=" << std::fixed << std::setprecision(3) << result.cold_ms
            << " hot_mean_ms=" << result.hot_mean_ms
            << " hot_min_ms=" << result.hot_min_ms
            << " hot_max_ms=" << result.hot_max_ms << "\n";
  std::cout << "  hot_tflops=" << std::setprecision(3) << result.hot_tflops
            << " hot_weight_gib_per_s=" << result.hot_weight_gib_per_s << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const BenchmarkOptions options = ParseOptions(argc, argv);
  if (!HasCudaDevice()) {
    std::cout << "nano_routed_up_bench: SKIP (no CUDA device)\n";
    return 0;
  }

  std::cout << "nano_routed_up_bench: hidden=" << kHiddenSize
            << " routed_intermediate=" << kRoutedIntermediateSize
            << " routed_experts=" << kRoutedExperts
            << " top_k=" << kTopK
            << " warmup_iterations=" << options.warmup_iterations
            << " hot_iterations=" << options.hot_iterations << "\n";

  bool ran_any = false;
  for (const auto& benchmark_case : DefaultCases()) {
    if (!ShouldRunCase(options, benchmark_case.name)) {
      continue;
    }
    ran_any = true;
    const auto result = RunCase(benchmark_case, options);
    if (!result.has_value()) {
      std::cerr << "nano_routed_up_bench: case failed: " << benchmark_case.name << "\n";
      return 1;
    }
    PrintResult(*result);
  }

  if (!ran_any) {
    std::cerr << "nano_routed_up_bench: no cases selected\n";
    return 1;
  }

  return 0;
}
