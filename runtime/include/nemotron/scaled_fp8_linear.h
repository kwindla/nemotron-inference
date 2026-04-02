#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/dense_weight.h"
#include "nemotron/dense_gemm_runner.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"

namespace nemotron {

struct ScaledFp8LinearConfig {
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  const std::uint8_t* packed_weight_data = nullptr;
  std::size_t packed_weight_nbytes = 0;
  std::string tensor_name;
  float weight_scale = 0.0f;
  float input_scale = 0.0f;
  bool allow_dequantized_dense_fastpath = false;
};

std::optional<std::vector<float>> DequantizeScaledFp8WeightToHostFp32(
    const ScaledFp8LinearConfig& config);

bool QuantizeFp32ToScaledFp8RoundTrip(
    const DeviceTensorFp32& input,
    float input_scale,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

class ScaledFp8LinearOp {
 public:
  static std::unique_ptr<ScaledFp8LinearOp> Create(const ScaledFp8LinearConfig& config);
  static std::unique_ptr<ScaledFp8LinearOp> CreateView(
      const ScaledFp8LinearConfig& config,
      std::unique_ptr<DeviceDenseWeightFp32> weight_view);
  static std::unique_ptr<ScaledFp8LinearOp> CreateView(
      const ScaledFp8LinearConfig& config,
      std::unique_ptr<DeviceDenseWeightFp32> weight_view,
      std::unique_ptr<DeviceTensorFp8E4M3> packed_weight_view);

  ScaledFp8LinearOp(ScaledFp8LinearOp&&) noexcept;
  ScaledFp8LinearOp& operator=(ScaledFp8LinearOp&&) noexcept;
  ~ScaledFp8LinearOp();

  ScaledFp8LinearOp(const ScaledFp8LinearOp&) = delete;
  ScaledFp8LinearOp& operator=(const ScaledFp8LinearOp&) = delete;

  bool valid() const;
  std::size_t output_rows() const;
  std::size_t input_cols() const;
  float input_scale() const;
  float weight_scale() const;

  bool Run(
      CublasLtHandle& handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorFp32& activations,
      DeviceTensorFp32* output,
      cudaStream_t stream = nullptr) const;

  bool Run(
      CublasLtHandle& handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorBf16& activations,
      DeviceTensorFp32* output,
      cudaStream_t stream = nullptr) const;

  bool Run(
      CublasLtHandle& handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorBf16& activations,
      DeviceTensorBf16* output,
      cudaStream_t stream = nullptr) const;

 private:
  struct Impl;

  explicit ScaledFp8LinearOp(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
