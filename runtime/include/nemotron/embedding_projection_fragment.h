#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "nemotron/cublaslt_handle.h"
#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/dense_gemm_runner.h"
#include "nemotron/dense_weight.h"
#include "nemotron/device_tensor.h"
#include "nemotron/embedding_table.h"

namespace nemotron {

struct EmbeddingProjectionStats {
  std::size_t token_count = 0;
  std::size_t embedding_dim = 0;
  DenseRowMajorDeviceStats projection;
};

std::optional<EmbeddingProjectionStats> RunEmbeddingThenDenseProjectionFp32(
    const DeviceEmbeddingTableFp32& embedding_table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    const DeviceDenseWeightFp32& projection_weight,
    CublasLtHandle& handle,
    const CublasLtGemmPlan& projection_plan,
    DeviceTensorFp32* output);

std::optional<EmbeddingProjectionStats> RunEmbeddingThenDenseProjectionFp32(
    const DeviceEmbeddingTableFp32& embedding_table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    CublasLtHandle& handle,
    const CublasLtGemmPlan& projection_plan,
    DeviceTensorFp32* output);

}  // namespace nemotron
