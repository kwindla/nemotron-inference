#include "nemotron/embedding_projection_fragment.h"

namespace nemotron {

std::optional<EmbeddingProjectionStats> RunEmbeddingThenDenseProjectionFp32(
    const DeviceEmbeddingTableFp32& embedding_table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    const DeviceDenseWeightFp32& projection_weight,
    CublasLtHandle& handle,
    const CublasLtGemmPlan& projection_plan,
    DeviceTensorFp32* output) {
  if (!embedding_table.valid() ||
      !projection_weight.valid() ||
      host_token_ids == nullptr ||
      token_count == 0 ||
      output == nullptr ||
      !output->valid()) {
    return std::nullopt;
  }

  auto embeddings = DeviceTensorFp32::Create({token_count, embedding_table.embedding_dim()});
  if (!embeddings || !embeddings->valid()) {
    return std::nullopt;
  }

  const auto lookup = LookupEmbeddingRowsFp32(
      embedding_table,
      host_token_ids,
      token_count,
      embeddings.get());
  if (!lookup.has_value()) {
    return std::nullopt;
  }

  const auto projection = RunDenseRowMajorFp32ToDevice(
      handle,
      projection_plan,
      projection_weight,
      *embeddings,
      output);
  if (!projection.has_value()) {
    return std::nullopt;
  }

  return EmbeddingProjectionStats{
      lookup->token_count,
      lookup->embedding_dim,
      *projection,
  };
}

std::optional<EmbeddingProjectionStats> RunEmbeddingThenDenseProjectionFp32(
    const DeviceEmbeddingTableFp32& embedding_table,
    const std::int32_t* host_token_ids,
    std::size_t token_count,
    CublasLtHandle& handle,
    const CublasLtGemmPlan& projection_plan,
    DeviceTensorFp32* output) {
  if (!embedding_table.valid() ||
      host_token_ids == nullptr ||
      token_count == 0 ||
      output == nullptr ||
      !output->valid()) {
    return std::nullopt;
  }

  auto embeddings = DeviceTensorFp32::Create({token_count, embedding_table.embedding_dim()});
  if (!embeddings || !embeddings->valid()) {
    return std::nullopt;
  }

  const auto lookup = LookupEmbeddingRowsFp32(
      embedding_table,
      host_token_ids,
      token_count,
      embeddings.get());
  if (!lookup.has_value()) {
    return std::nullopt;
  }

  const auto projection = RunDenseRowMajorFp32ToDevice(
      handle,
      projection_plan,
      *embeddings,
      output);
  if (!projection.has_value()) {
    return std::nullopt;
  }

  return EmbeddingProjectionStats{
      lookup->token_count,
      lookup->embedding_dim,
      *projection,
  };
}

}  // namespace nemotron
