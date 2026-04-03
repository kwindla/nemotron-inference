#include "nemotron/mamba_gated_group_norm.h"

#include <cuda_runtime.h>

#include <cstddef>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = fused_decode::kThreadsPerBlock / kWarpSize;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
  for (int offset = static_cast<int>(kWarpSize) / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__global__ void MambaGatedGroupNormKernel(
    const float* __restrict__ projected,
    std::size_t projected_stride,
    float* __restrict__ ssm_output,
    std::size_t ssm_output_stride,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t n_groups,
    float mixer_rms_epsilon,
    const float* __restrict__ mixer_norm_weight) {
  __shared__ float warp_sums[kWarpsPerBlock];
  __shared__ float shared_rstd;

  const std::size_t group = blockIdx.x;
  const std::size_t token_index = blockIdx.y;
  if (group >= n_groups || token_index >= token_count) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane = tid % kWarpSize;
  const std::size_t warp = tid / kWarpSize;
  const std::size_t mixer_group_size = intermediate_size / n_groups;
  const std::size_t begin = group * mixer_group_size;
  const std::size_t end = begin + mixer_group_size;

  const float* gate = projected + (token_index * projected_stride);
  float* output_row = ssm_output + (token_index * ssm_output_stride);

  float variance_sum = 0.0f;
  for (std::size_t hidden_index = begin + tid; hidden_index < end; hidden_index += blockDim.x) {
    const float gated = output_row[hidden_index] * fused_decode::SiLU(gate[hidden_index]);
    output_row[hidden_index] = gated;
    variance_sum = fmaf(gated, gated, variance_sum);
  }

  variance_sum = WarpReduceSum(variance_sum);
  if (lane == 0) {
    warp_sums[warp] = variance_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0f;
    block_sum = WarpReduceSum(block_sum);
    if (lane == 0) {
      const float variance = block_sum / static_cast<float>(mixer_group_size);
      shared_rstd = rsqrtf(variance + mixer_rms_epsilon);
    }
  }
  __syncthreads();

  for (std::size_t hidden_index = begin + tid; hidden_index < end; hidden_index += blockDim.x) {
    output_row[hidden_index] =
        output_row[hidden_index] * shared_rstd * mixer_norm_weight[hidden_index];
  }
}

}  // namespace

bool RunMambaGatedGroupNorm(
    const MambaGatedGroupNormParams& params,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* ssm_output) {
  if (!projected.valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] == 0 ||
      ssm_output == nullptr ||
      !ssm_output->valid() ||
      ssm_output->shape().size() != 2 ||
      params.intermediate_size == 0 ||
      params.n_groups == 0 ||
      (params.intermediate_size % params.n_groups) != 0 ||
      params.mixer_rms_epsilon <= 0.0f ||
      params.mixer_norm_weight == nullptr) {
    return false;
  }

  if (projected.shape()[0] != ssm_output->shape()[0] ||
      projected.shape()[1] < params.intermediate_size ||
      ssm_output->shape()[1] != params.intermediate_size) {
    return false;
  }

  const std::size_t token_count = projected.shape()[0];
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned>(params.n_groups),
      static_cast<unsigned>(token_count));
  MambaGatedGroupNormKernel<<<grid, block>>>(
      projected.data(),
      projected.shape()[1],
      ssm_output->data(),
      ssm_output->shape()[1],
      token_count,
      params.intermediate_size,
      params.n_groups,
      params.mixer_rms_epsilon,
      params.mixer_norm_weight);
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
