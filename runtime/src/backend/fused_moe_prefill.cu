#include "nemotron/fused_moe_prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "nemotron/device_tensor.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/linear_op_trace.h"
#include <cutlass/arch/barrier.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor_impl.hpp>

#include "nemotron/device_nvfp4_matrix.h"
#include "fused_decode_common.cuh"
#include "routed_p5_tma_descriptor.cuh"

namespace nemotron {

bool RunLaunchPlannedNvfp4ExpertMatVecBf16(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output);

namespace {
#include "fused_moe_prefill/common_helpers.cuh"
#include "fused_moe_prefill/nvfp4_cute.cuh"
#include "fused_moe_prefill/nvfp4_bridge.cuh"
#include "fused_moe_prefill/trt_helpers_pre_nano_epilogue.cuh"
#include "fused_moe_prefill/nano_p1_epilogue.cuh"
#include "fused_moe_prefill/trt_helpers_post_nano_epilogue.cuh"
#include "fused_moe_prefill/nano_p1_kernel.cuh"
}  // namespace
#include "fused_moe_prefill/public_exports.cuh"
}  // namespace nemotron
