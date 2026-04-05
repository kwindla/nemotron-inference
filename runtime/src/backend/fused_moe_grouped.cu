#include "nemotron/fused_moe_grouped.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cute/tensor.hpp>

#pragma GCC diagnostic pop

namespace nemotron {

using namespace cute;

// Architectural config
using ArchTag = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Data types
using ElementInput = cutlass::float_e2m1_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementC = void;
using ElementD = float;
using ElementAccum = float;
using ElementCompute = float;
using ElementSF = cutlass::float_ue4m3_t;

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Alignments
constexpr int AlignA = 32;
constexpr int AlignB = 32;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Shapes
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using MmaTileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// Epilogue
using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAccum, ElementC, ElementAccum>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementC, LayoutC, AlignD,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOp>::CollectiveOp;

// Mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccum,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Host workspace definition
struct WorkspaceArrays {
    cute::Shape<int, int, int>* problem_sizes;
    const ElementA** ptr_A;
    const ElementB** ptr_B;
    ElementD** ptr_D;
    const ElementSF** ptr_SFA;
    const ElementSF** ptr_SFB;
    float* alpha;
    typename GemmKernel::StrideA* stride_A;
    typename GemmKernel::StrideB* stride_B;
    typename GemmKernel::StrideD* stride_D;
    typename GemmKernel::CollectiveMainloop::InternalLayoutSFA* layout_SFA;
    typename GemmKernel::CollectiveMainloop::InternalLayoutSFB* layout_SFB;
};

__global__ void SetupGroupedGemmWorkspaceKernel(
    WorkspaceArrays ws,
    const std::uint8_t* const* packed_activations_device,
    const std::uint8_t* const* activation_scales_device,
    const float* const* activation_global_scale_device,
    const std::uint8_t* const* packed_weights_device,
    const std::uint8_t* const* weight_scales_device,
    const float* const* weight_global_scales_device,
    float* const* output_activations_device,
    const int32_t* expert_token_counts,
    int hidden_size,
    int intermediate_size,
    int num_experts) {
    
    int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_idx >= num_experts) return;

    int m = expert_token_counts[expert_idx];
    int n = intermediate_size; 
    int k = hidden_size;

    ws.problem_sizes[expert_idx] = cute::make_shape(m, n, k);

    ws.ptr_A[expert_idx] = reinterpret_cast<const ElementA*>(packed_activations_device[expert_idx]);
    ws.stride_A[expert_idx] = cute::make_int_tuple_from<typename GemmKernel::StrideA>(k, 0);

    ws.ptr_B[expert_idx] = reinterpret_cast<const ElementB*>(packed_weights_device[expert_idx]);
    ws.stride_B[expert_idx] = cute::make_int_tuple_from<typename GemmKernel::StrideB>(k, 0);

    ws.ptr_D[expert_idx] = reinterpret_cast<ElementD*>(output_activations_device[expert_idx]);
    ws.stride_D[expert_idx] = cute::make_int_tuple_from<typename GemmKernel::StrideD>(n, 0);

    ws.ptr_SFA[expert_idx] = reinterpret_cast<const ElementSF*>(activation_scales_device[expert_idx]);
    ws.ptr_SFB[expert_idx] = reinterpret_cast<const ElementSF*>(weight_scales_device[expert_idx]);
    
    // Scale layout configuration (pads M to 128 under the hood for 128x4 swizzled layout)
    ws.layout_SFA[expert_idx] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    ws.layout_SFB[expert_idx] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    // Alpha calculation: alpha = act_global * weight_global
    float act_scale = 1.0f;
    if (activation_global_scale_device && activation_global_scale_device[expert_idx]) {
        act_scale = *activation_global_scale_device[expert_idx];
    }
    float weight_scale = 1.0f;
    if (weight_global_scales_device && weight_global_scales_device[expert_idx]) {
        weight_scale = *weight_global_scales_device[expert_idx];
    }
    ws.alpha[expert_idx] = act_scale * weight_scale;
}

std::size_t GetNvfp4GroupedMoEWorkspaceSize(int num_experts) {
    std::size_t size = 0;
    size += sizeof(cute::Shape<int, int, int>) * num_experts;
    size += sizeof(const ElementA*) * num_experts;
    size += sizeof(const ElementB*) * num_experts;
    size += sizeof(ElementD*) * num_experts;
    size += sizeof(const ElementSF*) * num_experts;
    size += sizeof(const ElementSF*) * num_experts;
    size += sizeof(float) * num_experts; // alpha per expert
    size += sizeof(float*) * num_experts; // array of pointers to alpha
    size += sizeof(typename GemmKernel::StrideA) * num_experts;
    size += sizeof(typename GemmKernel::StrideB) * num_experts;
    size += sizeof(typename GemmKernel::StrideD) * num_experts;
    size += sizeof(typename GemmKernel::CollectiveMainloop::InternalLayoutSFA) * num_experts;
    size += sizeof(typename GemmKernel::CollectiveMainloop::InternalLayoutSFB) * num_experts;
    
    size += Gemm::get_workspace_size(typename Gemm::Arguments{});
    
    return (size + 127) & ~127;
}

__global__ void SetupAlphaPtrArray(float** ptr_array, float* alpha_data, int num_experts) {
    int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_idx >= num_experts) return;
    ptr_array[expert_idx] = &alpha_data[expert_idx];
}

bool RunNvfp4GroupedMoEFp32AccumToDevice(
    const std::uint8_t* const* packed_activations_device,
    const std::uint8_t* const* activation_scales_device,
    const float* const* activation_global_scale_device,
    const std::uint8_t* const* packed_weights_device,
    const std::uint8_t* const* weight_scales_device,
    const float* const* weight_global_scales_device,
    float* const* output_activations_device,
    const int32_t* expert_token_counts_device,
    int hidden_size,
    int intermediate_size,
    int num_experts,
    Nvfp4GroupedMoEWorkspace& workspace,
    cudaStream_t stream) {
    
    if (!workspace.data || workspace.nbytes < GetNvfp4GroupedMoEWorkspaceSize(num_experts)) {
        return false;
    }

    std::uint8_t* ws_ptr = static_cast<std::uint8_t*>(workspace.data);
    WorkspaceArrays ws;
    ws.problem_sizes = reinterpret_cast<cute::Shape<int, int, int>*>(ws_ptr); ws_ptr += sizeof(cute::Shape<int, int, int>) * num_experts;
    ws.ptr_A = reinterpret_cast<const ElementA**>(ws_ptr); ws_ptr += sizeof(const ElementA*) * num_experts;
    ws.ptr_B = reinterpret_cast<const ElementB**>(ws_ptr); ws_ptr += sizeof(const ElementB*) * num_experts;
    ws.ptr_D = reinterpret_cast<ElementD**>(ws_ptr); ws_ptr += sizeof(ElementD*) * num_experts;
    ws.ptr_SFA = reinterpret_cast<const ElementSF**>(ws_ptr); ws_ptr += sizeof(const ElementSF*) * num_experts;
    ws.ptr_SFB = reinterpret_cast<const ElementSF**>(ws_ptr); ws_ptr += sizeof(const ElementSF*) * num_experts;
    ws.alpha = reinterpret_cast<float*>(ws_ptr); ws_ptr += sizeof(float) * num_experts;
    float** alpha_ptr_array = reinterpret_cast<float**>(ws_ptr); ws_ptr += sizeof(float*) * num_experts;
    ws.stride_A = reinterpret_cast<typename GemmKernel::StrideA*>(ws_ptr); ws_ptr += sizeof(typename GemmKernel::StrideA) * num_experts;
    ws.stride_B = reinterpret_cast<typename GemmKernel::StrideB*>(ws_ptr); ws_ptr += sizeof(typename GemmKernel::StrideB) * num_experts;
    ws.stride_D = reinterpret_cast<typename GemmKernel::StrideD*>(ws_ptr); ws_ptr += sizeof(typename GemmKernel::StrideD) * num_experts;
    ws.layout_SFA = reinterpret_cast<typename GemmKernel::CollectiveMainloop::InternalLayoutSFA*>(ws_ptr); ws_ptr += sizeof(typename GemmKernel::CollectiveMainloop::InternalLayoutSFA) * num_experts;
    ws.layout_SFB = reinterpret_cast<typename GemmKernel::CollectiveMainloop::InternalLayoutSFB*>(ws_ptr); ws_ptr += sizeof(typename GemmKernel::CollectiveMainloop::InternalLayoutSFB) * num_experts;
    
    void* internal_ws = ws_ptr;

    int threads = 128;
    int blocks = (num_experts + threads - 1) / threads;
    
    SetupGroupedGemmWorkspaceKernel<<<blocks, threads, 0, stream>>>(
        ws,
        packed_activations_device,
        activation_scales_device,
        activation_global_scale_device,
        packed_weights_device,
        weight_scales_device,
        weight_global_scales_device,
        output_activations_device,
        expert_token_counts_device,
        hidden_size,
        intermediate_size,
        num_experts);
    
    if (cudaGetLastError() != cudaSuccess) return false;

    SetupAlphaPtrArray<<<blocks, threads, 0, stream>>>(
        alpha_ptr_array, ws.alpha, num_experts);

    if (cudaGetLastError() != cudaSuccess) return false;

    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape = ProblemShape{num_experts, reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(ws.problem_sizes), nullptr};
    
    args.mainloop.ptr_A = ws.ptr_A;
    args.mainloop.stride_A = ws.stride_A;
    args.mainloop.ptr_B = ws.ptr_B;
    args.mainloop.stride_B = ws.stride_B;
    args.mainloop.ptr_SFA = ws.ptr_SFA;
    args.mainloop.layout_SFA = ws.layout_SFA;
    args.mainloop.ptr_SFB = ws.ptr_SFB;
    args.mainloop.layout_SFB = ws.layout_SFB;

    args.epilogue.ptr_C = nullptr;
    args.epilogue.dC = ws.stride_D;
    args.epilogue.ptr_D = ws.ptr_D;
    args.epilogue.dD = ws.stride_D;
    args.epilogue.thread.alpha_ptr_array = alpha_ptr_array;
    args.epilogue.thread.alpha = 1.0f; // not used since alpha_ptr_array is provided
    args.epilogue.thread.beta = 0.0f; 

    args.hw_info.cluster_shape = dim3(1, 1, 1);
    args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    Gemm gemm;
    auto can_impl = gemm.can_implement(args);
    if (can_impl != cutlass::Status::kSuccess) {
        return false;
    }

    std::size_t required_internal_ws = gemm.get_workspace_size(args);
    std::size_t internal_ws_available = workspace.nbytes - (ws_ptr - static_cast<std::uint8_t*>(workspace.data));
    if (internal_ws_available < required_internal_ws) {
        return false;
    }

    auto init_status = gemm.initialize(args, internal_ws, stream);
    if (init_status != cutlass::Status::kSuccess) {
        return false;
    }

    auto run_status = gemm.run(args, internal_ws, stream);
    if (run_status != cutlass::Status::kSuccess) {
        return false;
    }

    return true;
}

}  // namespace nemotron