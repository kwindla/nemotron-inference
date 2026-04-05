// SM120 CUTLASS grouped FP4×FP4→FP32 GEMM for MoE expert dispatch.
// One kernel launch for all routed experts.

#include "nemotron/fused_moe_grouped.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cute/tensor.hpp>

#pragma GCC diagnostic pop

#include <vector>

namespace nemotron {
namespace {

using namespace cute;

// ── CUTLASS type configuration (SM120 grouped FP4 block-scaled) ─────────

using ArchTag = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = void;
using ElementD = float;
using ElementAccum = float;
using ElementCompute = float;
using ElementSF = cutlass::float_ue4m3_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignA = 32;
constexpr int AlignB = 32;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;
using MmaTileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAccum, ElementC, ElementAccum>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementC, LayoutD*, AlignD,
    ElementD, LayoutD*, AlignD,
    EpilogueSchedule,
    FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA*, AlignA,
    ElementB, LayoutB*, AlignB,
    ElementAccum,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using BlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using InternalStrideA = typename GemmKernel::InternalStrideA;
using InternalStrideB = typename GemmKernel::InternalStrideB;
using InternalStrideD = typename GemmKernel::InternalStrideD;
using InternalLayoutSFA = typename GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using InternalLayoutSFB = typename GemmKernel::CollectiveMainloop::InternalLayoutSFB;

using ElemFP4 = cutlass::float_e2m1_t;

template <typename IntT>
CUTLASS_HOST_DEVICE
cute::Stride<IntT, cute::Int<1>, cute::Int<0>>
MakeGroupedPackedStride(
    cute::Stride<IntT, cute::Int<1>, cute::Int<0>>,
    int64_t dim0,
    int64_t dim1) {
  auto stride = cute::Stride<IntT, cute::Int<1>, cute::Int<0>>{};
  cute::get<0>(stride) = static_cast<IntT>(dim1);
  return stride;
}

template <typename IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Int<1>, IntT, cute::Int<0>>
MakeGroupedPackedStride(
    cute::Stride<cute::Int<1>, IntT, cute::Int<0>>,
    int64_t dim0,
    int64_t dim1) {
  auto stride = cute::Stride<cute::Int<1>, IntT, cute::Int<0>>{};
  cute::get<1>(stride) = static_cast<IntT>(dim0);
  return stride;
}

// ── Device kernel to set up per-expert workspace arrays ─────────────────

struct WorkspaceLayout {
  typename ProblemShape::UnderlyingProblemShape* shapes;
  const ElemFP4** ptr_A;
  const ElemFP4** ptr_B;
  float** ptr_D;
  const ElementSF** ptr_SFA;
  const ElementSF** ptr_SFB;
  InternalStrideA* dA;
  InternalStrideB* dB;
  InternalStrideD* dD;
  InternalLayoutSFA* layout_SFA;
  InternalLayoutSFB* layout_SFB;
  float* alpha_values;
  const float** alpha_ptrs;
};

// ── Workspace size calculation ──────────────────────────────────────────

struct SlabAllocator {
  std::uint8_t* base;
  std::size_t offset = 0;

  template <typename T>
  T* alloc(int count, std::size_t alignment = 16) {
    offset = (offset + alignment - 1) & ~(alignment - 1);
    T* ptr = reinterpret_cast<T*>(base + offset);
    offset += sizeof(T) * count;
    return ptr;
  }
};

std::size_t ArraysBytes(int n) {
  std::size_t s = 0;
  auto align = [&](std::size_t a = 16) { s = (s + a - 1) & ~(a - 1); };
  s += sizeof(typename ProblemShape::UnderlyingProblemShape) * n; align();
  s += sizeof(const ElemFP4*) * n; align();
  s += sizeof(const ElemFP4*) * n; align();
  s += sizeof(float*) * n; align();
  s += sizeof(const ElementSF*) * n; align();
  s += sizeof(const ElementSF*) * n; align();
  s += sizeof(InternalStrideA) * n; align();
  s += sizeof(InternalStrideB) * n; align();
  s += sizeof(InternalStrideD) * n; align();
  s += sizeof(InternalLayoutSFA) * n; align();
  s += sizeof(InternalLayoutSFB) * n; align();
  s += sizeof(float) * n; align();       // alpha_values
  s += sizeof(const float*) * n; align(); // alpha_ptrs
  return s;
}

}  // namespace

std::size_t GroupedMoeWorkspaceBytes(int max_experts) {
  std::size_t bytes = ArraysBytes(max_experts);
  // Add CUTLASS internal workspace (query with dummy args)
  bytes = (bytes + 127) & ~127;
  // Conservative: CUTLASS grouped workspace scales with num_groups
  bytes += 256 * 1024;  // 256 KB headroom
  return bytes;
}

bool RunGroupedFp4Gemm(
    const std::uint8_t* const* packed_A,
    const std::uint8_t* const* scales_A,
    const float* const* global_A,
    const std::uint8_t* const* packed_B,
    const std::uint8_t* const* scales_B,
    const float* const* global_B,
    float* const* output_D,
    const std::int32_t* token_counts,
    int K, int N, int num_experts,
    GroupedMoeWorkspace& workspace,
    cudaStream_t stream) {

  const std::size_t required = GroupedMoeWorkspaceBytes(num_experts);
  if (workspace.data == nullptr || workspace.nbytes < required) {
    return false;
  }

  std::vector<std::int32_t> token_counts_host(num_experts, 0);
  if (num_experts > 0 &&
      cudaMemcpy(
          token_counts_host.data(),
          token_counts,
          sizeof(std::int32_t) * static_cast<std::size_t>(num_experts),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    return false;
  }

  std::vector<typename ProblemShape::UnderlyingProblemShape> host_shapes(
      num_experts);
  std::vector<const std::uint8_t*> packed_A_host(num_experts, nullptr);
  std::vector<const std::uint8_t*> scales_A_host(num_experts, nullptr);
  std::vector<const float*> global_A_host(num_experts, nullptr);
  std::vector<const std::uint8_t*> packed_B_host(num_experts, nullptr);
  std::vector<const std::uint8_t*> scales_B_host(num_experts, nullptr);
  std::vector<const float*> global_B_host(num_experts, nullptr);
  std::vector<float*> output_D_host(num_experts, nullptr);
  if (num_experts > 0 &&
      (cudaMemcpy(
           packed_A_host.data(),
           packed_A,
           sizeof(const std::uint8_t*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(
           scales_A_host.data(),
           scales_A,
           sizeof(const std::uint8_t*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(
           global_A_host.data(),
           global_A,
           sizeof(const float*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(
           packed_B_host.data(),
           packed_B,
           sizeof(const std::uint8_t*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(
           scales_B_host.data(),
           scales_B,
           sizeof(const std::uint8_t*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(
           global_B_host.data(),
           global_B,
           sizeof(const float*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyDeviceToHost) != cudaSuccess ||
       cudaMemcpy(
           output_D_host.data(),
           output_D,
           sizeof(float*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyDeviceToHost) != cudaSuccess)) {
    return false;
  }

  std::vector<const ElemFP4*> ptr_A_host(num_experts, nullptr);
  std::vector<const ElemFP4*> ptr_B_host(num_experts, nullptr);
  std::vector<float*> ptr_D_host(num_experts, nullptr);
  std::vector<const ElementSF*> ptr_SFA_host(num_experts, nullptr);
  std::vector<const ElementSF*> ptr_SFB_host(num_experts, nullptr);
  std::vector<InternalStrideA> dA_host(num_experts);
  std::vector<InternalStrideB> dB_host(num_experts);
  std::vector<InternalStrideD> dD_host(num_experts);
  std::vector<InternalLayoutSFA> layout_SFA_host(num_experts);
  std::vector<InternalLayoutSFB> layout_SFB_host(num_experts);
  std::vector<float> alpha_values_host(num_experts, 1.0f);
  for (int i = 0; i < num_experts; ++i) {
    const auto index = static_cast<std::size_t>(i);
    const int M = token_counts_host[index];
    host_shapes[index] = cute::make_shape(int64_t(M), int64_t(N), int64_t(K));
    ptr_A_host[index] = reinterpret_cast<const ElemFP4*>(packed_A_host[index]);
    ptr_B_host[index] = reinterpret_cast<const ElemFP4*>(packed_B_host[index]);
    ptr_D_host[index] = output_D_host[index];
    ptr_SFA_host[index] = reinterpret_cast<const ElementSF*>(scales_A_host[index]);
    ptr_SFB_host[index] = reinterpret_cast<const ElementSF*>(scales_B_host[index]);
    dA_host[index] =
        MakeGroupedPackedStride(InternalStrideA{}, int64_t(M), int64_t(K));
    dB_host[index] =
        MakeGroupedPackedStride(InternalStrideB{}, int64_t(N), int64_t(K));
    dD_host[index] =
        MakeGroupedPackedStride(InternalStrideD{}, int64_t(M), int64_t(N));
    layout_SFA_host[index] =
        BlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    layout_SFB_host[index] =
        BlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    float act_scale = 1.0f;
    float wt_scale = 1.0f;
    if (global_A_host[index] != nullptr &&
        cudaMemcpy(
            &act_scale,
            global_A_host[index],
            sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    if (global_B_host[index] != nullptr &&
        cudaMemcpy(
            &wt_scale,
            global_B_host[index],
            sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    alpha_values_host[index] = act_scale * wt_scale;
  }

  // Sub-allocate device arrays from workspace slab
  SlabAllocator slab{static_cast<std::uint8_t*>(workspace.data)};
  WorkspaceLayout ws;
  ws.shapes = slab.alloc<typename ProblemShape::UnderlyingProblemShape>(num_experts);
  ws.ptr_A = slab.alloc<const ElemFP4*>(num_experts);
  ws.ptr_B = slab.alloc<const ElemFP4*>(num_experts);
  ws.ptr_D = slab.alloc<float*>(num_experts);
  ws.ptr_SFA = slab.alloc<const ElementSF*>(num_experts);
  ws.ptr_SFB = slab.alloc<const ElementSF*>(num_experts);
  ws.dA = slab.alloc<InternalStrideA>(num_experts);
  ws.dB = slab.alloc<InternalStrideB>(num_experts);
  ws.dD = slab.alloc<InternalStrideD>(num_experts);
  ws.layout_SFA = slab.alloc<InternalLayoutSFA>(num_experts);
  ws.layout_SFB = slab.alloc<InternalLayoutSFB>(num_experts);
  ws.alpha_values = slab.alloc<float>(num_experts);
  ws.alpha_ptrs = slab.alloc<const float*>(num_experts);

  void* cutlass_ws = slab.base + ((slab.offset + 127) & ~127);
  std::size_t cutlass_ws_bytes = workspace.nbytes - ((slab.offset + 127) & ~127);

  std::vector<const float*> alpha_ptrs_host(num_experts, nullptr);
  for (int i = 0; i < num_experts; ++i) {
    alpha_ptrs_host[static_cast<std::size_t>(i)] = ws.alpha_values + i;
  }

  if (num_experts > 0 &&
      (cudaMemcpy(
           ws.shapes,
           host_shapes.data(),
           sizeof(typename ProblemShape::UnderlyingProblemShape) *
               static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.ptr_A,
           ptr_A_host.data(),
           sizeof(const ElemFP4*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.ptr_B,
           ptr_B_host.data(),
           sizeof(const ElemFP4*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.ptr_D,
           ptr_D_host.data(),
           sizeof(float*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.ptr_SFA,
           ptr_SFA_host.data(),
           sizeof(const ElementSF*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.ptr_SFB,
           ptr_SFB_host.data(),
           sizeof(const ElementSF*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.dA,
           dA_host.data(),
           sizeof(InternalStrideA) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.dB,
           dB_host.data(),
           sizeof(InternalStrideB) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.dD,
           dD_host.data(),
           sizeof(InternalStrideD) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.layout_SFA,
           layout_SFA_host.data(),
           sizeof(InternalLayoutSFA) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.layout_SFB,
           layout_SFB_host.data(),
           sizeof(InternalLayoutSFB) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.alpha_values,
           alpha_values_host.data(),
           sizeof(float) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(
           ws.alpha_ptrs,
           alpha_ptrs_host.data(),
           sizeof(const float*) * static_cast<std::size_t>(num_experts),
           cudaMemcpyHostToDevice) != cudaSuccess)) {
    return false;
  }

  // Build CUTLASS arguments
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape =
      ProblemShape{num_experts, ws.shapes, host_shapes.data()};

  args.mainloop.ptr_A = ws.ptr_A;
  args.mainloop.dA = ws.dA;
  args.mainloop.ptr_B = ws.ptr_B;
  args.mainloop.dB = ws.dB;
  args.mainloop.ptr_SFA = ws.ptr_SFA;
  args.mainloop.layout_SFA = ws.layout_SFA;
  args.mainloop.ptr_SFB = ws.ptr_SFB;
  args.mainloop.layout_SFB = ws.layout_SFB;

  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = ws.dD;
  args.epilogue.ptr_D = ws.ptr_D;
  args.epilogue.dD = ws.dD;
  args.epilogue.thread.alpha_ptr = nullptr;
  args.epilogue.thread.beta_ptr = nullptr;
  args.epilogue.thread.alpha_ptr_array = ws.alpha_ptrs;
  args.epilogue.thread.beta_ptr_array = nullptr;
  args.epilogue.thread.dAlpha = {_0{}, _0{}, 1};
  args.epilogue.thread.dBeta = {_0{}, _0{}, 0};
  args.epilogue.thread.alpha = 0.0f;
  args.epilogue.thread.beta = 0.0f;

  args.hw_info.device_id = 0;
  cudaDeviceGetAttribute(&args.hw_info.sm_count, cudaDevAttrMultiProcessorCount, 0);
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

  Gemm gemm;

  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return false;
  }

  const std::size_t needed_ws = gemm.get_workspace_size(args);
  if (needed_ws > cutlass_ws_bytes) {
    return false;
  }

  if (gemm.initialize(args, cutlass_ws, stream) != cutlass::Status::kSuccess) {
    return false;
  }

  if (gemm.run(args, cutlass_ws, stream) != cutlass::Status::kSuccess) {
    return false;
  }

  return true;
}

}  // namespace nemotron
