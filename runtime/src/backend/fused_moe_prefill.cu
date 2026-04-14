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

namespace wmma = nvcuda::wmma;

template <class Layout>
std::string LayoutString(Layout const& layout) {
  std::ostringstream oss;
  oss << layout;
  return oss.str();
}

__host__ __device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout);

__host__ __device__ std::size_t RoundUp(std::size_t value, std::size_t alignment);
bool LaunchZeroBuffer(float* data, std::size_t count);
bool LaunchZeroBf16Buffer(__nv_bfloat16* data, std::size_t count);
bool LaunchGatherRows(
    const float* input,
    const int* indices,
    const int* active_output_rows,
    std::size_t output_rows,
    std::size_t input_rows,
    std::size_t cols,
    float* output);
bool LaunchRoutedBf16Relu2Pack(
    const __nv_bfloat16* source,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    DeviceNvfp4Matrix* output_pack,
    float* output_dequant_scales);
bool LaunchPlannedPackedInputMatVecBf16(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output);

namespace cute = ::cute;

namespace nvfp4_cute {

using ElementAB = cute::float_e2m1_t;
using ElementSFCompute = cute::float_ue4m3_t;
static constexpr int kScaleVecSize = 16;

using MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    ElementAB,
    ElementAB,
    float,
    ElementSFCompute,
    kScaleVecSize>;
}  // namespace nvfp4_cute

namespace nvfp4_bridge {

using ARegister = std::remove_extent_t<typename nvfp4_cute::MmaOp::ARegisters>;
using BRegister = std::remove_extent_t<typename nvfp4_cute::MmaOp::BRegisters>;
using CRegister = std::remove_extent_t<typename nvfp4_cute::MmaOp::CRegisters>;
using SFRegister = std::remove_extent_t<typename nvfp4_cute::MmaOp::SFARegisters>;
using Atom = cute::MMA_Atom<nvfp4_cute::MmaOp>;
using SingleAtomTiledMma = cute::TiledMMA<Atom, cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>>;
using TracedP5AtomLayoutMNK = cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>;
using TracedP5PermTileN =
    cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_2>, cute::Stride<cute::_1, cute::_16, cute::_8>>;
using TracedP5LayoutA = cutlass::layout::RowMajor;
using TracedP5LayoutB = cutlass::layout::ColumnMajor;
using TracedP5LayoutC = cutlass::layout::ColumnMajor;
using TracedP5LayoutD = cutlass::layout::ColumnMajor;
using TracedP5MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using TracedP5ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using TracedP5EpilogueTensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using TracedP5TensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using TracedP5ElementD = __nv_bfloat16;
using TracedP5Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5EpilogueTensorOp,
    TracedP5MmaTileShape,
    TracedP5ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    TracedP5ElementD,
    TracedP5LayoutC*,
    32,
    TracedP5ElementD,
    TracedP5LayoutD*,
    32,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using TracedP5StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename TracedP5Epilogue::SharedStorage))>;
using TracedP5CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5TensorOp,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutA*,
    32,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutB*,
    32,
    float,
    TracedP5MmaTileShape,
    TracedP5ClusterShape,
    TracedP5StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using TracedP5TiledMma = typename TracedP5CollectiveMainloop::TiledMma;
using TracedP5SmemLayoutA = typename TracedP5CollectiveMainloop::SmemLayoutA;
using TracedP5SmemLayoutB = typename TracedP5CollectiveMainloop::SmemLayoutB;
using TracedP5SmemLayoutSFA = typename TracedP5CollectiveMainloop::SmemLayoutSFA;
using TracedP5SmemLayoutSFB = typename TracedP5CollectiveMainloop::SmemLayoutSFB;
using TracedP5SmemCopyAtomA = typename TracedP5CollectiveMainloop::SmemCopyAtomA;
using TracedP5SmemCopyAtomB = typename TracedP5CollectiveMainloop::SmemCopyAtomB;
using TracedP5SmemCopyAtomSFA = typename TracedP5CollectiveMainloop::SmemCopyAtomSFA;
using TracedP5SmemCopyAtomSFB = typename TracedP5CollectiveMainloop::SmemCopyAtomSFB;
// Definitive host-side probe facts from artifacts/tmp/trt_p5_runtime_layout_dump.cu:
// - Stages == 4
// - cosize(SmemLayoutSFA) == cosize(SmemLayoutSFB) == 4096
// - cosize(tCrSFA) == 16
// - cosize(tCrSFB) == 64
// These values prove the traced P5 scale path is not representable by the
// older row-wise scale_words + scale[1] abstraction.
constexpr int kTracedP5ScaleSmemCosizeA = cute::cosize_v<TracedP5SmemLayoutSFA>;
constexpr int kTracedP5ScaleSmemCosizeB = cute::cosize_v<TracedP5SmemLayoutSFB>;
constexpr int kTracedP5ScaleFragmentCosizeA = 16;
constexpr int kTracedP5ScaleFragmentCosizeB = 64;
constexpr int kTracedP5DirectStageCols = 128;
constexpr int kTracedP5DirectStageRows = 32;
constexpr int kTracedP5DirectBlocksPerRow =
    kTracedP5DirectStageCols / fused_decode::kNvfp4BlockWidth;
using TracedP5AccumProfileLayout = decltype(
    cute::partition_fragment_C(
        TracedP5TiledMma{},
        cute::make_shape(cute::Int<128>{}, cute::Int<128>{}))
        .layout());
constexpr int kTracedP5AccumProfileCosize = cute::cosize_v<TracedP5AccumProfileLayout>;
static_assert(fused_decode::kNvfp4BlockWidth == 16, "FP4 block width must be 16");
static_assert(
    kTracedP5DirectStageCols == 128 && kTracedP5DirectStageRows == 32,
    "P5 direct stage tile must be 128x32");
using TracedP13MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>;
using TracedP13ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using TracedP13Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5EpilogueTensorOp,
    TracedP13MmaTileShape,
    TracedP13ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    TracedP5ElementD,
    TracedP5LayoutC*,
    32,
    TracedP5ElementD,
    TracedP5LayoutD*,
    32,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using TracedP13StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename TracedP13Epilogue::SharedStorage))>;
using TracedP13CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5TensorOp,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutA*,
    32,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutB*,
    32,
    float,
    TracedP13MmaTileShape,
    TracedP13ClusterShape,
    TracedP13StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using TracedP13TiledMma = typename TracedP13CollectiveMainloop::TiledMma;
using TracedP13SmemLayoutA = typename TracedP13CollectiveMainloop::SmemLayoutA;
using TracedP13SmemLayoutB = typename TracedP13CollectiveMainloop::SmemLayoutB;
using TracedP13SmemLayoutSFA = typename TracedP13CollectiveMainloop::SmemLayoutSFA;
using TracedP13SmemLayoutSFB = typename TracedP13CollectiveMainloop::SmemLayoutSFB;
using TracedP13SmemCopyAtomA = typename TracedP13CollectiveMainloop::SmemCopyAtomA;
using TracedP13SmemCopyAtomB = typename TracedP13CollectiveMainloop::SmemCopyAtomB;
using TracedP13SmemCopyAtomSFA = typename TracedP13CollectiveMainloop::SmemCopyAtomSFA;
using TracedP13SmemCopyAtomSFB = typename TracedP13CollectiveMainloop::SmemCopyAtomSFB;
// Definitive host-side probe facts from artifacts/tmp/trt_p13_runtime_layout_dump.cu:
// - Stages == 9
// - size(TiledMma) == 256 and tile_mnk == (128, 32, 64)
// - cosize(SmemLayoutSFA) == cosize(SmemLayoutSFB) == 4608
// - cosize(tCrSFA) == 8
// - cosize(tCrSFB) == 32
// - accum_profile.layout == ((_2,_2),_2,(_2,_4)):((_1@1,_8@0),_64@0,(_8@1,_32@1))
// These values prove low-row FC2 P13 cannot safely reuse the older traced P5
// scale-smem assumptions.
constexpr int kTracedP13ScaleSmemCosizeA = cute::cosize_v<TracedP13SmemLayoutSFA>;
constexpr int kTracedP13ScaleSmemCosizeB = cute::cosize_v<TracedP13SmemLayoutSFB>;
constexpr int kTracedP13ScaleFragmentCosizeA = 8;
constexpr int kTracedP13ScaleFragmentCosizeB = 32;
using TracedP15MmaTileShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
using TracedP15ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using TracedP15Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5EpilogueTensorOp,
    TracedP15MmaTileShape,
    TracedP15ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    TracedP5ElementD,
    TracedP5LayoutC*,
    32,
    TracedP5ElementD,
    TracedP5LayoutD*,
    32,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using TracedP15StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename TracedP15Epilogue::SharedStorage))>;
using TracedP15CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5TensorOp,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutA*,
    32,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutB*,
    32,
    float,
    TracedP15MmaTileShape,
    TracedP15ClusterShape,
    TracedP15StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using TracedP15TiledMma = typename TracedP15CollectiveMainloop::TiledMma;
using TracedP15SmemLayoutA = typename TracedP15CollectiveMainloop::SmemLayoutA;
using TracedP15SmemLayoutB = typename TracedP15CollectiveMainloop::SmemLayoutB;
using TracedP15SmemLayoutSFA = typename TracedP15CollectiveMainloop::SmemLayoutSFA;
using TracedP15SmemLayoutSFB = typename TracedP15CollectiveMainloop::SmemLayoutSFB;
using TracedP15SmemCopyAtomA = typename TracedP15CollectiveMainloop::SmemCopyAtomA;
using TracedP15SmemCopyAtomB = typename TracedP15CollectiveMainloop::SmemCopyAtomB;
using TracedP15SmemCopyAtomSFA = typename TracedP15CollectiveMainloop::SmemCopyAtomSFA;
using TracedP15SmemCopyAtomSFB = typename TracedP15CollectiveMainloop::SmemCopyAtomSFB;
using TracedP15TensorStorage = typename TracedP15CollectiveMainloop::TensorStorage;
// Definitive host-side probe facts from artifacts/tmp/trt_p15_runtime_layout_dump.cu:
// - Stages == 6
// - size(TiledMma) == 256 and tile_mnk == (128, 32, 64)
// - cosize(SmemLayoutSFA) == 6144
// - cosize(SmemLayoutSFB) == 3072
// - cosize(tCrSFA) == 16
// - cosize(tCrSFB) == 32
// Definitive copy-view/register probe facts from
// artifacts/tmp/trt_p15_copy_to_reg_dump.cu:
// - tCrA_copy_view.layout = ((_32,_2),_2,_1):((_1,_32),_64,_0), size == 128
// - tCrB_copy_view.layout = ((_32,_1),_4,_1):((_1,_0),_32,_0), size == 128
// - rA_from_copy_view.layout = ((_4,_2),_2,_1):((_1,_4),_8,_0), size == 16
// - rB_from_copy_view.layout = ((_4,_1),_4,_1):((_1,_0),_4,_0), size == 16
// These facts mean traced P15 should be rewritten around real
// tCrA/tCrB/tCrSFA/tCrSFB/tCrC tensors plus cute::gemm(...), not another
// hand-packed AFragment64/BFragment64/CFragment64 variant.
// Follow-up exact-probe fact:
// - the 4-arg cute::gemm(...) call is correct, but a naive builder-level
//   thread_mma.make_fragment_C(partition_C(dense_sC)) accumulator is still
//   not the live P15 FrgTensorC contract
// - the missing helpers (AccumulatorPipelineStageCount, IsOverlappingAccum,
//   partition_accumulator_shape, slice_accumulator) live at the kernel layer,
//   not on the builder CollectiveMainloop type
// - definitive blk-shape probe facts from
//   artifacts/tmp/trt_p15_blk_shape_accum_dump.cu:
//   - atom tile is (128,32,64), but traced P15 profile shape is (256,128,64)
//   - partition_fragment_C(tiled_mma, (256,128)) is the first accumulator
//     shape whose M/N ranks match the traced full-profile kernel intent
//   - definitive slice probe facts from
//     artifacts/tmp/trt_p15_accum_slices_dump.cu:
//     - partition_fragment_C(tiled_mma, (256,128)) has size<2> == 8
//     - each slice is a 16-value accumulator tile with
//       tCrC_slice.layout == ((_2,_2),_4):((_1,_2),_4)
//     - so traced P15 is eight atom-sized accumulator slices per thread, not
//       one larger CFragment64 or the older CFragment64[2][2] model
//   - the dormant exact P15 path therefore needs a full-profile accumulator
//     storage contract, not the older CFragment64[2][2] model
//   - definitive dense operand probe facts from
//     artifacts/tmp/trt_p15_true_dense_operand_coords_dump.cu:
//     - true traced operand copy views are flat dense coords over the full
//       (256,64) A tile and (128,64) B tile
//     - copy_view_a covers row bands 0/64/128/192, not one 128-row subtile
//     - copy_view_b covers row bands 0/32/64/96, not one 32-row atom only
//     - so the exact P15 path must be rewritten around the full-profile
//       operand contract, not the older 2x2 manual subtile decomposition
//   - definitive dense store/accumulator probe facts from
//     artifacts/tmp/trt_p15_true_dense_store_coords_dump.cu:
//     - part_c_dense.layout == ((_2,_2),_4,(_2,_4))
//     - accum_profile.layout == ((_2,_2),_4,(_2,_4))
//     - traced flat output coords span the same 0/64/128/192 row bands and
//       0/32/64/96 column bands as the dense operand probes
//     - so the remaining P15 rewrite should replace the older 2x2
//       subtile accumulator/store model with the full traced 4x4x8 profile
//   - definitive full-profile fragment-shape probe facts from
//     artifacts/tmp/trt_p15_full_profile_fragment_dump.cu:
//     - tCrA.layout == ((_8,_2,_2),_4,_1), sizes = (32,4,1)
//     - tCrB.layout == ((_8,_2),(_2,_4),_1), sizes = (16,8,1)
//     - tCrC.layout == ((_2,_2),_4,(_2,_4)), sizes = (4,4,8)
//     - this matches the failed local-CUTE static assertions exactly:
//       size<1>(A) must match size<1>(C), and size<1>(B) must match size<2>(C)
//     - therefore the next exact P15 rewrite must stop using the current
//       hand-stitched rA_from_copy_view / rB_from_copy_view tensors and build
//       the real full-profile tCrA / tCrB / tCrC contract directly
//   - traced CUTLASS source in sm120_mma_array_tma_blockwise_scaling.hpp
//     shows the operand path is:
//       partition_S(as_position_independent_swizzle_tensor(sA/sB))
//       -> copy into tCrA_copy_view/tCrB_copy_view
//       -> fp4_shift_A/B on the copy views
//       -> cute::gemm(tCrA(_,_,k_block), tCrB(_,_,k_block), tmp_accum)
//   - standalone compile probes showed neither a naive recast path nor a
//     naive make_fragment_A/B + copy path reproduces that contract cleanly
//   - so the next exact P15 rewrite must replace this dormant path's
//     row-packed operand staging with real SmemLayoutA/B-shaped staged shared
//     storage before another activation attempt
//   - maintained all-thread probe/codegen pipeline now exists:
//     proj-2026-04-05-1704/trt_p15_full_thread_contract_dump.cu +
//     proj-2026-04-05-1704/generate_p15_probe_tables.py
//   - generated artifacts live under artifacts/tmp and artifacts/benchmarks;
//     they are the new ground truth for per-thread dense operand/store and
//     scale-stage contracts, instead of relying on thread-0-only notes
//   - standalone backend proof now exists at:
//     testing/backend/p15_swizzled_pipeline_test.cu
//   - that test passes with real SmemLayoutA/B/SFA/SFB staging plus:
//       partition_S(as_position_independent_swizzle_tensor(...))
//       -> copy -> fp4_shift -> make_zip_tensor -> cute::gemm
//   - future exact P15 work should transplant that mechanism into the live
//     kernel and must not preserve the older row-packed a_packed/b_packed path
constexpr int kTracedP15ScaleSmemCosizeA = cute::cosize_v<TracedP15SmemLayoutSFA>;
constexpr int kTracedP15ScaleSmemCosizeB = cute::cosize_v<TracedP15SmemLayoutSFB>;

using NanoP1ElementAct = cutlass::nv_float4_t<nvfp4_cute::ElementAB>;
using NanoP1ElementWeight = cutlass::nv_float4_t<nvfp4_cute::ElementAB>;
using NanoP1ElementAccum = float;
constexpr int NanoP1PipelineStages = 4;
constexpr int NanoP1ThreadsPerCta = 256;
using NanoP1MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using NanoP1ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
// The logical kernel contract is nv_float4_t<e2m1>, but the vendored SM120
// arch atom is specialized on the unpacked e2m1 MMA element type.
using NanoP1MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    nvfp4_cute::ElementAB,
    nvfp4_cute::ElementAB,
    NanoP1ElementAccum,
    nvfp4_cute::ElementSFCompute,
    nvfp4_cute::kScaleVecSize>;
using NanoP1MmaAtom = cute::MMA_Atom<NanoP1MmaOp>;
using NanoP1AtomLayoutMNK = cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>;
using NanoP1ValLayoutMNK =
    cute::Tile<cute::Int<128>, TracedP5PermTileN, cute::Int<64>>;
using NanoP1TiledMma =
    cute::TiledMMA<NanoP1MmaAtom, NanoP1AtomLayoutMNK, NanoP1ValLayoutMNK>;
using NanoP1SmemLayoutAtomA = cute::UMMA::Layout_K_SW64_Atom<typename NanoP1TiledMma::ValTypeA>;
using NanoP1SmemLayoutAtomB = cute::UMMA::Layout_K_SW64_Atom<typename NanoP1TiledMma::ValTypeB>;
using NanoP1SmemLayoutA = decltype(cute::tile_to_shape(
    NanoP1SmemLayoutAtomA{},
    cute::make_shape(
        cute::size<0>(NanoP1MmaTileShape{}) * cute::size<0>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{}),
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using NanoP1SmemLayoutB = decltype(cute::tile_to_shape(
    NanoP1SmemLayoutAtomB{},
    cute::make_shape(
        cute::size<1>(NanoP1MmaTileShape{}) * cute::size<1>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{}),
    cute::Step<cute::_2, cute::_1, cute::_3>{}));
using NanoP1ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<nvfp4_cute::kScaleVecSize>;
using NanoP1SmemLayoutSFA = decltype(NanoP1ScaleConfig::tile_atom_to_shape_SFA(
    cute::make_shape(
        cute::size<0>(NanoP1MmaTileShape{}) * cute::size<0>(NanoP1ClusterShape{}),
        cute::size<1>(NanoP1MmaTileShape{}) * cute::size<1>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{})));
using NanoP1SmemLayoutSFB = decltype(NanoP1ScaleConfig::tile_atom_to_shape_SFB(
    cute::make_shape(
        cute::size<0>(NanoP1MmaTileShape{}) * cute::size<0>(NanoP1ClusterShape{}),
        cute::size<1>(NanoP1MmaTileShape{}) * cute::size<1>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{})));
using NanoP1AccumLayout = decltype(
    cute::partition_fragment_C(
        NanoP1TiledMma{},
        cute::make_shape(
            cute::size<0>(NanoP1MmaTileShape{}),
            cute::size<1>(NanoP1MmaTileShape{})))
        .layout());
using NanoP1SmemAllocA = typename NanoP1TiledMma::ValTypeA;
using NanoP1SmemAllocB = typename NanoP1TiledMma::ValTypeB;
using NanoP1SmemCopyAtomA = cute::Copy_Atom<
    decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_A<
             NanoP1ElementAct,
             NanoP1ElementWeight,
             false>()),
    NanoP1SmemAllocA>;
using NanoP1SmemCopyAtomB = cute::Copy_Atom<
    decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_B<
             NanoP1ElementAct,
             NanoP1ElementWeight,
             false>()),
    NanoP1SmemAllocB>;
using NanoP1SmemCopyAtomSF = cute::Copy_Atom<
    cute::UniversalCopy<nvfp4_cute::ElementSFCompute>,
    nvfp4_cute::ElementSFCompute>;
using NanoP1SmemCopyAtomSFA = NanoP1SmemCopyAtomSF;
using NanoP1SmemCopyAtomSFB = NanoP1SmemCopyAtomSF;
constexpr int kNanoP1SwizzledAElems = cute::size(cute::take<0, 2>(NanoP1SmemLayoutA{}));
constexpr int kNanoP1SwizzledBElems = cute::size(cute::take<0, 2>(NanoP1SmemLayoutB{}));
constexpr int kNanoP1ScaleStageElemsA = cute::cosize(cute::take<0, 2>(NanoP1SmemLayoutSFA{}));
constexpr int kNanoP1ScaleStageElemsB = cute::cosize(cute::take<0, 2>(NanoP1SmemLayoutSFB{}));
constexpr int kNanoP1AccumCoordCount = cute::cosize_v<NanoP1AccumLayout>;
constexpr int kNanoP1NFragments = cute::size<1>(NanoP1AccumLayout{});
constexpr int kNanoP1MFragments = cute::size<2>(NanoP1AccumLayout{});
static_assert(cute::size(NanoP1TiledMma{}) == NanoP1ThreadsPerCta);
static_assert(cute::size(typename NanoP1TiledMma::AtomThrID{}) == 32);
static_assert(cute::cosize_v<NanoP1SmemLayoutA> == 65536);
static_assert(cute::cosize_v<NanoP1SmemLayoutB> == 65536);
static_assert(cute::cosize_v<NanoP1SmemLayoutSFA> == 4096);
static_assert(cute::cosize_v<NanoP1SmemLayoutSFB> == 4096);
static_assert(kNanoP1AccumCoordCount == 64);
static_assert(kNanoP1NFragments == 2);
static_assert(kNanoP1MFragments == 8);
static_assert(kNanoP1ScaleStageElemsA * NanoP1PipelineStages == cute::cosize_v<NanoP1SmemLayoutSFA>);
static_assert(kNanoP1ScaleStageElemsB * NanoP1PipelineStages == cute::cosize_v<NanoP1SmemLayoutSFB>);

enum class NanoP1EpilogueMode {
  kBf16Dense,
  kDirectPack,
};

template <class SFATensor, class AtomT, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto NanoP1ThrfrgSFA(
    SFATensor&& sfatensor,
    cute::TiledMMA<AtomT, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(cute::rank(sfatensor) >= cute::Int<2>{});

  using AtomShape_MNK = typename AtomT::Shape_MNK;
  using AtomLayoutSFA_TV = typename AtomT::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile = cute::make_tile(cute::get<0>(permutation_mnk), cute::get<2>(permutation_mnk));
  auto t_tensor = cute::logical_divide(sfatensor, t_tile);

  auto a_tile = cute::make_tile(
      cute::make_layout(cute::size<0>(AtomShape_MNK{})),
      cute::make_layout(cute::size<2>(AtomShape_MNK{})));
  auto a_tensor = cute::zipped_divide(t_tensor, a_tile);

  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, cute::_);
  auto thr_tile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(cute::size<1>(thr_layout_vmnk)),
          cute::make_layout(cute::size<3>(thr_layout_vmnk))));
  return cute::zipped_divide(tv_tensor, thr_tile);
}

template <class SFBTensor, class AtomT, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto NanoP1ThrfrgSFB(
    SFBTensor&& sfbtensor,
    cute::TiledMMA<AtomT, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(cute::rank(sfbtensor) >= cute::Int<2>{});

  using AtomShape_MNK = typename AtomT::Shape_MNK;
  using AtomLayoutSFB_TV = typename AtomT::Traits::SFBLayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile = cute::make_tile(cute::get<1>(permutation_mnk), cute::get<2>(permutation_mnk));
  auto t_tensor = cute::logical_divide(sfbtensor, t_tile);

  auto a_tile = cute::make_tile(
      cute::make_layout(cute::size<1>(AtomShape_MNK{})),
      cute::make_layout(cute::size<2>(AtomShape_MNK{})));
  auto a_tensor = cute::zipped_divide(t_tensor, a_tile);

  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, cute::_);
  auto thr_tile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(cute::size<2>(thr_layout_vmnk)),
          cute::make_layout(cute::size<3>(thr_layout_vmnk))));
  return cute::zipped_divide(tv_tensor, thr_tile);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto NanoP1GetLayoutSFATV(TiledMma& mma) {
  auto tile_shape_mnk = cute::tile_shape(mma);
  auto ref_a = cute::make_layout(
      cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto thr_tensor = NanoP1ThrfrgSFA(ref_a, mma);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(
              cute::make_shape(cute::size<1>(thr_layout_vmnk), cute::size<2>(thr_layout_vmnk)),
              cute::make_stride(cute::Int<1>{}, cute::Int<0>{})),
          cute::_));
  auto thridx_2_thrid = cute::right_inverse(thr_layout_vmnk);
  return thr_tensor.compose(atile, cute::_).compose(thridx_2_thrid, cute::_);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto NanoP1GetLayoutSFBTV(TiledMma& mma) {
  auto tile_shape_mnk = cute::tile_shape(mma);
  auto ref_b = cute::make_layout(
      cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto thr_tensor = NanoP1ThrfrgSFB(ref_b, mma);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(
              cute::make_shape(cute::size<1>(thr_layout_vmnk), cute::size<2>(thr_layout_vmnk)),
              cute::make_stride(cute::Int<0>{}, cute::Int<1>{})),
          cute::_));
  auto thridx_2_thrid = cute::right_inverse(thr_layout_vmnk);
  return thr_tensor.compose(btile, cute::_).compose(thridx_2_thrid, cute::_);
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto NanoP1PartitionScaleA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = cute::make_tensor(
      static_cast<SFATensor&&>(sfatensor).data(),
      NanoP1ThrfrgSFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk =
      cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
  auto partition_sfa =
      thr_tensor(thr_vmk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
  return cute::make_fragment_like<ValTypeSF>(partition_sfa);
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto NanoP1PartitionScaleB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = cute::make_tensor(
      static_cast<SFBTensor&&>(sfbtensor).data(),
      NanoP1ThrfrgSFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk =
      cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<2>(thr_vmnk), cute::get<3>(thr_vmnk)));
  auto partition_sfb =
      thr_tensor(thr_vnk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
  return cute::make_fragment_like<ValTypeSF>(partition_sfb);
}

struct AFragment64 {
  ARegister regs[4];
  SFRegister scale[1];
};

struct BFragment64 {
  BRegister regs[2];
  SFRegister scale[1];
};

struct CFragment64 {
  CRegister regs[4];
};

template <typename OutputType>
__device__ __forceinline__ void StoreNanoP1CFragmentsRowMajor(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    std::size_t output_stride,
    OutputType* output);

__device__ __forceinline__ void AccumulateNanoP1DirectPackRowMaxAbs(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int row_start,
    int valid_rows,
    int valid_cols,
    float* activation_output_scales);

__device__ __forceinline__ void StoreNanoP1DirectPackCFragments(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    int num_rows_global,
    int inter_size_global,
    std::uint8_t* packed_bytes,
    std::uint8_t* block_scales,
    std::uint8_t* matmul_block_scales,
    float* activation_output_scales,
    float* staging_tile);

constexpr int kTiledCopyCoordCapacityA = 32;
constexpr int kTiledCopyCoordCapacityB = 16;
constexpr int kTiledCopyCoordCapacityScale = 4;
constexpr int kTracedP5ACopyCoordCapacity = 128;
constexpr int kTracedP5BCopyCoordCapacity = 64;
constexpr int kTracedP5CCopyCoordCapacity = 32;
constexpr int kTracedP5MFragments = 4;
constexpr int kTracedP5NFragments = 2;
constexpr int kTracedP13ACopyCoordCapacity = 64;
constexpr int kTracedP13BCopyCoordCapacity = 32;
constexpr int kTracedP13CCopyCoordCapacity = 16;
constexpr int kTracedP13MFragments = 2;
constexpr int kTracedP13NFragments = 2;

CUTE_HOST_DEVICE nvfp4_cute::ElementAB MakeElementAB(std::uint8_t nibble) {
  return nvfp4_cute::ElementAB::bitcast(static_cast<typename nvfp4_cute::ElementAB::Storage>(nibble & 0x0f));
}

CUTE_HOST_DEVICE nvfp4_cute::ElementSFCompute MakeScaleElement(std::uint8_t byte) {
  return nvfp4_cute::ElementSFCompute::bitcast(
      static_cast<typename nvfp4_cute::ElementSFCompute::Storage>(byte));
}

CUTE_HOST_DEVICE constexpr auto GetSingleAtomTiledMma() {
  return SingleAtomTiledMma{};
}

CUTE_HOST_DEVICE constexpr auto GetTracedP5TiledMma() {
  return TracedP5TiledMma{};
}

CUTE_HOST_DEVICE constexpr auto GetTracedP13LayoutSFATV(auto const& mma) {
  auto collective_mainloop = TracedP13CollectiveMainloop{};
  auto mma_copy = mma;
  return collective_mainloop.get_layoutSFA_TV(mma_copy);
}

CUTE_HOST_DEVICE constexpr auto GetTracedP13LayoutSFBTV(auto const& mma) {
  auto collective_mainloop = TracedP13CollectiveMainloop{};
  auto mma_copy = mma;
  return collective_mainloop.get_layoutSFB_TV(mma_copy);
}

CUTE_HOST_DEVICE constexpr auto GetTracedP5LayoutSFATV(auto const& mma) {
  auto collective_mainloop = TracedP5CollectiveMainloop{};
  auto mma_copy = mma;
  return collective_mainloop.get_layoutSFA_TV(mma_copy);
}

CUTE_HOST_DEVICE constexpr auto GetTracedP5LayoutSFBTV(auto const& mma) {
  auto collective_mainloop = TracedP5CollectiveMainloop{};
  auto mma_copy = mma;
  return collective_mainloop.get_layoutSFB_TV(mma_copy);
}

CUTE_HOST_DEVICE constexpr auto GetTracedP15LayoutSFATV(auto const& mma) {
  auto collective_mainloop = TracedP15CollectiveMainloop{};
  auto mma_copy = mma;
  return collective_mainloop.get_layoutSFA_TV(mma_copy);
}

CUTE_HOST_DEVICE constexpr auto GetTracedP15LayoutSFBTV(auto const& mma) {
  auto collective_mainloop = TracedP15CollectiveMainloop{};
  auto mma_copy = mma;
  return collective_mainloop.get_layoutSFB_TV(mma_copy);
}

enum class UnifiedRoutedFp4Profile {
  kP1,
  kP5,
  kP12,
  kP13,
  kP15,
};

enum class P5EpilogueMode {
  kBf16,
  kFp4Direct,
};

template <UnifiedRoutedFp4Profile Profile>
struct UnifiedRoutedFp4Traits;

template <class CoordTensor>
CUTE_HOST_DEVICE void FillPhysicalCoordMapCopyViewLimited(
    CoordTensor const& coord_tensor,
    int limit,
    int* row_coords,
    int* col_coords);

template <>
struct UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP5> {
  using CollectiveMainloop = TracedP5CollectiveMainloop;
  using TiledMma = TracedP5TiledMma;
  using SmemLayoutA = TracedP5SmemLayoutA;
  using SmemLayoutB = TracedP5SmemLayoutB;
  using SmemLayoutSFA = TracedP5SmemLayoutSFA;
  using SmemLayoutSFB = TracedP5SmemLayoutSFB;
  using SmemCopyAtomA = TracedP5SmemCopyAtomA;
  using SmemCopyAtomB = TracedP5SmemCopyAtomB;
  using SmemCopyAtomSFA = TracedP5SmemCopyAtomSFA;
  using SmemCopyAtomSFB = TracedP5SmemCopyAtomSFB;
  using SmemAllocA = typename TiledMma::ValTypeA;
  using SmemAllocB = typename TiledMma::ValTypeB;
  using AccumLayout = TracedP5AccumProfileLayout;
  static constexpr int kOutputTile = 128;
  static constexpr int kTokenRows = cute::tile_size<1>(TiledMma{});
  static constexpr int kMacroTileK = 128;
  static constexpr int kScaleSmemCosizeA = kTracedP5ScaleSmemCosizeA;
  static constexpr int kScaleSmemCosizeB = kTracedP5ScaleSmemCosizeB;
  static constexpr int kAccumProfileCosize = kTracedP5AccumProfileCosize;
  static constexpr int kSwizzledAElems = cute::size(cute::take<0, 2>(SmemLayoutA{}));
  static constexpr int kSwizzledBElems = cute::size(cute::take<0, 2>(SmemLayoutB{}));
  static constexpr bool kEnableP15ScaleTrace = false;
  static constexpr const char* kName = "P5";
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFATV(auto const& mma) {
    return GetTracedP5LayoutSFATV(mma);
  }
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFBTV(auto const& mma) {
    return GetTracedP5LayoutSFBTV(mma);
  }
};

template <>
struct UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP12> {
  using CollectiveMainloop = TracedP5CollectiveMainloop;
  using TiledMma = TracedP5TiledMma;
  using SmemLayoutA = TracedP5SmemLayoutA;
  using SmemLayoutB = TracedP5SmemLayoutB;
  using SmemLayoutSFA = TracedP5SmemLayoutSFA;
  using SmemLayoutSFB = TracedP5SmemLayoutSFB;
  using SmemCopyAtomA = TracedP5SmemCopyAtomA;
  using SmemCopyAtomB = TracedP5SmemCopyAtomB;
  using SmemCopyAtomSFA = TracedP5SmemCopyAtomSFA;
  using SmemCopyAtomSFB = TracedP5SmemCopyAtomSFB;
  using SmemAllocA = typename TiledMma::ValTypeA;
  using SmemAllocB = typename TiledMma::ValTypeB;
  using AccumLayout = TracedP5AccumProfileLayout;
  static constexpr int kOutputTile = 128;
  static constexpr int kTokenRows = cute::tile_size<1>(TiledMma{});
  static constexpr int kMacroTileK = 128;
  static constexpr int kScaleSmemCosizeA = kTracedP5ScaleSmemCosizeA;
  static constexpr int kScaleSmemCosizeB = kTracedP5ScaleSmemCosizeB;
  static constexpr int kAccumProfileCosize = kTracedP5AccumProfileCosize;
  static constexpr int kSwizzledAElems = cute::size(cute::take<0, 2>(SmemLayoutA{}));
  static constexpr int kSwizzledBElems = cute::size(cute::take<0, 2>(SmemLayoutB{}));
  static constexpr bool kEnableP15ScaleTrace = false;
  static constexpr const char* kName = "P12";
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFATV(auto const& mma) {
    return GetTracedP5LayoutSFATV(mma);
  }
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFBTV(auto const& mma) {
    return GetTracedP5LayoutSFBTV(mma);
  }
};

template <>
struct UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP13> {
  using CollectiveMainloop = TracedP13CollectiveMainloop;
  using TiledMma = TracedP13TiledMma;
  using SmemLayoutA = TracedP13SmemLayoutA;
  using SmemLayoutB = TracedP13SmemLayoutB;
  using SmemLayoutSFA = TracedP13SmemLayoutSFA;
  using SmemLayoutSFB = TracedP13SmemLayoutSFB;
  using SmemCopyAtomA = TracedP13SmemCopyAtomA;
  using SmemCopyAtomB = TracedP13SmemCopyAtomB;
  using SmemCopyAtomSFA = TracedP13SmemCopyAtomSFA;
  using SmemCopyAtomSFB = TracedP13SmemCopyAtomSFB;
  using SmemAllocA = typename TiledMma::ValTypeA;
  using SmemAllocB = typename TiledMma::ValTypeB;
  using AccumLayout = decltype(
      cute::partition_fragment_C(
          TiledMma{},
          cute::make_shape(cute::Int<128>{}, cute::Int<128>{}))
          .layout());
  static constexpr int kOutputTile = 128;
  static constexpr int kTokenRows = cute::tile_size<1>(TiledMma{});
  static constexpr int kMacroTileK = 64;
  static constexpr int kScaleSmemCosizeA = kTracedP13ScaleSmemCosizeA;
  static constexpr int kScaleSmemCosizeB = kTracedP13ScaleSmemCosizeB;
  static constexpr int kAccumProfileCosize = cute::cosize_v<AccumLayout>;
  static constexpr int kSwizzledAElems = cute::size(cute::take<0, 2>(SmemLayoutA{}));
  static constexpr int kSwizzledBElems = cute::size(cute::take<0, 2>(SmemLayoutB{}));
  static constexpr bool kEnableP15ScaleTrace = false;
  static constexpr const char* kName = "P13";
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFATV(auto const& mma) {
    return GetTracedP13LayoutSFATV(mma);
  }
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFBTV(auto const& mma) {
    return GetTracedP13LayoutSFBTV(mma);
  }
};

template <>
struct UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP15> {
  using CollectiveMainloop = TracedP15CollectiveMainloop;
  using TiledMma = TracedP15TiledMma;
  using SmemLayoutA = TracedP15SmemLayoutA;
  using SmemLayoutB = TracedP15SmemLayoutB;
  using SmemLayoutSFA = TracedP15SmemLayoutSFA;
  using SmemLayoutSFB = TracedP15SmemLayoutSFB;
  using SmemCopyAtomA = TracedP15SmemCopyAtomA;
  using SmemCopyAtomB = TracedP15SmemCopyAtomB;
  using SmemCopyAtomSFA = TracedP15SmemCopyAtomSFA;
  using SmemCopyAtomSFB = TracedP15SmemCopyAtomSFB;
  using SmemAllocA = typename TiledMma::ValTypeA;
  using SmemAllocB = typename TiledMma::ValTypeB;
  using AccumLayout = decltype(
      cute::partition_fragment_C(
          TiledMma{},
          cute::make_shape(cute::Int<256>{}, cute::Int<128>{}))
          .layout());
  static constexpr int kOutputTile = 256;
  static constexpr int kTokenRows = cute::tile_size<1>(TiledMma{});
  static constexpr int kMacroTileK = 64;
  static constexpr int kScaleSmemCosizeA = kTracedP15ScaleSmemCosizeA;
  static constexpr int kScaleSmemCosizeB = kTracedP15ScaleSmemCosizeB;
  static constexpr int kAccumProfileCosize = cute::cosize_v<AccumLayout>;
  static constexpr int kSwizzledAElems = cute::size(cute::take<0, 2>(SmemLayoutA{}));
  static constexpr int kSwizzledBElems = cute::size(cute::take<0, 2>(SmemLayoutB{}));
  static constexpr bool kEnableP15ScaleTrace = true;
  static constexpr const char* kName = "P15";
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFATV(auto const& mma) {
    return GetTracedP15LayoutSFATV(mma);
  }
  CUTE_HOST_DEVICE static constexpr auto GetLayoutSFBTV(auto const& mma) {
    return GetTracedP15LayoutSFBTV(mma);
  }
};

__device__ __forceinline__ void StoreUnifiedRoutedFp4DirectPack(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    const float* per_row_tensor_scales,
    float weight_tensor_scale,
    std::uint8_t* fp4_packed_data,
    std::uint8_t* fp4_block_scales,
    std::uint8_t* fp4_matmul_block_scales,
    float* fp4_activation_output_scale,
    std::size_t fp4_cols,
    std::size_t fp4_padded_blocks_per_row,
    Nvfp4ScaleLayout fp4_scale_layout) {
  if (fp4_cols == 0 ||
      fp4_cols != output_rows_per_expert ||
      (fp4_cols % fused_decode::kNvfp4BlockWidth) != 0) {
    return;
  }

  auto accum_tensor = cute::make_tensor(
      reinterpret_cast<CRegister*>(const_cast<CRegister*>(accum_storage)),
      TracedP5AccumProfileLayout{});

  const int warp_id = thread_idx / 32;
  const int lane_id = thread_idx & 31;
  const int g_m = warp_id & 3;
  const int g_n = warp_id / 4;
  const int q = lane_id / 4;
  const int r = lane_id & 3;
  const unsigned int subgroup_mask = 0x11111111u << r;

  const int n_coords[4] = {
      16 * g_n + 2 * r,
      16 * g_n + 2 * r + 1,
      16 * g_n + 8 + 2 * r,
      16 * g_n + 9 + 2 * r};

  const std::size_t blocks_per_row = fp4_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_bytes = fp4_cols / 2u;
  const std::size_t output_limit =
      static_cast<std::size_t>(output_row_base + output_rows_this_tile);
  constexpr int kLinearToBlockHalf[16] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  constexpr int kLinearToTokenGroup[16] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3};
  constexpr int kLinearToMHalf[16] = {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};
  constexpr int kBlockElemMh0[8] = {0, 1, 8, 9, 4, 5, 12, 13};
  constexpr int kBlockElemMh1[8] = {2, 3, 10, 11, 6, 7, 14, 15};

  float activated[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int block_half = kLinearToBlockHalf[i];
    const int token_group = kLinearToTokenGroup[i];
    const int m_half = kLinearToMHalf[i];
    const int m_coord =
        (block_half == 0 ? 0 : 64) + 16 * g_m + (m_half == 0 ? q : 8 + q);
    const int n_coord = n_coords[token_group];
    const std::size_t output_col =
        static_cast<std::size_t>(output_row_base + m_coord);

    activated[i] = 0.0f;
    if (n_coord < valid_rows &&
        output_col < output_limit &&
        output_col < fp4_cols) {
      const std::size_t input_row = row_start + static_cast<std::size_t>(n_coord);
      const float row_alpha =
          per_row_tensor_scales != nullptr
              ? per_row_tensor_scales[input_row] * weight_tensor_scale
              : alpha;
      const float scaled = accum_tensor(i) * row_alpha;
      // Native direct path: keep accumulation in FP32 through activation and
      // NVFP4 repack instead of recreating the legacy BF16 boundary.
      activated[i] = fused_decode::Relu2(scaled);
    }
  }

#pragma unroll
  for (int b = 0; b < 8; ++b) {
    const int block_half = b / 4;
    const int token_group = b & 3;
    const int idx_mh0 = kBlockElemMh0[b];
    const int idx_mh1 = kBlockElemMh1[b];
    const float val_mh0 = activated[idx_mh0];
    const float val_mh1 = activated[idx_mh1];

    const int n_coord = n_coords[token_group];
    const int block_start_m = (block_half == 0 ? 0 : 64) + 16 * g_m;
    const std::size_t output_col_start =
        static_cast<std::size_t>(output_row_base + block_start_m);

    const bool block_valid =
        (output_col_start + static_cast<std::size_t>(fused_decode::kNvfp4BlockWidth) <= fp4_cols) &&
        (output_col_start < output_limit);
    if (!block_valid) {
      continue;
    }

    const std::size_t input_row = row_start + static_cast<std::size_t>(n_coord);
    const std::size_t block_index =
        output_col_start / fused_decode::kNvfp4BlockWidth;
    const std::size_t scale_index = input_row * blocks_per_row + block_index;
    const std::size_t matmul_scale_index = ExecutionScaleOffset(
        input_row,
        block_index,
        fp4_padded_blocks_per_row,
        fp4_scale_layout);
    const std::size_t packed_offset_base =
        input_row * packed_row_bytes + (output_col_start / 2u);

    if (n_coord >= valid_rows) {
      if (q == 0) {
        fp4_activation_output_scale[scale_index] = 0.0f;
        fp4_block_scales[scale_index] = 0u;
        fp4_matmul_block_scales[matmul_scale_index] = 0u;
      }
      if ((q & 1) == 0) {
        const int k = q / 2;
        fp4_packed_data[packed_offset_base + static_cast<std::size_t>(k)] = 0u;
        fp4_packed_data[packed_offset_base + 4u + static_cast<std::size_t>(k)] = 0u;
      }
      continue;
    }

    float local_max = fmaxf(val_mh0, val_mh1);
    float partner = __shfl_xor_sync(subgroup_mask, local_max, 4);
    local_max = fmaxf(local_max, partner);
    partner = __shfl_xor_sync(subgroup_mask, local_max, 8);
    local_max = fmaxf(local_max, partner);
    partner = __shfl_xor_sync(subgroup_mask, local_max, 16);
    local_max = fmaxf(local_max, partner);

    float block_scale = 1.0f;
    if (local_max > 0.0f) {
      block_scale = fused_decode::ClampNvfp4Scale(
          local_max / fused_decode::kNvfp4Fp4MaxFinite);
    }
    const std::uint8_t encoded_scale =
        fused_decode::EncodeFp8Scale(block_scale);

    const std::uint8_t nibble_mh0 =
        fused_decode::EncodeFp4(val_mh0 / block_scale);
    const std::uint8_t nibble_mh1 =
        fused_decode::EncodeFp4(val_mh1 / block_scale);
    const std::uint8_t partner_nibble_mh0 = static_cast<std::uint8_t>(
        __shfl_down_sync(subgroup_mask, static_cast<int>(nibble_mh0), 4));
    const std::uint8_t partner_nibble_mh1 = static_cast<std::uint8_t>(
        __shfl_down_sync(subgroup_mask, static_cast<int>(nibble_mh1), 4));

    if (q == 0) {
      fp4_activation_output_scale[scale_index] = block_scale;
      fp4_block_scales[scale_index] = encoded_scale;
      fp4_matmul_block_scales[matmul_scale_index] = encoded_scale;
    }

    if ((q & 1) == 0) {
      const int k = q / 2;
      fp4_packed_data[packed_offset_base + static_cast<std::size_t>(k)] =
          static_cast<std::uint8_t>(
              (nibble_mh0 & 0x0Fu) | ((partner_nibble_mh0 & 0x0Fu) << 4u));
      fp4_packed_data[packed_offset_base + 4u + static_cast<std::size_t>(k)] =
          static_cast<std::uint8_t>(
              (nibble_mh1 & 0x0Fu) | ((partner_nibble_mh1 & 0x0Fu) << 4u));
    }
  }
}

template <UnifiedRoutedFp4Profile Profile, P5EpilogueMode P5Mode = P5EpilogueMode::kBf16, typename OutputType>
__device__ __forceinline__ void StoreUnifiedRoutedFp4Output(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output,
    const float* per_row_tensor_scales = nullptr,
    float weight_tensor_scale = 1.0f,
    std::uint8_t* fp4_packed_data = nullptr,
    std::uint8_t* fp4_block_scales = nullptr,
    std::uint8_t* fp4_matmul_block_scales = nullptr,
    float* fp4_activation_output_scale = nullptr,
    std::size_t fp4_cols = 0,
    std::size_t fp4_padded_blocks_per_row = 0,
    Nvfp4ScaleLayout fp4_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4) {
  if constexpr (Profile == UnifiedRoutedFp4Profile::kP5) {
    if constexpr (P5Mode == P5EpilogueMode::kFp4Direct) {
      (void)output;
      StoreUnifiedRoutedFp4DirectPack(
          alpha,
          accum_storage,
          thread_idx,
          output_row_base,
          valid_rows,
          output_rows_this_tile,
          row_start,
          output_rows_per_expert,
          per_row_tensor_scales,
          weight_tensor_scale,
          fp4_packed_data,
          fp4_block_scales,
          fp4_matmul_block_scales,
          fp4_activation_output_scale,
          fp4_cols,
          fp4_padded_blocks_per_row,
          fp4_scale_layout);
      return;
    }
    (void)fp4_packed_data;
    (void)fp4_block_scales;
    (void)fp4_matmul_block_scales;
    (void)fp4_activation_output_scale;
    (void)fp4_cols;
    (void)fp4_padded_blocks_per_row;
    (void)fp4_scale_layout;
    int row_coords[kTracedP5CCopyCoordCapacity];
    int col_coords[kTracedP5CCopyCoordCapacity];
    auto mma = TracedP5TiledMma{};
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto ref_c = cute::make_identity_tensor(
        cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
    auto part_c = thr_mma.partition_C(ref_c);
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<CRegister*>(const_cast<CRegister*>(accum_storage)),
        TracedP5AccumProfileLayout{});
    FillPhysicalCoordMapCopyViewLimited(
        part_c,
        kTracedP5CCopyCoordCapacity,
        row_coords,
        col_coords);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < kTracedP5MFragments; ++m_fragment) {
#pragma unroll
        for (int n_fragment = 0; n_fragment < kTracedP5NFragments; ++n_fragment) {
          const int physical = reg * 8 + m_fragment * 2 + n_fragment;
          const int row = row_coords[physical];
          const int token = col_coords[physical];
          if (row >= output_rows_this_tile || token >= valid_rows) {
            continue;
          }
          const std::size_t input_row = row_start + static_cast<std::size_t>(token);
          const std::size_t output_col = static_cast<std::size_t>(output_row_base + row);
          const float row_alpha =
              per_row_tensor_scales != nullptr
                  ? per_row_tensor_scales[input_row] * weight_tensor_scale
                  : alpha;
          const float value = accum_tensor(reg, m_fragment, n_fragment) * row_alpha;
          if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
            output[input_row * output_rows_per_expert + output_col] = __float2bfloat16(value);
          } else {
            output[input_row * output_rows_per_expert + output_col] = value;
          }
        }
      }
    }
  } else if constexpr (Profile == UnifiedRoutedFp4Profile::kP13) {
    (void)fp4_packed_data;
    (void)fp4_block_scales;
    (void)fp4_matmul_block_scales;
    (void)fp4_activation_output_scale;
    (void)fp4_cols;
    (void)fp4_padded_blocks_per_row;
    (void)fp4_scale_layout;
    int row_coords[kTracedP13CCopyCoordCapacity];
    int col_coords[kTracedP13CCopyCoordCapacity];
    auto mma = TracedP13TiledMma{};
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto ref_c = cute::make_identity_tensor(
        cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
    auto part_c = thr_mma.partition_C(ref_c);
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<CRegister*>(const_cast<CRegister*>(accum_storage)),
        typename UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP13>::AccumLayout{});
    FillPhysicalCoordMapCopyViewLimited(
        part_c,
        kTracedP13CCopyCoordCapacity,
        row_coords,
        col_coords);
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < kTracedP13MFragments; ++m_fragment) {
#pragma unroll
        for (int n_fragment = 0; n_fragment < kTracedP13NFragments; ++n_fragment) {
          const int physical = n_fragment * 8 + m_fragment * 4 + reg;
          const int output_col_offset = row_coords[physical];
          const int token_row = col_coords[physical];
          if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
            continue;
          }
          const std::size_t input_row = row_start + static_cast<std::size_t>(token_row);
          const std::size_t output_col =
              static_cast<std::size_t>(output_row_base + output_col_offset);
          const float row_alpha =
              per_row_tensor_scales != nullptr
                  ? per_row_tensor_scales[input_row] * weight_tensor_scale
                  : alpha;
          const float value = accum_tensor(reg, m_fragment, n_fragment) * row_alpha;
          if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
            output[input_row * output_rows_per_expert + output_col] = __float2bfloat16(value);
          } else {
            output[input_row * output_rows_per_expert + output_col] = value;
          }
        }
      }
    }
  } else {
    (void)fp4_packed_data;
    (void)fp4_block_scales;
    (void)fp4_matmul_block_scales;
    (void)fp4_activation_output_scale;
    (void)fp4_cols;
    (void)fp4_padded_blocks_per_row;
    (void)fp4_scale_layout;
    using Traits = UnifiedRoutedFp4Traits<Profile>;
    using TiledMma = typename Traits::TiledMma;
    auto tiled_mma = TiledMma{};
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<CRegister*>(const_cast<CRegister*>(accum_storage)),
        typename Traits::AccumLayout{});
    auto dense_c = cute::make_identity_tensor(
        cute::make_shape(cute::Int<Traits::kOutputTile>{}, cute::Int<Traits::kTokenRows>{}));
    auto part_c = thread_mma.partition_C(dense_c);
    for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
      auto coord = part_c(i);
      const int output_col_offset = static_cast<int>(cute::get<0>(coord));
      const int token_row = static_cast<int>(cute::get<1>(coord));
      if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
        continue;
      }
      const std::size_t input_row = row_start + static_cast<std::size_t>(token_row);
      const std::size_t output_col = static_cast<std::size_t>(output_row_base + output_col_offset);
      const float row_alpha =
          per_row_tensor_scales != nullptr
              ? per_row_tensor_scales[input_row] * weight_tensor_scale
              : alpha;
      if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
        output[input_row * output_rows_per_expert + output_col] =
            __float2bfloat16(accum_tensor(i) * row_alpha);
      } else {
        output[input_row * output_rows_per_expert + output_col] = accum_tensor(i) * row_alpha;
      }
    }
  }
}

template <class Coord>
CUTE_HOST_DEVICE constexpr void FlattenCoordPair(Coord const& coord, int& row, int& col, int& count) {
  if (count >= 2) {
    return;
  }
  if constexpr (cute::is_tuple<Coord>::value) {
    constexpr std::size_t kTupleSize = std::tuple_size_v<std::remove_cvref_t<Coord>>;
    FlattenCoordPair(cute::get<0>(coord), row, col, count);
    if constexpr (kTupleSize > 1) {
      if (count < 2) {
        FlattenCoordPair(cute::get<1>(coord), row, col, count);
      }
    }
    if constexpr (kTupleSize > 2) {
      if (count < 2) {
        FlattenCoordPair(cute::get<2>(coord), row, col, count);
      }
    }
  } else {
    if (count == 0) {
      row = static_cast<int>(coord);
    } else {
      col = static_cast<int>(coord);
    }
    ++count;
  }
}

template <class Coord>
CUTE_HOST_DEVICE constexpr int CoordGet0(Coord const& coord) {
  int row = 0;
  int col = 0;
  int count = 0;
  FlattenCoordPair(coord, row, col, count);
  (void)col;
  return row;
}

template <class Coord>
CUTE_HOST_DEVICE constexpr int CoordGet1(Coord const& coord) {
  int row = 0;
  int col = 0;
  int count = 0;
  FlattenCoordPair(coord, row, col, count);
  (void)row;
  return count >= 2 ? col : 0;
}

template <class SFATensor, class AtomT, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto ThrfrgSFA(
    SFATensor&& sfatensor,
    cute::TiledMMA<AtomT, TiledThr, TiledPerm>& mma) {
  auto permutation_mnk = TiledPerm{};
  auto t_tile = cute::make_tile(cute::get<0>(permutation_mnk), cute::_1{});
  auto tiled_sfa = cute::logical_divide(sfatensor, t_tile);

  using AtomShape_MNK = typename AtomT::Shape_MNK;
  auto atom_tile =
      cute::make_tile(cute::make_layout(cute::size<0>(AtomShape_MNK{})), cute::make_layout(cute::_1{}));
  auto tiled_atom_sfa = cute::zipped_divide(tiled_sfa, atom_tile);
  using AtomLayoutSFA_TV = typename cute::MMA_Traits<nvfp4_cute::MmaOp>::SFALayout;
  auto tv_atom_sfa = tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, cute::_);

  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto thr_tile = cute::make_tile(
      cute::_,
      cute::make_tile(cute::make_layout(cute::size<1>(thr_layout_vmnk)),
          cute::make_layout(cute::size<3>(thr_layout_vmnk))));
  return cute::zipped_divide(tv_atom_sfa, thr_tile);
}

template <class SFBTensor, class AtomT, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto ThrfrgSFB(
    SFBTensor&& sfbtensor,
    cute::TiledMMA<AtomT, TiledThr, TiledPerm>& mma) {
  auto permutation_mnk = TiledPerm{};
  auto t_tile = cute::make_tile(cute::get<1>(permutation_mnk), cute::_1{});
  auto tiled_sfb = cute::logical_divide(sfbtensor, t_tile);

  using AtomShape_MNK = typename AtomT::Shape_MNK;
  auto atom_tile =
      cute::make_tile(cute::make_layout(cute::size<1>(AtomShape_MNK{})), cute::make_layout(cute::_1{}));
  auto tiled_atom_sfb = cute::zipped_divide(tiled_sfb, atom_tile);
  using AtomLayoutSFB_TV = typename cute::MMA_Traits<nvfp4_cute::MmaOp>::SFBLayout;
  auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, cute::_);

  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto thr_tile = cute::make_tile(
      cute::_,
      cute::make_tile(cute::make_layout(cute::size<2>(thr_layout_vmnk)),
          cute::make_layout(cute::size<3>(thr_layout_vmnk))));
  return cute::zipped_divide(tv_atom_sfb, thr_tile);
}

CUTE_HOST_DEVICE constexpr auto GetLayoutATV(auto const& mma) {
  return mma.get_layoutA_TV();
}

CUTE_HOST_DEVICE constexpr auto GetLayoutBTV(auto const& mma) {
  return mma.get_layoutB_TV();
}

CUTE_HOST_DEVICE constexpr auto GetLayoutCTV(auto const& mma) {
  return mma.get_layoutC_TV();
}

CUTE_HOST_DEVICE constexpr auto GetLayoutSFATV(auto const& mma) {
  auto mma_copy = mma;
  auto ref_a = cute::make_layout(cute::make_shape(cute::size<0>(typename Atom::Shape_MNK{}), cute::_1{}));
  auto thr_tensor = ThrfrgSFA(ref_a, mma_copy);
  auto thr_layout_vmnk = mma_copy.get_thr_layout_vmnk();
  auto atile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(cute::make_shape(cute::size<1>(thr_layout_vmnk), cute::size<2>(thr_layout_vmnk)),
              cute::make_stride(cute::Int<1>{}, cute::Int<0>{})),
          cute::_));
  auto tv_sfa = thr_tensor.compose(atile, cute::_);
  auto thridx_2_thrid = cute::right_inverse(thr_layout_vmnk);
  return tv_sfa.compose(thridx_2_thrid, cute::_);
}

CUTE_HOST_DEVICE constexpr auto GetLayoutSFBTV(auto const& mma) {
  auto mma_copy = mma;
  auto ref_b = cute::make_layout(cute::make_shape(cute::size<1>(typename Atom::Shape_MNK{}), cute::_1{}));
  auto thr_tensor = ThrfrgSFB(ref_b, mma_copy);
  auto thr_layout_vmnk = mma_copy.get_thr_layout_vmnk();
  auto btile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(cute::make_shape(cute::size<1>(thr_layout_vmnk), cute::size<2>(thr_layout_vmnk)),
              cute::make_stride(cute::Int<0>{}, cute::Int<1>{})),
          cute::_));
  auto tv_sfb = thr_tensor.compose(btile, cute::_);
  auto thridx_2_thrid = cute::right_inverse(thr_layout_vmnk);
  return tv_sfb.compose(thridx_2_thrid, cute::_);
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto PartitionScaleA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  auto thr_tensor =
      cute::make_tensor(static_cast<SFATensor&&>(sfatensor).data(), ThrfrgSFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto PartitionScaleB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  auto thr_tensor =
      cute::make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), ThrfrgSFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
}

template <class CopyOperation, class CopyInternalType>
struct LocalCopyAtom : cute::Copy_Traits<CopyOperation> {
  using Traits = cute::Copy_Traits<CopyOperation>;
  using ThrID = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;
  using ValType = CopyInternalType;
  using ValLayoutSrc = decltype(cute::recast_layout<cute::uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(cute::recast_layout<cute::uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(cute::recast_layout<cute::uint1_t, ValType>(BitLayoutRef{}));
  static constexpr int NumValSrc = cute::size<1>(ValLayoutSrc{});
  static constexpr int NumValDst = cute::size<1>(ValLayoutDst{});
};

template <class TiledCopy, class ThrIdx>
struct LocalThrCopy;

template <class CopyAtom, class LayoutCopyTV, class ShapeTilerMN>
struct LocalTiledCopy : CopyAtom {
  using AtomLayoutSrc = typename CopyAtom::ValLayoutSrc;
  using AtomLayoutDst = typename CopyAtom::ValLayoutDst;
  using AtomLayoutRef = typename CopyAtom::ValLayoutRef;
  using AtomNumThr = decltype(cute::size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(cute::size<1>(AtomLayoutRef{}));
  using TilerMN = ShapeTilerMN;
  using TiledLayoutTV = LayoutCopyTV;
  using TiledNumThr = decltype(cute::size<0>(TiledLayoutTV{}));
  using TiledNumVal = decltype(cute::size<1>(TiledLayoutTV{}));

  template <class Tensor, class Ref2TrgLayout>
  CUTE_HOST_DEVICE constexpr static auto TileToThrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg) {
    auto atom_layout_tv = cute::zipped_divide(TiledLayoutTV{}, cute::make_shape(AtomNumThr{}, AtomNumVal{}));
    auto trg_layout_tv = atom_layout_tv.compose(ref2trg, cute::_);
    auto thrval2mn = cute::coalesce(cute::zip(trg_layout_tv), cute::Shape<cute::_1, cute::Shape<cute::_1, cute::_1>>{});
    auto tv_tensor = tensor.compose(thrval2mn, cute::_);
    return tv_tensor(cute::make_coord(cute::_, cute::_), cute::_);
  }

  template <class DTensor>
  CUTE_HOST_DEVICE constexpr static auto Retile(DTensor&& dtensor) {
    constexpr int R = std::remove_cvref_t<DTensor>::rank;
    auto v = cute::size<0>(dtensor);
    auto frg_layout_mn =
        cute::upcast<TiledNumThr{} * v>(cute::right_inverse(TiledLayoutTV{}).with_shape(cute::shape(TilerMN{})));
    auto frg_layout_v = cute::zipped_divide(
        cute::logical_product(cute::make_layout(v), cute::right_inverse(frg_layout_mn)),
        cute::make_layout(AtomNumVal{}));
    auto t_tensor = cute::zipped_divide(dtensor, cute::prepend(cute::product_each(cute::shape(frg_layout_mn)), v));
    auto v_tensor = t_tensor.compose(frg_layout_v, cute::_);
    return v_tensor(cute::_, cute::append<R>(cute::Int<0>{}, cute::_));
  }

  template <class ThrIdx_, __CUTE_REQUIRES(cute::is_integral<ThrIdx_>::value)>
  CUTE_HOST_DEVICE constexpr static auto get_thread_slice(ThrIdx_ const& thr_idx) {
    return LocalThrCopy<LocalTiledCopy, ThrIdx_>{thr_idx};
  }
};

template <class TiledCopy, class ThrIdx>
struct LocalThrCopy {
  ThrIdx thr_idx_;

  template <class DTensor>
  CUTE_HOST_DEVICE auto retile_D(DTensor&& dtensor) const {
    auto thr_tensor =
        cute::make_tensor(static_cast<DTensor&&>(dtensor).data(), TiledCopy::Retile(dtensor.layout()));
    return thr_tensor;
  }
};

template <class CopyOperation, class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeLocalTiledCopyA(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<CopyOperation, nvfp4_cute::ElementAB>;
  using LayoutTV = decltype(GetLayoutATV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class CopyOperation, class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeLocalTiledCopyB(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<CopyOperation, nvfp4_cute::ElementAB>;
  using LayoutTV = decltype(GetLayoutBTV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<1>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeLocalTiledCopySFA(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetLayoutSFATV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeLocalTiledCopySFB(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetLayoutSFBTV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<1>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

CUTE_HOST_DEVICE void FillSingleAtomCLayoutCoords(
    int thread_idx,
    int token_rows[4],
    int output_rows[4]) {
  auto mma = SingleAtomTiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_c = cute::make_identity_tensor(
      cute::make_shape(cute::Int<16>{}, cute::Int<8>{}));
  auto part_c = thr_mma.partition_C(ref_c);
  static_assert(
      decltype(cute::size(part_c))::value == 4,
      "Single-atom P1 debug expects four C outputs per lane");
  for (int reg = 0; reg < 4; ++reg) {
    auto coord = part_c(reg);
    token_rows[reg] = static_cast<int>(cute::get<0>(coord));
    output_rows[reg] = static_cast<int>(cute::get<1>(coord));
  }
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeTracedP13TiledCopySFA(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetTracedP13LayoutSFATV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeTracedP13TiledCopySFB(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetTracedP13LayoutSFBTV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<1>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeTracedP5TiledCopySFA(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetTracedP5LayoutSFATV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeTracedP5TiledCopySFB(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetTracedP5LayoutSFBTV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<1>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeTracedP15TiledCopySFA(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetTracedP15LayoutSFATV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto MakeTracedP15TiledCopySFB(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<nvfp4_cute::ElementSFCompute>, nvfp4_cute::ElementSFCompute>;
  using LayoutTV = decltype(GetTracedP15LayoutSFBTV(mma));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<1>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class CoordTensor, class Layout>
CUTE_HOST_DEVICE constexpr void FillPhysicalCoordMapA(
    CoordTensor const& coord_tensor,
    Layout const& fragment_layout,
    int* row_coords,
    int* col_coords) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        auto logical = cute::make_coord(cute::make_coord(i, j, k), cute::Int<0>{}, cute::Int<0>{});
        auto coord = coord_tensor(logical);
        const int physical = static_cast<int>(fragment_layout(logical));
        row_coords[physical] = CoordGet0(coord);
        col_coords[physical] = CoordGet1(coord);
      }
    }
  }
}

template <class CoordTensor>
CUTE_HOST_DEVICE constexpr void FillPhysicalCoordMapCopyView(
    CoordTensor const& coord_tensor,
    int* row_coords,
    int* col_coords) {
  int physical = 0;
#pragma unroll
  for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
#pragma unroll
    for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
#pragma unroll
      for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
        auto logical = cute::make_coord(i, j, k);
        auto coord = coord_tensor(logical);
        row_coords[physical] = CoordGet0(coord);
        col_coords[physical] = CoordGet1(coord);
        ++physical;
      }
    }
  }
}

template <class CoordTensor>
CUTE_HOST_DEVICE void FillPhysicalCoordMapCopyViewLimited(
    CoordTensor const& coord_tensor,
    int limit,
    int* row_coords,
    int* col_coords) {
  int physical = 0;
  if constexpr (std::remove_cvref_t<CoordTensor>::rank == 2) {
    for (int i = 0; i < cute::size<0>(coord_tensor) && physical < limit; ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor) && physical < limit; ++j) {
        auto logical = cute::make_coord(i, j);
        auto coord = coord_tensor(logical);
        row_coords[physical] = CoordGet0(coord);
        col_coords[physical] = CoordGet1(coord);
        ++physical;
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 3) {
    for (int i = 0; i < cute::size<0>(coord_tensor) && physical < limit; ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor) && physical < limit; ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor) && physical < limit; ++k) {
          auto logical = cute::make_coord(i, j, k);
          auto coord = coord_tensor(logical);
          row_coords[physical] = CoordGet0(coord);
          col_coords[physical] = CoordGet1(coord);
          ++physical;
        }
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 4) {
    for (int i = 0; i < cute::size<0>(coord_tensor) && physical < limit; ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor) && physical < limit; ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor) && physical < limit; ++k) {
          for (int l = 0; l < cute::size<3>(coord_tensor) && physical < limit; ++l) {
            auto logical = cute::make_coord(i, j, k, l);
            auto coord = coord_tensor(logical);
            row_coords[physical] = CoordGet0(coord);
            col_coords[physical] = CoordGet1(coord);
            ++physical;
          }
        }
      }
    }
  } else {
    static_assert(
        std::remove_cvref_t<CoordTensor>::rank <= 4,
        "FillPhysicalCoordMapCopyViewLimited only supports copy-view tensors up to rank 4");
  }
}

template <class CoordTensor, class Layout>
CUTE_HOST_DEVICE constexpr void FillPhysicalCoordMapB(
    CoordTensor const& coord_tensor,
    Layout const& fragment_layout,
    int* row_coords,
    int* col_coords) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      auto logical = cute::make_coord(cute::make_coord(i, j), cute::Int<0>{}, cute::Int<0>{});
      auto coord = coord_tensor(logical);
      const int physical = static_cast<int>(fragment_layout(logical));
      row_coords[physical] = CoordGet0(coord);
      col_coords[physical] = CoordGet1(coord);
    }
  }
}

CUTE_HOST_DEVICE constexpr std::uint8_t LoadScaleByte(std::uint32_t packed_scale_word, int byte_index) {
  return static_cast<std::uint8_t>((packed_scale_word >> (byte_index * 8)) & 0xFFu);
}

template <int kRowsPerTile>
struct PackedTile64 {
  std::uint8_t* packed_rows;
  std::uint32_t* scale_words;
};

template <int kRowsPerTile>
__device__ __forceinline__ PackedTile64<kRowsPerTile> MakePackedTile64(
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words);

template <int kRowsPerTile>
__device__ __forceinline__ void ZeroRow(
    const PackedTile64<kRowsPerTile>& tile,
    int row);

template <int kRowsPerTile>
__device__ __forceinline__ void CopyWeightRow64(
    const PackedTile64<kRowsPerTile>& tile,
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* matmul_scales,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    int row);

template <int kRowsPerTile>
__device__ __forceinline__ void CopyActivationRow64(
    const PackedTile64<kRowsPerTile>& tile,
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* block_scales,
    std::size_t block_base,
    std::size_t blocks_per_row,
    int row);

template <class ScaleTensor>
__device__ __forceinline__ void ZeroTracedP5ScaleRow(
    ScaleTensor& scale_tensor,
    int row);

template <class ScaleTensor>
__device__ __forceinline__ void StoreTracedP5ScaleWordK64(
    ScaleTensor& scale_tensor,
    std::uint32_t scale_word,
    int row);

template <class ScaleTensor, int kScaleByteCount>
__device__ __forceinline__ void StoreTracedScaleBytes(
    ScaleTensor& scale_tensor,
    const std::uint8_t (&scale_bytes)[kScaleByteCount],
    int row);

template <class ScaleTensor>
__device__ __forceinline__ void StoreTracedScaleWordsK128(
    ScaleTensor& scale_tensor,
    std::uint32_t scale_word_lo,
    std::uint32_t scale_word_hi,
    int row);

template <int kRowsPerTile>
__device__ __forceinline__ AFragment64 LoadFragmentA_RowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int row_base,
    int lane_id);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ AFragment64 LoadFragmentA_RowMajor16x64Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int row_base,
    int thread_idx);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ AFragment64 LoadFragmentA_RowMajor16x64TracedScaleTiled(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int row_base,
    int thread_idx);

template <int kRowsPerTile>
__device__ __forceinline__ BFragment64 LoadFragmentB_ColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int lane_id,
    int n_base);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ BFragment64 LoadFragmentB_ColMajor64x8Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int thread_idx,
    int n_base);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ AFragment64 LoadFragmentA_RowMajor16x64TracedScaleTiledP13(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int row_base,
    int thread_idx);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ BFragment64 LoadFragmentB_ColMajor64x8TracedScaleTiledP13(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int n_base);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ BFragment64 LoadFragmentB_ColMajor64x8TracedScaleTiled(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int n_base);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void LoadTracedP13AFragmentsRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int row_base,
    AFragment64 (&fragments)[kTracedP13MFragments]);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void LoadTracedP13BFragmentsColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    BFragment64 (&fragments)[kTracedP13NFragments]);

template <class TiledMma, typename OutputType>
__device__ __forceinline__ void StoreTracedP13CFragmentsRowMajor(
    float alpha,
    const CFragment64 (&accum)[kTracedP13MFragments][kTracedP13NFragments],
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output);

template <class TiledMma, int kRowsPerTile, int kFragmentCount>
__device__ __forceinline__ void LoadFragmentBTileColMajor64x8Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int thread_idx,
    BFragment64 (&fragments)[kFragmentCount]);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void LoadTracedP5AFragmentsRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    AFragment64 (&fragments)[kTracedP5MFragments]);

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void LoadTracedP5BFragmentsColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    BFragment64 (&fragments)[kTracedP5NFragments]);

template <class TiledMma, typename OutputType>
__device__ __forceinline__ void StoreTracedP5CFragmentsTranspose(
    float alpha,
    const CFragment64 (&accum)[kTracedP5MFragments][kTracedP5NFragments],
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output);



__device__ __forceinline__ void Clear(CFragment64& fragment);

__device__ __forceinline__ void Gemm(
    CFragment64& accum,
    const AFragment64& a,
    const BFragment64& b);

__device__ __forceinline__ std::uint32_t MakePackedUnitScaleWord();

template <int kRowsPerTile, typename OutputType>
__device__ __forceinline__ void StoreFragmentC_RowMajor16x8(
    float alpha,
    const CFragment64& accum,
    int lane_id,
    int col_base,
    int row_base,
    int valid_rows,
    int valid_cols,
    std::size_t row_start,
    std::size_t output_row_base,
    std::size_t output_rows_per_expert,
    OutputType* output);

template <typename OutputType>
__device__ __forceinline__ void StoreFragmentC_Transpose16x8(
    float alpha,
    const CFragment64& accum,
    int lane_id,
    int output_row_base,
    int m_base,
    int token_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output);

}  // namespace nvfp4_bridge

constexpr int kGroupedTokenTile = static_cast<int>(kMoeLaunchPlanTokenTile);
// Match TRT-LLM's grouped routed shape more closely: tileTokensDim=16 and an
// epilogue/output tile of 128 rows per CTA when transposeMmaOutput=true.
constexpr int kPlannedOutputTile = 128;
constexpr int kPlannedThreadsPerBlock = 256;
constexpr int kPlannedWmmaTileM = 16;
constexpr int kPlannedWmmaTileN = 16;
constexpr int kPlannedWmmaTileK = 16;
constexpr int kPlannedWmmaWarpsPerBlock = 8;
// The traced TRT routed grouped kernels launch with BlockX=384 on SM120.
// Keep eight consumer warps for the 128-row WMMA tile and add four extra
// staging warps so the grouped decode/stage path is closer to TRT's larger
// warp-specialized CTA shape.
constexpr int kGroupedThreadsPerBlock = 384;
constexpr int kGroupedConsumerWarpsPerBlock = 8;
constexpr int kGroupedProducerWarpsPerBlock =
    (kGroupedThreadsPerBlock / 32) - kGroupedConsumerWarpsPerBlock;
constexpr int kFp4MmaTileN = 8;
constexpr int kRoutedLargeOutputTile = 256;
constexpr int kRoutedThreadsPerBlock = kGroupedThreadsPerBlock;
constexpr int kContiguousSmallOutputTile = 32;
constexpr int kContiguousMediumOutputTile = 64;
constexpr int kContiguousLargeOutputTile = 128;
constexpr int kContiguousSmallThreadsPerBlock = 64;
constexpr int kContiguousMediumThreadsPerBlock = 128;
constexpr int kContiguousLargeThreadsPerBlock = 256;
constexpr float kNvfp4ActivationMaxFinite = 6.0f * 448.0f;
constexpr float kNvfp4MinTensorScale = 1.0f / 1024.0f;
constexpr std::size_t kNvfp4ScaleBlockTile = 4u;
constexpr int kSharedContiguousP5ConsumerWarps =
    cute::size(nvfp4_bridge::TracedP5TiledMma{}) / 32;
constexpr int kSharedContiguousP5ThreadsPerBlock =
    (kSharedContiguousP5ConsumerWarps + 1) * 32;

static_assert(kGroupedTokenTile == static_cast<int>(kMoeLaunchPlanTokenTile));



enum class RoutedGemm1Profile {
  kLegacy,
  kP0_128x128x128_SwapFalse,
  kP1_128x128x64_SwapFalse,
  kP4_128x128x128_SwapTrue,
  kP5_128x128x64_SwapTrue,
  kP7_256x128x64_SwapTrue,
};

enum class RoutedGemm2Profile {
  kLegacy,
  kP12_128x128x128_SwapTrue,
  kP13_128x128x64_SwapTrue,
  kP15_256x128x64_SwapTrue,
};

struct SharedContiguousPreparedLaunch {
  GemmLaunchPlan launch_plan;
  PreparedGemmExecution execution;
  const void* p5_tma_load_b_descriptors = nullptr;
  const void* p5_tma_load_sfb_descriptors = nullptr;
};

struct P15ScaleTrace {
  int valid = 0;
  int row_start = -1;
  int valid_rows = -1;
  int output_row_base = -1;
  int a_base_rows[4] = {-1, -1, -1, -1};
  int b_base_rows[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  std::uint32_t a_source_words[4] = {0, 0, 0, 0};
  std::uint32_t b_source_words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::uint32_t a_fragment_words[4] = {0, 0, 0, 0};
  std::uint32_t b_fragment_words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

struct P5ScaleTrace {
  int valid = 0;
  int row_start = -1;
  int valid_rows = -1;
  int output_row_base = -1;
  int a_base_rows[4] = {-1, -1, -1, -1};
  int b_base_rows[2] = {-1, -1};
  std::uint32_t a_source_words[4] = {0, 0, 0, 0};
  std::uint32_t b_source_words[2] = {0, 0};
  std::uint32_t a_fragment_words[4] = {0, 0, 0, 0};
  std::uint32_t b_fragment_words[2] = {0, 0};
};

struct P13ScaleTrace {
  int valid = 0;
  int row_start = -1;
  int valid_rows = -1;
  int output_row_base = -1;
  int a_base_rows[2] = {-1, -1};
  int b_base_rows[2] = {-1, -1};
  std::uint32_t a_source_words[2] = {0, 0};
  std::uint32_t b_source_words[2] = {0, 0};
  std::uint32_t a_smem_words[2] = {0, 0};
  std::uint32_t b_smem_words[2] = {0, 0};
  std::uint32_t a_fragment_words[2] = {0, 0};
  std::uint32_t b_fragment_words[2] = {0, 0};
  std::uint32_t a_loaded_words[2] = {0, 0};
  std::uint32_t b_loaded_words[2] = {0, 0};
  int a_scale_rows[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int a_scale_cols[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int b_scale_rows[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int b_scale_cols[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  std::uint8_t a_scale_raw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t b_scale_raw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t a_post_store_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t b_post_store_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t a_logical_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t b_logical_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct P1DirectTrace {
  int valid = 0;
  int warp_ids[2] = {0, 0};
  int n_base[2] = {0, 0};
  std::uint32_t a_scale[2] = {0u, 0u};
  std::uint32_t a_regs[2][4] = {};
  std::uint32_t b_scale[2] = {0u, 0u};
  std::uint32_t b_regs[2][2] = {};
};

__device__ __managed__ P15ScaleTrace g_p15_scale_trace;
__device__ __managed__ P5ScaleTrace g_p5_scale_trace;
__device__ __managed__ P13ScaleTrace g_p13_scale_trace;
__device__ __managed__ P1DirectTrace g_p1_direct_trace;
__device__ __managed__ P13DebugTrace g_p13_debug_trace;
__device__ __managed__ int g_enable_p15_scale_trace = 0;
__device__ __managed__ int g_enable_p5_scale_trace = 0;
__device__ __managed__ int g_enable_p13_scale_trace = 0;
__device__ __managed__ int g_enable_p13_scale_map_probe = 0;
__device__ __managed__ int g_enable_p13_debug_trace = 0;
__device__ __managed__ int g_enable_p1_direct_trace = 0;
__device__ __managed__ int g_p13_debug_trace_target_valid_rows = -1;
__device__ __managed__ int g_p13_debug_trace_claimed = 0;
__device__ __managed__ int g_p13_debug_trace_target_thread_id = -1;

template <class ScaleAtomTensor>
__device__ std::uint32_t PackP15ScaleFragmentWord(ScaleAtomTensor const& scale_atom) {
  constexpr int kGroupSize = 16;
  const auto raw0 = static_cast<std::uint8_t>(scale_atom(0).raw());
  const auto raw1 = static_cast<std::uint8_t>(scale_atom(kGroupSize).raw());
  const auto raw2 = static_cast<std::uint8_t>(scale_atom(2 * kGroupSize).raw());
  const auto raw3 = static_cast<std::uint8_t>(scale_atom(3 * kGroupSize).raw());
  return static_cast<std::uint32_t>(raw0) |
         (static_cast<std::uint32_t>(raw1) << 8) |
         (static_cast<std::uint32_t>(raw2) << 16) |
         (static_cast<std::uint32_t>(raw3) << 24);
}

template <class ScaleAtomTensor>
__device__ void FillP15ScaleFragmentWord(ScaleAtomTensor const& scale_atom, std::uint32_t packed_scale_word) {
  constexpr int kGroupSize = 16;
  static_assert(cute::size(ScaleAtomTensor{}) == 64);
#pragma unroll
  for (int byte_index = 0; byte_index < 4; ++byte_index) {
    const std::uint8_t value =
        static_cast<std::uint8_t>((packed_scale_word >> (byte_index * 8)) & 0xFFu);
    scale_atom(byte_index * kGroupSize) = nvfp4_bridge::MakeScaleElement(value);
  }
}

void PrintP15ScaleTrace() {
  const auto& trace = g_p15_scale_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p15_scale_trace: invalid\n");
    return;
  }
  std::fprintf(
      stderr,
      "p15_scale_trace: row_start=%d valid_rows=%d output_row_base=%d\n",
      trace.row_start,
      trace.valid_rows,
      trace.output_row_base);
  for (int i = 0; i < 4; ++i) {
    std::fprintf(
        stderr,
        "p15_scale_trace: A m=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.a_base_rows[i],
        trace.a_source_words[i],
        trace.a_fragment_words[i]);
  }
  for (int i = 0; i < 8; ++i) {
    std::fprintf(
        stderr,
        "p15_scale_trace: B n=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.b_base_rows[i],
        trace.b_source_words[i],
        trace.b_fragment_words[i]);
  }
}

void PrintP5ScaleTrace() {
  const auto& trace = g_p5_scale_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p5_scale_trace: invalid\n");
    return;
  }
  std::fprintf(
      stderr,
      "p5_scale_trace: row_start=%d valid_rows=%d output_row_base=%d\n",
      trace.row_start,
      trace.valid_rows,
      trace.output_row_base);
  for (int i = 0; i < 4; ++i) {
    std::fprintf(
        stderr,
        "p5_scale_trace: A m=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.a_base_rows[i],
        trace.a_source_words[i],
        trace.a_fragment_words[i]);
  }
  for (int i = 0; i < 2; ++i) {
    std::fprintf(
        stderr,
        "p5_scale_trace: B n=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.b_base_rows[i],
        trace.b_source_words[i],
        trace.b_fragment_words[i]);
  }
}

bool CheckCuda(cudaError_t status) {
  if (status != cudaSuccess && std::getenv("NEMOTRON_ROUTED_PROFILE_DEBUG") != nullptr) {
    std::fprintf(stderr, "fused_moe_prefill cuda error: %s\n", cudaGetErrorString(status));
  }
  return status == cudaSuccess;
}

bool RoutedEnvEnabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool CopyDeviceFloatBufferToHost(
    const float* source,
    std::size_t count,
    std::vector<float>* host_output) {
  if (source == nullptr || host_output == nullptr) {
    return false;
  }
  host_output->assign(count, 0.0f);
  if (count == 0) {
    return true;
  }
  return CheckCuda(cudaMemcpy(
      host_output->data(),
      source,
      count * sizeof(float),
      cudaMemcpyDeviceToHost));
}

bool CopyDeviceBf16BufferToHost(
    const __nv_bfloat16* source,
    std::size_t count,
    std::vector<float>* host_output) {
  if (source == nullptr || host_output == nullptr) {
    return false;
  }
  std::vector<__nv_bfloat16> host_bf16(count);
  if (count != 0 &&
      !CheckCuda(cudaMemcpy(
          host_bf16.data(),
          source,
          count * sizeof(__nv_bfloat16),
          cudaMemcpyDeviceToHost))) {
    return false;
  }
  host_output->resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    (*host_output)[index] = __bfloat162float(host_bf16[index]);
  }
  return true;
}

void PrintPackedDataMismatches(
    const char* compare_label,
    const std::vector<std::uint8_t>& reference,
    const std::vector<std::uint8_t>& direct,
    std::size_t rows,
    std::size_t cols,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  const std::size_t packed_row_bytes = cols / 2u;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t pair = 0; pair < packed_row_bytes; ++pair) {
      const std::size_t index = row * packed_row_bytes + pair;
      if (reference[index] == direct[index]) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s packed_data mismatch[%zu]: row=%zu col=%zu reference=0x%02x direct=0x%02x\n",
            compare_label,
            *mismatch_count,
            row,
            pair * 2u,
            static_cast<unsigned int>(reference[index]),
            static_cast<unsigned int>(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintPackedScaleMismatches(
    const char* compare_label,
    const char* label,
    const std::vector<std::uint8_t>& reference,
    const std::vector<std::uint8_t>& direct,
    std::size_t rows,
    std::size_t blocks_per_row,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t index = row * blocks_per_row + block;
      if (reference[index] == direct[index]) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s %s mismatch[%zu]: row=%zu col=%zu reference=0x%02x direct=0x%02x\n",
            compare_label,
            label,
            *mismatch_count,
            row,
            block * fused_decode::kNvfp4BlockWidth,
            static_cast<unsigned int>(reference[index]),
            static_cast<unsigned int>(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintMatmulScaleMismatches(
    const char* compare_label,
    const std::vector<std::uint8_t>& reference,
    const std::vector<std::uint8_t>& direct,
    std::size_t rows,
    std::size_t blocks_per_row,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t index =
          ExecutionScaleOffset(row, block, padded_blocks_per_row, scale_layout);
      if (reference[index] == direct[index]) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s matmul_block_scales_data mismatch[%zu]: row=%zu col=%zu reference=0x%02x direct=0x%02x\n",
            compare_label,
            *mismatch_count,
            row,
            block * fused_decode::kNvfp4BlockWidth,
            static_cast<unsigned int>(reference[index]),
            static_cast<unsigned int>(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintActivationScaleMismatches(
    const char* compare_label,
    const std::vector<float>& reference,
    const std::vector<float>& direct,
    std::size_t rows,
    std::size_t blocks_per_row,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t index = row * blocks_per_row + block;
      if (FloatBits(reference[index]) == FloatBits(direct[index])) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s activation_output_scale mismatch[%zu]: row=%zu col=%zu reference=%g (0x%08x) direct=%g (0x%08x)\n",
            compare_label,
            *mismatch_count,
            row,
            block * fused_decode::kNvfp4BlockWidth,
            static_cast<double>(reference[index]),
            FloatBits(reference[index]),
            static_cast<double>(direct[index]),
            FloatBits(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintDenseFloatMismatches(
    const char* compare_label,
    const char* label,
    const std::vector<float>& reference,
    const std::vector<float>& direct,
    std::size_t rows,
    std::size_t cols,
    float tolerance,
    std::size_t limit,
    std::size_t* mismatch_count,
    float* max_abs_diff) {
  *mismatch_count = 0;
  *max_abs_diff = 0.0f;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const std::size_t index = row * cols + col;
      const float diff = fabsf(reference[index] - direct[index]);
      *max_abs_diff = fmaxf(*max_abs_diff, diff);
      if (diff <= tolerance) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s %s mismatch[%zu]: row=%zu col=%zu reference=%g direct=%g diff=%g\n",
            compare_label,
            label,
            *mismatch_count,
            row,
            col,
            static_cast<double>(reference[index]),
            static_cast<double>(direct[index]),
            static_cast<double>(diff));
      }
      ++(*mismatch_count);
    }
  }
}

// Host fp64 NVFP4 GEMM reference for the first FC1 dispatch.  Independent of
// any kernel — used as ground truth in MaybeCompareFirstFp4DirectFc1Output to
// detect bugs in either the BF16 reference path or the direct FP4 kernel
// without trusting either of them as a baseline.
//
// Computes prepack[input_row, output_col] = (sum_k a*b) * a_tscale * b_tscale,
// where the dot product is in fp64 and the input row → expert mapping comes
// from the launch plan's cta_idx_xy_to_batch_idx + cta_row_starts/valid_rows.
//
// Returns false if the host reference cannot be computed (missing pointers,
// shape mismatch, etc.).  Caller is expected to ensure scales are configured
// in one of the supported ways: per-row tensor scales, per-expert tensor
// scales, or a single global tensor scale (all FP4-direct dispatches use one
// of these).
bool ComputeFp4DirectHostReference(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales_dev,
    const float* input_per_row_tensor_scales_dev,
    const DeviceMoeLaunchPlan* launch_plan,
    const FusedNvfp4WeightView* weights_dev,
    std::size_t output_rows_per_expert,
    std::vector<float>* host_reference) {
  if (host_reference == nullptr ||
      !input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      weights_dev == nullptr) {
    return false;
  }

  const std::size_t input_rows = input_pack.rows();
  const std::size_t hidden_size = input_pack.cols();
  if (hidden_size == 0 || (hidden_size % 16u) != 0 ||
      output_rows_per_expert == 0 || (output_rows_per_expert % 16u) != 0) {
    return false;
  }
  const std::size_t blocks_per_row = hidden_size / 16u;
  const std::size_t row_packed_bytes = hidden_size / 2u;

  // Snapshot the input pack on the host.
  std::vector<std::uint8_t> input_packed_host;
  std::vector<std::uint8_t> input_scales_host;
  if (!input_pack.CopyPackedToHost(&input_packed_host) ||
      !input_pack.CopyBlockScalesToHost(&input_scales_host)) {
    return false;
  }
  if (input_packed_host.size() < input_rows * row_packed_bytes ||
      input_scales_host.size() < input_rows * blocks_per_row) {
    return false;
  }

  // Resolve per-row input tensor scales.
  std::vector<float> input_row_tscales(input_rows, 1.0f);
  if (input_per_row_tensor_scales_dev != nullptr) {
    if (cudaMemcpy(
            input_row_tscales.data(),
            input_per_row_tensor_scales_dev,
            input_rows * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
  } else {
    float global_scale = 0.0f;
    if (!input_pack.CopyTensorScaleToHost(&global_scale)) {
      return false;
    }
    if (global_scale == 0.0f) {
      global_scale = 1.0f;
    }
    std::fill(input_row_tscales.begin(), input_row_tscales.end(), global_scale);
  }

  // Resolve the active CTA → expert mapping and the input-row partitioning.
  const int exact_cta_count = launch_plan->exact_cta_count_host();
  if (exact_cta_count <= 0) {
    return false;
  }
  std::vector<int> cta_batch_host(static_cast<std::size_t>(exact_cta_count), -1);
  if (cudaMemcpy(
          cta_batch_host.data(),
          launch_plan->cta_idx_xy_to_batch_idx(),
          static_cast<std::size_t>(exact_cta_count) * sizeof(int),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    return false;
  }
  const int* cta_row_starts_host = launch_plan->cta_row_starts_host();
  const int* cta_valid_rows_host = launch_plan->cta_valid_rows_host();
  if (cta_row_starts_host == nullptr || cta_valid_rows_host == nullptr) {
    return false;
  }

  // For every input row covered by some CTA, record which expert it belongs
  // to.  Rows not covered by any CTA are left at expert -1 and the host
  // reference leaves their output at zero (matching the kernel's
  // LaunchZeroBf16Buffer behavior on the prepack workspace).
  std::vector<int> row_to_expert(input_rows, -1);
  for (int cta = 0; cta < exact_cta_count; ++cta) {
    const int expert = cta_batch_host[cta];
    const int row_start = cta_row_starts_host[cta];
    const int valid_rows = cta_valid_rows_host[cta];
    if (expert < 0 || row_start < 0 || valid_rows <= 0) {
      continue;
    }
    for (int r = 0; r < valid_rows; ++r) {
      const std::size_t idx = static_cast<std::size_t>(row_start + r);
      if (idx < input_rows) {
        row_to_expert[idx] = expert;
      }
    }
  }

  // Snapshot every distinct expert weight that we actually need.  Caches by
  // expert id so we only copy each expert's weight from device once.
  std::vector<bool> expert_loaded;
  std::vector<FusedNvfp4WeightView> expert_view_host;
  std::vector<std::vector<std::uint8_t>> expert_packed_host;
  std::vector<std::vector<std::uint8_t>> expert_scales_host;
  std::vector<float> expert_tensor_scale_host;
  const std::size_t weight_packed_bytes = (output_rows_per_expert * hidden_size) / 2u;
  const std::size_t weight_scales_bytes = output_rows_per_expert * blocks_per_row;

  auto load_expert = [&](int expert_id) -> bool {
    if (expert_id < 0) {
      return false;
    }
    if (static_cast<std::size_t>(expert_id) >= expert_loaded.size()) {
      const std::size_t new_size = static_cast<std::size_t>(expert_id) + 1u;
      expert_loaded.resize(new_size, false);
      expert_view_host.resize(new_size);
      expert_packed_host.resize(new_size);
      expert_scales_host.resize(new_size);
      expert_tensor_scale_host.resize(new_size, 0.0f);
    }
    if (expert_loaded[expert_id]) {
      return true;
    }
    FusedNvfp4WeightView view{};
    if (cudaMemcpy(
            &view,
            &weights_dev[expert_id],
            sizeof(FusedNvfp4WeightView),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    if (view.packed_data == nullptr ||
        view.matmul_block_scales_data == nullptr ||
        view.tensor_scale_data == nullptr ||
        view.input_cols != hidden_size ||
        view.output_rows != output_rows_per_expert) {
      return false;
    }
    expert_view_host[expert_id] = view;
    expert_packed_host[expert_id].assign(weight_packed_bytes, 0u);
    // Decode the weight's swizzled scales into a row-major host buffer to
    // simplify the dot-product loop.  Production weights only populate the
    // matmul (swizzled) layout, so we go through ExecutionScaleOffset to
    // un-swizzle.
    const std::size_t weight_padded_blocks_per_row =
        RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
    const std::size_t weight_swizzled_bytes =
        ((output_rows_per_expert + 127u) / 128u) * 128u *
        weight_padded_blocks_per_row;
    std::vector<std::uint8_t> swizzled_scratch(weight_swizzled_bytes, 0u);
    if (cudaMemcpy(
            expert_packed_host[expert_id].data(),
            view.packed_data,
            weight_packed_bytes,
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            swizzled_scratch.data(),
            view.matmul_block_scales_data,
            weight_swizzled_bytes,
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            &expert_tensor_scale_host[expert_id],
            view.tensor_scale_data,
            sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    expert_scales_host[expert_id].assign(weight_scales_bytes, 0u);
    for (std::size_t row = 0; row < output_rows_per_expert; ++row) {
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const std::size_t swizzled_offset = ExecutionScaleOffset(
            row, block, weight_padded_blocks_per_row,
            Nvfp4ScaleLayout::kSwizzled128x4);
        expert_scales_host[expert_id][row * blocks_per_row + block] =
            swizzled_scratch[swizzled_offset];
      }
    }
    expert_loaded[expert_id] = true;
    return true;
  };

  for (std::size_t r = 0; r < input_rows; ++r) {
    const int expert = row_to_expert[r];
    if (expert >= 0 && !load_expert(expert)) {
      return false;
    }
  }

  // If we have per-expert input tensor scales, copy them to host and apply
  // them per row according to the row→expert mapping.
  if (input_expert_tensor_scales_dev != nullptr) {
    // Copy enough entries to cover the highest expert we touched.
    int max_expert = -1;
    for (int e : row_to_expert) {
      if (e > max_expert) {
        max_expert = e;
      }
    }
    if (max_expert >= 0) {
      std::vector<float> expert_input_tscales(static_cast<std::size_t>(max_expert + 1), 1.0f);
      if (cudaMemcpy(
              expert_input_tscales.data(),
              input_expert_tensor_scales_dev,
              expert_input_tscales.size() * sizeof(float),
              cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
      for (std::size_t r = 0; r < input_rows; ++r) {
        const int expert = row_to_expert[r];
        if (expert >= 0) {
          input_row_tscales[r] = expert_input_tscales[expert];
        }
      }
    }
  }

  // Compute the dot products in fp64.
  host_reference->assign(input_rows * output_rows_per_expert, 0.0f);

  auto decode_fp4 = [](std::uint8_t nibble) -> float {
    __nv_fp4_e2m1 v;
    v.__x = nibble & 0x0Fu;
    return static_cast<float>(v);
  };
  auto decode_fp8 = [](std::uint8_t byte) -> float {
    __nv_fp8_e4m3 v;
    v.__x = byte;
    return static_cast<float>(v);
  };

  for (std::size_t row = 0; row < input_rows; ++row) {
    const int expert = row_to_expert[row];
    if (expert < 0) {
      continue;
    }
    const float a_tscale = input_row_tscales[row];
    const float b_tscale = expert_tensor_scale_host[expert];
    const std::uint8_t* input_row_packed =
        input_packed_host.data() + row * row_packed_bytes;
    const std::uint8_t* input_row_scales =
        input_scales_host.data() + row * blocks_per_row;
    const std::uint8_t* expert_packed = expert_packed_host[expert].data();
    const std::uint8_t* expert_scales = expert_scales_host[expert].data();

    for (std::size_t n = 0; n < output_rows_per_expert; ++n) {
      const std::uint8_t* weight_row_packed = expert_packed + n * row_packed_bytes;
      const std::uint8_t* weight_row_scales = expert_scales + n * blocks_per_row;
      double acc = 0.0;
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const float ab_block_scale =
            decode_fp8(input_row_scales[block]) *
            decode_fp8(weight_row_scales[block]);
        double block_acc = 0.0;
        const std::size_t k_byte_start = block * 8u;
        for (std::size_t kb = 0; kb < 8u; ++kb) {
          const std::uint8_t a_byte = input_row_packed[k_byte_start + kb];
          const std::uint8_t b_byte = weight_row_packed[k_byte_start + kb];
          const float a_lo = decode_fp4(a_byte & 0x0Fu);
          const float a_hi = decode_fp4((a_byte >> 4) & 0x0Fu);
          const float b_lo = decode_fp4(b_byte & 0x0Fu);
          const float b_hi = decode_fp4((b_byte >> 4) & 0x0Fu);
          block_acc += static_cast<double>(a_lo) * static_cast<double>(b_lo);
          block_acc += static_cast<double>(a_hi) * static_cast<double>(b_hi);
        }
        acc += block_acc * static_cast<double>(ab_block_scale);
      }
      const double tscale = static_cast<double>(a_tscale) * static_cast<double>(b_tscale);
      (*host_reference)[row * output_rows_per_expert + n] =
          static_cast<float>(acc * tscale);
    }
  }
  return true;
}

bool MaybeCompareFirstFp4DirectFc1Output(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const DeviceNvfp4Matrix& direct_output_pack,
    const float* direct_activation_output_scale,
    __nv_bfloat16* bf16_workspace) {
  if (!RoutedEnvEnabled("NEMOTRON_DEBUG_COMPARE_FP4_DIRECT")) {
    return true;
  }

  constexpr const char* kCompareLabel = "fp4_direct_compare";
  constexpr const char* kPrepackCompareLabel = "fp4_direct_prepack_compare";

  static std::mutex compare_mutex;
  static bool compared_once = false;
  {
    std::lock_guard<std::mutex> lock(compare_mutex);
    if (compared_once) {
      return true;
    }
    compared_once = true;
  }

  if (!input_pack.valid() ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      weights == nullptr ||
      !direct_output_pack.valid() ||
      direct_activation_output_scale == nullptr ||
      bf16_workspace == nullptr) {
    std::fprintf(stderr, "%s: invalid inputs\n", kCompareLabel);
    return false;
  }

  const std::size_t rows = direct_output_pack.rows();
  const std::size_t cols = direct_output_pack.cols();
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::size_t bf16_workspace_count =
      launch_plan->padded_row_capacity() * output_rows_per_expert;
  constexpr std::size_t kMaxReportedMismatches = 16;
  constexpr float kBf16CompareTolerance = 1.0e-2f;

  auto reference_pack =
      DeviceNvfp4Matrix::Create(rows, cols, direct_output_pack.scale_layout());
  auto reference_activation_output_scale =
      DeviceTensorFp32::Create({rows, blocks_per_row});
  if (reference_pack == nullptr ||
      !reference_pack->valid() ||
      reference_activation_output_scale == nullptr ||
      !reference_activation_output_scale->valid()) {
    std::fprintf(stderr, "%s: failed to allocate reference buffers\n", kCompareLabel);
    return false;
  }

  std::vector<float> direct_prepack_output;
  if (!CheckCuda(cudaDeviceSynchronize()) ||
      !CopyDeviceBf16BufferToHost(
          bf16_workspace,
          bf16_workspace_count,
          &direct_prepack_output) ||
      !LaunchZeroBf16Buffer(bf16_workspace, bf16_workspace_count) ||
      !LaunchPlannedPackedInputMatVecBf16(
          input_pack,
          input_expert_tensor_scales,
          input_dq_scales,
          input_per_row_tensor_scales,
          launch_plan,
          dispatch_rows,
          active_selection_count,
          weights,
          output_rows_per_expert,
          bf16_workspace) ||
      !LaunchRoutedBf16Relu2Pack(
          bf16_workspace,
          routing,
          launch_plan,
          active_selection_count,
          reference_pack.get(),
          reference_activation_output_scale->data()) ||
      !CheckCuda(cudaDeviceSynchronize())) {
    std::fprintf(stderr, "%s: failed to build BF16 reference\n", kCompareLabel);
    return false;
  }

  std::vector<float> reference_prepack_output;
  if (!CopyDeviceBf16BufferToHost(
          bf16_workspace,
          bf16_workspace_count,
          &reference_prepack_output)) {
    std::fprintf(stderr, "%s: failed to copy BF16 prepack outputs to host\n", kPrepackCompareLabel);
    return false;
  }

  std::vector<std::uint8_t> direct_packed;
  std::vector<std::uint8_t> reference_packed;
  std::vector<std::uint8_t> direct_block_scales;
  std::vector<std::uint8_t> reference_block_scales;
  std::vector<std::uint8_t> direct_matmul_block_scales;
  std::vector<std::uint8_t> reference_matmul_block_scales;
  std::vector<float> direct_activation_scales;
  std::vector<float> reference_activation_scales;
  if (!direct_output_pack.CopyPackedToHost(&direct_packed) ||
      !reference_pack->CopyPackedToHost(&reference_packed) ||
      !direct_output_pack.CopyBlockScalesToHost(&direct_block_scales) ||
      !reference_pack->CopyBlockScalesToHost(&reference_block_scales) ||
      !direct_output_pack.CopyMatmulBlockScalesToHost(&direct_matmul_block_scales) ||
      !reference_pack->CopyMatmulBlockScalesToHost(&reference_matmul_block_scales) ||
      !CopyDeviceFloatBufferToHost(
          direct_activation_output_scale,
          rows * blocks_per_row,
          &direct_activation_scales) ||
      !CopyDeviceFloatBufferToHost(
          reference_activation_output_scale->data(),
          rows * blocks_per_row,
          &reference_activation_scales)) {
    std::fprintf(stderr, "%s: failed to copy outputs to host\n", kCompareLabel);
    return false;
  }

  std::size_t packed_mismatches = 0;
  std::size_t block_scale_mismatches = 0;
  std::size_t matmul_scale_mismatches = 0;
  std::size_t activation_scale_mismatches = 0;
  std::size_t prepack_mismatches = 0;
  float prepack_max_abs_diff = 0.0f;
  PrintDenseFloatMismatches(
      kPrepackCompareLabel,
      "bf16_output",
      reference_prepack_output,
      direct_prepack_output,
      rows,
      cols,
      kBf16CompareTolerance,
      kMaxReportedMismatches,
      &prepack_mismatches,
      &prepack_max_abs_diff);
  PrintPackedDataMismatches(
      kCompareLabel,
      reference_packed,
      direct_packed,
      rows,
      cols,
      kMaxReportedMismatches,
      &packed_mismatches);
  PrintPackedScaleMismatches(
      kCompareLabel,
      "block_scales_data",
      reference_block_scales,
      direct_block_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &block_scale_mismatches);
  PrintMatmulScaleMismatches(
      kCompareLabel,
      reference_matmul_block_scales,
      direct_matmul_block_scales,
      rows,
      blocks_per_row,
      padded_blocks_per_row,
      direct_output_pack.scale_layout(),
      kMaxReportedMismatches,
      &matmul_scale_mismatches);
  PrintActivationScaleMismatches(
      kCompareLabel,
      reference_activation_scales,
      direct_activation_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &activation_scale_mismatches);

  std::fprintf(
      stderr,
      "%s summary: bf16_output=%zu max_abs_diff=%g tolerance=%g\n",
      kPrepackCompareLabel,
      prepack_mismatches,
      static_cast<double>(prepack_max_abs_diff),
      static_cast<double>(kBf16CompareTolerance));
  std::fprintf(
      stderr,
      "%s summary: packed_data=%zu block_scales_data=%zu matmul_block_scales_data=%zu activation_output_scale=%zu\n",
      kCompareLabel,
      packed_mismatches,
      block_scale_mismatches,
      matmul_scale_mismatches,
      activation_scale_mismatches);

  // Independent host fp64 NVFP4 GEMM reference: ground truth that does not
  // depend on the BF16 reference path.  Lets us tell whether a divergence
  // is in the direct kernel, in the BF16 reference, or in both.
  constexpr const char* kHostOracleLabel = "fp4_direct_host_oracle";
  std::vector<float> host_reference_output;
  const bool host_reference_ok = ComputeFp4DirectHostReference(
      input_pack,
      input_expert_tensor_scales,
      input_per_row_tensor_scales,
      launch_plan,
      weights,
      output_rows_per_expert,
      &host_reference_output);
  if (host_reference_ok) {
    std::size_t host_vs_direct_mismatches = 0;
    float host_vs_direct_max = 0.0f;
    PrintDenseFloatMismatches(
        kHostOracleLabel,
        "direct_prepack",
        host_reference_output,
        direct_prepack_output,
        rows,
        cols,
        kBf16CompareTolerance,
        kMaxReportedMismatches,
        &host_vs_direct_mismatches,
        &host_vs_direct_max);
    std::size_t host_vs_reference_mismatches = 0;
    float host_vs_reference_max = 0.0f;
    PrintDenseFloatMismatches(
        kHostOracleLabel,
        "bf16_reference",
        host_reference_output,
        reference_prepack_output,
        rows,
        cols,
        kBf16CompareTolerance,
        kMaxReportedMismatches,
        &host_vs_reference_mismatches,
        &host_vs_reference_max);
    std::fprintf(
        stderr,
        "%s summary: direct_prepack=%zu max_abs_diff=%g  bf16_reference=%zu max_abs_diff=%g\n",
        kHostOracleLabel,
        host_vs_direct_mismatches,
        static_cast<double>(host_vs_direct_max),
        host_vs_reference_mismatches,
        static_cast<double>(host_vs_reference_max));
  } else {
    std::fprintf(stderr, "%s summary: failed_to_compute_host_reference\n", kHostOracleLabel);
  }

  return true;
}

bool MaybeCompareFirstGroupedBf16Fc1Output(
    const float* normalized,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t token_count,
    std::size_t active_selection_count,
    std::size_t hidden_size,
    float* reference_gather_scratch,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const DeviceNvfp4Matrix& direct_output_pack,
    const float* direct_activation_output_scale,
    __nv_bfloat16* reference_bf16_workspace) {
  if (!RoutedEnvEnabled("NEMOTRON_DEBUG_COMPARE_GROUPED_BF16_FC1")) {
    return true;
  }

  constexpr const char* kCompareLabel = "grouped_bf16_fc1_compare";

  static std::mutex compare_mutex;
  static bool compared_once = false;
  {
    std::lock_guard<std::mutex> lock(compare_mutex);
    if (compared_once) {
      return true;
    }
    compared_once = true;
  }

  if (normalized == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      reference_gather_scratch == nullptr ||
      weights == nullptr ||
      !direct_output_pack.valid() ||
      direct_activation_output_scale == nullptr ||
      reference_bf16_workspace == nullptr) {
    std::fprintf(stderr, "%s: invalid inputs\n", kCompareLabel);
    return false;
  }

  const std::size_t rows = direct_output_pack.rows();
  const std::size_t cols = direct_output_pack.cols();
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::size_t gather_count =
      launch_plan->padded_row_capacity() * hidden_size;
  const std::size_t bf16_workspace_count =
      launch_plan->padded_row_capacity() * output_rows_per_expert;
  constexpr std::size_t kMaxReportedMismatches = 16;

  auto reference_pack =
      DeviceNvfp4Matrix::Create(rows, cols, direct_output_pack.scale_layout());
  auto reference_activation_output_scale =
      DeviceTensorFp32::Create({rows, blocks_per_row});
  if (reference_pack == nullptr ||
      !reference_pack->valid() ||
      reference_activation_output_scale == nullptr ||
      !reference_activation_output_scale->valid()) {
    std::fprintf(stderr, "%s: failed to allocate reference buffers\n", kCompareLabel);
    return false;
  }

  if (!LaunchZeroBuffer(reference_gather_scratch, gather_count) ||
      !LaunchGatherRows(
          normalized,
          launch_plan->permuted_idx_to_token_idx(),
          launch_plan->total_num_padded_tokens(),
          launch_plan->padded_row_capacity(),
          token_count,
          hidden_size,
          reference_gather_scratch) ||
      !LaunchZeroBf16Buffer(reference_bf16_workspace, bf16_workspace_count) ||
      !RunLaunchPlannedNvfp4ExpertMatVecBf16(
          reference_gather_scratch,
          launch_plan,
          active_selection_count,
          weights,
          output_rows_per_expert,
          reference_bf16_workspace) ||
      !LaunchRoutedBf16Relu2Pack(
          reference_bf16_workspace,
          routing,
          launch_plan,
          active_selection_count,
          reference_pack.get(),
          reference_activation_output_scale->data()) ||
      !CheckCuda(cudaDeviceSynchronize())) {
    std::fprintf(stderr, "%s: failed to build gathered BF16 reference\n", kCompareLabel);
    return false;
  }

  std::vector<std::uint8_t> direct_packed;
  std::vector<std::uint8_t> reference_packed;
  std::vector<std::uint8_t> direct_block_scales;
  std::vector<std::uint8_t> reference_block_scales;
  std::vector<std::uint8_t> direct_matmul_block_scales;
  std::vector<std::uint8_t> reference_matmul_block_scales;
  std::vector<float> direct_activation_scales;
  std::vector<float> reference_activation_scales;
  if (!direct_output_pack.CopyPackedToHost(&direct_packed) ||
      !reference_pack->CopyPackedToHost(&reference_packed) ||
      !direct_output_pack.CopyBlockScalesToHost(&direct_block_scales) ||
      !reference_pack->CopyBlockScalesToHost(&reference_block_scales) ||
      !direct_output_pack.CopyMatmulBlockScalesToHost(&direct_matmul_block_scales) ||
      !reference_pack->CopyMatmulBlockScalesToHost(&reference_matmul_block_scales) ||
      !CopyDeviceFloatBufferToHost(
          direct_activation_output_scale,
          rows * blocks_per_row,
          &direct_activation_scales) ||
      !CopyDeviceFloatBufferToHost(
          reference_activation_output_scale->data(),
          rows * blocks_per_row,
          &reference_activation_scales)) {
    std::fprintf(stderr, "%s: failed to copy outputs to host\n", kCompareLabel);
    return false;
  }

  std::size_t packed_mismatches = 0;
  std::size_t block_scale_mismatches = 0;
  std::size_t matmul_scale_mismatches = 0;
  std::size_t activation_scale_mismatches = 0;
  PrintPackedDataMismatches(
      kCompareLabel,
      reference_packed,
      direct_packed,
      rows,
      cols,
      kMaxReportedMismatches,
      &packed_mismatches);
  PrintPackedScaleMismatches(
      kCompareLabel,
      "block_scales_data",
      reference_block_scales,
      direct_block_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &block_scale_mismatches);
  PrintMatmulScaleMismatches(
      kCompareLabel,
      reference_matmul_block_scales,
      direct_matmul_block_scales,
      rows,
      blocks_per_row,
      padded_blocks_per_row,
      direct_output_pack.scale_layout(),
      kMaxReportedMismatches,
      &matmul_scale_mismatches);
  PrintActivationScaleMismatches(
      kCompareLabel,
      reference_activation_scales,
      direct_activation_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &activation_scale_mismatches);

  std::fprintf(
      stderr,
      "%s summary: packed_data=%zu block_scales_data=%zu matmul_block_scales_data=%zu activation_output_scale=%zu\n",
      kCompareLabel,
      packed_mismatches,
      block_scale_mismatches,
      matmul_scale_mismatches,
      activation_scale_mismatches);
  return true;
}

bool RoutedProfileDebugEnabled() {
  return std::getenv("NEMOTRON_ROUTED_PROFILE_DEBUG") != nullptr;
}

bool SharedProfileDebugEnabled() {
  return std::getenv("NEMOTRON_SHARED_PROFILE_DEBUG") != nullptr;
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight);

void AppendSharedContiguousTraceEntry(
    const std::string& tensor_name,
    bool plan_build_ok,
    bool execute_ok) {
  if (!IsLinearOpTraceEnabled()) {
    return;
  }
  auto& trace = GetLinearOpTrace();
  std::lock_guard<std::mutex> lock(trace.mutex);
  trace.entries.push_back(LinearOpTraceEntry{
      tensor_name,
      GemmKernelFamily::kSm120ContiguousSharedNvfp4,
      LinearOpPath::kFastpath,
      plan_build_ok,
      execute_ok,
  });
}

std::optional<SharedContiguousPreparedLaunch> PrepareSharedContiguousLaunch(
    const std::string& tensor_name,
    const DeviceNvfp4Matrix& input_pack,
    std::size_t activation_rows,
    const FusedNvfp4WeightView& weight,
    GemmHeuristicCache* heuristic_cache) {
  if (!input_pack.valid() ||
      !ValidFusedNvfp4WeightView(weight) ||
      activation_rows == 0 ||
      activation_rows > input_pack.rows() ||
      input_pack.cols() != weight.input_cols ||
      input_pack.scale_layout() != Nvfp4ScaleLayout::kSwizzled128x4) {
    return std::nullopt;
  }

  const auto launch_plan = BuildSharedNvfp4ContiguousLaunchPlan(
      SharedNvfp4ContiguousPlanRequest{
          tensor_name,
          activation_rows,
          weight.output_rows,
          weight.input_cols,
          ByteRangeView{weight.packed_data, (weight.output_rows * weight.input_cols) / 2u},
          ByteRangeView{
              weight.matmul_block_scales_data != nullptr
                  ? weight.matmul_block_scales_data
                  : weight.block_scales_data,
              1u,
          },
          ByteRangeView{reinterpret_cast<const std::uint8_t*>(weight.tensor_scale_data), sizeof(float)},
      });
  if (!launch_plan.has_value()) {
    return std::nullopt;
  }

  const auto execution = PrepareGemmExecution(*launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    return std::nullopt;
  }

  const void* p5_tma_load_b_descriptors =
      input_pack.shared_p5_tma_load_b_descriptors(*launch_plan);
  const void* p5_tma_load_sfb_descriptors =
      input_pack.shared_p5_tma_load_sfb_descriptors(*launch_plan);
  if (p5_tma_load_b_descriptors == nullptr || p5_tma_load_sfb_descriptors == nullptr) {
    return std::nullopt;
  }

  if (SharedProfileDebugEnabled()) {
    const std::string bucket_upper =
        launch_plan->token_bucket_upper == std::numeric_limits<std::size_t>::max()
            ? "inf"
            : std::to_string(launch_plan->token_bucket_upper);
    std::fprintf(
        stderr,
        "shared_contiguous profile tensor=%s rows=%zu profile=%s tile=%zux%zux%zu bucket=%zu-%s cta=%zux%zu backend=%s algorithm=%llu cached=%d\n",
        tensor_name.c_str(),
        activation_rows,
        launch_plan->profile_name.c_str(),
        launch_plan->tile_m,
        launch_plan->tile_n,
        launch_plan->tile_k,
        launch_plan->token_bucket_lower,
        bucket_upper.c_str(),
        launch_plan->cta_m_count,
        launch_plan->cta_n_count,
        ToString(execution->backend_kind),
        static_cast<unsigned long long>(execution->algorithm_id),
        execution->algorithm_from_cache ? 1 : 0);
  }

  SharedContiguousPreparedLaunch prepared;
  prepared.launch_plan = *launch_plan;
  prepared.execution = *execution;
  prepared.p5_tma_load_b_descriptors = p5_tma_load_b_descriptors;
  prepared.p5_tma_load_sfb_descriptors = p5_tma_load_sfb_descriptors;
  return prepared;
}

const char* RoutedGemm1ProfileName(RoutedGemm1Profile profile) {
  switch (profile) {
    case RoutedGemm1Profile::kLegacy:
      return "legacy";
    case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
      return "p0_128x128x128_swap_false";
    case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
      return "p1_128x128x64_swap_false";
    case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
      return "p4_128x128x128_swap_true";
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue:
      return "p5_128x128x64_swap_true";
    case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
      return "p7_256x128x64_swap_true";
  }
  return "unknown";
}

const char* RoutedGemm1ProfileClassName(RoutedGemm1Profile profile) {
  switch (profile) {
    case RoutedGemm1Profile::kLegacy:
      return "legacy";
    case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
    case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
    case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
      return "bf16_boundary_grouped";
    case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
      return "native_direct";
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue:
      return "native_unified_direct";
  }
  return "unknown";
}

const char* RoutedGemm2ProfileName(RoutedGemm2Profile profile) {
  switch (profile) {
    case RoutedGemm2Profile::kLegacy:
      return "legacy";
    case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
      return "p12_128x128x128_swap_true";
    case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
      return "p13_128x128x64_swap_true";
    case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
      return "p15_256x128x64_swap_true";
  }
  return "unknown";
}

const char* RoutedGemm2ProfileClassName(RoutedGemm2Profile profile) {
  switch (profile) {
    case RoutedGemm2Profile::kLegacy:
      return "legacy";
    case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
    case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
    case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
      return "native_unified";
  }
  return "unknown";
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         (weight.block_scales_data != nullptr ||
          weight.matmul_block_scales_data != nullptr) &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

__host__ __device__ fused_decode::Nvfp4WeightView MakeDeviceWeightView(
    const FusedNvfp4WeightView& weight) {
  return fused_decode::Nvfp4WeightView{
      weight.packed_data,
      weight.block_scales_data,
      weight.matmul_block_scales_data,
      weight.tensor_scale_data,
      weight.output_rows,
      weight.input_cols,
  };
}

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return (numerator + denominator - 1u) / denominator;
}

__host__ __device__ std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
}

__host__ __device__ std::size_t RowTile(Nvfp4ScaleLayout scale_layout) {
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4:
      return 128u;
    case Nvfp4ScaleLayout::kSwizzled8x4:
      return 8u;
  }
  return 128u;
}

__host__ __device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  const std::size_t num_k_tiles = padded_blocks_per_row / kNvfp4ScaleBlockTile;
  const std::size_t k_tile = block_col / kNvfp4ScaleBlockTile;
  const std::size_t inner_k = block_col & 3u;
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4: {
      const std::size_t m_tile = row / 128u;
      const std::size_t outer_m = row & 31u;
      const std::size_t inner_m = (row >> 5u) & 3u;
      return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
              (outer_m << 4u) |
              (inner_m << 2u) |
              inner_k);
    }
    case Nvfp4ScaleLayout::kSwizzled8x4: {
      const std::size_t m_tile = row / 8u;
      const std::size_t inner_m = row & 7u;
      return (((m_tile * num_k_tiles) + k_tile) << 5u) |
             (inner_m << 2u) |
             inner_k;
    }
  }
  return 0;
}

int GetMultiProcessorCount() {
  int device = 0;
  if (!CheckCuda(cudaGetDevice(&device))) {
    return 0;
  }
  int multiprocessor_count = 0;
  if (!CheckCuda(cudaDeviceGetAttribute(
          &multiprocessor_count, cudaDevAttrMultiProcessorCount, device))) {
    return 0;
  }
  return multiprocessor_count;
}

template <typename KernelFn, typename... Args>
bool LaunchProgrammaticKernel(
    dim3 grid,
    dim3 block,
    KernelFn kernel,
    Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = cudaStream_t{};
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = &attr;
  return CheckCuda(cudaLaunchKernelEx(&config, kernel, args...)) &&
         CheckCuda(cudaGetLastError());
}

int SelectContiguousOutputTile(std::size_t output_rows, std::size_t input_row_count) {
  const int multiprocessor_count = GetMultiProcessorCount();
  if (multiprocessor_count <= 0) {
    return kContiguousMediumOutputTile;
  }
  const std::size_t input_row_tiles =
      CeilDiv(input_row_count, static_cast<std::size_t>(kPlannedWmmaTileM));
  const auto meets_target = [&](int output_tile) {
    const std::size_t output_row_tiles =
        CeilDiv(output_rows, static_cast<std::size_t>(output_tile));
    return output_row_tiles * input_row_tiles >=
           static_cast<std::size_t>(multiprocessor_count);
  };
  if (meets_target(kContiguousLargeOutputTile)) {
    return kContiguousLargeOutputTile;
  }
  if (meets_target(kContiguousMediumOutputTile)) {
    return kContiguousMediumOutputTile;
  }
  return kContiguousSmallOutputTile;
}

RoutedGemm1Profile SelectRoutedGemm1Profile(std::size_t num_rows) {
  if (RoutedEnvEnabled("NEMOTRON_DEBUG_FORCE_LEGACY_P1")) {
    return RoutedGemm1Profile::kLegacy;
  }
  if (num_rows >= 1u && num_rows <= 8u) {
    // Use the unified FP4 MMA kernel for all small row counts.  The non-unified
    // P0/P7 profiles decode FP4→BF16 before the WMMA multiply, which loses
    // precision compared to the native block-scaled FP4 MMA used by P5.  At
    // small dispatch sizes (1-3 rows per expert) this precision loss accumulates
    // across layers and produces divergent output.
    return RoutedGemm1Profile::kP5_128x128x64_SwapTrue;
  }
  if ((num_rows >= 9u && num_rows <= 15u) || num_rows == 16u ||
      (num_rows >= 24u && num_rows <= 31u) || num_rows == 320u ||
      (num_rows >= 448u && num_rows <= 511u) || num_rows >= 1024u) {
    return RoutedGemm1Profile::kP1_128x128x64_SwapFalse;
  }
  if (num_rows >= 32u && num_rows <= 112u) {
    return RoutedGemm1Profile::kP0_128x128x128_SwapFalse;
  }
  if ((num_rows >= 120u && num_rows <= 127u) ||
      (num_rows >= 200u && num_rows <= 248u)) {
    return RoutedGemm1Profile::kP4_128x128x128_SwapTrue;
  }
  if ((num_rows >= 128u && num_rows <= 192u) || num_rows == 256u ||
      num_rows == 384u || (num_rows >= 512u && num_rows <= 992u)) {
    return RoutedGemm1Profile::kP5_128x128x64_SwapTrue;
  }
  return RoutedGemm1Profile::kLegacy;
}

RoutedGemm2Profile SelectRoutedGemm2Profile(std::size_t num_rows) {
  if (RoutedEnvEnabled("NEMOTRON_DEBUG_FORCE_LEGACY_P13")) {
    return RoutedGemm2Profile::kLegacy;
  }
  if (num_rows >= 2u && num_rows <= 7u) {
    return RoutedGemm2Profile::kP15_256x128x64_SwapTrue;
  }
  if (num_rows == 16u) {
    return RoutedGemm2Profile::kP15_256x128x64_SwapTrue;
  }
  if (num_rows == 1u || num_rows == 8u || (num_rows >= 24u && num_rows <= 127u) ||
      num_rows == 256u || num_rows == 384u || num_rows >= 1024u) {
    return RoutedGemm2Profile::kP13_128x128x64_SwapTrue;
  }
  if ((num_rows >= 9u && num_rows <= 15u) || (num_rows >= 128u && num_rows <= 248u) ||
      num_rows == 320u || (num_rows >= 448u && num_rows <= 992u)) {
    return RoutedGemm2Profile::kP12_128x128x128_SwapTrue;
  }
  return RoutedGemm2Profile::kLegacy;
}

int RoutedOutputTile(RoutedGemm1Profile profile) {
  switch (profile) {
    case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
      return kRoutedLargeOutputTile;
    case RoutedGemm1Profile::kLegacy:
    case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
    case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
    case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue:
      return kPlannedOutputTile;
  }
  return kPlannedOutputTile;
}

int RoutedOutputTile(RoutedGemm2Profile profile) {
  switch (profile) {
    case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
      return kRoutedLargeOutputTile;
    case RoutedGemm2Profile::kLegacy:
    case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
    case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
      return kPlannedOutputTile;
  }
  return kPlannedOutputTile;
}

__global__ void ZeroBufferKernel(float* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = 0.0f;
}

__global__ void ZeroBf16BufferKernel(__nv_bfloat16* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = __float2bfloat16(0.0f);
}

__global__ void FillFloatBufferKernel(float* data, std::size_t count, float value) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = value;
}

__global__ void GatherRowsKernel(
    const float* input,
    const int* row_indices,
    const int* active_output_rows,
    float* output,
    std::size_t output_row_capacity,
    std::size_t input_rows,
    std::size_t cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = output_row_capacity * cols;
  if (index >= count) {
    return;
  }

  const std::size_t row = index / cols;
  if (active_output_rows != nullptr &&
      row >= static_cast<std::size_t>(active_output_rows[0])) {
    return;
  }
  const std::size_t col = index % cols;
  const int input_row = row_indices[row];
  if (input_row < 0 || static_cast<std::size_t>(input_row) >= input_rows) {
    output[index] = 0.0f;
    return;
  }
  output[index] = input[static_cast<std::size_t>(input_row) * cols + col];
}

__global__ void QuantizeDequantizeRowsKernel(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= row_capacity) {
    return;
  }
  if (active_row_count != nullptr &&
      row >= static_cast<std::size_t>(active_row_count[0])) {
    return;
  }
  fused_decode::QuantizeDequantizeNvfp4Row(
      data + row * cols,
      data + row * cols,
      cols);
}

__global__ void Relu2Kernel(float* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = fused_decode::Relu2(data[index]);
}

__global__ void Relu2RowsKernel(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = row_capacity * cols;
  if (index >= count) {
    return;
  }
  const std::size_t row = index / cols;
  if (active_row_count != nullptr &&
      row >= static_cast<std::size_t>(active_row_count[0])) {
    return;
  }
  data[index] = fused_decode::Relu2(data[index]);
}

__device__ float ClampNvfp4TensorScale(float value) {
  if (!isfinite(value) || value < kNvfp4MinTensorScale) {
    return kNvfp4MinTensorScale;
  }
  return value;
}

__global__ void ComputeExpertActivationScalesKernel(
    const float* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x);
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      expert_index >= n_experts) {
    return;
  }

  const int row_begin = expert_first_token_offsets[expert_index];
  const int row_end = expert_first_token_offsets[expert_index + 1];
  if (row_end <= row_begin) {
    if (threadIdx.x == 0) {
      output_scales[expert_index] = 1.0f;
    }
    return;
  }

  float thread_max_abs = 0.0f;
  const std::size_t row_count = static_cast<std::size_t>(row_end - row_begin);
  const std::size_t numel = row_count * cols;
  for (std::size_t linear_index = threadIdx.x;
       linear_index < numel;
       linear_index += blockDim.x) {
    const std::size_t row = linear_index / cols;
    const std::size_t col = linear_index % cols;
    const float value = input[(static_cast<std::size_t>(row_begin) + row) * cols + col];
    const float abs_value = fabsf(value);
    if (abs_value > thread_max_abs) {
      thread_max_abs = abs_value;
    }
  }

  __shared__ float shared_max_abs[fused_decode::kThreadsPerBlock];
  shared_max_abs[threadIdx.x] = thread_max_abs;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max_abs[threadIdx.x + stride] > shared_max_abs[threadIdx.x]) {
      shared_max_abs[threadIdx.x] = shared_max_abs[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float tensor_scale = 1.0f;
    if (shared_max_abs[0] > kNvfp4ActivationMaxFinite) {
      tensor_scale = ClampNvfp4TensorScale(shared_max_abs[0] / kNvfp4ActivationMaxFinite);
    }
    output_scales[expert_index] = tensor_scale;
  }
}

__global__ void ComputeExpertActivationScalesBf16Kernel(
    const __nv_bfloat16* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x);
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      expert_index >= n_experts) {
    return;
  }

  const int row_begin = expert_first_token_offsets[expert_index];
  const int row_end = expert_first_token_offsets[expert_index + 1];
  if (row_end <= row_begin) {
    if (threadIdx.x == 0) {
      output_scales[expert_index] = 1.0f;
    }
    return;
  }

  float thread_max_abs = 0.0f;
  const std::size_t row_count = static_cast<std::size_t>(row_end - row_begin);
  const std::size_t numel = row_count * cols;
  for (std::size_t linear_index = threadIdx.x;
       linear_index < numel;
       linear_index += blockDim.x) {
    const std::size_t row = linear_index / cols;
    const std::size_t col = linear_index % cols;
    const float value = fused_decode::Relu2(__bfloat162float(
        input[(static_cast<std::size_t>(row_begin) + row) * cols + col]));
    const float abs_value = value;
    if (abs_value > thread_max_abs) {
      thread_max_abs = abs_value;
    }
  }

  __shared__ float shared_max_abs[fused_decode::kThreadsPerBlock];
  shared_max_abs[threadIdx.x] = thread_max_abs;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max_abs[threadIdx.x + stride] > shared_max_abs[threadIdx.x]) {
      shared_max_abs[threadIdx.x] = shared_max_abs[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float tensor_scale = 1.0f;
    if (shared_max_abs[0] > kNvfp4ActivationMaxFinite) {
      tensor_scale = ClampNvfp4TensorScale(shared_max_abs[0] / kNvfp4ActivationMaxFinite);
    }
    output_scales[expert_index] = tensor_scale;
  }
}

__global__ void MaxBitsToTensorScalesKernel(
    const unsigned int* max_bits,
    std::size_t count,
    float* output_scales) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float max_abs = __uint_as_float(max_bits[index]);
  float tensor_scale = 1.0f;
  if (max_abs > kNvfp4ActivationMaxFinite) {
    tensor_scale = ClampNvfp4TensorScale(max_abs / kNvfp4ActivationMaxFinite);
  }
  output_scales[index] = tensor_scale;
}

__device__ __forceinline__ int FindExpertForPaddedRow(
    const int* padded_expert_offsets,
    int n_experts,
    int row) {
  int low = 0;
  int high = n_experts;
  while (low < high) {
    const int mid = low + ((high - low) >> 1);
    if (padded_expert_offsets[mid + 1] <= row) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

template <int kProcessRows>
__global__ __launch_bounds__(fused_decode::kThreadsPerBlock)
void RoutedBf16Relu2PackKernel(
    const __nv_bfloat16* source,
    const int* actual_expert_offsets,
    const int* padded_expert_offsets,
    int n_experts,
    std::size_t cols,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    float* output_dequant_scales,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales) {
  cudaGridDependencySynchronize();

  if (source == nullptr ||
      actual_expert_offsets == nullptr ||
      padded_expert_offsets == nullptr ||
      output_dequant_scales == nullptr ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr ||
      cols == 0 ||
      (cols % fused_decode::kNvfp4BlockWidth) != 0 ||
      n_experts <= 0) {
    return;
  }

  const int64_t total_padded_rows = padded_expert_offsets[n_experts];
  if (total_padded_rows <= 0) {
    return;
  }

  const int64_t blocks_per_row =
      static_cast<int64_t>(cols / fused_decode::kNvfp4BlockWidth);
  const int64_t block_col = static_cast<int64_t>(blockIdx.y);
  if (block_col >= blocks_per_row) {
    return;
  }

  constexpr int64_t rows_per_cta =
      static_cast<int64_t>(kProcessRows) * fused_decode::kThreadsPerBlock;
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * rows_per_cta;

  for (int64_t row_base =
           static_cast<int64_t>(blockIdx.x) * rows_per_cta + threadIdx.x;
       row_base < total_padded_rows;
       row_base += grid_stride) {
    int expert = FindExpertForPaddedRow(
        padded_expert_offsets,
        n_experts,
        static_cast<int>(row_base));

#pragma unroll
    for (int row_iter = 0; row_iter < kProcessRows; ++row_iter) {
      const int64_t row =
          row_base + static_cast<int64_t>(row_iter) * fused_decode::kThreadsPerBlock;
      if (row >= total_padded_rows) {
        break;
      }

      while (row_iter > 0 &&
             (expert + 1) < n_experts &&
             padded_expert_offsets[expert + 1] <= row) {
        ++expert;
      }

      const int64_t padded_row_begin = padded_expert_offsets[expert];
      const int64_t actual_row_begin = actual_expert_offsets[expert];
      const int64_t actual_row_end = actual_expert_offsets[expert + 1];
      const int64_t active_row_end =
          padded_row_begin + (actual_row_end - actual_row_begin);

      const std::size_t scale_index =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(blocks_per_row) +
          static_cast<std::size_t>(block_col);
      const std::size_t matmul_scale_index = ExecutionScaleOffset(
          static_cast<std::size_t>(row),
          static_cast<std::size_t>(block_col),
          padded_blocks_per_row,
          scale_layout);
      const std::size_t packed_offset =
          static_cast<std::size_t>(row) * (cols / 2u) +
          static_cast<std::size_t>(block_col) *
              (fused_decode::kNvfp4BlockWidth / 2u);

      if (row >= active_row_end) {
        output_dequant_scales[scale_index] = 0.0f;
        output_block_scales[scale_index] = 0u;
        output_matmul_scales[matmul_scale_index] = 0u;
        for (std::size_t pair = 0; pair < (fused_decode::kNvfp4BlockWidth / 2u); ++pair) {
          output_packed[packed_offset + pair] = 0u;
        }
        continue;
      }

      const std::size_t input_offset =
          static_cast<std::size_t>(row) * cols +
          static_cast<std::size_t>(block_col) * fused_decode::kNvfp4BlockWidth;
      float activated[fused_decode::kNvfp4BlockWidth];
      float block_max_abs = 0.0f;
      for (std::size_t col = 0; col < fused_decode::kNvfp4BlockWidth; ++col) {
        const float value = __bfloat162float(source[input_offset + col]);
        const float relu2 = fused_decode::Relu2(value);
        activated[col] = relu2;
        if (relu2 > block_max_abs) {
          block_max_abs = relu2;
        }
      }

      float block_scale = 1.0f;
      if (block_max_abs > 0.0f) {
        block_scale = fused_decode::ClampNvfp4Scale(
            block_max_abs / fused_decode::kNvfp4Fp4MaxFinite);
      }
      const std::uint8_t encoded_block_scale =
          fused_decode::EncodeFp8Scale(block_scale);
      output_dequant_scales[scale_index] = block_scale;
      output_block_scales[scale_index] = encoded_block_scale;
      output_matmul_scales[matmul_scale_index] = encoded_block_scale;

      for (std::size_t pair = 0; pair < (fused_decode::kNvfp4BlockWidth / 2u); ++pair) {
        const std::uint8_t lhs =
            fused_decode::EncodeFp4(activated[pair * 2u] / block_scale);
        const std::uint8_t rhs =
            fused_decode::EncodeFp4(activated[pair * 2u + 1u] / block_scale);
        output_packed[packed_offset + pair] =
            static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
      }
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

__device__ __forceinline__ std::uint8_t LoadExecutionScaleByte(
    const std::uint8_t* scale_bytes,
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout);

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const FusedNvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col);

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const fused_decode::Nvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col);

__device__ __forceinline__ float DecodeNvfp4WeightElement(
    const FusedNvfp4WeightView& weight,
    std::size_t output_row,
    std::size_t col) {
  const std::size_t pairs_per_row = weight.input_cols / 2u;
  const std::size_t pair_index = col / 2u;
  const std::size_t block = col / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * pairs_per_row;
  const float block_scale = LoadNvfp4WeightBlockScale(weight, output_row, block);
  const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
  const std::uint8_t nibble =
      (col & 1u) == 0u ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu);
  return fused_decode::DecodeFp4(nibble) * block_scale;
}

__device__ __forceinline__ float DecodeGroupedPackedInputElement(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_dq_scales,
    float input_tensor_scale,
    std::size_t cols,
    std::size_t input_row,
    std::size_t col) {
  const std::size_t pairs_per_row = cols / 2u;
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t pair_index = col / 2u;
  const std::size_t block = col / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = input_row * pairs_per_row;
  const std::size_t scale_row_offset = input_row * blocks_per_row;
  const float block_scale =
      input_dq_scales != nullptr
          ? input_dq_scales[scale_row_offset + block]
          : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                input_tensor_scale;
  const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
  const std::uint8_t nibble =
      (col & 1u) == 0u ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu);
  return fused_decode::DecodeFp4(nibble) * block_scale;
}

__device__ __forceinline__ void ZeroBf16Block16(__nv_bfloat16* output) {
#pragma unroll
  for (int i = 0; i < static_cast<int>(fused_decode::kNvfp4BlockWidth); ++i) {
    output[i] = __float2bfloat16(0.0f);
  }
}

__device__ __forceinline__ void DecodePackedNvfp4BlockToBf16(
    const std::uint8_t* packed_block,
    float block_scale,
    __nv_bfloat16* output) {
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t packed = packed_block[pair];
    output[pair * 2] =
        __float2bfloat16(fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale);
    output[pair * 2 + 1] =
        __float2bfloat16(fused_decode::DecodeFp4((packed >> 4u) & 0x0Fu) * block_scale);
  }
}

__device__ __forceinline__ void DecodeNvfp4WeightBlockBf16(
    const FusedNvfp4WeightView& weight,
    std::size_t output_row,
    std::size_t block_index,
    __nv_bfloat16* output) {
  const std::size_t blocks_per_row =
      weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset =
      output_row * (weight.input_cols / 2u) + block_index * (fused_decode::kNvfp4BlockWidth / 2u);
  const float block_scale = LoadNvfp4WeightBlockScale(weight, output_row, block_index);
  DecodePackedNvfp4BlockToBf16(weight.packed_data + packed_row_offset, block_scale, output);
}

__device__ __forceinline__ void DecodeGroupedPackedInputBlockBf16(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_dq_scales,
    float input_tensor_scale,
    std::size_t cols,
    std::size_t input_row,
    std::size_t block_index,
    __nv_bfloat16* output) {
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset =
      input_row * (cols / 2u) + block_index * (fused_decode::kNvfp4BlockWidth / 2u);
  const std::size_t scale_row_offset = input_row * blocks_per_row + block_index;
  const float block_scale =
      input_dq_scales != nullptr
          ? input_dq_scales[scale_row_offset]
          : fused_decode::DecodeFp8(input_block_scales[scale_row_offset]) * input_tensor_scale;
  DecodePackedNvfp4BlockToBf16(packed_input + packed_row_offset, block_scale, output);
}

__device__ __forceinline__ std::uint8_t LoadPackedFp4Nibble(
    const std::uint8_t* packed_row,
    int col) {
  const std::uint8_t packed = packed_row[col >> 1];
  return static_cast<std::uint8_t>(((col & 1) == 0) ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu));
}

__device__ __forceinline__ std::uint32_t PackFp4Register8(
    std::uint8_t v0,
    std::uint8_t v1,
    std::uint8_t v2,
    std::uint8_t v3,
    std::uint8_t v4,
    std::uint8_t v5,
    std::uint8_t v6,
    std::uint8_t v7) {
  std::uint32_t reg = static_cast<std::uint32_t>(v0 & 0x0Fu);
  reg |= static_cast<std::uint32_t>(v1 & 0x0Fu) << 4u;
  reg |= static_cast<std::uint32_t>(v2 & 0x0Fu) << 8u;
  reg |= static_cast<std::uint32_t>(v3 & 0x0Fu) << 12u;
  reg |= static_cast<std::uint32_t>(v4 & 0x0Fu) << 16u;
  reg |= static_cast<std::uint32_t>(v5 & 0x0Fu) << 20u;
  reg |= static_cast<std::uint32_t>(v6 & 0x0Fu) << 24u;
  reg |= static_cast<std::uint32_t>(v7 & 0x0Fu) << 28u;
  return reg;
}

__device__ __forceinline__ std::uint32_t PackScaleWord4(
    std::uint8_t s0,
    std::uint8_t s1,
    std::uint8_t s2,
    std::uint8_t s3) {
  std::uint32_t reg = static_cast<std::uint32_t>(s0);
  reg |= static_cast<std::uint32_t>(s1) << 8u;
  reg |= static_cast<std::uint32_t>(s2) << 16u;
  reg |= static_cast<std::uint32_t>(s3) << 24u;
  return reg;
}

__device__ __forceinline__ std::uint32_t MakeP13ProbeScaleWord(int row) {
  const std::uint8_t base = static_cast<std::uint8_t>(row & 0xff);
  return PackScaleWord4(
      base,
      static_cast<std::uint8_t>(base + 1u),
      static_cast<std::uint8_t>(base + 2u),
      static_cast<std::uint8_t>(base + 3u));
}

template <class T>
__device__ __forceinline__ std::uint8_t ScaleByteValue(T const& value) {
  using ValueType = std::remove_cvref_t<T>;
  if constexpr (std::is_integral_v<ValueType>) {
    return static_cast<std::uint8_t>(value);
  } else {
    return static_cast<std::uint8_t>(value.raw());
  }
}

template <int kWordCount, class CopyViewTensor>
__device__ __forceinline__ void PackLinearFp4CopyViewWords(
    CopyViewTensor const& copy_view,
    std::uint32_t (&words)[kWordCount]) {
  constexpr int kPhysicalCount = decltype(cute::size(copy_view))::value;
  static_assert(
      kPhysicalCount == kWordCount * 8,
      "Packed FP4 copy view size must match the requested word count");
#pragma unroll
  for (int word = 0; word < kWordCount; ++word) {
    std::uint32_t packed = 0u;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int physical = word * 8 + elem;
      packed |= static_cast<std::uint32_t>(ScaleByteValue(copy_view(physical)) & 0x0Fu)
                << (elem * 4);
    }
    words[word] = packed;
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void FindScaleTensorCoordByRaw(
    ScaleTensor const& scale_tensor,
    std::uint8_t raw,
    int& row_out,
    int& col_out) {
  row_out = -1;
  col_out = -1;
  for (int row = 0; row < static_cast<int>(cute::size<0>(ScaleTensor{})); ++row) {
    for (int col = 0; col < static_cast<int>(cute::size<1>(ScaleTensor{})); ++col) {
      if (ScaleByteValue(scale_tensor(row, col, cute::Int<0>{})) == raw) {
        row_out = row;
        col_out = col;
        return;
      }
    }
  }
}

__device__ __forceinline__ void ApplySm120Fp4ShiftA(
    std::uint32_t& a0,
    std::uint32_t& a1,
    std::uint32_t& a2,
    std::uint32_t& a3) {
  a0 <<= 2u;
  a1 <<= 2u;
  a2 <<= 2u;
  a3 <<= 2u;
}

__device__ __forceinline__ void ApplySm120Fp4ShiftB(
    std::uint32_t& b0,
    std::uint32_t& b1) {
  b0 <<= 2u;
  b1 <<= 2u;
}

__device__ __forceinline__ std::uint32_t LoadExecutionScaleWord(
    const std::uint8_t* scale_bytes,
    std::size_t row,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  return PackScaleWord4(
      scale_bytes[ExecutionScaleOffset(row, block_base + 0u, padded_blocks_per_row, scale_layout)],
      scale_bytes[ExecutionScaleOffset(row, block_base + 1u, padded_blocks_per_row, scale_layout)],
      scale_bytes[ExecutionScaleOffset(row, block_base + 2u, padded_blocks_per_row, scale_layout)],
      scale_bytes[ExecutionScaleOffset(row, block_base + 3u, padded_blocks_per_row, scale_layout)]);
}

__device__ __forceinline__ std::uint8_t LoadExecutionScaleByte(
    const std::uint8_t* scale_bytes,
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  return scale_bytes[ExecutionScaleOffset(row, block_col, padded_blocks_per_row, scale_layout)];
}

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const FusedNvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col) {
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  if (weight.block_scales_data != nullptr) {
    return fused_decode::DecodeFp8(weight.block_scales_data[row * blocks_per_row + block_col]) *
           tensor_scale;
  }
  const std::size_t padded_blocks_per_row = (blocks_per_row + 3u) & ~std::size_t{3u};
  return fused_decode::DecodeFp8(
             LoadExecutionScaleByte(
                 weight.matmul_block_scales_data,
                 row,
                 block_col,
                 padded_blocks_per_row,
                 Nvfp4ScaleLayout::kSwizzled128x4)) *
         tensor_scale;
}

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const fused_decode::Nvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col) {
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  if (weight.block_scales_data != nullptr) {
    return fused_decode::DecodeFp8(weight.block_scales_data[row * blocks_per_row + block_col]) *
           tensor_scale;
  }
  const std::size_t padded_blocks_per_row = (blocks_per_row + 3u) & ~std::size_t{3u};
  return fused_decode::DecodeFp8(
             LoadExecutionScaleByte(
                 weight.matmul_block_scales_data,
                 row,
                 block_col,
                 padded_blocks_per_row,
                 Nvfp4ScaleLayout::kSwizzled128x4)) *
         tensor_scale;
}

__device__ __forceinline__ void Sm120BlockScaledFp4Mma(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    std::uint32_t a0,
    std::uint32_t a1,
    std::uint32_t a2,
    std::uint32_t a3,
    std::uint32_t b0,
    std::uint32_t b1,
    std::uint32_t sfa,
    std::uint32_t sfb) {
  static constexpr std::uint16_t kBidA = 0;
  static constexpr std::uint16_t kTidA = 0;
  static constexpr std::uint16_t kBidB = 0;
  static constexpr std::uint16_t kTidB = 0;
  const float c0 = d0;
  const float c1 = d1;
  const float c2 = d2;
  const float c3 = d3;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa), "h"(kBidA), "h"(kTidA),
        "r"(sfb), "h"(kBidB), "h"(kTidB));
}

template <int kRowsPerTile>
__device__ __forceinline__ void ZeroPackedTileRows(
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words,
    int row) {
  if (row >= kRowsPerTile) {
    return;
  }
#pragma unroll
  for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
    packed_rows[row * (64 / 2) + byte_index] = 0u;
  }
  if (scale_words != nullptr) {
    scale_words[row] = 0u;
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void CopyPackedTileRow64(
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* matmul_scales,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words,
    int row) {
  if (row >= kRowsPerTile) {
    return;
  }
  const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
  std::uint8_t* dst = packed_rows + row * (64 / 2);
#pragma unroll
  for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
    dst[byte_index] = packed_data[src_offset + static_cast<std::size_t>(byte_index)];
  }
  if (scale_words != nullptr) {
    scale_words[row] = LoadExecutionScaleWord(
        matmul_scales,
        source_row,
        block_base,
        padded_blocks_per_row,
        scale_layout);
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void CopyPackedTileRow64FromRowMajorScales(
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* block_scales,
    std::size_t block_base,
    std::size_t blocks_per_row,
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words,
    int row) {
  if (row >= kRowsPerTile) {
    return;
  }
  const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
  std::uint8_t* dst = packed_rows + row * (64 / 2);
#pragma unroll
  for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
    dst[byte_index] = packed_data[src_offset + static_cast<std::size_t>(byte_index)];
  }
  if (scale_words != nullptr) {
    const std::size_t scale_offset = source_row * blocks_per_row + block_base;
    scale_words[row] = PackScaleWord4(
        block_scales[scale_offset + 0u],
        block_scales[scale_offset + 1u],
        block_scales[scale_offset + 2u],
        block_scales[scale_offset + 3u]);
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void LoadFp4ARegistersRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int lane_id,
    std::uint32_t& a0,
    std::uint32_t& a1,
    std::uint32_t& a2,
    std::uint32_t& a3,
    std::uint32_t& sfa) {
  const int col_base = lane_id >> 2;
  const int row_pair_base = (lane_id & 3) * 2;
  const std::uint8_t* row0 = packed_rows + row_pair_base * (64 / 2);
  const std::uint8_t* row1 = packed_rows + (row_pair_base + 1) * (64 / 2);
  const std::uint8_t* row8 = packed_rows + (row_pair_base + 8) * (64 / 2);
  const std::uint8_t* row9 = packed_rows + (row_pair_base + 9) * (64 / 2);
  a0 = PackFp4Register8(
      LoadPackedFp4Nibble(row0, col_base + 0),
      LoadPackedFp4Nibble(row0, col_base + 16),
      LoadPackedFp4Nibble(row0, col_base + 32),
      LoadPackedFp4Nibble(row0, col_base + 48),
      LoadPackedFp4Nibble(row1, col_base + 0),
      LoadPackedFp4Nibble(row1, col_base + 16),
      LoadPackedFp4Nibble(row1, col_base + 32),
      LoadPackedFp4Nibble(row1, col_base + 48));
  a1 = PackFp4Register8(
      LoadPackedFp4Nibble(row0, col_base + 8),
      LoadPackedFp4Nibble(row0, col_base + 24),
      LoadPackedFp4Nibble(row0, col_base + 40),
      LoadPackedFp4Nibble(row0, col_base + 56),
      LoadPackedFp4Nibble(row1, col_base + 8),
      LoadPackedFp4Nibble(row1, col_base + 24),
      LoadPackedFp4Nibble(row1, col_base + 40),
      LoadPackedFp4Nibble(row1, col_base + 56));
  a2 = PackFp4Register8(
      LoadPackedFp4Nibble(row8, col_base + 0),
      LoadPackedFp4Nibble(row8, col_base + 16),
      LoadPackedFp4Nibble(row8, col_base + 32),
      LoadPackedFp4Nibble(row8, col_base + 48),
      LoadPackedFp4Nibble(row9, col_base + 0),
      LoadPackedFp4Nibble(row9, col_base + 16),
      LoadPackedFp4Nibble(row9, col_base + 32),
      LoadPackedFp4Nibble(row9, col_base + 48));
  a3 = PackFp4Register8(
      LoadPackedFp4Nibble(row8, col_base + 8),
      LoadPackedFp4Nibble(row8, col_base + 24),
      LoadPackedFp4Nibble(row8, col_base + 40),
      LoadPackedFp4Nibble(row8, col_base + 56),
      LoadPackedFp4Nibble(row9, col_base + 8),
      LoadPackedFp4Nibble(row9, col_base + 24),
      LoadPackedFp4Nibble(row9, col_base + 40),
      LoadPackedFp4Nibble(row9, col_base + 56));
  const int scale_row = (lane_id >> 2) + ((lane_id & 1) ? 8 : 0);
  sfa = scale_words[scale_row];
}

template <int kRowsPerTile>
__device__ __forceinline__ void LoadFp4BRegistersColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int lane_id,
    int n_base,
    std::uint32_t& b0,
    std::uint32_t& b1,
    std::uint32_t& sfb) {
  const int row_group = lane_id & 3;
  const int k_base = lane_id >> 2;
  const std::uint8_t* row0 = packed_rows + (n_base + row_group) * (64 / 2);
  const std::uint8_t* row4 = packed_rows + (n_base + row_group + 4) * (64 / 2);
  b0 = PackFp4Register8(
      LoadPackedFp4Nibble(row0, k_base + 0),
      LoadPackedFp4Nibble(row0, k_base + 8),
      LoadPackedFp4Nibble(row0, k_base + 16),
      LoadPackedFp4Nibble(row0, k_base + 24),
      LoadPackedFp4Nibble(row0, k_base + 32),
      LoadPackedFp4Nibble(row0, k_base + 40),
      LoadPackedFp4Nibble(row0, k_base + 48),
      LoadPackedFp4Nibble(row0, k_base + 56));
  b1 = PackFp4Register8(
      LoadPackedFp4Nibble(row4, k_base + 0),
      LoadPackedFp4Nibble(row4, k_base + 8),
      LoadPackedFp4Nibble(row4, k_base + 16),
      LoadPackedFp4Nibble(row4, k_base + 24),
      LoadPackedFp4Nibble(row4, k_base + 32),
      LoadPackedFp4Nibble(row4, k_base + 40),
      LoadPackedFp4Nibble(row4, k_base + 48),
      LoadPackedFp4Nibble(row4, k_base + 56));
  sfb = scale_words[n_base + (lane_id >> 2)];
}

template <int kRowsPerTile>
__device__ __forceinline__ void StoreFp4AccumulatorTileRowMajor16x8(
    float alpha,
    const float c0,
    const float c1,
    const float c2,
    const float c3,
    int lane_id,
    int col_base,
    int row_base,
    int valid_rows,
    int valid_cols,
    std::size_t row_start,
    std::size_t output_row_base,
    std::size_t output_rows_per_expert,
    float* output) {
  if (col_base >= valid_cols) {
    return;
  }
  const int row_group = (lane_id >> 3) * 4;
  const int row0 = row_group + 0;
  const int row1 = row_group + 2;
  const int row2 = row_group + 1;
  const int row3 = row_group + 3;
  if (row0 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row0)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c0 * alpha;
  }
  if (row1 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row1)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c1 * alpha;
  }
  if (row2 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row2)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c2 * alpha;
  }
  if (row3 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row3)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c3 * alpha;
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void StoreFp4AccumulatorTileRowMajor16x8(
    float alpha,
    const float c0,
    const float c1,
    const float c2,
    const float c3,
    int lane_id,
    int col_base,
    int row_base,
    int valid_rows,
    int valid_cols,
    std::size_t row_start,
    std::size_t output_row_base,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (col_base >= valid_cols) {
    return;
  }
  const int row_group = (lane_id >> 3) * 4;
  const int row0 = row_group + 0;
  const int row1 = row_group + 2;
  const int row2 = row_group + 1;
  const int row3 = row_group + 3;
  if (row0 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row0)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c0 * alpha);
  }
  if (row1 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row1)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c1 * alpha);
  }
  if (row2 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row2)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c2 * alpha);
  }
  if (row3 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row3)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c3 * alpha);
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::PackedTile64<kRowsPerTile>
nvfp4_bridge::MakePackedTile64(
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words) {
  return PackedTile64<kRowsPerTile>{packed_rows, scale_words};
}

template <int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::ZeroRow(
    const PackedTile64<kRowsPerTile>& tile,
    int row) {
  ZeroPackedTileRows<kRowsPerTile>(tile.packed_rows, tile.scale_words, row);
}

template <int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::CopyWeightRow64(
    const PackedTile64<kRowsPerTile>& tile,
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* matmul_scales,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    int row) {
  CopyPackedTileRow64<kRowsPerTile>(
      packed_data,
      packed_row_bytes,
      source_row,
      packed_byte_offset,
      matmul_scales,
      block_base,
      padded_blocks_per_row,
      scale_layout,
      tile.packed_rows,
      tile.scale_words,
      row);
}

template <int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::CopyActivationRow64(
    const PackedTile64<kRowsPerTile>& tile,
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* block_scales,
    std::size_t block_base,
    std::size_t blocks_per_row,
    int row) {
  CopyPackedTileRow64FromRowMajorScales<kRowsPerTile>(
      packed_data,
      packed_row_bytes,
      source_row,
      packed_byte_offset,
      block_scales,
      block_base,
      blocks_per_row,
      tile.packed_rows,
      tile.scale_words,
      row);
}

template <class ScaleTensor>
__device__ __forceinline__ void StoreScaleTensorByte(
    ScaleTensor& scale_tensor,
    int row,
    int scale_col,
    std::uint8_t value) {
  if constexpr (std::remove_cvref_t<ScaleTensor>::rank == 2) {
    auto elem = scale_tensor(row, scale_col);
    if constexpr (requires { elem.storage; }) {
      scale_tensor(row, scale_col).storage = value;
    } else {
      scale_tensor(row, scale_col) = value;
    }
  } else {
    static_assert(std::remove_cvref_t<ScaleTensor>::rank == 3);
    auto elem = scale_tensor(row, scale_col, cute::Int<0>{});
    if constexpr (requires { elem.storage; }) {
      scale_tensor(row, scale_col, cute::Int<0>{}).storage = value;
    } else {
      scale_tensor(row, scale_col, cute::Int<0>{}) = value;
    }
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::ZeroTracedP5ScaleRow(
    ScaleTensor& scale_tensor,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
#pragma unroll
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    StoreScaleTensorByte(scale_tensor, row, scale_col, 0u);
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedP5ScaleWordK64(
    ScaleTensor& scale_tensor,
    std::uint32_t scale_word,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
#pragma unroll
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    const std::uint8_t value = scale_col < 4 ? LoadScaleByte(scale_word, scale_col) : std::uint8_t{0};
    StoreScaleTensorByte(scale_tensor, row, scale_col, value);
  }
}

template <class ScaleTensor, int kScaleByteCount>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedScaleBytes(
    ScaleTensor& scale_tensor,
    const std::uint8_t (&scale_bytes)[kScaleByteCount],
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
  const int logical_cols = static_cast<int>(cute::size<1>(scale_tensor));
  const int segment_len = logical_cols > 0 ? max(1, logical_cols / kScaleByteCount) : 1;
#pragma unroll
  for (int scale_col = 0; scale_col < logical_cols; ++scale_col) {
    const int byte_index = min(scale_col / segment_len, kScaleByteCount - 1);
    StoreScaleTensorByte(scale_tensor, row, scale_col, scale_bytes[byte_index]);
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedScaleWordsK128(
    ScaleTensor& scale_tensor,
    std::uint32_t scale_word_lo,
    std::uint32_t scale_word_hi,
    int row) {
  const std::uint8_t scale_bytes[8] = {
      LoadScaleByte(scale_word_lo, 0),
      LoadScaleByte(scale_word_lo, 1),
      LoadScaleByte(scale_word_lo, 2),
      LoadScaleByte(scale_word_lo, 3),
      LoadScaleByte(scale_word_hi, 0),
      LoadScaleByte(scale_word_hi, 1),
      LoadScaleByte(scale_word_hi, 2),
      LoadScaleByte(scale_word_hi, 3),
  };
  StoreTracedScaleBytes(scale_tensor, scale_bytes, row);
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int row_base,
    int thread_idx) {
  AFragment64 fragment{};
  int row_coords[kTiledCopyCoordCapacityA];
  int col_coords[kTiledCopyCoordCapacityA];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_a = cute::make_identity_tensor(
      cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::size<2>(typename TiledMma::AtomShape_MNK{})));
  auto part_a = thr_mma.partition_A(ref_a);
  auto smem_tiled_copy_a = MakeLocalTiledCopyA<cute::SM75_U32x4_LDSM_N>(mma);
  auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(thread_idx);
  auto copy_view_a = smem_thr_copy_a.retile_D(part_a);
  FillPhysicalCoordMapCopyViewLimited(copy_view_a, 32, row_coords, col_coords);
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    std::uint32_t packed = 0;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int physical = reg * 8 + elem;
      const int row = row_coords[physical];
      const int col = col_coords[physical];
      const std::uint8_t nibble =
          LoadPackedFp4Nibble(packed_rows + static_cast<std::size_t>(row_base + row) * (64 / 2), col);
      packed |= static_cast<std::uint32_t>(nibble) << (elem * 4);
    }
    fragment.regs[reg] = packed;
  }
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  std::uint32_t packed_scale = 0u;
  if (scale_words != nullptr) {
    auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
    auto smem_tiled_copy_sfa = MakeLocalTiledCopySFA(mma);
    auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
    auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
    int scale_rows[kTiledCopyCoordCapacityScale];
    int scale_cols[kTiledCopyCoordCapacityScale];
    FillPhysicalCoordMapCopyViewLimited(copy_view_sfa, 4, scale_rows, scale_cols);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int scale_row = scale_rows[i];
      const int scale_col = scale_cols[i];
      packed_scale |= static_cast<std::uint32_t>(LoadScaleByte(scale_words[scale_row], scale_col)) << (i * 8);
    }
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  ApplySm120Fp4ShiftA(fragment.regs[0], fragment.regs[1], fragment.regs[2], fragment.regs[3]);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64TracedScaleTiled(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int row_base,
    int thread_idx) {
  auto fragment =
      LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, row_base, thread_idx);
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP5SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP5TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP5ScaleFragmentCosizeA];
  int scale_cols[kTracedP5ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP5ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  const int fragment_index = row_base / 16;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int row_base,
    int lane_id) {
  return LoadFragmentA_RowMajor16x64Tiled<SingleAtomTiledMma, kRowsPerTile>(
      packed_rows,
      scale_words,
      row_base,
      lane_id);
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int thread_idx,
    int n_base) {
  BFragment64 fragment{};
  int row_coords[kTiledCopyCoordCapacityB];
  int col_coords[kTiledCopyCoordCapacityB];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_b = cute::make_identity_tensor(
      cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::size<2>(typename TiledMma::AtomShape_MNK{})));
  auto part_b = thr_mma.partition_B(ref_b);
  auto smem_tiled_copy_b = MakeLocalTiledCopyB<cute::SM75_U32x4_LDSM_N>(mma);
  auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(thread_idx);
  auto copy_view_b = smem_thr_copy_b.retile_D(part_b);
  FillPhysicalCoordMapCopyViewLimited(copy_view_b, 16, row_coords, col_coords);
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    std::uint32_t packed = 0;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int physical = reg * 8 + elem;
      const int row = row_coords[physical];
      const int col = col_coords[physical];
      const std::uint8_t nibble =
          LoadPackedFp4Nibble(packed_rows + static_cast<std::size_t>(n_base + row) * (64 / 2), col);
      packed |= static_cast<std::uint32_t>(nibble) << (elem * 4);
    }
    fragment.regs[reg] = packed;
  }
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  std::uint32_t packed_scale = 0u;
  if (scale_words != nullptr) {
    auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
    auto smem_tiled_copy_sfb = MakeLocalTiledCopySFB(mma);
    auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
    auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
    int scale_rows[kTiledCopyCoordCapacityScale];
    int scale_cols[kTiledCopyCoordCapacityScale];
    FillPhysicalCoordMapCopyViewLimited(copy_view_sfb, 4, scale_rows, scale_cols);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int scale_row = scale_rows[i];
      const int scale_col = scale_cols[i];
      packed_scale |= static_cast<std::uint32_t>(LoadScaleByte(scale_words[n_base + scale_row], scale_col)) << (i * 8);
    }
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  ApplySm120Fp4ShiftB(fragment.regs[0], fragment.regs[1]);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8TracedScaleTiled(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int n_base) {
  auto fragment =
      LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, thread_idx, n_base);
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP5SmemLayoutSFB{});
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = MakeTracedP5TiledCopySFB(mma);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP5ScaleFragmentCosizeB];
  int scale_cols[kTracedP5ScaleFragmentCosizeB];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfb,
      kTracedP5ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  const int fragment_index = n_base / 8;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8TracedScaleTiledP13(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int n_base) {
  auto fragment =
      LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, thread_idx, n_base);
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP13SmemLayoutSFB{});
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = MakeTracedP13TiledCopySFB(mma);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP13ScaleFragmentCosizeB];
  int scale_cols[kTracedP13ScaleFragmentCosizeB];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfb,
      kTracedP13ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  const int fragment_index = n_base / 8;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64TracedScaleTiledP13(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int row_base,
    int thread_idx) {
  auto fragment =
      LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, row_base, thread_idx);
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP13SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP13TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP13ScaleFragmentCosizeA];
  int scale_cols[kTracedP13ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP13ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  const int fragment_index = row_base / 16;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP5AFragmentsRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    AFragment64 (&fragments)[kTracedP5MFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP5SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP5TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP5ScaleFragmentCosizeA];
  int scale_cols[kTracedP5ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP5ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  for (int m_fragment = 0; m_fragment < kTracedP5MFragments; ++m_fragment) {
    fragments[m_fragment] =
        LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            m_fragment * 16,
            thread_idx);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = m_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[m_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP13AFragmentsRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int row_base,
    AFragment64 (&fragments)[kTracedP13MFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP13SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP13TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP13ScaleFragmentCosizeA];
  int scale_cols[kTracedP13ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP13ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  for (int m_fragment = 0; m_fragment < kTracedP13MFragments; ++m_fragment) {
    fragments[m_fragment] =
        LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            row_base + m_fragment * 64,
            thread_idx);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = m_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[m_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedP13CFragmentsRowMajor(
    float alpha,
    const CFragment64 (&accum)[kTracedP13MFragments][kTracedP13NFragments],
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  int row_coords[kTracedP13CCopyCoordCapacity];
  int col_coords[kTracedP13CCopyCoordCapacity];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_c = cute::make_identity_tensor(
      cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
  auto part_c = thr_mma.partition_C(ref_c);
  FillPhysicalCoordMapCopyViewLimited(
      part_c,
      kTracedP13CCopyCoordCapacity,
      row_coords,
      col_coords);
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < kTracedP13MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < kTracedP13NFragments; ++n_fragment) {
        const int physical = n_fragment * 8 + m_fragment * 4 + reg;
        const int output_col_offset = row_coords[physical];
        const int token_row = col_coords[physical];
        if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
          continue;
        }
        const std::size_t input_row = row_start + static_cast<std::size_t>(token_row);
        const std::size_t output_col =
            static_cast<std::size_t>(output_row_base + output_col_offset);
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          output[input_row * output_rows_per_expert + output_col] =
              __float2bfloat16(accum[m_fragment][n_fragment].regs[reg] * alpha);
        } else {
          output[input_row * output_rows_per_expert + output_col] =
              accum[m_fragment][n_fragment].regs[reg] * alpha;
        }
      }
    }
  }
}

template <typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreNanoP1CFragmentsRowMajor(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    std::size_t output_stride,
    OutputType* output) {
  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(thread_idx);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);
  auto accum_tensor = cute::make_tensor(
      const_cast<CRegister*>(accum_storage),
      NanoP1AccumLayout{});
  static_assert(cute::size<0>(decltype(part_c){}) == cute::size<0>(NanoP1AccumLayout{}));
  static_assert(cute::size<1>(decltype(part_c){}) == cute::size<1>(NanoP1AccumLayout{}));
  static_assert(cute::size<2>(decltype(part_c){}) == cute::size<2>(NanoP1AccumLayout{}));
#pragma unroll
  for (int reg = 0; reg < static_cast<int>(cute::size<0>(NanoP1AccumLayout{})); ++reg) {
#pragma unroll
    for (int n_fragment = 0; n_fragment < static_cast<int>(cute::size<1>(NanoP1AccumLayout{})); ++n_fragment) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < static_cast<int>(cute::size<2>(NanoP1AccumLayout{})); ++m_fragment) {
        auto coord = part_c(cute::make_coord(reg, n_fragment, m_fragment));
        const int output_col_offset = CoordGet0(coord);
        const int token_row = CoordGet1(coord);
        if (token_row < 0 ||
            token_row >= valid_rows ||
            output_col_offset < 0 ||
            output_col_offset >= valid_cols) {
          continue;
        }
        const std::size_t row_index =
            static_cast<std::size_t>(row_start + token_row);
        const std::size_t col_index =
            static_cast<std::size_t>(output_col_base + output_col_offset);
        const float value = alpha * accum_tensor(reg, n_fragment, m_fragment);
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          output[row_index * output_stride + col_index] =
              __float2bfloat16(value);
        } else {
          output[row_index * output_stride + col_index] = value;
        }
      }
    }
  }
}

__device__ __forceinline__ void nvfp4_bridge::AccumulateNanoP1DirectPackRowMaxAbs(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int row_start,
    int valid_rows,
    int valid_cols,
    float* activation_output_scales) {
  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(thread_idx);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);
  auto accum_tensor = cute::make_tensor(
      const_cast<CRegister*>(accum_storage),
      NanoP1AccumLayout{});
#pragma unroll
  for (int reg = 0; reg < static_cast<int>(cute::size<0>(NanoP1AccumLayout{})); ++reg) {
#pragma unroll
    for (int n_fragment = 0; n_fragment < static_cast<int>(cute::size<1>(NanoP1AccumLayout{})); ++n_fragment) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < static_cast<int>(cute::size<2>(NanoP1AccumLayout{})); ++m_fragment) {
        auto coord = part_c(cute::make_coord(reg, n_fragment, m_fragment));
        const int output_col_offset = CoordGet0(coord);
        const int token_row = CoordGet1(coord);
        if (token_row < 0 ||
            token_row >= valid_rows ||
            output_col_offset < 0 ||
            output_col_offset >= valid_cols) {
          continue;
        }
        const float activated = fused_decode::Relu2(
            alpha * accum_tensor(reg, n_fragment, m_fragment));
        atomicMax(
            reinterpret_cast<unsigned int*>(
                activation_output_scales + static_cast<std::size_t>(row_start + token_row)),
            __float_as_uint(activated));
      }
    }
  }
}

__device__ __forceinline__ void nvfp4_bridge::StoreNanoP1DirectPackCFragments(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    int num_rows_global,
    int inter_size_global,
    std::uint8_t* packed_bytes,
    std::uint8_t* block_scales,
    std::uint8_t* matmul_block_scales,
    float* activation_output_scales,
    float* staging_tile) {
  if (packed_bytes == nullptr ||
      block_scales == nullptr ||
      matmul_block_scales == nullptr ||
      activation_output_scales == nullptr ||
      staging_tile == nullptr ||
      inter_size_global <= 0 ||
      (inter_size_global % fused_decode::kNvfp4BlockWidth) != 0) {
    return;
  }

  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(thread_idx);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);
  auto accum_tensor = cute::make_tensor(
      const_cast<CRegister*>(accum_storage),
      NanoP1AccumLayout{});
  constexpr int kTileCols = cute::size<1>(NanoP1MmaTileShape{});
  const std::size_t blocks_per_row =
      static_cast<std::size_t>(inter_size_global) / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::size_t packed_row_bytes = static_cast<std::size_t>(inter_size_global) / 2u;

#pragma unroll
  for (int reg = 0; reg < static_cast<int>(cute::size<0>(NanoP1AccumLayout{})); ++reg) {
#pragma unroll
    for (int n_fragment = 0; n_fragment < static_cast<int>(cute::size<1>(NanoP1AccumLayout{})); ++n_fragment) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < static_cast<int>(cute::size<2>(NanoP1AccumLayout{})); ++m_fragment) {
        auto coord = part_c(cute::make_coord(reg, n_fragment, m_fragment));
        const int output_col_offset = CoordGet0(coord);
        const int token_row = CoordGet1(coord);
        if (token_row < 0 ||
            token_row >= valid_rows ||
            output_col_offset < 0 ||
            output_col_offset >= valid_cols) {
          continue;
        }
        staging_tile[
            static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
            static_cast<std::size_t>(output_col_offset)] =
            fused_decode::Relu2(alpha * accum_tensor(reg, n_fragment, m_fragment));
      }
    }
  }
  __syncthreads();

  const int blocks_this_tile =
      (valid_cols + fused_decode::kNvfp4BlockWidth - 1) / fused_decode::kNvfp4BlockWidth;
  for (int row_block = thread_idx;
       row_block < valid_rows * blocks_this_tile;
       row_block += NanoP1ThreadsPerCta) {
    const int token_row = row_block / blocks_this_tile;
    const int block_in_tile = row_block % blocks_this_tile;
    const int col_in_tile = block_in_tile * fused_decode::kNvfp4BlockWidth;
    const int global_col = output_col_base + col_in_tile;
    const int global_row = row_start + token_row;
    if (global_row < 0 ||
        global_row >= num_rows_global ||
        global_col < 0 ||
        (global_col + fused_decode::kNvfp4BlockWidth) > inter_size_global) {
      continue;
    }

    const float row_scale = activation_output_scales[static_cast<std::size_t>(global_row)];
    float block_max_abs = 0.0f;
#pragma unroll
    for (int offset = 0; offset < fused_decode::kNvfp4BlockWidth; ++offset) {
      const float value = staging_tile[
          static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
          static_cast<std::size_t>(col_in_tile + offset)];
      block_max_abs = fmaxf(block_max_abs, fabsf(value));
    }

    const float raw_block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / fused_decode::kNvfp4Fp4MaxFinite);
    const float stabilized_block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * row_scale));
    const std::uint8_t raw_encoded = fused_decode::EncodeFp8Scale(raw_block_scale);
    const std::uint8_t stabilized_encoded =
        fused_decode::EncodeFp8Scale(stabilized_block_scale);
    const std::size_t global_block =
        static_cast<std::size_t>(global_col) / fused_decode::kNvfp4BlockWidth;
    const std::size_t scale_offset = ExecutionScaleOffset(
        static_cast<std::size_t>(global_row),
        global_block,
        padded_blocks_per_row,
        Nvfp4ScaleLayout::kSwizzled128x4);
    block_scales[scale_offset] = raw_encoded;
    matmul_block_scales[scale_offset] = stabilized_encoded;

    const float pack_scale = row_scale * stabilized_block_scale;
    const std::size_t packed_offset =
        static_cast<std::size_t>(global_row) * packed_row_bytes +
        static_cast<std::size_t>(global_col) / 2u;
#pragma unroll
    for (int pair = 0; pair < (fused_decode::kNvfp4BlockWidth / 2); ++pair) {
      const float lhs = staging_tile[
          static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
          static_cast<std::size_t>(col_in_tile + pair * 2 + 0)];
      const float rhs = staging_tile[
          static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
          static_cast<std::size_t>(col_in_tile + pair * 2 + 1)];
      const std::uint8_t lhs_nibble = fused_decode::EncodeFp4(lhs / pack_scale);
      const std::uint8_t rhs_nibble = fused_decode::EncodeFp4(rhs / pack_scale);
      packed_bytes[packed_offset + static_cast<std::size_t>(pair)] =
          static_cast<std::uint8_t>(
              (lhs_nibble & 0x0Fu) | ((rhs_nibble & 0x0Fu) << 4u));
    }
  }
  __syncthreads();
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP13BFragmentsColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    BFragment64 (&fragments)[kTracedP13NFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP13SmemLayoutSFB{});
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = MakeTracedP13TiledCopySFB(mma);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP13ScaleFragmentCosizeB];
  int scale_cols[kTracedP13ScaleFragmentCosizeB];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfb,
      kTracedP13ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  for (int n_fragment = 0; n_fragment < kTracedP13NFragments; ++n_fragment) {
    fragments[n_fragment] =
        LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            thread_idx,
            n_fragment * 8);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = n_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[n_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP5BFragmentsColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    BFragment64 (&fragments)[kTracedP5NFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP5SmemLayoutSFB{});
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = MakeTracedP5TiledCopySFB(mma);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP5ScaleFragmentCosizeB];
  int scale_cols[kTracedP5ScaleFragmentCosizeB];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfb,
      kTracedP5ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  for (int n_fragment = 0; n_fragment < kTracedP5NFragments; ++n_fragment) {
    fragments[n_fragment] =
        LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            thread_idx,
            n_fragment * 8);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = n_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[n_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedP5CFragmentsTranspose(
    float alpha,
    const CFragment64 (&accum)[kTracedP5MFragments][kTracedP5NFragments],
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  int row_coords[kTracedP5CCopyCoordCapacity];
  int col_coords[kTracedP5CCopyCoordCapacity];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_c = cute::make_identity_tensor(
      cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
  auto part_c = thr_mma.partition_C(ref_c);
  FillPhysicalCoordMapCopyViewLimited(
      part_c,
      kTracedP5CCopyCoordCapacity,
      row_coords,
      col_coords);
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < kTracedP5MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < kTracedP5NFragments; ++n_fragment) {
        const int physical = reg * 8 + m_fragment * 2 + n_fragment;
        const int row = row_coords[physical];
        const int token = col_coords[physical];
        if (row >= output_rows_this_tile || token >= valid_rows) {
          continue;
        }
        const std::size_t input_row = row_start + static_cast<std::size_t>(token);
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row)] =
              __float2bfloat16(accum[m_fragment][n_fragment].regs[reg] * alpha);
        } else {
          output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row)] =
              accum[m_fragment][n_fragment].regs[reg] * alpha;
        }
      }
    }
  }
}



template <int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int lane_id,
    int n_base) {
  return LoadFragmentB_ColMajor64x8Tiled<SingleAtomTiledMma, kRowsPerTile>(
      packed_rows,
      scale_words,
      lane_id,
      n_base);
}

__device__ __forceinline__ void nvfp4_bridge::Clear(CFragment64& fragment) {
  fragment.regs[0] = 0.0f;
  fragment.regs[1] = 0.0f;
  fragment.regs[2] = 0.0f;
  fragment.regs[3] = 0.0f;
}

__device__ __forceinline__ std::uint32_t nvfp4_bridge::MakePackedUnitScaleWord() {
  const auto one = static_cast<std::uint8_t>(cute::float_ue4m3_t(1.0f).raw());
  return static_cast<std::uint32_t>(one) |
         (static_cast<std::uint32_t>(one) << 8) |
         (static_cast<std::uint32_t>(one) << 16) |
         (static_cast<std::uint32_t>(one) << 24);
}

__device__ __forceinline__ void nvfp4_bridge::Gemm(
    CFragment64& accum,
    const AFragment64& a,
    const BFragment64& b) {
  Sm120BlockScaledFp4Mma(
      accum.regs[0],
      accum.regs[1],
      accum.regs[2],
      accum.regs[3],
      a.regs[0],
      a.regs[1],
      a.regs[2],
      a.regs[3],
      b.regs[0],
      b.regs[1],
      static_cast<std::uint32_t>(a.scale[0]),
      static_cast<std::uint32_t>(b.scale[0]));
}

template <int kRowsPerTile, typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreFragmentC_RowMajor16x8(
    float alpha,
    const CFragment64& accum,
    int lane_id,
    int col_base,
    int row_base,
    int valid_rows,
    int valid_cols,
    std::size_t row_start,
    std::size_t output_row_base,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  StoreFp4AccumulatorTileRowMajor16x8<kRowsPerTile>(
      alpha,
      accum.regs[0],
      accum.regs[1],
      accum.regs[2],
      accum.regs[3],
      lane_id,
      col_base,
      row_base,
      valid_rows,
      valid_cols,
      row_start,
      output_row_base,
      output_rows_per_expert,
      output);
}

template <typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreFragmentC_Transpose16x8(
    float alpha,
    const CFragment64& accum,
    int lane_id,
    int output_row_base,
    int m_base,
    int token_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  if (token_base >= valid_rows) {
    return;
  }
  const int row_group = (lane_id >> 3) * 4;
  const int row0 = m_base + row_group + 0;
  const int row1 = m_base + row_group + 2;
  const int row2 = m_base + row_group + 1;
  const int row3 = m_base + row_group + 3;
  const std::size_t input_row = row_start + static_cast<std::size_t>(token_base);
  if (row0 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row0)] =
          __float2bfloat16(accum.regs[0] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row0)] =
          accum.regs[0] * alpha;
    }
  }
  if (row1 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row1)] =
          __float2bfloat16(accum.regs[1] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row1)] =
          accum.regs[1] * alpha;
    }
  }
  if (row2 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row2)] =
          __float2bfloat16(accum.regs[2] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row2)] =
          accum.regs[2] * alpha;
    }
  }
  if (row3 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row3)] =
          __float2bfloat16(accum.regs[3] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row3)] =
          accum.regs[3] * alpha;
    }
  }
}

void PrintP13ScaleTrace() {
  const auto& trace = g_p13_scale_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p13_scale_trace: invalid\n");
    return;
  }
  std::fprintf(
      stderr,
      "p13_scale_trace: row_start=%d valid_rows=%d output_row_base=%d\n",
      trace.row_start,
      trace.valid_rows,
      trace.output_row_base);
  for (int i = 0; i < 2; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: A m=%d base_row=%d source=0x%08x loaded=0x%08x smem=0x%08x fragment=0x%08x\n",
        i,
        trace.a_base_rows[i],
        trace.a_source_words[i],
        trace.a_loaded_words[i],
        trace.a_smem_words[i],
        trace.a_fragment_words[i]);
  }
  for (int i = 0; i < 2; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: B n=%d base_row=%d source=0x%08x loaded=0x%08x smem=0x%08x fragment=0x%08x\n",
        i,
        trace.b_base_rows[i],
        trace.b_source_words[i],
        trace.b_loaded_words[i],
        trace.b_smem_words[i],
        trace.b_fragment_words[i]);
  }
  for (int i = 0; i < 8; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: A coord[%d]=(%d,%d) raw=0x%02x\n",
        i,
        trace.a_scale_rows[i],
        trace.a_scale_cols[i],
        static_cast<unsigned>(trace.a_scale_raw[i]));
  }
  for (int i = 0; i < 8; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: B coord[%d]=(%d,%d) raw=0x%02x\n",
        i,
        trace.b_scale_rows[i],
        trace.b_scale_cols[i],
        static_cast<unsigned>(trace.b_scale_raw[i]));
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: A post-store row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.a_post_store_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: B post-store row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.b_post_store_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: A logical row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.a_logical_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: B logical row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.b_logical_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
}

void PrintP1DirectTrace() {
  const auto& trace = g_p1_direct_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p1_direct_trace: no capture\n");
    return;
  }
  for (int slot = 0; slot < 2; ++slot) {
    std::fprintf(
        stderr,
        "p1_direct_trace[%d]: warp=%d n_base=%d a_scale=0x%08x "
        "a={0x%08x,0x%08x,0x%08x,0x%08x} "
        "b_scale=0x%08x b={0x%08x,0x%08x}\n",
        slot,
        trace.warp_ids[slot],
        trace.n_base[slot],
        trace.a_scale[slot],
        trace.a_regs[slot][0],
        trace.a_regs[slot][1],
        trace.a_regs[slot][2],
        trace.a_regs[slot][3],
        trace.b_scale[slot],
        trace.b_regs[slot][0],
        trace.b_regs[slot][1]);
  }
}

__global__ void Nvfp4GroupedExpertMatVecRowsKernel(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x) / output_rows_per_expert;
  const std::size_t output_row = static_cast<std::size_t>(blockIdx.x) % output_rows_per_expert;
  if (expert_index >= n_experts) {
    return;
  }

  const int begin_row = expert_offsets[expert_index];
  const int end_row = expert_offsets[expert_index + 1];
  if (begin_row >= end_row) {
    return;
  }

  const fused_decode::Nvfp4WeightView weight =
      MakeDeviceWeightView(weights[expert_index]);
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * (weight.input_cols / 2);
  const std::size_t scale_row_offset = output_row * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;

  for (int tile_begin = begin_row; tile_begin < end_row; tile_begin += kGroupedTokenTile) {
    const int remaining_rows = end_row - tile_begin;
    const int valid_rows =
        remaining_rows < kGroupedTokenTile ? remaining_rows : kGroupedTokenTile;
    float accum[kGroupedTokenTile] = {0.0f};

    for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
         pair_index < pairs_per_row;
         pair_index += blockDim.x) {
      const std::size_t block = pair_index / 8u;
      const std::size_t pair_in_block = pair_index % 8u;
      const std::size_t col = block * fused_decode::kNvfp4BlockWidth + pair_in_block * 2u;
      const float block_scale =
          LoadNvfp4WeightBlockScale(weight, output_row, block);
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale;
      const float w1 = fused_decode::DecodeFp4((packed >> 4) & 0x0Fu) * block_scale;
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        const float* input_row =
            input + static_cast<std::size_t>(tile_begin + tile_row) * weight.input_cols;
        accum[tile_row] += input_row[col] * w0;
        accum[tile_row] += input_row[col + 1] * w1;
      }
    }

    for (int tile_row = 0; tile_row < kGroupedTokenTile; ++tile_row) {
      partial_sums[tile_row * blockDim.x + threadIdx.x] =
          tile_row < valid_rows ? accum[tile_row] : 0.0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
          partial_sums[tile_row * blockDim.x + threadIdx.x] +=
              partial_sums[tile_row * blockDim.x + threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        output[static_cast<std::size_t>(tile_begin + tile_row) * output_rows_per_expert +
               output_row] = partial_sums[tile_row * blockDim.x];
      }
    }
    __syncthreads();
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRowsKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  if (expert_index < 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      valid_rows <= 0 ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRowsBf16Kernel(
    const float* input,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_m_limits,
    const int* expert_first_token_offsets,
    int token_tile_dim,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int expert_index = cta_batch_indices[cta_index];
  const int batch_row_begin = expert_first_token_offsets[expert_index];
  const int batch_cta_begin = batch_row_begin / token_tile_dim;
  const int row_start =
      batch_row_begin + (cta_index - batch_cta_begin) * token_tile_dim;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows = max(0, min(m_limit - row_start, token_tile_dim));
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  if (expert_index < 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      valid_rows <= 0 ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        __float2bfloat16(c_tile[tile_output_row][tile_token]);
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRelu2MaxAbsKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    unsigned int* expert_max_bits) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];
  __shared__ float shared_max[kPlannedThreadsPerBlock];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count || expert_max_bits == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  float thread_max = 0.0f;
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const float value = fused_decode::Relu2(c_tile[tile_output_row][tile_token]);
    thread_max = fmaxf(thread_max, value);
  }

  shared_max[tid] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
      shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(&expert_max_bits[expert_index], __float_as_uint(shared_max[0]));
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRelu2PackKernel(
    const float* input,
    const float* output_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    std::size_t output_row_capacity,
    Nvfp4ScaleLayout output_scale_layout,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      output_expert_tensor_scales == nullptr ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_input_row = weight.input_cols / 2;
  const std::size_t input_blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_input_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * input_blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  if (lane_id >= valid_rows || warp_row >= output_rows_this_tile) {
    return;
  }

  const std::size_t output_row = static_cast<std::size_t>(row_start + lane_id);
  if (output_row >= output_row_capacity) {
    return;
  }
  const std::size_t blocks_per_output_row =
      output_rows_per_expert / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_output_row, kNvfp4ScaleBlockTile);
  const std::size_t block_col =
      static_cast<std::size_t>(output_row_base + warp_row) / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = output_expert_tensor_scales[expert_index];

  float activated[16];
  float block_max_abs = 0.0f;
  for (int col = 0; col < 16; ++col) {
    const float value = fused_decode::Relu2(c_tile[warp_row + col][lane_id]);
    activated[col] = value;
    if (value > block_max_abs) {
      block_max_abs = value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * tensor_scale));
  }
  const std::uint8_t encoded_block_scale = fused_decode::EncodeFp8Scale(block_scale);
  const std::size_t block_scale_offset = output_row * blocks_per_output_row + block_col;
  output_block_scales[block_scale_offset] = encoded_block_scale;
  output_matmul_scales[ExecutionScaleOffset(
      output_row, block_col, padded_blocks_per_row, output_scale_layout)] = encoded_block_scale;

  const float pack_scale = tensor_scale * block_scale;
  const std::size_t packed_row_offset =
      output_row * (output_rows_per_expert / 2u) +
      block_col * (fused_decode::kNvfp4BlockWidth / 2u);
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t lhs = fused_decode::EncodeFp4(activated[pair * 2] / pack_scale);
    const std::uint8_t rhs = fused_decode::EncodeFp4(activated[pair * 2 + 1] / pack_scale);
    output_packed[packed_row_offset + static_cast<std::size_t>(pair)] =
        static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
  }
}

template <typename OutputType, int kOutputTile, int kMacroTileK>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalse(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kWarpSubTiles =
      kOutputTile / (kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN);
  static_assert(kWarpSubTiles >= 1);
  __shared__ __nv_bfloat16 a_tile[2][kPlannedWmmaTileM][kMacroTileK];
  __shared__ __nv_bfloat16 b_tile[2][kOutputTile][kMacroTileK];
  __shared__ float c_tile_storage[kPlannedWmmaTileM * kOutputTile];
  auto* c_tile = reinterpret_cast<float (*)[kOutputTile]>(c_tile_storage);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const bool mma_warp = warp_id < kGroupedConsumerWarpsPerBlock;
  const bool producer_warp = warp_id >= kGroupedConsumerWarpsPerBlock;
  const int producer_tid = tid - (kGroupedConsumerWarpsPerBlock * 32);
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frags[kWarpSubTiles];
  if (mma_warp) {
    for (int i = 0; i < kWarpSubTiles; ++i) {
      wmma::fill_fragment(c_frags[i], 0.0f);
    }
  }

  auto stage_macro_tile = [&](int buffer_index, std::size_t k_base) {
    if (!producer_warp) {
      return;
    }
    const int producer_threads = kGroupedProducerWarpsPerBlock * 32;
    const int macro_k = static_cast<int>(
        min(static_cast<std::size_t>(kMacroTileK), weight.input_cols - k_base));
    const int macro_blocks = macro_k / static_cast<int>(fused_decode::kNvfp4BlockWidth);
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    for (int linear_block = producer_tid;
         linear_block < (kPlannedWmmaTileM * macro_blocks);
         linear_block += producer_threads) {
      const int tile_token = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &a_tile[buffer_index][tile_token]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      if (tile_token < valid_rows) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float row_ts =
            input_dq_scales != nullptr
                ? 1.0f
                : (input_expert_tensor_scales != nullptr
                       ? input_expert_tensor_scales[expert_index]
                       : (input_per_row_tensor_scales != nullptr
                              ? input_per_row_tensor_scales[input_row]
                              : *input_tensor_scale_data));
        DecodeGroupedPackedInputBlockBf16(
            packed_input,
            input_block_scales,
            input_dq_scales,
            row_ts,
            weight.input_cols,
            input_row,
            block_base + static_cast<std::size_t>(tile_block),
            dst);
      } else {
        ZeroBf16Block16(dst);
      }
    }
    for (int linear_block = producer_tid;
         linear_block < (output_rows_this_tile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &b_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      DecodeNvfp4WeightBlockBf16(
          weight,
          static_cast<std::size_t>(output_row_base + tile_output_row),
          block_base + static_cast<std::size_t>(tile_block),
          dst);
    }
    for (int linear_block = producer_tid + output_rows_this_tile * macro_blocks;
         linear_block < (kOutputTile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &b_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      ZeroBf16Block16(dst);
    }
  };

  const std::size_t k_step = static_cast<std::size_t>(kMacroTileK);
  int buffer_index = 0;
  stage_macro_tile(buffer_index, 0);
  __syncthreads();
  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += k_step) {
    const int macro_k = static_cast<int>(
        min(k_step, weight.input_cols - k_base));
    const int next_buffer = buffer_index ^ 1;
    const std::size_t next_k_base = k_base + k_step;
    if (next_k_base < weight.input_cols) {
      stage_macro_tile(next_buffer, next_k_base);
    }
    if (mma_warp) {
      wmma::fragment<
          wmma::matrix_a,
          kPlannedWmmaTileM,
          kPlannedWmmaTileN,
          kPlannedWmmaTileK,
          __nv_bfloat16,
          wmma::row_major>
          a_frag;
      for (int tile_k_base = 0; tile_k_base < macro_k; tile_k_base += kPlannedWmmaTileK) {
        wmma::load_matrix_sync(a_frag, &a_tile[buffer_index][0][tile_k_base], kMacroTileK);
        for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
          const int warp_col = warp_id * kPlannedWmmaTileN +
              subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
          if (warp_col < output_rows_this_tile) {
            wmma::fragment<
                wmma::matrix_b,
                kPlannedWmmaTileM,
                kPlannedWmmaTileN,
                kPlannedWmmaTileK,
                __nv_bfloat16,
                wmma::col_major>
                b_frag;
            wmma::load_matrix_sync(
                b_frag, &b_tile[buffer_index][warp_col][tile_k_base], kMacroTileK);
            wmma::mma_sync(c_frags[subtile], a_frag, b_frag, c_frags[subtile]);
          }
        }
      }
    }
    __syncthreads();
    buffer_index ^= 1;
  }

  if (mma_warp) {
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      const int warp_col = warp_id * kPlannedWmmaTileN +
          subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
      if (warp_col < output_rows_this_tile) {
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          wmma::store_matrix_sync(
              &c_tile[0][warp_col], c_frags[subtile], kOutputTile, wmma::mem_row_major);
        } else {
          wmma::store_matrix_sync(
              &c_tile[0][warp_col],
              c_frags[subtile],
              kOutputTile,
              wmma::mem_row_major);
        }
      }
    }
  }
  __syncthreads();
  for (int linear_index = tid;
       linear_index < (valid_rows * output_rows_this_tile);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_token = linear_index / output_rows_this_tile;
    const int tile_output_row = linear_index % output_rows_this_tile;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + output_row] =
          __float2bfloat16(c_tile[tile_token][tile_output_row]);
    } else {
      output[input_row * output_rows_per_expert + output_row] =
          c_tile[tile_token][tile_output_row];
    }
  }
}

template <typename OutputType, int kOutputTile, int kMacroTileK>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kWarpSubTiles =
      kOutputTile / (kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN);
  static_assert(kWarpSubTiles >= 1);
  __shared__ __nv_bfloat16 a_tile[2][kOutputTile][kMacroTileK];
  __shared__ __nv_bfloat16 b_tile[2][kPlannedWmmaTileM][kMacroTileK];
  __shared__ float c_tile_storage[kOutputTile * kPlannedWmmaTileM];
  auto* c_tile = reinterpret_cast<float (*)[kPlannedWmmaTileM]>(c_tile_storage);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const bool mma_warp = warp_id < kGroupedConsumerWarpsPerBlock;
  const bool producer_warp = warp_id >= kGroupedConsumerWarpsPerBlock;
  const int producer_tid = tid - (kGroupedConsumerWarpsPerBlock * 32);
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frags[kWarpSubTiles];
  if (mma_warp) {
    for (int i = 0; i < kWarpSubTiles; ++i) {
      wmma::fill_fragment(c_frags[i], 0.0f);
    }
  }

  auto stage_macro_tile = [&](int buffer_index, std::size_t k_base) {
    if (!producer_warp) {
      return;
    }
    const int producer_threads = kGroupedProducerWarpsPerBlock * 32;
    const int macro_k = static_cast<int>(
        min(static_cast<std::size_t>(kMacroTileK), weight.input_cols - k_base));
    const int macro_blocks = macro_k / static_cast<int>(fused_decode::kNvfp4BlockWidth);
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    for (int linear_block = producer_tid;
         linear_block < (output_rows_this_tile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &a_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      DecodeNvfp4WeightBlockBf16(
          weight,
          static_cast<std::size_t>(output_row_base + tile_output_row),
          block_base + static_cast<std::size_t>(tile_block),
          dst);
    }
    for (int linear_block = producer_tid + output_rows_this_tile * macro_blocks;
         linear_block < (kOutputTile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &a_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      ZeroBf16Block16(dst);
    }
    for (int linear_block = producer_tid;
         linear_block < (kPlannedWmmaTileM * macro_blocks);
         linear_block += producer_threads) {
      const int tile_token = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &b_tile[buffer_index][tile_token]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      if (tile_token < valid_rows) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float row_ts =
            input_dq_scales != nullptr
                ? 1.0f
                : (input_expert_tensor_scales != nullptr
                       ? input_expert_tensor_scales[expert_index]
                       : (input_per_row_tensor_scales != nullptr
                              ? input_per_row_tensor_scales[input_row]
                              : *input_tensor_scale_data));
        DecodeGroupedPackedInputBlockBf16(
            packed_input,
            input_block_scales,
            input_dq_scales,
            row_ts,
            weight.input_cols,
            input_row,
            block_base + static_cast<std::size_t>(tile_block),
            dst);
      } else {
        ZeroBf16Block16(dst);
      }
    }
  };

  const std::size_t k_step = static_cast<std::size_t>(kMacroTileK);
  int buffer_index = 0;
  stage_macro_tile(buffer_index, 0);
  __syncthreads();
  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += k_step) {
    const int macro_k = static_cast<int>(
        min(k_step, weight.input_cols - k_base));
    const int next_buffer = buffer_index ^ 1;
    const std::size_t next_k_base = k_base + k_step;
    if (next_k_base < weight.input_cols) {
      stage_macro_tile(next_buffer, next_k_base);
    }
    if (mma_warp) {
      for (int tile_k_base = 0; tile_k_base < macro_k; tile_k_base += kPlannedWmmaTileK) {
        for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
          const int warp_row = warp_id * kPlannedWmmaTileN +
              subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
          if (warp_row < output_rows_this_tile) {
            wmma::fragment<
                wmma::matrix_a,
                kPlannedWmmaTileM,
                kPlannedWmmaTileN,
                kPlannedWmmaTileK,
                __nv_bfloat16,
                wmma::row_major>
                a_frag;
            wmma::fragment<
                wmma::matrix_b,
                kPlannedWmmaTileM,
                kPlannedWmmaTileN,
                kPlannedWmmaTileK,
                __nv_bfloat16,
                wmma::col_major>
                b_frag;
            wmma::load_matrix_sync(
                a_frag, &a_tile[buffer_index][warp_row][tile_k_base], kMacroTileK);
            wmma::load_matrix_sync(
                b_frag, &b_tile[buffer_index][0][tile_k_base], kMacroTileK);
            wmma::mma_sync(c_frags[subtile], a_frag, b_frag, c_frags[subtile]);
          }
        }
      }
    }
    __syncthreads();
    buffer_index ^= 1;
  }

  if (mma_warp) {
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      const int warp_row = warp_id * kPlannedWmmaTileN +
          subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
      if (warp_row < output_rows_this_tile) {
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          wmma::store_matrix_sync(
              &c_tile[warp_row][0], c_frags[subtile], kPlannedWmmaTileM, wmma::mem_row_major);
        } else {
          wmma::store_matrix_sync(
              &c_tile[warp_row][0],
              c_frags[subtile],
              kPlannedWmmaTileM,
              wmma::mem_row_major);
        }
      }
    }
  }
  __syncthreads();
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + output_row] =
          __float2bfloat16(c_tile[tile_output_row][tile_token]);
    } else {
      output[input_row * output_rows_per_expert + output_row] =
          c_tile[tile_output_row][tile_token];
    }
  }
}

template <
    nvfp4_bridge::UnifiedRoutedFp4Profile Profile,
    typename OutputType,
    nvfp4_bridge::P5EpilogueMode P5Mode = nvfp4_bridge::P5EpilogueMode::kBf16,
    bool UseFp4DirectOutput =
        (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
         P5Mode == nvfp4_bridge::P5EpilogueMode::kFp4Direct)>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_matmul_block_scales,
    Nvfp4ScaleLayout input_scale_layout,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const routed_p5_tma::P5TmaLoadB* p5_tma_load_b_descriptors,
    const routed_p5_tma::P5TmaLoadSFB* p5_tma_load_sfb_descriptors,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output,
    std::uint8_t* fp4_packed_data,
    std::uint8_t* fp4_block_scales,
    std::uint8_t* fp4_matmul_block_scales,
    float* fp4_activation_output_scale,
    std::size_t fp4_cols,
    std::size_t fp4_padded_blocks_per_row,
    Nvfp4ScaleLayout fp4_scale_layout) {
  constexpr bool kUseSpecializedP5Fp4DirectOutput =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      P5Mode == nvfp4_bridge::P5EpilogueMode::kFp4Direct;
  using Traits = nvfp4_bridge::UnifiedRoutedFp4Traits<Profile>;
  using TiledMma = typename Traits::TiledMma;
  using CollectiveMainloop = typename Traits::CollectiveMainloop;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemCopyAtomA = typename Traits::SmemCopyAtomA;
  using SmemCopyAtomB = typename Traits::SmemCopyAtomB;
  using SmemCopyAtomSFA = typename Traits::SmemCopyAtomSFA;
  using SmemCopyAtomSFB = typename Traits::SmemCopyAtomSFB;
  constexpr int kOutputTile = Traits::kOutputTile;
  constexpr int kProfileTokenRows = Traits::kTokenRows;
  constexpr int kMacroTileK = Traits::kMacroTileK;
  constexpr int kMacroTileBytes = kMacroTileK / 2;
  constexpr int kMacroScaleBytes = kMacroTileK / fused_decode::kNvfp4BlockWidth;
  constexpr int kFp4ConsumerWarps = cute::size(TiledMma{}) / 32;
  constexpr int kP5ScaleTmaRowTile = 128;
  constexpr int kP5PipelineStages =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 ? 2 : 1;
  constexpr int kP5SfbTmaStageElems =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 ? Traits::kScaleSmemCosizeB : 1;

  static_assert(
      !kUseSpecializedP5Fp4DirectOutput ||
          (kFp4ConsumerWarps * 32 == fused_decode::kThreadsPerBlock &&
           Traits::kOutputTile == nvfp4_bridge::kTracedP5DirectStageCols &&
           Traits::kTokenRows == nvfp4_bridge::kTracedP5DirectStageRows),
      "P5 direct pack assumes a full 32x128 CTA tile");

  __shared__ alignas(1024) cute::array_aligned<
      typename Traits::SmemAllocA,
      Traits::kSwizzledAElems * kP5PipelineStages>
      smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<
      typename Traits::SmemAllocB,
      Traits::kSwizzledBElems * kP5PipelineStages>
      smem_swizzled_b_storage;
  __shared__ alignas(1024) cute::array_aligned<
      nvfp4_cute::ElementSFCompute,
      Traits::kScaleSmemCosizeA * kP5PipelineStages>
      a_scale_smem_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeB>
      b_scale_smem_storage;
  __shared__ alignas(128) cute::array_aligned<nvfp4_cute::ElementSFCompute, kP5SfbTmaStageElems>
      p5_sfb_tma_smem_storage;
  __shared__ alignas(16)
      cutlass::arch::ClusterTransactionBarrier::ValueType
          p5_tma_full_mbar_storage[kP5PipelineStages];
  __shared__ alignas(16)
      cutlass::arch::ClusterBarrier::ValueType
          p5_tma_empty_mbar_storage[kP5PipelineStages];
  __shared__ int p13_debug_capture_cta;

  auto* smem_swizzled_a = smem_swizzled_a_storage.data();
  auto* smem_swizzled_b = smem_swizzled_b_storage.data();
  auto* a_scale_smem = a_scale_smem_storage.data();
  auto* b_scale_smem = b_scale_smem_storage.data();
  auto* p5_sfb_tma_smem = p5_sfb_tma_smem_storage.data();
  auto* p5_tma_full_mbar =
      cute::recast_ptr<cutlass::arch::ClusterTransactionBarrier>(&p5_tma_full_mbar_storage[0]);
  auto* p5_tma_empty_mbar =
      cute::recast_ptr<cutlass::arch::ClusterBarrier>(&p5_tma_empty_mbar_storage[0]);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_matmul_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      (!UseFp4DirectOutput && output == nullptr) ||
      (UseFp4DirectOutput &&
       (fp4_packed_data == nullptr ||
        fp4_block_scales == nullptr ||
        fp4_matmul_block_scales == nullptr ||
        fp4_activation_output_scale == nullptr ||
        fp4_cols == 0 ||
        fp4_padded_blocks_per_row == 0))) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const bool is_fp4_consumer_thread = warp_id < kFp4ConsumerWarps;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }
  if (tid == 0) {
    int capture = 0;
    if (g_enable_p13_debug_trace != 0 &&
        Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13 &&
        blockIdx.x == 0 &&
        (g_p13_debug_trace_target_valid_rows < 0 ||
         valid_rows == g_p13_debug_trace_target_valid_rows) &&
        atomicCAS(&g_p13_debug_trace_claimed, 0, 1) == 0) {
      capture = 1;
    }
    p13_debug_capture_cta = capture;
  }
  __syncthreads();

  const FusedNvfp4WeightView weight = weights[expert_index];
  const bool use_p5_tma_a =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      weight.p5_tma_load_a != nullptr &&
      (output_row_base % 128) == 0 &&
      (weight.output_rows % 128u) == 0 &&
      (weight.input_cols % 128u) == 0;
  const auto* p5_tma_load_a_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadA*>(weight.p5_tma_load_a);
  const auto* p5_tma_load_sfa_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadSFA*>(weight.p5_tma_load_sfa);
  const bool use_p5_tma_b =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      p5_tma_load_b_descriptors != nullptr &&
      (weight.input_cols % 128u) == 0;
  const bool use_p5_tma_sfa =
      use_p5_tma_a &&
      p5_tma_load_sfa_ptr != nullptr;
  const bool use_p5_direct_gmem_sfb =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      input_matmul_block_scales != nullptr &&
      input_scale_layout == Nvfp4ScaleLayout::kSwizzled128x4;
  const bool use_p5_pipeline =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      use_p5_tma_a &&
      use_p5_tma_sfa &&
      use_p5_tma_b &&
      use_p5_direct_gmem_sfb &&
      (weight.input_cols % static_cast<std::size_t>(kMacroTileK)) == 0;
  const bool use_p5_tma_sfb =
      !use_p5_pipeline &&
      !use_p5_direct_gmem_sfb &&
      use_p5_tma_b &&
      p5_tma_load_sfb_descriptors != nullptr &&
      input_scale_layout == Nvfp4ScaleLayout::kSwizzled128x4 &&
      (row_start >= 0) &&
      ((row_start & (kP5ScaleTmaRowTile - 1)) + valid_rows <= kP5ScaleTmaRowTile);
  const int p5_sfb_row_offset =
      use_p5_tma_sfb ? (row_start & (kP5ScaleTmaRowTile - 1)) : 0;
  const bool use_p5_tma_sfb_direct = use_p5_tma_sfb && p5_sfb_row_offset == 0;
  const bool use_p5_tma_sfb_remap = use_p5_tma_sfb && !use_p5_tma_sfb_direct;
  const bool p5_sfb_within_single_chunk =
      valid_rows > 0 &&
      (p5_sfb_row_offset / 32) ==
          ((p5_sfb_row_offset + valid_rows - 1) / 32);
  const bool use_p5_tma_sfb_direct_slice =
      use_p5_tma_sfb_remap &&
      p5_sfb_row_offset < 32 &&
      (p5_sfb_row_offset + valid_rows) <= 32;
  const bool use_p5_tma_sfb_register_assembly =
      use_p5_tma_sfb_remap &&
      !use_p5_tma_sfb_direct_slice &&
      p5_sfb_within_single_chunk;
  const bool use_p5_tma_sfb_remap_fallback =
      use_p5_tma_sfb_remap &&
      !use_p5_tma_sfb_direct_slice &&
      !use_p5_tma_sfb_register_assembly;
  const int lane_predicate = cute::elect_one_sync();
  const bool is_p5_tma_thread =
      (use_p5_tma_a || use_p5_tma_b) &&
      warp_id == kFp4ConsumerWarps &&
      lane_predicate != 0;
  if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
    if (is_p5_tma_thread) {
      if (use_p5_tma_a) {
        cute::prefetch_tma_descriptor(p5_tma_load_a_ptr->get_tma_descriptor());
      }
      if (use_p5_tma_sfa) {
        cute::prefetch_tma_descriptor(p5_tma_load_sfa_ptr->get_tma_descriptor());
      }
      if (use_p5_tma_b) {
        cute::prefetch_tma_descriptor(
            p5_tma_load_b_descriptors[cta_index].get_tma_descriptor());
      }
      if (use_p5_tma_sfb) {
        cute::prefetch_tma_descriptor(
            p5_tma_load_sfb_descriptors[cta_index].get_tma_descriptor());
      }
    }
  }
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int macro_k_tile_count = static_cast<int>(weight.input_cols / static_cast<std::size_t>(kMacroTileK));
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * weight_tensor_scale;

  nvfp4_bridge::CRegister accum_storage[Traits::kAccumProfileCosize];
  if (warp_id < kFp4ConsumerWarps) {
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        typename Traits::AccumLayout{});
    cute::clear(accum_tensor);
  }

  if (use_p5_pipeline) {
    if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
    constexpr uint32_t kP5PipelineStageBytes = static_cast<uint32_t>(
        cutlass::bits_to_bytes(
            cute::size(cute::take<0, 2>(SmemLayoutA{})) *
                cute::sizeof_bits_v<routed_p5_tma::ElementAB> +
            cute::size(cute::take<0, 2>(SmemLayoutB{})) *
                cute::sizeof_bits_v<routed_p5_tma::ElementAB> +
            cute::cosize(cute::take<0, 2>(SmemLayoutSFA{})) *
                cute::sizeof_bits_v<routed_p5_tma::ElementSF>));

    auto advance_pipeline_state = [](int& stage, uint32_t& phase) {
      stage = (stage + 1) & 1;
      if (stage == 0) {
        phase ^= 1u;
      }
    };

    if (tid == 0) {
      for (int stage = 0; stage < kP5PipelineStages; ++stage) {
        p5_tma_full_mbar[stage].init(1);
        p5_tma_empty_mbar[stage].init(kFp4ConsumerWarps);
      }
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (is_p5_tma_thread) {
      using X = cute::Underscore;
      using ProducerBarrierType = typename cutlass::arch::ClusterTransactionBarrier::ValueType;
      const int output_tile_coord = output_row_base / 128;

      const auto& tma_load_a = *p5_tma_load_a_ptr;
      auto mA_mkl = tma_load_a.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(weight.output_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gA_mkl = cute::local_tile(
          mA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_a = tma_load_a.get_slice(0);
      auto gA = gA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgA = block_tma_a.partition_S(gA);
      auto sA_stage0 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 0 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto sA_stage1 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 1 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto tAsA_stage0 =
          block_tma_a.partition_D(cute::as_position_independent_swizzle_tensor(sA_stage0));
      auto tAsA_stage1 =
          block_tma_a.partition_D(cute::as_position_independent_swizzle_tensor(sA_stage1));

      const auto& tma_load_sfa = *p5_tma_load_sfa_ptr;
      auto mSFA_mkl = tma_load_sfa.get_tma_tensor(cute::shape(
          routed_p5_tma::MakeP5ScaleLayoutSFA(
              static_cast<int32_t>(weight.output_rows),
              static_cast<int32_t>(weight.input_cols))));
      auto gSFA_mkl = cute::local_tile(
          mSFA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_sfa = tma_load_sfa.get_slice(0);
      auto gSFA = gSFA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgSFA = block_tma_sfa.partition_S(gSFA);
      auto sSFA_stage0 = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 0 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto sSFA_stage1 = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 1 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto tAsSFA_stage0 =
          block_tma_sfa.partition_D(cute::as_position_independent_swizzle_tensor(sSFA_stage0));
      auto tAsSFA_stage1 =
          block_tma_sfa.partition_D(cute::as_position_independent_swizzle_tensor(sSFA_stage1));

      const auto& tma_load_b = p5_tma_load_b_descriptors[cta_index];
      auto mB_nkl = tma_load_b.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(valid_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gB_nkl = cute::local_tile(
          mB_nkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<X, cute::_1, cute::_1>{});
      auto block_tma_b = tma_load_b.get_slice(0);
      auto gB = gB_nkl(cute::_, cute::_, 0, cute::_, 0);
      auto tBgB = block_tma_b.partition_S(gB);
      auto sB_stage0 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 0 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto sB_stage1 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 1 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto tBsB_stage0 =
          block_tma_b.partition_D(cute::as_position_independent_swizzle_tensor(sB_stage0));
      auto tBsB_stage1 =
          block_tma_b.partition_D(cute::as_position_independent_swizzle_tensor(sB_stage1));

      int producer_stage = 0;
      uint32_t producer_phase = 1;
      for (int tile = 0; tile < macro_k_tile_count; ++tile) {
        p5_tma_empty_mbar[producer_stage].wait(producer_phase);
        p5_tma_full_mbar[producer_stage].arrive_and_expect_tx(kP5PipelineStageBytes);
        auto& barrier = p5_tma_full_mbar[producer_stage];
        if (producer_stage == 0) {
          auto tma_copy_a =
              tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_sfa =
              tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_b =
              tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_a,
              tAgA(cute::_, cute::_, cute::_, tile),
              tAsA_stage0(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_sfa,
              tAgSFA(cute::_, cute::_, cute::_, tile),
              tAsSFA_stage0(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_b,
              tBgB(cute::_, cute::_, cute::_, tile),
              tBsB_stage0(cute::_, cute::_, cute::_, cute::Int<0>{}));
        } else {
          auto tma_copy_a =
              tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_sfa =
              tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_b =
              tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_a,
              tAgA(cute::_, cute::_, cute::_, tile),
              tAsA_stage1(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_sfa,
              tAgSFA(cute::_, cute::_, cute::_, tile),
              tAsSFA_stage1(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_b,
              tBgB(cute::_, cute::_, cute::_, tile),
              tBsB_stage1(cute::_, cute::_, cute::_, cute::Int<0>{}));
        }
        advance_pipeline_state(producer_stage, producer_phase);
      }
      const int outstanding_stages =
          macro_k_tile_count < kP5PipelineStages ? macro_k_tile_count : kP5PipelineStages;
      for (int drain = 0; drain < outstanding_stages; ++drain) {
        p5_tma_empty_mbar[producer_stage].wait(producer_phase);
        advance_pipeline_state(producer_stage, producer_phase);
      }
    }

    if (is_fp4_consumer_thread) {
      auto tiled_mma = TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_stage0_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 0 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto sA_stage1_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 1 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto sB_stage0_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 0 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto sB_stage1_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 1 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto sSFA_stage0_ = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 0 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto sSFA_stage1_ = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 1 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});

      auto sA_stage0 = cute::as_position_independent_swizzle_tensor(sA_stage0_);
      auto sA_stage1 = cute::as_position_independent_swizzle_tensor(sA_stage1_);
      auto sB_stage0 = cute::as_position_independent_swizzle_tensor(sB_stage0_);
      auto sB_stage1 = cute::as_position_independent_swizzle_tensor(sB_stage1_);
      auto sScaleA_stage0 = cute::as_position_independent_swizzle_tensor(sSFA_stage0_);
      auto sScaleA_stage1 = cute::as_position_independent_swizzle_tensor(sSFA_stage1_);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

      auto tCrA = thread_mma.partition_fragment_A(sA_stage0(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB_stage0(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
          sSFA_stage0_(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA_stage0 = s2r_thr_A.partition_S(sA_stage0);
      auto tCsA_stage1 = s2r_thr_A.partition_S(sA_stage1);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB_stage0 = s2r_thr_B.partition_S(sB_stage0);
      auto tCsB_stage1 = s2r_thr_B.partition_S(sB_stage1);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          SmemCopyAtomSFA{},
          Traits::GetLayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA_stage0 = s2r_thr_SFA.partition_S(sScaleA_stage0);
      auto tCsSFA_stage1 = s2r_thr_SFA.partition_S(sScaleA_stage1);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          SmemCopyAtomSFB{},
          Traits::GetLayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto scale_coords = cute::make_identity_tensor(cute::shape(sScaleB));
      auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords);
      auto dense_c = cute::make_identity_tensor(
          cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
      auto part_c = thread_mma.partition_C(dense_c);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          typename Traits::AccumLayout{});
      using MMAOp = typename TiledMma::MMA_Op;
      typename TiledMma::Atom mma_atom;
      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
      constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
      const int blocks_per_k_block = kMacroScaleBytes / SfbKBlocks;

      int consumer_stage = 0;
      uint32_t consumer_phase = 0;
      for (int tile = 0; tile < macro_k_tile_count; ++tile) {
        const std::size_t block_base = static_cast<std::size_t>(tile) * kMacroScaleBytes;
        const int curr_stage = consumer_stage;
        p5_tma_full_mbar[curr_stage].wait(consumer_phase);

        auto load_sfb_kblock = [&](int k_block) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto sfb_atom = tCrSFB(cute::_, n, k_block);
            auto row_anchor = tCsSFB_coords(cute::_, n, 0, cute::Int<0>{});
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (b_base_row >= 0 && b_base_row < valid_rows &&
                local_row0 >= 0 && local_row0 < valid_rows) {
              packed_scale_word = LoadExecutionScaleWord(
                  input_matmul_block_scales,
                  static_cast<std::size_t>(row_start + local_row0),
                  block_base + static_cast<std::size_t>(k_block * blocks_per_k_block),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
            FillP15ScaleFragmentWord(sfb_atom, packed_scale_word);
          }
        };

        auto copy_kblock = [&](int k_block) {
          if (curr_stage == 0) {
            cute::copy(
                s2r_copy_A,
                tCsA_stage0(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrA_cv(cute::_, cute::_, k_block));
            cute::copy(
                s2r_copy_B,
                tCsB_stage0(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrB_cv(cute::_, cute::_, k_block));
            cute::copy(
                tCsSFA_stage0(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrSFA_cv(cute::_, cute::_, k_block));
          } else {
            cute::copy(
                s2r_copy_A,
                tCsA_stage1(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrA_cv(cute::_, cute::_, k_block));
            cute::copy(
                s2r_copy_B,
                tCsB_stage1(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrB_cv(cute::_, cute::_, k_block));
            cute::copy(
                tCsSFA_stage1(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrSFA_cv(cute::_, cute::_, k_block));
          }
          cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k_block));
          cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k_block));
          load_sfb_kblock(k_block);
        };

        copy_kblock(0);

        if (g_enable_p5_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          g_p5_scale_trace.valid = 1;
          g_p5_scale_trace.row_start = row_start;
          g_p5_scale_trace.valid_rows = valid_rows;
          g_p5_scale_trace.output_row_base = output_row_base;
          for (int m = 0; m < M_tiles; ++m) {
            auto c_atom_coords = part_c(cute::_, m, 0);
            auto coord0 = c_atom_coords(0);
            const int a_base_row = static_cast<int>(cute::get<0>(coord0));
            g_p5_scale_trace.a_base_rows[m] = a_base_row;
            g_p5_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            g_p5_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < N_tiles; ++n) {
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            g_p5_scale_trace.b_base_rows[n] = b_base_row;
            g_p5_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            g_p5_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }

        for (int k = 0; k < K_blocks; ++k) {
          const int next_k = k + 1;
          if (next_k < K_blocks) {
            copy_kblock(next_k);
          }
          for (int n = 0; n < N_tiles; ++n) {
            for (int m = 0; m < M_tiles; ++m) {
              auto a_atom = tCrA(cute::_, m, k);
              auto b_atom = tCrB(cute::_, n, k);
              auto c_atom = accum_tensor(cute::_, m, n);
              auto sfa_atom = tCrSFA(cute::_, m, k);
              auto sfb_atom = tCrSFB(cute::_, n, k);
              auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
              auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
              mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
            }
          }
        }

        __syncwarp();
        if (lane_id == 0) {
          p5_tma_empty_mbar[curr_stage].arrive();
        }
        advance_pipeline_state(consumer_stage, consumer_phase);
      }
    }
    }
  } else {
    for (std::size_t macro_k_base = 0; macro_k_base < weight.input_cols; macro_k_base += kMacroTileK) {
    const std::size_t remaining_k = weight.input_cols - macro_k_base;
    const std::size_t macro_k =
        remaining_k < static_cast<std::size_t>(kMacroTileK)
            ? remaining_k
            : static_cast<std::size_t>(kMacroTileK);
    const std::size_t available_bytes = macro_k / 2u;
    const std::size_t available_blocks = macro_k / fused_decode::kNvfp4BlockWidth;
    const std::size_t block_base = macro_k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = macro_k_base / 2u;

    auto a_scale_tensor_raw = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
    auto b_scale_tensor_raw = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
    auto a_scale_tensor = cute::as_position_independent_swizzle_tensor(a_scale_tensor_raw);
    auto b_scale_tensor = cute::as_position_independent_swizzle_tensor(b_scale_tensor_raw);
    auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
    auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
    auto stage0_SFA = SmemLayoutSFA{}(cute::_, cute::_, cute::Int<0>{});
    auto stage0_SFB = SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{});
    auto* sw_a = reinterpret_cast<std::uint8_t*>(smem_swizzled_a);
    auto* sw_b = reinterpret_cast<std::uint8_t*>(smem_swizzled_b);

    if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
      for (int i = tid; i < Traits::kScaleSmemCosizeA; i += blockDim.x) {
        a_scale_smem[i] = nvfp4_bridge::MakeScaleElement(0u);
      }
      for (int i = tid; i < Traits::kScaleSmemCosizeB; i += blockDim.x) {
        b_scale_smem[i] = nvfp4_bridge::MakeScaleElement(0u);
      }
      __syncthreads();
    }

    if ((use_p5_tma_a || use_p5_tma_b) && is_p5_tma_thread) {
      p5_tma_full_mbar[0].init(1);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
      if (is_p5_tma_thread) {
        using X = cute::Underscore;
        using ProducerBarrierType = typename cutlass::arch::ClusterTransactionBarrier::ValueType;
        auto& barrier = p5_tma_full_mbar[0];
        uint32_t stage_bytes = 0u;
        const int output_tile_coord = output_row_base / 128;
        const int k_tile_coord = static_cast<int>(macro_k_base / 128u);
        if (use_p5_tma_a) {
          const auto& tma_load_a = *p5_tma_load_a_ptr;
          auto mA_mkl = tma_load_a.get_tma_tensor(cute::make_shape(
              static_cast<int32_t>(weight.output_rows),
              static_cast<int32_t>(weight.input_cols),
              cute::Int<1>{}));
          auto gA_mkl = cute::local_tile(
              mA_mkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<cute::_1, X, cute::_1>{});
          auto block_tma_a = tma_load_a.get_slice(0);
          auto gA = gA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
          auto tAgA = block_tma_a.partition_S(gA);
          auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
          auto sA = cute::as_position_independent_swizzle_tensor(sA_);
          auto tAsA = block_tma_a.partition_D(sA);
          auto tma_copy_a =
              tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_a,
              tAgA(cute::_, cute::_, cute::_, k_tile_coord),
              tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Operand tiles use compact staged layouts, so size(...) matches the
          // actual TMA transaction footprint for barrier accounting.
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::size(cute::take<0, 2>(SmemLayoutA{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementAB>));
        }
        if (use_p5_tma_sfa) {
          const auto& tma_load_sfa = *p5_tma_load_sfa_ptr;
          auto mSFA_mkl = tma_load_sfa.get_tma_tensor(cute::shape(
              routed_p5_tma::MakeP5ScaleLayoutSFA(
                  static_cast<int32_t>(weight.output_rows),
                  static_cast<int32_t>(weight.input_cols))));
          auto gSFA_mkl = cute::local_tile(
              mSFA_mkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<cute::_1, X, cute::_1>{});
          auto block_tma_sfa = tma_load_sfa.get_slice(0);
          auto gSFA = gSFA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
          auto tAgSFA = block_tma_sfa.partition_S(gSFA);
          auto sSFA_ = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
          auto sSFA = cute::as_position_independent_swizzle_tensor(sSFA_);
          auto tAsSFA = block_tma_sfa.partition_D(sSFA);
          auto tma_copy_sfa =
              tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_sfa,
              tAgSFA(cute::_, cute::_, cute::_, k_tile_coord),
              tAsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Scale layouts can have padded/non-compact strides, so barrier
          // bytes must use cosize(...) rather than size(...).
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::cosize(cute::take<0, 2>(SmemLayoutSFA{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementSF>));
        }
        if (use_p5_tma_b) {
          const auto& tma_load_b = p5_tma_load_b_descriptors[cta_index];
          auto mB_nkl = tma_load_b.get_tma_tensor(cute::make_shape(
              static_cast<int32_t>(valid_rows),
              static_cast<int32_t>(weight.input_cols),
              cute::Int<1>{}));
          auto gB_nkl = cute::local_tile(
              mB_nkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<X, cute::_1, cute::_1>{});
          auto block_tma_b = tma_load_b.get_slice(0);
          auto gB = gB_nkl(cute::_, cute::_, 0, cute::_, 0);
          auto tBgB = block_tma_b.partition_S(gB);
          auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
          auto sB = cute::as_position_independent_swizzle_tensor(sB_);
          auto tBsB = block_tma_b.partition_D(sB);
          auto tma_copy_b =
              tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_b,
              tBgB(cute::_, cute::_, cute::_, cute::Int<0>{}),
              tBsB(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Operand tiles use compact staged layouts, so size(...) matches the
          // actual TMA transaction footprint for barrier accounting.
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::size(cute::take<0, 2>(SmemLayoutB{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementAB>));
        }
        if (use_p5_tma_sfb) {
          const auto& tma_load_sfb = p5_tma_load_sfb_descriptors[cta_index];
          auto mSFB_nkl = tma_load_sfb.get_tma_tensor(cute::shape(
              routed_p5_tma::MakeP5ScaleLayoutSFB(
                  static_cast<int32_t>(kP5ScaleTmaRowTile),
                  static_cast<int32_t>(weight.input_cols))));
          auto gSFB_nkl = cute::local_tile(
              mSFB_nkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<X, cute::_1, cute::_1>{});
          auto block_tma_sfb = tma_load_sfb.get_slice(0);
          auto gSFB = gSFB_nkl(cute::_, cute::_, 0, cute::_, 0);
          auto tBgSFB = block_tma_sfb.partition_S(gSFB);
          auto sSFB_ = cute::make_tensor(
              cute::make_smem_ptr(
                  (use_p5_tma_sfb_direct || use_p5_tma_sfb_direct_slice)
                      ? b_scale_smem
                      : p5_sfb_tma_smem),
              SmemLayoutSFB{});
          auto sSFB = cute::as_position_independent_swizzle_tensor(sSFB_);
          auto tBsSFB = block_tma_sfb.partition_D(sSFB);
          auto tma_copy_sfb =
              tma_load_sfb.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_sfb,
              tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
              tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Scale layouts can have padded/non-compact strides, so barrier
          // bytes must use cosize(...) rather than size(...).
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementSF>));
        }
        if (stage_bytes != 0u) {
          p5_tma_full_mbar[0].arrive_and_expect_tx(stage_bytes);
        }
      }
    }

    if (!use_p5_tma_a || !use_p5_tma_sfa) {
      for (int row = tid; row < kOutputTile; row += blockDim.x) {
        const bool row_valid = row < output_rows_this_tile;
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        if (!use_p5_tma_a) {
#pragma unroll
          for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
            std::uint8_t value = 0u;
            if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
              value = weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)];
            }
            auto elem_offset = stage0_A(row, byte_index * 2);
            sw_a[static_cast<int>(elem_offset) / 2] = value;
          }
        }
        if (!use_p5_tma_sfa && row_valid) {
          std::uint8_t scale_bytes[kMacroScaleBytes] = {};
#pragma unroll
          for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
            if (static_cast<std::size_t>(scale_index) < available_blocks) {
              scale_bytes[scale_index] = LoadExecutionScaleByte(
                  weight.matmul_block_scales_data,
                  source_row,
                  block_base + static_cast<std::size_t>(scale_index),
                  padded_blocks_per_row,
                  Nvfp4ScaleLayout::kSwizzled128x4);
            }
          }
          if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              g_p13_scale_trace.a_loaded_words[row] = PackScaleWord4(
                  scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]);
            }
            for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(a_scale_tensor));
                 ++scale_col) {
              const std::uint8_t value =
                  scale_col < 4 ? scale_bytes[scale_col] : std::uint8_t{0};
              StoreScaleTensorByte(a_scale_tensor, row, scale_col, value);
            }
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              for (int col = 0; col < 8; ++col) {
                g_p13_scale_trace.a_post_store_raw[row * 8 + col] = static_cast<std::uint8_t>(
                    a_scale_tensor(row, col, cute::Int<0>{}).raw());
              }
            }
          } else {
            nvfp4_bridge::StoreTracedScaleBytes(a_scale_tensor, scale_bytes, row);
          }
        } else if (!use_p5_tma_sfa) {
          nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
        }
      }
    }

    if (!use_p5_tma_b || (!use_p5_tma_sfb && !use_p5_direct_gmem_sfb)) {
      for (int row = tid; row < kProfileTokenRows; row += blockDim.x) {
        const bool row_valid = row < valid_rows;
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        if (!use_p5_tma_b) {
#pragma unroll
          for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
            std::uint8_t value = 0u;
            if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
              value = packed_input[src_offset + static_cast<std::size_t>(byte_index)];
            }
            auto elem_offset = stage0_B(row, byte_index * 2);
            sw_b[static_cast<int>(elem_offset) / 2] = value;
          }
        }
        if (!use_p5_tma_sfb && !use_p5_direct_gmem_sfb && row_valid) {
          std::uint8_t scale_bytes[kMacroScaleBytes] = {};
#pragma unroll
          for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
            if (static_cast<std::size_t>(scale_index) < available_blocks) {
              scale_bytes[scale_index] = LoadExecutionScaleByte(
                  input_matmul_block_scales,
                  source_row,
                  block_base + static_cast<std::size_t>(scale_index),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
          }
          if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              g_p13_scale_trace.b_loaded_words[row] = PackScaleWord4(
                  scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]);
            }
            for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(b_scale_tensor));
                 ++scale_col) {
              const std::uint8_t value =
                  scale_col < 4 ? scale_bytes[scale_col] : std::uint8_t{0};
              StoreScaleTensorByte(b_scale_tensor, row, scale_col, value);
            }
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              for (int col = 0; col < 8; ++col) {
                g_p13_scale_trace.b_post_store_raw[row * 8 + col] = static_cast<std::uint8_t>(
                    b_scale_tensor(row, col, cute::Int<0>{}).raw());
              }
            }
          } else {
            nvfp4_bridge::StoreTracedScaleBytes(b_scale_tensor, scale_bytes, row);
          }
        } else if (!use_p5_tma_sfb &&
                   !use_p5_direct_gmem_sfb) {
          nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
        }
      }
    }
    const bool need_p5_tma_wait =
        (use_p5_tma_a || use_p5_tma_b) &&
        (is_fp4_consumer_thread || use_p5_tma_sfb_remap_fallback);
    if (need_p5_tma_wait) {
      p5_tma_full_mbar[0].wait(0);
    }
    if (use_p5_tma_sfb_remap_fallback) {
      auto p5_sSFB_tma = cute::make_tensor(
          cute::make_smem_ptr(p5_sfb_tma_smem), SmemLayoutSFB{});
      auto p5_sSFB_tma_logical = cute::as_position_independent_swizzle_tensor(p5_sSFB_tma);
      auto b_scale_logical = cute::as_position_independent_swizzle_tensor(b_scale_tensor_raw);
      constexpr int kLogicalCols = cute::size<1>(SmemLayoutSFB{});
      for (int row = tid; row < kProfileTokenRows; row += blockDim.x) {
        if (row < valid_rows) {
#pragma unroll
          for (int col = 0; col < kLogicalCols; ++col) {
            b_scale_logical(row, col, cute::Int<0>{}) =
                p5_sSFB_tma_logical(p5_sfb_row_offset + row, col, cute::Int<0>{});
          }
        } else {
#pragma unroll
          for (int col = 0; col < kLogicalCols; ++col) {
            b_scale_logical(row, col, cute::Int<0>{}).storage = 0u;
          }
        }
      }
    }
    const bool need_stage_thread_sync =
        !use_p5_tma_a || !use_p5_tma_sfa || !use_p5_tma_b ||
        (!use_p5_tma_sfb && !use_p5_direct_gmem_sfb) ||
        use_p5_tma_sfb_remap_fallback;
    if (need_stage_thread_sync) {
      __syncthreads();
    }

    if (is_fp4_consumer_thread) {
      auto tiled_mma = TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
      auto sSFA = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
      auto p5_sSFB_tma =
          cute::make_tensor(cute::make_smem_ptr(p5_sfb_tma_smem), SmemLayoutSFB{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);
      auto p5_sScaleB_tma = cute::as_position_independent_swizzle_tensor(p5_sSFB_tma);

      auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
          sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA = s2r_thr_A.partition_S(sA);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB = s2r_thr_B.partition_S(sB);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          SmemCopyAtomSFA{},
          Traits::GetLayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          SmemCopyAtomSFB{},
          Traits::GetLayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          typename Traits::AccumLayout{});

      cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
      cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        auto dense_c = cute::make_identity_tensor(
            cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
        auto part_c = thread_mma.partition_C(dense_c);
        constexpr int SfaMTiles = cute::size<1>(decltype(tCrSFA){});
        constexpr int SfaKBlocks = cute::size<2>(decltype(tCrSFA){});
        constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
        constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
        const int sfa_blocks_per_k_block = static_cast<int>(available_blocks / SfaKBlocks);
        const int sfb_blocks_per_k_block = static_cast<int>(available_blocks / SfbKBlocks);
        auto scale_coords_sfa = cute::make_identity_tensor(cute::shape(sScaleA));
        auto tCsSFA_coords = s2r_thr_SFA.partition_S(scale_coords_sfa);
        auto scale_coords_sfb = cute::make_identity_tensor(cute::shape(sScaleB));
        auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords_sfb);
        for (int k = 0; k < SfaKBlocks; ++k) {
          for (int m = 0; m < SfaMTiles; ++m) {
            auto row_anchor = tCsSFA_coords(cute::_, m, 0, cute::Int<0>{});
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (local_row0 >= 0 && local_row0 < output_rows_this_tile) {
              packed_scale_word = LoadExecutionScaleWord(
                  weight.matmul_block_scales_data,
                  static_cast<std::size_t>(output_row_base + local_row0),
                  block_base + static_cast<std::size_t>(k * sfa_blocks_per_k_block),
                  padded_blocks_per_row,
                  Nvfp4ScaleLayout::kSwizzled128x4);
            }
            FillP15ScaleFragmentWord(tCrSFA(cute::_, m, k), packed_scale_word);
          }
        }
        for (int k = 0; k < SfbKBlocks; ++k) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto row_anchor = tCsSFB_coords(cute::_, n, 0, cute::Int<0>{});
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (local_row0 >= 0 && local_row0 < valid_rows) {
              packed_scale_word = LoadExecutionScaleWord(
                  input_matmul_block_scales,
                  static_cast<std::size_t>(row_start + local_row0),
                  block_base + static_cast<std::size_t>(k * sfb_blocks_per_k_block),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
            FillP15ScaleFragmentWord(tCrSFB(cute::_, n, k), packed_scale_word);
          }
        }
      } else {
        cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
      }
      if (use_p5_direct_gmem_sfb) {
        auto scale_coords = cute::make_identity_tensor(cute::shape(sScaleB));
        auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords);
        auto dense_c = cute::make_identity_tensor(
            cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
        auto part_c = thread_mma.partition_C(dense_c);
        constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
        constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
        const int blocks_per_k_block = static_cast<int>(available_blocks / SfbKBlocks);
        for (int k = 0; k < SfbKBlocks; ++k) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto sfb_atom = tCrSFB(cute::_, n, k);
            auto row_anchor = tCsSFB_coords(cute::_, n, 0, cute::Int<0>{});
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (local_row0 >= 0 && local_row0 < valid_rows) {
              packed_scale_word = LoadExecutionScaleWord(
                  input_matmul_block_scales,
                  static_cast<std::size_t>(row_start + local_row0),
                  block_base + static_cast<std::size_t>(k * blocks_per_k_block),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
            FillP15ScaleFragmentWord(sfb_atom, packed_scale_word);
          }
        }
      } else if (use_p5_tma_sfb_direct_slice) {
        auto p5_sScaleB_src = cute::domain_offset(
            cute::make_coord(p5_sfb_row_offset, 0, 0),
            p5_sScaleB_tma);
        auto tCsSFB = s2r_thr_SFB.partition_S(p5_sScaleB_src);
        cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);
      } else if (use_p5_tma_sfb_register_assembly) {
        auto p5_sScaleB_src = cute::domain_offset(
            cute::make_coord(p5_sfb_row_offset, 0, 0),
            p5_sScaleB_tma);
        auto scale_coords = cute::make_identity_tensor(cute::shape(sScaleB));
        auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords);
        auto dense_c = cute::make_identity_tensor(
            cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
        auto part_c = thread_mma.partition_C(dense_c);
        constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
        constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
        for (int k = 0; k < SfbKBlocks; ++k) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto c_atom_coords = part_c(cute::_, k, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            auto sfb_atom = tCrSFB(cute::_, n, k);
            if (b_base_row >= 0 && b_base_row < valid_rows) {
              auto sfb_atom_coords =
                  tCsSFB_coords(cute::_, n, k, cute::Int<0>{});
              for (int elem = 0; elem < static_cast<int>(cute::size(sfb_atom)); ++elem) {
                sfb_atom(elem) = p5_sScaleB_src(sfb_atom_coords(elem));
              }
            } else {
              for (int elem = 0; elem < static_cast<int>(cute::size(sfb_atom)); ++elem) {
                sfb_atom(elem) = cute::float_ue4m3_t::bitcast(std::uint8_t{0});
              }
            }
          }
        }
      } else if constexpr (Profile != nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
        cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);
      }

      using MMAOp = typename TiledMma::MMA_Op;
      for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
        cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
        cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
      }

      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
        if (g_enable_p5_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          g_p5_scale_trace.valid = 1;
          g_p5_scale_trace.row_start = row_start;
          g_p5_scale_trace.valid_rows = valid_rows;
          g_p5_scale_trace.output_row_base = output_row_base;
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int m = 0; m < M_tiles; ++m) {
            auto c_atom_coords = part_c(cute::_, m, 0);
            auto coord0 = c_atom_coords(0);
            const int a_base_row = static_cast<int>(cute::get<0>(coord0));
            g_p5_scale_trace.a_base_rows[m] = a_base_row;
            g_p5_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            g_p5_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < N_tiles; ++n) {
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            g_p5_scale_trace.b_base_rows[n] = b_base_row;
            g_p5_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            g_p5_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
      } else if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        if (g_enable_p13_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          auto scale_coords_sfa = cute::make_identity_tensor(cute::shape(sScaleA));
          auto scale_coords_sfb = cute::make_identity_tensor(cute::shape(sScaleB));
          auto tCsSFA_coords = s2r_thr_SFA.partition_S(scale_coords_sfa);
          auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords_sfb);
          auto tCsSFA_coords_stage0 =
              tCsSFA_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          auto tCsSFB_coords_stage0 =
              tCsSFB_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          int a_scale_rows[nvfp4_bridge::kTracedP13ScaleFragmentCosizeA];
          int a_scale_cols[nvfp4_bridge::kTracedP13ScaleFragmentCosizeA];
          int b_scale_rows[nvfp4_bridge::kTracedP13ScaleFragmentCosizeB];
          int b_scale_cols[nvfp4_bridge::kTracedP13ScaleFragmentCosizeB];
          {
            int physical = 0;
            for (int i = 0;
                 i < cute::size<0>(tCsSFA_coords_stage0) &&
                 physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeA;
                 ++i) {
              for (int j = 0;
                   j < cute::size<1>(tCsSFA_coords_stage0) &&
                   physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeA;
                   ++j) {
                for (int k = 0;
                     k < cute::size<2>(tCsSFA_coords_stage0) &&
                     physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeA;
                     ++k) {
                  auto coord = tCsSFA_coords_stage0(cute::make_coord(i, j, k));
                  a_scale_rows[physical] = nvfp4_bridge::CoordGet0(coord);
                  a_scale_cols[physical] = nvfp4_bridge::CoordGet1(coord);
                  ++physical;
                }
              }
            }
          }
          {
            int physical = 0;
            for (int i = 0;
                 i < cute::size<0>(tCsSFB_coords_stage0) &&
                 physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeB;
                 ++i) {
              for (int j = 0;
                   j < cute::size<1>(tCsSFB_coords_stage0) &&
                   physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeB;
                   ++j) {
                for (int k = 0;
                     k < cute::size<2>(tCsSFB_coords_stage0) &&
                     physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeB;
                     ++k) {
                  auto coord = tCsSFB_coords_stage0(cute::make_coord(i, j, k));
                  b_scale_rows[physical] = nvfp4_bridge::CoordGet0(coord);
                  b_scale_cols[physical] = nvfp4_bridge::CoordGet1(coord);
                  ++physical;
                }
              }
            }
          }
          g_p13_scale_trace.valid = 1;
          g_p13_scale_trace.row_start = row_start;
          g_p13_scale_trace.valid_rows = valid_rows;
          g_p13_scale_trace.output_row_base = output_row_base;
          for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 8; ++col) {
              g_p13_scale_trace.a_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  sScaleA(row, col, cute::Int<0>{}).raw());
              g_p13_scale_trace.b_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  sScaleB(row, col, cute::Int<0>{}).raw());
            }
          }
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int m = 0; m < min(M_tiles, 2); ++m) {
            const int a_base_row = a_scale_rows[m * 4];
            g_p13_scale_trace.a_base_rows[m] = a_base_row;
            g_p13_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            std::uint32_t packed_from_smem = 0u;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = m * 4 + elem;
              const int scale_row = a_scale_rows[physical];
              const int scale_col = a_scale_cols[physical];
              const auto raw = static_cast<std::uint8_t>(
                  sScaleA(scale_row, scale_col, cute::Int<0>{}).raw());
              g_p13_scale_trace.a_scale_rows[physical] = scale_row;
              g_p13_scale_trace.a_scale_cols[physical] = scale_col;
              g_p13_scale_trace.a_scale_raw[physical] = raw;
              packed_from_smem |= static_cast<std::uint32_t>(raw) << (elem * 8);
            }
            g_p13_scale_trace.a_smem_words[m] = packed_from_smem;
            g_p13_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < min(N_tiles, 2); ++n) {
            const int b_base_row = b_scale_rows[n * 4];
            g_p13_scale_trace.b_base_rows[n] = b_base_row;
            g_p13_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            std::uint32_t packed_from_smem = 0u;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = n * 4 + elem;
              const int scale_row = b_scale_rows[physical];
              const int scale_col = b_scale_cols[physical];
              const auto raw = static_cast<std::uint8_t>(
                  sScaleB(scale_row, scale_col, cute::Int<0>{}).raw());
              g_p13_scale_trace.b_scale_rows[physical] = scale_row;
              g_p13_scale_trace.b_scale_cols[physical] = scale_col;
              g_p13_scale_trace.b_scale_raw[physical] = raw;
              packed_from_smem |= static_cast<std::uint32_t>(raw) << (elem * 8);
            }
            g_p13_scale_trace.b_smem_words[n] = packed_from_smem;
            g_p13_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
        if (p13_debug_capture_cta != 0 &&
            block_base == 0 &&
            ((g_p13_debug_trace_target_thread_id < 0 && thread_id == 0) ||
             thread_id == g_p13_debug_trace_target_thread_id)) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          auto a_coords = cute::make_identity_tensor(cute::shape(sA));
          auto tCsA_coords = s2r_thr_A.partition_S(a_coords);
          auto tCsA_coords_stage0 =
              tCsA_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          auto b_coords = cute::make_identity_tensor(cute::shape(sB));
          auto tCsB_coords = s2r_thr_B.partition_S(b_coords);
          auto tCsB_coords_stage0 =
              tCsB_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          int row_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
          int col_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
          int a_copy_rows[32];
          int a_copy_cols[32];
          int b_copy_rows[16];
          int b_copy_cols[16];
          nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
              part_c,
              nvfp4_bridge::kTracedP13CCopyCoordCapacity,
              row_coords,
              col_coords);
          nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
              tCsA_coords_stage0,
              32,
              a_copy_rows,
              a_copy_cols);
          nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
              tCsB_coords_stage0,
              16,
              b_copy_rows,
              b_copy_cols);
          g_p13_debug_trace.valid = 1;
          g_p13_debug_trace.row_start = row_start;
          g_p13_debug_trace.valid_rows = valid_rows;
          g_p13_debug_trace.output_row_base = output_row_base;
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int physical = 0; physical < nvfp4_bridge::kTracedP13CCopyCoordCapacity; ++physical) {
            g_p13_debug_trace.store_rows[physical] = row_coords[physical];
            g_p13_debug_trace.store_cols[physical] = col_coords[physical];
          }
          for (int physical = 0; physical < 32; ++physical) {
            g_p13_debug_trace.a_copy_rows[physical] = a_copy_rows[physical];
            g_p13_debug_trace.a_copy_cols[physical] = a_copy_cols[physical];
            std::uint8_t raw = 0u;
            if (a_copy_rows[physical] >= 0 &&
                a_copy_rows[physical] < valid_rows &&
                a_copy_cols[physical] >= 0 &&
                a_copy_cols[physical] < static_cast<int>(weight.input_cols)) {
              raw = LoadPackedFp4Nibble(
                  packed_input +
                      (static_cast<std::size_t>(row_start + a_copy_rows[physical]) *
                       packed_row_bytes),
                  a_copy_cols[physical]);
            }
            g_p13_debug_trace.a_copy_raw[physical] = raw;
          }
          for (int physical = 0; physical < 16; ++physical) {
            g_p13_debug_trace.b_copy_rows[physical] = b_copy_rows[physical];
            g_p13_debug_trace.b_copy_cols[physical] = b_copy_cols[physical];
          }
          for (int m = 0; m < min(M_tiles, 2); ++m) {
            g_p13_debug_trace.a_scale_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < min(N_tiles, 2); ++n) {
            g_p13_debug_trace.b_scale_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
      } else if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP15) {
        if (g_enable_p15_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          g_p15_scale_trace.valid = 1;
          g_p15_scale_trace.row_start = row_start;
          g_p15_scale_trace.valid_rows = valid_rows;
          g_p15_scale_trace.output_row_base = output_row_base;
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int m = 0; m < M_tiles; ++m) {
            auto c_atom_coords = part_c(cute::_, m, 0);
            auto coord0 = c_atom_coords(0);
            const int a_base_row = static_cast<int>(cute::get<0>(coord0));
            g_p15_scale_trace.a_base_rows[m] = a_base_row;
            g_p15_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            g_p15_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < N_tiles; ++n) {
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            g_p15_scale_trace.b_base_rows[n] = b_base_row;
            g_p15_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            g_p15_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
      }

      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      typename TiledMma::Atom mma_atom;
      for (int k = 0; k < K_blocks; ++k) {
        for (int n = 0; n < N_tiles; ++n) {
          for (int m = 0; m < M_tiles; ++m) {
            auto a_atom = tCrA(cute::_, m, k);
            auto b_atom = tCrB(cute::_, n, k);
            auto c_atom = accum_tensor(cute::_, m, n);
            auto sfa_atom = tCrSFA(cute::_, m, k);
            auto sfb_atom = tCrSFB(cute::_, n, k);
            auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
            auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
            mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
          }
        }
      }
      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        if (p13_debug_capture_cta != 0 &&
            block_base == 0 &&
            ((g_p13_debug_trace_target_thread_id < 0 && thread_id == 0) ||
             thread_id == g_p13_debug_trace_target_thread_id)) {
          for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 2; ++n) {
              for (int reg = 0; reg < 4; ++reg) {
                g_p13_debug_trace.block0_accum_regs[m][n][reg] =
                    accum_tensor(reg, m, n);
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
  }



  if (warp_id < kFp4ConsumerWarps) {
      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        if (p13_debug_capture_cta != 0 &&
            ((g_p13_debug_trace_target_thread_id < 0 && warp_id == 0 && lane_id == 0) ||
             tid == g_p13_debug_trace_target_thread_id)) {
          auto accum_tensor = cute::make_tensor(
            reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
            typename Traits::AccumLayout{});
        for (int m = 0; m < 2; ++m) {
          for (int n = 0; n < 2; ++n) {
            for (int reg = 0; reg < 4; ++reg) {
              g_p13_debug_trace.accum_regs[m][n][reg] =
                  accum_tensor(reg, m, n);
            }
            }
          }
        }
      }
      nvfp4_bridge::StoreUnifiedRoutedFp4Output<Profile, P5Mode>(
          output_alpha,
          &accum_storage[0],
          warp_id * 32 + lane_id,
          output_row_base,
          valid_rows,
          output_rows_this_tile,
          static_cast<std::size_t>(row_start),
          output_rows_per_expert,
          output,
          input_per_row_tensor_scales,
          weight_tensor_scale,
          fp4_packed_data,
          fp4_block_scales,
          fp4_matmul_block_scales,
          fp4_activation_output_scale,
          fp4_cols,
          fp4_padded_blocks_per_row,
          fp4_scale_layout);
  }
}

template <typename OutputType, int kOutputTile>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmem(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const std::uint8_t* input_matmul_block_scales,
    Nvfp4ScaleLayout input_scale_layout,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kFp4ConsumerWarps = cute::size(nvfp4_bridge::TracedP13TiledMma{}) / 32;
  __shared__ std::uint8_t a_packed[kPlannedWmmaTileM][64 / 2];
  __shared__ std::uint8_t a_scale_smem[nvfp4_bridge::kTracedP13ScaleSmemCosizeA];
  __shared__ std::uint8_t b_packed[kOutputTile][64 / 2];
  __shared__ std::uint8_t b_scale_smem[nvfp4_bridge::kTracedP13ScaleSmemCosizeB];
  const auto a_tile_view =
      nvfp4_bridge::MakePackedTile64<kPlannedWmmaTileM>(&a_packed[0][0], nullptr);
  const auto b_tile_view =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&b_packed[0][0], nullptr);
  auto a_scale_tensor_raw =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[0]), nvfp4_bridge::TracedP13SmemLayoutSFA{});
  auto b_scale_tensor_raw =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[0]), nvfp4_bridge::TracedP13SmemLayoutSFB{});
  auto a_scale_tensor = cute::as_position_independent_swizzle_tensor(a_scale_tensor_raw);
  auto b_scale_tensor = cute::as_position_independent_swizzle_tensor(b_scale_tensor_raw);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * (*weight.tensor_scale_data);
  const bool p13_scale_map_probe = (g_enable_p13_scale_map_probe != 0);

  nvfp4_bridge::CFragment64 accum[nvfp4_bridge::kTracedP13MFragments][nvfp4_bridge::kTracedP13NFragments];
  if (warp_id < kFp4ConsumerWarps) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
        nvfp4_bridge::Clear(accum[m_fragment][n_fragment]);
      }
    }
  }

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += 64u) {
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = k_base / 2u;

    for (int row = tid; row < kPlannedWmmaTileM; row += blockDim.x) {
      if (row < valid_rows) {
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = a_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = packed_input[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            a_scale_tensor,
            PackScaleWord4(
                input_block_scales[scale_offset + 0u],
                input_block_scales[scale_offset + 1u],
                input_block_scales[scale_offset + 2u],
                input_block_scales[scale_offset + 3u]),
            row);
      } else {
        nvfp4_bridge::ZeroRow(a_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
      }
    }
    for (int row = tid; row < kOutputTile; row += blockDim.x) {
      if (row < output_rows_this_tile) {
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = b_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)];
        }
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            b_scale_tensor,
            LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                source_row,
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4),
            row);
      } else {
        nvfp4_bridge::ZeroRow(b_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
      }
    }
    if (p13_scale_map_probe) {
      for (int logical_row = tid;
           logical_row < static_cast<int>(cute::size<0>(a_scale_tensor));
           logical_row += blockDim.x) {
        const std::uint8_t value = static_cast<std::uint8_t>((logical_row + 1) & 0xff);
        for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(a_scale_tensor)); ++scale_col) {
          StoreScaleTensorByte(a_scale_tensor, logical_row, scale_col, value);
        }
      }
      for (int logical_row = tid;
           logical_row < static_cast<int>(cute::size<0>(b_scale_tensor));
           logical_row += blockDim.x) {
        const std::uint8_t value = static_cast<std::uint8_t>((logical_row + 1) & 0xff);
        for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(b_scale_tensor)); ++scale_col) {
          StoreScaleTensorByte(b_scale_tensor, logical_row, scale_col, value);
        }
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      nvfp4_bridge::AFragment64 a_fragments[nvfp4_bridge::kTracedP13MFragments];
      nvfp4_bridge::BFragment64 b_fragments[nvfp4_bridge::kTracedP13NFragments];
      if (p13_scale_map_probe) {
        nvfp4_bridge::LoadTracedP13AFragmentsRowMajor16x64<nvfp4_bridge::TracedP13TiledMma, kPlannedWmmaTileM>(
            &a_packed[0][0],
            a_scale_smem,
            warp_id * 32 + lane_id,
            0,
            a_fragments);
        nvfp4_bridge::LoadTracedP13BFragmentsColMajor64x8<nvfp4_bridge::TracedP13TiledMma, kOutputTile>(
            &b_packed[0][0],
            b_scale_smem,
            warp_id * 32 + lane_id,
            b_fragments);
        if (g_enable_p13_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            warp_id == 0 &&
            lane_id == 0) {
          g_p13_scale_trace.valid = 1;
          g_p13_scale_trace.row_start = row_start;
          g_p13_scale_trace.valid_rows = valid_rows;
          g_p13_scale_trace.output_row_base = output_row_base;
          for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 8; ++col) {
              g_p13_scale_trace.a_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  ScaleByteValue(a_scale_tensor(row, col, cute::Int<0>{})));
              g_p13_scale_trace.b_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  ScaleByteValue(b_scale_tensor(row, col, cute::Int<0>{})));
              g_p13_scale_trace.a_post_store_raw[row * 8 + col] =
                  g_p13_scale_trace.a_logical_raw[row * 8 + col];
              g_p13_scale_trace.b_post_store_raw[row * 8 + col] =
                  g_p13_scale_trace.b_logical_raw[row * 8 + col];
            }
          }
          for (int m = 0; m < 2; ++m) {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(a_fragments[m].scale[0]);
            g_p13_scale_trace.a_fragment_words[m] = packed;
            g_p13_scale_trace.a_smem_words[m] = packed;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = m * 4 + elem;
              const std::uint8_t raw =
                  static_cast<std::uint8_t>((packed >> (elem * 8)) & 0xffu);
              g_p13_scale_trace.a_scale_raw[physical] = raw;
              FindScaleTensorCoordByRaw(
                  a_scale_tensor,
                  raw,
                  g_p13_scale_trace.a_scale_rows[physical],
                  g_p13_scale_trace.a_scale_cols[physical]);
            }
            g_p13_scale_trace.a_base_rows[m] = g_p13_scale_trace.a_scale_rows[m * 4];
          }
          for (int n = 0; n < 2; ++n) {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(b_fragments[n].scale[0]);
            g_p13_scale_trace.b_fragment_words[n] = packed;
            g_p13_scale_trace.b_smem_words[n] = packed;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = n * 4 + elem;
              const std::uint8_t raw =
                  static_cast<std::uint8_t>((packed >> (elem * 8)) & 0xffu);
              g_p13_scale_trace.b_scale_raw[physical] = raw;
              FindScaleTensorCoordByRaw(
                  b_scale_tensor,
                  raw,
                  g_p13_scale_trace.b_scale_rows[physical],
                  g_p13_scale_trace.b_scale_cols[physical]);
            }
            g_p13_scale_trace.b_base_rows[n] = g_p13_scale_trace.b_scale_rows[n * 4];
          }
        }
      } else {
#pragma unroll
        for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
          const int row_base = m_fragment * 64;
          a_fragments[m_fragment] =
              nvfp4_bridge::LoadFragmentA_RowMajor16x64Tiled<
                  nvfp4_bridge::TracedP13TiledMma,
                  kPlannedWmmaTileM>(&a_packed[0][0], nullptr, row_base, warp_id * 32 + lane_id);
          a_fragments[m_fragment].scale[0] = static_cast<nvfp4_bridge::SFRegister>(
              LoadExecutionScaleWord(
                  weight.matmul_block_scales_data,
                  static_cast<std::size_t>(output_row_base),
                  block_base,
                  padded_blocks_per_row,
                  Nvfp4ScaleLayout::kSwizzled128x4));
        }
#pragma unroll
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          const int n_base = n_fragment * 8;
          b_fragments[n_fragment] =
              nvfp4_bridge::LoadFragmentB_ColMajor64x8Tiled<
                  nvfp4_bridge::TracedP13TiledMma,
                  kOutputTile>(&b_packed[0][0], nullptr, warp_id * 32 + lane_id, n_base);
          std::uint32_t packed_scale_word = 0u;
          if (valid_rows > 0) {
            packed_scale_word = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
          }
          b_fragments[n_fragment].scale[0] =
              static_cast<nvfp4_bridge::SFRegister>(packed_scale_word);
        }
      }
      if (g_enable_p13_debug_trace != 0 &&
          cta_index == 0 &&
          blockIdx.x == 0 &&
          block_base == 0 &&
          warp_id == 0 &&
          lane_id == 0) {
        auto mma = nvfp4_bridge::TracedP13TiledMma{};
        auto thr_mma = mma.get_thread_slice(warp_id * 32 + lane_id);
        auto ref_c = cute::make_identity_tensor(
            cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
        auto part_c = thr_mma.partition_C(ref_c);
        int row_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
        int col_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
        nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
            part_c,
            nvfp4_bridge::kTracedP13CCopyCoordCapacity,
            row_coords,
            col_coords);
        g_p13_debug_trace.valid = 1;
        g_p13_debug_trace.row_start = row_start;
        g_p13_debug_trace.valid_rows = valid_rows;
        g_p13_debug_trace.output_row_base = output_row_base;
        for (int physical = 0; physical < nvfp4_bridge::kTracedP13CCopyCoordCapacity; ++physical) {
          g_p13_debug_trace.store_rows[physical] = row_coords[physical];
          g_p13_debug_trace.store_cols[physical] = col_coords[physical];
        }
        for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
          for (int reg = 0; reg < 4; ++reg) {
            g_p13_debug_trace.a_regs[m_fragment][reg] = a_fragments[m_fragment].regs[reg];
          }
          g_p13_debug_trace.a_scale_words[m_fragment] =
              static_cast<std::uint32_t>(a_fragments[m_fragment].scale[0]);
        }
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          for (int reg = 0; reg < 2; ++reg) {
            g_p13_debug_trace.b_regs[n_fragment][reg] = b_fragments[n_fragment].regs[reg];
          }
          g_p13_debug_trace.b_scale_words[n_fragment] =
              static_cast<std::uint32_t>(b_fragments[n_fragment].scale[0]);
        }
      }
#pragma unroll
      for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
#pragma unroll
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          nvfp4_bridge::Gemm(accum[m_fragment][n_fragment], a_fragments[m_fragment], b_fragments[n_fragment]);
        }
      }
      if (g_enable_p13_debug_trace != 0 &&
          cta_index == 0 &&
          blockIdx.x == 0 &&
          block_base == 0 &&
          warp_id == 0 &&
          lane_id == 0) {
        for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
          for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
            for (int reg = 0; reg < 4; ++reg) {
              g_p13_debug_trace.block0_accum_regs[m_fragment][n_fragment][reg] =
                  accum[m_fragment][n_fragment].regs[reg];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    if (g_enable_p13_debug_trace != 0 &&
        cta_index == 0 &&
        blockIdx.x == 0 &&
        warp_id == 0 &&
        lane_id == 0) {
      for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          for (int reg = 0; reg < 4; ++reg) {
            g_p13_debug_trace.accum_regs[m_fragment][n_fragment][reg] =
                accum[m_fragment][n_fragment].regs[reg];
          }
        }
      }
    }
    nvfp4_bridge::StoreTracedP13CFragmentsRowMajor<nvfp4_bridge::TracedP13TiledMma>(
        output_alpha,
        accum,
        warp_id * 32 + lane_id,
        output_row_base,
        valid_rows,
        output_rows_this_tile,
        static_cast<std::size_t>(row_start),
        output_rows_per_expert,
        output);
  }
}

template <typename OutputType>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK128P12(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kOutputTile = 128;
  constexpr int kProfileTokenRows = 128;
  constexpr int kMacroTileK = 128;
  constexpr int kMacroTileBytes = kMacroTileK / 2;
  constexpr int kMacroScaleBytes = kMacroTileK / fused_decode::kNvfp4BlockWidth;
  using P12TiledMma = nvfp4_bridge::TracedP5TiledMma;
  using P12CollectiveMainloop = nvfp4_bridge::TracedP5CollectiveMainloop;
  using P12SmemLayoutA = nvfp4_bridge::TracedP5SmemLayoutA;
  using P12SmemLayoutB = nvfp4_bridge::TracedP5SmemLayoutB;
  using P12SmemLayoutSFA = nvfp4_bridge::TracedP5SmemLayoutSFA;
  using P12SmemLayoutSFB = nvfp4_bridge::TracedP5SmemLayoutSFB;
  using P12SmemCopyAtomA = nvfp4_bridge::TracedP5SmemCopyAtomA;
  using P12SmemCopyAtomB = nvfp4_bridge::TracedP5SmemCopyAtomB;
  using P12SmemCopyAtomSFA = nvfp4_bridge::TracedP5SmemCopyAtomSFA;
  using P12SmemCopyAtomSFB = nvfp4_bridge::TracedP5SmemCopyAtomSFB;
  constexpr int kFp4ConsumerWarps = cute::size(P12TiledMma{}) / 32;
  __shared__ std::uint8_t a_packed[kOutputTile][kMacroTileBytes];
  __shared__ alignas(1024) cute::array_aligned<
      nvfp4_cute::ElementSFCompute,
      nvfp4_bridge::kTracedP5ScaleSmemCosizeA>
      a_scale_smem_storage;
  __shared__ std::uint8_t b_packed[kProfileTokenRows][kMacroTileBytes];
  __shared__ alignas(1024) cute::array_aligned<
      nvfp4_cute::ElementSFCompute,
      nvfp4_bridge::kTracedP5ScaleSmemCosizeB>
      b_scale_smem_storage;
  using P12SmemAllocA = typename P12TiledMma::ValTypeA;
  using P12SmemAllocB = typename P12TiledMma::ValTypeB;
  static constexpr int kP12SwizzledAElems = cute::size(cute::take<0, 2>(P12SmemLayoutA{}));
  static constexpr int kP12SwizzledBElems = cute::size(cute::take<0, 2>(P12SmemLayoutB{}));
  __shared__ alignas(1024) cute::array_aligned<P12SmemAllocA, kP12SwizzledAElems> smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<P12SmemAllocB, kP12SwizzledBElems> smem_swizzled_b_storage;
  auto* a_scale_smem = a_scale_smem_storage.data();
  auto* b_scale_smem = b_scale_smem_storage.data();
  auto* smem_swizzled_a = smem_swizzled_a_storage.data();
  auto* smem_swizzled_b = smem_swizzled_b_storage.data();

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * (*weight.tensor_scale_data);

  nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kTracedP5AccumProfileCosize];
  if (warp_id < kFp4ConsumerWarps) {
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        nvfp4_bridge::TracedP5AccumProfileLayout{});
    cute::clear(accum_tensor);
  }

  for (std::size_t macro_k_base = 0; macro_k_base < weight.input_cols; macro_k_base += kMacroTileK) {
    const std::size_t remaining_k = weight.input_cols - macro_k_base;
    const std::size_t macro_k =
        remaining_k < static_cast<std::size_t>(kMacroTileK)
            ? remaining_k
            : static_cast<std::size_t>(kMacroTileK);
    const std::size_t available_bytes = macro_k / 2u;
    const std::size_t available_blocks = macro_k / fused_decode::kNvfp4BlockWidth;
    const std::size_t block_base = macro_k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = macro_k_base / 2u;

    auto a_scale_tensor = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), P12SmemLayoutSFA{});
    auto b_scale_tensor = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), P12SmemLayoutSFB{});

    for (int row = tid; row < kOutputTile; row += blockDim.x) {
      if (row < output_rows_this_tile) {
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = &a_packed[row][0];
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          dst[byte_index] =
              static_cast<std::size_t>(byte_index) < available_bytes
                  ? weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)]
                  : std::uint8_t{0};
        }
        std::uint8_t scale_bytes[kMacroScaleBytes] = {};
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          if (static_cast<std::size_t>(scale_index) < available_blocks) {
            scale_bytes[scale_index] = LoadExecutionScaleByte(
                weight.matmul_block_scales_data,
                source_row,
                block_base + static_cast<std::size_t>(scale_index),
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
          }
        }
        nvfp4_bridge::StoreTracedScaleBytes(a_scale_tensor, scale_bytes, row);
      } else {
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          a_packed[row][byte_index] = 0u;
        }
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
      }
    }
    for (int row = tid; row < kProfileTokenRows; row += blockDim.x) {
      if (row < valid_rows) {
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = &b_packed[row][0];
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          dst[byte_index] =
              static_cast<std::size_t>(byte_index) < available_bytes
                  ? packed_input[src_offset + static_cast<std::size_t>(byte_index)]
                  : std::uint8_t{0};
        }
        std::uint8_t scale_bytes[kMacroScaleBytes] = {};
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          if (static_cast<std::size_t>(scale_index) < available_blocks) {
            scale_bytes[scale_index] =
                input_block_scales[scale_offset + static_cast<std::size_t>(scale_index)];
          }
        }
        nvfp4_bridge::StoreTracedScaleBytes(b_scale_tensor, scale_bytes, row);
      } else {
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          b_packed[row][byte_index] = 0u;
        }
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
      }
    }
    {
      auto stage0_A = P12SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
      auto* sw_a = reinterpret_cast<std::uint8_t*>(smem_swizzled_a);
      constexpr int a_total_bytes = kOutputTile * kMacroTileBytes;
      for (int i = tid; i < a_total_bytes; i += blockDim.x) {
        const int row = i / kMacroTileBytes;
        const int col_byte = i % kMacroTileBytes;
        auto elem_offset = stage0_A(row, col_byte * 2);
        sw_a[static_cast<int>(elem_offset) / 2] = a_packed[row][col_byte];
      }

      auto stage0_B = P12SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
      auto* sw_b = reinterpret_cast<std::uint8_t*>(smem_swizzled_b);
      constexpr int b_total_bytes = kProfileTokenRows * kMacroTileBytes;
      for (int i = tid; i < b_total_bytes; i += blockDim.x) {
        const int row = i / kMacroTileBytes;
        const int col_byte = i % kMacroTileBytes;
        auto elem_offset = stage0_B(row, col_byte * 2);
        sw_b[static_cast<int>(elem_offset) / 2] = b_packed[row][col_byte];
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      auto tiled_mma = P12TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), P12SmemLayoutA{});
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), P12SmemLayoutB{});
      auto sSFA = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), P12SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), P12SmemLayoutSFB{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

      auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = P12CollectiveMainloop{}.partition_fragment_SFA(
          sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = P12CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(P12SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA = s2r_thr_A.partition_S(sA);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(P12SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB = s2r_thr_B.partition_S(sB);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          P12SmemCopyAtomSFA{},
          nvfp4_bridge::GetTracedP5LayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          P12SmemCopyAtomSFB{},
          nvfp4_bridge::GetTracedP5LayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
      auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          nvfp4_bridge::TracedP5AccumProfileLayout{});

      cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
      cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
      cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
      cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

      using P12MmaOp = typename P12TiledMma::MMA_Op;
      for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
        cute::fp4_shift_A(P12MmaOp{}, tCrA_cv(cute::_, cute::_, k));
        cute::fp4_shift_B(P12MmaOp{}, tCrB_cv(cute::_, cute::_, k));
      }

      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      typename P12TiledMma::Atom mma_atom;
      for (int k = 0; k < K_blocks; ++k) {
        for (int n = 0; n < N_tiles; ++n) {
          for (int m = 0; m < M_tiles; ++m) {
            auto a_atom = tCrA(cute::_, m, k);
            auto b_atom = tCrB(cute::_, n, k);
            auto c_atom = accum_tensor(cute::_, m, n);
            auto sfa_atom = tCrSFA(cute::_, m, k);
            auto sfb_atom = tCrSFB(cute::_, n, k);
            auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
            auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
            mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
          }
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    auto tiled_mma = P12TiledMma{};
    auto thread_mma = tiled_mma.get_thread_slice(warp_id * 32 + lane_id);
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        nvfp4_bridge::TracedP5AccumProfileLayout{});
    auto dense_c =
        cute::make_identity_tensor(cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
    auto part_c = thread_mma.partition_C(dense_c);
    for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
      auto coord = part_c(i);
      const int output_col_offset = static_cast<int>(cute::get<0>(coord));
      const int token_row = static_cast<int>(cute::get<1>(coord));
      if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
        continue;
      }
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_row);
      const std::size_t output_col =
          static_cast<std::size_t>(output_row_base + output_col_offset);
      if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
        output[input_row * output_rows_per_expert + output_col] =
            __float2bfloat16(accum_tensor(i) * output_alpha);
      } else {
        output[input_row * output_rows_per_expert + output_col] = accum_tensor(i) * output_alpha;
      }
    }
  }
}

template <typename OutputType, int kOutputTile>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kTracedP5TileN = cute::tile_size<1>(nvfp4_bridge::TracedP5TiledMma{});
  constexpr int kFp4ConsumerWarps = cute::size(nvfp4_bridge::TracedP5TiledMma{}) / 32;
  static_assert(kOutputTile == cute::tile_size<0>(nvfp4_bridge::TracedP5TiledMma{}));
  __shared__ std::uint8_t a_packed[kOutputTile][64 / 2];
  __shared__ std::uint8_t a_scale_smem[nvfp4_bridge::kTracedP5ScaleSmemCosizeA];
  __shared__ std::uint8_t b_packed[kTracedP5TileN][64 / 2];
  __shared__ std::uint8_t b_scale_smem[nvfp4_bridge::kTracedP5ScaleSmemCosizeB];
  const auto a_tile_view =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&a_packed[0][0], nullptr);
  const auto b_tile_view =
      nvfp4_bridge::MakePackedTile64<kTracedP5TileN>(&b_packed[0][0], nullptr);
  auto a_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFA{});
  auto b_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFB{});

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * (*weight.tensor_scale_data);

  nvfp4_bridge::CFragment64 accum[nvfp4_bridge::kTracedP5MFragments][nvfp4_bridge::kTracedP5NFragments];
  if (warp_id < kFp4ConsumerWarps) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP5MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP5NFragments; ++n_fragment) {
        nvfp4_bridge::Clear(accum[m_fragment][n_fragment]);
      }
    }
  }

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += 64u) {
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = k_base / 2u;

    for (int row = tid; row < kOutputTile; row += blockDim.x) {
      if (row < output_rows_this_tile) {
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = a_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)];
        }
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            a_scale_tensor,
            LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                source_row,
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4),
            row);
      } else {
        nvfp4_bridge::ZeroRow(a_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
      }
    }
    for (int row = tid; row < kTracedP5TileN; row += blockDim.x) {
      if (row < valid_rows) {
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = b_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = packed_input[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            b_scale_tensor,
            PackScaleWord4(
                input_block_scales[scale_offset + 0u],
                input_block_scales[scale_offset + 1u],
                input_block_scales[scale_offset + 2u],
                input_block_scales[scale_offset + 3u]),
            row);
      } else {
        nvfp4_bridge::ZeroRow(b_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      nvfp4_bridge::AFragment64 a_fragments[nvfp4_bridge::kTracedP5MFragments];
      nvfp4_bridge::BFragment64 b_fragments[nvfp4_bridge::kTracedP5NFragments];
      nvfp4_bridge::LoadTracedP5AFragmentsRowMajor16x64<
          nvfp4_bridge::TracedP5TiledMma,
          kOutputTile>(&a_packed[0][0], a_scale_smem, warp_id * 32 + lane_id, a_fragments);
      nvfp4_bridge::LoadTracedP5BFragmentsColMajor64x8<
          nvfp4_bridge::TracedP5TiledMma,
          kTracedP5TileN>(&b_packed[0][0], b_scale_smem, warp_id * 32 + lane_id, b_fragments);
#pragma unroll
      for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP5MFragments; ++m_fragment) {
#pragma unroll
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP5NFragments; ++n_fragment) {
          nvfp4_bridge::Gemm(accum[m_fragment][n_fragment], a_fragments[m_fragment], b_fragments[n_fragment]);
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    nvfp4_bridge::StoreTracedP5CFragmentsTranspose<nvfp4_bridge::TracedP5TiledMma>(
        output_alpha,
        accum,
        warp_id * 32 + lane_id,
        output_row_base,
        valid_rows,
        output_rows_this_tile,
        static_cast<std::size_t>(row_start),
        output_rows_per_expert,
        output);
  }
}

__device__ __forceinline__ float LoadContiguousInputTensorScale(
    const float* input_per_row_tensor_scales,
    const float* input_tensor_scale_data,
    std::size_t input_row) {
  if (input_per_row_tensor_scales != nullptr) {
    return input_per_row_tensor_scales[input_row];
  }
  if (input_tensor_scale_data != nullptr) {
    return *input_tensor_scale_data;
  }
  return 1.0f;
}

__device__ __forceinline__ float LoadGroupedInputTensorScale(
    const float* input_expert_tensor_scales,
    const float* input_per_row_tensor_scales,
    const float* input_tensor_scale_data,
    const float* input_dq_scales,
    int expert_index,
    std::size_t input_row) {
  if (input_dq_scales != nullptr) {
    return 1.0f;
  }
  if (input_expert_tensor_scales != nullptr) {
    return input_expert_tensor_scales[expert_index];
  }
  if (input_per_row_tensor_scales != nullptr) {
    return input_per_row_tensor_scales[input_row];
  }
  if (input_tensor_scale_data != nullptr) {
    return *input_tensor_scale_data;
  }
  return 1.0f;
}

__global__ void Nvfp4ContiguousSharedFp4P5Kernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_matmul_block_scales,
    Nvfp4ScaleLayout input_scale_layout,
    const float* input_tensor_scale_data,
    const float* input_per_row_tensor_scales,
    std::size_t input_row_count,
    const routed_p5_tma::P5TmaLoadB* p5_tma_load_b_descriptors,
    const routed_p5_tma::P5TmaLoadSFB* p5_tma_load_sfb_descriptors,
    FusedNvfp4WeightView weight,
    float* output) {
  using Traits = nvfp4_bridge::UnifiedRoutedFp4Traits<nvfp4_bridge::UnifiedRoutedFp4Profile::kP5>;
  using TiledMma = typename Traits::TiledMma;
  using CollectiveMainloop = typename Traits::CollectiveMainloop;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemCopyAtomA = typename Traits::SmemCopyAtomA;
  using SmemCopyAtomB = typename Traits::SmemCopyAtomB;
  using SmemCopyAtomSFA = typename Traits::SmemCopyAtomSFA;
  using SmemCopyAtomSFB = typename Traits::SmemCopyAtomSFB;
  constexpr int kOutputTile = 128;
  constexpr int kRowTile = 128;
  constexpr int kMacroTileK = 128;
  constexpr int kFp4ConsumerWarps = cute::size(TiledMma{}) / 32;

  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocA, Traits::kSwizzledAElems>
      smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocB, Traits::kSwizzledBElems>
      smem_swizzled_b_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeA>
      a_scale_smem_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeB>
      b_scale_smem_storage;
  __shared__ alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType p5_tma_full_mbar_storage;

  auto* smem_swizzled_a = smem_swizzled_a_storage.data();
  auto* smem_swizzled_b = smem_swizzled_b_storage.data();
  auto* a_scale_smem = a_scale_smem_storage.data();
  auto* b_scale_smem = b_scale_smem_storage.data();
  auto* p5_tma_full_mbar =
      cute::recast_ptr<cutlass::arch::ClusterTransactionBarrier>(&p5_tma_full_mbar_storage);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int cta_index = static_cast<int>(blockIdx.y);
  const int row_start = cta_index * kRowTile;
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (packed_input == nullptr ||
      input_matmul_block_scales == nullptr ||
      p5_tma_load_b_descriptors == nullptr ||
      p5_tma_load_sfb_descriptors == nullptr ||
      weight.p5_tma_load_a == nullptr ||
      weight.p5_tma_load_sfa == nullptr ||
      output == nullptr ||
      input_scale_layout != Nvfp4ScaleLayout::kSwizzled128x4 ||
      static_cast<std::size_t>(row_start) >= input_row_count ||
      static_cast<std::size_t>(output_row_base) >= weight.output_rows) {
    return;
  }

  const auto* p5_tma_load_a_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadA*>(weight.p5_tma_load_a);
  const auto* p5_tma_load_sfa_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadSFA*>(weight.p5_tma_load_sfa);
  const int valid_rows = static_cast<int>(
      min(input_row_count - static_cast<std::size_t>(row_start), static_cast<std::size_t>(kRowTile)));
  const int output_rows_this_tile = static_cast<int>(
      min(weight.output_rows - static_cast<std::size_t>(output_row_base), static_cast<std::size_t>(kOutputTile)));
  if (valid_rows <= 0 || output_rows_this_tile <= 0) {
    return;
  }

  const bool is_fp4_consumer_thread = warp_id < kFp4ConsumerWarps;
  const int lane_predicate = cute::elect_one_sync();
  const bool is_p5_tma_thread =
      warp_id == kFp4ConsumerWarps &&
      lane_predicate != 0;

  if (is_p5_tma_thread) {
    cute::prefetch_tma_descriptor(p5_tma_load_a_ptr->get_tma_descriptor());
    cute::prefetch_tma_descriptor(p5_tma_load_sfa_ptr->get_tma_descriptor());
    cute::prefetch_tma_descriptor(p5_tma_load_b_descriptors[cta_index].get_tma_descriptor());
    cute::prefetch_tma_descriptor(p5_tma_load_sfb_descriptors[cta_index].get_tma_descriptor());
  }

  nvfp4_bridge::CRegister accum_storage[Traits::kAccumProfileCosize];
  if (is_fp4_consumer_thread) {
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        typename Traits::AccumLayout{});
    cute::clear(accum_tensor);
  }

  const float weight_tensor_scale =
      weight.tensor_scale_data != nullptr ? *weight.tensor_scale_data : 1.0f;
  const int macro_k_tile_count = static_cast<int>(weight.input_cols / static_cast<std::size_t>(kMacroTileK));
  for (int k_tile_coord = 0; k_tile_coord < macro_k_tile_count; ++k_tile_coord) {
    if (is_p5_tma_thread) {
      p5_tma_full_mbar[0].init(1);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (is_p5_tma_thread) {
      using X = cute::Underscore;
      using ProducerBarrierType = typename cutlass::arch::ClusterTransactionBarrier::ValueType;
      auto& barrier = p5_tma_full_mbar[0];
      std::uint32_t stage_bytes = 0u;
      const int output_tile_coord = output_row_base / kOutputTile;

      const auto& tma_load_a = *p5_tma_load_a_ptr;
      auto mA_mkl = tma_load_a.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(weight.output_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gA_mkl = cute::local_tile(
          mA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_a = tma_load_a.get_slice(0);
      auto gA = gA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgA = block_tma_a.partition_S(gA);
      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto tAsA = block_tma_a.partition_D(sA);
      auto tma_copy_a =
          tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_a,
          tAgA(cute::_, cute::_, cute::_, k_tile_coord),
          tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::size(cute::take<0, 2>(SmemLayoutA{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementAB>));

      const auto& tma_load_sfa = *p5_tma_load_sfa_ptr;
      auto mSFA_mkl = tma_load_sfa.get_tma_tensor(cute::shape(
          routed_p5_tma::MakeP5ScaleLayoutSFA(
              static_cast<int32_t>(weight.output_rows),
              static_cast<int32_t>(weight.input_cols))));
      auto gSFA_mkl = cute::local_tile(
          mSFA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_sfa = tma_load_sfa.get_slice(0);
      auto gSFA = gSFA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgSFA = block_tma_sfa.partition_S(gSFA);
      auto sSFA_ = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
      auto sSFA = cute::as_position_independent_swizzle_tensor(sSFA_);
      auto tAsSFA = block_tma_sfa.partition_D(sSFA);
      auto tma_copy_sfa =
          tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_sfa,
          tAgSFA(cute::_, cute::_, cute::_, k_tile_coord),
          tAsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::cosize(cute::take<0, 2>(SmemLayoutSFA{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementSF>));

      const auto& tma_load_b = p5_tma_load_b_descriptors[cta_index];
      auto mB_nkl = tma_load_b.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(valid_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gB_nkl = cute::local_tile(
          mB_nkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<X, cute::_1, cute::_1>{});
      auto block_tma_b = tma_load_b.get_slice(0);
      auto gB = gB_nkl(cute::_, cute::_, 0, cute::_, 0);
      auto tBgB = block_tma_b.partition_S(gB);
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto tBsB = block_tma_b.partition_D(sB);
      auto tma_copy_b =
          tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_b,
          tBgB(cute::_, cute::_, cute::_, k_tile_coord),
          tBsB(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::size(cute::take<0, 2>(SmemLayoutB{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementAB>));

      const auto& tma_load_sfb = p5_tma_load_sfb_descriptors[cta_index];
      auto mSFB_nkl = tma_load_sfb.get_tma_tensor(cute::shape(
          routed_p5_tma::MakeP5ScaleLayoutSFB(
              static_cast<int32_t>(kRowTile),
              static_cast<int32_t>(weight.input_cols))));
      auto gSFB_nkl = cute::local_tile(
          mSFB_nkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<X, cute::_1, cute::_1>{});
      auto block_tma_sfb = tma_load_sfb.get_slice(0);
      auto gSFB = gSFB_nkl(cute::_, cute::_, 0, cute::_, 0);
      auto tBgSFB = block_tma_sfb.partition_S(gSFB);
      auto sSFB_ = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
      auto sSFB = cute::as_position_independent_swizzle_tensor(sSFB_);
      auto tBsSFB = block_tma_sfb.partition_D(sSFB);
      auto tma_copy_sfb =
          tma_load_sfb.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_sfb,
          tBgSFB(cute::_, cute::_, cute::_, k_tile_coord),
          tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementSF>));

      p5_tma_full_mbar[0].arrive_and_expect_tx(stage_bytes);
    }

    if (is_fp4_consumer_thread) {
      p5_tma_full_mbar[0].wait(0);

      auto tiled_mma = TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
      auto sSFA = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

      auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
          sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA = s2r_thr_A.partition_S(sA);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB = s2r_thr_B.partition_S(sB);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          SmemCopyAtomSFA{},
          Traits::GetLayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          SmemCopyAtomSFB{},
          Traits::GetLayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
      auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          typename Traits::AccumLayout{});

      cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
      cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
      cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
      cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

      using MMAOp = typename TiledMma::MMA_Op;
      for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
        cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
        cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
      }

      using MMAAtom = typename TiledMma::Atom;
      MMAAtom mma_atom;
      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      for (int k = 0; k < K_blocks; ++k) {
        for (int n = 0; n < N_tiles; ++n) {
          for (int m = 0; m < M_tiles; ++m) {
            auto a_atom = tCrA(cute::_, m, k);
            auto b_atom = tCrB(cute::_, n, k);
            auto c_atom = accum_tensor(cute::_, m, n);
            auto sfa_atom = tCrSFA(cute::_, m, k);
            auto sfb_atom = tCrSFB(cute::_, n, k);
            mma_atom.call(
                c_atom,
                cute::make_zip_tensor(a_atom, sfa_atom),
                cute::make_zip_tensor(b_atom, sfb_atom),
                c_atom);
          }
        }
      }
    }
    __syncthreads();
  }

  if (is_fp4_consumer_thread) {
    auto tiled_mma = TiledMma{};
    auto thread_mma = tiled_mma.get_thread_slice(warp_id * 32 + lane_id);
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        typename Traits::AccumLayout{});
    auto dense_c = cute::make_identity_tensor(
        cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kRowTile>{}));
    auto part_c = thread_mma.partition_C(dense_c);
    for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
      auto coord = part_c(i);
      const int output_col_offset = static_cast<int>(cute::get<0>(coord));
      const int token_row = static_cast<int>(cute::get<1>(coord));
      if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
        continue;
      }
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_row);
      const float output_alpha =
          weight_tensor_scale *
          LoadContiguousInputTensorScale(
              input_per_row_tensor_scales,
              input_tensor_scale_data,
              input_row);
      output[input_row * weight.output_rows +
             static_cast<std::size_t>(output_row_base + output_col_offset)] =
          accum_tensor(i) * output_alpha;
    }
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_m_limits,
    const int* expert_first_token_offsets,
    int token_tile_dim,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr)) {
    return;
  }
  (void) input_per_row_tensor_scales;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_batch_indices[cta_index];
  const int batch_row_begin = expert_first_token_offsets[expert_index];
  const int batch_cta_begin = batch_row_begin / token_tile_dim;
  const int row_start =
      batch_row_begin + (cta_index - batch_cta_begin) * token_tile_dim;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows = max(0, min(m_limit - row_start, token_tile_dim));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            input_dq_scales != nullptr
                ? input_dq_scales[scale_row_offset + block]
                : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                      input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRowsBf16Kernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_m_limits,
    const int* expert_first_token_offsets,
    int token_tile_dim,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }
  (void) input_per_row_tensor_scales;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_batch_indices[cta_index];
  const int batch_row_begin = expert_first_token_offsets[expert_index];
  const int batch_cta_begin = batch_row_begin / token_tile_dim;
  const int row_start =
      batch_row_begin + (cta_index - batch_cta_begin) * token_tile_dim;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows = max(0, min(m_limit - row_start, token_tile_dim));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }

    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            input_dq_scales != nullptr
                ? input_dq_scales[scale_row_offset + block]
                : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                      input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        __float2bfloat16(c_tile[tile_output_row][tile_token]);
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2MaxAbsKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_m_limits,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    unsigned int* expert_max_bits) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];
  __shared__ float shared_max[kPlannedThreadsPerBlock];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      input_tensor_scale_data == nullptr ||
      expert_max_bits == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_index * kGroupedTokenTile;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows =
      max(0, min(m_limit - row_start, kGroupedTokenTile));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                            : *input_tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
            input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  float thread_max = 0.0f;
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const float value = fused_decode::Relu2(c_tile[tile_output_row][tile_token]);
    thread_max = fmaxf(thread_max, value);
  }

  shared_max[tid] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
      shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(&expert_max_bits[expert_index], __float_as_uint(shared_max[0]));
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2PackKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    std::size_t output_row_capacity,
    Nvfp4ScaleLayout output_scale_layout,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales,
    float* output_activation_scales) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr ||
      output_activation_scales == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_input_row = weight.input_cols / 2;
  const std::size_t input_blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float expert_input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_input_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * input_blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_input_row;
        const std::size_t scale_row_offset = input_row * input_blocks_per_row;
        const float block_scale =
            input_dq_scales != nullptr
                ? input_dq_scales[scale_row_offset + block]
                : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                      (input_per_row_tensor_scales != nullptr
                           ? input_per_row_tensor_scales[input_row]
                           : expert_input_tensor_scale);
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  if (lane_id >= valid_rows || warp_row >= output_rows_this_tile) {
    return;
  }

  const std::size_t output_row = static_cast<std::size_t>(row_start + lane_id);
  if (output_row >= output_row_capacity) {
    return;
  }
  const std::size_t blocks_per_output_row =
      output_rows_per_expert / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_output_row, kNvfp4ScaleBlockTile);
  const std::size_t block_col =
      static_cast<std::size_t>(output_row_base + warp_row) / fused_decode::kNvfp4BlockWidth;
  const float output_tensor_scale = 1.0f;

  float activated[16];
  float block_max_abs = 0.0f;
  for (int col = 0; col < 16; ++col) {
    const float value = fused_decode::Relu2(c_tile[warp_row + col][lane_id]);
    activated[col] = value;
    if (value > block_max_abs) {
      block_max_abs = value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * output_tensor_scale));
  }
  const std::uint8_t encoded_block_scale = fused_decode::EncodeFp8Scale(block_scale);
  const std::size_t block_scale_offset = output_row * blocks_per_output_row + block_col;
  output_activation_scales[block_scale_offset] = output_tensor_scale * block_scale;
  output_block_scales[block_scale_offset] = encoded_block_scale;
  output_matmul_scales[ExecutionScaleOffset(
      output_row, block_col, padded_blocks_per_row, output_scale_layout)] = encoded_block_scale;

  const float pack_scale = output_tensor_scale * block_scale;
  const std::size_t packed_row_offset =
      output_row * (output_rows_per_expert / 2u) +
      block_col * (fused_decode::kNvfp4BlockWidth / 2u);
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t lhs = fused_decode::EncodeFp4(activated[pair * 2] / pack_scale);
    const std::uint8_t rhs = fused_decode::EncodeFp4(activated[pair * 2 + 1] / pack_scale);
    output_packed[packed_row_offset + static_cast<std::size_t>(pair)] =
        static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
  }
}

template <bool kGroupedRows>
__global__ void Nvfp4MatVecRowsKernel(
    const float* input,
    std::size_t input_row_count,
    const int* expert_offsets,
    int expert_index,
    fused_decode::Nvfp4WeightView weight,
    float* output) {
  __shared__ float partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

  const std::size_t output_row = static_cast<std::size_t>(blockIdx.x);
  if (output_row >= weight.output_rows) {
    return;
  }

  int begin_row = 0;
  int end_row = static_cast<int>(input_row_count);
  if constexpr (kGroupedRows) {
    begin_row = expert_offsets[expert_index];
    end_row = expert_offsets[expert_index + 1];
  }
  if (begin_row >= end_row) {
    return;
  }

  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * (weight.input_cols / 2);
  const std::size_t scale_row_offset = output_row * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;

  for (int tile_begin = begin_row; tile_begin < end_row; tile_begin += kGroupedTokenTile) {
    const int remaining_rows = end_row - tile_begin;
    const int valid_rows =
        remaining_rows < kGroupedTokenTile ? remaining_rows : kGroupedTokenTile;
    float accum[kGroupedTokenTile] = {0.0f};

    for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
         pair_index < pairs_per_row;
         pair_index += blockDim.x) {
      const std::size_t block = pair_index / 8u;
      const std::size_t pair_in_block = pair_index % 8u;
      const std::size_t col = block * fused_decode::kNvfp4BlockWidth + pair_in_block * 2u;
      const float block_scale =
          LoadNvfp4WeightBlockScale(weight, output_row, block);
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale;
      const float w1 = fused_decode::DecodeFp4((packed >> 4) & 0x0Fu) * block_scale;
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        const float* input_row =
            input + static_cast<std::size_t>(tile_begin + tile_row) * weight.input_cols;
        accum[tile_row] += input_row[col] * w0;
        accum[tile_row] += input_row[col + 1] * w1;
      }
    }

    for (int tile_row = 0; tile_row < kGroupedTokenTile; ++tile_row) {
      partial_sums[tile_row * blockDim.x + threadIdx.x] =
          tile_row < valid_rows ? accum[tile_row] : 0.0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
          partial_sums[tile_row * blockDim.x + threadIdx.x] +=
              partial_sums[tile_row * blockDim.x + threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        output[static_cast<std::size_t>(tile_begin + tile_row) * weight.output_rows + output_row] =
            partial_sums[tile_row * blockDim.x];
      }
    }
    __syncthreads();
  }
}

template <int kOutputTile, int kThreadsPerBlock>
__global__ void Nvfp4ContiguousWmmaMatVecRowsKernel(
    const float* input,
    std::size_t input_row_count,
    FusedNvfp4WeightView weight,
    float* output) {
  constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

  __shared__ __nv_bfloat16 a_tile[kOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kOutputTile][kPlannedWmmaTileM];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  const int row_start = static_cast<int>(blockIdx.y) * kPlannedWmmaTileM;
  if (warp_id >= kWarpsPerBlock ||
      static_cast<std::size_t>(output_row_base) >= weight.output_rows ||
      static_cast<std::size_t>(row_start) >= input_row_count) {
    return;
  }

  const std::size_t remaining_input_rows =
      input_row_count - static_cast<std::size_t>(row_start);
  const int valid_rows = static_cast<int>(
      remaining_input_rows < static_cast<std::size_t>(kPlannedWmmaTileM)
          ? remaining_input_rows
          : static_cast<std::size_t>(kPlannedWmmaTileM));
  const std::size_t remaining_output_rows =
      weight.output_rows - static_cast<std::size_t>(output_row_base);
  const int output_rows_this_tile = static_cast<int>(
      remaining_output_rows < static_cast<std::size_t>(kOutputTile)
          ? remaining_output_rows
          : static_cast<std::size_t>(kOutputTile));
    const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * weight.output_rows + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void ReduceSelectionOutputsKernel(
    const float* grouped_output,
    const int* selection_to_sorted,
    const int* sorted_to_permuted,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    std::size_t hidden_size,
    float* output,
    float* routed_output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = token_count * hidden_size;
  if (index >= count) {
    return;
  }

  const std::size_t token_index = index / hidden_size;
  const std::size_t hidden_index = index % hidden_size;
  double accum = 0.0;
  for (std::size_t slot = 0; slot < top_k; ++slot) {
    const std::size_t selection_index = token_index * top_k + slot;
    const int sorted_index = selection_to_sorted[selection_index];
    if (sorted_index < 0) {
      continue;
    }
    const int output_row =
        sorted_to_permuted != nullptr ? sorted_to_permuted[sorted_index] : sorted_index;
    if (output_row < 0) {
      continue;
    }
    const std::size_t grouped_offset =
        static_cast<std::size_t>(output_row) * hidden_size + hidden_index;
    accum += static_cast<double>(selected_weights[selection_index]) *
             static_cast<double>(grouped_output[grouped_offset]);
  }
  const float reduced = static_cast<float>(accum);
  output[index] = reduced;
  if (routed_output != nullptr) {
    routed_output[index] = reduced;
  }
}

__global__ void AccumulateSharedOutputKernel(
    const float* shared_output,
    std::size_t count,
    float* output,
    float* shared_output_copy) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float value = shared_output[index];
  output[index] += value;
  if (shared_output_copy != nullptr) {
    shared_output_copy[index] = value;
  }
}

bool LaunchZeroBuffer(float* data, std::size_t count) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ZeroBufferKernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchZeroBf16Buffer(__nv_bfloat16* data, std::size_t count) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ZeroBf16BufferKernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchFillFloatBuffer(float* data, std::size_t count, float value) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  FillFloatBufferKernel<<<grid, block>>>(data, count, value);
  return CheckCuda(cudaGetLastError());
}

bool ClearDeviceNvfp4Matrix(
    DeviceNvfp4Matrix* matrix,
    float tensor_scale = 1.0f,
    float per_row_tensor_scale = 0.0f) {
  if (matrix == nullptr || !matrix->valid()) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(matrix->packed_data()), 0, matrix->packed_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(matrix->block_scales_data()),
          0,
          matrix->block_scales_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(matrix->matmul_block_scales_data()),
          0,
          matrix->matmul_block_scales_nbytes()))) {
    return false;
  }

  float* per_row_scales = const_cast<float*>(matrix->per_row_tensor_scales());
  if (per_row_scales == nullptr) {
    return false;
  }
  if (per_row_tensor_scale == 0.0f) {
    if (!CheckCuda(cudaMemset(
            per_row_scales,
            0,
            matrix->rows() * sizeof(float)))) {
      return false;
    }
  } else if (!LaunchFillFloatBuffer(
                 per_row_scales,
                 matrix->rows(),
                 per_row_tensor_scale)) {
    return false;
  }

  return CheckCuda(cudaMemcpy(
      const_cast<std::uint8_t*>(matrix->tensor_scale_data()),
      &tensor_scale,
      sizeof(tensor_scale),
      cudaMemcpyHostToDevice));
}

bool LaunchGatherRows(
    const float* input,
    const int* row_indices,
    const int* active_output_rows,
    std::size_t output_row_capacity,
    std::size_t input_rows,
    std::size_t cols,
    float* output) {
  const std::size_t count = output_row_capacity * cols;
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  GatherRowsKernel<<<grid, block>>>(
      input,
      row_indices,
      active_output_rows,
      output,
      output_row_capacity,
      input_rows,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchComputeExpertActivationScales(
    const float* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      n_experts == 0 ||
      cols == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts));
  ComputeExpertActivationScalesKernel<<<grid, block>>>(
      input,
      expert_first_token_offsets,
      n_experts,
      cols,
      output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchComputeExpertActivationScalesBf16(
    const __nv_bfloat16* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      n_experts == 0 ||
      cols == 0 ||
      output_scales == nullptr) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts));
  ComputeExpertActivationScalesBf16Kernel<<<grid, block>>>(
      input,
      expert_first_token_offsets,
      n_experts,
      cols,
      output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRoutedBf16Relu2Pack(
    const __nv_bfloat16* source,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    DeviceNvfp4Matrix* output_pack,
    float* output_dequant_scales) {
  if (source == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      output_dequant_scales == nullptr) {
    return false;
  }

  const auto current_padded_row_capacity =
      DeviceMoeLaunchPlan::PaddedRowCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_padded_row_capacity.has_value() ||
      *current_padded_row_capacity == 0 ||
      *current_padded_row_capacity > output_pack->rows()) {
    return false;
  }

  const std::size_t blocks_per_row =
      output_pack->cols() / fused_decode::kNvfp4BlockWidth;
  const std::size_t scale_count = (*current_padded_row_capacity) * blocks_per_row;
  if (blocks_per_row == 0 ||
      !ClearDeviceNvfp4Matrix(output_pack) ||
      !LaunchZeroBuffer(output_dequant_scales, scale_count)) {
    return false;
  }

  const auto launch_rows = [&](auto rows_per_cta_tag) -> bool {
    constexpr int kProcessRows = decltype(rows_per_cta_tag)::value;
    const dim3 block(fused_decode::kThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(
            (*current_padded_row_capacity + (kProcessRows * block.x) - 1u) /
            (kProcessRows * block.x)),
        static_cast<unsigned int>(blocks_per_row));
    return LaunchProgrammaticKernel(
        grid,
        block,
        RoutedBf16Relu2PackKernel<kProcessRows>,
        source,
        routing->expert_offsets(),
        launch_plan->expert_first_token_offsets(),
        static_cast<int>(launch_plan->n_experts()),
        output_pack->cols(),
        RoundUp(blocks_per_row, kNvfp4ScaleBlockTile),
        output_pack->scale_layout(),
        output_dequant_scales,
        const_cast<std::uint8_t*>(output_pack->packed_data()),
        const_cast<std::uint8_t*>(output_pack->block_scales_data()),
        const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()));
  };

  const std::size_t workload = (*current_padded_row_capacity) * blocks_per_row;
  if (workload < (256u * 256u)) {
    return launch_rows(std::integral_constant<int, 1>{});
  }
  if (workload < (256u * 512u)) {
    return launch_rows(std::integral_constant<int, 2>{});
  }
  return launch_rows(std::integral_constant<int, 4>{});
}

bool LaunchMaxBitsToTensorScales(
    const unsigned int* max_bits,
    std::size_t count,
    float* output_scales) {
  if (max_bits == nullptr || output_scales == nullptr || count == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  MaxBitsToTensorScalesKernel<<<grid, block>>>(max_bits, count, output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchQuantizeDequantizeRows(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(row_capacity));
  QuantizeDequantizeRowsKernel<<<grid, block>>>(
      data,
      active_row_count,
      row_capacity,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRelu2(float* data, std::size_t count) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  Relu2Kernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRelu2Rows(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(((row_capacity * cols) + block.x - 1u) / block.x));
  Relu2RowsKernel<<<grid, block>>>(
      data,
      active_row_count,
      row_capacity,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchGroupedMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (weights == nullptr || n_experts == 0 || output_rows_per_expert == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts * output_rows_per_expert));
  Nvfp4GroupedExpertMatVecRowsKernel<<<grid, block>>>(
      input,
      expert_offsets,
      n_experts,
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchContiguousFp4MatVec(
    const DeviceNvfp4Matrix& input_pack,
    const SharedContiguousPreparedLaunch& prepared_launch,
    const FusedNvfp4WeightView& weight,
    float* output) {
  if (!input_pack.valid() ||
      !ValidFusedNvfp4WeightView(weight) ||
      output == nullptr) {
    return false;
  }

  const auto& launch_plan = prepared_launch.launch_plan;
  const auto& execution = prepared_launch.execution;
  if (execution.backend_kind != GemmBackendKind::kSm120ContiguousSharedNvfp4 ||
      launch_plan.kernel_family != GemmKernelFamily::kSm120ContiguousSharedNvfp4 ||
      launch_plan.m == 0 ||
      launch_plan.n != weight.output_rows ||
      launch_plan.k != weight.input_cols ||
      launch_plan.m > input_pack.rows() ||
      input_pack.cols() != weight.input_cols ||
      input_pack.scale_layout() != Nvfp4ScaleLayout::kSwizzled128x4 ||
      launch_plan.tile_m != 128u ||
      launch_plan.tile_n != 128u ||
      launch_plan.cta_m_count == 0 ||
      launch_plan.cta_n_count == 0) {
    return false;
  }

  if ((weight.input_cols % 128u) != 0 ||
      weight.p5_tma_load_a == nullptr ||
      weight.p5_tma_load_sfa == nullptr ||
      prepared_launch.p5_tma_load_b_descriptors == nullptr ||
      prepared_launch.p5_tma_load_sfb_descriptors == nullptr) {
    return false;
  }

  const dim3 block(kSharedContiguousP5ThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(launch_plan.cta_n_count),
      static_cast<unsigned int>(launch_plan.cta_m_count));
  const auto* p5_tma_load_b_descriptors =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadB*>(
          prepared_launch.p5_tma_load_b_descriptors);
  const auto* p5_tma_load_sfb_descriptors =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadSFB*>(
          prepared_launch.p5_tma_load_sfb_descriptors);
  if (launch_plan.uses_programmatic_launch) {
    return LaunchProgrammaticKernel(
        grid,
        block,
        Nvfp4ContiguousSharedFp4P5Kernel,
        input_pack.packed_data(),
        input_pack.matmul_block_scales_data(),
        input_pack.scale_layout(),
        input_pack.device_tensor_scale_ptr(),
        input_pack.per_row_tensor_scales(),
        launch_plan.m,
        p5_tma_load_b_descriptors,
        p5_tma_load_sfb_descriptors,
        weight,
        output);
  }

  Nvfp4ContiguousSharedFp4P5Kernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.matmul_block_scales_data(),
      input_pack.scale_layout(),
      input_pack.device_tensor_scale_ptr(),
      input_pack.per_row_tensor_scales(),
      launch_plan.m,
      p5_tma_load_b_descriptors,
      p5_tma_load_sfb_descriptors,
      weight,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchContiguousMatVec(
    const float* input,
    std::size_t input_row_count,
    const FusedNvfp4WeightView& weight,
    float* output) {
  if (!ValidFusedNvfp4WeightView(weight)) {
    return false;
  }
  if (input_row_count == 0) {
    return true;
  }
  const int output_tile = SelectContiguousOutputTile(weight.output_rows, input_row_count);
  const std::size_t input_row_tile_count =
      CeilDiv(input_row_count, static_cast<std::size_t>(kPlannedWmmaTileM));

  if (output_tile == kContiguousLargeOutputTile) {
    const dim3 block(kContiguousLargeThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(CeilDiv(
            weight.output_rows,
            static_cast<std::size_t>(kContiguousLargeOutputTile))),
        static_cast<unsigned int>(input_row_tile_count));
    Nvfp4ContiguousWmmaMatVecRowsKernel<
        kContiguousLargeOutputTile,
        kContiguousLargeThreadsPerBlock><<<grid, block>>>(
        input,
        input_row_count,
        weight,
        output);
    return CheckCuda(cudaGetLastError());
  }

  if (output_tile == kContiguousMediumOutputTile) {
    const dim3 block(kContiguousMediumThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(CeilDiv(
            weight.output_rows,
            static_cast<std::size_t>(kContiguousMediumOutputTile))),
        static_cast<unsigned int>(input_row_tile_count));
    Nvfp4ContiguousWmmaMatVecRowsKernel<
        kContiguousMediumOutputTile,
        kContiguousMediumThreadsPerBlock><<<grid, block>>>(
        input,
        input_row_count,
        weight,
        output);
    return CheckCuda(cudaGetLastError());
  }

  const dim3 block(kContiguousSmallThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(CeilDiv(
          weight.output_rows,
          static_cast<std::size_t>(kContiguousSmallOutputTile))),
      static_cast<unsigned int>(input_row_tile_count));
  Nvfp4ContiguousWmmaMatVecRowsKernel<
      kContiguousSmallOutputTile,
      kContiguousSmallThreadsPerBlock><<<grid, block>>>(
      input,
      input_row_count,
      weight,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRowsKernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVecBf16(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRowsBf16Kernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      launch_plan->expert_first_token_offsets(),
      static_cast<int>(launch_plan->selected_token_tile()),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVecRelu2ExpertScales(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output_expert_scales) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          reinterpret_cast<void*>(output_expert_scales),
          0,
          launch_plan->n_experts() * sizeof(float)))) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRelu2MaxAbsKernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      reinterpret_cast<unsigned int*>(output_expert_scales));
  return CheckCuda(cudaGetLastError()) &&
         LaunchMaxBitsToTensorScales(
             reinterpret_cast<const unsigned int*>(output_expert_scales),
             launch_plan->n_experts(),
             output_expert_scales);
}

bool LaunchPlannedMatVecRelu2Pack(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const float* output_expert_scales,
    DeviceNvfp4Matrix* output_pack) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      output_pack->rows() < launch_plan->padded_row_capacity() ||
      output_pack->cols() != output_rows_per_expert) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0 || !ClearDeviceNvfp4Matrix(output_pack)) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRelu2PackKernel<<<grid, block>>>(
      input,
      output_expert_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output_pack->rows(),
      output_pack->scale_layout(),
      const_cast<std::uint8_t*>(output_pack->packed_data()),
      const_cast<std::uint8_t*>(output_pack->block_scales_data()),
      const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedPackedInputMatVec(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const auto launch_profile = [&](RoutedGemm2Profile profile) -> bool {
    if (RoutedProfileDebugEnabled()) {
      std::fprintf(
          stderr,
          "routed_gemm2 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
          dispatch_rows,
          active_selection_count,
          RoutedGemm2ProfileName(profile));
    }
    const int output_tile = RoutedOutputTile(profile);
    const std::size_t output_row_tile_count =
        (output_rows_per_expert + static_cast<std::size_t>(output_tile) - 1u) /
        static_cast<std::size_t>(output_tile);
    if (output_row_tile_count == 0) {
      return false;
    }
    const dim3 block(kRoutedThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(output_row_tile_count),
        static_cast<unsigned int>(*current_cta_capacity));
    switch (profile) {
      case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
            nvfp4_bridge::UnifiedRoutedFp4Profile::kP12,
            float><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.matmul_block_scales_data(),
            input_pack.scale_layout(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            nullptr,
            nullptr,
            weights,
            output_rows_per_expert,
            output,
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<std::size_t>(0),
            static_cast<std::size_t>(0),
            Nvfp4ScaleLayout::kSwizzled128x4);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
        g_enable_p13_scale_trace = std::getenv("NEMOTRON_P13_SCALE_DEBUG") != nullptr ? 1 : 0;
        g_enable_p13_scale_map_probe =
            std::getenv("NEMOTRON_P13_SCALE_MAP_PROBE") != nullptr ? 1 : 0;
        g_enable_p13_debug_trace = std::getenv("NEMOTRON_P13_DEBUG_TRACE") != nullptr ? 1 : 0;
        g_p13_debug_trace_target_valid_rows = -1;
        g_p13_debug_trace_target_thread_id = -1;
        if (const char* target_valid_rows_env =
                std::getenv("NEMOTRON_P13_DEBUG_TRACE_TARGET_VALID_ROWS")) {
          g_p13_debug_trace_target_valid_rows = std::atoi(target_valid_rows_env);
        }
        if (const char* target_thread_env =
                std::getenv("NEMOTRON_P13_DEBUG_TRACE_TARGET_THREAD_ID")) {
          g_p13_debug_trace_target_thread_id = std::atoi(target_thread_env);
        }
        if (g_enable_p13_scale_trace != 0 ||
            g_enable_p13_scale_map_probe != 0 ||
            g_enable_p13_debug_trace != 0) {
          g_p13_scale_trace = {};
          g_p13_debug_trace = {};
          g_p13_debug_trace_claimed = 0;
        }
        if (RoutedEnvEnabled("NEMOTRON_DEBUG_USE_OLD_P13_KERNEL")) {
          Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmem<
              float,
              128><<<grid, block>>>(
              input_pack.packed_data(),
              input_pack.block_scales_data(),
              input_pack.matmul_block_scales_data(),
              input_pack.scale_layout(),
              input_pack.device_tensor_scale_ptr(),
              input_expert_tensor_scales,
              input_dq_scales,
              launch_plan->num_non_exiting_ctas(),
              launch_plan->cta_idx_xy_to_batch_idx(),
              launch_plan->cta_row_starts(),
              launch_plan->cta_valid_rows(),
              weights,
              output_rows_per_expert,
              output);
        } else {
          Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
              nvfp4_bridge::UnifiedRoutedFp4Profile::kP13,
              float><<<grid, block>>>(
              input_pack.packed_data(),
              input_pack.matmul_block_scales_data(),
              input_pack.scale_layout(),
              input_pack.device_tensor_scale_ptr(),
              input_expert_tensor_scales,
              input_dq_scales,
              input_per_row_tensor_scales,
              launch_plan->num_non_exiting_ctas(),
              launch_plan->cta_idx_xy_to_batch_idx(),
              launch_plan->cta_row_starts(),
              launch_plan->cta_valid_rows(),
              nullptr,
              nullptr,
              weights,
              output_rows_per_expert,
              output,
              static_cast<std::uint8_t*>(nullptr),
              static_cast<std::uint8_t*>(nullptr),
              static_cast<std::uint8_t*>(nullptr),
              static_cast<float*>(nullptr),
              static_cast<std::size_t>(0),
              static_cast<std::size_t>(0),
              Nvfp4ScaleLayout::kSwizzled128x4);
        }
        if (g_enable_p13_scale_trace != 0) {
          if (!CheckCuda(cudaDeviceSynchronize())) {
            return false;
          }
          PrintP13ScaleTrace();
        }
        return CheckCuda(cudaGetLastError());
      case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
        g_enable_p15_scale_trace = std::getenv("NEMOTRON_P15_SCALE_DEBUG") != nullptr ? 1 : 0;
        if (g_enable_p15_scale_trace != 0) {
          g_p15_scale_trace = {};
        }
        Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
            nvfp4_bridge::UnifiedRoutedFp4Profile::kP15,
            float><<<grid, dim3(256)>>>(
            input_pack.packed_data(),
            input_pack.matmul_block_scales_data(),
            input_pack.scale_layout(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            nullptr,
            nullptr,
            weights,
            output_rows_per_expert,
            output,
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<std::size_t>(0),
            static_cast<std::size_t>(0),
            Nvfp4ScaleLayout::kSwizzled128x4);
        if (!CheckCuda(cudaGetLastError())) {
          return false;
        }
        if (g_enable_p15_scale_trace != 0) {
          if (!CheckCuda(cudaDeviceSynchronize())) {
            return false;
          }
          PrintP15ScaleTrace();
        }
        return true;
      case RoutedGemm2Profile::kLegacy:
        break;
    }
    return false;
  };

  const RoutedGemm2Profile profile = SelectRoutedGemm2Profile(dispatch_rows);
  if (profile != RoutedGemm2Profile::kLegacy && launch_profile(profile)) {
    return true;
  }
  if (RoutedProfileDebugEnabled()) {
    std::fprintf(
        stderr,
        "routed_gemm2 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
        dispatch_rows,
        active_selection_count,
        RoutedGemm2ProfileName(RoutedGemm2Profile::kLegacy));
  }

  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      input_dq_scales,
      input_per_row_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      launch_plan->expert_first_token_offsets(),
      static_cast<int>(launch_plan->selected_token_tile()),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedPackedInputMatVecBf16(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const auto launch_profile = [&](RoutedGemm1Profile profile) -> bool {
    if (RoutedProfileDebugEnabled()) {
      std::fprintf(
          stderr,
          "routed_gemm1 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
          dispatch_rows,
          active_selection_count,
          RoutedGemm1ProfileName(profile));
    }
    const int output_tile = RoutedOutputTile(profile);
    const std::size_t output_row_tile_count =
        (output_rows_per_expert + static_cast<std::size_t>(output_tile) - 1u) /
        static_cast<std::size_t>(output_tile);
    if (output_row_tile_count == 0) {
      return false;
    }
    const dim3 block(kRoutedThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(output_row_tile_count),
        static_cast<unsigned int>(*current_cta_capacity));
    switch (profile) {
      case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalse<__nv_bfloat16, 128, 128><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            weights,
            output_rows_per_expert,
            output);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
        // kP1 has no BF16-output kernel during the rebuild. Fall through to
        // `default:` → returns false. Non-env-gated kP1 rows will fail
        // dispatch until step 6 wires the new NanoP1 kernel.
        break;
      case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue<__nv_bfloat16, 128, 128><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            weights,
            output_rows_per_expert,
            output);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP5_128x128x64_SwapTrue: {
        g_enable_p5_scale_trace = std::getenv("NEMOTRON_P5_SCALE_DEBUG") != nullptr ? 1 : 0;
        if (g_enable_p5_scale_trace != 0) {
          g_p5_scale_trace = {};
        }
        const auto* p5_tma_load_b_descriptors =
            reinterpret_cast<const routed_p5_tma::P5TmaLoadB*>(
                input_pack.p5_tma_load_b_descriptors(*launch_plan));
        const auto* p5_tma_load_sfb_descriptors =
            reinterpret_cast<const routed_p5_tma::P5TmaLoadSFB*>(
                input_pack.p5_tma_load_sfb_descriptors(*launch_plan));
        if (!LaunchProgrammaticKernel(
            grid,
            block,
            Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
                nvfp4_bridge::UnifiedRoutedFp4Profile::kP5,
                __nv_bfloat16>,
            input_pack.packed_data(),
            input_pack.matmul_block_scales_data(),
            input_pack.scale_layout(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            p5_tma_load_b_descriptors,
            p5_tma_load_sfb_descriptors,
            weights,
            output_rows_per_expert,
            output,
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<std::size_t>(0),
            static_cast<std::size_t>(0),
            Nvfp4ScaleLayout::kSwizzled128x4)) {
          return false;
        }
        if (g_enable_p5_scale_trace != 0) {
          if (!CheckCuda(cudaDeviceSynchronize())) {
            return false;
          }
          PrintP5ScaleTrace();
        }
        return true;
      }
      case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue<__nv_bfloat16, 256, 64><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            weights,
            output_rows_per_expert,
            output);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kLegacy:
        break;
    }
    return false;
  };

  const RoutedGemm1Profile profile = SelectRoutedGemm1Profile(dispatch_rows);
  if (profile != RoutedGemm1Profile::kLegacy && launch_profile(profile)) {
    return true;
  }
  if (RoutedProfileDebugEnabled()) {
    std::fprintf(
        stderr,
        "routed_gemm1 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
        dispatch_rows,
        active_selection_count,
        RoutedGemm1ProfileName(RoutedGemm1Profile::kLegacy));
  }

  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRowsBf16Kernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      input_dq_scales,
      input_per_row_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      launch_plan->expert_first_token_offsets(),
      static_cast<int>(launch_plan->selected_token_tile()),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

struct DirectFp4OutputBufferState {
  std::size_t output_row_tile_count = 0;
  std::size_t blocks_per_row = 0;
  std::size_t padded_blocks_per_row = 0;
  std::size_t scale_count = 0;
};

bool PrepareDirectFp4OutputBuffers(
    const DeviceMoeLaunchPlan& launch_plan,
    std::size_t output_rows_per_expert,
    DeviceNvfp4Matrix* output_pack,
    float* activation_output_scale,
    __nv_bfloat16* debug_prepack_output,
    DirectFp4OutputBufferState* state) {
  if (output_pack == nullptr ||
      activation_output_scale == nullptr ||
      state == nullptr) {
    return false;
  }

  state->output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  state->blocks_per_row = output_pack->cols() / fused_decode::kNvfp4BlockWidth;
  state->padded_blocks_per_row =
      RoundUp(state->blocks_per_row, kNvfp4ScaleBlockTile);
  state->scale_count = launch_plan.padded_row_capacity() * state->blocks_per_row;

  if (state->output_row_tile_count == 0 ||
      state->blocks_per_row == 0 ||
      !ClearDeviceNvfp4Matrix(output_pack, 1.0f, 1.0f) ||
      (debug_prepack_output != nullptr &&
       !LaunchZeroBf16Buffer(
           debug_prepack_output,
           launch_plan.padded_row_capacity() * output_rows_per_expert)) ||
      !LaunchZeroBuffer(activation_output_scale, state->scale_count)) {
    return false;
  }

  return true;
}

bool LaunchPlannedPackedInputMatVecFp4Direct(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    DeviceNvfp4Matrix* output_pack,
    float* activation_output_scale,
    __nv_bfloat16* debug_prepack_output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      activation_output_scale == nullptr ||
      output_pack->rows() < launch_plan->padded_row_capacity() ||
      output_pack->cols() != output_rows_per_expert) {
    return false;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }

  const RoutedGemm1Profile profile = SelectRoutedGemm1Profile(dispatch_rows);
  switch (profile) {
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue: {
      if (RoutedProfileDebugEnabled()) {
        std::fprintf(
            stderr,
            "routed_gemm1 dispatch_rows=%zu active_selection_count=%zu profile=%s mode=fp4_direct\n",
            dispatch_rows,
            active_selection_count,
            RoutedGemm1ProfileName(profile));
      }

      DirectFp4OutputBufferState output_state;
      if (!PrepareDirectFp4OutputBuffers(
              *launch_plan,
              output_rows_per_expert,
              output_pack,
              activation_output_scale,
              nullptr,
              &output_state)) {
        return false;
      }

      const dim3 block(kRoutedThreadsPerBlock);
      const dim3 grid(
          static_cast<unsigned int>(output_state.output_row_tile_count),
          static_cast<unsigned int>(*current_cta_capacity));

      g_enable_p5_scale_trace = std::getenv("NEMOTRON_P5_SCALE_DEBUG") != nullptr ? 1 : 0;
      if (g_enable_p5_scale_trace != 0) {
        g_p5_scale_trace = {};
      }

      const auto* p5_tma_load_b_descriptors =
          reinterpret_cast<const routed_p5_tma::P5TmaLoadB*>(
              input_pack.p5_tma_load_b_descriptors(*launch_plan));
      const auto* p5_tma_load_sfb_descriptors =
          reinterpret_cast<const routed_p5_tma::P5TmaLoadSFB*>(
              input_pack.p5_tma_load_sfb_descriptors(*launch_plan));

      if (!LaunchProgrammaticKernel(
              grid,
              block,
              Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
                  nvfp4_bridge::UnifiedRoutedFp4Profile::kP5,
                  __nv_bfloat16,
                  nvfp4_bridge::P5EpilogueMode::kFp4Direct>,
              input_pack.packed_data(),
              input_pack.matmul_block_scales_data(),
              input_pack.scale_layout(),
              input_pack.device_tensor_scale_ptr(),
              input_expert_tensor_scales,
              input_dq_scales,
              input_per_row_tensor_scales,
              launch_plan->num_non_exiting_ctas(),
              launch_plan->cta_idx_xy_to_batch_idx(),
              launch_plan->cta_row_starts(),
              launch_plan->cta_valid_rows(),
              p5_tma_load_b_descriptors,
              p5_tma_load_sfb_descriptors,
              weights,
              output_rows_per_expert,
              static_cast<__nv_bfloat16*>(nullptr),
              const_cast<std::uint8_t*>(output_pack->packed_data()),
              const_cast<std::uint8_t*>(output_pack->block_scales_data()),
              const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()),
              activation_output_scale,
              output_pack->cols(),
              output_state.padded_blocks_per_row,
              output_pack->scale_layout())) {
        return false;
      }

      if (g_enable_p5_scale_trace != 0) {
        if (!CheckCuda(cudaDeviceSynchronize())) {
          return false;
        }
        PrintP5ScaleTrace();
      }
      return true;
    }
    default:
      return false;
  }
}

[[maybe_unused]] bool LaunchPlannedPackedInputMatVecRelu2ExpertScales(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output_expert_scales) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(launch_plan->n_experts(), active_selection_count);
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          reinterpret_cast<void*>(output_expert_scales),
          0,
          launch_plan->n_experts() * sizeof(float)))) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2MaxAbsKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      weights,
      output_rows_per_expert,
      reinterpret_cast<unsigned int*>(output_expert_scales));
  return CheckCuda(cudaGetLastError()) &&
         LaunchMaxBitsToTensorScales(
             reinterpret_cast<const unsigned int*>(output_expert_scales),
             launch_plan->n_experts(),
             output_expert_scales);
}

bool LaunchReduceSelectionOutputs(
    const float* grouped_output,
    const int* selection_to_sorted,
    const int* sorted_to_permuted,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    std::size_t hidden_size,
    float* output,
    float* routed_output) {
  const std::size_t count = token_count * hidden_size;
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ReduceSelectionOutputsKernel<<<grid, block>>>(
      grouped_output,
      selection_to_sorted,
      sorted_to_permuted,
      selected_weights,
      token_count,
      top_k,
      hidden_size,
      output,
      routed_output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchAccumulateSharedOutput(
    const float* shared_output,
    std::size_t count,
    float* output,
    float* shared_output_copy) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  AccumulateSharedOutputKernel<<<grid, block>>>(
      shared_output,
      count,
      output,
      shared_output_copy);
  return CheckCuda(cudaGetLastError());
}

constexpr int kNanoP1MicroTileK = 64;
constexpr int kNanoP1MicroTileBytes = kNanoP1MicroTileK / 2;
constexpr int kNanoP1MicroScaleBytes = kNanoP1MicroTileK / fused_decode::kNvfp4BlockWidth;
constexpr int kNanoP1ProbeDebugOffset =
    nvfp4_bridge::NanoP1ThreadsPerCta * nvfp4_bridge::kNanoP1AccumCoordCount;
constexpr int kNanoP1ProbeFinalThread0Offset = kNanoP1ProbeDebugOffset + 8;
constexpr int kNanoP1ProbeDeadLaneDebugOffset = kNanoP1ProbeFinalThread0Offset + 64;
constexpr int kNanoP1ProbeFinalThread2Offset = kNanoP1ProbeDeadLaneDebugOffset + 8;
static_assert(kNanoP1MicroScaleBytes == 4);

struct NanoP1SharedStorage {
  alignas(1024) cute::array_aligned<nvfp4_bridge::NanoP1SmemAllocA, nvfp4_bridge::kNanoP1SwizzledAElems> smem_A;
  alignas(1024) cute::array_aligned<nvfp4_bridge::NanoP1SmemAllocB, nvfp4_bridge::kNanoP1SwizzledBElems> smem_B;
  alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, nvfp4_bridge::kNanoP1ScaleStageElemsA> smem_SFA;
  alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, nvfp4_bridge::kNanoP1ScaleStageElemsB> smem_SFB;
  alignas(16) cute::array_aligned<std::uint8_t, cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{}) * kNanoP1MicroTileBytes>
      row_major_A;
  alignas(16) cute::array_aligned<std::uint8_t, cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}) * kNanoP1MicroTileBytes>
      row_major_B;
  alignas(16) cute::array_aligned<std::uint32_t, cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{})> scale_words_A;
  alignas(16) cute::array_aligned<std::uint32_t, cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})> scale_words_B;
};

__device__ __forceinline__ void NanoP1CtaBarrier() {
  asm volatile("bar.sync 0;\n" : : : "memory");
}

template <bool kCaptureAccumulatorScratch>
__device__ __forceinline__ void ComputeNanoP1AccumTile(
    NanoP1SharedStorage& shared,
    const std::uint8_t* input_packed,
    const std::uint8_t* weight_packed,
    const std::uint8_t* input_exec_scales,
    const std::uint8_t* weight_exec_scales,
    int tid,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    int64_t hidden_size,
    float* accumulator_scratch,
    nvfp4_bridge::CRegister* accum_storage) {
  using NanoP1TiledMma = nvfp4_bridge::NanoP1TiledMma;
  using NanoP1AccumLayout = nvfp4_bridge::NanoP1AccumLayout;

  constexpr int kTileM = cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kTileN = cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kTileK = cute::size<2>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kMacroTileBytes = kTileK / 2;
  constexpr int kMacroScaleBytes = kTileK / fused_decode::kNvfp4BlockWidth;

  const std::size_t packed_row_bytes = static_cast<std::size_t>(hidden_size) / 2u;
  const std::size_t blocks_per_row =
      static_cast<std::size_t>(hidden_size) / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::uint32_t unit_scale_word = nvfp4_bridge::MakePackedUnitScaleWord();
  const std::uint8_t unit_scale_byte = nvfp4_bridge::LoadScaleByte(unit_scale_word, 0);

  auto accum_tensor = cute::make_tensor(accum_storage, NanoP1AccumLayout{});
  cute::clear(accum_tensor);

  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(tid);
  auto const stage0_A = nvfp4_bridge::NanoP1SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
  auto const stage0_B = nvfp4_bridge::NanoP1SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()),
      nvfp4_bridge::NanoP1SmemLayoutA{});
  auto sB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_B.data()),
      nvfp4_bridge::NanoP1SmemLayoutB{});
  auto sSFA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFA.data()),
      nvfp4_bridge::NanoP1SmemLayoutSFA{});
  auto sSFB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()),
      nvfp4_bridge::NanoP1SmemLayoutSFB{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA_);
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB_);
  auto* swizzled_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
  auto* swizzled_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());

  auto sA_stage0 = sA(cute::_, cute::_, cute::Int<0>{});
  auto sB_stage0 = sB(cute::_, cute::_, cute::Int<0>{});
  auto sSFA_stage0 = sSFA_(cute::_, cute::_, cute::Int<0>{});
  auto sSFB_stage0 = sSFB_(cute::_, cute::_, cute::Int<0>{});

  auto tCrA = thread_mma.partition_fragment_A(sA_stage0);
  auto tCrB = thread_mma.partition_fragment_B(sB_stage0);
  auto tCrSFA = nvfp4_bridge::NanoP1PartitionScaleA(sSFA_stage0, thread_mma);
  auto tCrSFB = nvfp4_bridge::NanoP1PartitionScaleB(sSFB_stage0, thread_mma);

  auto s2r_copy_A = cute::make_tiled_copy_A(nvfp4_bridge::NanoP1SmemCopyAtomA{}, mma);
  auto s2r_thr_A = s2r_copy_A.get_thread_slice(tid);
  auto tCsA = s2r_thr_A.partition_S(sA);
  auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

  auto s2r_copy_B = cute::make_tiled_copy_B(nvfp4_bridge::NanoP1SmemCopyAtomB{}, mma);
  auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
  auto tCsB = s2r_thr_B.partition_S(sB);
  auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

  auto tile_shape_mnk = cute::tile_shape(mma);
  auto s2r_copy_SFA = cute::make_tiled_copy_impl(
      nvfp4_bridge::NanoP1SmemCopyAtomSFA{},
      nvfp4_bridge::NanoP1GetLayoutSFATV(mma),
      cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(tid);
  auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
  auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

  auto s2r_copy_SFB = cute::make_tiled_copy_impl(
      nvfp4_bridge::NanoP1SmemCopyAtomSFB{},
      nvfp4_bridge::NanoP1GetLayoutSFBTV(mma),
      cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(tid);
  auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
  auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

  static_assert(cute::size<1>(decltype(tCrA){}) == cute::size<1>(decltype(tCrSFA){}));
  static_assert(cute::size<1>(decltype(tCrB){}) == cute::size<1>(decltype(tCrSFB){}));

  for (int64_t k_base = 0; k_base < hidden_size; k_base += kTileK) {
    const int64_t remaining_k = hidden_size - k_base;
    const std::size_t available_k = static_cast<std::size_t>(
        remaining_k < static_cast<int64_t>(kTileK) ? remaining_k : kTileK);
    const std::size_t available_bytes = available_k / 2u;
    const std::size_t available_blocks =
        available_k / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = static_cast<std::size_t>(k_base) / 2u;
    const std::size_t block_base =
        static_cast<std::size_t>(k_base) / fused_decode::kNvfp4BlockWidth;

    for (int row = tid; row < kTileN; row += blockDim.x) {
      const bool row_valid = row < valid_cols;
      const std::size_t source_row = static_cast<std::size_t>(output_col_base + row);
      const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
      for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
        std::uint8_t value = 0u;
        if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
          value = weight_packed[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const auto elem_offset = stage0_A(row, byte_index * 2);
        swizzled_a_bytes[static_cast<int>(elem_offset) / 2] = value;
      }
      std::uint8_t scale_bytes[kMacroScaleBytes];
      if (row_valid) {
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          std::uint8_t value = unit_scale_byte;
          if (static_cast<std::size_t>(scale_index) >= available_blocks) {
            value = 0u;
          } else if (weight_exec_scales != nullptr) {
            value = LoadExecutionScaleByte(
                weight_exec_scales,
                source_row,
                block_base + static_cast<std::size_t>(scale_index),
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
          }
          scale_bytes[scale_index] = value;
        }
        nvfp4_bridge::StoreTracedScaleBytes(sSFA_stage0, scale_bytes, row);
      } else {
        nvfp4_bridge::ZeroTracedP5ScaleRow(sSFA_stage0, row);
      }
    }

    for (int row = tid; row < kTileM; row += blockDim.x) {
      const bool row_valid = row < valid_rows;
      const std::size_t source_row = static_cast<std::size_t>(row_start + row);
      const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
      for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
        std::uint8_t value = 0u;
        if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
          value = input_packed[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const auto elem_offset = stage0_B(row, byte_index * 2);
        swizzled_b_bytes[static_cast<int>(elem_offset) / 2] = value;
      }
      std::uint8_t scale_bytes[kMacroScaleBytes];
      if (row_valid) {
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          std::uint8_t value = unit_scale_byte;
          if (static_cast<std::size_t>(scale_index) >= available_blocks) {
            value = 0u;
          } else if (input_exec_scales != nullptr) {
            value = LoadExecutionScaleByte(
                input_exec_scales,
                source_row,
                block_base + static_cast<std::size_t>(scale_index),
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
          }
          scale_bytes[scale_index] = value;
        }
        nvfp4_bridge::StoreTracedScaleBytes(sSFB_stage0, scale_bytes, row);
      } else {
        nvfp4_bridge::ZeroTracedP5ScaleRow(sSFB_stage0, row);
      }
    }

    NanoP1CtaBarrier();

    cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
    cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
    cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
    cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

    using MMAOp = typename NanoP1TiledMma::MMA_Op;
    for (int k_block = 0; k_block < cute::size<2>(tCrA_cv); ++k_block) {
      cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k_block));
      cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k_block));
    }

    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          k_base == 0 &&
          output_col_base == 0 &&
          row_start == 0 &&
          (tid == 0 || tid == 2)) {
        auto rA = cute::recast<nvfp4_bridge::ARegister>(tCrA);
        auto rB = cute::recast<nvfp4_bridge::BRegister>(tCrB);
        auto rSFA = cute::recast<nvfp4_bridge::SFRegister>(cute::filter_zeros(tCrSFA));
        auto rSFB = cute::recast<nvfp4_bridge::SFRegister>(cute::filter_zeros(tCrSFB));
        const int debug_offset =
            tid == 0 ? kNanoP1ProbeDebugOffset : kNanoP1ProbeDeadLaneDebugOffset;
        accumulator_scratch[debug_offset + 0] =
            __uint_as_float(static_cast<std::uint32_t>(rA(0)));
        accumulator_scratch[debug_offset + 1] =
            __uint_as_float(static_cast<std::uint32_t>(rA(1)));
        accumulator_scratch[debug_offset + 2] =
            __uint_as_float(static_cast<std::uint32_t>(rA(2)));
        accumulator_scratch[debug_offset + 3] =
            __uint_as_float(static_cast<std::uint32_t>(rA(3)));
        accumulator_scratch[debug_offset + 4] =
            __uint_as_float(static_cast<std::uint32_t>(rSFA(0)));
        accumulator_scratch[debug_offset + 5] =
            __uint_as_float(static_cast<std::uint32_t>(rB(0)));
        accumulator_scratch[debug_offset + 6] =
            __uint_as_float(static_cast<std::uint32_t>(rB(1)));
        accumulator_scratch[debug_offset + 7] =
            __uint_as_float(static_cast<std::uint32_t>(rSFB(0)));
      }
    }

    constexpr int kMmaKBlocks = cute::size<2>(decltype(tCrA){});
    for (int k_block = 0; k_block < kMmaKBlocks; ++k_block) {
      cute::gemm(
          mma,
          cute::make_zip_tensor(tCrA(cute::_, cute::_, k_block), tCrSFA(cute::_, cute::_, k_block)),
          cute::make_zip_tensor(tCrB(cute::_, cute::_, k_block), tCrSFB(cute::_, cute::_, k_block)),
          accum_tensor);
      if constexpr (kCaptureAccumulatorScratch) {
        if (accumulator_scratch != nullptr &&
            k_base == 0 &&
            k_block == 0 &&
            output_col_base == 0 &&
            row_start == 0) {
          const std::size_t thread_offset =
              static_cast<std::size_t>(threadIdx.x) *
              static_cast<std::size_t>(nvfp4_bridge::kNanoP1AccumCoordCount);
#pragma unroll
          for (int physical = 0; physical < nvfp4_bridge::kNanoP1AccumCoordCount; ++physical) {
            accumulator_scratch[thread_offset + static_cast<std::size_t>(physical)] =
                accum_storage[physical];
          }
        }
      }
    }

    NanoP1CtaBarrier();
  }

  if constexpr (kCaptureAccumulatorScratch) {
    if (accumulator_scratch != nullptr &&
        output_col_base == 0 &&
        row_start == 0 &&
        tid == 0) {
      for (int physical = 0; physical < nvfp4_bridge::kNanoP1AccumCoordCount; ++physical) {
        accumulator_scratch[kNanoP1ProbeFinalThread0Offset + physical] = accum_storage[physical];
      }
    }
    if (accumulator_scratch != nullptr &&
        output_col_base == 0 &&
        row_start == 0 &&
        tid == 2) {
      for (int physical = 0; physical < nvfp4_bridge::kNanoP1AccumCoordCount; ++physical) {
        accumulator_scratch[kNanoP1ProbeFinalThread2Offset + physical] = accum_storage[physical];
      }
    }
  }
}

template <nvfp4_bridge::NanoP1EpilogueMode Mode>
__global__ __launch_bounds__(nvfp4_bridge::NanoP1ThreadsPerCta) void NanoP1Kernel(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    __nv_bfloat16* bf16_output,
    std::uint8_t* packed_output,
    std::uint8_t* block_scales_output,
    std::uint8_t* matmul_block_scales_output,
    float* activation_output_scales,
    float* direct_pack_staging,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    float* accumulator_scratch) {
  constexpr int kTileM = cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kTileN = cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{});
  static_assert(fused_decode::kNvfp4BlockWidth == 16);

  __shared__ NanoP1SharedStorage shared;

  if (input_fp4 == nullptr ||
      weight_fp4 == nullptr ||
      g1_alphas == nullptr ||
      num_rows <= 0 ||
      hidden_size <= 0 ||
      inter_size <= 0 ||
      (hidden_size % fused_decode::kNvfp4BlockWidth) != 0) {
    return;
  }
  if constexpr (Mode == nvfp4_bridge::NanoP1EpilogueMode::kBf16Dense) {
    if (bf16_output == nullptr) {
      return;
    }
  } else {
    if (packed_output == nullptr ||
        block_scales_output == nullptr ||
        matmul_block_scales_output == nullptr ||
        activation_output_scales == nullptr ||
        direct_pack_staging == nullptr ||
        (inter_size % fused_decode::kNvfp4BlockWidth) != 0) {
      return;
    }
  }

  auto const* input_packed = static_cast<std::uint8_t const*>(input_fp4);
  auto const* weight_packed = static_cast<std::uint8_t const*>(weight_fp4);
  auto const* input_exec_scales = static_cast<std::uint8_t const*>(input_sf);
  auto const* weight_exec_scales = static_cast<std::uint8_t const*>(weight_sf);
  const float alpha = g1_alphas[0];
  const int tid = static_cast<int>(threadIdx.x);
  const int row_start = static_cast<int>(blockIdx.y) * kTileM;
  const int valid_rows =
      max(0, min(static_cast<int>(num_rows) - row_start, kTileM));
  if (valid_rows <= 0) {
    return;
  }

  if constexpr (Mode == nvfp4_bridge::NanoP1EpilogueMode::kBf16Dense) {
    const int output_col_base = static_cast<int>(blockIdx.x) * kTileN;
    const int valid_cols =
        max(0, min(static_cast<int>(inter_size) - output_col_base, kTileN));
    if (valid_cols <= 0) {
      return;
    }

    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kNanoP1AccumCoordCount];
    ComputeNanoP1AccumTile<true>(
        shared,
        input_packed,
        weight_packed,
        input_exec_scales,
        weight_exec_scales,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        hidden_size,
        accumulator_scratch,
        accum_storage);
    nvfp4_bridge::StoreNanoP1CFragmentsRowMajor(
        alpha,
        accum_storage,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        static_cast<std::size_t>(inter_size),
        bf16_output);
    return;
  }

  if (blockIdx.x != 0) {
    return;
  }

  for (int output_col_base = 0; output_col_base < inter_size; output_col_base += kTileN) {
    const int valid_cols =
        max(0, min(static_cast<int>(inter_size) - output_col_base, kTileN));
    if (valid_cols <= 0) {
      continue;
    }
    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kNanoP1AccumCoordCount];
    ComputeNanoP1AccumTile<false>(
        shared,
        input_packed,
        weight_packed,
        input_exec_scales,
        weight_exec_scales,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        hidden_size,
        nullptr,
        accum_storage);
    nvfp4_bridge::AccumulateNanoP1DirectPackRowMaxAbs(
        alpha,
        accum_storage,
        tid,
        row_start,
        valid_rows,
        valid_cols,
        activation_output_scales);
  }
  __syncthreads();

  for (int row = tid; row < valid_rows; row += blockDim.x) {
    const std::size_t row_index = static_cast<std::size_t>(row_start + row);
    const float max_abs = activation_output_scales[row_index];
    float row_scale = 1.0f;
    if (max_abs > kNvfp4ActivationMaxFinite) {
      row_scale = ClampNvfp4TensorScale(max_abs / kNvfp4ActivationMaxFinite);
    }
    activation_output_scales[row_index] = row_scale;
  }
  __syncthreads();

  constexpr std::size_t kStagingTileElements =
      static_cast<std::size_t>(kTileM) * static_cast<std::size_t>(kTileN);
  float* staging_tile =
      direct_pack_staging + (static_cast<std::size_t>(blockIdx.y) * kStagingTileElements);
  for (int output_col_base = 0; output_col_base < inter_size; output_col_base += kTileN) {
    const int valid_cols =
        max(0, min(static_cast<int>(inter_size) - output_col_base, kTileN));
    if (valid_cols <= 0) {
      continue;
    }
    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kNanoP1AccumCoordCount];
    ComputeNanoP1AccumTile<false>(
        shared,
        input_packed,
        weight_packed,
        input_exec_scales,
        weight_exec_scales,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        hidden_size,
        nullptr,
        accum_storage);
    nvfp4_bridge::StoreNanoP1DirectPackCFragments(
        alpha,
        accum_storage,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        static_cast<int>(num_rows),
        static_cast<int>(inter_size),
        packed_output,
        block_scales_output,
        matmul_block_scales_output,
        activation_output_scales,
        staging_tile);
  }
}

bool RunNanoP1KernelForTestingImpl(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    __nv_bfloat16* bf16_output,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    cudaStream_t stream,
    float* accumulator_scratch) {
  if (input_fp4 == nullptr ||
      weight_fp4 == nullptr ||
      bf16_output == nullptr ||
      g1_alphas == nullptr ||
      num_rows <= 0 ||
      hidden_size <= 0 ||
      inter_size <= 0 ||
      (hidden_size % fused_decode::kNvfp4BlockWidth) != 0) {
    return false;
  }

  const dim3 block(nvfp4_bridge::NanoP1ThreadsPerCta);
  const dim3 grid(
      static_cast<unsigned int>(
          (static_cast<std::size_t>(inter_size) +
           static_cast<std::size_t>(cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
          static_cast<std::size_t>(cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{}))),
      static_cast<unsigned int>(
          (static_cast<std::size_t>(num_rows) +
           static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
          static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}))));
  if (grid.x == 0 || grid.y == 0) {
    return false;
  }

  NanoP1Kernel<nvfp4_bridge::NanoP1EpilogueMode::kBf16Dense><<<grid, block, 0, stream>>>(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      bf16_output,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      accumulator_scratch);
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaStreamSynchronize(stream));
}

bool RunNanoP1DirectPackKernelForTestingImpl(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    std::uint8_t* packed_output,
    std::uint8_t* block_scales_output,
    std::uint8_t* matmul_block_scales_output,
    float* activation_output_scales,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    cudaStream_t stream) {
  if (input_fp4 == nullptr ||
      weight_fp4 == nullptr ||
      packed_output == nullptr ||
      block_scales_output == nullptr ||
      matmul_block_scales_output == nullptr ||
      activation_output_scales == nullptr ||
      g1_alphas == nullptr ||
      num_rows <= 0 ||
      hidden_size <= 0 ||
      inter_size <= 0 ||
      (hidden_size % fused_decode::kNvfp4BlockWidth) != 0 ||
      (inter_size % fused_decode::kNvfp4BlockWidth) != 0) {
    return false;
  }

  const std::size_t packed_nbytes =
      static_cast<std::size_t>(num_rows) * (static_cast<std::size_t>(inter_size) / 2u);
  const std::size_t padded_blocks_per_row = RoundUp(
      static_cast<std::size_t>(inter_size) / fused_decode::kNvfp4BlockWidth,
      kNvfp4ScaleBlockTile);
  const std::size_t scale_nbytes = static_cast<std::size_t>(num_rows) * padded_blocks_per_row;
  const std::size_t row_tile_count =
      (static_cast<std::size_t>(num_rows) +
       static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
      static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}));
  float* direct_pack_staging_dev = nullptr;
  const std::size_t staging_count =
      row_tile_count *
      static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) *
      static_cast<std::size_t>(cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{}));
  if (!CheckCuda(cudaMalloc(
          reinterpret_cast<void**>(&direct_pack_staging_dev),
          staging_count * sizeof(float)))) {
    return false;
  }
  if (!CheckCuda(cudaMemsetAsync(packed_output, 0, packed_nbytes, stream)) ||
      !CheckCuda(cudaMemsetAsync(block_scales_output, 0, scale_nbytes, stream)) ||
      !CheckCuda(cudaMemsetAsync(matmul_block_scales_output, 0, scale_nbytes, stream)) ||
      !CheckCuda(cudaMemsetAsync(
          activation_output_scales,
          0,
          static_cast<std::size_t>(num_rows) * sizeof(float),
          stream))) {
    cudaFree(direct_pack_staging_dev);
    return false;
  }

  const dim3 block(nvfp4_bridge::NanoP1ThreadsPerCta);
  const dim3 grid(
      1u,
      static_cast<unsigned int>(
          (static_cast<std::size_t>(num_rows) +
           static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
          static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}))));
  if (grid.y == 0) {
    return false;
  }

  NanoP1Kernel<nvfp4_bridge::NanoP1EpilogueMode::kDirectPack><<<grid, block, 0, stream>>>(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      nullptr,
      packed_output,
      block_scales_output,
      matmul_block_scales_output,
      activation_output_scales,
      direct_pack_staging_dev,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      nullptr);
  const bool ok =
      CheckCuda(cudaGetLastError()) && CheckCuda(cudaStreamSynchronize(stream));
  cudaFree(direct_pack_staging_dev);
  return ok;
}

}  // namespace

bool CopyP13DebugTrace(P13DebugTrace* out) {
  if (out == nullptr) {
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return false;
  }
  *out = g_p13_debug_trace;
  return true;
}

void ResetP13DebugTrace() {
  g_p13_debug_trace = {};
}

const char* SelectRoutedGemm1ProfileNameForTesting(std::size_t num_rows) {
  return RoutedGemm1ProfileName(SelectRoutedGemm1Profile(num_rows));
}

const char* ClassifyRoutedGemm1ProfileForTesting(std::size_t num_rows) {
  return RoutedGemm1ProfileClassName(SelectRoutedGemm1Profile(num_rows));
}

const char* SelectRoutedGemm2ProfileNameForTesting(std::size_t num_rows) {
  return RoutedGemm2ProfileName(SelectRoutedGemm2Profile(num_rows));
}

const char* ClassifyRoutedGemm2ProfileForTesting(std::size_t num_rows) {
  return RoutedGemm2ProfileClassName(SelectRoutedGemm2Profile(num_rows));
}

bool RunGroupedNvfp4ExpertMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (input == nullptr ||
      expert_offsets == nullptr ||
      n_experts == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchGroupedMatVec(
      input,
      expert_offsets,
      n_experts,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedNvfp4ExpertMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchPlannedMatVec(
      input,
      launch_plan,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedPackedNvfp4ExpertMatVecBf16(
    const DeviceNvfp4Matrix& input_pack,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchPlannedPackedInputMatVecBf16(
      input_pack,
      nullptr,
      nullptr,
      input_pack.per_row_tensor_scales(),
      launch_plan,
      dispatch_rows,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedNvfp4ExpertMatVecBf16(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchPlannedMatVecBf16(
      input,
      launch_plan,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunNanoP1KernelForTesting(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    __nv_bfloat16* bf16_output,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    cudaStream_t stream,
    float* accumulator_scratch) {
  return RunNanoP1KernelForTestingImpl(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      bf16_output,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      stream,
      accumulator_scratch);
}

bool RunNanoP1DirectPackKernelForTesting(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    std::uint8_t* packed_output,
    std::uint8_t* block_scales_output,
    std::uint8_t* matmul_block_scales_output,
    float* activation_output_scales,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    cudaStream_t stream) {
  return RunNanoP1DirectPackKernelForTestingImpl(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      packed_output,
      block_scales_output,
      matmul_block_scales_output,
      activation_output_scales,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      stream);
}

__global__ void P5NativeDirectPackOracleKernel(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  constexpr int kTileRows = 128;
  constexpr int kTileCols = 128;
  constexpr int kPassRows = nvfp4_bridge::kTracedP5DirectStageRows;

  static_assert(cute::size(nvfp4_bridge::TracedP5TiledMma{}) == 256);
  auto mma = nvfp4_bridge::TracedP5TiledMma{};
  auto thr_mma = mma.get_thread_slice(threadIdx.x);
  auto ref_c = cute::make_identity_tensor(
      cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
  auto part_c = thr_mma.partition_C(ref_c);
  static_assert(
      decltype(cute::size(part_c))::value == 16,
      "P5 warp-local direct pack expects 16 logical accum coords per thread");
  constexpr int kLinearToBlockHalf[16] = {
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  constexpr int kLinearToTokenGroup[16] = {
      0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3};
  constexpr int kLinearToMHalf[16] = {
      0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};

  for (std::size_t row = threadIdx.x; row < static_cast<std::size_t>(kTileRows);
       row += blockDim.x) {
    per_row_tensor_scales[row] =
        per_row_tensor_scales_input != nullptr ? per_row_tensor_scales_input[row]
                                               : 1.0f;
  }
  if (threadIdx.x == 0) {
    *tensor_scale_data = 1.0f;
  }
  __syncthreads();

  for (int pass = 0; pass < (kTileRows / kPassRows); ++pass) {
    const int row_base = pass * kPassRows;
    const int pass_valid_rows =
        max(0, min(valid_rows - row_base, kPassRows));
    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kTracedP5AccumProfileCosize];
#pragma unroll
    for (int i = 0; i < nvfp4_bridge::kTracedP5AccumProfileCosize; ++i) {
      accum_storage[i] = 0.0f;
    }

    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        nvfp4_bridge::TracedP5AccumProfileLayout{});

#pragma unroll
    for (int linear = 0; linear < static_cast<int>(cute::size(part_c)); ++linear) {
      auto coord = part_c(linear);
      const int m_coord = static_cast<int>(cute::get<0>(coord));
      const int n_coord = static_cast<int>(cute::get<1>(coord));
      const int block_half = kLinearToBlockHalf[linear];
      const int token_group = kLinearToTokenGroup[linear];
      const int m_half = kLinearToMHalf[linear];
      const int warp_id = threadIdx.x / 32;
      const int lane_id = threadIdx.x & 31;
      const int g_m = warp_id & 3;
      const int g_n = warp_id / 4;
      const int q = lane_id / 4;
      const int r = lane_id & 3;
      const int expected_n_coords[4] = {
          16 * g_n + 2 * r,
          16 * g_n + 2 * r + 1,
          16 * g_n + 8 + 2 * r,
          16 * g_n + 9 + 2 * r};
      const int expected_m_coord =
          (block_half == 0 ? 0 : 64) + 16 * g_m + (m_half == 0 ? q : 8 + q);
      const int expected_n_coord = expected_n_coords[token_group];
      if (m_coord != expected_m_coord || n_coord != expected_n_coord) {
        printf(
            "p5_native_direct_pack_oracle mapping mismatch thread=%d linear=%d actual=(%d,%d) expected=(%d,%d)\n",
            static_cast<int>(threadIdx.x),
            linear,
            m_coord,
            n_coord,
            expected_m_coord,
            expected_n_coord);
        asm("trap;");
      }
      if (n_coord < pass_valid_rows && m_coord < kTileCols) {
        const int abs_row = row_base + n_coord;
        if (abs_row < kTileRows) {
          accum_tensor(linear) =
              input[static_cast<std::size_t>(abs_row) * kTileCols + m_coord];
        }
      }
    }

    nvfp4_bridge::StoreUnifiedRoutedFp4DirectPack(
        1.0f,
        &accum_storage[0],
        threadIdx.x,
        0,
        pass_valid_rows,
        kTileCols,
        static_cast<std::size_t>(row_base),
        static_cast<std::size_t>(kTileCols),
        per_row_tensor_scales_input,
        1.0f,
        packed_data,
        block_scales_data,
        matmul_block_scales_data,
        activation_output_scale,
        static_cast<std::size_t>(kTileCols),
        padded_blocks_per_row,
        scale_layout);
    __syncthreads();
  }
}

bool RunP5NativeDirectPackOracleForTesting(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  if (input == nullptr ||
      activation_output_scale == nullptr ||
      packed_data == nullptr ||
      block_scales_data == nullptr ||
      matmul_block_scales_data == nullptr ||
      tensor_scale_data == nullptr ||
      per_row_tensor_scales == nullptr ||
      valid_rows < 0 ||
      valid_rows > 128) {
    return false;
  }

  P5NativeDirectPackOracleKernel<<<1, 256>>>(
      input,
      per_row_tensor_scales_input,
      valid_rows,
      padded_blocks_per_row,
      scale_layout,
      activation_output_scale,
      packed_data,
      block_scales_data,
      matmul_block_scales_data,
      tensor_scale_data,
      per_row_tensor_scales);
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize());
}

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  const std::size_t selection_count = params.token_count * params.top_k;
  DeviceNvfp4Matrix* shared_fc1_pack =
      params.shared_fc1_pack != nullptr ? params.shared_fc1_pack
                                        : const_cast<DeviceNvfp4Matrix*>(params.normalized_pack);
  DeviceNvfp4Matrix* shared_fc2_pack = params.shared_fc2_pack;
  const bool use_packed_fc1_source =
      params.normalized_pack != nullptr &&
      params.normalized_pack->valid() &&
      params.fc1_grouped_pack != nullptr &&
      params.fc1_grouped_pack->valid();
  const float* fc1_input_expert_scales =
      use_packed_fc1_source ? nullptr : params.fc1_expert_activation_scales;
  DeviceNvfp4Matrix* gemm1_output =
      params.gemm1_output != nullptr ? params.gemm1_output : params.fc2_grouped_pack;
  float* gemm1_output_scale =
      params.gemm1_output_scale != nullptr ? params.gemm1_output_scale
                                           : params.fc2_expert_activation_scales;
  float* activation_output_scale =
      params.activation_output_scale != nullptr ? params.activation_output_scale
                                                : gemm1_output_scale;
  const bool use_legacy_packed_fc1 =
      params.fc1_grouped_pack != nullptr && fc1_input_expert_scales != nullptr;
  const bool use_grouped_gemm1_output =
      gemm1_output != nullptr &&
      gemm1_output_scale != nullptr &&
      activation_output_scale != nullptr;
  const bool use_packed_fc2 =
      gemm1_output != nullptr && activation_output_scale != nullptr;

  if (params.token_count == 0 ||
      params.hidden_size == 0 ||
      params.hidden_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.routed_expert_intermediate_size == 0 ||
      params.routed_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.shared_expert_intermediate_size == 0 ||
      params.shared_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.n_routed_experts == 0 ||
      params.top_k == 0 ||
      params.top_k > params.n_routed_experts ||
      !ValidFusedNvfp4WeightView(params.shared_up) ||
      !ValidFusedNvfp4WeightView(params.shared_down) ||
      params.routed_up_device == nullptr ||
      params.routed_down_device == nullptr ||
      params.selected_indices == nullptr ||
      params.selected_weights == nullptr ||
      params.input == nullptr ||
      params.normalized == nullptr ||
      params.routing == nullptr ||
      params.launch_plan == nullptr ||
      params.routed_gather_scratch == nullptr ||
      (!use_grouped_gemm1_output && params.routed_up_scratch == nullptr) ||
      params.shared_up_scratch == nullptr ||
      params.output == nullptr) {
    return false;
  }

  if (!params.routing->valid() ||
      params.routing->n_experts() != params.n_routed_experts ||
      params.routing->selection_count() < selection_count ||
      !params.launch_plan->valid() ||
      params.launch_plan->n_experts() != params.n_routed_experts ||
      params.launch_plan->selection_count() < selection_count) {
    return false;
  }

  if (params.token_count >
          (std::numeric_limits<std::size_t>::max() / params.top_k) ||
      params.token_count >
          (std::numeric_limits<std::size_t>::max() / params.hidden_size) ||
      params.token_count >
          static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) ||
      params.shared_up.input_cols != params.hidden_size ||
      params.shared_up.output_rows != params.shared_expert_intermediate_size ||
      params.shared_down.input_cols != params.shared_expert_intermediate_size ||
      params.shared_down.output_rows != params.hidden_size ||
      (shared_fc1_pack != nullptr &&
       (!shared_fc1_pack->valid() ||
        shared_fc1_pack->rows() < params.token_count ||
        shared_fc1_pack->cols() != params.hidden_size)) ||
      (shared_fc2_pack != nullptr &&
       (!shared_fc2_pack->valid() ||
        shared_fc2_pack->rows() < params.token_count ||
        shared_fc2_pack->cols() != params.shared_expert_intermediate_size))) {
    return false;
  }

  const std::size_t token_hidden_count = params.token_count * params.hidden_size;
  const std::size_t shared_intermediate_count =
      params.token_count * params.shared_expert_intermediate_size;
  const auto current_padded_row_capacity =
      DeviceMoeLaunchPlan::PaddedRowCapacity(params.n_routed_experts, selection_count);
  if (!current_padded_row_capacity.has_value()) {
    return false;
  }
  if (!RunDeviceExpertRouting(
          params.selected_indices,
          params.selected_weights,
          params.token_count,
          params.top_k,
          params.routing) ||
      !BuildDeviceMoeLaunchPlan(*params.routing, selection_count, params.launch_plan) ||
      !LaunchZeroBuffer(params.output, token_hidden_count) ||
      !LaunchZeroBuffer(params.routed_output, token_hidden_count) ||
      !LaunchZeroBuffer(params.shared_output, token_hidden_count)) {
    return false;
  }

  const std::size_t packed_dispatch_rows =
      RoutedEnvEnabled("NEMOTRON_DEBUG_USE_SELECTED_TOKEN_TILE_FOR_DISPATCH")
          ? params.launch_plan->selected_token_tile()
          : params.token_count;
  const RoutedGemm1Profile grouped_gemm1_profile =
      SelectRoutedGemm1Profile(packed_dispatch_rows);
  const bool use_fp4_direct_fc1 =
      use_grouped_gemm1_output &&
      use_packed_fc1_source &&
      (grouped_gemm1_profile == RoutedGemm1Profile::kP1_128x128x64_SwapFalse ||
       grouped_gemm1_profile == RoutedGemm1Profile::kP5_128x128x64_SwapTrue);

  if ((!use_packed_fc1_source &&
       !LaunchGatherRows(
           params.normalized,
           params.launch_plan->permuted_idx_to_token_idx(),
           params.launch_plan->total_num_padded_tokens(),
           params.launch_plan->padded_row_capacity(),
           params.token_count,
           params.hidden_size,
           params.routed_gather_scratch)) ||
      (use_packed_fc1_source &&
       !GatherDeviceNvfp4Rows(
           *params.normalized_pack,
           params.launch_plan->permuted_idx_to_token_idx(),
           params.launch_plan->padded_row_capacity(),
           params.fc1_grouped_pack)) ||
      (use_legacy_packed_fc1 && !use_grouped_gemm1_output &&
       (!LaunchComputeExpertActivationScales(
            params.routed_gather_scratch,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            params.hidden_size,
            const_cast<float*>(fc1_input_expert_scales)) ||
        !PackDeviceRowMajorFp32ToNvfp4PerExpert(
            params.routed_gather_scratch,
            params.launch_plan->padded_row_capacity(),
            params.hidden_size,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            fc1_input_expert_scales,
            params.fc1_grouped_pack))) ||
      (!use_legacy_packed_fc1 && !use_grouped_gemm1_output &&
       !LaunchQuantizeDequantizeRows(
           params.routed_gather_scratch,
           params.launch_plan->total_num_padded_tokens(),
           params.launch_plan->padded_row_capacity(),
           params.hidden_size))) {
    return false;
  }

  if (use_grouped_gemm1_output) {
    if (use_fp4_direct_fc1) {
      __nv_bfloat16* fp4_direct_compare_workspace =
          RoutedEnvEnabled("NEMOTRON_DEBUG_COMPARE_FP4_DIRECT")
              ? params.gemm1_output_bf16
              : nullptr;
      if (!LaunchPlannedPackedInputMatVecFp4Direct(
              *params.fc1_grouped_pack,
              nullptr,
              nullptr,
              params.fc1_grouped_pack->per_row_tensor_scales(),
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_up_device,
              params.routed_expert_intermediate_size,
              gemm1_output,
              activation_output_scale,
              fp4_direct_compare_workspace) ||
          !MaybeCompareFirstFp4DirectFc1Output(
              *params.fc1_grouped_pack,
              nullptr,
              nullptr,
              params.fc1_grouped_pack->per_row_tensor_scales(),
              params.routing,
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_up_device,
              params.routed_expert_intermediate_size,
              *gemm1_output,
              activation_output_scale,
              params.gemm1_output_bf16) ||
          !LaunchZeroBuffer(
              params.routed_gather_scratch,
              params.launch_plan->padded_row_capacity() * params.hidden_size) ||
          !LaunchPlannedPackedInputMatVec(
              *gemm1_output,
              nullptr,
              activation_output_scale,
              nullptr,
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_down_device,
              params.hidden_size,
              params.routed_gather_scratch)) {
        return false;
      }
    } else {
      if (params.gemm1_output_bf16 == nullptr ||
          !LaunchZeroBf16Buffer(
              params.gemm1_output_bf16,
              params.launch_plan->padded_row_capacity() *
                  params.routed_expert_intermediate_size) ||
          !(use_packed_fc1_source
                ? LaunchPlannedPackedInputMatVecBf16(
                      *params.fc1_grouped_pack,
                      nullptr,
                      nullptr,
                      params.fc1_grouped_pack->per_row_tensor_scales(),
                      params.launch_plan,
                      packed_dispatch_rows,
                      selection_count,
                      params.routed_up_device,
                      params.routed_expert_intermediate_size,
                      params.gemm1_output_bf16)
                : RunLaunchPlannedNvfp4ExpertMatVecBf16(
                      params.routed_gather_scratch,
                      params.launch_plan,
                      selection_count,
                      params.routed_up_device,
                      params.routed_expert_intermediate_size,
                      params.gemm1_output_bf16)) ||
          !LaunchRoutedBf16Relu2Pack(
              params.gemm1_output_bf16,
              params.routing,
              params.launch_plan,
              selection_count,
              gemm1_output,
              activation_output_scale) ||
          (use_packed_fc1_source &&
           !MaybeCompareFirstGroupedBf16Fc1Output(
               params.normalized,
               params.routing,
               params.launch_plan,
               params.token_count,
               selection_count,
               params.hidden_size,
               params.routed_gather_scratch,
               params.routed_up_device,
               params.routed_expert_intermediate_size,
               *gemm1_output,
               activation_output_scale,
               params.gemm1_output_bf16)) ||
          !LaunchZeroBuffer(
              params.routed_gather_scratch,
              params.launch_plan->padded_row_capacity() * params.hidden_size) ||
          !LaunchPlannedPackedInputMatVec(
              *gemm1_output,
              nullptr,
              activation_output_scale,
              nullptr,
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_down_device,
              params.hidden_size,
              params.routed_gather_scratch)) {
        return false;
      }
    }
  } else {
    if (!LaunchZeroBuffer(
            params.routed_up_scratch,
            *current_padded_row_capacity * params.routed_expert_intermediate_size) ||
        !(use_legacy_packed_fc1
              ? LaunchPlannedPackedInputMatVec(
                    *params.fc1_grouped_pack,
                    fc1_input_expert_scales,
                    nullptr,
                    params.fc1_grouped_pack->per_row_tensor_scales(),
                    params.launch_plan,
                    packed_dispatch_rows,
                    selection_count,
                    params.routed_up_device,
                    params.routed_expert_intermediate_size,
                    params.routed_up_scratch)
              : RunLaunchPlannedNvfp4ExpertMatVec(
                    params.routed_gather_scratch,
                    params.launch_plan,
                    selection_count,
                    params.routed_up_device,
                    params.routed_expert_intermediate_size,
                    params.routed_up_scratch)) ||
        !LaunchRelu2Rows(
            params.routed_up_scratch,
            params.launch_plan->total_num_padded_tokens(),
            params.launch_plan->padded_row_capacity(),
            params.routed_expert_intermediate_size) ||
        (activation_output_scale != nullptr &&
         !LaunchComputeExpertActivationScales(
             params.routed_up_scratch,
             params.launch_plan->expert_first_token_offsets(),
             params.n_routed_experts,
             params.routed_expert_intermediate_size,
             activation_output_scale)) ||
        (gemm1_output != nullptr &&
         !PackDeviceRowMajorFp32ToNvfp4PerExpert(
             params.routed_up_scratch,
             params.launch_plan->padded_row_capacity(),
             params.routed_expert_intermediate_size,
             params.launch_plan->expert_first_token_offsets(),
             params.n_routed_experts,
             activation_output_scale,
             gemm1_output)) ||
        (gemm1_output == nullptr &&
         !LaunchQuantizeDequantizeRows(
             params.routed_up_scratch,
             params.launch_plan->total_num_padded_tokens(),
             params.launch_plan->padded_row_capacity(),
             params.routed_expert_intermediate_size))) {
      return false;
    }

    if (!(use_packed_fc2
              ? LaunchPlannedPackedInputMatVec(
                    *gemm1_output,
                    params.fc2_expert_activation_scales,
                    nullptr,
                    nullptr,
                    params.launch_plan,
                    packed_dispatch_rows,
                    selection_count,
                    params.routed_down_device,
                    params.hidden_size,
                    params.routed_gather_scratch)
              : RunLaunchPlannedNvfp4ExpertMatVec(
                    params.routed_up_scratch,
                    params.launch_plan,
                    selection_count,
                    params.routed_down_device,
                    params.hidden_size,
                    params.routed_gather_scratch))) {
      return false;
    }
  }

  if (!LaunchReduceSelectionOutputs(
          params.routed_gather_scratch,
          params.routing->selection_to_sorted(),
          params.launch_plan->sorted_to_permuted_indices(),
          params.selected_weights,
          params.token_count,
          params.top_k,
          params.hidden_size,
          params.output,
          params.routed_output)) {
    return false;
  }

  const bool should_prepare_shared_up_plan =
      shared_fc1_pack != nullptr && shared_fc1_pack->valid();
  const bool should_prepare_shared_down_plan =
      shared_fc2_pack != nullptr && shared_fc2_pack->valid();
  const auto shared_up_prepared_launch =
      should_prepare_shared_up_plan
          ? PrepareSharedContiguousLaunch(
                "shared_up_prefill",
                *shared_fc1_pack,
                params.token_count,
                params.shared_up,
                params.heuristic_cache)
          : std::nullopt;
  const auto shared_down_prepared_launch =
      should_prepare_shared_down_plan
          ? PrepareSharedContiguousLaunch(
                "shared_down_prefill",
                *shared_fc2_pack,
                params.token_count,
                params.shared_down,
                params.heuristic_cache)
          : std::nullopt;
  if (should_prepare_shared_up_plan) {
    AppendSharedContiguousTraceEntry(
        "shared_up_prefill",
        shared_up_prepared_launch.has_value(),
        shared_up_prepared_launch.has_value());
  }
  if (should_prepare_shared_down_plan) {
    AppendSharedContiguousTraceEntry(
        "shared_down_prefill",
        shared_down_prepared_launch.has_value(),
        shared_down_prepared_launch.has_value());
  }

  const bool use_shared_fp4_path =
      shared_up_prepared_launch.has_value() && shared_down_prepared_launch.has_value();
  if (use_shared_fp4_path) {
    Nvfp4PackOptions shared_pack_options;
    shared_pack_options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
    auto shared_fc1_source = DeviceTensorFp32::CreateView(
        {params.token_count, params.hidden_size},
        const_cast<float*>(params.normalized));
    auto shared_fc2_source = DeviceTensorFp32::CreateView(
        {params.token_count, params.shared_expert_intermediate_size},
        params.shared_up_scratch);
    if (shared_fc1_source == nullptr ||
        shared_fc2_source == nullptr ||
        !shared_fc1_pack->PackIntoPerRow(*shared_fc1_source, shared_pack_options) ||
        !LaunchContiguousFp4MatVec(
            *shared_fc1_pack,
            *shared_up_prepared_launch,
            params.shared_up,
            params.shared_up_scratch) ||
        !LaunchRelu2(params.shared_up_scratch, shared_intermediate_count) ||
        !shared_fc2_pack->PackIntoPerRow(*shared_fc2_source, shared_pack_options) ||
        !LaunchContiguousFp4MatVec(
            *shared_fc2_pack,
            *shared_down_prepared_launch,
            params.shared_down,
            params.routed_gather_scratch) ||
        !LaunchAccumulateSharedOutput(
            params.routed_gather_scratch,
            token_hidden_count,
            params.output,
            params.shared_output)) {
      return false;
    }
  } else if (!CheckCuda(cudaMemcpyAsync(
                 params.routed_gather_scratch,
                 params.normalized,
                 token_hidden_count * sizeof(float),
                 cudaMemcpyDeviceToDevice)) ||
             !LaunchQuantizeDequantizeRows(
                 params.routed_gather_scratch,
                 nullptr,
                 params.token_count,
                 params.hidden_size) ||
             !LaunchContiguousMatVec(
                 params.routed_gather_scratch,
                 params.token_count,
                 params.shared_up,
                 params.shared_up_scratch) ||
             !LaunchRelu2(params.shared_up_scratch, shared_intermediate_count) ||
             !LaunchQuantizeDequantizeRows(
                 params.shared_up_scratch,
                 nullptr,
                 params.token_count,
                 params.shared_expert_intermediate_size) ||
             !LaunchContiguousMatVec(
                 params.shared_up_scratch,
                 params.token_count,
                 params.shared_down,
                 params.routed_gather_scratch) ||
             !LaunchAccumulateSharedOutput(
                 params.routed_gather_scratch,
                 token_hidden_count,
                 params.output,
                 params.shared_output)) {
    return false;
  }

  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
