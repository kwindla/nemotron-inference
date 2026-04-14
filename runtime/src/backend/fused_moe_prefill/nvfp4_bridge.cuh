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

CUTE_HOST_DEVICE constexpr int NanoP1PermuteBSourceRow(int row) {
  const int row_block = row & ~31;
  const int row_in_block = row & 31;
  if (row_in_block < 8) {
    return row;
  }
  if (row_in_block < 16) {
    return row_block + row_in_block + 8;
  }
  if (row_in_block < 24) {
    return row_block + row_in_block - 8;
  }
  return row;
}

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
