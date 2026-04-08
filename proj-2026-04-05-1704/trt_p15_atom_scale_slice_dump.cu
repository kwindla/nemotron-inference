#include <iostream>
#include <sstream>
#include <type_traits>

#include <cuda_bf16.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_zip.hpp>

using ArchTag = cutlass::arch::Sm120;
using OutputType = __nv_bfloat16;
constexpr bool IsSM120 = true;
using InputClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using ClusterShape = InputClusterShape;
using MmaTileShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
using ElementAct = cutlass::float_e2m1_t;
using ElementWeight = cutlass::float_e2m1_t;
using ElementD = OutputType;
using ElementCSafe = ElementD;
using ElementAccumulator = float;
using ElementSF = cutlass::float_ue4m3_t;
using ElementActBlockScaled = cutlass::nv_float4_t<ElementAct>;
using ElementWeightBlockScaled = cutlass::nv_float4_t<ElementWeight>;
using EpilogueScheduleSM90 = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using EpilogueScheduleSM120 = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueSchedule = std::conditional_t<IsSM120, EpilogueScheduleSM120, EpilogueScheduleSM90>;
using EpilogueTensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using EpilogueSubTile = cutlass::epilogue::collective::EpilogueTileAuto;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    EpilogueTensorOp,
    MmaTileShape,
    ClusterShape,
    EpilogueSubTile,
    ElementAccumulator,
    ElementAccumulator,
    ElementCSafe,
    LayoutC*,
    32,
    ElementD,
    LayoutD*,
    32,
    EpilogueSchedule>::CollectiveOp;
using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
using TensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    TensorOp,
    ElementWeightBlockScaled,
    LayoutA*,
    32,
    ElementActBlockScaled,
    LayoutB*,
    32,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    StageCountAutoCarveout,
    KernelSchedule>::CollectiveOp;
using TiledMma = typename CollectiveMainloop::TiledMma;
using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
using SmemLayoutSFA = typename CollectiveMainloop::SmemLayoutSFA;
using SmemLayoutSFB = typename CollectiveMainloop::SmemLayoutSFB;
using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
using SmemCopyAtomB = typename CollectiveMainloop::SmemCopyAtomB;
using SmemCopyAtomSFA = typename CollectiveMainloop::SmemCopyAtomSFA;
using SmemCopyAtomSFB = typename CollectiveMainloop::SmemCopyAtomSFB;

template <class Tensor>
void dump_tensor(std::string const& name, Tensor const& tensor) {
  std::cout << name << ".layout: " << tensor.layout() << "\n";
  std::cout << name << ".size=" << cute::size(tensor)
            << " cosize=" << cute::cosize(tensor.layout()) << "\n";
}

int main() {
  using namespace cute;

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(0);

  auto sA = make_tensor(make_smem_ptr(static_cast<ElementAct*>(nullptr)), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(static_cast<ElementAct*>(nullptr)), SmemLayoutB{});
  auto sSFA = make_tensor(make_smem_ptr(static_cast<ElementSF*>(nullptr)), SmemLayoutSFA{});
  auto sSFB = make_tensor(make_smem_ptr(static_cast<ElementSF*>(nullptr)), SmemLayoutSFB{});

  auto tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
  auto tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));
  auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
  auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

  auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(0);
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

  auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(0);
  auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto smem_tiled_copy_SFA = make_tiled_copy_impl(
      SmemCopyAtomSFA{}, CollectiveMainloop{}.get_layoutSFA_TV(tiled_mma),
      make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(0);
  auto tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

  auto smem_tiled_copy_SFB = make_tiled_copy_impl(
      SmemCopyAtomSFB{}, CollectiveMainloop{}.get_layoutSFB_TV(tiled_mma),
      make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(0);
  auto tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

  using RegTypeSF = typename std::remove_extent<typename TiledMma::MMA_Op::SFARegisters>::type;
  auto rSFA_from_copy_view = recast<RegTypeSF>(tCrSFA_copy_view);
  auto rSFB_from_copy_view = recast<RegTypeSF>(tCrSFB_copy_view);

  auto a_atom = tCrA(_, Int<0>{}, Int<0>{});
  auto b_atom = tCrB(_, Int<0>{}, Int<0>{});
  auto sfa_atom_from_recast = rSFA_from_copy_view(_, Int<0>{}, Int<0>{});
  auto sfb_atom_from_recast = rSFB_from_copy_view(_, Int<0>{}, Int<0>{});
  auto sfa_atom_raw = tCrSFA(_, Int<0>{}, Int<0>{});
  auto sfb_atom_raw = tCrSFB(_, Int<0>{}, Int<0>{});
  auto sfa_copy_atom_raw = tCrSFA_copy_view(_, Int<0>{}, Int<0>{});
  auto sfb_copy_atom_raw = tCrSFB_copy_view(_, Int<0>{}, Int<0>{});

  dump_tensor("a_atom", a_atom);
  dump_tensor("b_atom", b_atom);
  dump_tensor("tCrSFA", tCrSFA);
  dump_tensor("tCrSFB", tCrSFB);
  dump_tensor("sfa_atom_from_recast", sfa_atom_from_recast);
  dump_tensor("sfb_atom_from_recast", sfb_atom_from_recast);
  dump_tensor("sfa_atom_raw", sfa_atom_raw);
  dump_tensor("sfb_atom_raw", sfb_atom_raw);
  dump_tensor("sfa_copy_atom_raw", sfa_copy_atom_raw);
  dump_tensor("sfb_copy_atom_raw", sfb_copy_atom_raw);
  dump_tensor("filter_zeros(sfa_atom_from_recast)", filter_zeros(sfa_atom_from_recast));
  dump_tensor("filter_zeros(sfb_atom_from_recast)", filter_zeros(sfb_atom_from_recast));
  dump_tensor("filter_zeros(sfa_atom_raw)", filter_zeros(sfa_atom_raw));
  dump_tensor("filter_zeros(sfb_atom_raw)", filter_zeros(sfb_atom_raw));

  return 0;
}
