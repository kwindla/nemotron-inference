#include <iostream>

#include <cuda_bf16.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

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
constexpr int AlignmentAct = 128 / cutlass::sizeof_bits<ElementAct>::value;
constexpr int AlignmentWeight = 128 / cutlass::sizeof_bits<ElementWeight>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementAct>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using EpilogueScheduleSM90 = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using EpilogueScheduleSM120 = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueSchedule = std::conditional_t<IsSM120, EpilogueScheduleSM120, EpilogueScheduleSM90>;
using EpilogueElementC = ElementCSafe;
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
    EpilogueElementC,
    LayoutC*,
    AlignmentC,
    ElementD,
    LayoutD*,
    AlignmentD,
    EpilogueSchedule>::CollectiveOp;
using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
using TensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using MainloopElementAct = ElementActBlockScaled;
using MainloopElementWeight = ElementWeightBlockScaled;
using SwappedMainloopElementA = MainloopElementWeight;
using SwappedMainloopElementB = MainloopElementAct;
constexpr int SwappedAlignmentA = AlignmentWeight;
constexpr int SwappedAlignmentB = AlignmentAct;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    TensorOp,
    SwappedMainloopElementA,
    LayoutA*,
    SwappedAlignmentA,
    SwappedMainloopElementB,
    LayoutB*,
    SwappedAlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    StageCountAutoCarveout,
    KernelSchedule>::CollectiveOp;
using TiledMma = typename CollectiveMainloop::TiledMma;
using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
using SmemCopyAtomB = typename CollectiveMainloop::SmemCopyAtomB;

template <class Coord>
int coord_get0(Coord const& coord) {
  return static_cast<int>(cute::get<0>(coord));
}

template <class Coord>
int coord_get1(Coord const& coord) {
  return static_cast<int>(cute::get<1>(coord));
}

template <class Coord>
int coord_nested_get00(Coord const& coord) {
  return static_cast<int>(cute::get<0>(cute::get<0>(coord)));
}

template <class Coord>
int coord_nested_get01(Coord const& coord) {
  return static_cast<int>(cute::get<1>(cute::get<0>(coord)));
}

template <class Coord>
int coord_nested_get10(Coord const& coord) {
  return static_cast<int>(cute::get<0>(cute::get<1>(coord)));
}

int main() {
  using namespace cute;

  TiledMma tiled_mma;
  auto dense_a = make_identity_tensor(make_shape(Int<256>{}, Int<64>{}));
  auto dense_b = make_identity_tensor(make_shape(Int<128>{}, Int<64>{}));
  auto dense_c = make_identity_tensor(make_shape(Int<256>{}, Int<128>{}));

  using TileShape = MmaTileShape;
  constexpr int ScaleGranularityM = cute::size<0, 0>(typename CollectiveMainloop::InternalLayoutSFA{});
  constexpr int ScaleGranularityN = cute::size<0, 0>(typename CollectiveMainloop::InternalLayoutSFB{});
  constexpr int ScaleMsPerTile = cute::size<0>(TileShape{}) / ScaleGranularityM;
  constexpr int ScaleNsPerTile = cute::size<1>(TileShape{}) / ScaleGranularityN;
  auto cScaleAViewAsC = make_identity_tensor(make_shape(
      make_shape(Int<ScaleGranularityM>{}, Int<ScaleMsPerTile>{}),
      tuple_element_t<1, TileShape>{},
      Int<CollectiveMainloop::DispatchPolicy::Stages>{}));
  auto cScaleBViewAsC = make_identity_tensor(make_shape(
      tuple_element_t<0, TileShape>{},
      make_shape(Int<ScaleGranularityN>{}, Int<ScaleNsPerTile>{}),
      Int<CollectiveMainloop::DispatchPolicy::Stages>{}));

  std::cout << "META threads " << cute::size(tiled_mma) << "\n";
  std::cout << "META stages " << CollectiveMainloop::DispatchPolicy::Stages << "\n";
  std::cout << "META a_size " << cute::size(dense_a) << "\n";
  std::cout << "META b_size " << cute::size(dense_b) << "\n";
  std::cout << "META c_size " << cute::size(dense_c) << "\n";
  std::cout << "META a_copy_size 128\n";
  std::cout << "META b_copy_size 128\n";
  std::cout << "META c_store_size 128\n";
  std::cout << "META sfa_stage0_size 128\n";
  std::cout << "META sfb_stage0_size 128\n";

  for (int thread_idx = 0; thread_idx < static_cast<int>(cute::size(tiled_mma)); ++thread_idx) {
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto part_a = thread_mma.partition_A(dense_a);
    auto part_b = thread_mma.partition_B(dense_b);
    auto part_c = thread_mma.partition_C(dense_c);

    auto smem_tiled_copy_a = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(thread_idx);
    auto copy_view_a = smem_thr_copy_a.retile_D(part_a);

    auto smem_tiled_copy_b = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(thread_idx);
    auto copy_view_b = smem_thr_copy_b.retile_D(part_b);

    auto tCcScaleAViewAsC = thread_mma.partition_C(cScaleAViewAsC);
    auto tCcScaleBViewAsC = thread_mma.partition_C(cScaleBViewAsC);
    auto tCcScaleAStage0 = tCcScaleAViewAsC(_, _, _, _0{});
    auto tCcScaleBStage0 = tCcScaleBViewAsC(_, _, _, _0{});

    std::cout << "THREAD " << thread_idx << "\n";
    for (int i = 0; i < static_cast<int>(cute::size(copy_view_a)); ++i) {
      auto coord = copy_view_a(i);
      std::cout << "A " << i << " " << coord_get0(coord) << " " << coord_get1(coord) << "\n";
    }
    for (int i = 0; i < static_cast<int>(cute::size(copy_view_b)); ++i) {
      auto coord = copy_view_b(i);
      std::cout << "B " << i << " " << coord_get0(coord) << " " << coord_get1(coord) << "\n";
    }
    for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
      auto coord = part_c(i);
      std::cout << "C " << i << " " << coord_get0(coord) << " " << coord_get1(coord) << "\n";
    }
    for (int i = 0; i < static_cast<int>(cute::size(tCcScaleAStage0)); ++i) {
      auto coord = tCcScaleAStage0(i);
      std::cout << "SFA " << i << " "
                << coord_nested_get00(coord) << " "
                << coord_nested_get01(coord) << " "
                << coord_get1(coord) << "\n";
    }
    for (int i = 0; i < static_cast<int>(cute::size(tCcScaleBStage0)); ++i) {
      auto coord = tCcScaleBStage0(i);
      std::cout << "SFB " << i << " "
                << coord_get0(coord) << " "
                << coord_nested_get10(coord) << "\n";
    }
  }

  return 0;
}
