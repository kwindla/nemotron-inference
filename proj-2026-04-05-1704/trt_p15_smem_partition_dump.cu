#include <iostream>

#include <cuda_bf16.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

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
    AlignmentC,
    ElementD,
    LayoutD*,
    AlignmentD,
    EpilogueSchedule>::CollectiveOp;
using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
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
using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;

int main() {
  using namespace cute;

  TiledMma tiled_mma;
  auto sA = make_tensor(make_smem_ptr(static_cast<ElementAct*>(nullptr)), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(static_cast<ElementWeight*>(nullptr)), SmemLayoutB{});
  using SwizzleA = get_swizzle_t<decltype(sA)>;
  using SwizzleB = get_swizzle_t<decltype(sB)>;
  auto cA = [&] {
    if constexpr (SwizzleA::num_bits == 0) {
      return make_coord_tensor(sA.layout());
    } else {
      auto layout = composition(recast_layout<uint8_t, ElementAct>(SwizzleA{}), Int<0>{}, sA.layout());
      return make_tensor(make_inttuple_iter(coprofile(sA.layout())), layout);
    }
  }();
  auto cB = [&] {
    if constexpr (SwizzleB::num_bits == 0) {
      return make_coord_tensor(sB.layout());
    } else {
      auto layout = composition(recast_layout<uint8_t, ElementWeight>(SwizzleB{}), Int<0>{}, sB.layout());
      return make_tensor(make_inttuple_iter(coprofile(sB.layout())), layout);
    }
  }();

  std::cout << "META threads " << size(tiled_mma) << "\n";

  for (int thread_idx = 0; thread_idx < static_cast<int>(size(tiled_mma)); ++thread_idx) {
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    auto smem_tiled_copy_a = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(thread_idx);
    auto tCsA = smem_thr_copy_a.partition_S(cA);
    auto tCsAStage0 = tCsA(_, _, _, _0{});

    auto smem_tiled_copy_b = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(thread_idx);
    auto tCsB = smem_thr_copy_b.partition_S(cB);
    auto tCsBStage0 = tCsB(_, _, _, _0{});

    if (thread_idx == 0) {
      std::cout << "META tcsa_stage0_size " << size(tCsAStage0) << "\n";
      std::cout << "META tcsb_stage0_size " << size(tCsBStage0) << "\n";
    }

    std::cout << "THREAD " << thread_idx << "\n";
    for (int i = 0; i < static_cast<int>(size(tCsAStage0)); ++i) {
      auto physical = tCsAStage0(i);
      auto coord = sA.layout().get_hier_coord(physical);
      std::cout << "TCSA " << i << " ";
      cute::print(coord);
      std::cout << "\n";
    }
    for (int i = 0; i < static_cast<int>(size(tCsBStage0)); ++i) {
      auto physical = tCsBStage0(i);
      auto coord = sB.layout().get_hier_coord(physical);
      std::cout << "TCSB " << i << " ";
      cute::print(coord);
      std::cout << "\n";
    }
  }

  return 0;
}
