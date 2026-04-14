#include <cuda_bf16.h>
#include <iostream>
#include <cstdio>
#include <tuple>
#include <type_traits>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/tensor_impl.hpp>

namespace {

namespace cute = ::cute;

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

using ArchTag = cutlass::arch::Sm120;
using ElementAct = cutlass::float_e2m1_t;
using ElementWeight = cutlass::float_e2m1_t;
using ElementAccumulator = float;
using ElementD = __nv_bfloat16;
using MainloopElementA = cutlass::nv_float4_t<ElementAct>;
using MainloopElementB = cutlass::nv_float4_t<ElementWeight>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementAct>::value;
constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementWeight>::value;
constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementD,
    LayoutC*,
    kAlignmentC,
    ElementD,
    LayoutD*,
    kAlignmentD,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassBlockScaledTensorOp,
    MainloopElementA,
    LayoutA*,
    kAlignmentA,
    MainloopElementB,
    LayoutB*,
    kAlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using TiledMma = typename CollectiveMainloop::TiledMma;
using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;

constexpr int kTrackedTidCount = 4;
constexpr int kTrackedTids[kTrackedTidCount] = {0, 1, 128, 129};

}  // namespace

int main() {
  TiledMma tiled_mma;

  auto dense_a = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(MmaTileShape{}),
          cute::Int<128>{}));
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::Int<128>{},
          cute::Int<128>{}));
  auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});

  std::printf(
      "nano_p1_reference_a_operand_probe: dense reference A copy-view contract for the TRT-LLM-equivalent P1 mainloop\n");
  std::printf(
      "nano_p1_reference_a_operand_probe: MmaTileShape=(%d,%d,%d) threads=%d stages=%d\n",
      static_cast<int>(cute::size<0>(MmaTileShape{})),
      static_cast<int>(cute::size<1>(MmaTileShape{})),
      static_cast<int>(cute::size<2>(MmaTileShape{})),
      static_cast<int>(cute::size(tiled_mma)),
      static_cast<int>(CollectiveMainloop::DispatchPolicy::Stages));

  bool printed_layouts = false;
  for (int tid_slot = 0; tid_slot < kTrackedTidCount; ++tid_slot) {
    const int tid = kTrackedTids[tid_slot];
    auto thread_mma = tiled_mma.get_thread_slice(tid);
    auto part_a = thread_mma.partition_B(dense_a);
    auto smem_tiled_copy_a = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(tid);
    auto copy_view_a = smem_thr_copy_a.retile_D(part_a);
    auto part_c = thread_mma.partition_C(dense_c);

    constexpr int kCopyTiles = cute::size<1>(decltype(copy_view_a){});
    constexpr int kKBlocks = cute::size<2>(decltype(copy_view_a){});

    if (!printed_layouts) {
      printed_layouts = true;
      std::printf(
          "nano_p1_reference_a_operand_probe: copy_view_a sizes=(%d,%d,%d) part_c sizes=(%d,%d,%d)\n",
          static_cast<int>(cute::size<0>(copy_view_a)),
          static_cast<int>(cute::size<1>(copy_view_a)),
          static_cast<int>(cute::size<2>(copy_view_a)),
          static_cast<int>(cute::size<0>(part_c)),
          static_cast<int>(cute::size<1>(part_c)),
          static_cast<int>(cute::size<2>(part_c)));
      std::cout << "nano_p1_reference_a_operand_probe: copy_view_a.layout=" << copy_view_a.layout() << "\n";
      std::cout << "nano_p1_reference_a_operand_probe: part_c.layout=" << part_c.layout() << "\n";
    }

    std::printf("nano_p1_reference_a_operand_probe: ---- tid=%d ----\n", tid);
    for (int copy_tile = 0; copy_tile < kCopyTiles; ++copy_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        auto coords = copy_view_a(cute::_, copy_tile, k_block);
        int part_token_row0 = -1;
        int part_output_col0 = -1;
        if constexpr (cute::size<2>(decltype(part_c){}) >= 8) {
          auto c_atom_coords = part_c(cute::_, 0, copy_tile);
          auto c0 = c_atom_coords(0);
          part_output_col0 = CoordGet0(c0);
          part_token_row0 = CoordGet1(c0);
        }
        std::printf(
            "  copy_tile=%d k_block=%d part_token_row0=%d part_output_col0=%d anchors=",
            copy_tile,
            k_block,
            part_token_row0,
            part_output_col0);
        constexpr int kCoordCount = cute::size<0>(decltype(copy_view_a){});
        constexpr int kStride = kCoordCount >= 8 ? (kCoordCount / 8) : 1;
        constexpr int kPrintCoords = kCoordCount < 8 ? kCoordCount : 8;
        for (int i = 0; i < kPrintCoords; ++i) {
          auto coord = coords(i * kStride);
          const int dense_row = CoordGet0(coord);
          const int dense_byte = CoordGet1(coord);
          const int stage0_offset = static_cast<int>(stage0_A(dense_row, dense_byte * 2));
          std::printf(
              "%s(%d,%d@%d)",
              i == 0 ? "" : " ",
              dense_row,
              dense_byte,
              stage0_offset);
        }
        std::printf("\n");
      }
    }
  }

  return 0;
}
