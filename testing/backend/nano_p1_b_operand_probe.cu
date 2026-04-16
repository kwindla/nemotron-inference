#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/arch/barrier.h>
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

using TracedP5PermTileN =
    cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_2>, cute::Stride<cute::_1, cute::_16, cute::_8>>;

using NanoP1ElementAct = cutlass::nv_float4_t<nvfp4_cute::ElementAB>;
using NanoP1ElementWeight = cutlass::nv_float4_t<nvfp4_cute::ElementAB>;
using NanoP1ElementAccum = float;
constexpr int NanoP1PipelineStages = 4;
constexpr int NanoP1ThreadsPerCta = 256;
using NanoP1MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using NanoP1ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using NanoP1MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    nvfp4_cute::ElementAB,
    nvfp4_cute::ElementAB,
    NanoP1ElementAccum,
    nvfp4_cute::ElementSFCompute,
    nvfp4_cute::kScaleVecSize>;
using NanoP1MmaAtom = cute::MMA_Atom<NanoP1MmaOp>;
using NanoP1AtomLayoutMNK = cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>;
using NanoP1ValLayoutMNK = cute::Tile<cute::Int<128>, TracedP5PermTileN, cute::Int<64>>;
using NanoP1TiledMma = cute::TiledMMA<NanoP1MmaAtom, NanoP1AtomLayoutMNK, NanoP1ValLayoutMNK>;
using NanoP1SmemLayoutAtomB = cute::UMMA::Layout_K_SW64_Atom<typename NanoP1TiledMma::ValTypeB>;
using NanoP1ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<nvfp4_cute::kScaleVecSize>;
using BRegister = std::remove_extent_t<typename NanoP1MmaOp::BRegisters>;
using NanoP1AccumLayout = decltype(
    cute::partition_fragment_C(
        NanoP1TiledMma{},
        cute::make_shape(
            cute::size<0>(NanoP1MmaTileShape{}),
            cute::size<1>(NanoP1MmaTileShape{})))
        .layout());
using NanoP1SmemLayoutB = decltype(cute::tile_to_shape(
    NanoP1SmemLayoutAtomB{},
    cute::make_shape(
        cute::size<1>(NanoP1MmaTileShape{}) * cute::size<1>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{}),
    cute::Step<cute::_2, cute::_1, cute::_3>{}));
using NanoP1SmemLayoutSFB = decltype(NanoP1ScaleConfig::tile_atom_to_shape_SFB(
    cute::make_shape(
        cute::size<0>(NanoP1MmaTileShape{}) * cute::size<0>(NanoP1ClusterShape{}),
        cute::size<1>(NanoP1MmaTileShape{}) * cute::size<1>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{})));
using NanoP1SmemAllocB = typename NanoP1TiledMma::ValTypeB;
using NanoP1SmemAllocSFB = nvfp4_cute::ElementSFCompute;
using NanoP1SmemCopyAtomB = cute::Copy_Atom<
    decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_B<
             NanoP1ElementAct,
             NanoP1ElementWeight,
             false>()),
    NanoP1SmemAllocB>;
using NanoP1SmemCopyAtomBAlt = cute::Copy_Atom<
    decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_B<
             NanoP1ElementAct,
             NanoP1ElementWeight,
             true>()),
    NanoP1SmemAllocB>;
using NanoP1SmemCopyAtomSFB = cute::Copy_Atom<
    cute::UniversalCopy<nvfp4_cute::ElementSFCompute>,
    nvfp4_cute::ElementSFCompute>;
constexpr int kNanoP1SwizzledBElems = cute::size(cute::take<0, 2>(NanoP1SmemLayoutB{}));
constexpr int kNanoP1ScaleStageElemsB = cute::cosize(cute::take<0, 2>(NanoP1SmemLayoutSFB{}));

static_assert(cute::size(NanoP1TiledMma{}) == NanoP1ThreadsPerCta);
static_assert(cute::cosize_v<NanoP1SmemLayoutB> == 65536);
static_assert(cute::cosize_v<NanoP1SmemLayoutSFB> == 4096);
static_assert(cute::size<1>(NanoP1AccumLayout{}) == 2);
static_assert(cute::size<2>(NanoP1AccumLayout{}) == 8);

constexpr int kNanoP1TileRows = cute::size<0>(NanoP1MmaTileShape{});
constexpr int kNanoP1MacroTileBytes = cute::size<2>(NanoP1MmaTileShape{}) / 2;

enum class SourceRowMode : int {
  kIdentity = -1,
  kRowOnlyPermute = 0,
  kPositionMap = 1,
  kPositionMapOddPlus4 = 2,
};

enum class InputStageMode : int {
  kPackedBytes = 0,
  kUnpackNibbles = 1,
  kUnpackNibblesSwapHalves = 2,
};

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

CUTE_HOST_DEVICE constexpr int NanoP1SourceRowForStageOffset(int stage_offset) {
  const int band_offset = stage_offset & 0x1ff;
  const int window = band_offset & ~31;
  const int x = (band_offset & 31) >> 1;
  const int permuted =
      ((x & 0x1) << 4) |
      (((x >> 1) & 0x1) << 3) |
      (((x >> 2) & 0x1) << 5) |
      (((x >> 3) & 0x1) << 6);
  if (window == 0) {
    return permuted;
  }
  if (window == 128) {
    const int permuted_rot =
        (((x ^ 0x8) & 0x1) << 4) |
        ((((x ^ 0x8) >> 1) & 0x1) << 3) |
        ((((x ^ 0x8) >> 2) & 0x1) << 5) |
        ((((x ^ 0x8) >> 3) & 0x1) << 6);
    return 2 + permuted_rot;
  }
  if (window == 288) {
    return 4 + permuted;
  }
  if (window == 416) {
    const int permuted_rot =
        (((x ^ 0x8) & 0x1) << 4) |
        ((((x ^ 0x8) >> 1) & 0x1) << 3) |
        ((((x ^ 0x8) >> 2) & 0x1) << 5) |
        ((((x ^ 0x8) >> 3) & 0x1) << 6);
    return 6 + permuted_rot;
  }
  return -1;
}

constexpr char const* SourceRowModeName(SourceRowMode mode) {
  switch (mode) {
    case SourceRowMode::kIdentity:
      return "identity";
    case SourceRowMode::kRowOnlyPermute:
      return "row_only_permute";
    case SourceRowMode::kPositionMap:
      return "position_map";
    case SourceRowMode::kPositionMapOddPlus4:
      return "position_map_odd_plus4";
  }
  return "unknown";
}

constexpr char const* InputStageModeName(InputStageMode mode) {
  switch (mode) {
    case InputStageMode::kPackedBytes:
      return "packed_bytes";
    case InputStageMode::kUnpackNibbles:
      return "unpack_nibbles";
    case InputStageMode::kUnpackNibblesSwapHalves:
      return "unpack_nibbles_swap_halves";
  }
  return "unknown";
}

CUTE_HOST_DEVICE constexpr int DecodeSourceRow(
    SourceRowMode mode,
    int row,
    int stage_offset) {
  switch (mode) {
    case SourceRowMode::kIdentity:
      return row;
    case SourceRowMode::kRowOnlyPermute:
      return NanoP1PermuteBSourceRow(row);
    case SourceRowMode::kPositionMap:
      return NanoP1SourceRowForStageOffset(stage_offset);
    case SourceRowMode::kPositionMapOddPlus4: {
      int const base = NanoP1SourceRowForStageOffset(stage_offset);
      if (base < 0) {
        return base;
      }
      if ((stage_offset & 0x1) == 0) {
        return base;
      }
      int const biased = base + 4;
      return biased < kNanoP1TileRows ? biased : -1;
    }
  }
  return -1;
}

template <class SFBTensor, class AtomT, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto NanoP1ThrfrgSFB(
    SFBTensor&& sfbtensor,
    cute::TiledMMA<AtomT, TiledThr, TiledPerm>& mma) {
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

template <class CoordTensor, class F>
CUTE_HOST_DEVICE void ForEachCopyViewCoord(CoordTensor const& coord_tensor, F&& f) {
  if constexpr (std::remove_cvref_t<CoordTensor>::rank == 1) {
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      f(coord_tensor(cute::make_coord(i)), i);
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 2) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        f(coord_tensor(cute::make_coord(i, j)), physical++);
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 3) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          f(coord_tensor(cute::make_coord(i, j, k)), physical++);
        }
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 4) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          for (int l = 0; l < cute::size<3>(coord_tensor); ++l) {
            f(coord_tensor(cute::make_coord(i, j, k, l)), physical++);
          }
        }
      }
    }
  } else {
    static_assert(
        std::remove_cvref_t<CoordTensor>::rank <= 4,
        "ForEachCopyViewCoord only supports rank <= 4");
  }
}

template <class CoordTensor, class ValueTensor, class F>
CUTE_HOST_DEVICE void ForEachCopyViewCoordValue(
    CoordTensor const& coord_tensor,
    ValueTensor const& value_tensor,
    F&& f) {
  static_assert(
      std::remove_cvref_t<CoordTensor>::rank == std::remove_cvref_t<ValueTensor>::rank,
      "coord/value tensor rank mismatch");
  if constexpr (std::remove_cvref_t<CoordTensor>::rank == 1) {
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      auto const coord = coord_tensor(cute::make_coord(i));
      auto const value = value_tensor(cute::make_coord(i));
      f(coord, value, i);
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 2) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        auto const ij = cute::make_coord(i, j);
        f(coord_tensor(ij), value_tensor(ij), physical++);
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 3) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          auto const ijk = cute::make_coord(i, j, k);
          f(coord_tensor(ijk), value_tensor(ijk), physical++);
        }
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 4) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          for (int l = 0; l < cute::size<3>(coord_tensor); ++l) {
            auto const ijkl = cute::make_coord(i, j, k, l);
            f(coord_tensor(ijkl), value_tensor(ijkl), physical++);
          }
        }
      }
    }
  } else {
    static_assert(
        std::remove_cvref_t<CoordTensor>::rank <= 4,
        "ForEachCopyViewCoordValue only supports rank <= 4");
  }
}

template <class ValueTensor, class F>
CUTE_HOST_DEVICE void ForEachTensorValue(ValueTensor const& value_tensor, F&& f) {
  if constexpr (std::remove_cvref_t<ValueTensor>::rank == 1) {
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      f(value_tensor(cute::make_coord(i)), i);
    }
  } else if constexpr (std::remove_cvref_t<ValueTensor>::rank == 2) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(value_tensor); ++j) {
        f(value_tensor(cute::make_coord(i, j)), physical++);
      }
    }
  } else if constexpr (std::remove_cvref_t<ValueTensor>::rank == 3) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(value_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(value_tensor); ++k) {
          f(value_tensor(cute::make_coord(i, j, k)), physical++);
        }
      }
    }
  } else if constexpr (std::remove_cvref_t<ValueTensor>::rank == 4) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(value_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(value_tensor); ++k) {
          for (int l = 0; l < cute::size<3>(value_tensor); ++l) {
            f(value_tensor(cute::make_coord(i, j, k, l)), physical++);
          }
        }
      }
    }
  } else {
    static_assert(
        std::remove_cvref_t<ValueTensor>::rank <= 4,
        "ForEachTensorValue only supports rank <= 4");
  }
}

template <class PackedValue>
__device__ __forceinline__ std::uint8_t LoadPackedValueByte(PackedValue const& value) {
  if constexpr (requires { value.raw(); }) {
    return static_cast<std::uint8_t>(value.raw());
  } else if constexpr (requires { value.get(); }) {
    auto const unpacked = value.get();
    if constexpr (requires { unpacked.raw(); }) {
      return static_cast<std::uint8_t>(unpacked.raw());
    } else if constexpr (requires { unpacked.storage; }) {
      return static_cast<std::uint8_t>(unpacked.storage);
    } else {
      return static_cast<std::uint8_t>(unpacked);
    }
  } else if constexpr (requires { value.storage; }) {
    return static_cast<std::uint8_t>(value.storage);
  } else {
    return static_cast<std::uint8_t>(value);
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void StoreScaleTensorByte(
    ScaleTensor& scale_tensor,
    int row,
    int scale_col,
    std::uint8_t value) {
  auto elem = scale_tensor(row, scale_col);
  if constexpr (requires { elem.storage; }) {
    scale_tensor(row, scale_col).storage = value;
  } else {
    scale_tensor(row, scale_col) = value;
  }
}

template <class ScaleTensor>
__device__ __forceinline__ std::uint8_t LoadScaleTensorByte(
    ScaleTensor const& scale_tensor,
    int row,
    int scale_col) {
  auto elem = scale_tensor(row, scale_col);
  if constexpr (requires { elem.storage; }) {
    return static_cast<std::uint8_t>(elem.storage);
  } else {
    return static_cast<std::uint8_t>(elem);
  }
}

template <class ScaleAtomTensor>
CUTE_HOST_DEVICE std::uint32_t PackScaleFragmentWordLocal(ScaleAtomTensor const& scale_atom) {
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

enum class ProbeTagPattern {
  kSourceRow,
  kByteIndex,
};

enum class ProbeScaleTagPattern {
  kLogicalRow,
  kLogicalCol,
};

CUTE_HOST_DEVICE constexpr int WrapStageByteSlot(int slot) {
  int wrapped = slot % kNanoP1SwizzledBElems;
  if (wrapped < 0) {
    wrapped += kNanoP1SwizzledBElems;
  }
  return wrapped;
}

template <ProbeTagPattern kPattern, class Stage0Layout>
__device__ __forceinline__ void FillProbeBStage0(
    std::uint8_t* swizzled_b_bytes,
    Stage0Layout const& stage0_B,
    SourceRowMode source_row_mode,
    int source_row_bit = -1,
    int stage_slot_bit = -1,
    int stage_slot_low_byte = 0,
    bool synthetic_linear_row8 = false,
    int stage_slot_bias = 0) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kNanoP1TileRows * kNanoP1MacroTileBytes;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kNanoP1MacroTileBytes;
    const int byte_index = linear % kNanoP1MacroTileBytes;
    const auto elem_offset = stage0_B(row, byte_index * 2);
    const int stage_offset = static_cast<int>(elem_offset);
    const int source_row = DecodeSourceRow(source_row_mode, row, stage_offset);
    std::uint8_t value = 0u;
    if constexpr (kPattern == ProbeTagPattern::kSourceRow) {
      if (stage_slot_low_byte != 0) {
        value = static_cast<std::uint8_t>((stage_offset / 2) & 0xff);
      } else if (stage_slot_bit >= 0) {
        value = static_cast<std::uint8_t>(((stage_offset / 2) >> stage_slot_bit) & 0x1);
      } else if (source_row_bit >= 0) {
        value = static_cast<std::uint8_t>((source_row >> source_row_bit) & 0x1);
      } else {
        value = static_cast<std::uint8_t>(source_row);
      }
    } else {
      if (synthetic_linear_row8 && source_row >= 0) {
        value = static_cast<std::uint8_t>((source_row * 8 + byte_index) & 0xff);
      } else {
        value = static_cast<std::uint8_t>(byte_index);
      }
    }
    const int physical_stage_slot = WrapStageByteSlot(stage_offset / 2 + stage_slot_bias);
    swizzled_b_bytes[physical_stage_slot] = value;
  }
  __syncthreads();
}

template <ProbeScaleTagPattern kPattern, class ScaleTensor>
__device__ __forceinline__ void FillProbeSFBStage0(ScaleTensor& scale_tensor) {
  constexpr int kRows = cute::size<0>(ScaleTensor{});
  constexpr int kCols = cute::size<1>(ScaleTensor{});
  for (int linear = static_cast<int>(threadIdx.x); linear < kRows * kCols; linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kCols;
    const int col = linear % kCols;
    const std::uint8_t value = kPattern == ProbeScaleTagPattern::kLogicalRow
        ? static_cast<std::uint8_t>(row)
        : static_cast<std::uint8_t>(col);
    StoreScaleTensorByte(scale_tensor, row, col, value);
  }
  __syncthreads();
}

constexpr int kTrackedTidCount = 8;
constexpr int kDefaultTrackedTidCount = 4;
constexpr int kDefaultTrackedTids[kDefaultTrackedTidCount] = {0, 1, 128, 129};
constexpr int kCopyViewSignatureLen = 8;
constexpr int kCopyViewRawSignatureLen = 32;

template <std::size_t N>
void PrintIntArray(char const* label, int const (&values)[N], int count) {
  std::printf("%s=[", label);
  for (int idx = 0; idx < count; ++idx) {
    std::printf("%s%d", idx == 0 ? "" : ",", values[idx]);
  }
  std::printf("]");
}

struct ProbeConfig {
  int tracked_tid_count = kDefaultTrackedTidCount;
  int tracked_tids[kTrackedTidCount] = {0, 1, 128, 129, -1, -1, -1, -1};
  SourceRowMode source_row_mode = SourceRowMode::kRowOnlyPermute;
  int dump_slot_map = 0;
  int source_row_bit = -1;
  int stage_slot_bit = -1;
  int stage_slot_low_byte = 0;
  int use_input_file = 0;
  InputStageMode input_stage_mode = InputStageMode::kPackedBytes;
  int input_rows = kNanoP1TileRows;
  int packed_row_bytes = kNanoP1MacroTileBytes;
  int packed_byte_offset = 0;
  int odd_stage_offset_xor32 = 0;
  int synthetic_linear_row8 = 0;
  int stage_slot_bias = 0;
  int live_consumer_order = 0;
};

template <class Stage0Tensor, class Stage0Layout>
__device__ __forceinline__ void FillProbeBStage0FromInput(
    Stage0Tensor& sB_stage0,
    std::uint8_t* swizzled_b_bytes,
    Stage0Layout const& stage0_B,
    ProbeConfig const& config,
    std::uint8_t const* input_bytes) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kNanoP1TileRows * kNanoP1MacroTileBytes;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kNanoP1MacroTileBytes;
    const int byte_index = linear % kNanoP1MacroTileBytes;
    const auto elem_offset = stage0_B(row, byte_index * 2);
    const int stage_offset = static_cast<int>(elem_offset);
    const int source_row = DecodeSourceRow(config.source_row_mode, row, stage_offset);
    std::uint8_t value = 0u;
    if (source_row >= 0 &&
        source_row < config.input_rows &&
        input_bytes != nullptr) {
      int packed_byte_index = config.packed_byte_offset + byte_index;
      if (config.odd_stage_offset_xor32 != 0 && (stage_offset & 0x1) != 0) {
        packed_byte_index ^= 32;
      }
      if (packed_byte_index >= 0 && packed_byte_index < config.packed_row_bytes) {
        const std::size_t src_offset =
            static_cast<std::size_t>(source_row) * static_cast<std::size_t>(config.packed_row_bytes) +
            static_cast<std::size_t>(packed_byte_index);
        value = input_bytes[src_offset];
      }
    }
    if (config.input_stage_mode == InputStageMode::kPackedBytes) {
      const int physical_stage_slot =
          WrapStageByteSlot(stage_offset / 2 + config.stage_slot_bias);
      swizzled_b_bytes[physical_stage_slot] = value;
      continue;
    }
    const std::uint8_t low = static_cast<std::uint8_t>(value & 0x0fu);
    const std::uint8_t high = static_cast<std::uint8_t>((value >> 4u) & 0x0fu);
    const bool swap_halves = config.input_stage_mode == InputStageMode::kUnpackNibblesSwapHalves;
    sB_stage0(row, byte_index * 2) = NanoP1SmemAllocB(swap_halves ? high : low);
    sB_stage0(row, byte_index * 2 + 1) = NanoP1SmemAllocB(swap_halves ? low : high);
  }
  __syncthreads();
}

template <class CopyOp, class SrcTensor, class DstTensor>
__device__ __forceinline__ void CopyProbeBPerKBlock(
    CopyOp const& copy_op,
    SrcTensor const& src,
    DstTensor& dst) {
  constexpr int kKBlocks = cute::size<2>(DstTensor{});
  for (int k_block = 0; k_block < kKBlocks; ++k_block) {
    cute::copy(
        copy_op,
        src(cute::_, cute::_, k_block, cute::Int<0>{}),
        dst(cute::_, cute::_, k_block));
  }
}

struct ProbeEntry {
  int tid = -1;
  int atom_n = -1;
  int lane_n = -1;
  int n_tile = -1;
  int mf_inner = -1;
  int mf_outer = -1;
  int k_block = -1;
  int part_output_col0 = -1;
  int part_token_row0 = -1;
  int local_row0 = -1;
  int local_col0 = -1;
  int local_row1 = -1;
  int local_col1 = -1;
  int staged_source_row0 = -1;
  int staged_source_row1 = -1;
  int stage0_offset0 = -1;
  int stage0_offset1 = -1;
  int consumed_source_row0 = -1;
  int consumed_source_row1 = -1;
  int copy_view_row_mismatches = -1;
  int first_copy_mismatch_elem = -1;
  int first_copy_mismatch_row = -1;
  int first_copy_mismatch_col = -1;
  int first_copy_mismatch_stage_offset = -1;
  int first_copy_expected_source_row = -1;
  int first_copy_consumed_source_row = -1;
  int copy_view_byte_mismatches = -1;
  int first_copy_byte_mismatch_elem = -1;
  int first_copy_byte_mismatch_row = -1;
  int first_copy_byte_mismatch_col = -1;
  int first_copy_byte_mismatch_stage_offset = -1;
  int first_copy_expected_byte_index = -1;
  int first_copy_consumed_byte_index = -1;
  int copy_view_sig_count = 0;
  int copy_view_sig_stage_offsets[kCopyViewSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int copy_view_sig_stage_bytes[kCopyViewSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int copy_view_unique_stage_slot_count = 0;
  int copy_view_stage_byte_slots[kCopyViewSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int copy_view_expected_byte_indices[kCopyViewSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int copy_view_raw_signature_count = 0;
  int copy_view_raw_expected_byte_indices[kCopyViewRawSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int copy_view_raw_consumed_bytes[kCopyViewRawSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int copy_view_raw_stage_offsets[kCopyViewRawSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int frag_raw_signature_count = 0;
  int frag_raw_consumed_bytes[kCopyViewRawSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int consumed_byte_index0 = -1;
  int consumed_byte_index1 = -1;
  int copy_view_consumed_sig_count = 0;
  int copy_view_consumed_sig_stage_offsets[kCopyViewSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int copy_view_consumed_sig_bytes[kCopyViewSignatureLen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int scale_local_row0 = -1;
  int scale_local_col0 = -1;
  int scale_stage0_offset0 = -1;
  std::uint32_t consumed_scale_row_word0 = 0u;
  std::uint32_t consumed_scale_col_word0 = 0u;
  std::uint32_t source_row_reg0_pre = 0u;
  std::uint32_t source_row_reg1_pre = 0u;
  std::uint32_t byte_tag_reg0_pre = 0u;
  std::uint32_t byte_tag_reg1_pre = 0u;
  std::uint32_t byte_tag_reg0_pre_alt = 0u;
  std::uint32_t byte_tag_reg1_pre_alt = 0u;
  std::uint32_t byte_tag_reg0_post = 0u;
  std::uint32_t byte_tag_reg1_post = 0u;
  std::uint32_t actual_scale_word0 = 0u;
};

struct SurfaceStats {
  int invalid_count = 0;
  int first_invalid_row = -1;
  int first_invalid_byte = -1;
  int first_invalid_offset = -1;
  int band_counts[16] = {0};
  int consumed_invalid_count = 0;
  int first_consumed_invalid_tid = -1;
  int first_consumed_invalid_n_tile = -1;
  int first_consumed_invalid_k_block = -1;
  int first_consumed_invalid_elem = -1;
  int first_consumed_invalid_row = -1;
  int first_consumed_invalid_col = -1;
  int first_consumed_invalid_offset = -1;
  int consumed_mod512_counts[512] = {0};
  int consumed_slot_source_rows[2048] = {0};
  int consumed_slot_mod512[2048] = {0};
  int consumed_slot_source_row_conflicts = 0;
  int first_slot_source_row_conflict_slot = -1;
  int first_slot_source_row_conflict_prev = -1;
  int first_slot_source_row_conflict_new = -1;
  int slot_write_counts[2048] = {0};
  int slot_invalid_write_counts[2048] = {0};
  int consumed_slot_ref_counts[2048] = {0};
};

constexpr int kProbeEntryCount =
    kTrackedTidCount * 8 * 2;  // max tracked tids * 8 n tiles * 2 k blocks.

ProbeConfig LoadProbeConfig() {
  ProbeConfig config{};
  if (char const* source_row_mode_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_SOURCE_ROW_MODE");
      source_row_mode_env != nullptr &&
      source_row_mode_env[0] != '\0') {
    if (std::strcmp(source_row_mode_env, "identity") == 0) {
      config.source_row_mode = SourceRowMode::kIdentity;
    } else if (std::strcmp(source_row_mode_env, "row_only_permute") == 0 ||
        std::strcmp(source_row_mode_env, "row_only") == 0) {
      config.source_row_mode = SourceRowMode::kRowOnlyPermute;
    } else if (std::strcmp(source_row_mode_env, "position_map") == 0) {
      config.source_row_mode = SourceRowMode::kPositionMap;
    } else if (std::strcmp(source_row_mode_env, "position_map_odd_plus4") == 0 ||
               std::strcmp(source_row_mode_env, "position_map_odd_bias4") == 0) {
      config.source_row_mode = SourceRowMode::kPositionMapOddPlus4;
    }
  }
  if (char const* use_position_map_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP");
      use_position_map_env != nullptr &&
      use_position_map_env[0] != '\0' &&
      use_position_map_env[0] != '0') {
    config.source_row_mode = SourceRowMode::kPositionMap;
  }
  if (char const* dump_slot_map_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_DUMP_SLOT_MAP");
      dump_slot_map_env != nullptr &&
      dump_slot_map_env[0] != '\0' &&
      dump_slot_map_env[0] != '0') {
    config.dump_slot_map = 1;
  }
  if (char const* source_row_bit_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_SOURCE_ROW_BIT");
      source_row_bit_env != nullptr &&
      source_row_bit_env[0] != '\0') {
    config.source_row_bit = std::atoi(source_row_bit_env);
  }
  if (char const* stage_slot_bit_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_STAGE_SLOT_BIT");
      stage_slot_bit_env != nullptr &&
      stage_slot_bit_env[0] != '\0') {
    config.stage_slot_bit = std::atoi(stage_slot_bit_env);
  }
  if (char const* stage_slot_low_byte_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_STAGE_SLOT_LOW_BYTE");
      stage_slot_low_byte_env != nullptr &&
      stage_slot_low_byte_env[0] != '\0' &&
      stage_slot_low_byte_env[0] != '0') {
    config.stage_slot_low_byte = 1;
  }
  if (char const* input_rows_env = std::getenv("NEMOTRON_NANO_P1_B_INPUT_ROWS");
      input_rows_env != nullptr &&
      input_rows_env[0] != '\0') {
    config.input_rows = std::atoi(input_rows_env);
  }
  if (char const* packed_row_bytes_env = std::getenv("NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES");
      packed_row_bytes_env != nullptr &&
      packed_row_bytes_env[0] != '\0') {
    config.packed_row_bytes = std::atoi(packed_row_bytes_env);
  }
  if (char const* packed_byte_offset_env = std::getenv("NEMOTRON_NANO_P1_B_PACKED_BYTE_OFFSET");
      packed_byte_offset_env != nullptr &&
      packed_byte_offset_env[0] != '\0') {
    config.packed_byte_offset = std::atoi(packed_byte_offset_env);
  }
  if (char const* stage_slot_bias_env = std::getenv("NEMOTRON_NANO_P1_B_STAGE_SLOT_BIAS");
      stage_slot_bias_env != nullptr &&
      stage_slot_bias_env[0] != '\0') {
    config.stage_slot_bias = std::atoi(stage_slot_bias_env);
  }
  if (char const* input_file_env = std::getenv("NEMOTRON_NANO_P1_B_INPUT_FILE");
      input_file_env != nullptr &&
      input_file_env[0] != '\0') {
    config.use_input_file = 1;
  }
  if (char const* input_stage_mode_env = std::getenv("NEMOTRON_NANO_P1_B_INPUT_STAGE_MODE");
      input_stage_mode_env != nullptr &&
      input_stage_mode_env[0] != '\0') {
    if (std::strcmp(input_stage_mode_env, "packed_bytes") == 0 ||
        std::strcmp(input_stage_mode_env, "packed") == 0) {
      config.input_stage_mode = InputStageMode::kPackedBytes;
    } else if (std::strcmp(input_stage_mode_env, "unpack_nibbles") == 0 ||
               std::strcmp(input_stage_mode_env, "unpacked_nibbles") == 0) {
      config.input_stage_mode = InputStageMode::kUnpackNibbles;
    } else if (std::strcmp(input_stage_mode_env, "unpack_nibbles_swap_halves") == 0 ||
               std::strcmp(input_stage_mode_env, "unpack_nibbles_swap") == 0) {
      config.input_stage_mode = InputStageMode::kUnpackNibblesSwapHalves;
    }
  }
  if (char const* odd_stage_offset_xor32_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_ODD_STAGE_OFFSET_XOR32");
      odd_stage_offset_xor32_env != nullptr &&
      odd_stage_offset_xor32_env[0] != '\0' &&
      odd_stage_offset_xor32_env[0] != '0') {
    config.odd_stage_offset_xor32 = 1;
  }
  if (char const* synthetic_linear_row8_env = std::getenv("NEMOTRON_NANO_P1_B_SYNTHETIC_LINEAR_ROW8");
      synthetic_linear_row8_env != nullptr &&
      synthetic_linear_row8_env[0] != '\0' &&
      synthetic_linear_row8_env[0] != '0') {
    config.synthetic_linear_row8 = 1;
  }
  if (char const* live_consumer_order_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_LIVE_ORDER");
      live_consumer_order_env != nullptr &&
      live_consumer_order_env[0] != '\0' &&
      live_consumer_order_env[0] != '0') {
    config.live_consumer_order = 1;
  }
  char const* tids_env = std::getenv("NEMOTRON_NANO_P1_B_PROBE_TIDS");
  if (tids_env == nullptr || tids_env[0] == '\0') {
    return config;
  }

  int parsed_count = 0;
  char const* cursor = tids_env;
  while (*cursor != '\0' && parsed_count < kTrackedTidCount) {
    while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
      ++cursor;
    }
    if (*cursor == '\0') {
      break;
    }
    char* end_ptr = nullptr;
    long const parsed_tid = std::strtol(cursor, &end_ptr, 10);
    if (end_ptr == cursor) {
      break;
    }
    config.tracked_tids[parsed_count++] = static_cast<int>(parsed_tid);
    cursor = end_ptr;
  }
  if (parsed_count > 0) {
    config.tracked_tid_count = parsed_count;
    for (int idx = parsed_count; idx < kTrackedTidCount; ++idx) {
      config.tracked_tids[idx] = -1;
    }
  }
  return config;
}

bool CheckCuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::printf(
      "nano_p1_b_operand_probe: CUDA failure at %s: %s\n",
      what,
      cudaGetErrorString(status));
  return false;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

__device__ int ProbeThreadSlot(ProbeConfig const& config, int tid) {
  for (int idx = 0; idx < config.tracked_tid_count; ++idx) {
    if (config.tracked_tids[idx] == tid) {
      return idx;
    }
  }
  return -1;
}

__global__ void NanoP1BOperandProbeKernel(
    ProbeEntry* entries,
    SurfaceStats* surface_stats,
    ProbeConfig config,
    std::uint8_t const* input_bytes) {
  __shared__ cute::array_aligned<NanoP1SmemAllocB, kNanoP1SwizzledBElems> smem_B;
  __shared__ cute::array_aligned<NanoP1SmemAllocSFB, kNanoP1ScaleStageElemsB> smem_SFB;

  auto mma = NanoP1TiledMma{};
  const int tid = static_cast<int>(threadIdx.x);
  const int probe_slot = ProbeThreadSlot(config, tid);

  auto thread_mma = mma.get_thread_slice(tid);
  auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_B.data()), NanoP1SmemLayoutB{});
  auto sSFB_ = cute::make_tensor(cute::make_smem_ptr(smem_SFB.data()), NanoP1SmemLayoutSFB{});
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sB_stage0 = sB(cute::_, cute::_, cute::Int<0>{});
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB_);
  auto stage0_B = NanoP1SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
  auto sSFB_stage0 = sSFB_(cute::_, cute::_, cute::Int<0>{});
  auto* swizzled_b_bytes = reinterpret_cast<std::uint8_t*>(smem_B.data());

  auto s2r_copy_B = cute::make_tiled_copy_B(NanoP1SmemCopyAtomB{}, mma);
  auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
  auto tCsB = s2r_thr_B.partition_S(sB);
  auto b_coords = cute::make_identity_tensor(cute::shape(sB));
  auto tCsB_coords = s2r_thr_B.partition_S(b_coords);
  auto tCrB = thread_mma.partition_fragment_B(sB_stage0);
  auto tCrB_cv = s2r_thr_B.retile_D(tCrB);
  auto s2r_copy_B_alt = cute::make_tiled_copy_B(NanoP1SmemCopyAtomBAlt{}, mma);
  auto s2r_thr_B_alt = s2r_copy_B_alt.get_thread_slice(tid);
  auto tCsB_alt = s2r_thr_B_alt.partition_S(sB);
  auto tCrB_alt = thread_mma.partition_fragment_B(sB_stage0);
  auto tCrB_cv_alt = s2r_thr_B_alt.retile_D(tCrB_alt);
  auto tCrSFB = NanoP1PartitionScaleB(sSFB_stage0, thread_mma);
  auto s2r_copy_SFB = cute::make_tiled_copy_impl(
      NanoP1SmemCopyAtomSFB{},
      NanoP1GetLayoutSFBTV(mma),
      cute::make_shape(
          cute::size<1>(cute::tile_shape(mma)),
          cute::size<2>(cute::tile_shape(mma))));
  auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(tid);
  auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
  auto scale_coords = cute::make_identity_tensor(cute::shape(sScaleB));
  auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords);
  auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);

  constexpr int kNTiles = cute::size<2>(NanoP1AccumLayout{});
  constexpr int kKBlocks = cute::size<2>(decltype(tCsB_coords){});
  static_assert(kNTiles == 8);
  static_assert(kKBlocks == 2);

  if (surface_stats != nullptr && tid == 0) {
    *surface_stats = SurfaceStats{};
    for (int slot = 0; slot < 2048; ++slot) {
      surface_stats->consumed_slot_source_rows[slot] = -2;
      surface_stats->consumed_slot_mod512[slot] = -1;
    }
  }
  __syncthreads();

  if (surface_stats != nullptr) {
    for (int linear = tid; linear < kNanoP1TileRows * kNanoP1MacroTileBytes; linear += blockDim.x) {
      const int row = linear / kNanoP1MacroTileBytes;
      const int byte_index = linear % kNanoP1MacroTileBytes;
      const int stage_offset = static_cast<int>(stage0_B(row, byte_index * 2));
      const int band = (stage_offset & 0x1ff) >> 5;
      const int slot = stage_offset / 2;
      atomicAdd(&surface_stats->band_counts[band], 1);
      atomicAdd(&surface_stats->slot_write_counts[slot], 1);
      if (DecodeSourceRow(config.source_row_mode, row, stage_offset) < 0) {
        atomicAdd(&surface_stats->invalid_count, 1);
        atomicAdd(&surface_stats->slot_invalid_write_counts[slot], 1);
        atomicCAS(&surface_stats->first_invalid_row, -1, row);
        atomicCAS(&surface_stats->first_invalid_byte, -1, byte_index);
        atomicCAS(&surface_stats->first_invalid_offset, -1, stage_offset);
      }
    }
  }
  __syncthreads();

  if (surface_stats != nullptr) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        auto copy_view = tCsB_coords(cute::_, n_tile, k_block, cute::Int<0>{});
        ForEachCopyViewCoord(copy_view, [&](auto const& coord, int elem) {
          const int row = CoordGet0(coord);
          const int col = CoordGet1(coord);
          const int stage_offset = static_cast<int>(stage0_B(row, col));
          const int slot = stage_offset / 2;
          const int mod = stage_offset & 0x1ff;
          const int decoded = DecodeSourceRow(config.source_row_mode, row, stage_offset);
          atomicAdd(&surface_stats->consumed_slot_ref_counts[slot], 1);
          atomicAdd(&surface_stats->consumed_mod512_counts[mod], 1);
          const int prev_row = atomicCAS(&surface_stats->consumed_slot_source_rows[slot], -2, decoded);
          if (prev_row != -2 && prev_row != decoded) {
            atomicAdd(&surface_stats->consumed_slot_source_row_conflicts, 1);
            atomicCAS(&surface_stats->first_slot_source_row_conflict_slot, -1, slot);
            atomicCAS(&surface_stats->first_slot_source_row_conflict_prev, -1, prev_row);
            atomicCAS(&surface_stats->first_slot_source_row_conflict_new, -1, decoded);
          }
          atomicCAS(&surface_stats->consumed_slot_mod512[slot], -1, mod);
          if (decoded < 0) {
            atomicAdd(&surface_stats->consumed_invalid_count, 1);
            atomicCAS(&surface_stats->first_consumed_invalid_tid, -1, tid);
            atomicCAS(&surface_stats->first_consumed_invalid_n_tile, -1, n_tile);
            atomicCAS(&surface_stats->first_consumed_invalid_k_block, -1, k_block);
            atomicCAS(&surface_stats->first_consumed_invalid_elem, -1, elem);
            atomicCAS(&surface_stats->first_consumed_invalid_row, -1, row);
            atomicCAS(&surface_stats->first_consumed_invalid_col, -1, col);
            atomicCAS(&surface_stats->first_consumed_invalid_offset, -1, stage_offset);
          }
        });
      }
    }
  }
  __syncthreads();

  FillProbeBStage0<ProbeTagPattern::kSourceRow>(
      swizzled_b_bytes,
      stage0_B,
      config.source_row_mode,
      config.source_row_bit,
      config.stage_slot_bit,
      config.stage_slot_low_byte,
      false,
      config.stage_slot_bias);

  CopyProbeBPerKBlock(s2r_copy_B, tCsB, tCrB_cv);

  if (probe_slot >= 0) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      auto c_atom_coords = part_c(cute::_, 0, n_tile);
      auto coord0 = c_atom_coords(0);
      const int part_output_col0 = CoordGet0(coord0);
      const int part_token_row0 = CoordGet1(coord0);

      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        auto row_anchor = tCsB_coords(cute::_, n_tile, k_block, cute::Int<0>{});
        auto copy_coord0 = row_anchor(0);
        auto copy_coord1 = row_anchor(1);

        const int local_row0 = CoordGet0(copy_coord0);
        const int local_col0 = CoordGet1(copy_coord0);
        const int local_row1 = CoordGet0(copy_coord1);
        const int local_col1 = CoordGet1(copy_coord1);
        const int stage0_offset0 = static_cast<int>(stage0_B(local_row0, local_col0));
        const int stage0_offset1 = static_cast<int>(stage0_B(local_row1, local_col1));
        const int staged_source_row0 =
            DecodeSourceRow(config.source_row_mode, local_row0, stage0_offset0);
        const int staged_source_row1 =
            DecodeSourceRow(config.source_row_mode, local_row1, stage0_offset1);

        const int entry_index =
            probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
        entries[entry_index] = ProbeEntry{
            .tid = tid,
            .atom_n = tid / 128,
            .lane_n = tid % 4,
            .n_tile = n_tile,
            .mf_inner = n_tile % 2,
            .mf_outer = n_tile / 2,
            .k_block = k_block,
            .part_output_col0 = part_output_col0,
            .part_token_row0 = part_token_row0,
            .local_row0 = local_row0,
            .local_col0 = local_col0,
            .local_row1 = local_row1,
            .local_col1 = local_col1,
            .staged_source_row0 = staged_source_row0,
            .staged_source_row1 = staged_source_row1,
            .stage0_offset0 = stage0_offset0,
            .stage0_offset1 = stage0_offset1,
            .consumed_source_row0 = static_cast<int>(swizzled_b_bytes[stage0_offset0 / 2]),
            .consumed_source_row1 = static_cast<int>(swizzled_b_bytes[stage0_offset1 / 2]),
        };
        auto b_atom_words = cute::recast<BRegister>(tCrB(cute::_, n_tile, k_block));
        entries[entry_index].source_row_reg0_pre =
            static_cast<std::uint32_t>(b_atom_words(0));
        entries[entry_index].source_row_reg1_pre =
            static_cast<std::uint32_t>(b_atom_words(1));

        int copy_view_row_mismatches = 0;
        int first_copy_mismatch_elem = -1;
        int first_copy_mismatch_row = -1;
        int first_copy_mismatch_col = -1;
        int first_copy_mismatch_stage_offset = -1;
        int first_copy_expected_source_row = -1;
        int first_copy_consumed_source_row = -1;
        ForEachCopyViewCoord(row_anchor, [&](auto const& coord, int elem) {
          const int row = CoordGet0(coord);
          const int col = CoordGet1(coord);
          const int stage_offset = static_cast<int>(stage0_B(row, col));
          const int expected_source_row =
              DecodeSourceRow(config.source_row_mode, row, stage_offset);
          const int consumed_source_row =
              static_cast<int>(swizzled_b_bytes[stage_offset / 2]);
          if (consumed_source_row != expected_source_row) {
            ++copy_view_row_mismatches;
            if (first_copy_mismatch_elem < 0) {
              first_copy_mismatch_elem = elem;
              first_copy_mismatch_row = row;
              first_copy_mismatch_col = col;
              first_copy_mismatch_stage_offset = stage_offset;
              first_copy_expected_source_row = expected_source_row;
              first_copy_consumed_source_row = consumed_source_row;
            }
          }
        });
        entries[entry_index].copy_view_row_mismatches = copy_view_row_mismatches;
        entries[entry_index].first_copy_mismatch_elem = first_copy_mismatch_elem;
        entries[entry_index].first_copy_mismatch_row = first_copy_mismatch_row;
        entries[entry_index].first_copy_mismatch_col = first_copy_mismatch_col;
        entries[entry_index].first_copy_mismatch_stage_offset = first_copy_mismatch_stage_offset;
        entries[entry_index].first_copy_expected_source_row = first_copy_expected_source_row;
        entries[entry_index].first_copy_consumed_source_row = first_copy_consumed_source_row;
      }
    }
  }
  __syncthreads();

  if (config.use_input_file != 0) {
    FillProbeBStage0FromInput(
        sB_stage0,
        swizzled_b_bytes,
        stage0_B,
        config,
        input_bytes);
  } else {
    FillProbeBStage0<ProbeTagPattern::kByteIndex>(
        swizzled_b_bytes,
        stage0_B,
        config.source_row_mode,
        config.source_row_bit,
        config.stage_slot_bit,
        config.stage_slot_low_byte,
        config.synthetic_linear_row8 != 0,
        config.stage_slot_bias);
  }

  CopyProbeBPerKBlock(s2r_copy_B, tCsB, tCrB_cv);

  if (probe_slot >= 0) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        const int entry_index =
            probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
        ProbeEntry& entry = entries[entry_index];
        auto row_anchor = tCsB_coords(cute::_, n_tile, k_block, cute::Int<0>{});
        auto byte_view = tCrB_cv(cute::_, n_tile, k_block);
        int copy_view_byte_mismatches = 0;
        int first_copy_byte_mismatch_elem = -1;
        int first_copy_byte_mismatch_row = -1;
        int first_copy_byte_mismatch_col = -1;
        int first_copy_byte_mismatch_stage_offset = -1;
        int first_copy_expected_byte_index = -1;
        int first_copy_consumed_byte_index = -1;
        ForEachCopyViewCoordValue(row_anchor, byte_view, [&](auto const& coord, auto const& value, int elem) {
          const int row = CoordGet0(coord);
          const int col = CoordGet1(coord);
          const int stage_offset = static_cast<int>(stage0_B(row, col));
          const int stage_byte_slot = stage_offset / 2;
          const int stage_byte = static_cast<int>(swizzled_b_bytes[stage_byte_slot]);
          const int expected_byte_index = col / 2;
          const int consumed_byte_index = static_cast<int>(LoadPackedValueByte(value));
          if (elem < kCopyViewSignatureLen) {
            entry.copy_view_sig_count = elem + 1;
            entry.copy_view_sig_stage_offsets[elem] = stage_offset;
            entry.copy_view_sig_stage_bytes[elem] = stage_byte;
            entry.copy_view_consumed_sig_count = elem + 1;
            entry.copy_view_consumed_sig_stage_offsets[elem] = stage_offset;
            entry.copy_view_consumed_sig_bytes[elem] = consumed_byte_index;
          }
          if (elem < kCopyViewRawSignatureLen) {
            entry.copy_view_raw_signature_count = elem + 1;
            entry.copy_view_raw_expected_byte_indices[elem] = expected_byte_index;
            entry.copy_view_raw_consumed_bytes[elem] = consumed_byte_index;
            entry.copy_view_raw_stage_offsets[elem] = stage_offset;
          }
          bool have_stage_slot = false;
          for (int sig = 0; sig < entry.copy_view_unique_stage_slot_count; ++sig) {
            if (entry.copy_view_stage_byte_slots[sig] == stage_byte_slot) {
              have_stage_slot = true;
              break;
            }
          }
          if (!have_stage_slot && entry.copy_view_unique_stage_slot_count < kCopyViewSignatureLen) {
            const int sig = entry.copy_view_unique_stage_slot_count++;
            entry.copy_view_stage_byte_slots[sig] = stage_byte_slot;
            entry.copy_view_expected_byte_indices[sig] = expected_byte_index;
          }
          if (consumed_byte_index != expected_byte_index) {
            ++copy_view_byte_mismatches;
            if (first_copy_byte_mismatch_elem < 0) {
              first_copy_byte_mismatch_elem = elem;
              first_copy_byte_mismatch_row = row;
              first_copy_byte_mismatch_col = col;
              first_copy_byte_mismatch_stage_offset = stage_offset;
              first_copy_expected_byte_index = expected_byte_index;
              first_copy_consumed_byte_index = consumed_byte_index;
            }
          }
        });
        entry.copy_view_byte_mismatches = copy_view_byte_mismatches;
        entry.first_copy_byte_mismatch_elem = first_copy_byte_mismatch_elem;
        entry.first_copy_byte_mismatch_row = first_copy_byte_mismatch_row;
        entry.first_copy_byte_mismatch_col = first_copy_byte_mismatch_col;
        entry.first_copy_byte_mismatch_stage_offset = first_copy_byte_mismatch_stage_offset;
        entry.first_copy_expected_byte_index = first_copy_expected_byte_index;
        entry.first_copy_consumed_byte_index = first_copy_consumed_byte_index;
        auto frag_view = tCrB(cute::_, n_tile, k_block);
        ForEachTensorValue(frag_view, [&](auto const& value, int elem) {
          if (elem < kCopyViewRawSignatureLen) {
            entry.frag_raw_signature_count = elem + 1;
            entry.frag_raw_consumed_bytes[elem] = static_cast<int>(LoadPackedValueByte(value));
          }
        });
        entry.consumed_byte_index0 =
            static_cast<int>(swizzled_b_bytes[entry.stage0_offset0 / 2]);
        entry.consumed_byte_index1 =
            static_cast<int>(swizzled_b_bytes[entry.stage0_offset1 / 2]);
      }
    }
  }

  __syncthreads();
  FillProbeSFBStage0<ProbeScaleTagPattern::kLogicalRow>(sSFB_stage0);

  if (probe_slot >= 0) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        const int entry_index =
            probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
        ProbeEntry& entry = entries[entry_index];
        auto scale_anchor = tCsSFB_coords(cute::_, n_tile, k_block, cute::Int<0>{});
        auto scale_coord0 = scale_anchor(0);
        entry.scale_local_row0 = CoordGet0(scale_coord0);
        entry.scale_local_col0 = CoordGet1(scale_coord0);
        entry.scale_stage0_offset0 =
            static_cast<int>(sSFB_stage0.layout()(entry.scale_local_row0, entry.scale_local_col0));
      }
    }
  }
  cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);
  if (probe_slot >= 0) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        const int entry_index =
            probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
        entries[entry_index].consumed_scale_row_word0 =
            PackScaleFragmentWordLocal(tCrSFB(cute::_, n_tile, k_block));
      }
    }
  }

  __syncthreads();
  FillProbeSFBStage0<ProbeScaleTagPattern::kLogicalCol>(sSFB_stage0);
  cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);
  if (probe_slot >= 0) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        const int entry_index =
            probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
        entries[entry_index].consumed_scale_col_word0 =
            PackScaleFragmentWordLocal(tCrSFB(cute::_, n_tile, k_block));
      }
    }
  }

  __syncthreads();
  using MMAOp = typename NanoP1TiledMma::MMA_Op;
  if (config.live_consumer_order != 0) {
    for (int k_block = 0; k_block < kKBlocks; ++k_block) {
      cute::copy(
          s2r_copy_B,
          tCsB(cute::_, cute::_, k_block, cute::Int<0>{}),
          tCrB_cv(cute::_, cute::_, k_block));
      cute::copy(
          s2r_copy_B_alt,
          tCsB_alt(cute::_, cute::_, k_block, cute::Int<0>{}),
          tCrB_cv_alt(cute::_, cute::_, k_block));

      if (probe_slot >= 0) {
        for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
          auto b_atom_words = cute::recast<BRegister>(tCrB(cute::_, n_tile, k_block));
          auto b_atom_words_alt = cute::recast<BRegister>(tCrB_alt(cute::_, n_tile, k_block));
          const int entry_index =
              probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
          entries[entry_index].byte_tag_reg0_pre =
              static_cast<std::uint32_t>(b_atom_words(0));
          entries[entry_index].byte_tag_reg1_pre =
              static_cast<std::uint32_t>(b_atom_words(1));
          entries[entry_index].byte_tag_reg0_pre_alt =
              static_cast<std::uint32_t>(b_atom_words_alt(0));
          entries[entry_index].byte_tag_reg1_pre_alt =
              static_cast<std::uint32_t>(b_atom_words_alt(1));
        }
      }

      cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k_block));

      if (probe_slot >= 0) {
        for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
          auto b_atom_words = cute::recast<BRegister>(tCrB(cute::_, n_tile, k_block));
          const int entry_index =
              probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
          entries[entry_index].byte_tag_reg0_post =
              static_cast<std::uint32_t>(b_atom_words(0));
          entries[entry_index].byte_tag_reg1_post =
              static_cast<std::uint32_t>(b_atom_words(1));
        }
      }

      cute::copy(
          tCsSFB(cute::_, cute::_, k_block, cute::Int<0>{}),
          tCrSFB_cv(cute::_, cute::_, k_block));

      if (probe_slot >= 0) {
        for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
          const int entry_index =
              probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
          entries[entry_index].actual_scale_word0 =
              PackScaleFragmentWordLocal(tCrSFB(cute::_, n_tile, k_block));
        }
      }
    }
  } else {
    CopyProbeBPerKBlock(s2r_copy_B, tCsB, tCrB_cv);
    CopyProbeBPerKBlock(s2r_copy_B_alt, tCsB_alt, tCrB_cv_alt);
    cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

    if (probe_slot >= 0) {
      for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
        for (int k_block = 0; k_block < kKBlocks; ++k_block) {
          auto b_atom_words = cute::recast<BRegister>(tCrB(cute::_, n_tile, k_block));
          auto b_atom_words_alt = cute::recast<BRegister>(tCrB_alt(cute::_, n_tile, k_block));
          const int entry_index =
              probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
          entries[entry_index].byte_tag_reg0_pre =
              static_cast<std::uint32_t>(b_atom_words(0));
          entries[entry_index].byte_tag_reg1_pre =
              static_cast<std::uint32_t>(b_atom_words(1));
          entries[entry_index].byte_tag_reg0_pre_alt =
              static_cast<std::uint32_t>(b_atom_words_alt(0));
          entries[entry_index].byte_tag_reg1_pre_alt =
              static_cast<std::uint32_t>(b_atom_words_alt(1));
          entries[entry_index].actual_scale_word0 =
              PackScaleFragmentWordLocal(tCrSFB(cute::_, n_tile, k_block));
        }
      }
    }

    for (int k_block = 0; k_block < kKBlocks; ++k_block) {
      cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k_block));
    }

    if (probe_slot >= 0) {
      for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
        for (int k_block = 0; k_block < kKBlocks; ++k_block) {
          auto b_atom_words = cute::recast<BRegister>(tCrB(cute::_, n_tile, k_block));
          const int entry_index =
              probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
          entries[entry_index].byte_tag_reg0_post =
              static_cast<std::uint32_t>(b_atom_words(0));
          entries[entry_index].byte_tag_reg1_post =
              static_cast<std::uint32_t>(b_atom_words(1));
        }
      }
    }
  }
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::printf("nano_p1_b_operand_probe: SKIP (no CUDA device available)\n");
    return 0;
  }

  ProbeConfig const config = LoadProbeConfig();
  std::vector<std::uint8_t> host_input_bytes;
  std::uint8_t* device_input_bytes = nullptr;
  if (config.use_input_file != 0) {
    char const* input_file_path = std::getenv("NEMOTRON_NANO_P1_B_INPUT_FILE");
    if (input_file_path == nullptr || input_file_path[0] == '\0') {
      std::printf("nano_p1_b_operand_probe: missing input file path\n");
      return 1;
    }
    std::ifstream input_stream(input_file_path, std::ios::binary);
    if (!input_stream) {
      std::printf("nano_p1_b_operand_probe: failed to open input file %s\n", input_file_path);
      return 1;
    }
    host_input_bytes.assign(
        std::istreambuf_iterator<char>(input_stream),
        std::istreambuf_iterator<char>());
    std::size_t const required_bytes =
        static_cast<std::size_t>(config.input_rows) * static_cast<std::size_t>(config.packed_row_bytes);
    if (host_input_bytes.size() < required_bytes) {
      std::printf(
          "nano_p1_b_operand_probe: input file too small (%zu < %zu)\n",
          host_input_bytes.size(),
          required_bytes);
      return 1;
    }
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_input_bytes), required_bytes), "cudaMalloc input_bytes") ||
        !CheckCuda(
            cudaMemcpy(
                device_input_bytes,
                host_input_bytes.data(),
                required_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy input_bytes")) {
      cudaFree(device_input_bytes);
      return 1;
    }
  }
  ProbeEntry* probe_dev = nullptr;
  SurfaceStats* surface_stats_dev = nullptr;
  if (!CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&probe_dev), sizeof(ProbeEntry) * kProbeEntryCount),
          "cudaMalloc probe_dev") ||
      !CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&surface_stats_dev), sizeof(SurfaceStats)),
          "cudaMalloc surface_stats_dev") ||
      !CheckCuda(
          cudaMemset(probe_dev, 0xff, sizeof(ProbeEntry) * kProbeEntryCount),
          "cudaMemset probe_dev")) {
    cudaFree(probe_dev);
    cudaFree(surface_stats_dev);
    cudaFree(device_input_bytes);
    return 1;
  }

  NanoP1BOperandProbeKernel<<<1, NanoP1ThreadsPerCta>>>(probe_dev, surface_stats_dev, config, device_input_bytes);
  if (!CheckCuda(cudaGetLastError(), "launch probe kernel") ||
      !CheckCuda(cudaDeviceSynchronize(), "synchronize probe kernel")) {
    cudaFree(probe_dev);
    cudaFree(surface_stats_dev);
    cudaFree(device_input_bytes);
    return 1;
  }

  std::vector<ProbeEntry> probe(static_cast<std::size_t>(kProbeEntryCount));
  SurfaceStats surface_stats{};
  if (!CheckCuda(
          cudaMemcpy(
              probe.data(),
              probe_dev,
              sizeof(ProbeEntry) * kProbeEntryCount,
              cudaMemcpyDeviceToHost),
          "copy probe_dev") ||
      !CheckCuda(
          cudaMemcpy(
              &surface_stats,
              surface_stats_dev,
              sizeof(SurfaceStats),
              cudaMemcpyDeviceToHost),
          "copy surface_stats_dev")) {
    cudaFree(probe_dev);
    cudaFree(surface_stats_dev);
    cudaFree(device_input_bytes);
    return 1;
  }
  cudaFree(probe_dev);
  cudaFree(surface_stats_dev);
  cudaFree(device_input_bytes);

  std::printf(
      "nano_p1_b_operand_probe: current B-side staging writes source_row tags and %s into stage-0 B\n",
      config.use_input_file != 0 ? "input-file payloads" : "byte-index tags");
  std::printf(
      "nano_p1_b_operand_probe: check whether consumed_source_row0 matches part_token_row0 for each (tid, n_tile, k_block)\n");
  std::printf(
      "nano_p1_b_operand_probe: source_row_mode=%s\n",
      SourceRowModeName(config.source_row_mode));
  std::printf(
      "nano_p1_b_operand_probe: synthetic_linear_row8=%d live_consumer_order=%d\n",
      config.synthetic_linear_row8,
      config.live_consumer_order);
  if (config.use_input_file != 0) {
    std::printf(
        "nano_p1_b_operand_probe: payload_mode=input_file input_stage_mode=%s input_rows=%d packed_row_bytes=%d packed_byte_offset=%d stage_slot_bias=%d odd_stage_offset_xor32=%d\n",
        InputStageModeName(config.input_stage_mode),
        config.input_rows,
        config.packed_row_bytes,
        config.packed_byte_offset,
        config.stage_slot_bias,
        config.odd_stage_offset_xor32);
  }
  std::printf(
      "nano_p1_b_operand_probe: full_stage0_surface invalid_source_rows=%d first_invalid=(row=%d byte=%d offset=%d)\n",
      surface_stats.invalid_count,
      surface_stats.first_invalid_row,
      surface_stats.first_invalid_byte,
      surface_stats.first_invalid_offset);
  std::printf("nano_p1_b_operand_probe: full_stage0_surface band_counts=");
  for (int band = 0; band < 16; ++band) {
    std::printf("%s%d", band == 0 ? "" : ",", surface_stats.band_counts[band]);
  }
  std::printf("\n");
  int consumed_unique = 0;
  int consumed_invalid_unique = 0;
  int consumed_band_counts[16] = {0};
  int consumed_unique_slots = 0;
  int consumed_slots_with_invalid_writes = 0;
  int first_conflicting_slot = -1;
  int first_conflicting_slot_total_writes = -1;
  int first_conflicting_slot_invalid_writes = -1;
  for (int mod = 0; mod < 512; ++mod) {
    if (surface_stats.consumed_mod512_counts[mod] > 0) {
      ++consumed_unique;
      ++consumed_band_counts[mod >> 5];
      if (DecodeSourceRow(config.source_row_mode, 0, mod) < 0) {
        ++consumed_invalid_unique;
      }
    }
  }
  for (int slot = 0; slot < 2048; ++slot) {
    if (surface_stats.consumed_slot_ref_counts[slot] > 0) {
      ++consumed_unique_slots;
      if (surface_stats.slot_invalid_write_counts[slot] > 0) {
        ++consumed_slots_with_invalid_writes;
        if (first_conflicting_slot < 0) {
          first_conflicting_slot = slot;
          first_conflicting_slot_total_writes = surface_stats.slot_write_counts[slot];
      first_conflicting_slot_invalid_writes = surface_stats.slot_invalid_write_counts[slot];
        }
      }
    }
  }
  std::printf(
      "nano_p1_b_operand_probe: consumed_copy_view_surface unique_mod512=%d invalid_refs=%d invalid_unique_mod512=%d first_invalid=(tid=%d n_tile=%d k_block=%d elem=%d row=%d col=%d offset=%d)\n",
      consumed_unique,
      surface_stats.consumed_invalid_count,
      consumed_invalid_unique,
      surface_stats.first_consumed_invalid_tid,
      surface_stats.first_consumed_invalid_n_tile,
      surface_stats.first_consumed_invalid_k_block,
      surface_stats.first_consumed_invalid_elem,
      surface_stats.first_consumed_invalid_row,
      surface_stats.first_consumed_invalid_col,
      surface_stats.first_consumed_invalid_offset);
  std::printf("nano_p1_b_operand_probe: consumed_copy_view_surface band_counts=");
  for (int band = 0; band < 16; ++band) {
    std::printf("%s%d", band == 0 ? "" : ",", consumed_band_counts[band]);
  }
  std::printf("\n");
  std::printf(
      "nano_p1_b_operand_probe: consumed_byte_slot_surface unique_slots=%d slots_with_invalid_writes=%d first_conflict=(slot=%d total_writes=%d invalid_writes=%d)\n",
      consumed_unique_slots,
      consumed_slots_with_invalid_writes,
      first_conflicting_slot,
      first_conflicting_slot_total_writes,
      first_conflicting_slot_invalid_writes);
  std::printf(
      "nano_p1_b_operand_probe: consumed_byte_slot_surface source_row_conflicts=%d first_conflict=(slot=%d prev=%d new=%d)\n",
      surface_stats.consumed_slot_source_row_conflicts,
      surface_stats.first_slot_source_row_conflict_slot,
      surface_stats.first_slot_source_row_conflict_prev,
      surface_stats.first_slot_source_row_conflict_new);
  if (config.dump_slot_map != 0) {
    for (int slot = 0; slot < 2048; ++slot) {
      if (surface_stats.consumed_slot_ref_counts[slot] <= 0) {
        continue;
      }
      std::printf(
          "nano_p1_b_operand_probe: slot_map slot=%d mod512=%d source_row=%d consumed_refs=%d total_writes=%d invalid_writes=%d\n",
          slot,
          surface_stats.consumed_slot_mod512[slot],
          surface_stats.consumed_slot_source_rows[slot],
          surface_stats.consumed_slot_ref_counts[slot],
          surface_stats.slot_write_counts[slot],
          surface_stats.slot_invalid_write_counts[slot]);
    }
  }
  std::printf("nano_p1_b_operand_probe: tracked_tids=");
  for (int idx = 0; idx < config.tracked_tid_count; ++idx) {
    std::printf("%s%d", idx == 0 ? "" : ",", config.tracked_tids[idx]);
  }
  std::printf("\n");

  int total_mismatches = 0;
  for (int tid_slot = 0; tid_slot < config.tracked_tid_count; ++tid_slot) {
    const int tid = config.tracked_tids[tid_slot];
    int thread_mismatches = 0;
    std::printf("nano_p1_b_operand_probe: ---- tid=%d ----\n", tid);
    for (int n_tile = 0; n_tile < 8; ++n_tile) {
      for (int k_block = 0; k_block < 2; ++k_block) {
        const ProbeEntry& entry =
            probe[static_cast<std::size_t>(tid_slot * 16 + n_tile * 2 + k_block)];
        const bool row_match = entry.consumed_source_row0 == entry.part_token_row0;
        if (!row_match) {
          ++thread_mismatches;
          ++total_mismatches;
        }
        std::printf(
            "  n_tile=%d mf_inner=%d mf_outer=%d k_block=%d atom_n=%d lane_n=%d "
            "part_token_row0=%d part_output_col0=%d local_row0=%d local_col0=%d "
            "local_row1=%d local_col1=%d staged_source_row0=%d staged_source_row1=%d "
            "consumed_source_row0=%d consumed_source_row1=%d "
            "copy_view_row_mismatches=%d first_copy_mismatch=(elem=%d row=%d col=%d offset=%d expected=%d got=%d) "
            "copy_view_byte_mismatches=%d first_copy_byte_mismatch=(elem=%d row=%d col=%d offset=%d expected=%d got=%d) "
            "copy_sig_count=%d "
            "copy_view_unique_stage_slot_count=%d copy_view_stage_byte_slots=[%d,%d,%d,%d,%d,%d,%d,%d] "
            "copy_view_expected_byte_indices=[%d,%d,%d,%d,%d,%d,%d,%d] "
            "copy_view_raw_signature_count=%d "
            "consumed_byte_index0=%d consumed_byte_index1=%d "
            "scale_local_row0=%d scale_local_col0=%d scale_stage0_offset0=%d "
            "consumed_scale_row_word0=0x%08x consumed_scale_col_word0=0x%08x actual_scale_word0=0x%08x "
            "stage0_offset0=%d stage0_offset1=%d "
            "source_row_reg0_pre=0x%08x source_row_reg1_pre=0x%08x "
            "byte_tag_reg0_pre=0x%08x byte_tag_reg1_pre=0x%08x "
            "byte_tag_reg0_pre_alt=0x%08x byte_tag_reg1_pre_alt=0x%08x "
            "byte_tag_reg0_post=0x%08x byte_tag_reg1_post=0x%08x %s\n",
            entry.n_tile,
            entry.mf_inner,
            entry.mf_outer,
            entry.k_block,
            entry.atom_n,
            entry.lane_n,
            entry.part_token_row0,
            entry.part_output_col0,
            entry.local_row0,
            entry.local_col0,
            entry.local_row1,
            entry.local_col1,
            entry.staged_source_row0,
            entry.staged_source_row1,
            entry.consumed_source_row0,
            entry.consumed_source_row1,
            entry.copy_view_row_mismatches,
            entry.first_copy_mismatch_elem,
            entry.first_copy_mismatch_row,
            entry.first_copy_mismatch_col,
            entry.first_copy_mismatch_stage_offset,
            entry.first_copy_expected_source_row,
            entry.first_copy_consumed_source_row,
            entry.copy_view_byte_mismatches,
            entry.first_copy_byte_mismatch_elem,
            entry.first_copy_byte_mismatch_row,
            entry.first_copy_byte_mismatch_col,
            entry.first_copy_byte_mismatch_stage_offset,
            entry.first_copy_expected_byte_index,
            entry.first_copy_consumed_byte_index,
            entry.copy_view_sig_count,
            entry.copy_view_unique_stage_slot_count,
            entry.copy_view_stage_byte_slots[0],
            entry.copy_view_stage_byte_slots[1],
            entry.copy_view_stage_byte_slots[2],
            entry.copy_view_stage_byte_slots[3],
            entry.copy_view_stage_byte_slots[4],
            entry.copy_view_stage_byte_slots[5],
            entry.copy_view_stage_byte_slots[6],
            entry.copy_view_stage_byte_slots[7],
            entry.copy_view_expected_byte_indices[0],
            entry.copy_view_expected_byte_indices[1],
            entry.copy_view_expected_byte_indices[2],
            entry.copy_view_expected_byte_indices[3],
            entry.copy_view_expected_byte_indices[4],
            entry.copy_view_expected_byte_indices[5],
            entry.copy_view_expected_byte_indices[6],
            entry.copy_view_expected_byte_indices[7],
            entry.copy_view_raw_signature_count,
            entry.consumed_byte_index0,
            entry.consumed_byte_index1,
            entry.scale_local_row0,
            entry.scale_local_col0,
            entry.scale_stage0_offset0,
            entry.consumed_scale_row_word0,
            entry.consumed_scale_col_word0,
            entry.actual_scale_word0,
            entry.stage0_offset0,
            entry.stage0_offset1,
            entry.source_row_reg0_pre,
            entry.source_row_reg1_pre,
            entry.byte_tag_reg0_pre,
            entry.byte_tag_reg1_pre,
            entry.byte_tag_reg0_pre_alt,
            entry.byte_tag_reg1_pre_alt,
            entry.byte_tag_reg0_post,
            entry.byte_tag_reg1_post,
            row_match ? "ROW_MATCH" : "ROW_MISMATCH");
        if (entry.copy_view_sig_count > 0) {
          std::printf("    copy_sig=");
          for (int sig = 0; sig < entry.copy_view_sig_count; ++sig) {
            std::printf(
                "%s%d:%d",
                sig == 0 ? "" : ",",
                entry.copy_view_sig_stage_offsets[sig],
                entry.copy_view_sig_stage_bytes[sig]);
          }
          std::printf("\n");
        }
        if (entry.copy_view_consumed_sig_count > 0) {
          std::printf("    copy_consumed_sig=");
          for (int sig = 0; sig < entry.copy_view_consumed_sig_count; ++sig) {
            std::printf(
                "%s%d:%d",
                sig == 0 ? "" : ",",
                entry.copy_view_consumed_sig_stage_offsets[sig],
                entry.copy_view_consumed_sig_bytes[sig]);
          }
          std::printf("\n");
        }
        if (entry.copy_view_raw_signature_count > 0) {
          std::printf("    ");
          PrintIntArray(
              "copy_view_raw_expected_byte_indices",
              entry.copy_view_raw_expected_byte_indices,
              entry.copy_view_raw_signature_count);
          std::printf(" ");
          PrintIntArray(
              "copy_view_raw_consumed_bytes",
              entry.copy_view_raw_consumed_bytes,
              entry.copy_view_raw_signature_count);
          std::printf(" ");
          PrintIntArray(
              "copy_view_raw_stage_offsets",
              entry.copy_view_raw_stage_offsets,
              entry.copy_view_raw_signature_count);
          if (entry.frag_raw_signature_count > 0) {
            std::printf(" ");
            PrintIntArray(
                "frag_raw_consumed_bytes",
                entry.frag_raw_consumed_bytes,
                entry.frag_raw_signature_count);
          }
          std::printf("\n");
        }
      }
    }
    std::printf(
        "nano_p1_b_operand_probe: tid=%d row_mismatches=%d/16\n",
        tid,
        thread_mismatches);
  }

  std::printf(
      "nano_p1_b_operand_probe: total_row_mismatches=%d/%d\n",
      total_mismatches,
      kProbeEntryCount);
  return 0;
}
