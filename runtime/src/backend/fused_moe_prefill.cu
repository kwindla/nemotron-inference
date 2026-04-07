#include "nemotron/fused_moe_prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor_impl.hpp>
#endif

#include "nemotron/device_nvfp4_matrix.h"
#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

namespace wmma = nvcuda::wmma;

#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#endif

namespace nvfp4_bridge {

#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
using TracedP5SmemLayoutSFA = typename TracedP5CollectiveMainloop::SmemLayoutSFA;
using TracedP5SmemLayoutSFB = typename TracedP5CollectiveMainloop::SmemLayoutSFB;
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
#else
using ARegister = std::uint32_t;
using BRegister = std::uint32_t;
using CRegister = float;
using SFRegister = std::uint32_t;
#endif

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

constexpr int kTiledCopyCoordCapacityA = 32;
constexpr int kTiledCopyCoordCapacityB = 16;
constexpr int kTiledCopyCoordCapacityScale = 4;
constexpr int kTracedP5ACopyCoordCapacity = 128;
constexpr int kTracedP5BCopyCoordCapacity = 64;
constexpr int kTracedP5CCopyCoordCapacity = 32;
constexpr int kTracedP5MFragments = 4;
constexpr int kTracedP5NFragments = 2;

#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
CUTE_HOST_DEVICE constexpr auto GetSingleAtomTiledMma() {
  return SingleAtomTiledMma{};
}

CUTE_HOST_DEVICE constexpr auto GetTracedP5TiledMma() {
  return TracedP5TiledMma{};
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

template <class Coord>
CUTE_HOST_DEVICE constexpr int CoordGet0(Coord const& coord) {
  if constexpr (cute::is_tuple<Coord>::value) {
    return static_cast<int>(cute::get<0>(coord));
  } else {
    return static_cast<int>(coord);
  }
}

template <class Coord>
CUTE_HOST_DEVICE constexpr int CoordGet1(Coord const& coord) {
  if constexpr (cute::is_tuple<Coord>::value) {
    return static_cast<int>(cute::get<1>(coord));
  } else {
    return 0;
  }
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
#endif

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

#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
template <class ScaleTensor>
__device__ __forceinline__ void ZeroTracedP5ScaleRow(
    const ScaleTensor& scale_tensor,
    int row);

template <class ScaleTensor>
__device__ __forceinline__ void StoreTracedP5ScaleWordK64(
    const ScaleTensor& scale_tensor,
    std::uint32_t scale_word,
    int row);
#endif

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
__device__ __forceinline__ BFragment64 LoadFragmentB_ColMajor64x8TracedScaleTiled(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int n_base);

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

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         weight.block_scales_data != nullptr &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

__host__ __device__ fused_decode::Nvfp4WeightView MakeDeviceWeightView(
    const FusedNvfp4WeightView& weight) {
  return fused_decode::Nvfp4WeightView{
      weight.packed_data,
      weight.block_scales_data,
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
  if (num_rows == 1u) {
    return RoutedGemm1Profile::kP0_128x128x128_SwapFalse;
  }
  if (num_rows >= 2u && num_rows <= 3u) {
    return RoutedGemm1Profile::kP7_256x128x64_SwapTrue;
  }
  if (num_rows >= 4u && num_rows <= 7u) {
    return RoutedGemm1Profile::kP5_128x128x64_SwapTrue;
  }
  if (num_rows == 8u || (num_rows >= 9u && num_rows <= 15u) || num_rows == 16u ||
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

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

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ float DecodeNvfp4WeightElement(
    const FusedNvfp4WeightView& weight,
    std::size_t output_row,
    std::size_t col) {
  const std::size_t pairs_per_row = weight.input_cols / 2u;
  const std::size_t blocks_per_row =
      weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t pair_index = col / 2u;
  const std::size_t block = col / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * pairs_per_row;
  const std::size_t scale_row_offset = output_row * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;
  const float block_scale =
      fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
      tensor_scale;
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
  const std::size_t scale_row_offset = output_row * blocks_per_row + block_index;
  const float block_scale =
      fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset]) *
      (*weight.tensor_scale_data);
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
  nvfp4_cute::MmaOp::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3, sfa, sfb);
#else
  static constexpr std::uint16_t kBidA = 0;
  static constexpr std::uint16_t kTidA = 0;
  static constexpr std::uint16_t kBidB = 0;
  static constexpr std::uint16_t kTidB = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%0, %1, %2, %3},"
      "{%10},"
      "{%11, %12},"
      "{%13},"
      "{%14, %15};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(sfa), "h"(kBidA), "h"(kTidA),
        "r"(sfb), "h"(kBidB), "h"(kTidB));
#endif
#endif
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

#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::ZeroTracedP5ScaleRow(
    const ScaleTensor& scale_tensor,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
#pragma unroll
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    scale_tensor(row, scale_col, cute::Int<0>{}) = std::uint8_t{0};
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedP5ScaleWordK64(
    const ScaleTensor& scale_tensor,
    std::uint32_t scale_word,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
#pragma unroll
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    const std::uint8_t value = scale_col < 4 ? LoadScaleByte(scale_word, scale_col) : std::uint8_t{0};
    scale_tensor(row, scale_col, cute::Int<0>{}) = value;
  }
}
#endif

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int row_base,
    int thread_idx) {
  AFragment64 fragment{};
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#else
  LoadFp4ARegistersRowMajor16x64<kRowsPerTile>(
      packed_rows + static_cast<std::size_t>(row_base) * (64 / 2),
      scale_words + static_cast<std::size_t>(row_base),
      lane_id,
      fragment.regs[0],
      fragment.regs[1],
      fragment.regs[2],
      fragment.regs[3],
      fragment.scale[0]);
#endif
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#else
  (void)scale_smem;
#endif
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#else
  LoadFp4BRegistersColMajor64x8<kRowsPerTile>(
      packed_rows,
      scale_words,
      thread_idx,
      n_base,
      fragment.regs[0],
      fragment.regs[1],
      fragment.scale[0]);
#endif
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#else
  (void)scale_smem;
#endif
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP5AFragmentsRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    AFragment64 (&fragments)[kTracedP5MFragments]) {
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#else
  for (int m_fragment = 0; m_fragment < kTracedP5MFragments; ++m_fragment) {
    fragments[m_fragment] =
        LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            reinterpret_cast<const std::uint32_t*>(scale_smem),
            m_fragment * 16,
            thread_idx);
  }
#endif
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP5BFragmentsColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    BFragment64 (&fragments)[kTracedP5NFragments]) {
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#else
  for (int n_fragment = 0; n_fragment < kTracedP5NFragments; ++n_fragment) {
    fragments[n_fragment] =
        LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            reinterpret_cast<const std::uint32_t*>(scale_smem),
            thread_idx,
            n_fragment * 8);
  }
#endif
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
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
#else
  for (int m_fragment = 0; m_fragment < kTracedP5MFragments; ++m_fragment) {
    for (int n_fragment = 0; n_fragment < kTracedP5NFragments; ++n_fragment) {
      StoreFragmentC_Transpose16x8(
          alpha,
          accum[m_fragment][n_fragment],
          thread_idx & 31,
          output_row_base,
          m_fragment * 16,
          n_fragment * 8 + (thread_idx & 7),
          valid_rows,
          output_rows_this_tile,
          row_start,
          output_rows_per_expert,
          output);
    }
  }
#endif
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
          fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
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
        DecodeGroupedPackedInputBlockBf16(
            packed_input,
            input_block_scales,
            input_dq_scales,
            input_tensor_scale,
            weight.input_cols,
            static_cast<std::size_t>(row_start + tile_token),
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
        DecodeGroupedPackedInputBlockBf16(
            packed_input,
            input_block_scales,
            input_dq_scales,
            input_tensor_scale,
            weight.input_cols,
            static_cast<std::size_t>(row_start + tile_token),
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

template <typename OutputType, int kOutputTile>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64(
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
  constexpr int kFp4ConsumerWarps = cute::size(nvfp4_bridge::TracedP5TiledMma{}) / 32;
  static_assert(kOutputTile % (kFp4ConsumerWarps * kFp4MmaTileN) == 0);
  constexpr int kWarpSubTiles = kOutputTile / (kFp4ConsumerWarps * kFp4MmaTileN);
  __shared__ std::uint8_t a_packed[kPlannedWmmaTileM][64 / 2];
  __shared__ std::uint32_t a_scale_words[kPlannedWmmaTileM];
  __shared__ std::uint8_t b_packed[kOutputTile][64 / 2];
  __shared__ std::uint32_t b_scale_words[kOutputTile];
  const auto a_tile_view =
      nvfp4_bridge::MakePackedTile64<kPlannedWmmaTileM>(&a_packed[0][0], a_scale_words);
  const auto b_tile_view =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&b_packed[0][0], b_scale_words);

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

  nvfp4_bridge::CFragment64 accum[kWarpSubTiles];
  if (warp_id < kGroupedConsumerWarpsPerBlock) {
#pragma unroll
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      nvfp4_bridge::Clear(accum[subtile]);
    }
  }

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += 64u) {
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = k_base / 2u;

    for (int row = tid; row < kPlannedWmmaTileM; row += blockDim.x) {
      if (row < valid_rows) {
        nvfp4_bridge::CopyActivationRow64(
            a_tile_view,
            packed_input,
            packed_row_bytes,
            static_cast<std::size_t>(row_start + row),
            packed_byte_offset,
            input_block_scales,
            block_base,
            blocks_per_row,
            row);
      } else {
        nvfp4_bridge::ZeroRow(a_tile_view, row);
      }
    }
    for (int row = tid; row < kOutputTile; row += blockDim.x) {
      if (row < output_rows_this_tile) {
        nvfp4_bridge::CopyWeightRow64(
            b_tile_view,
            weight.packed_data,
            packed_row_bytes,
            static_cast<std::size_t>(output_row_base + row),
            packed_byte_offset,
            weight.matmul_block_scales_data,
            block_base,
            padded_blocks_per_row,
            Nvfp4ScaleLayout::kSwizzled128x4,
            row);
      } else {
        nvfp4_bridge::ZeroRow(b_tile_view, row);
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      const auto a_fragment =
          nvfp4_bridge::LoadFragmentA_RowMajor16x64Tiled<nvfp4_bridge::TracedP5TiledMma, kPlannedWmmaTileM>(
              &a_packed[0][0], a_scale_words, 0, warp_id * 32 + lane_id);
#pragma unroll
      for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
        const int n_base = warp_id * kFp4MmaTileN + subtile * kFp4ConsumerWarps * kFp4MmaTileN;
        if (n_base < output_rows_this_tile) {
          const auto b_fragment =
              nvfp4_bridge::LoadFragmentB_ColMajor64x8Tiled<nvfp4_bridge::TracedP5TiledMma, kOutputTile>(
                  &b_packed[0][0], b_scale_words, warp_id * 32 + lane_id, n_base);
          nvfp4_bridge::Gemm(accum[subtile], a_fragment, b_fragment);
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    const int col_in_subtile = lane_id & 7;
#pragma unroll
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      const int n_base = warp_id * kFp4MmaTileN + subtile * kFp4ConsumerWarps * kFp4MmaTileN;
      nvfp4_bridge::StoreFragmentC_RowMajor16x8<kPlannedWmmaTileM>(
          output_alpha,
          accum[subtile],
          lane_id,
          n_base + col_in_subtile,
          0,
          valid_rows,
          output_rows_this_tile,
          static_cast<std::size_t>(row_start),
          static_cast<std::size_t>(output_row_base),
          output_rows_per_expert,
          output);
    }
  }
}

template <typename OutputType>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64P13(
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
  constexpr int kFp4ConsumerWarps = cute::size(nvfp4_bridge::TracedP5TiledMma{}) / 32;
  constexpr int kWarpSubTiles = kOutputTile / (kFp4ConsumerWarps * kFp4MmaTileN);
  static_assert(kWarpSubTiles >= 1);
  __shared__ std::uint8_t a_packed[kPlannedWmmaTileM][64 / 2];
  __shared__ std::uint8_t a_scale_smem[nvfp4_bridge::kTracedP5ScaleSmemCosizeA];
  __shared__ std::uint8_t b_packed[kOutputTile][64 / 2];
  __shared__ std::uint8_t b_scale_smem[nvfp4_bridge::kTracedP5ScaleSmemCosizeB];
  const auto a_tile_view =
      nvfp4_bridge::MakePackedTile64<kPlannedWmmaTileM>(&a_packed[0][0], nullptr);
  const auto b_tile_view =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&b_packed[0][0], nullptr);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
  auto a_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFA{});
  auto b_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFB{});
#endif

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

  nvfp4_bridge::CFragment64 accum[kWarpSubTiles];
  if (warp_id < kFp4ConsumerWarps) {
#pragma unroll
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      nvfp4_bridge::Clear(accum[subtile]);
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            a_scale_tensor,
            PackScaleWord4(
                input_block_scales[scale_offset + 0u],
                input_block_scales[scale_offset + 1u],
                input_block_scales[scale_offset + 2u],
                input_block_scales[scale_offset + 3u]),
            row);
#endif
      } else {
        nvfp4_bridge::ZeroRow(a_tile_view, row);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
#endif
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            b_scale_tensor,
            LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                source_row,
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4),
            row);
#endif
      } else {
        nvfp4_bridge::ZeroRow(b_tile_view, row);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
#endif
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      const auto a_fragment =
          nvfp4_bridge::LoadFragmentA_RowMajor16x64TracedScaleTiled<nvfp4_bridge::TracedP5TiledMma, kPlannedWmmaTileM>(
              &a_packed[0][0], a_scale_smem, 0, warp_id * 32 + lane_id);
#pragma unroll
      for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
        const int n_base = warp_id * kFp4MmaTileN + subtile * kFp4ConsumerWarps * kFp4MmaTileN;
        if (n_base < output_rows_this_tile) {
          const auto b_fragment =
              nvfp4_bridge::LoadFragmentB_ColMajor64x8TracedScaleTiled<nvfp4_bridge::TracedP5TiledMma, kOutputTile>(
                  &b_packed[0][0], b_scale_smem, warp_id * 32 + lane_id, n_base);
          nvfp4_bridge::Gemm(accum[subtile], a_fragment, b_fragment);
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    const int col_in_subtile = lane_id & 7;
#pragma unroll
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      const int n_base = warp_id * kFp4MmaTileN + subtile * kFp4ConsumerWarps * kFp4MmaTileN;
      nvfp4_bridge::StoreFragmentC_RowMajor16x8<kPlannedWmmaTileM>(
          output_alpha,
          accum[subtile],
          lane_id,
          n_base + col_in_subtile,
          0,
          valid_rows,
          output_rows_this_tile,
          static_cast<std::size_t>(row_start),
          static_cast<std::size_t>(output_row_base),
          output_rows_per_expert,
          output);
    }
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
  constexpr int kFp4ConsumerWarps = cute::size(nvfp4_bridge::TracedP5TiledMma{}) / 32;
  constexpr int kWarpSubTiles = kOutputTile / (kFp4ConsumerWarps * kFp4MmaTileN);
  static_assert(kWarpSubTiles >= 1);
  __shared__ std::uint8_t a_packed[2][kPlannedWmmaTileM][64 / 2];
  __shared__ std::uint8_t a_scale_smem[2][nvfp4_bridge::kTracedP5ScaleSmemCosizeA];
  __shared__ std::uint8_t b_packed[2][kOutputTile][64 / 2];
  __shared__ std::uint8_t b_scale_smem[2][nvfp4_bridge::kTracedP5ScaleSmemCosizeB];
  const auto a_tile_view0 =
      nvfp4_bridge::MakePackedTile64<kPlannedWmmaTileM>(&a_packed[0][0][0], nullptr);
  const auto a_tile_view1 =
      nvfp4_bridge::MakePackedTile64<kPlannedWmmaTileM>(&a_packed[1][0][0], nullptr);
  const auto b_tile_view0 =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&b_packed[0][0][0], nullptr);
  const auto b_tile_view1 =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&b_packed[1][0][0], nullptr);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
  auto a_scale_tensor0 =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[0][0]), nvfp4_bridge::TracedP5SmemLayoutSFA{});
  auto a_scale_tensor1 =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[1][0]), nvfp4_bridge::TracedP5SmemLayoutSFA{});
  auto b_scale_tensor0 =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[0][0]), nvfp4_bridge::TracedP5SmemLayoutSFB{});
  auto b_scale_tensor1 =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[1][0]), nvfp4_bridge::TracedP5SmemLayoutSFB{});
#endif

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

  nvfp4_bridge::CFragment64 accum[kWarpSubTiles];
  if (warp_id < kFp4ConsumerWarps) {
#pragma unroll
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      nvfp4_bridge::Clear(accum[subtile]);
    }
  }

  for (std::size_t macro_k_base = 0; macro_k_base < weight.input_cols; macro_k_base += 128u) {
    const std::size_t remaining_k = weight.input_cols - macro_k_base;
    const int macro_k = static_cast<int>(remaining_k < 128u ? remaining_k : 128u);

    for (int sub = 0; sub < 2; ++sub) {
      const std::size_t k_base = macro_k_base + static_cast<std::size_t>(sub) * 64u;
      const bool sub_active = static_cast<int>(k_base - macro_k_base) < macro_k;
      const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
      const std::size_t packed_byte_offset = k_base / 2u;
      const auto a_tile_view = sub == 0 ? a_tile_view0 : a_tile_view1;
      const auto b_tile_view = sub == 0 ? b_tile_view0 : b_tile_view1;
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
      auto& a_scale_tensor = sub == 0 ? a_scale_tensor0 : a_scale_tensor1;
      auto& b_scale_tensor = sub == 0 ? b_scale_tensor0 : b_scale_tensor1;
#endif

      for (int row = tid; row < kPlannedWmmaTileM; row += blockDim.x) {
        if (sub_active && row < valid_rows) {
          const std::size_t source_row = static_cast<std::size_t>(row_start + row);
          const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
          std::uint8_t* dst = a_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
          for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
            dst[byte_index] = packed_input[src_offset + static_cast<std::size_t>(byte_index)];
          }
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
          const std::size_t scale_offset = source_row * blocks_per_row + block_base;
          nvfp4_bridge::StoreTracedP5ScaleWordK64(
              a_scale_tensor,
              PackScaleWord4(
                  input_block_scales[scale_offset + 0u],
                  input_block_scales[scale_offset + 1u],
                  input_block_scales[scale_offset + 2u],
                  input_block_scales[scale_offset + 3u]),
              row);
#endif
        } else {
          nvfp4_bridge::ZeroRow(a_tile_view, row);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
          nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
#endif
        }
      }
      for (int row = tid; row < kOutputTile; row += blockDim.x) {
        if (sub_active && row < output_rows_this_tile) {
          const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
          const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
          std::uint8_t* dst = b_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
          for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
            dst[byte_index] = weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)];
          }
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
          nvfp4_bridge::StoreTracedP5ScaleWordK64(
              b_scale_tensor,
              LoadExecutionScaleWord(
                  weight.matmul_block_scales_data,
                  source_row,
                  block_base,
                  padded_blocks_per_row,
                  Nvfp4ScaleLayout::kSwizzled128x4),
              row);
#endif
        } else {
          nvfp4_bridge::ZeroRow(b_tile_view, row);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
          nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
#endif
        }
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        if (sub * 64 >= macro_k) {
          continue;
        }
        const std::uint8_t* a_scale = sub == 0 ? &a_scale_smem[0][0] : &a_scale_smem[1][0];
        const std::uint8_t* b_scale = sub == 0 ? &b_scale_smem[0][0] : &b_scale_smem[1][0];
        const std::uint8_t* a_pack = &a_packed[sub][0][0];
        const std::uint8_t* b_pack = &b_packed[sub][0][0];
        const auto a_fragment =
            nvfp4_bridge::LoadFragmentA_RowMajor16x64TracedScaleTiled<nvfp4_bridge::TracedP5TiledMma, kPlannedWmmaTileM>(
                a_pack, a_scale, 0, warp_id * 32 + lane_id);
#pragma unroll
        for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
          const int n_base = warp_id * kFp4MmaTileN + subtile * kFp4ConsumerWarps * kFp4MmaTileN;
          if (n_base < output_rows_this_tile) {
            const auto b_fragment =
                nvfp4_bridge::LoadFragmentB_ColMajor64x8TracedScaleTiled<nvfp4_bridge::TracedP5TiledMma, kOutputTile>(
                    b_pack, b_scale, warp_id * 32 + lane_id, n_base);
            nvfp4_bridge::Gemm(accum[subtile], a_fragment, b_fragment);
          }
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    const int col_in_subtile = lane_id & 7;
#pragma unroll
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      const int n_base = warp_id * kFp4MmaTileN + subtile * kFp4ConsumerWarps * kFp4MmaTileN;
      nvfp4_bridge::StoreFragmentC_RowMajor16x8<kPlannedWmmaTileM>(
          output_alpha,
          accum[subtile],
          lane_id,
          n_base + col_in_subtile,
          0,
          valid_rows,
          output_rows_this_tile,
          static_cast<std::size_t>(row_start),
          static_cast<std::size_t>(output_row_base),
          output_rows_per_expert,
          output);
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
  auto a_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFA{});
  auto b_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFB{});
#endif

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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            a_scale_tensor,
            LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                source_row,
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4),
            row);
#endif
      } else {
        nvfp4_bridge::ZeroRow(a_tile_view, row);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
#endif
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
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            b_scale_tensor,
            PackScaleWord4(
                input_block_scales[scale_offset + 0u],
                input_block_scales[scale_offset + 1u],
                input_block_scales[scale_offset + 2u],
                input_block_scales[scale_offset + 3u]),
            row);
#endif
      } else {
        nvfp4_bridge::ZeroRow(b_tile_view, row);
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
#endif
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

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
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
    const float* output_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_m_limits,
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
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      input_tensor_scale_data == nullptr ||
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
  const std::size_t pairs_per_input_row = weight.input_cols / 2;
  const std::size_t input_blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
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
            static_cast<std::size_t>(output_row) * pairs_per_input_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * input_blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
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

  if (lane_id >= valid_rows || warp_row >= output_rows_this_tile) {
    return;
  }

  const std::size_t output_row = static_cast<std::size_t>(row_start + lane_id);
  if (output_row >= output_row_capacity) {
    return;
  }
  const std::size_t blocks_per_output_row =
      output_rows_per_expert / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_rows = RoundUp(output_row_capacity, RowTile(output_scale_layout));
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
          fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
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
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
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

bool ClearDeviceNvfp4Matrix(DeviceNvfp4Matrix* matrix, float tensor_scale = 1.0f) {
  if (matrix == nullptr || !matrix->valid()) {
    return false;
  }
  return CheckCuda(cudaMemset(
             const_cast<std::uint8_t*>(matrix->packed_data()), 0, matrix->packed_nbytes())) &&
         CheckCuda(cudaMemset(
             const_cast<std::uint8_t*>(matrix->block_scales_data()),
             0,
             matrix->block_scales_nbytes())) &&
         CheckCuda(cudaMemset(
             const_cast<std::uint8_t*>(matrix->matmul_block_scales_data()),
             0,
             matrix->matmul_block_scales_nbytes())) &&
         CheckCuda(cudaMemcpy(
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
        Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK128P12<float><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
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
        return CheckCuda(cudaGetLastError());
      case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64P13<float><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
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
        return CheckCuda(cudaGetLastError());
      case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue<float, 256, 64><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
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
        return CheckCuda(cudaGetLastError());
      case RoutedGemm2Profile::kLegacy:
        break;
    }
    return false;
  };

  const RoutedGemm2Profile profile = SelectRoutedGemm2Profile(dispatch_rows);
  if (profile != RoutedGemm2Profile::kLegacy && launch_profile(profile)) {
    return true;
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
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            weights,
            output_rows_per_expert,
            output);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
        Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64<__nv_bfloat16, 128><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
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
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue<__nv_bfloat16, 128, 128><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
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
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP5_128x128x64_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64<__nv_bfloat16, 128><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
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
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue<__nv_bfloat16, 256, 64><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
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

[[maybe_unused]] bool LaunchPlannedPackedInputMatVecRelu2Pack(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const float* output_expert_scales,
    DeviceNvfp4Matrix* output_pack) {
  if (!input_pack.valid() ||
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
      DeviceMoeLaunchPlan::CtaCapacity(launch_plan->n_experts(), active_selection_count);
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
  Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2PackKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      output_expert_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      weights,
      output_rows_per_expert,
      output_pack->rows(),
      output_pack->scale_layout(),
      const_cast<std::uint8_t*>(output_pack->packed_data()),
      const_cast<std::uint8_t*>(output_pack->block_scales_data()),
      const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
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

}  // namespace

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

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  const std::size_t selection_count = params.token_count * params.top_k;
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
      (use_grouped_gemm1_output && params.gemm1_output_bf16 == nullptr) ||
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
      params.shared_down.output_rows != params.hidden_size) {
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
    if (!LaunchZeroBf16Buffer(
            params.gemm1_output_bf16,
            params.launch_plan->padded_row_capacity() *
                params.routed_expert_intermediate_size) ||
        !(use_packed_fc1_source
              ? LaunchPlannedPackedInputMatVecBf16(
                    *params.fc1_grouped_pack,
                    nullptr,
                    nullptr,
                    params.launch_plan,
                    params.token_count,
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
            activation_output_scale
        ) ||
        !LaunchPlannedPackedInputMatVec(
            *gemm1_output,
            nullptr,
            activation_output_scale,
            params.launch_plan,
            params.token_count,
            selection_count,
            params.routed_down_device,
            params.hidden_size,
            params.routed_gather_scratch)) {
      return false;
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
                    params.launch_plan,
                    params.token_count,
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
                    params.launch_plan,
                    params.token_count,
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

  if (!CheckCuda(cudaMemcpyAsync(
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
