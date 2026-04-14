// Scale fragment helpers for the P15 (256x128x64) SM120 FP4 block-scaled MoE GEMM.
//
// Adapted from TRT-LLM sm120_utils.cuh partition_fragment_SFA/SFB and
// transform_fragment_for_qmma patterns for our specific FP4 element and scale
// types (float_e2m1_t operands, float_ue4m3_t scales, SFVecSize=16).
#pragma once

#include <cstdint>

#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

namespace nemotron::p15_scale {

using ElementSFLoad = int32_t;
using ElementSFCompute = cute::float_ue4m3_t;
constexpr int kScaleGranularityM = 128;
constexpr int kScaleGranularityN = 128;

using AtomScaleLayout = cute::Layout<
    cute::Shape<cute::Shape<cute::_16, cute::_4>>,
    cute::Stride<cute::Stride<cute::_0, cute::_1>>>;

static_assert(cute::size(AtomScaleLayout{}) == 64);
static_assert(cute::cosize(AtomScaleLayout{}) == 4);

struct AtomBaseRows {
  int a_base_row;
  int b_base_row;
};

struct AtomScaleWords {
  std::uint32_t a_word;
  std::uint32_t b_word;
};

CUTE_HOST_DEVICE constexpr std::uint8_t
encode_scale_byte(float value) {
  return static_cast<std::uint8_t>(ElementSFCompute(value).raw());
}

CUTE_HOST_DEVICE constexpr float
decode_scale_byte(std::uint8_t raw) {
  return static_cast<float>(ElementSFCompute::bitcast(raw));
}

CUTE_HOST_DEVICE constexpr std::uint32_t
pack_scale_word4_bytes(
    std::uint8_t s0,
    std::uint8_t s1,
    std::uint8_t s2,
    std::uint8_t s3) {
  return static_cast<std::uint32_t>(s0) |
         (static_cast<std::uint32_t>(s1) << 8) |
         (static_cast<std::uint32_t>(s2) << 16) |
         (static_cast<std::uint32_t>(s3) << 24);
}

CUTE_HOST_DEVICE constexpr std::uint8_t
load_scale_byte(std::uint32_t packed_scale_word, int byte_index) {
  return static_cast<std::uint8_t>((packed_scale_word >> (byte_index * 8)) & 0xFFu);
}

template <class AtomCoordTensor>
CUTE_HOST_DEVICE constexpr AtomBaseRows
get_atom_base_rows(AtomCoordTensor const& c_atom_coords) {
  auto coord0 = c_atom_coords(0);
  return {
      static_cast<int>(cute::get<0>(coord0)),
      static_cast<int>(cute::get<1>(coord0))};
}

CUTE_HOST_DEVICE constexpr bool
atom_base_rows_in_bounds(AtomBaseRows rows, int a_rows, int b_rows) {
  return rows.a_base_row >= 0 && rows.a_base_row < a_rows &&
         rows.b_base_row >= 0 && rows.b_base_row < b_rows;
}

template <class AtomCoordTensor>
CUTE_HOST_DEVICE constexpr bool
atom_stays_in_single_scale_band(
    AtomCoordTensor const& c_atom_coords,
    int scale_granularity_m = kScaleGranularityM,
    int scale_granularity_n = kScaleGranularityN) {
  int min_row = 1 << 30;
  int max_row = -(1 << 30);
  int min_col = 1 << 30;
  int max_col = -(1 << 30);
  for (int i = 0; i < static_cast<int>(cute::size(c_atom_coords)); ++i) {
    auto coord = c_atom_coords(i);
    const int row = static_cast<int>(cute::get<0>(coord));
    const int col = static_cast<int>(cute::get<1>(coord));
    min_row = row < min_row ? row : min_row;
    max_row = row > max_row ? row : max_row;
    min_col = col < min_col ? col : min_col;
    max_col = col > max_col ? col : max_col;
  }
  return (min_row / scale_granularity_m) == (max_row / scale_granularity_m) &&
         (min_col / scale_granularity_n) == (max_col / scale_granularity_n);
}

template <class AtomCoordTensor>
CUTE_HOST_DEVICE constexpr AtomScaleWords
load_atom_scale_words(
    AtomCoordTensor const& c_atom_coords,
    std::uint32_t const* a_scale_words,
    std::uint32_t const* b_scale_words) {
  auto rows = get_atom_base_rows(c_atom_coords);
  return {a_scale_words[rows.a_base_row], b_scale_words[rows.b_base_row]};
}

template <class WordRef>
CUTE_HOST_DEVICE constexpr auto
make_atom_scale_tensor(WordRef& packed_word) {
  return cute::make_tensor(
      reinterpret_cast<ElementSFCompute*>(&packed_word),
      AtomScaleLayout{});
}

template <class ScaleTensor>
CUTE_HOST_DEVICE void
zero_traced_scale_row(ScaleTensor const& scale_tensor, int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    scale_tensor(row, scale_col, cute::Int<0>{}) = ElementSFCompute::bitcast(std::uint8_t{0});
  }
}

template <class ScaleTensor>
CUTE_HOST_DEVICE void
store_traced_scale_word_k64(
    ScaleTensor const& scale_tensor,
    std::uint32_t scale_word,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    const std::uint8_t value = scale_col < 4 ? load_scale_byte(scale_word, scale_col) : std::uint8_t{0};
    scale_tensor(row, scale_col, cute::Int<0>{}) = ElementSFCompute::bitcast(value);
  }
}

template <class Coord>
CUTE_HOST_DEVICE constexpr void
flatten_coord_pair(Coord const& coord, int& row, int& col, int& count) {
  if (count >= 2) {
    return;
  }
  if constexpr (cute::is_tuple<Coord>::value) {
    constexpr std::size_t kTupleSize = std::tuple_size_v<std::remove_cvref_t<Coord>>;
    flatten_coord_pair(cute::get<0>(coord), row, col, count);
    if constexpr (kTupleSize > 1) {
      if (count < 2) {
        flatten_coord_pair(cute::get<1>(coord), row, col, count);
      }
    }
    if constexpr (kTupleSize > 2) {
      if (count < 2) {
        flatten_coord_pair(cute::get<2>(coord), row, col, count);
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
CUTE_HOST_DEVICE constexpr int
coord_get_0(Coord const& coord) {
  int row = 0;
  int col = 0;
  int count = 0;
  flatten_coord_pair(coord, row, col, count);
  (void)col;
  return row;
}

template <class Coord>
CUTE_HOST_DEVICE constexpr int
coord_get_1(Coord const& coord) {
  int row = 0;
  int col = 0;
  int count = 0;
  flatten_coord_pair(coord, row, col, count);
  (void)row;
  return count >= 2 ? col : 0;
}

template <class CoordTensor>
CUTE_HOST_DEVICE void
fill_physical_coord_map_copy_view_limited(
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
        row_coords[physical] = coord_get_0(coord);
        col_coords[physical] = coord_get_1(coord);
        ++physical;
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 3) {
    for (int i = 0; i < cute::size<0>(coord_tensor) && physical < limit; ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor) && physical < limit; ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor) && physical < limit; ++k) {
          auto logical = cute::make_coord(i, j, k);
          auto coord = coord_tensor(logical);
          row_coords[physical] = coord_get_0(coord);
          col_coords[physical] = coord_get_1(coord);
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
            row_coords[physical] = coord_get_0(coord);
            col_coords[physical] = coord_get_1(coord);
            ++physical;
          }
        }
      }
    }
  }
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
  using AtomNumThr = decltype(cute::size<0>(ValLayoutRef{}));
  using AtomNumVal = decltype(cute::size<1>(ValLayoutRef{}));
};

template <class TiledCopy, class ThrIdx>
struct LocalThrCopy;

template <class CopyAtom, class LayoutCopyTV, class ShapeTilerMN>
struct LocalTiledCopy : CopyAtom {
  using AtomNumThr = typename CopyAtom::AtomNumThr;
  using AtomNumVal = typename CopyAtom::AtomNumVal;
  using TilerMN = ShapeTilerMN;
  using TiledLayoutTV = LayoutCopyTV;
  using TiledNumThr = decltype(cute::size<0>(TiledLayoutTV{}));

  template <class DTensor>
  CUTE_HOST_DEVICE constexpr static auto
  Retile(DTensor&& dtensor) {
    constexpr int R = std::remove_cvref_t<DTensor>::rank;
    auto v = cute::size<0>(dtensor);
    auto frg_layout_mn = cute::upcast<TiledNumThr{} * v>(
        cute::right_inverse(TiledLayoutTV{}).with_shape(cute::shape(TilerMN{})));
    auto frg_layout_v = cute::zipped_divide(
        cute::logical_product(cute::make_layout(v), cute::right_inverse(frg_layout_mn)),
        cute::make_layout(AtomNumVal{}));
    auto t_tensor = cute::zipped_divide(
        dtensor,
        cute::prepend(cute::product_each(cute::shape(frg_layout_mn)), v));
    auto v_tensor = t_tensor.compose(frg_layout_v, cute::_);
    return v_tensor(cute::_, cute::append<R>(cute::Int<0>{}, cute::_));
  }

  template <class ThrIdx_, __CUTE_REQUIRES(cute::is_integral<ThrIdx_>::value)>
  CUTE_HOST_DEVICE constexpr static auto
  get_thread_slice(ThrIdx_ const& thr_idx) {
    return LocalThrCopy<LocalTiledCopy, ThrIdx_>{thr_idx};
  }
};

template <class TiledCopy, class ThrIdx>
struct LocalThrCopy {
  ThrIdx thr_idx_;

  template <class DTensor>
  CUTE_HOST_DEVICE auto
  retile_D(DTensor&& dtensor) const {
    auto thr_tensor =
        cute::make_tensor(static_cast<DTensor&&>(dtensor).data(), TiledCopy::Retile(dtensor.layout()));
    return thr_tensor;
  }
};

template <class CollectiveMainloop, class TiledMma>
CUTE_HOST_DEVICE constexpr auto
make_traced_p15_tiled_copy_sfa(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<ElementSFCompute>, ElementSFCompute>;
  auto collective_mainloop = CollectiveMainloop{};
  auto mma_copy = mma;
  using LayoutTV = decltype(collective_mainloop.get_layoutSFA_TV(mma_copy));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

template <class CollectiveMainloop, class TiledMma>
CUTE_HOST_DEVICE constexpr auto
make_traced_p15_tiled_copy_sfb(TiledMma const& mma) {
  using CopyAtom = LocalCopyAtom<cute::UniversalCopy<ElementSFCompute>, ElementSFCompute>;
  auto collective_mainloop = CollectiveMainloop{};
  auto mma_copy = mma;
  using LayoutTV = decltype(collective_mainloop.get_layoutSFB_TV(mma_copy));
  using TilerMN = decltype(cute::make_shape(cute::tile_size<1>(mma), cute::tile_size<2>(mma)));
  return LocalTiledCopy<CopyAtom, LayoutTV, TilerMN>{};
}

constexpr int kTracedP15ScaleFragmentCosizeA = 16;
constexpr int kTracedP15ScaleFragmentCosizeB = 32;

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_scale_A(
    SFATensor&& sfatensor,
    ThrMma& thread_mma);

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_scale_B(
    SFBTensor&& sfbtensor,
    ThrMma& thread_mma);

template <class CollectiveMainloop, class TiledMma>
CUTE_HOST_DEVICE std::uint32_t
load_traced_p15_sfa_word(
    TiledMma const& mma,
    int thread_idx,
    int m_fragment,
    std::uint32_t const* a_scale_words) {
  auto mma_copy = mma;
  auto thr_mma = mma_copy.get_thread_slice(thread_idx);
  auto ref_sfa = cute::make_identity_tensor(
      cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = partition_scale_A(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = make_traced_p15_tiled_copy_sfa<CollectiveMainloop>(mma_copy);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP15ScaleFragmentCosizeA];
  int scale_cols[kTracedP15ScaleFragmentCosizeA];
  fill_physical_coord_map_copy_view_limited(
      copy_view_sfa,
      kTracedP15ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  std::uint32_t packed_scale = 0u;
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = m_fragment * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        load_scale_byte(a_scale_words[scale_rows[physical]], scale_cols[physical]))
                    << (elem * 8);
  }
  return packed_scale;
}

template <class CollectiveMainloop, class TiledMma, class ScaleTensor>
CUTE_HOST_DEVICE std::uint32_t
load_traced_p15_sfa_word_from_tensor(
    TiledMma const& mma,
    int thread_idx,
    int m_fragment,
    ScaleTensor const& scale_tensor) {
  auto mma_copy = mma;
  auto thr_mma = mma_copy.get_thread_slice(thread_idx);
  auto ref_sfa = cute::make_identity_tensor(
      cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = partition_scale_A(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = make_traced_p15_tiled_copy_sfa<CollectiveMainloop>(mma_copy);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP15ScaleFragmentCosizeA];
  int scale_cols[kTracedP15ScaleFragmentCosizeA];
  fill_physical_coord_map_copy_view_limited(
      copy_view_sfa,
      kTracedP15ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  std::uint32_t packed_scale = 0u;
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = m_fragment * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        static_cast<std::uint8_t>(
                            scale_tensor(scale_rows[physical], scale_cols[physical], cute::Int<0>{})))
                    << (elem * 8);
  }
  return packed_scale;
}

template <class CollectiveMainloop, class TiledMma>
CUTE_HOST_DEVICE std::uint32_t
load_traced_p15_sfb_word(
    TiledMma const& mma,
    int thread_idx,
    int n_fragment,
    std::uint32_t const* b_scale_words) {
  auto mma_copy = mma;
  auto thr_mma = mma_copy.get_thread_slice(thread_idx);
  auto ref_sfb = cute::make_identity_tensor(
      cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = partition_scale_B(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = make_traced_p15_tiled_copy_sfb<CollectiveMainloop>(mma_copy);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP15ScaleFragmentCosizeB];
  int scale_cols[kTracedP15ScaleFragmentCosizeB];
  fill_physical_coord_map_copy_view_limited(
      copy_view_sfb,
      kTracedP15ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  std::uint32_t packed_scale = 0u;
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = n_fragment * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        load_scale_byte(b_scale_words[scale_rows[physical]], scale_cols[physical]))
                    << (elem * 8);
  }
  return packed_scale;
}

template <class CollectiveMainloop, class TiledMma, class ScaleTensor>
CUTE_HOST_DEVICE std::uint32_t
load_traced_p15_sfb_word_from_tensor(
    TiledMma const& mma,
    int thread_idx,
    int n_fragment,
    ScaleTensor const& scale_tensor) {
  auto mma_copy = mma;
  auto thr_mma = mma_copy.get_thread_slice(thread_idx);
  auto ref_sfb = cute::make_identity_tensor(
      cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = partition_scale_B(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = make_traced_p15_tiled_copy_sfb<CollectiveMainloop>(mma_copy);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP15ScaleFragmentCosizeB];
  int scale_cols[kTracedP15ScaleFragmentCosizeB];
  fill_physical_coord_map_copy_view_limited(
      copy_view_sfb,
      kTracedP15ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  std::uint32_t packed_scale = 0u;
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = n_fragment * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        static_cast<std::uint8_t>(
                            scale_tensor(scale_rows[physical], scale_cols[physical], cute::Int<0>{})))
                    << (elem * 8);
  }
  return packed_scale;
}

// --------------------------------------------------------------------------
// thrfrg_SFA / thrfrg_SFB
//
// Build the thread-fragment layout for scale factors.
// These accept the TiledMMA type (not the per-thread ThrMMA slice).
// --------------------------------------------------------------------------

// SFA atom layout: (T32, V) -> (M16, K64) from MMA_Traits SFALayout
using AtomLayoutSFA_TV = cute::Layout<
    cute::Shape<cute::Shape<cute::_2, cute::_2, cute::_8>, cute::_64>,
    cute::Stride<cute::Stride<cute::_8, cute::_0, cute::_1>, cute::_16>>;

// SFB atom layout: (T32, V) -> (N8, K64) from MMA_Traits SFBLayout
using AtomLayoutSFB_TV = cute::Layout<
    cute::Shape<cute::Shape<cute::_4, cute::_8>, cute::_64>,
    cute::Stride<cute::Stride<cute::_0, cute::_1>, cute::_8>>;

template <class SFTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
thrfrg_SFA(SFTensor&& sftensor, cute::TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  using namespace cute;
  CUTE_STATIC_ASSERT_V(rank(sftensor) >= Int<2>{});
  auto permutation_mnk = TiledPerm{};
  auto t_tile = make_tile(get<0>(permutation_mnk), _1{});
  auto tiled_sf = logical_divide(sftensor, t_tile);
  using AtomShape_MNK = typename Atom::Shape_MNK;
  auto atom_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(_1{}));
  auto tiled_atom_sf = zipped_divide(tiled_sf, atom_tile);
  auto tv_atom_sf = tiled_atom_sf.compose(AtomLayoutSFA_TV{}, _);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto thr_tile = make_tile(
      _,
      make_tile(
          make_layout(size<1>(thr_layout_vmnk)),
          make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_atom_sf, thr_tile);
}

template <class SFTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
thrfrg_SFB(SFTensor&& sftensor, cute::TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  using namespace cute;
  CUTE_STATIC_ASSERT_V(rank(sftensor) >= Int<2>{});
  auto permutation_mnk = TiledPerm{};
  auto t_tile = make_tile(get<1>(permutation_mnk), _1{});
  auto tiled_sf = logical_divide(sftensor, t_tile);
  using AtomShape_MNK = typename Atom::Shape_MNK;
  auto atom_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(_1{}));
  auto tiled_atom_sf = zipped_divide(tiled_sf, atom_tile);
  auto tv_atom_sf = tiled_atom_sf.compose(AtomLayoutSFB_TV{}, _);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto thr_tile = make_tile(
      _,
      make_tile(
          make_layout(size<2>(thr_layout_vmnk)),
          make_layout(size<3>(thr_layout_vmnk))));
  return zipped_divide(tv_atom_sf, thr_tile);
}

// --------------------------------------------------------------------------
// partition_fragment_SFA / SFB
//
// These take a scale smem tensor and a ThrMMA (per-thread slice).
// They also need the TiledMMA for the thrfrg layout.
// --------------------------------------------------------------------------

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_scale_A(SFATensor&& sfatensor, ThrMma& thread_mma) {
  auto thr_tensor =
      cute::make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk =
      cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_scale_B(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  auto thr_tensor =
      cute::make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk =
      cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
}

template <class SFATensor, class Atom, class TiledThr, class TiledPerm, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_fragment_SFA(
    SFATensor&& sfatensor,
    cute::TiledMMA<Atom, TiledThr, TiledPerm>& tiled_mma,
    ThrMma& thread_mma) {
  using namespace cute;
  auto thr_tensor = make_tensor(
      static_cast<SFATensor&&>(sfatensor).data(),
      thrfrg_SFA(sfatensor.layout(), tiled_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ElementSFLoad>(partition);
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm, class ThrMma>
CUTE_HOST_DEVICE constexpr auto
partition_fragment_SFB(
    SFBTensor&& sfbtensor,
    cute::TiledMMA<Atom, TiledThr, TiledPerm>& tiled_mma,
    ThrMma& thread_mma) {
  using namespace cute;
  auto thr_tensor = make_tensor(
      static_cast<SFBTensor&&>(sfbtensor).data(),
      thrfrg_SFB(sfbtensor.layout(), tiled_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  auto partition = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
  return make_fragment_like<ElementSFLoad>(partition);
}

// --------------------------------------------------------------------------
// transform_fragment_for_fp4_qmma
//
// FP4-specific variant of TRT-LLM's transform_fragment_for_qmma.
//
// For FP4 (SM120_16x8x64_TN_VS, SFVecSize=16):
//   - The MMA atom has K=64, needs 64/16 = 4 scale values per instruction
//   - All 4 scale values come from ONE int32 word (4 packed ue4m3 bytes)
//   - MMA_K=1 (one k-block per tile), so no cross-k-tile broadcasting
//   - RegNumSFA=1: one uint32_t register holding all 4 packed scale bytes
//
// The transformed fragment is rank-3 to match tCrA's rank-3 (MMA, MMA_M, MMA_K).
// When zipped and sliced to the atom level, the SFA part must satisfy
// mma_unpack's assertions: size(SFA)==64, cosize==4.
//
// Layout: (_32, num_mn, _4) with stride (_0, _4, _1)
//   Mode 0 (_32, stride 0): broadcast across 32-thread atom
//   Mode 1 (num_mn, stride 4): each MN partition gets its own 4-byte word
//   Mode 2 (_4, stride 1): the 4 ue4m3 bytes within the int32 word
// --------------------------------------------------------------------------

template <class Tensor>
CUTE_HOST_DEVICE constexpr auto
transform_fragment_for_fp4_qmma(Tensor&& tensor) {
  using namespace cute;
  CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
  auto old_ptr = tensor.data();
  auto new_ptr = recast_ptr<ElementSFCompute>(old_ptr);
  auto num_mn = size<1>(shape(tensor.layout()));
  CUTE_STATIC_ASSERT_V(size<2>(shape(tensor.layout())) == Int<1>{});
  auto new_layout = make_layout(
      make_shape(_32{}, num_mn, _4{}),
      make_stride(_0{}, _4{}, _1{}));
  return make_tensor(new_ptr, new_layout);
}

// --------------------------------------------------------------------------
// get_layoutSFA_TV / get_layoutSFB_TV
// --------------------------------------------------------------------------

template <class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
get_layoutSFA_TV(cute::TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  using namespace cute;
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), _1{}));
  auto thr_tensor = thrfrg_SFA(ref_A, mma);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = make_tile(
      _,
      make_tile(
          make_layout(
              make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
              make_stride(Int<1>{}, Int<0>{})),
          _));
  auto tv_sfa = thr_tensor.compose(atile, _);
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return tv_sfa.compose(thridx_2_thrid, _);
}

template <class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto
get_layoutSFB_TV(cute::TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  using namespace cute;
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
  auto thr_tensor = thrfrg_SFB(ref_B, mma);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto btile = make_tile(
      _,
      make_tile(
          make_layout(
              make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
              make_stride(Int<0>{}, Int<1>{})),
          _));
  auto tv_sfb = thr_tensor.compose(btile, _);
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return tv_sfb.compose(thridx_2_thrid, _);
}

}  // namespace nemotron::p15_scale
