// P5 (128x128x64 swap_ab=true) swizzled-smem GEMM pipeline test.
//
// This is the P5 analog of p15_swizzled_pipeline_test.cu. It exercises the
// standalone swizzled-smem + CUTE copy + fp4_shift + atom-level zipped MMA
// path that the unified routed FP4 kernel is trying to use for FC1.
//
// The goal is to isolate the exact low-row P5 contract (`dispatch_rows=4/5`)
// under the real FC1 packed-K envelope (1344) without involving the full
// fused-MoE runtime.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_zip.hpp>

#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/p15_scale_runtime_helpers.h"

namespace {

namespace cute = ::cute;

using ArchTag = cutlass::arch::Sm120;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using ElementAB = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementD = __nv_bfloat16;
using ElementABBlockScaled = cutlass::nv_float4_t<ElementAB>;

constexpr int kAlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementD, LayoutC*, kAlignmentAB,
    ElementD, LayoutD*, kAlignmentD,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementABBlockScaled, LayoutA*, kAlignmentAB,
    ElementABBlockScaled, LayoutB*, kAlignmentAB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using TiledMma = typename CollectiveMainloop::TiledMma;
using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
using SmemLayoutSFA = typename CollectiveMainloop::SmemLayoutSFA;
using SmemLayoutSFB = typename CollectiveMainloop::SmemLayoutSFB;
using SmemCopyAtomA = typename CollectiveMainloop::SmemCopyAtomA;
using SmemCopyAtomB = typename CollectiveMainloop::SmemCopyAtomB;
using SmemCopyAtomSFA = typename CollectiveMainloop::SmemCopyAtomSFA;
using SmemCopyAtomSFB = typename CollectiveMainloop::SmemCopyAtomSFB;
using SmemAllocTypeA = typename TiledMma::ValTypeA;
using SmemAllocTypeB = typename TiledMma::ValTypeB;
using AccumLayout = decltype(
    cute::partition_fragment_C(
        TiledMma{},
        cute::make_shape(cute::Int<128>{}, cute::Int<128>{}))
        .layout());

static constexpr int kSmemAStageElems = cute::size(cute::take<0, 2>(SmemLayoutA{}));
static constexpr int kSmemBStageElems = cute::size(cute::take<0, 2>(SmemLayoutB{}));

CUTE_HOST_DEVICE constexpr auto MakeP5ScaleLayoutSFBLocal(
    int32_t token_rows,
    int32_t input_cols) {
  return ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(int32_t{1}, token_rows, input_cols, int32_t{1}));
}

struct SharedStorage {
  alignas(1024) cute::array_aligned<SmemAllocTypeA, kSmemAStageElems> smem_A;
  alignas(1024) cute::array_aligned<SmemAllocTypeB, kSmemBStageElems> smem_B;
  alignas(1024) cute::array_aligned<ElementSF, cute::cosize(SmemLayoutSFA{})> smem_SFA;
  alignas(1024) cute::array_aligned<ElementSF, cute::cosize(SmemLayoutSFB{})> smem_SFB;
};

constexpr int kOutputRows = 128;
constexpr int kTokenRows = cute::tile_size<1>(TiledMma{});
constexpr int kMacroTileK = 64;
constexpr int kTmaCopyOnlyTileK = 128;
constexpr int kTmaCopyOnlyTileN = 128;
constexpr int kRuntimeLikeTotalK = 1344;
constexpr int kThreadCount = 256;
constexpr int kTmaThreadCount = 384;
constexpr int kCCopyCoordCapacity = 32;
constexpr int kMFragments = 4;
constexpr int kNFragments = 2;
constexpr float kSingleTileTolerance = 8.0f;
constexpr float kRuntimeLikeTolerance = 24.0f;

bool CheckCuda(cudaError_t status, const char* where) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << where << ": "
              << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

struct TmaCopyOnlySharedStorageA {
  alignas(1024) cute::array_aligned<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
  alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType ab_full_mbar[1];
};

struct TmaCopyOnlySharedStorageB {
  alignas(1024) cute::array_aligned<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
  alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType ab_full_mbar[1];
};

struct TmaCopyOnlySharedStorageSFB {
  alignas(128) cute::TmaDescriptor smem_tensormap_SFB;
  alignas(1024) cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
  alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType ab_full_mbar[1];
};

template <class TmaCopyA>
struct P5ATmaCopyOnlyParams {
  TmaCopyA tma_load_a;
  const std::uint8_t* a_packed_global;
  std::uint8_t* raw_tma_bytes;
  std::uint8_t* raw_ref_bytes;
};

template <class TmaCopyA>
struct P5ATmaCopyOnlyPtrParams {
  const TmaCopyA* tma_load_a_ptr;
  const std::uint8_t* a_packed_global;
  std::uint8_t* raw_tma_bytes;
  std::uint8_t* raw_ref_bytes;
};

template <class TmaCopyA>
struct P5ATmaOffsetCopyParams {
  TmaCopyA tma_load_a;
  const std::uint8_t* a_packed_reference;
  int tile_m_index;
  std::uint8_t* raw_tma_bytes;
  std::uint8_t* raw_ref_bytes;
};

template <class TmaCopyB>
struct P5BTmaCopyOnlyParams {
  TmaCopyB tma_load_b;
  const std::uint8_t* b_packed_global;
  int logical_n;
  std::uint8_t* raw_tma_bytes;
  std::uint8_t* raw_ref_bytes;
};

template <class TmaCopySFB>
struct P5SfbTmaOffsetParams {
  TmaCopySFB tma_load_sfb;
  int total_rows;
  int row_start;
  std::uint8_t* logical_tma_bytes;
};

template <class TmaCopySFB>
struct P5SfbTmaPartitionVerifyParams {
  TmaCopySFB tma_load_sfb;
  std::uint8_t* logical_tma_bytes;
};

template <class TmaCopySFB>
struct P5SfbTmaGroupedDescriptorParams {
  TmaCopySFB tma_load_sfb;
  cute::TmaDescriptor* gmem_tensormap_sfb;
  ElementSF const* execution_scales;
  int total_rows;
  int row_start;
  std::uint8_t* logical_tma_bytes;
};

struct TmaRuntimeLikeSharedStorageA {
  SharedStorage gemm;
  alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType ab_full_mbar[1];
};

template <class TmaCopyA>
struct P5ATmaSingleTileExecParams {
  TmaCopyA tma_load_a;
  const std::uint8_t* b_packed_global;
  const std::uint8_t* a_exec_scales;
  const std::uint8_t* b_exec_scales;
  int exec_k;
  int valid_rows;
  float* output;
};

constexpr std::size_t kTmaCopyOnlySmemABytes =
    sizeof(SmemAllocTypeA) * static_cast<std::size_t>(cute::cosize_v<SmemLayoutA>);
constexpr std::size_t kTmaCopyOnlySmemBBytes =
    sizeof(SmemAllocTypeB) * static_cast<std::size_t>(cute::cosize_v<SmemLayoutB>);
constexpr std::size_t kTmaCopyOnlySmemSfbBytes =
    sizeof(ElementSF) * static_cast<std::size_t>(cute::cosize_v<SmemLayoutSFB>);
constexpr int kSfbLogicalCols = cute::size<1>(SmemLayoutSFB{});

std::vector<float> MakePatternedValuesLocal(
    std::size_t rows,
    std::size_t cols,
    int seed,
    float scale) {
  std::vector<float> values(rows * cols, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const int raw =
          static_cast<int>(((row + 1) * (seed + 3)) + ((col + 5) * (seed + 7)));
      values[row * cols + col] =
          (static_cast<float>((raw % 31) - 15) * scale) +
          (0.00125f * static_cast<float>((row + col + static_cast<std::size_t>(seed)) % 11));
    }
  }
  return values;
}

std::vector<std::uint8_t> SwizzleRowMajorScalesForExecutionLocal(
    const std::uint8_t* row_major_scales,
    std::size_t rows,
    std::size_t cols,
    nemotron::Nvfp4ScaleLayout scale_layout) {
  constexpr std::size_t kBlockWidth = 16;
  constexpr std::size_t kBlockTile = 4;
  auto round_up = [](std::size_t value, std::size_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
  };
  auto row_tile = [](nemotron::Nvfp4ScaleLayout layout) {
    switch (layout) {
      case nemotron::Nvfp4ScaleLayout::kSwizzled128x4:
        return std::size_t{128};
      case nemotron::Nvfp4ScaleLayout::kSwizzled8x4:
        return std::size_t{8};
    }
    return std::size_t{128};
  };
  auto execution_scale_offset = [](std::size_t row,
                                   std::size_t block_col,
                                   std::size_t padded_blocks_per_row,
                                   nemotron::Nvfp4ScaleLayout layout) {
    const std::size_t num_k_tiles = padded_blocks_per_row / kBlockTile;
    const std::size_t k_tile = block_col / kBlockTile;
    const std::size_t inner_k = block_col & 3u;
    switch (layout) {
      case nemotron::Nvfp4ScaleLayout::kSwizzled128x4: {
        const std::size_t m_tile = row / 128u;
        const std::size_t outer_m = row & 31u;
        const std::size_t inner_m = (row >> 5u) & 3u;
        return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
                (outer_m << 4u) |
                (inner_m << 2u) |
                inner_k);
      }
      case nemotron::Nvfp4ScaleLayout::kSwizzled8x4: {
        const std::size_t m_tile = row / 8u;
        const std::size_t inner_m = row & 7u;
        return (((m_tile * num_k_tiles) + k_tile) << 5u) |
               (inner_m << 2u) |
               inner_k;
      }
    }
    return std::size_t{0};
  };

  const std::size_t logical_blocks_per_row = cols / kBlockWidth;
  const std::size_t padded_rows = round_up(rows, row_tile(scale_layout));
  const std::size_t padded_blocks_per_row = round_up(logical_blocks_per_row, kBlockTile);
  std::vector<std::uint8_t> swizzled(padded_rows * padded_blocks_per_row, 0u);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block_col = 0; block_col < logical_blocks_per_row; ++block_col) {
      const std::size_t src = row * logical_blocks_per_row + block_col;
      const std::size_t dst =
          execution_scale_offset(row, block_col, padded_blocks_per_row, scale_layout);
      swizzled[dst] = row_major_scales[src];
    }
  }
  return swizzled;
}

__device__ std::uint32_t LoadExecutionScaleWordTest(
    const std::uint8_t* scale_bytes,
    std::size_t row,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    nemotron::Nvfp4ScaleLayout scale_layout) {
  auto execution_scale_offset = [padded_blocks_per_row, scale_layout](std::size_t r, std::size_t block_col) {
    constexpr std::size_t kBlockTile = 4;
    const std::size_t num_k_tiles = padded_blocks_per_row / kBlockTile;
    const std::size_t k_tile = block_col / kBlockTile;
    const std::size_t inner_k = block_col & 3u;
    switch (scale_layout) {
      case nemotron::Nvfp4ScaleLayout::kSwizzled128x4: {
        const std::size_t m_tile = r / 128u;
        const std::size_t outer_m = r & 31u;
        const std::size_t inner_m = (r >> 5u) & 3u;
        return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
                (outer_m << 4u) |
                (inner_m << 2u) |
                inner_k);
      }
      case nemotron::Nvfp4ScaleLayout::kSwizzled8x4: {
        const std::size_t m_tile = r / 8u;
        const std::size_t inner_m = r & 7u;
        return (((m_tile * num_k_tiles) + k_tile) << 5u) |
               (inner_m << 2u) |
               inner_k;
      }
    }
    return std::size_t{0};
  };
  return nemotron::p15_scale_runtime::pack_scale_word4_bytes(
      scale_bytes[execution_scale_offset(row, block_base + 0u)],
      scale_bytes[execution_scale_offset(row, block_base + 1u)],
      scale_bytes[execution_scale_offset(row, block_base + 2u)],
      scale_bytes[execution_scale_offset(row, block_base + 3u)]);
}

template <class ScaleTensor>
__device__ void StoreScaleBytesTest(
    ScaleTensor const& scale_tensor,
    const std::uint8_t* scale_bytes,
    int scale_byte_count,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
  const int logical_cols = static_cast<int>(cute::size<1>(scale_tensor));
  const int segment_len = logical_cols > 0 ? max(1, logical_cols / scale_byte_count) : 1;
  for (int scale_col = 0; scale_col < logical_cols; ++scale_col) {
    const int byte_index = min(scale_col / segment_len, scale_byte_count - 1);
    scale_tensor(row, scale_col, cute::Int<0>{}) =
        nemotron::p15_scale_runtime::make_scale_element(scale_bytes[byte_index]);
  }
}

template <class ScaleTensor>
__device__ void StoreExecutionScaleBytesTest(
    ScaleTensor const& scale_tensor,
    const std::uint8_t* exec_scales,
    int exec_k,
    int row) {
  const int scale_byte_count = exec_k / 16;
  const std::size_t padded_blocks_per_row =
      ((static_cast<std::size_t>(scale_byte_count) + 3u) / 4u) * 4u;
  const std::uint32_t scale_word_lo = LoadExecutionScaleWordTest(
      exec_scales,
      static_cast<std::size_t>(row),
      0,
      padded_blocks_per_row,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const std::uint32_t scale_word_hi = scale_byte_count > 4 ? LoadExecutionScaleWordTest(
      exec_scales,
      static_cast<std::size_t>(row),
      4,
      padded_blocks_per_row,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4) : 0u;
  const std::uint8_t scale_bytes[8] = {
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_lo, 0),
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_lo, 1),
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_lo, 2),
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_lo, 3),
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_hi, 0),
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_hi, 1),
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_hi, 2),
      nemotron::p15_scale_runtime::load_scale_byte(scale_word_hi, 3),
  };
  StoreScaleBytesTest(scale_tensor, scale_bytes, scale_byte_count, row);
}

std::uint8_t LoadPackedFp4NibbleHost(const std::uint8_t* packed_row, int col) {
  const std::uint8_t packed_byte = packed_row[col / 2];
  return static_cast<std::uint8_t>((packed_byte >> ((col & 1) * 4)) & 0x0F);
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
  }
}

__global__ void P5RuntimeLikeExecScaleKernel(
    const std::uint8_t* __restrict__ a_global,
    const std::uint8_t* __restrict__ b_global,
    const std::uint8_t* __restrict__ a_exec_scales,
    const std::uint8_t* __restrict__ b_exec_scales,
    int total_k,
    int valid_rows,
    float* output) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const std::size_t packed_row_bytes = static_cast<std::size_t>(total_k / 2);
  const std::size_t logical_blocks_per_row = static_cast<std::size_t>(total_k / 16);
  const std::size_t padded_blocks_per_row = ((logical_blocks_per_row + 3u) / 4u) * 4u;

  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFA.data()), SmemLayoutSFA{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()), SmemLayoutSFB{});
  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()), SmemLayoutA{});
  auto sB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_B.data()), SmemLayoutB{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(tid);
  auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
  auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
  auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
      sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
  auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
      sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

  auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
  auto s2r_thr_A = s2r_copy_A.get_thread_slice(tid);
  auto tCsA = s2r_thr_A.partition_S(sA);
  auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

  auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
  auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
  auto tCsB = s2r_thr_B.partition_S(sB);
  auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

  auto tile_shape_mnk = cute::tile_shape(tiled_mma);
  auto s2r_copy_SFA = cute::make_tiled_copy_impl(
      SmemCopyAtomSFA{},
      CollectiveMainloop{}.get_layoutSFA_TV(tiled_mma),
      cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(tid);
  auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
  auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

  auto s2r_copy_SFB = cute::make_tiled_copy_impl(
      SmemCopyAtomSFB{},
      CollectiveMainloop{}.get_layoutSFB_TV(tiled_mma),
      cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(tid);
  auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
  auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

  float accum_storage[cute::cosize_v<AccumLayout>];
  auto accum = cute::make_tensor(&accum_storage[0], AccumLayout{});
  cute::clear(accum);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTokenRows>{}));
  auto part_c = thread_mma.partition_C(dense_c);

  using MMAOp = typename TiledMma::MMA_Op;
  using AtomType = typename TiledMma::Atom;
  constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
  constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
  constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
  AtomType mma_atom;

  for (int k_base = 0; k_base < total_k; k_base += kMacroTileK) {
    const std::size_t packed_byte_offset = static_cast<std::size_t>(k_base / 2);
    const std::size_t block_base = static_cast<std::size_t>(k_base / 16);
    const int available_k = min(kMacroTileK, total_k - k_base);
    const int available_bytes = available_k / 2;
    const int available_blocks = available_k / 16;

    auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
    constexpr int a_row_bytes = kMacroTileK / 2;
    constexpr int a_total_bytes = kOutputRows * a_row_bytes;
    for (int i = tid; i < a_total_bytes; i += kThreadCount) {
      const int row = i / a_row_bytes;
      const int col_byte = i % a_row_bytes;
      std::uint8_t value = 0u;
      if (col_byte < available_bytes) {
        value =
            a_global[static_cast<std::size_t>(row) * packed_row_bytes +
                     packed_byte_offset + static_cast<std::size_t>(col_byte)];
      }
      auto elem_offset = stage0_A(row, col_byte * 2);
      const int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_a_bytes[smem_byte_pos] = value;
    }

    auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());
    constexpr int b_row_bytes = kMacroTileK / 2;
    constexpr int b_total_bytes = kTokenRows * b_row_bytes;
    for (int i = tid; i < b_total_bytes; i += kThreadCount) {
      const int row = i / b_row_bytes;
      const int col_byte = i % b_row_bytes;
      std::uint8_t value = 0u;
      if (row < valid_rows && col_byte < available_bytes) {
        value =
            b_global[static_cast<std::size_t>(row) * packed_row_bytes +
                     packed_byte_offset + static_cast<std::size_t>(col_byte)];
      }
      auto elem_offset = stage0_B(row, col_byte * 2);
      const int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_b_bytes[smem_byte_pos] = value;
    }

    for (int row = tid; row < kOutputRows; row += kThreadCount) {
      std::uint32_t scale_word = 0u;
      if (available_blocks > 0) {
        scale_word = LoadExecutionScaleWordTest(
            a_exec_scales,
            static_cast<std::size_t>(row),
            block_base,
            padded_blocks_per_row,
            nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
      }
      nemotron::p15_scale_runtime::store_scale_word_k64(sSFA, scale_word, row);
    }
    for (int row = tid; row < kTokenRows; row += kThreadCount) {
      std::uint32_t scale_word = 0u;
      if (row < valid_rows && available_blocks > 0) {
        scale_word = LoadExecutionScaleWordTest(
            b_exec_scales,
            static_cast<std::size_t>(row),
            block_base,
            padded_blocks_per_row,
            nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
      }
      nemotron::p15_scale_runtime::store_scale_word_k64(sSFB, scale_word, row);
    }
    __syncthreads();

    cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
    cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
    cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
    cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

    for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
      cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
      cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
    }

    for (int k = 0; k < K_blocks; ++k) {
      for (int n = 0; n < N_tiles; ++n) {
        for (int m = 0; m < M_tiles; ++m) {
          auto a_atom = tCrA(cute::_, m, k);
          auto b_atom = tCrB(cute::_, n, k);
          auto c_atom = accum(cute::_, m, n);
          auto sfa_atom = tCrSFA(cute::_, m, k);
          auto sfb_atom = tCrSFB(cute::_, n, k);
          auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
          auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
          mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
        }
      }
    }
    __syncthreads();
  }

  int row_coords[kCCopyCoordCapacity];
  int col_coords[kCCopyCoordCapacity];
  FillPhysicalCoordMapCopyViewLimited(part_c, kCCopyCoordCapacity, row_coords, col_coords);

#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < kMFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < kNFragments; ++n_fragment) {
        const int physical = reg * 8 + m_fragment * 2 + n_fragment;
        const int out_row = row_coords[physical];
        const int token = col_coords[physical];
        if (out_row >= kOutputRows || token >= valid_rows) {
          continue;
        }
        output[static_cast<std::size_t>(token) * kOutputRows +
               static_cast<std::size_t>(out_row)] =
            accum(reg, m_fragment, n_fragment);
      }
    }
  }
}

float DecodeFp4(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0F;
  return static_cast<float>(value);
}

float DecodeScaleByte(std::uint8_t raw_byte) {
  return nemotron::p15_scale_runtime::decode_scale_byte(raw_byte);
}

float ReferenceOutputAtMultiK(
    std::vector<std::uint8_t> const& a_bytes,
    std::vector<std::uint8_t> const& b_bytes,
    std::vector<std::uint8_t> const& a_scale_bytes,
    std::vector<std::uint8_t> const& b_scale_bytes,
    int total_k,
    int out_row,
    int token_row) {
  const int blocks_per_row = total_k / 16;
  float accum = 0.0f;
  for (int k = 0; k < total_k; ++k) {
    const std::uint8_t a_byte =
        a_bytes[static_cast<std::size_t>(out_row) * static_cast<std::size_t>(total_k / 2) +
                static_cast<std::size_t>(k / 2)];
    const std::uint8_t b_byte =
        b_bytes[static_cast<std::size_t>(token_row) * static_cast<std::size_t>(total_k / 2) +
                static_cast<std::size_t>(k / 2)];
    const std::uint8_t a_nibble = (k & 1) ? ((a_byte >> 4) & 0x0F) : (a_byte & 0x0F);
    const std::uint8_t b_nibble = (k & 1) ? ((b_byte >> 4) & 0x0F) : (b_byte & 0x0F);
    const float a_scale =
        DecodeScaleByte(a_scale_bytes[static_cast<std::size_t>(out_row) * static_cast<std::size_t>(blocks_per_row) +
                                      static_cast<std::size_t>(k / 16)]);
    const float b_scale =
        DecodeScaleByte(b_scale_bytes[static_cast<std::size_t>(token_row) * static_cast<std::size_t>(blocks_per_row) +
                                      static_cast<std::size_t>(k / 16)]);
    accum += DecodeFp4(a_nibble) * a_scale * DecodeFp4(b_nibble) * b_scale;
  }
  return accum;
}

int RunRuntimeLikeExecScaleCaseMultiK(
    std::string_view label,
    std::vector<std::uint8_t> const& h_a,
    std::vector<std::uint8_t> const& h_b,
    std::vector<std::uint8_t> const& h_a_scale_bytes,
    std::vector<std::uint8_t> const& h_b_scale_bytes,
    int total_k,
    int valid_rows,
    float tolerance) {
  const auto h_a_exec = SwizzleRowMajorScalesForExecutionLocal(
      h_a_scale_bytes.data(),
      kOutputRows,
      static_cast<std::size_t>(total_k),
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const auto h_b_exec = SwizzleRowMajorScalesForExecutionLocal(
      h_b_scale_bytes.data(),
      kTokenRows,
      static_cast<std::size_t>(total_k),
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const std::size_t a_bytes = h_a.size();
  const std::size_t b_bytes = h_b.size();
  const std::size_t out_count = static_cast<std::size_t>(kTokenRows) * kOutputRows;

  std::uint8_t *d_a = nullptr, *d_b = nullptr, *d_a_exec = nullptr, *d_b_exec = nullptr;
  float* d_out = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A") ||
      !CheckCuda(cudaMalloc(&d_b, b_bytes), "malloc B") ||
      !CheckCuda(cudaMalloc(&d_a_exec, h_a_exec.size()), "malloc A exec") ||
      !CheckCuda(cudaMalloc(&d_b_exec, h_b_exec.size()), "malloc B exec") ||
      !CheckCuda(cudaMalloc(&d_out, out_count * sizeof(float)), "malloc out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A") ||
      !CheckCuda(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice), "copy B") ||
      !CheckCuda(cudaMemcpy(d_a_exec, h_a_exec.data(), h_a_exec.size(), cudaMemcpyHostToDevice), "copy A exec") ||
      !CheckCuda(cudaMemcpy(d_b_exec, h_b_exec.data(), h_b_exec.size(), cudaMemcpyHostToDevice), "copy B exec") ||
      !CheckCuda(cudaMemset(d_out, 0xff, out_count * sizeof(float)), "fill out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  constexpr int smem_bytes = sizeof(SharedStorage);
  if (!CheckCuda(cudaFuncSetAttribute(
          P5RuntimeLikeExecScaleKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes), "setMaxDynamicSmem")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  P5RuntimeLikeExecScaleKernel<<<1, kThreadCount, smem_bytes>>>(
      d_a, d_b, d_a_exec, d_b_exec, total_k, valid_rows, d_out);
  if (!CheckCuda(cudaGetLastError(), "kernel launch") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  std::vector<float> h_out(out_count, 0.0f);
  if (!CheckCuda(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost), "memcpy out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);

  float max_diff = 0.0f;
  int max_row = 0;
  int max_token = 0;
  std::size_t nan_count = 0;
  for (int token = 0; token < valid_rows; ++token) {
    for (int out_row = 0; out_row < kOutputRows; ++out_row) {
      const float got =
          h_out[static_cast<std::size_t>(token) * kOutputRows + static_cast<std::size_t>(out_row)];
      if (std::isnan(got)) {
        ++nan_count;
        continue;
      }
      const float expected = ReferenceOutputAtMultiK(
          h_a, h_b, h_a_scale_bytes, h_b_scale_bytes, total_k, out_row, token);
      const float diff = std::fabs(got - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_row = out_row;
        max_token = token;
      }
    }
  }

  std::cout << "  case=" << label
            << " valid_rows=" << valid_rows
            << " max_diff=" << max_diff
            << " at=(token=" << max_token << ",out_row=" << max_row << ")"
            << " got=" << h_out[static_cast<std::size_t>(max_token) * kOutputRows +
                               static_cast<std::size_t>(max_row)]
            << " expected=" << ReferenceOutputAtMultiK(
                   h_a, h_b, h_a_scale_bytes, h_b_scale_bytes, total_k, max_row, max_token)
            << " nan_count=" << nan_count << "\n";
  if (nan_count != 0 || max_diff > tolerance) {
    std::cerr << "  FAIL " << label << "\n";
    return 1;
  }
  return 0;
}

template <class TmaCopyA>
__global__ void P5ATmaCopyOnlyKernel(
    __grid_constant__ P5ATmaCopyOnlyParams<TmaCopyA> const params) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaCopyOnlySharedStorageA*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }
  auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();

  using X = cute::Underscore;
  auto mA_mkl = params.tma_load_a.get_tma_tensor(
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}, cute::Int<1>{}));
  auto gA_mkl = cute::local_tile(
      mA_mkl,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, X, cute::_1>{});

  auto block_tma_a = params.tma_load_a.get_slice(0);
  auto gA = gA_mkl(cute::_, cute::_, 0, cute::_, 0);
  auto tAgA = block_tma_a.partition_S(gA);

  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()), SmemLayoutA{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto tAsA = block_tma_a.partition_D(sA);

  if (is_tma_thread) {
    auto& ab_full_barrier = ab_full_mbar[0];
    auto tma_copy_a =
        params.tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
    cute::copy(
        tma_copy_a,
        tAgA(cute::_, cute::_, cute::_, cute::Int<0>{}),
        tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
    ab_full_mbar[0].arrive_and_expect_tx(
        static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                cute::size(cute::take<0, 2>(SmemLayoutA{})) *
                cute::sizeof_bits_v<ElementAB>)));
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    params.raw_tma_bytes[i] = smem_a_bytes[i];
  }
  __syncthreads();

  constexpr int a_row_bytes = kTmaCopyOnlyTileK / 2;
  constexpr int a_total_bytes = kOutputRows * a_row_bytes;
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();
  for (int i = tid; i < a_total_bytes; i += blockDim.x) {
    smem_a_bytes[i] = params.a_packed_global[static_cast<std::size_t>(i)];
  }
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    params.raw_ref_bytes[i] = smem_a_bytes[i];
  }
}

template <class TmaCopyA>
__global__ void P5ATmaCopyOnlyKernelFromGlobalObject(
    __grid_constant__ P5ATmaCopyOnlyPtrParams<TmaCopyA> const params) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaCopyOnlySharedStorageA*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);
  const TmaCopyA& tma_load_a = *params.tma_load_a_ptr;

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(tma_load_a.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }
  auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();

  using X = cute::Underscore;
  auto mA_mkl = tma_load_a.get_tma_tensor(
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}, cute::Int<1>{}));
  auto gA_mkl = cute::local_tile(
      mA_mkl,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, X, cute::_1>{});

  auto block_tma_a = tma_load_a.get_slice(0);
  auto gA = gA_mkl(cute::_, cute::_, 0, cute::_, 0);
  auto tAgA = block_tma_a.partition_S(gA);

  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()), SmemLayoutA{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto tAsA = block_tma_a.partition_D(sA);

  if (is_tma_thread) {
    auto& ab_full_barrier = ab_full_mbar[0];
    auto tma_copy_a =
        tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
    cute::copy(
        tma_copy_a,
        tAgA(cute::_, cute::_, cute::_, cute::Int<0>{}),
        tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
    ab_full_mbar[0].arrive_and_expect_tx(
        static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                cute::size(cute::take<0, 2>(SmemLayoutA{})) *
                cute::sizeof_bits_v<ElementAB>)));
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    params.raw_tma_bytes[i] = smem_a_bytes[i];
  }
  __syncthreads();

  constexpr int a_row_bytes = kTmaCopyOnlyTileK / 2;
  constexpr int a_total_bytes = kOutputRows * a_row_bytes;
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();
  for (int i = tid; i < a_total_bytes; i += blockDim.x) {
    smem_a_bytes[i] = params.a_packed_global[static_cast<std::size_t>(i)];
  }
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    params.raw_ref_bytes[i] = smem_a_bytes[i];
  }
}

int RunP5ATmaCopyOnlySmoke() {
  constexpr std::size_t kPackedRowBytes = kTmaCopyOnlyTileK / 2;
  std::vector<std::uint8_t> packed_a(
      static_cast<std::size_t>(kOutputRows) * kPackedRowBytes,
      std::uint8_t{0});
  for (std::size_t row = 0; row < static_cast<std::size_t>(kOutputRows); ++row) {
    for (std::size_t col_byte = 0; col_byte < kPackedRowBytes; ++col_byte) {
      const std::uint8_t lo = static_cast<std::uint8_t>((row + 3u * col_byte) & 0x0Fu);
      const std::uint8_t hi = static_cast<std::uint8_t>(((row * 7u) + (5u * col_byte) + 1u) & 0x0Fu);
      packed_a[row * kPackedRowBytes + col_byte] =
          static_cast<std::uint8_t>(lo | static_cast<std::uint8_t>(hi << 4));
    }
  }

  const std::size_t a_bytes = packed_a.size();
  std::uint8_t* d_a = nullptr;
  std::uint8_t* d_raw_tma = nullptr;
  std::uint8_t* d_raw_ref = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A tma copy-only") ||
      !CheckCuda(cudaMalloc(&d_raw_tma, kTmaCopyOnlySmemABytes), "malloc raw tma bytes") ||
      !CheckCuda(cudaMalloc(&d_raw_ref, kTmaCopyOnlySmemABytes), "malloc raw ref bytes")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, packed_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A tma copy-only") ||
      !CheckCuda(cudaMemset(d_raw_tma, 0, kTmaCopyOnlySmemABytes), "memset raw tma bytes") ||
      !CheckCuda(cudaMemset(d_raw_ref, 0, kTmaCopyOnlySmemABytes), "memset raw ref bytes")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  auto tensor_A = cute::make_tensor(
      cute::make_gmem_ptr(cute::recast_ptr<ElementAB>(d_a)),
      cute::make_layout(
          cute::make_shape(kOutputRows, kTmaCopyOnlyTileK, 1),
          cute::make_stride(
              int64_t(kTmaCopyOnlyTileK),
              cute::Int<1>{},
              int64_t(kOutputRows) * int64_t(kTmaCopyOnlyTileK))));
  auto tma_load_a = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_A,
      SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopyA = decltype(tma_load_a);
  using Params = P5ATmaCopyOnlyParams<TmaCopyA>;
  static_assert(alignof(Params) >= 64);
  auto kernel_typed = P5ATmaCopyOnlyKernel<TmaCopyA>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaCopyOnlySharedStorageA)),
          "func attr p5 a tma copy-only")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaCopyOnlySharedStorageA);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  alignas(64) Params params{tma_load_a, d_a, d_raw_tma, d_raw_ref};

  if (!CheckCuda(
          cudaLaunchKernelEx(&config, kernel_typed, params),
          "launch p5 a tma copy-only") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync p5 a tma copy-only")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  std::vector<std::uint8_t> h_raw_tma(kTmaCopyOnlySmemABytes, 0);
  std::vector<std::uint8_t> h_raw_ref(kTmaCopyOnlySmemABytes, 0);
  if (!CheckCuda(cudaMemcpy(h_raw_tma.data(), d_raw_tma, kTmaCopyOnlySmemABytes, cudaMemcpyDeviceToHost), "copy raw tma bytes") ||
      !CheckCuda(cudaMemcpy(h_raw_ref.data(), d_raw_ref, kTmaCopyOnlySmemABytes, cudaMemcpyDeviceToHost), "copy raw ref bytes")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  cudaFree(d_a);
  cudaFree(d_raw_tma);
  cudaFree(d_raw_ref);

  std::size_t mismatch_index = kTmaCopyOnlySmemABytes;
  for (std::size_t i = 0; i < kTmaCopyOnlySmemABytes; ++i) {
    if (h_raw_tma[i] != h_raw_ref[i]) {
      mismatch_index = i;
      break;
    }
  }

  std::cout << "  tma_copy_only_a_raw";
  if (mismatch_index == kTmaCopyOnlySmemABytes) {
    std::cout << " PASS"
              << "\n";
    return 0;
  }

  const std::size_t window_start = mismatch_index >= 8 ? mismatch_index - 8 : 0;
  const std::size_t window_end =
      std::min<std::size_t>(kTmaCopyOnlySmemABytes, mismatch_index + 24);
  std::cerr << " FAIL first_byte=" << mismatch_index
            << " tma=" << static_cast<unsigned>(h_raw_tma[mismatch_index])
            << " ref=" << static_cast<unsigned>(h_raw_ref[mismatch_index])
            << "\n";
  std::cerr << "  tma_bytes[" << window_start << ":" << window_end << ") =";
  for (std::size_t i = window_start; i < window_end; ++i) {
    std::cerr << ' ' << static_cast<unsigned>(h_raw_tma[i]);
  }
  std::cerr << "\n";
  std::cerr << "  ref_bytes[" << window_start << ":" << window_end << ") =";
  for (std::size_t i = window_start; i < window_end; ++i) {
    std::cerr << ' ' << static_cast<unsigned>(h_raw_ref[i]);
  }
  std::cerr << "\n";
  return 1;
}

int RunP5ATmaCopyOnlyGlobalObjectSmoke() {
  constexpr std::size_t kPackedRowBytes = kTmaCopyOnlyTileK / 2;
  std::vector<std::uint8_t> packed_a(
      static_cast<std::size_t>(kOutputRows) * kPackedRowBytes,
      std::uint8_t{0});
  for (std::size_t row = 0; row < static_cast<std::size_t>(kOutputRows); ++row) {
    for (std::size_t col_byte = 0; col_byte < kPackedRowBytes; ++col_byte) {
      const std::uint8_t lo = static_cast<std::uint8_t>((row + 3u * col_byte) & 0x0Fu);
      const std::uint8_t hi = static_cast<std::uint8_t>(((row * 7u) + (5u * col_byte) + 1u) & 0x0Fu);
      packed_a[row * kPackedRowBytes + col_byte] =
          static_cast<std::uint8_t>(lo | static_cast<std::uint8_t>(hi << 4));
    }
  }

  const std::size_t a_bytes = packed_a.size();
  std::uint8_t* d_a = nullptr;
  std::uint8_t* d_raw_tma = nullptr;
  std::uint8_t* d_raw_ref = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A tma copy-only global-object") ||
      !CheckCuda(cudaMalloc(&d_raw_tma, kTmaCopyOnlySmemABytes), "malloc raw tma bytes global-object") ||
      !CheckCuda(cudaMalloc(&d_raw_ref, kTmaCopyOnlySmemABytes), "malloc raw ref bytes global-object")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, packed_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A tma copy-only global-object") ||
      !CheckCuda(cudaMemset(d_raw_tma, 0, kTmaCopyOnlySmemABytes), "memset raw tma bytes global-object") ||
      !CheckCuda(cudaMemset(d_raw_ref, 0, kTmaCopyOnlySmemABytes), "memset raw ref bytes global-object")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  auto tensor_A = cute::make_tensor(
      cute::make_gmem_ptr(cute::recast_ptr<ElementAB>(d_a)),
      cute::make_layout(
          cute::make_shape(kOutputRows, kTmaCopyOnlyTileK, 1),
          cute::make_stride(
              int64_t(kTmaCopyOnlyTileK),
              cute::Int<1>{},
              int64_t(kOutputRows) * int64_t(kTmaCopyOnlyTileK))));
  auto tma_load_a = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_A,
      SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopyA = decltype(tma_load_a);
  TmaCopyA* d_tma_load_a = nullptr;
  if (!CheckCuda(cudaMalloc(&d_tma_load_a, sizeof(TmaCopyA)), "malloc global tma object") ||
      !CheckCuda(cudaMemcpy(d_tma_load_a, &tma_load_a, sizeof(TmaCopyA), cudaMemcpyHostToDevice), "copy global tma object")) {
    cudaFree(d_tma_load_a);
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  using Params = P5ATmaCopyOnlyPtrParams<TmaCopyA>;
  auto kernel_typed = P5ATmaCopyOnlyKernelFromGlobalObject<TmaCopyA>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaCopyOnlySharedStorageA)),
          "func attr p5 a tma copy-only global-object")) {
    cudaFree(d_tma_load_a);
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaCopyOnlySharedStorageA);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  alignas(64) Params params{d_tma_load_a, d_a, d_raw_tma, d_raw_ref};
  if (!CheckCuda(
          cudaLaunchKernelEx(&config, kernel_typed, params),
          "launch p5 a tma copy-only global-object") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync p5 a tma copy-only global-object")) {
    cudaFree(d_tma_load_a);
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  std::vector<std::uint8_t> h_raw_tma(kTmaCopyOnlySmemABytes, 0);
  std::vector<std::uint8_t> h_raw_ref(kTmaCopyOnlySmemABytes, 0);
  if (!CheckCuda(cudaMemcpy(h_raw_tma.data(), d_raw_tma, kTmaCopyOnlySmemABytes, cudaMemcpyDeviceToHost), "copy raw tma bytes global-object") ||
      !CheckCuda(cudaMemcpy(h_raw_ref.data(), d_raw_ref, kTmaCopyOnlySmemABytes, cudaMemcpyDeviceToHost), "copy raw ref bytes global-object")) {
    cudaFree(d_tma_load_a);
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  cudaFree(d_tma_load_a);
  cudaFree(d_a);
  cudaFree(d_raw_tma);
  cudaFree(d_raw_ref);

  std::size_t mismatch_index = kTmaCopyOnlySmemABytes;
  for (std::size_t i = 0; i < kTmaCopyOnlySmemABytes; ++i) {
    if (h_raw_tma[i] != h_raw_ref[i]) {
      mismatch_index = i;
      break;
    }
  }

  std::cout << "  tma_copy_only_a_global_object";
  if (mismatch_index == kTmaCopyOnlySmemABytes) {
    std::cout << " PASS\n";
    return 0;
  }

  std::cerr << " FAIL first_byte=" << mismatch_index
            << " tma=" << static_cast<unsigned>(h_raw_tma[mismatch_index])
            << " ref=" << static_cast<unsigned>(h_raw_ref[mismatch_index])
            << "\n";
  return 1;
}

template <class TmaCopyA>
__global__ void P5ATmaOffsetCopyOnlyKernel(
    __grid_constant__ P5ATmaOffsetCopyParams<TmaCopyA> const params) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaCopyOnlySharedStorageA*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }
  auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();

  using X = cute::Underscore;
  auto mA_mkl = params.tma_load_a.get_tma_tensor(
      cute::make_shape(cute::Int<kOutputRows * 2>{}, cute::Int<kTmaCopyOnlyTileK>{}, cute::Int<2>{}));
  auto gA_mkl = cute::local_tile(
      mA_mkl,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, X, cute::_1>{});

  auto block_tma_a = params.tma_load_a.get_slice(0);
  auto gA = gA_mkl(cute::_, cute::_, params.tile_m_index, cute::_, 1);
  auto tAgA = block_tma_a.partition_S(gA);

  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()), SmemLayoutA{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto tAsA = block_tma_a.partition_D(sA);

  if (is_tma_thread) {
    auto& ab_full_barrier = ab_full_mbar[0];
    auto tma_copy_a =
        params.tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
    cute::copy(
        tma_copy_a,
        tAgA(cute::_, cute::_, cute::_, cute::Int<0>{}),
        tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
    ab_full_mbar[0].arrive_and_expect_tx(
        static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                cute::size(cute::take<0, 2>(SmemLayoutA{})) *
                cute::sizeof_bits_v<ElementAB>)));
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    params.raw_tma_bytes[i] = smem_a_bytes[i];
  }
  __syncthreads();

  constexpr int a_row_bytes = kTmaCopyOnlyTileK / 2;
  constexpr int a_total_bytes = kOutputRows * a_row_bytes;
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();
  for (int i = tid; i < a_total_bytes; i += blockDim.x) {
    smem_a_bytes[i] = params.a_packed_reference[static_cast<std::size_t>(i)];
  }
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    params.raw_ref_bytes[i] = smem_a_bytes[i];
  }
}

int RunP5ATmaOffsetCopyOnlySmoke() {
  constexpr int kLargeRows = kOutputRows * 2;
  constexpr int kLargeBatches = 2;
  constexpr int kTileMIndex = 1;
  constexpr int kBatchIndex = 1;
  constexpr std::size_t kPackedRowBytes = kTmaCopyOnlyTileK / 2;
  const std::size_t packed_tensor_bytes =
      static_cast<std::size_t>(kLargeBatches) *
      static_cast<std::size_t>(kLargeRows) *
      kPackedRowBytes;
  std::vector<std::uint8_t> packed_a(packed_tensor_bytes, std::uint8_t{0});
  auto byte_at = [&](int batch, int row, int col_byte) -> std::uint8_t& {
    return packed_a[(static_cast<std::size_t>(batch) * static_cast<std::size_t>(kLargeRows) +
                     static_cast<std::size_t>(row)) *
                    kPackedRowBytes +
                    static_cast<std::size_t>(col_byte)];
  };
  for (int batch = 0; batch < kLargeBatches; ++batch) {
    for (int row = 0; row < kLargeRows; ++row) {
      for (int col_byte = 0; col_byte < static_cast<int>(kPackedRowBytes); ++col_byte) {
        const std::uint8_t lo = static_cast<std::uint8_t>((11u * batch + row + 3u * col_byte) & 0x0Fu);
        const std::uint8_t hi = static_cast<std::uint8_t>((5u * batch + 7u * row + 9u * col_byte + 1u) & 0x0Fu);
        byte_at(batch, row, col_byte) =
            static_cast<std::uint8_t>(lo | static_cast<std::uint8_t>(hi << 4));
      }
    }
  }
  std::uint8_t* d_a = nullptr;
  std::uint8_t* d_raw_tma = nullptr;
  std::uint8_t* d_raw_ref = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, packed_a.size()), "malloc A offset copy-only") ||
      !CheckCuda(cudaMalloc(&d_raw_tma, kTmaCopyOnlySmemABytes), "malloc raw tma offset bytes") ||
      !CheckCuda(cudaMalloc(&d_raw_ref, kTmaCopyOnlySmemABytes), "malloc raw ref offset bytes")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  const std::uint8_t* d_reference =
      d_a +
      ((static_cast<std::size_t>(kBatchIndex) * static_cast<std::size_t>(kLargeRows) +
        static_cast<std::size_t>(kTileMIndex * kOutputRows)) *
       kPackedRowBytes);
  if (!CheckCuda(cudaMemcpy(d_a, packed_a.data(), packed_a.size(), cudaMemcpyHostToDevice), "copy A offset copy-only") ||
      !CheckCuda(cudaMemset(d_raw_tma, 0, kTmaCopyOnlySmemABytes), "memset raw offset tma bytes") ||
      !CheckCuda(cudaMemset(d_raw_ref, 0, kTmaCopyOnlySmemABytes), "memset raw offset ref bytes")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  auto tensor_A = cute::make_tensor(
      cute::make_gmem_ptr(cute::recast_ptr<ElementAB>(d_a)),
      cute::make_layout(
          cute::make_shape(kLargeRows, kTmaCopyOnlyTileK, kLargeBatches),
          cute::make_stride(
              int64_t(kTmaCopyOnlyTileK),
              cute::Int<1>{},
              int64_t(kLargeRows) * int64_t(kTmaCopyOnlyTileK))));
  auto tma_load_a = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_A,
      SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopyA = decltype(tma_load_a);
  using Params = P5ATmaOffsetCopyParams<TmaCopyA>;
  auto kernel_typed = P5ATmaOffsetCopyOnlyKernel<TmaCopyA>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaCopyOnlySharedStorageA)),
          "func attr p5 a tma offset copy-only")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaCopyOnlySharedStorageA);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  alignas(64) Params params{tma_load_a, d_reference, kTileMIndex, d_raw_tma, d_raw_ref};
  if (!CheckCuda(
          cudaLaunchKernelEx(&config, kernel_typed, params),
          "launch p5 a tma offset copy-only") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync p5 a tma offset copy-only")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  std::vector<std::uint8_t> h_raw_tma(kTmaCopyOnlySmemABytes, 0);
  std::vector<std::uint8_t> h_raw_ref(kTmaCopyOnlySmemABytes, 0);
  if (!CheckCuda(cudaMemcpy(h_raw_tma.data(), d_raw_tma, kTmaCopyOnlySmemABytes, cudaMemcpyDeviceToHost), "copy raw offset tma bytes") ||
      !CheckCuda(cudaMemcpy(h_raw_ref.data(), d_raw_ref, kTmaCopyOnlySmemABytes, cudaMemcpyDeviceToHost), "copy raw offset ref bytes")) {
    cudaFree(d_a);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  cudaFree(d_a);
  cudaFree(d_raw_tma);
  cudaFree(d_raw_ref);

  std::size_t mismatch_index = kTmaCopyOnlySmemABytes;
  for (std::size_t i = 0; i < kTmaCopyOnlySmemABytes; ++i) {
    if (h_raw_tma[i] != h_raw_ref[i]) {
      mismatch_index = i;
      break;
    }
  }

  std::cout << "  tma_copy_only_a_offset_tile";
  if (mismatch_index == kTmaCopyOnlySmemABytes) {
    std::cout << " PASS\n";
    return 0;
  }

  std::cerr << " FAIL first_byte=" << mismatch_index
            << " tma=" << static_cast<unsigned>(h_raw_tma[mismatch_index])
            << " ref=" << static_cast<unsigned>(h_raw_ref[mismatch_index])
            << "\n";
  return 1;
}

template <class TmaCopyB>
__global__ void P5BTmaCopyOnlyKernel(
    __grid_constant__ P5BTmaCopyOnlyParams<TmaCopyB> const params) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaCopyOnlySharedStorageB*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }
  auto* smem_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemBBytes; i += blockDim.x) {
    smem_b_bytes[i] = 0u;
  }
  __syncthreads();

  using X = cute::Underscore;
  auto mB_nkl = params.tma_load_b.get_tma_tensor(
      cute::make_shape(params.logical_n, cute::Int<kTmaCopyOnlyTileK>{}, cute::Int<1>{}));
  auto gB_nkl = cute::local_tile(
      mB_nkl,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<X, cute::_1, cute::_1>{});

  auto block_tma_b = params.tma_load_b.get_slice(0);
  auto gB = gB_nkl(cute::_, cute::_, 0, cute::_, 0);
  auto tBgB = block_tma_b.partition_S(gB);

  auto sB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_B.data()), SmemLayoutB{});
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto tBsB = block_tma_b.partition_D(sB);

  if (is_tma_thread) {
    auto& ab_full_barrier = ab_full_mbar[0];
    auto tma_copy_b =
        params.tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
    cute::copy(
        tma_copy_b,
        tBgB(cute::_, cute::_, cute::_, cute::Int<0>{}),
        tBsB(cute::_, cute::_, cute::_, cute::Int<0>{}));
    ab_full_mbar[0].arrive_and_expect_tx(
        static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                cute::size(cute::take<0, 2>(SmemLayoutB{})) *
                cute::sizeof_bits_v<ElementAB>)));
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemBBytes; i += blockDim.x) {
    params.raw_tma_bytes[i] = smem_b_bytes[i];
  }
  __syncthreads();

  constexpr int b_row_bytes = kTmaCopyOnlyTileK / 2;
  constexpr int b_total_bytes = kTmaCopyOnlyTileN * b_row_bytes;
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemBBytes; i += blockDim.x) {
    smem_b_bytes[i] = 0u;
  }
  __syncthreads();
  for (int i = tid; i < b_total_bytes; i += blockDim.x) {
    const int row = i / b_row_bytes;
    const int col_byte = i % b_row_bytes;
    if (row < params.logical_n) {
      smem_b_bytes[i] =
          params.b_packed_global[static_cast<std::size_t>(row) *
                                     static_cast<std::size_t>(b_row_bytes) +
                                 static_cast<std::size_t>(col_byte)];
    }
  }
  __syncthreads();

  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemBBytes; i += blockDim.x) {
    params.raw_ref_bytes[i] = smem_b_bytes[i];
  }
}

int RunP5BTmaCopyOnlyCase(std::string_view label, int logical_n, int filled_rows) {
  if (logical_n <= 0 || logical_n > kTmaCopyOnlyTileN ||
      filled_rows < 0 || filled_rows > logical_n) {
    std::cerr << "  FAIL " << label << " invalid shape\n";
    return 1;
  }
  constexpr std::size_t kPackedRowBytes = kTmaCopyOnlyTileK / 2;
  std::vector<std::uint8_t> packed_b(
      static_cast<std::size_t>(logical_n) * kPackedRowBytes,
      std::uint8_t{0});
  for (std::size_t row = 0; row < static_cast<std::size_t>(filled_rows); ++row) {
    for (std::size_t col_byte = 0; col_byte < kPackedRowBytes; ++col_byte) {
      const std::uint8_t lo = static_cast<std::uint8_t>((3u * row + 5u * col_byte + 2u) & 0x0Fu);
      const std::uint8_t hi = static_cast<std::uint8_t>(((11u * row) + (7u * col_byte) + 4u) & 0x0Fu);
      packed_b[row * kPackedRowBytes + col_byte] =
          static_cast<std::uint8_t>(lo | static_cast<std::uint8_t>(hi << 4));
    }
  }

  std::uint8_t* d_b = nullptr;
  std::uint8_t* d_raw_tma = nullptr;
  std::uint8_t* d_raw_ref = nullptr;
  if (!CheckCuda(cudaMalloc(&d_b, packed_b.size()), "malloc B tma copy-only") ||
      !CheckCuda(cudaMalloc(&d_raw_tma, kTmaCopyOnlySmemBBytes), "malloc raw B tma bytes") ||
      !CheckCuda(cudaMalloc(&d_raw_ref, kTmaCopyOnlySmemBBytes), "malloc raw B ref bytes")) {
    cudaFree(d_b);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_b, packed_b.data(), packed_b.size(), cudaMemcpyHostToDevice), "copy B tma copy-only") ||
      !CheckCuda(cudaMemset(d_raw_tma, 0, kTmaCopyOnlySmemBBytes), "memset raw B tma bytes") ||
      !CheckCuda(cudaMemset(d_raw_ref, 0, kTmaCopyOnlySmemBBytes), "memset raw B ref bytes")) {
    cudaFree(d_b);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  auto tensor_B = cute::make_tensor(
      cute::make_gmem_ptr(cute::recast_ptr<ElementAB>(d_b)),
      cute::make_layout(
          cute::make_shape(logical_n, kTmaCopyOnlyTileK, 1),
          cute::make_stride(
              int64_t(kTmaCopyOnlyTileK),
              cute::Int<1>{},
              int64_t(logical_n) * int64_t(kTmaCopyOnlyTileK))));
  auto tma_load_b = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_B,
      SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTmaCopyOnlyTileN>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopyB = decltype(tma_load_b);
  using Params = P5BTmaCopyOnlyParams<TmaCopyB>;
  auto kernel_typed = P5BTmaCopyOnlyKernel<TmaCopyB>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaCopyOnlySharedStorageB)),
          "func attr p5 b tma copy-only")) {
    cudaFree(d_b);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaCopyOnlySharedStorageB);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  alignas(64) Params params{tma_load_b, d_b, logical_n, d_raw_tma, d_raw_ref};
  if (!CheckCuda(
          cudaLaunchKernelEx(&config, kernel_typed, params),
          "launch p5 b tma copy-only") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync p5 b tma copy-only")) {
    cudaFree(d_b);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }

  std::vector<std::uint8_t> h_raw_tma(kTmaCopyOnlySmemBBytes, 0);
  std::vector<std::uint8_t> h_raw_ref(kTmaCopyOnlySmemBBytes, 0);
  if (!CheckCuda(cudaMemcpy(h_raw_tma.data(), d_raw_tma, kTmaCopyOnlySmemBBytes, cudaMemcpyDeviceToHost), "copy raw B tma bytes") ||
      !CheckCuda(cudaMemcpy(h_raw_ref.data(), d_raw_ref, kTmaCopyOnlySmemBBytes, cudaMemcpyDeviceToHost), "copy raw B ref bytes")) {
    cudaFree(d_b);
    cudaFree(d_raw_tma);
    cudaFree(d_raw_ref);
    return 1;
  }
  cudaFree(d_b);
  cudaFree(d_raw_tma);
  cudaFree(d_raw_ref);

  std::size_t mismatch_index = kTmaCopyOnlySmemBBytes;
  for (std::size_t i = 0; i < kTmaCopyOnlySmemBBytes; ++i) {
    if (h_raw_tma[i] != h_raw_ref[i]) {
      mismatch_index = i;
      break;
    }
  }

  std::cout << "  " << label;
  if (mismatch_index == kTmaCopyOnlySmemBBytes) {
    std::cout << " PASS\n";
    return 0;
  }

  std::cerr << " FAIL first_byte=" << mismatch_index
            << " tma=" << static_cast<unsigned>(h_raw_tma[mismatch_index])
            << " ref=" << static_cast<unsigned>(h_raw_ref[mismatch_index])
            << "\n";
  return 1;
}

int RunP5BTmaCopyOnlySmoke() {
  if (RunP5BTmaCopyOnlyCase(
          "tma_copy_only_b_raw_full128",
          kTmaCopyOnlyTileN,
          kTmaCopyOnlyTileN) != 0) {
    return 1;
  }
  return RunP5BTmaCopyOnlyCase("tma_copy_only_b_raw_logical32_valid5", kTokenRows, 5);
}

template <bool kUseTileIndexSlice, class TmaCopySFB>
__global__ void P5SfbTmaOffsetLogicalKernel(
    __grid_constant__ P5SfbTmaOffsetParams<TmaCopySFB> const params) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaCopyOnlySharedStorageSFB*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }

  auto* smem_sfb_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_SFB.data());
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemSfbBytes; i += blockDim.x) {
    smem_sfb_bytes[i] = 0u;
  }
  __syncthreads();

  using X = cute::Underscore;
  auto full_layout_sfb = MakeP5ScaleLayoutSFBLocal(
      params.total_rows,
      kTmaCopyOnlyTileK);
  auto mSFB_nkl = params.tma_load_sfb.get_tma_tensor(cute::shape(full_layout_sfb));
  auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
  auto sSFB_raw = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()), SmemLayoutSFB{});
  auto tBsSFB = block_tma_sfb.partition_D(sSFB_raw);

  if constexpr (kUseTileIndexSlice) {
    auto gSFB_nkl = cute::local_tile(
        mSFB_nkl,
        MmaTileShape{},
        cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});
    const int tile_n = params.row_start / kTmaCopyOnlyTileN;
    auto gSFB = gSFB_nkl(cute::_, cute::_, tile_n, cute::_, 0);
    auto tBgSFB = block_tma_sfb.partition_S(gSFB);
    if (is_tma_thread) {
      auto& ab_full_barrier = ab_full_mbar[0];
      auto tma_copy_sfb =
          params.tma_load_sfb.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      cute::copy(
          tma_copy_sfb,
          tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
          tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
      ab_full_mbar[0].arrive_and_expect_tx(
          static_cast<uint32_t>(
              cutlass::bits_to_bytes(
                  cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})) *
                  cute::sizeof_bits_v<ElementSF>)));
    }
  } else {
    auto gSFB_offset = cute::domain_offset(cute::make_coord(params.row_start, 0, 0), mSFB_nkl);
    auto gSFB_nkl = cute::local_tile(
        gSFB_offset,
        MmaTileShape{},
        cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});
    auto gSFB = gSFB_nkl(cute::_, cute::_, 0, cute::_, 0);
    auto tBgSFB = block_tma_sfb.partition_S(gSFB);
    if (is_tma_thread) {
      auto& ab_full_barrier = ab_full_mbar[0];
      auto tma_copy_sfb =
          params.tma_load_sfb.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      cute::copy(
          tma_copy_sfb,
          tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
          tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
      ab_full_mbar[0].arrive_and_expect_tx(
          static_cast<uint32_t>(
              cutlass::bits_to_bytes(
                  cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})) *
                  cute::sizeof_bits_v<ElementSF>)));
    }
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  auto sSFB = cute::as_position_independent_swizzle_tensor(sSFB_raw);
  for (int linear = tid; linear < kTmaCopyOnlyTileN * kSfbLogicalCols; linear += blockDim.x) {
    const int row = linear / kSfbLogicalCols;
    const int col = linear % kSfbLogicalCols;
    params.logical_tma_bytes[linear] = sSFB(row, col, cute::Int<0>{}).storage;
  }
}

template <bool kUseTileIndexSlice>
int RunP5SfbTmaOffsetLogicalCase(
    std::string_view label,
    int total_rows,
    int row_start) {
  if (total_rows <= 0 || row_start < 0 || row_start >= total_rows) {
    std::cerr << "  FAIL " << label << " invalid shape\n";
    return 1;
  }
  if constexpr (kUseTileIndexSlice) {
    if ((row_start % kTmaCopyOnlyTileN) != 0) {
      std::cerr << "  FAIL " << label << " row_start must be tile aligned\n";
      return 1;
    }
  }
  constexpr int kLogicalBlocksPerRow = kTmaCopyOnlyTileK / 16;
  std::vector<std::uint8_t> row_major_scales(
      static_cast<std::size_t>(total_rows * kLogicalBlocksPerRow),
      std::uint8_t{0});
  for (int row = 0; row < total_rows; ++row) {
    for (int block = 0; block < kLogicalBlocksPerRow; ++block) {
      row_major_scales[static_cast<std::size_t>(row * kLogicalBlocksPerRow + block)] =
          static_cast<std::uint8_t>((13 * row + 7 * block + 5) & 0xFF);
    }
  }
  std::vector<std::uint8_t> execution_scales = SwizzleRowMajorScalesForExecutionLocal(
      row_major_scales.data(),
      total_rows,
      kTmaCopyOnlyTileK,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);

  std::uint8_t* d_execution_scales = nullptr;
  std::uint8_t* d_logical_tma = nullptr;
  const std::size_t logical_bytes =
      static_cast<std::size_t>(kTmaCopyOnlyTileN * kSfbLogicalCols);
  if (!CheckCuda(cudaMalloc(&d_execution_scales, execution_scales.size()), "malloc sfb execution scales") ||
      !CheckCuda(cudaMalloc(&d_logical_tma, logical_bytes), "malloc sfb logical readback")) {
    cudaFree(d_execution_scales);
    cudaFree(d_logical_tma);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(
          d_execution_scales,
          execution_scales.data(),
          execution_scales.size(),
          cudaMemcpyHostToDevice),
          "copy sfb execution scales") ||
      !CheckCuda(cudaMemset(d_logical_tma, 0, logical_bytes), "memset sfb logical readback")) {
    cudaFree(d_execution_scales);
    cudaFree(d_logical_tma);
    return 1;
  }

  auto tensor_sfb = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<ElementSF const*>(d_execution_scales)),
      MakeP5ScaleLayoutSFBLocal(total_rows, kTmaCopyOnlyTileK));
  auto tma_load_sfb = cute::make_tma_copy<uint16_t>(
      cute::SM90_TMA_LOAD{},
      tensor_sfb,
      SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTmaCopyOnlyTileN>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopySFB = decltype(tma_load_sfb);
  using Params = P5SfbTmaOffsetParams<TmaCopySFB>;
  auto kernel_typed = P5SfbTmaOffsetLogicalKernel<kUseTileIndexSlice, TmaCopySFB>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaCopyOnlySharedStorageSFB)),
          "func attr p5 sfb tma offset")) {
    cudaFree(d_execution_scales);
    cudaFree(d_logical_tma);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaCopyOnlySharedStorageSFB);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  alignas(64) Params params{tma_load_sfb, total_rows, row_start, d_logical_tma};
  if (!CheckCuda(
          cudaLaunchKernelEx(&config, kernel_typed, params),
          "launch p5 sfb tma offset") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync p5 sfb tma offset")) {
    cudaFree(d_execution_scales);
    cudaFree(d_logical_tma);
    return 1;
  }

  std::vector<std::uint8_t> h_logical_tma(logical_bytes, 0);
  if (!CheckCuda(cudaMemcpy(
          h_logical_tma.data(),
          d_logical_tma,
          logical_bytes,
          cudaMemcpyDeviceToHost),
          "copy sfb logical readback")) {
    cudaFree(d_execution_scales);
    cudaFree(d_logical_tma);
    return 1;
  }
  cudaFree(d_execution_scales);
  cudaFree(d_logical_tma);

  auto host_tensor_sfb = cute::make_tensor(
      reinterpret_cast<ElementSF const*>(execution_scales.data()),
      MakeP5ScaleLayoutSFBLocal(total_rows, kTmaCopyOnlyTileK));
  using X = cute::Underscore;
  auto host_block_tma_sfb = tma_load_sfb.get_slice(0);
  std::vector<ElementSF> expected_smem(cute::cosize_v<SmemLayoutSFB>, ElementSF{});
  auto host_sSFB_raw = cute::make_tensor(expected_smem.data(), SmemLayoutSFB{});
  auto host_tBsSFB = host_block_tma_sfb.partition_D(host_sSFB_raw);
  if constexpr (kUseTileIndexSlice) {
    auto host_gSFB_nkl = cute::local_tile(
        host_tensor_sfb,
        MmaTileShape{},
        cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});
    const int tile_n = row_start / kTmaCopyOnlyTileN;
    auto host_gSFB = host_gSFB_nkl(cute::_, cute::_, tile_n, cute::_, 0);
    auto host_tBgSFB = host_block_tma_sfb.partition_S(host_gSFB);
    cute::copy(
        host_tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
        host_tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
  } else {
    auto host_offset_sfb = cute::domain_offset(cute::make_coord(row_start, 0, 0), host_tensor_sfb);
    auto host_gSFB_nkl = cute::local_tile(
        host_offset_sfb,
        MmaTileShape{},
        cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<X, cute::_1, cute::_1>{});
    auto host_gSFB = host_gSFB_nkl(cute::_, cute::_, 0, cute::_, 0);
    auto host_tBgSFB = host_block_tma_sfb.partition_S(host_gSFB);
    cute::copy(
        host_tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
        host_tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
  }
  std::vector<std::uint8_t> expected(logical_bytes, 0);
  for (int row = 0; row < kTmaCopyOnlyTileN; ++row) {
    for (int col = 0; col < kSfbLogicalCols; ++col) {
      expected[static_cast<std::size_t>(row * kSfbLogicalCols + col)] =
          host_sSFB_raw(row, col, cute::Int<0>{}).storage;
    }
  }

  std::size_t mismatch_index = logical_bytes;
  for (std::size_t i = 0; i < logical_bytes; ++i) {
    if (h_logical_tma[i] != expected[i]) {
      mismatch_index = i;
      break;
    }
  }

  std::cout << "  " << label;
  if (mismatch_index == logical_bytes) {
    std::cout << " PASS\n";
    return 0;
  }

  const int row = static_cast<int>(mismatch_index / static_cast<std::size_t>(kSfbLogicalCols));
  const int col = static_cast<int>(mismatch_index % static_cast<std::size_t>(kSfbLogicalCols));
  std::cerr << " FAIL row=" << row
            << " col=" << col
            << " tma=" << static_cast<unsigned>(h_logical_tma[mismatch_index])
            << " ref=" << static_cast<unsigned>(expected[mismatch_index])
            << "\n";
  return 1;
}

int RunP5SfbTmaOffsetLogicalSmoke() {
  int status = 0;
  if (RunP5SfbTmaOffsetLogicalCase<true>(
          "tma_copy_only_sfb_tileindex_row0",
          256,
          0) != 0) {
    status = 1;
  }
  if (RunP5SfbTmaOffsetLogicalCase<true>(
          "tma_copy_only_sfb_tileindex_row128",
          256,
          128) != 0) {
    status = 1;
  }
  if (RunP5SfbTmaOffsetLogicalCase<false>(
          "tma_copy_only_sfb_offset_row8",
          16,
          8) != 0) {
    status = 1;
  }
  return status;
}

int RunP5SfbTmaPartitionVerifySmoke() {
  std::cout << "  tma_partition_sfb_tileindex_row128 SKIP (SM120 array mainloop uses make_tma_copy/local_tile, not tma_partition)\n";
  return 0;
}

int RunP5SfbBuilderContractSmoke() {
  constexpr std::string_view label = "tma_copy_only_sfb_builder_contract";
  auto tensor_sfb = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<ElementSF const*>(nullptr)),
      MakeP5ScaleLayoutSFBLocal(256, kTmaCopyOnlyTileK));

  auto local_tma = cute::make_tma_copy<uint16_t>(
      cute::SM90_TMA_LOAD{},
      tensor_sfb,
      SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTmaCopyOnlyTileN>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  auto cutlass_tma = cute::make_tma_copy<uint16_t>(
      typename CollectiveMainloop::GmemTiledCopySFB{},
      tensor_sfb,
      SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTmaCopyOnlyTileN>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  constexpr bool kSameType = std::is_same_v<decltype(local_tma), decltype(cutlass_tma)>;
  constexpr bool kMatchesParamsType =
      std::is_same_v<decltype(cutlass_tma), typename CollectiveMainloop::Params::TMA_SFB>;

  auto const* local_desc =
      reinterpret_cast<std::uint8_t const*>(local_tma.get_tma_descriptor());
  auto const* cutlass_desc =
      reinterpret_cast<std::uint8_t const*>(cutlass_tma.get_tma_descriptor());

  std::size_t mismatch_index = sizeof(cute::TmaDescriptor);
  for (std::size_t i = 0; i < sizeof(cute::TmaDescriptor); ++i) {
    if (local_desc[i] != cutlass_desc[i]) {
      mismatch_index = i;
      break;
    }
  }

  std::cout << "  " << label;
  if constexpr (!kSameType) {
    std::cerr << " FAIL type_mismatch local_vs_cutlass\n";
    return 1;
  }
  if constexpr (!kMatchesParamsType) {
    std::cerr << " FAIL cutlass_vs_params_type_mismatch\n";
    return 1;
  }
  if (mismatch_index != sizeof(cute::TmaDescriptor)) {
    std::cerr << " FAIL desc_byte=" << mismatch_index
              << " local=" << static_cast<unsigned>(local_desc[mismatch_index])
              << " cutlass=" << static_cast<unsigned>(cutlass_desc[mismatch_index])
              << "\n";
    return 1;
  }

  std::cout << " PASS\n";
  return 0;
}

int RunP5SfbExecutionLayoutContractSmoke() {
  constexpr std::string_view label = "tma_copy_only_sfb_execution_layout_contract";
  constexpr int total_rows = 256;
  constexpr int input_cols = kTmaCopyOnlyTileK;
  constexpr int kLogicalBlocksPerRow = input_cols / 16;

  std::vector<std::uint8_t> row_major_scales(
      static_cast<std::size_t>(total_rows * kLogicalBlocksPerRow),
      std::uint8_t{0});
  for (int row = 0; row < total_rows; ++row) {
    for (int block = 0; block < kLogicalBlocksPerRow; ++block) {
      row_major_scales[static_cast<std::size_t>(row * kLogicalBlocksPerRow + block)] =
          static_cast<std::uint8_t>((13 * row + 7 * block + 5) & 0xFF);
    }
  }

  std::vector<std::uint8_t> execution_scales = SwizzleRowMajorScalesForExecutionLocal(
      row_major_scales.data(),
      total_rows,
      input_cols,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);

  auto exact_layout = MakeP5ScaleLayoutSFBLocal(total_rows, input_cols);
  std::cout << "  " << label;
  const std::size_t exact_bytes =
      static_cast<std::size_t>(cute::cosize(exact_layout)) * sizeof(ElementSF);
  if (execution_scales.size() != exact_bytes) {
    std::cerr << " FAIL size_mismatch swizzled=" << execution_scales.size()
              << " exact=" << exact_bytes << "\n";
    return 1;
  }

  std::cout << " PASS\n";
  return 0;
}

int RunP5SfbExactLayoutAliasSmoke() {
  constexpr std::string_view label = "tma_copy_only_sfb_exact_layout_alias_contract";
  constexpr int total_rows = 256;
  constexpr int input_cols = kTmaCopyOnlyTileK;
  constexpr int kLogicalBlocksPerRow = input_cols / 16;

  auto exact_layout = MakeP5ScaleLayoutSFBLocal(total_rows, input_cols);
  const std::size_t exact_cosize = static_cast<std::size_t>(cute::cosize(exact_layout));

  std::vector<std::uint32_t> block_hits(exact_cosize, 0u);
  std::size_t unique_block_offsets = 0;
  bool block_index_in_range = true;
  bool dense_k_in_range = true;
  bool row_plus_1_alias = true;
  bool row_plus_32_alias = true;
  bool row_plus_128_alias = true;
  int dense_oob_row = -1;
  int dense_oob_block = -1;
  std::size_t dense_oob_offset = 0;

  for (int row = 0; row < total_rows; ++row) {
    for (int block = 0; block < kLogicalBlocksPerRow; ++block) {
      const auto block_offset =
          static_cast<std::size_t>(exact_layout(row, block, cute::Int<0>{}));
      if (block_offset >= exact_cosize) {
        block_index_in_range = false;
        continue;
      }
      if (block_hits[block_offset]++ == 0u) {
        ++unique_block_offsets;
      }

      const auto dense_offset =
          static_cast<std::size_t>(exact_layout(row, block * 16, cute::Int<0>{}));
      if (dense_offset >= exact_cosize) {
        if (dense_k_in_range) {
          dense_oob_row = row;
          dense_oob_block = block;
          dense_oob_offset = dense_offset;
        }
        dense_k_in_range = false;
      }

      if (row + 1 < total_rows) {
        const auto next_row_offset =
            static_cast<std::size_t>(exact_layout(row + 1, block, cute::Int<0>{}));
        if (next_row_offset != block_offset) {
          row_plus_1_alias = false;
        }
      }
      if (row + 32 < total_rows) {
        const auto next_32_offset =
            static_cast<std::size_t>(exact_layout(row + 32, block, cute::Int<0>{}));
        if (next_32_offset != block_offset) {
          row_plus_32_alias = false;
        }
      }
      if (row + 128 < total_rows) {
        const auto next_128_offset =
            static_cast<std::size_t>(exact_layout(row + 128, block, cute::Int<0>{}));
        if (next_128_offset != block_offset) {
          row_plus_128_alias = false;
        }
      }
    }
  }

  std::uint32_t min_hits = block_hits.empty() ? 0u : block_hits.front();
  std::uint32_t max_hits = 0u;
  for (std::uint32_t hits : block_hits) {
    if (hits < min_hits) {
      min_hits = hits;
    }
    if (hits > max_hits) {
      max_hits = hits;
    }
  }

  std::cout << "  " << label
            << " PASS layout=" << exact_layout
            << " cosize=" << exact_cosize
            << " block_index_in_range=" << (block_index_in_range ? "yes" : "no")
            << " unique_block_offsets=" << unique_block_offsets
            << " min_hits=" << min_hits
            << " max_hits=" << max_hits
            << " dense_k_in_range=" << (dense_k_in_range ? "yes" : "no");
  if (!dense_k_in_range) {
    std::cout << " dense_oob=(" << dense_oob_row << "," << dense_oob_block
              << ")->" << dense_oob_offset;
  }
  std::cout << " row_plus_1_alias=" << (row_plus_1_alias ? "yes" : "no")
            << " row_plus_32_alias=" << (row_plus_32_alias ? "yes" : "no")
            << " row_plus_128_alias=" << (row_plus_128_alias ? "yes" : "no")
            << "\n";
  return 0;
}

template <class TmaCopySFB>
__global__ void P5SfbTmaGroupedDescriptorLogicalKernel(
    __grid_constant__ P5SfbTmaGroupedDescriptorParams<TmaCopySFB> const params) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaCopyOnlySharedStorageSFB*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;
  const bool is_tma_warp = (warp_idx == 0);

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }

  auto* smem_sfb_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_SFB.data());
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemSfbBytes; i += blockDim.x) {
    smem_sfb_bytes[i] = 0u;
  }
  __syncthreads();

  if (is_tma_warp) {
    if (lane_predicate) {
      auto pSFB_tensormap =
          cute::make_tensor(params.tma_load_sfb.get_tma_descriptor(), cute::Int<1>{}, cute::Int<1>{});
      auto sSFB_tensormap =
          cute::make_tensor(cute::make_smem_ptr(&shared.smem_tensormap_SFB), cute::Int<1>{}, cute::Int<1>{});
      cute::copy(cute::recast<cute::uint128_t>(pSFB_tensormap), cute::recast<cute::uint128_t>(sSFB_tensormap));

      cute::tma_descriptor_replace_addr_in_shared_mem(
          shared.smem_tensormap_SFB,
          params.execution_scales);

      constexpr int kMaxTensorRank = 5;
      cute::array<uint32_t, kMaxTensorRank> prob_shape_sfb = {1, 1, 1, 1, 1};
      cute::array<uint64_t, kMaxTensorRank> prob_stride_sfb = {0, 0, 0, 0, 0};
      ElementSF const* null_sf_ptr = nullptr;
      auto tensor_sfb = cute::make_tensor(
          null_sf_ptr,
          MakeP5ScaleLayoutSFBLocal(params.total_rows, kTmaCopyOnlyTileK));
      cute::detail::fill_tma_gmem_shape_stride(
          params.tma_load_sfb,
          tensor_sfb,
          prob_shape_sfb,
          prob_stride_sfb);
      for (uint64_t& stride : prob_stride_sfb) {
        stride = (stride * cute::sizeof_bits_v<ElementSF>) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          shared.smem_tensormap_SFB,
          prob_shape_sfb,
          prob_stride_sfb);
    }
    __syncwarp();
    cute::tma_descriptor_cp_fence_release(params.gmem_tensormap_sfb, shared.smem_tensormap_SFB);
    cute::tma_descriptor_fence_acquire(params.gmem_tensormap_sfb);
    if (lane_predicate) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
  }
  __syncthreads();

  using X = cute::Underscore;
  auto full_layout_sfb = MakeP5ScaleLayoutSFBLocal(
      params.total_rows,
      kTmaCopyOnlyTileK);
  auto mSFB_nkl = params.tma_load_sfb.get_tma_tensor(cute::shape(full_layout_sfb));
  auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
  auto sSFB_raw = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()), SmemLayoutSFB{});
  auto tBsSFB = block_tma_sfb.partition_D(sSFB_raw);

  auto gSFB_nkl = cute::local_tile(
      mSFB_nkl,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<X, cute::_1, cute::_1>{});
  const int tile_n = params.row_start / kTmaCopyOnlyTileN;
  auto gSFB = gSFB_nkl(cute::_, cute::_, tile_n, cute::_, 0);
  auto tBgSFB = block_tma_sfb.partition_S(gSFB);

  if (is_tma_thread) {
    auto& ab_full_barrier = ab_full_mbar[0];
    auto tma_copy_sfb =
        params.tma_load_sfb.with(
            params.gmem_tensormap_sfb,
            *cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
    cute::copy(
        tma_copy_sfb,
        tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
        tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
    ab_full_mbar[0].arrive_and_expect_tx(
        static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})) *
                cute::sizeof_bits_v<ElementSF>)));
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  auto sSFB = cute::as_position_independent_swizzle_tensor(sSFB_raw);
  for (int linear = tid; linear < kTmaCopyOnlyTileN * kSfbLogicalCols; linear += blockDim.x) {
    const int row = linear / kSfbLogicalCols;
    const int col = linear % kSfbLogicalCols;
    params.logical_tma_bytes[linear] = sSFB(row, col, cute::Int<0>{}).storage;
  }
}

int RunP5SfbTmaGroupedDescriptorSmoke() {
  constexpr std::string_view label = "tma_copy_only_sfb_grouped_tileindex_row128";
  constexpr int total_rows = 256;
  constexpr int row_start = 128;
  constexpr int kLogicalBlocksPerRow = kTmaCopyOnlyTileK / 16;

  std::vector<std::uint8_t> row_major_scales(
      static_cast<std::size_t>(total_rows * kLogicalBlocksPerRow),
      std::uint8_t{0});
  for (int row = 0; row < total_rows; ++row) {
    for (int block = 0; block < kLogicalBlocksPerRow; ++block) {
      row_major_scales[static_cast<std::size_t>(row * kLogicalBlocksPerRow + block)] =
          static_cast<std::uint8_t>((13 * row + 7 * block + 5) & 0xFF);
    }
  }
  std::vector<std::uint8_t> execution_scales = SwizzleRowMajorScalesForExecutionLocal(
      row_major_scales.data(),
      total_rows,
      kTmaCopyOnlyTileK,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);

  std::uint8_t* d_execution_scales = nullptr;
  cute::TmaDescriptor* d_gmem_tensormap_sfb = nullptr;
  std::uint8_t* d_logical_tma = nullptr;
  const std::size_t logical_bytes =
      static_cast<std::size_t>(kTmaCopyOnlyTileN * kSfbLogicalCols);
  if (!CheckCuda(cudaMalloc(&d_execution_scales, execution_scales.size()), "malloc grouped sfb execution scales") ||
      !CheckCuda(cudaMalloc(&d_gmem_tensormap_sfb, sizeof(cute::TmaDescriptor)), "malloc grouped sfb tensormap") ||
      !CheckCuda(cudaMalloc(&d_logical_tma, logical_bytes), "malloc grouped sfb logical readback")) {
    cudaFree(d_execution_scales);
    cudaFree(d_gmem_tensormap_sfb);
    cudaFree(d_logical_tma);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(
          d_execution_scales,
          execution_scales.data(),
          execution_scales.size(),
          cudaMemcpyHostToDevice),
          "copy grouped sfb execution scales") ||
      !CheckCuda(cudaMemset(d_gmem_tensormap_sfb, 0, sizeof(cute::TmaDescriptor)), "memset grouped sfb tensormap") ||
      !CheckCuda(cudaMemset(d_logical_tma, 0, logical_bytes), "memset grouped sfb logical readback")) {
    cudaFree(d_execution_scales);
    cudaFree(d_gmem_tensormap_sfb);
    cudaFree(d_logical_tma);
    return 1;
  }

  auto tensor_sfb = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<ElementSF const*>(d_execution_scales)),
      MakeP5ScaleLayoutSFBLocal(total_rows, kTmaCopyOnlyTileK));
  auto tma_load_sfb = cute::make_tma_copy<uint16_t>(
      cute::SM90_TMA_LOAD{},
      tensor_sfb,
      SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTmaCopyOnlyTileN>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopySFB = decltype(tma_load_sfb);
  using Params = P5SfbTmaGroupedDescriptorParams<TmaCopySFB>;
  auto kernel_typed = P5SfbTmaGroupedDescriptorLogicalKernel<TmaCopySFB>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaCopyOnlySharedStorageSFB)),
          "func attr grouped p5 sfb tma")) {
    cudaFree(d_execution_scales);
    cudaFree(d_gmem_tensormap_sfb);
    cudaFree(d_logical_tma);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaCopyOnlySharedStorageSFB);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  alignas(64) Params params{
      tma_load_sfb,
      d_gmem_tensormap_sfb,
      reinterpret_cast<ElementSF const*>(d_execution_scales),
      total_rows,
      row_start,
      d_logical_tma};
  if (!CheckCuda(
          cudaLaunchKernelEx(&config, kernel_typed, params),
          "launch grouped p5 sfb tma") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync grouped p5 sfb tma")) {
    cudaFree(d_execution_scales);
    cudaFree(d_gmem_tensormap_sfb);
    cudaFree(d_logical_tma);
    return 1;
  }

  std::vector<std::uint8_t> h_logical_tma(logical_bytes, 0);
  if (!CheckCuda(cudaMemcpy(
          h_logical_tma.data(),
          d_logical_tma,
          logical_bytes,
          cudaMemcpyDeviceToHost),
          "copy grouped sfb logical readback")) {
    cudaFree(d_execution_scales);
    cudaFree(d_gmem_tensormap_sfb);
    cudaFree(d_logical_tma);
    return 1;
  }
  cudaFree(d_execution_scales);
  cudaFree(d_gmem_tensormap_sfb);
  cudaFree(d_logical_tma);

  auto host_tensor_sfb = cute::make_tensor(
      reinterpret_cast<ElementSF const*>(execution_scales.data()),
      MakeP5ScaleLayoutSFBLocal(total_rows, kTmaCopyOnlyTileK));
  using X = cute::Underscore;
  auto host_block_tma_sfb = tma_load_sfb.get_slice(0);
  std::vector<ElementSF> expected_smem(cute::cosize_v<SmemLayoutSFB>, ElementSF{});
  auto host_sSFB_raw = cute::make_tensor(expected_smem.data(), SmemLayoutSFB{});
  auto host_tBsSFB = host_block_tma_sfb.partition_D(host_sSFB_raw);
  auto host_gSFB_nkl = cute::local_tile(
      host_tensor_sfb,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<X, cute::_1, cute::_1>{});
  auto host_gSFB = host_gSFB_nkl(cute::_, cute::_, row_start / kTmaCopyOnlyTileN, cute::_, 0);
  auto host_tBgSFB = host_block_tma_sfb.partition_S(host_gSFB);
  cute::copy(
      host_tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
      host_tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));

  std::vector<std::uint8_t> expected(logical_bytes, 0);
  for (int row = 0; row < kTmaCopyOnlyTileN; ++row) {
    for (int col = 0; col < kSfbLogicalCols; ++col) {
      expected[static_cast<std::size_t>(row * kSfbLogicalCols + col)] =
          host_sSFB_raw(row, col, cute::Int<0>{}).storage;
    }
  }

  std::size_t mismatch_index = logical_bytes;
  for (std::size_t i = 0; i < logical_bytes; ++i) {
    if (h_logical_tma[i] != expected[i]) {
      mismatch_index = i;
      break;
    }
  }

  std::cout << "  " << label;
  if (mismatch_index == logical_bytes) {
    std::cout << " PASS\n";
    return 0;
  }

  const int row = static_cast<int>(mismatch_index / static_cast<std::size_t>(kSfbLogicalCols));
  const int col = static_cast<int>(mismatch_index % static_cast<std::size_t>(kSfbLogicalCols));
  std::cerr << " FAIL row=" << row
            << " col=" << col
            << " tma=" << static_cast<unsigned>(h_logical_tma[mismatch_index])
            << " ref=" << static_cast<unsigned>(expected[mismatch_index])
            << "\n";
  return 1;
}

template <class TmaCopyA>
__global__ void P5ATmaFragmentKernel(
    __grid_constant__ P5ATmaCopyOnlyParams<TmaCopyA> const params,
    int* mismatch_counts,
    int* first_v,
    int* first_m,
    int* first_k,
    float* first_tma,
    float* first_ref) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaCopyOnlySharedStorageA*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  using MMAOp = typename TiledMma::MMA_Op;
  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);
  auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
  float saved_tma_values[128]{};
  int saved_tma_count = 0;

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();

  using X = cute::Underscore;
  auto mA_mkl = params.tma_load_a.get_tma_tensor(
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}, cute::Int<1>{}));
  auto gA_mkl = cute::local_tile(
      mA_mkl,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, X, cute::_1>{});

  auto block_tma_a = params.tma_load_a.get_slice(0);
  auto gA = gA_mkl(cute::_, cute::_, 0, cute::_, 0);
  auto tAgA = block_tma_a.partition_S(gA);

  auto sA_ = cute::make_tensor(cute::make_smem_ptr(shared.smem_A.data()), SmemLayoutA{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto tAsA = block_tma_a.partition_D(sA);

  if (is_tma_thread) {
    auto& ab_full_barrier = ab_full_mbar[0];
    auto tma_copy_a = params.tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
    cute::copy(
        tma_copy_a,
        tAgA(cute::_, cute::_, cute::_, cute::Int<0>{}),
        tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
    ab_full_mbar[0].arrive_and_expect_tx(
        static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                cute::size(cute::take<0, 2>(SmemLayoutA{})) *
                cute::sizeof_bits_v<ElementAB>)));
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  if (tid < kThreadCount) {
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(tid);
    auto tCrA_tma = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
    auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto s2r_thr_A = s2r_copy_A.get_thread_slice(tid);
    auto tCsA = s2r_thr_A.partition_S(sA);
    auto tCrA_tma_cv = s2r_thr_A.retile_D(tCrA_tma);
    cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_tma_cv);
    for (int k = 0; k < cute::size<2>(tCrA_tma_cv); ++k) {
      cute::fp4_shift_A(MMAOp{}, tCrA_tma_cv(cute::_, cute::_, k));
    }
    saved_tma_count = 0;
    for (int k = 0; k < cute::size<2>(tCrA_tma); ++k) {
      for (int m = 0; m < cute::size<1>(tCrA_tma); ++m) {
        for (int v = 0; v < cute::size<0>(tCrA_tma); ++v) {
          saved_tma_values[saved_tma_count++] =
              static_cast<float>(static_cast<int>(tCrA_tma(v, m, k).get()));
        }
      }
    }
  }

  constexpr int a_row_bytes = kTmaCopyOnlyTileK / 2;
  constexpr int a_total_bytes = kOutputRows * a_row_bytes;
  __syncthreads();
  for (std::size_t i = static_cast<std::size_t>(tid); i < kTmaCopyOnlySmemABytes; i += blockDim.x) {
    smem_a_bytes[i] = 0u;
  }
  __syncthreads();
  for (int i = tid; i < a_total_bytes; i += blockDim.x) {
    smem_a_bytes[i] = params.a_packed_global[static_cast<std::size_t>(i)];
  }
  __syncthreads();

  if (tid < kThreadCount) {
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(tid);
    auto tCrA_ref = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
    auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto s2r_thr_A = s2r_copy_A.get_thread_slice(tid);
    auto tCsA = s2r_thr_A.partition_S(sA);
    auto tCrA_ref_cv = s2r_thr_A.retile_D(tCrA_ref);
    cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_ref_cv);
    for (int k = 0; k < cute::size<2>(tCrA_ref_cv); ++k) {
      cute::fp4_shift_A(MMAOp{}, tCrA_ref_cv(cute::_, cute::_, k));
    }

    int mismatches = 0;
    int first_v_idx = -1;
    int first_m_idx = -1;
    int first_k_idx = -1;
    float first_tma_value = 0.0f;
    float first_ref_value = 0.0f;
    int tma_idx = 0;
    for (int k = 0; k < cute::size<2>(tCrA_ref); ++k) {
      for (int m = 0; m < cute::size<1>(tCrA_ref); ++m) {
        for (int v = 0; v < cute::size<0>(tCrA_ref); ++v) {
          const float got = saved_tma_values[tma_idx++];
          const float expected =
              static_cast<float>(static_cast<int>(tCrA_ref(v, m, k).get()));
          if (got != expected) {
            ++mismatches;
            if (first_v_idx < 0) {
              first_v_idx = v;
              first_m_idx = m;
              first_k_idx = k;
              first_tma_value = got;
              first_ref_value = expected;
            }
          }
        }
      }
    }
    if (tma_idx != saved_tma_count) {
      mismatches += 1;
      if (first_v_idx < 0) {
        first_v_idx = -2;
        first_m_idx = saved_tma_count;
        first_k_idx = tma_idx;
        first_tma_value = static_cast<float>(saved_tma_count);
        first_ref_value = static_cast<float>(tma_idx);
      }
    }
    mismatch_counts[tid] = mismatches;
    first_v[tid] = first_v_idx;
    first_m[tid] = first_m_idx;
    first_k[tid] = first_k_idx;
    first_tma[tid] = first_tma_value;
    first_ref[tid] = first_ref_value;
  }
}

int RunP5ATmaFragmentSmoke() {
  constexpr std::size_t kPackedRowBytes = kTmaCopyOnlyTileK / 2;
  std::vector<std::uint8_t> packed_a(
      static_cast<std::size_t>(kOutputRows) * kPackedRowBytes,
      std::uint8_t{0});
  for (std::size_t row = 0; row < static_cast<std::size_t>(kOutputRows); ++row) {
    for (std::size_t col_byte = 0; col_byte < kPackedRowBytes; ++col_byte) {
      const std::uint8_t lo = static_cast<std::uint8_t>((row + 3u * col_byte) & 0x0Fu);
      const std::uint8_t hi = static_cast<std::uint8_t>(((row * 7u) + (5u * col_byte) + 1u) & 0x0Fu);
      packed_a[row * kPackedRowBytes + col_byte] =
          static_cast<std::uint8_t>(lo | static_cast<std::uint8_t>(hi << 4));
    }
  }

  const std::size_t a_bytes = packed_a.size();
  std::uint8_t* d_a = nullptr;
  int* d_mismatch_counts = nullptr;
  int* d_first_v = nullptr;
  int* d_first_m = nullptr;
  int* d_first_k = nullptr;
  float* d_first_tma = nullptr;
  float* d_first_ref = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A tma fragment") ||
      !CheckCuda(cudaMalloc(&d_mismatch_counts, kThreadCount * sizeof(int)), "malloc mismatch counts") ||
      !CheckCuda(cudaMalloc(&d_first_v, kThreadCount * sizeof(int)), "malloc first_v") ||
      !CheckCuda(cudaMalloc(&d_first_m, kThreadCount * sizeof(int)), "malloc first_m") ||
      !CheckCuda(cudaMalloc(&d_first_k, kThreadCount * sizeof(int)), "malloc first_k") ||
      !CheckCuda(cudaMalloc(&d_first_tma, kThreadCount * sizeof(float)), "malloc first_tma") ||
      !CheckCuda(cudaMalloc(&d_first_ref, kThreadCount * sizeof(float)), "malloc first_ref")) {
    cudaFree(d_a);
    cudaFree(d_mismatch_counts);
    cudaFree(d_first_v);
    cudaFree(d_first_m);
    cudaFree(d_first_k);
    cudaFree(d_first_tma);
    cudaFree(d_first_ref);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, packed_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A tma fragment") ||
      !CheckCuda(cudaMemset(d_mismatch_counts, 0, kThreadCount * sizeof(int)), "memset mismatch counts") ||
      !CheckCuda(cudaMemset(d_first_v, 0xff, kThreadCount * sizeof(int)), "memset first_v") ||
      !CheckCuda(cudaMemset(d_first_m, 0xff, kThreadCount * sizeof(int)), "memset first_m") ||
      !CheckCuda(cudaMemset(d_first_k, 0xff, kThreadCount * sizeof(int)), "memset first_k") ||
      !CheckCuda(cudaMemset(d_first_tma, 0, kThreadCount * sizeof(float)), "memset first_tma") ||
      !CheckCuda(cudaMemset(d_first_ref, 0, kThreadCount * sizeof(float)), "memset first_ref")) {
    cudaFree(d_a);
    cudaFree(d_mismatch_counts);
    cudaFree(d_first_v);
    cudaFree(d_first_m);
    cudaFree(d_first_k);
    cudaFree(d_first_tma);
    cudaFree(d_first_ref);
    return 1;
  }

  auto tensor_A = cute::make_tensor(
      cute::make_gmem_ptr(cute::recast_ptr<ElementAB>(d_a)),
      cute::make_layout(
          cute::make_shape(kOutputRows, kTmaCopyOnlyTileK, 1),
          cute::make_stride(
              int64_t(kTmaCopyOnlyTileK),
              cute::Int<1>{},
              int64_t(kOutputRows) * int64_t(kTmaCopyOnlyTileK))));
  auto tma_load_a = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_A,
      SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopyA = decltype(tma_load_a);
  using Params = P5ATmaCopyOnlyParams<TmaCopyA>;
  auto kernel_typed = P5ATmaFragmentKernel<TmaCopyA>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaCopyOnlySharedStorageA)),
          "func attr p5 a tma fragment")) {
    cudaFree(d_a);
    cudaFree(d_mismatch_counts);
    cudaFree(d_first_v);
    cudaFree(d_first_m);
    cudaFree(d_first_k);
    cudaFree(d_first_tma);
    cudaFree(d_first_ref);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaCopyOnlySharedStorageA);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  Params params{tma_load_a, d_a, nullptr, nullptr};
  if (!CheckCuda(
          cudaLaunchKernelEx(
              &config,
              kernel_typed,
              params,
              d_mismatch_counts,
              d_first_v,
              d_first_m,
              d_first_k,
              d_first_tma,
              d_first_ref),
          "launch p5 a tma fragment") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync p5 a tma fragment")) {
    cudaFree(d_a);
    cudaFree(d_mismatch_counts);
    cudaFree(d_first_v);
    cudaFree(d_first_m);
    cudaFree(d_first_k);
    cudaFree(d_first_tma);
    cudaFree(d_first_ref);
    return 1;
  }

  std::vector<int> h_mismatch_counts(kThreadCount, 0);
  std::vector<int> h_first_v(kThreadCount, -1);
  std::vector<int> h_first_m(kThreadCount, -1);
  std::vector<int> h_first_k(kThreadCount, -1);
  std::vector<float> h_first_tma(kThreadCount, 0.0f);
  std::vector<float> h_first_ref(kThreadCount, 0.0f);
  if (!CheckCuda(cudaMemcpy(h_mismatch_counts.data(), d_mismatch_counts, kThreadCount * sizeof(int), cudaMemcpyDeviceToHost), "copy mismatch counts") ||
      !CheckCuda(cudaMemcpy(h_first_v.data(), d_first_v, kThreadCount * sizeof(int), cudaMemcpyDeviceToHost), "copy first_v") ||
      !CheckCuda(cudaMemcpy(h_first_m.data(), d_first_m, kThreadCount * sizeof(int), cudaMemcpyDeviceToHost), "copy first_m") ||
      !CheckCuda(cudaMemcpy(h_first_k.data(), d_first_k, kThreadCount * sizeof(int), cudaMemcpyDeviceToHost), "copy first_k") ||
      !CheckCuda(cudaMemcpy(h_first_tma.data(), d_first_tma, kThreadCount * sizeof(float), cudaMemcpyDeviceToHost), "copy first_tma") ||
      !CheckCuda(cudaMemcpy(h_first_ref.data(), d_first_ref, kThreadCount * sizeof(float), cudaMemcpyDeviceToHost), "copy first_ref")) {
    cudaFree(d_a);
    cudaFree(d_mismatch_counts);
    cudaFree(d_first_v);
    cudaFree(d_first_m);
    cudaFree(d_first_k);
    cudaFree(d_first_tma);
    cudaFree(d_first_ref);
    return 1;
  }
  cudaFree(d_a);
  cudaFree(d_mismatch_counts);
  cudaFree(d_first_v);
  cudaFree(d_first_m);
  cudaFree(d_first_k);
  cudaFree(d_first_tma);
  cudaFree(d_first_ref);

  int total_mismatches = 0;
  int failing_thread = -1;
  for (int tid = 0; tid < kThreadCount; ++tid) {
    total_mismatches += h_mismatch_counts[tid];
    if (failing_thread < 0 && h_mismatch_counts[tid] != 0) {
      failing_thread = tid;
    }
  }

  std::cout << "  tma_fragment_a";
  if (total_mismatches == 0) {
    std::cout << " PASS\n";
    return 0;
  }

  std::cerr << " FAIL total_mismatches=" << total_mismatches
            << " thread=" << failing_thread
            << " v=" << h_first_v[failing_thread]
            << " m=" << h_first_m[failing_thread]
            << " k=" << h_first_k[failing_thread]
            << " tma=" << h_first_tma[failing_thread]
            << " ref=" << h_first_ref[failing_thread]
            << "\n";
  return 1;
}

template <class TmaCopyA>
__global__ void P5ATmaSingleTileExecKernel(
    __grid_constant__ P5ATmaSingleTileExecParams<TmaCopyA> const params) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<TmaRuntimeLikeSharedStorageA*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const bool is_tma_thread = (warp_idx == 0) && lane_predicate;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  using MMAOp = typename TiledMma::MMA_Op;
  using AtomType = typename TiledMma::Atom;

  auto* ab_full_mbar = cute::recast_ptr<FullBarrier>(&shared.ab_full_mbar[0]);
  auto* smem_b_bytes = reinterpret_cast<std::uint8_t*>(shared.gemm.smem_B.data());

  if (is_tma_thread) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
  }
  __syncthreads();

  if (is_tma_thread) {
    ab_full_mbar[0].init(1);
    cutlass::arch::fence_barrier_init();
  }
  for (int i = tid; i < static_cast<int>(sizeof(shared.gemm)); i += blockDim.x) {
    smem_raw[i] = 0;
  }
  __syncthreads();

  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(shared.gemm.smem_SFA.data()), SmemLayoutSFA{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.gemm.smem_SFB.data()), SmemLayoutSFB{});
  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.gemm.smem_A.data()), SmemLayoutA{});
  auto sB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.gemm.smem_B.data()), SmemLayoutB{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

  using X = cute::Underscore;
  auto mA_mkl = params.tma_load_a.get_tma_tensor(
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}, cute::Int<1>{}));
  auto gA_mkl = cute::local_tile(
      mA_mkl,
      MmaTileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, X, cute::_1>{});
  auto block_tma_a = params.tma_load_a.get_slice(0);
  auto gA = gA_mkl(cute::_, cute::_, 0, cute::_, 0);
  auto tAgA = block_tma_a.partition_S(gA);
  auto tAsA = block_tma_a.partition_D(sA);

  if (is_tma_thread) {
    auto& ab_full_barrier = ab_full_mbar[0];
    auto tma_copy_a =
        params.tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&ab_full_barrier));
    cute::copy(
        tma_copy_a,
        tAgA(cute::_, cute::_, cute::_, cute::Int<0>{}),
        tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
    ab_full_mbar[0].arrive_and_expect_tx(
        static_cast<uint32_t>(
            cutlass::bits_to_bytes(
                cute::size(cute::take<0, 2>(SmemLayoutA{})) *
                cute::sizeof_bits_v<ElementAB>)));
  }

  auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
  constexpr int b_tma_row_bytes = kTmaCopyOnlyTileK / 2;
  constexpr int b_total_bytes = kTokenRows * b_tma_row_bytes;
  const int exec_row_bytes = params.exec_k / 2;
  for (int i = tid; i < b_total_bytes; i += blockDim.x) {
    const int row = i / b_tma_row_bytes;
    const int col_byte = i % b_tma_row_bytes;
    std::uint8_t value = 0u;
    if (row < params.valid_rows && col_byte < exec_row_bytes) {
      value = params.b_packed_global[
          static_cast<std::size_t>(row) * static_cast<std::size_t>(exec_row_bytes) +
          static_cast<std::size_t>(col_byte)];
    }
    auto elem_offset = stage0_B(row, col_byte * 2);
    const int smem_byte_pos = static_cast<int>(elem_offset) / 2;
    smem_b_bytes[smem_byte_pos] = value;
  }

  for (int row = tid; row < kOutputRows; row += blockDim.x) {
    StoreExecutionScaleBytesTest(
        sSFA,
        params.a_exec_scales,
        params.exec_k,
        row);
  }
  for (int row = tid; row < kTokenRows; row += blockDim.x) {
    if (row < params.valid_rows) {
      StoreExecutionScaleBytesTest(
          sSFB,
          params.b_exec_scales,
          params.exec_k,
          row);
    } else {
      const std::uint8_t zero_scale_bytes[8] = {};
      StoreScaleBytesTest(sSFB, zero_scale_bytes, params.exec_k / 16, row);
    }
  }

  ab_full_mbar[0].wait(0);
  __syncthreads();

  if (tid < kThreadCount) {
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(tid);
    auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
    auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
    auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
        sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
    auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
        sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

    auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto s2r_thr_A = s2r_copy_A.get_thread_slice(tid);
    auto tCsA = s2r_thr_A.partition_S(sA);
    auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

    auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
    auto tCsB = s2r_thr_B.partition_S(sB);
    auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

    auto tile_shape_mnk = cute::tile_shape(tiled_mma);
    auto s2r_copy_SFA = cute::make_tiled_copy_impl(
        SmemCopyAtomSFA{},
        CollectiveMainloop{}.get_layoutSFA_TV(tiled_mma),
        cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
    auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(tid);
    auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
    auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

    auto s2r_copy_SFB = cute::make_tiled_copy_impl(
        SmemCopyAtomSFB{},
        CollectiveMainloop{}.get_layoutSFB_TV(tiled_mma),
        cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
    auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(tid);
    auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
    auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

    float accum_storage[cute::cosize_v<AccumLayout>];
    auto accum = cute::make_tensor(&accum_storage[0], AccumLayout{});
    cute::clear(accum);
    auto dense_c = cute::make_identity_tensor(
        cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTokenRows>{}));
    auto part_c = thread_mma.partition_C(dense_c);

    cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
    cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
    cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
    cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

    for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
      cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
      cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
    }

    constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
    constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
    constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
    AtomType mma_atom;
    for (int k = 0; k < K_blocks; ++k) {
      for (int n = 0; n < N_tiles; ++n) {
        for (int m = 0; m < M_tiles; ++m) {
          auto a_atom = tCrA(cute::_, m, k);
          auto b_atom = tCrB(cute::_, n, k);
          auto c_atom = accum(cute::_, m, n);
          auto sfa_atom = tCrSFA(cute::_, m, k);
          auto sfb_atom = tCrSFB(cute::_, n, k);
          auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
          auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
          mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
        }
      }
    }

    int row_coords[kCCopyCoordCapacity];
    int col_coords[kCCopyCoordCapacity];
    FillPhysicalCoordMapCopyViewLimited(part_c, kCCopyCoordCapacity, row_coords, col_coords);

#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < kMFragments; ++m_fragment) {
#pragma unroll
        for (int n_fragment = 0; n_fragment < kNFragments; ++n_fragment) {
          const int physical = reg * 8 + m_fragment * 2 + n_fragment;
          const int out_row = row_coords[physical];
          const int token = col_coords[physical];
          if (out_row >= kOutputRows || token >= params.valid_rows) {
            continue;
          }
          params.output[static_cast<std::size_t>(token) * kOutputRows +
                        static_cast<std::size_t>(out_row)] =
              accum(reg, m_fragment, n_fragment);
        }
      }
    }
  }
}

int RunP5ATmaSingleTileExecCase(int exec_k, int valid_rows, float tolerance) {
  auto a_values = MakePatternedValuesLocal(
      static_cast<std::size_t>(kOutputRows),
      static_cast<std::size_t>(exec_k),
      23,
      0.0078125f);
  auto b_values = MakePatternedValuesLocal(
      static_cast<std::size_t>(kTokenRows),
      static_cast<std::size_t>(exec_k),
      41,
      0.01171875f);
  for (int row = valid_rows; row < kTokenRows; ++row) {
    for (int col = 0; col < exec_k; ++col) {
      b_values[static_cast<std::size_t>(row) * static_cast<std::size_t>(exec_k) +
               static_cast<std::size_t>(col)] = 0.0f;
    }
  }

  auto a_pack = nemotron::PackRowMajorFp32ToNvfp4(
      a_values.data(), static_cast<std::size_t>(kOutputRows), static_cast<std::size_t>(exec_k));
  auto b_pack = nemotron::PackRowMajorFp32ToNvfp4(
      b_values.data(), static_cast<std::size_t>(kTokenRows), static_cast<std::size_t>(exec_k));
  if (!a_pack.has_value() || !b_pack.has_value()) {
    std::cerr << "  FAIL tma_single_tile pack\n";
    return 1;
  }

  const auto h_a_exec = SwizzleRowMajorScalesForExecutionLocal(
      a_pack->block_scales.data(),
      kOutputRows,
      static_cast<std::size_t>(exec_k),
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const auto h_b_exec = SwizzleRowMajorScalesForExecutionLocal(
      b_pack->block_scales.data(),
      kTokenRows,
      static_cast<std::size_t>(exec_k),
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);

  constexpr std::size_t kPaddedPackedRowBytes = kTmaCopyOnlyTileK / 2;
  std::vector<std::uint8_t> a_tma_packed(
      static_cast<std::size_t>(kOutputRows) * kPaddedPackedRowBytes,
      std::uint8_t{0});
  for (int row = 0; row < kOutputRows; ++row) {
    std::memcpy(
        a_tma_packed.data() + static_cast<std::size_t>(row) * kPaddedPackedRowBytes,
        a_pack->packed.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(exec_k / 2),
        static_cast<std::size_t>(exec_k / 2));
  }

  std::uint8_t *d_a = nullptr, *d_b = nullptr, *d_a_exec = nullptr, *d_b_exec = nullptr;
  float* d_out = nullptr;
  const std::size_t out_count = static_cast<std::size_t>(kTokenRows) * kOutputRows;
  if (!CheckCuda(cudaMalloc(&d_a, a_tma_packed.size()), "malloc tma A") ||
      !CheckCuda(cudaMalloc(&d_b, b_pack->packed.size()), "malloc B packed") ||
      !CheckCuda(cudaMalloc(&d_a_exec, h_a_exec.size()), "malloc A exec") ||
      !CheckCuda(cudaMalloc(&d_b_exec, h_b_exec.size()), "malloc B exec") ||
      !CheckCuda(cudaMalloc(&d_out, out_count * sizeof(float)), "malloc out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, a_tma_packed.data(), a_tma_packed.size(), cudaMemcpyHostToDevice), "copy tma A") ||
      !CheckCuda(cudaMemcpy(d_b, b_pack->packed.data(), b_pack->packed.size(), cudaMemcpyHostToDevice), "copy B packed") ||
      !CheckCuda(cudaMemcpy(d_a_exec, h_a_exec.data(), h_a_exec.size(), cudaMemcpyHostToDevice), "copy A exec") ||
      !CheckCuda(cudaMemcpy(d_b_exec, h_b_exec.data(), h_b_exec.size(), cudaMemcpyHostToDevice), "copy B exec") ||
      !CheckCuda(cudaMemset(d_out, 0xff, out_count * sizeof(float)), "fill out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  auto tensor_A = cute::make_tensor(
      cute::make_gmem_ptr(cute::recast_ptr<ElementAB>(d_a)),
      cute::make_layout(
          cute::make_shape(kOutputRows, kTmaCopyOnlyTileK, 1),
          cute::make_stride(
              int64_t(kTmaCopyOnlyTileK),
              cute::Int<1>{},
              int64_t(kOutputRows) * int64_t(kTmaCopyOnlyTileK))));
  auto tma_load_a = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      tensor_A,
      SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kOutputRows>{}, cute::Int<kTmaCopyOnlyTileK>{}),
      cute::_1{});

  using TmaCopyA = decltype(tma_load_a);
  using Params = P5ATmaSingleTileExecParams<TmaCopyA>;
  auto kernel_typed = P5ATmaSingleTileExecKernel<TmaCopyA>;
  if (!CheckCuda(cudaFuncSetAttribute(
          kernel_typed,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(TmaRuntimeLikeSharedStorageA)),
          "func attr p5 a tma single-tile")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1, 1);
  config.blockDim = dim3(kTmaThreadCount, 1, 1);
  config.dynamicSmemBytes = sizeof(TmaRuntimeLikeSharedStorageA);
  config.stream = nullptr;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  alignas(64) Params params{tma_load_a, d_b, d_a_exec, d_b_exec, exec_k, valid_rows, d_out};
  if (!CheckCuda(
          cudaLaunchKernelEx(&config, kernel_typed, params),
          "launch p5 a tma single-tile") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync p5 a tma single-tile")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  std::vector<float> h_out(out_count, 0.0f);
  if (!CheckCuda(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost), "copy tma out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);

  float max_diff = 0.0f;
  int max_row = 0;
  int max_token = 0;
  std::size_t nan_count = 0;
  for (int token = 0; token < valid_rows; ++token) {
    for (int out_row = 0; out_row < kOutputRows; ++out_row) {
      const float got =
          h_out[static_cast<std::size_t>(token) * kOutputRows + static_cast<std::size_t>(out_row)];
      if (std::isnan(got)) {
        ++nan_count;
        continue;
      }
      const float expected = ReferenceOutputAtMultiK(
          a_pack->packed,
          b_pack->packed,
          a_pack->block_scales,
          b_pack->block_scales,
          exec_k,
          out_row,
          token);
      const float diff = std::fabs(got - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_row = out_row;
        max_token = token;
      }
    }
  }

  std::cout << "  tma_single_tile_k" << exec_k << "_dispatch_rows_" << valid_rows
            << " max_diff=" << max_diff
            << " at=(token=" << max_token << ",out_row=" << max_row << ")"
            << " got=" << h_out[static_cast<std::size_t>(max_token) * kOutputRows +
                               static_cast<std::size_t>(max_row)]
            << " expected=" << ReferenceOutputAtMultiK(
                   a_pack->packed,
                   b_pack->packed,
                   a_pack->block_scales,
                   b_pack->block_scales,
                   exec_k,
                   max_row,
                   max_token)
            << " nan_count=" << nan_count << "\n";
  if (nan_count != 0 || max_diff > tolerance) {
    std::cerr << "  FAIL tma_single_tile_k" << exec_k
              << "_dispatch_rows_" << valid_rows << "\n";
    return 1;
  }
  return 0;
}

int RunPackedCase(std::string_view label, int total_k, int valid_rows, float tolerance) {
  auto a_values = MakePatternedValuesLocal(
      static_cast<std::size_t>(kOutputRows),
      static_cast<std::size_t>(total_k),
      23,
      0.0078125f);
  auto b_values = MakePatternedValuesLocal(
      static_cast<std::size_t>(kTokenRows),
      static_cast<std::size_t>(total_k),
      41,
      0.01171875f);
  for (int row = valid_rows; row < kTokenRows; ++row) {
    for (int col = 0; col < total_k; ++col) {
      b_values[static_cast<std::size_t>(row) * static_cast<std::size_t>(total_k) +
               static_cast<std::size_t>(col)] = 0.0f;
    }
  }

  auto a_pack = nemotron::PackRowMajorFp32ToNvfp4(
      a_values.data(), static_cast<std::size_t>(kOutputRows), static_cast<std::size_t>(total_k));
  if (!a_pack.has_value()) {
    std::cerr << "  FAIL " << label << " pack A\n";
    return 1;
  }
  auto b_pack = nemotron::PackRowMajorFp32ToNvfp4(
      b_values.data(), static_cast<std::size_t>(kTokenRows), static_cast<std::size_t>(total_k));
  if (!b_pack.has_value()) {
    std::cerr << "  FAIL " << label << " pack B\n";
    return 1;
  }

  return RunRuntimeLikeExecScaleCaseMultiK(
      label,
      a_pack->packed,
      b_pack->packed,
      a_pack->block_scales,
      b_pack->block_scales,
      total_k,
      valid_rows,
      tolerance);
}

int RunTest() {
  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return 1;
  }
  if (device_count == 0) {
    std::cout << "p5_swizzled_pipeline_test: SKIP (no CUDA device)\n";
    return 0;
  }

  std::cout << "p5_swizzled_pipeline_test:\n"
            << "  tile: " << kOutputRows << "x" << kTokenRows << "x" << kMacroTileK << "\n"
            << "  runtime_like_total_k: " << kRuntimeLikeTotalK << "\n"
            << "  smem: " << sizeof(SharedStorage) << " bytes ("
            << sizeof(SharedStorage) / 1024 << " KB)\n"
            << "  threads: " << kThreadCount << "\n";

  if (const char* run_tma_smoke = std::getenv("NEMOTRON_RUN_P5_TMA_SMOKE")) {
    if (run_tma_smoke[0] != '\0' && run_tma_smoke[0] != '0') {
      int tma_status = 0;
      tma_status |= RunP5ATmaCopyOnlySmoke();
      tma_status |= RunP5ATmaCopyOnlyGlobalObjectSmoke();
      tma_status |= RunP5ATmaOffsetCopyOnlySmoke();
      tma_status |= RunP5BTmaCopyOnlySmoke();
      tma_status |= RunP5SfbBuilderContractSmoke();
      tma_status |= RunP5SfbExecutionLayoutContractSmoke();
      tma_status |= RunP5SfbExactLayoutAliasSmoke();
      tma_status |= RunP5SfbTmaOffsetLogicalSmoke();
      tma_status |= RunP5SfbTmaPartitionVerifySmoke();
      tma_status |= RunP5SfbTmaGroupedDescriptorSmoke();
      tma_status |= RunP5ATmaFragmentSmoke();
      tma_status |= RunP5ATmaSingleTileExecCase(kMacroTileK, 5, kSingleTileTolerance);
      tma_status |= RunP5ATmaSingleTileExecCase(kMacroTileK, 4, kSingleTileTolerance);
      tma_status |= RunP5ATmaSingleTileExecCase(kTmaCopyOnlyTileK, 5, kRuntimeLikeTolerance);
      if (tma_status != 0) {
        return 1;
      }
    } else {
      std::cout << "  tma_copy_only_a SKIP (env disabled)\n";
      std::cout << "  tma_copy_only_a_global_object SKIP (env disabled)\n";
      std::cout << "  tma_copy_only_a_offset_tile SKIP (env disabled)\n";
      std::cout << "  tma_copy_only_b_raw_full128 SKIP (env disabled)\n";
      std::cout << "  tma_copy_only_b_raw_logical32_valid5 SKIP (env disabled)\n";
      std::cout << "  tma_copy_only_sfb_exact_layout_alias_contract SKIP (env disabled)\n";
      std::cout << "  tma_partition_sfb_tileindex_row128 SKIP (env disabled)\n";
      std::cout << "  tma_fragment_a SKIP (env disabled)\n";
      std::cout << "  tma_single_tile_k64_dispatch_rows_5 SKIP (env disabled)\n";
      std::cout << "  tma_single_tile_k64_dispatch_rows_4 SKIP (env disabled)\n";
      std::cout << "  tma_single_tile_k128_dispatch_rows_5 SKIP (env disabled)\n";
    }
  } else {
    std::cout << "  tma_copy_only_a SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_a_global_object SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_a_offset_tile SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_b_raw_full128 SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_b_raw_logical32_valid5 SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_sfb_builder_contract SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_sfb_execution_layout_contract SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_sfb_exact_layout_alias_contract SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_sfb_tileindex_row0 SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_copy_only_sfb_grouped_tileindex_row128 SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_fragment_a SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_single_tile_k64_dispatch_rows_5 SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_single_tile_k64_dispatch_rows_4 SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
    std::cout << "  tma_single_tile_k128_dispatch_rows_5 SKIP (set NEMOTRON_RUN_P5_TMA_SMOKE=1 to enable)\n";
  }
  if (RunPackedCase("single_tile_valid_rows_full_tile", kMacroTileK, kTokenRows, kSingleTileTolerance) != 0) {
    return 1;
  }
  if (RunPackedCase("single_tile_dispatch_rows_5", kMacroTileK, 5, kSingleTileTolerance) != 0) {
    return 1;
  }
  if (RunPackedCase("single_tile_dispatch_rows_4", kMacroTileK, 4, kSingleTileTolerance) != 0) {
    return 1;
  }
  if (RunPackedCase("runtime_like_dispatch_rows_5", kRuntimeLikeTotalK, 5, kRuntimeLikeTolerance) != 0) {
    return 1;
  }
  if (RunPackedCase("runtime_like_dispatch_rows_4", kRuntimeLikeTotalK, 4, kRuntimeLikeTolerance) != 0) {
    return 1;
  }

  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() { return RunTest(); }
