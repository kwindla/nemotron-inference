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
#include <iostream>
#include <string_view>
#include <vector>

#include <cutlass/epilogue/collective/collective_builder.hpp>
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

struct SharedStorage {
  alignas(1024) cute::array_aligned<SmemAllocTypeA, kSmemAStageElems> smem_A;
  alignas(1024) cute::array_aligned<SmemAllocTypeB, kSmemBStageElems> smem_B;
  alignas(1024) cute::array_aligned<ElementSF, cute::cosize(SmemLayoutSFA{})> smem_SFA;
  alignas(1024) cute::array_aligned<ElementSF, cute::cosize(SmemLayoutSFB{})> smem_SFB;
};

constexpr int kOutputRows = 128;
constexpr int kTokenRows = cute::tile_size<1>(TiledMma{});
constexpr int kMacroTileK = 64;
constexpr int kRuntimeLikeTotalK = 1344;
constexpr int kThreadCount = 256;
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
