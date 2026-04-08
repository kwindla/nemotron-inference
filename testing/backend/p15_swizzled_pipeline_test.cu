// P15 (256x128x64) swizzled-smem GEMM pipeline test.
//
// Exercises the full CUTLASS SM120 block-scaled FP4 pipeline:
//   swizzled smem staging -> CUTE copy atoms -> fp4_shift -> make_zip_tensor -> cute::gemm
//
// Uses the TRT-LLM MoE kernel pattern (zipped operand+scale gemm) rather than
// the CUTLASS collective's C-view rescale pattern, because our TiledMma uses
// the BLOCKSCALED atom whose mma_unpack requires zipped inputs.
//
// The CUTLASS CollectiveMainloop type is used ONLY for:
//   - SmemLayoutA/B (swizzled operand smem layouts)
//   - SmemCopyAtomA/B (copy atoms for smem->register)
//   - TiledMma
//   - TensorStorage (smem allocation)
//
// Shared P15 scale helpers provide the production atom-scale contract:
// base-row extraction -> flat row-scale lookup -> atom-scale tensor.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

// ============================================================================
// Type definitions
// ============================================================================

using ArchTag = cutlass::arch::Sm120;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using MmaTileShape = cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>;
using ElementAB = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementSFLoad = int32_t;
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

// Use auto stage count for type extraction (TiledMma, SmemLayout*, etc.) but
// allocate only a single-stage smem buffer manually.  The auto-carveout gives
// 6 stages (146 KB) which exceeds SM120's 100 KB per-block limit, but we only
// need 1 stage for this test.
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

// Manually sized single-stage smem.  The collective's TensorStorage is too
// large (146 KB for 6 stages) but we only use stage 0.  Single-stage A+B
// is ~12 KB plus alignment.
using SmemAllocTypeA = typename TiledMma::ValTypeA;  // uint4_t for FP4
using SmemAllocTypeB = typename TiledMma::ValTypeB;
static constexpr int kSmemAStageElems = cute::size(cute::take<0, 2>(SmemLayoutA{}));
static constexpr int kSmemBStageElems = cute::size(cute::take<0, 2>(SmemLayoutB{}));

struct SharedStorage {
  alignas(1024) cute::array_aligned<SmemAllocTypeA, kSmemAStageElems> smem_A;
  alignas(1024) cute::array_aligned<SmemAllocTypeB, kSmemBStageElems> smem_B;
  alignas(1024) cute::array_aligned<ElementSF, cute::cosize(SmemLayoutSFA{})> smem_SFA;
  alignas(1024) cute::array_aligned<ElementSF, cute::cosize(SmemLayoutSFB{})> smem_SFB;
};

constexpr int kRows = 256;
constexpr int kCols = 128;
constexpr int kK = 64;
constexpr int kRuntimeLikeTotalK = 1856;
constexpr int kThreadCount = 256;

constexpr float kTolerance = 6.0f;  // FP4 tensor-core path vs scalar host reference
constexpr float kNvfp4ActivationMaxFiniteHost = 6.0f * 448.0f;
constexpr float kNvfp4MinTensorScaleHost = 1.0f / 1024.0f;

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
          static_cast<int>(((row + 1) * (seed + 5)) + ((col + 3) * (seed + 11)));
      values[row * cols + col] =
          (static_cast<float>((raw % 29) - 14) * scale) +
          (0.0015f * static_cast<float>((row + col + static_cast<std::size_t>(seed)) % 7));
    }
  }
  return values;
}

std::vector<float> RowMajorMatVecLocal(
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t cols,
    const std::vector<float>& activations) {
  std::vector<float> output(rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    float accum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
      accum += weights[row * cols + col] * activations[col];
    }
    output[row] = accum;
  }
  return output;
}

void Relu2InPlaceLocal(std::vector<float>* values) {
  if (values == nullptr) {
    return;
  }
  for (float& value : *values) {
    value = value > 0.0f ? value * value : 0.0f;
  }
}

float ClampNvfp4TensorScaleLocal(float value) {
  if (!std::isfinite(value) || value < kNvfp4MinTensorScaleHost) {
    return kNvfp4MinTensorScaleHost;
  }
  return value;
}

float ComputeTensorScaleForRowLocal(const std::vector<float>& row) {
  float max_abs = 0.0f;
  for (float value : row) {
    max_abs = std::max(max_abs, std::fabs(value));
  }
  if (max_abs > kNvfp4ActivationMaxFiniteHost) {
    return ClampNvfp4TensorScaleLocal(max_abs / kNvfp4ActivationMaxFiniteHost);
  }
  return 1.0f;
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
  auto execution_scale_offset = [](std::size_t row, std::size_t block_col, std::size_t padded_blocks_per_row, nemotron::Nvfp4ScaleLayout layout) {
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

// ============================================================================
// Kernel
// ============================================================================

__global__ void P15SwizzledPipelineKernel(
    const std::uint8_t* __restrict__ a_global,   // packed FP4, kRows * kK/2 bytes
    const std::uint8_t* __restrict__ b_global,   // packed FP4, kCols * kK/2 bytes
    const std::uint32_t* __restrict__ a_scale_words,
    const std::uint32_t* __restrict__ b_scale_words,
    float* output) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);

  // ----- Stage input data into swizzled SmemLayoutA/B -----
  // Strategy: copy the input bytes directly into the smem allocation.  The
  // input is row-major packed FP4 (2 elements per byte) and SmemLayoutA/B's
  // stage 0 also stores the same number of bytes with a swizzled byte order.
  // We use an identity tensor to compute the swizzled byte position for each
  // logical (row, col_pair) and scatter into smem.
  //
  // Since FP4 is subbyte, we work at byte granularity: each byte holds two
  // consecutive K-elements for the same row.  The swizzle only permutes bytes
  // (not nibbles within a byte), so we can copy at byte level.
  {
    // Byte-level identity tensor for stage 0: maps (row, byte) -> smem byte offset
    // SmemLayoutA stage 0 for FP4 has kRows * kK/2 logical bytes
    // The swizzled layout maps each byte to a physical smem byte offset.
    auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
    constexpr int a_row_bytes = kK / 2;
    constexpr int a_total_bytes = kRows * a_row_bytes;
    for (int i = tid; i < a_total_bytes; i += kThreadCount) {
      int row = i / a_row_bytes;
      int col_byte = i % a_row_bytes;
      // The swizzled element offset for the first nibble of this byte
      // Note: stage0_A maps (row, col_element) -> swizzled element offset
      // Two elements per byte, at col_element = col_byte*2 and col_byte*2+1
      auto elem_offset = stage0_A(row, col_byte * 2);
      // For FP4, element_offset / 2 = byte offset in swizzled smem
      int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_a_bytes[smem_byte_pos] = a_global[i];
    }

    auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());
    constexpr int b_row_bytes = kK / 2;
    constexpr int b_total_bytes = kCols * b_row_bytes;
    for (int i = tid; i < b_total_bytes; i += kThreadCount) {
      int row = i / b_row_bytes;
      int col_byte = i % b_row_bytes;
      auto elem_offset = stage0_B(row, col_byte * 2);
      int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_b_bytes[smem_byte_pos] = b_global[i];
    }
  }
  __syncthreads();

  // ----- Create smem tensors -----
  // Use the full SmemLayoutA/B (all 6 stages) but backed by single-stage
  // storage.  Only stage 0 is accessed.  The smem_A allocation is sized for
  // one stage's worth of elements; the layout's stage-0 footprint fits within.
  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()), SmemLayoutA{});
  auto sB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_B.data()), SmemLayoutB{});
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFA.data()), SmemLayoutSFA{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()), SmemLayoutSFB{});

  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

  for (int row = tid; row < kRows; row += kThreadCount) {
    nemotron::p15_scale_runtime::store_scale_word_k64(
        sSFA, a_scale_words[static_cast<std::size_t>(row)], row);
  }
  for (int row = tid; row < kCols; row += kThreadCount) {
    nemotron::p15_scale_runtime::store_scale_word_k64(
        sSFB, b_scale_words[static_cast<std::size_t>(row)], row);
  }
  __syncthreads();

  // ----- MMA setup -----
  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(tid);
  // A/B register fragments
  auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
  auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
  auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
  auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

  // A/B smem -> register copy
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

  // Accumulator
  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(MmaTileShape{}));
  cute::clear(accum);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kRows>{}, cute::Int<kCols>{}));
  auto part_c = thread_mma.partition_C(dense_c);
  // ----- Copy operands: smem -> register (stage 0) -----
  cute::copy(s2r_copy_A,
      tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}),
      tCrA_cv);
  cute::copy(s2r_copy_B,
      tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}),
      tCrB_cv);
  cute::copy(
      tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}),
      tCrSFA_cv);
  cute::copy(
      tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
      tCrSFB_cv);

  // ----- FP4 shift -----
  using MMAOp = typename TiledMma::MMA_Op;
  for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
    cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
    cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
  }

  // ----- Zipped GEMM (manual atom-level dispatch) -----
  //
  // The BLOCKSCALED MMA atom's mma_unpack expects zipped (operand, scale)
  // tensors where the SFA component has size==64 (atom K dimension).
  //
  // For FP4 with K=64 and SFVecSize=16: the atom needs 4 unique scale
  // values packed into a 64-element tensor (64/16=4, with broadcast).
  // But tCrA's mode-0 (V) is only 32 elements, while SFA needs 64.
  //
  // This makes cute::gemm(tiled_mma, make_zip_tensor(tCrA, SFA), ...)
  // structurally incompatible for FP4: the mode-0 sizes don't match.
  //
  // Solution: iterate over M and N tiles manually and construct atom-level
  // zipped tensors with the correct 64-element SFA/SFB shape.
  {
    constexpr int M_tiles = cute::size<1>(decltype(tCrA){});  // 4
    constexpr int N_tiles = cute::size<1>(decltype(tCrB){});  // 8
    constexpr int K_blocks = cute::size<2>(decltype(tCrA){}); // 1

    using AtomType = typename TiledMma::Atom;
    AtomType mma_atom;

    for (int k = 0; k < K_blocks; ++k) {
      for (int n = 0; n < N_tiles; ++n) {
        for (int m = 0; m < M_tiles; ++m) {
          // Atom-level operand slices
          auto a_atom = tCrA(cute::_, m, k);   // rank-1, size 32
          auto b_atom = tCrB(cute::_, n, k);   // rank-1, size 16
          auto c_atom = accum(cute::_, m, n);   // rank-1, size 4
          auto sfa_atom = tCrSFA(cute::_, m, k);
          auto sfb_atom = tCrSFB(cute::_, n, k);

          // Zip and call atom-level MMA
          auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
          auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);

          // MMA_Atom::call(D, A, B, C)
          mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
        }
      }
    }
  }

  // ----- Store output -----
  for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
    auto coord = part_c(i);
    int row = static_cast<int>(cute::get<0>(coord));
    int col = static_cast<int>(cute::get<1>(coord));
    output[static_cast<std::size_t>(row) * kCols + col] = accum(i);
  }
}

__global__ void P15SwizzledPipelineExecScaleKernel(
    const std::uint8_t* __restrict__ a_global,
    const std::uint8_t* __restrict__ b_global,
    const std::uint8_t* __restrict__ a_exec_scales,
    const std::uint8_t* __restrict__ b_exec_scales,
    float* output) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int tid = static_cast<int>(threadIdx.x);

  {
    auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
    constexpr int a_row_bytes = kK / 2;
    constexpr int a_total_bytes = kRows * a_row_bytes;
    for (int i = tid; i < a_total_bytes; i += kThreadCount) {
      int row = i / a_row_bytes;
      int col_byte = i % a_row_bytes;
      auto elem_offset = stage0_A(row, col_byte * 2);
      int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_a_bytes[smem_byte_pos] = a_global[i];
    }

    auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());
    constexpr int b_row_bytes = kK / 2;
    constexpr int b_total_bytes = kCols * b_row_bytes;
    for (int i = tid; i < b_total_bytes; i += kThreadCount) {
      int row = i / b_row_bytes;
      int col_byte = i % b_row_bytes;
      auto elem_offset = stage0_B(row, col_byte * 2);
      int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_b_bytes[smem_byte_pos] = b_global[i];
    }
  }
  __syncthreads();

  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()), SmemLayoutA{});
  auto sB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_B.data()), SmemLayoutB{});
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFA.data()), SmemLayoutSFA{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()), SmemLayoutSFB{});

  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

  constexpr std::size_t kPaddedBlocksPerRow = 4;
  for (int row = tid; row < kRows; row += kThreadCount) {
    nemotron::p15_scale_runtime::store_scale_word_k64(
        sSFA,
        LoadExecutionScaleWordTest(
            a_exec_scales,
            static_cast<std::size_t>(row),
            0u,
            kPaddedBlocksPerRow,
            nemotron::Nvfp4ScaleLayout::kSwizzled128x4),
        row);
  }
  for (int row = tid; row < kCols; row += kThreadCount) {
    nemotron::p15_scale_runtime::store_scale_word_k64(
        sSFB,
        LoadExecutionScaleWordTest(
            b_exec_scales,
            static_cast<std::size_t>(row),
            0u,
            kPaddedBlocksPerRow,
            nemotron::Nvfp4ScaleLayout::kSwizzled128x4),
        row);
  }
  __syncthreads();

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(tid);
  auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
  auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
  auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
  auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

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

  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(MmaTileShape{}));
  cute::clear(accum);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kRows>{}, cute::Int<kCols>{}));
  auto part_c = thread_mma.partition_C(dense_c);

  cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
  cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
  cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
  cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

  using MMAOp = typename TiledMma::MMA_Op;
  for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
    cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
    cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
  }

  {
    constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
    constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
    constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
    using AtomType = typename TiledMma::Atom;
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
  }

  for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
    auto coord = part_c(i);
    int row = static_cast<int>(cute::get<0>(coord));
    int col = static_cast<int>(cute::get<1>(coord));
    output[static_cast<std::size_t>(row) * kCols + col] = accum(i);
  }
}

__global__ void P15RuntimeLikeOperandExecScaleKernel(
    const std::uint8_t* __restrict__ a_global,
    const std::uint8_t* __restrict__ b_global,
    const std::uint8_t* __restrict__ a_exec_scales,
    const std::uint8_t* __restrict__ b_exec_scales,
    float* output) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);
  __shared__ std::uint8_t a_packed[kRows][kK / 2];
  __shared__ std::uint8_t b_packed[kCols][kK / 2];

  const int tid = static_cast<int>(threadIdx.x);

  constexpr int a_row_bytes = kK / 2;
  constexpr int a_total_bytes = kRows * a_row_bytes;
  for (int i = tid; i < a_total_bytes; i += kThreadCount) {
    int row = i / a_row_bytes;
    int col_byte = i % a_row_bytes;
    a_packed[row][col_byte] = a_global[i];
  }
  constexpr int b_row_bytes = kK / 2;
  constexpr int b_total_bytes = kCols * b_row_bytes;
  for (int i = tid; i < b_total_bytes; i += kThreadCount) {
    int row = i / b_row_bytes;
    int col_byte = i % b_row_bytes;
    b_packed[row][col_byte] = b_global[i];
  }

  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFA.data()), SmemLayoutSFA{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()), SmemLayoutSFB{});
  constexpr std::size_t kPaddedBlocksPerRow = 4;
  for (int row = tid; row < kRows; row += kThreadCount) {
    nemotron::p15_scale_runtime::store_scale_word_k64(
        sSFA,
        LoadExecutionScaleWordTest(
            a_exec_scales,
            static_cast<std::size_t>(row),
            0u,
            kPaddedBlocksPerRow,
            nemotron::Nvfp4ScaleLayout::kSwizzled128x4),
        row);
  }
  for (int row = tid; row < kCols; row += kThreadCount) {
    nemotron::p15_scale_runtime::store_scale_word_k64(
        sSFB,
        LoadExecutionScaleWordTest(
            b_exec_scales,
            static_cast<std::size_t>(row),
            0u,
            kPaddedBlocksPerRow,
            nemotron::Nvfp4ScaleLayout::kSwizzled128x4),
        row);
  }

  {
    auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
    for (int i = tid; i < a_total_bytes; i += kThreadCount) {
      int row = i / a_row_bytes;
      int col_byte = i % a_row_bytes;
      auto elem_offset = stage0_A(row, col_byte * 2);
      int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_a_bytes[smem_byte_pos] = a_packed[row][col_byte];
    }

    auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
    auto* smem_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());
    for (int i = tid; i < b_total_bytes; i += kThreadCount) {
      int row = i / b_row_bytes;
      int col_byte = i % b_row_bytes;
      auto elem_offset = stage0_B(row, col_byte * 2);
      int smem_byte_pos = static_cast<int>(elem_offset) / 2;
      smem_b_bytes[smem_byte_pos] = b_packed[row][col_byte];
    }
  }
  __syncthreads();

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
  auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
  auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

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

  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(MmaTileShape{}));
  cute::clear(accum);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kRows>{}, cute::Int<kCols>{}));
  auto part_c = thread_mma.partition_C(dense_c);

  cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
  cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
  cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
  cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

  using MMAOp = typename TiledMma::MMA_Op;
  for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
    cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
    cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
  }

  {
    constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
    constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
    constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
    using AtomType = typename TiledMma::Atom;
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
  }

  for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
    auto coord = part_c(i);
    int row = static_cast<int>(cute::get<0>(coord));
    int col = static_cast<int>(cute::get<1>(coord));
    output[static_cast<std::size_t>(row) * kCols + col] = accum(i);
  }
}

__global__ void P15RuntimeLikeOperandExecScaleKernelMultiK(
    const std::uint8_t* __restrict__ a_global,
    const std::uint8_t* __restrict__ b_global,
    const std::uint8_t* __restrict__ a_exec_scales,
    const std::uint8_t* __restrict__ b_exec_scales,
    int total_k,
    float* output) {
  extern __shared__ char smem_raw[];
  auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);
  __shared__ std::uint8_t a_packed[kRows][kK / 2];
  __shared__ std::uint8_t b_packed[kCols][kK / 2];

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

  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(MmaTileShape{}));
  cute::clear(accum);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kRows>{}, cute::Int<kCols>{}));
  auto part_c = thread_mma.partition_C(dense_c);

  using MMAOp = typename TiledMma::MMA_Op;
  using AtomType = typename TiledMma::Atom;
  constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
  constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
  constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
  AtomType mma_atom;

  for (int k_base = 0; k_base < total_k; k_base += 64) {
    const std::size_t packed_byte_offset = static_cast<std::size_t>(k_base / 2);
    const std::size_t block_base = static_cast<std::size_t>(k_base / 16);

    constexpr int a_row_bytes = kK / 2;
    constexpr int a_total_bytes = kRows * a_row_bytes;
    for (int i = tid; i < a_total_bytes; i += kThreadCount) {
      const int row = i / a_row_bytes;
      const int col_byte = i % a_row_bytes;
      a_packed[row][col_byte] =
          a_global[static_cast<std::size_t>(row) * packed_row_bytes +
                   packed_byte_offset +
                   static_cast<std::size_t>(col_byte)];
    }
    constexpr int b_row_bytes = kK / 2;
    constexpr int b_total_bytes = kCols * b_row_bytes;
    for (int i = tid; i < b_total_bytes; i += kThreadCount) {
      const int row = i / b_row_bytes;
      const int col_byte = i % b_row_bytes;
      b_packed[row][col_byte] =
          b_global[static_cast<std::size_t>(row) * packed_row_bytes +
                   packed_byte_offset +
                   static_cast<std::size_t>(col_byte)];
    }

    for (int row = tid; row < kRows; row += kThreadCount) {
      nemotron::p15_scale_runtime::store_scale_word_k64(
          sSFA,
          LoadExecutionScaleWordTest(
              a_exec_scales,
              static_cast<std::size_t>(row),
              block_base,
              padded_blocks_per_row,
              nemotron::Nvfp4ScaleLayout::kSwizzled128x4),
          row);
    }
    for (int row = tid; row < kCols; row += kThreadCount) {
      nemotron::p15_scale_runtime::store_scale_word_k64(
          sSFB,
          LoadExecutionScaleWordTest(
              b_exec_scales,
              static_cast<std::size_t>(row),
              block_base,
              padded_blocks_per_row,
              nemotron::Nvfp4ScaleLayout::kSwizzled128x4),
          row);
    }

    {
      auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
      auto* smem_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
      for (int i = tid; i < a_total_bytes; i += kThreadCount) {
        const int row = i / a_row_bytes;
        const int col_byte = i % a_row_bytes;
        auto elem_offset = stage0_A(row, col_byte * 2);
        const int smem_byte_pos = static_cast<int>(elem_offset) / 2;
        smem_a_bytes[smem_byte_pos] = a_packed[row][col_byte];
      }

      auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
      auto* smem_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());
      for (int i = tid; i < b_total_bytes; i += kThreadCount) {
        const int row = i / b_row_bytes;
        const int col_byte = i % b_row_bytes;
        auto elem_offset = stage0_B(row, col_byte * 2);
        const int smem_byte_pos = static_cast<int>(elem_offset) / 2;
        smem_b_bytes[smem_byte_pos] = b_packed[row][col_byte];
      }
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

  for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
    auto coord = part_c(i);
    int row = static_cast<int>(cute::get<0>(coord));
    int col = static_cast<int>(cute::get<1>(coord));
    output[static_cast<std::size_t>(row) * kCols + col] = accum(i);
  }
}

// ============================================================================
// Host driver
// ============================================================================

float DecodeFp4(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0F;
  return static_cast<float>(value);
}

std::uint8_t EncodeScaleByte(float value) {
  return nemotron::p15_scale_runtime::encode_scale_byte(value);
}

float DecodeScaleByte(std::uint8_t raw_byte) {
  return nemotron::p15_scale_runtime::decode_scale_byte(raw_byte);
}

std::uint8_t MakePackedByte(std::uint8_t low, std::uint8_t high) {
  return static_cast<std::uint8_t>((high << 4) | (low & 0x0F));
}

float ReferenceOutputAt(
    std::vector<std::uint8_t> const& a_bytes,
    std::vector<std::uint8_t> const& b_bytes,
    std::vector<std::uint32_t> const& a_scale_words,
    std::vector<std::uint32_t> const& b_scale_words,
    int row,
    int col) {
  float accum = 0.0f;
  for (int k = 0; k < kK; ++k) {
    const std::uint8_t a_byte = a_bytes[static_cast<std::size_t>(row) * (kK / 2) + static_cast<std::size_t>(k / 2)];
    const std::uint8_t b_byte = b_bytes[static_cast<std::size_t>(col) * (kK / 2) + static_cast<std::size_t>(k / 2)];
    const std::uint8_t a_nibble = (k & 1) ? ((a_byte >> 4) & 0x0F) : (a_byte & 0x0F);
    const std::uint8_t b_nibble = (k & 1) ? ((b_byte >> 4) & 0x0F) : (b_byte & 0x0F);
    const float a_scale = DecodeScaleByte(
        nemotron::p15_scale_runtime::load_scale_byte(a_scale_words[static_cast<std::size_t>(row)], k / 16));
    const float b_scale = DecodeScaleByte(
        nemotron::p15_scale_runtime::load_scale_byte(b_scale_words[static_cast<std::size_t>(col)], k / 16));
    accum += DecodeFp4(a_nibble) * a_scale * DecodeFp4(b_nibble) * b_scale;
  }
  return accum;
}

float ReferenceOutputAtMultiK(
    std::vector<std::uint8_t> const& a_bytes,
    std::vector<std::uint8_t> const& b_bytes,
    std::vector<std::uint8_t> const& a_scale_bytes,
    std::vector<std::uint8_t> const& b_scale_bytes,
    int total_k,
    int row,
    int col) {
  const int blocks_per_row = total_k / 16;
  float accum = 0.0f;
  for (int k = 0; k < total_k; ++k) {
    const std::uint8_t a_byte =
        a_bytes[static_cast<std::size_t>(row) * static_cast<std::size_t>(total_k / 2) +
                static_cast<std::size_t>(k / 2)];
    const std::uint8_t b_byte =
        b_bytes[static_cast<std::size_t>(col) * static_cast<std::size_t>(total_k / 2) +
                static_cast<std::size_t>(k / 2)];
    const std::uint8_t a_nibble = (k & 1) ? ((a_byte >> 4) & 0x0F) : (a_byte & 0x0F);
    const std::uint8_t b_nibble = (k & 1) ? ((b_byte >> 4) & 0x0F) : (b_byte & 0x0F);
    const float a_scale =
        DecodeScaleByte(a_scale_bytes[static_cast<std::size_t>(row) * static_cast<std::size_t>(blocks_per_row) +
                                      static_cast<std::size_t>(k / 16)]);
    const float b_scale =
        DecodeScaleByte(b_scale_bytes[static_cast<std::size_t>(col) * static_cast<std::size_t>(blocks_per_row) +
                                      static_cast<std::size_t>(k / 16)]);
    accum += DecodeFp4(a_nibble) * a_scale * DecodeFp4(b_nibble) * b_scale;
  }
  return accum;
}

int RunCase(
    std::string_view label,
    std::vector<std::uint8_t> const& h_a,
    std::vector<std::uint8_t> const& h_b,
    std::vector<std::uint32_t> const& h_a_scale_words,
    std::vector<std::uint32_t> const& h_b_scale_words,
    float tolerance) {
  const std::size_t a_bytes = h_a.size();
  const std::size_t b_bytes = h_b.size();
  const std::size_t out_count = static_cast<std::size_t>(kRows) * kCols;

  std::uint8_t *d_a = nullptr, *d_b = nullptr;
  std::uint32_t *d_a_scale_words = nullptr, *d_b_scale_words = nullptr;
  float* d_out = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A") ||
      !CheckCuda(cudaMalloc(&d_b, b_bytes), "malloc B") ||
      !CheckCuda(cudaMalloc(&d_a_scale_words, h_a_scale_words.size() * sizeof(std::uint32_t)), "malloc A scales") ||
      !CheckCuda(cudaMalloc(&d_b_scale_words, h_b_scale_words.size() * sizeof(std::uint32_t)), "malloc B scales") ||
      !CheckCuda(cudaMalloc(&d_out, out_count * sizeof(float)), "malloc out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_scale_words); cudaFree(d_b_scale_words); cudaFree(d_out);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A") ||
      !CheckCuda(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice), "copy B") ||
      !CheckCuda(cudaMemcpy(d_a_scale_words, h_a_scale_words.data(), h_a_scale_words.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "copy A scales") ||
      !CheckCuda(cudaMemcpy(d_b_scale_words, h_b_scale_words.data(), h_b_scale_words.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "copy B scales") ||
      !CheckCuda(cudaMemset(d_out, 0xff, out_count * sizeof(float)), "fill out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_scale_words); cudaFree(d_b_scale_words); cudaFree(d_out);
    return 1;
  }

  constexpr int smem_bytes = sizeof(SharedStorage);
  if (!CheckCuda(cudaFuncSetAttribute(
          P15SwizzledPipelineKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes), "setMaxDynamicSmem")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_scale_words); cudaFree(d_b_scale_words); cudaFree(d_out);
    return 1;
  }

  P15SwizzledPipelineKernel<<<1, kThreadCount, smem_bytes>>>(
      d_a, d_b, d_a_scale_words, d_b_scale_words, d_out);
  if (!CheckCuda(cudaGetLastError(), "kernel launch") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_scale_words); cudaFree(d_b_scale_words); cudaFree(d_out);
    return 1;
  }

  std::vector<float> h_out(out_count, 0.0f);
  if (!CheckCuda(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost), "memcpy out")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_scale_words); cudaFree(d_b_scale_words); cudaFree(d_out);
    return 1;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_a_scale_words);
  cudaFree(d_b_scale_words);
  cudaFree(d_out);

  float max_diff = 0.0f;
  int max_row = 0;
  int max_col = 0;
  std::size_t nan_count = 0;
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const float got = h_out[static_cast<std::size_t>(row) * kCols + col];
      if (std::isnan(got)) {
        ++nan_count;
        continue;
      }
      const float expected =
          ReferenceOutputAt(h_a, h_b, h_a_scale_words, h_b_scale_words, row, col);
      const float diff = std::fabs(got - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_row = row;
        max_col = col;
      }
    }
  }

  std::cout << "  case=" << label
            << " max_diff=" << max_diff
            << " at=(" << max_row << "," << max_col << ")"
            << " got=" << h_out[static_cast<std::size_t>(max_row) * kCols + max_col]
            << " expected=" << ReferenceOutputAt(h_a, h_b, h_a_scale_words, h_b_scale_words, max_row, max_col)
            << " nan_count=" << nan_count << "\n";
  if (nan_count != 0 || max_diff > tolerance) {
    std::cerr << "  FAIL " << label << "\n";
    return 1;
  }
  return 0;
}

int RunExecScaleCase(
    std::string_view label,
    std::vector<std::uint8_t> const& h_a,
    std::vector<std::uint8_t> const& h_b,
    std::vector<std::uint32_t> const& h_a_scale_words,
    std::vector<std::uint32_t> const& h_b_scale_words,
    float tolerance) {
  const auto h_a_exec = SwizzleRowMajorScalesForExecutionLocal(
      reinterpret_cast<const std::uint8_t*>(h_a_scale_words.data()),
      kRows,
      kK,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const auto h_b_exec = SwizzleRowMajorScalesForExecutionLocal(
      reinterpret_cast<const std::uint8_t*>(h_b_scale_words.data()),
      kCols,
      kK,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const std::size_t a_bytes = h_a.size();
  const std::size_t b_bytes = h_b.size();
  const std::size_t out_count = static_cast<std::size_t>(kRows) * kCols;

  std::uint8_t *d_a = nullptr, *d_b = nullptr, *d_a_exec = nullptr, *d_b_exec = nullptr;
  float* d_out = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A") ||
      !CheckCuda(cudaMalloc(&d_b, b_bytes), "malloc B") ||
      !CheckCuda(cudaMalloc(&d_a_exec, h_a_exec.size()), "malloc A exec scales") ||
      !CheckCuda(cudaMalloc(&d_b_exec, h_b_exec.size()), "malloc B exec scales") ||
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
          P15SwizzledPipelineExecScaleKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes), "setMaxDynamicSmem exec")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  P15SwizzledPipelineExecScaleKernel<<<1, kThreadCount, smem_bytes>>>(
      d_a, d_b, d_a_exec, d_b_exec, d_out);
  if (!CheckCuda(cudaGetLastError(), "kernel launch exec") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync exec")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  std::vector<float> h_out(out_count, 0.0f);
  if (!CheckCuda(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost), "memcpy out exec")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);

  float max_diff = 0.0f;
  int max_row = 0;
  int max_col = 0;
  std::size_t nan_count = 0;
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const float got = h_out[static_cast<std::size_t>(row) * kCols + col];
      if (std::isnan(got)) {
        ++nan_count;
        continue;
      }
      const float expected =
          ReferenceOutputAt(h_a, h_b, h_a_scale_words, h_b_scale_words, row, col);
      const float diff = std::fabs(got - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_row = row;
        max_col = col;
      }
    }
  }
  std::cout << "  case=" << label
            << " max_diff=" << max_diff
            << " at=(" << max_row << "," << max_col << ")"
            << " got=" << h_out[static_cast<std::size_t>(max_row) * kCols + max_col]
            << " expected=" << ReferenceOutputAt(h_a, h_b, h_a_scale_words, h_b_scale_words, max_row, max_col)
            << " nan_count=" << nan_count << "\n";
  if (nan_count != 0 || max_diff > tolerance) {
    std::cerr << "  FAIL " << label << "\n";
    return 1;
  }
  return 0;
}

int RunRuntimeLikeExecScaleCase(
    std::string_view label,
    std::vector<std::uint8_t> const& h_a,
    std::vector<std::uint8_t> const& h_b,
    std::vector<std::uint32_t> const& h_a_scale_words,
    std::vector<std::uint32_t> const& h_b_scale_words,
    float tolerance) {
  const auto h_a_exec = SwizzleRowMajorScalesForExecutionLocal(
      reinterpret_cast<const std::uint8_t*>(h_a_scale_words.data()),
      kRows,
      kK,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const auto h_b_exec = SwizzleRowMajorScalesForExecutionLocal(
      reinterpret_cast<const std::uint8_t*>(h_b_scale_words.data()),
      kCols,
      kK,
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const std::size_t a_bytes = h_a.size();
  const std::size_t b_bytes = h_b.size();
  const std::size_t out_count = static_cast<std::size_t>(kRows) * kCols;

  std::uint8_t *d_a = nullptr, *d_b = nullptr, *d_a_exec = nullptr, *d_b_exec = nullptr;
  float* d_out = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A rl") ||
      !CheckCuda(cudaMalloc(&d_b, b_bytes), "malloc B rl") ||
      !CheckCuda(cudaMalloc(&d_a_exec, h_a_exec.size()), "malloc A exec rl") ||
      !CheckCuda(cudaMalloc(&d_b_exec, h_b_exec.size()), "malloc B exec rl") ||
      !CheckCuda(cudaMalloc(&d_out, out_count * sizeof(float)), "malloc out rl")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A rl") ||
      !CheckCuda(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice), "copy B rl") ||
      !CheckCuda(cudaMemcpy(d_a_exec, h_a_exec.data(), h_a_exec.size(), cudaMemcpyHostToDevice), "copy A exec rl") ||
      !CheckCuda(cudaMemcpy(d_b_exec, h_b_exec.data(), h_b_exec.size(), cudaMemcpyHostToDevice), "copy B exec rl") ||
      !CheckCuda(cudaMemset(d_out, 0xff, out_count * sizeof(float)), "fill out rl")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  constexpr int smem_bytes = sizeof(SharedStorage);
  if (!CheckCuda(cudaFuncSetAttribute(
          P15RuntimeLikeOperandExecScaleKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes), "setMaxDynamicSmem rl")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  P15RuntimeLikeOperandExecScaleKernel<<<1, kThreadCount, smem_bytes>>>(
      d_a, d_b, d_a_exec, d_b_exec, d_out);
  if (!CheckCuda(cudaGetLastError(), "kernel launch rl") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync rl")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  std::vector<float> h_out(out_count, 0.0f);
  if (!CheckCuda(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost), "memcpy out rl")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);

  float max_diff = 0.0f;
  int max_row = 0;
  int max_col = 0;
  std::size_t nan_count = 0;
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const float got = h_out[static_cast<std::size_t>(row) * kCols + col];
      if (std::isnan(got)) {
        ++nan_count;
        continue;
      }
      const float expected =
          ReferenceOutputAt(h_a, h_b, h_a_scale_words, h_b_scale_words, row, col);
      const float diff = std::fabs(got - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_row = row;
        max_col = col;
      }
    }
  }
  std::cout << "  case=" << label
            << " max_diff=" << max_diff
            << " at=(" << max_row << "," << max_col << ")"
            << " got=" << h_out[static_cast<std::size_t>(max_row) * kCols + max_col]
            << " expected=" << ReferenceOutputAt(h_a, h_b, h_a_scale_words, h_b_scale_words, max_row, max_col)
            << " nan_count=" << nan_count << "\n";
  if (nan_count != 0 || max_diff > tolerance) {
    std::cerr << "  FAIL " << label << "\n";
    return 1;
  }
  return 0;
}

int RunRuntimeLikeExecScaleCaseMultiK(
    std::string_view label,
    std::vector<std::uint8_t> const& h_a,
    std::vector<std::uint8_t> const& h_b,
    std::vector<std::uint8_t> const& h_a_scale_bytes,
    std::vector<std::uint8_t> const& h_b_scale_bytes,
    int total_k,
    float tolerance) {
  const auto h_a_exec = SwizzleRowMajorScalesForExecutionLocal(
      h_a_scale_bytes.data(),
      kRows,
      static_cast<std::size_t>(total_k),
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const auto h_b_exec = SwizzleRowMajorScalesForExecutionLocal(
      h_b_scale_bytes.data(),
      kCols,
      static_cast<std::size_t>(total_k),
      nemotron::Nvfp4ScaleLayout::kSwizzled128x4);
  const std::size_t a_bytes = h_a.size();
  const std::size_t b_bytes = h_b.size();
  const std::size_t out_count = static_cast<std::size_t>(kRows) * kCols;

  std::uint8_t *d_a = nullptr, *d_b = nullptr, *d_a_exec = nullptr, *d_b_exec = nullptr;
  float* d_out = nullptr;
  if (!CheckCuda(cudaMalloc(&d_a, a_bytes), "malloc A multi-k") ||
      !CheckCuda(cudaMalloc(&d_b, b_bytes), "malloc B multi-k") ||
      !CheckCuda(cudaMalloc(&d_a_exec, h_a_exec.size()), "malloc A exec multi-k") ||
      !CheckCuda(cudaMalloc(&d_b_exec, h_b_exec.size()), "malloc B exec multi-k") ||
      !CheckCuda(cudaMalloc(&d_out, out_count * sizeof(float)), "malloc out multi-k")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  if (!CheckCuda(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A multi-k") ||
      !CheckCuda(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice), "copy B multi-k") ||
      !CheckCuda(cudaMemcpy(d_a_exec, h_a_exec.data(), h_a_exec.size(), cudaMemcpyHostToDevice), "copy A exec multi-k") ||
      !CheckCuda(cudaMemcpy(d_b_exec, h_b_exec.data(), h_b_exec.size(), cudaMemcpyHostToDevice), "copy B exec multi-k") ||
      !CheckCuda(cudaMemset(d_out, 0xff, out_count * sizeof(float)), "fill out multi-k")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  constexpr int smem_bytes = sizeof(SharedStorage);
  if (!CheckCuda(cudaFuncSetAttribute(
          P15RuntimeLikeOperandExecScaleKernelMultiK,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_bytes), "setMaxDynamicSmem multi-k")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  P15RuntimeLikeOperandExecScaleKernelMultiK<<<1, kThreadCount, smem_bytes>>>(
      d_a, d_b, d_a_exec, d_b_exec, total_k, d_out);
  if (!CheckCuda(cudaGetLastError(), "kernel launch multi-k") ||
      !CheckCuda(cudaDeviceSynchronize(), "sync multi-k")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }

  std::vector<float> h_out(out_count, 0.0f);
  if (!CheckCuda(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost), "memcpy out multi-k")) {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);
    return 1;
  }
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_a_exec); cudaFree(d_b_exec); cudaFree(d_out);

  float max_diff = 0.0f;
  int max_row = 0;
  int max_col = 0;
  std::size_t nan_count = 0;
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const float got = h_out[static_cast<std::size_t>(row) * kCols + col];
      if (std::isnan(got)) {
        ++nan_count;
        continue;
      }
      const float expected =
          ReferenceOutputAtMultiK(h_a, h_b, h_a_scale_bytes, h_b_scale_bytes, total_k, row, col);
      const float diff = std::fabs(got - expected);
      if (diff > max_diff) {
        max_diff = diff;
        max_row = row;
        max_col = col;
      }
    }
  }
  std::cout << "  case=" << label
            << " max_diff=" << max_diff
            << " at=(" << max_row << "," << max_col << ")"
            << " got=" << h_out[static_cast<std::size_t>(max_row) * kCols + max_col]
            << " expected=" << ReferenceOutputAtMultiK(
                   h_a, h_b, h_a_scale_bytes, h_b_scale_bytes, total_k, max_row, max_col)
            << " nan_count=" << nan_count << "\n";
  if (nan_count != 0 || max_diff > tolerance) {
    std::cerr << "  FAIL " << label << "\n";
    return 1;
  }
  return 0;
}

int RunTest() {
  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return 1;
  }
  if (device_count == 0) {
    std::cout << "p15_swizzled_pipeline_test: SKIP (no CUDA device)\n";
    return 0;
  }

  std::cout << "p15_swizzled_pipeline_test:\n"
            << "  tile: " << kRows << "x" << kCols << "x" << kK << "\n"
            << "  smem: " << sizeof(SharedStorage) << " bytes ("
            << sizeof(SharedStorage) / 1024 << " KB)\n"
            << "  threads: " << kThreadCount << "\n";

  constexpr std::size_t a_bytes = kRows * kK / 2;
  constexpr std::size_t b_bytes = kCols * kK / 2;

  std::vector<std::uint8_t> h_a(a_bytes, 0);
  std::vector<std::uint8_t> h_b(b_bytes, 0);
  std::vector<std::uint32_t> h_a_scale_words(kRows, 0);
  std::vector<std::uint32_t> h_b_scale_words(kCols, 0);

  for (int row = 0; row < kRows; ++row) {
    for (int col_byte = 0; col_byte < kK / 2; ++col_byte) {
      const std::uint8_t low =
          static_cast<std::uint8_t>(1 + ((row + col_byte) % 3));
      const std::uint8_t high =
          static_cast<std::uint8_t>(1 + (((row * 3) + (col_byte * 5) + 1) % 4));
      h_a[static_cast<std::size_t>(row) * (kK / 2) + col_byte] = MakePackedByte(low, high);
    }
    const float row_region_bias =
        0.125f * static_cast<float>(row & 0x3) +
        0.25f * static_cast<float>((row >> 2) & 0x1) +
        0.5f * static_cast<float>((row >> 3) & 0x1);
    h_a_scale_words[static_cast<std::size_t>(row)] = nemotron::p15_scale_runtime::pack_scale_word4_bytes(
        EncodeScaleByte(1.00f + row_region_bias),
        EncodeScaleByte(1.25f + row_region_bias),
        EncodeScaleByte(1.50f + row_region_bias),
        EncodeScaleByte(1.75f + row_region_bias));
  }

  for (int row = 0; row < kCols; ++row) {
    for (int col_byte = 0; col_byte < kK / 2; ++col_byte) {
      const std::uint8_t low =
          static_cast<std::uint8_t>(1 + (((row * 7) + col_byte + 2) % 4));
      const std::uint8_t high =
          static_cast<std::uint8_t>(1 + (((row * 5) + (col_byte * 3) + 1) % 3));
      h_b[static_cast<std::size_t>(row) * (kK / 2) + col_byte] = MakePackedByte(low, high);
    }
    const float row_region_bias =
        0.125f * static_cast<float>(row & 0x3) +
        0.25f * static_cast<float>((row >> 2) & 0x1);
    h_b_scale_words[static_cast<std::size_t>(row)] = nemotron::p15_scale_runtime::pack_scale_word4_bytes(
        EncodeScaleByte(0.75f + row_region_bias),
        EncodeScaleByte(1.00f + row_region_bias),
        EncodeScaleByte(1.25f + row_region_bias),
        EncodeScaleByte(1.50f + row_region_bias));
  }

  if (RunCase("non_uniform_scales", h_a, h_b, h_a_scale_words, h_b_scale_words, kTolerance) != 0) {
    return 1;
  }

  std::vector<std::uint8_t> h_b_rows2 = h_b;
  std::vector<std::uint32_t> h_b_scale_rows2 = h_b_scale_words;
  for (int row = 2; row < kCols; ++row) {
    for (int col_byte = 0; col_byte < kK / 2; ++col_byte) {
      h_b_rows2[static_cast<std::size_t>(row) * (kK / 2) + col_byte] = 0u;
    }
    h_b_scale_rows2[static_cast<std::size_t>(row)] = 0u;
  }
  if (RunCase("dispatch_rows_2_non_uniform", h_a, h_b_rows2, h_a_scale_words, h_b_scale_rows2, kTolerance) != 0) {
    return 1;
  }
  if (RunExecScaleCase("dispatch_rows_2_exec_scales", h_a, h_b_rows2, h_a_scale_words, h_b_scale_rows2, kTolerance) != 0) {
    return 1;
  }
  if (RunRuntimeLikeExecScaleCase("dispatch_rows_2_runtime_like", h_a, h_b_rows2, h_a_scale_words, h_b_scale_rows2, kTolerance) != 0) {
    return 1;
  }

  std::vector<std::uint8_t> h_b_rows1 = h_b;
  std::vector<std::uint32_t> h_b_scale_rows1 = h_b_scale_words;
  for (int row = 1; row < kCols; ++row) {
    for (int col_byte = 0; col_byte < kK / 2; ++col_byte) {
      h_b_rows1[static_cast<std::size_t>(row) * (kK / 2) + col_byte] = 0u;
    }
    h_b_scale_rows1[static_cast<std::size_t>(row)] = 0u;
  }
  if (RunCase("dispatch_rows_1_non_uniform", h_a, h_b_rows1, h_a_scale_words, h_b_scale_rows1, kTolerance) != 0) {
    return 1;
  }
  if (RunExecScaleCase("dispatch_rows_1_exec_scales", h_a, h_b_rows1, h_a_scale_words, h_b_scale_rows1, kTolerance) != 0) {
    return 1;
  }
  if (RunRuntimeLikeExecScaleCase("dispatch_rows_1_runtime_like", h_a, h_b_rows1, h_a_scale_words, h_b_scale_rows1, kTolerance) != 0) {
    return 1;
  }

  constexpr float kMultiKTolerance = 40.0f;
  const int multi_k = kRuntimeLikeTotalK;
  const int multi_k_row_bytes = multi_k / 2;
  const int multi_k_blocks = multi_k / 16;
  std::vector<std::uint8_t> h_a_multi(
      static_cast<std::size_t>(kRows) * static_cast<std::size_t>(multi_k_row_bytes), 0u);
  std::vector<std::uint8_t> h_b_multi(
      static_cast<std::size_t>(kCols) * static_cast<std::size_t>(multi_k_row_bytes), 0u);
  std::vector<std::uint8_t> h_a_multi_scale_bytes(
      static_cast<std::size_t>(kRows) * static_cast<std::size_t>(multi_k_blocks), 0u);
  std::vector<std::uint8_t> h_b_multi_scale_bytes(
      static_cast<std::size_t>(kCols) * static_cast<std::size_t>(multi_k_blocks), 0u);

  for (int row = 0; row < kRows; ++row) {
    for (int col_byte = 0; col_byte < multi_k_row_bytes; ++col_byte) {
      const std::uint8_t low = static_cast<std::uint8_t>(
          1 + ((row + col_byte + (col_byte / 7)) % 4));
      const std::uint8_t high = static_cast<std::uint8_t>(
          1 + (((row * 5) + (col_byte * 3) + (col_byte / 11) + 1) % 4));
      h_a_multi[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_row_bytes) +
                static_cast<std::size_t>(col_byte)] = MakePackedByte(low, high);
    }
    for (int block = 0; block < multi_k_blocks; ++block) {
      const float bias =
          0.0625f * static_cast<float>(row & 0x7) +
          0.03125f * static_cast<float>(block & 0x7) +
          0.125f * static_cast<float>((block >> 3) & 0x3);
      h_a_multi_scale_bytes[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_blocks) +
                            static_cast<std::size_t>(block)] =
          EncodeScaleByte(0.75f + bias);
    }
  }

  for (int row = 0; row < kCols; ++row) {
    for (int col_byte = 0; col_byte < multi_k_row_bytes; ++col_byte) {
      const std::uint8_t low = static_cast<std::uint8_t>(
          1 + (((row * 7) + col_byte + (col_byte / 5) + 2) % 4));
      const std::uint8_t high = static_cast<std::uint8_t>(
          1 + (((row * 3) + (col_byte * 5) + (col_byte / 13) + 1) % 4));
      h_b_multi[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_row_bytes) +
                static_cast<std::size_t>(col_byte)] = MakePackedByte(low, high);
    }
    for (int block = 0; block < multi_k_blocks; ++block) {
      const float bias =
          0.0625f * static_cast<float>(row & 0x3) +
          0.03125f * static_cast<float>(block & 0xF);
      h_b_multi_scale_bytes[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_blocks) +
                            static_cast<std::size_t>(block)] =
          EncodeScaleByte(0.50f + bias);
    }
  }

  std::vector<std::uint8_t> h_b_multi_rows2 = h_b_multi;
  std::vector<std::uint8_t> h_b_multi_scale_rows2 = h_b_multi_scale_bytes;
  for (int row = 2; row < kCols; ++row) {
    for (int col_byte = 0; col_byte < multi_k_row_bytes; ++col_byte) {
      h_b_multi_rows2[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_row_bytes) +
                      static_cast<std::size_t>(col_byte)] = 0u;
    }
    for (int block = 0; block < multi_k_blocks; ++block) {
      h_b_multi_scale_rows2[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_blocks) +
                            static_cast<std::size_t>(block)] = 0u;
    }
  }
  if (RunRuntimeLikeExecScaleCaseMultiK(
          "dispatch_rows_2_runtime_like_multi_k",
          h_a_multi,
          h_b_multi_rows2,
          h_a_multi_scale_bytes,
          h_b_multi_scale_rows2,
          multi_k,
          kMultiKTolerance) != 0) {
    return 1;
  }

  std::vector<std::uint8_t> h_b_multi_rows1 = h_b_multi;
  std::vector<std::uint8_t> h_b_multi_scale_rows1 = h_b_multi_scale_bytes;
  for (int row = 1; row < kCols; ++row) {
    for (int col_byte = 0; col_byte < multi_k_row_bytes; ++col_byte) {
      h_b_multi_rows1[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_row_bytes) +
                      static_cast<std::size_t>(col_byte)] = 0u;
    }
    for (int block = 0; block < multi_k_blocks; ++block) {
      h_b_multi_scale_rows1[static_cast<std::size_t>(row) * static_cast<std::size_t>(multi_k_blocks) +
                            static_cast<std::size_t>(block)] = 0u;
    }
  }
  if (RunRuntimeLikeExecScaleCaseMultiK(
          "dispatch_rows_1_runtime_like_multi_k",
          h_a_multi,
          h_b_multi_rows1,
          h_a_multi_scale_bytes,
          h_b_multi_scale_rows1,
          multi_k,
          kMultiKTolerance) != 0) {
    return 1;
  }

  const auto nano_routed_down =
      MakePatternedValuesLocal(kRows, static_cast<std::size_t>(multi_k), 53, 0.0078125f);
  const auto nano_routed_up =
      MakePatternedValuesLocal(static_cast<std::size_t>(multi_k), 2688, 47, 0.0078125f);
  const auto nano_a_pack =
      nemotron::PackRowMajorFp32ToNvfp4(nano_routed_down.data(), kRows, static_cast<std::size_t>(multi_k));
  if (!nano_a_pack.has_value()) {
    std::cerr << "  FAIL nano_like_runtime_like_multi_k pack\n";
    return 1;
  }
  auto run_nano_like_rows = [&](int valid_rows) -> int {
    const auto nano_normalized =
        MakePatternedValuesLocal(static_cast<std::size_t>(valid_rows), 2688, 43, 0.01171875f);
    std::vector<std::vector<float>> nano_rows(static_cast<std::size_t>(valid_rows));
    float nano_shared_scale = 1.0f;
    for (int row = 0; row < valid_rows; ++row) {
      std::vector<float> token(
          nano_normalized.begin() + static_cast<std::ptrdiff_t>(row * 2688),
          nano_normalized.begin() + static_cast<std::ptrdiff_t>((row + 1) * 2688));
      nano_rows[static_cast<std::size_t>(row)] = RowMajorMatVecLocal(
          nano_routed_up,
          static_cast<std::size_t>(multi_k),
          2688,
          token);
      Relu2InPlaceLocal(&nano_rows[static_cast<std::size_t>(row)]);
      for (float value : nano_rows[static_cast<std::size_t>(row)]) {
        nano_shared_scale = std::max(nano_shared_scale, std::fabs(value));
      }
    }
    if (nano_shared_scale > kNvfp4ActivationMaxFiniteHost) {
      nano_shared_scale =
          ClampNvfp4TensorScaleLocal(nano_shared_scale / kNvfp4ActivationMaxFiniteHost);
    } else {
      nano_shared_scale = 1.0f;
    }

    std::vector<std::uint8_t> h_b_nano(
        static_cast<std::size_t>(kCols) * static_cast<std::size_t>(multi_k_row_bytes), 0u);
    std::vector<std::uint8_t> h_b_nano_scales(
        static_cast<std::size_t>(kCols) * static_cast<std::size_t>(multi_k_blocks), 0u);
    nemotron::Nvfp4PackOptions nano_b_options;
    nano_b_options.fixed_tensor_scale = nano_shared_scale;
    for (int row = 0; row < valid_rows; ++row) {
      const auto nano_b_pack = nemotron::PackRowMajorFp32ToNvfp4(
          nano_rows[static_cast<std::size_t>(row)].data(),
          1,
          static_cast<std::size_t>(multi_k),
          nano_b_options);
      if (!nano_b_pack.has_value()) {
        std::cerr << "  FAIL nano_like_runtime_like_multi_k row pack\n";
        return 1;
      }
      std::copy(
          nano_b_pack->packed.begin(),
          nano_b_pack->packed.end(),
          h_b_nano.begin() + static_cast<std::ptrdiff_t>(row * multi_k_row_bytes));
      std::copy(
          nano_b_pack->block_scales.begin(),
          nano_b_pack->block_scales.end(),
          h_b_nano_scales.begin() + static_cast<std::ptrdiff_t>(row * multi_k_blocks));
    }

    const std::string label =
        "nano_like_dispatch_rows_" + std::to_string(valid_rows) + "_runtime_like_multi_k";
    return RunRuntimeLikeExecScaleCaseMultiK(
        label,
        nano_a_pack->packed,
        h_b_nano,
        nano_a_pack->block_scales,
        h_b_nano_scales,
        multi_k,
        kMultiKTolerance);
  };

  if (run_nano_like_rows(1) != 0 ||
      run_nano_like_rows(2) != 0 ||
      run_nano_like_rows(4) != 0) {
    return 1;
  }

  std::vector<std::uint8_t> h_a_shift(a_bytes, MakePackedByte(0x3, 0x3));
  std::vector<std::uint8_t> h_b_shift(b_bytes, MakePackedByte(0x3, 0x3));
  std::vector<std::uint32_t> h_a_shift_scales(kRows, nemotron::p15_scale_runtime::pack_scale_word4_bytes(
      EncodeScaleByte(1.0f), EncodeScaleByte(1.0f), EncodeScaleByte(1.0f), EncodeScaleByte(1.0f)));
  std::vector<std::uint32_t> h_b_shift_scales(kCols, nemotron::p15_scale_runtime::pack_scale_word4_bytes(
      EncodeScaleByte(1.0f), EncodeScaleByte(1.0f), EncodeScaleByte(1.0f), EncodeScaleByte(1.0f)));

  if (RunCase("fp4_shift_1p5", h_a_shift, h_b_shift, h_a_shift_scales, h_b_shift_scales, kTolerance) != 0) {
    return 1;
  }

  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() { return RunTest(); }
