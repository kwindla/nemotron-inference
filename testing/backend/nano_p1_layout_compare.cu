#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <type_traits>

#include <cutlass/arch/barrier.h>
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

#include "nemotron/fused_moe_prefill.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/linear_op_trace.h"
#include "nemotron/device_nvfp4_matrix.h"
#include "runtime/src/backend/fused_decode_common.cuh"
#include "runtime/src/backend/routed_p5_tma_descriptor.cuh"

namespace nemotron {
#include "runtime/src/backend/fused_moe_prefill/common_helpers.cuh"
#include "runtime/src/backend/fused_moe_prefill/nvfp4_cute.cuh"

__host__ __device__ std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
}

__host__ __device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  constexpr std::size_t kScaleBlockTile = 4u;
  const std::size_t num_k_tiles = padded_blocks_per_row / kScaleBlockTile;
  const std::size_t k_tile = block_col / kScaleBlockTile;
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

#include "runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh"
}  // namespace nemotron

namespace {

namespace cute = ::cute;

template <class T, class U>
constexpr int SameType() {
  return std::is_same_v<T, U> ? 1 : 0;
}

constexpr int kTrackedTidCount = 4;
constexpr int kTrackedTids[kTrackedTidCount] = {32, 48, 64, 80};
constexpr int kProbeNTiles = 8;
constexpr int kProbeKBlocks = 2;
using ScaleElement = decltype(nemotron::nvfp4_bridge::MakeScaleElement(std::uint8_t{}));

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

struct BFragmentCompareOutput {
  std::uint32_t traced_regs[kTrackedTidCount][kProbeNTiles][kProbeKBlocks][2];
  std::uint32_t nano_regs[kTrackedTidCount][kProbeNTiles][kProbeKBlocks][2];
  std::uint32_t traced_scale_regs[kTrackedTidCount][kProbeNTiles][kProbeKBlocks];
  std::uint32_t nano_scale_regs[kTrackedTidCount][kProbeNTiles][kProbeKBlocks];
};

__device__ int TrackedTidSlot(int tid) {
  switch (tid) {
    case 32:
      return 0;
    case 48:
      return 1;
    case 64:
      return 2;
    case 80:
      return 3;
    default:
      return -1;
  }
}

__global__ void CompareTracedVsNanoBFragments(BFragmentCompareOutput* output) {
  constexpr int kRows = cute::size<0>(nemotron::nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kTileK = cute::size<2>(nemotron::nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kPackedRowBytes = kTileK / 2;
  constexpr int kTracedSwizzledBElems =
      cute::size(cute::take<0, 2>(nemotron::nvfp4_bridge::TracedP5SmemLayoutB{}));
  constexpr int kNanoSwizzledBElems = nemotron::nvfp4_bridge::kNanoP1SwizzledBElems;
  constexpr int kTracedScaleStageElems =
      cute::cosize(cute::take<0, 2>(nemotron::nvfp4_bridge::TracedP5SmemLayoutSFB{}));
  constexpr int kNanoScaleStageElems = nemotron::nvfp4_bridge::kNanoP1ScaleStageElemsB;
  __shared__ cute::array_aligned<nemotron::nvfp4_bridge::NanoP1SmemAllocB, kTracedSwizzledBElems>
      traced_smem_b;
  __shared__ cute::array_aligned<nemotron::nvfp4_bridge::NanoP1SmemAllocB, kNanoSwizzledBElems>
      nano_smem_b;
  __shared__ cute::array_aligned<ScaleElement, kTracedScaleStageElems> traced_smem_sfb;
  __shared__ cute::array_aligned<ScaleElement, kNanoScaleStageElems> nano_smem_sfb;

  auto traced_sB_ = cute::make_tensor(
      cute::make_smem_ptr(traced_smem_b.data()),
      nemotron::nvfp4_bridge::TracedP5SmemLayoutB{});
  auto nano_sB_ = cute::make_tensor(
      cute::make_smem_ptr(nano_smem_b.data()),
      nemotron::nvfp4_bridge::NanoP1SmemLayoutB{});
  auto traced_sSFB = cute::make_tensor(
      cute::make_smem_ptr(traced_smem_sfb.data()),
      nemotron::nvfp4_bridge::TracedP5SmemLayoutSFB{});
  auto nano_sSFB = cute::make_tensor(
      cute::make_smem_ptr(nano_smem_sfb.data()),
      nemotron::nvfp4_bridge::NanoP1SmemLayoutSFB{});
  auto traced_sB = cute::as_position_independent_swizzle_tensor(traced_sB_);
  auto nano_sB = cute::as_position_independent_swizzle_tensor(nano_sB_);
  auto traced_sScaleB = cute::as_position_independent_swizzle_tensor(traced_sSFB);
  auto nano_sScaleB = cute::as_position_independent_swizzle_tensor(nano_sSFB);
  auto traced_sB_stage0 = traced_sB(cute::_, cute::_, cute::Int<0>{});
  auto nano_sB_stage0 = nano_sB(cute::_, cute::_, cute::Int<0>{});
  auto traced_sSFB_stage0 = traced_sSFB(cute::_, cute::_, cute::Int<0>{});
  auto nano_sSFB_stage0 = nano_sSFB(cute::_, cute::_, cute::Int<0>{});
  auto traced_stage0_B = nemotron::nvfp4_bridge::TracedP5SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
  auto nano_stage0_B = nemotron::nvfp4_bridge::NanoP1SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});

  auto* traced_swizzled_bytes = reinterpret_cast<std::uint8_t*>(traced_smem_b.data());
  auto* nano_swizzled_bytes = reinterpret_cast<std::uint8_t*>(nano_smem_b.data());
  auto* traced_scale_bytes = reinterpret_cast<std::uint8_t*>(traced_smem_sfb.data());
  auto* nano_scale_bytes = reinterpret_cast<std::uint8_t*>(nano_smem_sfb.data());
  const int tid = static_cast<int>(threadIdx.x);

  for (int i = tid; i < kTracedSwizzledBElems; i += blockDim.x) {
    traced_swizzled_bytes[i] = static_cast<nemotron::nvfp4_bridge::NanoP1SmemAllocB>(0u);
  }
  for (int i = tid; i < kNanoSwizzledBElems; i += blockDim.x) {
    nano_swizzled_bytes[i] = static_cast<nemotron::nvfp4_bridge::NanoP1SmemAllocB>(0u);
  }
  for (int i = tid; i < kTracedScaleStageElems * static_cast<int>(sizeof(ScaleElement)); i += blockDim.x) {
    traced_scale_bytes[i] = 0u;
  }
  for (int i = tid; i < kNanoScaleStageElems * static_cast<int>(sizeof(ScaleElement)); i += blockDim.x) {
    nano_scale_bytes[i] = 0u;
  }
  __syncthreads();

  for (int row = tid; row < kRows; row += blockDim.x) {
    for (int byte_index = 0; byte_index < kPackedRowBytes; ++byte_index) {
      const std::uint8_t packed =
          static_cast<std::uint8_t>(((row & 0x0f) << 4) | (byte_index & 0x0f));
      const int traced_stage_offset = static_cast<int>(traced_stage0_B(row, byte_index * 2));
      const int nano_stage_offset = static_cast<int>(nano_stage0_B(row, byte_index * 2));
      traced_swizzled_bytes[traced_stage_offset / 2] = packed;
      nano_swizzled_bytes[nano_stage_offset / 2] = packed;
    }
  }
  constexpr int kScaleRows = cute::size<0>(decltype(traced_sSFB_stage0){});
  constexpr int kScaleCols = cute::size<1>(decltype(traced_sSFB_stage0){});
  static_assert(kScaleRows == cute::size<0>(decltype(nano_sSFB_stage0){}));
  static_assert(kScaleCols == cute::size<1>(decltype(nano_sSFB_stage0){}));
  for (int linear = tid; linear < kScaleRows * kScaleCols; linear += blockDim.x) {
    const int row = linear / kScaleCols;
    const int col = linear % kScaleCols;
    const std::uint8_t value =
        static_cast<std::uint8_t>(((row & 0x0f) << 4) | (col & 0x0f));
    traced_sSFB_stage0(row, col) = nemotron::nvfp4_bridge::MakeScaleElement(value);
    nano_sSFB_stage0(row, col) = nemotron::nvfp4_bridge::MakeScaleElement(value);
  }
  __syncthreads();

  auto traced_mma = nemotron::nvfp4_bridge::TracedP5TiledMma{};
  auto nano_mma = nemotron::nvfp4_bridge::NanoP1TiledMma{};
  auto traced_thread = traced_mma.get_thread_slice(threadIdx.x);
  auto nano_thread = nano_mma.get_thread_slice(threadIdx.x);
  auto traced_collective = nemotron::nvfp4_bridge::TracedP5CollectiveMainloop{};
  auto traced_tCrB = traced_thread.partition_fragment_B(traced_sB_stage0);
  auto nano_tCrB = nano_thread.partition_fragment_B(nano_sB_stage0);
  auto traced_tCrSFB = traced_collective.partition_fragment_SFB(traced_sSFB_stage0, traced_thread);
  auto nano_tCrSFB = nemotron::nvfp4_bridge::NanoP1PartitionScaleB(nano_sSFB_stage0, nano_thread);
  auto traced_copy_B =
      cute::make_tiled_copy_B(nemotron::nvfp4_bridge::TracedP5SmemCopyAtomB{}, traced_mma);
  auto nano_copy_B =
      cute::make_tiled_copy_B(nemotron::nvfp4_bridge::NanoP1SmemCopyAtomB{}, nano_mma);
  auto traced_copy_SFB = cute::make_tiled_copy_impl(
      nemotron::nvfp4_bridge::TracedP5SmemCopyAtomSFB{},
      traced_collective.get_layoutSFB_TV(traced_mma),
      cute::make_shape(
          cute::size<1>(cute::tile_shape(traced_mma)),
          cute::size<2>(cute::tile_shape(traced_mma))));
  auto nano_copy_SFB = cute::make_tiled_copy_impl(
      nemotron::nvfp4_bridge::NanoP1SmemCopyAtomSFB{},
      nemotron::nvfp4_bridge::NanoP1GetLayoutSFBTV(nano_mma),
      cute::make_shape(
          cute::size<1>(cute::tile_shape(nano_mma)),
          cute::size<2>(cute::tile_shape(nano_mma))));
  auto traced_copy_thr_B = traced_copy_B.get_thread_slice(threadIdx.x);
  auto nano_copy_thr_B = nano_copy_B.get_thread_slice(threadIdx.x);
  auto traced_copy_thr_SFB = traced_copy_SFB.get_thread_slice(threadIdx.x);
  auto nano_copy_thr_SFB = nano_copy_SFB.get_thread_slice(threadIdx.x);
  auto traced_tCsB = traced_copy_thr_B.partition_S(traced_sB);
  auto nano_tCsB = nano_copy_thr_B.partition_S(nano_sB);
  auto traced_tCsSFB = traced_copy_thr_SFB.partition_S(traced_sScaleB);
  auto nano_tCsSFB = nano_copy_thr_SFB.partition_S(nano_sScaleB);
  auto traced_tCrB_cv = traced_copy_thr_B.retile_D(traced_tCrB);
  auto nano_tCrB_cv = nano_copy_thr_B.retile_D(nano_tCrB);
  auto traced_tCrSFB_cv = traced_copy_thr_SFB.retile_D(traced_tCrSFB);
  auto nano_tCrSFB_cv = nano_copy_thr_SFB.retile_D(nano_tCrSFB);

  cute::copy(
      traced_copy_B,
      traced_tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}),
      traced_tCrB_cv);
  cute::copy(
      nano_copy_B,
      nano_tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}),
      nano_tCrB_cv);
  cute::copy(
      traced_tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
      traced_tCrSFB_cv);
  cute::copy(
      nano_tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
      nano_tCrSFB_cv);

  const int slot = TrackedTidSlot(tid);
  if (slot < 0) {
    return;
  }

  static_assert(cute::size<1>(decltype(traced_tCrB){}) == kProbeNTiles);
  static_assert(cute::size<2>(decltype(traced_tCrB){}) == kProbeKBlocks);
  static_assert(cute::size<1>(decltype(nano_tCrB){}) == kProbeNTiles);
  static_assert(cute::size<2>(decltype(nano_tCrB){}) == kProbeKBlocks);
  for (int n_tile = 0; n_tile < kProbeNTiles; ++n_tile) {
    for (int k_block = 0; k_block < kProbeKBlocks; ++k_block) {
      auto traced_words =
          cute::recast<nemotron::nvfp4_bridge::BRegister>(traced_tCrB(cute::_, n_tile, k_block));
      auto nano_words =
          cute::recast<nemotron::nvfp4_bridge::BRegister>(nano_tCrB(cute::_, n_tile, k_block));
      output->traced_regs[slot][n_tile][k_block][0] =
          static_cast<std::uint32_t>(traced_words(0));
      output->traced_regs[slot][n_tile][k_block][1] =
          static_cast<std::uint32_t>(traced_words(1));
      output->nano_regs[slot][n_tile][k_block][0] =
          static_cast<std::uint32_t>(nano_words(0));
      output->nano_regs[slot][n_tile][k_block][1] =
          static_cast<std::uint32_t>(nano_words(1));
      output->traced_scale_regs[slot][n_tile][k_block] =
          PackScaleFragmentWordLocal(traced_tCrSFB(cute::_, n_tile, k_block));
      output->nano_scale_regs[slot][n_tile][k_block] =
          PackScaleFragmentWordLocal(nano_tCrSFB(cute::_, n_tile, k_block));
    }
  }
}

}  // namespace

int main() {
  auto traced_mma = nemotron::nvfp4_bridge::TracedP5TiledMma{};
  auto nano_mma = nemotron::nvfp4_bridge::NanoP1TiledMma{};

  auto traced_sA = cute::make_tensor(
      cute::make_smem_ptr(static_cast<nemotron::nvfp4_bridge::NanoP1SmemAllocA*>(nullptr)),
      nemotron::nvfp4_bridge::TracedP5SmemLayoutA{});
  auto nano_sA = cute::make_tensor(
      cute::make_smem_ptr(static_cast<nemotron::nvfp4_bridge::NanoP1SmemAllocA*>(nullptr)),
      nemotron::nvfp4_bridge::NanoP1SmemLayoutA{});
  auto traced_sB = cute::make_tensor(
      cute::make_smem_ptr(static_cast<nemotron::nvfp4_bridge::NanoP1SmemAllocB*>(nullptr)),
      nemotron::nvfp4_bridge::TracedP5SmemLayoutB{});
  auto nano_sB = cute::make_tensor(
      cute::make_smem_ptr(static_cast<nemotron::nvfp4_bridge::NanoP1SmemAllocB*>(nullptr)),
      nemotron::nvfp4_bridge::NanoP1SmemLayoutB{});

  auto traced_thread = traced_mma.get_thread_slice(0);
  auto nano_thread = nano_mma.get_thread_slice(0);

  auto traced_tCrA = traced_thread.partition_fragment_A(traced_sA(cute::_, cute::_, cute::Int<0>{}));
  auto nano_tCrA = nano_thread.partition_fragment_A(nano_sA(cute::_, cute::_, cute::Int<0>{}));
  auto traced_tCrB = traced_thread.partition_fragment_B(traced_sB(cute::_, cute::_, cute::Int<0>{}));
  auto nano_tCrB = nano_thread.partition_fragment_B(nano_sB(cute::_, cute::_, cute::Int<0>{}));

  auto traced_copy_A =
      cute::make_tiled_copy_A(nemotron::nvfp4_bridge::TracedP5SmemCopyAtomA{}, traced_mma);
  auto nano_copy_A =
      cute::make_tiled_copy_A(nemotron::nvfp4_bridge::NanoP1SmemCopyAtomA{}, nano_mma);
  auto traced_copy_B =
      cute::make_tiled_copy_B(nemotron::nvfp4_bridge::TracedP5SmemCopyAtomB{}, traced_mma);
  auto nano_copy_B =
      cute::make_tiled_copy_B(nemotron::nvfp4_bridge::NanoP1SmemCopyAtomB{}, nano_mma);

  auto traced_copy_thr_A = traced_copy_A.get_thread_slice(0);
  auto nano_copy_thr_A = nano_copy_A.get_thread_slice(0);
  auto traced_copy_thr_B = traced_copy_B.get_thread_slice(0);
  auto nano_copy_thr_B = nano_copy_B.get_thread_slice(0);

  auto traced_tCsA =
      traced_copy_thr_A.partition_S(cute::as_position_independent_swizzle_tensor(traced_sA));
  auto nano_tCsA =
      nano_copy_thr_A.partition_S(cute::as_position_independent_swizzle_tensor(nano_sA));
  auto traced_tCsB =
      traced_copy_thr_B.partition_S(cute::as_position_independent_swizzle_tensor(traced_sB));
  auto nano_tCsB =
      nano_copy_thr_B.partition_S(cute::as_position_independent_swizzle_tensor(nano_sB));
  auto traced_tCrA_cv = traced_copy_thr_A.retile_D(traced_tCrA);
  auto nano_tCrA_cv = nano_copy_thr_A.retile_D(nano_tCrA);
  auto traced_tCrB_cv = traced_copy_thr_B.retile_D(traced_tCrB);
  auto nano_tCrB_cv = nano_copy_thr_B.retile_D(nano_tCrB);

  using DenseShape = decltype(cute::make_shape(cute::Int<128>{}, cute::Int<128>{}));
  auto traced_part_c = cute::partition_fragment_C(traced_mma, DenseShape{});
  auto nano_part_c = cute::partition_fragment_C(nano_mma, DenseShape{});

  std::printf(
      "same_tiled_mma=%d same_smem_layout_a=%d same_smem_layout_b=%d "
      "same_smem_layout_sfa=%d same_smem_layout_sfb=%d\n",
      SameType<
          nemotron::nvfp4_bridge::TracedP5TiledMma,
          nemotron::nvfp4_bridge::NanoP1TiledMma>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemLayoutA,
          nemotron::nvfp4_bridge::NanoP1SmemLayoutA>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemLayoutB,
          nemotron::nvfp4_bridge::NanoP1SmemLayoutB>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemLayoutSFA,
          nemotron::nvfp4_bridge::NanoP1SmemLayoutSFA>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemLayoutSFB,
          nemotron::nvfp4_bridge::NanoP1SmemLayoutSFB>());
  std::printf(
      "same_copy_atom_a=%d same_copy_atom_b=%d same_accum_layout=%d "
      "same_part_c_layout=%d same_copy_atom_sfa=%d same_copy_atom_sfb=%d\n",
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemCopyAtomA,
          nemotron::nvfp4_bridge::NanoP1SmemCopyAtomA>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemCopyAtomB,
          nemotron::nvfp4_bridge::NanoP1SmemCopyAtomB>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5AccumProfileLayout,
          nemotron::nvfp4_bridge::NanoP1AccumLayout>(),
      SameType<decltype(traced_part_c.layout()), decltype(nano_part_c.layout())>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemCopyAtomSFA,
          nemotron::nvfp4_bridge::NanoP1SmemCopyAtomSFA>(),
      SameType<
          nemotron::nvfp4_bridge::TracedP5SmemCopyAtomSFB,
          nemotron::nvfp4_bridge::NanoP1SmemCopyAtomSFB>());
  std::printf(
      "same_tCrA_layout=%d same_tCrB_layout=%d same_tCsA_layout=%d same_tCsB_layout=%d "
      "same_tCrA_cv_layout=%d same_tCrB_cv_layout=%d\n",
      SameType<decltype(traced_tCrA.layout()), decltype(nano_tCrA.layout())>(),
      SameType<decltype(traced_tCrB.layout()), decltype(nano_tCrB.layout())>(),
      SameType<decltype(traced_tCsA.layout()), decltype(nano_tCsA.layout())>(),
      SameType<decltype(traced_tCsB.layout()), decltype(nano_tCsB.layout())>(),
      SameType<decltype(traced_tCrA_cv.layout()), decltype(nano_tCrA_cv.layout())>(),
      SameType<decltype(traced_tCrB_cv.layout()), decltype(nano_tCrB_cv.layout())>());
  std::printf(
      "scale_layout_meta traced=(sfa_cosize:%d sfb_cosize:%d sfa_stage:%d sfb_stage:%d) "
      "nano=(sfa_cosize:%d sfb_cosize:%d sfa_stage:%d sfb_stage:%d)\n",
      nemotron::nvfp4_bridge::kTracedP5ScaleSmemCosizeA,
      nemotron::nvfp4_bridge::kTracedP5ScaleSmemCosizeB,
      static_cast<int>(cute::cosize(cute::take<0, 2>(nemotron::nvfp4_bridge::TracedP5SmemLayoutSFA{}))),
      static_cast<int>(cute::cosize(cute::take<0, 2>(nemotron::nvfp4_bridge::TracedP5SmemLayoutSFB{}))),
      static_cast<int>(cute::cosize_v<nemotron::nvfp4_bridge::NanoP1SmemLayoutSFA>),
      static_cast<int>(cute::cosize_v<nemotron::nvfp4_bridge::NanoP1SmemLayoutSFB>),
      nemotron::nvfp4_bridge::kNanoP1ScaleStageElemsA,
      nemotron::nvfp4_bridge::kNanoP1ScaleStageElemsB);
  std::printf(
      "sizes traced=(part_c:%d tCrA:%d tCrB:%d tCsA:%d tCsB:%d tCrA_cv:%d tCrB_cv:%d)\n",
      static_cast<int>(cute::cosize(traced_part_c.layout())),
      static_cast<int>(cute::cosize(traced_tCrA.layout())),
      static_cast<int>(cute::cosize(traced_tCrB.layout())),
      static_cast<int>(cute::cosize(traced_tCsA.layout())),
      static_cast<int>(cute::cosize(traced_tCsB.layout())),
      static_cast<int>(cute::cosize(traced_tCrA_cv.layout())),
      static_cast<int>(cute::cosize(traced_tCrB_cv.layout())));
  std::printf(
      "sizes nano=(part_c:%d tCrA:%d tCrB:%d tCsA:%d tCsB:%d tCrA_cv:%d tCrB_cv:%d)\n",
      static_cast<int>(cute::cosize(nano_part_c.layout())),
      static_cast<int>(cute::cosize(nano_tCrA.layout())),
      static_cast<int>(cute::cosize(nano_tCrB.layout())),
      static_cast<int>(cute::cosize(nano_tCsA.layout())),
      static_cast<int>(cute::cosize(nano_tCsB.layout())),
      static_cast<int>(cute::cosize(nano_tCrA_cv.layout())),
      static_cast<int>(cute::cosize(nano_tCrB_cv.layout())));
  std::cout << "traced_smem_layout_sfa=" << nemotron::nvfp4_bridge::TracedP5SmemLayoutSFA{} << "\n";
  std::cout << "nano_smem_layout_sfa=" << nemotron::nvfp4_bridge::NanoP1SmemLayoutSFA{} << "\n";
  std::cout << "traced_smem_layout_sfb=" << nemotron::nvfp4_bridge::TracedP5SmemLayoutSFB{} << "\n";
  std::cout << "nano_smem_layout_sfb=" << nemotron::nvfp4_bridge::NanoP1SmemLayoutSFB{} << "\n";

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cout << "nano_p1_layout_compare: SKIP runtime fragment compare (no CUDA device)\n";
    return 0;
  }

  BFragmentCompareOutput* output = nullptr;
  cudaMallocManaged(&output, sizeof(BFragmentCompareOutput));
  if (output == nullptr) {
    std::cout << "nano_p1_layout_compare: failed to allocate compare output\n";
    return 1;
  }
  std::memset(output, 0, sizeof(BFragmentCompareOutput));
  CompareTracedVsNanoBFragments<<<1, nemotron::nvfp4_bridge::NanoP1ThreadsPerCta>>>(output);
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cout << "nano_p1_layout_compare: fragment compare kernel failed: "
              << cudaGetErrorString(status) << "\n";
    cudaFree(output);
    return 1;
  }

  std::cout << "runtime_b_fragment_compare:\n";
  int direct_pairs = 0;
  int swapped_pairs = 0;
  int other_pairs = 0;
  int scale_direct_pairs = 0;
  int scale_swapped_pairs = 0;
  int scale_other_pairs = 0;
  for (int tid_slot = 0; tid_slot < kTrackedTidCount; ++tid_slot) {
    for (int n_tile = 0; n_tile < kProbeNTiles; ++n_tile) {
      const bool direct_pair =
          output->traced_regs[tid_slot][n_tile][0][0] == output->nano_regs[tid_slot][n_tile][0][0] &&
          output->traced_regs[tid_slot][n_tile][0][1] == output->nano_regs[tid_slot][n_tile][0][1] &&
          output->traced_regs[tid_slot][n_tile][1][0] == output->nano_regs[tid_slot][n_tile][1][0] &&
          output->traced_regs[tid_slot][n_tile][1][1] == output->nano_regs[tid_slot][n_tile][1][1];
      const bool swapped_pair =
          output->traced_regs[tid_slot][n_tile][0][0] == output->nano_regs[tid_slot][n_tile][1][0] &&
          output->traced_regs[tid_slot][n_tile][0][1] == output->nano_regs[tid_slot][n_tile][1][1] &&
          output->traced_regs[tid_slot][n_tile][1][0] == output->nano_regs[tid_slot][n_tile][0][0] &&
          output->traced_regs[tid_slot][n_tile][1][1] == output->nano_regs[tid_slot][n_tile][0][1];
      if (direct_pair) {
        ++direct_pairs;
      } else if (swapped_pair) {
        ++swapped_pairs;
      } else {
        ++other_pairs;
      }
      const bool scale_direct_pair =
          output->traced_scale_regs[tid_slot][n_tile][0] == output->nano_scale_regs[tid_slot][n_tile][0] &&
          output->traced_scale_regs[tid_slot][n_tile][1] == output->nano_scale_regs[tid_slot][n_tile][1];
      const bool scale_swapped_pair =
          output->traced_scale_regs[tid_slot][n_tile][0] == output->nano_scale_regs[tid_slot][n_tile][1] &&
          output->traced_scale_regs[tid_slot][n_tile][1] == output->nano_scale_regs[tid_slot][n_tile][0];
      if (scale_direct_pair) {
        ++scale_direct_pairs;
      } else if (scale_swapped_pair) {
        ++scale_swapped_pairs;
      } else {
        ++scale_other_pairs;
      }
      std::printf(
          "  tid=%d n_tile=%d traced_vs_nano=%s traced_k0=(0x%08x,0x%08x) traced_k1=(0x%08x,0x%08x) "
          "nano_k0=(0x%08x,0x%08x) nano_k1=(0x%08x,0x%08x) "
          "scale_traced_vs_nano=%s traced_scale=(0x%08x,0x%08x) nano_scale=(0x%08x,0x%08x)\n",
          kTrackedTids[tid_slot],
          n_tile,
          direct_pair ? "direct" : (swapped_pair ? "kblock_swap" : "other"),
          output->traced_regs[tid_slot][n_tile][0][0],
          output->traced_regs[tid_slot][n_tile][0][1],
          output->traced_regs[tid_slot][n_tile][1][0],
          output->traced_regs[tid_slot][n_tile][1][1],
          output->nano_regs[tid_slot][n_tile][0][0],
          output->nano_regs[tid_slot][n_tile][0][1],
          output->nano_regs[tid_slot][n_tile][1][0],
          output->nano_regs[tid_slot][n_tile][1][1],
          scale_direct_pair ? "direct" : (scale_swapped_pair ? "kblock_swap" : "other"),
          output->traced_scale_regs[tid_slot][n_tile][0],
          output->traced_scale_regs[tid_slot][n_tile][1],
          output->nano_scale_regs[tid_slot][n_tile][0],
          output->nano_scale_regs[tid_slot][n_tile][1]);
    }
  }
  std::printf(
      "runtime_b_fragment_compare_summary direct_pairs=%d swapped_pairs=%d other_pairs=%d\n",
      direct_pairs,
      swapped_pairs,
      other_pairs);
  std::printf(
      "runtime_b_scale_fragment_compare_summary direct_pairs=%d swapped_pairs=%d other_pairs=%d\n",
      scale_direct_pairs,
      scale_swapped_pairs,
      scale_other_pairs);
  cudaFree(output);

  return 0;
}
