#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
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
using NanoP1SmemAllocB = typename NanoP1TiledMma::ValTypeB;
using NanoP1SmemCopyAtomB = cute::Copy_Atom<
    decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_B<
             NanoP1ElementAct,
             NanoP1ElementWeight,
             false>()),
    NanoP1SmemAllocB>;
constexpr int kNanoP1SwizzledBElems = cute::size(cute::take<0, 2>(NanoP1SmemLayoutB{}));

static_assert(cute::size(NanoP1TiledMma{}) == NanoP1ThreadsPerCta);
static_assert(cute::cosize_v<NanoP1SmemLayoutB> == 65536);
static_assert(cute::size<1>(NanoP1AccumLayout{}) == 2);
static_assert(cute::size<2>(NanoP1AccumLayout{}) == 8);

constexpr int kNanoP1TileRows = cute::size<0>(NanoP1MmaTileShape{});
constexpr int kNanoP1MacroTileBytes = cute::size<2>(NanoP1MmaTileShape{}) / 2;

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

enum class ProbeTagPattern {
  kSourceRow,
  kByteIndex,
};

template <ProbeTagPattern kPattern, class Stage0Layout>
__device__ __forceinline__ void FillProbeBStage0(
    std::uint8_t* swizzled_b_bytes,
    Stage0Layout const& stage0_B) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < kNanoP1TileRows * kNanoP1MacroTileBytes;
       linear += static_cast<int>(blockDim.x)) {
    const int row = linear / kNanoP1MacroTileBytes;
    const int byte_index = linear % kNanoP1MacroTileBytes;
    const int source_row = NanoP1PermuteBSourceRow(row);
    const std::uint8_t value = kPattern == ProbeTagPattern::kSourceRow
        ? static_cast<std::uint8_t>(source_row)
        : static_cast<std::uint8_t>(byte_index);
    const auto elem_offset = stage0_B(row, byte_index * 2);
    swizzled_b_bytes[static_cast<int>(elem_offset) / 2] = value;
  }
  __syncthreads();
}

constexpr int kTrackedTidCount = 4;
constexpr int kTrackedTids[kTrackedTidCount] = {0, 1, 128, 129};

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
  int consumed_byte_index0 = -1;
  int consumed_byte_index1 = -1;
  std::uint32_t byte_tag_reg0_pre = 0u;
  std::uint32_t byte_tag_reg1_pre = 0u;
  std::uint32_t byte_tag_reg0_post = 0u;
  std::uint32_t byte_tag_reg1_post = 0u;
};

constexpr int kProbeEntryCount =
    kTrackedTidCount * 8 * 2;  // 4 tracked tids * 8 n tiles * 2 k blocks.

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

__device__ int ProbeThreadSlot(int tid) {
  switch (tid) {
    case 0:
      return 0;
    case 1:
      return 1;
    case 128:
      return 2;
    case 129:
      return 3;
    default:
      return -1;
  }
}

__global__ void NanoP1BOperandProbeKernel(ProbeEntry* entries) {
  __shared__ cute::array_aligned<NanoP1SmemAllocB, kNanoP1SwizzledBElems> smem_B;

  auto mma = NanoP1TiledMma{};
  const int tid = static_cast<int>(threadIdx.x);
  const int probe_slot = ProbeThreadSlot(tid);

  auto thread_mma = mma.get_thread_slice(tid);
  auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_B.data()), NanoP1SmemLayoutB{});
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto stage0_B = NanoP1SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
  auto* swizzled_b_bytes = reinterpret_cast<std::uint8_t*>(smem_B.data());

  auto s2r_copy_B = cute::make_tiled_copy_B(NanoP1SmemCopyAtomB{}, mma);
  auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
  auto tCsB = s2r_thr_B.partition_S(sB);
  auto b_coords = cute::make_identity_tensor(cute::shape(sB));
  auto tCsB_coords = s2r_thr_B.partition_S(b_coords);
  auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
  auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);

  constexpr int kNTiles = cute::size<2>(NanoP1AccumLayout{});
  constexpr int kKBlocks = cute::size<2>(decltype(tCsB_coords){});
  static_assert(kNTiles == 8);
  static_assert(kKBlocks == 2);

  FillProbeBStage0<ProbeTagPattern::kSourceRow>(swizzled_b_bytes, stage0_B);

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
        const int staged_source_row0 = NanoP1PermuteBSourceRow(local_row0);
        const int staged_source_row1 = NanoP1PermuteBSourceRow(local_row1);

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
      }
    }
  }

  FillProbeBStage0<ProbeTagPattern::kByteIndex>(swizzled_b_bytes, stage0_B);

  if (probe_slot >= 0) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        const int entry_index =
            probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
        ProbeEntry& entry = entries[entry_index];
        entry.consumed_byte_index0 =
            static_cast<int>(swizzled_b_bytes[entry.stage0_offset0 / 2]);
        entry.consumed_byte_index1 =
            static_cast<int>(swizzled_b_bytes[entry.stage0_offset1 / 2]);
      }
    }
  }

  __syncthreads();
  cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);

  if (probe_slot >= 0) {
    for (int n_tile = 0; n_tile < kNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kKBlocks; ++k_block) {
        auto b_atom_words = cute::recast<BRegister>(tCrB(cute::_, n_tile, k_block));
        const int entry_index =
            probe_slot * (kNTiles * kKBlocks) + n_tile * kKBlocks + k_block;
        entries[entry_index].byte_tag_reg0_pre =
            static_cast<std::uint32_t>(b_atom_words(0));
        entries[entry_index].byte_tag_reg1_pre =
            static_cast<std::uint32_t>(b_atom_words(1));
      }
    }
  }

  using MMAOp = typename NanoP1TiledMma::MMA_Op;
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

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::printf("nano_p1_b_operand_probe: SKIP (no CUDA device available)\n");
    return 0;
  }

  ProbeEntry* probe_dev = nullptr;
  if (!CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&probe_dev), sizeof(ProbeEntry) * kProbeEntryCount),
          "cudaMalloc probe_dev") ||
      !CheckCuda(
          cudaMemset(probe_dev, 0xff, sizeof(ProbeEntry) * kProbeEntryCount),
          "cudaMemset probe_dev")) {
    cudaFree(probe_dev);
    return 1;
  }

  NanoP1BOperandProbeKernel<<<1, NanoP1ThreadsPerCta>>>(probe_dev);
  if (!CheckCuda(cudaGetLastError(), "launch probe kernel") ||
      !CheckCuda(cudaDeviceSynchronize(), "synchronize probe kernel")) {
    cudaFree(probe_dev);
    return 1;
  }

  std::vector<ProbeEntry> probe(static_cast<std::size_t>(kProbeEntryCount));
  if (!CheckCuda(
          cudaMemcpy(
              probe.data(),
              probe_dev,
              sizeof(ProbeEntry) * kProbeEntryCount,
              cudaMemcpyDeviceToHost),
          "copy probe_dev")) {
    cudaFree(probe_dev);
    return 1;
  }
  cudaFree(probe_dev);

  std::printf(
      "nano_p1_b_operand_probe: current B-side staging writes source_row tags and byte-index tags into stage-0 B\n");
  std::printf(
      "nano_p1_b_operand_probe: check whether consumed_source_row0 matches part_token_row0 for each (tid, n_tile, k_block)\n");

  int total_mismatches = 0;
  for (int tid_slot = 0; tid_slot < kTrackedTidCount; ++tid_slot) {
    const int tid = kTrackedTids[tid_slot];
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
            "consumed_byte_index0=%d consumed_byte_index1=%d "
            "stage0_offset0=%d stage0_offset1=%d "
            "byte_tag_reg0_pre=0x%08x byte_tag_reg1_pre=0x%08x "
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
            entry.consumed_byte_index0,
            entry.consumed_byte_index1,
            entry.stage0_offset0,
            entry.stage0_offset1,
            entry.byte_tag_reg0_pre,
            entry.byte_tag_reg1_pre,
            entry.byte_tag_reg0_post,
            entry.byte_tag_reg1_post,
            row_match ? "ROW_MATCH" : "ROW_MISMATCH");
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
