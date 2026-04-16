#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>

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
using NanoP1ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<nvfp4_cute::kScaleVecSize>;
using NanoP1SmemLayoutSFA = decltype(NanoP1ScaleConfig::tile_atom_to_shape_SFA(
    cute::make_shape(
        cute::size<0>(NanoP1MmaTileShape{}) * cute::size<0>(NanoP1ClusterShape{}),
        cute::size<1>(NanoP1MmaTileShape{}) * cute::size<1>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{})));
using NanoP1SmemCopyAtomSFA = cute::Copy_Atom<
    cute::UniversalCopy<nvfp4_cute::ElementSFCompute>,
    nvfp4_cute::ElementSFCompute>;
using NanoP1AccumLayout = decltype(
    cute::partition_fragment_C(
        NanoP1TiledMma{},
        cute::make_shape(
            cute::size<0>(NanoP1MmaTileShape{}),
            cute::size<1>(NanoP1MmaTileShape{})))
        .layout());

constexpr int kNanoP1ScaleStageElemsA = cute::cosize(cute::take<0, 2>(NanoP1SmemLayoutSFA{}));
static_assert(cute::size(NanoP1TiledMma{}) == NanoP1ThreadsPerCta);
static_assert(cute::cosize_v<NanoP1SmemLayoutSFA> == 4096);
static_assert(kNanoP1ScaleStageElemsA * NanoP1PipelineStages == cute::cosize_v<NanoP1SmemLayoutSFA>);

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
      auto const logical = cute::make_coord(i);
      f(coord_tensor(logical), value_tensor(logical), i);
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 2) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        auto const logical = cute::make_coord(i, j);
        f(coord_tensor(logical), value_tensor(logical), physical++);
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 3) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          auto const logical = cute::make_coord(i, j, k);
          f(coord_tensor(logical), value_tensor(logical), physical++);
        }
      }
    }
  } else if constexpr (std::remove_cvref_t<CoordTensor>::rank == 4) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          for (int l = 0; l < cute::size<3>(coord_tensor); ++l) {
            auto const logical = cute::make_coord(i, j, k, l);
            f(coord_tensor(logical), value_tensor(logical), physical++);
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

template <class Value>
__device__ __forceinline__ std::uint8_t LoadScaleValueByte(Value const& value) {
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

template <class SFATensor, class AtomT, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto NanoP1ThrfrgSFA(
    SFATensor&& sfatensor,
    cute::TiledMMA<AtomT, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(cute::rank(sfatensor) >= cute::Int<2>{});

  using AtomShape_MNK = typename AtomT::Shape_MNK;
  using AtomLayoutSFA_TV = typename AtomT::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile = cute::make_tile(cute::get<0>(permutation_mnk), cute::get<2>(permutation_mnk));
  auto t_tensor = cute::logical_divide(sfatensor, t_tile);

  auto a_tile = cute::make_tile(
      cute::make_layout(cute::size<0>(AtomShape_MNK{})),
      cute::make_layout(cute::size<2>(AtomShape_MNK{})));
  auto a_tensor = cute::zipped_divide(t_tensor, a_tile);

  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, cute::_);
  auto thr_tile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(cute::size<1>(thr_layout_vmnk)),
          cute::make_layout(cute::size<3>(thr_layout_vmnk))));
  return cute::zipped_divide(tv_tensor, thr_tile);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto NanoP1GetLayoutSFATV(TiledMma& mma) {
  auto tile_shape_mnk = cute::tile_shape(mma);
  auto ref_a = cute::make_layout(
      cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto thr_tensor = NanoP1ThrfrgSFA(ref_a, mma);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto atile = cute::make_tile(
      cute::_,
      cute::make_tile(
          cute::make_layout(
              cute::make_shape(cute::size<1>(thr_layout_vmnk), cute::size<2>(thr_layout_vmnk)),
              cute::make_stride(cute::Int<1>{}, cute::Int<0>{})),
          cute::_));
  auto thridx_2_thrid = cute::right_inverse(thr_layout_vmnk);
  return thr_tensor.compose(atile, cute::_).compose(thridx_2_thrid, cute::_);
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto NanoP1PartitionScaleA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  auto thr_tensor = cute::make_tensor(
      static_cast<SFATensor&&>(sfatensor).data(),
      NanoP1ThrfrgSFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk =
      cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
  auto partition_sfa =
      thr_tensor(thr_vmk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
  return cute::make_fragment_like<ValTypeSF>(partition_sfa);
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto NanoP1PartitionScaleAView(SFATensor&& sfatensor, ThrMma& thread_mma) {
  auto thr_tensor = cute::make_tensor(
      static_cast<SFATensor&&>(sfatensor).data(),
      NanoP1ThrfrgSFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk =
      cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
}

constexpr int kMaxTrackedTidCount = 64;
constexpr int kDefaultTrackedTidCount = 4;
constexpr int kDefaultTrackedTids[kDefaultTrackedTidCount] = {32, 48, 64, 80};
constexpr int kMaxNTiles = 8;
constexpr int kMaxKBlocks = 2;
constexpr int kCopySigLen = 16;
constexpr int kMaxRegWords = 4;
constexpr int kProbeEntryCount = kMaxTrackedTidCount * kMaxNTiles * kMaxKBlocks;

enum class PayloadMode : int {
  kInputFile = 0,
  kRowIndex = 1,
  kBlockIndex = 2,
  kRowIndexHighBits = 3,
  kBlockIndexHighBits = 4,
};

struct ProbeConfig {
  int hidden_size;
  int scale_blocks_per_row;
  int padded_blocks_per_row;
  int weight_rows;
  int output_col_base;
  int block_base;
  PayloadMode payload_mode;
  int tracked_tid_count;
  int tracked_tids[kMaxTrackedTidCount];
};

struct ProbeEntry {
  int valid;
  int tid;
  int n_tile;
  int k_block;
  int part_token_row0;
  int part_output_col0;
  int local_row0;
  int local_col0;
  int local_row1;
  int local_col1;
  int stage0_offset0;
  int stage0_offset1;
  int copy_sig_count;
  int copy_sig_stage_offsets[kCopySigLen];
  int copy_sig_stage_bytes[kCopySigLen];
  int copy_sig_source_rows[kCopySigLen];
  int copy_sig_source_blocks[kCopySigLen];
  int reg_word_count;
  std::uint32_t reg_words[kMaxRegWords];
};

constexpr char const* PayloadModeName(PayloadMode mode) {
  switch (mode) {
    case PayloadMode::kInputFile:
      return "input_file";
    case PayloadMode::kRowIndex:
      return "row_index";
    case PayloadMode::kBlockIndex:
      return "block_index";
    case PayloadMode::kRowIndexHighBits:
      return "row_index_high_bits";
    case PayloadMode::kBlockIndexHighBits:
      return "block_index_high_bits";
  }
  return "unknown";
}

bool CheckCuda(cudaError_t status, char const* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::printf(
      "nano_p1_a_scale_probe: CUDA failure at %s: %s\n",
      what,
      cudaGetErrorString(status));
  return false;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

int LoadEnvInt(char const* name, int default_value) {
  char const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  long const parsed = std::strtol(value, &end, 10);
  if (end == value || (end != nullptr && end[0] != '\0')) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
}

__host__ __device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row) {
  const std::size_t num_k_tiles = padded_blocks_per_row / 4u;
  const std::size_t k_tile = block_col / 4u;
  const std::size_t inner_k = block_col & 3u;
  const std::size_t m_tile = row / 128u;
  const std::size_t outer_m = row & 31u;
  const std::size_t inner_m = (row >> 5u) & 3u;
  return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
          (outer_m << 4u) |
          (inner_m << 2u) |
          inner_k);
}

ProbeConfig LoadProbeConfig() {
  ProbeConfig config{};
  config.hidden_size = LoadEnvInt("NEMOTRON_NANO_P1_A_SCALE_HIDDEN_SIZE", 2688);
  config.scale_blocks_per_row = config.hidden_size / 16;
  config.padded_blocks_per_row =
      static_cast<int>(RoundUp(static_cast<std::size_t>(config.scale_blocks_per_row), 4u));
  config.weight_rows = LoadEnvInt("NEMOTRON_NANO_P1_A_SCALE_WEIGHT_ROWS", 1920);
  config.output_col_base = LoadEnvInt("NEMOTRON_NANO_P1_A_SCALE_OUTPUT_COL_BASE", 0);
  config.block_base = LoadEnvInt("NEMOTRON_NANO_P1_A_SCALE_BLOCK_BASE", 0);
  config.payload_mode = PayloadMode::kInputFile;
  config.tracked_tid_count = kDefaultTrackedTidCount;
  for (int idx = 0; idx < kMaxTrackedTidCount; ++idx) {
    config.tracked_tids[idx] = -1;
  }
  for (int idx = 0; idx < kDefaultTrackedTidCount; ++idx) {
    config.tracked_tids[idx] = kDefaultTrackedTids[idx];
  }
  if (char const* mode_env = std::getenv("NEMOTRON_NANO_P1_A_SCALE_PAYLOAD_MODE");
      mode_env != nullptr && mode_env[0] != '\0') {
    if (std::strcmp(mode_env, "input_file") == 0 || std::strcmp(mode_env, "file") == 0) {
      config.payload_mode = PayloadMode::kInputFile;
    } else if (std::strcmp(mode_env, "row_index") == 0) {
      config.payload_mode = PayloadMode::kRowIndex;
    } else if (std::strcmp(mode_env, "block_index") == 0) {
      config.payload_mode = PayloadMode::kBlockIndex;
    } else if (std::strcmp(mode_env, "row_index_high_bits") == 0) {
      config.payload_mode = PayloadMode::kRowIndexHighBits;
    } else if (std::strcmp(mode_env, "block_index_high_bits") == 0) {
      config.payload_mode = PayloadMode::kBlockIndexHighBits;
    }
  }
  if (char const* tids_env = std::getenv("NEMOTRON_NANO_P1_A_SCALE_TRACKED_TIDS");
      tids_env != nullptr && tids_env[0] != '\0') {
    int parsed_count = 0;
    char const* cursor = tids_env;
    while (*cursor != '\0' && parsed_count < kMaxTrackedTidCount) {
      while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
        ++cursor;
      }
      if (*cursor == '\0') {
        break;
      }
      char* end = nullptr;
      long const parsed = std::strtol(cursor, &end, 10);
      if (end == cursor) {
        break;
      }
      config.tracked_tids[parsed_count++] = static_cast<int>(parsed);
      cursor = end;
    }
    if (parsed_count > 0) {
      config.tracked_tid_count = parsed_count;
      for (int idx = parsed_count; idx < kMaxTrackedTidCount; ++idx) {
        config.tracked_tids[idx] = -1;
      }
    }
  }
  return config;
}

__device__ __forceinline__ int ProbeTidSlot(ProbeConfig const& config, int tid) {
  int const tracked_count =
      config.tracked_tid_count < 0
          ? 0
          : (config.tracked_tid_count > kMaxTrackedTidCount
                 ? kMaxTrackedTidCount
                 : config.tracked_tid_count);
  for (int idx = 0; idx < tracked_count; ++idx) {
    if (config.tracked_tids[idx] == tid) {
      return idx;
    }
  }
  return -1;
}

__device__ __forceinline__ std::uint8_t LoadPayloadScaleByte(
    ProbeConfig const& config,
    std::uint8_t const* execution_scales,
    int source_row,
    int block_index) {
  switch (config.payload_mode) {
    case PayloadMode::kInputFile:
      if (execution_scales == nullptr ||
          source_row < 0 ||
          source_row >= config.weight_rows ||
          block_index < 0 ||
          block_index >= config.scale_blocks_per_row) {
        return 0u;
      }
      return execution_scales[ExecutionScaleOffset(
          static_cast<std::size_t>(source_row),
          static_cast<std::size_t>(block_index),
          static_cast<std::size_t>(config.padded_blocks_per_row))];
    case PayloadMode::kRowIndex:
      return static_cast<std::uint8_t>(source_row & 0xff);
    case PayloadMode::kBlockIndex:
      return static_cast<std::uint8_t>(block_index & 0xff);
    case PayloadMode::kRowIndexHighBits:
      return static_cast<std::uint8_t>((source_row >> 8) & 0xff);
    case PayloadMode::kBlockIndexHighBits:
      return static_cast<std::uint8_t>((block_index >> 8) & 0xff);
  }
  return 0u;
}

template <class ScaleTensor>
__device__ __forceinline__ void StoreScaleTensorByte(
    ScaleTensor& scale_tensor,
    int row,
    int scale_col,
    std::uint8_t value) {
  if constexpr (std::remove_cvref_t<ScaleTensor>::rank == 2) {
    auto elem = scale_tensor(row, scale_col);
    if constexpr (requires { elem.storage; }) {
      scale_tensor(row, scale_col).storage = value;
    } else {
      scale_tensor(row, scale_col) = value;
    }
  } else {
    static_assert(std::remove_cvref_t<ScaleTensor>::rank == 3);
    auto elem = scale_tensor(row, scale_col, cute::Int<0>{});
    if constexpr (requires { elem.storage; }) {
      scale_tensor(row, scale_col, cute::Int<0>{}).storage = value;
    } else {
      scale_tensor(row, scale_col, cute::Int<0>{}) = value;
    }
  }
}

template <class ScaleTensor, class StageLayout, int kScaleByteCount>
__device__ __forceinline__ void StoreTracedScaleBytesWithMeta(
    ScaleTensor& scale_tensor,
    StageLayout const& stage_layout,
    std::uint16_t* stage_source_rows,
    std::uint16_t* stage_source_blocks,
    std::uint8_t const (&scale_bytes)[kScaleByteCount],
    int row,
    int source_row,
    int block_base) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
  const int logical_cols = static_cast<int>(cute::size<1>(scale_tensor));
  const int segment_len = logical_cols > 0 ? ((logical_cols + kScaleByteCount - 1) / kScaleByteCount) : 1;
  for (int scale_col = 0; scale_col < logical_cols; ++scale_col) {
    int byte_index = scale_col / segment_len;
    if (byte_index >= kScaleByteCount) {
      byte_index = kScaleByteCount - 1;
    }
    StoreScaleTensorByte(scale_tensor, row, scale_col, scale_bytes[byte_index]);
    const int stage_offset = static_cast<int>(stage_layout(row, scale_col));
    stage_source_rows[stage_offset] = static_cast<std::uint16_t>(source_row);
    stage_source_blocks[stage_offset] =
        static_cast<std::uint16_t>(block_base + byte_index);
  }
}

struct SharedStorage {
  alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, kNanoP1ScaleStageElemsA> smem_SFA;
  alignas(4) cute::array_aligned<std::uint16_t, kNanoP1ScaleStageElemsA> stage_source_rows;
  alignas(4) cute::array_aligned<std::uint16_t, kNanoP1ScaleStageElemsA> stage_source_blocks;
};

__global__ void NanoP1AScaleProbeKernel(
    ProbeEntry* probe,
    ProbeConfig config,
    std::uint8_t const* execution_scales) {
  constexpr int kTileN = cute::size<1>(NanoP1MmaTileShape{});
  constexpr int kTileK = cute::size<2>(NanoP1MmaTileShape{});
  constexpr int kMacroScaleBytes = kTileK / 16;

  __shared__ SharedStorage shared;
  int const tid = static_cast<int>(threadIdx.x);

  for (int index = tid; index < kNanoP1ScaleStageElemsA; index += blockDim.x) {
    shared.stage_source_rows[index] = 0xffffu;
    shared.stage_source_blocks[index] = 0xffffu;
  }
  __syncthreads();

  auto const stage0_SFA = NanoP1SmemLayoutSFA{}(cute::_, cute::_, cute::Int<0>{});
  auto sScaleA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFA.data()),
      NanoP1SmemLayoutSFA{});
  auto sScaleA = cute::as_position_independent_swizzle_tensor(sScaleA_);
  auto sSFA_stage0 = sScaleA(cute::_, cute::_, cute::Int<0>{});

  for (int row = tid; row < kTileN; row += blockDim.x) {
    std::uint8_t scale_bytes[kMacroScaleBytes];
    int const source_row = config.output_col_base + row;
    for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
      scale_bytes[scale_index] = LoadPayloadScaleByte(
          config,
          execution_scales,
          source_row,
          config.block_base + scale_index);
    }
    StoreTracedScaleBytesWithMeta(
        sSFA_stage0,
        stage0_SFA,
        shared.stage_source_rows.data(),
        shared.stage_source_blocks.data(),
        scale_bytes,
        row,
        source_row,
        config.block_base);
  }

  __syncthreads();

  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(tid);
  auto tCrSFA = NanoP1PartitionScaleA(sSFA_stage0, thread_mma);

  auto tile_shape_mnk = cute::tile_shape(mma);
  auto s2r_copy_SFA = cute::make_tiled_copy_impl(
      NanoP1SmemCopyAtomSFA{},
      NanoP1GetLayoutSFATV(mma),
      cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(tid);
  auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
  auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);
  auto ref_sfa = cute::make_identity_tensor(
      cute::make_shape(cute::size<0>(typename NanoP1TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa_coords = NanoP1PartitionScaleAView(ref_sfa, thread_mma);
  auto tCrSFA_coords = s2r_thr_SFA.retile_D(part_sfa_coords);

  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);

  cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);

  int const tid_slot = ProbeTidSlot(config, tid);
  if (tid_slot < 0) {
    return;
  }

  int const n_tiles = static_cast<int>(cute::size<2>(part_c));
  int const k_blocks = static_cast<int>(cute::size<2>(tCrSFA));
  for (int n_tile = 0; n_tile < n_tiles && n_tile < kMaxNTiles; ++n_tile) {
    auto c_atom_coords = part_c(cute::_, 0, n_tile);
    auto coord0 = c_atom_coords(0);
    for (int k_block = 0; k_block < k_blocks && k_block < kMaxKBlocks; ++k_block) {
      auto coord_view = tCrSFA_coords(cute::_, n_tile, k_block);
      auto value_view = tCrSFA_cv(cute::_, n_tile, k_block);
      auto copy_coord0 = coord_view(0);
      auto copy_coord1 = coord_view(1);
      auto reg_words = cute::recast<std::uint32_t>(cute::filter_zeros(tCrSFA(cute::_, n_tile, k_block)));

      ProbeEntry& entry =
          probe[tid_slot * (kMaxNTiles * kMaxKBlocks) + n_tile * kMaxKBlocks + k_block];
      entry.valid = 1;
      entry.tid = tid;
      entry.n_tile = n_tile;
      entry.k_block = k_block;
      entry.part_output_col0 = CoordGet0(coord0);
      entry.part_token_row0 = CoordGet1(coord0);
      entry.local_row0 = CoordGet0(copy_coord0);
      entry.local_col0 = CoordGet1(copy_coord0);
      entry.local_row1 = CoordGet0(copy_coord1);
      entry.local_col1 = CoordGet1(copy_coord1);
      entry.stage0_offset0 = static_cast<int>(stage0_SFA(entry.local_row0, entry.local_col0));
      entry.stage0_offset1 = static_cast<int>(stage0_SFA(entry.local_row1, entry.local_col1));
      entry.copy_sig_count = 0;
      for (int sig = 0; sig < kCopySigLen; ++sig) {
        entry.copy_sig_stage_offsets[sig] = -1;
        entry.copy_sig_stage_bytes[sig] = -1;
        entry.copy_sig_source_rows[sig] = -1;
        entry.copy_sig_source_blocks[sig] = -1;
      }
      entry.reg_word_count = static_cast<int>(cute::size(reg_words));
      for (int word = 0; word < kMaxRegWords; ++word) {
        entry.reg_words[word] = 0u;
      }
      if (entry.reg_word_count > 0) {
        entry.reg_words[0] = static_cast<std::uint32_t>(reg_words(0));
      }
      if (entry.reg_word_count > 1) {
        entry.reg_words[1] = static_cast<std::uint32_t>(reg_words(1));
      }
      if (entry.reg_word_count > 2) {
        entry.reg_words[2] = static_cast<std::uint32_t>(reg_words(2));
      }
      if (entry.reg_word_count > 3) {
        entry.reg_words[3] = static_cast<std::uint32_t>(reg_words(3));
      }
      ForEachCopyViewCoordValue(coord_view, value_view, [&](auto const& coord, auto const& value, int physical) {
        if (physical >= kCopySigLen) {
          return;
        }
        int const row = CoordGet0(coord);
        int const col = CoordGet1(coord);
        int const stage0_offset = static_cast<int>(stage0_SFA(row, col));
        entry.copy_sig_count = physical + 1;
        entry.copy_sig_stage_offsets[physical] = stage0_offset;
        entry.copy_sig_stage_bytes[physical] = static_cast<int>(LoadScaleValueByte(value));
        entry.copy_sig_source_rows[physical] =
            shared.stage_source_rows[stage0_offset] == 0xffffu
                ? -1
                : static_cast<int>(shared.stage_source_rows[stage0_offset]);
        entry.copy_sig_source_blocks[physical] =
            shared.stage_source_blocks[stage0_offset] == 0xffffu
                ? -1
                : static_cast<int>(shared.stage_source_blocks[stage0_offset]);
      });
    }
  }
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::printf("nano_p1_a_scale_probe: SKIP (no CUDA device available)\n");
    return 0;
  }

  ProbeConfig const config = LoadProbeConfig();
  std::vector<std::uint8_t> host_scale_bytes;
  std::uint8_t* device_scale_bytes = nullptr;
  if (config.payload_mode == PayloadMode::kInputFile) {
    char const* input_file_path = std::getenv("NEMOTRON_NANO_P1_A_SCALE_INPUT_FILE");
    if (input_file_path == nullptr || input_file_path[0] == '\0') {
      std::printf("nano_p1_a_scale_probe: missing input file path\n");
      return 1;
    }
    std::ifstream input_stream(input_file_path, std::ios::binary);
    if (!input_stream) {
      std::printf("nano_p1_a_scale_probe: failed to open input file %s\n", input_file_path);
      return 1;
    }
    host_scale_bytes.assign(
        std::istreambuf_iterator<char>(input_stream),
        std::istreambuf_iterator<char>());
    std::size_t const required_bytes =
        static_cast<std::size_t>(config.weight_rows) *
        static_cast<std::size_t>(config.padded_blocks_per_row);
    if (host_scale_bytes.size() < required_bytes) {
      std::printf(
          "nano_p1_a_scale_probe: input file too small (%zu < %zu)\n",
          host_scale_bytes.size(),
          required_bytes);
      return 1;
    }
    if (!CheckCuda(
            cudaMalloc(reinterpret_cast<void**>(&device_scale_bytes), required_bytes),
            "cudaMalloc scale_bytes") ||
        !CheckCuda(
            cudaMemcpy(
                device_scale_bytes,
                host_scale_bytes.data(),
                required_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy scale_bytes")) {
      cudaFree(device_scale_bytes);
      return 1;
    }
  }

  ProbeEntry* probe_dev = nullptr;
  if (!CheckCuda(
          cudaMalloc(reinterpret_cast<void**>(&probe_dev), sizeof(ProbeEntry) * kProbeEntryCount),
          "cudaMalloc probe_dev") ||
      !CheckCuda(
          cudaMemset(probe_dev, 0, sizeof(ProbeEntry) * kProbeEntryCount),
          "cudaMemset probe_dev")) {
    cudaFree(probe_dev);
    cudaFree(device_scale_bytes);
    return 1;
  }

  NanoP1AScaleProbeKernel<<<1, NanoP1ThreadsPerCta>>>(
      probe_dev,
      config,
      device_scale_bytes);
  if (!CheckCuda(cudaGetLastError(), "launch probe kernel") ||
      !CheckCuda(cudaDeviceSynchronize(), "synchronize probe kernel")) {
    cudaFree(probe_dev);
    cudaFree(device_scale_bytes);
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
    cudaFree(device_scale_bytes);
    return 1;
  }

  cudaFree(probe_dev);
  cudaFree(device_scale_bytes);

  std::printf(
      "nano_p1_a_scale_probe: payload_mode=%s hidden_size=%d scale_blocks_per_row=%d "
      "weight_rows=%d output_col_base=%d block_base=%d\n",
      PayloadModeName(config.payload_mode),
      config.hidden_size,
      config.scale_blocks_per_row,
      config.weight_rows,
      config.output_col_base,
      config.block_base);
  std::printf("nano_p1_a_scale_probe: tracked_tids=");
  int const safe_tracked_tid_count =
      config.tracked_tid_count < 0
          ? 0
          : (config.tracked_tid_count > kMaxTrackedTidCount
                 ? kMaxTrackedTidCount
                 : config.tracked_tid_count);
  for (int idx = 0; idx < safe_tracked_tid_count; ++idx) {
    std::printf("%s%d", idx == 0 ? "" : ",", config.tracked_tids[idx]);
  }
  std::printf("\n");

  for (int tid_slot = 0; tid_slot < safe_tracked_tid_count; ++tid_slot) {
    int const tid = config.tracked_tids[tid_slot];
    std::printf("nano_p1_a_scale_probe: ---- tid=%d ----\n", tid);
    for (int n_tile = 0; n_tile < kMaxNTiles; ++n_tile) {
      for (int k_block = 0; k_block < kMaxKBlocks; ++k_block) {
        ProbeEntry const& entry =
            probe[static_cast<std::size_t>(tid_slot * (kMaxNTiles * kMaxKBlocks) + n_tile * kMaxKBlocks + k_block)];
        if (entry.valid == 0) {
          continue;
        }
        std::printf(
            "  n_tile=%d k_block=%d part_token_row0=%d part_output_col0=%d "
            "local_row0=%d local_col0=%d local_row1=%d local_col1=%d "
            "stage0_offset0=%d stage0_offset1=%d reg_word_count=%d",
            entry.n_tile,
            entry.k_block,
            entry.part_token_row0,
            entry.part_output_col0,
            entry.local_row0,
            entry.local_col0,
            entry.local_row1,
            entry.local_col1,
            entry.stage0_offset0,
            entry.stage0_offset1,
            entry.reg_word_count);
        if (entry.reg_word_count > 0) {
          std::printf(" reg_pre=(");
          for (int word = 0; word < entry.reg_word_count && word < kMaxRegWords; ++word) {
            std::printf("%s0x%08x", word == 0 ? "" : ",", entry.reg_words[word]);
          }
          std::printf(")");
        }
        std::printf("\n");
        if (entry.copy_sig_count > 0) {
          std::printf("    copy_sig=");
          for (int sig = 0; sig < entry.copy_sig_count; ++sig) {
            std::printf(
                "%s%d:%d[row=%d,block=%d]",
                sig == 0 ? "" : ",",
                entry.copy_sig_stage_offsets[sig],
                entry.copy_sig_stage_bytes[sig],
                entry.copy_sig_source_rows[sig],
                entry.copy_sig_source_blocks[sig]);
          }
          std::printf("\n");
        }
      }
    }
  }

  return 0;
}
