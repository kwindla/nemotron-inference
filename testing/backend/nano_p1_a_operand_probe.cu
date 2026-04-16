#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
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
using NanoP1SmemLayoutAtomA = cute::UMMA::Layout_K_SW64_Atom<typename NanoP1TiledMma::ValTypeA>;
using NanoP1SmemLayoutA = decltype(cute::tile_to_shape(
    NanoP1SmemLayoutAtomA{},
    cute::make_shape(
        cute::size<0>(NanoP1MmaTileShape{}) * cute::size<0>(NanoP1ClusterShape{}),
        cute::size<2>(NanoP1MmaTileShape{}) * cute::size<2>(NanoP1ClusterShape{}),
        cute::Int<NanoP1PipelineStages>{}),
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using NanoP1SmemAllocA = typename NanoP1TiledMma::ValTypeA;
using NanoP1SmemCopyAtomA = cute::Copy_Atom<
    decltype(cutlass::gemm::collective::detail::sm120_rr_smem_copy_selector_A<
             NanoP1ElementAct,
             NanoP1ElementWeight,
             false>()),
    NanoP1SmemAllocA>;
constexpr int kNanoP1SwizzledAElems = cute::size(cute::take<0, 2>(NanoP1SmemLayoutA{}));

static_assert(cute::size(NanoP1TiledMma{}) == NanoP1ThreadsPerCta);
static_assert(cute::cosize_v<NanoP1SmemLayoutA> == 65536);

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

constexpr int kMaxTrackedTidCount = 64;
constexpr int kDefaultTrackedTidCount = 4;
constexpr int kDefaultTrackedTids[kDefaultTrackedTidCount] = {32, 48, 64, 80};
constexpr int kMaxNTiles = 8;
constexpr int kMaxKBlocks = 2;
constexpr int kCopySigLen = 8;
constexpr int kProbeEntryCount = kMaxTrackedTidCount * kMaxNTiles * kMaxKBlocks;

enum class PayloadMode : int {
  kInputFile = 0,
  kRowIndex = 1,
  kByteIndex = 2,
  kRowIndexHighBits = 3,
  kByteIndexHighBits = 4,
};

struct ProbeConfig {
  int packed_row_bytes;
  int weight_rows;
  int output_col_base;
  int output_col_bases_enabled;
  int output_col_bases[4];
  int packed_byte_base;
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
  std::uint32_t reg0_pre;
  std::uint32_t reg1_pre;
  std::uint32_t reg0_post;
  std::uint32_t reg1_post;
};

constexpr char const* PayloadModeName(PayloadMode mode) {
  switch (mode) {
    case PayloadMode::kInputFile:
      return "input_file";
    case PayloadMode::kRowIndex:
      return "row_index";
    case PayloadMode::kByteIndex:
      return "byte_index";
    case PayloadMode::kRowIndexHighBits:
      return "row_index_high_bits";
    case PayloadMode::kByteIndexHighBits:
      return "byte_index_high_bits";
  }
  return "unknown";
}

bool CheckCuda(cudaError_t status, char const* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::printf(
      "nano_p1_a_operand_probe: CUDA failure at %s: %s\n",
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

ProbeConfig LoadProbeConfig() {
  ProbeConfig config{};
  config.packed_row_bytes = LoadEnvInt("NEMOTRON_NANO_P1_A_PACKED_ROW_BYTES", 1344);
  config.weight_rows = LoadEnvInt("NEMOTRON_NANO_P1_A_WEIGHT_ROWS", 128);
  config.output_col_base = LoadEnvInt("NEMOTRON_NANO_P1_A_OUTPUT_COL_BASE", 0);
  config.packed_byte_base = LoadEnvInt("NEMOTRON_NANO_P1_A_PACKED_BYTE_BASE", 0);
  config.output_col_bases_enabled = 0;
  for (int idx = 0; idx < 4; ++idx) {
    config.output_col_bases[idx] = config.output_col_base;
  }
  config.payload_mode = PayloadMode::kInputFile;
  config.tracked_tid_count = kDefaultTrackedTidCount;
  for (int idx = 0; idx < kMaxTrackedTidCount; ++idx) {
    config.tracked_tids[idx] = -1;
  }
  for (int idx = 0; idx < kDefaultTrackedTidCount; ++idx) {
    config.tracked_tids[idx] = kDefaultTrackedTids[idx];
  }
  if (char const* mode_env = std::getenv("NEMOTRON_NANO_P1_A_PAYLOAD_MODE");
      mode_env != nullptr && mode_env[0] != '\0') {
    if (std::strcmp(mode_env, "input_file") == 0 || std::strcmp(mode_env, "file") == 0) {
      config.payload_mode = PayloadMode::kInputFile;
    } else if (std::strcmp(mode_env, "row_index") == 0) {
      config.payload_mode = PayloadMode::kRowIndex;
    } else if (std::strcmp(mode_env, "byte_index") == 0) {
      config.payload_mode = PayloadMode::kByteIndex;
    } else if (std::strcmp(mode_env, "row_index_high_bits") == 0) {
      config.payload_mode = PayloadMode::kRowIndexHighBits;
    } else if (std::strcmp(mode_env, "byte_index_high_bits") == 0) {
      config.payload_mode = PayloadMode::kByteIndexHighBits;
    }
  }
  if (char const* tids_env = std::getenv("NEMOTRON_NANO_P1_A_TRACKED_TIDS");
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
  if (char const* bases_env = std::getenv("NEMOTRON_NANO_P1_A_OUTPUT_COL_BASES");
      bases_env != nullptr && bases_env[0] != '\0') {
    int parsed_count = 0;
    char const* cursor = bases_env;
    while (*cursor != '\0' && parsed_count < 4) {
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
      config.output_col_bases[parsed_count++] = static_cast<int>(parsed);
      cursor = end;
    }
    if (parsed_count == 4) {
      config.output_col_bases_enabled = 1;
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

__device__ __forceinline__ std::uint8_t LoadPayloadByte(
    ProbeConfig const& config,
    std::uint8_t const* packed_weight,
    int source_row,
    int byte_index) {
  switch (config.payload_mode) {
    case PayloadMode::kInputFile:
      if (packed_weight == nullptr ||
          source_row < 0 ||
          source_row >= config.weight_rows ||
          byte_index < 0 ||
          byte_index >= config.packed_row_bytes) {
        return 0u;
      }
      return packed_weight[
          static_cast<std::size_t>(source_row) * static_cast<std::size_t>(config.packed_row_bytes) +
          static_cast<std::size_t>(byte_index)];
    case PayloadMode::kRowIndex:
      return static_cast<std::uint8_t>(source_row & 0xff);
    case PayloadMode::kByteIndex:
      return static_cast<std::uint8_t>(byte_index & 0xff);
    case PayloadMode::kRowIndexHighBits:
      return static_cast<std::uint8_t>((source_row >> 8) & 0xff);
    case PayloadMode::kByteIndexHighBits:
      return static_cast<std::uint8_t>((byte_index >> 8) & 0xff);
  }
  return 0u;
}

__device__ __forceinline__ int SourceRowBaseForByteIndex(
    ProbeConfig const& config,
    int byte_index) {
  if (config.output_col_bases_enabled != 0) {
    return config.output_col_bases[byte_index & 0x3];
  }
  return config.output_col_base;
}

struct SharedStorage {
  alignas(1024) cute::array_aligned<NanoP1SmemAllocA, kNanoP1SwizzledAElems> smem_A;
};

__global__ void NanoP1AOperandProbeKernel(
    ProbeEntry* probe,
    ProbeConfig config,
    std::uint8_t const* packed_weight) {
  using TiledMma = NanoP1TiledMma;

  constexpr int kTileN = cute::size<1>(NanoP1MmaTileShape{});
  constexpr int kTileK = cute::size<2>(NanoP1MmaTileShape{});
  constexpr int kMacroTileBytes = kTileK / 2;

  __shared__ SharedStorage shared;
  int const tid = static_cast<int>(threadIdx.x);

  auto const stage0_A = NanoP1SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()),
      NanoP1SmemLayoutA{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto sA_stage0 = sA(cute::_, cute::_, cute::Int<0>{});
  auto* swizzled_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());

  for (int row = tid; row < kTileN; row += blockDim.x) {
    for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
    int const source_row = SourceRowBaseForByteIndex(config, byte_index) + row;
      int const source_byte_index = config.packed_byte_base + byte_index;
      std::uint8_t const value =
          LoadPayloadByte(config, packed_weight, source_row, source_byte_index);
      auto const elem_offset = stage0_A(row, byte_index * 2);
      swizzled_a_bytes[static_cast<int>(elem_offset) / 2] = value;
    }
  }

  __syncthreads();

  auto mma = TiledMma{};
  auto thread_mma = mma.get_thread_slice(tid);
  auto tCrA = thread_mma.partition_fragment_A(sA_stage0);
  auto smem_tiled_copy_A = cute::make_tiled_copy_A(NanoP1SmemCopyAtomA{}, mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tid);
  auto tCsA = smem_thr_copy_A.partition_S(sA);
  auto a_coords = cute::make_identity_tensor(cute::shape(sA));
  auto tCsA_coords = smem_thr_copy_A.partition_S(a_coords);
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);

  cute::copy(smem_tiled_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_copy_view);

  using MMAOp = typename TiledMma::MMA_Op;

  int const tid_slot = ProbeTidSlot(config, tid);
  if (tid_slot < 0) {
    return;
  }

  int const n_tiles = static_cast<int>(cute::size<2>(part_c));
  int const k_blocks = static_cast<int>(cute::size<2>(tCrA));
  for (int n_tile = 0; n_tile < n_tiles && n_tile < kMaxNTiles; ++n_tile) {
    auto c_atom_coords = part_c(cute::_, 0, n_tile);
    auto coord0 = c_atom_coords(0);
    for (int k_block = 0; k_block < k_blocks && k_block < kMaxKBlocks; ++k_block) {
      auto row_anchor = tCsA_coords(cute::_, n_tile, k_block, cute::Int<0>{});
      auto copy_coord0 = row_anchor(0);
      auto copy_coord1 = row_anchor(1);
      auto words = cute::recast<std::uint32_t>(tCrA(cute::_, n_tile, k_block));

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
      entry.stage0_offset0 = static_cast<int>(stage0_A(entry.local_row0, entry.local_col0));
      entry.stage0_offset1 = static_cast<int>(stage0_A(entry.local_row1, entry.local_col1));
      entry.copy_sig_count = 0;
      for (int sig = 0; sig < kCopySigLen; ++sig) {
        entry.copy_sig_stage_offsets[sig] = -1;
        entry.copy_sig_stage_bytes[sig] = -1;
      }
      ForEachCopyViewCoord(row_anchor, [&](auto const& coord, int physical) {
        if (physical >= kCopySigLen) {
          return;
        }
        int const row = CoordGet0(coord);
        int const col = CoordGet1(coord);
        int const stage0_offset = static_cast<int>(stage0_A(row, col));
        entry.copy_sig_count = physical + 1;
        entry.copy_sig_stage_offsets[physical] = stage0_offset;
        entry.copy_sig_stage_bytes[physical] = swizzled_a_bytes[stage0_offset / 2];
      });
      entry.reg0_pre = static_cast<std::uint32_t>(words(0));
      entry.reg1_pre = static_cast<std::uint32_t>(words(1));
      entry.reg0_post = 0u;
      entry.reg1_post = 0u;
    }
  }

  for (int k_block = 0; k_block < k_blocks; ++k_block) {
    cute::fp4_shift_A(MMAOp{}, tCrA_copy_view(cute::_, cute::_, k_block));
  }

  for (int n_tile = 0; n_tile < n_tiles && n_tile < kMaxNTiles; ++n_tile) {
    for (int k_block = 0; k_block < k_blocks && k_block < kMaxKBlocks; ++k_block) {
      auto shifted_words = cute::recast<std::uint32_t>(tCrA(cute::_, n_tile, k_block));
      ProbeEntry& entry =
          probe[tid_slot * (kMaxNTiles * kMaxKBlocks) + n_tile * kMaxKBlocks + k_block];
      entry.reg0_post = static_cast<std::uint32_t>(shifted_words(0));
      entry.reg1_post = static_cast<std::uint32_t>(shifted_words(1));
    }
  }
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::printf("nano_p1_a_operand_probe: SKIP (no CUDA device available)\n");
    return 0;
  }

  ProbeConfig const config = LoadProbeConfig();
  std::vector<std::uint8_t> host_weight_bytes;
  std::uint8_t* device_weight_bytes = nullptr;
  if (config.payload_mode == PayloadMode::kInputFile) {
    char const* input_file_path = std::getenv("NEMOTRON_NANO_P1_A_INPUT_FILE");
    if (input_file_path == nullptr || input_file_path[0] == '\0') {
      std::printf("nano_p1_a_operand_probe: missing input file path\n");
      return 1;
    }
    std::ifstream input_stream(input_file_path, std::ios::binary);
    if (!input_stream) {
      std::printf("nano_p1_a_operand_probe: failed to open input file %s\n", input_file_path);
      return 1;
    }
    host_weight_bytes.assign(
        std::istreambuf_iterator<char>(input_stream),
        std::istreambuf_iterator<char>());
    std::size_t const required_bytes =
        static_cast<std::size_t>(config.weight_rows) * static_cast<std::size_t>(config.packed_row_bytes);
    if (host_weight_bytes.size() < required_bytes) {
      std::printf(
          "nano_p1_a_operand_probe: input file too small (%zu < %zu)\n",
          host_weight_bytes.size(),
          required_bytes);
      return 1;
    }
    if (!CheckCuda(
            cudaMalloc(reinterpret_cast<void**>(&device_weight_bytes), required_bytes),
            "cudaMalloc weight_bytes") ||
        !CheckCuda(
            cudaMemcpy(
                device_weight_bytes,
                host_weight_bytes.data(),
                required_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy weight_bytes")) {
      cudaFree(device_weight_bytes);
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
    cudaFree(device_weight_bytes);
    return 1;
  }

  NanoP1AOperandProbeKernel<<<1, NanoP1ThreadsPerCta>>>(
      probe_dev,
      config,
      device_weight_bytes);
  if (!CheckCuda(cudaGetLastError(), "launch probe kernel") ||
      !CheckCuda(cudaDeviceSynchronize(), "synchronize probe kernel")) {
    cudaFree(probe_dev);
    cudaFree(device_weight_bytes);
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
    cudaFree(device_weight_bytes);
    return 1;
  }

  cudaFree(probe_dev);
  cudaFree(device_weight_bytes);

  std::printf(
      "nano_p1_a_operand_probe: payload_mode=%s packed_row_bytes=%d weight_rows=%d output_col_base=%d packed_byte_base=%d\n",
      PayloadModeName(config.payload_mode),
      config.packed_row_bytes,
      config.weight_rows,
      config.output_col_base,
      config.packed_byte_base);
  if (config.output_col_bases_enabled != 0) {
    std::printf(
        "nano_p1_a_operand_probe: output_col_bases=%d,%d,%d,%d\n",
        config.output_col_bases[0],
        config.output_col_bases[1],
        config.output_col_bases[2],
        config.output_col_bases[3]);
  }
  std::printf("nano_p1_a_operand_probe: tracked_tids=");
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
    std::printf("nano_p1_a_operand_probe: ---- tid=%d ----\n", tid);
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
            "stage0_offset0=%d stage0_offset1=%d reg_pre=(0x%08x,0x%08x) "
            "reg_post=(0x%08x,0x%08x)\n",
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
            entry.reg0_pre,
            entry.reg1_pre,
            entry.reg0_post,
            entry.reg1_post);
        if (entry.copy_sig_count > 0) {
          std::printf("    copy_sig=");
          for (int sig = 0; sig < entry.copy_sig_count; ++sig) {
            std::printf(
                "%s%d:%d",
                sig == 0 ? "" : ",",
                entry.copy_sig_stage_offsets[sig],
                entry.copy_sig_stage_bytes[sig]);
          }
          std::printf("\n");
        }
      }
    }
  }

  return 0;
}
