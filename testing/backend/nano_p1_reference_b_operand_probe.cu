#include <cxxabi.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <typeinfo>
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
#include "runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh"
}  // namespace nemotron

namespace {

namespace cute = ::cute;

constexpr auto kReferenceProfile = nemotron::nvfp4_bridge::UnifiedRoutedFp4Profile::kP5;
using Traits = nemotron::nvfp4_bridge::UnifiedRoutedFp4Traits<kReferenceProfile>;
using CollectiveMainloop = typename Traits::CollectiveMainloop;
using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
using TiledMma = typename Traits::TiledMma;
using SmemLayoutB = typename Traits::SmemLayoutB;
using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
using SmemLayoutAtomSFB = typename CollectiveMainloop::SmemLayoutAtomSFB;
using SmemCopyAtomB = typename Traits::SmemCopyAtomB;
using SmemCopyAtomSFB = typename Traits::SmemCopyAtomSFB;
using SmemAllocB = typename Traits::SmemAllocB;
using BRegister = nemotron::nvfp4_bridge::BRegister;

constexpr char const* ReferenceProfileName() {
  switch (kReferenceProfile) {
    case nemotron::nvfp4_bridge::UnifiedRoutedFp4Profile::kP5:
      return "kP5";
    case nemotron::nvfp4_bridge::UnifiedRoutedFp4Profile::kP12:
      return "kP12";
    case nemotron::nvfp4_bridge::UnifiedRoutedFp4Profile::kP13:
      return "kP13";
    case nemotron::nvfp4_bridge::UnifiedRoutedFp4Profile::kP15:
      return "kP15";
  }
  return "unknown";
}

constexpr int kThreadsPerCta = cute::size(TiledMma{});
constexpr int kTrackedTidCount = 8;
constexpr int kDefaultTrackedTidCount = 4;
constexpr int kProbeEntryCount = kTrackedTidCount * 8 * 2;
constexpr int kTileRows = Traits::kTokenRows;
constexpr int kMacroTileBytes = Traits::kMacroTileK / 2;
constexpr int kSwizzledBElems = Traits::kSwizzledBElems;
constexpr int kScaleSmemElemsB = Traits::kScaleSmemCosizeB;
constexpr int kMaxScaleRegWords = 4;

static_assert(kThreadsPerCta == 256);

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

struct ProbeConfig {
  int tracked_tid_count = kDefaultTrackedTidCount;
  int tracked_tids[kTrackedTidCount] = {32, 48, 64, 80, -1, -1, -1, -1};
  int stage_slot_bit = -1;
  int stage_slot_low_byte = 0;
  int use_position_map = 0;
  int use_input_file = 0;
  int input_rows = 128;
  int packed_row_bytes = 1344;
  int use_scale_file = 0;
  int hidden_size = 2688;
  int scale_blocks_per_row = 2688 / 16;
  int padded_blocks_per_row = ((2688 / 16) + 3) & ~3;
  int scale_block_base = 160;
};

struct ProbeEntry {
  int valid = 0;
  int tid = -1;
  int n_tile = -1;
  int k_block = -1;
  int tcrb_k_blocks = -1;
  int tcrb_copy_view_k_blocks = -1;
  int tcsb_coord_k_blocks = -1;
  int tcrsfb_k_blocks = -1;
  int tcrsfb_copy_view_k_blocks = -1;
  int tcssfb_coord_k_blocks = -1;
  int tcrb_dim0 = -1;
  int tcrb_dim1 = -1;
  int tcrb_copy_view_dim0 = -1;
  int tcrb_copy_view_dim1 = -1;
  int tcsb_coord_dim0 = -1;
  int tcsb_coord_dim1 = -1;
  int tcsb_coord_stage_count = -1;
  int tcrsfb_dim0 = -1;
  int tcrsfb_dim1 = -1;
  int tcrsfb_copy_view_dim0 = -1;
  int tcrsfb_copy_view_dim1 = -1;
  int tcssfb_coord_dim0 = -1;
  int tcssfb_coord_dim1 = -1;
  int tcssfb_coord_stage_count = -1;
  int sf_vec_size = -1;
  int smem_layout_atom_sfb_dim0 = -1;
  int smem_layout_atom_sfb_dim1 = -1;
  int sfb_stage_elems = -1;
  int sfb_total_elems = -1;
  int sfb_stage_shape_dim0 = -1;
  int sfb_stage_shape_dim1 = -1;
  int layout_sfb_tv_dim0 = -1;
  int layout_sfb_tv_dim1 = -1;
  int part_output_col0 = -1;
  int part_token_row0 = -1;
  int local_row0 = -1;
  int local_col0 = -1;
  int local_row1 = -1;
  int local_col1 = -1;
  int stage0_offset0 = -1;
  int stage0_offset1 = -1;
  int scale_local_row0 = -1;
  int scale_local_col0 = -1;
  int scale_stage0_offset0 = -1;
  int scale_reg_word_count_traced = 0;
  std::uint32_t scale_reg_words_traced[kMaxScaleRegWords] = {};
  int scale_reg_word_count_generic = 0;
  std::uint32_t scale_reg_words_generic[kMaxScaleRegWords] = {};
  int scale_reg_word_count_nano = 0;
  std::uint32_t scale_reg_words_nano[kMaxScaleRegWords] = {};
  std::uint32_t reg0_pre = 0u;
  std::uint32_t reg1_pre = 0u;
};

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
  if (char const* stage_slot_bit_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_BIT");
      stage_slot_bit_env != nullptr &&
      stage_slot_bit_env[0] != '\0') {
    config.stage_slot_bit = std::atoi(stage_slot_bit_env);
  }
  if (char const* stage_slot_low_byte_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_LOW_BYTE");
      stage_slot_low_byte_env != nullptr &&
      stage_slot_low_byte_env[0] != '\0' &&
      stage_slot_low_byte_env[0] != '0') {
    config.stage_slot_low_byte = 1;
  }
  if (char const* use_position_map_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_PROBE_USE_POSITION_MAP");
      use_position_map_env != nullptr &&
      use_position_map_env[0] != '\0' &&
      use_position_map_env[0] != '0') {
    config.use_position_map = 1;
  }
  if (char const* input_rows_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_INPUT_ROWS");
      input_rows_env != nullptr &&
      input_rows_env[0] != '\0') {
    config.input_rows = std::atoi(input_rows_env);
  }
  if (char const* packed_row_bytes_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_PACKED_ROW_BYTES");
      packed_row_bytes_env != nullptr &&
      packed_row_bytes_env[0] != '\0') {
    config.packed_row_bytes = std::atoi(packed_row_bytes_env);
  }
  if (char const* input_file_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_INPUT_FILE");
      input_file_env != nullptr &&
      input_file_env[0] != '\0') {
    config.use_input_file = 1;
  }
  if (char const* scale_file_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_SCALE_FILE");
      scale_file_env != nullptr &&
      scale_file_env[0] != '\0') {
    config.use_scale_file = 1;
  }
  if (char const* hidden_size_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_HIDDEN_SIZE");
      hidden_size_env != nullptr &&
      hidden_size_env[0] != '\0') {
    config.hidden_size = std::atoi(hidden_size_env);
    config.scale_blocks_per_row = config.hidden_size / 16;
    config.padded_blocks_per_row =
        static_cast<int>(RoundUp(static_cast<std::size_t>(config.scale_blocks_per_row), 4u));
  }
  if (char const* scale_block_base_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_SCALE_BLOCK_BASE");
      scale_block_base_env != nullptr &&
      scale_block_base_env[0] != '\0') {
    config.scale_block_base = std::atoi(scale_block_base_env);
  }
  char const* tids_env = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_PROBE_TIDS");
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

template <class ScaleTensor, int kScaleByteCount>
__device__ __forceinline__ void StoreTracedScaleBytesLocal(
    ScaleTensor& scale_tensor,
    const std::uint8_t (&scale_bytes)[kScaleByteCount],
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
  const int logical_cols = static_cast<int>(cute::size<1>(scale_tensor));
  const int segment_len = logical_cols > 0 ? max(1, logical_cols / kScaleByteCount) : 1;
#pragma unroll
  for (int scale_col = 0; scale_col < logical_cols; ++scale_col) {
    const int byte_index = min(scale_col / segment_len, kScaleByteCount - 1);
    StoreScaleTensorByte(scale_tensor, row, scale_col, scale_bytes[byte_index]);
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

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto NanoP1PartitionScaleBView(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  auto thr_tensor = cute::make_tensor(
      static_cast<SFBTensor&&>(sfbtensor).data(),
      nemotron::nvfp4_bridge::NanoP1ThrfrgSFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk =
      cute::make_coord(cute::get<0>(thr_vmnk), cute::make_coord(cute::get<2>(thr_vmnk), cute::get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
}

bool CheckCuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::printf(
      "nano_p1_reference_b_operand_probe: CUDA failure at %s: %s\n",
      what,
      cudaGetErrorString(status));
  return false;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

template <class T>
std::string DemangledTypeName() {
  int status = 0;
  char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
  if (status != 0 || demangled == nullptr) {
    return typeid(T).name();
  }
  std::string result(demangled);
  std::free(demangled);
  return result;
}

__device__ int ProbeThreadSlot(ProbeConfig const& config, int tid) {
  for (int idx = 0; idx < config.tracked_tid_count; ++idx) {
    if (config.tracked_tids[idx] == tid) {
      return idx;
    }
  }
  return -1;
}

__device__ __forceinline__ void ClearBOperandSource(std::uint8_t* swizzled_b_bytes) {
  for (int linear = static_cast<int>(threadIdx.x);
       linear < static_cast<int>(sizeof(SmemAllocB) * kSwizzledBElems);
       linear += static_cast<int>(blockDim.x)) {
    swizzled_b_bytes[linear] = 0u;
  }
}

template <class Stage0Layout>
__device__ __forceinline__ std::uint8_t BOperandProbeValueForCoord(
    ProbeConfig const& config,
    Stage0Layout const& stage0_B,
    std::uint8_t const* input_bytes,
    int row,
    int col) {
  int const stage0_offset = static_cast<int>(stage0_B(row, col));
  if (config.use_input_file != 0 && input_bytes != nullptr) {
    int source_row = row;
    if (config.use_position_map != 0) {
      source_row = nemotron::nvfp4_bridge::NanoP1SourceRowForBStageOffset(stage0_offset);
    }
    int const byte_index = col / 2;
    if (source_row >= 0 &&
        source_row < config.input_rows &&
        byte_index >= 0 &&
        byte_index < config.packed_row_bytes) {
      std::size_t const src_offset =
          static_cast<std::size_t>(source_row) * static_cast<std::size_t>(config.packed_row_bytes) +
          static_cast<std::size_t>(byte_index);
      return input_bytes[src_offset];
    }
    return 0u;
  }
  if (config.stage_slot_low_byte != 0) {
    return static_cast<std::uint8_t>((stage0_offset / 2) & 0xff);
  }
  if (config.stage_slot_bit >= 0) {
    return static_cast<std::uint8_t>(((stage0_offset / 2) >> config.stage_slot_bit) & 0x1);
  }
  return 0u;
}

template <class CoordTensor, class Stage0Layout>
__device__ __forceinline__ void OverwriteTrackedBOperandSource(
    CoordTensor const& coord_tensor,
    std::uint8_t* swizzled_b_bytes,
    Stage0Layout const& stage0_B,
    ProbeConfig const& config,
    std::uint8_t const* input_bytes) {
  using CoordTensorT = std::remove_cvref_t<CoordTensor>;
  if constexpr (CoordTensorT::rank == 1) {
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      auto const coord = coord_tensor(cute::make_coord(i));
      int const row = CoordGet0(coord);
      int const col = CoordGet1(coord);
      int const stage0_offset = static_cast<int>(stage0_B(row, col));
      swizzled_b_bytes[stage0_offset / 2] =
          BOperandProbeValueForCoord(config, stage0_B, input_bytes, row, col);
    }
  } else if constexpr (CoordTensorT::rank == 2) {
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        auto const coord = coord_tensor(cute::make_coord(i, j));
        int const row = CoordGet0(coord);
        int const col = CoordGet1(coord);
        int const stage0_offset = static_cast<int>(stage0_B(row, col));
        swizzled_b_bytes[stage0_offset / 2] =
            BOperandProbeValueForCoord(config, stage0_B, input_bytes, row, col);
      }
    }
  } else if constexpr (CoordTensorT::rank == 3) {
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          auto const coord = coord_tensor(cute::make_coord(i, j, k));
          int const row = CoordGet0(coord);
          int const col = CoordGet1(coord);
          int const stage0_offset = static_cast<int>(stage0_B(row, col));
          swizzled_b_bytes[stage0_offset / 2] =
              BOperandProbeValueForCoord(config, stage0_B, input_bytes, row, col);
        }
      }
    }
  } else if constexpr (CoordTensorT::rank == 4) {
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          for (int l = 0; l < cute::size<3>(coord_tensor); ++l) {
            auto const coord = coord_tensor(cute::make_coord(i, j, k, l));
            int const row = CoordGet0(coord);
            int const col = CoordGet1(coord);
            int const stage0_offset = static_cast<int>(stage0_B(row, col));
            swizzled_b_bytes[stage0_offset / 2] =
                BOperandProbeValueForCoord(config, stage0_B, input_bytes, row, col);
          }
        }
      }
    }
  }
}

__global__ void ReferenceBOperandProbeKernel(
    ProbeEntry* entries,
    ProbeConfig config,
    std::uint8_t const* input_bytes,
    std::uint8_t const* scale_bytes) {
  __shared__ cute::array_aligned<SmemAllocB, kSwizzledBElems> smem_B;
  __shared__ cute::array_aligned<nemotron::nvfp4_cute::ElementSFCompute, kScaleSmemElemsB> smem_SFB;

  int const tid = static_cast<int>(threadIdx.x);
  int const probe_slot = ProbeThreadSlot(config, tid);

  auto mma = TiledMma{};
  auto thread_mma = mma.get_thread_slice(tid);
  auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_B.data()), SmemLayoutB{});
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sSFB =
      cute::make_tensor(cute::make_smem_ptr(smem_SFB.data()), SmemLayoutSFB{});
  auto sSFB_stage0 = sSFB(cute::_, cute::_, cute::Int<0>{});
  auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
  auto* swizzled_b_bytes = reinterpret_cast<std::uint8_t*>(smem_B.data());

  auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
  auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, mma);
  auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
  auto tCsB = s2r_thr_B.partition_S(sB);
  auto tCrB_cv = s2r_thr_B.retile_D(tCrB);
  auto b_coords = cute::make_identity_tensor(cute::shape(sB));
  auto tCsB_coords = s2r_thr_B.partition_S(b_coords);
  auto tile_shape_mnk = cute::tile_shape(mma);
  auto layout_sfb_tv = Traits::GetLayoutSFBTV(mma);
  auto layout_sfb_tv_nano = nemotron::nvfp4_bridge::NanoP1GetLayoutSFBTV(mma);
  auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(sSFB_stage0, thread_mma);
  auto tCrSFB_nano = nemotron::nvfp4_bridge::NanoP1PartitionScaleB(sSFB_stage0, thread_mma);
  auto s2r_copy_SFB = cute::make_tiled_copy_impl(
      SmemCopyAtomSFB{},
      layout_sfb_tv,
      cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_copy_SFB_nano = cute::make_tiled_copy_impl(
      SmemCopyAtomSFB{},
      layout_sfb_tv_nano,
      cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(tid);
  auto s2r_thr_SFB_nano = s2r_copy_SFB_nano.get_thread_slice(tid);
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);
  auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
  auto tCsSFB_nano = s2r_thr_SFB_nano.partition_S(sScaleB);
  auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);
  auto tCrSFB_nano_cv = s2r_thr_SFB_nano.retile_D(tCrSFB_nano);
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto tCrSFB_coords =
      s2r_thr_SFB.retile_D(nemotron::nvfp4_bridge::PartitionScaleB(ref_sfb, thread_mma));
  auto tCrSFB_nano_coords =
      s2r_thr_SFB_nano.retile_D(NanoP1PartitionScaleBView(ref_sfb, thread_mma));

  auto dense_c = cute::make_identity_tensor(cute::make_shape(cute::Int<128>{}, cute::Int<128>{}));
  auto part_c = thread_mma.partition_C(dense_c);

  ClearBOperandSource(swizzled_b_bytes);
  __syncthreads();
  constexpr int kMacroScaleBytes = Traits::kMacroTileK / 16;
  for (int row = tid; row < kTileRows; row += blockDim.x) {
    std::uint8_t row_scale_bytes[kMacroScaleBytes] = {};
    if (config.use_scale_file != 0 && scale_bytes != nullptr) {
      for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
        int const block_index = config.scale_block_base + scale_index;
        if (block_index >= 0 && block_index < config.scale_blocks_per_row) {
          row_scale_bytes[scale_index] = scale_bytes[ExecutionScaleOffset(
              static_cast<std::size_t>(row),
              static_cast<std::size_t>(block_index),
              static_cast<std::size_t>(config.padded_blocks_per_row))];
        }
      }
    }
    StoreTracedScaleBytesLocal(sSFB_stage0, row_scale_bytes, row);
  }
  __syncthreads();
  cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);
  cute::copy(tCsSFB_nano(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_nano_cv);
  __syncthreads();
  for (int k_block = 0; k_block < cute::size<2>(tCrB_cv); ++k_block) {
    if (probe_slot >= 0) {
      OverwriteTrackedBOperandSource(
          tCsB_coords(cute::_, cute::_, k_block, cute::Int<0>{}),
          swizzled_b_bytes,
          stage0_B,
          config,
          input_bytes);
    }
    __syncthreads();
    cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, k_block, cute::Int<0>{}), tCrB_cv(cute::_, cute::_, k_block));
    __syncthreads();
  }

  if (probe_slot < 0) {
    return;
  }

  int const n_tiles = cute::size<2>(part_c);
  int const k_blocks = cute::size<2>(tCrB_cv);
  for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
    auto c_atom_coords = part_c(cute::_, 0, n_tile);
    auto coord0 = c_atom_coords(0);
    int const part_output_col0 = CoordGet0(coord0);
    int const part_token_row0 = CoordGet1(coord0);

    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      auto row_anchor = tCsB_coords(cute::_, n_tile, k_block, cute::Int<0>{});
      auto copy_coord0 = row_anchor(0);
      auto copy_coord1 = row_anchor(1);
      int const local_row0 = CoordGet0(copy_coord0);
      int const local_col0 = CoordGet1(copy_coord0);
      int const local_row1 = CoordGet0(copy_coord1);
      int const local_col1 = CoordGet1(copy_coord1);
      int const stage0_offset0 = static_cast<int>(stage0_B(local_row0, local_col0));
      int const stage0_offset1 = static_cast<int>(stage0_B(local_row1, local_col1));
      auto scale_anchor = tCsSFB(cute::_, n_tile, k_block, cute::Int<0>{});
      auto scale_coord0 = scale_anchor(0);

      auto b_atom_words = cute::recast<BRegister>(tCrB(cute::_, n_tile, k_block));
      const std::uint32_t packed_scale_traced =
          PackScaleFragmentWordLocal(tCrSFB(cute::_, n_tile, k_block));
      auto scale_coord_view_generic = tCrSFB_coords(cute::_, n_tile, k_block);
      auto scale_coord_view_nano = tCrSFB_nano_coords(cute::_, n_tile, k_block);
      std::uint32_t packed_scale_generic = 0u;
      const std::uint32_t packed_scale_nano =
          PackScaleFragmentWordLocal(tCrSFB_nano(cute::_, n_tile, k_block));
      int scale_word_count_traced = 0;
      int scale_word_count_generic = 0;
      int scale_rows_generic[4] = {0, 0, 0, 0};
      int scale_cols_generic[4] = {0, 0, 0, 0};
      nemotron::nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
          scale_coord_view_generic,
          4,
          scale_rows_generic,
          scale_cols_generic);
      for (int elem = 0; elem < 4; ++elem) {
        packed_scale_generic |= static_cast<std::uint32_t>(
                                    LoadScaleValueByte(
                                        sSFB(scale_rows_generic[elem], scale_cols_generic[elem], cute::Int<0>{})))
                                << (elem * 8);
        ++scale_word_count_traced;
        ++scale_word_count_generic;
      }

      int const entry_index = probe_slot * 16 + n_tile * 2 + k_block;
      entries[entry_index] = ProbeEntry{
          .valid = 1,
          .tid = tid,
          .n_tile = n_tile,
          .k_block = k_block,
          .tcrb_k_blocks = cute::size<2>(tCrB),
          .tcrb_copy_view_k_blocks = cute::size<2>(tCrB_cv),
          .tcsb_coord_k_blocks = cute::size<2>(tCsB_coords),
          .tcrsfb_k_blocks = cute::size<2>(tCrSFB_coords),
          .tcrsfb_copy_view_k_blocks = cute::size<2>(tCrSFB_coords),
          .tcssfb_coord_k_blocks = cute::size<2>(tCsSFB),
          .tcrb_dim0 = cute::size<0>(tCrB),
          .tcrb_dim1 = cute::size<1>(tCrB),
          .tcrb_copy_view_dim0 = cute::size<0>(tCrB_cv),
          .tcrb_copy_view_dim1 = cute::size<1>(tCrB_cv),
          .tcsb_coord_dim0 = cute::size<0>(tCsB_coords),
          .tcsb_coord_dim1 = cute::size<1>(tCsB_coords),
          .tcsb_coord_stage_count = cute::size<3>(tCsB_coords),
          .tcrsfb_dim0 = cute::size<0>(tCrSFB_coords),
          .tcrsfb_dim1 = cute::size<1>(tCrSFB_coords),
          .tcrsfb_copy_view_dim0 = cute::size<0>(tCrSFB_coords),
          .tcrsfb_copy_view_dim1 = cute::size<1>(tCrSFB_coords),
          .tcssfb_coord_dim0 = cute::size<0>(tCsSFB),
          .tcssfb_coord_dim1 = cute::size<1>(tCsSFB),
          .tcssfb_coord_stage_count = cute::size<3>(tCsSFB),
          .sf_vec_size = TiledMma::Traits::SFVecSize,
          .smem_layout_atom_sfb_dim0 = cute::size<0>(SmemLayoutAtomSFB{}),
          .smem_layout_atom_sfb_dim1 = cute::size<1>(SmemLayoutAtomSFB{}),
          .sfb_stage_elems = cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})),
          .sfb_total_elems = cute::cosize_v<SmemLayoutSFB>,
          .sfb_stage_shape_dim0 = cute::size<0>(SmemLayoutSFB{}),
          .sfb_stage_shape_dim1 = cute::size<1>(SmemLayoutSFB{}),
          .layout_sfb_tv_dim0 = cute::size<0>(layout_sfb_tv),
          .layout_sfb_tv_dim1 = cute::size<1>(layout_sfb_tv),
          .part_output_col0 = part_output_col0,
          .part_token_row0 = part_token_row0,
          .local_row0 = local_row0,
          .local_col0 = local_col0,
          .local_row1 = local_row1,
          .local_col1 = local_col1,
          .stage0_offset0 = stage0_offset0,
          .stage0_offset1 = stage0_offset1,
          .scale_local_row0 = CoordGet0(scale_coord0),
          .scale_local_col0 = CoordGet1(scale_coord0),
          .scale_stage0_offset0 =
              static_cast<int>(SmemLayoutSFB{}(
                  CoordGet0(scale_coord0),
                  CoordGet1(scale_coord0),
                  cute::Int<0>{})),
          .reg0_pre = static_cast<std::uint32_t>(b_atom_words(0)),
          .reg1_pre = static_cast<std::uint32_t>(b_atom_words(1)),
      };
      entries[entry_index].scale_reg_word_count_traced = scale_word_count_traced > 0 ? 1 : 0;
      entries[entry_index].scale_reg_word_count_generic = scale_word_count_generic > 0 ? 1 : 0;
      entries[entry_index].scale_reg_word_count_nano = scale_word_count_traced > 0 ? 1 : 0;
      for (int word = 0; word < kMaxScaleRegWords; ++word) {
        entries[entry_index].scale_reg_words_traced[word] = 0u;
        entries[entry_index].scale_reg_words_generic[word] = 0u;
        entries[entry_index].scale_reg_words_nano[word] = 0u;
      }
      entries[entry_index].scale_reg_words_traced[0] = packed_scale_traced;
      entries[entry_index].scale_reg_words_generic[0] = packed_scale_generic;
      entries[entry_index].scale_reg_words_nano[0] = packed_scale_nano;
    }
  }
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::printf("nano_p1_reference_b_operand_probe: SKIP (no CUDA device)\n");
    return 0;
  }

  ProbeConfig config = LoadProbeConfig();
  std::vector<std::uint8_t> host_input_bytes;
  std::uint8_t* device_input_bytes = nullptr;
  std::vector<std::uint8_t> host_scale_bytes;
  std::uint8_t* device_scale_bytes = nullptr;
  if (config.use_input_file != 0) {
    char const* input_file_path = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_INPUT_FILE");
    if (input_file_path == nullptr || input_file_path[0] == '\0') {
      std::printf("nano_p1_reference_b_operand_probe: missing input file path\n");
      return 1;
    }
    std::ifstream input_stream(input_file_path, std::ios::binary);
    if (!input_stream) {
      std::printf(
          "nano_p1_reference_b_operand_probe: failed to open input file %s\n",
          input_file_path);
      return 1;
    }
    host_input_bytes.assign(
        std::istreambuf_iterator<char>(input_stream),
        std::istreambuf_iterator<char>());
    std::size_t const required_bytes =
        static_cast<std::size_t>(config.input_rows) * static_cast<std::size_t>(config.packed_row_bytes);
    if (host_input_bytes.size() < required_bytes) {
      std::printf(
          "nano_p1_reference_b_operand_probe: input file too small (%zu < %zu)\n",
          host_input_bytes.size(),
          required_bytes);
      return 1;
    }
    if (!CheckCuda(cudaMalloc(&device_input_bytes, required_bytes), "cudaMalloc(input_bytes)") ||
        !CheckCuda(cudaMemcpy(
                       device_input_bytes,
                       host_input_bytes.data(),
                       required_bytes,
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy(input_bytes)")) {
      if (device_input_bytes != nullptr) {
        cudaFree(device_input_bytes);
      }
      return 1;
    }
  }
  if (config.use_scale_file != 0) {
    char const* scale_file_path = std::getenv("NEMOTRON_NANO_P1_REFERENCE_B_SCALE_FILE");
    if (scale_file_path == nullptr || scale_file_path[0] == '\0') {
      std::printf("nano_p1_reference_b_operand_probe: missing scale file path\n");
      if (device_input_bytes != nullptr) {
        cudaFree(device_input_bytes);
      }
      return 1;
    }
    std::ifstream scale_stream(scale_file_path, std::ios::binary);
    if (!scale_stream) {
      std::printf(
          "nano_p1_reference_b_operand_probe: failed to open scale file %s\n",
          scale_file_path);
      if (device_input_bytes != nullptr) {
        cudaFree(device_input_bytes);
      }
      return 1;
    }
    host_scale_bytes.assign(
        std::istreambuf_iterator<char>(scale_stream),
        std::istreambuf_iterator<char>());
    std::size_t const required_scale_bytes =
        static_cast<std::size_t>(config.input_rows) *
        static_cast<std::size_t>(config.padded_blocks_per_row);
    if (host_scale_bytes.size() < required_scale_bytes) {
      std::printf(
          "nano_p1_reference_b_operand_probe: scale file too small (%zu < %zu)\n",
          host_scale_bytes.size(),
          required_scale_bytes);
      if (device_input_bytes != nullptr) {
        cudaFree(device_input_bytes);
      }
      return 1;
    }
    if (!CheckCuda(cudaMalloc(&device_scale_bytes, required_scale_bytes), "cudaMalloc(scale_bytes)") ||
        !CheckCuda(cudaMemcpy(
                       device_scale_bytes,
                       host_scale_bytes.data(),
                       required_scale_bytes,
                       cudaMemcpyHostToDevice),
                   "cudaMemcpy(scale_bytes)")) {
      if (device_scale_bytes != nullptr) {
        cudaFree(device_scale_bytes);
      }
      if (device_input_bytes != nullptr) {
        cudaFree(device_input_bytes);
      }
      return 1;
    }
  }
  ProbeEntry* device_entries = nullptr;
  ProbeEntry host_entries[kProbeEntryCount] = {};
  if (!CheckCuda(cudaMalloc(&device_entries, sizeof(host_entries)), "cudaMalloc(entries)")) {
    if (device_scale_bytes != nullptr) {
      cudaFree(device_scale_bytes);
    }
    if (device_input_bytes != nullptr) {
      cudaFree(device_input_bytes);
    }
    return 1;
  }
  if (!CheckCuda(cudaMemset(device_entries, 0, sizeof(host_entries)), "cudaMemset(entries)")) {
    cudaFree(device_entries);
    if (device_scale_bytes != nullptr) {
      cudaFree(device_scale_bytes);
    }
    if (device_input_bytes != nullptr) {
      cudaFree(device_input_bytes);
    }
    return 1;
  }

  ReferenceBOperandProbeKernel<<<1, kThreadsPerCta>>>(
      device_entries,
      config,
      device_input_bytes,
      device_scale_bytes);
  if (!CheckCuda(cudaGetLastError(), "ReferenceBOperandProbeKernel launch") ||
      !CheckCuda(cudaDeviceSynchronize(), "ReferenceBOperandProbeKernel sync") ||
      !CheckCuda(cudaMemcpy(host_entries, device_entries, sizeof(host_entries), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(entries)")) {
    cudaFree(device_entries);
    if (device_scale_bytes != nullptr) {
      cudaFree(device_scale_bytes);
    }
    if (device_input_bytes != nullptr) {
      cudaFree(device_input_bytes);
    }
    return 1;
  }
  cudaFree(device_entries);
  if (device_scale_bytes != nullptr) {
    cudaFree(device_scale_bytes);
  }
  if (device_input_bytes != nullptr) {
    cudaFree(device_input_bytes);
  }

  std::printf("nano_p1_reference_b_operand_probe: tracked_tids=");
  for (int idx = 0; idx < config.tracked_tid_count; ++idx) {
    std::printf("%s%d", idx == 0 ? "" : ",", config.tracked_tids[idx]);
  }
  std::printf("\n");
  std::printf(
      "nano_p1_reference_b_operand_probe: profile=%s\n",
      ReferenceProfileName());
  std::printf(
      "nano_p1_reference_b_operand_probe: source_row_mode=%s payload_mode=%s\n",
      config.use_position_map != 0 ? "position_map" : "row_identity",
      config.use_input_file != 0 ? "input_file" :
      (config.stage_slot_low_byte != 0 ? "stage_slot_low_byte" :
       (config.stage_slot_bit >= 0 ? "stage_slot_bit" : "zero")));
  std::printf(
      "nano_p1_reference_b_operand_probe: scale_mode=%s hidden_size=%d scale_block_base=%d blocks_per_row=%d padded_blocks_per_row=%d\n",
      config.use_scale_file != 0 ? "input_file" : "zero",
      config.hidden_size,
      config.scale_block_base,
      config.scale_blocks_per_row,
      config.padded_blocks_per_row);
  std::printf(
      "nano_p1_reference_b_operand_probe: dispatch=(%d,%d) dispatch_policy=%s\n",
      DispatchPolicy::Stages,
      DispatchPolicy::SchedulerPipelineStageCount,
      DemangledTypeName<DispatchPolicy>().c_str());
  std::printf(
      "nano_p1_reference_b_operand_probe: smem_copy_atom_b=%s smem_alloc_b_size=%zu\n",
      DemangledTypeName<SmemCopyAtomB>().c_str(),
      sizeof(SmemAllocB));
  std::printf(
      "nano_p1_reference_b_operand_probe: shared_storage_bytes=(%zu,%zu)\n",
      sizeof(typename nemotron::nvfp4_bridge::TracedP13Epilogue::SharedStorage),
      sizeof(typename CollectiveMainloop::SharedStorage));

  for (int tid_idx = 0; tid_idx < config.tracked_tid_count; ++tid_idx) {
    int const tid = config.tracked_tids[tid_idx];
    std::printf("nano_p1_reference_b_operand_probe: ---- tid=%d ----\n", tid);
    for (int entry_idx = 0; entry_idx < kProbeEntryCount; ++entry_idx) {
      ProbeEntry const& entry = host_entries[entry_idx];
      if (entry.valid == 0 || entry.tid != tid) {
        continue;
      }
      std::printf(
          "  n_tile=%d k_block=%d dispatch=(%d,%d) k_counts=(%d,%d,%d) "
          "shapes=tCrB(%d,%d,%d) tCrBcv(%d,%d,%d) tCsB(%d,%d,%d,%d) "
          "sf_counts=(%d,%d,%d) "
          "sf_shapes=tCrSFB(%d,%d,%d) tCrSFBcv(%d,%d,%d) tCsSFB(%d,%d,%d,%d) "
          "sf_atom=(%d,%d,%d) "
          "sf_layout=(%d,%d,%d,%d,%d,%d) "
          "part_token_row0=%d part_output_col0=%d "
          "local_row0=%d local_col0=%d local_row1=%d local_col1=%d "
          "stage0_offset0=%d stage0_offset1=%d "
          "scale_local_row0=%d scale_local_col0=%d scale_stage0_offset0=%d "
          "scale_reg_word_count_traced=%d scale_reg_traced=(0x%08x,0x%08x,0x%08x,0x%08x) "
          "scale_reg_word_count_generic=%d scale_reg_generic=(0x%08x,0x%08x,0x%08x,0x%08x) "
          "scale_reg_word_count_nano=%d scale_reg_nano=(0x%08x,0x%08x,0x%08x,0x%08x) "
          "source_row_reg0_pre=0x%08x source_row_reg1_pre=0x%08x\n",
          entry.n_tile,
          entry.k_block,
          DispatchPolicy::Stages,
          DispatchPolicy::SchedulerPipelineStageCount,
          entry.tcrb_k_blocks,
          entry.tcrb_copy_view_k_blocks,
          entry.tcsb_coord_k_blocks,
          entry.tcrb_dim0,
          entry.tcrb_dim1,
          entry.tcrb_k_blocks,
          entry.tcrb_copy_view_dim0,
          entry.tcrb_copy_view_dim1,
          entry.tcrb_copy_view_k_blocks,
          entry.tcsb_coord_dim0,
          entry.tcsb_coord_dim1,
          entry.tcsb_coord_k_blocks,
          entry.tcsb_coord_stage_count,
          entry.tcrsfb_k_blocks,
          entry.tcrsfb_copy_view_k_blocks,
          entry.tcssfb_coord_k_blocks,
          entry.tcrsfb_dim0,
          entry.tcrsfb_dim1,
          entry.tcrsfb_k_blocks,
          entry.tcrsfb_copy_view_dim0,
          entry.tcrsfb_copy_view_dim1,
          entry.tcrsfb_copy_view_k_blocks,
          entry.tcssfb_coord_dim0,
          entry.tcssfb_coord_dim1,
          entry.tcssfb_coord_k_blocks,
          entry.tcssfb_coord_stage_count,
          entry.sf_vec_size,
          entry.smem_layout_atom_sfb_dim0,
          entry.smem_layout_atom_sfb_dim1,
          entry.sfb_stage_elems,
          entry.sfb_total_elems,
          entry.sfb_stage_shape_dim0,
          entry.sfb_stage_shape_dim1,
          entry.layout_sfb_tv_dim0,
          entry.layout_sfb_tv_dim1,
          entry.part_token_row0,
          entry.part_output_col0,
          entry.local_row0,
          entry.local_col0,
          entry.local_row1,
          entry.local_col1,
          entry.stage0_offset0,
          entry.stage0_offset1,
          entry.scale_local_row0,
          entry.scale_local_col0,
          entry.scale_stage0_offset0,
          entry.scale_reg_word_count_traced,
          entry.scale_reg_words_traced[0],
          entry.scale_reg_words_traced[1],
          entry.scale_reg_words_traced[2],
          entry.scale_reg_words_traced[3],
          entry.scale_reg_word_count_generic,
          entry.scale_reg_words_generic[0],
          entry.scale_reg_words_generic[1],
          entry.scale_reg_words_generic[2],
          entry.scale_reg_words_generic[3],
          entry.scale_reg_word_count_nano,
          entry.scale_reg_words_nano[0],
          entry.scale_reg_words_nano[1],
          entry.scale_reg_words_nano[2],
          entry.scale_reg_words_nano[3],
          entry.reg0_pre,
          entry.reg1_pre);
    }
  }

  return 0;
}
