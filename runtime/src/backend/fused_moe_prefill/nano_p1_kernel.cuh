constexpr int kNanoP1MicroTileK = 64;
constexpr int kNanoP1MicroTileBytes = kNanoP1MicroTileK / 2;
constexpr int kNanoP1MicroScaleBytes = kNanoP1MicroTileK / fused_decode::kNvfp4BlockWidth;
constexpr int kNanoP1ProbeDebugOffset =
    nvfp4_bridge::NanoP1ThreadsPerCta * nvfp4_bridge::kNanoP1AccumCoordCount;
constexpr int kNanoP1ProbeFinalThread0Offset = kNanoP1ProbeDebugOffset + 8;
constexpr int kNanoP1ProbeDeadLaneDebugOffset = kNanoP1ProbeFinalThread0Offset + 64;
constexpr int kNanoP1ProbeFinalThread2Offset = kNanoP1ProbeDeadLaneDebugOffset + 8;
constexpr int kNanoP1BOperandCopySigLen = 8;
constexpr int kNanoP1BOperandCopyRawSigLen = 32;
constexpr int kNanoP1BOperandFragRawSigLen = 16;
constexpr int kNanoP1BOperandScaleWordCount = 4;
constexpr int kNanoP1BOperandProbeEntryWords = 53;
constexpr int kNanoP1BOperandProbeTrackedTidCount = 4;
constexpr int kNanoP1BOperandProbeKStepCap = 2;
constexpr int kNanoP1BOperandProbeEntries =
    kNanoP1BOperandProbeTrackedTidCount * kNanoP1BOperandProbeKStepCap * 8 * 2;
constexpr int kNanoP1BOperandProbeOffset = kNanoP1ProbeFinalThread2Offset + 64;
constexpr int kNanoP1AScaleProbeEntryWords = 16;
constexpr int kNanoP1AScaleProbeTrackedTidCount = 2;
constexpr int kNanoP1AScaleProbeKStepCap = 32;
constexpr int kNanoP1AScaleProbeEntries =
    kNanoP1AScaleProbeTrackedTidCount * kNanoP1AScaleProbeKStepCap * 8 * 2;
constexpr int kNanoP1AScaleProbeOffset =
    kNanoP1BOperandProbeOffset + kNanoP1BOperandProbeEntries * kNanoP1BOperandProbeEntryWords;
constexpr int kNanoP1AOperandCopySigLen = 8;
constexpr int kNanoP1AOperandProbeEntryWords = 35;
constexpr int kNanoP1AOperandProbeTrackedTidCount = 4;
constexpr int kNanoP1AOperandProbeKStepCap = 2;
constexpr int kNanoP1AOperandProbeEntries =
    kNanoP1AOperandProbeTrackedTidCount * kNanoP1AOperandProbeKStepCap * 8 * 2;
constexpr int kNanoP1AOperandProbeOffset =
    kNanoP1AScaleProbeOffset + kNanoP1AScaleProbeEntries * kNanoP1AScaleProbeEntryWords;
constexpr int kNanoP1AScaleProbeScratchCount =
    kNanoP1AOperandProbeOffset + kNanoP1AOperandProbeEntries * kNanoP1AOperandProbeEntryWords;
static_assert(kNanoP1MicroScaleBytes == 4);

struct NanoP1SharedStorage {
  alignas(1024) cute::array_aligned<nvfp4_bridge::NanoP1SmemAllocA, nvfp4_bridge::kNanoP1SwizzledAElems> smem_A;
  alignas(1024) cute::array_aligned<nvfp4_bridge::NanoP1SmemAllocB, nvfp4_bridge::kNanoP1SwizzledBElems> smem_B;
  alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, nvfp4_bridge::kNanoP1ScaleStageElemsA> smem_SFA;
  alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, nvfp4_bridge::kNanoP1ScaleStageElemsB> smem_SFB;
  alignas(16) cute::array_aligned<std::uint8_t, cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{}) * kNanoP1MicroTileBytes>
      row_major_A;
  alignas(16) cute::array_aligned<std::uint8_t, cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}) * kNanoP1MicroTileBytes>
      row_major_B;
  alignas(16) cute::array_aligned<std::uint32_t, cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{})> scale_words_A;
  alignas(16) cute::array_aligned<std::uint32_t, cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})> scale_words_B;
};

__device__ __forceinline__ void NanoP1CtaBarrier() {
  asm volatile("bar.sync 0;\n" : : : "memory");
}

__device__ __forceinline__ int NanoP1BOperandProbeTidSlot(int tid) {
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

__device__ __forceinline__ int NanoP1AScaleProbeTidSlot(int tid) {
  switch (tid) {
    case 0:
      return 0;
    case 2:
      return 1;
    default:
      return -1;
  }
}

__device__ __forceinline__ int NanoP1AOperandProbeTidSlot(int tid) {
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

template <class CoordTensor, class F>
CUTE_HOST_DEVICE void NanoP1ForEachProbeCoord(CoordTensor const& coord_tensor, F&& f) {
  using CoordTensorT = std::remove_cvref_t<CoordTensor>;
  if constexpr (CoordTensorT::rank == 1) {
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      f(coord_tensor(cute::make_coord(i)), i);
    }
  } else if constexpr (CoordTensorT::rank == 2) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        f(coord_tensor(cute::make_coord(i, j)), physical++);
      }
    }
  } else if constexpr (CoordTensorT::rank == 3) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(coord_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(coord_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(coord_tensor); ++k) {
          f(coord_tensor(cute::make_coord(i, j, k)), physical++);
        }
      }
    }
  } else if constexpr (CoordTensorT::rank == 4) {
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
    static_assert(CoordTensorT::rank <= 4, "NanoP1ForEachProbeCoord only supports rank <= 4");
  }
}

template <class ValueTensor, class F>
CUTE_HOST_DEVICE void NanoP1ForEachProbeValue(ValueTensor const& value_tensor, F&& f) {
  using ValueTensorT = std::remove_cvref_t<ValueTensor>;
  if constexpr (ValueTensorT::rank == 1) {
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      f(value_tensor(cute::make_coord(i)), i);
    }
  } else if constexpr (ValueTensorT::rank == 2) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(value_tensor); ++j) {
        f(value_tensor(cute::make_coord(i, j)), physical++);
      }
    }
  } else if constexpr (ValueTensorT::rank == 3) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(value_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(value_tensor); ++k) {
          f(value_tensor(cute::make_coord(i, j, k)), physical++);
        }
      }
    }
  } else if constexpr (ValueTensorT::rank == 4) {
    int physical = 0;
    for (int i = 0; i < cute::size<0>(value_tensor); ++i) {
      for (int j = 0; j < cute::size<1>(value_tensor); ++j) {
        for (int k = 0; k < cute::size<2>(value_tensor); ++k) {
          for (int l = 0; l < cute::size<3>(value_tensor); ++l) {
            f(value_tensor(cute::make_coord(i, j, k, l)), physical++);
          }
        }
      }
    }
  } else {
    static_assert(ValueTensorT::rank <= 4, "NanoP1ForEachProbeValue only supports rank <= 4");
  }
}

template <class PackedValue>
CUTE_HOST_DEVICE std::uint8_t NanoP1LoadPackedValueByte(PackedValue const& value) {
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

template <bool kCaptureAccumulatorScratch>
__device__ __forceinline__ void ComputeNanoP1AccumTile(
    NanoP1SharedStorage& shared,
    const std::uint8_t* input_packed,
    const std::uint8_t* weight_packed,
    const std::uint8_t* input_exec_scales,
    const std::uint8_t* weight_exec_scales,
    int tid,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    int64_t hidden_size,
    float* accumulator_scratch,
    nvfp4_bridge::CRegister* accum_storage) {
  using NanoP1TiledMma = nvfp4_bridge::NanoP1TiledMma;
  using NanoP1AccumLayout = nvfp4_bridge::NanoP1AccumLayout;

  constexpr int kTileM = cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kTileN = cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kTileK = cute::size<2>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kMacroTileBytes = kTileK / 2;
  constexpr int kMacroScaleBytes = kTileK / fused_decode::kNvfp4BlockWidth;

  const std::size_t packed_row_bytes = static_cast<std::size_t>(hidden_size) / 2u;
  const std::size_t blocks_per_row =
      static_cast<std::size_t>(hidden_size) / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::uint32_t unit_scale_word = nvfp4_bridge::MakePackedUnitScaleWord();
  const std::uint8_t unit_scale_byte = nvfp4_bridge::LoadScaleByte(unit_scale_word, 0);

  auto accum_tensor = cute::make_tensor(accum_storage, NanoP1AccumLayout{});
  cute::clear(accum_tensor);

  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(tid);
  auto const stage0_A = nvfp4_bridge::NanoP1SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
  auto const stage0_B = nvfp4_bridge::NanoP1SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
  auto sA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_A.data()),
      nvfp4_bridge::NanoP1SmemLayoutA{});
  auto sB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_B.data()),
      nvfp4_bridge::NanoP1SmemLayoutB{});
  auto sSFA_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFA.data()),
      nvfp4_bridge::NanoP1SmemLayoutSFA{});
  auto sSFB_ = cute::make_tensor(
      cute::make_smem_ptr(shared.smem_SFB.data()),
      nvfp4_bridge::NanoP1SmemLayoutSFB{});
  auto sA = cute::as_position_independent_swizzle_tensor(sA_);
  auto sB = cute::as_position_independent_swizzle_tensor(sB_);
  auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA_);
  auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB_);
  auto* swizzled_a_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_A.data());
  auto* swizzled_b_bytes = reinterpret_cast<std::uint8_t*>(shared.smem_B.data());

  auto sA_stage0 = sA(cute::_, cute::_, cute::Int<0>{});
  auto sB_stage0 = sB(cute::_, cute::_, cute::Int<0>{});
  auto sSFA_stage0 = sSFA_(cute::_, cute::_, cute::Int<0>{});
  auto sSFB_stage0 = sSFB_(cute::_, cute::_, cute::Int<0>{});

  auto tCrA = thread_mma.partition_fragment_A(sA_stage0);
  auto tCrB = thread_mma.partition_fragment_B(sB_stage0);
  auto tCrSFA = nvfp4_bridge::NanoP1PartitionScaleA(sSFA_stage0, thread_mma);
  auto tCrSFB = nvfp4_bridge::NanoP1PartitionScaleB(sSFB_stage0, thread_mma);

  auto s2r_copy_A = cute::make_tiled_copy_A(nvfp4_bridge::NanoP1SmemCopyAtomA{}, mma);
  auto s2r_thr_A = s2r_copy_A.get_thread_slice(tid);
  auto tCsA = s2r_thr_A.partition_S(sA);
  auto a_coords = cute::make_identity_tensor(cute::shape(sA));
  auto tCsA_coords = s2r_thr_A.partition_S(a_coords);
  auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

  auto s2r_copy_B = cute::make_tiled_copy_B(nvfp4_bridge::NanoP1SmemCopyAtomB{}, mma);
  auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
  auto tCsB = s2r_thr_B.partition_S(sB);
  auto b_coords = cute::make_identity_tensor(cute::shape(sB));
  auto tCsB_coords = s2r_thr_B.partition_S(b_coords);
  auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

  auto tile_shape_mnk = cute::tile_shape(mma);
  auto s2r_copy_SFA = cute::make_tiled_copy_impl(
      nvfp4_bridge::NanoP1SmemCopyAtomSFA{},
      nvfp4_bridge::NanoP1GetLayoutSFATV(mma),
      cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(tid);
  auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
  auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

  auto s2r_copy_SFB = cute::make_tiled_copy_impl(
      nvfp4_bridge::NanoP1SmemCopyAtomSFB{},
      nvfp4_bridge::NanoP1GetLayoutSFBTV(mma),
      cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
  auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(tid);
  auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
  auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}),
          cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);

  static_assert(cute::size<1>(decltype(tCrA){}) == cute::size<1>(decltype(tCrSFA){}));
  static_assert(cute::size<1>(decltype(tCrB){}) == cute::size<1>(decltype(tCrSFB){}));


  for (int64_t k_base = 0; k_base < hidden_size; k_base += kTileK) {
    const int64_t remaining_k = hidden_size - k_base;
    const std::size_t available_k = static_cast<std::size_t>(
        remaining_k < static_cast<int64_t>(kTileK) ? remaining_k : kTileK);
    const std::size_t available_bytes = available_k / 2u;
    const std::size_t available_blocks =
        available_k / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = static_cast<std::size_t>(k_base) / 2u;
    const std::size_t block_base =
        static_cast<std::size_t>(k_base) / fused_decode::kNvfp4BlockWidth;

    for (int row = tid; row < kTileN; row += blockDim.x) {
      const bool row_valid = row < valid_cols;
      const std::size_t source_row = static_cast<std::size_t>(output_col_base + row);
      const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
      for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
        std::uint8_t value = 0u;
        if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
          value = weight_packed[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const auto elem_offset = stage0_A(row, byte_index * 2);
        swizzled_a_bytes[static_cast<int>(elem_offset) / 2] = value;
      }
      std::uint8_t scale_bytes[kMacroScaleBytes];
      if (row_valid) {
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          std::uint8_t value = unit_scale_byte;
          if (static_cast<std::size_t>(scale_index) >= available_blocks) {
            value = 0u;
          } else if (weight_exec_scales != nullptr) {
            value = LoadExecutionScaleByte(
                weight_exec_scales,
                source_row,
                block_base + static_cast<std::size_t>(scale_index),
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
          }
          scale_bytes[scale_index] = value;
        }
        nvfp4_bridge::StoreTracedScaleBytes(sSFA_stage0, scale_bytes, row);
      } else {
        nvfp4_bridge::ZeroTracedP5ScaleRow(sSFA_stage0, row);
      }
    }

    // Live tactic-5 activation row-index probes match identity source rows at
    // this boundary, so B staging should read the logical token row directly
    // instead of applying an extra source-row permutation here.
    for (int row = tid; row < kTileM; row += blockDim.x) {
      const bool row_valid = row < valid_rows;
      const std::size_t source_row = static_cast<std::size_t>(row_start + row);
      const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
      for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
        std::uint8_t packed = 0u;
        if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
          packed = input_packed[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const auto elem_offset = stage0_B(row, byte_index * 2);
        swizzled_b_bytes[static_cast<int>(elem_offset) / 2] = packed;
      }
      std::uint8_t scale_bytes[kMacroScaleBytes];
      if (row_valid) {
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          std::uint8_t value = unit_scale_byte;
          if (static_cast<std::size_t>(scale_index) >= available_blocks) {
            value = 0u;
          } else if (input_exec_scales != nullptr) {
            value = LoadExecutionScaleByte(
                input_exec_scales,
                source_row,
                block_base + static_cast<std::size_t>(scale_index),
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
          }
          scale_bytes[scale_index] = value;
        }
        nvfp4_bridge::StoreTracedScaleBytes(sSFB_stage0, scale_bytes, row);
      } else {
        nvfp4_bridge::ZeroTracedP5ScaleRow(sSFB_stage0, row);
      }
    }

    NanoP1CtaBarrier();

    cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
    int probe_k_step_slot = -1;
    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          output_col_base == 0 &&
          row_start == 0) {
        if (k_base == 0) {
          probe_k_step_slot = 0;
        } else if (k_base + kTileK >= hidden_size) {
          probe_k_step_slot = 1;
        }
      }
    }
    for (int k_block = 0; k_block < cute::size<2>(decltype(tCrB){}) ; ++k_block) {
      cute::copy(
          s2r_copy_B,
          tCsB(cute::_, cute::_, k_block, cute::Int<0>{}),
          tCrB_cv(cute::_, cute::_, k_block));
      if constexpr (kCaptureAccumulatorScratch) {
        if (probe_k_step_slot >= 0) {
          constexpr int kProbeNTiles = cute::size<2>(NanoP1AccumLayout{});
          constexpr int kProbeKBlocks = cute::size<2>(decltype(tCrB){});
          const int probe_tid_slot = NanoP1BOperandProbeTidSlot(tid);
          if (probe_tid_slot >= 0) {
            for (int n_tile = 0; n_tile < kProbeNTiles; ++n_tile) {
              auto b_words_after_own_copy =
                  cute::recast<nvfp4_bridge::BRegister>(tCrB(cute::_, n_tile, k_block));
              int const entry_index =
                  (((probe_tid_slot * kNanoP1BOperandProbeKStepCap + probe_k_step_slot) *
                    kProbeNTiles) +
                   n_tile) *
                      kProbeKBlocks +
                  k_block;
              int const entry_base =
                  kNanoP1BOperandProbeOffset + entry_index * kNanoP1BOperandProbeEntryWords;
              accumulator_scratch[entry_base + 51] =
                  __uint_as_float(static_cast<std::uint32_t>(b_words_after_own_copy(0)));
              accumulator_scratch[entry_base + 52] =
                  __uint_as_float(static_cast<std::uint32_t>(b_words_after_own_copy(1)));
            }
          }
        }
      }
    }
    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          output_col_base == 0 &&
          row_start == 0) {
        constexpr int kProbeNTiles = cute::size<2>(NanoP1AccumLayout{});
        constexpr int kProbeKBlocks = cute::size<2>(decltype(tCrB){});
        const int probe_tid_slot = NanoP1BOperandProbeTidSlot(tid);
        if (probe_tid_slot >= 0 && probe_k_step_slot >= 0) {
          for (int n_tile = 0; n_tile < kProbeNTiles; ++n_tile) {
            for (int k_block = 0; k_block < kProbeKBlocks; ++k_block) {
              auto b_words_after_all_b_copies =
                  cute::recast<nvfp4_bridge::BRegister>(tCrB(cute::_, n_tile, k_block));
              int const entry_index =
                  (((probe_tid_slot * kNanoP1BOperandProbeKStepCap + probe_k_step_slot) *
                    kProbeNTiles) +
                   n_tile) *
                      kProbeKBlocks +
                  k_block;
              int const entry_base =
                  kNanoP1BOperandProbeOffset + entry_index * kNanoP1BOperandProbeEntryWords;
              accumulator_scratch[entry_base + 33] =
                  __uint_as_float(static_cast<std::uint32_t>(b_words_after_all_b_copies(0)));
              accumulator_scratch[entry_base + 34] =
                  __uint_as_float(static_cast<std::uint32_t>(b_words_after_all_b_copies(1)));
            }
          }
        }
      }
    }
    cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
    cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          output_col_base == 0 &&
          row_start == 0) {
        if (tid == 0 || tid == 2) {
          auto rA = cute::recast<nvfp4_bridge::ARegister>(tCrA);
          auto rB = cute::recast<nvfp4_bridge::BRegister>(tCrB);
          auto rSFA = cute::recast<nvfp4_bridge::SFRegister>(cute::filter_zeros(tCrSFA));
          auto rSFB = cute::recast<nvfp4_bridge::SFRegister>(cute::filter_zeros(tCrSFB));
          const int debug_offset =
              tid == 0 ? kNanoP1ProbeDebugOffset : kNanoP1ProbeDeadLaneDebugOffset;
          accumulator_scratch[debug_offset + 0] =
              __uint_as_float(static_cast<std::uint32_t>(rA(0)));
          accumulator_scratch[debug_offset + 1] =
              __uint_as_float(static_cast<std::uint32_t>(rA(1)));
          accumulator_scratch[debug_offset + 2] =
              __uint_as_float(static_cast<std::uint32_t>(rA(2)));
          accumulator_scratch[debug_offset + 3] =
              __uint_as_float(static_cast<std::uint32_t>(rA(3)));
          accumulator_scratch[debug_offset + 4] =
              __uint_as_float(static_cast<std::uint32_t>(rSFA(0)));
          accumulator_scratch[debug_offset + 5] =
              __uint_as_float(static_cast<std::uint32_t>(rB(0)));
          accumulator_scratch[debug_offset + 6] =
              __uint_as_float(static_cast<std::uint32_t>(rB(1)));
          accumulator_scratch[debug_offset + 7] =
              __uint_as_float(static_cast<std::uint32_t>(rSFB(0)));
        }

        constexpr int kProbeNTiles = cute::size<2>(NanoP1AccumLayout{});
        constexpr int kProbeKBlocks = cute::size<2>(decltype(tCrB){});
        static_assert(kProbeNTiles == 8);
        static_assert(kProbeKBlocks == 2);
        const int probe_tid_slot = NanoP1BOperandProbeTidSlot(tid);
        if (probe_tid_slot >= 0 && probe_k_step_slot >= 0) {
          const int k_step = static_cast<int>(k_base / kTileK);
          for (int n_tile = 0; n_tile < kProbeNTiles; ++n_tile) {
            auto c_atom_coords = part_c(cute::_, 0, n_tile);
            auto coord0 = c_atom_coords(0);
            int const part_output_col0 = nvfp4_bridge::CoordGet0(coord0);
            int const part_token_row0 = nvfp4_bridge::CoordGet1(coord0);
            for (int k_block = 0; k_block < kProbeKBlocks; ++k_block) {
              auto row_anchor = tCsB_coords(cute::_, n_tile, k_block, cute::Int<0>{});
              auto copy_coord0 = row_anchor(0);
              auto copy_coord1 = row_anchor(1);
              int const local_row0 = nvfp4_bridge::CoordGet0(copy_coord0);
              int const local_col0 = nvfp4_bridge::CoordGet1(copy_coord0);
              int const local_row1 = nvfp4_bridge::CoordGet0(copy_coord1);
              int const local_col1 = nvfp4_bridge::CoordGet1(copy_coord1);
              int const stage0_offset0 = static_cast<int>(stage0_B(local_row0, local_col0));
              int const stage0_offset1 = static_cast<int>(stage0_B(local_row1, local_col1));
              auto b_words = cute::recast<nvfp4_bridge::BRegister>(tCrB(cute::_, n_tile, k_block));
              int const entry_index =
                  (((probe_tid_slot * kNanoP1BOperandProbeKStepCap + probe_k_step_slot) *
                    kProbeNTiles) +
                   n_tile) *
                      kProbeKBlocks +
                  k_block;
              int const entry_base =
                  kNanoP1BOperandProbeOffset + entry_index * kNanoP1BOperandProbeEntryWords;
              accumulator_scratch[entry_base + 0] = __uint_as_float(1u);
              accumulator_scratch[entry_base + 1] =
                  __uint_as_float(static_cast<std::uint32_t>(tid));
              accumulator_scratch[entry_base + 2] =
                  __uint_as_float(static_cast<std::uint32_t>(k_step));
              accumulator_scratch[entry_base + 3] =
                  __uint_as_float(static_cast<std::uint32_t>(k_base));
              accumulator_scratch[entry_base + 4] =
                  __uint_as_float(static_cast<std::uint32_t>(n_tile));
              accumulator_scratch[entry_base + 5] =
                  __uint_as_float(static_cast<std::uint32_t>(k_block));
              accumulator_scratch[entry_base + 6] =
                  __uint_as_float(static_cast<std::uint32_t>(part_output_col0));
              accumulator_scratch[entry_base + 7] =
                  __uint_as_float(static_cast<std::uint32_t>(part_token_row0));
              accumulator_scratch[entry_base + 8] =
                  __uint_as_float(static_cast<std::uint32_t>(local_row0));
              accumulator_scratch[entry_base + 9] =
                  __uint_as_float(static_cast<std::uint32_t>(local_col0));
              accumulator_scratch[entry_base + 10] =
                  __uint_as_float(static_cast<std::uint32_t>(local_row1));
              accumulator_scratch[entry_base + 11] =
                  __uint_as_float(static_cast<std::uint32_t>(local_col1));
              accumulator_scratch[entry_base + 12] =
                  __uint_as_float(static_cast<std::uint32_t>(stage0_offset0));
              accumulator_scratch[entry_base + 13] =
                  __uint_as_float(static_cast<std::uint32_t>(stage0_offset1));
              accumulator_scratch[entry_base + 14] =
                  __uint_as_float(static_cast<std::uint32_t>(b_words(0)));
              accumulator_scratch[entry_base + 15] =
                  __uint_as_float(static_cast<std::uint32_t>(b_words(1)));
              int copy_sig_count = 0;
              for (int sig = 0; sig < kNanoP1BOperandCopySigLen; ++sig) {
                accumulator_scratch[entry_base + 17 + sig * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(0xffffffffu));
                accumulator_scratch[entry_base + 18 + sig * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(0xffffffffu));
              }
              NanoP1ForEachProbeCoord(row_anchor, [&](auto const& coord, int physical) {
                if (physical >= kNanoP1BOperandCopySigLen) {
                  return;
                }
                int const sig_stage_offset =
                    static_cast<int>(stage0_B(
                        nvfp4_bridge::CoordGet0(coord),
                        nvfp4_bridge::CoordGet1(coord)));
                copy_sig_count = physical + 1;
                accumulator_scratch[entry_base + 17 + physical * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(sig_stage_offset));
                accumulator_scratch[entry_base + 18 + physical * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(
                        swizzled_b_bytes[sig_stage_offset / 2]));
              });
              accumulator_scratch[entry_base + 16] =
                  __uint_as_float(static_cast<std::uint32_t>(copy_sig_count));
              std::uint8_t copy_view_raw_bytes[kNanoP1BOperandCopyRawSigLen] = {};
              auto byte_view = tCrB_cv(cute::_, n_tile, k_block);
              NanoP1ForEachProbeValue(byte_view, [&](auto const& value, int physical) {
                if (physical < kNanoP1BOperandCopyRawSigLen) {
                  copy_view_raw_bytes[physical] = NanoP1LoadPackedValueByte(value);
                }
              });
              for (int packed_word = 0; packed_word < kNanoP1BOperandCopyRawSigLen / 4; ++packed_word) {
                std::uint32_t packed = 0u;
                packed |= static_cast<std::uint32_t>(copy_view_raw_bytes[packed_word * 4 + 0]) << 0;
                packed |= static_cast<std::uint32_t>(copy_view_raw_bytes[packed_word * 4 + 1]) << 8;
                packed |= static_cast<std::uint32_t>(copy_view_raw_bytes[packed_word * 4 + 2]) << 16;
                packed |= static_cast<std::uint32_t>(copy_view_raw_bytes[packed_word * 4 + 3]) << 24;
                accumulator_scratch[entry_base + 35 + packed_word] = __uint_as_float(packed);
              }
              std::uint8_t frag_raw_bytes[kNanoP1BOperandFragRawSigLen] = {};
              auto frag_view = tCrB(cute::_, n_tile, k_block);
              NanoP1ForEachProbeValue(frag_view, [&](auto const& value, int physical) {
                if (physical < kNanoP1BOperandFragRawSigLen) {
                  frag_raw_bytes[physical] = NanoP1LoadPackedValueByte(value);
                }
              });
              for (int packed_word = 0; packed_word < kNanoP1BOperandFragRawSigLen / 4; ++packed_word) {
                std::uint32_t packed = 0u;
                packed |= static_cast<std::uint32_t>(frag_raw_bytes[packed_word * 4 + 0]) << 0;
                packed |= static_cast<std::uint32_t>(frag_raw_bytes[packed_word * 4 + 1]) << 8;
                packed |= static_cast<std::uint32_t>(frag_raw_bytes[packed_word * 4 + 2]) << 16;
                packed |= static_cast<std::uint32_t>(frag_raw_bytes[packed_word * 4 + 3]) << 24;
                accumulator_scratch[entry_base + 43 + packed_word] = __uint_as_float(packed);
              }
              for (int word = 0; word < kNanoP1BOperandScaleWordCount; ++word) {
                accumulator_scratch[entry_base + 47 + word] =
                    __uint_as_float(static_cast<std::uint32_t>(0u));
              }
              auto scale_words_post =
                  cute::recast<std::uint32_t>(cute::filter_zeros(tCrSFB(cute::_, n_tile, k_block)));
              NanoP1ForEachProbeValue(scale_words_post, [&](auto const& value, int physical) {
                if (physical < kNanoP1BOperandScaleWordCount) {
                  accumulator_scratch[entry_base + 47 + physical] =
                      __uint_as_float(static_cast<std::uint32_t>(value));
                }
              });
            }
          }
        }
      }
    }

    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          output_col_base == 0 &&
          row_start == 0) {
        constexpr int kProbeScaleNTiles = cute::size<2>(NanoP1AccumLayout{});
        constexpr int kProbeScaleKBlocks = cute::size<2>(decltype(tCrSFA){});
        static_assert(kProbeScaleNTiles == 8);
        static_assert(kProbeScaleKBlocks == 2);
        const int probe_tid_slot = NanoP1AScaleProbeTidSlot(tid);
        const int k_step = static_cast<int>(k_base / kTileK);
        if (probe_tid_slot >= 0 && k_step < kNanoP1AScaleProbeKStepCap) {
          for (int n_tile = 0; n_tile < kProbeScaleNTiles; ++n_tile) {
            auto c_atom_coords = part_c(cute::_, 0, n_tile);
            auto coord0 = c_atom_coords(0);
            int const part_output_col0 = nvfp4_bridge::CoordGet0(coord0);
            int const part_token_row0 = nvfp4_bridge::CoordGet1(coord0);
            for (int k_block = 0; k_block < kProbeScaleKBlocks; ++k_block) {
              auto reg_words =
                  cute::recast<std::uint32_t>(cute::filter_zeros(tCrSFA(cute::_, n_tile, k_block)));
              int const entry_index =
                  (((probe_tid_slot * kNanoP1AScaleProbeKStepCap + k_step) * kProbeScaleNTiles) +
                   n_tile) *
                      kProbeScaleKBlocks +
                  k_block;
              int const entry_base =
                  kNanoP1AScaleProbeOffset + entry_index * kNanoP1AScaleProbeEntryWords;
              accumulator_scratch[entry_base + 0] = __uint_as_float(1u);
              accumulator_scratch[entry_base + 1] =
                  __uint_as_float(static_cast<std::uint32_t>(tid));
              accumulator_scratch[entry_base + 2] =
                  __uint_as_float(static_cast<std::uint32_t>(k_step));
              accumulator_scratch[entry_base + 3] =
                  __uint_as_float(static_cast<std::uint32_t>(k_base));
              accumulator_scratch[entry_base + 4] =
                  __uint_as_float(static_cast<std::uint32_t>(block_base));
              accumulator_scratch[entry_base + 5] =
                  __uint_as_float(static_cast<std::uint32_t>(n_tile));
              accumulator_scratch[entry_base + 6] =
                  __uint_as_float(static_cast<std::uint32_t>(k_block));
              accumulator_scratch[entry_base + 7] =
                  __uint_as_float(static_cast<std::uint32_t>(part_output_col0));
              accumulator_scratch[entry_base + 8] =
                  __uint_as_float(static_cast<std::uint32_t>(part_token_row0));
              accumulator_scratch[entry_base + 9] =
                  __uint_as_float(static_cast<std::uint32_t>(cute::size(reg_words)));
              for (int word = 0; word < 4; ++word) {
                std::uint32_t value = 0u;
                if (word < cute::size(reg_words)) {
                  value = static_cast<std::uint32_t>(reg_words(word));
                }
                accumulator_scratch[entry_base + 10 + word] = __uint_as_float(value);
              }
            }
          }
        }
      }
    }

    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          output_col_base == 0 &&
          row_start == 0) {
        constexpr int kProbeATiles = cute::size<1>(decltype(tCrA){});
        constexpr int kProbeAKBlocks = cute::size<2>(decltype(tCrA){});
        const int probe_tid_slot = NanoP1AOperandProbeTidSlot(tid);
        int probe_k_step_slot = -1;
        if (k_base == 0) {
          probe_k_step_slot = 0;
        } else if (k_base + kTileK >= hidden_size) {
          probe_k_step_slot = 1;
        }
        if (probe_tid_slot >= 0 && probe_k_step_slot >= 0) {
          const int k_step = static_cast<int>(k_base / kTileK);
          for (int n_tile = 0; n_tile < kProbeATiles; ++n_tile) {
            auto c_atom_coords = part_c(cute::_, 0, n_tile);
            auto coord0 = c_atom_coords(0);
            int const part_output_col0 = nvfp4_bridge::CoordGet0(coord0);
            int const part_token_row0 = nvfp4_bridge::CoordGet1(coord0);
            for (int k_block = 0; k_block < kProbeAKBlocks; ++k_block) {
              auto row_anchor = tCsA_coords(cute::_, n_tile, k_block, cute::Int<0>{});
              auto copy_coord0 = row_anchor(0);
              auto copy_coord1 = row_anchor(1);
              int const local_row0 = nvfp4_bridge::CoordGet0(copy_coord0);
              int const local_col0 = nvfp4_bridge::CoordGet1(copy_coord0);
              int const local_row1 = nvfp4_bridge::CoordGet0(copy_coord1);
              int const local_col1 = nvfp4_bridge::CoordGet1(copy_coord1);
              int const stage0_offset0 = static_cast<int>(stage0_A(local_row0, local_col0));
              int const stage0_offset1 = static_cast<int>(stage0_A(local_row1, local_col1));
              auto a_words_pre = cute::recast<nvfp4_bridge::ARegister>(tCrA(cute::_, n_tile, k_block));
              int const entry_index =
                  (((probe_tid_slot * kNanoP1AOperandProbeKStepCap + probe_k_step_slot) * kProbeATiles) +
                   n_tile) *
                      kProbeAKBlocks +
                  k_block;
              int const entry_base =
                  kNanoP1AOperandProbeOffset + entry_index * kNanoP1AOperandProbeEntryWords;
              accumulator_scratch[entry_base + 0] = __uint_as_float(1u);
              accumulator_scratch[entry_base + 1] =
                  __uint_as_float(static_cast<std::uint32_t>(tid));
              accumulator_scratch[entry_base + 2] =
                  __uint_as_float(static_cast<std::uint32_t>(k_step));
              accumulator_scratch[entry_base + 3] =
                  __uint_as_float(static_cast<std::uint32_t>(k_base));
              accumulator_scratch[entry_base + 4] =
                  __uint_as_float(static_cast<std::uint32_t>(n_tile));
              accumulator_scratch[entry_base + 5] =
                  __uint_as_float(static_cast<std::uint32_t>(k_block));
              accumulator_scratch[entry_base + 6] =
                  __uint_as_float(static_cast<std::uint32_t>(part_output_col0));
              accumulator_scratch[entry_base + 7] =
                  __uint_as_float(static_cast<std::uint32_t>(part_token_row0));
              accumulator_scratch[entry_base + 8] =
                  __uint_as_float(static_cast<std::uint32_t>(local_row0));
              accumulator_scratch[entry_base + 9] =
                  __uint_as_float(static_cast<std::uint32_t>(local_col0));
              accumulator_scratch[entry_base + 10] =
                  __uint_as_float(static_cast<std::uint32_t>(local_row1));
              accumulator_scratch[entry_base + 11] =
                  __uint_as_float(static_cast<std::uint32_t>(local_col1));
              accumulator_scratch[entry_base + 12] =
                  __uint_as_float(static_cast<std::uint32_t>(stage0_offset0));
              accumulator_scratch[entry_base + 13] =
                  __uint_as_float(static_cast<std::uint32_t>(stage0_offset1));
              accumulator_scratch[entry_base + 14] =
                  __uint_as_float(static_cast<std::uint32_t>(a_words_pre(0)));
              accumulator_scratch[entry_base + 15] =
                  __uint_as_float(static_cast<std::uint32_t>(a_words_pre(1)));
              accumulator_scratch[entry_base + 16] = __uint_as_float(0u);
              accumulator_scratch[entry_base + 17] = __uint_as_float(0u);
              accumulator_scratch[entry_base + 18] = __uint_as_float(0u);
              for (int sig = 0; sig < kNanoP1AOperandCopySigLen; ++sig) {
                accumulator_scratch[entry_base + 19 + sig * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(0xffffffffu));
                accumulator_scratch[entry_base + 20 + sig * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(0xffffffffu));
              }
              int copy_sig_count = 0;
              NanoP1ForEachProbeCoord(row_anchor, [&](auto const& coord, int physical) {
                if (physical >= kNanoP1AOperandCopySigLen) {
                  return;
                }
                int const sig_stage_offset =
                    static_cast<int>(stage0_A(
                        nvfp4_bridge::CoordGet0(coord),
                        nvfp4_bridge::CoordGet1(coord)));
                copy_sig_count = physical + 1;
                accumulator_scratch[entry_base + 19 + physical * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(sig_stage_offset));
                accumulator_scratch[entry_base + 20 + physical * 2] =
                    __uint_as_float(static_cast<std::uint32_t>(
                        swizzled_a_bytes[sig_stage_offset / 2]));
              });
              accumulator_scratch[entry_base + 18] =
                  __uint_as_float(static_cast<std::uint32_t>(copy_sig_count));
            }
          }
        }
      }
    }

    using MMAOp = typename NanoP1TiledMma::MMA_Op;
    for (int k_block = 0; k_block < cute::size<2>(tCrA_cv); ++k_block) {
      cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k_block));
      cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k_block));
    }

    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          output_col_base == 0 &&
          row_start == 0) {
        constexpr int kProbeATiles = cute::size<1>(decltype(tCrA){});
        constexpr int kProbeAKBlocks = cute::size<2>(decltype(tCrA){});
        const int probe_tid_slot = NanoP1AOperandProbeTidSlot(tid);
        int probe_k_step_slot = -1;
        if (k_base == 0) {
          probe_k_step_slot = 0;
        } else if (k_base + kTileK >= hidden_size) {
          probe_k_step_slot = 1;
        }
        if (probe_tid_slot >= 0 && probe_k_step_slot >= 0) {
          for (int n_tile = 0; n_tile < kProbeATiles; ++n_tile) {
            for (int k_block = 0; k_block < kProbeAKBlocks; ++k_block) {
              auto a_words_post = cute::recast<nvfp4_bridge::ARegister>(tCrA(cute::_, n_tile, k_block));
              int const entry_index =
                  (((probe_tid_slot * kNanoP1AOperandProbeKStepCap + probe_k_step_slot) * kProbeATiles) +
                   n_tile) *
                      kProbeAKBlocks +
                  k_block;
              int const entry_base =
                  kNanoP1AOperandProbeOffset + entry_index * kNanoP1AOperandProbeEntryWords;
              accumulator_scratch[entry_base + 16] =
                  __uint_as_float(static_cast<std::uint32_t>(a_words_post(0)));
              accumulator_scratch[entry_base + 17] =
                  __uint_as_float(static_cast<std::uint32_t>(a_words_post(1)));
            }
          }
        }
      }
    }

    constexpr int kMmaKBlocks = cute::size<2>(decltype(tCrA){});
    for (int k_block = 0; k_block < kMmaKBlocks; ++k_block) {
      cute::gemm(
          mma,
          cute::make_zip_tensor(tCrA(cute::_, cute::_, k_block), tCrSFA(cute::_, cute::_, k_block)),
          cute::make_zip_tensor(tCrB(cute::_, cute::_, k_block), tCrSFB(cute::_, cute::_, k_block)),
          accum_tensor);
      if constexpr (kCaptureAccumulatorScratch) {
        if (accumulator_scratch != nullptr &&
            k_base == 0 &&
            k_block == 0 &&
            output_col_base == 0 &&
            row_start == 0) {
          const std::size_t thread_offset =
              static_cast<std::size_t>(threadIdx.x) *
              static_cast<std::size_t>(nvfp4_bridge::kNanoP1AccumCoordCount);
#pragma unroll
          for (int physical = 0; physical < nvfp4_bridge::kNanoP1AccumCoordCount; ++physical) {
            accumulator_scratch[thread_offset + static_cast<std::size_t>(physical)] =
                accum_storage[physical];
          }
        }
      }
    }

    NanoP1CtaBarrier();
  }

  if constexpr (kCaptureAccumulatorScratch) {
    if (accumulator_scratch != nullptr &&
        output_col_base == 0 &&
        row_start == 0 &&
        tid == 0) {
      for (int physical = 0; physical < nvfp4_bridge::kNanoP1AccumCoordCount; ++physical) {
        accumulator_scratch[kNanoP1ProbeFinalThread0Offset + physical] = accum_storage[physical];
      }
    }
    if (accumulator_scratch != nullptr &&
        output_col_base == 0 &&
        row_start == 0 &&
        tid == 2) {
      for (int physical = 0; physical < nvfp4_bridge::kNanoP1AccumCoordCount; ++physical) {
        accumulator_scratch[kNanoP1ProbeFinalThread2Offset + physical] = accum_storage[physical];
      }
    }
  }
}

template <nvfp4_bridge::NanoP1EpilogueMode Mode>
__global__ __launch_bounds__(nvfp4_bridge::NanoP1ThreadsPerCta) void NanoP1Kernel(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    __nv_bfloat16* bf16_output,
    std::uint8_t* packed_output,
    std::uint8_t* block_scales_output,
    std::uint8_t* matmul_block_scales_output,
    float* activation_output_scales,
    float* direct_pack_staging,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    float* accumulator_scratch) {
  constexpr int kTileM = cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{});
  constexpr int kTileN = cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{});
  static_assert(fused_decode::kNvfp4BlockWidth == 16);

  __shared__ NanoP1SharedStorage shared;

  if (input_fp4 == nullptr ||
      weight_fp4 == nullptr ||
      g1_alphas == nullptr ||
      num_rows <= 0 ||
      hidden_size <= 0 ||
      inter_size <= 0 ||
      (hidden_size % fused_decode::kNvfp4BlockWidth) != 0) {
    return;
  }
  if constexpr (Mode == nvfp4_bridge::NanoP1EpilogueMode::kBf16Dense) {
    if (bf16_output == nullptr) {
      return;
    }
  } else {
    if (packed_output == nullptr ||
        block_scales_output == nullptr ||
        matmul_block_scales_output == nullptr ||
        activation_output_scales == nullptr ||
        direct_pack_staging == nullptr ||
        (inter_size % fused_decode::kNvfp4BlockWidth) != 0) {
      return;
    }
  }

  auto const* input_packed = static_cast<std::uint8_t const*>(input_fp4);
  auto const* weight_packed = static_cast<std::uint8_t const*>(weight_fp4);
  auto const* input_exec_scales = static_cast<std::uint8_t const*>(input_sf);
  auto const* weight_exec_scales = static_cast<std::uint8_t const*>(weight_sf);
  const float alpha = g1_alphas[0];
  const int tid = static_cast<int>(threadIdx.x);
  const int row_start = static_cast<int>(blockIdx.y) * kTileM;
  const int valid_rows =
      max(0, min(static_cast<int>(num_rows) - row_start, kTileM));
  if (valid_rows <= 0) {
    return;
  }

  if constexpr (Mode == nvfp4_bridge::NanoP1EpilogueMode::kBf16Dense) {
    const int output_col_base = static_cast<int>(blockIdx.x) * kTileN;
    const int valid_cols =
        max(0, min(static_cast<int>(inter_size) - output_col_base, kTileN));
    if (valid_cols <= 0) {
      return;
    }

    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kNanoP1AccumCoordCount];
    ComputeNanoP1AccumTile<true>(
        shared,
        input_packed,
        weight_packed,
        input_exec_scales,
        weight_exec_scales,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        hidden_size,
        accumulator_scratch,
        accum_storage);
    nvfp4_bridge::StoreNanoP1CFragmentsRowMajor(
        alpha,
        accum_storage,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        static_cast<std::size_t>(inter_size),
        bf16_output);
    return;
  }

  if (blockIdx.x != 0) {
    return;
  }

  for (int output_col_base = 0; output_col_base < inter_size; output_col_base += kTileN) {
    const int valid_cols =
        max(0, min(static_cast<int>(inter_size) - output_col_base, kTileN));
    if (valid_cols <= 0) {
      continue;
    }
    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kNanoP1AccumCoordCount];
    ComputeNanoP1AccumTile<false>(
        shared,
        input_packed,
        weight_packed,
        input_exec_scales,
        weight_exec_scales,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        hidden_size,
        nullptr,
        accum_storage);
    nvfp4_bridge::AccumulateNanoP1DirectPackRowMaxAbs(
        alpha,
        accum_storage,
        tid,
        row_start,
        valid_rows,
        valid_cols,
        activation_output_scales);
  }
  __syncthreads();

  for (int row = tid; row < valid_rows; row += blockDim.x) {
    const std::size_t row_index = static_cast<std::size_t>(row_start + row);
    const float max_abs = activation_output_scales[row_index];
    float row_scale = 1.0f;
    if (max_abs > kNvfp4ActivationMaxFinite) {
      row_scale = ClampNvfp4TensorScale(max_abs / kNvfp4ActivationMaxFinite);
    }
    activation_output_scales[row_index] = row_scale;
  }
  __syncthreads();

  constexpr std::size_t kStagingTileElements =
      static_cast<std::size_t>(kTileM) * static_cast<std::size_t>(kTileN);
  float* staging_tile =
      direct_pack_staging + (static_cast<std::size_t>(blockIdx.y) * kStagingTileElements);
  for (int output_col_base = 0; output_col_base < inter_size; output_col_base += kTileN) {
    const int valid_cols =
        max(0, min(static_cast<int>(inter_size) - output_col_base, kTileN));
    if (valid_cols <= 0) {
      continue;
    }
    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kNanoP1AccumCoordCount];
    ComputeNanoP1AccumTile<false>(
        shared,
        input_packed,
        weight_packed,
        input_exec_scales,
        weight_exec_scales,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        hidden_size,
        nullptr,
        accum_storage);
    nvfp4_bridge::StoreNanoP1DirectPackCFragments(
        alpha,
        accum_storage,
        tid,
        output_col_base,
        row_start,
        valid_rows,
        valid_cols,
        static_cast<int>(num_rows),
        static_cast<int>(inter_size),
        packed_output,
        block_scales_output,
        matmul_block_scales_output,
        activation_output_scales,
        staging_tile);
  }
}

bool RunNanoP1KernelForTestingImpl(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    __nv_bfloat16* bf16_output,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    cudaStream_t stream,
    float* accumulator_scratch) {
  if (input_fp4 == nullptr ||
      weight_fp4 == nullptr ||
      bf16_output == nullptr ||
      g1_alphas == nullptr ||
      num_rows <= 0 ||
      hidden_size <= 0 ||
      inter_size <= 0 ||
      (hidden_size % fused_decode::kNvfp4BlockWidth) != 0) {
    return false;
  }

  const dim3 block(nvfp4_bridge::NanoP1ThreadsPerCta);
  const dim3 grid(
      static_cast<unsigned int>(
          (static_cast<std::size_t>(inter_size) +
           static_cast<std::size_t>(cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
          static_cast<std::size_t>(cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{}))),
      static_cast<unsigned int>(
          (static_cast<std::size_t>(num_rows) +
           static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
          static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}))));
  if (grid.x == 0 || grid.y == 0) {
    return false;
  }

  NanoP1Kernel<nvfp4_bridge::NanoP1EpilogueMode::kBf16Dense><<<grid, block, 0, stream>>>(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      bf16_output,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      accumulator_scratch);
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaStreamSynchronize(stream));
}

bool RunNanoP1DirectPackKernelForTestingImpl(
    void const* input_fp4,
    void const* weight_fp4,
    void const* input_sf,
    void const* weight_sf,
    std::uint8_t* packed_output,
    std::uint8_t* block_scales_output,
    std::uint8_t* matmul_block_scales_output,
    float* activation_output_scales,
    float const* g1_alphas,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    cudaStream_t stream) {
  if (input_fp4 == nullptr ||
      weight_fp4 == nullptr ||
      packed_output == nullptr ||
      block_scales_output == nullptr ||
      matmul_block_scales_output == nullptr ||
      activation_output_scales == nullptr ||
      g1_alphas == nullptr ||
      num_rows <= 0 ||
      hidden_size <= 0 ||
      inter_size <= 0 ||
      (hidden_size % fused_decode::kNvfp4BlockWidth) != 0 ||
      (inter_size % fused_decode::kNvfp4BlockWidth) != 0) {
    return false;
  }

  const std::size_t packed_nbytes =
      static_cast<std::size_t>(num_rows) * (static_cast<std::size_t>(inter_size) / 2u);
  const std::size_t padded_blocks_per_row = RoundUp(
      static_cast<std::size_t>(inter_size) / fused_decode::kNvfp4BlockWidth,
      kNvfp4ScaleBlockTile);
  const std::size_t scale_nbytes = static_cast<std::size_t>(num_rows) * padded_blocks_per_row;
  const std::size_t row_tile_count =
      (static_cast<std::size_t>(num_rows) +
       static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
      static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}));
  float* direct_pack_staging_dev = nullptr;
  const std::size_t staging_count =
      row_tile_count *
      static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) *
      static_cast<std::size_t>(cute::size<1>(nvfp4_bridge::NanoP1MmaTileShape{}));
  if (!CheckCuda(cudaMalloc(
          reinterpret_cast<void**>(&direct_pack_staging_dev),
          staging_count * sizeof(float)))) {
    return false;
  }
  if (!CheckCuda(cudaMemsetAsync(packed_output, 0, packed_nbytes, stream)) ||
      !CheckCuda(cudaMemsetAsync(block_scales_output, 0, scale_nbytes, stream)) ||
      !CheckCuda(cudaMemsetAsync(matmul_block_scales_output, 0, scale_nbytes, stream)) ||
      !CheckCuda(cudaMemsetAsync(
          activation_output_scales,
          0,
          static_cast<std::size_t>(num_rows) * sizeof(float),
          stream))) {
    cudaFree(direct_pack_staging_dev);
    return false;
  }

  const dim3 block(nvfp4_bridge::NanoP1ThreadsPerCta);
  const dim3 grid(
      1u,
      static_cast<unsigned int>(
          (static_cast<std::size_t>(num_rows) +
           static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{})) - 1u) /
          static_cast<std::size_t>(cute::size<0>(nvfp4_bridge::NanoP1MmaTileShape{}))));
  if (grid.y == 0) {
    return false;
  }

  NanoP1Kernel<nvfp4_bridge::NanoP1EpilogueMode::kDirectPack><<<grid, block, 0, stream>>>(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      nullptr,
      packed_output,
      block_scales_output,
      matmul_block_scales_output,
      activation_output_scales,
      direct_pack_staging_dev,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      nullptr);
  const bool ok =
      CheckCuda(cudaGetLastError()) && CheckCuda(cudaStreamSynchronize(stream));
  cudaFree(direct_pack_staging_dev);
  return ok;
}
