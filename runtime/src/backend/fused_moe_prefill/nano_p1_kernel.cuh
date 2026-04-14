constexpr int kNanoP1MicroTileK = 64;
constexpr int kNanoP1MicroTileBytes = kNanoP1MicroTileK / 2;
constexpr int kNanoP1MicroScaleBytes = kNanoP1MicroTileK / fused_decode::kNvfp4BlockWidth;
constexpr int kNanoP1ProbeDebugOffset =
    nvfp4_bridge::NanoP1ThreadsPerCta * nvfp4_bridge::kNanoP1AccumCoordCount;
constexpr int kNanoP1ProbeFinalThread0Offset = kNanoP1ProbeDebugOffset + 8;
constexpr int kNanoP1ProbeDeadLaneDebugOffset = kNanoP1ProbeFinalThread0Offset + 64;
constexpr int kNanoP1ProbeFinalThread2Offset = kNanoP1ProbeDeadLaneDebugOffset + 8;
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
  auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

  auto s2r_copy_B = cute::make_tiled_copy_B(nvfp4_bridge::NanoP1SmemCopyAtomB{}, mma);
  auto s2r_thr_B = s2r_copy_B.get_thread_slice(tid);
  auto tCsB = s2r_thr_B.partition_S(sB);
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

    for (int row = tid; row < kTileM; row += blockDim.x) {
      const bool row_valid = row < valid_rows;
      const std::size_t source_row = static_cast<std::size_t>(row_start + row);
      const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
      for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
        std::uint8_t value = 0u;
        if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
          value = input_packed[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const auto elem_offset = stage0_B(row, byte_index * 2);
        swizzled_b_bytes[static_cast<int>(elem_offset) / 2] = value;
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
    cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
    cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
    cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

    using MMAOp = typename NanoP1TiledMma::MMA_Op;
    for (int k_block = 0; k_block < cute::size<2>(tCrA_cv); ++k_block) {
      cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k_block));
      cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k_block));
    }

    if constexpr (kCaptureAccumulatorScratch) {
      if (accumulator_scratch != nullptr &&
          k_base == 0 &&
          output_col_base == 0 &&
          row_start == 0 &&
          (tid == 0 || tid == 2)) {
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

