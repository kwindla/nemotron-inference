template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP13BFragmentsColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    BFragment64 (&fragments)[kTracedP13NFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP13SmemLayoutSFB{});
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = MakeTracedP13TiledCopySFB(mma);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP13ScaleFragmentCosizeB];
  int scale_cols[kTracedP13ScaleFragmentCosizeB];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfb,
      kTracedP13ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  for (int n_fragment = 0; n_fragment < kTracedP13NFragments; ++n_fragment) {
    fragments[n_fragment] =
        LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            thread_idx,
            n_fragment * 8);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = n_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[n_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP5BFragmentsColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    BFragment64 (&fragments)[kTracedP5NFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP5SmemLayoutSFB{});
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
  auto smem_tiled_copy_sfb = MakeTracedP5TiledCopySFB(mma);
  auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
  auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
  int scale_rows[kTracedP5ScaleFragmentCosizeB];
  int scale_cols[kTracedP5ScaleFragmentCosizeB];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfb,
      kTracedP5ScaleFragmentCosizeB,
      scale_rows,
      scale_cols);
  for (int n_fragment = 0; n_fragment < kTracedP5NFragments; ++n_fragment) {
    fragments[n_fragment] =
        LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            thread_idx,
            n_fragment * 8);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = n_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[n_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedP5CFragmentsTranspose(
    float alpha,
    const CFragment64 (&accum)[kTracedP5MFragments][kTracedP5NFragments],
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  int row_coords[kTracedP5CCopyCoordCapacity];
  int col_coords[kTracedP5CCopyCoordCapacity];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_c = cute::make_identity_tensor(
      cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
  auto part_c = thr_mma.partition_C(ref_c);
  FillPhysicalCoordMapCopyViewLimited(
      part_c,
      kTracedP5CCopyCoordCapacity,
      row_coords,
      col_coords);
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < kTracedP5MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < kTracedP5NFragments; ++n_fragment) {
        const int physical = reg * 8 + m_fragment * 2 + n_fragment;
        const int row = row_coords[physical];
        const int token = col_coords[physical];
        if (row >= output_rows_this_tile || token >= valid_rows) {
          continue;
        }
        const std::size_t input_row = row_start + static_cast<std::size_t>(token);
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row)] =
              __float2bfloat16(accum[m_fragment][n_fragment].regs[reg] * alpha);
        } else {
          output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row)] =
              accum[m_fragment][n_fragment].regs[reg] * alpha;
        }
      }
    }
  }
}



template <int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int lane_id,
    int n_base) {
  return LoadFragmentB_ColMajor64x8Tiled<SingleAtomTiledMma, kRowsPerTile>(
      packed_rows,
      scale_words,
      lane_id,
      n_base);
}

__device__ __forceinline__ void nvfp4_bridge::Clear(CFragment64& fragment) {
  fragment.regs[0] = 0.0f;
  fragment.regs[1] = 0.0f;
  fragment.regs[2] = 0.0f;
  fragment.regs[3] = 0.0f;
}

__device__ __forceinline__ std::uint32_t nvfp4_bridge::MakePackedUnitScaleWord() {
  const auto one = static_cast<std::uint8_t>(cute::float_ue4m3_t(1.0f).raw());
  return static_cast<std::uint32_t>(one) |
         (static_cast<std::uint32_t>(one) << 8) |
         (static_cast<std::uint32_t>(one) << 16) |
         (static_cast<std::uint32_t>(one) << 24);
}

__device__ __forceinline__ void nvfp4_bridge::Gemm(
    CFragment64& accum,
    const AFragment64& a,
    const BFragment64& b) {
  Sm120BlockScaledFp4Mma(
      accum.regs[0],
      accum.regs[1],
      accum.regs[2],
      accum.regs[3],
      a.regs[0],
      a.regs[1],
      a.regs[2],
      a.regs[3],
      b.regs[0],
      b.regs[1],
      static_cast<std::uint32_t>(a.scale[0]),
      static_cast<std::uint32_t>(b.scale[0]));
}

template <int kRowsPerTile, typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreFragmentC_RowMajor16x8(
    float alpha,
    const CFragment64& accum,
    int lane_id,
    int col_base,
    int row_base,
    int valid_rows,
    int valid_cols,
    std::size_t row_start,
    std::size_t output_row_base,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  StoreFp4AccumulatorTileRowMajor16x8<kRowsPerTile>(
      alpha,
      accum.regs[0],
      accum.regs[1],
      accum.regs[2],
      accum.regs[3],
      lane_id,
      col_base,
      row_base,
      valid_rows,
      valid_cols,
      row_start,
      output_row_base,
      output_rows_per_expert,
      output);
}

template <typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreFragmentC_Transpose16x8(
    float alpha,
    const CFragment64& accum,
    int lane_id,
    int output_row_base,
    int m_base,
    int token_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  if (token_base >= valid_rows) {
    return;
  }
  const int row_group = (lane_id >> 3) * 4;
  const int row0 = m_base + row_group + 0;
  const int row1 = m_base + row_group + 2;
  const int row2 = m_base + row_group + 1;
  const int row3 = m_base + row_group + 3;
  const std::size_t input_row = row_start + static_cast<std::size_t>(token_base);
  if (row0 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row0)] =
          __float2bfloat16(accum.regs[0] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row0)] =
          accum.regs[0] * alpha;
    }
  }
  if (row1 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row1)] =
          __float2bfloat16(accum.regs[1] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row1)] =
          accum.regs[1] * alpha;
    }
  }
  if (row2 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row2)] =
          __float2bfloat16(accum.regs[2] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row2)] =
          accum.regs[2] * alpha;
    }
  }
  if (row3 < output_rows_this_tile) {
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row3)] =
          __float2bfloat16(accum.regs[3] * alpha);
    } else {
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row_base + row3)] =
          accum.regs[3] * alpha;
    }
  }
}

void PrintP13ScaleTrace() {
  const auto& trace = g_p13_scale_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p13_scale_trace: invalid\n");
    return;
  }
  std::fprintf(
      stderr,
      "p13_scale_trace: row_start=%d valid_rows=%d output_row_base=%d\n",
      trace.row_start,
      trace.valid_rows,
      trace.output_row_base);
  for (int i = 0; i < 2; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: A m=%d base_row=%d source=0x%08x loaded=0x%08x smem=0x%08x fragment=0x%08x\n",
        i,
        trace.a_base_rows[i],
        trace.a_source_words[i],
        trace.a_loaded_words[i],
        trace.a_smem_words[i],
        trace.a_fragment_words[i]);
  }
  for (int i = 0; i < 2; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: B n=%d base_row=%d source=0x%08x loaded=0x%08x smem=0x%08x fragment=0x%08x\n",
        i,
        trace.b_base_rows[i],
        trace.b_source_words[i],
        trace.b_loaded_words[i],
        trace.b_smem_words[i],
        trace.b_fragment_words[i]);
  }
  for (int i = 0; i < 8; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: A coord[%d]=(%d,%d) raw=0x%02x\n",
        i,
        trace.a_scale_rows[i],
        trace.a_scale_cols[i],
        static_cast<unsigned>(trace.a_scale_raw[i]));
  }
  for (int i = 0; i < 8; ++i) {
    std::fprintf(
        stderr,
        "p13_scale_trace: B coord[%d]=(%d,%d) raw=0x%02x\n",
        i,
        trace.b_scale_rows[i],
        trace.b_scale_cols[i],
        static_cast<unsigned>(trace.b_scale_raw[i]));
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: A post-store row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.a_post_store_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: B post-store row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.b_post_store_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: A logical row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.a_logical_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
  for (int row = 0; row < 2; ++row) {
    std::fprintf(stderr, "p13_scale_trace: B logical row %d:", row);
    for (int col = 0; col < 8; ++col) {
      std::fprintf(stderr, " %02x", static_cast<unsigned>(trace.b_logical_raw[row * 8 + col]));
    }
    std::fprintf(stderr, "\n");
  }
}

void PrintP1DirectTrace() {
  const auto& trace = g_p1_direct_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p1_direct_trace: no capture\n");
    return;
  }
  for (int slot = 0; slot < 2; ++slot) {
    std::fprintf(
        stderr,
        "p1_direct_trace[%d]: warp=%d n_base=%d a_scale=0x%08x "
        "a={0x%08x,0x%08x,0x%08x,0x%08x} "
        "b_scale=0x%08x b={0x%08x,0x%08x}\n",
        slot,
        trace.warp_ids[slot],
        trace.n_base[slot],
        trace.a_scale[slot],
        trace.a_regs[slot][0],
        trace.a_regs[slot][1],
        trace.a_regs[slot][2],
        trace.a_regs[slot][3],
        trace.b_scale[slot],
        trace.b_regs[slot][0],
        trace.b_regs[slot][1]);
  }
}

__global__ void Nvfp4GroupedExpertMatVecRowsKernel(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x) / output_rows_per_expert;
  const std::size_t output_row = static_cast<std::size_t>(blockIdx.x) % output_rows_per_expert;
  if (expert_index >= n_experts) {
    return;
  }

  const int begin_row = expert_offsets[expert_index];
  const int end_row = expert_offsets[expert_index + 1];
  if (begin_row >= end_row) {
    return;
  }

  const fused_decode::Nvfp4WeightView weight =
      MakeDeviceWeightView(weights[expert_index]);
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * (weight.input_cols / 2);
  const std::size_t scale_row_offset = output_row * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;

  for (int tile_begin = begin_row; tile_begin < end_row; tile_begin += kGroupedTokenTile) {
    const int remaining_rows = end_row - tile_begin;
    const int valid_rows =
        remaining_rows < kGroupedTokenTile ? remaining_rows : kGroupedTokenTile;
    float accum[kGroupedTokenTile] = {0.0f};

    for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
         pair_index < pairs_per_row;
         pair_index += blockDim.x) {
      const std::size_t block = pair_index / 8u;
      const std::size_t pair_in_block = pair_index % 8u;
      const std::size_t col = block * fused_decode::kNvfp4BlockWidth + pair_in_block * 2u;
      const float block_scale =
          LoadNvfp4WeightBlockScale(weight, output_row, block);
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale;
      const float w1 = fused_decode::DecodeFp4((packed >> 4) & 0x0Fu) * block_scale;
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        const float* input_row =
            input + static_cast<std::size_t>(tile_begin + tile_row) * weight.input_cols;
        accum[tile_row] += input_row[col] * w0;
        accum[tile_row] += input_row[col + 1] * w1;
      }
    }

    for (int tile_row = 0; tile_row < kGroupedTokenTile; ++tile_row) {
      partial_sums[tile_row * blockDim.x + threadIdx.x] =
          tile_row < valid_rows ? accum[tile_row] : 0.0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
          partial_sums[tile_row * blockDim.x + threadIdx.x] +=
              partial_sums[tile_row * blockDim.x + threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        output[static_cast<std::size_t>(tile_begin + tile_row) * output_rows_per_expert +
               output_row] = partial_sums[tile_row * blockDim.x];
      }
    }
    __syncthreads();
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRowsKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  if (expert_index < 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      valid_rows <= 0 ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRowsBf16Kernel(
    const float* input,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_m_limits,
    const int* expert_first_token_offsets,
    int token_tile_dim,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int expert_index = cta_batch_indices[cta_index];
  const int batch_row_begin = expert_first_token_offsets[expert_index];
  const int batch_cta_begin = batch_row_begin / token_tile_dim;
  const int row_start =
      batch_row_begin + (cta_index - batch_cta_begin) * token_tile_dim;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows = max(0, min(m_limit - row_start, token_tile_dim));
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  if (expert_index < 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      valid_rows <= 0 ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        __float2bfloat16(c_tile[tile_output_row][tile_token]);
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRelu2MaxAbsKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    unsigned int* expert_max_bits) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];
  __shared__ float shared_max[kPlannedThreadsPerBlock];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count || expert_max_bits == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  float thread_max = 0.0f;
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const float value = fused_decode::Relu2(c_tile[tile_output_row][tile_token]);
    thread_max = fmaxf(thread_max, value);
  }

  shared_max[tid] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
      shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(&expert_max_bits[expert_index], __float_as_uint(shared_max[0]));
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRelu2PackKernel(
    const float* input,
    const float* output_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    std::size_t output_row_capacity,
    Nvfp4ScaleLayout output_scale_layout,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      output_expert_tensor_scales == nullptr ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_input_row = weight.input_cols / 2;
  const std::size_t input_blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_input_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * input_blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  if (lane_id >= valid_rows || warp_row >= output_rows_this_tile) {
    return;
  }

  const std::size_t output_row = static_cast<std::size_t>(row_start + lane_id);
  if (output_row >= output_row_capacity) {
    return;
  }
  const std::size_t blocks_per_output_row =
      output_rows_per_expert / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_output_row, kNvfp4ScaleBlockTile);
  const std::size_t block_col =
      static_cast<std::size_t>(output_row_base + warp_row) / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = output_expert_tensor_scales[expert_index];

  float activated[16];
  float block_max_abs = 0.0f;
  for (int col = 0; col < 16; ++col) {
    const float value = fused_decode::Relu2(c_tile[warp_row + col][lane_id]);
    activated[col] = value;
    if (value > block_max_abs) {
      block_max_abs = value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * tensor_scale));
  }
  const std::uint8_t encoded_block_scale = fused_decode::EncodeFp8Scale(block_scale);
  const std::size_t block_scale_offset = output_row * blocks_per_output_row + block_col;
  output_block_scales[block_scale_offset] = encoded_block_scale;
  output_matmul_scales[ExecutionScaleOffset(
      output_row, block_col, padded_blocks_per_row, output_scale_layout)] = encoded_block_scale;

  const float pack_scale = tensor_scale * block_scale;
  const std::size_t packed_row_offset =
      output_row * (output_rows_per_expert / 2u) +
      block_col * (fused_decode::kNvfp4BlockWidth / 2u);
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t lhs = fused_decode::EncodeFp4(activated[pair * 2] / pack_scale);
    const std::uint8_t rhs = fused_decode::EncodeFp4(activated[pair * 2 + 1] / pack_scale);
    output_packed[packed_row_offset + static_cast<std::size_t>(pair)] =
        static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
  }
}

template <typename OutputType, int kOutputTile, int kMacroTileK>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalse(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kWarpSubTiles =
      kOutputTile / (kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN);
  static_assert(kWarpSubTiles >= 1);
  __shared__ __nv_bfloat16 a_tile[2][kPlannedWmmaTileM][kMacroTileK];
  __shared__ __nv_bfloat16 b_tile[2][kOutputTile][kMacroTileK];
  __shared__ float c_tile_storage[kPlannedWmmaTileM * kOutputTile];
  auto* c_tile = reinterpret_cast<float (*)[kOutputTile]>(c_tile_storage);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const bool mma_warp = warp_id < kGroupedConsumerWarpsPerBlock;
  const bool producer_warp = warp_id >= kGroupedConsumerWarpsPerBlock;
  const int producer_tid = tid - (kGroupedConsumerWarpsPerBlock * 32);
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frags[kWarpSubTiles];
  if (mma_warp) {
    for (int i = 0; i < kWarpSubTiles; ++i) {
      wmma::fill_fragment(c_frags[i], 0.0f);
    }
  }

  auto stage_macro_tile = [&](int buffer_index, std::size_t k_base) {
    if (!producer_warp) {
      return;
    }
    const int producer_threads = kGroupedProducerWarpsPerBlock * 32;
    const int macro_k = static_cast<int>(
        min(static_cast<std::size_t>(kMacroTileK), weight.input_cols - k_base));
    const int macro_blocks = macro_k / static_cast<int>(fused_decode::kNvfp4BlockWidth);
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    for (int linear_block = producer_tid;
         linear_block < (kPlannedWmmaTileM * macro_blocks);
         linear_block += producer_threads) {
      const int tile_token = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &a_tile[buffer_index][tile_token]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      if (tile_token < valid_rows) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float row_ts =
            input_dq_scales != nullptr
                ? 1.0f
                : (input_expert_tensor_scales != nullptr
                       ? input_expert_tensor_scales[expert_index]
                       : (input_per_row_tensor_scales != nullptr
                              ? input_per_row_tensor_scales[input_row]
                              : *input_tensor_scale_data));
        DecodeGroupedPackedInputBlockBf16(
            packed_input,
            input_block_scales,
            input_dq_scales,
            row_ts,
            weight.input_cols,
            input_row,
            block_base + static_cast<std::size_t>(tile_block),
            dst);
      } else {
        ZeroBf16Block16(dst);
      }
    }
    for (int linear_block = producer_tid;
         linear_block < (output_rows_this_tile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &b_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      DecodeNvfp4WeightBlockBf16(
          weight,
          static_cast<std::size_t>(output_row_base + tile_output_row),
          block_base + static_cast<std::size_t>(tile_block),
          dst);
    }
    for (int linear_block = producer_tid + output_rows_this_tile * macro_blocks;
         linear_block < (kOutputTile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &b_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      ZeroBf16Block16(dst);
    }
  };

  const std::size_t k_step = static_cast<std::size_t>(kMacroTileK);
  int buffer_index = 0;
  stage_macro_tile(buffer_index, 0);
  __syncthreads();
  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += k_step) {
    const int macro_k = static_cast<int>(
        min(k_step, weight.input_cols - k_base));
    const int next_buffer = buffer_index ^ 1;
    const std::size_t next_k_base = k_base + k_step;
    if (next_k_base < weight.input_cols) {
      stage_macro_tile(next_buffer, next_k_base);
    }
    if (mma_warp) {
      wmma::fragment<
          wmma::matrix_a,
          kPlannedWmmaTileM,
          kPlannedWmmaTileN,
          kPlannedWmmaTileK,
          __nv_bfloat16,
          wmma::row_major>
          a_frag;
      for (int tile_k_base = 0; tile_k_base < macro_k; tile_k_base += kPlannedWmmaTileK) {
        wmma::load_matrix_sync(a_frag, &a_tile[buffer_index][0][tile_k_base], kMacroTileK);
        for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
          const int warp_col = warp_id * kPlannedWmmaTileN +
              subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
          if (warp_col < output_rows_this_tile) {
            wmma::fragment<
                wmma::matrix_b,
                kPlannedWmmaTileM,
                kPlannedWmmaTileN,
                kPlannedWmmaTileK,
                __nv_bfloat16,
                wmma::col_major>
                b_frag;
            wmma::load_matrix_sync(
                b_frag, &b_tile[buffer_index][warp_col][tile_k_base], kMacroTileK);
            wmma::mma_sync(c_frags[subtile], a_frag, b_frag, c_frags[subtile]);
          }
        }
      }
    }
    __syncthreads();
    buffer_index ^= 1;
  }

  if (mma_warp) {
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      const int warp_col = warp_id * kPlannedWmmaTileN +
          subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
      if (warp_col < output_rows_this_tile) {
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          wmma::store_matrix_sync(
              &c_tile[0][warp_col], c_frags[subtile], kOutputTile, wmma::mem_row_major);
        } else {
          wmma::store_matrix_sync(
              &c_tile[0][warp_col],
              c_frags[subtile],
              kOutputTile,
              wmma::mem_row_major);
        }
      }
    }
  }
  __syncthreads();
  for (int linear_index = tid;
       linear_index < (valid_rows * output_rows_this_tile);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_token = linear_index / output_rows_this_tile;
    const int tile_output_row = linear_index % output_rows_this_tile;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + output_row] =
          __float2bfloat16(c_tile[tile_token][tile_output_row]);
    } else {
      output[input_row * output_rows_per_expert + output_row] =
          c_tile[tile_token][tile_output_row];
    }
  }
}

template <typename OutputType, int kOutputTile, int kMacroTileK>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kWarpSubTiles =
      kOutputTile / (kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN);
  static_assert(kWarpSubTiles >= 1);
  __shared__ __nv_bfloat16 a_tile[2][kOutputTile][kMacroTileK];
  __shared__ __nv_bfloat16 b_tile[2][kPlannedWmmaTileM][kMacroTileK];
  __shared__ float c_tile_storage[kOutputTile * kPlannedWmmaTileM];
  auto* c_tile = reinterpret_cast<float (*)[kPlannedWmmaTileM]>(c_tile_storage);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const bool mma_warp = warp_id < kGroupedConsumerWarpsPerBlock;
  const bool producer_warp = warp_id >= kGroupedConsumerWarpsPerBlock;
  const int producer_tid = tid - (kGroupedConsumerWarpsPerBlock * 32);
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frags[kWarpSubTiles];
  if (mma_warp) {
    for (int i = 0; i < kWarpSubTiles; ++i) {
      wmma::fill_fragment(c_frags[i], 0.0f);
    }
  }

  auto stage_macro_tile = [&](int buffer_index, std::size_t k_base) {
    if (!producer_warp) {
      return;
    }
    const int producer_threads = kGroupedProducerWarpsPerBlock * 32;
    const int macro_k = static_cast<int>(
        min(static_cast<std::size_t>(kMacroTileK), weight.input_cols - k_base));
    const int macro_blocks = macro_k / static_cast<int>(fused_decode::kNvfp4BlockWidth);
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    for (int linear_block = producer_tid;
         linear_block < (output_rows_this_tile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &a_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      DecodeNvfp4WeightBlockBf16(
          weight,
          static_cast<std::size_t>(output_row_base + tile_output_row),
          block_base + static_cast<std::size_t>(tile_block),
          dst);
    }
    for (int linear_block = producer_tid + output_rows_this_tile * macro_blocks;
         linear_block < (kOutputTile * macro_blocks);
         linear_block += producer_threads) {
      const int tile_output_row = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &a_tile[buffer_index][tile_output_row]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      ZeroBf16Block16(dst);
    }
    for (int linear_block = producer_tid;
         linear_block < (kPlannedWmmaTileM * macro_blocks);
         linear_block += producer_threads) {
      const int tile_token = linear_block / macro_blocks;
      const int tile_block = linear_block % macro_blocks;
      __nv_bfloat16* dst =
          &b_tile[buffer_index][tile_token]
                 [tile_block * static_cast<int>(fused_decode::kNvfp4BlockWidth)];
      if (tile_token < valid_rows) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float row_ts =
            input_dq_scales != nullptr
                ? 1.0f
                : (input_expert_tensor_scales != nullptr
                       ? input_expert_tensor_scales[expert_index]
                       : (input_per_row_tensor_scales != nullptr
                              ? input_per_row_tensor_scales[input_row]
                              : *input_tensor_scale_data));
        DecodeGroupedPackedInputBlockBf16(
            packed_input,
            input_block_scales,
            input_dq_scales,
            row_ts,
            weight.input_cols,
            input_row,
            block_base + static_cast<std::size_t>(tile_block),
            dst);
      } else {
        ZeroBf16Block16(dst);
      }
    }
  };

  const std::size_t k_step = static_cast<std::size_t>(kMacroTileK);
  int buffer_index = 0;
  stage_macro_tile(buffer_index, 0);
  __syncthreads();
  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += k_step) {
    const int macro_k = static_cast<int>(
        min(k_step, weight.input_cols - k_base));
    const int next_buffer = buffer_index ^ 1;
    const std::size_t next_k_base = k_base + k_step;
    if (next_k_base < weight.input_cols) {
      stage_macro_tile(next_buffer, next_k_base);
    }
    if (mma_warp) {
      for (int tile_k_base = 0; tile_k_base < macro_k; tile_k_base += kPlannedWmmaTileK) {
        for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
          const int warp_row = warp_id * kPlannedWmmaTileN +
              subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
          if (warp_row < output_rows_this_tile) {
            wmma::fragment<
                wmma::matrix_a,
                kPlannedWmmaTileM,
                kPlannedWmmaTileN,
                kPlannedWmmaTileK,
                __nv_bfloat16,
                wmma::row_major>
                a_frag;
            wmma::fragment<
                wmma::matrix_b,
                kPlannedWmmaTileM,
                kPlannedWmmaTileN,
                kPlannedWmmaTileK,
                __nv_bfloat16,
                wmma::col_major>
                b_frag;
            wmma::load_matrix_sync(
                a_frag, &a_tile[buffer_index][warp_row][tile_k_base], kMacroTileK);
            wmma::load_matrix_sync(
                b_frag, &b_tile[buffer_index][0][tile_k_base], kMacroTileK);
            wmma::mma_sync(c_frags[subtile], a_frag, b_frag, c_frags[subtile]);
          }
        }
      }
    }
    __syncthreads();
    buffer_index ^= 1;
  }

  if (mma_warp) {
    for (int subtile = 0; subtile < kWarpSubTiles; ++subtile) {
      const int warp_row = warp_id * kPlannedWmmaTileN +
          subtile * kGroupedConsumerWarpsPerBlock * kPlannedWmmaTileN;
      if (warp_row < output_rows_this_tile) {
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          wmma::store_matrix_sync(
              &c_tile[warp_row][0], c_frags[subtile], kPlannedWmmaTileM, wmma::mem_row_major);
        } else {
          wmma::store_matrix_sync(
              &c_tile[warp_row][0],
              c_frags[subtile],
              kPlannedWmmaTileM,
              wmma::mem_row_major);
        }
      }
    }
  }
  __syncthreads();
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
      output[input_row * output_rows_per_expert + output_row] =
          __float2bfloat16(c_tile[tile_output_row][tile_token]);
    } else {
      output[input_row * output_rows_per_expert + output_row] =
          c_tile[tile_output_row][tile_token];
    }
  }
}

template <
    nvfp4_bridge::UnifiedRoutedFp4Profile Profile,
    typename OutputType,
    nvfp4_bridge::P5EpilogueMode P5Mode = nvfp4_bridge::P5EpilogueMode::kBf16,
    bool UseFp4DirectOutput =
        (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
         P5Mode == nvfp4_bridge::P5EpilogueMode::kFp4Direct)>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_matmul_block_scales,
    Nvfp4ScaleLayout input_scale_layout,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const routed_p5_tma::P5TmaLoadB* p5_tma_load_b_descriptors,
    const routed_p5_tma::P5TmaLoadSFB* p5_tma_load_sfb_descriptors,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output,
    std::uint8_t* fp4_packed_data,
    std::uint8_t* fp4_block_scales,
    std::uint8_t* fp4_matmul_block_scales,
    float* fp4_activation_output_scale,
    std::size_t fp4_cols,
    std::size_t fp4_padded_blocks_per_row,
    Nvfp4ScaleLayout fp4_scale_layout) {
  constexpr bool kUseSpecializedP5Fp4DirectOutput =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      P5Mode == nvfp4_bridge::P5EpilogueMode::kFp4Direct;
  using Traits = nvfp4_bridge::UnifiedRoutedFp4Traits<Profile>;
  using TiledMma = typename Traits::TiledMma;
  using CollectiveMainloop = typename Traits::CollectiveMainloop;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemCopyAtomA = typename Traits::SmemCopyAtomA;
  using SmemCopyAtomB = typename Traits::SmemCopyAtomB;
  using SmemCopyAtomSFA = typename Traits::SmemCopyAtomSFA;
  using SmemCopyAtomSFB = typename Traits::SmemCopyAtomSFB;
  constexpr int kOutputTile = Traits::kOutputTile;
  constexpr int kProfileTokenRows = Traits::kTokenRows;
  constexpr int kMacroTileK = Traits::kMacroTileK;
  constexpr int kMacroTileBytes = kMacroTileK / 2;
  constexpr int kMacroScaleBytes = kMacroTileK / fused_decode::kNvfp4BlockWidth;
  constexpr int kFp4ConsumerWarps = cute::size(TiledMma{}) / 32;
  constexpr int kP5ScaleTmaRowTile = 128;
  constexpr int kP5PipelineStages =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 ? 2 : 1;
  constexpr int kP5SfbTmaStageElems =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 ? Traits::kScaleSmemCosizeB : 1;

  static_assert(
      !kUseSpecializedP5Fp4DirectOutput ||
          (kFp4ConsumerWarps * 32 == fused_decode::kThreadsPerBlock &&
           Traits::kOutputTile == nvfp4_bridge::kTracedP5DirectStageCols &&
           Traits::kTokenRows == nvfp4_bridge::kTracedP5DirectStageRows),
      "P5 direct pack assumes a full 32x128 CTA tile");

  __shared__ alignas(1024) cute::array_aligned<
      typename Traits::SmemAllocA,
      Traits::kSwizzledAElems * kP5PipelineStages>
      smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<
      typename Traits::SmemAllocB,
      Traits::kSwizzledBElems * kP5PipelineStages>
      smem_swizzled_b_storage;
  __shared__ alignas(1024) cute::array_aligned<
      nvfp4_cute::ElementSFCompute,
      Traits::kScaleSmemCosizeA * kP5PipelineStages>
      a_scale_smem_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeB>
      b_scale_smem_storage;
  __shared__ alignas(128) cute::array_aligned<nvfp4_cute::ElementSFCompute, kP5SfbTmaStageElems>
      p5_sfb_tma_smem_storage;
  __shared__ alignas(16)
      cutlass::arch::ClusterTransactionBarrier::ValueType
          p5_tma_full_mbar_storage[kP5PipelineStages];
  __shared__ alignas(16)
      cutlass::arch::ClusterBarrier::ValueType
          p5_tma_empty_mbar_storage[kP5PipelineStages];
  __shared__ int p13_debug_capture_cta;

  auto* smem_swizzled_a = smem_swizzled_a_storage.data();
  auto* smem_swizzled_b = smem_swizzled_b_storage.data();
  auto* a_scale_smem = a_scale_smem_storage.data();
  auto* b_scale_smem = b_scale_smem_storage.data();
  auto* p5_sfb_tma_smem = p5_sfb_tma_smem_storage.data();
  auto* p5_tma_full_mbar =
      cute::recast_ptr<cutlass::arch::ClusterTransactionBarrier>(&p5_tma_full_mbar_storage[0]);
  auto* p5_tma_empty_mbar =
      cute::recast_ptr<cutlass::arch::ClusterBarrier>(&p5_tma_empty_mbar_storage[0]);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_matmul_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      (!UseFp4DirectOutput && output == nullptr) ||
      (UseFp4DirectOutput &&
       (fp4_packed_data == nullptr ||
        fp4_block_scales == nullptr ||
        fp4_matmul_block_scales == nullptr ||
        fp4_activation_output_scale == nullptr ||
        fp4_cols == 0 ||
        fp4_padded_blocks_per_row == 0))) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const bool is_fp4_consumer_thread = warp_id < kFp4ConsumerWarps;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }
  if (tid == 0) {
    int capture = 0;
    if (g_enable_p13_debug_trace != 0 &&
        Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13 &&
        blockIdx.x == 0 &&
        (g_p13_debug_trace_target_valid_rows < 0 ||
         valid_rows == g_p13_debug_trace_target_valid_rows) &&
        atomicCAS(&g_p13_debug_trace_claimed, 0, 1) == 0) {
      capture = 1;
    }
    p13_debug_capture_cta = capture;
  }
  __syncthreads();

  const FusedNvfp4WeightView weight = weights[expert_index];
  const bool use_p5_tma_a =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      weight.p5_tma_load_a != nullptr &&
      (output_row_base % 128) == 0 &&
      (weight.output_rows % 128u) == 0 &&
      (weight.input_cols % 128u) == 0;
  const auto* p5_tma_load_a_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadA*>(weight.p5_tma_load_a);
  const auto* p5_tma_load_sfa_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadSFA*>(weight.p5_tma_load_sfa);
  const bool use_p5_tma_b =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      p5_tma_load_b_descriptors != nullptr &&
      (weight.input_cols % 128u) == 0;
  const bool use_p5_tma_sfa =
      use_p5_tma_a &&
      p5_tma_load_sfa_ptr != nullptr;
  const bool use_p5_direct_gmem_sfb =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      input_matmul_block_scales != nullptr &&
      input_scale_layout == Nvfp4ScaleLayout::kSwizzled128x4;
  const bool use_p5_pipeline =
      Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5 &&
      use_p5_tma_a &&
      use_p5_tma_sfa &&
      use_p5_tma_b &&
      use_p5_direct_gmem_sfb &&
      (weight.input_cols % static_cast<std::size_t>(kMacroTileK)) == 0;
  const bool use_p5_tma_sfb =
      !use_p5_pipeline &&
      !use_p5_direct_gmem_sfb &&
      use_p5_tma_b &&
      p5_tma_load_sfb_descriptors != nullptr &&
      input_scale_layout == Nvfp4ScaleLayout::kSwizzled128x4 &&
      (row_start >= 0) &&
      ((row_start & (kP5ScaleTmaRowTile - 1)) + valid_rows <= kP5ScaleTmaRowTile);
  const int p5_sfb_row_offset =
      use_p5_tma_sfb ? (row_start & (kP5ScaleTmaRowTile - 1)) : 0;
  const bool use_p5_tma_sfb_direct = use_p5_tma_sfb && p5_sfb_row_offset == 0;
  const bool use_p5_tma_sfb_remap = use_p5_tma_sfb && !use_p5_tma_sfb_direct;
  const bool p5_sfb_within_single_chunk =
      valid_rows > 0 &&
      (p5_sfb_row_offset / 32) ==
          ((p5_sfb_row_offset + valid_rows - 1) / 32);
  const bool use_p5_tma_sfb_direct_slice =
      use_p5_tma_sfb_remap &&
      p5_sfb_row_offset < 32 &&
      (p5_sfb_row_offset + valid_rows) <= 32;
  const bool use_p5_tma_sfb_register_assembly =
      use_p5_tma_sfb_remap &&
      !use_p5_tma_sfb_direct_slice &&
      p5_sfb_within_single_chunk;
  const bool use_p5_tma_sfb_remap_fallback =
      use_p5_tma_sfb_remap &&
      !use_p5_tma_sfb_direct_slice &&
      !use_p5_tma_sfb_register_assembly;
  const int lane_predicate = cute::elect_one_sync();
  const bool is_p5_tma_thread =
      (use_p5_tma_a || use_p5_tma_b) &&
      warp_id == kFp4ConsumerWarps &&
      lane_predicate != 0;
  if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
    if (is_p5_tma_thread) {
      if (use_p5_tma_a) {
        cute::prefetch_tma_descriptor(p5_tma_load_a_ptr->get_tma_descriptor());
      }
      if (use_p5_tma_sfa) {
        cute::prefetch_tma_descriptor(p5_tma_load_sfa_ptr->get_tma_descriptor());
      }
      if (use_p5_tma_b) {
        cute::prefetch_tma_descriptor(
            p5_tma_load_b_descriptors[cta_index].get_tma_descriptor());
      }
      if (use_p5_tma_sfb) {
        cute::prefetch_tma_descriptor(
            p5_tma_load_sfb_descriptors[cta_index].get_tma_descriptor());
      }
    }
  }
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int macro_k_tile_count = static_cast<int>(weight.input_cols / static_cast<std::size_t>(kMacroTileK));
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * weight_tensor_scale;

  nvfp4_bridge::CRegister accum_storage[Traits::kAccumProfileCosize];
  if (warp_id < kFp4ConsumerWarps) {
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        typename Traits::AccumLayout{});
    cute::clear(accum_tensor);
  }

  if (use_p5_pipeline) {
    if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
    constexpr uint32_t kP5PipelineStageBytes = static_cast<uint32_t>(
        cutlass::bits_to_bytes(
            cute::size(cute::take<0, 2>(SmemLayoutA{})) *
                cute::sizeof_bits_v<routed_p5_tma::ElementAB> +
            cute::size(cute::take<0, 2>(SmemLayoutB{})) *
                cute::sizeof_bits_v<routed_p5_tma::ElementAB> +
            cute::cosize(cute::take<0, 2>(SmemLayoutSFA{})) *
                cute::sizeof_bits_v<routed_p5_tma::ElementSF>));

    auto advance_pipeline_state = [](int& stage, uint32_t& phase) {
      stage = (stage + 1) & 1;
      if (stage == 0) {
        phase ^= 1u;
      }
    };

    if (tid == 0) {
      for (int stage = 0; stage < kP5PipelineStages; ++stage) {
        p5_tma_full_mbar[stage].init(1);
        p5_tma_empty_mbar[stage].init(kFp4ConsumerWarps);
      }
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (is_p5_tma_thread) {
      using X = cute::Underscore;
      using ProducerBarrierType = typename cutlass::arch::ClusterTransactionBarrier::ValueType;
      const int output_tile_coord = output_row_base / 128;

      const auto& tma_load_a = *p5_tma_load_a_ptr;
      auto mA_mkl = tma_load_a.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(weight.output_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gA_mkl = cute::local_tile(
          mA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_a = tma_load_a.get_slice(0);
      auto gA = gA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgA = block_tma_a.partition_S(gA);
      auto sA_stage0 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 0 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto sA_stage1 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 1 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto tAsA_stage0 =
          block_tma_a.partition_D(cute::as_position_independent_swizzle_tensor(sA_stage0));
      auto tAsA_stage1 =
          block_tma_a.partition_D(cute::as_position_independent_swizzle_tensor(sA_stage1));

      const auto& tma_load_sfa = *p5_tma_load_sfa_ptr;
      auto mSFA_mkl = tma_load_sfa.get_tma_tensor(cute::shape(
          routed_p5_tma::MakeP5ScaleLayoutSFA(
              static_cast<int32_t>(weight.output_rows),
              static_cast<int32_t>(weight.input_cols))));
      auto gSFA_mkl = cute::local_tile(
          mSFA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_sfa = tma_load_sfa.get_slice(0);
      auto gSFA = gSFA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgSFA = block_tma_sfa.partition_S(gSFA);
      auto sSFA_stage0 = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 0 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto sSFA_stage1 = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 1 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto tAsSFA_stage0 =
          block_tma_sfa.partition_D(cute::as_position_independent_swizzle_tensor(sSFA_stage0));
      auto tAsSFA_stage1 =
          block_tma_sfa.partition_D(cute::as_position_independent_swizzle_tensor(sSFA_stage1));

      const auto& tma_load_b = p5_tma_load_b_descriptors[cta_index];
      auto mB_nkl = tma_load_b.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(valid_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gB_nkl = cute::local_tile(
          mB_nkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<X, cute::_1, cute::_1>{});
      auto block_tma_b = tma_load_b.get_slice(0);
      auto gB = gB_nkl(cute::_, cute::_, 0, cute::_, 0);
      auto tBgB = block_tma_b.partition_S(gB);
      auto sB_stage0 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 0 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto sB_stage1 = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 1 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto tBsB_stage0 =
          block_tma_b.partition_D(cute::as_position_independent_swizzle_tensor(sB_stage0));
      auto tBsB_stage1 =
          block_tma_b.partition_D(cute::as_position_independent_swizzle_tensor(sB_stage1));

      int producer_stage = 0;
      uint32_t producer_phase = 1;
      for (int tile = 0; tile < macro_k_tile_count; ++tile) {
        p5_tma_empty_mbar[producer_stage].wait(producer_phase);
        p5_tma_full_mbar[producer_stage].arrive_and_expect_tx(kP5PipelineStageBytes);
        auto& barrier = p5_tma_full_mbar[producer_stage];
        if (producer_stage == 0) {
          auto tma_copy_a =
              tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_sfa =
              tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_b =
              tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_a,
              tAgA(cute::_, cute::_, cute::_, tile),
              tAsA_stage0(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_sfa,
              tAgSFA(cute::_, cute::_, cute::_, tile),
              tAsSFA_stage0(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_b,
              tBgB(cute::_, cute::_, cute::_, tile),
              tBsB_stage0(cute::_, cute::_, cute::_, cute::Int<0>{}));
        } else {
          auto tma_copy_a =
              tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_sfa =
              tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          auto tma_copy_b =
              tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_a,
              tAgA(cute::_, cute::_, cute::_, tile),
              tAsA_stage1(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_sfa,
              tAgSFA(cute::_, cute::_, cute::_, tile),
              tAsSFA_stage1(cute::_, cute::_, cute::_, cute::Int<0>{}));
          cute::copy(
              tma_copy_b,
              tBgB(cute::_, cute::_, cute::_, tile),
              tBsB_stage1(cute::_, cute::_, cute::_, cute::Int<0>{}));
        }
        advance_pipeline_state(producer_stage, producer_phase);
      }
      const int outstanding_stages =
          macro_k_tile_count < kP5PipelineStages ? macro_k_tile_count : kP5PipelineStages;
      for (int drain = 0; drain < outstanding_stages; ++drain) {
        p5_tma_empty_mbar[producer_stage].wait(producer_phase);
        advance_pipeline_state(producer_stage, producer_phase);
      }
    }

    if (is_fp4_consumer_thread) {
      auto tiled_mma = TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_stage0_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 0 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto sA_stage1_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_a + 1 * Traits::kSwizzledAElems),
          SmemLayoutA{});
      auto sB_stage0_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 0 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto sB_stage1_ = cute::make_tensor(
          cute::make_smem_ptr(smem_swizzled_b + 1 * Traits::kSwizzledBElems),
          SmemLayoutB{});
      auto sSFA_stage0_ = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 0 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto sSFA_stage1_ = cute::make_tensor(
          cute::make_smem_ptr(a_scale_smem + 1 * Traits::kScaleSmemCosizeA),
          SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});

      auto sA_stage0 = cute::as_position_independent_swizzle_tensor(sA_stage0_);
      auto sA_stage1 = cute::as_position_independent_swizzle_tensor(sA_stage1_);
      auto sB_stage0 = cute::as_position_independent_swizzle_tensor(sB_stage0_);
      auto sB_stage1 = cute::as_position_independent_swizzle_tensor(sB_stage1_);
      auto sScaleA_stage0 = cute::as_position_independent_swizzle_tensor(sSFA_stage0_);
      auto sScaleA_stage1 = cute::as_position_independent_swizzle_tensor(sSFA_stage1_);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

      auto tCrA = thread_mma.partition_fragment_A(sA_stage0(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB_stage0(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
          sSFA_stage0_(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA_stage0 = s2r_thr_A.partition_S(sA_stage0);
      auto tCsA_stage1 = s2r_thr_A.partition_S(sA_stage1);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB_stage0 = s2r_thr_B.partition_S(sB_stage0);
      auto tCsB_stage1 = s2r_thr_B.partition_S(sB_stage1);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          SmemCopyAtomSFA{},
          Traits::GetLayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA_stage0 = s2r_thr_SFA.partition_S(sScaleA_stage0);
      auto tCsSFA_stage1 = s2r_thr_SFA.partition_S(sScaleA_stage1);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          SmemCopyAtomSFB{},
          Traits::GetLayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto scale_coords = cute::make_identity_tensor(cute::shape(sScaleB));
      auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords);
      auto dense_c = cute::make_identity_tensor(
          cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
      auto part_c = thread_mma.partition_C(dense_c);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          typename Traits::AccumLayout{});
      using MMAOp = typename TiledMma::MMA_Op;
      typename TiledMma::Atom mma_atom;
      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
      constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
      const int blocks_per_k_block = kMacroScaleBytes / SfbKBlocks;

      int consumer_stage = 0;
      uint32_t consumer_phase = 0;
      for (int tile = 0; tile < macro_k_tile_count; ++tile) {
        const std::size_t block_base = static_cast<std::size_t>(tile) * kMacroScaleBytes;
        const int curr_stage = consumer_stage;
        p5_tma_full_mbar[curr_stage].wait(consumer_phase);

        auto load_sfb_kblock = [&](int k_block) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto sfb_atom = tCrSFB(cute::_, n, k_block);
            auto row_anchor = tCsSFB_coords(cute::_, n, 0, cute::Int<0>{});
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (b_base_row >= 0 && b_base_row < valid_rows &&
                local_row0 >= 0 && local_row0 < valid_rows) {
              packed_scale_word = LoadExecutionScaleWord(
                  input_matmul_block_scales,
                  static_cast<std::size_t>(row_start + local_row0),
                  block_base + static_cast<std::size_t>(k_block * blocks_per_k_block),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
            FillP15ScaleFragmentWord(sfb_atom, packed_scale_word);
          }
        };

        auto copy_kblock = [&](int k_block) {
          if (curr_stage == 0) {
            cute::copy(
                s2r_copy_A,
                tCsA_stage0(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrA_cv(cute::_, cute::_, k_block));
            cute::copy(
                s2r_copy_B,
                tCsB_stage0(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrB_cv(cute::_, cute::_, k_block));
            cute::copy(
                tCsSFA_stage0(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrSFA_cv(cute::_, cute::_, k_block));
          } else {
            cute::copy(
                s2r_copy_A,
                tCsA_stage1(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrA_cv(cute::_, cute::_, k_block));
            cute::copy(
                s2r_copy_B,
                tCsB_stage1(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrB_cv(cute::_, cute::_, k_block));
            cute::copy(
                tCsSFA_stage1(cute::_, cute::_, k_block, cute::Int<0>{}),
                tCrSFA_cv(cute::_, cute::_, k_block));
          }
          cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k_block));
          cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k_block));
          load_sfb_kblock(k_block);
        };

        copy_kblock(0);

        if (g_enable_p5_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          g_p5_scale_trace.valid = 1;
          g_p5_scale_trace.row_start = row_start;
          g_p5_scale_trace.valid_rows = valid_rows;
          g_p5_scale_trace.output_row_base = output_row_base;
          for (int m = 0; m < M_tiles; ++m) {
            auto c_atom_coords = part_c(cute::_, m, 0);
            auto coord0 = c_atom_coords(0);
            const int a_base_row = static_cast<int>(cute::get<0>(coord0));
            g_p5_scale_trace.a_base_rows[m] = a_base_row;
            g_p5_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            g_p5_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < N_tiles; ++n) {
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            g_p5_scale_trace.b_base_rows[n] = b_base_row;
            g_p5_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            g_p5_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }

        for (int k = 0; k < K_blocks; ++k) {
          const int next_k = k + 1;
          if (next_k < K_blocks) {
            copy_kblock(next_k);
          }
          for (int n = 0; n < N_tiles; ++n) {
            for (int m = 0; m < M_tiles; ++m) {
              auto a_atom = tCrA(cute::_, m, k);
              auto b_atom = tCrB(cute::_, n, k);
              auto c_atom = accum_tensor(cute::_, m, n);
              auto sfa_atom = tCrSFA(cute::_, m, k);
              auto sfb_atom = tCrSFB(cute::_, n, k);
              auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
              auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
              mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
            }
          }
        }

        __syncwarp();
        if (lane_id == 0) {
          p5_tma_empty_mbar[curr_stage].arrive();
        }
        advance_pipeline_state(consumer_stage, consumer_phase);
      }
    }
    }
  } else {
    for (std::size_t macro_k_base = 0; macro_k_base < weight.input_cols; macro_k_base += kMacroTileK) {
    const std::size_t remaining_k = weight.input_cols - macro_k_base;
    const std::size_t macro_k =
        remaining_k < static_cast<std::size_t>(kMacroTileK)
            ? remaining_k
            : static_cast<std::size_t>(kMacroTileK);
    const std::size_t available_bytes = macro_k / 2u;
    const std::size_t available_blocks = macro_k / fused_decode::kNvfp4BlockWidth;
    const std::size_t block_base = macro_k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = macro_k_base / 2u;

    auto a_scale_tensor_raw = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
    auto b_scale_tensor_raw = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
    auto a_scale_tensor = cute::as_position_independent_swizzle_tensor(a_scale_tensor_raw);
    auto b_scale_tensor = cute::as_position_independent_swizzle_tensor(b_scale_tensor_raw);
    auto stage0_A = SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
    auto stage0_B = SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
    auto stage0_SFA = SmemLayoutSFA{}(cute::_, cute::_, cute::Int<0>{});
    auto stage0_SFB = SmemLayoutSFB{}(cute::_, cute::_, cute::Int<0>{});
    auto* sw_a = reinterpret_cast<std::uint8_t*>(smem_swizzled_a);
    auto* sw_b = reinterpret_cast<std::uint8_t*>(smem_swizzled_b);

    if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
      for (int i = tid; i < Traits::kScaleSmemCosizeA; i += blockDim.x) {
        a_scale_smem[i] = nvfp4_bridge::MakeScaleElement(0u);
      }
      for (int i = tid; i < Traits::kScaleSmemCosizeB; i += blockDim.x) {
        b_scale_smem[i] = nvfp4_bridge::MakeScaleElement(0u);
      }
      __syncthreads();
    }

    if ((use_p5_tma_a || use_p5_tma_b) && is_p5_tma_thread) {
      p5_tma_full_mbar[0].init(1);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
      if (is_p5_tma_thread) {
        using X = cute::Underscore;
        using ProducerBarrierType = typename cutlass::arch::ClusterTransactionBarrier::ValueType;
        auto& barrier = p5_tma_full_mbar[0];
        uint32_t stage_bytes = 0u;
        const int output_tile_coord = output_row_base / 128;
        const int k_tile_coord = static_cast<int>(macro_k_base / 128u);
        if (use_p5_tma_a) {
          const auto& tma_load_a = *p5_tma_load_a_ptr;
          auto mA_mkl = tma_load_a.get_tma_tensor(cute::make_shape(
              static_cast<int32_t>(weight.output_rows),
              static_cast<int32_t>(weight.input_cols),
              cute::Int<1>{}));
          auto gA_mkl = cute::local_tile(
              mA_mkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<cute::_1, X, cute::_1>{});
          auto block_tma_a = tma_load_a.get_slice(0);
          auto gA = gA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
          auto tAgA = block_tma_a.partition_S(gA);
          auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
          auto sA = cute::as_position_independent_swizzle_tensor(sA_);
          auto tAsA = block_tma_a.partition_D(sA);
          auto tma_copy_a =
              tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_a,
              tAgA(cute::_, cute::_, cute::_, k_tile_coord),
              tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Operand tiles use compact staged layouts, so size(...) matches the
          // actual TMA transaction footprint for barrier accounting.
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::size(cute::take<0, 2>(SmemLayoutA{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementAB>));
        }
        if (use_p5_tma_sfa) {
          const auto& tma_load_sfa = *p5_tma_load_sfa_ptr;
          auto mSFA_mkl = tma_load_sfa.get_tma_tensor(cute::shape(
              routed_p5_tma::MakeP5ScaleLayoutSFA(
                  static_cast<int32_t>(weight.output_rows),
                  static_cast<int32_t>(weight.input_cols))));
          auto gSFA_mkl = cute::local_tile(
              mSFA_mkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<cute::_1, X, cute::_1>{});
          auto block_tma_sfa = tma_load_sfa.get_slice(0);
          auto gSFA = gSFA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
          auto tAgSFA = block_tma_sfa.partition_S(gSFA);
          auto sSFA_ = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
          auto sSFA = cute::as_position_independent_swizzle_tensor(sSFA_);
          auto tAsSFA = block_tma_sfa.partition_D(sSFA);
          auto tma_copy_sfa =
              tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_sfa,
              tAgSFA(cute::_, cute::_, cute::_, k_tile_coord),
              tAsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Scale layouts can have padded/non-compact strides, so barrier
          // bytes must use cosize(...) rather than size(...).
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::cosize(cute::take<0, 2>(SmemLayoutSFA{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementSF>));
        }
        if (use_p5_tma_b) {
          const auto& tma_load_b = p5_tma_load_b_descriptors[cta_index];
          auto mB_nkl = tma_load_b.get_tma_tensor(cute::make_shape(
              static_cast<int32_t>(valid_rows),
              static_cast<int32_t>(weight.input_cols),
              cute::Int<1>{}));
          auto gB_nkl = cute::local_tile(
              mB_nkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<X, cute::_1, cute::_1>{});
          auto block_tma_b = tma_load_b.get_slice(0);
          auto gB = gB_nkl(cute::_, cute::_, 0, cute::_, 0);
          auto tBgB = block_tma_b.partition_S(gB);
          auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
          auto sB = cute::as_position_independent_swizzle_tensor(sB_);
          auto tBsB = block_tma_b.partition_D(sB);
          auto tma_copy_b =
              tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_b,
              tBgB(cute::_, cute::_, cute::_, cute::Int<0>{}),
              tBsB(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Operand tiles use compact staged layouts, so size(...) matches the
          // actual TMA transaction footprint for barrier accounting.
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::size(cute::take<0, 2>(SmemLayoutB{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementAB>));
        }
        if (use_p5_tma_sfb) {
          const auto& tma_load_sfb = p5_tma_load_sfb_descriptors[cta_index];
          auto mSFB_nkl = tma_load_sfb.get_tma_tensor(cute::shape(
              routed_p5_tma::MakeP5ScaleLayoutSFB(
                  static_cast<int32_t>(kP5ScaleTmaRowTile),
                  static_cast<int32_t>(weight.input_cols))));
          auto gSFB_nkl = cute::local_tile(
              mSFB_nkl,
              routed_p5_tma::MmaTileShape{},
              cute::make_coord(cute::_, cute::_, cute::_),
              cute::Step<X, cute::_1, cute::_1>{});
          auto block_tma_sfb = tma_load_sfb.get_slice(0);
          auto gSFB = gSFB_nkl(cute::_, cute::_, 0, cute::_, 0);
          auto tBgSFB = block_tma_sfb.partition_S(gSFB);
          auto sSFB_ = cute::make_tensor(
              cute::make_smem_ptr(
                  (use_p5_tma_sfb_direct || use_p5_tma_sfb_direct_slice)
                      ? b_scale_smem
                      : p5_sfb_tma_smem),
              SmemLayoutSFB{});
          auto sSFB = cute::as_position_independent_swizzle_tensor(sSFB_);
          auto tBsSFB = block_tma_sfb.partition_D(sSFB);
          auto tma_copy_sfb =
              tma_load_sfb.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
          cute::copy(
              tma_copy_sfb,
              tBgSFB(cute::_, cute::_, cute::_, cute::Int<0>{}),
              tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
          // Scale layouts can have padded/non-compact strides, so barrier
          // bytes must use cosize(...) rather than size(...).
          stage_bytes += static_cast<uint32_t>(cutlass::bits_to_bytes(
              cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})) *
              cute::sizeof_bits_v<routed_p5_tma::ElementSF>));
        }
        if (stage_bytes != 0u) {
          p5_tma_full_mbar[0].arrive_and_expect_tx(stage_bytes);
        }
      }
    }

    if (!use_p5_tma_a || !use_p5_tma_sfa) {
      for (int row = tid; row < kOutputTile; row += blockDim.x) {
        const bool row_valid = row < output_rows_this_tile;
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        if (!use_p5_tma_a) {
#pragma unroll
          for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
            std::uint8_t value = 0u;
            if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
              value = weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)];
            }
            auto elem_offset = stage0_A(row, byte_index * 2);
            sw_a[static_cast<int>(elem_offset) / 2] = value;
          }
        }
        if (!use_p5_tma_sfa && row_valid) {
          std::uint8_t scale_bytes[kMacroScaleBytes] = {};
#pragma unroll
          for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
            if (static_cast<std::size_t>(scale_index) < available_blocks) {
              scale_bytes[scale_index] = LoadExecutionScaleByte(
                  weight.matmul_block_scales_data,
                  source_row,
                  block_base + static_cast<std::size_t>(scale_index),
                  padded_blocks_per_row,
                  Nvfp4ScaleLayout::kSwizzled128x4);
            }
          }
          if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              g_p13_scale_trace.a_loaded_words[row] = PackScaleWord4(
                  scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]);
            }
            for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(a_scale_tensor));
                 ++scale_col) {
              const std::uint8_t value =
                  scale_col < 4 ? scale_bytes[scale_col] : std::uint8_t{0};
              StoreScaleTensorByte(a_scale_tensor, row, scale_col, value);
            }
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              for (int col = 0; col < 8; ++col) {
                g_p13_scale_trace.a_post_store_raw[row * 8 + col] = static_cast<std::uint8_t>(
                    a_scale_tensor(row, col, cute::Int<0>{}).raw());
              }
            }
          } else {
            nvfp4_bridge::StoreTracedScaleBytes(a_scale_tensor, scale_bytes, row);
          }
        } else if (!use_p5_tma_sfa) {
          nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
        }
      }
    }

    if (!use_p5_tma_b || (!use_p5_tma_sfb && !use_p5_direct_gmem_sfb)) {
      for (int row = tid; row < kProfileTokenRows; row += blockDim.x) {
        const bool row_valid = row < valid_rows;
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        if (!use_p5_tma_b) {
#pragma unroll
          for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
            std::uint8_t value = 0u;
            if (row_valid && static_cast<std::size_t>(byte_index) < available_bytes) {
              value = packed_input[src_offset + static_cast<std::size_t>(byte_index)];
            }
            auto elem_offset = stage0_B(row, byte_index * 2);
            sw_b[static_cast<int>(elem_offset) / 2] = value;
          }
        }
        if (!use_p5_tma_sfb && !use_p5_direct_gmem_sfb && row_valid) {
          std::uint8_t scale_bytes[kMacroScaleBytes] = {};
#pragma unroll
          for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
            if (static_cast<std::size_t>(scale_index) < available_blocks) {
              scale_bytes[scale_index] = LoadExecutionScaleByte(
                  input_matmul_block_scales,
                  source_row,
                  block_base + static_cast<std::size_t>(scale_index),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
          }
          if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              g_p13_scale_trace.b_loaded_words[row] = PackScaleWord4(
                  scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]);
            }
            for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(b_scale_tensor));
                 ++scale_col) {
              const std::uint8_t value =
                  scale_col < 4 ? scale_bytes[scale_col] : std::uint8_t{0};
              StoreScaleTensorByte(b_scale_tensor, row, scale_col, value);
            }
            if (g_enable_p13_scale_trace != 0 &&
                cta_index == 0 &&
                blockIdx.x == 0 &&
                block_base == 0 &&
                row < 2) {
              for (int col = 0; col < 8; ++col) {
                g_p13_scale_trace.b_post_store_raw[row * 8 + col] = static_cast<std::uint8_t>(
                    b_scale_tensor(row, col, cute::Int<0>{}).raw());
              }
            }
          } else {
            nvfp4_bridge::StoreTracedScaleBytes(b_scale_tensor, scale_bytes, row);
          }
        } else if (!use_p5_tma_sfb &&
                   !use_p5_direct_gmem_sfb) {
          nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
        }
      }
    }
    const bool need_p5_tma_wait =
        (use_p5_tma_a || use_p5_tma_b) &&
        (is_fp4_consumer_thread || use_p5_tma_sfb_remap_fallback);
    if (need_p5_tma_wait) {
      p5_tma_full_mbar[0].wait(0);
    }
    if (use_p5_tma_sfb_remap_fallback) {
      auto p5_sSFB_tma = cute::make_tensor(
          cute::make_smem_ptr(p5_sfb_tma_smem), SmemLayoutSFB{});
      auto p5_sSFB_tma_logical = cute::as_position_independent_swizzle_tensor(p5_sSFB_tma);
      auto b_scale_logical = cute::as_position_independent_swizzle_tensor(b_scale_tensor_raw);
      constexpr int kLogicalCols = cute::size<1>(SmemLayoutSFB{});
      for (int row = tid; row < kProfileTokenRows; row += blockDim.x) {
        if (row < valid_rows) {
#pragma unroll
          for (int col = 0; col < kLogicalCols; ++col) {
            b_scale_logical(row, col, cute::Int<0>{}) =
                p5_sSFB_tma_logical(p5_sfb_row_offset + row, col, cute::Int<0>{});
          }
        } else {
#pragma unroll
          for (int col = 0; col < kLogicalCols; ++col) {
            b_scale_logical(row, col, cute::Int<0>{}).storage = 0u;
          }
        }
      }
    }
    const bool need_stage_thread_sync =
        !use_p5_tma_a || !use_p5_tma_sfa || !use_p5_tma_b ||
        (!use_p5_tma_sfb && !use_p5_direct_gmem_sfb) ||
        use_p5_tma_sfb_remap_fallback;
    if (need_stage_thread_sync) {
      __syncthreads();
    }

    if (is_fp4_consumer_thread) {
      auto tiled_mma = TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
      auto sSFA = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
      auto p5_sSFB_tma =
          cute::make_tensor(cute::make_smem_ptr(p5_sfb_tma_smem), SmemLayoutSFB{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);
      auto p5_sScaleB_tma = cute::as_position_independent_swizzle_tensor(p5_sSFB_tma);

      auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
          sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA = s2r_thr_A.partition_S(sA);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB = s2r_thr_B.partition_S(sB);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          SmemCopyAtomSFA{},
          Traits::GetLayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          SmemCopyAtomSFB{},
          Traits::GetLayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          typename Traits::AccumLayout{});

      cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
      cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        auto dense_c = cute::make_identity_tensor(
            cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
        auto part_c = thread_mma.partition_C(dense_c);
        constexpr int SfaMTiles = cute::size<1>(decltype(tCrSFA){});
        constexpr int SfaKBlocks = cute::size<2>(decltype(tCrSFA){});
        constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
        constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
        const int sfa_blocks_per_k_block = static_cast<int>(available_blocks / SfaKBlocks);
        const int sfb_blocks_per_k_block = static_cast<int>(available_blocks / SfbKBlocks);
        auto scale_coords_sfa = cute::make_identity_tensor(cute::shape(sScaleA));
        auto tCsSFA_coords = s2r_thr_SFA.partition_S(scale_coords_sfa);
        auto scale_coords_sfb = cute::make_identity_tensor(cute::shape(sScaleB));
        auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords_sfb);
        for (int k = 0; k < SfaKBlocks; ++k) {
          for (int m = 0; m < SfaMTiles; ++m) {
            auto row_anchor = tCsSFA_coords(cute::_, m, 0, cute::Int<0>{});
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (local_row0 >= 0 && local_row0 < output_rows_this_tile) {
              packed_scale_word = LoadExecutionScaleWord(
                  weight.matmul_block_scales_data,
                  static_cast<std::size_t>(output_row_base + local_row0),
                  block_base + static_cast<std::size_t>(k * sfa_blocks_per_k_block),
                  padded_blocks_per_row,
                  Nvfp4ScaleLayout::kSwizzled128x4);
            }
            FillP15ScaleFragmentWord(tCrSFA(cute::_, m, k), packed_scale_word);
          }
        }
        for (int k = 0; k < SfbKBlocks; ++k) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto row_anchor = tCsSFB_coords(cute::_, n, 0, cute::Int<0>{});
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (local_row0 >= 0 && local_row0 < valid_rows) {
              packed_scale_word = LoadExecutionScaleWord(
                  input_matmul_block_scales,
                  static_cast<std::size_t>(row_start + local_row0),
                  block_base + static_cast<std::size_t>(k * sfb_blocks_per_k_block),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
            FillP15ScaleFragmentWord(tCrSFB(cute::_, n, k), packed_scale_word);
          }
        }
      } else {
        cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
      }
      if (use_p5_direct_gmem_sfb) {
        auto scale_coords = cute::make_identity_tensor(cute::shape(sScaleB));
        auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords);
        auto dense_c = cute::make_identity_tensor(
            cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
        auto part_c = thread_mma.partition_C(dense_c);
        constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
        constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
        const int blocks_per_k_block = static_cast<int>(available_blocks / SfbKBlocks);
        for (int k = 0; k < SfbKBlocks; ++k) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto sfb_atom = tCrSFB(cute::_, n, k);
            auto row_anchor = tCsSFB_coords(cute::_, n, 0, cute::Int<0>{});
            const int local_row0 = nvfp4_bridge::CoordGet0(row_anchor(0));
            std::uint32_t packed_scale_word = 0u;
            if (local_row0 >= 0 && local_row0 < valid_rows) {
              packed_scale_word = LoadExecutionScaleWord(
                  input_matmul_block_scales,
                  static_cast<std::size_t>(row_start + local_row0),
                  block_base + static_cast<std::size_t>(k * blocks_per_k_block),
                  padded_blocks_per_row,
                  input_scale_layout);
            }
            FillP15ScaleFragmentWord(sfb_atom, packed_scale_word);
          }
        }
      } else if (use_p5_tma_sfb_direct_slice) {
        auto p5_sScaleB_src = cute::domain_offset(
            cute::make_coord(p5_sfb_row_offset, 0, 0),
            p5_sScaleB_tma);
        auto tCsSFB = s2r_thr_SFB.partition_S(p5_sScaleB_src);
        cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);
      } else if (use_p5_tma_sfb_register_assembly) {
        auto p5_sScaleB_src = cute::domain_offset(
            cute::make_coord(p5_sfb_row_offset, 0, 0),
            p5_sScaleB_tma);
        auto scale_coords = cute::make_identity_tensor(cute::shape(sScaleB));
        auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords);
        auto dense_c = cute::make_identity_tensor(
            cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
        auto part_c = thread_mma.partition_C(dense_c);
        constexpr int SfbNTiles = cute::size<1>(decltype(tCrSFB){});
        constexpr int SfbKBlocks = cute::size<2>(decltype(tCrSFB){});
        for (int k = 0; k < SfbKBlocks; ++k) {
          for (int n = 0; n < SfbNTiles; ++n) {
            auto c_atom_coords = part_c(cute::_, k, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            auto sfb_atom = tCrSFB(cute::_, n, k);
            if (b_base_row >= 0 && b_base_row < valid_rows) {
              auto sfb_atom_coords =
                  tCsSFB_coords(cute::_, n, k, cute::Int<0>{});
              for (int elem = 0; elem < static_cast<int>(cute::size(sfb_atom)); ++elem) {
                sfb_atom(elem) = p5_sScaleB_src(sfb_atom_coords(elem));
              }
            } else {
              for (int elem = 0; elem < static_cast<int>(cute::size(sfb_atom)); ++elem) {
                sfb_atom(elem) = cute::float_ue4m3_t::bitcast(std::uint8_t{0});
              }
            }
          }
        }
      } else if constexpr (Profile != nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
        cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);
      }

      using MMAOp = typename TiledMma::MMA_Op;
      for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
        cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
        cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
      }

      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP5) {
        if (g_enable_p5_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          g_p5_scale_trace.valid = 1;
          g_p5_scale_trace.row_start = row_start;
          g_p5_scale_trace.valid_rows = valid_rows;
          g_p5_scale_trace.output_row_base = output_row_base;
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int m = 0; m < M_tiles; ++m) {
            auto c_atom_coords = part_c(cute::_, m, 0);
            auto coord0 = c_atom_coords(0);
            const int a_base_row = static_cast<int>(cute::get<0>(coord0));
            g_p5_scale_trace.a_base_rows[m] = a_base_row;
            g_p5_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            g_p5_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < N_tiles; ++n) {
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            g_p5_scale_trace.b_base_rows[n] = b_base_row;
            g_p5_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            g_p5_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
      } else if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        if (g_enable_p13_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          auto scale_coords_sfa = cute::make_identity_tensor(cute::shape(sScaleA));
          auto scale_coords_sfb = cute::make_identity_tensor(cute::shape(sScaleB));
          auto tCsSFA_coords = s2r_thr_SFA.partition_S(scale_coords_sfa);
          auto tCsSFB_coords = s2r_thr_SFB.partition_S(scale_coords_sfb);
          auto tCsSFA_coords_stage0 =
              tCsSFA_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          auto tCsSFB_coords_stage0 =
              tCsSFB_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          int a_scale_rows[nvfp4_bridge::kTracedP13ScaleFragmentCosizeA];
          int a_scale_cols[nvfp4_bridge::kTracedP13ScaleFragmentCosizeA];
          int b_scale_rows[nvfp4_bridge::kTracedP13ScaleFragmentCosizeB];
          int b_scale_cols[nvfp4_bridge::kTracedP13ScaleFragmentCosizeB];
          {
            int physical = 0;
            for (int i = 0;
                 i < cute::size<0>(tCsSFA_coords_stage0) &&
                 physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeA;
                 ++i) {
              for (int j = 0;
                   j < cute::size<1>(tCsSFA_coords_stage0) &&
                   physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeA;
                   ++j) {
                for (int k = 0;
                     k < cute::size<2>(tCsSFA_coords_stage0) &&
                     physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeA;
                     ++k) {
                  auto coord = tCsSFA_coords_stage0(cute::make_coord(i, j, k));
                  a_scale_rows[physical] = nvfp4_bridge::CoordGet0(coord);
                  a_scale_cols[physical] = nvfp4_bridge::CoordGet1(coord);
                  ++physical;
                }
              }
            }
          }
          {
            int physical = 0;
            for (int i = 0;
                 i < cute::size<0>(tCsSFB_coords_stage0) &&
                 physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeB;
                 ++i) {
              for (int j = 0;
                   j < cute::size<1>(tCsSFB_coords_stage0) &&
                   physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeB;
                   ++j) {
                for (int k = 0;
                     k < cute::size<2>(tCsSFB_coords_stage0) &&
                     physical < nvfp4_bridge::kTracedP13ScaleFragmentCosizeB;
                     ++k) {
                  auto coord = tCsSFB_coords_stage0(cute::make_coord(i, j, k));
                  b_scale_rows[physical] = nvfp4_bridge::CoordGet0(coord);
                  b_scale_cols[physical] = nvfp4_bridge::CoordGet1(coord);
                  ++physical;
                }
              }
            }
          }
          g_p13_scale_trace.valid = 1;
          g_p13_scale_trace.row_start = row_start;
          g_p13_scale_trace.valid_rows = valid_rows;
          g_p13_scale_trace.output_row_base = output_row_base;
          for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 8; ++col) {
              g_p13_scale_trace.a_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  sScaleA(row, col, cute::Int<0>{}).raw());
              g_p13_scale_trace.b_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  sScaleB(row, col, cute::Int<0>{}).raw());
            }
          }
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int m = 0; m < min(M_tiles, 2); ++m) {
            const int a_base_row = a_scale_rows[m * 4];
            g_p13_scale_trace.a_base_rows[m] = a_base_row;
            g_p13_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            std::uint32_t packed_from_smem = 0u;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = m * 4 + elem;
              const int scale_row = a_scale_rows[physical];
              const int scale_col = a_scale_cols[physical];
              const auto raw = static_cast<std::uint8_t>(
                  sScaleA(scale_row, scale_col, cute::Int<0>{}).raw());
              g_p13_scale_trace.a_scale_rows[physical] = scale_row;
              g_p13_scale_trace.a_scale_cols[physical] = scale_col;
              g_p13_scale_trace.a_scale_raw[physical] = raw;
              packed_from_smem |= static_cast<std::uint32_t>(raw) << (elem * 8);
            }
            g_p13_scale_trace.a_smem_words[m] = packed_from_smem;
            g_p13_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < min(N_tiles, 2); ++n) {
            const int b_base_row = b_scale_rows[n * 4];
            g_p13_scale_trace.b_base_rows[n] = b_base_row;
            g_p13_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            std::uint32_t packed_from_smem = 0u;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = n * 4 + elem;
              const int scale_row = b_scale_rows[physical];
              const int scale_col = b_scale_cols[physical];
              const auto raw = static_cast<std::uint8_t>(
                  sScaleB(scale_row, scale_col, cute::Int<0>{}).raw());
              g_p13_scale_trace.b_scale_rows[physical] = scale_row;
              g_p13_scale_trace.b_scale_cols[physical] = scale_col;
              g_p13_scale_trace.b_scale_raw[physical] = raw;
              packed_from_smem |= static_cast<std::uint32_t>(raw) << (elem * 8);
            }
            g_p13_scale_trace.b_smem_words[n] = packed_from_smem;
            g_p13_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
        if (p13_debug_capture_cta != 0 &&
            block_base == 0 &&
            ((g_p13_debug_trace_target_thread_id < 0 && thread_id == 0) ||
             thread_id == g_p13_debug_trace_target_thread_id)) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          auto a_coords = cute::make_identity_tensor(cute::shape(sA));
          auto tCsA_coords = s2r_thr_A.partition_S(a_coords);
          auto tCsA_coords_stage0 =
              tCsA_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          auto b_coords = cute::make_identity_tensor(cute::shape(sB));
          auto tCsB_coords = s2r_thr_B.partition_S(b_coords);
          auto tCsB_coords_stage0 =
              tCsB_coords(cute::_, cute::_, cute::_, cute::Int<0>{});
          int row_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
          int col_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
          int a_copy_rows[32];
          int a_copy_cols[32];
          int b_copy_rows[16];
          int b_copy_cols[16];
          nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
              part_c,
              nvfp4_bridge::kTracedP13CCopyCoordCapacity,
              row_coords,
              col_coords);
          nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
              tCsA_coords_stage0,
              32,
              a_copy_rows,
              a_copy_cols);
          nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
              tCsB_coords_stage0,
              16,
              b_copy_rows,
              b_copy_cols);
          g_p13_debug_trace.valid = 1;
          g_p13_debug_trace.row_start = row_start;
          g_p13_debug_trace.valid_rows = valid_rows;
          g_p13_debug_trace.output_row_base = output_row_base;
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int physical = 0; physical < nvfp4_bridge::kTracedP13CCopyCoordCapacity; ++physical) {
            g_p13_debug_trace.store_rows[physical] = row_coords[physical];
            g_p13_debug_trace.store_cols[physical] = col_coords[physical];
          }
          for (int physical = 0; physical < 32; ++physical) {
            g_p13_debug_trace.a_copy_rows[physical] = a_copy_rows[physical];
            g_p13_debug_trace.a_copy_cols[physical] = a_copy_cols[physical];
            std::uint8_t raw = 0u;
            if (a_copy_rows[physical] >= 0 &&
                a_copy_rows[physical] < valid_rows &&
                a_copy_cols[physical] >= 0 &&
                a_copy_cols[physical] < static_cast<int>(weight.input_cols)) {
              raw = LoadPackedFp4Nibble(
                  packed_input +
                      (static_cast<std::size_t>(row_start + a_copy_rows[physical]) *
                       packed_row_bytes),
                  a_copy_cols[physical]);
            }
            g_p13_debug_trace.a_copy_raw[physical] = raw;
          }
          for (int physical = 0; physical < 16; ++physical) {
            g_p13_debug_trace.b_copy_rows[physical] = b_copy_rows[physical];
            g_p13_debug_trace.b_copy_cols[physical] = b_copy_cols[physical];
          }
          for (int m = 0; m < min(M_tiles, 2); ++m) {
            g_p13_debug_trace.a_scale_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < min(N_tiles, 2); ++n) {
            g_p13_debug_trace.b_scale_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
      } else if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP15) {
        if (g_enable_p15_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            thread_id == 0) {
          auto dense_c = cute::make_identity_tensor(
              cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
          auto part_c = thread_mma.partition_C(dense_c);
          g_p15_scale_trace.valid = 1;
          g_p15_scale_trace.row_start = row_start;
          g_p15_scale_trace.valid_rows = valid_rows;
          g_p15_scale_trace.output_row_base = output_row_base;
          constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
          constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
          for (int m = 0; m < M_tiles; ++m) {
            auto c_atom_coords = part_c(cute::_, m, 0);
            auto coord0 = c_atom_coords(0);
            const int a_base_row = static_cast<int>(cute::get<0>(coord0));
            g_p15_scale_trace.a_base_rows[m] = a_base_row;
            g_p15_scale_trace.a_source_words[m] = LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                static_cast<std::size_t>(output_row_base + a_base_row),
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
            g_p15_scale_trace.a_fragment_words[m] =
                PackP15ScaleFragmentWord(tCrSFA(cute::_, m, 0));
          }
          for (int n = 0; n < N_tiles; ++n) {
            auto c_atom_coords = part_c(cute::_, 0, n);
            auto coord0 = c_atom_coords(0);
            const int b_base_row = static_cast<int>(cute::get<1>(coord0));
            g_p15_scale_trace.b_base_rows[n] = b_base_row;
            g_p15_scale_trace.b_source_words[n] = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start + b_base_row),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
            g_p15_scale_trace.b_fragment_words[n] =
                PackP15ScaleFragmentWord(tCrSFB(cute::_, n, 0));
          }
        }
      }

      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      typename TiledMma::Atom mma_atom;
      for (int k = 0; k < K_blocks; ++k) {
        for (int n = 0; n < N_tiles; ++n) {
          for (int m = 0; m < M_tiles; ++m) {
            auto a_atom = tCrA(cute::_, m, k);
            auto b_atom = tCrB(cute::_, n, k);
            auto c_atom = accum_tensor(cute::_, m, n);
            auto sfa_atom = tCrSFA(cute::_, m, k);
            auto sfb_atom = tCrSFB(cute::_, n, k);
            auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
            auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
            mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
          }
        }
      }
      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        if (p13_debug_capture_cta != 0 &&
            block_base == 0 &&
            ((g_p13_debug_trace_target_thread_id < 0 && thread_id == 0) ||
             thread_id == g_p13_debug_trace_target_thread_id)) {
          for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 2; ++n) {
              for (int reg = 0; reg < 4; ++reg) {
                g_p13_debug_trace.block0_accum_regs[m][n][reg] =
                    accum_tensor(reg, m, n);
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
  }



  if (warp_id < kFp4ConsumerWarps) {
      if constexpr (Profile == nvfp4_bridge::UnifiedRoutedFp4Profile::kP13) {
        if (p13_debug_capture_cta != 0 &&
            ((g_p13_debug_trace_target_thread_id < 0 && warp_id == 0 && lane_id == 0) ||
             tid == g_p13_debug_trace_target_thread_id)) {
          auto accum_tensor = cute::make_tensor(
            reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
            typename Traits::AccumLayout{});
        for (int m = 0; m < 2; ++m) {
          for (int n = 0; n < 2; ++n) {
            for (int reg = 0; reg < 4; ++reg) {
              g_p13_debug_trace.accum_regs[m][n][reg] =
                  accum_tensor(reg, m, n);
            }
            }
          }
        }
      }
      nvfp4_bridge::StoreUnifiedRoutedFp4Output<Profile, P5Mode>(
          output_alpha,
          &accum_storage[0],
          warp_id * 32 + lane_id,
          output_row_base,
          valid_rows,
          output_rows_this_tile,
          static_cast<std::size_t>(row_start),
          output_rows_per_expert,
          output,
          input_per_row_tensor_scales,
          weight_tensor_scale,
          fp4_packed_data,
          fp4_block_scales,
          fp4_matmul_block_scales,
          fp4_activation_output_scale,
          fp4_cols,
          fp4_padded_blocks_per_row,
          fp4_scale_layout);
  }
}

template <typename OutputType, int kOutputTile>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmem(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const std::uint8_t* input_matmul_block_scales,
    Nvfp4ScaleLayout input_scale_layout,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kFp4ConsumerWarps = cute::size(nvfp4_bridge::TracedP13TiledMma{}) / 32;
  __shared__ std::uint8_t a_packed[kPlannedWmmaTileM][64 / 2];
  __shared__ std::uint8_t a_scale_smem[nvfp4_bridge::kTracedP13ScaleSmemCosizeA];
  __shared__ std::uint8_t b_packed[kOutputTile][64 / 2];
  __shared__ std::uint8_t b_scale_smem[nvfp4_bridge::kTracedP13ScaleSmemCosizeB];
  const auto a_tile_view =
      nvfp4_bridge::MakePackedTile64<kPlannedWmmaTileM>(&a_packed[0][0], nullptr);
  const auto b_tile_view =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&b_packed[0][0], nullptr);
  auto a_scale_tensor_raw =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[0]), nvfp4_bridge::TracedP13SmemLayoutSFA{});
  auto b_scale_tensor_raw =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[0]), nvfp4_bridge::TracedP13SmemLayoutSFB{});
  auto a_scale_tensor = cute::as_position_independent_swizzle_tensor(a_scale_tensor_raw);
  auto b_scale_tensor = cute::as_position_independent_swizzle_tensor(b_scale_tensor_raw);

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * (*weight.tensor_scale_data);
  const bool p13_scale_map_probe = (g_enable_p13_scale_map_probe != 0);

  nvfp4_bridge::CFragment64 accum[nvfp4_bridge::kTracedP13MFragments][nvfp4_bridge::kTracedP13NFragments];
  if (warp_id < kFp4ConsumerWarps) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
        nvfp4_bridge::Clear(accum[m_fragment][n_fragment]);
      }
    }
  }

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += 64u) {
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = k_base / 2u;

    for (int row = tid; row < kPlannedWmmaTileM; row += blockDim.x) {
      if (row < valid_rows) {
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = a_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = packed_input[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            a_scale_tensor,
            PackScaleWord4(
                input_block_scales[scale_offset + 0u],
                input_block_scales[scale_offset + 1u],
                input_block_scales[scale_offset + 2u],
                input_block_scales[scale_offset + 3u]),
            row);
      } else {
        nvfp4_bridge::ZeroRow(a_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
      }
    }
    for (int row = tid; row < kOutputTile; row += blockDim.x) {
      if (row < output_rows_this_tile) {
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = b_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)];
        }
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            b_scale_tensor,
            LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                source_row,
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4),
            row);
      } else {
        nvfp4_bridge::ZeroRow(b_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
      }
    }
    if (p13_scale_map_probe) {
      for (int logical_row = tid;
           logical_row < static_cast<int>(cute::size<0>(a_scale_tensor));
           logical_row += blockDim.x) {
        const std::uint8_t value = static_cast<std::uint8_t>((logical_row + 1) & 0xff);
        for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(a_scale_tensor)); ++scale_col) {
          StoreScaleTensorByte(a_scale_tensor, logical_row, scale_col, value);
        }
      }
      for (int logical_row = tid;
           logical_row < static_cast<int>(cute::size<0>(b_scale_tensor));
           logical_row += blockDim.x) {
        const std::uint8_t value = static_cast<std::uint8_t>((logical_row + 1) & 0xff);
        for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(b_scale_tensor)); ++scale_col) {
          StoreScaleTensorByte(b_scale_tensor, logical_row, scale_col, value);
        }
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      nvfp4_bridge::AFragment64 a_fragments[nvfp4_bridge::kTracedP13MFragments];
      nvfp4_bridge::BFragment64 b_fragments[nvfp4_bridge::kTracedP13NFragments];
      if (p13_scale_map_probe) {
        nvfp4_bridge::LoadTracedP13AFragmentsRowMajor16x64<nvfp4_bridge::TracedP13TiledMma, kPlannedWmmaTileM>(
            &a_packed[0][0],
            a_scale_smem,
            warp_id * 32 + lane_id,
            0,
            a_fragments);
        nvfp4_bridge::LoadTracedP13BFragmentsColMajor64x8<nvfp4_bridge::TracedP13TiledMma, kOutputTile>(
            &b_packed[0][0],
            b_scale_smem,
            warp_id * 32 + lane_id,
            b_fragments);
        if (g_enable_p13_scale_trace != 0 &&
            cta_index == 0 &&
            blockIdx.x == 0 &&
            block_base == 0 &&
            warp_id == 0 &&
            lane_id == 0) {
          g_p13_scale_trace.valid = 1;
          g_p13_scale_trace.row_start = row_start;
          g_p13_scale_trace.valid_rows = valid_rows;
          g_p13_scale_trace.output_row_base = output_row_base;
          for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 8; ++col) {
              g_p13_scale_trace.a_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  ScaleByteValue(a_scale_tensor(row, col, cute::Int<0>{})));
              g_p13_scale_trace.b_logical_raw[row * 8 + col] = static_cast<std::uint8_t>(
                  ScaleByteValue(b_scale_tensor(row, col, cute::Int<0>{})));
              g_p13_scale_trace.a_post_store_raw[row * 8 + col] =
                  g_p13_scale_trace.a_logical_raw[row * 8 + col];
              g_p13_scale_trace.b_post_store_raw[row * 8 + col] =
                  g_p13_scale_trace.b_logical_raw[row * 8 + col];
            }
          }
          for (int m = 0; m < 2; ++m) {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(a_fragments[m].scale[0]);
            g_p13_scale_trace.a_fragment_words[m] = packed;
            g_p13_scale_trace.a_smem_words[m] = packed;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = m * 4 + elem;
              const std::uint8_t raw =
                  static_cast<std::uint8_t>((packed >> (elem * 8)) & 0xffu);
              g_p13_scale_trace.a_scale_raw[physical] = raw;
              FindScaleTensorCoordByRaw(
                  a_scale_tensor,
                  raw,
                  g_p13_scale_trace.a_scale_rows[physical],
                  g_p13_scale_trace.a_scale_cols[physical]);
            }
            g_p13_scale_trace.a_base_rows[m] = g_p13_scale_trace.a_scale_rows[m * 4];
          }
          for (int n = 0; n < 2; ++n) {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(b_fragments[n].scale[0]);
            g_p13_scale_trace.b_fragment_words[n] = packed;
            g_p13_scale_trace.b_smem_words[n] = packed;
            for (int elem = 0; elem < 4; ++elem) {
              const int physical = n * 4 + elem;
              const std::uint8_t raw =
                  static_cast<std::uint8_t>((packed >> (elem * 8)) & 0xffu);
              g_p13_scale_trace.b_scale_raw[physical] = raw;
              FindScaleTensorCoordByRaw(
                  b_scale_tensor,
                  raw,
                  g_p13_scale_trace.b_scale_rows[physical],
                  g_p13_scale_trace.b_scale_cols[physical]);
            }
            g_p13_scale_trace.b_base_rows[n] = g_p13_scale_trace.b_scale_rows[n * 4];
          }
        }
      } else {
#pragma unroll
        for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
          const int row_base = m_fragment * 64;
          a_fragments[m_fragment] =
              nvfp4_bridge::LoadFragmentA_RowMajor16x64Tiled<
                  nvfp4_bridge::TracedP13TiledMma,
                  kPlannedWmmaTileM>(&a_packed[0][0], nullptr, row_base, warp_id * 32 + lane_id);
          a_fragments[m_fragment].scale[0] = static_cast<nvfp4_bridge::SFRegister>(
              LoadExecutionScaleWord(
                  weight.matmul_block_scales_data,
                  static_cast<std::size_t>(output_row_base),
                  block_base,
                  padded_blocks_per_row,
                  Nvfp4ScaleLayout::kSwizzled128x4));
        }
#pragma unroll
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          const int n_base = n_fragment * 8;
          b_fragments[n_fragment] =
              nvfp4_bridge::LoadFragmentB_ColMajor64x8Tiled<
                  nvfp4_bridge::TracedP13TiledMma,
                  kOutputTile>(&b_packed[0][0], nullptr, warp_id * 32 + lane_id, n_base);
          std::uint32_t packed_scale_word = 0u;
          if (valid_rows > 0) {
            packed_scale_word = LoadExecutionScaleWord(
                input_matmul_block_scales,
                static_cast<std::size_t>(row_start),
                block_base,
                padded_blocks_per_row,
                input_scale_layout);
          }
          b_fragments[n_fragment].scale[0] =
              static_cast<nvfp4_bridge::SFRegister>(packed_scale_word);
        }
      }
      if (g_enable_p13_debug_trace != 0 &&
          cta_index == 0 &&
          blockIdx.x == 0 &&
          block_base == 0 &&
          warp_id == 0 &&
          lane_id == 0) {
        auto mma = nvfp4_bridge::TracedP13TiledMma{};
        auto thr_mma = mma.get_thread_slice(warp_id * 32 + lane_id);
        auto ref_c = cute::make_identity_tensor(
            cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
        auto part_c = thr_mma.partition_C(ref_c);
        int row_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
        int col_coords[nvfp4_bridge::kTracedP13CCopyCoordCapacity];
        nvfp4_bridge::FillPhysicalCoordMapCopyViewLimited(
            part_c,
            nvfp4_bridge::kTracedP13CCopyCoordCapacity,
            row_coords,
            col_coords);
        g_p13_debug_trace.valid = 1;
        g_p13_debug_trace.row_start = row_start;
        g_p13_debug_trace.valid_rows = valid_rows;
        g_p13_debug_trace.output_row_base = output_row_base;
        for (int physical = 0; physical < nvfp4_bridge::kTracedP13CCopyCoordCapacity; ++physical) {
          g_p13_debug_trace.store_rows[physical] = row_coords[physical];
          g_p13_debug_trace.store_cols[physical] = col_coords[physical];
        }
        for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
          for (int reg = 0; reg < 4; ++reg) {
            g_p13_debug_trace.a_regs[m_fragment][reg] = a_fragments[m_fragment].regs[reg];
          }
          g_p13_debug_trace.a_scale_words[m_fragment] =
              static_cast<std::uint32_t>(a_fragments[m_fragment].scale[0]);
        }
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          for (int reg = 0; reg < 2; ++reg) {
            g_p13_debug_trace.b_regs[n_fragment][reg] = b_fragments[n_fragment].regs[reg];
          }
          g_p13_debug_trace.b_scale_words[n_fragment] =
              static_cast<std::uint32_t>(b_fragments[n_fragment].scale[0]);
        }
      }
#pragma unroll
      for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
#pragma unroll
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          nvfp4_bridge::Gemm(accum[m_fragment][n_fragment], a_fragments[m_fragment], b_fragments[n_fragment]);
        }
      }
      if (g_enable_p13_debug_trace != 0 &&
          cta_index == 0 &&
          blockIdx.x == 0 &&
          block_base == 0 &&
          warp_id == 0 &&
          lane_id == 0) {
        for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
          for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
            for (int reg = 0; reg < 4; ++reg) {
              g_p13_debug_trace.block0_accum_regs[m_fragment][n_fragment][reg] =
                  accum[m_fragment][n_fragment].regs[reg];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    if (g_enable_p13_debug_trace != 0 &&
        cta_index == 0 &&
        blockIdx.x == 0 &&
        warp_id == 0 &&
        lane_id == 0) {
      for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP13MFragments; ++m_fragment) {
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP13NFragments; ++n_fragment) {
          for (int reg = 0; reg < 4; ++reg) {
            g_p13_debug_trace.accum_regs[m_fragment][n_fragment][reg] =
                accum[m_fragment][n_fragment].regs[reg];
          }
        }
      }
    }
    nvfp4_bridge::StoreTracedP13CFragmentsRowMajor<nvfp4_bridge::TracedP13TiledMma>(
        output_alpha,
        accum,
        warp_id * 32 + lane_id,
        output_row_base,
        valid_rows,
        output_rows_this_tile,
        static_cast<std::size_t>(row_start),
        output_rows_per_expert,
        output);
  }
}

template <typename OutputType>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK128P12(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kOutputTile = 128;
  constexpr int kProfileTokenRows = 128;
  constexpr int kMacroTileK = 128;
  constexpr int kMacroTileBytes = kMacroTileK / 2;
  constexpr int kMacroScaleBytes = kMacroTileK / fused_decode::kNvfp4BlockWidth;
  using P12TiledMma = nvfp4_bridge::TracedP5TiledMma;
  using P12CollectiveMainloop = nvfp4_bridge::TracedP5CollectiveMainloop;
  using P12SmemLayoutA = nvfp4_bridge::TracedP5SmemLayoutA;
  using P12SmemLayoutB = nvfp4_bridge::TracedP5SmemLayoutB;
  using P12SmemLayoutSFA = nvfp4_bridge::TracedP5SmemLayoutSFA;
  using P12SmemLayoutSFB = nvfp4_bridge::TracedP5SmemLayoutSFB;
  using P12SmemCopyAtomA = nvfp4_bridge::TracedP5SmemCopyAtomA;
  using P12SmemCopyAtomB = nvfp4_bridge::TracedP5SmemCopyAtomB;
  using P12SmemCopyAtomSFA = nvfp4_bridge::TracedP5SmemCopyAtomSFA;
  using P12SmemCopyAtomSFB = nvfp4_bridge::TracedP5SmemCopyAtomSFB;
  constexpr int kFp4ConsumerWarps = cute::size(P12TiledMma{}) / 32;
  __shared__ std::uint8_t a_packed[kOutputTile][kMacroTileBytes];
  __shared__ alignas(1024) cute::array_aligned<
      nvfp4_cute::ElementSFCompute,
      nvfp4_bridge::kTracedP5ScaleSmemCosizeA>
      a_scale_smem_storage;
  __shared__ std::uint8_t b_packed[kProfileTokenRows][kMacroTileBytes];
  __shared__ alignas(1024) cute::array_aligned<
      nvfp4_cute::ElementSFCompute,
      nvfp4_bridge::kTracedP5ScaleSmemCosizeB>
      b_scale_smem_storage;
  using P12SmemAllocA = typename P12TiledMma::ValTypeA;
  using P12SmemAllocB = typename P12TiledMma::ValTypeB;
  static constexpr int kP12SwizzledAElems = cute::size(cute::take<0, 2>(P12SmemLayoutA{}));
  static constexpr int kP12SwizzledBElems = cute::size(cute::take<0, 2>(P12SmemLayoutB{}));
  __shared__ alignas(1024) cute::array_aligned<P12SmemAllocA, kP12SwizzledAElems> smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<P12SmemAllocB, kP12SwizzledBElems> smem_swizzled_b_storage;
  auto* a_scale_smem = a_scale_smem_storage.data();
  auto* b_scale_smem = b_scale_smem_storage.data();
  auto* smem_swizzled_a = smem_swizzled_a_storage.data();
  auto* smem_swizzled_b = smem_swizzled_b_storage.data();

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * (*weight.tensor_scale_data);

  nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kTracedP5AccumProfileCosize];
  if (warp_id < kFp4ConsumerWarps) {
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        nvfp4_bridge::TracedP5AccumProfileLayout{});
    cute::clear(accum_tensor);
  }

  for (std::size_t macro_k_base = 0; macro_k_base < weight.input_cols; macro_k_base += kMacroTileK) {
    const std::size_t remaining_k = weight.input_cols - macro_k_base;
    const std::size_t macro_k =
        remaining_k < static_cast<std::size_t>(kMacroTileK)
            ? remaining_k
            : static_cast<std::size_t>(kMacroTileK);
    const std::size_t available_bytes = macro_k / 2u;
    const std::size_t available_blocks = macro_k / fused_decode::kNvfp4BlockWidth;
    const std::size_t block_base = macro_k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = macro_k_base / 2u;

    auto a_scale_tensor = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), P12SmemLayoutSFA{});
    auto b_scale_tensor = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), P12SmemLayoutSFB{});

    for (int row = tid; row < kOutputTile; row += blockDim.x) {
      if (row < output_rows_this_tile) {
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = &a_packed[row][0];
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          dst[byte_index] =
              static_cast<std::size_t>(byte_index) < available_bytes
                  ? weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)]
                  : std::uint8_t{0};
        }
        std::uint8_t scale_bytes[kMacroScaleBytes] = {};
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          if (static_cast<std::size_t>(scale_index) < available_blocks) {
            scale_bytes[scale_index] = LoadExecutionScaleByte(
                weight.matmul_block_scales_data,
                source_row,
                block_base + static_cast<std::size_t>(scale_index),
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4);
          }
        }
        nvfp4_bridge::StoreTracedScaleBytes(a_scale_tensor, scale_bytes, row);
      } else {
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          a_packed[row][byte_index] = 0u;
        }
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
      }
    }
    for (int row = tid; row < kProfileTokenRows; row += blockDim.x) {
      if (row < valid_rows) {
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = &b_packed[row][0];
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          dst[byte_index] =
              static_cast<std::size_t>(byte_index) < available_bytes
                  ? packed_input[src_offset + static_cast<std::size_t>(byte_index)]
                  : std::uint8_t{0};
        }
        std::uint8_t scale_bytes[kMacroScaleBytes] = {};
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
#pragma unroll
        for (int scale_index = 0; scale_index < kMacroScaleBytes; ++scale_index) {
          if (static_cast<std::size_t>(scale_index) < available_blocks) {
            scale_bytes[scale_index] =
                input_block_scales[scale_offset + static_cast<std::size_t>(scale_index)];
          }
        }
        nvfp4_bridge::StoreTracedScaleBytes(b_scale_tensor, scale_bytes, row);
      } else {
#pragma unroll
        for (int byte_index = 0; byte_index < kMacroTileBytes; ++byte_index) {
          b_packed[row][byte_index] = 0u;
        }
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
      }
    }
    {
      auto stage0_A = P12SmemLayoutA{}(cute::_, cute::_, cute::Int<0>{});
      auto* sw_a = reinterpret_cast<std::uint8_t*>(smem_swizzled_a);
      constexpr int a_total_bytes = kOutputTile * kMacroTileBytes;
      for (int i = tid; i < a_total_bytes; i += blockDim.x) {
        const int row = i / kMacroTileBytes;
        const int col_byte = i % kMacroTileBytes;
        auto elem_offset = stage0_A(row, col_byte * 2);
        sw_a[static_cast<int>(elem_offset) / 2] = a_packed[row][col_byte];
      }

      auto stage0_B = P12SmemLayoutB{}(cute::_, cute::_, cute::Int<0>{});
      auto* sw_b = reinterpret_cast<std::uint8_t*>(smem_swizzled_b);
      constexpr int b_total_bytes = kProfileTokenRows * kMacroTileBytes;
      for (int i = tid; i < b_total_bytes; i += blockDim.x) {
        const int row = i / kMacroTileBytes;
        const int col_byte = i % kMacroTileBytes;
        auto elem_offset = stage0_B(row, col_byte * 2);
        sw_b[static_cast<int>(elem_offset) / 2] = b_packed[row][col_byte];
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      auto tiled_mma = P12TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), P12SmemLayoutA{});
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), P12SmemLayoutB{});
      auto sSFA = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), P12SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), P12SmemLayoutSFB{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

      auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = P12CollectiveMainloop{}.partition_fragment_SFA(
          sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = P12CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(P12SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA = s2r_thr_A.partition_S(sA);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(P12SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB = s2r_thr_B.partition_S(sB);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          P12SmemCopyAtomSFA{},
          nvfp4_bridge::GetTracedP5LayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          P12SmemCopyAtomSFB{},
          nvfp4_bridge::GetTracedP5LayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
      auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          nvfp4_bridge::TracedP5AccumProfileLayout{});

      cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
      cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
      cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
      cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

      using P12MmaOp = typename P12TiledMma::MMA_Op;
      for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
        cute::fp4_shift_A(P12MmaOp{}, tCrA_cv(cute::_, cute::_, k));
        cute::fp4_shift_B(P12MmaOp{}, tCrB_cv(cute::_, cute::_, k));
      }

      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      typename P12TiledMma::Atom mma_atom;
      for (int k = 0; k < K_blocks; ++k) {
        for (int n = 0; n < N_tiles; ++n) {
          for (int m = 0; m < M_tiles; ++m) {
            auto a_atom = tCrA(cute::_, m, k);
            auto b_atom = tCrB(cute::_, n, k);
            auto c_atom = accum_tensor(cute::_, m, n);
            auto sfa_atom = tCrSFA(cute::_, m, k);
            auto sfb_atom = tCrSFB(cute::_, n, k);
            auto a_zipped = cute::make_zip_tensor(a_atom, sfa_atom);
            auto b_zipped = cute::make_zip_tensor(b_atom, sfb_atom);
            mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
          }
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    auto tiled_mma = P12TiledMma{};
    auto thread_mma = tiled_mma.get_thread_slice(warp_id * 32 + lane_id);
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        nvfp4_bridge::TracedP5AccumProfileLayout{});
    auto dense_c =
        cute::make_identity_tensor(cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kProfileTokenRows>{}));
    auto part_c = thread_mma.partition_C(dense_c);
    for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
      auto coord = part_c(i);
      const int output_col_offset = static_cast<int>(cute::get<0>(coord));
      const int token_row = static_cast<int>(cute::get<1>(coord));
      if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
        continue;
      }
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_row);
      const std::size_t output_col =
          static_cast<std::size_t>(output_row_base + output_col_offset);
      if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
        output[input_row * output_rows_per_expert + output_col] =
            __float2bfloat16(accum_tensor(i) * output_alpha);
      } else {
        output[input_row * output_rows_per_expert + output_col] = accum_tensor(i) * output_alpha;
      }
    }
  }
}

template <typename OutputType, int kOutputTile>
__global__ void Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  constexpr int kTracedP5TileN = cute::tile_size<1>(nvfp4_bridge::TracedP5TiledMma{});
  constexpr int kFp4ConsumerWarps = cute::size(nvfp4_bridge::TracedP5TiledMma{}) / 32;
  static_assert(kOutputTile == cute::tile_size<0>(nvfp4_bridge::TracedP5TiledMma{}));
  __shared__ std::uint8_t a_packed[kOutputTile][64 / 2];
  __shared__ std::uint8_t a_scale_smem[nvfp4_bridge::kTracedP5ScaleSmemCosizeA];
  __shared__ std::uint8_t b_packed[kTracedP5TileN][64 / 2];
  __shared__ std::uint8_t b_scale_smem[nvfp4_bridge::kTracedP5ScaleSmemCosizeB];
  const auto a_tile_view =
      nvfp4_bridge::MakePackedTile64<kOutputTile>(&a_packed[0][0], nullptr);
  const auto b_tile_view =
      nvfp4_bridge::MakePackedTile64<kTracedP5TileN>(&b_packed[0][0], nullptr);
  auto a_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&a_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFA{});
  auto b_scale_tensor =
      cute::make_tensor(cute::make_smem_ptr(&b_scale_smem[0]), nvfp4_bridge::TracedP5SmemLayoutSFB{});

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int expert_index = cta_batch_indices[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t packed_row_bytes = weight.input_cols / 2u;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const int output_rows_this_tile = static_cast<int>(
      min(output_rows_per_expert - static_cast<std::size_t>(output_row_base),
          static_cast<std::size_t>(kOutputTile)));
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;
  const float output_alpha = input_tensor_scale * (*weight.tensor_scale_data);

  nvfp4_bridge::CFragment64 accum[nvfp4_bridge::kTracedP5MFragments][nvfp4_bridge::kTracedP5NFragments];
  if (warp_id < kFp4ConsumerWarps) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP5MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP5NFragments; ++n_fragment) {
        nvfp4_bridge::Clear(accum[m_fragment][n_fragment]);
      }
    }
  }

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += 64u) {
    const std::size_t block_base = k_base / fused_decode::kNvfp4BlockWidth;
    const std::size_t packed_byte_offset = k_base / 2u;

    for (int row = tid; row < kOutputTile; row += blockDim.x) {
      if (row < output_rows_this_tile) {
        const std::size_t source_row = static_cast<std::size_t>(output_row_base + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = a_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = weight.packed_data[src_offset + static_cast<std::size_t>(byte_index)];
        }
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            a_scale_tensor,
            LoadExecutionScaleWord(
                weight.matmul_block_scales_data,
                source_row,
                block_base,
                padded_blocks_per_row,
                Nvfp4ScaleLayout::kSwizzled128x4),
            row);
      } else {
        nvfp4_bridge::ZeroRow(a_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(a_scale_tensor, row);
      }
    }
    for (int row = tid; row < kTracedP5TileN; row += blockDim.x) {
      if (row < valid_rows) {
        const std::size_t source_row = static_cast<std::size_t>(row_start + row);
        const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
        std::uint8_t* dst = b_tile_view.packed_rows + row * (64 / 2);
#pragma unroll
        for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
          dst[byte_index] = packed_input[src_offset + static_cast<std::size_t>(byte_index)];
        }
        const std::size_t scale_offset = source_row * blocks_per_row + block_base;
        nvfp4_bridge::StoreTracedP5ScaleWordK64(
            b_scale_tensor,
            PackScaleWord4(
                input_block_scales[scale_offset + 0u],
                input_block_scales[scale_offset + 1u],
                input_block_scales[scale_offset + 2u],
                input_block_scales[scale_offset + 3u]),
            row);
      } else {
        nvfp4_bridge::ZeroRow(b_tile_view, row);
        nvfp4_bridge::ZeroTracedP5ScaleRow(b_scale_tensor, row);
      }
    }
    __syncthreads();

    if (warp_id < kFp4ConsumerWarps) {
      nvfp4_bridge::AFragment64 a_fragments[nvfp4_bridge::kTracedP5MFragments];
      nvfp4_bridge::BFragment64 b_fragments[nvfp4_bridge::kTracedP5NFragments];
      nvfp4_bridge::LoadTracedP5AFragmentsRowMajor16x64<
          nvfp4_bridge::TracedP5TiledMma,
          kOutputTile>(&a_packed[0][0], a_scale_smem, warp_id * 32 + lane_id, a_fragments);
      nvfp4_bridge::LoadTracedP5BFragmentsColMajor64x8<
          nvfp4_bridge::TracedP5TiledMma,
          kTracedP5TileN>(&b_packed[0][0], b_scale_smem, warp_id * 32 + lane_id, b_fragments);
#pragma unroll
      for (int m_fragment = 0; m_fragment < nvfp4_bridge::kTracedP5MFragments; ++m_fragment) {
#pragma unroll
        for (int n_fragment = 0; n_fragment < nvfp4_bridge::kTracedP5NFragments; ++n_fragment) {
          nvfp4_bridge::Gemm(accum[m_fragment][n_fragment], a_fragments[m_fragment], b_fragments[n_fragment]);
        }
      }
    }
    __syncthreads();
  }

  if (warp_id < kFp4ConsumerWarps) {
    nvfp4_bridge::StoreTracedP5CFragmentsTranspose<nvfp4_bridge::TracedP5TiledMma>(
        output_alpha,
        accum,
        warp_id * 32 + lane_id,
        output_row_base,
        valid_rows,
        output_rows_this_tile,
        static_cast<std::size_t>(row_start),
        output_rows_per_expert,
        output);
  }
}

__device__ __forceinline__ float LoadContiguousInputTensorScale(
    const float* input_per_row_tensor_scales,
    const float* input_tensor_scale_data,
    std::size_t input_row) {
  if (input_per_row_tensor_scales != nullptr) {
    return input_per_row_tensor_scales[input_row];
  }
  if (input_tensor_scale_data != nullptr) {
    return *input_tensor_scale_data;
  }
  return 1.0f;
}

__device__ __forceinline__ float LoadGroupedInputTensorScale(
    const float* input_expert_tensor_scales,
    const float* input_per_row_tensor_scales,
    const float* input_tensor_scale_data,
    const float* input_dq_scales,
    int expert_index,
    std::size_t input_row) {
  if (input_dq_scales != nullptr) {
    return 1.0f;
  }
  if (input_expert_tensor_scales != nullptr) {
    return input_expert_tensor_scales[expert_index];
  }
  if (input_per_row_tensor_scales != nullptr) {
    return input_per_row_tensor_scales[input_row];
  }
  if (input_tensor_scale_data != nullptr) {
    return *input_tensor_scale_data;
  }
  return 1.0f;
}

__global__ void Nvfp4ContiguousSharedFp4P5Kernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_matmul_block_scales,
    Nvfp4ScaleLayout input_scale_layout,
    const float* input_tensor_scale_data,
    const float* input_per_row_tensor_scales,
    std::size_t input_row_count,
    const routed_p5_tma::P5TmaLoadB* p5_tma_load_b_descriptors,
    const routed_p5_tma::P5TmaLoadSFB* p5_tma_load_sfb_descriptors,
    FusedNvfp4WeightView weight,
    float* output) {
  using Traits = nvfp4_bridge::UnifiedRoutedFp4Traits<nvfp4_bridge::UnifiedRoutedFp4Profile::kP5>;
  using TiledMma = typename Traits::TiledMma;
  using CollectiveMainloop = typename Traits::CollectiveMainloop;
  using SmemLayoutA = typename Traits::SmemLayoutA;
  using SmemLayoutB = typename Traits::SmemLayoutB;
  using SmemLayoutSFA = typename Traits::SmemLayoutSFA;
  using SmemLayoutSFB = typename Traits::SmemLayoutSFB;
  using SmemCopyAtomA = typename Traits::SmemCopyAtomA;
  using SmemCopyAtomB = typename Traits::SmemCopyAtomB;
  using SmemCopyAtomSFA = typename Traits::SmemCopyAtomSFA;
  using SmemCopyAtomSFB = typename Traits::SmemCopyAtomSFB;
  constexpr int kOutputTile = 128;
  constexpr int kRowTile = 128;
  constexpr int kMacroTileK = 128;
  constexpr int kFp4ConsumerWarps = cute::size(TiledMma{}) / 32;

  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocA, Traits::kSwizzledAElems>
      smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocB, Traits::kSwizzledBElems>
      smem_swizzled_b_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeA>
      a_scale_smem_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeB>
      b_scale_smem_storage;
  __shared__ alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType p5_tma_full_mbar_storage;

  auto* smem_swizzled_a = smem_swizzled_a_storage.data();
  auto* smem_swizzled_b = smem_swizzled_b_storage.data();
  auto* a_scale_smem = a_scale_smem_storage.data();
  auto* b_scale_smem = b_scale_smem_storage.data();
  auto* p5_tma_full_mbar =
      cute::recast_ptr<cutlass::arch::ClusterTransactionBarrier>(&p5_tma_full_mbar_storage);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int cta_index = static_cast<int>(blockIdx.y);
  const int row_start = cta_index * kRowTile;
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  if (packed_input == nullptr ||
      input_matmul_block_scales == nullptr ||
      p5_tma_load_b_descriptors == nullptr ||
      p5_tma_load_sfb_descriptors == nullptr ||
      weight.p5_tma_load_a == nullptr ||
      weight.p5_tma_load_sfa == nullptr ||
      output == nullptr ||
      input_scale_layout != Nvfp4ScaleLayout::kSwizzled128x4 ||
      static_cast<std::size_t>(row_start) >= input_row_count ||
      static_cast<std::size_t>(output_row_base) >= weight.output_rows) {
    return;
  }

  const auto* p5_tma_load_a_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadA*>(weight.p5_tma_load_a);
  const auto* p5_tma_load_sfa_ptr =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadSFA*>(weight.p5_tma_load_sfa);
  const int valid_rows = static_cast<int>(
      min(input_row_count - static_cast<std::size_t>(row_start), static_cast<std::size_t>(kRowTile)));
  const int output_rows_this_tile = static_cast<int>(
      min(weight.output_rows - static_cast<std::size_t>(output_row_base), static_cast<std::size_t>(kOutputTile)));
  if (valid_rows <= 0 || output_rows_this_tile <= 0) {
    return;
  }

  const bool is_fp4_consumer_thread = warp_id < kFp4ConsumerWarps;
  const int lane_predicate = cute::elect_one_sync();
  const bool is_p5_tma_thread =
      warp_id == kFp4ConsumerWarps &&
      lane_predicate != 0;

  if (is_p5_tma_thread) {
    cute::prefetch_tma_descriptor(p5_tma_load_a_ptr->get_tma_descriptor());
    cute::prefetch_tma_descriptor(p5_tma_load_sfa_ptr->get_tma_descriptor());
    cute::prefetch_tma_descriptor(p5_tma_load_b_descriptors[cta_index].get_tma_descriptor());
    cute::prefetch_tma_descriptor(p5_tma_load_sfb_descriptors[cta_index].get_tma_descriptor());
  }

  nvfp4_bridge::CRegister accum_storage[Traits::kAccumProfileCosize];
  if (is_fp4_consumer_thread) {
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        typename Traits::AccumLayout{});
    cute::clear(accum_tensor);
  }

  const float weight_tensor_scale =
      weight.tensor_scale_data != nullptr ? *weight.tensor_scale_data : 1.0f;
  const int macro_k_tile_count = static_cast<int>(weight.input_cols / static_cast<std::size_t>(kMacroTileK));
  for (int k_tile_coord = 0; k_tile_coord < macro_k_tile_count; ++k_tile_coord) {
    if (is_p5_tma_thread) {
      p5_tma_full_mbar[0].init(1);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (is_p5_tma_thread) {
      using X = cute::Underscore;
      using ProducerBarrierType = typename cutlass::arch::ClusterTransactionBarrier::ValueType;
      auto& barrier = p5_tma_full_mbar[0];
      std::uint32_t stage_bytes = 0u;
      const int output_tile_coord = output_row_base / kOutputTile;

      const auto& tma_load_a = *p5_tma_load_a_ptr;
      auto mA_mkl = tma_load_a.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(weight.output_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gA_mkl = cute::local_tile(
          mA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_a = tma_load_a.get_slice(0);
      auto gA = gA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgA = block_tma_a.partition_S(gA);
      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto tAsA = block_tma_a.partition_D(sA);
      auto tma_copy_a =
          tma_load_a.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_a,
          tAgA(cute::_, cute::_, cute::_, k_tile_coord),
          tAsA(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::size(cute::take<0, 2>(SmemLayoutA{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementAB>));

      const auto& tma_load_sfa = *p5_tma_load_sfa_ptr;
      auto mSFA_mkl = tma_load_sfa.get_tma_tensor(cute::shape(
          routed_p5_tma::MakeP5ScaleLayoutSFA(
              static_cast<int32_t>(weight.output_rows),
              static_cast<int32_t>(weight.input_cols))));
      auto gSFA_mkl = cute::local_tile(
          mSFA_mkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<cute::_1, X, cute::_1>{});
      auto block_tma_sfa = tma_load_sfa.get_slice(0);
      auto gSFA = gSFA_mkl(cute::_, cute::_, output_tile_coord, cute::_, 0);
      auto tAgSFA = block_tma_sfa.partition_S(gSFA);
      auto sSFA_ = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
      auto sSFA = cute::as_position_independent_swizzle_tensor(sSFA_);
      auto tAsSFA = block_tma_sfa.partition_D(sSFA);
      auto tma_copy_sfa =
          tma_load_sfa.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_sfa,
          tAgSFA(cute::_, cute::_, cute::_, k_tile_coord),
          tAsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::cosize(cute::take<0, 2>(SmemLayoutSFA{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementSF>));

      const auto& tma_load_b = p5_tma_load_b_descriptors[cta_index];
      auto mB_nkl = tma_load_b.get_tma_tensor(cute::make_shape(
          static_cast<int32_t>(valid_rows),
          static_cast<int32_t>(weight.input_cols),
          cute::Int<1>{}));
      auto gB_nkl = cute::local_tile(
          mB_nkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<X, cute::_1, cute::_1>{});
      auto block_tma_b = tma_load_b.get_slice(0);
      auto gB = gB_nkl(cute::_, cute::_, 0, cute::_, 0);
      auto tBgB = block_tma_b.partition_S(gB);
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto tBsB = block_tma_b.partition_D(sB);
      auto tma_copy_b =
          tma_load_b.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_b,
          tBgB(cute::_, cute::_, cute::_, k_tile_coord),
          tBsB(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::size(cute::take<0, 2>(SmemLayoutB{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementAB>));

      const auto& tma_load_sfb = p5_tma_load_sfb_descriptors[cta_index];
      auto mSFB_nkl = tma_load_sfb.get_tma_tensor(cute::shape(
          routed_p5_tma::MakeP5ScaleLayoutSFB(
              static_cast<int32_t>(kRowTile),
              static_cast<int32_t>(weight.input_cols))));
      auto gSFB_nkl = cute::local_tile(
          mSFB_nkl,
          routed_p5_tma::MmaTileShape{},
          cute::make_coord(cute::_, cute::_, cute::_),
          cute::Step<X, cute::_1, cute::_1>{});
      auto block_tma_sfb = tma_load_sfb.get_slice(0);
      auto gSFB = gSFB_nkl(cute::_, cute::_, 0, cute::_, 0);
      auto tBgSFB = block_tma_sfb.partition_S(gSFB);
      auto sSFB_ = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
      auto sSFB = cute::as_position_independent_swizzle_tensor(sSFB_);
      auto tBsSFB = block_tma_sfb.partition_D(sSFB);
      auto tma_copy_sfb =
          tma_load_sfb.with(*cute::recast_ptr<ProducerBarrierType>(&barrier));
      cute::copy(
          tma_copy_sfb,
          tBgSFB(cute::_, cute::_, cute::_, k_tile_coord),
          tBsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}));
      stage_bytes += static_cast<std::uint32_t>(cutlass::bits_to_bytes(
          cute::cosize(cute::take<0, 2>(SmemLayoutSFB{})) *
          cute::sizeof_bits_v<routed_p5_tma::ElementSF>));

      p5_tma_full_mbar[0].arrive_and_expect_tx(stage_bytes);
    }

    if (is_fp4_consumer_thread) {
      p5_tma_full_mbar[0].wait(0);

      auto tiled_mma = TiledMma{};
      const int thread_id = warp_id * 32 + lane_id;
      auto thread_mma = tiled_mma.get_thread_slice(thread_id);

      auto sA_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_a), SmemLayoutA{});
      auto sB_ = cute::make_tensor(cute::make_smem_ptr(smem_swizzled_b), SmemLayoutB{});
      auto sSFA = cute::make_tensor(cute::make_smem_ptr(a_scale_smem), SmemLayoutSFA{});
      auto sSFB = cute::make_tensor(cute::make_smem_ptr(b_scale_smem), SmemLayoutSFB{});
      auto sA = cute::as_position_independent_swizzle_tensor(sA_);
      auto sB = cute::as_position_independent_swizzle_tensor(sB_);
      auto sScaleA = cute::as_position_independent_swizzle_tensor(sSFA);
      auto sScaleB = cute::as_position_independent_swizzle_tensor(sSFB);

      auto tCrA = thread_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
      auto tCrB = thread_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
      auto tCrSFA = CollectiveMainloop{}.partition_fragment_SFA(
          sSFA(cute::_, cute::_, cute::Int<0>{}), thread_mma);
      auto tCrSFB = CollectiveMainloop{}.partition_fragment_SFB(
          sSFB(cute::_, cute::_, cute::Int<0>{}), thread_mma);

      auto s2r_copy_A = cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
      auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_id);
      auto tCsA = s2r_thr_A.partition_S(sA);
      auto tCrA_cv = s2r_thr_A.retile_D(tCrA);

      auto s2r_copy_B = cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
      auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_id);
      auto tCsB = s2r_thr_B.partition_S(sB);
      auto tCrB_cv = s2r_thr_B.retile_D(tCrB);

      auto tile_shape_mnk = cute::tile_shape(tiled_mma);
      auto s2r_copy_SFA = cute::make_tiled_copy_impl(
          SmemCopyAtomSFA{},
          Traits::GetLayoutSFATV(tiled_mma),
          cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_id);
      auto tCsSFA = s2r_thr_SFA.partition_S(sScaleA);
      auto tCrSFA_cv = s2r_thr_SFA.retile_D(tCrSFA);

      auto s2r_copy_SFB = cute::make_tiled_copy_impl(
          SmemCopyAtomSFB{},
          Traits::GetLayoutSFBTV(tiled_mma),
          cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_id);
      auto tCsSFB = s2r_thr_SFB.partition_S(sScaleB);
      auto tCrSFB_cv = s2r_thr_SFB.retile_D(tCrSFB);

      auto accum_tensor = cute::make_tensor(
          reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
          typename Traits::AccumLayout{});

      cute::copy(s2r_copy_A, tCsA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrA_cv);
      cute::copy(s2r_copy_B, tCsB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrB_cv);
      cute::copy(tCsSFA(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFA_cv);
      cute::copy(tCsSFB(cute::_, cute::_, cute::_, cute::Int<0>{}), tCrSFB_cv);

      using MMAOp = typename TiledMma::MMA_Op;
      for (int k = 0; k < cute::size<2>(tCrA_cv); ++k) {
        cute::fp4_shift_A(MMAOp{}, tCrA_cv(cute::_, cute::_, k));
        cute::fp4_shift_B(MMAOp{}, tCrB_cv(cute::_, cute::_, k));
      }

      using MMAAtom = typename TiledMma::Atom;
      MMAAtom mma_atom;
      constexpr int M_tiles = cute::size<1>(decltype(tCrA){});
      constexpr int N_tiles = cute::size<1>(decltype(tCrB){});
      constexpr int K_blocks = cute::size<2>(decltype(tCrA){});
      for (int k = 0; k < K_blocks; ++k) {
        for (int n = 0; n < N_tiles; ++n) {
          for (int m = 0; m < M_tiles; ++m) {
            auto a_atom = tCrA(cute::_, m, k);
            auto b_atom = tCrB(cute::_, n, k);
            auto c_atom = accum_tensor(cute::_, m, n);
            auto sfa_atom = tCrSFA(cute::_, m, k);
            auto sfb_atom = tCrSFB(cute::_, n, k);
            mma_atom.call(
                c_atom,
                cute::make_zip_tensor(a_atom, sfa_atom),
                cute::make_zip_tensor(b_atom, sfb_atom),
                c_atom);
          }
        }
      }
    }
    __syncthreads();
  }

  if (is_fp4_consumer_thread) {
    auto tiled_mma = TiledMma{};
    auto thread_mma = tiled_mma.get_thread_slice(warp_id * 32 + lane_id);
    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        typename Traits::AccumLayout{});
    auto dense_c = cute::make_identity_tensor(
        cute::make_shape(cute::Int<kOutputTile>{}, cute::Int<kRowTile>{}));
    auto part_c = thread_mma.partition_C(dense_c);
    for (int i = 0; i < static_cast<int>(cute::size(part_c)); ++i) {
      auto coord = part_c(i);
      const int output_col_offset = static_cast<int>(cute::get<0>(coord));
      const int token_row = static_cast<int>(cute::get<1>(coord));
      if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
        continue;
      }
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_row);
      const float output_alpha =
          weight_tensor_scale *
          LoadContiguousInputTensorScale(
              input_per_row_tensor_scales,
              input_tensor_scale_data,
              input_row);
      output[input_row * weight.output_rows +
             static_cast<std::size_t>(output_row_base + output_col_offset)] =
          accum_tensor(i) * output_alpha;
    }
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_m_limits,
    const int* expert_first_token_offsets,
    int token_tile_dim,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr)) {
    return;
  }
  (void) input_per_row_tensor_scales;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_batch_indices[cta_index];
  const int batch_row_begin = expert_first_token_offsets[expert_index];
  const int batch_cta_begin = batch_row_begin / token_tile_dim;
  const int row_start =
      batch_row_begin + (cta_index - batch_cta_begin) * token_tile_dim;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows = max(0, min(m_limit - row_start, token_tile_dim));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            input_dq_scales != nullptr
                ? input_dq_scales[scale_row_offset + block]
                : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                      input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRowsBf16Kernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_batch_indices,
    const int* cta_m_limits,
    const int* expert_first_token_offsets,
    int token_tile_dim,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output == nullptr) {
    return;
  }
  (void) input_per_row_tensor_scales;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_batch_indices[cta_index];
  const int batch_row_begin = expert_first_token_offsets[expert_index];
  const int batch_cta_begin = batch_row_begin / token_tile_dim;
  const int row_start =
      batch_row_begin + (cta_index - batch_cta_begin) * token_tile_dim;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows = max(0, min(m_limit - row_start, token_tile_dim));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }

    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            input_dq_scales != nullptr
                ? input_dq_scales[scale_row_offset + block]
                : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                      input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        __float2bfloat16(c_tile[tile_output_row][tile_token]);
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2MaxAbsKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_m_limits,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    unsigned int* expert_max_bits) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];
  __shared__ float shared_max[kPlannedThreadsPerBlock];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      input_tensor_scale_data == nullptr ||
      expert_max_bits == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_index * kGroupedTokenTile;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows =
      max(0, min(m_limit - row_start, kGroupedTokenTile));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                            : *input_tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
            input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  float thread_max = 0.0f;
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const float value = fused_decode::Relu2(c_tile[tile_output_row][tile_token]);
    thread_max = fmaxf(thread_max, value);
  }

  shared_max[tid] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
      shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(&expert_max_bits[expert_index], __float_as_uint(shared_max[0]));
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2PackKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    std::size_t output_row_capacity,
    Nvfp4ScaleLayout output_scale_layout,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales,
    float* output_activation_scales) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      (input_tensor_scale_data == nullptr && input_dq_scales == nullptr) ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr ||
      output_activation_scales == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_input_row = weight.input_cols / 2;
  const std::size_t input_blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float expert_input_tensor_scale =
      input_dq_scales == nullptr
          ? (input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                                   : *input_tensor_scale_data)
          : 1.0f;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_input_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * input_blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_input_row;
        const std::size_t scale_row_offset = input_row * input_blocks_per_row;
        const float block_scale =
            input_dq_scales != nullptr
                ? input_dq_scales[scale_row_offset + block]
                : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                      (input_per_row_tensor_scales != nullptr
                           ? input_per_row_tensor_scales[input_row]
                           : expert_input_tensor_scale);
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  if (lane_id >= valid_rows || warp_row >= output_rows_this_tile) {
    return;
  }

  const std::size_t output_row = static_cast<std::size_t>(row_start + lane_id);
  if (output_row >= output_row_capacity) {
    return;
  }
  const std::size_t blocks_per_output_row =
      output_rows_per_expert / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_output_row, kNvfp4ScaleBlockTile);
  const std::size_t block_col =
      static_cast<std::size_t>(output_row_base + warp_row) / fused_decode::kNvfp4BlockWidth;
  const float output_tensor_scale = 1.0f;

  float activated[16];
  float block_max_abs = 0.0f;
  for (int col = 0; col < 16; ++col) {
    const float value = fused_decode::Relu2(c_tile[warp_row + col][lane_id]);
    activated[col] = value;
    if (value > block_max_abs) {
      block_max_abs = value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * output_tensor_scale));
  }
  const std::uint8_t encoded_block_scale = fused_decode::EncodeFp8Scale(block_scale);
  const std::size_t block_scale_offset = output_row * blocks_per_output_row + block_col;
  output_activation_scales[block_scale_offset] = output_tensor_scale * block_scale;
  output_block_scales[block_scale_offset] = encoded_block_scale;
  output_matmul_scales[ExecutionScaleOffset(
      output_row, block_col, padded_blocks_per_row, output_scale_layout)] = encoded_block_scale;

  const float pack_scale = output_tensor_scale * block_scale;
  const std::size_t packed_row_offset =
      output_row * (output_rows_per_expert / 2u) +
      block_col * (fused_decode::kNvfp4BlockWidth / 2u);
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t lhs = fused_decode::EncodeFp4(activated[pair * 2] / pack_scale);
    const std::uint8_t rhs = fused_decode::EncodeFp4(activated[pair * 2 + 1] / pack_scale);
    output_packed[packed_row_offset + static_cast<std::size_t>(pair)] =
        static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
  }
}

template <bool kGroupedRows>
__global__ void Nvfp4MatVecRowsKernel(
    const float* input,
    std::size_t input_row_count,
    const int* expert_offsets,
    int expert_index,
    fused_decode::Nvfp4WeightView weight,
    float* output) {
  __shared__ float partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

  const std::size_t output_row = static_cast<std::size_t>(blockIdx.x);
  if (output_row >= weight.output_rows) {
    return;
  }

  int begin_row = 0;
  int end_row = static_cast<int>(input_row_count);
  if constexpr (kGroupedRows) {
    begin_row = expert_offsets[expert_index];
    end_row = expert_offsets[expert_index + 1];
  }
  if (begin_row >= end_row) {
    return;
  }

  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * (weight.input_cols / 2);
  const std::size_t scale_row_offset = output_row * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;

  for (int tile_begin = begin_row; tile_begin < end_row; tile_begin += kGroupedTokenTile) {
    const int remaining_rows = end_row - tile_begin;
    const int valid_rows =
        remaining_rows < kGroupedTokenTile ? remaining_rows : kGroupedTokenTile;
    float accum[kGroupedTokenTile] = {0.0f};

    for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
         pair_index < pairs_per_row;
         pair_index += blockDim.x) {
      const std::size_t block = pair_index / 8u;
      const std::size_t pair_in_block = pair_index % 8u;
      const std::size_t col = block * fused_decode::kNvfp4BlockWidth + pair_in_block * 2u;
      const float block_scale =
          LoadNvfp4WeightBlockScale(weight, output_row, block);
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale;
      const float w1 = fused_decode::DecodeFp4((packed >> 4) & 0x0Fu) * block_scale;
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        const float* input_row =
            input + static_cast<std::size_t>(tile_begin + tile_row) * weight.input_cols;
        accum[tile_row] += input_row[col] * w0;
        accum[tile_row] += input_row[col + 1] * w1;
      }
    }

    for (int tile_row = 0; tile_row < kGroupedTokenTile; ++tile_row) {
      partial_sums[tile_row * blockDim.x + threadIdx.x] =
          tile_row < valid_rows ? accum[tile_row] : 0.0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
          partial_sums[tile_row * blockDim.x + threadIdx.x] +=
              partial_sums[tile_row * blockDim.x + threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        output[static_cast<std::size_t>(tile_begin + tile_row) * weight.output_rows + output_row] =
            partial_sums[tile_row * blockDim.x];
      }
    }
    __syncthreads();
  }
}

template <int kOutputTile, int kThreadsPerBlock>
__global__ void Nvfp4ContiguousWmmaMatVecRowsKernel(
    const float* input,
    std::size_t input_row_count,
    FusedNvfp4WeightView weight,
    float* output) {
  constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

  __shared__ __nv_bfloat16 a_tile[kOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kOutputTile][kPlannedWmmaTileM];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  const int row_start = static_cast<int>(blockIdx.y) * kPlannedWmmaTileM;
  if (warp_id >= kWarpsPerBlock ||
      static_cast<std::size_t>(output_row_base) >= weight.output_rows ||
      static_cast<std::size_t>(row_start) >= input_row_count) {
    return;
  }

  const std::size_t remaining_input_rows =
      input_row_count - static_cast<std::size_t>(row_start);
  const int valid_rows = static_cast<int>(
      remaining_input_rows < static_cast<std::size_t>(kPlannedWmmaTileM)
          ? remaining_input_rows
          : static_cast<std::size_t>(kPlannedWmmaTileM));
  const std::size_t remaining_output_rows =
      weight.output_rows - static_cast<std::size_t>(output_row_base);
  const int output_rows_this_tile = static_cast<int>(
      remaining_output_rows < static_cast<std::size_t>(kOutputTile)
          ? remaining_output_rows
          : static_cast<std::size_t>(kOutputTile));
    const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            LoadNvfp4WeightBlockScale(weight, static_cast<std::size_t>(output_row), block);
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * weight.output_rows + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void ReduceSelectionOutputsKernel(
    const float* grouped_output,
    const int* selection_to_sorted,
    const int* sorted_to_permuted,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    std::size_t hidden_size,
    float* output,
    float* routed_output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = token_count * hidden_size;
  if (index >= count) {
    return;
  }

  const std::size_t token_index = index / hidden_size;
  const std::size_t hidden_index = index % hidden_size;
  double accum = 0.0;
  for (std::size_t slot = 0; slot < top_k; ++slot) {
    const std::size_t selection_index = token_index * top_k + slot;
    const int sorted_index = selection_to_sorted[selection_index];
    if (sorted_index < 0) {
      continue;
    }
    const int output_row =
        sorted_to_permuted != nullptr ? sorted_to_permuted[sorted_index] : sorted_index;
    if (output_row < 0) {
      continue;
    }
    const std::size_t grouped_offset =
        static_cast<std::size_t>(output_row) * hidden_size + hidden_index;
    accum += static_cast<double>(selected_weights[selection_index]) *
             static_cast<double>(grouped_output[grouped_offset]);
  }
  const float reduced = static_cast<float>(accum);
  output[index] = reduced;
  if (routed_output != nullptr) {
    routed_output[index] = reduced;
  }
}

__global__ void AccumulateSharedOutputKernel(
    const float* shared_output,
    std::size_t count,
    float* output,
    float* shared_output_copy) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float value = shared_output[index];
  output[index] += value;
  if (shared_output_copy != nullptr) {
    shared_output_copy[index] = value;
  }
}

bool LaunchZeroBuffer(float* data, std::size_t count) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ZeroBufferKernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchZeroBf16Buffer(__nv_bfloat16* data, std::size_t count) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ZeroBf16BufferKernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchFillFloatBuffer(float* data, std::size_t count, float value) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  FillFloatBufferKernel<<<grid, block>>>(data, count, value);
  return CheckCuda(cudaGetLastError());
}

bool ClearDeviceNvfp4Matrix(
    DeviceNvfp4Matrix* matrix,
    float tensor_scale = 1.0f,
    float per_row_tensor_scale = 0.0f) {
  if (matrix == nullptr || !matrix->valid()) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(matrix->packed_data()), 0, matrix->packed_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(matrix->block_scales_data()),
          0,
          matrix->block_scales_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(matrix->matmul_block_scales_data()),
          0,
          matrix->matmul_block_scales_nbytes()))) {
    return false;
  }

  float* per_row_scales = const_cast<float*>(matrix->per_row_tensor_scales());
  if (per_row_scales == nullptr) {
    return false;
  }
  if (per_row_tensor_scale == 0.0f) {
    if (!CheckCuda(cudaMemset(
            per_row_scales,
            0,
            matrix->rows() * sizeof(float)))) {
      return false;
    }
  } else if (!LaunchFillFloatBuffer(
                 per_row_scales,
                 matrix->rows(),
                 per_row_tensor_scale)) {
    return false;
  }

  return CheckCuda(cudaMemcpy(
      const_cast<std::uint8_t*>(matrix->tensor_scale_data()),
      &tensor_scale,
      sizeof(tensor_scale),
      cudaMemcpyHostToDevice));
}

bool LaunchGatherRows(
    const float* input,
    const int* row_indices,
    const int* active_output_rows,
    std::size_t output_row_capacity,
    std::size_t input_rows,
    std::size_t cols,
    float* output) {
  const std::size_t count = output_row_capacity * cols;
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  GatherRowsKernel<<<grid, block>>>(
      input,
      row_indices,
      active_output_rows,
      output,
      output_row_capacity,
      input_rows,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchComputeExpertActivationScales(
    const float* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      n_experts == 0 ||
      cols == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts));
  ComputeExpertActivationScalesKernel<<<grid, block>>>(
      input,
      expert_first_token_offsets,
      n_experts,
      cols,
      output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchComputeExpertActivationScalesBf16(
    const __nv_bfloat16* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      n_experts == 0 ||
      cols == 0 ||
      output_scales == nullptr) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts));
  ComputeExpertActivationScalesBf16Kernel<<<grid, block>>>(
      input,
      expert_first_token_offsets,
      n_experts,
      cols,
      output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRoutedBf16Relu2Pack(
    const __nv_bfloat16* source,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    DeviceNvfp4Matrix* output_pack,
    float* output_dequant_scales) {
  if (source == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      output_dequant_scales == nullptr) {
    return false;
  }

  const auto current_padded_row_capacity =
      DeviceMoeLaunchPlan::PaddedRowCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_padded_row_capacity.has_value() ||
      *current_padded_row_capacity == 0 ||
      *current_padded_row_capacity > output_pack->rows()) {
    return false;
  }

  const std::size_t blocks_per_row =
      output_pack->cols() / fused_decode::kNvfp4BlockWidth;
  const std::size_t scale_count = (*current_padded_row_capacity) * blocks_per_row;
  if (blocks_per_row == 0 ||
      !ClearDeviceNvfp4Matrix(output_pack) ||
      !LaunchZeroBuffer(output_dequant_scales, scale_count)) {
    return false;
  }

  const auto launch_rows = [&](auto rows_per_cta_tag) -> bool {
    constexpr int kProcessRows = decltype(rows_per_cta_tag)::value;
    const dim3 block(fused_decode::kThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(
            (*current_padded_row_capacity + (kProcessRows * block.x) - 1u) /
            (kProcessRows * block.x)),
        static_cast<unsigned int>(blocks_per_row));
    return LaunchProgrammaticKernel(
        grid,
        block,
        RoutedBf16Relu2PackKernel<kProcessRows>,
        source,
        routing->expert_offsets(),
        launch_plan->expert_first_token_offsets(),
        static_cast<int>(launch_plan->n_experts()),
        output_pack->cols(),
        RoundUp(blocks_per_row, kNvfp4ScaleBlockTile),
        output_pack->scale_layout(),
        output_dequant_scales,
        const_cast<std::uint8_t*>(output_pack->packed_data()),
        const_cast<std::uint8_t*>(output_pack->block_scales_data()),
        const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()));
  };

  const std::size_t workload = (*current_padded_row_capacity) * blocks_per_row;
  if (workload < (256u * 256u)) {
    return launch_rows(std::integral_constant<int, 1>{});
  }
  if (workload < (256u * 512u)) {
    return launch_rows(std::integral_constant<int, 2>{});
  }
  return launch_rows(std::integral_constant<int, 4>{});
}

bool LaunchMaxBitsToTensorScales(
    const unsigned int* max_bits,
    std::size_t count,
    float* output_scales) {
  if (max_bits == nullptr || output_scales == nullptr || count == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  MaxBitsToTensorScalesKernel<<<grid, block>>>(max_bits, count, output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchQuantizeDequantizeRows(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(row_capacity));
  QuantizeDequantizeRowsKernel<<<grid, block>>>(
      data,
      active_row_count,
      row_capacity,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRelu2(float* data, std::size_t count) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  Relu2Kernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRelu2Rows(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(((row_capacity * cols) + block.x - 1u) / block.x));
  Relu2RowsKernel<<<grid, block>>>(
      data,
      active_row_count,
      row_capacity,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchGroupedMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (weights == nullptr || n_experts == 0 || output_rows_per_expert == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts * output_rows_per_expert));
  Nvfp4GroupedExpertMatVecRowsKernel<<<grid, block>>>(
      input,
      expert_offsets,
      n_experts,
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchContiguousFp4MatVec(
    const DeviceNvfp4Matrix& input_pack,
    const SharedContiguousPreparedLaunch& prepared_launch,
    const FusedNvfp4WeightView& weight,
    float* output) {
  if (!input_pack.valid() ||
      !ValidFusedNvfp4WeightView(weight) ||
      output == nullptr) {
    return false;
  }

  const auto& launch_plan = prepared_launch.launch_plan;
  const auto& execution = prepared_launch.execution;
  if (execution.backend_kind != GemmBackendKind::kSm120ContiguousSharedNvfp4 ||
      launch_plan.kernel_family != GemmKernelFamily::kSm120ContiguousSharedNvfp4 ||
      launch_plan.m == 0 ||
      launch_plan.n != weight.output_rows ||
      launch_plan.k != weight.input_cols ||
      launch_plan.m > input_pack.rows() ||
      input_pack.cols() != weight.input_cols ||
      input_pack.scale_layout() != Nvfp4ScaleLayout::kSwizzled128x4 ||
      launch_plan.tile_m != 128u ||
      launch_plan.tile_n != 128u ||
      launch_plan.cta_m_count == 0 ||
      launch_plan.cta_n_count == 0) {
    return false;
  }

  if ((weight.input_cols % 128u) != 0 ||
      weight.p5_tma_load_a == nullptr ||
      weight.p5_tma_load_sfa == nullptr ||
      prepared_launch.p5_tma_load_b_descriptors == nullptr ||
      prepared_launch.p5_tma_load_sfb_descriptors == nullptr) {
    return false;
  }

  const dim3 block(kSharedContiguousP5ThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(launch_plan.cta_n_count),
      static_cast<unsigned int>(launch_plan.cta_m_count));
  const auto* p5_tma_load_b_descriptors =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadB*>(
          prepared_launch.p5_tma_load_b_descriptors);
  const auto* p5_tma_load_sfb_descriptors =
      reinterpret_cast<const routed_p5_tma::P5TmaLoadSFB*>(
          prepared_launch.p5_tma_load_sfb_descriptors);
  if (launch_plan.uses_programmatic_launch) {
    return LaunchProgrammaticKernel(
        grid,
        block,
        Nvfp4ContiguousSharedFp4P5Kernel,
        input_pack.packed_data(),
        input_pack.matmul_block_scales_data(),
        input_pack.scale_layout(),
        input_pack.device_tensor_scale_ptr(),
        input_pack.per_row_tensor_scales(),
        launch_plan.m,
        p5_tma_load_b_descriptors,
        p5_tma_load_sfb_descriptors,
        weight,
        output);
  }

  Nvfp4ContiguousSharedFp4P5Kernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.matmul_block_scales_data(),
      input_pack.scale_layout(),
      input_pack.device_tensor_scale_ptr(),
      input_pack.per_row_tensor_scales(),
      launch_plan.m,
      p5_tma_load_b_descriptors,
      p5_tma_load_sfb_descriptors,
      weight,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchContiguousMatVec(
    const float* input,
    std::size_t input_row_count,
    const FusedNvfp4WeightView& weight,
    float* output) {
  if (!ValidFusedNvfp4WeightView(weight)) {
    return false;
  }
  if (input_row_count == 0) {
    return true;
  }
  const int output_tile = SelectContiguousOutputTile(weight.output_rows, input_row_count);
  const std::size_t input_row_tile_count =
      CeilDiv(input_row_count, static_cast<std::size_t>(kPlannedWmmaTileM));

  if (output_tile == kContiguousLargeOutputTile) {
    const dim3 block(kContiguousLargeThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(CeilDiv(
            weight.output_rows,
            static_cast<std::size_t>(kContiguousLargeOutputTile))),
        static_cast<unsigned int>(input_row_tile_count));
    Nvfp4ContiguousWmmaMatVecRowsKernel<
        kContiguousLargeOutputTile,
        kContiguousLargeThreadsPerBlock><<<grid, block>>>(
        input,
        input_row_count,
        weight,
        output);
    return CheckCuda(cudaGetLastError());
  }

  if (output_tile == kContiguousMediumOutputTile) {
    const dim3 block(kContiguousMediumThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(CeilDiv(
            weight.output_rows,
            static_cast<std::size_t>(kContiguousMediumOutputTile))),
        static_cast<unsigned int>(input_row_tile_count));
    Nvfp4ContiguousWmmaMatVecRowsKernel<
        kContiguousMediumOutputTile,
        kContiguousMediumThreadsPerBlock><<<grid, block>>>(
        input,
        input_row_count,
        weight,
        output);
    return CheckCuda(cudaGetLastError());
  }

  const dim3 block(kContiguousSmallThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(CeilDiv(
          weight.output_rows,
          static_cast<std::size_t>(kContiguousSmallOutputTile))),
      static_cast<unsigned int>(input_row_tile_count));
  Nvfp4ContiguousWmmaMatVecRowsKernel<
      kContiguousSmallOutputTile,
      kContiguousSmallThreadsPerBlock><<<grid, block>>>(
      input,
      input_row_count,
      weight,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRowsKernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVecBf16(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRowsBf16Kernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      launch_plan->expert_first_token_offsets(),
      static_cast<int>(launch_plan->selected_token_tile()),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVecRelu2ExpertScales(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output_expert_scales) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          reinterpret_cast<void*>(output_expert_scales),
          0,
          launch_plan->n_experts() * sizeof(float)))) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRelu2MaxAbsKernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      reinterpret_cast<unsigned int*>(output_expert_scales));
  return CheckCuda(cudaGetLastError()) &&
         LaunchMaxBitsToTensorScales(
             reinterpret_cast<const unsigned int*>(output_expert_scales),
             launch_plan->n_experts(),
             output_expert_scales);
}

bool LaunchPlannedMatVecRelu2Pack(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const float* output_expert_scales,
    DeviceNvfp4Matrix* output_pack) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      output_pack->rows() < launch_plan->padded_row_capacity() ||
      output_pack->cols() != output_rows_per_expert) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0 || !ClearDeviceNvfp4Matrix(output_pack)) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRelu2PackKernel<<<grid, block>>>(
      input,
      output_expert_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output_pack->rows(),
      output_pack->scale_layout(),
      const_cast<std::uint8_t*>(output_pack->packed_data()),
      const_cast<std::uint8_t*>(output_pack->block_scales_data()),
      const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedPackedInputMatVec(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const auto launch_profile = [&](RoutedGemm2Profile profile) -> bool {
    if (RoutedProfileDebugEnabled()) {
      std::fprintf(
          stderr,
          "routed_gemm2 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
          dispatch_rows,
          active_selection_count,
          RoutedGemm2ProfileName(profile));
    }
    const int output_tile = RoutedOutputTile(profile);
    const std::size_t output_row_tile_count =
        (output_rows_per_expert + static_cast<std::size_t>(output_tile) - 1u) /
        static_cast<std::size_t>(output_tile);
    if (output_row_tile_count == 0) {
      return false;
    }
    const dim3 block(kRoutedThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(output_row_tile_count),
        static_cast<unsigned int>(*current_cta_capacity));
    switch (profile) {
      case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
            nvfp4_bridge::UnifiedRoutedFp4Profile::kP12,
            float><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.matmul_block_scales_data(),
            input_pack.scale_layout(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            nullptr,
            nullptr,
            weights,
            output_rows_per_expert,
            output,
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<std::size_t>(0),
            static_cast<std::size_t>(0),
            Nvfp4ScaleLayout::kSwizzled128x4);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
        g_enable_p13_scale_trace = std::getenv("NEMOTRON_P13_SCALE_DEBUG") != nullptr ? 1 : 0;
        g_enable_p13_scale_map_probe =
            std::getenv("NEMOTRON_P13_SCALE_MAP_PROBE") != nullptr ? 1 : 0;
        g_enable_p13_debug_trace = std::getenv("NEMOTRON_P13_DEBUG_TRACE") != nullptr ? 1 : 0;
        g_p13_debug_trace_target_valid_rows = -1;
        g_p13_debug_trace_target_thread_id = -1;
        if (const char* target_valid_rows_env =
                std::getenv("NEMOTRON_P13_DEBUG_TRACE_TARGET_VALID_ROWS")) {
          g_p13_debug_trace_target_valid_rows = std::atoi(target_valid_rows_env);
        }
        if (const char* target_thread_env =
                std::getenv("NEMOTRON_P13_DEBUG_TRACE_TARGET_THREAD_ID")) {
          g_p13_debug_trace_target_thread_id = std::atoi(target_thread_env);
        }
        if (g_enable_p13_scale_trace != 0 ||
            g_enable_p13_scale_map_probe != 0 ||
            g_enable_p13_debug_trace != 0) {
          g_p13_scale_trace = {};
          g_p13_debug_trace = {};
          g_p13_debug_trace_claimed = 0;
        }
        if (RoutedEnvEnabled("NEMOTRON_DEBUG_USE_OLD_P13_KERNEL")) {
          Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmem<
              float,
              128><<<grid, block>>>(
              input_pack.packed_data(),
              input_pack.block_scales_data(),
              input_pack.matmul_block_scales_data(),
              input_pack.scale_layout(),
              input_pack.device_tensor_scale_ptr(),
              input_expert_tensor_scales,
              input_dq_scales,
              launch_plan->num_non_exiting_ctas(),
              launch_plan->cta_idx_xy_to_batch_idx(),
              launch_plan->cta_row_starts(),
              launch_plan->cta_valid_rows(),
              weights,
              output_rows_per_expert,
              output);
        } else {
          Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
              nvfp4_bridge::UnifiedRoutedFp4Profile::kP13,
              float><<<grid, block>>>(
              input_pack.packed_data(),
              input_pack.matmul_block_scales_data(),
              input_pack.scale_layout(),
              input_pack.device_tensor_scale_ptr(),
              input_expert_tensor_scales,
              input_dq_scales,
              input_per_row_tensor_scales,
              launch_plan->num_non_exiting_ctas(),
              launch_plan->cta_idx_xy_to_batch_idx(),
              launch_plan->cta_row_starts(),
              launch_plan->cta_valid_rows(),
              nullptr,
              nullptr,
              weights,
              output_rows_per_expert,
              output,
              static_cast<std::uint8_t*>(nullptr),
              static_cast<std::uint8_t*>(nullptr),
              static_cast<std::uint8_t*>(nullptr),
              static_cast<float*>(nullptr),
              static_cast<std::size_t>(0),
              static_cast<std::size_t>(0),
              Nvfp4ScaleLayout::kSwizzled128x4);
        }
        if (g_enable_p13_scale_trace != 0) {
          if (!CheckCuda(cudaDeviceSynchronize())) {
            return false;
          }
          PrintP13ScaleTrace();
        }
        return CheckCuda(cudaGetLastError());
      case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
        g_enable_p15_scale_trace = std::getenv("NEMOTRON_P15_SCALE_DEBUG") != nullptr ? 1 : 0;
        if (g_enable_p15_scale_trace != 0) {
          g_p15_scale_trace = {};
        }
        Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
            nvfp4_bridge::UnifiedRoutedFp4Profile::kP15,
            float><<<grid, dim3(256)>>>(
            input_pack.packed_data(),
            input_pack.matmul_block_scales_data(),
            input_pack.scale_layout(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            nullptr,
            nullptr,
            weights,
            output_rows_per_expert,
            output,
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<std::size_t>(0),
            static_cast<std::size_t>(0),
            Nvfp4ScaleLayout::kSwizzled128x4);
        if (!CheckCuda(cudaGetLastError())) {
          return false;
        }
        if (g_enable_p15_scale_trace != 0) {
          if (!CheckCuda(cudaDeviceSynchronize())) {
            return false;
          }
          PrintP15ScaleTrace();
        }
        return true;
      case RoutedGemm2Profile::kLegacy:
        break;
    }
    return false;
  };

  const RoutedGemm2Profile profile = SelectRoutedGemm2Profile(dispatch_rows);
  if (profile != RoutedGemm2Profile::kLegacy && launch_profile(profile)) {
    return true;
  }
  if (RoutedProfileDebugEnabled()) {
    std::fprintf(
        stderr,
        "routed_gemm2 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
        dispatch_rows,
        active_selection_count,
        RoutedGemm2ProfileName(RoutedGemm2Profile::kLegacy));
  }

  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      input_dq_scales,
      input_per_row_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      launch_plan->expert_first_token_offsets(),
      static_cast<int>(launch_plan->selected_token_tile()),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedPackedInputMatVecBf16(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const auto launch_profile = [&](RoutedGemm1Profile profile) -> bool {
    if (RoutedProfileDebugEnabled()) {
      std::fprintf(
          stderr,
          "routed_gemm1 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
          dispatch_rows,
          active_selection_count,
          RoutedGemm1ProfileName(profile));
    }
    const int output_tile = RoutedOutputTile(profile);
    const std::size_t output_row_tile_count =
        (output_rows_per_expert + static_cast<std::size_t>(output_tile) - 1u) /
        static_cast<std::size_t>(output_tile);
    if (output_row_tile_count == 0) {
      return false;
    }
    const dim3 block(kRoutedThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(output_row_tile_count),
        static_cast<unsigned int>(*current_cta_capacity));
    switch (profile) {
      case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalse<__nv_bfloat16, 128, 128><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            weights,
            output_rows_per_expert,
            output);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
        // kP1 has no BF16-output kernel during the rebuild. Fall through to
        // `default:` → returns false. Non-env-gated kP1 rows will fail
        // dispatch until step 6 wires the new NanoP1 kernel.
        break;
      case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue<__nv_bfloat16, 128, 128><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            weights,
            output_rows_per_expert,
            output);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kP5_128x128x64_SwapTrue: {
        g_enable_p5_scale_trace = std::getenv("NEMOTRON_P5_SCALE_DEBUG") != nullptr ? 1 : 0;
        if (g_enable_p5_scale_trace != 0) {
          g_p5_scale_trace = {};
        }
        const auto* p5_tma_load_b_descriptors =
            reinterpret_cast<const routed_p5_tma::P5TmaLoadB*>(
                input_pack.p5_tma_load_b_descriptors(*launch_plan));
        const auto* p5_tma_load_sfb_descriptors =
            reinterpret_cast<const routed_p5_tma::P5TmaLoadSFB*>(
                input_pack.p5_tma_load_sfb_descriptors(*launch_plan));
        if (!LaunchProgrammaticKernel(
            grid,
            block,
            Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
                nvfp4_bridge::UnifiedRoutedFp4Profile::kP5,
                __nv_bfloat16>,
            input_pack.packed_data(),
            input_pack.matmul_block_scales_data(),
            input_pack.scale_layout(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            p5_tma_load_b_descriptors,
            p5_tma_load_sfb_descriptors,
            weights,
            output_rows_per_expert,
            output,
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<std::uint8_t*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<std::size_t>(0),
            static_cast<std::size_t>(0),
            Nvfp4ScaleLayout::kSwizzled128x4)) {
          return false;
        }
        if (g_enable_p5_scale_trace != 0) {
          if (!CheckCuda(cudaDeviceSynchronize())) {
            return false;
          }
          PrintP5ScaleTrace();
        }
        return true;
      }
      case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
        Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue<__nv_bfloat16, 256, 64><<<grid, block>>>(
            input_pack.packed_data(),
            input_pack.block_scales_data(),
            input_pack.device_tensor_scale_ptr(),
            input_expert_tensor_scales,
            input_dq_scales,
            input_per_row_tensor_scales,
            launch_plan->num_non_exiting_ctas(),
            launch_plan->cta_idx_xy_to_batch_idx(),
            launch_plan->cta_row_starts(),
            launch_plan->cta_valid_rows(),
            weights,
            output_rows_per_expert,
            output);
        return CheckCuda(cudaGetLastError());
      case RoutedGemm1Profile::kLegacy:
        break;
    }
    return false;
  };

  const RoutedGemm1Profile profile = SelectRoutedGemm1Profile(dispatch_rows);
  if (profile != RoutedGemm1Profile::kLegacy && launch_profile(profile)) {
    return true;
  }
  if (RoutedProfileDebugEnabled()) {
    std::fprintf(
        stderr,
        "routed_gemm1 dispatch_rows=%zu active_selection_count=%zu profile=%s\n",
        dispatch_rows,
        active_selection_count,
        RoutedGemm1ProfileName(RoutedGemm1Profile::kLegacy));
  }

  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRowsBf16Kernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      input_dq_scales,
      input_per_row_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      launch_plan->expert_first_token_offsets(),
      static_cast<int>(launch_plan->selected_token_tile()),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

struct DirectFp4OutputBufferState {
  std::size_t output_row_tile_count = 0;
  std::size_t blocks_per_row = 0;
  std::size_t padded_blocks_per_row = 0;
  std::size_t scale_count = 0;
};

bool PrepareDirectFp4OutputBuffers(
    const DeviceMoeLaunchPlan& launch_plan,
    std::size_t output_rows_per_expert,
    DeviceNvfp4Matrix* output_pack,
    float* activation_output_scale,
    __nv_bfloat16* debug_prepack_output,
    DirectFp4OutputBufferState* state) {
  if (output_pack == nullptr ||
      activation_output_scale == nullptr ||
      state == nullptr) {
    return false;
  }

  state->output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  state->blocks_per_row = output_pack->cols() / fused_decode::kNvfp4BlockWidth;
  state->padded_blocks_per_row =
      RoundUp(state->blocks_per_row, kNvfp4ScaleBlockTile);
  state->scale_count = launch_plan.padded_row_capacity() * state->blocks_per_row;

  if (state->output_row_tile_count == 0 ||
      state->blocks_per_row == 0 ||
      !ClearDeviceNvfp4Matrix(output_pack, 1.0f, 1.0f) ||
      (debug_prepack_output != nullptr &&
       !LaunchZeroBf16Buffer(
           debug_prepack_output,
           launch_plan.padded_row_capacity() * output_rows_per_expert)) ||
      !LaunchZeroBuffer(activation_output_scale, state->scale_count)) {
    return false;
  }

  return true;
}

bool LaunchPlannedPackedInputMatVecFp4Direct(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    DeviceNvfp4Matrix* output_pack,
    float* activation_output_scale,
    __nv_bfloat16* debug_prepack_output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      activation_output_scale == nullptr ||
      output_pack->rows() < launch_plan->padded_row_capacity() ||
      output_pack->cols() != output_rows_per_expert) {
    return false;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }

  const RoutedGemm1Profile profile = SelectRoutedGemm1Profile(dispatch_rows);
  switch (profile) {
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue: {
      if (RoutedProfileDebugEnabled()) {
        std::fprintf(
            stderr,
            "routed_gemm1 dispatch_rows=%zu active_selection_count=%zu profile=%s mode=fp4_direct\n",
            dispatch_rows,
            active_selection_count,
            RoutedGemm1ProfileName(profile));
      }

      DirectFp4OutputBufferState output_state;
      if (!PrepareDirectFp4OutputBuffers(
              *launch_plan,
              output_rows_per_expert,
              output_pack,
              activation_output_scale,
              nullptr,
              &output_state)) {
        return false;
      }

      const dim3 block(kRoutedThreadsPerBlock);
      const dim3 grid(
          static_cast<unsigned int>(output_state.output_row_tile_count),
          static_cast<unsigned int>(*current_cta_capacity));

      g_enable_p5_scale_trace = std::getenv("NEMOTRON_P5_SCALE_DEBUG") != nullptr ? 1 : 0;
      if (g_enable_p5_scale_trace != 0) {
        g_p5_scale_trace = {};
      }

      const auto* p5_tma_load_b_descriptors =
          reinterpret_cast<const routed_p5_tma::P5TmaLoadB*>(
              input_pack.p5_tma_load_b_descriptors(*launch_plan));
      const auto* p5_tma_load_sfb_descriptors =
          reinterpret_cast<const routed_p5_tma::P5TmaLoadSFB*>(
              input_pack.p5_tma_load_sfb_descriptors(*launch_plan));

      if (!LaunchProgrammaticKernel(
              grid,
              block,
              Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
                  nvfp4_bridge::UnifiedRoutedFp4Profile::kP5,
                  __nv_bfloat16,
                  nvfp4_bridge::P5EpilogueMode::kFp4Direct>,
              input_pack.packed_data(),
              input_pack.matmul_block_scales_data(),
              input_pack.scale_layout(),
              input_pack.device_tensor_scale_ptr(),
              input_expert_tensor_scales,
              input_dq_scales,
              input_per_row_tensor_scales,
              launch_plan->num_non_exiting_ctas(),
              launch_plan->cta_idx_xy_to_batch_idx(),
              launch_plan->cta_row_starts(),
              launch_plan->cta_valid_rows(),
              p5_tma_load_b_descriptors,
              p5_tma_load_sfb_descriptors,
              weights,
              output_rows_per_expert,
              static_cast<__nv_bfloat16*>(nullptr),
              const_cast<std::uint8_t*>(output_pack->packed_data()),
              const_cast<std::uint8_t*>(output_pack->block_scales_data()),
              const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()),
              activation_output_scale,
              output_pack->cols(),
              output_state.padded_blocks_per_row,
              output_pack->scale_layout())) {
        return false;
      }

      if (g_enable_p5_scale_trace != 0) {
        if (!CheckCuda(cudaDeviceSynchronize())) {
          return false;
        }
        PrintP5ScaleTrace();
      }
      return true;
    }
    default:
      return false;
  }
}

[[maybe_unused]] bool LaunchPlannedPackedInputMatVecRelu2ExpertScales(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output_expert_scales) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(launch_plan->n_experts(), active_selection_count);
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          reinterpret_cast<void*>(output_expert_scales),
          0,
          launch_plan->n_experts() * sizeof(float)))) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2MaxAbsKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      weights,
      output_rows_per_expert,
      reinterpret_cast<unsigned int*>(output_expert_scales));
  return CheckCuda(cudaGetLastError()) &&
         LaunchMaxBitsToTensorScales(
             reinterpret_cast<const unsigned int*>(output_expert_scales),
             launch_plan->n_experts(),
             output_expert_scales);
}

bool LaunchReduceSelectionOutputs(
    const float* grouped_output,
    const int* selection_to_sorted,
    const int* sorted_to_permuted,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    std::size_t hidden_size,
    float* output,
    float* routed_output) {
  const std::size_t count = token_count * hidden_size;
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ReduceSelectionOutputsKernel<<<grid, block>>>(
      grouped_output,
      selection_to_sorted,
      sorted_to_permuted,
      selected_weights,
      token_count,
      top_k,
      hidden_size,
      output,
      routed_output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchAccumulateSharedOutput(
    const float* shared_output,
    std::size_t count,
    float* output,
    float* shared_output_copy) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  AccumulateSharedOutputKernel<<<grid, block>>>(
      shared_output,
      count,
      output,
      shared_output_copy);
  return CheckCuda(cudaGetLastError());
}

