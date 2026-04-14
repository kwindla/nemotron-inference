template <typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreNanoP1CFragmentsRowMajor(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    std::size_t output_stride,
    OutputType* output) {
  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(thread_idx);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);
  auto accum_tensor = cute::make_tensor(
      const_cast<CRegister*>(accum_storage),
      NanoP1AccumLayout{});
  static_assert(cute::size<0>(decltype(part_c){}) == cute::size<0>(NanoP1AccumLayout{}));
  static_assert(cute::size<1>(decltype(part_c){}) == cute::size<1>(NanoP1AccumLayout{}));
  static_assert(cute::size<2>(decltype(part_c){}) == cute::size<2>(NanoP1AccumLayout{}));
#pragma unroll
  for (int reg = 0; reg < static_cast<int>(cute::size<0>(NanoP1AccumLayout{})); ++reg) {
#pragma unroll
    for (int n_fragment = 0; n_fragment < static_cast<int>(cute::size<1>(NanoP1AccumLayout{})); ++n_fragment) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < static_cast<int>(cute::size<2>(NanoP1AccumLayout{})); ++m_fragment) {
        auto coord = part_c(cute::make_coord(reg, n_fragment, m_fragment));
        const int output_col_offset = CoordGet0(coord);
        const int token_row = CoordGet1(coord);
        if (token_row < 0 ||
            token_row >= valid_rows ||
            output_col_offset < 0 ||
            output_col_offset >= valid_cols) {
          continue;
        }
        const std::size_t row_index =
            static_cast<std::size_t>(row_start + token_row);
        const std::size_t col_index =
            static_cast<std::size_t>(output_col_base + output_col_offset);
        const float value = alpha * accum_tensor(reg, n_fragment, m_fragment);
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          output[row_index * output_stride + col_index] =
              __float2bfloat16(value);
        } else {
          output[row_index * output_stride + col_index] = value;
        }
      }
    }
  }
}

__device__ __forceinline__ void nvfp4_bridge::AccumulateNanoP1DirectPackRowMaxAbs(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int row_start,
    int valid_rows,
    int valid_cols,
    float* activation_output_scales) {
  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(thread_idx);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);
  auto accum_tensor = cute::make_tensor(
      const_cast<CRegister*>(accum_storage),
      NanoP1AccumLayout{});
#pragma unroll
  for (int reg = 0; reg < static_cast<int>(cute::size<0>(NanoP1AccumLayout{})); ++reg) {
#pragma unroll
    for (int n_fragment = 0; n_fragment < static_cast<int>(cute::size<1>(NanoP1AccumLayout{})); ++n_fragment) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < static_cast<int>(cute::size<2>(NanoP1AccumLayout{})); ++m_fragment) {
        auto coord = part_c(cute::make_coord(reg, n_fragment, m_fragment));
        const int output_col_offset = CoordGet0(coord);
        const int token_row = CoordGet1(coord);
        if (token_row < 0 ||
            token_row >= valid_rows ||
            output_col_offset < 0 ||
            output_col_offset >= valid_cols) {
          continue;
        }
        const float activated = fused_decode::Relu2(
            alpha * accum_tensor(reg, n_fragment, m_fragment));
        atomicMax(
            reinterpret_cast<unsigned int*>(
                activation_output_scales + static_cast<std::size_t>(row_start + token_row)),
            __float_as_uint(activated));
      }
    }
  }
}

__device__ __forceinline__ void nvfp4_bridge::StoreNanoP1DirectPackCFragments(
    float alpha,
    const CRegister* accum_storage,
    int thread_idx,
    int output_col_base,
    int row_start,
    int valid_rows,
    int valid_cols,
    int num_rows_global,
    int inter_size_global,
    std::uint8_t* packed_bytes,
    std::uint8_t* block_scales,
    std::uint8_t* matmul_block_scales,
    float* activation_output_scales,
    float* staging_tile) {
  if (packed_bytes == nullptr ||
      block_scales == nullptr ||
      matmul_block_scales == nullptr ||
      activation_output_scales == nullptr ||
      staging_tile == nullptr ||
      inter_size_global <= 0 ||
      (inter_size_global % fused_decode::kNvfp4BlockWidth) != 0) {
    return;
  }

  auto mma = NanoP1TiledMma{};
  auto thread_mma = mma.get_thread_slice(thread_idx);
  auto dense_c = cute::make_identity_tensor(
      cute::make_shape(
          cute::size<0>(NanoP1MmaTileShape{}),
          cute::size<1>(NanoP1MmaTileShape{})));
  auto part_c = thread_mma.partition_C(dense_c);
  auto accum_tensor = cute::make_tensor(
      const_cast<CRegister*>(accum_storage),
      NanoP1AccumLayout{});
  constexpr int kTileCols = cute::size<1>(NanoP1MmaTileShape{});
  const std::size_t blocks_per_row =
      static_cast<std::size_t>(inter_size_global) / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::size_t packed_row_bytes = static_cast<std::size_t>(inter_size_global) / 2u;

#pragma unroll
  for (int reg = 0; reg < static_cast<int>(cute::size<0>(NanoP1AccumLayout{})); ++reg) {
#pragma unroll
    for (int n_fragment = 0; n_fragment < static_cast<int>(cute::size<1>(NanoP1AccumLayout{})); ++n_fragment) {
#pragma unroll
      for (int m_fragment = 0; m_fragment < static_cast<int>(cute::size<2>(NanoP1AccumLayout{})); ++m_fragment) {
        auto coord = part_c(cute::make_coord(reg, n_fragment, m_fragment));
        const int output_col_offset = CoordGet0(coord);
        const int token_row = CoordGet1(coord);
        if (token_row < 0 ||
            token_row >= valid_rows ||
            output_col_offset < 0 ||
            output_col_offset >= valid_cols) {
          continue;
        }
        staging_tile[
            static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
            static_cast<std::size_t>(output_col_offset)] =
            fused_decode::Relu2(alpha * accum_tensor(reg, n_fragment, m_fragment));
      }
    }
  }
  __syncthreads();

  const int blocks_this_tile =
      (valid_cols + fused_decode::kNvfp4BlockWidth - 1) / fused_decode::kNvfp4BlockWidth;
  for (int row_block = thread_idx;
       row_block < valid_rows * blocks_this_tile;
       row_block += NanoP1ThreadsPerCta) {
    const int token_row = row_block / blocks_this_tile;
    const int block_in_tile = row_block % blocks_this_tile;
    const int col_in_tile = block_in_tile * fused_decode::kNvfp4BlockWidth;
    const int global_col = output_col_base + col_in_tile;
    const int global_row = row_start + token_row;
    if (global_row < 0 ||
        global_row >= num_rows_global ||
        global_col < 0 ||
        (global_col + fused_decode::kNvfp4BlockWidth) > inter_size_global) {
      continue;
    }

    const float row_scale = activation_output_scales[static_cast<std::size_t>(global_row)];
    float block_max_abs = 0.0f;
#pragma unroll
    for (int offset = 0; offset < fused_decode::kNvfp4BlockWidth; ++offset) {
      const float value = staging_tile[
          static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
          static_cast<std::size_t>(col_in_tile + offset)];
      block_max_abs = fmaxf(block_max_abs, fabsf(value));
    }

    const float raw_block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / fused_decode::kNvfp4Fp4MaxFinite);
    const float stabilized_block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * row_scale));
    const std::uint8_t raw_encoded = fused_decode::EncodeFp8Scale(raw_block_scale);
    const std::uint8_t stabilized_encoded =
        fused_decode::EncodeFp8Scale(stabilized_block_scale);
    const std::size_t global_block =
        static_cast<std::size_t>(global_col) / fused_decode::kNvfp4BlockWidth;
    const std::size_t scale_offset = ExecutionScaleOffset(
        static_cast<std::size_t>(global_row),
        global_block,
        padded_blocks_per_row,
        Nvfp4ScaleLayout::kSwizzled128x4);
    block_scales[scale_offset] = raw_encoded;
    matmul_block_scales[scale_offset] = stabilized_encoded;

    const float pack_scale = row_scale * stabilized_block_scale;
    const std::size_t packed_offset =
        static_cast<std::size_t>(global_row) * packed_row_bytes +
        static_cast<std::size_t>(global_col) / 2u;
#pragma unroll
    for (int pair = 0; pair < (fused_decode::kNvfp4BlockWidth / 2); ++pair) {
      const float lhs = staging_tile[
          static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
          static_cast<std::size_t>(col_in_tile + pair * 2 + 0)];
      const float rhs = staging_tile[
          static_cast<std::size_t>(token_row) * static_cast<std::size_t>(kTileCols) +
          static_cast<std::size_t>(col_in_tile + pair * 2 + 1)];
      const std::uint8_t lhs_nibble = fused_decode::EncodeFp4(lhs / pack_scale);
      const std::uint8_t rhs_nibble = fused_decode::EncodeFp4(rhs / pack_scale);
      packed_bytes[packed_offset + static_cast<std::size_t>(pair)] =
          static_cast<std::uint8_t>(
              (lhs_nibble & 0x0Fu) | ((rhs_nibble & 0x0Fu) << 4u));
    }
  }
  __syncthreads();
}

