
bool CopyP13DebugTrace(P13DebugTrace* out) {
  if (out == nullptr) {
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return false;
  }
  *out = g_p13_debug_trace;
  return true;
}

void ResetP13DebugTrace() {
  g_p13_debug_trace = {};
}

const char* SelectRoutedGemm1ProfileNameForTesting(std::size_t num_rows) {
  return RoutedGemm1ProfileName(SelectRoutedGemm1Profile(num_rows));
}

const char* ClassifyRoutedGemm1ProfileForTesting(std::size_t num_rows) {
  return RoutedGemm1ProfileClassName(SelectRoutedGemm1Profile(num_rows));
}

const char* SelectRoutedGemm2ProfileNameForTesting(std::size_t num_rows) {
  return RoutedGemm2ProfileName(SelectRoutedGemm2Profile(num_rows));
}

const char* ClassifyRoutedGemm2ProfileForTesting(std::size_t num_rows) {
  return RoutedGemm2ProfileClassName(SelectRoutedGemm2Profile(num_rows));
}

bool RunGroupedNvfp4ExpertMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (input == nullptr ||
      expert_offsets == nullptr ||
      n_experts == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchGroupedMatVec(
      input,
      expert_offsets,
      n_experts,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedNvfp4ExpertMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchPlannedMatVec(
      input,
      launch_plan,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedPackedNvfp4ExpertMatVecBf16(
    const DeviceNvfp4Matrix& input_pack,
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
  return LaunchPlannedPackedInputMatVecBf16(
      input_pack,
      nullptr,
      nullptr,
      input_pack.per_row_tensor_scales(),
      launch_plan,
      dispatch_rows,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedNvfp4ExpertMatVecBf16(
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
  return LaunchPlannedMatVecBf16(
      input,
      launch_plan,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunNanoP1KernelForTesting(
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
  return RunNanoP1KernelForTestingImpl(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      bf16_output,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      stream,
      accumulator_scratch);
}

bool RunNanoP1DirectPackKernelForTesting(
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
  return RunNanoP1DirectPackKernelForTestingImpl(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      packed_output,
      block_scales_output,
      matmul_block_scales_output,
      activation_output_scales,
      g1_alphas,
      num_rows,
      hidden_size,
      inter_size,
      stream);
}

__global__ void P5NativeDirectPackOracleKernel(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  constexpr int kTileRows = 128;
  constexpr int kTileCols = 128;
  constexpr int kPassRows = nvfp4_bridge::kTracedP5DirectStageRows;

  static_assert(cute::size(nvfp4_bridge::TracedP5TiledMma{}) == 256);
  auto mma = nvfp4_bridge::TracedP5TiledMma{};
  auto thr_mma = mma.get_thread_slice(threadIdx.x);
  auto ref_c = cute::make_identity_tensor(
      cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
  auto part_c = thr_mma.partition_C(ref_c);
  static_assert(
      decltype(cute::size(part_c))::value == 16,
      "P5 warp-local direct pack expects 16 logical accum coords per thread");
  constexpr int kLinearToBlockHalf[16] = {
      0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  constexpr int kLinearToTokenGroup[16] = {
      0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3};
  constexpr int kLinearToMHalf[16] = {
      0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};

  for (std::size_t row = threadIdx.x; row < static_cast<std::size_t>(kTileRows);
       row += blockDim.x) {
    per_row_tensor_scales[row] =
        per_row_tensor_scales_input != nullptr ? per_row_tensor_scales_input[row]
                                               : 1.0f;
  }
  if (threadIdx.x == 0) {
    *tensor_scale_data = 1.0f;
  }
  __syncthreads();

  for (int pass = 0; pass < (kTileRows / kPassRows); ++pass) {
    const int row_base = pass * kPassRows;
    const int pass_valid_rows =
        max(0, min(valid_rows - row_base, kPassRows));
    nvfp4_bridge::CRegister accum_storage[nvfp4_bridge::kTracedP5AccumProfileCosize];
#pragma unroll
    for (int i = 0; i < nvfp4_bridge::kTracedP5AccumProfileCosize; ++i) {
      accum_storage[i] = 0.0f;
    }

    auto accum_tensor = cute::make_tensor(
        reinterpret_cast<nvfp4_bridge::CRegister*>(&accum_storage[0]),
        nvfp4_bridge::TracedP5AccumProfileLayout{});

#pragma unroll
    for (int linear = 0; linear < static_cast<int>(cute::size(part_c)); ++linear) {
      auto coord = part_c(linear);
      const int m_coord = static_cast<int>(cute::get<0>(coord));
      const int n_coord = static_cast<int>(cute::get<1>(coord));
      const int block_half = kLinearToBlockHalf[linear];
      const int token_group = kLinearToTokenGroup[linear];
      const int m_half = kLinearToMHalf[linear];
      const int warp_id = threadIdx.x / 32;
      const int lane_id = threadIdx.x & 31;
      const int g_m = warp_id & 3;
      const int g_n = warp_id / 4;
      const int q = lane_id / 4;
      const int r = lane_id & 3;
      const int expected_n_coords[4] = {
          16 * g_n + 2 * r,
          16 * g_n + 2 * r + 1,
          16 * g_n + 8 + 2 * r,
          16 * g_n + 9 + 2 * r};
      const int expected_m_coord =
          (block_half == 0 ? 0 : 64) + 16 * g_m + (m_half == 0 ? q : 8 + q);
      const int expected_n_coord = expected_n_coords[token_group];
      if (m_coord != expected_m_coord || n_coord != expected_n_coord) {
        printf(
            "p5_native_direct_pack_oracle mapping mismatch thread=%d linear=%d actual=(%d,%d) expected=(%d,%d)\n",
            static_cast<int>(threadIdx.x),
            linear,
            m_coord,
            n_coord,
            expected_m_coord,
            expected_n_coord);
        asm("trap;");
      }
      if (n_coord < pass_valid_rows && m_coord < kTileCols) {
        const int abs_row = row_base + n_coord;
        if (abs_row < kTileRows) {
          accum_tensor(linear) =
              input[static_cast<std::size_t>(abs_row) * kTileCols + m_coord];
        }
      }
    }

    nvfp4_bridge::StoreUnifiedRoutedFp4DirectPack(
        1.0f,
        &accum_storage[0],
        threadIdx.x,
        0,
        pass_valid_rows,
        kTileCols,
        static_cast<std::size_t>(row_base),
        static_cast<std::size_t>(kTileCols),
        per_row_tensor_scales_input,
        1.0f,
        packed_data,
        block_scales_data,
        matmul_block_scales_data,
        activation_output_scale,
        static_cast<std::size_t>(kTileCols),
        padded_blocks_per_row,
        scale_layout);
    __syncthreads();
  }
}

bool RunP5NativeDirectPackOracleForTesting(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  if (input == nullptr ||
      activation_output_scale == nullptr ||
      packed_data == nullptr ||
      block_scales_data == nullptr ||
      matmul_block_scales_data == nullptr ||
      tensor_scale_data == nullptr ||
      per_row_tensor_scales == nullptr ||
      valid_rows < 0 ||
      valid_rows > 128) {
    return false;
  }

  P5NativeDirectPackOracleKernel<<<1, 256>>>(
      input,
      per_row_tensor_scales_input,
      valid_rows,
      padded_blocks_per_row,
      scale_layout,
      activation_output_scale,
      packed_data,
      block_scales_data,
      matmul_block_scales_data,
      tensor_scale_data,
      per_row_tensor_scales);
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize());
}

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  const std::size_t selection_count = params.token_count * params.top_k;
  DeviceNvfp4Matrix* shared_fc1_pack =
      params.shared_fc1_pack != nullptr ? params.shared_fc1_pack
                                        : const_cast<DeviceNvfp4Matrix*>(params.normalized_pack);
  DeviceNvfp4Matrix* shared_fc2_pack = params.shared_fc2_pack;
  const bool use_packed_fc1_source =
      params.normalized_pack != nullptr &&
      params.normalized_pack->valid() &&
      params.fc1_grouped_pack != nullptr &&
      params.fc1_grouped_pack->valid();
  const float* fc1_input_expert_scales =
      use_packed_fc1_source ? nullptr : params.fc1_expert_activation_scales;
  DeviceNvfp4Matrix* gemm1_output =
      params.gemm1_output != nullptr ? params.gemm1_output : params.fc2_grouped_pack;
  float* gemm1_output_scale =
      params.gemm1_output_scale != nullptr ? params.gemm1_output_scale
                                           : params.fc2_expert_activation_scales;
  float* activation_output_scale =
      params.activation_output_scale != nullptr ? params.activation_output_scale
                                                : gemm1_output_scale;
  const bool use_legacy_packed_fc1 =
      params.fc1_grouped_pack != nullptr && fc1_input_expert_scales != nullptr;
  const bool use_grouped_gemm1_output =
      gemm1_output != nullptr &&
      gemm1_output_scale != nullptr &&
      activation_output_scale != nullptr;
  const bool use_packed_fc2 =
      gemm1_output != nullptr && activation_output_scale != nullptr;

  if (params.token_count == 0 ||
      params.hidden_size == 0 ||
      params.hidden_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.routed_expert_intermediate_size == 0 ||
      params.routed_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.shared_expert_intermediate_size == 0 ||
      params.shared_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.n_routed_experts == 0 ||
      params.top_k == 0 ||
      params.top_k > params.n_routed_experts ||
      !ValidFusedNvfp4WeightView(params.shared_up) ||
      !ValidFusedNvfp4WeightView(params.shared_down) ||
      params.routed_up_device == nullptr ||
      params.routed_down_device == nullptr ||
      params.selected_indices == nullptr ||
      params.selected_weights == nullptr ||
      params.input == nullptr ||
      params.normalized == nullptr ||
      params.routing == nullptr ||
      params.launch_plan == nullptr ||
      params.routed_gather_scratch == nullptr ||
      (!use_grouped_gemm1_output && params.routed_up_scratch == nullptr) ||
      params.shared_up_scratch == nullptr ||
      params.output == nullptr) {
    return false;
  }

  if (!params.routing->valid() ||
      params.routing->n_experts() != params.n_routed_experts ||
      params.routing->selection_count() < selection_count ||
      !params.launch_plan->valid() ||
      params.launch_plan->n_experts() != params.n_routed_experts ||
      params.launch_plan->selection_count() < selection_count) {
    return false;
  }

  if (params.token_count >
          (std::numeric_limits<std::size_t>::max() / params.top_k) ||
      params.token_count >
          (std::numeric_limits<std::size_t>::max() / params.hidden_size) ||
      params.token_count >
          static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) ||
      params.shared_up.input_cols != params.hidden_size ||
      params.shared_up.output_rows != params.shared_expert_intermediate_size ||
      params.shared_down.input_cols != params.shared_expert_intermediate_size ||
      params.shared_down.output_rows != params.hidden_size ||
      (shared_fc1_pack != nullptr &&
       (!shared_fc1_pack->valid() ||
        shared_fc1_pack->rows() < params.token_count ||
        shared_fc1_pack->cols() != params.hidden_size)) ||
      (shared_fc2_pack != nullptr &&
       (!shared_fc2_pack->valid() ||
        shared_fc2_pack->rows() < params.token_count ||
        shared_fc2_pack->cols() != params.shared_expert_intermediate_size))) {
    return false;
  }

  const std::size_t token_hidden_count = params.token_count * params.hidden_size;
  const std::size_t shared_intermediate_count =
      params.token_count * params.shared_expert_intermediate_size;
  const auto current_padded_row_capacity =
      DeviceMoeLaunchPlan::PaddedRowCapacity(params.n_routed_experts, selection_count);
  if (!current_padded_row_capacity.has_value()) {
    return false;
  }
  if (!RunDeviceExpertRouting(
          params.selected_indices,
          params.selected_weights,
          params.token_count,
          params.top_k,
          params.routing) ||
      !BuildDeviceMoeLaunchPlan(*params.routing, selection_count, params.launch_plan) ||
      !LaunchZeroBuffer(params.output, token_hidden_count) ||
      !LaunchZeroBuffer(params.routed_output, token_hidden_count) ||
      !LaunchZeroBuffer(params.shared_output, token_hidden_count)) {
    return false;
  }

  const std::size_t packed_dispatch_rows =
      RoutedEnvEnabled("NEMOTRON_DEBUG_USE_SELECTED_TOKEN_TILE_FOR_DISPATCH")
          ? params.launch_plan->selected_token_tile()
          : params.token_count;
  const RoutedGemm1Profile grouped_gemm1_profile =
      SelectRoutedGemm1Profile(packed_dispatch_rows);
  const bool use_fp4_direct_fc1 =
      use_grouped_gemm1_output &&
      use_packed_fc1_source &&
      (grouped_gemm1_profile == RoutedGemm1Profile::kP1_128x128x64_SwapFalse ||
       grouped_gemm1_profile == RoutedGemm1Profile::kP5_128x128x64_SwapTrue);

  if ((!use_packed_fc1_source &&
       !LaunchGatherRows(
           params.normalized,
           params.launch_plan->permuted_idx_to_token_idx(),
           params.launch_plan->total_num_padded_tokens(),
           params.launch_plan->padded_row_capacity(),
           params.token_count,
           params.hidden_size,
           params.routed_gather_scratch)) ||
      (use_packed_fc1_source &&
       !GatherDeviceNvfp4Rows(
           *params.normalized_pack,
           params.launch_plan->permuted_idx_to_token_idx(),
           params.launch_plan->padded_row_capacity(),
           params.fc1_grouped_pack)) ||
      (use_legacy_packed_fc1 && !use_grouped_gemm1_output &&
       (!LaunchComputeExpertActivationScales(
            params.routed_gather_scratch,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            params.hidden_size,
            const_cast<float*>(fc1_input_expert_scales)) ||
        !PackDeviceRowMajorFp32ToNvfp4PerExpert(
            params.routed_gather_scratch,
            params.launch_plan->padded_row_capacity(),
            params.hidden_size,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            fc1_input_expert_scales,
            params.fc1_grouped_pack))) ||
      (!use_legacy_packed_fc1 && !use_grouped_gemm1_output &&
       !LaunchQuantizeDequantizeRows(
           params.routed_gather_scratch,
           params.launch_plan->total_num_padded_tokens(),
           params.launch_plan->padded_row_capacity(),
           params.hidden_size))) {
    return false;
  }

  if (use_grouped_gemm1_output) {
    if (use_fp4_direct_fc1) {
      __nv_bfloat16* fp4_direct_compare_workspace =
          RoutedEnvEnabled("NEMOTRON_DEBUG_COMPARE_FP4_DIRECT")
              ? params.gemm1_output_bf16
              : nullptr;
      if (!LaunchPlannedPackedInputMatVecFp4Direct(
              *params.fc1_grouped_pack,
              nullptr,
              nullptr,
              params.fc1_grouped_pack->per_row_tensor_scales(),
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_up_device,
              params.routed_expert_intermediate_size,
              gemm1_output,
              activation_output_scale,
              fp4_direct_compare_workspace) ||
          !MaybeCompareFirstFp4DirectFc1Output(
              *params.fc1_grouped_pack,
              nullptr,
              nullptr,
              params.fc1_grouped_pack->per_row_tensor_scales(),
              params.routing,
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_up_device,
              params.routed_expert_intermediate_size,
              *gemm1_output,
              activation_output_scale,
              params.gemm1_output_bf16) ||
          !LaunchZeroBuffer(
              params.routed_gather_scratch,
              params.launch_plan->padded_row_capacity() * params.hidden_size) ||
          !LaunchPlannedPackedInputMatVec(
              *gemm1_output,
              nullptr,
              activation_output_scale,
              nullptr,
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_down_device,
              params.hidden_size,
              params.routed_gather_scratch)) {
        return false;
      }
    } else {
      if (params.gemm1_output_bf16 == nullptr ||
          !LaunchZeroBf16Buffer(
              params.gemm1_output_bf16,
              params.launch_plan->padded_row_capacity() *
                  params.routed_expert_intermediate_size) ||
          !(use_packed_fc1_source
                ? LaunchPlannedPackedInputMatVecBf16(
                      *params.fc1_grouped_pack,
                      nullptr,
                      nullptr,
                      params.fc1_grouped_pack->per_row_tensor_scales(),
                      params.launch_plan,
                      packed_dispatch_rows,
                      selection_count,
                      params.routed_up_device,
                      params.routed_expert_intermediate_size,
                      params.gemm1_output_bf16)
                : RunLaunchPlannedNvfp4ExpertMatVecBf16(
                      params.routed_gather_scratch,
                      params.launch_plan,
                      selection_count,
                      params.routed_up_device,
                      params.routed_expert_intermediate_size,
                      params.gemm1_output_bf16)) ||
          !LaunchRoutedBf16Relu2Pack(
              params.gemm1_output_bf16,
              params.routing,
              params.launch_plan,
              selection_count,
              gemm1_output,
              activation_output_scale) ||
          (use_packed_fc1_source &&
           !MaybeCompareFirstGroupedBf16Fc1Output(
               params.normalized,
               params.routing,
               params.launch_plan,
               params.token_count,
               selection_count,
               params.hidden_size,
               params.routed_gather_scratch,
               params.routed_up_device,
               params.routed_expert_intermediate_size,
               *gemm1_output,
               activation_output_scale,
               params.gemm1_output_bf16)) ||
          !LaunchZeroBuffer(
              params.routed_gather_scratch,
              params.launch_plan->padded_row_capacity() * params.hidden_size) ||
          !LaunchPlannedPackedInputMatVec(
              *gemm1_output,
              nullptr,
              activation_output_scale,
              nullptr,
              params.launch_plan,
              packed_dispatch_rows,
              selection_count,
              params.routed_down_device,
              params.hidden_size,
              params.routed_gather_scratch)) {
        return false;
      }
    }
  } else {
    if (!LaunchZeroBuffer(
            params.routed_up_scratch,
            *current_padded_row_capacity * params.routed_expert_intermediate_size) ||
        !(use_legacy_packed_fc1
              ? LaunchPlannedPackedInputMatVec(
                    *params.fc1_grouped_pack,
                    fc1_input_expert_scales,
                    nullptr,
                    params.fc1_grouped_pack->per_row_tensor_scales(),
                    params.launch_plan,
                    packed_dispatch_rows,
                    selection_count,
                    params.routed_up_device,
                    params.routed_expert_intermediate_size,
                    params.routed_up_scratch)
              : RunLaunchPlannedNvfp4ExpertMatVec(
                    params.routed_gather_scratch,
                    params.launch_plan,
                    selection_count,
                    params.routed_up_device,
                    params.routed_expert_intermediate_size,
                    params.routed_up_scratch)) ||
        !LaunchRelu2Rows(
            params.routed_up_scratch,
            params.launch_plan->total_num_padded_tokens(),
            params.launch_plan->padded_row_capacity(),
            params.routed_expert_intermediate_size) ||
        (activation_output_scale != nullptr &&
         !LaunchComputeExpertActivationScales(
             params.routed_up_scratch,
             params.launch_plan->expert_first_token_offsets(),
             params.n_routed_experts,
             params.routed_expert_intermediate_size,
             activation_output_scale)) ||
        (gemm1_output != nullptr &&
         !PackDeviceRowMajorFp32ToNvfp4PerExpert(
             params.routed_up_scratch,
             params.launch_plan->padded_row_capacity(),
             params.routed_expert_intermediate_size,
             params.launch_plan->expert_first_token_offsets(),
             params.n_routed_experts,
             activation_output_scale,
             gemm1_output)) ||
        (gemm1_output == nullptr &&
         !LaunchQuantizeDequantizeRows(
             params.routed_up_scratch,
             params.launch_plan->total_num_padded_tokens(),
             params.launch_plan->padded_row_capacity(),
             params.routed_expert_intermediate_size))) {
      return false;
    }

    if (!(use_packed_fc2
              ? LaunchPlannedPackedInputMatVec(
                    *gemm1_output,
                    params.fc2_expert_activation_scales,
                    nullptr,
                    nullptr,
                    params.launch_plan,
                    packed_dispatch_rows,
                    selection_count,
                    params.routed_down_device,
                    params.hidden_size,
                    params.routed_gather_scratch)
              : RunLaunchPlannedNvfp4ExpertMatVec(
                    params.routed_up_scratch,
                    params.launch_plan,
                    selection_count,
                    params.routed_down_device,
                    params.hidden_size,
                    params.routed_gather_scratch))) {
      return false;
    }
  }

  if (!LaunchReduceSelectionOutputs(
          params.routed_gather_scratch,
          params.routing->selection_to_sorted(),
          params.launch_plan->sorted_to_permuted_indices(),
          params.selected_weights,
          params.token_count,
          params.top_k,
          params.hidden_size,
          params.output,
          params.routed_output)) {
    return false;
  }

  const bool should_prepare_shared_up_plan =
      shared_fc1_pack != nullptr && shared_fc1_pack->valid();
  const bool should_prepare_shared_down_plan =
      shared_fc2_pack != nullptr && shared_fc2_pack->valid();
  const auto shared_up_prepared_launch =
      should_prepare_shared_up_plan
          ? PrepareSharedContiguousLaunch(
                "shared_up_prefill",
                *shared_fc1_pack,
                params.token_count,
                params.shared_up,
                params.heuristic_cache)
          : std::nullopt;
  const auto shared_down_prepared_launch =
      should_prepare_shared_down_plan
          ? PrepareSharedContiguousLaunch(
                "shared_down_prefill",
                *shared_fc2_pack,
                params.token_count,
                params.shared_down,
                params.heuristic_cache)
          : std::nullopt;
  if (should_prepare_shared_up_plan) {
    AppendSharedContiguousTraceEntry(
        "shared_up_prefill",
        shared_up_prepared_launch.has_value(),
        shared_up_prepared_launch.has_value());
  }
  if (should_prepare_shared_down_plan) {
    AppendSharedContiguousTraceEntry(
        "shared_down_prefill",
        shared_down_prepared_launch.has_value(),
        shared_down_prepared_launch.has_value());
  }

  const bool use_shared_fp4_path =
      shared_up_prepared_launch.has_value() && shared_down_prepared_launch.has_value();
  if (use_shared_fp4_path) {
    Nvfp4PackOptions shared_pack_options;
    shared_pack_options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
    auto shared_fc1_source = DeviceTensorFp32::CreateView(
        {params.token_count, params.hidden_size},
        const_cast<float*>(params.normalized));
    auto shared_fc2_source = DeviceTensorFp32::CreateView(
        {params.token_count, params.shared_expert_intermediate_size},
        params.shared_up_scratch);
    if (shared_fc1_source == nullptr ||
        shared_fc2_source == nullptr ||
        !shared_fc1_pack->PackIntoPerRow(*shared_fc1_source, shared_pack_options) ||
        !LaunchContiguousFp4MatVec(
            *shared_fc1_pack,
            *shared_up_prepared_launch,
            params.shared_up,
            params.shared_up_scratch) ||
        !LaunchRelu2(params.shared_up_scratch, shared_intermediate_count) ||
        !shared_fc2_pack->PackIntoPerRow(*shared_fc2_source, shared_pack_options) ||
        !LaunchContiguousFp4MatVec(
            *shared_fc2_pack,
            *shared_down_prepared_launch,
            params.shared_down,
            params.routed_gather_scratch) ||
        !LaunchAccumulateSharedOutput(
            params.routed_gather_scratch,
            token_hidden_count,
            params.output,
            params.shared_output)) {
      return false;
    }
  } else if (!CheckCuda(cudaMemcpyAsync(
                 params.routed_gather_scratch,
                 params.normalized,
                 token_hidden_count * sizeof(float),
                 cudaMemcpyDeviceToDevice)) ||
             !LaunchQuantizeDequantizeRows(
                 params.routed_gather_scratch,
                 nullptr,
                 params.token_count,
                 params.hidden_size) ||
             !LaunchContiguousMatVec(
                 params.routed_gather_scratch,
                 params.token_count,
                 params.shared_up,
                 params.shared_up_scratch) ||
             !LaunchRelu2(params.shared_up_scratch, shared_intermediate_count) ||
             !LaunchQuantizeDequantizeRows(
                 params.shared_up_scratch,
                 nullptr,
                 params.token_count,
                 params.shared_expert_intermediate_size) ||
             !LaunchContiguousMatVec(
                 params.shared_up_scratch,
                 params.token_count,
                 params.shared_down,
                 params.routed_gather_scratch) ||
             !LaunchAccumulateSharedOutput(
                 params.routed_gather_scratch,
                 token_hidden_count,
                 params.output,
                 params.shared_output)) {
    return false;
  }

  return CheckCuda(cudaGetLastError());
}

