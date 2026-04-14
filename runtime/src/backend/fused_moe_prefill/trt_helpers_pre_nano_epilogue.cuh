constexpr int kGroupedTokenTile = static_cast<int>(kMoeLaunchPlanTokenTile);
// Match TRT-LLM's grouped routed shape more closely: tileTokensDim=16 and an
// epilogue/output tile of 128 rows per CTA when transposeMmaOutput=true.
constexpr int kPlannedOutputTile = 128;
constexpr int kPlannedThreadsPerBlock = 256;
constexpr int kPlannedWmmaTileM = 16;
constexpr int kPlannedWmmaTileN = 16;
constexpr int kPlannedWmmaTileK = 16;
constexpr int kPlannedWmmaWarpsPerBlock = 8;
// The traced TRT routed grouped kernels launch with BlockX=384 on SM120.
// Keep eight consumer warps for the 128-row WMMA tile and add four extra
// staging warps so the grouped decode/stage path is closer to TRT's larger
// warp-specialized CTA shape.
constexpr int kGroupedThreadsPerBlock = 384;
constexpr int kGroupedConsumerWarpsPerBlock = 8;
constexpr int kGroupedProducerWarpsPerBlock =
    (kGroupedThreadsPerBlock / 32) - kGroupedConsumerWarpsPerBlock;
constexpr int kFp4MmaTileN = 8;
constexpr int kRoutedLargeOutputTile = 256;
constexpr int kRoutedThreadsPerBlock = kGroupedThreadsPerBlock;
constexpr int kContiguousSmallOutputTile = 32;
constexpr int kContiguousMediumOutputTile = 64;
constexpr int kContiguousLargeOutputTile = 128;
constexpr int kContiguousSmallThreadsPerBlock = 64;
constexpr int kContiguousMediumThreadsPerBlock = 128;
constexpr int kContiguousLargeThreadsPerBlock = 256;
constexpr float kNvfp4ActivationMaxFinite = 6.0f * 448.0f;
constexpr float kNvfp4MinTensorScale = 1.0f / 1024.0f;
constexpr std::size_t kNvfp4ScaleBlockTile = 4u;
constexpr int kSharedContiguousP5ConsumerWarps =
    cute::size(nvfp4_bridge::TracedP5TiledMma{}) / 32;
constexpr int kSharedContiguousP5ThreadsPerBlock =
    (kSharedContiguousP5ConsumerWarps + 1) * 32;

static_assert(kGroupedTokenTile == static_cast<int>(kMoeLaunchPlanTokenTile));



enum class RoutedGemm1Profile {
  kLegacy,
  kP0_128x128x128_SwapFalse,
  kP1_128x128x64_SwapFalse,
  kP4_128x128x128_SwapTrue,
  kP5_128x128x64_SwapTrue,
  kP7_256x128x64_SwapTrue,
};

enum class RoutedGemm2Profile {
  kLegacy,
  kP12_128x128x128_SwapTrue,
  kP13_128x128x64_SwapTrue,
  kP15_256x128x64_SwapTrue,
};

struct SharedContiguousPreparedLaunch {
  GemmLaunchPlan launch_plan;
  PreparedGemmExecution execution;
  const void* p5_tma_load_b_descriptors = nullptr;
  const void* p5_tma_load_sfb_descriptors = nullptr;
};

struct P15ScaleTrace {
  int valid = 0;
  int row_start = -1;
  int valid_rows = -1;
  int output_row_base = -1;
  int a_base_rows[4] = {-1, -1, -1, -1};
  int b_base_rows[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  std::uint32_t a_source_words[4] = {0, 0, 0, 0};
  std::uint32_t b_source_words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::uint32_t a_fragment_words[4] = {0, 0, 0, 0};
  std::uint32_t b_fragment_words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

struct P5ScaleTrace {
  int valid = 0;
  int row_start = -1;
  int valid_rows = -1;
  int output_row_base = -1;
  int a_base_rows[4] = {-1, -1, -1, -1};
  int b_base_rows[2] = {-1, -1};
  std::uint32_t a_source_words[4] = {0, 0, 0, 0};
  std::uint32_t b_source_words[2] = {0, 0};
  std::uint32_t a_fragment_words[4] = {0, 0, 0, 0};
  std::uint32_t b_fragment_words[2] = {0, 0};
};

struct P13ScaleTrace {
  int valid = 0;
  int row_start = -1;
  int valid_rows = -1;
  int output_row_base = -1;
  int a_base_rows[2] = {-1, -1};
  int b_base_rows[2] = {-1, -1};
  std::uint32_t a_source_words[2] = {0, 0};
  std::uint32_t b_source_words[2] = {0, 0};
  std::uint32_t a_smem_words[2] = {0, 0};
  std::uint32_t b_smem_words[2] = {0, 0};
  std::uint32_t a_fragment_words[2] = {0, 0};
  std::uint32_t b_fragment_words[2] = {0, 0};
  std::uint32_t a_loaded_words[2] = {0, 0};
  std::uint32_t b_loaded_words[2] = {0, 0};
  int a_scale_rows[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int a_scale_cols[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int b_scale_rows[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int b_scale_cols[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  std::uint8_t a_scale_raw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t b_scale_raw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t a_post_store_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t b_post_store_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t a_logical_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::uint8_t b_logical_raw[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct P1DirectTrace {
  int valid = 0;
  int warp_ids[2] = {0, 0};
  int n_base[2] = {0, 0};
  std::uint32_t a_scale[2] = {0u, 0u};
  std::uint32_t a_regs[2][4] = {};
  std::uint32_t b_scale[2] = {0u, 0u};
  std::uint32_t b_regs[2][2] = {};
};

__device__ __managed__ P15ScaleTrace g_p15_scale_trace;
__device__ __managed__ P5ScaleTrace g_p5_scale_trace;
__device__ __managed__ P13ScaleTrace g_p13_scale_trace;
__device__ __managed__ P1DirectTrace g_p1_direct_trace;
__device__ __managed__ P13DebugTrace g_p13_debug_trace;
__device__ __managed__ int g_enable_p15_scale_trace = 0;
__device__ __managed__ int g_enable_p5_scale_trace = 0;
__device__ __managed__ int g_enable_p13_scale_trace = 0;
__device__ __managed__ int g_enable_p13_scale_map_probe = 0;
__device__ __managed__ int g_enable_p13_debug_trace = 0;
__device__ __managed__ int g_enable_p1_direct_trace = 0;
__device__ __managed__ int g_p13_debug_trace_target_valid_rows = -1;
__device__ __managed__ int g_p13_debug_trace_claimed = 0;
__device__ __managed__ int g_p13_debug_trace_target_thread_id = -1;

template <class ScaleAtomTensor>
__device__ std::uint32_t PackP15ScaleFragmentWord(ScaleAtomTensor const& scale_atom) {
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

template <class ScaleAtomTensor>
__device__ void FillP15ScaleFragmentWord(ScaleAtomTensor const& scale_atom, std::uint32_t packed_scale_word) {
  constexpr int kGroupSize = 16;
  static_assert(cute::size(ScaleAtomTensor{}) == 64);
#pragma unroll
  for (int byte_index = 0; byte_index < 4; ++byte_index) {
    const std::uint8_t value =
        static_cast<std::uint8_t>((packed_scale_word >> (byte_index * 8)) & 0xFFu);
    scale_atom(byte_index * kGroupSize) = nvfp4_bridge::MakeScaleElement(value);
  }
}

void PrintP15ScaleTrace() {
  const auto& trace = g_p15_scale_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p15_scale_trace: invalid\n");
    return;
  }
  std::fprintf(
      stderr,
      "p15_scale_trace: row_start=%d valid_rows=%d output_row_base=%d\n",
      trace.row_start,
      trace.valid_rows,
      trace.output_row_base);
  for (int i = 0; i < 4; ++i) {
    std::fprintf(
        stderr,
        "p15_scale_trace: A m=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.a_base_rows[i],
        trace.a_source_words[i],
        trace.a_fragment_words[i]);
  }
  for (int i = 0; i < 8; ++i) {
    std::fprintf(
        stderr,
        "p15_scale_trace: B n=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.b_base_rows[i],
        trace.b_source_words[i],
        trace.b_fragment_words[i]);
  }
}

void PrintP5ScaleTrace() {
  const auto& trace = g_p5_scale_trace;
  if (trace.valid == 0) {
    std::fprintf(stderr, "p5_scale_trace: invalid\n");
    return;
  }
  std::fprintf(
      stderr,
      "p5_scale_trace: row_start=%d valid_rows=%d output_row_base=%d\n",
      trace.row_start,
      trace.valid_rows,
      trace.output_row_base);
  for (int i = 0; i < 4; ++i) {
    std::fprintf(
        stderr,
        "p5_scale_trace: A m=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.a_base_rows[i],
        trace.a_source_words[i],
        trace.a_fragment_words[i]);
  }
  for (int i = 0; i < 2; ++i) {
    std::fprintf(
        stderr,
        "p5_scale_trace: B n=%d base_row=%d source=0x%08x fragment=0x%08x\n",
        i,
        trace.b_base_rows[i],
        trace.b_source_words[i],
        trace.b_fragment_words[i]);
  }
}

bool CheckCuda(cudaError_t status) {
  if (status != cudaSuccess && std::getenv("NEMOTRON_ROUTED_PROFILE_DEBUG") != nullptr) {
    std::fprintf(stderr, "fused_moe_prefill cuda error: %s\n", cudaGetErrorString(status));
  }
  return status == cudaSuccess;
}

bool RoutedEnvEnabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool CopyDeviceFloatBufferToHost(
    const float* source,
    std::size_t count,
    std::vector<float>* host_output) {
  if (source == nullptr || host_output == nullptr) {
    return false;
  }
  host_output->assign(count, 0.0f);
  if (count == 0) {
    return true;
  }
  return CheckCuda(cudaMemcpy(
      host_output->data(),
      source,
      count * sizeof(float),
      cudaMemcpyDeviceToHost));
}

bool CopyDeviceBf16BufferToHost(
    const __nv_bfloat16* source,
    std::size_t count,
    std::vector<float>* host_output) {
  if (source == nullptr || host_output == nullptr) {
    return false;
  }
  std::vector<__nv_bfloat16> host_bf16(count);
  if (count != 0 &&
      !CheckCuda(cudaMemcpy(
          host_bf16.data(),
          source,
          count * sizeof(__nv_bfloat16),
          cudaMemcpyDeviceToHost))) {
    return false;
  }
  host_output->resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    (*host_output)[index] = __bfloat162float(host_bf16[index]);
  }
  return true;
}

void PrintPackedDataMismatches(
    const char* compare_label,
    const std::vector<std::uint8_t>& reference,
    const std::vector<std::uint8_t>& direct,
    std::size_t rows,
    std::size_t cols,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  const std::size_t packed_row_bytes = cols / 2u;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t pair = 0; pair < packed_row_bytes; ++pair) {
      const std::size_t index = row * packed_row_bytes + pair;
      if (reference[index] == direct[index]) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s packed_data mismatch[%zu]: row=%zu col=%zu reference=0x%02x direct=0x%02x\n",
            compare_label,
            *mismatch_count,
            row,
            pair * 2u,
            static_cast<unsigned int>(reference[index]),
            static_cast<unsigned int>(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintPackedScaleMismatches(
    const char* compare_label,
    const char* label,
    const std::vector<std::uint8_t>& reference,
    const std::vector<std::uint8_t>& direct,
    std::size_t rows,
    std::size_t blocks_per_row,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t index = row * blocks_per_row + block;
      if (reference[index] == direct[index]) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s %s mismatch[%zu]: row=%zu col=%zu reference=0x%02x direct=0x%02x\n",
            compare_label,
            label,
            *mismatch_count,
            row,
            block * fused_decode::kNvfp4BlockWidth,
            static_cast<unsigned int>(reference[index]),
            static_cast<unsigned int>(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintMatmulScaleMismatches(
    const char* compare_label,
    const std::vector<std::uint8_t>& reference,
    const std::vector<std::uint8_t>& direct,
    std::size_t rows,
    std::size_t blocks_per_row,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t index =
          ExecutionScaleOffset(row, block, padded_blocks_per_row, scale_layout);
      if (reference[index] == direct[index]) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s matmul_block_scales_data mismatch[%zu]: row=%zu col=%zu reference=0x%02x direct=0x%02x\n",
            compare_label,
            *mismatch_count,
            row,
            block * fused_decode::kNvfp4BlockWidth,
            static_cast<unsigned int>(reference[index]),
            static_cast<unsigned int>(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintActivationScaleMismatches(
    const char* compare_label,
    const std::vector<float>& reference,
    const std::vector<float>& direct,
    std::size_t rows,
    std::size_t blocks_per_row,
    std::size_t limit,
    std::size_t* mismatch_count) {
  *mismatch_count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t index = row * blocks_per_row + block;
      if (FloatBits(reference[index]) == FloatBits(direct[index])) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s activation_output_scale mismatch[%zu]: row=%zu col=%zu reference=%g (0x%08x) direct=%g (0x%08x)\n",
            compare_label,
            *mismatch_count,
            row,
            block * fused_decode::kNvfp4BlockWidth,
            static_cast<double>(reference[index]),
            FloatBits(reference[index]),
            static_cast<double>(direct[index]),
            FloatBits(direct[index]));
      }
      ++(*mismatch_count);
    }
  }
}

void PrintDenseFloatMismatches(
    const char* compare_label,
    const char* label,
    const std::vector<float>& reference,
    const std::vector<float>& direct,
    std::size_t rows,
    std::size_t cols,
    float tolerance,
    std::size_t limit,
    std::size_t* mismatch_count,
    float* max_abs_diff) {
  *mismatch_count = 0;
  *max_abs_diff = 0.0f;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const std::size_t index = row * cols + col;
      const float diff = fabsf(reference[index] - direct[index]);
      *max_abs_diff = fmaxf(*max_abs_diff, diff);
      if (diff <= tolerance) {
        continue;
      }
      if (*mismatch_count < limit) {
        std::fprintf(
            stderr,
            "%s %s mismatch[%zu]: row=%zu col=%zu reference=%g direct=%g diff=%g\n",
            compare_label,
            label,
            *mismatch_count,
            row,
            col,
            static_cast<double>(reference[index]),
            static_cast<double>(direct[index]),
            static_cast<double>(diff));
      }
      ++(*mismatch_count);
    }
  }
}

// Host fp64 NVFP4 GEMM reference for the first FC1 dispatch.  Independent of
// any kernel — used as ground truth in MaybeCompareFirstFp4DirectFc1Output to
// detect bugs in either the BF16 reference path or the direct FP4 kernel
// without trusting either of them as a baseline.
//
// Computes prepack[input_row, output_col] = (sum_k a*b) * a_tscale * b_tscale,
// where the dot product is in fp64 and the input row → expert mapping comes
// from the launch plan's cta_idx_xy_to_batch_idx + cta_row_starts/valid_rows.
//
// Returns false if the host reference cannot be computed (missing pointers,
// shape mismatch, etc.).  Caller is expected to ensure scales are configured
// in one of the supported ways: per-row tensor scales, per-expert tensor
// scales, or a single global tensor scale (all FP4-direct dispatches use one
// of these).
bool ComputeFp4DirectHostReference(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales_dev,
    const float* input_per_row_tensor_scales_dev,
    const DeviceMoeLaunchPlan* launch_plan,
    const FusedNvfp4WeightView* weights_dev,
    std::size_t output_rows_per_expert,
    std::vector<float>* host_reference) {
  if (host_reference == nullptr ||
      !input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      weights_dev == nullptr) {
    return false;
  }

  const std::size_t input_rows = input_pack.rows();
  const std::size_t hidden_size = input_pack.cols();
  if (hidden_size == 0 || (hidden_size % 16u) != 0 ||
      output_rows_per_expert == 0 || (output_rows_per_expert % 16u) != 0) {
    return false;
  }
  const std::size_t blocks_per_row = hidden_size / 16u;
  const std::size_t row_packed_bytes = hidden_size / 2u;

  // Snapshot the input pack on the host.
  std::vector<std::uint8_t> input_packed_host;
  std::vector<std::uint8_t> input_scales_host;
  if (!input_pack.CopyPackedToHost(&input_packed_host) ||
      !input_pack.CopyBlockScalesToHost(&input_scales_host)) {
    return false;
  }
  if (input_packed_host.size() < input_rows * row_packed_bytes ||
      input_scales_host.size() < input_rows * blocks_per_row) {
    return false;
  }

  // Resolve per-row input tensor scales.
  std::vector<float> input_row_tscales(input_rows, 1.0f);
  if (input_per_row_tensor_scales_dev != nullptr) {
    if (cudaMemcpy(
            input_row_tscales.data(),
            input_per_row_tensor_scales_dev,
            input_rows * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
  } else {
    float global_scale = 0.0f;
    if (!input_pack.CopyTensorScaleToHost(&global_scale)) {
      return false;
    }
    if (global_scale == 0.0f) {
      global_scale = 1.0f;
    }
    std::fill(input_row_tscales.begin(), input_row_tscales.end(), global_scale);
  }

  // Resolve the active CTA → expert mapping and the input-row partitioning.
  const int exact_cta_count = launch_plan->exact_cta_count_host();
  if (exact_cta_count <= 0) {
    return false;
  }
  std::vector<int> cta_batch_host(static_cast<std::size_t>(exact_cta_count), -1);
  if (cudaMemcpy(
          cta_batch_host.data(),
          launch_plan->cta_idx_xy_to_batch_idx(),
          static_cast<std::size_t>(exact_cta_count) * sizeof(int),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    return false;
  }
  const int* cta_row_starts_host = launch_plan->cta_row_starts_host();
  const int* cta_valid_rows_host = launch_plan->cta_valid_rows_host();
  if (cta_row_starts_host == nullptr || cta_valid_rows_host == nullptr) {
    return false;
  }

  // For every input row covered by some CTA, record which expert it belongs
  // to.  Rows not covered by any CTA are left at expert -1 and the host
  // reference leaves their output at zero (matching the kernel's
  // LaunchZeroBf16Buffer behavior on the prepack workspace).
  std::vector<int> row_to_expert(input_rows, -1);
  for (int cta = 0; cta < exact_cta_count; ++cta) {
    const int expert = cta_batch_host[cta];
    const int row_start = cta_row_starts_host[cta];
    const int valid_rows = cta_valid_rows_host[cta];
    if (expert < 0 || row_start < 0 || valid_rows <= 0) {
      continue;
    }
    for (int r = 0; r < valid_rows; ++r) {
      const std::size_t idx = static_cast<std::size_t>(row_start + r);
      if (idx < input_rows) {
        row_to_expert[idx] = expert;
      }
    }
  }

  // Snapshot every distinct expert weight that we actually need.  Caches by
  // expert id so we only copy each expert's weight from device once.
  std::vector<bool> expert_loaded;
  std::vector<FusedNvfp4WeightView> expert_view_host;
  std::vector<std::vector<std::uint8_t>> expert_packed_host;
  std::vector<std::vector<std::uint8_t>> expert_scales_host;
  std::vector<float> expert_tensor_scale_host;
  const std::size_t weight_packed_bytes = (output_rows_per_expert * hidden_size) / 2u;
  const std::size_t weight_scales_bytes = output_rows_per_expert * blocks_per_row;

  auto load_expert = [&](int expert_id) -> bool {
    if (expert_id < 0) {
      return false;
    }
    if (static_cast<std::size_t>(expert_id) >= expert_loaded.size()) {
      const std::size_t new_size = static_cast<std::size_t>(expert_id) + 1u;
      expert_loaded.resize(new_size, false);
      expert_view_host.resize(new_size);
      expert_packed_host.resize(new_size);
      expert_scales_host.resize(new_size);
      expert_tensor_scale_host.resize(new_size, 0.0f);
    }
    if (expert_loaded[expert_id]) {
      return true;
    }
    FusedNvfp4WeightView view{};
    if (cudaMemcpy(
            &view,
            &weights_dev[expert_id],
            sizeof(FusedNvfp4WeightView),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    if (view.packed_data == nullptr ||
        view.matmul_block_scales_data == nullptr ||
        view.tensor_scale_data == nullptr ||
        view.input_cols != hidden_size ||
        view.output_rows != output_rows_per_expert) {
      return false;
    }
    expert_view_host[expert_id] = view;
    expert_packed_host[expert_id].assign(weight_packed_bytes, 0u);
    // Decode the weight's swizzled scales into a row-major host buffer to
    // simplify the dot-product loop.  Production weights only populate the
    // matmul (swizzled) layout, so we go through ExecutionScaleOffset to
    // un-swizzle.
    const std::size_t weight_padded_blocks_per_row =
        RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
    const std::size_t weight_swizzled_bytes =
        ((output_rows_per_expert + 127u) / 128u) * 128u *
        weight_padded_blocks_per_row;
    std::vector<std::uint8_t> swizzled_scratch(weight_swizzled_bytes, 0u);
    if (cudaMemcpy(
            expert_packed_host[expert_id].data(),
            view.packed_data,
            weight_packed_bytes,
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            swizzled_scratch.data(),
            view.matmul_block_scales_data,
            weight_swizzled_bytes,
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            &expert_tensor_scale_host[expert_id],
            view.tensor_scale_data,
            sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    expert_scales_host[expert_id].assign(weight_scales_bytes, 0u);
    for (std::size_t row = 0; row < output_rows_per_expert; ++row) {
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const std::size_t swizzled_offset = ExecutionScaleOffset(
            row, block, weight_padded_blocks_per_row,
            Nvfp4ScaleLayout::kSwizzled128x4);
        expert_scales_host[expert_id][row * blocks_per_row + block] =
            swizzled_scratch[swizzled_offset];
      }
    }
    expert_loaded[expert_id] = true;
    return true;
  };

  for (std::size_t r = 0; r < input_rows; ++r) {
    const int expert = row_to_expert[r];
    if (expert >= 0 && !load_expert(expert)) {
      return false;
    }
  }

  // If we have per-expert input tensor scales, copy them to host and apply
  // them per row according to the row→expert mapping.
  if (input_expert_tensor_scales_dev != nullptr) {
    // Copy enough entries to cover the highest expert we touched.
    int max_expert = -1;
    for (int e : row_to_expert) {
      if (e > max_expert) {
        max_expert = e;
      }
    }
    if (max_expert >= 0) {
      std::vector<float> expert_input_tscales(static_cast<std::size_t>(max_expert + 1), 1.0f);
      if (cudaMemcpy(
              expert_input_tscales.data(),
              input_expert_tensor_scales_dev,
              expert_input_tscales.size() * sizeof(float),
              cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
      for (std::size_t r = 0; r < input_rows; ++r) {
        const int expert = row_to_expert[r];
        if (expert >= 0) {
          input_row_tscales[r] = expert_input_tscales[expert];
        }
      }
    }
  }

  // Compute the dot products in fp64.
  host_reference->assign(input_rows * output_rows_per_expert, 0.0f);

  auto decode_fp4 = [](std::uint8_t nibble) -> float {
    __nv_fp4_e2m1 v;
    v.__x = nibble & 0x0Fu;
    return static_cast<float>(v);
  };
  auto decode_fp8 = [](std::uint8_t byte) -> float {
    __nv_fp8_e4m3 v;
    v.__x = byte;
    return static_cast<float>(v);
  };

  for (std::size_t row = 0; row < input_rows; ++row) {
    const int expert = row_to_expert[row];
    if (expert < 0) {
      continue;
    }
    const float a_tscale = input_row_tscales[row];
    const float b_tscale = expert_tensor_scale_host[expert];
    const std::uint8_t* input_row_packed =
        input_packed_host.data() + row * row_packed_bytes;
    const std::uint8_t* input_row_scales =
        input_scales_host.data() + row * blocks_per_row;
    const std::uint8_t* expert_packed = expert_packed_host[expert].data();
    const std::uint8_t* expert_scales = expert_scales_host[expert].data();

    for (std::size_t n = 0; n < output_rows_per_expert; ++n) {
      const std::uint8_t* weight_row_packed = expert_packed + n * row_packed_bytes;
      const std::uint8_t* weight_row_scales = expert_scales + n * blocks_per_row;
      double acc = 0.0;
      for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const float ab_block_scale =
            decode_fp8(input_row_scales[block]) *
            decode_fp8(weight_row_scales[block]);
        double block_acc = 0.0;
        const std::size_t k_byte_start = block * 8u;
        for (std::size_t kb = 0; kb < 8u; ++kb) {
          const std::uint8_t a_byte = input_row_packed[k_byte_start + kb];
          const std::uint8_t b_byte = weight_row_packed[k_byte_start + kb];
          const float a_lo = decode_fp4(a_byte & 0x0Fu);
          const float a_hi = decode_fp4((a_byte >> 4) & 0x0Fu);
          const float b_lo = decode_fp4(b_byte & 0x0Fu);
          const float b_hi = decode_fp4((b_byte >> 4) & 0x0Fu);
          block_acc += static_cast<double>(a_lo) * static_cast<double>(b_lo);
          block_acc += static_cast<double>(a_hi) * static_cast<double>(b_hi);
        }
        acc += block_acc * static_cast<double>(ab_block_scale);
      }
      const double tscale = static_cast<double>(a_tscale) * static_cast<double>(b_tscale);
      (*host_reference)[row * output_rows_per_expert + n] =
          static_cast<float>(acc * tscale);
    }
  }
  return true;
}

bool MaybeCompareFirstFp4DirectFc1Output(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const float* input_dq_scales,
    const float* input_per_row_tensor_scales,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const DeviceNvfp4Matrix& direct_output_pack,
    const float* direct_activation_output_scale,
    __nv_bfloat16* bf16_workspace) {
  if (!RoutedEnvEnabled("NEMOTRON_DEBUG_COMPARE_FP4_DIRECT")) {
    return true;
  }

  constexpr const char* kCompareLabel = "fp4_direct_compare";
  constexpr const char* kPrepackCompareLabel = "fp4_direct_prepack_compare";

  static std::mutex compare_mutex;
  static bool compared_once = false;
  {
    std::lock_guard<std::mutex> lock(compare_mutex);
    if (compared_once) {
      return true;
    }
    compared_once = true;
  }

  if (!input_pack.valid() ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      weights == nullptr ||
      !direct_output_pack.valid() ||
      direct_activation_output_scale == nullptr ||
      bf16_workspace == nullptr) {
    std::fprintf(stderr, "%s: invalid inputs\n", kCompareLabel);
    return false;
  }

  const std::size_t rows = direct_output_pack.rows();
  const std::size_t cols = direct_output_pack.cols();
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::size_t bf16_workspace_count =
      launch_plan->padded_row_capacity() * output_rows_per_expert;
  constexpr std::size_t kMaxReportedMismatches = 16;
  constexpr float kBf16CompareTolerance = 1.0e-2f;

  auto reference_pack =
      DeviceNvfp4Matrix::Create(rows, cols, direct_output_pack.scale_layout());
  auto reference_activation_output_scale =
      DeviceTensorFp32::Create({rows, blocks_per_row});
  if (reference_pack == nullptr ||
      !reference_pack->valid() ||
      reference_activation_output_scale == nullptr ||
      !reference_activation_output_scale->valid()) {
    std::fprintf(stderr, "%s: failed to allocate reference buffers\n", kCompareLabel);
    return false;
  }

  std::vector<float> direct_prepack_output;
  if (!CheckCuda(cudaDeviceSynchronize()) ||
      !CopyDeviceBf16BufferToHost(
          bf16_workspace,
          bf16_workspace_count,
          &direct_prepack_output) ||
      !LaunchZeroBf16Buffer(bf16_workspace, bf16_workspace_count) ||
      !LaunchPlannedPackedInputMatVecBf16(
          input_pack,
          input_expert_tensor_scales,
          input_dq_scales,
          input_per_row_tensor_scales,
          launch_plan,
          dispatch_rows,
          active_selection_count,
          weights,
          output_rows_per_expert,
          bf16_workspace) ||
      !LaunchRoutedBf16Relu2Pack(
          bf16_workspace,
          routing,
          launch_plan,
          active_selection_count,
          reference_pack.get(),
          reference_activation_output_scale->data()) ||
      !CheckCuda(cudaDeviceSynchronize())) {
    std::fprintf(stderr, "%s: failed to build BF16 reference\n", kCompareLabel);
    return false;
  }

  std::vector<float> reference_prepack_output;
  if (!CopyDeviceBf16BufferToHost(
          bf16_workspace,
          bf16_workspace_count,
          &reference_prepack_output)) {
    std::fprintf(stderr, "%s: failed to copy BF16 prepack outputs to host\n", kPrepackCompareLabel);
    return false;
  }

  std::vector<std::uint8_t> direct_packed;
  std::vector<std::uint8_t> reference_packed;
  std::vector<std::uint8_t> direct_block_scales;
  std::vector<std::uint8_t> reference_block_scales;
  std::vector<std::uint8_t> direct_matmul_block_scales;
  std::vector<std::uint8_t> reference_matmul_block_scales;
  std::vector<float> direct_activation_scales;
  std::vector<float> reference_activation_scales;
  if (!direct_output_pack.CopyPackedToHost(&direct_packed) ||
      !reference_pack->CopyPackedToHost(&reference_packed) ||
      !direct_output_pack.CopyBlockScalesToHost(&direct_block_scales) ||
      !reference_pack->CopyBlockScalesToHost(&reference_block_scales) ||
      !direct_output_pack.CopyMatmulBlockScalesToHost(&direct_matmul_block_scales) ||
      !reference_pack->CopyMatmulBlockScalesToHost(&reference_matmul_block_scales) ||
      !CopyDeviceFloatBufferToHost(
          direct_activation_output_scale,
          rows * blocks_per_row,
          &direct_activation_scales) ||
      !CopyDeviceFloatBufferToHost(
          reference_activation_output_scale->data(),
          rows * blocks_per_row,
          &reference_activation_scales)) {
    std::fprintf(stderr, "%s: failed to copy outputs to host\n", kCompareLabel);
    return false;
  }

  std::size_t packed_mismatches = 0;
  std::size_t block_scale_mismatches = 0;
  std::size_t matmul_scale_mismatches = 0;
  std::size_t activation_scale_mismatches = 0;
  std::size_t prepack_mismatches = 0;
  float prepack_max_abs_diff = 0.0f;
  PrintDenseFloatMismatches(
      kPrepackCompareLabel,
      "bf16_output",
      reference_prepack_output,
      direct_prepack_output,
      rows,
      cols,
      kBf16CompareTolerance,
      kMaxReportedMismatches,
      &prepack_mismatches,
      &prepack_max_abs_diff);
  PrintPackedDataMismatches(
      kCompareLabel,
      reference_packed,
      direct_packed,
      rows,
      cols,
      kMaxReportedMismatches,
      &packed_mismatches);
  PrintPackedScaleMismatches(
      kCompareLabel,
      "block_scales_data",
      reference_block_scales,
      direct_block_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &block_scale_mismatches);
  PrintMatmulScaleMismatches(
      kCompareLabel,
      reference_matmul_block_scales,
      direct_matmul_block_scales,
      rows,
      blocks_per_row,
      padded_blocks_per_row,
      direct_output_pack.scale_layout(),
      kMaxReportedMismatches,
      &matmul_scale_mismatches);
  PrintActivationScaleMismatches(
      kCompareLabel,
      reference_activation_scales,
      direct_activation_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &activation_scale_mismatches);

  std::fprintf(
      stderr,
      "%s summary: bf16_output=%zu max_abs_diff=%g tolerance=%g\n",
      kPrepackCompareLabel,
      prepack_mismatches,
      static_cast<double>(prepack_max_abs_diff),
      static_cast<double>(kBf16CompareTolerance));
  std::fprintf(
      stderr,
      "%s summary: packed_data=%zu block_scales_data=%zu matmul_block_scales_data=%zu activation_output_scale=%zu\n",
      kCompareLabel,
      packed_mismatches,
      block_scale_mismatches,
      matmul_scale_mismatches,
      activation_scale_mismatches);

  // Independent host fp64 NVFP4 GEMM reference: ground truth that does not
  // depend on the BF16 reference path.  Lets us tell whether a divergence
  // is in the direct kernel, in the BF16 reference, or in both.
  constexpr const char* kHostOracleLabel = "fp4_direct_host_oracle";
  std::vector<float> host_reference_output;
  const bool host_reference_ok = ComputeFp4DirectHostReference(
      input_pack,
      input_expert_tensor_scales,
      input_per_row_tensor_scales,
      launch_plan,
      weights,
      output_rows_per_expert,
      &host_reference_output);
  if (host_reference_ok) {
    std::size_t host_vs_direct_mismatches = 0;
    float host_vs_direct_max = 0.0f;
    PrintDenseFloatMismatches(
        kHostOracleLabel,
        "direct_prepack",
        host_reference_output,
        direct_prepack_output,
        rows,
        cols,
        kBf16CompareTolerance,
        kMaxReportedMismatches,
        &host_vs_direct_mismatches,
        &host_vs_direct_max);
    std::size_t host_vs_reference_mismatches = 0;
    float host_vs_reference_max = 0.0f;
    PrintDenseFloatMismatches(
        kHostOracleLabel,
        "bf16_reference",
        host_reference_output,
        reference_prepack_output,
        rows,
        cols,
        kBf16CompareTolerance,
        kMaxReportedMismatches,
        &host_vs_reference_mismatches,
        &host_vs_reference_max);
    std::fprintf(
        stderr,
        "%s summary: direct_prepack=%zu max_abs_diff=%g  bf16_reference=%zu max_abs_diff=%g\n",
        kHostOracleLabel,
        host_vs_direct_mismatches,
        static_cast<double>(host_vs_direct_max),
        host_vs_reference_mismatches,
        static_cast<double>(host_vs_reference_max));
  } else {
    std::fprintf(stderr, "%s summary: failed_to_compute_host_reference\n", kHostOracleLabel);
  }

  return true;
}

bool MaybeCompareFirstGroupedBf16Fc1Output(
    const float* normalized,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t token_count,
    std::size_t active_selection_count,
    std::size_t hidden_size,
    float* reference_gather_scratch,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const DeviceNvfp4Matrix& direct_output_pack,
    const float* direct_activation_output_scale,
    __nv_bfloat16* reference_bf16_workspace) {
  if (!RoutedEnvEnabled("NEMOTRON_DEBUG_COMPARE_GROUPED_BF16_FC1")) {
    return true;
  }

  constexpr const char* kCompareLabel = "grouped_bf16_fc1_compare";

  static std::mutex compare_mutex;
  static bool compared_once = false;
  {
    std::lock_guard<std::mutex> lock(compare_mutex);
    if (compared_once) {
      return true;
    }
    compared_once = true;
  }

  if (normalized == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      reference_gather_scratch == nullptr ||
      weights == nullptr ||
      !direct_output_pack.valid() ||
      direct_activation_output_scale == nullptr ||
      reference_bf16_workspace == nullptr) {
    std::fprintf(stderr, "%s: invalid inputs\n", kCompareLabel);
    return false;
  }

  const std::size_t rows = direct_output_pack.rows();
  const std::size_t cols = direct_output_pack.cols();
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_row, kNvfp4ScaleBlockTile);
  const std::size_t gather_count =
      launch_plan->padded_row_capacity() * hidden_size;
  const std::size_t bf16_workspace_count =
      launch_plan->padded_row_capacity() * output_rows_per_expert;
  constexpr std::size_t kMaxReportedMismatches = 16;

  auto reference_pack =
      DeviceNvfp4Matrix::Create(rows, cols, direct_output_pack.scale_layout());
  auto reference_activation_output_scale =
      DeviceTensorFp32::Create({rows, blocks_per_row});
  if (reference_pack == nullptr ||
      !reference_pack->valid() ||
      reference_activation_output_scale == nullptr ||
      !reference_activation_output_scale->valid()) {
    std::fprintf(stderr, "%s: failed to allocate reference buffers\n", kCompareLabel);
    return false;
  }

  if (!LaunchZeroBuffer(reference_gather_scratch, gather_count) ||
      !LaunchGatherRows(
          normalized,
          launch_plan->permuted_idx_to_token_idx(),
          launch_plan->total_num_padded_tokens(),
          launch_plan->padded_row_capacity(),
          token_count,
          hidden_size,
          reference_gather_scratch) ||
      !LaunchZeroBf16Buffer(reference_bf16_workspace, bf16_workspace_count) ||
      !RunLaunchPlannedNvfp4ExpertMatVecBf16(
          reference_gather_scratch,
          launch_plan,
          active_selection_count,
          weights,
          output_rows_per_expert,
          reference_bf16_workspace) ||
      !LaunchRoutedBf16Relu2Pack(
          reference_bf16_workspace,
          routing,
          launch_plan,
          active_selection_count,
          reference_pack.get(),
          reference_activation_output_scale->data()) ||
      !CheckCuda(cudaDeviceSynchronize())) {
    std::fprintf(stderr, "%s: failed to build gathered BF16 reference\n", kCompareLabel);
    return false;
  }

  std::vector<std::uint8_t> direct_packed;
  std::vector<std::uint8_t> reference_packed;
  std::vector<std::uint8_t> direct_block_scales;
  std::vector<std::uint8_t> reference_block_scales;
  std::vector<std::uint8_t> direct_matmul_block_scales;
  std::vector<std::uint8_t> reference_matmul_block_scales;
  std::vector<float> direct_activation_scales;
  std::vector<float> reference_activation_scales;
  if (!direct_output_pack.CopyPackedToHost(&direct_packed) ||
      !reference_pack->CopyPackedToHost(&reference_packed) ||
      !direct_output_pack.CopyBlockScalesToHost(&direct_block_scales) ||
      !reference_pack->CopyBlockScalesToHost(&reference_block_scales) ||
      !direct_output_pack.CopyMatmulBlockScalesToHost(&direct_matmul_block_scales) ||
      !reference_pack->CopyMatmulBlockScalesToHost(&reference_matmul_block_scales) ||
      !CopyDeviceFloatBufferToHost(
          direct_activation_output_scale,
          rows * blocks_per_row,
          &direct_activation_scales) ||
      !CopyDeviceFloatBufferToHost(
          reference_activation_output_scale->data(),
          rows * blocks_per_row,
          &reference_activation_scales)) {
    std::fprintf(stderr, "%s: failed to copy outputs to host\n", kCompareLabel);
    return false;
  }

  std::size_t packed_mismatches = 0;
  std::size_t block_scale_mismatches = 0;
  std::size_t matmul_scale_mismatches = 0;
  std::size_t activation_scale_mismatches = 0;
  PrintPackedDataMismatches(
      kCompareLabel,
      reference_packed,
      direct_packed,
      rows,
      cols,
      kMaxReportedMismatches,
      &packed_mismatches);
  PrintPackedScaleMismatches(
      kCompareLabel,
      "block_scales_data",
      reference_block_scales,
      direct_block_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &block_scale_mismatches);
  PrintMatmulScaleMismatches(
      kCompareLabel,
      reference_matmul_block_scales,
      direct_matmul_block_scales,
      rows,
      blocks_per_row,
      padded_blocks_per_row,
      direct_output_pack.scale_layout(),
      kMaxReportedMismatches,
      &matmul_scale_mismatches);
  PrintActivationScaleMismatches(
      kCompareLabel,
      reference_activation_scales,
      direct_activation_scales,
      rows,
      blocks_per_row,
      kMaxReportedMismatches,
      &activation_scale_mismatches);

  std::fprintf(
      stderr,
      "%s summary: packed_data=%zu block_scales_data=%zu matmul_block_scales_data=%zu activation_output_scale=%zu\n",
      kCompareLabel,
      packed_mismatches,
      block_scale_mismatches,
      matmul_scale_mismatches,
      activation_scale_mismatches);
  return true;
}

bool RoutedProfileDebugEnabled() {
  return std::getenv("NEMOTRON_ROUTED_PROFILE_DEBUG") != nullptr;
}

bool SharedProfileDebugEnabled() {
  return std::getenv("NEMOTRON_SHARED_PROFILE_DEBUG") != nullptr;
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight);

void AppendSharedContiguousTraceEntry(
    const std::string& tensor_name,
    bool plan_build_ok,
    bool execute_ok) {
  if (!IsLinearOpTraceEnabled()) {
    return;
  }
  auto& trace = GetLinearOpTrace();
  std::lock_guard<std::mutex> lock(trace.mutex);
  trace.entries.push_back(LinearOpTraceEntry{
      tensor_name,
      GemmKernelFamily::kSm120ContiguousSharedNvfp4,
      LinearOpPath::kFastpath,
      plan_build_ok,
      execute_ok,
  });
}

std::optional<SharedContiguousPreparedLaunch> PrepareSharedContiguousLaunch(
    const std::string& tensor_name,
    const DeviceNvfp4Matrix& input_pack,
    std::size_t activation_rows,
    const FusedNvfp4WeightView& weight,
    GemmHeuristicCache* heuristic_cache) {
  if (!input_pack.valid() ||
      !ValidFusedNvfp4WeightView(weight) ||
      activation_rows == 0 ||
      activation_rows > input_pack.rows() ||
      input_pack.cols() != weight.input_cols ||
      input_pack.scale_layout() != Nvfp4ScaleLayout::kSwizzled128x4) {
    return std::nullopt;
  }

  const auto launch_plan = BuildSharedNvfp4ContiguousLaunchPlan(
      SharedNvfp4ContiguousPlanRequest{
          tensor_name,
          activation_rows,
          weight.output_rows,
          weight.input_cols,
          ByteRangeView{weight.packed_data, (weight.output_rows * weight.input_cols) / 2u},
          ByteRangeView{
              weight.matmul_block_scales_data != nullptr
                  ? weight.matmul_block_scales_data
                  : weight.block_scales_data,
              1u,
          },
          ByteRangeView{reinterpret_cast<const std::uint8_t*>(weight.tensor_scale_data), sizeof(float)},
      });
  if (!launch_plan.has_value()) {
    return std::nullopt;
  }

  const auto execution = PrepareGemmExecution(*launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    return std::nullopt;
  }

  const void* p5_tma_load_b_descriptors =
      input_pack.shared_p5_tma_load_b_descriptors(*launch_plan);
  const void* p5_tma_load_sfb_descriptors =
      input_pack.shared_p5_tma_load_sfb_descriptors(*launch_plan);
  if (p5_tma_load_b_descriptors == nullptr || p5_tma_load_sfb_descriptors == nullptr) {
    return std::nullopt;
  }

  if (SharedProfileDebugEnabled()) {
    const std::string bucket_upper =
        launch_plan->token_bucket_upper == std::numeric_limits<std::size_t>::max()
            ? "inf"
            : std::to_string(launch_plan->token_bucket_upper);
    std::fprintf(
        stderr,
        "shared_contiguous profile tensor=%s rows=%zu profile=%s tile=%zux%zux%zu bucket=%zu-%s cta=%zux%zu backend=%s algorithm=%llu cached=%d\n",
        tensor_name.c_str(),
        activation_rows,
        launch_plan->profile_name.c_str(),
        launch_plan->tile_m,
        launch_plan->tile_n,
        launch_plan->tile_k,
        launch_plan->token_bucket_lower,
        bucket_upper.c_str(),
        launch_plan->cta_m_count,
        launch_plan->cta_n_count,
        ToString(execution->backend_kind),
        static_cast<unsigned long long>(execution->algorithm_id),
        execution->algorithm_from_cache ? 1 : 0);
  }

  SharedContiguousPreparedLaunch prepared;
  prepared.launch_plan = *launch_plan;
  prepared.execution = *execution;
  prepared.p5_tma_load_b_descriptors = p5_tma_load_b_descriptors;
  prepared.p5_tma_load_sfb_descriptors = p5_tma_load_sfb_descriptors;
  return prepared;
}

const char* RoutedGemm1ProfileName(RoutedGemm1Profile profile) {
  switch (profile) {
    case RoutedGemm1Profile::kLegacy:
      return "legacy";
    case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
      return "p0_128x128x128_swap_false";
    case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
      return "p1_128x128x64_swap_false";
    case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
      return "p4_128x128x128_swap_true";
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue:
      return "p5_128x128x64_swap_true";
    case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
      return "p7_256x128x64_swap_true";
  }
  return "unknown";
}

const char* RoutedGemm1ProfileClassName(RoutedGemm1Profile profile) {
  switch (profile) {
    case RoutedGemm1Profile::kLegacy:
      return "legacy";
    case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
    case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
    case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
      return "bf16_boundary_grouped";
    case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
      return "native_direct";
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue:
      return "native_unified_direct";
  }
  return "unknown";
}

const char* RoutedGemm2ProfileName(RoutedGemm2Profile profile) {
  switch (profile) {
    case RoutedGemm2Profile::kLegacy:
      return "legacy";
    case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
      return "p12_128x128x128_swap_true";
    case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
      return "p13_128x128x64_swap_true";
    case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
      return "p15_256x128x64_swap_true";
  }
  return "unknown";
}

const char* RoutedGemm2ProfileClassName(RoutedGemm2Profile profile) {
  switch (profile) {
    case RoutedGemm2Profile::kLegacy:
      return "legacy";
    case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
    case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
    case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
      return "native_unified";
  }
  return "unknown";
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         (weight.block_scales_data != nullptr ||
          weight.matmul_block_scales_data != nullptr) &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

__host__ __device__ fused_decode::Nvfp4WeightView MakeDeviceWeightView(
    const FusedNvfp4WeightView& weight) {
  return fused_decode::Nvfp4WeightView{
      weight.packed_data,
      weight.block_scales_data,
      weight.matmul_block_scales_data,
      weight.tensor_scale_data,
      weight.output_rows,
      weight.input_cols,
  };
}

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return (numerator + denominator - 1u) / denominator;
}

__host__ __device__ std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
}

__host__ __device__ std::size_t RowTile(Nvfp4ScaleLayout scale_layout) {
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4:
      return 128u;
    case Nvfp4ScaleLayout::kSwizzled8x4:
      return 8u;
  }
  return 128u;
}

__host__ __device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  const std::size_t num_k_tiles = padded_blocks_per_row / kNvfp4ScaleBlockTile;
  const std::size_t k_tile = block_col / kNvfp4ScaleBlockTile;
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

int GetMultiProcessorCount() {
  int device = 0;
  if (!CheckCuda(cudaGetDevice(&device))) {
    return 0;
  }
  int multiprocessor_count = 0;
  if (!CheckCuda(cudaDeviceGetAttribute(
          &multiprocessor_count, cudaDevAttrMultiProcessorCount, device))) {
    return 0;
  }
  return multiprocessor_count;
}

template <typename KernelFn, typename... Args>
bool LaunchProgrammaticKernel(
    dim3 grid,
    dim3 block,
    KernelFn kernel,
    Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = cudaStream_t{};
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = &attr;
  return CheckCuda(cudaLaunchKernelEx(&config, kernel, args...)) &&
         CheckCuda(cudaGetLastError());
}

int SelectContiguousOutputTile(std::size_t output_rows, std::size_t input_row_count) {
  const int multiprocessor_count = GetMultiProcessorCount();
  if (multiprocessor_count <= 0) {
    return kContiguousMediumOutputTile;
  }
  const std::size_t input_row_tiles =
      CeilDiv(input_row_count, static_cast<std::size_t>(kPlannedWmmaTileM));
  const auto meets_target = [&](int output_tile) {
    const std::size_t output_row_tiles =
        CeilDiv(output_rows, static_cast<std::size_t>(output_tile));
    return output_row_tiles * input_row_tiles >=
           static_cast<std::size_t>(multiprocessor_count);
  };
  if (meets_target(kContiguousLargeOutputTile)) {
    return kContiguousLargeOutputTile;
  }
  if (meets_target(kContiguousMediumOutputTile)) {
    return kContiguousMediumOutputTile;
  }
  return kContiguousSmallOutputTile;
}

RoutedGemm1Profile SelectRoutedGemm1Profile(std::size_t num_rows) {
  if (RoutedEnvEnabled("NEMOTRON_DEBUG_FORCE_LEGACY_P1")) {
    return RoutedGemm1Profile::kLegacy;
  }
  if (num_rows >= 1u && num_rows <= 8u) {
    // Use the unified FP4 MMA kernel for all small row counts.  The non-unified
    // P0/P7 profiles decode FP4→BF16 before the WMMA multiply, which loses
    // precision compared to the native block-scaled FP4 MMA used by P5.  At
    // small dispatch sizes (1-3 rows per expert) this precision loss accumulates
    // across layers and produces divergent output.
    return RoutedGemm1Profile::kP5_128x128x64_SwapTrue;
  }
  if ((num_rows >= 9u && num_rows <= 15u) || num_rows == 16u ||
      (num_rows >= 24u && num_rows <= 31u) || num_rows == 320u ||
      (num_rows >= 448u && num_rows <= 511u) || num_rows >= 1024u) {
    return RoutedGemm1Profile::kP1_128x128x64_SwapFalse;
  }
  if (num_rows >= 32u && num_rows <= 112u) {
    return RoutedGemm1Profile::kP0_128x128x128_SwapFalse;
  }
  if ((num_rows >= 120u && num_rows <= 127u) ||
      (num_rows >= 200u && num_rows <= 248u)) {
    return RoutedGemm1Profile::kP4_128x128x128_SwapTrue;
  }
  if ((num_rows >= 128u && num_rows <= 192u) || num_rows == 256u ||
      num_rows == 384u || (num_rows >= 512u && num_rows <= 992u)) {
    return RoutedGemm1Profile::kP5_128x128x64_SwapTrue;
  }
  return RoutedGemm1Profile::kLegacy;
}

RoutedGemm2Profile SelectRoutedGemm2Profile(std::size_t num_rows) {
  if (RoutedEnvEnabled("NEMOTRON_DEBUG_FORCE_LEGACY_P13")) {
    return RoutedGemm2Profile::kLegacy;
  }
  if (num_rows >= 2u && num_rows <= 7u) {
    return RoutedGemm2Profile::kP15_256x128x64_SwapTrue;
  }
  if (num_rows == 16u) {
    return RoutedGemm2Profile::kP15_256x128x64_SwapTrue;
  }
  if (num_rows == 1u || num_rows == 8u || (num_rows >= 24u && num_rows <= 127u) ||
      num_rows == 256u || num_rows == 384u || num_rows >= 1024u) {
    return RoutedGemm2Profile::kP13_128x128x64_SwapTrue;
  }
  if ((num_rows >= 9u && num_rows <= 15u) || (num_rows >= 128u && num_rows <= 248u) ||
      num_rows == 320u || (num_rows >= 448u && num_rows <= 992u)) {
    return RoutedGemm2Profile::kP12_128x128x128_SwapTrue;
  }
  return RoutedGemm2Profile::kLegacy;
}

int RoutedOutputTile(RoutedGemm1Profile profile) {
  switch (profile) {
    case RoutedGemm1Profile::kP7_256x128x64_SwapTrue:
      return kRoutedLargeOutputTile;
    case RoutedGemm1Profile::kLegacy:
    case RoutedGemm1Profile::kP0_128x128x128_SwapFalse:
    case RoutedGemm1Profile::kP1_128x128x64_SwapFalse:
    case RoutedGemm1Profile::kP4_128x128x128_SwapTrue:
    case RoutedGemm1Profile::kP5_128x128x64_SwapTrue:
      return kPlannedOutputTile;
  }
  return kPlannedOutputTile;
}

int RoutedOutputTile(RoutedGemm2Profile profile) {
  switch (profile) {
    case RoutedGemm2Profile::kP15_256x128x64_SwapTrue:
      return kRoutedLargeOutputTile;
    case RoutedGemm2Profile::kLegacy:
    case RoutedGemm2Profile::kP12_128x128x128_SwapTrue:
    case RoutedGemm2Profile::kP13_128x128x64_SwapTrue:
      return kPlannedOutputTile;
  }
  return kPlannedOutputTile;
}

__global__ void ZeroBufferKernel(float* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = 0.0f;
}

__global__ void ZeroBf16BufferKernel(__nv_bfloat16* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = __float2bfloat16(0.0f);
}

__global__ void FillFloatBufferKernel(float* data, std::size_t count, float value) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = value;
}

__global__ void GatherRowsKernel(
    const float* input,
    const int* row_indices,
    const int* active_output_rows,
    float* output,
    std::size_t output_row_capacity,
    std::size_t input_rows,
    std::size_t cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = output_row_capacity * cols;
  if (index >= count) {
    return;
  }

  const std::size_t row = index / cols;
  if (active_output_rows != nullptr &&
      row >= static_cast<std::size_t>(active_output_rows[0])) {
    return;
  }
  const std::size_t col = index % cols;
  const int input_row = row_indices[row];
  if (input_row < 0 || static_cast<std::size_t>(input_row) >= input_rows) {
    output[index] = 0.0f;
    return;
  }
  output[index] = input[static_cast<std::size_t>(input_row) * cols + col];
}

__global__ void QuantizeDequantizeRowsKernel(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= row_capacity) {
    return;
  }
  if (active_row_count != nullptr &&
      row >= static_cast<std::size_t>(active_row_count[0])) {
    return;
  }
  fused_decode::QuantizeDequantizeNvfp4Row(
      data + row * cols,
      data + row * cols,
      cols);
}

__global__ void Relu2Kernel(float* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = fused_decode::Relu2(data[index]);
}

__global__ void Relu2RowsKernel(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = row_capacity * cols;
  if (index >= count) {
    return;
  }
  const std::size_t row = index / cols;
  if (active_row_count != nullptr &&
      row >= static_cast<std::size_t>(active_row_count[0])) {
    return;
  }
  data[index] = fused_decode::Relu2(data[index]);
}

__device__ float ClampNvfp4TensorScale(float value) {
  if (!isfinite(value) || value < kNvfp4MinTensorScale) {
    return kNvfp4MinTensorScale;
  }
  return value;
}

__global__ void ComputeExpertActivationScalesKernel(
    const float* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x);
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      expert_index >= n_experts) {
    return;
  }

  const int row_begin = expert_first_token_offsets[expert_index];
  const int row_end = expert_first_token_offsets[expert_index + 1];
  if (row_end <= row_begin) {
    if (threadIdx.x == 0) {
      output_scales[expert_index] = 1.0f;
    }
    return;
  }

  float thread_max_abs = 0.0f;
  const std::size_t row_count = static_cast<std::size_t>(row_end - row_begin);
  const std::size_t numel = row_count * cols;
  for (std::size_t linear_index = threadIdx.x;
       linear_index < numel;
       linear_index += blockDim.x) {
    const std::size_t row = linear_index / cols;
    const std::size_t col = linear_index % cols;
    const float value = input[(static_cast<std::size_t>(row_begin) + row) * cols + col];
    const float abs_value = fabsf(value);
    if (abs_value > thread_max_abs) {
      thread_max_abs = abs_value;
    }
  }

  __shared__ float shared_max_abs[fused_decode::kThreadsPerBlock];
  shared_max_abs[threadIdx.x] = thread_max_abs;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max_abs[threadIdx.x + stride] > shared_max_abs[threadIdx.x]) {
      shared_max_abs[threadIdx.x] = shared_max_abs[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float tensor_scale = 1.0f;
    if (shared_max_abs[0] > kNvfp4ActivationMaxFinite) {
      tensor_scale = ClampNvfp4TensorScale(shared_max_abs[0] / kNvfp4ActivationMaxFinite);
    }
    output_scales[expert_index] = tensor_scale;
  }
}

__global__ void ComputeExpertActivationScalesBf16Kernel(
    const __nv_bfloat16* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x);
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      expert_index >= n_experts) {
    return;
  }

  const int row_begin = expert_first_token_offsets[expert_index];
  const int row_end = expert_first_token_offsets[expert_index + 1];
  if (row_end <= row_begin) {
    if (threadIdx.x == 0) {
      output_scales[expert_index] = 1.0f;
    }
    return;
  }

  float thread_max_abs = 0.0f;
  const std::size_t row_count = static_cast<std::size_t>(row_end - row_begin);
  const std::size_t numel = row_count * cols;
  for (std::size_t linear_index = threadIdx.x;
       linear_index < numel;
       linear_index += blockDim.x) {
    const std::size_t row = linear_index / cols;
    const std::size_t col = linear_index % cols;
    const float value = fused_decode::Relu2(__bfloat162float(
        input[(static_cast<std::size_t>(row_begin) + row) * cols + col]));
    const float abs_value = value;
    if (abs_value > thread_max_abs) {
      thread_max_abs = abs_value;
    }
  }

  __shared__ float shared_max_abs[fused_decode::kThreadsPerBlock];
  shared_max_abs[threadIdx.x] = thread_max_abs;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max_abs[threadIdx.x + stride] > shared_max_abs[threadIdx.x]) {
      shared_max_abs[threadIdx.x] = shared_max_abs[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float tensor_scale = 1.0f;
    if (shared_max_abs[0] > kNvfp4ActivationMaxFinite) {
      tensor_scale = ClampNvfp4TensorScale(shared_max_abs[0] / kNvfp4ActivationMaxFinite);
    }
    output_scales[expert_index] = tensor_scale;
  }
}

__global__ void MaxBitsToTensorScalesKernel(
    const unsigned int* max_bits,
    std::size_t count,
    float* output_scales) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float max_abs = __uint_as_float(max_bits[index]);
  float tensor_scale = 1.0f;
  if (max_abs > kNvfp4ActivationMaxFinite) {
    tensor_scale = ClampNvfp4TensorScale(max_abs / kNvfp4ActivationMaxFinite);
  }
  output_scales[index] = tensor_scale;
}

__device__ __forceinline__ int FindExpertForPaddedRow(
    const int* padded_expert_offsets,
    int n_experts,
    int row) {
  int low = 0;
  int high = n_experts;
  while (low < high) {
    const int mid = low + ((high - low) >> 1);
    if (padded_expert_offsets[mid + 1] <= row) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

template <int kProcessRows>
__global__ __launch_bounds__(fused_decode::kThreadsPerBlock)
void RoutedBf16Relu2PackKernel(
    const __nv_bfloat16* source,
    const int* actual_expert_offsets,
    const int* padded_expert_offsets,
    int n_experts,
    std::size_t cols,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    float* output_dequant_scales,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales) {
  cudaGridDependencySynchronize();

  if (source == nullptr ||
      actual_expert_offsets == nullptr ||
      padded_expert_offsets == nullptr ||
      output_dequant_scales == nullptr ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr ||
      cols == 0 ||
      (cols % fused_decode::kNvfp4BlockWidth) != 0 ||
      n_experts <= 0) {
    return;
  }

  const int64_t total_padded_rows = padded_expert_offsets[n_experts];
  if (total_padded_rows <= 0) {
    return;
  }

  const int64_t blocks_per_row =
      static_cast<int64_t>(cols / fused_decode::kNvfp4BlockWidth);
  const int64_t block_col = static_cast<int64_t>(blockIdx.y);
  if (block_col >= blocks_per_row) {
    return;
  }

  constexpr int64_t rows_per_cta =
      static_cast<int64_t>(kProcessRows) * fused_decode::kThreadsPerBlock;
  const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * rows_per_cta;

  for (int64_t row_base =
           static_cast<int64_t>(blockIdx.x) * rows_per_cta + threadIdx.x;
       row_base < total_padded_rows;
       row_base += grid_stride) {
    int expert = FindExpertForPaddedRow(
        padded_expert_offsets,
        n_experts,
        static_cast<int>(row_base));

#pragma unroll
    for (int row_iter = 0; row_iter < kProcessRows; ++row_iter) {
      const int64_t row =
          row_base + static_cast<int64_t>(row_iter) * fused_decode::kThreadsPerBlock;
      if (row >= total_padded_rows) {
        break;
      }

      while (row_iter > 0 &&
             (expert + 1) < n_experts &&
             padded_expert_offsets[expert + 1] <= row) {
        ++expert;
      }

      const int64_t padded_row_begin = padded_expert_offsets[expert];
      const int64_t actual_row_begin = actual_expert_offsets[expert];
      const int64_t actual_row_end = actual_expert_offsets[expert + 1];
      const int64_t active_row_end =
          padded_row_begin + (actual_row_end - actual_row_begin);

      const std::size_t scale_index =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(blocks_per_row) +
          static_cast<std::size_t>(block_col);
      const std::size_t matmul_scale_index = ExecutionScaleOffset(
          static_cast<std::size_t>(row),
          static_cast<std::size_t>(block_col),
          padded_blocks_per_row,
          scale_layout);
      const std::size_t packed_offset =
          static_cast<std::size_t>(row) * (cols / 2u) +
          static_cast<std::size_t>(block_col) *
              (fused_decode::kNvfp4BlockWidth / 2u);

      if (row >= active_row_end) {
        output_dequant_scales[scale_index] = 0.0f;
        output_block_scales[scale_index] = 0u;
        output_matmul_scales[matmul_scale_index] = 0u;
        for (std::size_t pair = 0; pair < (fused_decode::kNvfp4BlockWidth / 2u); ++pair) {
          output_packed[packed_offset + pair] = 0u;
        }
        continue;
      }

      const std::size_t input_offset =
          static_cast<std::size_t>(row) * cols +
          static_cast<std::size_t>(block_col) * fused_decode::kNvfp4BlockWidth;
      float activated[fused_decode::kNvfp4BlockWidth];
      float block_max_abs = 0.0f;
      for (std::size_t col = 0; col < fused_decode::kNvfp4BlockWidth; ++col) {
        const float value = __bfloat162float(source[input_offset + col]);
        const float relu2 = fused_decode::Relu2(value);
        activated[col] = relu2;
        if (relu2 > block_max_abs) {
          block_max_abs = relu2;
        }
      }

      float block_scale = 1.0f;
      if (block_max_abs > 0.0f) {
        block_scale = fused_decode::ClampNvfp4Scale(
            block_max_abs / fused_decode::kNvfp4Fp4MaxFinite);
      }
      const std::uint8_t encoded_block_scale =
          fused_decode::EncodeFp8Scale(block_scale);
      output_dequant_scales[scale_index] = block_scale;
      output_block_scales[scale_index] = encoded_block_scale;
      output_matmul_scales[matmul_scale_index] = encoded_block_scale;

      for (std::size_t pair = 0; pair < (fused_decode::kNvfp4BlockWidth / 2u); ++pair) {
        const std::uint8_t lhs =
            fused_decode::EncodeFp4(activated[pair * 2u] / block_scale);
        const std::uint8_t rhs =
            fused_decode::EncodeFp4(activated[pair * 2u + 1u] / block_scale);
        output_packed[packed_offset + pair] =
            static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
      }
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

__device__ __forceinline__ std::uint8_t LoadExecutionScaleByte(
    const std::uint8_t* scale_bytes,
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout);

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const FusedNvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col);

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const fused_decode::Nvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col);

__device__ __forceinline__ float DecodeNvfp4WeightElement(
    const FusedNvfp4WeightView& weight,
    std::size_t output_row,
    std::size_t col) {
  const std::size_t pairs_per_row = weight.input_cols / 2u;
  const std::size_t pair_index = col / 2u;
  const std::size_t block = col / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * pairs_per_row;
  const float block_scale = LoadNvfp4WeightBlockScale(weight, output_row, block);
  const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
  const std::uint8_t nibble =
      (col & 1u) == 0u ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu);
  return fused_decode::DecodeFp4(nibble) * block_scale;
}

__device__ __forceinline__ float DecodeGroupedPackedInputElement(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_dq_scales,
    float input_tensor_scale,
    std::size_t cols,
    std::size_t input_row,
    std::size_t col) {
  const std::size_t pairs_per_row = cols / 2u;
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t pair_index = col / 2u;
  const std::size_t block = col / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = input_row * pairs_per_row;
  const std::size_t scale_row_offset = input_row * blocks_per_row;
  const float block_scale =
      input_dq_scales != nullptr
          ? input_dq_scales[scale_row_offset + block]
          : fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
                input_tensor_scale;
  const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
  const std::uint8_t nibble =
      (col & 1u) == 0u ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu);
  return fused_decode::DecodeFp4(nibble) * block_scale;
}

__device__ __forceinline__ void ZeroBf16Block16(__nv_bfloat16* output) {
#pragma unroll
  for (int i = 0; i < static_cast<int>(fused_decode::kNvfp4BlockWidth); ++i) {
    output[i] = __float2bfloat16(0.0f);
  }
}

__device__ __forceinline__ void DecodePackedNvfp4BlockToBf16(
    const std::uint8_t* packed_block,
    float block_scale,
    __nv_bfloat16* output) {
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t packed = packed_block[pair];
    output[pair * 2] =
        __float2bfloat16(fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale);
    output[pair * 2 + 1] =
        __float2bfloat16(fused_decode::DecodeFp4((packed >> 4u) & 0x0Fu) * block_scale);
  }
}

__device__ __forceinline__ void DecodeNvfp4WeightBlockBf16(
    const FusedNvfp4WeightView& weight,
    std::size_t output_row,
    std::size_t block_index,
    __nv_bfloat16* output) {
  const std::size_t blocks_per_row =
      weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset =
      output_row * (weight.input_cols / 2u) + block_index * (fused_decode::kNvfp4BlockWidth / 2u);
  const float block_scale = LoadNvfp4WeightBlockScale(weight, output_row, block_index);
  DecodePackedNvfp4BlockToBf16(weight.packed_data + packed_row_offset, block_scale, output);
}

__device__ __forceinline__ void DecodeGroupedPackedInputBlockBf16(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_dq_scales,
    float input_tensor_scale,
    std::size_t cols,
    std::size_t input_row,
    std::size_t block_index,
    __nv_bfloat16* output) {
  const std::size_t blocks_per_row = cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset =
      input_row * (cols / 2u) + block_index * (fused_decode::kNvfp4BlockWidth / 2u);
  const std::size_t scale_row_offset = input_row * blocks_per_row + block_index;
  const float block_scale =
      input_dq_scales != nullptr
          ? input_dq_scales[scale_row_offset]
          : fused_decode::DecodeFp8(input_block_scales[scale_row_offset]) * input_tensor_scale;
  DecodePackedNvfp4BlockToBf16(packed_input + packed_row_offset, block_scale, output);
}

__device__ __forceinline__ std::uint8_t LoadPackedFp4Nibble(
    const std::uint8_t* packed_row,
    int col) {
  const std::uint8_t packed = packed_row[col >> 1];
  return static_cast<std::uint8_t>(((col & 1) == 0) ? (packed & 0x0Fu) : ((packed >> 4u) & 0x0Fu));
}

__device__ __forceinline__ std::uint32_t PackFp4Register8(
    std::uint8_t v0,
    std::uint8_t v1,
    std::uint8_t v2,
    std::uint8_t v3,
    std::uint8_t v4,
    std::uint8_t v5,
    std::uint8_t v6,
    std::uint8_t v7) {
  std::uint32_t reg = static_cast<std::uint32_t>(v0 & 0x0Fu);
  reg |= static_cast<std::uint32_t>(v1 & 0x0Fu) << 4u;
  reg |= static_cast<std::uint32_t>(v2 & 0x0Fu) << 8u;
  reg |= static_cast<std::uint32_t>(v3 & 0x0Fu) << 12u;
  reg |= static_cast<std::uint32_t>(v4 & 0x0Fu) << 16u;
  reg |= static_cast<std::uint32_t>(v5 & 0x0Fu) << 20u;
  reg |= static_cast<std::uint32_t>(v6 & 0x0Fu) << 24u;
  reg |= static_cast<std::uint32_t>(v7 & 0x0Fu) << 28u;
  return reg;
}

__device__ __forceinline__ std::uint32_t PackScaleWord4(
    std::uint8_t s0,
    std::uint8_t s1,
    std::uint8_t s2,
    std::uint8_t s3) {
  std::uint32_t reg = static_cast<std::uint32_t>(s0);
  reg |= static_cast<std::uint32_t>(s1) << 8u;
  reg |= static_cast<std::uint32_t>(s2) << 16u;
  reg |= static_cast<std::uint32_t>(s3) << 24u;
  return reg;
}

__device__ __forceinline__ std::uint32_t MakeP13ProbeScaleWord(int row) {
  const std::uint8_t base = static_cast<std::uint8_t>(row & 0xff);
  return PackScaleWord4(
      base,
      static_cast<std::uint8_t>(base + 1u),
      static_cast<std::uint8_t>(base + 2u),
      static_cast<std::uint8_t>(base + 3u));
}

template <class T>
__device__ __forceinline__ std::uint8_t ScaleByteValue(T const& value) {
  using ValueType = std::remove_cvref_t<T>;
  if constexpr (std::is_integral_v<ValueType>) {
    return static_cast<std::uint8_t>(value);
  } else {
    return static_cast<std::uint8_t>(value.raw());
  }
}

template <int kWordCount, class CopyViewTensor>
__device__ __forceinline__ void PackLinearFp4CopyViewWords(
    CopyViewTensor const& copy_view,
    std::uint32_t (&words)[kWordCount]) {
  constexpr int kPhysicalCount = decltype(cute::size(copy_view))::value;
  static_assert(
      kPhysicalCount == kWordCount * 8,
      "Packed FP4 copy view size must match the requested word count");
#pragma unroll
  for (int word = 0; word < kWordCount; ++word) {
    std::uint32_t packed = 0u;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int physical = word * 8 + elem;
      packed |= static_cast<std::uint32_t>(ScaleByteValue(copy_view(physical)) & 0x0Fu)
                << (elem * 4);
    }
    words[word] = packed;
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void FindScaleTensorCoordByRaw(
    ScaleTensor const& scale_tensor,
    std::uint8_t raw,
    int& row_out,
    int& col_out) {
  row_out = -1;
  col_out = -1;
  for (int row = 0; row < static_cast<int>(cute::size<0>(ScaleTensor{})); ++row) {
    for (int col = 0; col < static_cast<int>(cute::size<1>(ScaleTensor{})); ++col) {
      if (ScaleByteValue(scale_tensor(row, col, cute::Int<0>{})) == raw) {
        row_out = row;
        col_out = col;
        return;
      }
    }
  }
}

__device__ __forceinline__ void ApplySm120Fp4ShiftA(
    std::uint32_t& a0,
    std::uint32_t& a1,
    std::uint32_t& a2,
    std::uint32_t& a3) {
  a0 <<= 2u;
  a1 <<= 2u;
  a2 <<= 2u;
  a3 <<= 2u;
}

__device__ __forceinline__ void ApplySm120Fp4ShiftB(
    std::uint32_t& b0,
    std::uint32_t& b1) {
  b0 <<= 2u;
  b1 <<= 2u;
}

__device__ __forceinline__ std::uint32_t LoadExecutionScaleWord(
    const std::uint8_t* scale_bytes,
    std::size_t row,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  return PackScaleWord4(
      scale_bytes[ExecutionScaleOffset(row, block_base + 0u, padded_blocks_per_row, scale_layout)],
      scale_bytes[ExecutionScaleOffset(row, block_base + 1u, padded_blocks_per_row, scale_layout)],
      scale_bytes[ExecutionScaleOffset(row, block_base + 2u, padded_blocks_per_row, scale_layout)],
      scale_bytes[ExecutionScaleOffset(row, block_base + 3u, padded_blocks_per_row, scale_layout)]);
}

__device__ __forceinline__ std::uint8_t LoadExecutionScaleByte(
    const std::uint8_t* scale_bytes,
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  return scale_bytes[ExecutionScaleOffset(row, block_col, padded_blocks_per_row, scale_layout)];
}

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const FusedNvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col) {
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  if (weight.block_scales_data != nullptr) {
    return fused_decode::DecodeFp8(weight.block_scales_data[row * blocks_per_row + block_col]) *
           tensor_scale;
  }
  const std::size_t padded_blocks_per_row = (blocks_per_row + 3u) & ~std::size_t{3u};
  return fused_decode::DecodeFp8(
             LoadExecutionScaleByte(
                 weight.matmul_block_scales_data,
                 row,
                 block_col,
                 padded_blocks_per_row,
                 Nvfp4ScaleLayout::kSwizzled128x4)) *
         tensor_scale;
}

__device__ __forceinline__ float LoadNvfp4WeightBlockScale(
    const fused_decode::Nvfp4WeightView& weight,
    std::size_t row,
    std::size_t block_col) {
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  if (weight.block_scales_data != nullptr) {
    return fused_decode::DecodeFp8(weight.block_scales_data[row * blocks_per_row + block_col]) *
           tensor_scale;
  }
  const std::size_t padded_blocks_per_row = (blocks_per_row + 3u) & ~std::size_t{3u};
  return fused_decode::DecodeFp8(
             LoadExecutionScaleByte(
                 weight.matmul_block_scales_data,
                 row,
                 block_col,
                 padded_blocks_per_row,
                 Nvfp4ScaleLayout::kSwizzled128x4)) *
         tensor_scale;
}

__device__ __forceinline__ void Sm120BlockScaledFp4Mma(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    std::uint32_t a0,
    std::uint32_t a1,
    std::uint32_t a2,
    std::uint32_t a3,
    std::uint32_t b0,
    std::uint32_t b1,
    std::uint32_t sfa,
    std::uint32_t sfb) {
  static constexpr std::uint16_t kBidA = 0;
  static constexpr std::uint16_t kTidA = 0;
  static constexpr std::uint16_t kBidB = 0;
  static constexpr std::uint16_t kTidB = 0;
  const float c0 = d0;
  const float c1 = d1;
  const float c2 = d2;
  const float c3 = d3;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa), "h"(kBidA), "h"(kTidA),
        "r"(sfb), "h"(kBidB), "h"(kTidB));
}

template <int kRowsPerTile>
__device__ __forceinline__ void ZeroPackedTileRows(
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words,
    int row) {
  if (row >= kRowsPerTile) {
    return;
  }
#pragma unroll
  for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
    packed_rows[row * (64 / 2) + byte_index] = 0u;
  }
  if (scale_words != nullptr) {
    scale_words[row] = 0u;
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void CopyPackedTileRow64(
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* matmul_scales,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words,
    int row) {
  if (row >= kRowsPerTile) {
    return;
  }
  const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
  std::uint8_t* dst = packed_rows + row * (64 / 2);
#pragma unroll
  for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
    dst[byte_index] = packed_data[src_offset + static_cast<std::size_t>(byte_index)];
  }
  if (scale_words != nullptr) {
    scale_words[row] = LoadExecutionScaleWord(
        matmul_scales,
        source_row,
        block_base,
        padded_blocks_per_row,
        scale_layout);
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void CopyPackedTileRow64FromRowMajorScales(
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* block_scales,
    std::size_t block_base,
    std::size_t blocks_per_row,
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words,
    int row) {
  if (row >= kRowsPerTile) {
    return;
  }
  const std::size_t src_offset = source_row * packed_row_bytes + packed_byte_offset;
  std::uint8_t* dst = packed_rows + row * (64 / 2);
#pragma unroll
  for (int byte_index = 0; byte_index < (64 / 2); ++byte_index) {
    dst[byte_index] = packed_data[src_offset + static_cast<std::size_t>(byte_index)];
  }
  if (scale_words != nullptr) {
    const std::size_t scale_offset = source_row * blocks_per_row + block_base;
    scale_words[row] = PackScaleWord4(
        block_scales[scale_offset + 0u],
        block_scales[scale_offset + 1u],
        block_scales[scale_offset + 2u],
        block_scales[scale_offset + 3u]);
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void LoadFp4ARegistersRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int lane_id,
    std::uint32_t& a0,
    std::uint32_t& a1,
    std::uint32_t& a2,
    std::uint32_t& a3,
    std::uint32_t& sfa) {
  const int col_base = lane_id >> 2;
  const int row_pair_base = (lane_id & 3) * 2;
  const std::uint8_t* row0 = packed_rows + row_pair_base * (64 / 2);
  const std::uint8_t* row1 = packed_rows + (row_pair_base + 1) * (64 / 2);
  const std::uint8_t* row8 = packed_rows + (row_pair_base + 8) * (64 / 2);
  const std::uint8_t* row9 = packed_rows + (row_pair_base + 9) * (64 / 2);
  a0 = PackFp4Register8(
      LoadPackedFp4Nibble(row0, col_base + 0),
      LoadPackedFp4Nibble(row0, col_base + 16),
      LoadPackedFp4Nibble(row0, col_base + 32),
      LoadPackedFp4Nibble(row0, col_base + 48),
      LoadPackedFp4Nibble(row1, col_base + 0),
      LoadPackedFp4Nibble(row1, col_base + 16),
      LoadPackedFp4Nibble(row1, col_base + 32),
      LoadPackedFp4Nibble(row1, col_base + 48));
  a1 = PackFp4Register8(
      LoadPackedFp4Nibble(row0, col_base + 8),
      LoadPackedFp4Nibble(row0, col_base + 24),
      LoadPackedFp4Nibble(row0, col_base + 40),
      LoadPackedFp4Nibble(row0, col_base + 56),
      LoadPackedFp4Nibble(row1, col_base + 8),
      LoadPackedFp4Nibble(row1, col_base + 24),
      LoadPackedFp4Nibble(row1, col_base + 40),
      LoadPackedFp4Nibble(row1, col_base + 56));
  a2 = PackFp4Register8(
      LoadPackedFp4Nibble(row8, col_base + 0),
      LoadPackedFp4Nibble(row8, col_base + 16),
      LoadPackedFp4Nibble(row8, col_base + 32),
      LoadPackedFp4Nibble(row8, col_base + 48),
      LoadPackedFp4Nibble(row9, col_base + 0),
      LoadPackedFp4Nibble(row9, col_base + 16),
      LoadPackedFp4Nibble(row9, col_base + 32),
      LoadPackedFp4Nibble(row9, col_base + 48));
  a3 = PackFp4Register8(
      LoadPackedFp4Nibble(row8, col_base + 8),
      LoadPackedFp4Nibble(row8, col_base + 24),
      LoadPackedFp4Nibble(row8, col_base + 40),
      LoadPackedFp4Nibble(row8, col_base + 56),
      LoadPackedFp4Nibble(row9, col_base + 8),
      LoadPackedFp4Nibble(row9, col_base + 24),
      LoadPackedFp4Nibble(row9, col_base + 40),
      LoadPackedFp4Nibble(row9, col_base + 56));
  const int scale_row = (lane_id >> 2) + ((lane_id & 1) ? 8 : 0);
  sfa = scale_words[scale_row];
}

template <int kRowsPerTile>
__device__ __forceinline__ void LoadFp4BRegistersColMajor64x8(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int lane_id,
    int n_base,
    std::uint32_t& b0,
    std::uint32_t& b1,
    std::uint32_t& sfb) {
  const int row_group = lane_id & 3;
  const int k_base = lane_id >> 2;
  const std::uint8_t* row0 = packed_rows + (n_base + row_group) * (64 / 2);
  const std::uint8_t* row4 = packed_rows + (n_base + row_group + 4) * (64 / 2);
  b0 = PackFp4Register8(
      LoadPackedFp4Nibble(row0, k_base + 0),
      LoadPackedFp4Nibble(row0, k_base + 8),
      LoadPackedFp4Nibble(row0, k_base + 16),
      LoadPackedFp4Nibble(row0, k_base + 24),
      LoadPackedFp4Nibble(row0, k_base + 32),
      LoadPackedFp4Nibble(row0, k_base + 40),
      LoadPackedFp4Nibble(row0, k_base + 48),
      LoadPackedFp4Nibble(row0, k_base + 56));
  b1 = PackFp4Register8(
      LoadPackedFp4Nibble(row4, k_base + 0),
      LoadPackedFp4Nibble(row4, k_base + 8),
      LoadPackedFp4Nibble(row4, k_base + 16),
      LoadPackedFp4Nibble(row4, k_base + 24),
      LoadPackedFp4Nibble(row4, k_base + 32),
      LoadPackedFp4Nibble(row4, k_base + 40),
      LoadPackedFp4Nibble(row4, k_base + 48),
      LoadPackedFp4Nibble(row4, k_base + 56));
  sfb = scale_words[n_base + (lane_id >> 2)];
}

template <int kRowsPerTile>
__device__ __forceinline__ void StoreFp4AccumulatorTileRowMajor16x8(
    float alpha,
    const float c0,
    const float c1,
    const float c2,
    const float c3,
    int lane_id,
    int col_base,
    int row_base,
    int valid_rows,
    int valid_cols,
    std::size_t row_start,
    std::size_t output_row_base,
    std::size_t output_rows_per_expert,
    float* output) {
  if (col_base >= valid_cols) {
    return;
  }
  const int row_group = (lane_id >> 3) * 4;
  const int row0 = row_group + 0;
  const int row1 = row_group + 2;
  const int row2 = row_group + 1;
  const int row3 = row_group + 3;
  if (row0 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row0)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c0 * alpha;
  }
  if (row1 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row1)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c1 * alpha;
  }
  if (row2 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row2)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c2 * alpha;
  }
  if (row3 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row3)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] = c3 * alpha;
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ void StoreFp4AccumulatorTileRowMajor16x8(
    float alpha,
    const float c0,
    const float c1,
    const float c2,
    const float c3,
    int lane_id,
    int col_base,
    int row_base,
    int valid_rows,
    int valid_cols,
    std::size_t row_start,
    std::size_t output_row_base,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (col_base >= valid_cols) {
    return;
  }
  const int row_group = (lane_id >> 3) * 4;
  const int row0 = row_group + 0;
  const int row1 = row_group + 2;
  const int row2 = row_group + 1;
  const int row3 = row_group + 3;
  if (row0 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row0)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c0 * alpha);
  }
  if (row1 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row1)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c1 * alpha);
  }
  if (row2 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row2)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c2 * alpha);
  }
  if (row3 < valid_rows) {
    output[(row_start + static_cast<std::size_t>(row3)) * output_rows_per_expert +
           (output_row_base + static_cast<std::size_t>(col_base))] =
        __float2bfloat16(c3 * alpha);
  }
}

template <int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::PackedTile64<kRowsPerTile>
nvfp4_bridge::MakePackedTile64(
    std::uint8_t* packed_rows,
    std::uint32_t* scale_words) {
  return PackedTile64<kRowsPerTile>{packed_rows, scale_words};
}

template <int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::ZeroRow(
    const PackedTile64<kRowsPerTile>& tile,
    int row) {
  ZeroPackedTileRows<kRowsPerTile>(tile.packed_rows, tile.scale_words, row);
}

template <int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::CopyWeightRow64(
    const PackedTile64<kRowsPerTile>& tile,
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* matmul_scales,
    std::size_t block_base,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    int row) {
  CopyPackedTileRow64<kRowsPerTile>(
      packed_data,
      packed_row_bytes,
      source_row,
      packed_byte_offset,
      matmul_scales,
      block_base,
      padded_blocks_per_row,
      scale_layout,
      tile.packed_rows,
      tile.scale_words,
      row);
}

template <int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::CopyActivationRow64(
    const PackedTile64<kRowsPerTile>& tile,
    const std::uint8_t* packed_data,
    std::size_t packed_row_bytes,
    std::size_t source_row,
    std::size_t packed_byte_offset,
    const std::uint8_t* block_scales,
    std::size_t block_base,
    std::size_t blocks_per_row,
    int row) {
  CopyPackedTileRow64FromRowMajorScales<kRowsPerTile>(
      packed_data,
      packed_row_bytes,
      source_row,
      packed_byte_offset,
      block_scales,
      block_base,
      blocks_per_row,
      tile.packed_rows,
      tile.scale_words,
      row);
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

template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::ZeroTracedP5ScaleRow(
    ScaleTensor& scale_tensor,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
#pragma unroll
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    StoreScaleTensorByte(scale_tensor, row, scale_col, 0u);
  }
}

template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedP5ScaleWordK64(
    ScaleTensor& scale_tensor,
    std::uint32_t scale_word,
    int row) {
  if (row < 0 || row >= static_cast<int>(cute::size<0>(scale_tensor))) {
    return;
  }
#pragma unroll
  for (int scale_col = 0; scale_col < static_cast<int>(cute::size<1>(scale_tensor)); ++scale_col) {
    const std::uint8_t value = scale_col < 4 ? LoadScaleByte(scale_word, scale_col) : std::uint8_t{0};
    StoreScaleTensorByte(scale_tensor, row, scale_col, value);
  }
}

template <class ScaleTensor, int kScaleByteCount>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedScaleBytes(
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

template <class ScaleTensor>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedScaleWordsK128(
    ScaleTensor& scale_tensor,
    std::uint32_t scale_word_lo,
    std::uint32_t scale_word_hi,
    int row) {
  const std::uint8_t scale_bytes[8] = {
      LoadScaleByte(scale_word_lo, 0),
      LoadScaleByte(scale_word_lo, 1),
      LoadScaleByte(scale_word_lo, 2),
      LoadScaleByte(scale_word_lo, 3),
      LoadScaleByte(scale_word_hi, 0),
      LoadScaleByte(scale_word_hi, 1),
      LoadScaleByte(scale_word_hi, 2),
      LoadScaleByte(scale_word_hi, 3),
  };
  StoreTracedScaleBytes(scale_tensor, scale_bytes, row);
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int row_base,
    int thread_idx) {
  AFragment64 fragment{};
  int row_coords[kTiledCopyCoordCapacityA];
  int col_coords[kTiledCopyCoordCapacityA];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_a = cute::make_identity_tensor(
      cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::size<2>(typename TiledMma::AtomShape_MNK{})));
  auto part_a = thr_mma.partition_A(ref_a);
  auto smem_tiled_copy_a = MakeLocalTiledCopyA<cute::SM75_U32x4_LDSM_N>(mma);
  auto smem_thr_copy_a = smem_tiled_copy_a.get_thread_slice(thread_idx);
  auto copy_view_a = smem_thr_copy_a.retile_D(part_a);
  FillPhysicalCoordMapCopyViewLimited(copy_view_a, 32, row_coords, col_coords);
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    std::uint32_t packed = 0;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int physical = reg * 8 + elem;
      const int row = row_coords[physical];
      const int col = col_coords[physical];
      const std::uint8_t nibble =
          LoadPackedFp4Nibble(packed_rows + static_cast<std::size_t>(row_base + row) * (64 / 2), col);
      packed |= static_cast<std::uint32_t>(nibble) << (elem * 4);
    }
    fragment.regs[reg] = packed;
  }
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  std::uint32_t packed_scale = 0u;
  if (scale_words != nullptr) {
    auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
    auto smem_tiled_copy_sfa = MakeLocalTiledCopySFA(mma);
    auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
    auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
    int scale_rows[kTiledCopyCoordCapacityScale];
    int scale_cols[kTiledCopyCoordCapacityScale];
    FillPhysicalCoordMapCopyViewLimited(copy_view_sfa, 4, scale_rows, scale_cols);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int scale_row = scale_rows[i];
      const int scale_col = scale_cols[i];
      packed_scale |= static_cast<std::uint32_t>(LoadScaleByte(scale_words[scale_row], scale_col)) << (i * 8);
    }
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  ApplySm120Fp4ShiftA(fragment.regs[0], fragment.regs[1], fragment.regs[2], fragment.regs[3]);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64TracedScaleTiled(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int row_base,
    int thread_idx) {
  auto fragment =
      LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, row_base, thread_idx);
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP5SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP5TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP5ScaleFragmentCosizeA];
  int scale_cols[kTracedP5ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP5ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  const int fragment_index = row_base / 16;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int row_base,
    int lane_id) {
  return LoadFragmentA_RowMajor16x64Tiled<SingleAtomTiledMma, kRowsPerTile>(
      packed_rows,
      scale_words,
      row_base,
      lane_id);
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8Tiled(
    const std::uint8_t* packed_rows,
    const std::uint32_t* scale_words,
    int thread_idx,
    int n_base) {
  BFragment64 fragment{};
  int row_coords[kTiledCopyCoordCapacityB];
  int col_coords[kTiledCopyCoordCapacityB];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_b = cute::make_identity_tensor(
      cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::size<2>(typename TiledMma::AtomShape_MNK{})));
  auto part_b = thr_mma.partition_B(ref_b);
  auto smem_tiled_copy_b = MakeLocalTiledCopyB<cute::SM75_U32x4_LDSM_N>(mma);
  auto smem_thr_copy_b = smem_tiled_copy_b.get_thread_slice(thread_idx);
  auto copy_view_b = smem_thr_copy_b.retile_D(part_b);
  FillPhysicalCoordMapCopyViewLimited(copy_view_b, 16, row_coords, col_coords);
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    std::uint32_t packed = 0;
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const int physical = reg * 8 + elem;
      const int row = row_coords[physical];
      const int col = col_coords[physical];
      const std::uint8_t nibble =
          LoadPackedFp4Nibble(packed_rows + static_cast<std::size_t>(n_base + row) * (64 / 2), col);
      packed |= static_cast<std::uint32_t>(nibble) << (elem * 4);
    }
    fragment.regs[reg] = packed;
  }
  auto ref_sfb =
      cute::make_identity_tensor(cute::make_shape(cute::size<1>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  std::uint32_t packed_scale = 0u;
  if (scale_words != nullptr) {
    auto part_sfb = PartitionScaleB(ref_sfb, thr_mma);
    auto smem_tiled_copy_sfb = MakeLocalTiledCopySFB(mma);
    auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
    auto copy_view_sfb = smem_thr_copy_sfb.retile_D(part_sfb);
    int scale_rows[kTiledCopyCoordCapacityScale];
    int scale_cols[kTiledCopyCoordCapacityScale];
    FillPhysicalCoordMapCopyViewLimited(copy_view_sfb, 4, scale_rows, scale_cols);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int scale_row = scale_rows[i];
      const int scale_col = scale_cols[i];
      packed_scale |= static_cast<std::uint32_t>(LoadScaleByte(scale_words[n_base + scale_row], scale_col)) << (i * 8);
    }
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  ApplySm120Fp4ShiftB(fragment.regs[0], fragment.regs[1]);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8TracedScaleTiled(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int n_base) {
  auto fragment =
      LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, thread_idx, n_base);
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
  const int fragment_index = n_base / 8;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::BFragment64
nvfp4_bridge::LoadFragmentB_ColMajor64x8TracedScaleTiledP13(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int n_base) {
  auto fragment =
      LoadFragmentB_ColMajor64x8Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, thread_idx, n_base);
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
  const int fragment_index = n_base / 8;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFB(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ nvfp4_bridge::AFragment64
nvfp4_bridge::LoadFragmentA_RowMajor16x64TracedScaleTiledP13(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int row_base,
    int thread_idx) {
  auto fragment =
      LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(packed_rows, nullptr, row_base, thread_idx);
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP13SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP13TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP13ScaleFragmentCosizeA];
  int scale_cols[kTracedP13ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP13ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  const int fragment_index = row_base / 16;
  std::uint32_t packed_scale = 0u;
#pragma unroll
  for (int elem = 0; elem < 4; ++elem) {
    const int physical = fragment_index * 4 + elem;
    packed_scale |= static_cast<std::uint32_t>(
                        sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                    << (elem * 8);
  }
  fragment.scale[0] = static_cast<SFRegister>(packed_scale);
  return fragment;
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP5AFragmentsRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    AFragment64 (&fragments)[kTracedP5MFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP5SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP5TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP5ScaleFragmentCosizeA];
  int scale_cols[kTracedP5ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP5ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  for (int m_fragment = 0; m_fragment < kTracedP5MFragments; ++m_fragment) {
    fragments[m_fragment] =
        LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            m_fragment * 16,
            thread_idx);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = m_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[m_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, int kRowsPerTile>
__device__ __forceinline__ void nvfp4_bridge::LoadTracedP13AFragmentsRowMajor16x64(
    const std::uint8_t* packed_rows,
    const std::uint8_t* scale_smem,
    int thread_idx,
    int row_base,
    AFragment64 (&fragments)[kTracedP13MFragments]) {
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(const_cast<std::uint8_t*>(scale_smem)),
      TracedP13SmemLayoutSFA{});
  auto ref_sfa =
      cute::make_identity_tensor(cute::make_shape(cute::size<0>(typename TiledMma::AtomShape_MNK{}), cute::Int<4>{}));
  auto part_sfa = PartitionScaleA(ref_sfa, thr_mma);
  auto smem_tiled_copy_sfa = MakeTracedP13TiledCopySFA(mma);
  auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
  auto copy_view_sfa = smem_thr_copy_sfa.retile_D(part_sfa);
  int scale_rows[kTracedP13ScaleFragmentCosizeA];
  int scale_cols[kTracedP13ScaleFragmentCosizeA];
  FillPhysicalCoordMapCopyViewLimited(
      copy_view_sfa,
      kTracedP13ScaleFragmentCosizeA,
      scale_rows,
      scale_cols);
  for (int m_fragment = 0; m_fragment < kTracedP13MFragments; ++m_fragment) {
    fragments[m_fragment] =
        LoadFragmentA_RowMajor16x64Tiled<TiledMma, kRowsPerTile>(
            packed_rows,
            nullptr,
            row_base + m_fragment * 64,
            thread_idx);
    std::uint32_t packed_scale = 0;
#pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
      const int physical = m_fragment * 4 + elem;
      packed_scale |= static_cast<std::uint32_t>(
                          sSFA(scale_rows[physical], scale_cols[physical], cute::Int<0>{}))
                      << (elem * 8);
    }
    fragments[m_fragment].scale[0] = static_cast<SFRegister>(packed_scale);
  }
}

template <class TiledMma, typename OutputType>
__device__ __forceinline__ void nvfp4_bridge::StoreTracedP13CFragmentsRowMajor(
    float alpha,
    const CFragment64 (&accum)[kTracedP13MFragments][kTracedP13NFragments],
    int thread_idx,
    int output_row_base,
    int valid_rows,
    int output_rows_this_tile,
    std::size_t row_start,
    std::size_t output_rows_per_expert,
    OutputType* output) {
  int row_coords[kTracedP13CCopyCoordCapacity];
  int col_coords[kTracedP13CCopyCoordCapacity];
  auto mma = TiledMma{};
  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto ref_c = cute::make_identity_tensor(
      cute::make_shape(cute::tile_size<0>(mma), cute::tile_size<1>(mma)));
  auto part_c = thr_mma.partition_C(ref_c);
  FillPhysicalCoordMapCopyViewLimited(
      part_c,
      kTracedP13CCopyCoordCapacity,
      row_coords,
      col_coords);
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
#pragma unroll
    for (int m_fragment = 0; m_fragment < kTracedP13MFragments; ++m_fragment) {
#pragma unroll
      for (int n_fragment = 0; n_fragment < kTracedP13NFragments; ++n_fragment) {
        const int physical = n_fragment * 8 + m_fragment * 4 + reg;
        const int output_col_offset = row_coords[physical];
        const int token_row = col_coords[physical];
        if (token_row >= valid_rows || output_col_offset >= output_rows_this_tile) {
          continue;
        }
        const std::size_t input_row = row_start + static_cast<std::size_t>(token_row);
        const std::size_t output_col =
            static_cast<std::size_t>(output_row_base + output_col_offset);
        if constexpr (std::is_same_v<OutputType, __nv_bfloat16>) {
          output[input_row * output_rows_per_expert + output_col] =
              __float2bfloat16(accum[m_fragment][n_fragment].regs[reg] * alpha);
        } else {
          output[input_row * output_rows_per_expert + output_col] =
              accum[m_fragment][n_fragment].regs[reg] * alpha;
        }
      }
    }
  }
}

