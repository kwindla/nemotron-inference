
namespace wmma = nvcuda::wmma;

template <class Layout>
std::string LayoutString(Layout const& layout) {
  std::ostringstream oss;
  oss << layout;
  return oss.str();
}

__host__ __device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout);

__host__ __device__ std::size_t RoundUp(std::size_t value, std::size_t alignment);
bool LaunchZeroBuffer(float* data, std::size_t count);
bool LaunchZeroBf16Buffer(__nv_bfloat16* data, std::size_t count);
bool LaunchGatherRows(
    const float* input,
    const int* indices,
    const int* active_output_rows,
    std::size_t output_rows,
    std::size_t input_rows,
    std::size_t cols,
    float* output);
bool LaunchRoutedBf16Relu2Pack(
    const __nv_bfloat16* source,
    const DeviceExpertRouting* routing,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    DeviceNvfp4Matrix* output_pack,
    float* output_dequant_scales);
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
    __nv_bfloat16* output);

namespace cute = ::cute;

