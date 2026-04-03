#include "nemotron/linear_op.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "nemotron/runtime_stats.h"
#include "storage_conversion.h"

namespace nemotron {

struct UploadedLinearOp::Impl {
  GemmDescriptor descriptor;
  std::unique_ptr<DeviceDenseWeightFp32> dense_weight;
  std::unique_ptr<DeviceTensorBf16> dense_weight_bf16;
  std::unique_ptr<DeviceNvfp4Weight> nvfp4_weight;
  mutable std::unique_ptr<DeviceTensorFp32> dense_weight_fp32_storage_;
  mutable std::unique_ptr<DeviceTensorFp32> bf16_activation_scratch_;
  mutable std::unique_ptr<DeviceTensorFp32> bf16_output_scratch_;
  mutable std::mutex dense_rows1_plan_mutex;
  mutable bool dense_rows1_plan_attempted = false;
  mutable std::optional<CublasLtGemmPlan> dense_rows1_plan;
  mutable std::unique_ptr<CachedCublasLtMatmulState> dense_matmul_state_;
  mutable std::mutex dense_matmul_state_mutex_;
  mutable bool dense_matmul_state_attempted_ = false;
};

namespace {

std::optional<CublasLtGemmPlan> BuildRuntimeGemmPlan(
    const GemmDescriptor& descriptor,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache) {
  const auto launch_plan = BuildGemmLaunchPlan(descriptor, rows);
  if (!launch_plan.has_value()) {
    return std::nullopt;
  }
  const auto execution = PrepareGemmExecution(*launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    return std::nullopt;
  }
  return BuildCublasLtGemmPlan(*execution);
}

bool IsBf16StorageType(std::string_view storage_dtype) {
  return storage_dtype == "bf16" || storage_dtype == "bfloat16";
}

bool IsBf16DenseDescriptor(const GemmDescriptor& descriptor) {
  return descriptor.kernel_family == GemmKernelFamily::kDenseRowMajor &&
         descriptor.output_rows != 0 &&
         descriptor.input_cols != 0 &&
         descriptor.layout_tag == "row_major" &&
         !descriptor.is_scaled() &&
         descriptor.packed_bytes().valid() &&
         IsBf16StorageType(descriptor.storage_dtype) &&
         descriptor.packed_nbytes ==
             descriptor.output_rows * descriptor.input_cols * sizeof(__nv_bfloat16);
}

bool ExperimentalDenseDevicePlanSurfaceEnabled() {
  static const bool disabled =
      std::getenv("NEMOTRON_DISABLE_DENSE_DEVICE_PLAN_SURFACE") != nullptr;
  return !disabled;
}

const char* ExperimentalDenseDevicePlanSurfaceFamilyFilter() {
  static const char* const kFilter =
      std::getenv("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_FAMILY");
  return kFilter;
}

const char* ExperimentalDenseDevicePlanSurfaceTensorFilter() {
  static const char* const kFilter =
      std::getenv("NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_TENSORS");
  return kFilter;
}

DenseRuntimeOpFamily ClassifyDenseRuntimeOpFamily(const GemmDescriptor& descriptor) {
  const std::string_view op_class = descriptor.op_class;
  const std::string_view tensor_name = descriptor.tensor_name;
  const auto contains = [](std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
  };
  if (contains(op_class, "attention") ||
      contains(tensor_name, "q_proj") ||
      contains(tensor_name, "k_proj") ||
      contains(tensor_name, "v_proj") ||
      contains(tensor_name, "o_proj")) {
    return DenseRuntimeOpFamily::kAttention;
  }
  if (contains(op_class, "expert") ||
      contains(op_class, "moe") ||
      contains(tensor_name, "gate.weight") ||
      contains(tensor_name, "fc1_latent_proj.weight") ||
      contains(tensor_name, "fc2_latent_proj.weight") ||
      contains(tensor_name, "shared_expert") ||
      contains(tensor_name, ".experts.")) {
    return DenseRuntimeOpFamily::kExpert;
  }
  return DenseRuntimeOpFamily::kOther;
}

bool ExperimentalDenseDevicePlanSurfaceTensorMatches(const GemmDescriptor& descriptor) {
  const char* filter = ExperimentalDenseDevicePlanSurfaceTensorFilter();
  if (filter == nullptr || *filter == '\0') {
    return true;
  }

  std::stringstream stream(filter);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::size_t begin = token.find_first_not_of(" \t");
    if (begin == std::string::npos) {
      continue;
    }
    const std::size_t end = token.find_last_not_of(" \t");
    const std::string_view needle(token.data() + begin, end - begin + 1);
    if (!needle.empty() &&
        std::string_view(descriptor.tensor_name).find(needle) != std::string_view::npos) {
      return true;
    }
  }
  return false;
}

bool ExperimentalDenseDevicePlanSurfaceEnabledForDescriptor(
    const GemmDescriptor& descriptor,
    DenseRuntimeOpFamily family) {
  if (!ExperimentalDenseDevicePlanSurfaceEnabled()) {
    return false;
  }
  const char* filter = ExperimentalDenseDevicePlanSurfaceFamilyFilter();
  if (filter == nullptr || *filter == '\0') {
    return true;
  }
  const std::string_view filter_view(filter);
  if (filter_view == "all") {
    return true;
  }
  switch (family) {
    case DenseRuntimeOpFamily::kAttention:
      return filter_view == "attention";
    case DenseRuntimeOpFamily::kExpert:
      return filter_view == "expert";
    case DenseRuntimeOpFamily::kOther:
      if (filter_view != "other") {
        return false;
      }
      break;
  }
  return ExperimentalDenseDevicePlanSurfaceTensorMatches(descriptor);
}

std::unique_ptr<DeviceTensorBf16> UploadDenseWeightToDeviceBf16(const GemmDescriptor& descriptor) {
  if (!IsBf16DenseDescriptor(descriptor)) {
    return nullptr;
  }
  auto weight = DeviceTensorBf16::Create({descriptor.output_rows, descriptor.input_cols});
  if (!weight || !weight->CopyFromHost(
                     reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data),
                     descriptor.output_rows * descriptor.input_cols)) {
    return nullptr;
  }
  return weight;
}

GemmDescriptor BuildDenseRuntimeDescriptor(
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    DenseRuntimeOpFamily dense_family) {
  GemmDescriptor runtime_descriptor = descriptor;
  if (ExperimentalDenseDevicePlanSurfaceEnabledForDescriptor(descriptor, dense_family)) {
    if (dense_weight_bf16 != nullptr && dense_weight_bf16->valid()) {
      runtime_descriptor.storage_dtype = "bf16";
      runtime_descriptor.packed_data =
          reinterpret_cast<const std::uint8_t*>(dense_weight_bf16->data());
      runtime_descriptor.packed_nbytes =
          descriptor.output_rows * descriptor.input_cols * sizeof(__nv_bfloat16);
    } else if (dense_weight != nullptr && dense_weight->valid()) {
      runtime_descriptor.storage_dtype = "fp32";
      runtime_descriptor.packed_data =
          reinterpret_cast<const std::uint8_t*>(dense_weight->data());
      runtime_descriptor.packed_nbytes =
          descriptor.output_rows * descriptor.input_cols * sizeof(float);
    }
  }
  return runtime_descriptor;
}

template <typename ActivationTensorT>
std::optional<CublasLtGemmPlan> ResolveDensePlan(
    std::mutex& dense_rows1_plan_mutex,
    bool* dense_rows1_plan_attempted,
    std::optional<CublasLtGemmPlan>* dense_rows1_plan,
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    const ActivationTensorT& activations,
    GemmHeuristicCache* heuristic_cache,
    DenseRuntimeOpFamily dense_family) {
  const GemmDescriptor runtime_descriptor =
      BuildDenseRuntimeDescriptor(descriptor, dense_weight, dense_weight_bf16, dense_family);
  std::optional<CublasLtGemmPlan> plan;
  if (activations.shape().at(0) == 1) {
    std::lock_guard<std::mutex> lock(dense_rows1_plan_mutex);
    if (!*dense_rows1_plan_attempted) {
      *dense_rows1_plan =
          BuildRuntimeGemmPlan(runtime_descriptor, activations.shape().at(0), heuristic_cache);
      *dense_rows1_plan_attempted = true;
      if (!dense_rows1_plan->has_value()) {
        RecordDensePlanBuildFailure(dense_family);
      }
    } else if (dense_rows1_plan->has_value()) {
      RecordDensePlanCacheHit();
    }
    if (dense_rows1_plan->has_value()) {
      plan = *dense_rows1_plan;
    }
  } else {
    plan = BuildRuntimeGemmPlan(runtime_descriptor, activations.shape().at(0), heuristic_cache);
    if (!plan.has_value()) {
      RecordDensePlanBuildFailure(dense_family);
    }
  }
  return plan;
}

bool EnsureDenseWeightFp32(
    const GemmDescriptor& descriptor,
    const DeviceTensorBf16* dense_weight_bf16,
    std::unique_ptr<DeviceTensorFp32>* dense_weight_fp32_storage,
    std::unique_ptr<DeviceDenseWeightFp32>* dense_weight,
    cudaStream_t stream) {
  if (*dense_weight && (*dense_weight)->valid()) {
    return true;
  }
  if (dense_weight_bf16 != nullptr && dense_weight_bf16->valid()) {
    if (dense_weight_fp32_storage == nullptr) {
      return false;
    }
    const std::vector<std::size_t> shape = {descriptor.output_rows, descriptor.input_cols};
    if (!*dense_weight_fp32_storage ||
        !(*dense_weight_fp32_storage)->valid() ||
        (*dense_weight_fp32_storage)->shape() != shape) {
      *dense_weight_fp32_storage = DeviceTensorFp32::Create(shape);
    }
    if (!*dense_weight_fp32_storage ||
        !(*dense_weight_fp32_storage)->valid() ||
        !ConvertDeviceBf16ToFp32(
            dense_weight_bf16->data(),
            descriptor.output_rows * descriptor.input_cols,
            (*dense_weight_fp32_storage)->data(),
            stream)) {
      return false;
    }
    *dense_weight = DeviceDenseWeightFp32::CreateView(
        descriptor.output_rows,
        descriptor.input_cols,
        (*dense_weight_fp32_storage)->data());
    return *dense_weight && (*dense_weight)->valid();
  }
  if (descriptor.packed_data == nullptr) {
    return false;
  }
  *dense_weight = DeviceDenseWeightFp32::Upload(descriptor);
  return *dense_weight && (*dense_weight)->valid();
}

bool EnsureDenseWeightBf16(
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    std::unique_ptr<DeviceTensorBf16>* dense_weight_bf16,
    bool* dense_rows1_plan_attempted,
    std::optional<CublasLtGemmPlan>* dense_rows1_plan,
    std::mutex* dense_matmul_state_mutex,
    bool* dense_matmul_state_attempted,
    std::unique_ptr<CachedCublasLtMatmulState>* dense_matmul_state) {
  if (*dense_weight_bf16 && (*dense_weight_bf16)->valid()) {
    return true;
  }
  if (dense_weight == nullptr || !dense_weight->valid()) {
    return false;
  }
  auto converted = DeviceTensorBf16::Create({descriptor.output_rows, descriptor.input_cols});
  if (!converted ||
      !ConvertDeviceFp32ToBf16(
          dense_weight->data(),
          descriptor.output_rows * descriptor.input_cols,
          converted->data())) {
    return false;
  }
  *dense_weight_bf16 = std::move(converted);
  *dense_rows1_plan_attempted = false;
  dense_rows1_plan->reset();
  if (dense_matmul_state_mutex != nullptr &&
      dense_matmul_state_attempted != nullptr &&
      dense_matmul_state != nullptr) {
    std::lock_guard<std::mutex> lock(*dense_matmul_state_mutex);
    dense_matmul_state->reset();
    *dense_matmul_state_attempted = false;
  }
  return true;
}

DeviceTensorFp32* EnsureFp32Scratch(
    std::unique_ptr<DeviceTensorFp32>* scratch,
    const std::vector<std::size_t>& shape) {
  if (*scratch && (*scratch)->valid() && (*scratch)->shape() == shape) {
    return scratch->get();
  }
  *scratch = DeviceTensorFp32::Create(shape);
  if (!*scratch || !(*scratch)->valid()) {
    return nullptr;
  }
  return scratch->get();
}

bool CheckCudaStatus(cudaError_t status) {
  return status == cudaSuccess;
}

bool CheckCublasStatus(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS;
}

cudaDataType_t DenseTensorDataType(const DeviceDenseWeightFp32&) {
  return CUDA_R_32F;
}

cudaDataType_t DenseTensorDataType(const DeviceTensorFp32&) {
  return CUDA_R_32F;
}

cudaDataType_t DenseTensorDataType(const DeviceTensorBf16&) {
  return CUDA_R_16BF;
}

cublasOperation_t ToCublasOp(CublasLtTransform transform) {
  switch (transform) {
    case CublasLtTransform::kNone:
      return CUBLAS_OP_N;
    case CublasLtTransform::kTranspose:
      return CUBLAS_OP_T;
  }
  return CUBLAS_OP_N;
}

cublasLtOrder_t ToCublasOrder(CublasLtMatrixOrder order) {
  switch (order) {
    case CublasLtMatrixOrder::kRowMajor:
      return CUBLASLT_ORDER_ROW;
    case CublasLtMatrixOrder::kColumnMajor:
      return CUBLASLT_ORDER_COL;
  }
  return CUBLASLT_ORDER_ROW;
}

struct MatrixLayoutShape {
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  std::int64_t ld = 0;
};

MatrixLayoutShape ActivationLayoutShape(
    const CublasLtGemmPlan& plan,
    std::size_t m,
    std::size_t k) {
  switch (plan.contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.lda)};
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(k),
          static_cast<std::uint64_t>(m),
          static_cast<std::int64_t>(plan.lda)};
  }
  return {};
}

MatrixLayoutShape WeightLayoutShape(
    const CublasLtGemmPlan& plan,
    std::size_t n,
    std::size_t k) {
  switch (plan.contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(n),
          static_cast<std::uint64_t>(k),
          static_cast<std::int64_t>(plan.ldb)};
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(k),
          static_cast<std::uint64_t>(n),
          static_cast<std::int64_t>(plan.ldb)};
  }
  return {};
}

MatrixLayoutShape OutputLayoutShape(
    const CublasLtGemmPlan& plan,
    std::size_t m,
    std::size_t n) {
  switch (plan.contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(m),
          static_cast<std::uint64_t>(n),
          static_cast<std::int64_t>(plan.ldc)};
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      return MatrixLayoutShape{
          static_cast<std::uint64_t>(n),
          static_cast<std::uint64_t>(m),
          static_cast<std::int64_t>(plan.ldc)};
  }
  return {};
}

bool BuildDenseCachedCublasLtMatmulState(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    cudaDataType_t activations_type,
    cudaDataType_t weights_type,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    cudaDataType_t output_type,
    CachedCublasLtMatmulState* state) {
  if (!handle.valid() || state == nullptr) {
    return false;
  }

  *state = CachedCublasLtMatmulState{};
  state->activations_type = activations_type;
  state->weights_type = weights_type;
  state->output_type = output_type;
  state->alpha_scale = 1.0f;
  state->fast_accum = false;
  state->m = m;
  state->n = n;
  state->k = k;

  cublasLtMatmulPreference_t preference = nullptr;
  int returned_results = 0;
  bool ok = true;
  const MatrixLayoutShape a_shape = ActivationLayoutShape(plan, m, k);
  const MatrixLayoutShape b_shape = WeightLayoutShape(plan, n, k);
  const MatrixLayoutShape c_shape = OutputLayoutShape(plan, m, n);
  const cublasOperation_t trans_a = ToCublasOp(plan.transform_a);
  const cublasOperation_t trans_b = ToCublasOp(plan.transform_b);
  const cublasLtOrder_t order_a = ToCublasOrder(plan.order_a);
  const cublasLtOrder_t order_b = ToCublasOrder(plan.order_b);
  const cublasLtOrder_t order_c = ToCublasOrder(plan.order_c);

  ok = CheckCublasStatus(
           cublasLtMatmulDescCreate(&state->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)) &&
       CheckCublasStatus(
           cublasLtMatmulDescSetAttribute(
               state->op_desc,
               CUBLASLT_MATMUL_DESC_TRANSA,
               &trans_a,
               sizeof(trans_a))) &&
       CheckCublasStatus(
           cublasLtMatmulDescSetAttribute(
               state->op_desc,
               CUBLASLT_MATMUL_DESC_TRANSB,
               &trans_b,
               sizeof(trans_b))) &&
       CheckCublasStatus(
           cublasLtMatrixLayoutCreate(
               &state->a_desc,
               activations_type,
               a_shape.rows,
               a_shape.cols,
               a_shape.ld)) &&
       CheckCublasStatus(
           cublasLtMatrixLayoutCreate(
               &state->b_desc,
               weights_type,
               b_shape.rows,
               b_shape.cols,
               b_shape.ld)) &&
       CheckCublasStatus(
           cublasLtMatrixLayoutCreate(
               &state->c_desc,
               output_type,
               c_shape.rows,
               c_shape.cols,
               c_shape.ld)) &&
       CheckCublasStatus(
           cublasLtMatrixLayoutSetAttribute(
               state->a_desc,
               CUBLASLT_MATRIX_LAYOUT_ORDER,
               &order_a,
               sizeof(order_a))) &&
       CheckCublasStatus(
           cublasLtMatrixLayoutSetAttribute(
               state->b_desc,
               CUBLASLT_MATRIX_LAYOUT_ORDER,
               &order_b,
               sizeof(order_b))) &&
       CheckCublasStatus(
           cublasLtMatrixLayoutSetAttribute(
               state->c_desc,
               CUBLASLT_MATRIX_LAYOUT_ORDER,
               &order_c,
               sizeof(order_c))) &&
       CheckCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

  if (ok) {
    const std::size_t max_workspace = handle.workspace_bytes();
    ok = CheckCublasStatus(
             cublasLtMatmulPreferenceSetAttribute(
                 preference,
                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                 &max_workspace,
                 sizeof(max_workspace))) &&
         CheckCublasStatus(
             cublasLtMatmulAlgoGetHeuristic(
                 handle.handle(),
                 state->op_desc,
                 state->a_desc,
                 state->b_desc,
                 state->c_desc,
                 state->c_desc,
                 preference,
                 1,
                 &state->heuristic,
                 &returned_results)) &&
         returned_results > 0;
  }

  if (preference != nullptr) {
    cublasLtMatmulPreferenceDestroy(preference);
  }
  if (!ok) {
    *state = CachedCublasLtMatmulState{};
    return false;
  }

  state->heuristic_count = returned_results;
  return true;
}

bool DenseWeightMatchesPlan(const DeviceDenseWeightFp32& weights, std::size_t n, std::size_t k) {
  return weights.valid() && weights.output_rows() == n && weights.input_cols() == k;
}

bool DenseWeightMatchesPlan(const DeviceTensorBf16& weights, std::size_t n, std::size_t k) {
  return weights.valid() &&
         weights.shape().size() == 2 &&
         weights.shape()[0] == n &&
         weights.shape()[1] == k;
}

template <typename WeightTensorT, typename ActivationTensorT, typename OutputTensorT>
std::unique_ptr<CachedCublasLtMatmulState> CreateDenseRowMajorMatmulState(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const WeightTensorT& weights,
    const ActivationTensorT& activations,
    const OutputTensorT& output) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      !output.valid() ||
      activations.shape().size() != 2 ||
      output.shape().size() != 2) {
    return nullptr;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = plan.execution.launch_plan.n;
  const std::size_t k = plan.execution.launch_plan.k;
  if (activations.shape()[1] != k ||
      output.shape()[0] != m ||
      output.shape()[1] != n ||
      !DenseWeightMatchesPlan(weights, n, k)) {
    return nullptr;
  }

  auto state = std::make_unique<CachedCublasLtMatmulState>();
  if (!BuildDenseCachedCublasLtMatmulState(
          handle,
          plan,
          DenseTensorDataType(activations),
          DenseTensorDataType(weights),
          m,
          n,
          k,
          DenseTensorDataType(output),
          state.get())) {
    return nullptr;
  }
  return state;
}

template <typename WeightTensorT, typename ActivationTensorT, typename OutputTensorT>
const CachedCublasLtMatmulState* ResolveDenseMatmulState(
    std::mutex* dense_matmul_state_mutex,
    bool* dense_matmul_state_attempted,
    std::unique_ptr<CachedCublasLtMatmulState>* dense_matmul_state,
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const WeightTensorT& weights,
    const ActivationTensorT& activations,
    const OutputTensorT& output,
    const GemmDescriptor& descriptor,
    bool debug) {
  if (dense_matmul_state_mutex == nullptr ||
      dense_matmul_state_attempted == nullptr ||
      dense_matmul_state == nullptr ||
      activations.shape().empty() ||
      activations.shape()[0] != 1) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(*dense_matmul_state_mutex);
  if (!*dense_matmul_state_attempted) {
    *dense_matmul_state =
        CreateDenseRowMajorMatmulState(handle, plan, weights, activations, output);
    *dense_matmul_state_attempted = true;
    if (!*dense_matmul_state && debug) {
      std::cerr << "linear_op: dense matmul-state cache build failed for "
                << descriptor.tensor_name << "\n";
    }
  }
  return dense_matmul_state->get();
}

template <typename WeightTensorT, typename ActivationTensorT, typename OutputTensorT>
std::optional<DenseRowMajorDeviceStats> RunDenseWithCachedMatmulState(
    CublasLtHandle& handle,
    const WeightTensorT& weights,
    const ActivationTensorT& activations,
    OutputTensorT* output,
    const CachedCublasLtMatmulState* cached_matmul_state,
    cudaStream_t stream) {
  if (!handle.valid() ||
      !weights.valid() ||
      !activations.valid() ||
      output == nullptr ||
      !output->valid() ||
      cached_matmul_state == nullptr ||
      activations.shape().size() != 2 ||
      output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t m = activations.shape()[0];
  const std::size_t n = cached_matmul_state->n;
  const std::size_t k = cached_matmul_state->k;
  if (activations.shape()[1] != k ||
      output->shape()[0] != m ||
      output->shape()[1] != n ||
      !DenseWeightMatchesPlan(weights, n, k) ||
      cached_matmul_state->op_desc == nullptr ||
      cached_matmul_state->a_desc == nullptr ||
      cached_matmul_state->b_desc == nullptr ||
      cached_matmul_state->c_desc == nullptr ||
      cached_matmul_state->activations_type != DenseTensorDataType(activations) ||
      cached_matmul_state->weights_type != DenseTensorDataType(weights) ||
      cached_matmul_state->output_type != DenseTensorDataType(*output) ||
      cached_matmul_state->activations_scale_data != nullptr ||
      cached_matmul_state->weights_scale_data != nullptr ||
      cached_matmul_state->alpha_scale != 1.0f ||
      cached_matmul_state->fast_accum ||
      cached_matmul_state->m != m) {
    return std::nullopt;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (!CheckCublasStatus(
          cublasLtMatmul(
              handle.handle(),
              cached_matmul_state->op_desc,
              &alpha,
              activations.data(),
              cached_matmul_state->a_desc,
              weights.data(),
              cached_matmul_state->b_desc,
              &beta,
              output->data(),
              cached_matmul_state->c_desc,
              output->data(),
              cached_matmul_state->c_desc,
              &cached_matmul_state->heuristic.algo,
              handle.workspace(),
              handle.workspace_bytes(),
              stream)) ||
      !CheckCudaStatus(cudaGetLastError())) {
    return std::nullopt;
  }

  return DenseRowMajorDeviceStats{
      m,
      n,
      cached_matmul_state->heuristic.workspaceSize,
      cached_matmul_state->heuristic_count,
  };
}

template <typename ActivationTensorT, typename OutputTensorT>
std::optional<DenseRowMajorDeviceStats> RunDenseNativeDispatch(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceTensorBf16& weights,
    const ActivationTensorT& activations,
    OutputTensorT* output,
    cudaStream_t stream) {
  if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorFp32> &&
                std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorBf16ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorBf16ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorBf16>) {
    return RunDenseRowMajorBf16ToDevice(handle, plan, weights, activations, output, stream);
  } else {
    return std::nullopt;
  }
}

template <typename ActivationTensorT, typename OutputTensorT>
std::optional<DenseRowMajorDeviceStats> RunDenseNativeDispatch(
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const DeviceDenseWeightFp32& weights,
    const ActivationTensorT& activations,
    OutputTensorT* output,
    cudaStream_t stream) {
  if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorFp32> &&
                std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorFp32ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorFp32ToDevice(handle, plan, weights, activations, output, stream);
  } else if constexpr (std::is_same_v<ActivationTensorT, DeviceTensorBf16> &&
                       std::is_same_v<OutputTensorT, DeviceTensorBf16>) {
    return RunDenseRowMajorFp32ToDevice(handle, plan, weights, activations, output, stream);
  } else {
    return std::nullopt;
  }
}

template <typename ActivationTensorT, typename OutputTensorT>
bool TryRunDenseNative(
    const GemmDescriptor& descriptor,
    const DeviceDenseWeightFp32* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    std::mutex* dense_matmul_state_mutex,
    bool* dense_matmul_state_attempted,
    std::unique_ptr<CachedCublasLtMatmulState>* dense_matmul_state,
    CublasLtHandle& handle,
    const CublasLtGemmPlan& plan,
    const ActivationTensorT& activations,
    OutputTensorT* output,
    DenseRuntimeOpFamily dense_family,
    bool debug,
    cudaStream_t stream) {
  if (dense_weight_bf16 != nullptr && dense_weight_bf16->valid()) {
    const CachedCublasLtMatmulState* cached_matmul_state = ResolveDenseMatmulState(
        dense_matmul_state_mutex,
        dense_matmul_state_attempted,
        dense_matmul_state,
        handle,
        plan,
        *dense_weight_bf16,
        activations,
        *output,
        descriptor,
        debug);
    if (const auto cached_stats = RunDenseWithCachedMatmulState(
            handle,
            *dense_weight_bf16,
            activations,
            output,
            cached_matmul_state,
            stream);
        cached_stats.has_value()) {
      RecordDenseNativeSuccess(dense_family);
      return true;
    }
    if (const auto stats = RunDenseNativeDispatch(
            handle, plan, *dense_weight_bf16, activations, output, stream);
        stats.has_value()) {
      RecordDenseNativeSuccess(dense_family);
      return true;
    }
    RecordDenseBf16NativeFailure();
    if (debug) {
      std::cerr << "linear_op: dense BF16 cuBLASLt path failed for "
                << descriptor.tensor_name << "\n";
    }
    return false;
  }
  if (dense_weight != nullptr && dense_weight->valid()) {
    const CachedCublasLtMatmulState* cached_matmul_state = ResolveDenseMatmulState(
        dense_matmul_state_mutex,
        dense_matmul_state_attempted,
        dense_matmul_state,
        handle,
        plan,
        *dense_weight,
        activations,
        *output,
        descriptor,
        debug);
    if (const auto cached_stats = RunDenseWithCachedMatmulState(
            handle,
            *dense_weight,
            activations,
            output,
            cached_matmul_state,
            stream);
        cached_stats.has_value()) {
      RecordDenseNativeSuccess(dense_family);
      return true;
    }
    if (const auto stats =
            RunDenseNativeDispatch(handle, plan, *dense_weight, activations, output, stream);
        stats.has_value()) {
      RecordDenseNativeSuccess(dense_family);
      return true;
    }
    RecordDenseFp32NativeFailure();
    if (debug) {
      std::cerr << "linear_op: dense FP32-weight cuBLASLt path failed for "
                << descriptor.tensor_name << "\n";
    }
  }
  return false;
}

template <typename OutputTensorT>
bool RunDenseReferenceFallback(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceDenseWeightFp32>* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    std::unique_ptr<DeviceTensorFp32>* dense_weight_fp32_storage,
    const DeviceTensorFp32& activations,
    OutputTensorT* output,
    std::unique_ptr<DeviceTensorFp32>* output_fp32_scratch,
    cudaStream_t stream) {
  if (!EnsureDenseWeightFp32(
          descriptor,
          dense_weight_bf16,
          dense_weight_fp32_storage,
          dense_weight,
          stream)) {
    return false;
  }
  if constexpr (std::is_same_v<OutputTensorT, DeviceTensorFp32>) {
    return RunDenseRowMajorFp32ReferenceToDevice(**dense_weight, activations, output).has_value();
  } else {
    DeviceTensorFp32* output_fp32 = EnsureFp32Scratch(output_fp32_scratch, output->shape());
    return output_fp32 != nullptr &&
           RunDenseRowMajorFp32ReferenceToDevice(
               **dense_weight,
               activations,
               output_fp32)
               .has_value() &&
           ConvertDeviceFp32ToBf16(
               output_fp32->data(),
               output_fp32->numel(),
               output->data(),
               stream);
  }
}

template <typename OutputTensorT>
bool RunDenseReferenceFallback(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceDenseWeightFp32>* dense_weight,
    const DeviceTensorBf16* dense_weight_bf16,
    std::unique_ptr<DeviceTensorFp32>* dense_weight_fp32_storage,
    const DeviceTensorBf16& activations,
    OutputTensorT* output,
    std::unique_ptr<DeviceTensorFp32>* activations_fp32_scratch,
    std::unique_ptr<DeviceTensorFp32>* output_fp32_scratch,
    cudaStream_t stream) {
  DeviceTensorFp32* activations_fp32 =
      EnsureFp32Scratch(activations_fp32_scratch, activations.shape());
  if (activations_fp32 == nullptr ||
      !ConvertDeviceBf16ToFp32(
          activations.data(),
          activations.numel(),
          activations_fp32->data(),
          stream)) {
    return false;
  }
  return RunDenseReferenceFallback(
      descriptor,
      dense_weight,
      dense_weight_bf16,
      dense_weight_fp32_storage,
      *activations_fp32,
      output,
      output_fp32_scratch,
      stream);
}

}  // namespace

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::Create(const GemmDescriptor& descriptor) {
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  switch (descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      if (IsBf16DenseDescriptor(descriptor)) {
        impl->dense_weight_bf16 = UploadDenseWeightToDeviceBf16(descriptor);
        if (!impl->dense_weight_bf16 || !impl->dense_weight_bf16->valid()) {
          return nullptr;
        }
      } else {
        impl->dense_weight = DeviceDenseWeightFp32::Upload(descriptor);
        if (!impl->dense_weight || !impl->dense_weight->valid()) {
          return nullptr;
        }
      }
      break;
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      impl->nvfp4_weight = DeviceNvfp4Weight::Upload(descriptor);
      if (!impl->nvfp4_weight || !impl->nvfp4_weight->valid()) {
        return nullptr;
      }
      break;
  }
  return std::unique_ptr<UploadedLinearOp>(new UploadedLinearOp(std::move(impl)));
}

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::CreateDenseView(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceDenseWeightFp32> weight_view) {
  if (descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor ||
      !weight_view || !weight_view->valid()) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  impl->dense_weight = std::move(weight_view);
  return std::unique_ptr<UploadedLinearOp>(new UploadedLinearOp(std::move(impl)));
}

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::CreateDenseBf16View(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceTensorBf16> weight_view) {
  if (descriptor.kernel_family != GemmKernelFamily::kDenseRowMajor ||
      !weight_view || !weight_view->valid()) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  impl->descriptor.storage_dtype = "bf16";
  impl->dense_weight_bf16 = std::move(weight_view);
  return std::unique_ptr<UploadedLinearOp>(new UploadedLinearOp(std::move(impl)));
}

std::unique_ptr<UploadedLinearOp> UploadedLinearOp::CreateNvfp4View(
    const GemmDescriptor& descriptor,
    std::unique_ptr<DeviceNvfp4Weight> weight_view) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      !weight_view || !weight_view->valid()) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->descriptor = descriptor;
  impl->nvfp4_weight = std::move(weight_view);
  return std::unique_ptr<UploadedLinearOp>(new UploadedLinearOp(std::move(impl)));
}

UploadedLinearOp::UploadedLinearOp(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

UploadedLinearOp::UploadedLinearOp(UploadedLinearOp&&) noexcept = default;
UploadedLinearOp& UploadedLinearOp::operator=(UploadedLinearOp&&) noexcept = default;
UploadedLinearOp::~UploadedLinearOp() = default;

bool UploadedLinearOp::valid() const {
  if (!impl_) {
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor:
      return (impl_->dense_weight && impl_->dense_weight->valid()) ||
             (impl_->dense_weight_bf16 && impl_->dense_weight_bf16->valid());
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      return impl_->nvfp4_weight && impl_->nvfp4_weight->valid();
  }
  return false;
}

std::size_t UploadedLinearOp::output_rows() const {
  return impl_ ? impl_->descriptor.output_rows : 0;
}

std::size_t UploadedLinearOp::input_cols() const {
  return impl_ ? impl_->descriptor.input_cols : 0;
}

GemmKernelFamily UploadedLinearOp::kernel_family() const {
  return impl_ ? impl_->descriptor.kernel_family : GemmKernelFamily::kDenseRowMajor;
}

const DeviceNvfp4Weight* UploadedLinearOp::nvfp4_weight() const {
  return impl_ ? impl_->nvfp4_weight.get() : nullptr;
}

bool UploadedLinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) const {
  static const bool kDebug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool debug = kDebug;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    if (debug) {
      std::cerr << "linear_op: invalid run state for " << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      const DenseRuntimeOpFamily dense_family = ClassifyDenseRuntimeOpFamily(impl_->descriptor);
      const auto plan = ResolveDensePlan(
          impl_->dense_rows1_plan_mutex,
          &impl_->dense_rows1_plan_attempted,
          &impl_->dense_rows1_plan,
          impl_->descriptor,
          impl_->dense_weight.get(),
          impl_->dense_weight_bf16.get(),
          activations,
          heuristic_cache,
          dense_family);
      if (plan.has_value() &&
          TryRunDenseNative(
              impl_->descriptor,
              impl_->dense_weight.get(),
              impl_->dense_weight_bf16.get(),
              &impl_->dense_matmul_state_mutex_,
              &impl_->dense_matmul_state_attempted_,
              &impl_->dense_matmul_state_,
              handle,
              *plan,
              activations,
              output,
              dense_family,
              debug,
              stream)) {
        return true;
      }
      if (!plan.has_value() && debug) {
        std::cerr << "linear_op: dense plan build failed for " << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      } else if (debug) {
        std::cerr << "linear_op: dense cuBLASLt path failed for "
                  << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      }
      RecordDenseReferenceFallback(dense_family);
      return RunDenseReferenceFallback(
          impl_->descriptor,
          &impl_->dense_weight,
          impl_->dense_weight_bf16.get(),
          &impl_->dense_weight_fp32_storage_,
          activations,
          output,
          &impl_->bf16_output_scratch_,
          stream);
    }
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled:
      {
        const auto plan = BuildRuntimeGemmPlan(impl_->descriptor, activations.shape().at(0), heuristic_cache);
        if (!plan.has_value()) {
          if (debug) {
            std::cerr << "linear_op: plan build failed for NVFP4 op "
                      << impl_->descriptor.tensor_name << "\n";
          }
          return false;
        }
        return RunNvfp4RowMajorFp32SourceToDevice(
                   handle,
                   *plan,
                   activations,
                   *impl_->nvfp4_weight,
                   output,
                   {},
                   stream)
            .has_value();
      }
  }
  return false;
}

bool UploadedLinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorBf16& activations,
    DeviceTensorFp32* output,
    cudaStream_t stream) const {
  static const bool kDebug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool debug = kDebug;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    if (debug) {
      std::cerr << "linear_op: invalid BF16->FP32 run state for "
                << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      if ((!impl_->dense_weight_bf16 || !impl_->dense_weight_bf16->valid()) &&
          impl_->dense_weight != nullptr &&
          impl_->dense_weight->valid()) {
        DeviceTensorFp32* activations_fp32 =
            EnsureFp32Scratch(&impl_->bf16_activation_scratch_, activations.shape());
        if (activations_fp32 == nullptr ||
            !ConvertDeviceBf16ToFp32(
                activations.data(),
                activations.numel(),
                activations_fp32->data(),
                stream)) {
          return false;
        }
        return Run(handle, heuristic_cache, *activations_fp32, output, stream);
      }
      const DenseRuntimeOpFamily dense_family = ClassifyDenseRuntimeOpFamily(impl_->descriptor);
      const auto plan = ResolveDensePlan(
          impl_->dense_rows1_plan_mutex,
          &impl_->dense_rows1_plan_attempted,
          &impl_->dense_rows1_plan,
          impl_->descriptor,
          impl_->dense_weight.get(),
          impl_->dense_weight_bf16.get(),
          activations,
          heuristic_cache,
          dense_family);
      if (plan.has_value() &&
          TryRunDenseNative(
              impl_->descriptor,
              impl_->dense_weight.get(),
              impl_->dense_weight_bf16.get(),
              &impl_->dense_matmul_state_mutex_,
              &impl_->dense_matmul_state_attempted_,
              &impl_->dense_matmul_state_,
              handle,
              *plan,
              activations,
              output,
              dense_family,
              debug,
              stream)) {
        return true;
      }
      if (!plan.has_value() && debug) {
        std::cerr << "linear_op: dense BF16-input plan build failed for "
                  << impl_->descriptor.tensor_name << "\n";
      } else if (debug) {
        std::cerr << "linear_op: dense BF16-input cuBLASLt path failed for "
                  << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      }
      RecordDenseReferenceFallback(dense_family);
      return RunDenseReferenceFallback(
          impl_->descriptor,
          &impl_->dense_weight,
          impl_->dense_weight_bf16.get(),
          &impl_->dense_weight_fp32_storage_,
          activations,
          output,
          &impl_->bf16_activation_scratch_,
          &impl_->bf16_output_scratch_,
          stream);
    }
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled: {
      DeviceTensorFp32* activations_fp32 =
          EnsureFp32Scratch(&impl_->bf16_activation_scratch_, activations.shape());
      if (activations_fp32 == nullptr ||
          !ConvertDeviceBf16ToFp32(
              activations.data(),
              activations.numel(),
              activations_fp32->data(),
              stream)) {
        return false;
      }
      const auto plan = BuildRuntimeGemmPlan(impl_->descriptor, activations.shape().at(0), heuristic_cache);
      if (!plan.has_value()) {
        if (debug) {
          std::cerr << "linear_op: plan build failed for BF16-input NVFP4 op "
                    << impl_->descriptor.tensor_name << "\n";
        }
        return false;
      }
      return RunNvfp4RowMajorFp32SourceToDevice(
                 handle,
                 *plan,
                 *activations_fp32,
                 *impl_->nvfp4_weight,
                 output,
                 {},
                 stream)
          .has_value();
    }
  }
  return false;
}

bool UploadedLinearOp::Run(
    CublasLtHandle& handle,
    GemmHeuristicCache* heuristic_cache,
    const DeviceTensorBf16& activations,
    DeviceTensorBf16* output,
    cudaStream_t stream) const {
  static const bool kDebug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool debug = kDebug;
  if (!valid() || !handle.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    if (debug) {
      std::cerr << "linear_op: invalid BF16->BF16 run state for "
                << impl_->descriptor.tensor_name << "\n";
    }
    return false;
  }
  switch (impl_->descriptor.kernel_family) {
    case GemmKernelFamily::kDenseRowMajor: {
      if (!EnsureDenseWeightBf16(
              impl_->descriptor,
              impl_->dense_weight.get(),
              &impl_->dense_weight_bf16,
              &impl_->dense_rows1_plan_attempted,
              &impl_->dense_rows1_plan,
              &impl_->dense_matmul_state_mutex_,
              &impl_->dense_matmul_state_attempted_,
              &impl_->dense_matmul_state_) &&
          debug &&
          impl_->dense_weight != nullptr &&
          impl_->dense_weight->valid()) {
        std::cerr << "linear_op: failed to materialize BF16 dense weight for "
                  << impl_->descriptor.tensor_name << "\n";
      }
      const DenseRuntimeOpFamily dense_family = ClassifyDenseRuntimeOpFamily(impl_->descriptor);
      const auto plan = ResolveDensePlan(
          impl_->dense_rows1_plan_mutex,
          &impl_->dense_rows1_plan_attempted,
          &impl_->dense_rows1_plan,
          impl_->descriptor,
          impl_->dense_weight.get(),
          impl_->dense_weight_bf16.get(),
          activations,
          heuristic_cache,
          dense_family);
      if (plan.has_value() &&
          TryRunDenseNative(
              impl_->descriptor,
              impl_->dense_weight.get(),
              impl_->dense_weight_bf16.get(),
              &impl_->dense_matmul_state_mutex_,
              &impl_->dense_matmul_state_attempted_,
              &impl_->dense_matmul_state_,
              handle,
              *plan,
              activations,
              output,
              dense_family,
              debug,
              stream)) {
        return true;
      }
      if (!plan.has_value() && debug) {
        std::cerr << "linear_op: dense BF16 surface plan build failed for "
                  << impl_->descriptor.tensor_name << "\n";
      } else if (debug) {
        std::cerr << "linear_op: dense BF16 surface cuBLASLt path failed for "
                  << impl_->descriptor.tensor_name
                  << ", falling back to device reference\n";
      }
      RecordDenseReferenceFallback(dense_family);
      return RunDenseReferenceFallback(
          impl_->descriptor,
          &impl_->dense_weight,
          impl_->dense_weight_bf16.get(),
          &impl_->dense_weight_fp32_storage_,
          activations,
          output,
          &impl_->bf16_activation_scratch_,
          &impl_->bf16_output_scratch_,
          stream);
    }
    case GemmKernelFamily::kCublasLtNvfp4BlockScaled: {
      DeviceTensorFp32* output_fp32 =
          EnsureFp32Scratch(&impl_->bf16_output_scratch_, output->shape());
      if (output_fp32 == nullptr) {
        return false;
      }
      if (!Run(handle, heuristic_cache, activations, output_fp32, stream)) {
        return false;
      }
      return ConvertDeviceFp32ToBf16(
          output_fp32->data(),
          output_fp32->numel(),
          output->data(),
          stream);
    }
  }
  return false;
}

std::optional<std::vector<float>> ReadVectorWeightToHostFp32(
    const KernelTensorDescriptor& descriptor) {
  if (descriptor.logical_shape.size() != 1 || descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  const std::size_t count = descriptor.logical_shape[0];
  if (count == 0) {
    return std::nullopt;
  }

  std::vector<float> values(count, 0.0f);
  if (descriptor.storage_dtype == "fp32") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return values;
  }
  if (descriptor.storage_dtype == "bf16") {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return std::nullopt;
    }
    const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = __bfloat162float(src[i]);
    }
    return values;
  }
  return std::nullopt;
}

std::unique_ptr<DeviceTensorFp32> UploadVectorWeightToDeviceFp32(
    const KernelTensorDescriptor& descriptor) {
  if (descriptor.logical_shape.size() != 1 || descriptor.packed_data == nullptr) {
    return nullptr;
  }
  const std::size_t count = descriptor.logical_shape[0];
  if (count == 0) {
    return nullptr;
  }

  PackedFloatStorage storage;
  if (descriptor.storage_dtype == "fp32") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return nullptr;
    }
    storage = PackedFloatStorage::kFp32;
  } else if (descriptor.storage_dtype == "bf16") {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return nullptr;
    }
    storage = PackedFloatStorage::kBf16;
  } else {
    return nullptr;
  }

  auto device = DeviceTensorFp32::Create({count});
  if (!device ||
      !UploadPackedFloatToDeviceFp32(
          descriptor.packed_data,
          count,
          storage,
          1.0f,
          device->data())) {
    return nullptr;
  }
  return device;
}

}  // namespace nemotron
