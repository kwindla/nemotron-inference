#include "nemotron/request_context.h"

#include <limits>
#include <utility>

#include "nemotron/routed_expert_runtime.h"

namespace nemotron {
namespace {

constexpr std::size_t kMoeNvfp4ScaleBlockWidth = 16;

std::optional<std::size_t> CheckedMul(std::size_t lhs, std::size_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return std::size_t{0};
  }
  if (lhs > (std::numeric_limits<std::size_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

std::optional<std::size_t> CheckedAdd(std::size_t lhs, std::size_t rhs) {
  if (lhs > (std::numeric_limits<std::size_t>::max() - rhs)) {
    return std::nullopt;
  }
  return lhs + rhs;
}

std::optional<std::size_t> MatrixBytes(
    std::size_t rows,
    std::size_t cols,
    std::size_t element_bytes) {
  const auto numel = CheckedMul(rows, cols);
  if (!numel.has_value()) {
    return std::nullopt;
  }
  return CheckedMul(*numel, element_bytes);
}

std::optional<std::size_t> Nvfp4MatrixBytes(
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout) {
  std::size_t total = 0;
  const auto add_bytes = [&](std::size_t bytes) -> bool {
    const auto next_total = CheckedAdd(total, bytes);
    if (!next_total.has_value()) {
      return false;
    }
    total = *next_total;
    return true;
  };
  return add_bytes(PackedFp4Bytes(rows, cols)) &&
                 add_bytes(RowMajorNvfp4ScaleBytes(rows, cols)) &&
                 add_bytes(ExecutionNvfp4ScaleBytes(rows, cols, scale_layout)) &&
                 add_bytes(sizeof(float))
             ? std::optional<std::size_t>(total)
             : std::nullopt;
}

Nvfp4ScaleLayout MoePrefillPackScaleLayout() {
  return Nvfp4ScaleLayout::kSwizzled128x4;
}

bool WorkspaceConfigSupported(const MoePrefillWorkspaceConfig& config) {
  const std::size_t routed_expert_intermediate_size =
      ResolveRoutedExpertIntermediateSizeExecution(
          config.routed_expert_intermediate_size,
          config.routed_expert_intermediate_size_padded);
  return config.hidden_size != 0 &&
         config.num_experts != 0 &&
         config.num_experts <= kMaxDeviceExpertRoutingExperts &&
         config.top_k != 0 &&
         config.top_k <= config.num_experts &&
         RoutedExpertIntermediateSizePaddingValid(
             config.routed_expert_intermediate_size,
             config.routed_expert_intermediate_size_padded) &&
         routed_expert_intermediate_size % kMoeNvfp4ScaleBlockWidth == 0 &&
         config.shared_expert_intermediate_size != 0;
}

std::optional<std::size_t> DeviceExpertRoutingBytes(
    std::size_t n_experts,
    std::size_t selection_count) {
  std::size_t total = 0;
  const auto add_bytes = [&](std::size_t count, std::size_t element_bytes) -> bool {
    const auto bytes = CheckedMul(count, element_bytes);
    if (!bytes.has_value()) {
      return false;
    }
    const auto next_total = CheckedAdd(total, *bytes);
    if (!next_total.has_value()) {
      return false;
    }
    total = *next_total;
    return true;
  };
  return add_bytes(n_experts, sizeof(int)) &&
                 add_bytes(n_experts + 1, sizeof(int)) &&
                 add_bytes(selection_count, sizeof(int)) &&
                 add_bytes(selection_count, sizeof(float))
             ? std::optional<std::size_t>(total)
             : std::nullopt;
}

std::optional<std::size_t> DeviceMoeLaunchPlanBytes(
    std::size_t n_experts,
    std::size_t selection_count,
    std::size_t max_output_rows_per_expert) {
  return DeviceMoeLaunchPlan::Bytes(
      n_experts,
      selection_count,
      max_output_rows_per_expert);
}

std::optional<std::size_t> PaddedSelectionCapacity(
    std::size_t n_experts,
    std::size_t selection_count) {
  return DeviceMoeLaunchPlan::PaddedRowCapacity(n_experts, selection_count);
}

bool HandlesSatisfyLayerInvariant(
    const AttentionKvCacheConfig& config,
    const std::vector<KvPageHandle>& pages,
    std::size_t layer_index) {
  const auto geometry = BuildAttentionKvPageGeometry(config);
  if (!geometry.has_value() || layer_index >= config.layer_count) {
    return false;
  }

  std::size_t previous_page_id = 0;
  bool have_previous_page = false;
  for (const KvPageHandle& handle : pages) {
    const std::size_t expected_byte_offset = handle.page_id * geometry->bytes_per_page;
    if (handle.layer_index != layer_index ||
        handle.tokens_per_page != config.tokens_per_page ||
        handle.byte_offset != expected_byte_offset ||
        (have_previous_page && handle.page_id <= previous_page_id)) {
      return false;
    }
    previous_page_id = handle.page_id;
    have_previous_page = true;
  }
  return true;
}

bool TensorMatchesShape(
    const DeviceTensorBf16* tensor,
    std::size_t dim0,
    std::size_t dim1) {
  return tensor != nullptr &&
         tensor->valid() &&
         tensor->shape().size() == 2 &&
         tensor->shape()[0] == dim0 &&
         tensor->shape()[1] == dim1;
}

bool TensorMatchesShape(
    const DeviceTensorFp32* tensor,
    std::size_t dim0,
    std::size_t dim1) {
  return tensor != nullptr &&
         tensor->valid() &&
         tensor->shape().size() == 2 &&
         tensor->shape()[0] == dim0 &&
         tensor->shape()[1] == dim1;
}

bool TensorMatchesShape(
    const DeviceTensorInt32* tensor,
    std::size_t dim0,
    std::size_t dim1) {
  return tensor != nullptr &&
         tensor->valid() &&
         tensor->shape().size() == 2 &&
         tensor->shape()[0] == dim0 &&
         tensor->shape()[1] == dim1;
}

bool SameWorkspaceConfig(
    const MoePrefillWorkspaceConfig& lhs,
    const MoePrefillWorkspaceConfig& rhs) {
  return lhs.hidden_size == rhs.hidden_size &&
         lhs.num_experts == rhs.num_experts &&
         lhs.top_k == rhs.top_k &&
         ResolveRoutedExpertIntermediateSizeExecution(
             lhs.routed_expert_intermediate_size,
             lhs.routed_expert_intermediate_size_padded) ==
             ResolveRoutedExpertIntermediateSizeExecution(
                 rhs.routed_expert_intermediate_size,
                 rhs.routed_expert_intermediate_size_padded) &&
         lhs.shared_expert_intermediate_size == rhs.shared_expert_intermediate_size;
}

}  // namespace

std::optional<std::size_t> MoePrefillWorkspace::BytesForTokenCapacity(
    std::size_t token_capacity,
    const MoePrefillWorkspaceConfig& config) {
  if (token_capacity == 0 || !WorkspaceConfigSupported(config)) {
    return std::nullopt;
  }
  const std::size_t routed_expert_intermediate_size =
      ResolveRoutedExpertIntermediateSizeExecution(
          config.routed_expert_intermediate_size,
          config.routed_expert_intermediate_size_padded);

  const auto selection_capacity = CheckedMul(token_capacity, config.top_k);
  const auto padded_selection_capacity =
      selection_capacity.has_value()
          ? PaddedSelectionCapacity(config.num_experts, *selection_capacity)
          : std::nullopt;
  if (!selection_capacity.has_value()) {
    return std::nullopt;
  }
  if (!padded_selection_capacity.has_value()) {
    return std::nullopt;
  }
  const Nvfp4ScaleLayout pack_scale_layout = MoePrefillPackScaleLayout();

  std::size_t total = 0;
  const auto add_bytes = [&](std::optional<std::size_t> bytes) -> bool {
    if (!bytes.has_value()) {
      return false;
    }
    const auto next_total = CheckedAdd(total, *bytes);
    if (!next_total.has_value()) {
      return false;
    }
    total = *next_total;
    return true;
  };

  return add_bytes(MatrixBytes(token_capacity, config.hidden_size, sizeof(__nv_bfloat16))) &&
                 add_bytes(MatrixBytes(token_capacity, config.hidden_size, sizeof(float))) &&
                 add_bytes(MatrixBytes(token_capacity, config.hidden_size, sizeof(float))) &&
                 add_bytes(MatrixBytes(token_capacity, config.num_experts, sizeof(float))) &&
                 add_bytes(MatrixBytes(token_capacity, config.hidden_size, sizeof(float))) &&
                 add_bytes(MatrixBytes(token_capacity, config.top_k, sizeof(int))) &&
                 add_bytes(MatrixBytes(token_capacity, config.top_k, sizeof(float))) &&
                 add_bytes(DeviceExpertRoutingBytes(config.num_experts, *selection_capacity)) &&
                 add_bytes(DeviceMoeLaunchPlanBytes(
                     config.num_experts,
                     *selection_capacity,
                     config.hidden_size)) &&
                 add_bytes(MatrixBytes(token_capacity, config.hidden_size, sizeof(float))) &&
                 add_bytes(
                     MatrixBytes(*padded_selection_capacity, config.hidden_size, sizeof(float))) &&
                 add_bytes(MatrixBytes(config.num_experts, 1, sizeof(float))) &&
                 add_bytes(MatrixBytes(config.num_experts, 1, sizeof(float))) &&
                 add_bytes(MatrixBytes(
                     *padded_selection_capacity,
                     routed_expert_intermediate_size,
                     sizeof(__nv_bfloat16))) &&
                 add_bytes(MatrixBytes(
                     *padded_selection_capacity,
                     routed_expert_intermediate_size / kMoeNvfp4ScaleBlockWidth,
                     sizeof(float))) &&
                 add_bytes(MatrixBytes(
                     *padded_selection_capacity,
                     routed_expert_intermediate_size,
                     sizeof(float))) &&
                 add_bytes(MatrixBytes(
                     token_capacity,
                     config.shared_expert_intermediate_size,
                     sizeof(float))) &&
                 add_bytes(Nvfp4MatrixBytes(
                     *padded_selection_capacity,
                     config.hidden_size,
                     pack_scale_layout)) &&
                 add_bytes(Nvfp4MatrixBytes(
                     *padded_selection_capacity,
                     routed_expert_intermediate_size,
                     pack_scale_layout)) &&
                 add_bytes(Nvfp4MatrixBytes(
                     token_capacity,
                     config.shared_expert_intermediate_size,
                     pack_scale_layout))
             ? std::optional<std::size_t>(total)
             : std::nullopt;
}

std::unique_ptr<MoePrefillWorkspace> MoePrefillWorkspace::Create(
    std::size_t token_capacity,
    const MoePrefillWorkspaceConfig& config) {
  if (!BytesForTokenCapacity(token_capacity, config).has_value()) {
    return nullptr;
  }
  const std::size_t routed_expert_intermediate_size =
      ResolveRoutedExpertIntermediateSizeExecution(
          config.routed_expert_intermediate_size,
          config.routed_expert_intermediate_size_padded);

  const std::size_t selection_capacity = token_capacity * config.top_k;
  const auto padded_selection_capacity =
      PaddedSelectionCapacity(config.num_experts, selection_capacity);
  if (!padded_selection_capacity.has_value()) {
    return nullptr;
  }
  const Nvfp4ScaleLayout pack_scale_layout = MoePrefillPackScaleLayout();
  auto workspace = std::make_unique<MoePrefillWorkspace>();
  workspace->config = config;
  workspace->token_capacity_value = token_capacity;
  workspace->normalized_bf16 = DeviceTensorBf16::Create({token_capacity, config.hidden_size});
  workspace->input_fp32 = DeviceTensorFp32::Create({token_capacity, config.hidden_size});
  workspace->normalized = DeviceTensorFp32::Create({token_capacity, config.hidden_size});
  workspace->router_logits = DeviceTensorFp32::Create({token_capacity, config.num_experts});
  workspace->output_fp32 = DeviceTensorFp32::Create({token_capacity, config.hidden_size});
  workspace->topk_ids = DeviceTensorInt32::Create({token_capacity, config.top_k});
  workspace->topk_weights = DeviceTensorFp32::Create({token_capacity, config.top_k});
  workspace->fused_prefill_routing =
      DeviceExpertRouting::Create(config.num_experts, selection_capacity);
  workspace->fused_prefill_launch_plan =
      DeviceMoeLaunchPlan::Create(
          config.num_experts,
          selection_capacity,
          config.hidden_size);
  workspace->fused_prefill_routed_output_scratch =
      DeviceTensorFp32::Create({token_capacity, config.hidden_size});
  workspace->fused_prefill_gather_scratch =
      DeviceTensorFp32::Create({*padded_selection_capacity, config.hidden_size});
  workspace->fused_prefill_fc1_activation_scales =
      DeviceTensorFp32::Create({config.num_experts, 1});
  workspace->fused_prefill_fc2_activation_scales =
      DeviceTensorFp32::Create({config.num_experts, 1});
  workspace->fused_prefill_gemm1_output_bf16 =
      DeviceTensorBf16::Create(
          {*padded_selection_capacity, routed_expert_intermediate_size});
  workspace->fused_prefill_gemm1_output_scales =
      DeviceTensorFp32::Create(
          {*padded_selection_capacity,
           routed_expert_intermediate_size / kMoeNvfp4ScaleBlockWidth});
  workspace->fused_prefill_expert_up_scratch =
      DeviceTensorFp32::Create(
          {*padded_selection_capacity, routed_expert_intermediate_size});
  workspace->fused_prefill_shared_up_scratch =
      DeviceTensorFp32::Create({token_capacity, config.shared_expert_intermediate_size});
  // Reuse the normalized/source pack as the shared FC1 activation pack.
  workspace->fused_prefill_normalized_pack =
      DeviceNvfp4Matrix::Create(
          token_capacity,
          config.hidden_size,
          pack_scale_layout);
  workspace->fused_prefill_gather_pack =
      DeviceNvfp4Matrix::Create(
          *padded_selection_capacity,
          config.hidden_size,
          pack_scale_layout);
  workspace->fused_prefill_expert_up_pack = DeviceNvfp4Matrix::Create(
      *padded_selection_capacity,
      routed_expert_intermediate_size,
      pack_scale_layout);
  // Reuse the shared-up activation pack as the shared FC2 activation pack.
  workspace->fused_prefill_shared_up_pack = DeviceNvfp4Matrix::Create(
      token_capacity,
      config.shared_expert_intermediate_size,
      pack_scale_layout);
  if (!workspace->valid()) {
    return nullptr;
  }
  return workspace;
}

bool MoePrefillWorkspace::valid() const {
  if (token_capacity_value == 0 ||
      !WorkspaceConfigSupported(config) ||
      token_capacity_value > (std::numeric_limits<std::size_t>::max() / config.top_k)) {
    return false;
  }

  const std::size_t selection_capacity = token_capacity_value * config.top_k;
  const auto padded_selection_capacity =
      PaddedSelectionCapacity(config.num_experts, selection_capacity);
  if (!padded_selection_capacity.has_value()) {
    return false;
  }
  const std::size_t routed_expert_intermediate_size =
      ResolveRoutedExpertIntermediateSizeExecution(
          config.routed_expert_intermediate_size,
          config.routed_expert_intermediate_size_padded);
  return TensorMatchesShape(normalized_bf16.get(), token_capacity_value, config.hidden_size) &&
         TensorMatchesShape(input_fp32.get(), token_capacity_value, config.hidden_size) &&
         TensorMatchesShape(normalized.get(), token_capacity_value, config.hidden_size) &&
         TensorMatchesShape(router_logits.get(), token_capacity_value, config.num_experts) &&
         TensorMatchesShape(output_fp32.get(), token_capacity_value, config.hidden_size) &&
         TensorMatchesShape(topk_ids.get(), token_capacity_value, config.top_k) &&
         TensorMatchesShape(topk_weights.get(), token_capacity_value, config.top_k) &&
         fused_prefill_routing != nullptr &&
         fused_prefill_routing->valid() &&
         fused_prefill_routing->n_experts() == config.num_experts &&
         fused_prefill_routing->selection_count() >= selection_capacity &&
         fused_prefill_launch_plan != nullptr &&
         fused_prefill_launch_plan->valid() &&
         fused_prefill_launch_plan->n_experts() == config.num_experts &&
         fused_prefill_launch_plan->selection_count() >= selection_capacity &&
         fused_prefill_launch_plan->cta_capacity() > 0 &&
         TensorMatchesShape(
             fused_prefill_routed_output_scratch.get(),
             token_capacity_value,
             config.hidden_size) &&
         TensorMatchesShape(
             fused_prefill_gather_scratch.get(),
             *padded_selection_capacity,
             config.hidden_size) &&
         TensorMatchesShape(
             fused_prefill_fc1_activation_scales.get(),
             config.num_experts,
             1) &&
         TensorMatchesShape(
             fused_prefill_fc2_activation_scales.get(),
             config.num_experts,
             1) &&
         TensorMatchesShape(
             fused_prefill_gemm1_output_bf16.get(),
             *padded_selection_capacity,
             routed_expert_intermediate_size) &&
         TensorMatchesShape(
             fused_prefill_gemm1_output_scales.get(),
             *padded_selection_capacity,
             routed_expert_intermediate_size / kMoeNvfp4ScaleBlockWidth) &&
         TensorMatchesShape(
             fused_prefill_expert_up_scratch.get(),
             *padded_selection_capacity,
             routed_expert_intermediate_size) &&
         TensorMatchesShape(
             fused_prefill_shared_up_scratch.get(),
             token_capacity_value,
             config.shared_expert_intermediate_size) &&
         fused_prefill_normalized_pack != nullptr &&
         fused_prefill_normalized_pack->valid() &&
         fused_prefill_normalized_pack->rows() >= token_capacity_value &&
         fused_prefill_normalized_pack->cols() == config.hidden_size &&
         fused_prefill_gather_pack != nullptr &&
         fused_prefill_gather_pack->valid() &&
         fused_prefill_gather_pack->rows() >= *padded_selection_capacity &&
         fused_prefill_gather_pack->cols() == config.hidden_size &&
         fused_prefill_expert_up_pack != nullptr &&
         fused_prefill_expert_up_pack->valid() &&
         fused_prefill_expert_up_pack->rows() >= *padded_selection_capacity &&
         fused_prefill_expert_up_pack->cols() == routed_expert_intermediate_size &&
         fused_prefill_shared_up_pack != nullptr &&
         fused_prefill_shared_up_pack->valid() &&
         fused_prefill_shared_up_pack->rows() >= token_capacity_value &&
         fused_prefill_shared_up_pack->cols() == config.shared_expert_intermediate_size;
}

std::size_t MoePrefillWorkspace::token_capacity() const {
  return token_capacity_value;
}

std::unique_ptr<RequestExecutionContext> RequestExecutionContext::Create(
    const RequestExecutionConfig& config) {
  if (config.hidden_size == 0 || config.max_tokens == 0) {
    return nullptr;
  }
  const std::size_t scratch_tokens = config.scratch_tokens == 0 ? config.max_tokens : config.scratch_tokens;

  auto hidden = DeviceTensorBf16::Create({config.max_tokens, config.hidden_size});
  auto residual = DeviceTensorBf16::Create({config.max_tokens, config.hidden_size});
  auto scratch = DeviceTensorBf16::Create({scratch_tokens, config.hidden_size});
  if (!hidden || !residual || !scratch) {
    return nullptr;
  }
  if (!hidden->FillZero() || !residual->FillZero() || !scratch->FillZero()) {
    return nullptr;
  }

  std::unique_ptr<DeviceTensorFp32> mamba_state;
  std::unique_ptr<DeviceTensorFp32> mamba_conv_state;
  if (config.mamba_conv_state_bytes_fp32 != 0) {
    if (config.mamba_conv_state_bytes_fp32 % sizeof(float) != 0) {
      return nullptr;
    }
    const std::size_t state_numel = config.mamba_conv_state_bytes_fp32 / sizeof(float);
    mamba_conv_state = DeviceTensorFp32::Create({state_numel});
    if (!mamba_conv_state || !mamba_conv_state->FillZero()) {
      return nullptr;
    }
  }
  if (config.mamba_state_bytes_fp32 != 0) {
    if (config.mamba_state_bytes_fp32 % sizeof(float) != 0) {
      return nullptr;
    }
    const std::size_t state_numel = config.mamba_state_bytes_fp32 / sizeof(float);
    mamba_state = DeviceTensorFp32::Create({state_numel});
    if (!mamba_state || !mamba_state->FillZero()) {
      return nullptr;
    }
  }

  std::unique_ptr<DeviceTensorBf16> key_cache;
  std::unique_ptr<DeviceTensorBf16> value_cache;
  auto greedy_token_id_scratch = DeviceTensorInt32::Create({1});
  if (!greedy_token_id_scratch || !greedy_token_id_scratch->FillZero()) {
    return nullptr;
  }
  std::optional<PagedKvCacheArena> kv_arena;
  if (config.attention_kv_cache.layer_count != 0) {
    kv_arena = PagedKvCacheArena::Create(config.attention_kv_cache, config.attention_total_pages);
    if (!kv_arena.has_value()) {
      return nullptr;
    }
    key_cache = DeviceTensorBf16::Create({
        config.attention_total_pages,
        config.attention_kv_cache.kv_head_count,
        config.attention_kv_cache.tokens_per_page,
        config.attention_kv_cache.head_dim,
    });
    value_cache = DeviceTensorBf16::Create({
        config.attention_total_pages,
        config.attention_kv_cache.kv_head_count,
        config.attention_kv_cache.tokens_per_page,
        config.attention_kv_cache.head_dim,
    });
    if (!key_cache || !value_cache || !key_cache->FillZero() || !value_cache->FillZero()) {
      return nullptr;
    }
  }

  std::unique_ptr<MoePrefillWorkspace> moe_prefill_workspace;
  if (config.moe_prefill_capacity_tokens > 1) {
    moe_prefill_workspace = MoePrefillWorkspace::Create(
        config.moe_prefill_capacity_tokens,
        config.moe_prefill_workspace_config);
    if (!moe_prefill_workspace) {
      return nullptr;
    }
  }

  return std::unique_ptr<RequestExecutionContext>(new RequestExecutionContext(
      config,
      std::move(hidden),
      std::move(residual),
      std::move(scratch),
      std::move(mamba_conv_state),
      std::move(mamba_state),
      std::move(key_cache),
      std::move(value_cache),
      std::move(greedy_token_id_scratch),
      std::move(moe_prefill_workspace),
      std::move(kv_arena)));
}

RequestExecutionContext::RequestExecutionContext(
    RequestExecutionConfig config,
    std::unique_ptr<DeviceTensorBf16> hidden,
    std::unique_ptr<DeviceTensorBf16> residual,
    std::unique_ptr<DeviceTensorBf16> scratch,
    std::unique_ptr<DeviceTensorFp32> mamba_conv_state,
    std::unique_ptr<DeviceTensorFp32> mamba_state,
    std::unique_ptr<DeviceTensorBf16> key_cache,
    std::unique_ptr<DeviceTensorBf16> value_cache,
    std::unique_ptr<DeviceTensorInt32> greedy_token_id_scratch,
    std::unique_ptr<MoePrefillWorkspace> moe_prefill_workspace,
    std::optional<PagedKvCacheArena> kv_arena)
    : config_(std::move(config)),
      hidden_(std::move(hidden)),
      residual_(std::move(residual)),
      scratch_(std::move(scratch)),
      mamba_conv_state_(std::move(mamba_conv_state)),
      mamba_state_(std::move(mamba_state)),
      key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)),
      greedy_token_id_scratch_(std::move(greedy_token_id_scratch)),
      moe_prefill_workspace_(std::move(moe_prefill_workspace)),
      kv_arena_(std::move(kv_arena)),
      kv_pages_by_layer_(config_.attention_kv_cache.layer_count) {}

RequestExecutionContext::RequestExecutionContext(RequestExecutionContext&&) noexcept = default;
RequestExecutionContext& RequestExecutionContext::operator=(RequestExecutionContext&&) noexcept = default;
RequestExecutionContext::~RequestExecutionContext() = default;

bool RequestExecutionContext::valid() const {
  if (!hidden_ || !hidden_->valid() || !residual_ || !residual_->valid() || !scratch_ || !scratch_->valid()) {
    return false;
  }
  if (!greedy_token_id_scratch_ || !greedy_token_id_scratch_->valid() ||
      greedy_token_id_scratch_->numel() != 1) {
    return false;
  }
  if (config_.mamba_state_bytes_fp32 != 0 && (!mamba_state_ || !mamba_state_->valid())) {
    return false;
  }
  if (config_.mamba_conv_state_bytes_fp32 != 0 &&
      (!mamba_conv_state_ || !mamba_conv_state_->valid())) {
    return false;
  }
  if (config_.attention_kv_cache.layer_count != 0 && !kv_arena_.has_value()) {
    return false;
  }
  if (config_.attention_kv_cache.layer_count != 0 &&
      (!key_cache_ || !key_cache_->valid() || !value_cache_ || !value_cache_->valid())) {
    return false;
  }
  if (config_.moe_prefill_capacity_tokens > 1 &&
      (moe_prefill_workspace_ == nullptr ||
       !moe_prefill_workspace_->valid() ||
       !SameWorkspaceConfig(moe_prefill_workspace_->config, config_.moe_prefill_workspace_config) ||
       moe_prefill_workspace_->token_capacity() < config_.moe_prefill_capacity_tokens)) {
    return false;
  }
  if (moe_prefill_workspace_ != nullptr && !moe_prefill_workspace_->valid()) {
    return false;
  }
  return true;
}

const RequestExecutionConfig& RequestExecutionContext::config() const {
  return config_;
}

std::size_t RequestExecutionContext::sequence_length() const {
  return sequence_length_;
}

std::size_t RequestExecutionContext::decode_position() const {
  return decode_position_;
}

DeviceTensorBf16* RequestExecutionContext::hidden() {
  return hidden_.get();
}

const DeviceTensorBf16* RequestExecutionContext::hidden() const {
  return hidden_.get();
}

DeviceTensorBf16* RequestExecutionContext::residual() {
  return residual_.get();
}

const DeviceTensorBf16* RequestExecutionContext::residual() const {
  return residual_.get();
}

DeviceTensorBf16* RequestExecutionContext::scratch() {
  return scratch_.get();
}

const DeviceTensorBf16* RequestExecutionContext::scratch() const {
  return scratch_.get();
}

DeviceTensorFp32* RequestExecutionContext::mamba_state() {
  return mamba_state_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_state() const {
  return mamba_state_.get();
}

DeviceTensorFp32* RequestExecutionContext::mamba_conv_state() {
  return mamba_conv_state_.get();
}

const DeviceTensorFp32* RequestExecutionContext::mamba_conv_state() const {
  return mamba_conv_state_.get();
}

DeviceTensorBf16* RequestExecutionContext::key_cache() {
  return key_cache_.get();
}

const DeviceTensorBf16* RequestExecutionContext::key_cache() const {
  return key_cache_.get();
}

DeviceTensorBf16* RequestExecutionContext::value_cache() {
  return value_cache_.get();
}

const DeviceTensorBf16* RequestExecutionContext::value_cache() const {
  return value_cache_.get();
}

DeviceTensorInt32* RequestExecutionContext::greedy_token_id_scratch() {
  return greedy_token_id_scratch_.get();
}

const DeviceTensorInt32* RequestExecutionContext::greedy_token_id_scratch() const {
  return greedy_token_id_scratch_.get();
}

MoePrefillWorkspace* RequestExecutionContext::moe_prefill_workspace() {
  return moe_prefill_workspace_.get();
}

const MoePrefillWorkspace* RequestExecutionContext::moe_prefill_workspace() const {
  return moe_prefill_workspace_.get();
}

bool RequestExecutionContext::EnsureAttentionTokens(std::size_t token_count) {
  if (!kv_arena_.has_value()) {
    return token_count == 0;
  }
  if (token_count > config_.max_tokens) {
    return false;
  }

  const std::size_t required_pages =
      RequiredPagesForTokens(config_.attention_kv_cache, token_count);
  for (std::size_t layer_index = 0; layer_index < kv_pages_by_layer_.size(); ++layer_index) {
    std::vector<KvPageHandle>& pages = kv_pages_by_layer_[layer_index];
    if (!HandlesSatisfyLayerInvariant(config_.attention_kv_cache, pages, layer_index)) {
      return false;
    }
    if (pages.size() >= required_pages) {
      continue;
    }
    const std::size_t additional_pages = required_pages - pages.size();
    const auto allocated = kv_arena_->AllocatePages(layer_index, additional_pages);
    if (!allocated.has_value()) {
      return false;
    }
    pages.insert(pages.end(), allocated->begin(), allocated->end());
    if (!HandlesSatisfyLayerInvariant(config_.attention_kv_cache, pages, layer_index)) {
      return false;
    }
  }
  return true;
}

bool RequestExecutionContext::EnsureMoePrefillWorkspace(
    std::size_t token_capacity,
    const MoePrefillWorkspaceConfig& config) {
  if (token_capacity == 0) {
    return false;
  }
  if (moe_prefill_workspace_ != nullptr &&
      moe_prefill_workspace_->valid() &&
      SameWorkspaceConfig(moe_prefill_workspace_->config, config) &&
      moe_prefill_workspace_->token_capacity() >= token_capacity) {
    return true;
  }

  auto workspace = MoePrefillWorkspace::Create(token_capacity, config);
  if (!workspace) {
    return false;
  }
  moe_prefill_workspace_ = std::move(workspace);
  return true;
}

std::size_t RequestExecutionContext::allocated_kv_pages() const {
  std::size_t total = 0;
  for (const auto& layer_pages : kv_pages_by_layer_) {
    total += layer_pages.size();
  }
  return total;
}

std::size_t RequestExecutionContext::allocated_kv_pages(std::size_t layer_index) const {
  if (layer_index >= kv_pages_by_layer_.size()) {
    return 0;
  }
  return kv_pages_by_layer_[layer_index].size();
}

const std::vector<KvPageHandle>* RequestExecutionContext::kv_pages(std::size_t layer_index) const {
  if (layer_index >= kv_pages_by_layer_.size()) {
    return nullptr;
  }
  return &kv_pages_by_layer_[layer_index];
}

bool RequestExecutionContext::SetSequenceLength(std::size_t sequence_length) {
  if (sequence_length > config_.max_tokens) {
    return false;
  }
  sequence_length_ = sequence_length;
  decode_position_ = sequence_length;
  return EnsureAttentionTokens(sequence_length);
}

bool RequestExecutionContext::AdvanceDecodePosition(std::size_t token_count) {
  if (token_count == 0) {
    return true;
  }
  if (decode_position_ > config_.max_tokens || token_count > (config_.max_tokens - decode_position_)) {
    return false;
  }
  decode_position_ += token_count;
  if (decode_position_ > sequence_length_) {
    sequence_length_ = decode_position_;
  }
  return EnsureAttentionTokens(sequence_length_);
}

bool RequestExecutionContext::ResetForNewRequest() {
  bool ok = hidden_->FillZero() && residual_->FillZero() && scratch_->FillZero();
  if (mamba_conv_state_) {
    ok = ok && mamba_conv_state_->FillZero();
  }
  if (mamba_state_) {
    ok = ok && mamba_state_->FillZero();
  }
  if (key_cache_) {
    ok = ok && key_cache_->FillZero();
  }
  if (value_cache_) {
    ok = ok && value_cache_->FillZero();
  }
  if (greedy_token_id_scratch_) {
    ok = ok && greedy_token_id_scratch_->FillZero();
  }

  if (kv_arena_.has_value()) {
    std::vector<std::size_t> page_ids;
    page_ids.reserve(allocated_kv_pages());
    for (std::size_t layer_index = 0; layer_index < kv_pages_by_layer_.size(); ++layer_index) {
      const auto& layer_pages = kv_pages_by_layer_[layer_index];
      ok = ok && HandlesSatisfyLayerInvariant(config_.attention_kv_cache, layer_pages, layer_index);
      for (const KvPageHandle& handle : layer_pages) {
        page_ids.push_back(handle.page_id);
      }
    }
    if (!page_ids.empty()) {
      ok = ok && kv_arena_->ReleasePages(page_ids);
    }
  }

  for (auto& layer_pages : kv_pages_by_layer_) {
    layer_pages.clear();
  }
  sequence_length_ = 0;
  decode_position_ = 0;
  return ok;
}

}  // namespace nemotron
