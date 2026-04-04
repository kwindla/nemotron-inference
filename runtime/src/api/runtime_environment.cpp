#include "nemotron/runtime_environment.h"

#include "nemotron/artifact_loader.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/model_schedule.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <optional>
#include <utility>

namespace nemotron {
namespace {

struct ValidatedManifestBootstrap {
  PackedModelManifest manifest;
  std::unique_ptr<ArtifactLoader> artifact_loader;
};

struct PlannedManifestBootstrap {
  LoaderPlan loader_plan;
  RuntimeConfig config;
  std::size_t effective_shared_cache_budget_bytes = 0;
  std::unique_ptr<ArtifactLoader> artifact_loader;
  std::unique_ptr<TensorCatalog> tensor_catalog;
  std::unique_ptr<ModelSchedule> model_schedule;
  std::unique_ptr<WeightArenaPlan> weight_arena_plan;
  std::unique_ptr<WeightArena> weight_arena;
  std::unique_ptr<KernelCatalog> kernel_catalog;
  std::unique_ptr<GemmCatalog> gemm_catalog;
  std::unique_ptr<EmbeddingCatalog> embedding_catalog;
  std::unique_ptr<GemmHeuristicCache> gemm_heuristic_cache;
  std::unique_ptr<ReusableStateArena> reusable_state_arena;
  std::unique_ptr<PrefixCache> prefix_cache;
};

ServiceMemoryTarget ResolveServiceMemoryTarget(const RuntimeBootstrapOptions& options) {
  ServiceMemoryTarget target = options.service_target;
  if (!options.prefer_host_memory_snapshot) {
    return target;
  }

  const HostMemorySnapshot snapshot =
      ReadHostMemorySnapshotFromProcMeminfo(options.proc_meminfo_path);
  if (snapshot.valid) {
    target.host_memory_snapshot = snapshot;
    target.prefer_host_memory_snapshot = true;
  }
  return target;
}

std::optional<ValidatedManifestBootstrap> ValidateManifestBootstrap(
    const std::filesystem::path& manifest_path,
    const RuntimeBootstrapOptions& options) {
  const ManifestLoadResult load_result =
      options.verify_manifest_files
          ? LoadVerifiedManifestFromJsonFile(manifest_path)
          : LoadManifestFromJsonFile(manifest_path);
  if (!load_result.ok) {
    return std::nullopt;
  }

  auto artifact_loader = ArtifactLoader::OpenVerifiedWithMode(
      load_result.manifest,
      manifest_path,
      options.artifact_load_mode);
  if (!artifact_loader) {
    return std::nullopt;
  }

  return ValidatedManifestBootstrap{
      std::move(load_result.manifest),
      std::move(artifact_loader),
  };
}

std::optional<PlannedManifestBootstrap> PlanManifestBootstrap(
    ValidatedManifestBootstrap&& validated,
    const RuntimeBootstrapOptions& options) {
  auto tensor_catalog = std::make_unique<TensorCatalog>(
      BuildTensorCatalog(validated.manifest, *validated.artifact_loader));
  if (!tensor_catalog->valid()) {
    return std::nullopt;
  }

  auto model_schedule = std::make_unique<ModelSchedule>(
      BuildModelSchedule(validated.manifest));
  if (!model_schedule->valid()) {
    return std::nullopt;
  }

  std::unique_ptr<WeightArenaPlan> weight_arena_plan;
  if (options.materialize_weight_arena) {
    auto plan = std::make_unique<WeightArenaPlan>(BuildWeightArenaPlan(*tensor_catalog));
    if (!plan->valid()) {
      return std::nullopt;
    }
    weight_arena_plan = std::move(plan);
  }

  const ServiceMemoryTarget resolved_target = ResolveServiceMemoryTarget(options);
  LoaderPlan loader_plan = BuildLoaderPlan(
      validated.manifest,
      resolved_target,
      options.use_fp16_mamba_state,
      options.reusable_node_metadata_bytes);
  if (!loader_plan.valid) {
    return std::nullopt;
  }

  RuntimeConfig config = LoadRuntimeConfigFromEnv();
  const std::size_t effective_shared_cache_budget_bytes =
      config.prefix_cache_enabled ? loader_plan.memory_budget.shared_cache_budget_bytes : 0;

  return PlannedManifestBootstrap{
      std::move(loader_plan),
      std::move(config),
      effective_shared_cache_budget_bytes,
      std::move(validated.artifact_loader),
      std::move(tensor_catalog),
      std::move(model_schedule),
      std::move(weight_arena_plan),
  };
}

std::optional<PlannedManifestBootstrap> AssembleManifestBootstrap(
    PlannedManifestBootstrap&& plan,
    const RuntimeBootstrapOptions& options) {
  KernelCatalog kernel_catalog;
  if (options.materialize_weight_arena) {
    if (plan.weight_arena_plan == nullptr || !plan.weight_arena_plan->valid()) {
      return std::nullopt;
    }
    plan.weight_arena = WeightArena::CreateFromPlan(*plan.weight_arena_plan);
    if (!plan.weight_arena) {
      return std::nullopt;
    }
    kernel_catalog = BuildKernelCatalog(*plan.tensor_catalog, *plan.weight_arena);
  } else {
    kernel_catalog = BuildKernelCatalog(*plan.tensor_catalog);
  }
  if (!kernel_catalog.valid()) {
    return std::nullopt;
  }

  GemmCatalog gemm_catalog = BuildGemmCatalog(kernel_catalog);
  if (!gemm_catalog.valid()) {
    return std::nullopt;
  }
  EmbeddingCatalog embedding_catalog = BuildEmbeddingCatalog(kernel_catalog);
  if (!embedding_catalog.valid()) {
    return std::nullopt;
  }

  auto reusable_state_arena =
      std::make_unique<ReusableStateArena>(plan.effective_shared_cache_budget_bytes);
  auto prefix_cache = std::make_unique<PrefixCache>(
      plan.effective_shared_cache_budget_bytes,
      reusable_state_arena.get());
  auto gemm_heuristic_cache = std::make_unique<GemmHeuristicCache>();
  ApplyRuntimeConfig(plan.config, prefix_cache.get());

  plan.kernel_catalog = std::make_unique<KernelCatalog>(std::move(kernel_catalog));
  plan.gemm_catalog = std::make_unique<GemmCatalog>(std::move(gemm_catalog));
  plan.embedding_catalog = std::make_unique<EmbeddingCatalog>(std::move(embedding_catalog));
  plan.gemm_heuristic_cache = std::move(gemm_heuristic_cache);
  plan.reusable_state_arena = std::move(reusable_state_arena);
  plan.prefix_cache = std::move(prefix_cache);
  return std::move(plan);
}

}  // namespace

std::unique_ptr<RuntimeEnvironment> RuntimeEnvironment::Build(
    const PackedModelManifest& manifest,
    const RuntimeBootstrapOptions& options) {
  const ServiceMemoryTarget resolved_target = ResolveServiceMemoryTarget(options);
  LoaderPlan loader_plan = BuildLoaderPlan(
      manifest,
      resolved_target,
      options.use_fp16_mamba_state,
      options.reusable_node_metadata_bytes);
  if (!loader_plan.valid) {
    return nullptr;
  }

  RuntimeConfig config = LoadRuntimeConfigFromEnv();
  const std::size_t effective_shared_cache_budget_bytes =
      config.prefix_cache_enabled ? loader_plan.memory_budget.shared_cache_budget_bytes : 0;

  auto reusable_state_arena =
      std::make_unique<ReusableStateArena>(effective_shared_cache_budget_bytes);
  auto prefix_cache =
      std::make_unique<PrefixCache>(effective_shared_cache_budget_bytes, reusable_state_arena.get());
  auto gemm_heuristic_cache = std::make_unique<GemmHeuristicCache>();
  ModelSchedule model_schedule = BuildModelSchedule(manifest);
  if (!model_schedule.valid()) {
    return nullptr;
  }
  ApplyRuntimeConfig(config, prefix_cache.get());

  return std::unique_ptr<RuntimeEnvironment>(new RuntimeEnvironment(
      std::move(config),
      std::move(loader_plan),
      effective_shared_cache_budget_bytes,
      nullptr,
      nullptr,
      std::make_unique<ModelSchedule>(std::move(model_schedule)),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      std::move(gemm_heuristic_cache),
      std::move(reusable_state_arena),
      std::move(prefix_cache)));
}

std::unique_ptr<RuntimeEnvironment> RuntimeEnvironment::BuildFromManifestFile(
    const std::filesystem::path& manifest_path,
    const RuntimeBootstrapOptions& options) {
  auto validated = ValidateManifestBootstrap(manifest_path, options);
  if (!validated.has_value()) {
    return nullptr;
  }

  auto planned = PlanManifestBootstrap(std::move(*validated), options);
  if (!planned.has_value()) {
    return nullptr;
  }

  auto assembled = AssembleManifestBootstrap(std::move(*planned), options);
  if (!assembled.has_value()) {
    return nullptr;
  }

  PlannedManifestBootstrap plan = std::move(*assembled);
  return std::unique_ptr<RuntimeEnvironment>(new RuntimeEnvironment(
      std::move(plan.config),
      std::move(plan.loader_plan),
      plan.effective_shared_cache_budget_bytes,
      std::move(plan.artifact_loader),
      std::move(plan.tensor_catalog),
      std::move(plan.model_schedule),
      std::move(plan.weight_arena_plan),
      std::move(plan.weight_arena),
      std::move(plan.kernel_catalog),
      std::move(plan.gemm_catalog),
      std::move(plan.embedding_catalog),
      std::move(plan.gemm_heuristic_cache),
      std::move(plan.reusable_state_arena),
      std::move(plan.prefix_cache)));
}

RuntimeEnvironment::RuntimeEnvironment(
    RuntimeConfig config,
    LoaderPlan loader_plan,
    std::size_t effective_shared_cache_budget_bytes,
    std::unique_ptr<ArtifactLoader> artifact_loader,
    std::unique_ptr<TensorCatalog> tensor_catalog,
    std::unique_ptr<ModelSchedule> model_schedule,
    std::unique_ptr<WeightArenaPlan> weight_arena_plan,
    std::unique_ptr<WeightArena> weight_arena,
    std::unique_ptr<KernelCatalog> kernel_catalog,
    std::unique_ptr<GemmCatalog> gemm_catalog,
    std::unique_ptr<EmbeddingCatalog> embedding_catalog,
    std::unique_ptr<GemmHeuristicCache> gemm_heuristic_cache,
    std::unique_ptr<ReusableStateArena> reusable_state_arena,
    std::unique_ptr<PrefixCache> prefix_cache)
    : config_(std::move(config)),
      loader_plan_(std::move(loader_plan)),
      effective_shared_cache_budget_bytes_(effective_shared_cache_budget_bytes),
      artifact_loader_(std::move(artifact_loader)),
      tensor_catalog_(std::move(tensor_catalog)),
      model_schedule_(std::move(model_schedule)),
      weight_arena_plan_(std::move(weight_arena_plan)),
      weight_arena_(std::move(weight_arena)),
      kernel_catalog_(std::move(kernel_catalog)),
      gemm_catalog_(std::move(gemm_catalog)),
      embedding_catalog_(std::move(embedding_catalog)),
      gemm_heuristic_cache_(std::move(gemm_heuristic_cache)),
      reusable_state_arena_(std::move(reusable_state_arena)),
      prefix_cache_(std::move(prefix_cache)) {}

RuntimeEnvironment::RuntimeEnvironment(RuntimeEnvironment&&) noexcept = default;
RuntimeEnvironment& RuntimeEnvironment::operator=(RuntimeEnvironment&&) noexcept = default;
RuntimeEnvironment::~RuntimeEnvironment() = default;

const RuntimeConfig& RuntimeEnvironment::config() const {
  return config_;
}

const LoaderPlan& RuntimeEnvironment::loader_plan() const {
  return loader_plan_;
}

std::size_t RuntimeEnvironment::effective_shared_cache_budget_bytes() const {
  return effective_shared_cache_budget_bytes_;
}

bool RuntimeEnvironment::has_artifact_loader() const {
  return static_cast<bool>(artifact_loader_);
}

ArtifactLoader* RuntimeEnvironment::artifact_loader() {
  return artifact_loader_.get();
}

const ArtifactLoader* RuntimeEnvironment::artifact_loader() const {
  return artifact_loader_.get();
}

bool RuntimeEnvironment::has_tensor_catalog() const {
  return static_cast<bool>(tensor_catalog_);
}

TensorCatalog* RuntimeEnvironment::tensor_catalog() {
  return tensor_catalog_.get();
}

const TensorCatalog* RuntimeEnvironment::tensor_catalog() const {
  return tensor_catalog_.get();
}

bool RuntimeEnvironment::has_model_schedule() const {
  return static_cast<bool>(model_schedule_);
}

ModelSchedule* RuntimeEnvironment::model_schedule() {
  return model_schedule_.get();
}

const ModelSchedule* RuntimeEnvironment::model_schedule() const {
  return model_schedule_.get();
}

bool RuntimeEnvironment::has_weight_arena_plan() const {
  return static_cast<bool>(weight_arena_plan_);
}

WeightArenaPlan* RuntimeEnvironment::weight_arena_plan() {
  return weight_arena_plan_.get();
}

const WeightArenaPlan* RuntimeEnvironment::weight_arena_plan() const {
  return weight_arena_plan_.get();
}

bool RuntimeEnvironment::has_weight_arena() const {
  return static_cast<bool>(weight_arena_);
}

WeightArena* RuntimeEnvironment::weight_arena() {
  return weight_arena_.get();
}

const WeightArena* RuntimeEnvironment::weight_arena() const {
  return weight_arena_.get();
}

bool RuntimeEnvironment::has_kernel_catalog() const {
  return static_cast<bool>(kernel_catalog_);
}

KernelCatalog* RuntimeEnvironment::kernel_catalog() {
  return kernel_catalog_.get();
}

const KernelCatalog* RuntimeEnvironment::kernel_catalog() const {
  return kernel_catalog_.get();
}

bool RuntimeEnvironment::has_gemm_catalog() const {
  return static_cast<bool>(gemm_catalog_);
}

GemmCatalog* RuntimeEnvironment::gemm_catalog() {
  return gemm_catalog_.get();
}

const GemmCatalog* RuntimeEnvironment::gemm_catalog() const {
  return gemm_catalog_.get();
}

bool RuntimeEnvironment::has_embedding_catalog() const {
  return static_cast<bool>(embedding_catalog_);
}

EmbeddingCatalog* RuntimeEnvironment::embedding_catalog() {
  return embedding_catalog_.get();
}

const EmbeddingCatalog* RuntimeEnvironment::embedding_catalog() const {
  return embedding_catalog_.get();
}

bool RuntimeEnvironment::has_gemm_heuristic_cache() const {
  return static_cast<bool>(gemm_heuristic_cache_);
}

GemmHeuristicCache* RuntimeEnvironment::gemm_heuristic_cache() {
  return gemm_heuristic_cache_.get();
}

const GemmHeuristicCache* RuntimeEnvironment::gemm_heuristic_cache() const {
  return gemm_heuristic_cache_.get();
}

PrefixCache& RuntimeEnvironment::prefix_cache() {
  return *prefix_cache_;
}

const PrefixCache& RuntimeEnvironment::prefix_cache() const {
  return *prefix_cache_;
}

ReusableStateArena& RuntimeEnvironment::reusable_state_arena() {
  return *reusable_state_arena_;
}

const ReusableStateArena& RuntimeEnvironment::reusable_state_arena() const {
  return *reusable_state_arena_;
}

}  // namespace nemotron
