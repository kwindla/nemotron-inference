#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "nemotron/artifact_loader.h"
#include "nemotron/loader.h"
#include "nemotron/prefix_cache.h"
#include "nemotron/reusable_state.h"
#include "nemotron/runtime_config.h"

namespace nemotron {

class EmbeddingCatalog;
class GemmCatalog;
class GemmHeuristicCache;
class KernelCatalog;
class ModelSchedule;
class TensorCatalog;
class WeightArena;
class WeightArenaPlan;

struct RuntimeBootstrapOptions {
  ServiceMemoryTarget service_target;
  bool use_fp16_mamba_state = false;
  std::size_t reusable_node_metadata_bytes = 0;
  bool prefer_host_memory_snapshot = false;
  std::filesystem::path proc_meminfo_path = "/proc/meminfo";
  ArtifactLoadMode artifact_load_mode = ArtifactLoadMode::kMmap;
  bool verify_manifest_files = true;
  bool materialize_weight_arena = true;
};

class RuntimeEnvironment {
 public:
  static std::unique_ptr<RuntimeEnvironment> Build(
      const PackedModelManifest& manifest,
      const RuntimeBootstrapOptions& options);
  static std::unique_ptr<RuntimeEnvironment> BuildFromManifestFile(
      const std::filesystem::path& manifest_path,
      const RuntimeBootstrapOptions& options);

  RuntimeEnvironment(RuntimeEnvironment&&) noexcept;
  RuntimeEnvironment& operator=(RuntimeEnvironment&&) noexcept;
  ~RuntimeEnvironment();

  RuntimeEnvironment(const RuntimeEnvironment&) = delete;
  RuntimeEnvironment& operator=(const RuntimeEnvironment&) = delete;

  const RuntimeConfig& config() const;
  const LoaderPlan& loader_plan() const;
  std::size_t effective_shared_cache_budget_bytes() const;
  bool has_artifact_loader() const;
  ArtifactLoader* artifact_loader();
  const ArtifactLoader* artifact_loader() const;
  bool has_tensor_catalog() const;
  TensorCatalog* tensor_catalog();
  const TensorCatalog* tensor_catalog() const;
  bool has_model_schedule() const;
  ModelSchedule* model_schedule();
  const ModelSchedule* model_schedule() const;
  bool has_weight_arena_plan() const;
  WeightArenaPlan* weight_arena_plan();
  const WeightArenaPlan* weight_arena_plan() const;
  bool has_weight_arena() const;
  WeightArena* weight_arena();
  const WeightArena* weight_arena() const;
  bool has_kernel_catalog() const;
  KernelCatalog* kernel_catalog();
  const KernelCatalog* kernel_catalog() const;
  bool has_gemm_catalog() const;
  GemmCatalog* gemm_catalog();
  const GemmCatalog* gemm_catalog() const;
  bool has_embedding_catalog() const;
  EmbeddingCatalog* embedding_catalog();
  const EmbeddingCatalog* embedding_catalog() const;
  bool has_gemm_heuristic_cache() const;
  GemmHeuristicCache* gemm_heuristic_cache();
  const GemmHeuristicCache* gemm_heuristic_cache() const;
  PrefixCache& prefix_cache();
  const PrefixCache& prefix_cache() const;
  ReusableStateArena& reusable_state_arena();
  const ReusableStateArena& reusable_state_arena() const;

 private:
  RuntimeEnvironment(
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
      std::unique_ptr<PrefixCache> prefix_cache);

  RuntimeConfig config_;
  LoaderPlan loader_plan_;
  std::size_t effective_shared_cache_budget_bytes_ = 0;
  std::unique_ptr<ArtifactLoader> artifact_loader_;
  std::unique_ptr<TensorCatalog> tensor_catalog_;
  std::unique_ptr<ModelSchedule> model_schedule_;
  std::unique_ptr<WeightArenaPlan> weight_arena_plan_;
  std::unique_ptr<WeightArena> weight_arena_;
  std::unique_ptr<KernelCatalog> kernel_catalog_;
  std::unique_ptr<GemmCatalog> gemm_catalog_;
  std::unique_ptr<EmbeddingCatalog> embedding_catalog_;
  std::unique_ptr<GemmHeuristicCache> gemm_heuristic_cache_;
  std::unique_ptr<ReusableStateArena> reusable_state_arena_;
  std::unique_ptr<PrefixCache> prefix_cache_;
};

}  // namespace nemotron
