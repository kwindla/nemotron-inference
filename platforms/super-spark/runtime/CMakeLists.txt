find_package(CUDAToolkit REQUIRED)

# CUTLASS (header-only, used by the NVFP4 grouped GEMM kernel)
include(FetchContent)
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v4.4.2
  GIT_SHALLOW TRUE
)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cutlass)

set(NEMOTRON_CUDNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cudnn-frontend/include")
find_path(
  NEMOTRON_CUDNN_INCLUDE_DIR
  cudnn.h
  PATHS
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
)
find_library(
  NEMOTRON_CUDNN_LIBRARY
  NAMES cudnn libcudnn.so.9
  PATHS
    /usr/lib
    /usr/lib64
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)

set(NEMOTRON_RUNTIME_HAVE_CUDNN_FE FALSE)
if(EXISTS "${NEMOTRON_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h" AND
   NEMOTRON_CUDNN_INCLUDE_DIR AND
   NEMOTRON_CUDNN_LIBRARY)
  set(NEMOTRON_RUNTIME_HAVE_CUDNN_FE TRUE)
endif()
set(NEMOTRON_RUNTIME_HAVE_CUDNN_FE ${NEMOTRON_RUNTIME_HAVE_CUDNN_FE} PARENT_SCOPE)

add_library(
  nemotron_runtime_state
  src/attention/paged_attention_plan.cpp
  src/attention/paged_kv_cache.cpp
  src/cache/reusable_state.cpp
)

target_include_directories(
  nemotron_runtime_state
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(nemotron_runtime_state PUBLIC cxx_std_17)

add_library(
  nemotron_runtime_cache
  src/cache/prefix_cache.cpp
)

target_include_directories(
  nemotron_runtime_cache
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  nemotron_runtime_cache
  PUBLIC
    nemotron_runtime_state
)

target_compile_features(nemotron_runtime_cache PUBLIC cxx_std_17)

add_library(
  nemotron_runtime_planning
  src/planning/memory_budget.cpp
)

target_include_directories(
  nemotron_runtime_planning
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(nemotron_runtime_planning PUBLIC cxx_std_17)

add_library(
  nemotron_runtime_loader
  src/loader/artifact_loader.cpp
  src/loader/cublaslt_gemm_plan.cpp
  src/loader/embedding_catalog.cpp
  src/loader/gemm_catalog.cpp
  src/loader/gemm_execution.cpp
  src/loader/gemm_planner.cpp
  src/loader/kernel_catalog.cpp
  src/loader/manifest.cpp
  src/loader/model_schedule.cpp
  src/loader/loader.cpp
  src/loader/tensor_catalog.cpp
  src/loader/weight_arena.cpp
  src/loader/weight_arena_plan.cpp
)

target_include_directories(
  nemotron_runtime_loader
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  nemotron_runtime_loader
  PUBLIC
    nemotron_runtime_planning
)

target_compile_features(nemotron_runtime_loader PUBLIC cxx_std_17)

add_library(
  nemotron_runtime_backend
  src/backend/cublaslt_handle.cpp
  src/backend/attention_layout.cu
  src/backend/attention_layer.cpp
  src/backend/dense_reference_gemm.cu
  src/backend/dense_weight.cpp
  src/backend/device_nvfp4_matrix.cu
  src/backend/device_tensor.cpp
  src/backend/dense_gemm_runner.cpp
  src/backend/embedding_projection_fragment.cpp
  src/backend/embedding_table.cu
  src/backend/expert_layer.cpp
  src/backend/expert_ops.cu
  src/backend/flashinfer_layout.cpp
  src/backend/flashinfer_moe_backend.cpp
  src/backend/linear_op.cpp
  src/backend/mamba_layer.cpp
  src/backend/mamba_ops.cu
  src/backend/nvfp4_gemm_runner.cpp
  src/backend/nvfp4_packing.cpp
  src/backend/nvfp4_scale_layout.cpp
  src/backend/nvfp4_weight.cpp
  src/backend/primitive_ops.cu
  src/backend/request_context.cpp
  src/api/runtime_stats.cpp
  src/backend/scaled_fp8_linear.cu
  src/backend/storage_conversion.cu
  src/backend/cutlass_fp8_gemm.cu
  src/backend/cutlass_nvfp4_grouped_gemm.cu
)

# The CUTLASS GEMM files need special compile flags:
# - sm_121a for conditional MMA arch features
# - CUTLASS include paths (heavy headers, isolated to this one file)
set_source_files_properties(
  src/backend/cutlass_fp8_gemm.cu
  src/backend/cutlass_nvfp4_grouped_gemm.cu
  PROPERTIES
    COMPILE_OPTIONS "--expt-relaxed-constexpr;-DCUTLASS_SM100_FAMILY_ARCHS_ENABLED"
)
target_include_directories(
  nemotron_runtime_backend
  PRIVATE
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/tools/util/include
)

if(NEMOTRON_RUNTIME_HAVE_CUDNN_FE)
  target_sources(
    nemotron_runtime_backend
    PRIVATE
      src/backend/cudnn_handle.cpp
      src/backend/cudnn_paged_attention.cpp
  )
endif()

target_include_directories(
  nemotron_runtime_backend
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(NEMOTRON_RUNTIME_HAVE_CUDNN_FE)
  target_include_directories(
    nemotron_runtime_backend
    PRIVATE
      ${NEMOTRON_CUDNN_INCLUDE_DIR}
      ${NEMOTRON_CUDNN_FRONTEND_INCLUDE_DIR}
  )
endif()

target_link_libraries(
  nemotron_runtime_backend
  PUBLIC
    nemotron_runtime_state
    nemotron_runtime_loader
    CUDA::cudart
    CUDA::cublasLt
    ${CMAKE_DL_LIBS}
)

if(NEMOTRON_RUNTIME_HAVE_CUDNN_FE)
  target_link_libraries(
    nemotron_runtime_backend
    PUBLIC
      ${NEMOTRON_CUDNN_LIBRARY}
      CUDA::nvrtc
  )
endif()

target_compile_features(nemotron_runtime_backend PUBLIC cxx_std_17)

option(
  NEMOTRON_RUNTIME_BUILD_FLASHINFER_PLUGIN
  "Build the optional repo-local FlashInfer routed MoE plugin when external sources are available"
  ON
)

set(
  NEMOTRON_FLASHINFER_SOURCE_ROOT
  "/tmp/flashinfer-src"
  CACHE PATH
  "Path to the local FlashInfer source tree used for the routed MoE plugin"
)
set(
  NEMOTRON_FLASHINFER_CUTLASS_GENERATED_ROOT
  "/home/khkramer/.cache/flashinfer/0.6.5/121a/generated/cutlass_instantiations/120"
  CACHE PATH
  "Path to the local FlashInfer generated CUTLASS fused-MoE instantiations"
)
set(
  NEMOTRON_TRTLLM_SHALLOW_ROOT
  "/tmp/tensorrt-llm-shallow"
  CACHE PATH
  "Path to the local TensorRT-LLM shallow tree that provides generated batched GEMM exports"
)

set(
  NEMOTRON_TRTLLM_BMM_EXPORT_ROOT
  "${NEMOTRON_TRTLLM_SHALLOW_ROOT}/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export"
)

set(NEMOTRON_FLASHINFER_HAVE_CUTLASS_FUSED_MOE FALSE)
if(
  EXISTS "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu" AND
  EXISTS "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu" AND
  EXISTS "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
)
  file(
    GLOB_RECURSE NEMOTRON_FLASHINFER_CUTLASS_GENERATED_SOURCES
    CONFIGURE_DEPENDS
    "${NEMOTRON_FLASHINFER_CUTLASS_GENERATED_ROOT}/*.generated.cu"
  )
  if(NEMOTRON_FLASHINFER_CUTLASS_GENERATED_SOURCES)
    set(NEMOTRON_FLASHINFER_HAVE_CUTLASS_FUSED_MOE TRUE)
    message(
      STATUS
      "FlashInfer CUTLASS fused-MoE sources detected at ${NEMOTRON_FLASHINFER_CUTLASS_GENERATED_ROOT}"
    )
  else()
    message(
      STATUS
      "FlashInfer CUTLASS fused-MoE source tree found, but no generated instantiations were detected under ${NEMOTRON_FLASHINFER_CUTLASS_GENERATED_ROOT}"
    )
  endif()
endif()

if(
  NEMOTRON_RUNTIME_BUILD_FLASHINFER_PLUGIN AND
  EXISTS "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/include/flashinfer/trtllm/fused_moe/runner.h" AND
  EXISTS "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/trtllm_batched_gemm_runner.cu" AND
  EXISTS "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/trtllm_fused_moe_runner.cu" AND
  EXISTS "${NEMOTRON_TRTLLM_BMM_EXPORT_ROOT}/BatchedGemmInterface.h"
)
  set(NEMOTRON_FLASHINFER_EXPORT_OVERLAY "${CMAKE_CURRENT_BINARY_DIR}/generated/flashinfer_overlay")
  file(MAKE_DIRECTORY "${NEMOTRON_FLASHINFER_EXPORT_OVERLAY}/flashinfer/trtllm/batched_gemm")
  file(CREATE_LINK
    "${NEMOTRON_TRTLLM_BMM_EXPORT_ROOT}"
    "${NEMOTRON_FLASHINFER_EXPORT_OVERLAY}/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export"
    SYMBOLIC
    COPY_ON_ERROR
  )

  file(
    GLOB NEMOTRON_FLASHINFER_BMM_CUBIN_SOURCES
    CONFIGURE_DEPENDS
    "${NEMOTRON_TRTLLM_BMM_EXPORT_ROOT}/cubins/*.cpp"
  )

  file(
    GLOB NEMOTRON_FLASHINFER_ROUTING_SOURCES
    CONFIGURE_DEPENDS
    "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/trtllm_backend/routingDeepSeek/*.cu"
    "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/trtllm_backend/routingRenormalize/*.cu"
  )

  if(NEMOTRON_FLASHINFER_BMM_CUBIN_SOURCES)
    add_library(
      nemotron_flashinfer_moe_plugin
      SHARED
      src/backend/flashinfer_moe_plugin.cu
      src/backend/flashinfer_cuda_driver_wrapper.cpp
      src/backend/flashinfer_debug_config.cpp
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/trtllm_batched_gemm_runner.cu"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/trtllm_fused_moe_runner.cu"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_renormalize.cu"
      ${NEMOTRON_FLASHINFER_ROUTING_SOURCES}
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/nv_internal/cpp/common/envUtils.cpp"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/nv_internal/cpp/common/logger.cpp"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/nv_internal/cpp/common/stringUtils.cpp"
      "${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/nv_internal/cpp/common/tllmException.cpp"
      ${NEMOTRON_FLASHINFER_BMM_CUBIN_SOURCES}
    )

    target_include_directories(
      nemotron_flashinfer_moe_plugin
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
      PRIVATE
        ${NEMOTRON_FLASHINFER_SOURCE_ROOT}/include
        ${NEMOTRON_FLASHINFER_SOURCE_ROOT}/3rdparty/cutlass/include
        ${NEMOTRON_FLASHINFER_SOURCE_ROOT}/3rdparty/cutlass/tools/util/include
        ${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc
        ${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/nv_internal/include
        ${NEMOTRON_FLASHINFER_SOURCE_ROOT}/csrc/nv_internal
        ${NEMOTRON_FLASHINFER_EXPORT_OVERLAY}
        ${NEMOTRON_TRTLLM_SHALLOW_ROOT}/cpp/include
        ${NEMOTRON_TRTLLM_SHALLOW_ROOT}/cpp
    )

    target_compile_definitions(
      nemotron_flashinfer_moe_plugin
      PRIVATE
        TLLM_GEN_EXPORT_INTERFACE
        TLLM_ENABLE_CUDA
    )

    target_link_libraries(
      nemotron_flashinfer_moe_plugin
      PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
    )

    target_compile_features(nemotron_flashinfer_moe_plugin PUBLIC cxx_std_17)
    target_compile_options(
      nemotron_flashinfer_moe_plugin
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    set_target_properties(
      nemotron_flashinfer_moe_plugin
      PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME nemotron_flashinfer_moe
        POSITION_INDEPENDENT_CODE ON
    )
  else()
    message(
      STATUS
      "Skipping nemotron_flashinfer_moe_plugin: no compatible generated BMM cubin wrappers were found"
    )
  endif()
else()
  message(
    STATUS
    "Skipping nemotron_flashinfer_moe_plugin: FlashInfer or TRT-LLM export sources are unavailable"
  )
endif()

add_library(
  nemotron_runtime_api
  src/api/model_cache.cpp
  src/api/single_token_forward_model.cpp
  src/api/runtime_config.cpp
  src/api/runtime_environment.cpp
)

target_include_directories(
  nemotron_runtime_api
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  nemotron_runtime_api
  PUBLIC
    nemotron_runtime_backend
    nemotron_runtime_cache
    nemotron_runtime_loader
)

target_compile_features(nemotron_runtime_api PUBLIC cxx_std_17)
