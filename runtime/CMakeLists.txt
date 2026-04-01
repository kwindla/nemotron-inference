find_package(CUDAToolkit REQUIRED)

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

target_link_libraries(
  nemotron_runtime_state
  PUBLIC
    CUDA::cudart
)

target_include_directories(
  nemotron_runtime_state
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(nemotron_runtime_state PUBLIC cxx_std_17)

add_library(
  nemotron_runtime_cache
  src/api/state_snapshot.cpp
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
  src/backend/attention_layer.cpp
  src/backend/attention_device_fallback.cu
  src/backend/dense_weight.cpp
  src/backend/device_nvfp4_matrix.cu
  src/backend/device_tensor.cpp
  src/backend/dense_gemm_runner.cpp
  src/backend/embedding_projection_fragment.cpp
  src/backend/embedding_table.cu
  src/backend/expert_layer.cpp
  src/backend/fused_mamba_decode.cu
  src/backend/fused_moe_decode.cu
  src/backend/linear_op.cpp
  src/backend/linear_op_counters.cpp
  src/backend/linear_reference_kernels.cu
  src/backend/mamba_layer.cpp
  src/backend/nvfp4_gemm_runner.cpp
  src/backend/nvfp4_packing.cpp
  src/backend/nvfp4_scale_layout.cpp
  src/backend/nvfp4_weight.cpp
  src/backend/primitive_ops.cu
  src/backend/request_context.cpp
  src/backend/scaled_fp8_linear.cu
)

if(NEMOTRON_RUNTIME_HAVE_CUDNN_FE)
  target_sources(
    nemotron_runtime_backend
    PRIVATE
      src/backend/cudnn_handle.cpp
      src/backend/cudnn_paged_attention.cpp
  )
else()
  target_sources(
    nemotron_runtime_backend
    PRIVATE
      src/backend/cudnn_handle_stub.cpp
      src/backend/cudnn_paged_attention_stub.cpp
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

target_link_libraries(
  nemotron_runtime_cache
  PUBLIC
    nemotron_runtime_backend
)

add_library(
  nemotron_runtime_api
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
