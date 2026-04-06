set(NEMOTRON_TEST_ENVIRONMENT "")
if(EXISTS "/usr/local/cuda/compat")
  set(NEMOTRON_TEST_ENVIRONMENT "LD_LIBRARY_PATH=/usr/local/cuda/compat:$ENV{LD_LIBRARY_PATH}")
elseif(EXISTS "/usr/local/cuda-13.2/compat")
  set(NEMOTRON_TEST_ENVIRONMENT "LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:$ENV{LD_LIBRARY_PATH}")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(nemotron_add_test name)
  add_test(NAME ${name} COMMAND ${name})
  if(NEMOTRON_TEST_ENVIRONMENT)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${NEMOTRON_TEST_ENVIRONMENT}")
  endif()
endfunction()

add_executable(
  prefix_cache_test
  cache/prefix_cache_test.cpp
)

target_link_libraries(prefix_cache_test PRIVATE nemotron_runtime_cache)
target_compile_features(prefix_cache_test PRIVATE cxx_std_17)

nemotron_add_test(prefix_cache_test)

add_executable(
  reusable_state_test
  cache/reusable_state_test.cpp
)

target_link_libraries(reusable_state_test PRIVATE nemotron_runtime_state)
target_compile_features(reusable_state_test PRIVATE cxx_std_17)

nemotron_add_test(reusable_state_test)

add_executable(
  paged_kv_cache_test
  attention/paged_kv_cache_test.cpp
)

target_link_libraries(paged_kv_cache_test PRIVATE nemotron_runtime_state)
target_compile_features(paged_kv_cache_test PRIVATE cxx_std_17)

nemotron_add_test(paged_kv_cache_test)

add_executable(
  paged_attention_plan_test
  attention/paged_attention_plan_test.cpp
)

target_link_libraries(paged_attention_plan_test PRIVATE nemotron_runtime_state)
target_compile_features(paged_attention_plan_test PRIVATE cxx_std_17)

nemotron_add_test(paged_attention_plan_test)

add_executable(
  memory_budget_test
  planning/memory_budget_test.cpp
)

target_link_libraries(memory_budget_test PRIVATE nemotron_runtime_planning)
target_compile_features(memory_budget_test PRIVATE cxx_std_17)

nemotron_add_test(memory_budget_test)

add_executable(
  loader_plan_test
  loader/loader_plan_test.cpp
)

target_link_libraries(loader_plan_test PRIVATE nemotron_runtime_loader)
target_compile_features(loader_plan_test PRIVATE cxx_std_17)

nemotron_add_test(loader_plan_test)

add_executable(
  manifest_io_test
  loader/manifest_io_test.cpp
)

target_link_libraries(manifest_io_test PRIVATE nemotron_runtime_loader nemotron_runtime_api)
target_compile_features(manifest_io_test PRIVATE cxx_std_17)

nemotron_add_test(manifest_io_test)

add_executable(
  model_schedule_test
  loader/model_schedule_test.cpp
)

target_link_libraries(model_schedule_test PRIVATE nemotron_runtime_loader)
target_compile_features(model_schedule_test PRIVATE cxx_std_17)

nemotron_add_test(model_schedule_test)

add_executable(
  artifact_loader_test
  loader/artifact_loader_test.cpp
)

target_link_libraries(artifact_loader_test PRIVATE nemotron_runtime_loader)
target_compile_features(artifact_loader_test PRIVATE cxx_std_17)

nemotron_add_test(artifact_loader_test)

add_executable(
  tensor_catalog_test
  loader/tensor_catalog_test.cpp
)

target_link_libraries(tensor_catalog_test PRIVATE nemotron_runtime_loader)
target_compile_features(tensor_catalog_test PRIVATE cxx_std_17)

nemotron_add_test(tensor_catalog_test)

add_executable(
  weight_arena_plan_test
  loader/weight_arena_plan_test.cpp
)

target_link_libraries(weight_arena_plan_test PRIVATE nemotron_runtime_loader)
target_compile_features(weight_arena_plan_test PRIVATE cxx_std_17)

nemotron_add_test(weight_arena_plan_test)

add_executable(
  weight_arena_test
  loader/weight_arena_test.cpp
)

target_link_libraries(weight_arena_test PRIVATE nemotron_runtime_loader)
target_compile_features(weight_arena_test PRIVATE cxx_std_17)

nemotron_add_test(weight_arena_test)

add_executable(
  kernel_catalog_test
  loader/kernel_catalog_test.cpp
)

target_link_libraries(kernel_catalog_test PRIVATE nemotron_runtime_loader)
target_compile_features(kernel_catalog_test PRIVATE cxx_std_17)

nemotron_add_test(kernel_catalog_test)

add_executable(
  gemm_catalog_test
  loader/gemm_catalog_test.cpp
)

target_link_libraries(gemm_catalog_test PRIVATE nemotron_runtime_loader)
target_compile_features(gemm_catalog_test PRIVATE cxx_std_17)

nemotron_add_test(gemm_catalog_test)

add_executable(
  embedding_catalog_test
  loader/embedding_catalog_test.cpp
)

target_link_libraries(embedding_catalog_test PRIVATE nemotron_runtime_loader)
target_compile_features(embedding_catalog_test PRIVATE cxx_std_17)

nemotron_add_test(embedding_catalog_test)

add_executable(
  gemm_planner_test
  loader/gemm_planner_test.cpp
)

target_link_libraries(gemm_planner_test PRIVATE nemotron_runtime_loader)
target_compile_features(gemm_planner_test PRIVATE cxx_std_17)

nemotron_add_test(gemm_planner_test)

add_executable(
  gemm_execution_test
  loader/gemm_execution_test.cpp
)

target_link_libraries(gemm_execution_test PRIVATE nemotron_runtime_loader)
target_compile_features(gemm_execution_test PRIVATE cxx_std_17)

nemotron_add_test(gemm_execution_test)

add_executable(
  cublaslt_gemm_plan_test
  loader/cublaslt_gemm_plan_test.cpp
)

target_link_libraries(cublaslt_gemm_plan_test PRIVATE nemotron_runtime_loader)
target_compile_features(cublaslt_gemm_plan_test PRIVATE cxx_std_17)

nemotron_add_test(cublaslt_gemm_plan_test)

add_executable(
  device_tensor_test
  backend/device_tensor_test.cpp
)

target_link_libraries(device_tensor_test PRIVATE nemotron_runtime_backend)
target_compile_features(device_tensor_test PRIVATE cxx_std_17)

nemotron_add_test(device_tensor_test)

add_executable(
  device_argmax_test
  backend/device_argmax_test.cpp
)

target_link_libraries(device_argmax_test PRIVATE nemotron_runtime_backend)
target_compile_features(device_argmax_test PRIVATE cxx_std_17)

nemotron_add_test(device_argmax_test)

add_executable(
  expert_routing_device_test
  backend/expert_routing_device_test.cpp
)

target_link_libraries(expert_routing_device_test PRIVATE nemotron_runtime_backend)
target_compile_features(expert_routing_device_test PRIVATE cxx_std_17)

nemotron_add_test(expert_routing_device_test)

add_executable(
  expert_layer_fastpath_test
  backend/expert_layer_fastpath_test.cpp
)

target_link_libraries(expert_layer_fastpath_test PRIVATE nemotron_runtime_backend)
target_compile_features(expert_layer_fastpath_test PRIVATE cxx_std_17)

nemotron_add_test(expert_layer_fastpath_test)

add_executable(
  fused_moe_prefill_test
  backend/fused_moe_prefill_test.cpp
)

target_link_libraries(fused_moe_prefill_test PRIVATE nemotron_runtime_backend)
target_compile_features(fused_moe_prefill_test PRIVATE cxx_std_17)

nemotron_add_test(fused_moe_prefill_test)

add_executable(
  device_nvfp4_matrix_test
  backend/device_nvfp4_matrix_test.cpp
)

target_link_libraries(device_nvfp4_matrix_test PRIVATE nemotron_runtime_backend)
target_compile_features(device_nvfp4_matrix_test PRIVATE cxx_std_17)

nemotron_add_test(device_nvfp4_matrix_test)

add_executable(
  dense_weight_test
  backend/dense_weight_test.cpp
)

target_link_libraries(dense_weight_test PRIVATE nemotron_runtime_backend)
target_compile_features(dense_weight_test PRIVATE cxx_std_17)

nemotron_add_test(dense_weight_test)

add_executable(
  linear_op_test
  backend/linear_op_test.cpp
)

target_link_libraries(linear_op_test PRIVATE nemotron_runtime_backend)
target_compile_features(linear_op_test PRIVATE cxx_std_17)

nemotron_add_test(linear_op_test)

add_executable(
  scaled_fp8_linear_test
  backend/scaled_fp8_linear_test.cpp
)

target_link_libraries(scaled_fp8_linear_test PRIVATE nemotron_runtime_backend)
target_compile_features(scaled_fp8_linear_test PRIVATE cxx_std_17)

nemotron_add_test(scaled_fp8_linear_test)

add_executable(
  scaled_fp8_linear_compare_test
  backend/scaled_fp8_linear_compare_test.cpp
)

target_link_libraries(scaled_fp8_linear_compare_test PRIVATE nemotron_runtime_backend)
target_compile_features(scaled_fp8_linear_compare_test PRIVATE cxx_std_17)

nemotron_add_test(scaled_fp8_linear_compare_test)

# Super-model oracle fixtures were removed from this Nano branch because they
# were generated from incompatible Super checkpoints. Regenerate them from a
# Super checkpoint if those oracle tests are needed again.

add_executable(
  request_context_test
  backend/request_context_test.cpp
)

target_link_libraries(request_context_test PRIVATE nemotron_runtime_backend)
target_compile_features(request_context_test PRIVATE cxx_std_17)

nemotron_add_test(request_context_test)

add_executable(
  nano_paged_attention_test
  backend/nano_paged_attention_test.cpp
)

target_include_directories(
  nano_paged_attention_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks/nano_paged_attention
)

target_link_libraries(
  nano_paged_attention_test
  PRIVATE
    nemotron_runtime_backend
    ${CMAKE_DL_LIBS}
)

target_compile_features(nano_paged_attention_test PRIVATE cxx_std_17)

nemotron_add_test(nano_paged_attention_test)

add_executable(
  nano_attention_layer_integration_test
  backend/nano_attention_layer_integration_test.cpp
)

target_link_libraries(
  nano_attention_layer_integration_test
  PRIVATE
    nemotron_runtime_backend
)

target_compile_features(nano_attention_layer_integration_test PRIVATE cxx_std_17)

nemotron_add_test(nano_attention_layer_integration_test)

add_executable(
  primitive_ops_test
  backend/primitive_ops_test.cpp
)

target_link_libraries(primitive_ops_test PRIVATE nemotron_runtime_backend)
target_compile_features(primitive_ops_test PRIVATE cxx_std_17)

nemotron_add_test(primitive_ops_test)

add_executable(
  attention_layer_test
  backend/attention_layer_test.cpp
)

target_link_libraries(attention_layer_test PRIVATE nemotron_runtime_backend)
target_compile_features(attention_layer_test PRIVATE cxx_std_17)

nemotron_add_test(attention_layer_test)

add_executable(
  nvfp4_weight_test
  backend/nvfp4_weight_test.cpp
)

target_link_libraries(nvfp4_weight_test PRIVATE nemotron_runtime_backend)
target_compile_features(nvfp4_weight_test PRIVATE cxx_std_17)

nemotron_add_test(nvfp4_weight_test)

add_executable(
  nvfp4_packing_test
  backend/nvfp4_packing_test.cpp
)

target_link_libraries(nvfp4_packing_test PRIVATE nemotron_runtime_backend)
target_compile_features(nvfp4_packing_test PRIVATE cxx_std_17)

nemotron_add_test(nvfp4_packing_test)

add_executable(
  nvfp4_pack_compare_test
  backend/nvfp4_pack_compare_test.cpp
)

target_link_libraries(nvfp4_pack_compare_test PRIVATE nemotron_runtime_backend)
target_compile_features(nvfp4_pack_compare_test PRIVATE cxx_std_17)

nemotron_add_test(nvfp4_pack_compare_test)

add_executable(
  nvfp4_linear_compare_test
  backend/nvfp4_linear_compare_test.cpp
)

target_link_libraries(nvfp4_linear_compare_test PRIVATE nemotron_runtime_backend)
target_compile_features(nvfp4_linear_compare_test PRIVATE cxx_std_17)

nemotron_add_test(nvfp4_linear_compare_test)

add_executable(
  nvfp4_gemm_runner_test
  backend/nvfp4_gemm_runner_test.cpp
)

target_link_libraries(nvfp4_gemm_runner_test PRIVATE nemotron_runtime_backend)
target_compile_features(nvfp4_gemm_runner_test PRIVATE cxx_std_17)

nemotron_add_test(nvfp4_gemm_runner_test)

add_executable(
  nvfp4_vllm_golden_test
  backend/nvfp4_vllm_golden_test.cpp
)

target_link_libraries(nvfp4_vllm_golden_test PRIVATE nemotron_runtime_backend)
target_compile_features(nvfp4_vllm_golden_test PRIVATE cxx_std_17)
target_compile_definitions(
  nvfp4_vllm_golden_test
  PRIVATE
    NEMOTRON_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

nemotron_add_test(nvfp4_vllm_golden_test)

add_test(
  NAME nvfp4_manifest_generation_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/backend/nvfp4_manifest_generation_test.py
)

add_executable(
  mamba_conv_prefill_test
  backend/mamba_conv_prefill_test.cpp
)

target_link_libraries(mamba_conv_prefill_test PRIVATE nemotron_runtime_backend)
target_compile_features(mamba_conv_prefill_test PRIVATE cxx_std_17)

nemotron_add_test(mamba_conv_prefill_test)

add_executable(
  mamba_ssd_prefill_test
  backend/mamba_ssd_prefill_test.cpp
)

target_link_libraries(mamba_ssd_prefill_test PRIVATE nemotron_runtime_backend)
target_compile_features(mamba_ssd_prefill_test PRIVATE cxx_std_17)

nemotron_add_test(mamba_ssd_prefill_test)

add_executable(
  mamba_gated_group_norm_test
  backend/mamba_gated_group_norm_test.cpp
)

target_link_libraries(mamba_gated_group_norm_test PRIVATE nemotron_runtime_backend)
target_compile_features(mamba_gated_group_norm_test PRIVATE cxx_std_17)

nemotron_add_test(mamba_gated_group_norm_test)

add_executable(
  embedding_lookup_test
  backend/embedding_lookup_test.cpp
)

target_link_libraries(embedding_lookup_test PRIVATE nemotron_runtime_backend)
target_compile_features(embedding_lookup_test PRIVATE cxx_std_17)

nemotron_add_test(embedding_lookup_test)

add_executable(
  dense_gemm_runner_test
  backend/dense_gemm_runner_test.cpp
)

target_link_libraries(dense_gemm_runner_test PRIVATE nemotron_runtime_backend)
target_compile_features(dense_gemm_runner_test PRIVATE cxx_std_17)

nemotron_add_test(dense_gemm_runner_test)

add_executable(
  embedding_projection_fragment_test
  backend/embedding_projection_fragment_test.cpp
)

target_link_libraries(embedding_projection_fragment_test PRIVATE nemotron_runtime_backend)
target_compile_features(embedding_projection_fragment_test PRIVATE cxx_std_17)

nemotron_add_test(embedding_projection_fragment_test)

if(NEMOTRON_RUNTIME_HAVE_CUDNN_FE)
  add_executable(
    cudnn_paged_attention_test
    backend/cudnn_paged_attention_test.cpp
  )

  target_link_libraries(cudnn_paged_attention_test PRIVATE nemotron_runtime_backend)
  target_compile_features(cudnn_paged_attention_test PRIVATE cxx_std_17)

  nemotron_add_test(cudnn_paged_attention_test)
endif()

add_executable(
  runtime_config_test
  api/runtime_config_test.cpp
)

target_link_libraries(runtime_config_test PRIVATE nemotron_runtime_api nemotron_runtime_cache nemotron_runtime_state)
target_compile_features(runtime_config_test PRIVATE cxx_std_17)

nemotron_add_test(runtime_config_test)

add_executable(
  runtime_environment_test
  api/runtime_environment_test.cpp
)

target_link_libraries(runtime_environment_test PRIVATE nemotron_runtime_api)
target_compile_features(runtime_environment_test PRIVATE cxx_std_17)

nemotron_add_test(runtime_environment_test)

add_executable(
  cudnn_feasibility_test
  api/cudnn_feasibility_test.cpp
)

target_link_libraries(cudnn_feasibility_test PRIVATE nemotron_runtime_api)
target_compile_features(cudnn_feasibility_test PRIVATE cxx_std_17)

nemotron_add_test(cudnn_feasibility_test)

add_executable(
  state_snapshot_test
  api/state_snapshot_test.cpp
)

target_link_libraries(state_snapshot_test PRIVATE nemotron_runtime_api)
target_compile_features(state_snapshot_test PRIVATE cxx_std_17)

nemotron_add_test(state_snapshot_test)

add_executable(
  full_forward_manifest_smoke_test
  api/full_forward_manifest_smoke_test.cpp
)

target_link_libraries(full_forward_manifest_smoke_test PRIVATE nemotron_runtime_api)
target_compile_features(full_forward_manifest_smoke_test PRIVATE cxx_std_17)

nemotron_add_test(full_forward_manifest_smoke_test)

add_executable(
  nano_global_root_prefix_cache_test
  api/nano_global_root_prefix_cache_test.cpp
)

target_link_libraries(nano_global_root_prefix_cache_test PRIVATE nemotron_runtime_api)
target_compile_features(nano_global_root_prefix_cache_test PRIVATE cxx_std_17)

nemotron_add_test(nano_global_root_prefix_cache_test)

add_executable(
  multi_turn_prefix_reuse_test
  api/multi_turn_prefix_reuse_test.cpp
)

target_link_libraries(multi_turn_prefix_reuse_test PRIVATE nemotron_runtime_api)
target_compile_features(multi_turn_prefix_reuse_test PRIVATE cxx_std_17)
target_compile_definitions(
  multi_turn_prefix_reuse_test
  PRIVATE
    NEMOTRON_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

nemotron_add_test(multi_turn_prefix_reuse_test)

add_executable(
  split_prefill_localization_test
  api/split_prefill_localization_test.cpp
)

target_link_libraries(split_prefill_localization_test PRIVATE nemotron_runtime_api)
target_compile_features(split_prefill_localization_test PRIVATE cxx_std_17)
target_compile_definitions(
  split_prefill_localization_test
  PRIVATE
    NEMOTRON_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

add_executable(
  nano_loader_budget_test
  api/nano_loader_budget_test.cpp
)

target_link_libraries(nano_loader_budget_test PRIVATE nemotron_runtime_api)
target_compile_features(nano_loader_budget_test PRIVATE cxx_std_17)
target_compile_definitions(
  nano_loader_budget_test
  PRIVATE
    NEMOTRON_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

nemotron_add_test(nano_loader_budget_test)

add_executable(
  nano_16_token_correctness_test
  api/nano_16_token_correctness_test.cpp
)

target_link_libraries(nano_16_token_correctness_test PRIVATE nemotron_runtime_api)
target_compile_features(nano_16_token_correctness_test PRIVATE cxx_std_17)

nemotron_add_test(nano_16_token_correctness_test)

add_executable(
  nano_save_prompt_oracle
  api/nano_save_prompt_oracle.cpp
)

target_link_libraries(nano_save_prompt_oracle PRIVATE nemotron_runtime_api)
target_compile_features(nano_save_prompt_oracle PRIVATE cxx_std_17)

nemotron_add_test(nano_save_prompt_oracle)

add_executable(
  single_token_forward_model_test
  api/single_token_forward_model_test.cpp
)

target_link_libraries(single_token_forward_model_test PRIVATE nemotron_runtime_api)
target_compile_features(single_token_forward_model_test PRIVATE cxx_std_17)

nemotron_add_test(single_token_forward_model_test)

add_executable(
  prefill_prefix_oracle_test
  api/prefill_prefix_oracle_test.cpp
)

target_link_libraries(prefill_prefix_oracle_test PRIVATE nemotron_runtime_api)
target_compile_features(prefill_prefix_oracle_test PRIVATE cxx_std_17)
target_compile_definitions(
  prefill_prefix_oracle_test
  PRIVATE
    NEMOTRON_PREFILL_PREFIX_ORACLE_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/testing/oracle/prefix_prefill_short_chat_layer7_t4_oracle"
)

nemotron_add_test(prefill_prefix_oracle_test)

add_executable(
  single_token_decode_oracle_test
  api/single_token_decode_oracle_test.cpp
)

target_link_libraries(single_token_decode_oracle_test PRIVATE nemotron_runtime_api)
target_compile_features(single_token_decode_oracle_test PRIVATE cxx_std_17)
target_compile_definitions(
  single_token_decode_oracle_test
  PRIVATE
    NEMOTRON_SINGLE_TOKEN_DECODE_ORACLE_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/testing/oracle/full_model_single_token_short_chat_cuda_v3"
)

nemotron_add_test(single_token_decode_oracle_test)

add_executable(
  expert_layer_oracle_test
  backend/expert_layer_oracle_test.cpp
)

target_link_libraries(expert_layer_oracle_test PRIVATE nemotron_runtime_backend)
target_compile_features(expert_layer_oracle_test PRIVATE cxx_std_17)
target_compile_definitions(
  expert_layer_oracle_test
  PRIVATE
    NEMOTRON_EXPERT_LAYER_ORACLE_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/testing/oracle/expert_layer1_decode_block"
)

nemotron_add_test(expert_layer_oracle_test)

add_executable(
  expert_layer8_oracle_test
  backend/expert_layer_oracle_test.cpp
)

target_link_libraries(expert_layer8_oracle_test PRIVATE nemotron_runtime_backend)
target_compile_features(expert_layer8_oracle_test PRIVATE cxx_std_17)
target_compile_definitions(
  expert_layer8_oracle_test
  PRIVATE
    NEMOTRON_EXPERT_LAYER_ORACLE_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/testing/oracle/expert_layer8_decode_block"
)

nemotron_add_test(expert_layer8_oracle_test)

add_executable(
  mamba_layer_oracle_test
  backend/mamba_layer_oracle_test.cpp
)

target_link_libraries(mamba_layer_oracle_test PRIVATE nemotron_runtime_api)
target_compile_features(mamba_layer_oracle_test PRIVATE cxx_std_17)
target_compile_definitions(
  mamba_layer_oracle_test
  PRIVATE
    NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/testing/oracle/mamba_layer0_decode_block"
)

nemotron_add_test(mamba_layer_oracle_test)

add_executable(
  mamba_layer9_oracle_test
  backend/mamba_layer_oracle_test.cpp
)

target_link_libraries(mamba_layer9_oracle_test PRIVATE nemotron_runtime_api)
target_compile_features(mamba_layer9_oracle_test PRIVATE cxx_std_17)
target_compile_definitions(
  mamba_layer9_oracle_test
  PRIVATE
    NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/testing/oracle/mamba_layer9_decode_block"
)

nemotron_add_test(mamba_layer9_oracle_test)
