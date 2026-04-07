find_package(CUDAToolkit REQUIRED)

add_executable(
  gb10_mamba_cache_bench
  gb10_mamba_cache_bench.cu
)

target_link_libraries(gb10_mamba_cache_bench PRIVATE CUDA::cudart)
target_compile_features(gb10_mamba_cache_bench PRIVATE cxx_std_17)
target_compile_definitions(
  gb10_mamba_cache_bench
  PRIVATE
    NEMOTRON_MAMBA_ORACLE_FIXTURE_ROOT="${PROJECT_SOURCE_DIR}/testing/oracle/mamba_layer0_decode_update"
    NEMOTRON_MAMBA_TARGET_TRACE_ROOT="${PROJECT_SOURCE_DIR}/testing/oracle/mamba_layer0_target_chat_trace"
)
