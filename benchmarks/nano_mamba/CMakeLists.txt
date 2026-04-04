add_executable(
  nano_mamba_bench
  nano_mamba_bench.cpp
)

target_link_libraries(nano_mamba_bench PRIVATE nemotron_runtime_cache)
target_compile_features(nano_mamba_bench PRIVATE cxx_std_17)
target_compile_definitions(
  nano_mamba_bench
  PRIVATE
    NEMOTRON_MAMBA_LAYER_BENCH_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/testing/oracle/mamba_layer0_decode_block"
)
