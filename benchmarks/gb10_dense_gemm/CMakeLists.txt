add_executable(
  gb10_dense_gemm_bench
  gb10_dense_gemm_bench.cpp
)

target_link_libraries(gb10_dense_gemm_bench PRIVATE nemotron_runtime_backend)
target_compile_features(gb10_dense_gemm_bench PRIVATE cxx_std_17)
