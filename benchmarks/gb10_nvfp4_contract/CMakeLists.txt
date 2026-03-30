find_package(CUDAToolkit REQUIRED)

add_executable(
  gb10_nvfp4_contract_bench
  gb10_nvfp4_contract_bench.cu
)

target_link_libraries(gb10_nvfp4_contract_bench PRIVATE CUDA::cudart CUDA::cublasLt)
target_compile_features(gb10_nvfp4_contract_bench PRIVATE cxx_std_17)
