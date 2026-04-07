find_package(CUDAToolkit REQUIRED)

add_executable(cublaslt_probe cublaslt_probe.cu)
target_link_libraries(cublaslt_probe PRIVATE CUDA::cudart CUDA::cublasLt)
target_compile_features(cublaslt_probe PRIVATE cxx_std_17)
