add_executable(
  nano_routed_up_bench
  nano_routed_up_bench.cpp
)

target_link_libraries(nano_routed_up_bench PRIVATE nemotron_runtime_backend)
set_source_files_properties(
  nano_routed_up_bench.cpp
  PROPERTIES
    LANGUAGE CUDA
)
target_compile_features(nano_routed_up_bench PRIVATE cxx_std_17 cuda_std_17)
