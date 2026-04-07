add_executable(
  gb10_loader_modes_bench
  gb10_loader_modes_bench.cpp
)

target_link_libraries(
  gb10_loader_modes_bench
  PRIVATE
    nemotron_runtime_loader
)

target_compile_features(gb10_loader_modes_bench PRIVATE cxx_std_17)
