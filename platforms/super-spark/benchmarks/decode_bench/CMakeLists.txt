add_executable(
  single_token_decode_bench
  single_token_decode_bench.cpp
)

target_link_libraries(
  single_token_decode_bench
  PRIVATE
    nemotron_runtime_api
)

target_compile_features(single_token_decode_bench PRIVATE cxx_std_17)
