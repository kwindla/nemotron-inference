add_executable(
  nano_paged_attention_bench
  nano_paged_attention_bench.cpp
)

target_include_directories(
  nano_paged_attention_bench
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  nano_paged_attention_bench
  PRIVATE
    nemotron_runtime_backend
    ${CMAKE_DL_LIBS}
)

target_compile_features(nano_paged_attention_bench PRIVATE cxx_std_17)
