add_executable(
  prefill_trace_dump
  prefill_trace_dump.cpp
)

target_link_libraries(
  prefill_trace_dump
  PRIVATE
    nemotron_runtime_api
)

target_compile_features(prefill_trace_dump PRIVATE cxx_std_17)
