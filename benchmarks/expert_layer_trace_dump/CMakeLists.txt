add_executable(
  expert_layer_trace_dump
  expert_layer_trace_dump.cpp
)

target_link_libraries(
  expert_layer_trace_dump
  PRIVATE
    nemotron_runtime_api
    nemotron_runtime_backend
    nemotron_runtime_loader
    nemotron_runtime_planning
    nemotron_runtime_state
)

target_compile_features(expert_layer_trace_dump PRIVATE cxx_std_17)
