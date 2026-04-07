if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Must use 121a (not 121) to enable conditional MMA arch features
  # required by CUTLASS block-scaled TensorOps on SM121.
  set(CMAKE_CUDA_ARCHITECTURES 121a)
endif()

option(NEMOTRON_RUNTIME_ENABLE_PROBES "Build pre-implementation feasibility probes" ON)
option(NEMOTRON_RUNTIME_BUILD_TESTING "Build runtime tests" ON)
option(NEMOTRON_RUNTIME_BUILD_BENCHMARKS "Build runtime benchmarks" ON)
option(
  NEMOTRON_RUNTIME_BUILD_INTERACTIVE_FORWARD
  "Build the interactive forward helper and server"
  ON
)
option(
  NEMOTRON_RUNTIME_ENFORCE_MANIFEST_BACKED_TESTS
  "Require NEMOTRON_FORWARD_MANIFEST before running manifest-backed tests under CTest"
  OFF
)

add_subdirectory(runtime)

if(BUILD_TESTING AND NEMOTRON_RUNTIME_BUILD_TESTING)
  add_subdirectory(testing)
endif()

if(NEMOTRON_RUNTIME_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(NEMOTRON_RUNTIME_ENABLE_PROBES)
  add_subdirectory(tools/probes)
endif()

if(NEMOTRON_RUNTIME_BUILD_INTERACTIVE_FORWARD)
  add_subdirectory(tools/interactive_forward)
endif()
