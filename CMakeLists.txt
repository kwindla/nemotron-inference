cmake_minimum_required(VERSION 3.28)

# CMAKE_CUDA_ARCHITECTURES must be set BEFORE project(... LANGUAGES CUDA) or
# CMake's CUDA-compiler detection will auto-populate it based on whatever nvcc
# picks as its default (e.g. 75), and the fallback below would silently never
# run. That mismatch builds for the wrong arch, which manifests as incomplete
# wmma fragment types at compile time and wrong-arch PTX / perf regressions at
# runtime. Treat this as a hard invariant of the build: single-target repo,
# single CUDA architecture, do not let auto-detection win.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Target is GeForce RTX 5090 (SM120). The "a" suffix enables architecture-
  # specific features (block-scaled FP4 MMA via SM120_16x8x64_TN_VS) that the
  # nemotron-inference hot path depends on.
  # GB10/DGX Spark-specific builds should override this explicitly to 121a.
  set(CMAKE_CUDA_ARCHITECTURES 120a)
endif()

project(nemotron_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CTest)

option(NEMOTRON_RUNTIME_ENABLE_PROBES "Build pre-implementation feasibility probes" ON)
option(NEMOTRON_RUNTIME_BUILD_TESTING "Build runtime tests" ON)
option(NEMOTRON_RUNTIME_BUILD_BENCHMARKS "Build runtime benchmarks" ON)
option(NEMOTRON_RUNTIME_BUILD_TOOLS "Build runtime tools" ON)
option(NEMOTRON_PRODUCTION_BUILD "Compile out debug-only runtime scaffolding" OFF)

add_subdirectory(runtime)

if(BUILD_TESTING AND NEMOTRON_RUNTIME_BUILD_TESTING)
  add_subdirectory(testing)
endif()

if(NEMOTRON_RUNTIME_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(NEMOTRON_RUNTIME_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
