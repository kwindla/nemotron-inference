cmake_minimum_required(VERSION 3.28)

project(nemotron_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CTest)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 121)
endif()

option(NEMOTRON_RUNTIME_ENABLE_PROBES "Build pre-implementation feasibility probes" ON)
option(NEMOTRON_RUNTIME_BUILD_TESTING "Build runtime tests" ON)
option(NEMOTRON_RUNTIME_BUILD_BENCHMARKS "Build runtime benchmarks" ON)

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
