cmake_minimum_required(VERSION 3.28)

project(nemotron_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CTest)

# Select which platform specialization(s) to build.
# Override on the command line: -DNEMOTRON_PLATFORM=nano-5090
set(NEMOTRON_PLATFORM "super-spark" CACHE STRING
    "Platform specialization to build (super-spark, nano-5090)")

add_subdirectory(platforms/${NEMOTRON_PLATFORM})
