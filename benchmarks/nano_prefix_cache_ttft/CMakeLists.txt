add_executable(
  nano_prefix_cache_ttft_bench
  nano_prefix_cache_ttft_bench.cpp
)

target_link_libraries(nano_prefix_cache_ttft_bench PRIVATE nemotron_runtime_api)
target_compile_features(nano_prefix_cache_ttft_bench PRIVATE cxx_std_17)

if(BUILD_TESTING)
  set(NEMOTRON_TTFT_TEST_ENVIRONMENT "")
  if(EXISTS "/usr/local/cuda/compat")
    set(NEMOTRON_TTFT_TEST_ENVIRONMENT "LD_LIBRARY_PATH=/usr/local/cuda/compat:$ENV{LD_LIBRARY_PATH}")
  elseif(EXISTS "/usr/local/cuda-13.2/compat")
    set(NEMOTRON_TTFT_TEST_ENVIRONMENT "LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:$ENV{LD_LIBRARY_PATH}")
  endif()

  function(nemotron_add_ttft_smoke_test name)
    add_test(
      NAME ${name}
      COMMAND
        ${CMAKE_COMMAND} -E env
          NEMOTRON_FORWARD_MANIFEST=${CMAKE_SOURCE_DIR}/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json
          $<TARGET_FILE:nano_prefix_cache_ttft_bench>
          ${ARGN}
    )
    if(NEMOTRON_TTFT_TEST_ENVIRONMENT)
      set_tests_properties(${name} PROPERTIES ENVIRONMENT "${NEMOTRON_TTFT_TEST_ENVIRONMENT}")
    endif()
    set_tests_properties(${name} PROPERTIES RUN_SERIAL TRUE)
  endfunction()

  nemotron_add_ttft_smoke_test(
    nano_prefix_cache_ttft_prefix128_tail4_smoke
    --warmup 0
    --iterations 5
    --tail-token-count 4
    --prefix-length 128
    --prefix-length 4096
    --case cold_prefill_prefix128
    --case cached_committed_head_prefix128_tail4
    --case cached_global_root_prefix128_tail4
  )

  nemotron_add_ttft_smoke_test(
    nano_prefix_cache_ttft_prefix256_tail128_committed_smoke
    --warmup 0
    --iterations 5
    --tail-token-count 128
    --prefix-length 256
    --prefix-length 4096
    --case cached_committed_head_prefix256_tail128
  )

  nemotron_add_ttft_smoke_test(
    nano_prefix_cache_ttft_prefix256_tail128_global_root_smoke
    --warmup 0
    --iterations 5
    --tail-token-count 128
    --prefix-length 256
    --prefix-length 4096
    --case cached_global_root_prefix256_tail128
  )
endif()
