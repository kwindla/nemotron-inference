add_subdirectory(gb10_dense_gemm)
add_subdirectory(gb10_loader_modes)
add_subdirectory(gb10_mamba_cache)
add_subdirectory(gb10_nvfp4_gemm)
add_subdirectory(gb10_nvfp4_contract)

if(NEMOTRON_RUNTIME_HAVE_CUDNN_FE)
  add_subdirectory(gb10_paged_attention)
endif()
