# Baseline Reference

Captured on `2026-04-11` before implementing the warp-local P5 direct-FP4 epilogue.

## Forced-P5 compare reference

- Command intent: force routed GEMM1 onto `P5` and capture the current direct-FP4 boundary compare.
- Artifact: `forced_p5_compare_relwithdebinfo_forward_debug.log`
- Status: `exit_code=1` in `forced_p5_compare_relwithdebinfo_forward_debug.status.txt`
- Why the nonzero exit is still useful: this forced-dispatch diagnostic changes normal selection semantics, so it is a code-path probe, not a correctness oracle for the full test.
- Key evidence from the log:
  - P5 direct path is active: `routed_gemm1 dispatch_rows=8 active_selection_count=138 profile=p5_128x128x64_swap_true mode=fp4_direct`
  - Current staged-direct compare summary:
    - `packed_data=4150`
    - `block_scales_data=8305`
    - `matmul_block_scales_data=8284`
    - `activation_output_scale=9154`

## Long-prompt benchmark reference

- Artifacts:
  - `manual_bench/prompt_0384.json`
  - `manual_bench/prompt_0512.json`
- These were captured by running `nano_fused_decode_bench --mode=phased` directly with:
  - `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1`
  - `NEMOTRON_ENABLE_FP4_DIRECT_FC1=1`

### Prompt 384

- `cold_prefill_ms = 885.247876`
- `hot_prefill_mean_ms = 783.608886`
- `cold_first_token_ms = 929.095388`
- `hot_first_token_mean_ms = 825.789506`
- `full_decode_tokens_per_second = 1.210919`

### Prompt 512

- `cold_prefill_ms = 1164.751821`
- `hot_prefill_mean_ms = 1059.598123`
- `cold_first_token_ms = 1221.956369`
- `hot_first_token_mean_ms = 1115.096786`
- `full_decode_tokens_per_second = 0.896763`

## Important limitation

- The long-prompt benchmark is not a direct measurement of the P5 epilogue we plan to replace.
- `staged_direct_bench_0384_forward_debug.log` shows that for prompt `384` the routed MoE window is still dispatched as `23` rows:
  - `routed_gemm1 dispatch_rows=23 active_selection_count=138 profile=legacy`
- So `384/512` should be treated as end-to-end TTFT/TPS guardrails, not as direct P5 direct-FP4 performance evidence.
