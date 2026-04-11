# Post-default-direct validation

Captured on `2026-04-11` after:
- removing the `NEMOTRON_ENABLE_FP4_DIRECT_FC1` production gate for the supported packed-input/grouped-output `P5` routed-MoE FC1 path
- removing Nano's hard-coded `moe_prefill_window_tokens = 23` clamp and falling back to workspace-backed routing capacity

On this RTX 5090 setup, the default routed-MoE prefill window now resolves to `4096`.

## Sequential test sweep

- Command: `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -j1`
- Result: `69/72` passed
- Same pre-existing failures:
  - `nvfp4_weight_test`
  - `expert_layer_oracle_test`
  - `expert_layer8_oracle_test`

## Routed-profile proof

- Artifact: `diagnostics/prompt_0384_routed_profile.stderr.txt`
- Command shape: `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 NEMOTRON_ROUTED_PROFILE_DEBUG=1 nano_fused_decode_bench --mode=phased --prompt-tokens 384 --decode-tokens 1`
- Key evidence:
  - `routed_gemm1 dispatch_rows=384 active_selection_count=2304 profile=p5_128x128x64_swap_true mode=fp4_direct`
  - `routed_gemm2 dispatch_rows=384 active_selection_count=2304 profile=p13_128x128x64_swap_true`

This is the main change from the previous default runtime regime, which chunked routed-MoE prefill into `23`-token legacy windows.

## Shared-prefill throughput

- Artifact: `shared_prefill/nano_shared_prefill_default_direct.json`
- Raw runs: `shared_prefill/raw/`
- Result at `384` tokens:
  - `hot_prefill_mean_ms = 188.403850`
  - `hot_first_token_mean_ms = 229.611812`
- Result at `512` tokens:
  - `hot_prefill_mean_ms = 227.061275`
  - `hot_first_token_mean_ms = 282.020398`

Versus the saved pre-change default guardrail:
- `384`: hot prefill `783.608886 -> 188.403850 ms` (`-75.96%`), hot first token `825.789506 -> 229.611812 ms` (`-72.19%`)
- `512`: hot prefill `1059.598123 -> 227.061275 ms` (`-78.57%`), hot first token `1115.096786 -> 282.020398 ms` (`-74.71%`)

Unlike the earlier guardrail run, these numbers now reflect the actual native direct `P5` routed-MoE path.

## TTFT

- Artifact: `ttft/nano_prefix_cache_ttft_default_direct.stdout.txt`
- Stderr: `ttft/nano_prefix_cache_ttft_default_direct.stderr.txt`
- The benchmark header reports:
  - `moe_prefill_window_tokens_cli=0`
  - `resolved_runtime_moe_prefill_capacity_tokens=4096`
  - `resolved_runtime_moe_prefill_window_tokens=4096`

### Versus the old default-window runtime

- `cold_prefill_prefix256`: `666.521 -> 167.024 ms` (`-74.94%`)
- `cold_prefill_prefix1024`: `2684.803 -> 494.160 ms` (`-81.59%`)
- `cold_prefill_prefix4096`: `12179.834 -> 2186.433 ms` (`-82.05%`)
- `cached_committed_head_prefix256_tail32`: hot-prefix TTFT `122.426 -> 97.807 ms` (`-20.11%`), tail prefill `107.301 -> 82.902 ms` (`-22.74%`), first-token decode `14.761 -> 14.626 ms` (`-0.91%`)
- `cached_committed_head_prefix1024_tail32`: hot-prefix TTFT `136.342 -> 106.269 ms` (`-22.06%`), tail prefill `117.161 -> 87.273 ms` (`-25.51%`), first-token decode `18.752 -> 18.589 ms` (`-0.87%`)
- `cached_committed_head_prefix4096_tail32`: hot-prefix TTFT `183.702 -> 139.445 ms` (`-24.09%`), tail prefill `148.351 -> 104.267 ms` (`-29.72%`), first-token decode `34.732 -> 34.568 ms` (`-0.47%`)

The large win is in routed-MoE prefill. First-token decode stays effectively flat.

### Versus the earlier forced-`4096` diagnostic

These numbers are now effectively the same as the earlier manual wide-window run:
- `cold_prefill_prefix256`: `166.604 -> 167.024 ms` (`+0.25%`)
- `cold_prefill_prefix1024`: `494.662 -> 494.160 ms` (`-0.10%`)
- `cold_prefill_prefix4096`: `2186.216 -> 2186.433 ms` (`+0.01%`)
- `cached_committed_head_prefix4096_tail32`: hot-prefix TTFT `138.974 -> 139.445 ms` (`+0.34%`)

That confirms the default runtime is now using the intended wide-window direct path rather than the old `23`-token legacy fallback.
