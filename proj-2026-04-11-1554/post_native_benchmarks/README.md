## Post-native validation

Captured on `2026-04-11` after committing the native P5 direct-FP4 activation/pack path.

Historical note:
- This README reflects the state before removing Nano's hard-coded `23`-token routed-MoE prefill clamp.
- The current default-runtime measurements live under `../post_default_direct_benchmarks/README.md`.

Metric terminology note:
- `hot` in the phased fused-decode benchmark means a warmed uncached run in the same process, not explicit prefix-cache reuse.
- Actual cached-prefix measurements are the `hot-prefix TTFT` cases in the prefix-cache benchmark.

### Sequential test sweep

- Command: `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -j1`
- Result: `69/72` passed
- Same pre-existing failures:
  - `nvfp4_weight_test`
  - `expert_layer_oracle_test`
  - `expert_layer8_oracle_test`

### Shared-prefill throughput guardrail

- Artifact: `shared_prefill/nano_shared_prefill_native_direct.json`
- Command shape: `benchmarks/nano_shared_prefill/run_default_bench.sh --native-env NEMOTRON_ENABLE_FP4_DIRECT_FC1=1`
  - Historical only: the direct-FC1 path was still opt-in at this point.
- `384`: `hot_prefill_mean_ms = 785.876749`, `hot_first_token_mean_ms = 828.139879`
- `512`: `hot_prefill_mean_ms = 1062.848968`, `hot_first_token_mean_ms = 1119.214678`
- Versus the saved pre-native guardrail, both are slightly slower:
  - `384`: hot prefill `+0.29%`, hot first token `+0.28%`
  - `512`: hot prefill `+0.31%`, hot first token `+0.37%`

Important limitation:
- This benchmark is still an end-to-end guardrail, not a direct measurement of the new P5 direct path.
- A direct diagnostic run at `384` tokens still reports:
  - `routed_gemm1 dispatch_rows=23 active_selection_count=138 profile=legacy`
- The current Nano runtime default still clamps `moe_prefill_window_tokens` to `23`, so natural long-prompt runs do not reach the new P5 direct FC1 path.

### TTFT

- Default-window artifact: `ttft/nano_prefix_cache_ttft_native_direct.stdout.txt`
- Forced-wide-window artifact: `ttft/nano_prefix_cache_ttft_native_direct_window4096.stdout.txt`

The default TTFT run is also not a direct-path benchmark:
- It resolves `moe_prefill_window_tokens` to `23`
- It therefore reflects the current production runtime regime, not the new wide-window direct path

The `--moe-prefill-window-tokens 4096` run is the relevant direct-path signal:
- `cold_prefill_prefix256`: `166.604 ms`
- `cold_prefill_prefix1024`: `494.662 ms`
- `cold_prefill_prefix4096`: `2186.216 ms`
- `cached_committed_head_prefix256_tail32`: `hot-prefix TTFT = 97.598 ms`
- `cached_committed_head_prefix1024_tail32`: `hot-prefix TTFT = 105.788 ms`
- `cached_committed_head_prefix4096_tail32`: `hot-prefix TTFT = 138.974 ms`
- `cached_global_root_prefix256_tail32`: `hot-prefix TTFT = 97.467 ms`
- `cached_global_root_prefix1024_tail32`: `hot-prefix TTFT = 105.952 ms`
- `cached_global_root_prefix4096_tail32`: `hot-prefix TTFT = 139.063 ms`

Comparing current default-window TTFT to current `4096`-window TTFT isolates the runtime-window effect:
- `cached_committed_head_prefix256_tail32`: `122.426 -> 97.598 ms`
- `cached_committed_head_prefix1024_tail32`: `136.342 -> 105.788 ms`
- `cached_committed_head_prefix4096_tail32`: `183.702 -> 138.974 ms`
- `cached_global_root_prefix256_tail32`: `122.701 -> 97.467 ms`
- `cached_global_root_prefix1024_tail32`: `136.458 -> 105.952 ms`
- `cached_global_root_prefix4096_tail32`: `183.710 -> 139.063 ms`

First-token decode latency is effectively unchanged across those runs; the improvement comes from prefill.

### Implication

- The native direct path itself looks healthy.
- The current runtime default is the bottleneck to seeing that benefit in natural TTFT/TPS benchmarks.
- The next code change should remove the P5-specific BF16-preserving gate and then revisit the `23`-token Nano default so natural runs can use the direct path.
