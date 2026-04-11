# Step 8 Benchmark Summary

Artifacts
- Baseline shared-prefill: `nano_shared_prefill_20260411T152413Z_cuda130.json`
- FP4-direct shared-prefill: `nano_shared_prefill_20260411T152541Z_cuda130.json`
- Shared-prefill comparison: `shared_prefill_compare_20260411T152413Z_vs_20260411T152541Z.{json,md}`
- vLLM parity: `chat_runtime_parity_native_vllm_fp4_direct_20260411T1528.json`
- Prompt-length sweep: `prompt_length_sweep_native_fp4_direct_20260411T1529.json`

Notes
- Baseline run env: `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1`
- FP4-direct run env: `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 NEMOTRON_ENABLE_FP4_DIRECT_FC1=1`
- The direct flag is recorded in `nano_shared_prefill_20260411T152541Z_cuda130.env.txt`. The benchmark JSON's `options.native_env` only records script-provided overrides, not all inherited shell env.
- Short-prompt TTFT below uses `hot_first_token_mean_ms` at 4 prompt tokens.
- Long-prompt TPS below is derived as `prompt_tokens / hot_prefill_mean_ms` at 256 prompt tokens.

## Shared Prefill Comparison

Summary
- Hot prefill improved at 6/8 prompt lengths, but the best win was only `1.022675x` at 8 tokens.
- Cold prefill improved at only 1/8 prompt lengths.
- Short-prompt TTFT was effectively flat: `20.435434 ms -> 20.437996 ms` at 4 tokens.
- Long-prompt prefill TPS was effectively flat: `484.230 -> 484.396 tokens/s` at 256 tokens.
- Cold setup overhead mostly regressed: delta range `-0.107311 ms` to `+5.537399 ms`.

| Prompt | BF16 Hot ms | FP4 Hot ms | Hot Speedup | BF16 Cold ms | FP4 Cold ms | BF16 TTFT ms | FP4 TTFT ms | BF16 TPS | FP4 TPS | BF16 Setup ms | FP4 Setup ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 19.874273 | 19.902648 | 0.998574x | 128.791981 | 129.330505 | 20.435434 | 20.437996 | 201.265 | 200.978 | 108.917708 | 109.427857 |
| 8 | 38.815955 | 37.955320 | 1.022675x | 141.340296 | 143.759777 | 39.565953 | 38.726089 | 206.101 | 210.774 | 102.524341 | 105.804457 |
| 16 | 39.286714 | 39.188639 | 1.002503x | 142.582893 | 142.377507 | 40.491438 | 40.338927 | 407.262 | 408.282 | 103.296179 | 103.188868 |
| 24 | 59.521341 | 59.579750 | 0.999020x | 165.914385 | 166.816960 | 61.076680 | 61.162781 | 403.217 | 402.821 | 106.393044 | 107.237210 |
| 32 | 88.306652 | 88.128449 | 1.002022x | 189.000223 | 191.650733 | 90.275155 | 90.125914 | 362.374 | 363.106 | 100.693571 | 103.522284 |
| 64 | 133.404025 | 133.049161 | 1.002667x | 233.761853 | 235.068605 | 140.503999 | 139.839047 | 479.746 | 481.025 | 100.357828 | 102.019444 |
| 128 | 274.504876 | 274.218428 | 1.001045x | 375.651533 | 376.348361 | 288.497850 | 287.925474 | 466.294 | 466.781 | 101.146657 | 102.129933 |
| 256 | 528.674927 | 528.493657 | 1.000343x | 631.427913 | 636.784042 | 556.100824 | 556.992462 | 484.230 | 484.396 | 102.752986 | 108.290385 |

## Nsight Notes

Built-in shared-kernel metric
- The benchmark harness tracks `Nvfp4ContiguousSharedFp4P5Kernel`.
- That metric moved only within noise: `2.685310 -> 2.692928 ms` at 4 tokens and `32.383817 -> 32.338910 ms` at 256 tokens.

Routed grouped-kernel search in raw Nsight CSVs
- `RoutedBf16Relu2PackKernel` still appears in both the BF16 baseline and the FP4-direct run.
- `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<..., float>` also appears in both traces, so kernel-name matching alone does not isolate the direct path cleanly.
- Example at 256 tokens:
  - baseline BF16 grouped + pack: `6.377402 ms`
  - FP4-direct BF16 grouped + pack: `6.460926 ms`
  - baseline grouped float kernel: `4.785841 ms`
  - FP4-direct grouped float kernel: `4.883119 ms`
- Conclusion: the existing shared-prefill benchmark does not show a clean "fused P5 replaced old P5 + BF16Relu2Pack" transition. The routed traces still look mixed.

## vLLM Parity

Summary
- Normalized-text matches: `2 / 8`
- Visible-token matches: `2 / 8`
- Matching turns: `short_exact` turn 0, `multi_turn_codeword` turn 1

Notable mismatches
- `boundary_exact_reply`: native returned `1.1.1.1:443` instead of `4`
- `json_exact`: native returned `"Answer: Yes"` instead of `{"answer":4}`
- `multi_turn_codeword` turn 0: native returned free-form text instead of `OK`
- `multi_turn_recall_open` diverged on all 3 turns, including one empty visible reply on turn 1

## Prompt-Length Sweep

Summary
- All 16 prompt lengths completed with no runtime errors.
- Unique-token ratio range: `0.422727` to `1.0`
- One-token prompt produced empty visible text after normalization (`<|im_end|>`)

Lowest unique-token ratios
- 22 user tokens / 51 total prompt tokens: `93 / 220` unique tokens (`0.422727`)
- 2 user tokens / 31 total prompt tokens: `69 / 156` unique tokens (`0.442308`)
- 64 user tokens / 93 total prompt tokens: `114 / 219` unique tokens (`0.520548`)

## Concerns

- The FP4-direct flag produced no material end-to-end prefill win on this suite. Hot prefill changes were small and cold setup was generally worse.
- Nsight traces still show `RoutedBf16Relu2PackKernel`, so the direct path does not appear to have cleanly displaced the old BF16 pack path in this benchmark.
- vLLM parity remains weak (`2 / 8` matching turns), so there is still a behavioral blocker beyond the microbenchmark noise floor.
