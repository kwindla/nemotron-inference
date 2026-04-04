# Tier 2 Verification — Phase 1 (post step 6)

Date: 2026-04-04
Commit: f824cbf
Branch: khk/wip-nano

## Gate 1: Oracle Generation — PASS
- Fresh oracle generated successfully with new provenance fields
- manifest_path, build_dir, git_revision, backend_flags all populated
- boundary_token_id=1047, 16 decode tokens

## Gate 2: Local Correctness — PASS
- Default backend vs unified fused: exact match (max_abs_diff=0.0)
- nano_16_token_correctness_test: PASS (16/16 tokens match oracle)
- host_routing_adapter_calls=0 (unified fused is the default)
- total_bytes_uploaded=0 (no on-demand weight uploads)

## Gate 3: Decode Performance — PASS (no regression)
| Backend | decode_mean_ms | tokens/sec | delta vs default |
|---------|---------------|------------|------------------|
| default (unified fused) | 75.58 | 13.23 | baseline |
| unified_fused (explicit) | 75.31 | 13.28 | -0.35% |
| decode_cublaslt | 69.07 | 14.48 | -8.62% |

DecodeCublasLt remains ~8.6% faster at single-token decode.
No performance regression from steps 1-6.

## Gate 4: vLLM Exact-Token Parity — FAIL
- first_mismatch_index=0
- Runtime: token 1047 ("/imagine")
- vLLM: token 1010 ("\n")
- This is a PRE-EXISTING divergence, not a regression from this project.
  Older oracles (pre-step-1) also produce boundary_token_id=1047.
- The divergence is at the first generated token, suggesting a difference
  in how the runtime and vLLM handle the prompt boundary logits — possibly
  related to RoPE position handling, attention mask shape, or quantization
  path differences in the first-token computation.
- Investigation needed but should not block this alignment project.

## Counters (from correctness run)
- dense_reference_fallback=960 (all dense GEMMs use reference path)
- nvfp4_reference_fallback=544 (all NVFP4 use reference path)
- nvfp4_fastpath_plan_fail=544 (NVFP4 fastpath not yet available)
- scaled_fp8_reference_fallback=0 (no FP8 execution)
- host_routing_adapter_calls=0 (unified fused working)
