# Unified Fused Backend Validation Results

RTX 5090 (32GB), Nemotron-3 Nano 30B A3B NVFP4

## Correctness

| Test | Result |
|------|--------|
| Oracle comparison (default vs unified) | **PASS** — bit-identical output |
| Boundary token match | **PASS** — same token ID (1047) |
| Top-5 logit max absolute diff | **0.0** (exact match) |
| Generated token sequence (16 tokens) | **PASS** — identical |
| nano_16_token_correctness_test | **PASS** |

## Decode Performance (token_count == 1)

| Backend | Mean (ms) | Tokens/sec | vs Scalar |
|---------|----------|------------|-----------|
| Scalar fused decode | 1825.5 | 0.55 | baseline |
| **Unified fused** | **67.5** | **14.8** | **-96.3%** |
| Decode cuBLASLt | 67.2 | 14.9 | -96.3% |

The scalar decode kernel is **27x slower** than both the unified fused and cuBLASLt backends. The unified fused backend matches cuBLASLt to within ~0.3ms (0.5%).

**Decision**: The scalar decode kernel (`FusedDecodeBackend`) should be removed. The unified fused backend achieves identical latency to the cuBLASLt backend while providing a single code path for both decode and prefill.

## Prefill Performance

Not yet measured. The existing `nano_prefix_cache_ttft_bench` uses 32-token tail prefill. The unified fused backend was verified correct for 16-token prefill (via oracle), and the infrastructure for 64/128-token chunked prefill is in place via `moe_prefill_window_tokens`.

## vLLM External Baseline

**Blocked**: Local vLLM v0.19.0 checkout is not compiled (missing `vllm._C`). System vLLM is 0.17.1 (too old). FlashInfer has version mismatch (0.6.3 cubin vs 0.6.4 package). Requires either:
- Compile `third_party/vllm` from source
- Or install vLLM >= 0.19.0 with matching FlashInfer

## SM120 Custom Backend Decision

**NO-GO at this time** — not because the unified backend is sufficient (it may be), but because we cannot make the decision without the vLLM external baseline comparison.

The prerequisites from `SM120_BACKEND_DECISION.md`:
- [x] Unified fused backend is correct
- [x] Load-time weight preparation in place
- [x] Chunking is runner-level
- [ ] **Benchmark comparison vs vLLM completed** — BLOCKED
- [ ] Measured performance gap exists — UNKNOWN

**Action items**:
1. Get vLLM v0.19.0 running with flashinfer_cutlass on this RTX 5090
2. Run `bench_vllm_nano.py` for decode and prefill at 32/64/128 tokens
3. Compare against our unified fused backend numbers
4. Then apply the GO/NO-GO framework

## Bug Found and Fixed

During step 1 sanity check, a segfault was discovered in `RunFusedMoePrefill()`: `params.routed_up` and `params.routed_down` are device arrays but were dereferenced on the host. Fixed by copying the view arrays to host before the per-expert loop (commit `1dac5b3`).
