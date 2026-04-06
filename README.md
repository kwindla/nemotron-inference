# Nemotron Runtime

This repo is the active implementation for **Nemotron 3 Super on DGX Spark**.

## Current Status

- Direct `SingleTokenForwardModel::Create(...)` prompt parity is green against the pinned `vLLM` oracle.
- `SingleTokenForwardModel::CreateFromCache(...)` is not yet qualified.
- The interactive forward tool works on the direct path only.

## Start Here

```bash
cmake -S . -B build
cmake --build build -j2
export NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_unverified.json
ctest --test-dir build --output-on-failure -R '^full_forward_manifest_smoke_test$'
```

Then use:

- [docs/RUNBOOK.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/RUNBOOK.md) for correctness tests, profiling, and the interactive forward tool
- [tools/interactive_forward/INTERACTIVE_FORWARD_DESIGN.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/interactive_forward/INTERACTIVE_FORWARD_DESIGN.md) for helper design details
- start with the `Repo Health Check` section in [docs/RUNBOOK.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/RUNBOOK.md)

## Quick Commands

Correctness:

```bash
ctest --test-dir build --output-on-failure -R '^full_forward_manifest_smoke_test$'
ctest --test-dir build --output-on-failure -R '^prompt_matched_parity_test$'
ctest --test-dir build --output-on-failure -R '^cache_backed_dense_regression_test$'
ctest --test-dir build --output-on-failure -R '^single_token_forward_model_test$'
ctest --test-dir build --output-on-failure -R '^attention_layer_oracle_test$'
ctest --test-dir build --output-on-failure -R '^attention_layer_decode_oracle_test$'
```

Current red diagnostics are still available, but they are not acceptance gates:

```bash
ctest --test-dir build --output-on-failure -R '^prefill_prefix_oracle_test$'
ctest --test-dir build --output-on-failure -R '^single_token_decode_oracle_test$'
```

Profiling:

```bash
./build/benchmarks/decode_bench/single_token_decode_bench \
  --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json \
  --iterations 1

./benchmarks/decode_bench/run_bench.sh
```

Interactive forward:

```bash
cmake --build build --target nemotron_interactive_forward_server -j2
./tools/interactive_forward/nemotron_interactive_forward.py
./tools/interactive_forward/nemotron_interactive_forward.py --once "Hello"
```

## Oracle

The active parity oracle is the pinned image in [tools/oracle/generate_vllm_trace.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_vllm_trace.sh):

- `nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065`

This repo is not automatically tracking upstream `vLLM` `main`.

## Repo Layout

- `runtime/`: serving runtime implementation
- `kernels/`: CUDA kernels
- `testing/`: tests and oracle fixtures
- `benchmarks/`: profiling and performance harnesses
- `tools/`: offline utilities and interactive forward helper
- `docs/`: current reference docs

Historical project notes under the workspace root `proj-*` directories are not the primary instructions for this repo.
