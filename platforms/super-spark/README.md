# Nemotron 3 Super on DGX Spark

From-scratch C++/CUDA inference for **Nemotron 3 Super NVFP4** on **DGX Spark** (GB10, SM 121).

## Current Status

See [docs/RUNBOOK.md](docs/RUNBOOK.md#current-status) for the current status and latest validation results.

## Build

```bash
# From repo root
cmake -S . -B build -DNEMOTRON_PLATFORM=super-spark
cmake --build build -j2
```

## Quick Commands

Set the manifest:

```bash
export NEMOTRON_FORWARD_MANIFEST=$PWD/platforms/super-spark/artifacts/manifests/forward_runtime_manifest_unverified.json
```

Correctness:

```bash
ctest --test-dir build --output-on-failure -R '^full_forward_manifest_smoke_test$'
ctest --test-dir build --output-on-failure -R '^prompt_matched_parity_test$'
ctest --test-dir build --output-on-failure -R '^cache_backed_dense_regression_test$'
ctest --test-dir build --output-on-failure -R '^single_token_forward_model_test$'
ctest --test-dir build --output-on-failure -R '^attention_layer_oracle_test$'
ctest --test-dir build --output-on-failure -R '^attention_layer_decode_oracle_test$'
```

Profiling:

```bash
./build/platforms/super-spark/benchmarks/decode_bench/single_token_decode_bench \
  --manifest ./platforms/super-spark/artifacts/manifests/forward_runtime_manifest_unverified.json \
  --iterations 1
```

Interactive forward:

```bash
cmake --build build --target nemotron_interactive_forward_server -j2
./platforms/super-spark/tools/interactive_forward/nemotron_interactive_forward.py
./platforms/super-spark/tools/interactive_forward/nemotron_interactive_forward.py --once "Hello"
```

## Oracle

The active parity oracle is the pinned image in [tools/oracle/generate_vllm_trace.sh](tools/oracle/generate_vllm_trace.sh):

- `nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065`

## Layout

- `runtime/` — serving runtime implementation
- `testing/` — tests and oracle fixtures
- `benchmarks/` — profiling and performance harnesses
- `tools/` — oracle generators, interactive forward, manifest tooling
- `docs/` — platform-specific reference docs
- `scripts/` — helper scripts
- `vllm-reference/` — local vLLM image build files
