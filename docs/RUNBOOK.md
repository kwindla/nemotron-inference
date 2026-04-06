# Nemotron 3 Super DGX Spark Runbook

This is the canonical runbook for working in [nemotron-runtime](/home/khkramer/src/nemotron-march-2026/nemotron-runtime).

Scope:

- target model: Nemotron 3 Super
- target platform: DGX Spark
- target correctness oracle: pinned `vLLM` image in [generate_vllm_trace.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_vllm_trace.sh)

This runbook covers:

- building the repo
- correctness tests
- performance profiling
- the interactive forward tool

Historical `proj-*` notes are not the canonical instructions.

## Current Status

- Direct `SingleTokenForwardModel::Create(...)` prompt parity is green against the pinned `vLLM` oracle.
- `SingleTokenForwardModel::CreateFromCache(...)` is not yet qualified.
- The interactive forward tool uses the direct path only.
- Latest runbook validation on April 6, 2026:
  - `full_forward_manifest_smoke_test`: passed
  - `prompt_matched_parity_test`: passed
  - sequential full suite: `59` passed, `16` failed

## Common Setup

Work from [nemotron-runtime](/home/khkramer/src/nemotron-march-2026/nemotron-runtime).

```bash
cd /home/khkramer/src/nemotron-march-2026/nemotron-runtime
cmake -S . -B build
cmake --build build -j2
export NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_unverified.json
```

Notes:

- `-j2` is the safe default on DGX Spark.
- Run one heavy manifest-backed test or benchmark at a time.
- Treat `MemAvailable + SwapFree` from `/proc/meminfo` as the real host-memory signal on this machine.

Quick checks:

```bash
ctest --test-dir build -N
awk '/MemAvailable:|SwapFree:/{print}' /proc/meminfo
```

## Repo Health Check

If you want to answer "is this repo healthy on this machine right now?", run these in order.

Healthy means:

- the repo builds
- the green acceptance gates pass
- the green targeted regressions pass
- the profiling wrapper completes and writes a JSON artifact

Healthy does not currently mean "all 75 tests are green". Some oracle and subsystem diagnostics are intentionally still red and are listed later in this runbook.

### 1. Build And Manifest Setup

```bash
cd /home/khkramer/src/nemotron-march-2026/nemotron-runtime
cmake -S . -B build
cmake --build build -j2
export NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_unverified.json
ctest --test-dir build -N
```

Expected result:

- build completes
- `ctest -N` lists the registered suite

### 2. Green Acceptance Gates

Run both of these:

```bash
ctest --test-dir build --output-on-failure -R '^full_forward_manifest_smoke_test$'
ctest --test-dir build --output-on-failure -R '^prompt_matched_parity_test$'
```

Expected result:

- both commands pass

If either fails, the repo is not healthy.

### 3. Green Targeted Regressions

Run these next:

```bash
ctest --test-dir build --output-on-failure -R '^cache_backed_dense_regression_test$'
ctest --test-dir build --output-on-failure -R '^single_token_forward_model_test$'
ctest --test-dir build --output-on-failure -R '^attention_layer_oracle_test$'
ctest --test-dir build --output-on-failure -R '^attention_layer_decode_oracle_test$'
```

Expected result:

- all four commands pass

If one of these fails, the repo is not healthy enough for normal forward-path work.

### 4. Profiling Sanity Check

Run the wrapper:

```bash
./benchmarks/decode_bench/run_bench.sh
```

Expected result:

- the wrapper exits successfully
- a fresh JSON artifact appears under `artifacts/benchmarks/`

Latest validated example:

- [single_token_decode_20260406T152730Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260406T152730Z_cuda132.json)

Read the JSON artifact for the real result. Wrapper stdout is still noisy because backend path-identity lines are printed during model build.

### 5. Optional: Direct Interactive Tool Check

If you are about to work on the interactive tool, also run:

```bash
cmake --build build --target nemotron_interactive_forward_server -j2
./tools/interactive_forward/nemotron_interactive_forward.py --help
```

Expected result:

- server target builds
- helper prints usage and exits

### 6. Optional: Current Red Diagnostics

These tests are useful when debugging their specific subsystems, but they are not part of the healthy-repo gate today:

```bash
ctest --test-dir build --output-on-failure -R '^prefill_prefix_oracle_test$'
ctest --test-dir build --output-on-failure -R '^single_token_decode_oracle_test$'
```

Treat failures here as known current issues, not as immediate proof that the whole repo is broken.

## Correctness

Document these test tiers going forward:

- Green acceptance gates:
  - `full_forward_manifest_smoke_test`
  - `prompt_matched_parity_test`
- Green targeted regression tests:
  - `cache_backed_dense_regression_test`
  - `single_token_forward_model_test`
  - `attention_layer_oracle_test`
  - `attention_layer_decode_oracle_test`
- Current red diagnostic tests:
  - `prefill_prefix_oracle_test`
  - `single_token_decode_oracle_test`
  - `expert_layer_oracle_test`
  - `expert_layer3_oracle_test`
  - `expert_layer3_prefix_input_oracle_test`
  - `expert_layer19_decode_input_oracle_test`
  - `expert_layer19_runtime_input_oracle_test`
  - `expert_layer8_runtime_input_oracle_test`
  - `expert_layer14_runtime_input_oracle_test`
  - `mamba_layer_oracle_test`
  - `mamba_layer18_runtime_input_oracle_test`
  - `mamba_layer13_runtime_input_oracle_test`
  - `mamba_layer9_runtime_input_oracle_test`
  - `mamba_layer15_runtime_input_oracle_test`
  - `expert_layer3_prefix_replay_test`
  - `expert_layer3_manifest_binding_compare_test`
- Full correctness suite:
  - all tests registered by `ctest --test-dir build -N`
  - current suite size: `75`

Use the green acceptance gates for fast confidence, the green targeted regression tests when working in one subsystem, the current red diagnostics only when you are actively debugging those surfaces, and the full suite before landing risky changes.

Start with a cheap smoke test:

```bash
ctest --test-dir build --output-on-failure -R '^full_forward_manifest_smoke_test$'
```

Then run the main manifest-backed correctness gates one at a time:

```bash
ctest --test-dir build --output-on-failure -R '^prompt_matched_parity_test$'
```

Interpretation:

- [full_forward_manifest_smoke_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/api/full_forward_manifest_smoke_test.cpp): basic manifest-backed bring-up
- [prompt_matched_parity_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/api/prompt_matched_parity_test.cpp): end-to-end acceptance gate

Run the current red manifest-backed diagnostics only when you are working on those paths:

```bash
ctest --test-dir build --output-on-failure -R '^prefill_prefix_oracle_test$'
ctest --test-dir build --output-on-failure -R '^single_token_decode_oracle_test$'
```

Those two tests are useful, but they are not green today and should not be treated as the primary acceptance bar.

Safe sequential suite run:

```bash
set -euo pipefail
TESTS=$(ctest --test-dir build --show-only=json-v1 | jq -r '.tests[].name')
PASS=0
FAIL=0
FAIL_NAMES=""

for test_name in $TESTS; do
  mem_avail_kb=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
  swap_free_kb=$(awk '/SwapFree:/{print $2}' /proc/meminfo)
  total_free_kb=$((mem_avail_kb + swap_free_kb))

  if [ "$total_free_kb" -lt $((30 * 1024 * 1024)) ]; then
    echo "ABORT low host budget before $test_name: MemAvailable=${mem_avail_kb}kB SwapFree=${swap_free_kb}kB"
    exit 125
  fi

  echo "RUN $test_name (MemAvailable=${mem_avail_kb}kB SwapFree=${swap_free_kb}kB)"
  if ctest --test-dir build --output-on-failure -R "^${test_name}$"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    FAIL_NAMES="$FAIL_NAMES $test_name"
  fi
  sleep 1
done

echo "PASSED: $PASS  FAILED: $FAIL"
if [ -n "$FAIL_NAMES" ]; then
  echo "Failed tests:$FAIL_NAMES"
fi
```

Important:

- keep `set -euo pipefail` — it still protects the memory check and `jq` pipeline
- the `if ctest ...` wrapper lets known-red tests fail without aborting the loop
- the summary at the end gives the real pass/fail count

Pinned oracle note:

- active image: `nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065`
- this repo is not automatically using upstream `vllm` `main`
- if you change the oracle, regenerate the trace artifacts before using new comparisons

## Profiling And Performance

Use the decode bench for system-level runtime timing:

```bash
./build/benchmarks/decode_bench/single_token_decode_bench \
  --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json \
  --iterations 1
```

Use the wrapper when you want captured output in `artifacts/benchmarks`:

```bash
./benchmarks/decode_bench/run_bench.sh
./benchmarks/decode_bench/run_bench.sh ./artifacts/manifests/forward_runtime_manifest_unverified.json
```

Latest validated profiling artifact:

- [single_token_decode_20260406T152730Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260406T152730Z_cuda132.json)
- `environment_build_ms ~= 1189.5`
- `model_build_ms ~= 199686.1`
- `hot_mean_ms ~= 87.1`
- `predicted_token_id = 1044`

Important:

- `single_token_decode_bench` is a profiling surface, not the prompt-parity acceptance test
- use [prompt_matched_parity_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/api/prompt_matched_parity_test.cpp) for end-to-end correctness
- read the emitted JSON artifact for the real profiling result; wrapper stdout is still noisy because backend path-identity lines are printed during model build

Useful startup and loader probes:

```bash
NEMOTRON_FORWARD_BUILD_DEBUG=1 \
  ./build/benchmarks/decode_bench/single_token_decode_bench \
  --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json \
  --warmup 0 --iterations 0

NEMOTRON_FORWARD_BUILD_THREADS=1 \
  ./build/benchmarks/decode_bench/single_token_decode_bench \
  --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json \
  --warmup 0 --iterations 0
```

Useful operator-level benches:

```bash
./build/benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench --list-cases
./build/benchmarks/gb10_nvfp4_gemm/gb10_nvfp4_gemm_bench --list-cases
./build/benchmarks/gb10_paged_attention/gb10_paged_attention_bench --list-cases
./build/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench --help
```

Use those when you already know which subsystem is slow. Start with the decode bench first.

## Interactive Forward Tool

The interactive helper is:

- [nemotron_interactive_forward.py](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/interactive_forward/nemotron_interactive_forward.py)
- [nemotron_interactive_forward_server.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/interactive_forward/nemotron_interactive_forward_server.cpp)

Build the server:

```bash
cmake --build build --target nemotron_interactive_forward_server -j2
```

Run it:

```bash
./tools/interactive_forward/nemotron_interactive_forward.py
./tools/interactive_forward/nemotron_interactive_forward.py --once "Hello"
```

Current contract:

- direct `Create(...)` path only
- full prompt rerender and prefill each turn
- greedy decode
- visible thinking preserved
- no `CreateFromCache(...)`
- no prefix-cache restore/commit
- forward CUDA graph capture force-disabled inside the helper

Useful flags:

```bash
./tools/interactive_forward/nemotron_interactive_forward.py --show-step-stats
./tools/interactive_forward/nemotron_interactive_forward.py --show-token-ids
./tools/interactive_forward/nemotron_interactive_forward.py --target-context-tokens 8192 --max-new-tokens 2048
```

Logs:

- helper server stderr: [artifacts/interactive_forward/server-stderr.log](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/interactive_forward/server-stderr.log)
- design note: [INTERACTIVE_FORWARD_DESIGN.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/interactive_forward/INTERACTIVE_FORWARD_DESIGN.md)

## Useful Environment Variables

- `NEMOTRON_FORWARD_MANIFEST`: manifest path used by manifest-backed tests and tools
- `NEMOTRON_FORWARD_BUILD_DEBUG=1`: print environment/model build timing
- `NEMOTRON_FORWARD_BUILD_THREADS=<n>`: override layer-build concurrency
- `NEMOTRON_FORWARD_DEBUG=1`: emit stage and layer progress on correctness-first forward paths
- `NEMOTRON_PREFIX_CACHE=0`: force uncached execution

Avoid adding new "default" instructions outside this runbook and the two READMEs. Keep onboarding here.
