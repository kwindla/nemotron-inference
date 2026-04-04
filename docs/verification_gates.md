# Verification Gates

This branch uses four canonical verification gates. They answer different
questions and must stay separate so one passing signal does not hide another
failing signal.

## Gate 1: Local Correctness / Oracle Replay

Purpose: prove that the local runtime still reproduces saved oracle behavior and
local replay expectations without depending on vLLM.

Inputs:

- a manifest and matching local build
- runtime oracle JSON from `testing/nano_save_prompt_oracle`
- local replay tests such as `nano_16_token_correctness_test` and targeted cache /
  snapshot regressions

Pass criteria:

- required oracle fields replay cleanly
- `boundary_token_id`, `boundary_top5`, and `generated_token_ids` match the local
  replay result
- if the optional deeper regression payload is present, its dense prompt-boundary
  logits are treated as additional localization evidence, not a replacement for
  the compact oracle fields

Artifact format:

- compact oracle JSON with additive provenance fields:
  `model_id`, `manifest_path`, `build_dir`, `backend_flags`, `git_revision`,
  `prompt_token_ids`, `boundary_top5`, `boundary_token_id`,
  `generated_token_ids`, `route`, `timestamp_utc`
- optional `deep_regression_payload` object for dense prompt-boundary logits

## Gate 2: Exact-Token vLLM Parity

Purpose: prove that the runtime and local vLLM generate the exact same token
sequence from the exact same prompt token IDs.

Inputs:

- a runtime oracle JSON or an on-demand oracle generation request
- the exact `prompt_token_ids` used by the runtime
- the working local vLLM source tree and Python environment

Pass criteria:

- exact generated token agreement between runtime and vLLM
- on mismatch, the gate fails and records the first mismatch index instead of
  collapsing the result into a generic failure

Artifact format:

- parity JSON with `prompt_source`, `prompt_token_ids`, `decode_token_count`,
  `mismatch_index`, runtime token stream, vLLM token stream, runtime backend
  flags, and vLLM backend-selection metadata

## Gate 3: Internal Performance

Purpose: measure the runtime on its own terms, with explicit backend selection
and explicit MoE window settings.

Inputs:

- built runtime benchmark binaries
- manifest and model id
- runtime backend-selection env vars
- decode benchmark runs plus resumed-prefix TTFT runs for 32, 64, and 128-token
  tails

Pass criteria:

- every required case produces a machine-readable artifact with complete
  provenance
- regressions are judged only against internal performance baselines or
  thresholds; this gate does not decide correctness

Artifact format:

- JSON artifacts for decode and prefix-cache TTFT runs
- each artifact records the exact binary path, build dir, manifest path, model
  id, backend-selection env vars, and the vLLM source-tree / revision used for
  external comparison context

## Gate 4: External TTFT

Purpose: measure the local vLLM baseline on the known-working RTX 5090 setup.

Inputs:

- `run_bench_vllm.sh` / `bench_vllm_nano.py`
- local `third_party/vllm` checkout and Python interpreter
- vLLM backend env vars and requested TTFT token counts

Pass criteria:

- the decode and prefill TTFT cases complete and emit structured results
- regressions are judged only against external TTFT baselines; this gate does
  not override correctness gates

Artifact format:

- vLLM TTFT JSON with case summaries plus explicit provenance for Python path,
  source tree, git revision, requested backend, and backend-affecting env vars

## Reporting Rule

Correctness failures and performance regressions are reported separately.

- A correctness failure in gate 1 or gate 2 is a failed correctness result even
  if gate 3 or gate 4 is faster.
- A performance regression in gate 3 or gate 4 is a performance issue even if
  gate 1 and gate 2 still pass.
- Summary artifacts may list all four gates together, but they must not merge
  correctness and performance into one pass/fail bit.
