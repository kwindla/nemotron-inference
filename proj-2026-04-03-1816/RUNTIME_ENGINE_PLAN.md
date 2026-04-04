# Plan: Runtime Engine and Control-Plane Alignment

Project directory: `./proj-2026-04-03-1816`

## Objective

Close accidental divergences between this Nano-on-RTX-5090 branch and the vLLM reference architecture in the runtime engine and control plane.

The priority is to make the branch feel like a complete, polished engine rather than a monolithic model runner with embedded orchestration. Preserve intentional divergences only where they are required for exact prefix reuse or are clearly justified by branch-specific constraints.

## Current State

- Bootstrap is manifest-backed and real, but it is still monolithic. `RuntimeEnvironment::BuildFromManifestFile()` assembles the loader plan, catalogs, weight arena, model schedule, reusable state arena, prefix cache, and heuristic cache in one path in [runtime_environment.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/runtime_environment.cpp:80).
- The runtime environment is a container plus cache/service bootstrap, not an engine boundary in the vLLM sense. There is no explicit split between input processing, request scheduling, worker execution, and output processing.
- `SingleTokenForwardModel` owns request lifecycle, layer execution orchestration, prefix-cache lookup and publication, and MoE chunking policy in [single_token_forward_model.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/single_token_forward_model.cpp:988).
- `RequestExecutionContext` owns compute buffers, attention KV pages, and Mamba state, and it also performs request reset and page release in [request_context.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/request_context.cpp:6) and [request_context.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/request_context.cpp:267).
- `ModelSchedule` is useful, but its role detection is still heuristic and name-driven, not a fully explicit manifest contract in [model_schedule.cpp](/home/khkramer/src/nemotron-inference/runtime/src/loader/model_schedule.cpp:30).

## vLLM Reference Shape

- vLLM splits control flow into `LLMEngine`, `InputProcessor`, `OutputProcessor`, and `EngineCoreClient` in [llm_engine.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/v1/engine/llm_engine.py:48) and [core_client.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/v1/engine/core_client.py:69).
- vLLM keeps request/scheduler state explicit. `EngineCoreRequest` and `EngineCoreOutput` define the request boundary in [__init__.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/v1/engine/__init__.py:66) and [__init__.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/v1/engine/__init__.py:140).
- vLLM places waiting/running request queues, KV-cache manager ownership, and cache-related scheduling policy in `Scheduler` in [scheduler.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/v1/core/sched/scheduler.py:67).
- vLLM’s prefix caching is cache-manager driven and hash-based over KV blocks, not whole-request snapshots, in [prefix_caching.md](/home/khkramer/src/nemotron-inference/third_party/vllm/docs/design/prefix_caching.md:5).

## Alignment

- The branch already has the right high-level ingredients: manifest-backed bootstrap, request-local state, reusable cache publication, and layer-wise execution.
- `RunGreedyConversationTurn()` already does the right kind of end-to-end work for a conversation turn: lookup cache, restore state, run uncached tail, decode, and publish the committed head in [single_token_forward_model.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/single_token_forward_model.cpp:988).
- `RunTokens()` already provides a single execution path across attention, Mamba, and expert layers in [single_token_forward_model.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/single_token_forward_model.cpp:1179).
- The current branch is closer to a real engine than a prototype, but it still lacks the explicit boundaries that make vLLM easier to reason about and easier to evolve.

## Accidental Divergences To Close

- The control plane is embedded inside model execution. `RunGreedyConversationTurn()` and `RunTokens()` mix request policy, cache policy, and layer orchestration instead of routing through explicit request and scheduling components.
- Request state ownership is too broad. `RequestExecutionContext` owns both persistent per-request state and execution scratch, which makes the execution model harder to separate from the control plane.
- Scheduling policy is implicit. There is no clear equivalent of vLLM’s waiting/running queues, request status transitions, or scheduler output objects in the current branch.
- `ModelSchedule` relies on tensor-name substring heuristics to classify attention, Mamba, and expert roles in [model_schedule.cpp](/home/khkramer/src/nemotron-inference/runtime/src/loader/model_schedule.cpp:39). That is pragmatic, but it is not a polished source-of-truth contract.
- MoE chunking policy is embedded in the forward loop in [single_token_forward_model.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/single_token_forward_model.cpp:1386). The behavior is likely correct, but its placement is an orchestration smell relative to vLLM’s scheduler-centered chunking.
- Bootstrap failure modes are coarse. `RuntimeEnvironment::BuildFromManifestFile()` does a lot of work before returning a valid environment, so the boundaries between validation, planning, and runtime assembly are still blurred.

## Intentional Divergences To Keep

- Exact-prefix reuse of KV plus Mamba state is intentional and should remain. vLLM’s cache model is hash-based KV-block reuse, while this branch needs exact-prefix snapshots in [prefix_caching.md](/home/khkramer/src/nemotron-inference/third_party/vllm/docs/design/prefix_caching.md:5) and [state_snapshot.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/state_snapshot.cpp:52).
- A single-process C++ runtime is acceptable for this branch if it stays clean and measurable. The goal is not to copy vLLM’s multiprocessing architecture verbatim, but to stop hiding control-plane responsibilities inside model code.
- Bounded MoE windowing is acceptable as an implementation choice, but it should be expressed through explicit engine-level policy rather than hardwired inside the layer loop.

## Prioritized Implementation Tasks

1. Extract an explicit request lifecycle boundary around the current forward path. The branch needs a first-class request object and request-state transition layer, even if it stays in-process.
2. Split bootstrap into phases. Separate manifest validation, execution-plan construction, cache initialization, and runtime instantiation so failure modes and responsibilities are obvious.
3. Move scheduling policy out of `SingleTokenForwardModel`. The current layer runner should execute a plan, not decide the plan.
4. Formalize request-owned state versus execution scratch. `RequestExecutionContext` should stop being the place where every per-request concern accumulates.
5. Replace heuristic layer-role inference with explicit manifest or loader-plan roles. If the manifest is the source of truth, the engine should stop inferring architecture from tensor names.
6. Make MoE chunking a named engine policy. The chunk window should be configurable and visible at the orchestration layer, not only as an internal loop detail.
7. Add explicit request/output objects or adapters that mirror the vLLM boundary shape. The target is not API compatibility, but architectural clarity.
8. Quarantine any remaining fallback behavior so the hot path is obviously the production path and the fallback path is obviously non-primary.

## Missing Evidence

- We do not yet have a control-plane profile that quantifies how much time is spent in bootstrap, cache lookup, request restore, and layer orchestration versus kernel execution.
- We do not yet have an explicit comparison showing which parts of the current monolithic path are actually worth preserving for Nano on 5090 and which parts are just convenience debt.
- We do not yet know whether splitting request lifecycle and scheduling policy will change correctness or materially affect latency on this branch.
- We do not yet have a fully explicit manifest-role schema to prove that `ModelSchedule` can be made non-heuristic without losing flexibility.

## Sequencing Advice

1. Clean up the control-plane boundary before touching micro-optimizations.
2. Make request lifecycle explicit before refactoring cache or layer code around it.
3. Replace heuristic schedule inference before widening the orchestration split, so later refactors have a stable source of truth.
4. Only after the engine boundary is explicit should we decide which fallback paths stay, which are hidden behind flags, and which should be removed.
5. Treat prefix-cache semantics as intentional, then align everything else around that decision instead of letting the cache design dictate the entire engine structure.
