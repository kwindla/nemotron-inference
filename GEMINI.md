# Gemini Mandates: Nemotron Runtime

This document contains foundational mandates for Gemini CLI. These instructions take absolute precedence over general workflows.

## 1. High-Level Architectural Goals
- **Hardware Target**: All optimizations and kernel implementations must be tailored for **NVIDIA Blackwell (RTX 5090 / SM120)**. Avoid generic or datacenter-only abstractions that do not benefit this specific consumer stack.
- **Language Boundary**: The serving runtime (`runtime/`, `kernels/`) is strictly **C++/CUDA**. Python is reserved for `tools/`, `testing/`, `benchmarks/`, and oracle generation.
- **Optimization Priority**: Prioritize **Time-To-First-Token (TTFT)** and **Prefix-Cache reuse** for long, multi-turn conversations over high-concurrency throughput.
- **Blackwell Native**: Favor native SM120 features (e.g., NVFP4 tensor cores, specific MMA instructions) for performance-critical paths.

## 2. Engineering & Validation Standards
- **Oracle-Backed Correctness**: No behavioral change is complete without passing the **oracle validation gate**. Functional agreement (top-k token overlap) is the final arbiter of correctness.
- **State Integrity**: Maintain the strict separation between request-local state (KV pages, Mamba states) and shared prefix-cache nodes.
- **Performance Regression**: Every optimization must be verified using the `benchmarks/` harnesses to ensure real-world speedup on the target hardware.

## 3. Workflow & Organization
- **Workspace Discipline**: Continue organizing substantial milestones or research tasks within `proj-YYYY-MM-DD-HHMM/` directories to maintain a clean implementation history.
- **Source of Truth**: Always treat `PROGRESS.md` as the authoritative log of implementation history and `docs/*.md` as the definitive architectural contracts.
