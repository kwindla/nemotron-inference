# Nemotron Inference Runtime

From-scratch C++/CUDA inference engine for Nemotron-3 hybrid Mamba-attention-MoE models on RTX 5090.

## Development Workflow

When starting a substantial new development focus (new kernels, new subsystems, multi-file changes, performance optimization), follow the plan-init → implement workflow:

### 1. Write the plan (`/plan-init`)

Create `proj-YYYY-MM-DD-HHMM/PLAN.md` with:
- **Context**: problem, target outcome, why this matters
- **Reference implementations**: specific file:line links to vLLM/SGLang/TRT-LLM or other codebases we're aligning with. State divergences explicitly with rationale.
- **Current state**: exact file paths and line numbers for code being changed
- **Domain details**: dimensions, math formulas, state layouts, data flow — everything needed to implement without guessing
- **Steps**: 3-15 ordered, independently committable units. Each lists key files.
- **Progress table**: status, commit hash, notes per step

### 2. Codex review (background)

Send plan to `/cx-delegate --background --fresh` for deep review against actual codebase and reference implementations. Codex checks: factual accuracy (line numbers, signatures), algorithm fidelity vs reference, missing details (transforms, activations, state formats), step ordering risks.

### 3. Incorporate findings

Fix every factual error. Add every missing detail. Edit the plan text — don't just acknowledge.

### 4. Second review pass

Re-read the plan yourself focused on reference alignment. Then send for a final Codex review targeting remaining gaps. Only report issues, not confirmations.

### 5. Commit the plan

The reviewed plan goes into git before any implementation starts.

### 6. `/implement` execution loop

For each step in order:
- Mark in-progress in the plan
- Delegate to Codex with focused prompt: step description, key files to read, verification instructions, reference to the full plan and analysis notes
- **Review every changed file** when Codex completes — read the actual diff, not the summary
- Fix issues directly, then commit with `[step N]: description`
- Update progress table with commit hash
- Immediately proceed to next step

### Key principles

- **The plan is ground truth.** Implementation follows the plan's intent.
- **One commit per step.** Clean git history maps 1:1 to plan steps.
- **Diagnose before fixing.** When something doesn't work, add a diagnostic step first. Don't guess at fixes.
- **Run on hardware.** Compile-only verification is necessary but not sufficient. Benchmarks reveal real bottlenecks.
- **Update the plan with findings.** When diagnosis reveals a different problem, rewrite the steps — don't implement the wrong fix.
- **Reference implementations are guardrails, not blueprints.** Align with the reference algorithm but justify divergences explicitly.
- **Always pass the analysis notes file** (`/home/khkramer/.claude/plans/*.md`) to Codex delegations so it has full context.
- **Stream benchmark output.** Use `stdbuf -oL` and `std::unitbuf` so long runs can be monitored.
