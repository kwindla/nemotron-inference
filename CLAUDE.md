# Nemotron Inference Runtime

From-scratch C++/CUDA inference engine for Nemotron-3 hybrid Mamba-attention-MoE models on RTX 5090.

## Development Workflow

When starting a substantial new development focus (new kernels, new subsystems, multi-file changes, performance optimization), use `/plan-init` → `/implement`:

1. `/plan-init <description>` — creates `proj-YYYY-MM-DD-HHMM/PLAN.md` with context, references, rules, and steps
2. Review the plan with the user; send to Codex for factual review (`/cx-delegate --background --fresh`)
3. Commit the reviewed plan before implementation starts
4. `/implement <project-dir>/PLAN.md` — executes all steps: delegate → review → fix → test → commit → next

Shared rules and the test protocol live in `PLAN_RULES.md` at the project root. Both commands read it.

### Key principles

- **The plan is ground truth.** Implementation follows the plan's intent.
- **One commit per step.** Clean git history maps 1:1 to plan steps.
- **Diagnose before fixing.** When something doesn't work, add a diagnostic step first. Don't guess at fixes.
- **Run on hardware.** Compile-only verification is necessary but not sufficient. Benchmarks reveal real bottlenecks.
- **Update the plan with findings.** When diagnosis reveals a different problem, rewrite the steps — don't implement the wrong fix.
- **Reference implementations are guardrails, not blueprints.** Align with the reference algorithm but justify divergences explicitly.
- **Always pass the analysis notes file** (`/home/khkramer/.claude/plans/*.md`) to Codex delegations so it has full context.
- **Stream benchmark output.** Use `stdbuf -oL` and `std::unitbuf` so long runs can be monitored.
