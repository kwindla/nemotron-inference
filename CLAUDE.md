# CLAUDE.md

## Repo Structure

This repo contains self-contained platform specializations under `platforms/`.
Each platform has its own runtime, tests, benchmarks, tools, and docs.

Build with: `cmake -S . -B build -DNEMOTRON_PLATFORM=super-spark`

## Plan-Init / Implement Workflow

1. **`/plan-init`** — Create `proj-YYYY-MM-DD-HHMM/PLAN.md` with context, reference implementations
   (with `file:line` links), current state, domain details, 3-15 ordered steps, and a progress table.
2. **Codex background review** — Send the plan to `/cx-delegate --background --fresh` for deep review
   checking factual accuracy, algorithm fidelity, missing details, and step ordering.
3. **Incorporate findings** — Fix every error, add every missing detail directly in the plan text.
4. **Second review pass** — Self-review for reference alignment, then a final Codex review targeting
   remaining gaps.
5. **Commit the plan** — Plan goes into git before any implementation.
6. **`/implement` loop** — For each step: mark in-progress, delegate to Codex, read every changed file
   (not just summaries), fix issues, commit as `[step N]: description`, update progress table, proceed
   immediately.

### Key Principles

- Plan is ground truth.
- One commit per step.
- Diagnose before fixing (add diagnostic steps, don't guess).
- Run on hardware (compile-only isn't sufficient).
- Update the plan when diagnosis reveals different problems.
- Reference implementations are guardrails, not blueprints.
- Always pass analysis notes to Codex delegations.
- Stream benchmark output with `stdbuf -oL` and `std::unitbuf`.
