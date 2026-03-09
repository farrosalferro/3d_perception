# Subagent Workflow Guide

## Purpose
Define a repeatable multi-role workflow for onboarding each task-model from `repos/` + `papers/` into pure PyTorch implementation, test coverage, markdown study documentation, and generated study notebooks.

## Runtime Convention
Before any code execution:

```bash
conda activate 3d_perception
```

## Roles
1. Paper Analyst
   - Extract core equations, symbols, and assumptions.
   - Output: symbol/equation table with unresolved ambiguities.
2. Source Reverse Engineer
   - Map original implementation blocks to minimal pure-PyTorch blocks.
   - Output: module lineage map and required tensor contracts.
3. Implementer
   - Implement forward path under `pytorch_implementation/<task>/<model>/`.
   - Output: runnable module(s) with explicit interfaces.
4. Tester
   - Add hook-based intermediate tensor tests in `tests/<task>/<model>.py`.
   - Output: shape assertions + finite-value assertions + task-specific checks:
     - perception: decode/postprocess contracts
     - prediction: horizon/time-axis consistency and trajectory metrics smoke checks
     - planning: kinematic feasibility and collision/safety checks
5. Documentation Writer
   - Create/update `study/markdown/<task>/<model>_paper_to_code.md`.
   - Generate `study/notebook/<task>/<model>_paper_to_code.ipynb` from markdown.
   - Output: equation -> symbol -> code mapping narrative and synced notebook.
6. Integrator
   - Verify definition-of-done checklist and update `progress.md`.
   - Output: milestone entry with status and artifact paths.
7. Branch Verifier
   - Compare development branch against `main` before merge.
   - Output: branch verification report with conflict risk, test status, and merge decision.

## Handoff Contract
Each handoff must include:
- Model key (`<task>/<model>`) and task type (`perception`, `prediction`, `planning`)
- Input/output tensor signatures
- Assumptions and open questions
- File paths of produced artifacts

## Required Milestone Update
When a milestone completes, append/update `progress.md` with:
- date
- task
- model
- model key (`<task>/<model>`)
- branch (or `N/A` if repository branch is unavailable)
- status (`planned`, `in-progress`, `completed`, `blocked`)
- deliverable paths (`repos`, `papers`, `pytorch_implementation`, `tests`, `study/markdown`, `study/notebook`)
- notes on key decisions and blockers

Notebook generation checkpoint for documentation milestones:
```bash
conda activate 3d_perception
python study/notebook/_generate_notebooks.py
```

## Pre-Merge Gate
Before merging any model branch to `main`, run Branch Verifier:
- Skill: `.cursor/skills/branch-verifier/SKILL.md`
- Minimum checks:
  - branch diff summary (`main...HEAD`)
  - potential merge conflict scan
  - relevant test run(s) using `conda activate 3d_perception`
