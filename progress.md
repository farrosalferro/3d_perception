# 3D Perception Reimplementation Progress

This file tracks model-level implementation progress and deliverables.

## Environment
Run code/tests in:

```bash
conda activate 3d_perception
```

## Status Table

| Date | Model | Branch | Status | Implementation | Tests | Study Doc | Paper | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-03-08 | BEVFormer | N/A (workspace not git-initialized) | completed (forward + intermediate tensor checks) | `pytorch_implementation/bevformer/` | `tests/bevformer.py` | `study/bevformer_paper_to_code.md` | `papers/BEVFormer.pdf` (if present) | Baseline reference for future model onboarding. |
| 2026-03-08 | PETR / PETRv2 | N/A (workspace not git-initialized) | planned | `pytorch_implementation/petr/` (planned) | `tests/petr/` (planned) | `study/petr_paper_to_code.md` (planned) | `papers/PETR*.pdf` (planned) | Next target model for first pilot pass. |
| 2026-03-08 | MapTR | N/A (workspace not git-initialized) | planned | `pytorch_implementation/maptr/` (planned) | `tests/maptr/` (planned) | `study/maptr_paper_to_code.md` (planned) | `papers/MapTR*.pdf` (planned) | Second pilot after PETR workflow refinement. |

## Entry Template

Use this template for new rows:

| Date | Model | Branch | Status | Implementation | Tests | Study Doc | Paper | Notes |
|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | <model> | <branch-or-N/A> | planned/in-progress/completed/blocked | `pytorch_implementation/<model>/` | `tests/<model>.py` | `study/<model>_paper_to_code.md` | `papers/<paper>.pdf` | Key decisions, blockers, and next step. |
