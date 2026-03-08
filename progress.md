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
| 2026-03-08 | PETR | N/A (workspace not git-initialized) | completed (forward + intermediate tensor checks) | `pytorch_implementation/petr/` | `tests/petr/test_intermediate_tensors.py` | `study/petr_paper_to_code.md` | `papers/PETR.pdf` | Pure-PyTorch PETR forward path onboarded with geometry-aware positional embedding and decoder-stage tensor validation. |
| 2026-03-08 | MapTR | N/A (workspace not git-initialized) | planned | `pytorch_implementation/maptr/` (planned) | `tests/maptr/` (planned) | `study/maptr_paper_to_code.md` (planned) | `papers/MapTR*.pdf` (planned) | Second pilot after PETR workflow refinement. |
| 2026-03-08 | FBBEV | `HEAD (no branch)` | completed (forward + intermediate tensor checks) | `pytorch_implementation/fbbev/` | `tests/fbbev/test_intermediate_tensors.py` | `study/fbbev_paper_to_code.md` | `/media/farrosalferro/College/study/3d_perception/papers/FB-BEV.pdf` | Pure-PyTorch FB-BEV 3D detection forward path onboarded with temporal fusion, depth-aware refinement, and validated intermediate tensor contracts. |

## Entry Template

Use this template for new rows:

| Date | Model | Branch | Status | Implementation | Tests | Study Doc | Paper | Notes |
|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | <model> | <branch-or-N/A> | planned/in-progress/completed/blocked | `pytorch_implementation/<model>/` | `tests/<model>.py` | `study/<model>_paper_to_code.md` | `papers/<paper>.pdf` | Key decisions, blockers, and next step. |
