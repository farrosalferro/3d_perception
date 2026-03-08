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
| 2026-03-08 | MapTR | detached@ff1c2db | completed (forward + intermediate tensor checks) | `pytorch_implementation/maptr/` | `tests/maptr/test_intermediate_tensors.py` | `study/maptr_paper_to_code.md` | `papers/MapTR.pdf` | Pure-PyTorch MapTR forward path onboarded with hierarchical instance-point queries, BEV token encoding, and intermediate tensor validation. |
| 2026-03-08 | PolarFormer | `HEAD` (detached) | completed (forward + intermediate tensor checks) | `pytorch_implementation/polarformer/` | `tests/polarformer/test_intermediate_tensors.py` | `study/polarformer_paper_to_code.md` | `papers/PolarFormer.pdf` | Pure-PyTorch PolarFormer-lite forward path onboarded with polar ray projection, multi-level decoder outputs, and finite/shape validation. |
<<<<<<< HEAD
| 2026-03-08 | sparse4d | master | completed (forward + intermediate tensor checks) | `pytorch_implementation/sparse4d/` | `tests/sparse4d/test_intermediate_tensors.py` | `study/sparse4d_paper_to_code.md` | `papers/Sparse4D.pdf` | Pure-PyTorch Sparse4D-style forward path onboarded from upstream reference patterns; no MMDet3D/MMCV runtime dependency. |
=======
>>>>>>> main

## Entry Template

Use this template for new rows:

| Date | Model | Branch | Status | Implementation | Tests | Study Doc | Paper | Notes |
|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | <model> | <branch-or-N/A> | planned/in-progress/completed/blocked | `pytorch_implementation/<model>/` | `tests/<model>.py` | `study/<model>_paper_to_code.md` | `papers/<paper>.pdf` | Key decisions, blockers, and next step. |
