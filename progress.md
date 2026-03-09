# 3D Perception Reimplementation Progress

This file tracks model-level implementation progress and deliverables.

## Environment
Run code/tests in:

```bash
conda activate 3d_perception
```

## Status Table

| Date | Model | Branch | Status | Implementation | Tests | Study Doc (Markdown) | Notebook | Paper | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-03-08 | BEVFormer | N/A (workspace not git-initialized) | completed (forward + intermediate tensor checks) | `pytorch_implementation/bevformer/` | `tests/bevformer.py` | `study/markdown/bevformer_paper_to_code.md` | `study/notebook/bevformer_paper_to_code.ipynb` | `papers/BEVFormer.pdf` (if present) | Baseline reference for future model onboarding. |
| 2026-03-08 | PETR | N/A (workspace not git-initialized) | completed (forward + intermediate tensor checks) | `pytorch_implementation/petr/` | `tests/petr/test_intermediate_tensors.py` | `study/markdown/petr_paper_to_code.md` | `study/notebook/petr_paper_to_code.ipynb` | `papers/PETR.pdf` | Pure-PyTorch PETR forward path onboarded with geometry-aware positional embedding and decoder-stage tensor validation. |
| 2026-03-08 | MapTR | detached@ff1c2db | completed (forward + intermediate tensor checks) | `pytorch_implementation/maptr/` | `tests/maptr/test_intermediate_tensors.py` | `study/markdown/maptr_paper_to_code.md` | `study/notebook/maptr_paper_to_code.ipynb` | `papers/MapTR.pdf` | Pure-PyTorch MapTR forward path onboarded with hierarchical instance-point queries, BEV token encoding, and intermediate tensor validation. |
| 2026-03-08 | PolarFormer | `HEAD` (detached) | completed (forward + intermediate tensor checks) | `pytorch_implementation/polarformer/` | `tests/polarformer/test_intermediate_tensors.py` | `study/markdown/polarformer_paper_to_code.md` | `study/notebook/polarformer_paper_to_code.ipynb` | `papers/PolarFormer.pdf` | Pure-PyTorch PolarFormer-lite forward path onboarded with polar ray projection, multi-level decoder outputs, and finite/shape validation. |
| 2026-03-08 | sparse4d | master | completed (forward + intermediate tensor checks) | `pytorch_implementation/sparse4d/` | `tests/sparse4d/test_intermediate_tensors.py` | `study/markdown/sparse4d_paper_to_code.md` | `study/notebook/sparse4d_paper_to_code.ipynb` | `papers/Sparse4D.pdf` | Pure-PyTorch Sparse4D-style forward path onboarded from upstream reference patterns; no MMDet3D/MMCV runtime dependency. |
| 2026-03-08 | FBBEV | create-fbbef-model | completed (forward + intermediate tensor checks) | `pytorch_implementation/fbbev/` | `tests/fbbev/test_intermediate_tensors.py` | `study/markdown/fbbev_paper_to_code.md` | `study/notebook/fbbev_paper_to_code.ipynb` | `/media/farrosalferro/College/study/3d_perception/papers/FB-BEV.pdf` | Pure-PyTorch FB-BEV 3D detection forward path onboarded with temporal fusion, depth-aware refinement, and validated intermediate tensor contracts. |
| 2026-03-08 | StreamPETR | create-streampetr-model | completed (forward + temporal memory + intermediate tensor checks) | `pytorch_implementation/streampetr/` | `tests/streampetr/test_intermediate_tensors.py` | `study/markdown/streampetr_paper_to_code.md` | `study/notebook/streampetr_paper_to_code.ipynb` | `papers/StreamPETR.pdf` (remote source used) | Pure-PyTorch StreamPETR-style temporal query propagation implemented from public references: https://arxiv.org/abs/2303.11926 and https://github.com/exiawsh/StreamPETR. |

## Study Materials Update

| Date | Change | Details |
|---|---|---|
| 2026-03-08 | Markdown study docs created | All 7 models: `study/markdown/<model>_paper_to_code.md` with chunks 0-N, sections 1-2 |
| 2026-03-08 | Jupyter notebooks generated | All 7 models: `study/notebook/<model>_paper_to_code.ipynb` with executable code cells |
| 2026-03-09 | Sections 3-7 added to all models | Dataflow diagram, tensor trace, study drills, reading order, simplifications — for PETR, MapTR, FB-BEV, PolarFormer, Sparse4D, StreamPETR (BEVFormer already had them) |
| 2026-03-09 | Notebooks regenerated | All 7 notebooks updated with new sections 3-7 |

## Entry Template

Use this template for new rows:

| Date | Model | Branch | Status | Implementation | Tests | Study Doc (Markdown) | Notebook | Paper | Notes |
|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | <model> | <branch-or-N/A> | planned/in-progress/completed/blocked | `pytorch_implementation/<model>/` | `tests/<model>/test_intermediate_tensors.py` | `study/markdown/<model>_paper_to_code.md` | `study/notebook/<model>_paper_to_code.ipynb` | `papers/<paper>.pdf` | Key decisions, blockers, and next step. |
