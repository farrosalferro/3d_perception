# Project Agents Guide

## Mission
Build educational, pure-PyTorch reimplementations of autonomous-driving perception models so readers can understand math and mechanisms without MMDet3D/MMCV abstractions.

## Environment Setup
Use this environment before running any Python code or tests:

```bash
conda activate 3d_perception
```

## Source-of-Truth Layout
- Reference implementations: `repos/` (read for lineage, do not depend on at runtime)
- Paper assets: `papers/`
- Pure reimplementations: `pytorch_implementation/`
- Validation tests: `tests/`
- Paper-to-code study notes: `study/`
- Project progress log: `progress.md`

## Non-Negotiables
1. Reimplemented modules in `pytorch_implementation/` must run without MMDet3D/MMCV runtime dependencies.
2. Every model must include intermediate tensor checks (not only final output checks).
3. Every model must include a paper-to-code mapping document in `study/`.
4. Path naming must be consistent across code, tests, and docs.
5. `progress.md` must be updated at each milestone.
6. Any development branch must pass branch verification against `main` before merge.

## Canonical Per-Model Deliverables
For model `<model>`:
- `pytorch_implementation/<model>/`
- `tests/<model>.py`
- `study/<model>_paper_to_code.md`
- `papers/<model>.pdf` (or documented external link if file is excluded)

## Definition of Done
A model is done only when all items pass:
- Deterministic debug forward pass runs.
- Intermediate tensor dimensions are asserted at key layers.
- Intermediate and final tensors pass finite-value checks.
- Paper equations/symbols are mapped to concrete code tensors.
- `progress.md` contains the latest status and deliverable links.

## Recommended Run Commands
```bash
conda activate 3d_perception
pytest tests/<model>.py -q
```

## Naming and Consistency
- Keep model IDs lowercase with underscores (example: `bevformer`).
- Keep terminology consistent in docs (same symbol names and tensor names throughout).
- If a path alias changes, update code/tests/study docs in the same change.
