# Project Agents Guide

## Mission
Build educational, pure-PyTorch reimplementations of autonomous-driving models across perception, prediction, and planning so readers can understand math and mechanisms without MMDet3D/MMCV abstractions.

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
- Paper-to-code study markdown: `study/markdown/`
- Generated study notebooks: `study/notebook/`
- Project progress log: `progress.md`

## Canonical Model Key
- Use `<task>/<model>` for all artifacts.
- Supported task namespaces: `perception`, `prediction`, `planning`.

## Non-Negotiables
1. Reimplemented modules in `pytorch_implementation/` must run without MMDet3D/MMCV runtime dependencies.
2. Every model must include intermediate tensor checks (not only final output checks).
3. Every model must include a paper-to-code markdown document in `study/markdown/`.
4. Every model must include a generated notebook in `study/notebook/` that matches the markdown source.
5. Path naming must be consistent across code, tests, and docs.
6. `progress.md` must be updated at each milestone.
7. Any development branch must pass branch verification against `main` before merge.

## Canonical Per-Model Deliverables
For model key `<task>/<model>`:
- `repos/<task>/<model>/` (reference lineage)
- `papers/<task>/<model>.pdf` (or documented external link if file is excluded)
- `pytorch_implementation/<task>/<model>/`
- `tests/<task>/<model>.py`
- `study/markdown/<task>/<model>_paper_to_code.md`
- `study/notebook/<task>/<model>_paper_to_code.ipynb`

## Definition of Done
A model is done only when all items pass:
- Deterministic debug forward pass runs.
- Intermediate tensor dimensions are asserted at key layers.
- Intermediate and final tensors pass finite-value checks.
- Paper equations/symbols are mapped to concrete code tensors in markdown.
- Notebook is generated from the current markdown source in the same change.
- Task-specific checks pass:
  - Perception: decode/postprocess contract checks.
  - Prediction: horizon/time-axis integrity and trajectory metric smoke checks.
  - Planning: kinematic feasibility and collision/safety guard checks.
- `progress.md` contains the latest status and deliverable links.

## Recommended Run Commands
```bash
conda activate 3d_perception
pytest tests/<task>/<model>.py -q
python study/notebook/_generate_notebooks.py
```

## Naming and Consistency
- Keep task and model IDs lowercase with underscores (example: `perception/bevformer`).
- Keep terminology consistent in docs (same symbol names and tensor names throughout).
- Keep study artifacts synchronized as a pair: `study/markdown/<task>/<model>_paper_to_code.md` and `study/notebook/<task>/<model>_paper_to_code.ipynb`.
- If a path alias changes, update code/tests/study docs/notebooks in the same change.
