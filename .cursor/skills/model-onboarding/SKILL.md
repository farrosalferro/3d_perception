---
name: model-onboarding
description: Onboards an autonomous-driving model (perception, prediction, or planning) from reference repo and paper into pure PyTorch implementation, intermediate tensor tests, markdown study docs, generated notebooks, and progress logging. Use when adding a new model or planning a model port.
---

# Model Onboarding

## Quick Start
1. Select `task` (`perception`, `prediction`, or `planning`) and model ID.
2. Define model key as `<task>/<model>`.
3. Identify target model in `repos/<task>/<model>/` and matching paper in `papers/<task>/<model>.pdf` (or link).
4. Activate runtime environment:
   - `conda activate 3d_perception`
5. Fill [template_model_contract.md](template_model_contract.md).
6. Fill [template_test_matrix.md](template_test_matrix.md).
7. Implement minimal forward path in `pytorch_implementation/<task>/<model>/`.
8. Add intermediate tensor tests in `tests/<task>/<model>.py`.
9. Author `study/markdown/<task>/<model>_paper_to_code.md`.
10. Generate `study/notebook/<task>/<model>_paper_to_code.ipynb`:
   - `python study/notebook/_generate_notebooks.py`
11. Update `progress.md`.

## Required Outputs
- Pure-PyTorch implementation files under `pytorch_implementation/<task>/<model>/`
- Intermediate tensor validation tests under `tests/<task>/<model>.py`
- Paper-to-code markdown study note under `study/markdown/<task>/<model>_paper_to_code.md`
- Generated paper-to-code notebook under `study/notebook/<task>/<model>_paper_to_code.ipynb`
- Progress entry with status and artifact paths

## Quality Gates
- Shared gates:
  - No MMDet3D/MMCV runtime dependency in new implementation modules
  - Deterministic debug forward path
  - Shape assertions at key module boundaries
  - Finite-value checks for captured intermediates and outputs
- Task-specific gates:
  - perception: decode/postprocess contract checks
  - prediction: horizon/time-axis integrity and trajectory metric smoke checks
  - planning: kinematic feasibility and collision/safety checks
