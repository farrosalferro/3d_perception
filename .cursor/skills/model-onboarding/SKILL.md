---
name: model-onboarding
description: Onboards an autonomous-driving perception model from reference repo and paper into pure PyTorch implementation, intermediate tensor tests, markdown study docs, generated notebooks, and progress logging. Use when adding a new model or planning a model port.
---

# Model Onboarding

## Quick Start
1. Identify target model in `repos/` and matching paper in `papers/`.
2. Activate runtime environment:
   - `conda activate 3d_perception`
3. Fill [template_model_contract.md](template_model_contract.md).
4. Fill [template_test_matrix.md](template_test_matrix.md).
5. Implement minimal forward path in `pytorch_implementation/<model>/`.
6. Add intermediate tensor tests in `tests/<model>/test_intermediate_tensors.py`.
7. Author `study/markdown/<model>_paper_to_code.md`.
8. Generate `study/notebook/<model>_paper_to_code.ipynb`:
   - `python study/notebook/_generate_notebooks.py`
9. Update `progress.md`.

## Required Outputs
- Pure-PyTorch implementation files
- Intermediate tensor validation tests
- Paper-to-code markdown study note
- Generated paper-to-code notebook
- Progress entry with status and artifact paths

## Quality Gates
- No MMDet3D/MMCV runtime dependency in new implementation modules
- Deterministic debug forward path
- Shape assertions at key module boundaries
- Finite-value checks for captured intermediates and outputs
