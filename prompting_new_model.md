# Prompting Guide: Add a New Model

Use this guide when you want the agent to implement a new model under this repository workflow.

## Before Prompting
- Confirm reference source exists in `repos/<model_repo>/`.
- Confirm paper exists in `papers/` (or provide link/title).
- Decide canonical model ID (lowercase, underscore style), for example: `petr`.
- Use this environment for run/test steps:

```bash
conda activate 3d_perception
```

## Recommended Prompt Pattern
Copy and edit:

```text
Please onboard model <model_id> using the project workflow.

Context:
- Reference code: repos/<model_repo>/
- Paper: papers/<paper_file>.pdf
- Target task: <3d detection / map construction / segmentation>

Requirements:
1) Implement pure PyTorch forward path in pytorch_implementation/<model_id>/ without MMDet3D/MMCV runtime dependency.
2) Add intermediate tensor tests in tests/<model_id>/test_intermediate_tensors.py (shape checks + finite checks).
3) Add markdown study document in study/markdown/<model_id>_paper_to_code.md mapping equations/symbols to code tensors.
4) Generate notebook in study/notebook/<model_id>_paper_to_code.ipynb from the markdown source using:
   - conda activate 3d_perception
   - python study/notebook/_generate_notebooks.py
5) Update progress.md with date, model, status, branch, and deliverable paths (implementation/tests/markdown/notebook/paper).
6) Use conda activate 3d_perception for run/test commands.

Start with a short plan, then execute.
```

## Follow-Up Prompt Templates

### 1) Tighten Scope to MVP
```text
Limit this iteration to a minimal forward-only MVP with one debug config and one intermediate tensor test file.
```

### 2) Add Missing Coverage
```text
Please expand tests to cover encoder, decoder, and head intermediate tensor dimensions, then update the study doc accordingly.
```

### 3) Documentation Quality Pass
```text
Please refine study/markdown/<model_id>_paper_to_code.md to ensure each section has: goal, equations, symbol table, code mapping, and one sanity check, then regenerate study/notebook/<model_id>_paper_to_code.ipynb.
```

### 4) Progress Update Enforcement
```text
Before finishing, update progress.md with current status and exact artifact paths.
```

### 5) Pre-Merge Branch Verification
```text
Before merging to main, run branch verification for this branch and return:
- changed files summary
- conflict risk
- relevant test results
- final decision (ready / ready-with-warnings / blocked)
```

## Expected Deliverables Checklist
- `pytorch_implementation/<model_id>/`
- `tests/<model_id>/test_intermediate_tensors.py`
- `study/markdown/<model_id>_paper_to_code.md`
- `study/notebook/<model_id>_paper_to_code.ipynb`
- `progress.md` updated
- branch verification report (before merge to `main`)
