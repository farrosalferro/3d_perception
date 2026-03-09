# Agent Runbook: Coordinator + Per-Model Subagents

Use this runbook when onboarding many models (especially `prediction`) with a coordinator agent and per-model agents.

## Canonical Conventions
- Model key format: `<task>/<model>`
- Supported tasks: `perception`, `prediction`, `planning`
- Example model key: `prediction/mtr`

Canonical artifact paths:
- Reference lineage: `repos/<task>/<model>/`
- Paper: `papers/<task>/<model>.pdf` (or documented external link)
- Implementation: `pytorch_implementation/<task>/<model>/`
- Tests: `tests/<task>/<model>.py`
- Study doc: `study/markdown/<task>/<model>_paper_to_code.md`
- Notebook: `study/notebook/<task>/<model>_paper_to_code.ipynb`

## Execution Strategy
- Use one coordinator agent to discover and schedule.
- Use one per-model agent per model for implementation.
- Limit parallel per-model agents to 2-3 at a time.
- Prefer one branch/worktree per model to avoid merge conflicts.

## Environment
Always use:

```bash
conda activate 3d_perception
```

## Step 1: Start the Coordinator Agent
Copy and edit:

```text
Act as coordinator only for onboarding models in task=<task>.

Do not implement model code yet.
1) Discover candidate models from:
   - repos/<task>/
   - papers/<task>/
2) Normalize IDs to model key format <task>/<model>.
3) Output a worklist table:
   - model_key
   - repo_path
   - paper_path
   - priority
   - risk
4) Propose execution order and branch names.
5) For each model, include expected deliverables:
   - pytorch_implementation/<task>/<model>/
   - tests/<task>/<model>.py
   - study/markdown/<task>/<model>_paper_to_code.md
   - study/notebook/<task>/<model>_paper_to_code.ipynb
   - progress.md update
```

## Step 2: Start Per-Model Agents
Run one agent per model key from the coordinator worklist.

Per-model prompt template:

```text
Implement model <task>/<model> end-to-end following repository rules under .cursor/.

Required outputs:
1) Implementation in pytorch_implementation/<task>/<model>/ (pure PyTorch only; no MMDet3D/MMCV runtime dependency)
2) Tests in tests/<task>/<model>.py including:
   - intermediate hook captures
   - shape assertions at critical boundaries
   - finite-value checks
   - task-specific checks:
     - perception: decode/postprocess contract checks
     - prediction: horizon/time-axis integrity, trajectory consistency, metric smoke checks
     - planning: kinematic feasibility and collision/safety checks
3) Study markdown in study/markdown/<task>/<model>_paper_to_code.md
4) Generated notebook in study/notebook/<task>/<model>_paper_to_code.ipynb from markdown source
5) progress.md update with date, model key, status, and artifact paths

Run/test guidance:
- Use conda activate 3d_perception
- Keep naming and paths consistent with <task>/<model>
- If blocked, document blocker clearly and still deliver best partial artifacts
```

## Step 3: Conflict Control Rules
- Do not let multiple agents edit the same file at once when avoidable.
- Highest conflict risk files:
  - `progress.md`
  - shared generators/scripts
- Good practice:
  - each per-model branch updates `progress.md` independently
  - integrate branches one-by-one and resolve `progress.md` centrally

## Step 4: Integration and Verification
After each per-model branch is ready:
1) Run relevant tests for that model.
2) Verify markdown and notebook are synchronized.
3) Run branch verification before merge.
4) Merge one model branch at a time.

Branch verification prompt:

```text
Run branch verification for this branch and report:
- changed files summary
- conflict risk
- relevant test results
- final decision (ready / ready-with-warnings / blocked)
```

## Coordinator Progress Check Prompt
Use this to monitor the whole program:

```text
Summarize current onboarding status for task=<task>:
- completed models
- in-progress models
- blocked models and blockers
- artifact coverage per model
- recommended next model to run
```

## Definition of Done Per Model
- Deterministic debug forward pass
- Intermediate shape assertions
- Finite checks for intermediates and outputs
- Task-specific validation checks pass
- Study markdown and generated notebook both updated
- `progress.md` updated with exact artifact paths
- Branch verification returns ready or ready-with-warnings

## One-Shot Prompt (Branch + Conflict Automation)
Copy-paste this when you want one agent to run the whole workflow with automatic branch handling.

```text
Act as a coordinator + implementer for task=prediction in this repository.

You must execute end-to-end with git branch automation and conflict handling.

Repository rules:
- Follow .cursor/AGENTS.md and .cursor rules/skills.
- Use canonical model key format <task>/<model> with task=prediction.
- Use conda activate 3d_perception before run/test commands.
- Do not use MMDet3D/MMCV runtime dependencies in reimplementations.

Goal:
Onboard all prediction models discovered from:
- repos/prediction/
- papers/prediction/

For each discovered model, produce:
1) pytorch_implementation/prediction/<model>/
2) tests/prediction/<model>.py
   - intermediate hooks
   - shape assertions
   - finite checks
   - horizon/time-axis integrity
   - trajectory consistency
   - metric smoke check (ADE/FDE or equivalent)
3) study/markdown/prediction/<model>_paper_to_code.md
4) study/notebook/prediction/<model>_paper_to_code.ipynb (generated from markdown)
5) progress.md update

Git workflow requirements:
1) Start from latest main.
2) For each model, create a dedicated branch:
   - feat/prediction-<model>
3) Implement and validate on that model branch only.
4) Commit model artifacts on its branch.
5) Rebase branch onto latest main before integration.
6) If rebase/merge conflict occurs:
   - resolve conflicts carefully
   - never use destructive reset
   - preserve both valid changes
   - continue rebase/merge and rerun relevant tests
7) Integrate model branches one-by-one into an integration branch:
   - feat/prediction-batch
8) Never force-push.

Conflict policy:
- Treat progress.md as high-conflict:
  - keep deterministic, append-only updates per model
  - resolve ordering conflicts without dropping entries
- If shared generator/script conflicts occur:
  - keep behavior backward compatible across already completed models

Execution policy:
- Work model-by-model; if one model is blocked, log blocker and continue.
- Do not stop after planning; execute.
- Do not modify unrelated files.

Final output:
Return a table with:
- model
- branch
- status (completed/blocked)
- tests run
- key artifacts
- conflicts encountered and how resolved
- remaining blockers
```
