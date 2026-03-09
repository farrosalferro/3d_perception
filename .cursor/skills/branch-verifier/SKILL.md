---
name: branch-verifier
description: Verifies that a development branch is safe to merge into main by checking diff scope, potential merge conflicts, and relevant tests. Use when validating branch readiness before merge or pull request.
---

# Branch Verifier

## Goal
Check whether a development branch can be merged to `main` with low risk and clear visibility.

## Environment
```bash
conda activate 3d_perception
```

## Inputs
- Base branch (default: `main`)
- Development branch (default: current `HEAD`)

## Verification Workflow
1. Confirm repository state:
   - verify this is a git repository
   - detect current branch
2. Refresh branch data:
   - fetch base branch from remote when available
3. Inspect branch delta:
   - changed files and change types (`main...HEAD`)
   - commit list unique to development branch
4. Check conflict risk:
   - compute merge base
   - run merge-tree style check to detect conflict markers
5. Run relevant tests:
   - if `pytorch_implementation/<task>/<model>/` changed, run `pytest tests/<task>/<model>.py -q`
   - otherwise run the closest impacted test subset
6. Produce a decision report:
   - `ready`, `ready-with-warnings`, or `blocked`
   - blockers and recommended actions

## Report Format
Use this structure:
- Branch Pair: `<dev>` -> `<base>`
- Summary: one-line risk statement
- Changed Areas: key directories/files
- Conflict Risk: low/medium/high + evidence
- Test Results: command(s) + pass/fail
- Decision: ready / ready-with-warnings / blocked
- Next Actions: concrete steps
