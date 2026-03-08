# Branch Verifier Usage

Use this when validating a model branch before merging to `main`.

## Environment
```bash
conda activate 3d_perception
```

## Agent Prompt (Copy/Paste)
```text
Run branch verification for my current branch against main.

Checks required:
1) Summarize commits and file changes in main...HEAD.
2) Estimate merge conflict risk and report evidence.
3) Run relevant tests for impacted model paths.
4) Return a decision: ready / ready-with-warnings / blocked.
5) Provide next actions to make it merge-ready.
```

## Optional Prompt (Explicit Branch Names)
```text
Run branch verification from <dev_branch> into main.
Use the branch-verifier subagent workflow and return a structured report.
```

## Expected Report Fields
- Branch Pair
- Changed Areas
- Conflict Risk
- Test Results
- Decision
- Next Actions
