---
name: paper-to-code
description: Produces structured paper-to-code study markdown and generated notebooks that map equations and symbols to implementation tensors and module paths. Use when creating or updating study notes for a task-model.
---

# Paper-to-Code Documentation

## Objective
Create a study artifact pair (markdown + notebook) that links paper math to concrete tensors, modules, and tests.

## Workflow
1. Confirm model key `<task>/<model>` and canonical paths.
2. Use section scaffold from [section_template.md](section_template.md).
3. For each major block, write equation IDs and symbol mapping.
4. Map symbols to concrete code tensors and file paths.
5. Add one sanity check per section, including task-specific checks where needed:
   - prediction: horizon/time indexing, trajectory coordinates, and metric definitions
   - planning: action/control semantics, constraints, and safety terms
6. Generate notebook from markdown source:
   - `conda activate 3d_perception`
   - `python study/notebook/_generate_notebooks.py`
7. Verify consistency across implementation, tests, markdown, and notebook.

## Required References
- Paper source in `papers/<task>/<model>.pdf` (or documented external link)
- Implementation in `pytorch_implementation/<task>/<model>/`
- Tests in `tests/<task>/<model>.py`
- Markdown study note in `study/markdown/<task>/<model>_paper_to_code.md`
- Generated notebook in `study/notebook/<task>/<model>_paper_to_code.ipynb`

## Style Rules
- Keep terminology consistent from start to end.
- Prefer explicit shapes and tensor names over vague descriptions.
- Keep equations and IDs stable after publication.
- Explicitly map temporal/planning symbols when the task is not perception-only.
- Keep markdown and notebook content synchronized in the same change.
