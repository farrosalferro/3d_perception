---
name: paper-to-code
description: Produces structured paper-to-code study documents that map equations and symbols to implementation tensors and module paths. Use when creating or updating study notes for a model.
---

# Paper-to-Code Documentation

## Objective
Create a study note that links paper math to concrete tensors, modules, and tests.

## Workflow
1. Confirm model and canonical paths.
2. Use section scaffold from [section_template.md](section_template.md).
3. For each major block, write equation IDs and symbol mapping.
4. Map symbols to concrete code tensors and file paths.
5. Add one sanity check per section.
6. Verify consistency with implementation and tests.

## Required References
- Paper source in `papers/`
- Implementation in `pytorch_implementation/<model>/`
- Tests in `tests/<model>.py`

## Style Rules
- Keep terminology consistent from start to end.
- Prefer explicit shapes and tensor names over vague descriptions.
- Keep equations and IDs stable after publication.
