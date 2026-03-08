---
name: tensor-shape-validation
description: Builds intermediate tensor validation tests with forward hooks, shape assertions, and finite checks for pure PyTorch model implementations. Use when adding or extending model tests.
---

# Tensor Shape Validation

## Objective
Create reliable tests that verify tensor flow at intermediate layers and final outputs.

## Environment
```bash
conda activate 3d_perception
```

## Workflow
1. Define compact debug config for quick test runtime.
2. Register forward hooks on major modules.
3. Run one no-grad forward pass and collect captures.
4. Assert shapes for critical intermediate tensors.
5. Assert finite values on all captured tensors and outputs.
6. Keep tests isolated per model in `tests/<model>.py`.

## Template
- Start from [hook_capture_template.py](hook_capture_template.py).

## Required Assertions
- At least one assertion per major stage (backbone, encoder, decoder/head).
- End-to-end output shape assertions.
- Finite-value assertions.
