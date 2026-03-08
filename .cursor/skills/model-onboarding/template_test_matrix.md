# Test Matrix Template

## Environment
```bash
conda activate 3d_perception
```

## Matrix

| Test ID | Purpose | Input Setup | Assertions | Expected Result | Status |
|---|---|---|---|---|---|
| T01 | Forward smoke test | Tiny debug config | Output keys exist | Pass | planned |
| T02 | Backbone shape checks | Hook backbone layers | Exact tensor dims | Pass | planned |
| T03 | Encoder shape checks | Hook encoder blocks | Exact tensor dims | Pass | planned |
| T04 | Decoder/head shape checks | Hook decoder + heads | Exact tensor dims | Pass | planned |
| T05 | Finite-value checks | Iterate tensors | `torch.isfinite(...).all()` | Pass | planned |

## Notes
- Keep debug config small for fast runtime.
- Add model-specific tests if the architecture has unique components.
