# Test Matrix Template

## Environment
```bash
conda activate 3d_perception
```

## Artifact
- Test file: `tests/<task>/<model>.py`

## Matrix

| Test ID | Purpose | Input Setup | Assertions | Expected Result | Status |
|---|---|---|---|---|---|
| T01 | Forward smoke test | Tiny debug config | Output keys exist | Pass | planned |
| T02 | Intermediate shape checks | Hook major modules | Exact tensor dims at critical boundaries | Pass | planned |
| T03 | Finite-value checks | Iterate tensors | `torch.isfinite(...).all()` for intermediates and outputs | Pass | planned |
| T04 | Perception decode contract | Detection/map debug config | Decode outputs have expected keys/shapes/dtypes | Pass | planned |
| T05 | Prediction horizon integrity | Multi-step prediction debug config | Time axis length, mask validity, trajectory shape consistency | Pass | planned |
| T06 | Prediction metric smoke | Simple deterministic trajectories | ADE/FDE (or task metric) computes finite values | Pass | planned |
| T07 | Planning feasibility checks | Planning debug rollout | Kinematic bounds satisfied (speed/accel/curvature) | Pass | planned |
| T08 | Planning safety checks | Scenario with static/dynamic agents | Collision/safety guard checks pass | Pass | planned |

## Notes
- Keep debug config small for fast runtime.
- Keep task-irrelevant rows as `N/A` with rationale.
- Add model-specific tests if the architecture has unique components.
