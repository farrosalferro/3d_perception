# VIP3D Paper-to-Code (Prediction)

## 0) Scope and Artifacts

- Model key: `prediction/vip3d`
- Task: trajectory prediction
- Implementation: `pytorch_implementation/prediction/vip3d/`
- Test: `tests/prediction/vip3d.py`
- Notebook: `study/notebook/prediction/vip3d_paper_to_code.ipynb`
- Paper/source: external reference (local PDF not currently present): [VIP3D paper](https://arxiv.org/abs/2203.08376)

This reimplementation is an educational, pure-PyTorch forward path that keeps the core trajectory-prediction structure:
1) temporal agent encoding,
2) map token encoding,
3) agent-map fusion,
4) multimodal future decoding,
5) ADE/FDE metric smoke validation.

## Chunk 0: End-to-End Prediction Contract

### Goal
Predict `M` candidate trajectories for each agent over a future horizon `T_f`, plus mode confidence scores.

### Interface
- Input history tensor: `H \in R^{B x A x T_h x F_a}`
- Input map tensor: `P \in R^{B x N_m x S_m x F_m}`
- Output trajectories: `Y_hat \in R^{B x A x M x T_f x 2}`
- Output mode logits: `L \in R^{B x A x M}`
- Output mode probs: `pi = softmax(L)`

### Code Mapping
- `H` -> `agent_history` in `VIP3DLite.forward`
- `P` -> `map_polylines` in `VIP3DLite.forward`
- `Y_hat` -> `outputs["trajectories"]`
- `pi` -> `outputs["mode_probs"]`

### Sanity Check
- `mode_probs.sum(-1) == 1` for each `(b, a)`.
- `trajectories.shape[-2] == future_steps` (horizon integrity).

## Chunk 1: Agent Temporal Encoding

### Goal
Encode each agent history token sequence into a compact latent token.

### Equations
`z_{b,a,t} = W_h h_{b,a,t} + b_h`  
`u_{b,a,1:T_h} = TransformerEncoder(z_{b,a,1:T_h}, mask)`  
`q_{b,a} = Mean_t(u_{b,a,t}, valid_mask)`

### Symbol Table
- `h_{b,a,t}`: raw motion features (x, y, vx, vy)
- `z_{b,a,t}`: projected history token
- `u_{b,a,t}`: temporally encoded token
- `q_{b,a}`: agent latent token

### Code Mapping
- `W_h` -> `history_input_proj`
- `TransformerEncoder` -> `history_encoder`
- masked mean -> `_masked_temporal_mean`

### Sanity Check
- `history_tokens` preserves time axis `[B, A, T_h, C]`.
- Masked pooling never divides by zero (`clamp(min=1)`).

## Chunk 2: Map Tokenization

### Goal
Convert polyline points into fixed map tokens for cross-attention.

### Equations
`m_{b,n,s} = MLP_map(p_{b,n,s})`  
`k_{b,n} = W_k ( (1/S_m) * sum_s m_{b,n,s} )`

### Symbol Table
- `p_{b,n,s}`: map point feature
- `m_{b,n,s}`: encoded map point
- `k_{b,n}`: map token

### Code Mapping
- `MLP_map` -> `map_point_mlp`
- token projection `W_k` -> `map_token_proj`

### Sanity Check
- `map_tokens.shape == [B, N_m, C]`.
- All map tensors pass finite checks in `tests/prediction/vip3d.py`.

## Chunk 3: Agent-Map Fusion

### Goal
Fuse each agent token with map context via cross-attention.

### Equation
`f_{b,a} = LN( q_{b,a} + CrossAttn(query=q_{b,a}, key=k_{b,*}, value=k_{b,*}) )`

### Symbol Table
- `q_{b,a}`: agent token
- `k_{b,*}`: all map tokens in scene
- `f_{b,a}`: fused token for trajectory decoding

### Code Mapping
- `CrossAttn` -> `agent_map_attention`
- residual + normalization -> `fusion_norm`

### Sanity Check
- `fused_tokens.shape == [B, A, C]`.
- Attention output is captured with hooks (`fusion.cross_attention`).

## Chunk 4: Multimodal Trajectory Decoding

### Goal
Decode fused tokens into mode scores and future 2D trajectories.

### Equations
`r_{b,a} = DecoderMLP(f_{b,a})`  
`L_{b,a,:} = W_mode r_{b,a}`  
`Delta_{b,a,m,1:T_f} = reshape(W_delta r_{b,a})`  
`Y_hat_{b,a,m,t} = x_{b,a}^{last} + sum_{tau=1..t} Delta_{b,a,m,tau}`

### Symbol Table
- `r_{b,a}`: decoder latent
- `L_{b,a,:}`: mode logits
- `Delta`: per-step trajectory deltas
- `x^{last}`: last valid observed position
- `Y_hat`: absolute predicted trajectory

### Code Mapping
- decoder trunk -> `decoder.trunk`
- mode logits -> `decoder.mode_head`
- deltas -> `decoder.delta_head`
- cumulative integration -> `deltas.cumsum(dim=3)`
- best mode index -> `outputs["best_mode"]`

### Sanity Check
- Trajectory consistency:
  `Y_hat[..., t] - Y_hat[..., t-1] == Delta[..., t]` for `t >= 1`.

## Chunk 5: Prediction Metrics (ADE/FDE Smoke)

### Goal
Validate that trajectory metrics are computable and finite.

### Equations
`d_{b,a,m,t} = ||Y_hat_{b,a,m,t} - Y_{b,a,t}||_2`  
`ADE_{b,a,m} = (1/T_valid) * sum_t d_{b,a,m,t}`  
`FDE_{b,a,m} = d_{b,a,m,t_last_valid}`  
`mode* = argmin_m FDE_{b,a,m}`

### Code Mapping
- metric function -> `compute_ade_fde(...)`
- best-of-K selection -> `argmin` over per-mode FDE

### Sanity Check
- `ade` and `fde` are scalar, finite, non-negative.
- Horizon and validity masks are respected (`gt_valid`).

## Implementation Notes

- This module intentionally avoids MMDet3D/MMCV runtime dependencies.
- The forward path is deterministic in debug config (`dropout=0.0`).
- The test file validates:
  - intermediate hook captures,
  - shape contracts at major boundaries,
  - finite-value checks,
  - prediction-specific horizon and metric checks.
