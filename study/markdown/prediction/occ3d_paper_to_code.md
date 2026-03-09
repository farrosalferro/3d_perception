# Occ3D (Prediction) Paper-to-Code Notes

Model key: `prediction/occ3d`  
Implementation: `pytorch_implementation/prediction/occ3d/`  
Tests: `tests/prediction/occ3d.py`  
Primary paper reference: [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/html/2304.14365v3)  
Project page: [Occ3D Benchmark](https://tsinghua-mars-lab.github.io/Occ3D/)

This implementation is a compact, educational pure-PyTorch predictor inspired by the Occ3D task setup. It focuses on temporal occupancy forecasting and trajectory consistency checks, not on reproducing leaderboard performance.

## Chunk 0 - End-to-End Prediction Contract

### Goal
Define the forward contract from historical BEV features to future occupancy volumes and trajectories.

### Paper Concept / Equation
- Future occupancy is predicted over time steps `t = 1..T`.
- Equation ID(s): `OCC3D-E0`, `OCC3D-E1`.

### Explicit Equations
`X_hist in R^{B x T_h x C_in x H x W}`  
`S = f_backbone(X_hist) in R^{B x T_h x C x H x W}`  
`H_fut = f_future(f_temporal(S)) in R^{B x T x C_t}`  
`Y_occ = f_occ(H_fut) in R^{B x T x Z x H x W}`  
`Delta P = f_traj(H_fut) in R^{B x A x T x 2}`  
`P_t = sum_{k=1..t} Delta P_k`

### Symbol Table
- `B` -> batch size -> number of scenes
- `T_h` -> `history_frames` -> observed history length
- `T` -> `future_horizon` -> prediction horizon
- `C_in` -> `input_channels` -> BEV feature channels
- `C` -> `embed_dims` -> spatial embedding channels
- `C_t` -> `temporal_hidden_dims` -> temporal state channels
- `A` -> `num_agents` -> number of predicted trajectories
- `Z,H,W` -> `bev_z, bev_h, bev_w` -> voxel grid dimensions

### Code Mapping
- File/module path(s): `pytorch_implementation/prediction/occ3d/model.py`
- Function/class names: `Occ3DLite.forward`
- Key tensor transitions:
  - `bev_history -> spatial_features`
  - `spatial_features -> temporal_encoded -> future_states`
  - `future_states -> occupancy_logits`
  - `future_states -> trajectory_deltas -> trajectories`

### Input Tensors
- `bev_history`: `[B, T_h, C_in, H, W]`, historical BEV feature stack.

### Output Tensors
- `occupancy_logits`: `[B, T, Z, H, W]`, unnormalized occupancy logits.
- `trajectory_deltas`: `[B, A, T, 2]`, per-step XY displacement.
- `trajectories`: `[B, A, T, 2]`, cumulative trajectories.
- `time_indices`: `[B, T]`, explicit time axis.

### Math Intuition
The model compresses history into one temporal context vector, then unfolds that context across future steps to jointly predict occupancy and trajectories.

### Sanity Check
`tests/prediction/occ3d.py` verifies that `time_indices` is strictly increasing by 1 and has exact length `future_horizon`.

## Chunk 1 - Spatial Backbone over Historical Frames

### Goal
Encode each historical frame into a shared latent BEV feature space.

### Paper Concept / Equation
- Shared spatial encoding across history frames.
- Equation ID(s): `OCC3D-E2`.

### Explicit Equations
For each history step `tau`:  
`S_tau = ConvBlock(X_tau)`  
Stacked output: `S = [S_1, ..., S_{T_h}]`

### Symbol Table
- `X_tau` -> one history BEV frame
- `S_tau` -> per-frame latent spatial feature map

### Code Mapping
- File/module path(s): `pytorch_implementation/prediction/occ3d/model.py`
- Function/class names: `SpatialBackbone.forward`
- Key tensor transitions:
  - Flatten history into batch: `[B*T_h, C_in, H, W]`
  - Apply conv stem + residual-style block
  - Reshape back to `[B, T_h, C, H, W]`

### Input Tensors
- Flattened history: `[B*T_h, C_in, H, W]`.

### Output Tensors
- `spatial_features`: `[B, T_h, C, H, W]`.

### Math Intuition
Using one shared backbone across all history frames enforces feature alignment before temporal aggregation.

### Sanity Check
The test hooks `backbone.stem` and `backbone.block` and asserts exact `[B*T_h, C, H, W]` shapes.

## Chunk 2 - Temporal Encoding and Future Decoding

### Goal
Convert spatial history into a temporal context and decode future latent states.

### Paper Concept / Equation
- Sequence modeling with recurrent temporal state.
- Equation ID(s): `OCC3D-E3`, `OCC3D-E4`.

### Explicit Equations
`u_tau = MeanPool(S_tau) in R^C`  
`z_tau = W_u u_tau`  
`h_1, ..., h_{T_h} = GRU_enc(z_1, ..., z_{T_h})`  
`q_t = Emb_time(t) + h_{T_h}`  
`g_1, ..., g_T = GRU_dec(q_1, ..., q_T; h_{T_h})`

### Symbol Table
- `u_tau` -> pooled spatial token
- `z_tau` -> projected token
- `h_tau` -> encoder hidden sequence
- `g_t` -> decoded future state
- `Emb_time` -> learned horizon-step embedding

### Code Mapping
- File/module path(s): `pytorch_implementation/prediction/occ3d/model.py`
- Function/class names: `TemporalEncoder`, `FutureDecoder`
- Key tensor transitions:
  - `spatial_features.mean(-2,-1) -> temporal_tokens`
  - `temporal_tokens -> temporal_encoded/context`
  - `context + time_embedding -> future_states`

### Input Tensors
- `spatial_features`: `[B, T_h, C, H, W]`.

### Output Tensors
- `temporal_tokens`: `[B, T_h, C_t]`
- `temporal_encoded`: `[B, T_h, C_t]`
- `future_states`: `[B, T, C_t]`

### Math Intuition
Encoder GRU summarizes observed dynamics; decoder GRU unfolds this summary over a fixed horizon with learned time anchors.

### Sanity Check
Tests assert `future_decoder.time_embedding` shape is `[T, C_t]` and `future_decoder.gru` output is `[B, T, C_t]`.

## Chunk 3 - Occupancy and Trajectory Heads

### Goal
Map future latent states to voxel occupancy and motion trajectories.

### Paper Concept / Equation
- Multi-head prediction from shared future latent states.
- Equation ID(s): `OCC3D-E5`, `OCC3D-E6`.

### Explicit Equations
`Y_occ_t = W_occ g_t` reshaped to `[Z, H, W]`  
`Delta P_t = W_traj g_t` reshaped to `[A, 2]`  
`P_t = P_{t-1} + Delta P_t`

### Symbol Table
- `Y_occ_t` -> occupancy logits at horizon step `t`
- `Delta P_t` -> trajectory displacement at step `t`
- `P_t` -> integrated trajectory position at step `t`

### Code Mapping
- File/module path(s): `pytorch_implementation/prediction/occ3d/model.py`
- Function/class names: `OccupancyHead.forward`, `TrajectoryHead.forward`
- Key tensor transitions:
  - `future_states -> occupancy_logits`
  - `future_states -> trajectory_deltas`
  - `trajectory_deltas.cumsum -> trajectories`

### Input Tensors
- `future_states`: `[B, T, C_t]`.

### Output Tensors
- `occupancy_logits`: `[B, T, Z, H, W]`.
- `trajectory_deltas`: `[B, A, T, 2]`.
- `trajectories`: `[B, A, T, 2]`.

### Math Intuition
A shared latent state provides temporal coherence; occupancy and trajectory branches specialize into scene-centric and agent-centric predictions.

### Sanity Check
Tests reconstruct trajectories with `trajectory_deltas.cumsum(dim=2)` and assert equality with `trajectories`.

## Chunk 4 - Decode Contract and Metrics Smoke Checks

### Goal
Define task-level postprocess outputs and lightweight metric checks.

### Paper Concept / Equation
- Occupancy probabilities and binary masks.
- Trajectory quality using standard ADE/FDE smoke metrics.
- Equation ID(s): `OCC3D-E7`, `OCC3D-E8`.

### Explicit Equations
`P_occ = sigmoid(Y_occ)`  
`M_occ = 1[P_occ >= theta]`  
`ADE = (1/N) sum ||P_pred - P_gt||_2`  
`FDE = (1/N) sum ||P_pred(T) - P_gt(T)||_2`

### Symbol Table
- `theta` -> occupancy threshold
- `P_occ` -> occupancy probability
- `M_occ` -> occupancy binary mask
- `ADE/FDE` -> trajectory error metrics

### Code Mapping
- File/module path(s):
  - `pytorch_implementation/prediction/occ3d/postprocess.py`
  - `pytorch_implementation/prediction/occ3d/metrics.py`
  - `tests/prediction/occ3d.py`
- Function/class names:
  - `Occ3DPostProcessorLite.decode`
  - `trajectory_ade_fde`
- Key tensor transitions:
  - `occupancy_logits -> occupancy_probs -> occupancy_binary`
  - `trajectories + gt -> ade/fde`

### Input Tensors
- `occupancy_logits`: `[B, T, Z, H, W]`
- `trajectories`: `[B, A, T, 2]`

### Output Tensors
- `occupancy_probs`: `[B, T, Z, H, W]`
- `occupancy_binary`: `[B, T, Z, H, W]` (bool)
- `metrics`: scalar `ade`, `fde`

### Math Intuition
Postprocess converts raw outputs into directly consumable contracts, while ADE/FDE smoke checks ensure metric plumbing is numerically stable.

### Sanity Check
`tests/prediction/occ3d.py` checks decoded tensor shapes/dtypes and verifies ADE/FDE are finite and non-negative.
