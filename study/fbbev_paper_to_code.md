# FB-BEV Paper-to-Code Study Guide

This note maps FB-BEV paper symbols/equations to the pure-PyTorch forward implementation in this repository.

Primary references:
- Paper: `/media/farrosalferro/College/study/3d_perception/papers/FB-BEV.pdf`
- Reference implementation: `/media/farrosalferro/College/study/3d_perception/repos/FB-BEV/`
- Pure-PyTorch implementation: `pytorch_implementation/fbbev/`
- Intermediate tensor tests: `tests/fbbev/test_intermediate_tensors.py`

## 1) Canonical study setup (fixed debug run)

Use one setup so equation-to-tensor mapping stays stable across sections.

- Config:
  - `debug_forward_config(max_num=48, depth_bins=6, bev_h=20, bev_w=20, bev_z=3)`
- Input image:
  - `img`: `[B, Ncam, C, H, W] = [2, 6, 3, 96, 160]`
- Metadata (`img_metas`) per sample:
  - `sequence_group_idx`
  - `start_of_sequence`
  - `curr_to_prev_ego_rt`: `4 x 4` ego transform matrix

Core dimensions under this setup:
- `embed_dims = 128`
- `num_classes = 10`
- `depth_bins = 6`
- `bev = [Hbev, Wbev, Zbev] = [20, 20, 3]`
- `max_num = 48`

Expected model outputs:
- `all_cls_scores`: `[L, B, Q, num_classes] = [1, 2, 48, 10]`
- `all_bbox_preds`: `[L, B, Q, code_size] = [1, 2, 48, 9]`

These are verified in `tests/fbbev/test_intermediate_tensors.py`.

## 2) Symbol dictionary (paper -> code tensors)

- `F` (image-view feature tensor) -> `camera_feat` / `context` in `FBBEVLite.forward`
- `D` (depth distribution) -> `depth` in `FBBEVDepthNetLite.forward`
- `B` (lifted BEV volume) -> `bev_volume` in `ForwardProjectionLite.forward`
- `B'` (depth-aware refined BEV) -> `bev_refined` in `BackwardProjectionLite.forward`
- `M` (temporal memory/history) -> `self.history_bev` in `TemporalFusionLite`
- `t_f` (temporal offset embedding) -> `time_map` in `TemporalFusionLite.forward`
- `Q_{x,y}` (BEV query locations) -> dense BEV grid positions in `FBBEVDetectionHeadLite._meshgrid`
- `N_c` (camera count) -> `cfg.num_cams`
- `N_ref` (depth reference bins) -> `cfg.depth_bins`
- `w_c` (depth consistency weight) -> `depth_weight` in `DepthAwareAttentionLite.forward`

Equation IDs below are stable and use `E<section>.<index>`.

---

## Chunk 0 - End-to-end forward contract

### Goal
Bind FB-BEV high-level 3D detection pipeline to concrete module calls.

### Paper concept/equation
FB-BEV combines forward projection, backward depth-aware refinement, and temporal BEV fusion before detection.

### Explicit equations
`(E0.1)` Pipeline:

$$
F = \mathrm{ImageEncoder}(I),\;
D = \mathrm{DepthNet}(F),\;
B = \mathrm{ForwardProj}(F, D),\;
B' = \mathrm{BackwardProj}(B, F, D)
$$

`(E0.2)` Temporal fusion and detection:

$$
\tilde{B} = \mathrm{TemporalFuse}(B', M, t_f),\quad
\hat{Y} = \mathrm{DetHead}(\mathrm{BEVEncoder}(\tilde{B}))
$$

### Symbol table (E0.*)
- `I`: multi-camera image input
- `M`: queued historical BEV features
- `\hat{Y}`: class and box predictions

### Code mapping
- `FBBEVLite.forward` in `pytorch_implementation/fbbev/model.py`
- `FBBEVDepthNetLite.forward` in `pytorch_implementation/fbbev/depth_net.py`
- `ForwardProjectionLite.forward` in `pytorch_implementation/fbbev/forward_projection.py`
- `BackwardProjectionLite.forward` in `pytorch_implementation/fbbev/backward_projection.py`
- `TemporalFusionLite.forward` in `pytorch_implementation/fbbev/temporal_fusion.py`
- `FBBEVDetectionHeadLite.forward` in `pytorch_implementation/fbbev/detection_head.py`

### One sanity check
`tests/fbbev/test_intermediate_tensors.py` checks final output shapes for the debug config.

---

## Chunk 1 - Camera projection and depth weighting

### Goal
Connect depth probabilities to BEV volume construction.

### Paper concept/equation
Use depth logits to weight per-camera context, then aggregate across cameras.

### Explicit equations
`(E1.1)` Depth probabilities:

$$
D = \mathrm{softmax}(W_d * F)
$$

`(E1.2)` Lift and aggregate:

$$
B = \frac{1}{N_c}\sum_{i=1}^{N_c} \left(F_i \odot D_i\right)
$$

### Symbol table (E1.*)
- `W_d`: depth prediction conv
- `\odot`: element-wise depth weighting

### Code mapping
- `FBBEVDepthNetLite` (`context_proj`, `depth_logits`) in `pytorch_implementation/fbbev/depth_net.py`
- `ForwardProjectionLite.forward` in `pytorch_implementation/fbbev/forward_projection.py`

### Tensor shape notes
- `context`: `[B, Ncam, C, Hf, Wf]`
- `depth`: `[B, Ncam, D, Hf, Wf]`
- `bev_volume`: `[B, C, Hbev, Wbev, Zbev]`

### One sanity check
Tests assert `depth_net.context_proj`, `depth_net.depth_logits`, and `forward_projection` output shapes.

---

## Chunk 2 - Depth-aware backward projection

### Goal
Map depth-consistency weighting to BEV refinement.

### Paper concept/equation
Depth-aware attention reweights context-to-BEV aggregation using consistency term `w_c`.

### Explicit equations
`(E2.1)` Depth-consistency weight:

$$
w_c = \sigma\left(\langle q, k \rangle\right)\cdot d_{prior}
$$

`(E2.2)` Depth-aware update:

$$
B' = \mathrm{Conv}\left(q + w_c \cdot v\right)
$$

### Symbol table (E2.*)
- `q, k, v`: projected BEV/context features
- `d_{prior}`: reduced depth confidence map

### Code mapping
- `DepthAwareAttentionLite.forward` in `pytorch_implementation/fbbev/depth_aware_attention.py`
- `BackwardProjectionLite.forward` in `pytorch_implementation/fbbev/backward_projection.py`

### Tensor shape notes
- `bev_2d`: `[B, C, Hbev, Wbev]`
- `depth_weight`: `[B, 1, Hbev, Wbev]`
- `bev_refined`: `[B, C, Hbev, Wbev]`

### One sanity check
Tests assert `backward.depth_attention` and `backward.post` intermediate shapes.

---

## Chunk 3 - Temporal fusion with ego-motion alignment

### Goal
Tie history queue update to temporal transform input.

### Paper concept/equation
Current BEV is fused with aligned history using relative ego-motion and temporal offset embeddings.

### Explicit equations
`(E3.1)` History alignment:

$$
M_t^{align} = \mathcal{W}(M_{t-1}, T_{t\rightarrow t-1})
$$

`(E3.2)` Time-aware fusion:

$$
\tilde{B}_t = \phi\left(\left[B_t,\; M_t^{align},\; t_f\right]\right)
$$

### Symbol table (E3.*)
- `\mathcal{W}`: warping/alignment operator
- `T_{t\rightarrow t-1}`: `curr_to_prev_ego_rt`
- `\phi`: small conv fusion network

### Code mapping
- `TemporalFusionLite._warp_single` and `TemporalFusionLite.forward` in `pytorch_implementation/fbbev/temporal_fusion.py`

### Tensor shape notes
- history queue: `[B, T, C, Hbev, Wbev]`
- fused output: `[B, C, Hbev, Wbev]`

### One sanity check
Tests assert `temporal.time_conv` and `temporal.cat_conv` outputs.

---

## Chunk 4 - Detection outputs and decoded 3D boxes

### Goal
Map BEV head outputs to query-style tensors and decoded boxes.

### Paper concept/equation
A dense BEV head predicts class logits and box fields per cell; top-k cells become query outputs.

### Explicit equations
`(E4.1)` Dense heads:

$$
H_{cls}, H_{reg} = f_{det}(\tilde{B})
$$

`(E4.2)` Metric conversion:

$$
x = \left(\sigma(\Delta x) + u\right)\cdot s_x + x_{min},\quad
y = \left(\sigma(\Delta y) + v\right)\cdot s_y + y_{min}
$$

`(E4.3)` Top-k decode:

$$
\hat{Y} = \mathrm{TopK}\left(\sigma(H_{cls}), H_{reg}\right)
$$

### Symbol table (E4.*)
- `(u, v)`: BEV grid coordinates
- `s_x, s_y`: metric scale from `pc_range`

### Code mapping
- `FBBEVDetectionHeadLite._decode_reg_map` in `pytorch_implementation/fbbev/detection_head.py`
- `FBBEVDetectionHeadLite._select_topk_queries` in `pytorch_implementation/fbbev/detection_head.py`
- `FBBEVBoxCoderLite.decode` in `pytorch_implementation/fbbev/postprocess.py`

### Tensor shape notes
- `all_cls_scores`: `[1, B, Q, num_classes]`
- `all_bbox_preds`: `[1, B, Q, 9]`

### One sanity check
Tests assert class/box output dimensions and finite values for all captures and outputs.
