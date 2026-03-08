# PETR Paper-to-Code Study Guide

This note maps PETR paper symbols/equations to the pure-PyTorch forward implementation in this repository.

Primary references:
- Paper: `papers/PETR.pdf`
- Implementation: `pytorch_implementation/petr/`
- Intermediate tensor tests: `tests/petr/test_intermediate_tensors.py`

## 1) Canonical study setup (fixed debug run)

Use one setup so equation-to-tensor mapping stays stable across sections.

- Config:
  - `debug_forward_config(num_queries=48, decoder_layers=2, depth_num=6)`
- Input image:
  - `img`: `[B, Ncam, C, H, W] = [1, 6, 3, 96, 160]`
- Metadata (`img_metas`):
  - `img_shape`: per-camera `(96, 160, 3)`
  - `pad_shape`: per-camera `(96, 160, 3)`
  - `lidar2img`: `6 x (4x4)` projection matrices

Core dimensions under this setup:
- `embed_dims = 256`
- `num_classes = 10`
- `num_decoder_layers = 2`
- `num_queries = 48`
- `depth_num = 6`

Expected model outputs:
- `all_cls_scores`: `[L, B, Q, num_classes] = [2, 1, 48, 10]`
- `all_bbox_preds`: `[L, B, Q, code_size] = [2, 1, 48, 10]`

These are verified in `tests/petr/test_intermediate_tensors.py`.

## 2) Symbol dictionary (paper -> code tensors)

- `F_t^i` (camera feature for camera `i`) -> `mlvl_feats[0][:, i]`
- `P_img` (image-space positional signal) -> `sin_embed`
- `P_3d` (3D-aware positional signal) -> `coords_position_embeding`
- `P` (final memory key position) -> `pos_embed = P_3d + P_img`
- `r_q` (3D reference points for queries) -> `reference_points`
- `e_q` (query embedding) -> `query_embeds = query_embedding(pos2posemb3d(reference_points))`
- `H_l` (decoder hidden at layer `l`) -> `outs_dec[l]`
- `\hat{c}_l` (class logits layer `l`) -> `outputs_classes[l]`
- `\hat{b}_l` (box prediction layer `l`) -> `outputs_coords[l]`

Equation IDs below are stable and use `E<section>.<index>`.

---

## Chunk 0 - End-to-end forward contract

### Goal
Bind PETR high-level pipeline to concrete module calls.

### Paper concept/equation
PETR performs object-query decoding from multi-view image memory with 3D position cues.

### Explicit equations
`(E0.1)` Feature extraction and decoding:

$$
F_t = \mathrm{ImageEncoder}(I_t), \quad H = \mathrm{Decoder}(Q, F_t, P), \quad \hat{Y} = \mathrm{Head}(H)
$$

`(E0.2)` Layer-wise outputs:

$$
\hat{Y} = \{(\hat{c}_l, \hat{b}_l)\}_{l=1}^{L}
$$

### Symbol table (E0.*)
- `I_t`: multi-camera image tensor
- `F_t`: multi-camera memory features
- `Q`: object queries
- `P`: positional embedding added to memory keys
- `\hat{Y}`: class and box predictions across decoder layers

### Code mapping
- `PETRLite.forward` in `pytorch_implementation/petr/model.py`
- `PETRLite.extract_img_feat` in `pytorch_implementation/petr/model.py`
- `PETRHeadLite.forward` in `pytorch_implementation/petr/head.py`

### Tensor shape notes
- Input image: `[B, Ncam, 3, H, W]`
- Head outputs: `all_cls_scores [L, B, Q, Ccls]`, `all_bbox_preds [L, B, Q, Cbox]`

### One sanity check
`tests/petr/test_intermediate_tensors.py` asserts final output shapes for the debug config.

---

## Chunk 1 - Image features (backbone + neck)

### Goal
Understand how camera images are flattened, encoded, and restored to camera-major shape.

### Paper concept/equation
Each camera image is processed by shared CNN weights, then fused as a multi-view memory set.

### Explicit equations
`(E1.1)` Camera-batch flattening:

$$
I_t \in \mathbb{R}^{B\times N_{cam}\times 3\times H\times W}
\rightarrow
I'_t \in \mathbb{R}^{(B\cdot N_{cam})\times 3\times H\times W}
$$

`(E1.2)` Feature reshape back to camera axis:

$$
F'_t \in \mathbb{R}^{(B\cdot N_{cam})\times C\times H_f\times W_f}
\rightarrow
F_t \in \mathbb{R}^{B\times N_{cam}\times C\times H_f\times W_f}
$$

### Symbol table (E1.*)
- `N_cam`: number of cameras
- `H_f, W_f`: feature-map size after backbone/neck
- `C`: neck output channels

### Code mapping
- `BackboneNeck` in `pytorch_implementation/petr/backbone_neck.py`
- `extract_img_feat` in `pytorch_implementation/petr/model.py`

### Tensor shape notes
- Debug run yields `fpn.output0` shape `[6, 256, 6, 10]`
- After reshape: `[1, 6, 256, 6, 10]`

### One sanity check
`tests/petr/test_intermediate_tensors.py` validates backbone stage and `fpn.output0` shapes.

---

## Chunk 2 - 3D position embedding from geometry

### Goal
Connect PETR's 3D coordinate lifting with implemented tensor operations.

### Paper concept/equation
Image-grid points with sampled depths are lifted through inverse camera projection to 3D and normalized in a predefined range.

### Explicit equations
`(E2.1)` Pixel-depth homogeneous point:

$$
\tilde{p}(u,v,d) = [u\cdot d, v\cdot d, d, 1]^T
$$

`(E2.2)` Lift to 3D (lidar/world frame proxy):

$$
p_{3d} = T^{-1}_{lidar2img}\,\tilde{p}
$$

`(E2.3)` Range normalization:

$$
\bar{p}_{3d} = \frac{p_{3d} - p_{min}}{p_{max} - p_{min}}
$$

### Symbol table (E2.*)
- `(u,v)`: image coordinates at feature resolution
- `d`: sampled depth bin
- `T_{lidar2img}`: camera projection matrix from metadata
- `p_min, p_max`: `position_range` bounds

### Code mapping
- `PETRHeadLite.position_embeding` in `pytorch_implementation/petr/head.py`
- `inverse_sigmoid` in `pytorch_implementation/petr/utils.py`
- `position_encoder` in `pytorch_implementation/petr/head.py`

### Tensor shape notes
- Pre-conv geometry tensor: `[B*Ncam, 3*D, H_f, W_f]`
- Encoded 3D positional feature: `[B, Ncam, C, H_f, W_f]`

### One sanity check
`head.position_encoder` hook is asserted to output `[B*Ncam, C, H_f, W_f]`.

---

## Chunk 3 - Query construction from 3D reference points

### Goal
Map learned reference anchors to decoder query embeddings.

### Paper concept/equation
PETR uses learnable 3D reference points and sinusoidal embedding to parameterize object queries.

### Explicit equations
`(E3.1)` Learnable reference points:

$$
r_q \in \mathbb{R}^{Q\times 3}
$$

`(E3.2)` Query embedding:

$$
e_q = \mathrm{MLP}(\mathrm{PE}_{3d}(r_q))
$$

### Symbol table (E3.*)
- `Q`: number of object queries
- `r_q`: normalized 3D reference points
- `\mathrm{PE}_{3d}`: sinusoidal embedding (`pos2posemb3d`)
- `e_q`: decoder query positional embedding

### Code mapping
- `reference_points` and `query_embedding` in `pytorch_implementation/petr/head.py`
- `pos2posemb3d` in `pytorch_implementation/petr/utils.py`

### Tensor shape notes
- `reference_points.weight`: `[Q, 3]`
- `query_embeds`: `[Q, C]`

### One sanity check
Tests assert `head.reference_points` and `head.query_embedding` output shapes.

---

## Chunk 4 - Transformer decoder over multi-view memory

### Goal
Describe how flattened memory and query tokens interact through self/cross attention.

### Paper concept/equation
PETR decodes object tokens using decoder layers: self-attention, cross-attention to image memory, and FFN refinement.

### Explicit equations
`(E4.1)` Memory flattening:

$$
M \in \mathbb{R}^{(N_{cam}H_fW_f)\times B\times C}
$$

`(E4.2)` Decoder layer update:

$$
H_l = \mathrm{FFN}(\mathrm{CrossAttn}(\mathrm{SelfAttn}(H_{l-1})))
$$

### Symbol table (E4.*)
- `M`: flattened memory tokens from camera features
- `H_l`: decoder hidden state at layer `l`
- `L`: number of decoder layers

### Code mapping
- `PETRTransformerLite.forward` in `pytorch_implementation/petr/transformer.py`
- `PETRTransformerDecoderLayerLite` in `pytorch_implementation/petr/transformer.py`

### Tensor shape notes
- Decoder hidden per layer: `[Q, B, C]`
- Stacked decoder output: `[L, Q, B, C]`

### One sanity check
Tests verify each `decoder.layer*`, `self_attn`, `cross_attn`, and `ffn` output has shape `[Q, B, C]`.

---

## Chunk 5 - Class/box heads and metric-space decoding

### Goal
Map decoder states to class logits and 3D box predictions.

### Paper concept/equation
Each decoder layer predicts class scores and box parameters; center dimensions tied to reference points use inverse-sigmoid residual updates.

### Explicit equations
`(E5.1)` Per-layer predictions:

$$
\hat{c}_l = f_{cls}(H_l), \quad \hat{b}_l = f_{reg}(H_l)
$$

`(E5.2)` Reference-aware center update:

$$
\hat{x},\hat{y}=\sigma(\Delta_{xy}+\sigma^{-1}(r_{xy})), \quad
\hat{z}=\sigma(\Delta_{z}+\sigma^{-1}(r_{z}))
$$

`(E5.3)` Scale normalized centers to metric range:

$$
x = \hat{x}(x_{max}-x_{min}) + x_{min}
$$

and similarly for `y, z`.

### Symbol table (E5.*)
- `f_cls, f_reg`: layer-specific MLP heads
- `r_xy, r_z`: reference point components
- `\Delta_{xy}, \Delta_z`: residual outputs from regression branch

### Code mapping
- Layer branches in `PETRHeadLite._build_branches`
- Forward update equations in `PETRHeadLite.forward`
- Top-k decode in `NMSFreeCoderLite` (`pytorch_implementation/petr/postprocess.py`)

### Tensor shape notes
- `all_cls_scores`: `[L, B, Q, num_classes]`
- `all_bbox_preds`: `[L, B, Q, code_size]`
- Decoded inference set uses the last layer (`[-1]`) and top-k over `Q * num_classes`.

### One sanity check
Tests assert class/box branch output dimensions for every decoder layer and finite values for all captured outputs.

