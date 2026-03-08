# Sparse4D Paper-to-Code Study Guide

This note maps Sparse4D paper symbols/equations to the pure-PyTorch forward implementation in this repository.

Primary references:
- Paper: `papers/Sparse4D.pdf`
- Paper (online mirror): [Sparse4D v2: Recurrent Temporal Fusion with Sparse Model (arXiv 2305.14018)](https://arxiv.org/pdf/2305.14018.pdf)
- Reference code (online): [linxuewu/Sparse4D](https://github.com/linxuewu/Sparse4D)
- Implementation: `pytorch_implementation/sparse4d/`
- Intermediate tensor tests: `tests/sparse4d/test_intermediate_tensors.py`

## 1) Canonical study setup (fixed debug run)

Use one setup so equation-to-tensor mapping stays stable across sections.

- Config:
  - `debug_forward_config(num_queries=48, decoder_layers=2)`
- Input image:
  - `img`: `[B, Ncam, C, H, W] = [1, 6, 3, 96, 160]`
- Metadata:
  - `projection_mat`: `[B, Ncam, 4, 4]`
  - `image_wh`: `[B, Ncam, 2]`

Core dimensions:
- `embed_dims = 256`
- `num_classes = 10`
- `num_decoder_layers = 2`
- `num_queries = 48`
- `box_code_size = 11`

Expected model outputs:
- `all_cls_scores`: `[L, B, Q, num_classes] = [2, 1, 48, 10]`
- `all_bbox_preds`: `[L, B, Q, 11] = [2, 1, 48, 11]`

These are verified in `tests/sparse4d/test_intermediate_tensors.py`.

## 2) Symbol dictionary (paper -> code tensors)

- `I_t` (multi-view image) -> `img`
- `F_t` (multi-level image features) -> `mlvl_feats`
- `Q_t` (instance query features) -> `instance_feature`
- `A_t` (instance anchors) -> `anchors`
- `E(A_t)` (anchor embedding) -> `anchor_embed`
- `H_l` (decoder hidden at layer `l`) -> `query` after `decoder.layers[l]`
- `\hat{c}_l` (class logits, layer `l`) -> `all_cls_scores[l]`
- `\hat{b}_l` (box prediction, layer `l`) -> `all_bbox_preds[l]`

Equation IDs below use `E<section>.<index>`.

---

## Chunk 0 - End-to-end forward contract

### Goal
Bind Sparse4D high-level pipeline to concrete module calls.

### Explicit equations
`(E0.1)` Forward path:

$$
F_t = \mathrm{ImageEncoder}(I_t),\;
(Q_t, A_t) = \mathrm{InstanceBank}(),\;
\hat{Y} = \mathrm{Head}(F_t, Q_t, A_t)
$$

`(E0.2)` Layer-wise outputs:

$$
\hat{Y} = \{(\hat{c}_l, \hat{b}_l)\}_{l=1}^{L}
$$

### Code mapping
- `Sparse4DLite.forward` and `Sparse4DLite.extract_img_feat` in `pytorch_implementation/sparse4d/model.py`
- `Sparse4DHeadLite.forward` in `pytorch_implementation/sparse4d/head.py`

### One sanity check
`tests/sparse4d/test_intermediate_tensors.py` asserts final output shapes for the debug config.

---

## Chunk 1 - Image feature extraction

### Goal
Map multi-camera image flattening and multi-level feature construction.

### Explicit equations
`(E1.1)` Camera-batch flattening:

$$
I_t \in \mathbb{R}^{B\times N_{cam}\times 3\times H\times W}
\rightarrow
I'_t \in \mathbb{R}^{(B\cdot N_{cam})\times 3\times H\times W}
$$

`(E1.2)` Multi-level features with camera reshape:

$$
F'_t{}^{(k)} \in \mathbb{R}^{(B\cdot N_{cam})\times C\times H_k\times W_k}
\rightarrow
F_t^{(k)} \in \mathbb{R}^{B\times N_{cam}\times C\times H_k\times W_k}
$$

### Code mapping
- `BackboneNeck` in `pytorch_implementation/sparse4d/backbone_neck.py`
- `extract_img_feat` in `pytorch_implementation/sparse4d/model.py`

### One sanity check
Tests validate `backbone.stage*` and `neck.output*` tensor shapes at each level.

---

## Chunk 2 - Instance bank and anchor encoding

### Goal
Connect Sparse4D sparse query initialization to concrete tensors.

### Explicit equations
`(E2.1)` Learnable sparse instance state:

$$
Q_t = \mathrm{Repeat}(Q_0, B),\;
A_t = \mathrm{Repeat}(A_0, B)
$$

`(E2.2)` Anchor encoding:

$$
E(A_t) = \mathrm{MLP}(A_t)
$$

### Code mapping
- `InstanceBankLite` in `pytorch_implementation/sparse4d/instance_bank.py`
- `SparseBox3DEncoderLite` in `pytorch_implementation/sparse4d/blocks.py`
- Assembly in `Sparse4DHeadLite.forward` (`pytorch_implementation/sparse4d/head.py`)

### One sanity check
Tests assert shapes for `head.instance_bank` and `head.anchor_encoder`.

---

## Chunk 3 - Decoder updates with image aggregation

### Goal
Map per-layer Sparse4D update blocks to the implemented decoder.

### Explicit equations
`(E3.1)` Query initialization:

$$
H_0 = Q_t + E(A_t)
$$

`(E3.2)` Decoder layer update:

$$
H_l = \mathrm{FFN}\Big(\mathrm{CrossAgg}(\mathrm{SelfAttn}(H_{l-1}, H_{l-1}, H_{l-1}), F_t)\Big)
$$

### Code mapping
- `SparseDecoderLayerLite` in `pytorch_implementation/sparse4d/blocks.py`
- `Sparse4DDecoderLite` loop in `pytorch_implementation/sparse4d/decoder.py`
- `DeformableFeatureAggregationLite` as a pure-PyTorch image-context surrogate in `pytorch_implementation/sparse4d/blocks.py`

### One sanity check
Tests verify `decoder.layer*`, `self_attn`, `cross_attn`, and `ffn` output shapes.

---

## Chunk 4 - Layer-wise class/box refinement and decode

### Goal
Map query states to class logits, box refinement, and final top-k decode.

### Explicit equations
`(E4.1)` Per-layer predictions:

$$
\hat{c}_l = f_{cls}(H_l),\;
\Delta b_l = f_{reg}(H_l),\;
\hat{b}_l = A_{l-1} + \Delta b_l
$$

`(E4.2)` Iterative anchor update:

$$
A_l = \mathrm{stopgrad}(\hat{b}_l)
$$

`(E4.3)` NMS-free top-k decode:

$$
(\mathrm{score}, \mathrm{label}, \mathrm{box}) =
\mathrm{TopK}(\sigma(\hat{c}_L), \hat{b}_L)
$$

### Code mapping
- `SparseBox3DRefinementLite` in `pytorch_implementation/sparse4d/blocks.py`
- Iterative updates in `Sparse4DDecoderLite.forward` (`pytorch_implementation/sparse4d/decoder.py`)
- `SparseBox3DDecoderLite.decode` in `pytorch_implementation/sparse4d/decoder.py`
- `Sparse4DHeadLite.get_bboxes` in `pytorch_implementation/sparse4d/head.py`

### One sanity check
Tests assert head branch output shapes and finite values for all captured intermediates/final tensors.
