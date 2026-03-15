# Strict Parity Contracts (Pure PyTorch)

This document freezes behavior-level parity contracts for the 11 in-scope models.

Reference anchors are frozen in `study/markdown/strict_parity_anchor_manifest.md`.

## Global Runtime Rules

- Implementation code under `pytorch_implementation/` may only use:
  - `torch`, `torch.nn`, `torch.nn.functional`
  - Python standard library
- Forbidden runtime dependencies:
  - `mmcv`, `mmdet`, `mmdet3d`, `mmengine`, compiled custom CUDA ops.
- Custom upstream kernels must be replaced by explicit PyTorch equivalents with matching tensor contracts.

## Global Metadata Contract

Perception models must accept `metas` as `list[dict]` (batch-aligned) and validate presence of fields required by each model, including:
- camera geometry tensors (`lidar2img` and/or intrinsics/extrinsics),
- image padding/shape metadata (`img_shape`, `pad_shape`),
- temporal transform/can-bus fields when temporal fusion is enabled.

Prediction models must validate temporal axis semantics:
- `history` and `future` axis length checks,
- monotonic time indices for future rollouts,
- finite trajectory and occupancy outputs before decode.

## Per-Model Strict Contracts

### `perception/bevformer`
- **Behavior parity**
  - Temporal BEV state update with optional previous BEV rotation/shift from ego-motion.
  - Encoder uses temporal self-attention + spatial cross-attention over projected BEV references.
  - Decoder produces layer-wise box/class predictions and last-layer decode output.
- **Decode parity**
  - NMS-free decode semantics (`topk` over class logits, center/size/yaw/vel parameter layout).
- **PyTorch replacements**
  - Multi-scale deformable attention -> batched `grid_sample` + learned offsets + weighted reduction.
  - CUDA custom attention kernels -> pure tensor gather/scatter and interpolation.

### `perception/petr`
- **Behavior parity**
  - 3D position-aware query embeddings with camera-aware positional encoding.
  - Transformer decoder with per-layer prediction heads and reference update semantics.
- **Decode parity**
  - PETR/NMS-free coder-compatible denormalization and score/label/box output contract.
- **PyTorch replacements**
  - Registry/build wrappers -> direct `nn.Module` composition.
  - Any plugin attention wrappers -> native `nn.MultiheadAttention` or explicit QKV matmul path.

### `perception/maptr`
- **Behavior parity**
  - Hierarchical instance-point query structure with vectorized map element decoding.
  - BEV encoder + transformer decoder parity including point-set refinement per decoder layer.
- **Decode parity**
  - Ordered polyline point decoding with class-conditioned confidence selection.
- **PyTorch replacements**
  - Deformable attention plugins -> pure PyTorch sampled cross-attention.
  - Framework coders -> explicit tensor decode utilities in local postprocess.

### `perception/polarformer`
- **Behavior parity**
  - Polar BEV/range discretization and transformer fusion from multi-view camera features.
  - Query-wise decoder refinement preserving polar-to-Cartesian geometry consistency.
- **Decode parity**
  - 3D bbox decode contract aligned with polar head outputs and metric-space projection.
- **PyTorch replacements**
  - Polar transformer plugin ops -> explicit indexing and interpolation with torch ops only.

### `perception/sparse4d`
- **Behavior parity**
  - Persistent instance bank (anchor + feature memory), temporal fusion, iterative refinement.
  - Decoder stage updates anchors and features each layer with projection-aware sampling.
- **Decode parity**
  - Anchor-parameter decode to final metric boxes with confidence ranking.
- **PyTorch replacements**
  - Sparse plugin operators -> dense masked tensor ops, batched matmul, and gather.

### `perception/fbbev`
- **Behavior parity**
  - Forward projection and backward projection stages with depth-aware feature lifting.
  - Temporal fusion and occupancy/detection head outputs from BEV volume features.
- **Decode parity**
  - Occupancy logits and detection branches preserve channel/time axis conventions.
- **PyTorch replacements**
  - Depth projection kernels -> explicit frustum/grid transforms + `grid_sample`.
  - BEVFormer utils wrappers -> local pure-PyTorch attention/encoder components.

### `perception/streampetr`
- **Behavior parity**
  - Streaming temporal memory bank propagation and top-k proposal carry-over.
  - Decoder with propagated + current-frame queries and per-layer refinement.
- **Decode parity**
  - NMS-free decode contract with memory-aware query indexing preserved.
- **PyTorch replacements**
  - Queue/memory helper plugins -> local tensor ring-buffer utilities.

### `prediction/beverse`
- **Behavior parity**
  - Shared BEV trunk with multi-task heads (detection/map/motion occupancy) using temporal features.
  - Motion branch emits multimodal trajectories with mode probabilities over horizon.
- **Decode parity**
  - Future trajectory tensors follow `[B, A, M, T, 2]`-style semantics (agent/mode/time/xy).
  - Metric smoke checks include finite ADE/FDE.
- **PyTorch replacements**
  - Task-head registry modules -> direct composed pure-PyTorch heads.

### `prediction/surroundocc`
- **Behavior parity**
  - Multi-view 3D occupancy prediction with temporal BEV context and volumetric decode.
  - Transformer encoder path preserves camera-view fusion and voxel-grid semantics.
- **Decode parity**
  - Occupancy logits -> probabilities/masks with stable voxel axis ordering.
  - Optional trajectory branch keeps strict horizon/time indexing.
- **PyTorch replacements**
  - Multi-scale deformable attention ops -> pure torch sampling/aggregation.

### `prediction/vip3d`
- **Behavior parity**
  - Agent-centric temporal memory bank and interaction-aware trajectory decoding.
  - Query update path preserves mode-wise predictions and temporal consistency.
- **Decode parity**
  - Trajectory decode yields mode scores + cumulative displacement to absolute positions.
  - ADE/FDE smoke checks and monotonic time-index checks required.
- **PyTorch replacements**
  - Custom bbox coder/plugin layers -> local decode utilities and tensor-only predictors.

### `prediction/flashocc`
- **Behavior parity**
  - BEV depth/view transformation + occupancy branch + trajectory head contracts.
  - Temporal stacking/warping path respects history-frame alignment semantics.
- **Decode parity**
  - Occupancy voxel logits and trajectory outputs preserve upstream tensor ordering.
  - Metric smoke checks include occupancy IoU-style and trajectory ADE/FDE finite checks.
- **PyTorch replacements**
  - View-transform CUDA plugins -> torch frustum projection and `grid_sample` warping.

## Gate Definition

A model only passes strict parity gate when all are true:
- anchor files and SHA map to this document,
- implementation path is pure PyTorch at runtime,
- decode/postprocess semantics match upstream behavior contract,
- temporal/memory/state logic matches upstream contract,
- tests cover intermediate tensor contracts and task-specific smoke metrics.
