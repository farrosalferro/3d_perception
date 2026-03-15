# Cross-Model Duplicate Inventory (Strict Parity Refactor)

Date: 2026-03-15  
Baseline validation before refactor: `pytest tests/perception/*.py tests/prediction/*.py -q` -> `67 passed`

## Scope

- Runtime refactor scope: exact + near-duplicate helpers/modules across perception and prediction.
- Constraints:
  - Preserve strict behavior parity.
  - Preserve pure-PyTorch runtime.
  - Preserve existing import paths via wrappers/re-exports.

## Locked Extraction Order

1. Low-risk runtime extraction.
2. Medium-risk runtime extraction.
3. High-risk runtime extraction.
4. Test helper deduplication.
5. Notebook generator deduplication.
6. Full parity + compatibility validation.

## Duplicate Clusters

### Low Risk

1. Shared numerics
   - `inverse_sigmoid` duplicates:
     - `pytorch_implementation/perception/bevformer/utils/math.py`
     - `pytorch_implementation/perception/petr/utils.py`
     - `pytorch_implementation/perception/maptr/utils.py`
     - `pytorch_implementation/perception/polarformer/utils.py`
     - `pytorch_implementation/perception/streampetr/utils.py`
2. Shared positional encoding
   - `SinePositionalEncoding2D` duplicates:
     - `pytorch_implementation/perception/petr/utils.py`
     - `pytorch_implementation/perception/maptr/utils.py`
     - `pytorch_implementation/perception/polarformer/utils.py`
     - `pytorch_implementation/perception/streampetr/utils.py`
3. Simple backbone/FPN scaffold
   - `_ConvBlock`, `SimpleBackbone`, `SimpleFPN`, `BackboneNeck` near-identical implementations:
     - `pytorch_implementation/perception/bevformer/backbone_neck.py`
     - `pytorch_implementation/perception/petr/backbone_neck.py`
     - `pytorch_implementation/perception/maptr/backbone_neck.py`
     - `pytorch_implementation/perception/streampetr/backbone_neck.py`
4. Prediction time-axis + metric primitives
   - Time axis builders and ADE/FDE kernels:
     - `pytorch_implementation/prediction/beverse/model.py`
     - `pytorch_implementation/prediction/beverse/metrics.py`
     - `pytorch_implementation/prediction/flashocc/head.py`
     - `pytorch_implementation/prediction/flashocc/metrics.py`
     - `pytorch_implementation/prediction/vip3d/model.py`
     - `pytorch_implementation/prediction/surroundocc/postprocess.py`

### Medium Risk

1. Metadata validators and camera matrix parsing
   - `validate_*_img_metas`, shape coercion helpers:
     - `pytorch_implementation/perception/petr/utils.py`
     - `pytorch_implementation/perception/polarformer/utils.py`
     - `pytorch_implementation/perception/streampetr/utils.py`
     - `pytorch_implementation/perception/fbbev/model.py`
     - `pytorch_implementation/prediction/beverse/model.py`
     - `pytorch_implementation/prediction/surroundocc/model.py`
     - `pytorch_implementation/prediction/vip3d/model.py`
2. Decode/top-k helper flows
   - NMS-free style selection and label/score filtering:
     - `pytorch_implementation/perception/bevformer/postprocess/nms_free_coder.py`
     - `pytorch_implementation/perception/petr/postprocess.py`
     - `pytorch_implementation/perception/polarformer/postprocess.py`
     - `pytorch_implementation/perception/streampetr/postprocess.py`
     - `pytorch_implementation/prediction/flashocc/postprocess.py`
     - `pytorch_implementation/prediction/vip3d/model.py`

### High Risk

1. Geometry/projection and reference-point builders
   - Cross-model projection/reference logic:
     - `pytorch_implementation/perception/bevformer/utils/geometry.py`
     - `pytorch_implementation/perception/fbbev/forward_projection.py`
     - `pytorch_implementation/perception/fbbev/backward_projection.py`
     - `pytorch_implementation/perception/polarformer/backbone_neck.py`
2. Attention/decoder shared internals
   - Deformable/temporal attention and decoder layer skeletons:
     - `pytorch_implementation/perception/bevformer/modules/deformable_attention.py`
     - `pytorch_implementation/perception/bevformer/modules/temporal_self_attention.py`
     - `pytorch_implementation/perception/maptr/transformer.py`
     - `pytorch_implementation/perception/petr/transformer.py`
     - `pytorch_implementation/perception/polarformer/transformer.py`
     - `pytorch_implementation/perception/streampetr/transformer.py`
     - `pytorch_implementation/perception/sparse4d/blocks.py`

## Planned Shared Module Targets

- `pytorch_implementation/common/utils/numerics.py`
- `pytorch_implementation/common/utils/positional_encoding.py`
- `pytorch_implementation/common/backbone/simple_backbone_fpn.py`
- `pytorch_implementation/common/meta/validators.py`
- `pytorch_implementation/common/postprocess/topk.py`
- `pytorch_implementation/prediction/common/time_contracts.py`
- `pytorch_implementation/prediction/common/trajectory_metrics.py`
- `tests/_shared/tensor_helpers.py`
- `tests/_shared/hook_helpers.py`
- `tests/_shared/parity_helpers.py`

## Compatibility Strategy

- Keep existing per-model import surfaces stable.
- Delegate internals to shared modules.
- Preserve signatures and return-key contracts.
- Validate each extraction wave with targeted tests, then full suite.
