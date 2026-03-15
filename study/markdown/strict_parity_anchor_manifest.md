# Strict Parity Upstream Anchor Manifest

This manifest freezes the upstream source anchors used for strict behavioral parity migration.

Hard runtime rule for all implementations under `pytorch_implementation/`: pure PyTorch only (`torch`, `torch.nn`, `torch.nn.functional`, and standard Python). No MMDet3D/MMCV runtime imports, no custom CUDA ops.

## Perception

### `perception/bevformer`
- Upstream repo: `repos/perception/BEVFormer`
- Frozen SHA: `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`
- Anchor files:
  - `repos/perception/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`
  - `repos/perception/BEVFormer/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`
  - `repos/perception/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/transformer.py`
  - `repos/perception/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/encoder.py`
  - `repos/perception/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`
  - `repos/perception/BEVFormer/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`

### `perception/petr`
- Upstream repo: `repos/perception/PETR`
- Frozen SHA: `f7525f93467a33707ef401c587a52d5e7b34de74`
- Anchor files:
  - `repos/perception/PETR/projects/mmdet3d_plugin/models/detectors/petr3d.py`
  - `repos/perception/PETR/projects/mmdet3d_plugin/models/detectors/mspetr3d.py`
  - `repos/perception/PETR/projects/mmdet3d_plugin/models/dense_heads/petr_head.py`
  - `repos/perception/PETR/projects/mmdet3d_plugin/models/dense_heads/petrv2_head.py`
  - `repos/perception/PETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py`
  - `repos/perception/PETR/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`

### `perception/maptr`
- Upstream repo: `repos/perception/MapTR`
- Frozen SHA: `a6872d8d9670bde17b4b01560f1221f88b443d55`
- Anchor files:
  - `repos/perception/MapTR/projects/mmdet3d_plugin/maptr/detectors/maptr.py`
  - `repos/perception/MapTR/projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py`
  - `repos/perception/MapTR/projects/mmdet3d_plugin/maptr/modules/transformer.py`
  - `repos/perception/MapTR/projects/mmdet3d_plugin/maptr/modules/encoder.py`
  - `repos/perception/MapTR/projects/mmdet3d_plugin/maptr/modules/decoder.py`
  - `repos/perception/MapTR/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`

### `perception/polarformer`
- Upstream repo: `repos/perception/PolarFormer`
- Frozen SHA: `dadd1bfd213e00ddf3e6c77c4733acc089131142`
- Anchor files:
  - `repos/perception/PolarFormer/projects/mmdet3d_plugin/models/detectors/polarformer.py`
  - `repos/perception/PolarFormer/projects/mmdet3d_plugin/models/dense_heads/polarformer_head.py`
  - `repos/perception/PolarFormer/projects/mmdet3d_plugin/models/utils/polar_transformer.py`
  - `repos/perception/PolarFormer/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`

### `perception/sparse4d`
- Upstream repo: `repos/perception/Sparse4D`
- Frozen SHA: `c41df4bbf7bc82490f11ff55173abfcb3fb91425`
- Anchor files:
  - `repos/perception/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py`
  - `repos/perception/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py`
  - `repos/perception/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py`
  - `repos/perception/Sparse4D/projects/mmdet3d_plugin/models/blocks.py`
  - `repos/perception/Sparse4D/projects/mmdet3d_plugin/models/detection3d/decoder.py`

### `perception/fbbev`
- Upstream repo: `repos/perception/FB-BEV`
- Frozen SHA: `6e25469256d98e7fcb52cc43efe812dc2fd2b446`
- Anchor files:
  - `repos/perception/FB-BEV/mmdet3d/models/fbbev/detectors/fbocc.py`
  - `repos/perception/FB-BEV/mmdet3d/models/fbbev/heads/occupancy_head.py`
  - `repos/perception/FB-BEV/mmdet3d/models/fbbev/view_transformation/forward_projection/view_transformer.py`
  - `repos/perception/FB-BEV/mmdet3d/models/fbbev/view_transformation/backward_projection/bevformer_utils/bevformer_encoder.py`
  - `repos/perception/FB-BEV/mmdet3d/models/fbbev/view_transformation/backward_projection/bevformer_utils/spatial_cross_attention_depth.py`

### `perception/streampetr`
- Upstream repo: `repos/perception/StreamPETR`
- Frozen SHA: `95f64702306ccdb7a78889578b2a55b5deb35b2a`
- Anchor files:
  - `repos/perception/StreamPETR/projects/mmdet3d_plugin/models/detectors/petr3d.py`
  - `repos/perception/StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py`
  - `repos/perception/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py`
  - `repos/perception/StreamPETR/projects/mmdet3d_plugin/models/utils/misc.py`
  - `repos/perception/StreamPETR/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`

## Prediction

### `prediction/beverse`
- Upstream repo: `repos/prediction/BEVerse`
- Frozen SHA: `5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be`
- Anchor files:
  - `repos/prediction/BEVerse/projects/mmdet3d_plugin/models/detectors/beverse.py`
  - `repos/prediction/BEVerse/projects/mmdet3d_plugin/models/dense_heads/mtl_head.py`
  - `repos/prediction/BEVerse/projects/mmdet3d_plugin/models/dense_heads/det_head.py`
  - `repos/prediction/BEVerse/projects/mmdet3d_plugin/models/dense_heads/motion_head.py`
  - `repos/prediction/BEVerse/projects/mmdet3d_plugin/models/necks/temporal.py`

### `prediction/surroundocc`
- Upstream repo: `repos/prediction/SurroundOcc`
- Frozen SHA: `419bf5b47ee07ef0610950cce5ba99168c506753`
- Anchor files:
  - `repos/prediction/SurroundOcc/projects/mmdet3d_plugin/surroundocc/detectors/surroundocc.py`
  - `repos/prediction/SurroundOcc/projects/mmdet3d_plugin/surroundocc/dense_heads/occ_head.py`
  - `repos/prediction/SurroundOcc/projects/mmdet3d_plugin/surroundocc/modules/transformer.py`
  - `repos/prediction/SurroundOcc/projects/mmdet3d_plugin/surroundocc/modules/encoder.py`
  - `repos/prediction/SurroundOcc/projects/mmdet3d_plugin/surroundocc/modules/spatial_cross_attention.py`

### `prediction/vip3d`
- Upstream repo: `repos/prediction/ViP3D`
- Frozen SHA: `da59b556f4e8fdfe6b02997ed8c28c8ec0ba3324`
- Anchor files:
  - `repos/prediction/ViP3D/plugin/vip3d/models/vip3d.py`
  - `repos/prediction/ViP3D/plugin/vip3d/models/head_plus_raw.py`
  - `repos/prediction/ViP3D/plugin/vip3d/models/transformer.py`
  - `repos/prediction/ViP3D/plugin/vip3d/models/memory_bank.py`
  - `repos/prediction/ViP3D/plugin/vip3d/models/predictor_decoder.py`
  - `repos/prediction/ViP3D/plugin/vip3d/bbox_coder.py`

### `prediction/flashocc`
- Upstream repo: `repos/prediction/FlashOCC`
- Frozen SHA: `4084861d8d605bb01df55fcbc8072036055aa625`
- Anchor files:
  - `repos/prediction/FlashOCC/projects/mmdet3d_plugin/models/detectors/bevdet_occ.py`
  - `repos/prediction/FlashOCC/projects/mmdet3d_plugin/models/detectors/bevdet4d.py`
  - `repos/prediction/FlashOCC/projects/mmdet3d_plugin/models/detectors/bevdepth4d.py`
  - `repos/prediction/FlashOCC/projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py`
  - `repos/prediction/FlashOCC/projects/mmdet3d_plugin/models/necks/view_transformer.py`
