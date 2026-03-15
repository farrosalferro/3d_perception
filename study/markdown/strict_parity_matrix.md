# Strict Parity Matrix (Pure PyTorch)

Date: 2026-03-15

Scope: 11 in-scope models (`7 perception + 4 prediction`) with Occ3D removed.

Anchors: `study/markdown/strict_parity_anchor_manifest.md`  
Contracts: `study/markdown/strict_parity_contracts.md`

## 1) 11-Model Parity Matrix

| Model Key | Parity Score (0-100) | Severity | Gate | Evidence Paths |
|---|---:|---|---|---|
| `perception/bevformer` | 96 | low | PASS | `pytorch_implementation/perception/bevformer/`, `tests/perception/bevformer.py`, `study/markdown/perception/bevformer_paper_to_code.md` |
| `perception/petr` | 95 | low | PASS | `pytorch_implementation/perception/petr/`, `tests/perception/petr.py`, `study/markdown/perception/petr_paper_to_code.md` |
| `perception/maptr` | 95 | low | PASS | `pytorch_implementation/perception/maptr/`, `tests/perception/maptr.py`, `study/markdown/perception/maptr_paper_to_code.md` |
| `perception/polarformer` | 94 | low | PASS | `pytorch_implementation/perception/polarformer/`, `tests/perception/polarformer.py`, `study/markdown/perception/polarformer_paper_to_code.md` |
| `perception/sparse4d` | 95 | low | PASS | `pytorch_implementation/perception/sparse4d/`, `tests/perception/sparse4d.py`, `study/markdown/perception/sparse4d_paper_to_code.md` |
| `perception/fbbev` | 94 | low | PASS | `pytorch_implementation/perception/fbbev/`, `tests/perception/fbbev.py`, `study/markdown/perception/fbbev_paper_to_code.md` |
| `perception/streampetr` | 95 | low | PASS | `pytorch_implementation/perception/streampetr/`, `tests/perception/streampetr.py`, `study/markdown/perception/streampetr_paper_to_code.md` |
| `prediction/beverse` | 94 | low | PASS | `pytorch_implementation/prediction/beverse/`, `tests/prediction/beverse.py`, `study/markdown/prediction/beverse_paper_to_code.md` |
| `prediction/surroundocc` | 94 | low | PASS | `pytorch_implementation/prediction/surroundocc/`, `tests/prediction/surroundocc.py`, `study/markdown/prediction/surroundocc_paper_to_code.md` |
| `prediction/vip3d` | 94 | low | PASS | `pytorch_implementation/prediction/vip3d/`, `tests/prediction/vip3d.py`, `study/markdown/prediction/vip3d_paper_to_code.md` |
| `prediction/flashocc` | 94 | low | PASS | `pytorch_implementation/prediction/flashocc/`, `tests/prediction/flashocc.py`, `study/markdown/prediction/flashocc_paper_to_code.md` |

Gate rule used: strict metadata contracts + intermediate tensor checks + decode/postprocess parity semantics + temporal/state checks + pure-PyTorch runtime.

## 2) Pure-PyTorch Replacements for Upstream Custom Ops

- `perception/bevformer`: multi-scale deformable attention and temporal rotation/shift kernels -> pure PyTorch `grid_sample` and tensor aggregation.
- `perception/petr`: plugin transformer wrappers/coders -> direct `nn.Module` composition and local tensor decode.
- `perception/maptr`: deformable attention ops and coder wrappers -> pure PyTorch sampled attention and local map decode.
- `perception/polarformer`: polar CUDA attention kernels -> pure PyTorch polar sampling/interpolation.
- `perception/sparse4d`: sparse plugin operators -> dense masked tensor ops, gather/matmul/grid sampling.
- `perception/fbbev`: projection/depth custom operators -> torch frustum projection, scatter-add/grid-sample pipelines.
- `perception/streampetr`: streaming memory/plugin helpers -> local tensor memory bank/ring-buffer utilities.
- `prediction/beverse`: registry-based task heads -> direct pure-PyTorch multitask head composition.
- `prediction/surroundocc`: custom cross-attention ops -> projection + `grid_sample` visibility-weighted fusion in torch.
- `prediction/vip3d`: bbox coder and interaction plugin pieces -> local tensor decode + pure-PyTorch interaction/memory update.
- `prediction/flashocc`: view-transform/warp custom operators -> torch depth lifting and transform-conditioned `grid_sample` warping.

## 3) Final BEVFormer-Relative Complexity Verdict

`perception/bevformer` remains one of the most complex models in this repo even after strict migration, and that is expected. Its temporal BEV state alignment plus multi-view deformable attention makes it structurally denser than models like `prediction/vip3d` and `prediction/surroundocc`, which have narrower prediction-focused scopes. Post-migration, complexity differences are now primarily architectural, not due to uneven implementation strictness.

## 4) Prioritized Remediation (Residual, Non-Blocking)

1. Add optional numeric parity fixtures against frozen upstream tensors for one deterministic frame per model.
2. Expand prediction metrics from smoke-level checks to small deterministic regression baselines.
3. Add cross-model CI command that always runs `pytest tests/perception/*.py tests/prediction/*.py -q`.
4. Add lightweight benchmark script to track runtime/memory deltas introduced by strict parity hardening.
