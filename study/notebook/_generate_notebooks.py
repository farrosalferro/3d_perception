"""Generate study notebooks from markdown sources and pytorch implementations."""

from __future__ import annotations

import os
import re
import textwrap

import nbformat as nbf

try:
    from ._builder_primitives import new_notebook as _shared_new_notebook
except ImportError:  # pragma: no cover - script execution fallback
    from _builder_primitives import new_notebook as _shared_new_notebook

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_DIR = os.path.join(WORKSPACE, "study", "markdown")
NB_DIR = os.path.join(WORKSPACE, "study", "notebook")
os.makedirs(NB_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md_cell(source):
    return nbf.v4.new_markdown_cell(source=textwrap.dedent(source).strip())


def code_cell(source):
    return nbf.v4.new_code_cell(source=textwrap.dedent(source).strip())


def code_cell_raw(source):
    """Create code cell from already-formatted source (no dedent)."""
    return nbf.v4.new_code_cell(source=source.strip())


def _split_markdown_sections(text):
    """Split markdown into (heading, body) pairs.  Top-level content before
    the first heading gets heading=''."""
    chunks = []
    current_heading = ""
    current_lines = []
    for line in text.splitlines(True):
        if re.match(r"^#{1,3}\s", line):
            if current_lines:
                chunks.append((current_heading, "".join(current_lines)))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        chunks.append((current_heading, "".join(current_lines)))
    return chunks


def _read_markdown(model_name: str) -> str:
    path = os.path.join(MD_DIR, f"{model_name}_paper_to_code.md")
    with open(path) as fh:
        return fh.read()


def _write_notebook(model_name: str, nb: nbf.NotebookNode) -> str:
    path = os.path.join(NB_DIR, f"{model_name}_paper_to_code.ipynb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        nbf.write(nb, fh)
    return path


def _chunk_id_from_heading(heading: str):
    m = re.match(r"^##\s+Chunk\s+(\d+)", heading)
    return int(m.group(1)) if m else None


def _new_notebook() -> nbf.NotebookNode:
    return _shared_new_notebook()


COMMON_HELPERS = '''\
from typing import Any

def _first_tensor(value: Any):
    """Extract the first tensor from a nested structure."""
    import torch
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(value, dict):
        for item in value.values():
            t = _first_tensor(item)
            if t is not None:
                return t
    return None

def _iter_tensors(value: Any):
    """Iterate over all tensors in a nested structure."""
    import torch
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)

def _register_hook(module, name: str, capture: dict, handles: list) -> None:
    """Register a forward hook that stores output in capture[name]."""
    def _hook(_module, _inputs, output):
        capture[name] = output
    handles.append(module.register_forward_hook(_hook))

def _print_shape(label: str, value) -> None:
    """Print shape of first tensor in value."""
    t = _first_tensor(value)
    if t is not None:
        print(f"  {label}: {tuple(t.shape)}")
    else:
        print(f"  {label}: <no tensor>")

def _check_finite(capture: dict, outputs: dict) -> None:
    """Assert all captured intermediates and outputs are finite."""
    import torch
    for name, value in capture.items():
        for t in _iter_tensors(value):
            assert torch.isfinite(t).all(), f"Non-finite in {name}"
    for name, value in outputs.items():
        if value is None:
            continue
        for t in _iter_tensors(value):
            assert torch.isfinite(t).all(), f"Non-finite in output {name}"
    print("All intermediate and final tensors are finite.")
'''


# ---------------------------------------------------------------------------
# Per-model definitions
# ---------------------------------------------------------------------------

def _make_petr_notebook():
    nb = _new_notebook()
    md = _read_markdown("perception/petr")
    sections = _split_markdown_sections(md)

    # --- Preamble ---
    nb.cells.append(code_cell("""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.perception.petr.config import debug_forward_config
        from pytorch_implementation.perception.petr.model import PETRLite

        cfg = debug_forward_config(num_queries=48, decoder_layers=2, depth_num=6)
        model = PETRLite(cfg).eval()

        batch_size = 1
        height, width = 96, 160
        img = torch.randn(batch_size, cfg.num_cams, 3, height, width)

        def _build_dummy_img_metas(batch_size, num_cams, height, width):
            metas = []
            for batch_idx in range(batch_size):
                lidar2img = []
                for cam_idx in range(num_cams):
                    projection = [
                        [1.0, 0.0, 0.0, float(width) * (0.5 + cam_idx * 0.01)],
                        [0.0, 1.0, 0.0, float(height) * 0.5],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                    lidar2img.append(projection)
                metas.append({
                    "sample_idx": f"sample_{batch_idx}",
                    "lidar2img": lidar2img,
                    "img_shape": [(height, width, 3)] * num_cams,
                    "pad_shape": [(height, width, 3)] * num_cams,
                })
            return metas

        img_metas = _build_dummy_img_metas(batch_size, cfg.num_cams, height, width)

        print(f"Model: {type(model).__name__}")
        print(f"Config: {cfg.name}")
        print(f"Input image: {tuple(img.shape)}")
        print(f"embed_dims={cfg.embed_dims}, num_queries={cfg.num_queries}, "
              f"decoder_layers={cfg.num_decoder_layers}, depth_num={cfg.depth_num}")
    """))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    # Collect markdown sections and insert code after each chunk
    i = 0
    while i < len(sections):
        heading, body = sections[i]
        full_md = (heading + "\n" + body).strip() if heading else body.strip()

        chunk_id = _chunk_id_from_heading(heading)

        # For section headers that start chunks, also gather sub-headings
        if heading.startswith("## Chunk") or heading.startswith("## 1)") or heading.startswith("## 2)"):
            sub_parts = [full_md]
            j = i + 1
            while j < len(sections):
                h2, b2 = sections[j]
                if h2.startswith("## "):
                    break
                sub_parts.append((h2 + "\n" + b2).strip())
                j += 1
            nb.cells.append(md_cell("\n\n".join(sub_parts)))
            i = j
        elif heading.startswith("## "):
            # Other top-level sections (dataflow, study drills, etc.)
            sub_parts = [full_md]
            j = i + 1
            while j < len(sections):
                h2, b2 = sections[j]
                if h2.startswith("## "):
                    break
                sub_parts.append((h2 + "\n" + b2).strip())
                j += 1
            nb.cells.append(md_cell("\n\n".join(sub_parts)))
            i = j
        elif heading:
            nb.cells.append(md_cell(full_md))
            i += 1
        else:
            if full_md:
                nb.cells.append(md_cell(full_md))
            i += 1

        # Add code cell after specific chunks
        if chunk_id == 0:
            nb.cells.append(code_cell("""\
                # Chunk 0: End-to-end forward pass
                # Source: pytorch_implementation/perception/petr/model.py - PETRLite.forward

                with torch.no_grad():
                    outputs = model(img, img_metas, decode=False)

                print("=== End-to-end output shapes ===")
                for key, val in outputs.items():
                    if val is not None and torch.is_tensor(val):
                        print(f"  {key}: {tuple(val.shape)}")
                    elif val is None:
                        print(f"  {key}: None")

                assert outputs["all_cls_scores"].shape == (cfg.num_decoder_layers, batch_size, cfg.num_queries, cfg.num_classes)
                assert outputs["all_bbox_preds"].shape == (cfg.num_decoder_layers, batch_size, cfg.num_queries, cfg.code_size)
                print("\\nShape assertions passed.")
            """))
        elif chunk_id == 1:
            nb.cells.append(code_cell("""\
                # Chunk 1: Image features (backbone + neck)
                # Source: pytorch_implementation/perception/petr/model.py - PETRLite.extract_img_feat
                # Source: pytorch_implementation/perception/petr/backbone_neck.py - BackboneNeck

                capture, handles = {}, []

                _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
                for idx, stage in enumerate(model.backbone_neck.backbone.stages):
                    _register_hook(stage, f"backbone.stage{idx}", capture, handles)
                _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)

                with torch.no_grad():
                    img_feats = model.extract_img_feat(img)

                for h in handles:
                    h.remove()

                print("=== Backbone stage shapes (B*Ncam flattened) ===")
                _print_shape("backbone.stem", capture["backbone.stem"])
                for idx in range(4):
                    _print_shape(f"backbone.stage{idx}", capture[f"backbone.stage{idx}"])

                print("\\n=== FPN output ===")
                _print_shape("fpn.output0", capture["fpn.output0"])

                print("\\n=== Camera-major feature (after reshape) ===")
                for lvl, feat in enumerate(img_feats):
                    print(f"  mlvl_feats[{lvl}]: {tuple(feat.shape)}")
                    assert feat.shape[0] == batch_size
                    assert feat.shape[1] == cfg.num_cams
            """))
        elif chunk_id == 2:
            nb.cells.append(code_cell("""\
                # Chunk 2: 3D position embedding from geometry
                # Source: pytorch_implementation/perception/petr/head.py - PETRHeadLite.position_embeding
                # Source: pytorch_implementation/perception/petr/utils.py - inverse_sigmoid, SinePositionalEncoding2D

                capture, handles = {}, []
                _register_hook(model.head.input_proj, "head.input_proj", capture, handles)
                _register_hook(model.head.position_encoder, "head.position_encoder", capture, handles)
                _register_hook(model.head.adapt_pos3d, "head.adapt_pos3d", capture, handles)

                with torch.no_grad():
                    outputs = model(img, img_metas, decode=False)

                for h in handles:
                    h.remove()

                print("=== 3D position embedding shapes ===")
                _print_shape("head.input_proj (B*Ncam, C, Hf, Wf)", capture["head.input_proj"])
                _print_shape("head.position_encoder (B*Ncam, C, Hf, Wf)", capture["head.position_encoder"])
                _print_shape("head.adapt_pos3d (B*Ncam, C, Hf, Wf)", capture["head.adapt_pos3d"])

                # Replay the geometry step standalone
                x = img_feats[0]
                import torch.nn.functional as F
                masks = model.head._build_img_masks(img_metas, batch_size=batch_size,
                                                     num_cams=cfg.num_cams, device=x.device)
                x_proj = model.head.input_proj(x.flatten(0, 1)).view(
                    batch_size, cfg.num_cams, cfg.embed_dims, x.shape[-2], x.shape[-1])
                masks_resized = F.interpolate(masks.float(), size=x_proj.shape[-2:], mode="nearest").to(torch.bool)

                coords_pos, coords_mask = model.head.position_embeding(x_proj, img_metas, masks_resized)
                print(f"\\n=== Standalone geometry replay ===")
                print(f"  coords_position_embedding (P_3d): {tuple(coords_pos.shape)}")
                print(f"  coords_mask: {tuple(coords_mask.shape)}")
            """))
        elif chunk_id == 3:
            nb.cells.append(code_cell("""\
                # Chunk 3: Query construction from 3D reference points
                # Source: pytorch_implementation/perception/petr/head.py - reference_points, query_embedding
                # Source: pytorch_implementation/perception/petr/utils.py - pos2posemb3d

                from pytorch_implementation.perception.petr.utils import pos2posemb3d

                capture, handles = {}, []
                _register_hook(model.head.reference_points, "head.reference_points", capture, handles)
                _register_hook(model.head.query_embedding, "head.query_embedding", capture, handles)

                with torch.no_grad():
                    outputs = model(img, img_metas, decode=False)

                for h in handles:
                    h.remove()

                print("=== Query construction shapes ===")
                _print_shape("reference_points (r_q): [Q, 3]", capture["head.reference_points"])
                _print_shape("query_embedding (e_q): [Q, C]", capture["head.query_embedding"])

                # Replay standalone: show the pos2posemb3d step
                ref_pts = model.head.reference_points(torch.arange(cfg.num_queries, device=img.device))
                print(f"\\n=== Standalone query replay ===")
                print(f"  raw reference_points (learned, before sigmoid): {tuple(ref_pts.shape)}")
                pos_emb_3d = pos2posemb3d(ref_pts, num_pos_feats=cfg.embed_dims // 2)
                print(f"  pos2posemb3d output: {tuple(pos_emb_3d.shape)}")
                query_embeds = model.head.query_embedding(pos_emb_3d)
                print(f"  query_embedding output: {tuple(query_embeds.shape)}")
            """))
        elif chunk_id == 4:
            nb.cells.append(code_cell("""\
                # Chunk 4: Transformer decoder over multi-view memory
                # Source: pytorch_implementation/perception/petr/transformer.py
                #   PETRTransformerLite.forward, PETRTransformerDecoderLayerLite.forward

                capture, handles = {}, []
                for idx, layer in enumerate(model.head.transformer.decoder.layers):
                    _register_hook(layer, f"decoder.layer{idx}", capture, handles)
                    _register_hook(layer.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
                    _register_hook(layer.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
                    _register_hook(layer.ffn, f"decoder.layer{idx}.ffn", capture, handles)

                with torch.no_grad():
                    outputs = model(img, img_metas, decode=False)

                for h in handles:
                    h.remove()

                print("=== Decoder layer shapes ===")
                for idx in range(cfg.num_decoder_layers):
                    print(f"\\n  --- Layer {idx} ---")
                    _print_shape(f"  self_attn  [Q, B, C]", capture[f"decoder.layer{idx}.self_attn"])
                    _print_shape(f"  cross_attn [Q, B, C]", capture[f"decoder.layer{idx}.cross_attn"])
                    _print_shape(f"  ffn        [Q, B, C]", capture[f"decoder.layer{idx}.ffn"])
                    _print_shape(f"  layer_out  [Q, B, C]", capture[f"decoder.layer{idx}"])

                    t = _first_tensor(capture[f"decoder.layer{idx}"])
                    assert t.shape == (cfg.num_queries, batch_size, cfg.embed_dims), \\
                        f"Unexpected decoder.layer{idx} shape: {tuple(t.shape)}"

                print("\\nAll decoder layer shape assertions passed.")
            """))
        elif chunk_id == 5:
            nb.cells.append(code_cell("""\
                # Chunk 5: Class/box heads and metric-space decoding
                # Source: pytorch_implementation/perception/petr/head.py - PETRHeadLite._build_branches, forward
                # Source: pytorch_implementation/perception/petr/postprocess.py - NMSFreeCoderLite

                capture, handles = {}, []
                for idx, branch in enumerate(model.head.cls_branches):
                    _register_hook(branch, f"head.cls_branch{idx}", capture, handles)
                for idx, branch in enumerate(model.head.reg_branches):
                    _register_hook(branch, f"head.reg_branch{idx}", capture, handles)

                with torch.no_grad():
                    outputs = model(img, img_metas, decode=False)

                for h in handles:
                    h.remove()

                print("=== Per-layer branch output shapes ===")
                for idx in range(cfg.num_decoder_layers):
                    _print_shape(f"  cls_branch{idx} [B, Q, num_classes]", capture[f"head.cls_branch{idx}"])
                    _print_shape(f"  reg_branch{idx} [B, Q, code_size]", capture[f"head.reg_branch{idx}"])

                print("\\n=== Stacked outputs ===")
                print(f"  all_cls_scores: {tuple(outputs['all_cls_scores'].shape)}")
                print(f"  all_bbox_preds: {tuple(outputs['all_bbox_preds'].shape)}")

                # Run decode (NMS-free top-k)
                with torch.no_grad():
                    decoded_outputs = model(img, img_metas, decode=True)

                decoded = decoded_outputs["decoded"]
                print(f"\\n=== Decoded predictions (sample 0) ===")
                print(f"  bboxes: {tuple(decoded[0]['bboxes'].shape)}")
                print(f"  scores: {tuple(decoded[0]['scores'].shape)}")
                print(f"  labels: {tuple(decoded[0]['labels'].shape)}")
                print(f"  labels dtype: {decoded[0]['labels'].dtype}")
            """))

    # Final finiteness check cell
    nb.cells.append(code_cell("""\
        # Final validation: all intermediate and output tensors are finite

        capture, handles = {}, []
        _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
        for idx, stage in enumerate(model.backbone_neck.backbone.stages):
            _register_hook(stage, f"backbone.stage{idx}", capture, handles)
        _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)
        _register_hook(model.head.input_proj, "head.input_proj", capture, handles)
        _register_hook(model.head.position_encoder, "head.position_encoder", capture, handles)
        _register_hook(model.head.adapt_pos3d, "head.adapt_pos3d", capture, handles)
        _register_hook(model.head.reference_points, "head.reference_points", capture, handles)
        _register_hook(model.head.query_embedding, "head.query_embedding", capture, handles)
        for idx, layer in enumerate(model.head.transformer.decoder.layers):
            _register_hook(layer, f"decoder.layer{idx}", capture, handles)
            _register_hook(layer.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
            _register_hook(layer.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
            _register_hook(layer.ffn, f"decoder.layer{idx}.ffn", capture, handles)
        for idx, branch in enumerate(model.head.cls_branches):
            _register_hook(branch, f"head.cls_branch{idx}", capture, handles)
        for idx, branch in enumerate(model.head.reg_branches):
            _register_hook(branch, f"head.reg_branch{idx}", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        _check_finite(capture, outputs)
    """))

    return nb


def _make_bevformer_notebook():
    nb = _new_notebook()
    md = _read_markdown("perception/bevformer")

    # --- Preamble ---
    nb.cells.append(code_cell("""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.perception.bevformer.config import debug_forward_config
        from pytorch_implementation.perception.bevformer.model import BEVFormerLite

        cfg = debug_forward_config("tiny", bev_hw=(12, 12), num_queries=48,
                                    encoder_layers=2, decoder_layers=2)
        model = BEVFormerLite(cfg).eval()

        batch_size = 1
        height, width = 96, 160
        img = torch.randn(batch_size, cfg.num_cams, 3, height, width)

        def _build_dummy_img_metas(batch_size, num_cams, height, width, can_bus_dims=18):
            metas = []
            for batch_idx in range(batch_size):
                can_bus = [0.0] * can_bus_dims
                lidar2img = []
                for cam_idx in range(num_cams):
                    projection = [
                        [1.0, 0.0, 0.0, float(width) * (0.5 + cam_idx * 0.005)],
                        [0.0, 1.0, 0.0, float(height) * 0.5],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                    lidar2img.append(projection)
                metas.append({
                    "scene_token": f"scene_{batch_idx}",
                    "can_bus": can_bus,
                    "lidar2img": lidar2img,
                    "img_shape": [(height, width, 3)] * num_cams,
                })
            return metas

        img_metas = _build_dummy_img_metas(batch_size, cfg.num_cams, height, width)

        print(f"Model: {type(model).__name__}")
        print(f"Config: {cfg.name}")
        print(f"Input: {tuple(img.shape)}")
        print(f"embed_dims={cfg.embed_dims}, bev_h={cfg.bev_h}, bev_w={cfg.bev_w}, "
              f"num_queries={cfg.num_queries}")
        print(f"encoder_layers={cfg.num_encoder_layers}, decoder_layers={cfg.num_decoder_layers}")
    """))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    # Parse markdown into sections and group by ## headings
    sections = _split_markdown_sections(md)
    chunk_ids_seen = set()

    i = 0
    while i < len(sections):
        heading, body = sections[i]
        full_md = (heading + "\n" + body).strip() if heading else body.strip()
        chunk_id = _chunk_id_from_heading(heading)

        if heading.startswith("## "):
            sub_parts = [full_md]
            j = i + 1
            while j < len(sections):
                h2, b2 = sections[j]
                if h2.startswith("## "):
                    break
                sub_parts.append((h2 + "\n" + b2).strip())
                j += 1
            nb.cells.append(md_cell("\n\n".join(sub_parts)))
            i = j
        else:
            if full_md:
                nb.cells.append(md_cell(full_md))
            i += 1

        if chunk_id is not None and chunk_id not in chunk_ids_seen:
            chunk_ids_seen.add(chunk_id)
            if chunk_id == 0:
                nb.cells.append(code_cell("""\
                    # Chunk 0: End-to-end forward pass
                    # Source: pytorch_implementation/perception/bevformer/model.py - BEVFormerLite.forward

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)

                    print("=== End-to-end output shapes ===")
                    for key, val in outputs.items():
                        if val is not None and torch.is_tensor(val):
                            print(f"  {key}: {tuple(val.shape)}")
                        elif val is None:
                            print(f"  {key}: None")

                    # Test with prev_bev=None (temporal branch degenerates)
                    with torch.no_grad():
                        outputs_no_prev = model(img, img_metas, decode=False, prev_bev=None)
                    print("\\nWith prev_bev=None, model still runs:")
                    print(f"  bev_embed: {tuple(outputs_no_prev['bev_embed'].shape)}")
                """))
            elif chunk_id == 1:
                nb.cells.append(code_cell("""\
                    # Chunk 1: Image features (backbone + neck)
                    # Source: pytorch_implementation/perception/bevformer/model.py - extract_img_feat
                    # Source: pytorch_implementation/perception/bevformer/backbone_neck.py

                    capture, handles = {}, []
                    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
                    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
                        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
                    for idx, conv in enumerate(model.backbone_neck.neck.output_convs):
                        _register_hook(conv, f"fpn.output{idx}", capture, handles)

                    with torch.no_grad():
                        img_feats = model.extract_img_feat(img)

                    for h in handles:
                        h.remove()

                    print("=== Backbone stages (B*Ncam flattened) ===")
                    _print_shape("stem", capture["backbone.stem"])
                    for idx in range(4):
                        _print_shape(f"stage{idx}", capture[f"backbone.stage{idx}"])

                    print("\\n=== FPN outputs ===")
                    for key in sorted(k for k in capture if k.startswith("fpn.")):
                        _print_shape(key, capture[key])

                    print("\\n=== Camera-major features ===")
                    for lvl, feat in enumerate(img_feats):
                        print(f"  mlvl_feats[{lvl}]: {tuple(feat.shape)}")
                """))
            elif chunk_id == 2:
                nb.cells.append(code_cell("""\
                    # Chunk 2: BEV query initialization
                    # Source: pytorch_implementation/perception/bevformer/head.py - BEVFormerHeadLite.forward
                    # Source: pytorch_implementation/perception/bevformer/utils/positional_encoding.py

                    print("=== BEV query shapes ===")
                    print(f"  bev_embedding.weight: {tuple(model.head.bev_embedding.weight.shape)}")
                    print(f"    = [bev_h*bev_w, embed_dims] = [{cfg.bev_h*cfg.bev_w}, {cfg.embed_dims}]")
                    print(f"  query_embedding.weight: {tuple(model.head.query_embedding.weight.shape)}")
                    print(f"    = [num_queries, 2*embed_dims] = [{cfg.num_queries}, {2*cfg.embed_dims}]")

                    bev_mask = torch.zeros(batch_size, cfg.bev_h, cfg.bev_w, dtype=torch.bool)
                    bev_pos = model.head.positional_encoding(bev_mask)
                    print(f"\\n  positional_encoding output: {tuple(bev_pos.shape)}")
                    print(f"    = [B, C, bev_h, bev_w]")

                    can_bus_tensor = torch.tensor(img_metas[0]["can_bus"], dtype=torch.float32).unsqueeze(0)
                    can_bus_emb = model.head.can_bus_mlp(can_bus_tensor)
                    print(f"  can_bus MLP output: {tuple(can_bus_emb.shape)}")
                """))
            elif chunk_id == 3:
                nb.cells.append(code_cell("""\
                    # Chunk 3: Geometry and projection bridge
                    # Source: pytorch_implementation/perception/bevformer/utils/geometry.py
                    #   get_reference_points_3d, get_reference_points_2d, point_sampling

                    from pytorch_implementation.perception.bevformer.utils.geometry import (
                        get_reference_points_3d, get_reference_points_2d, point_sampling,
                    )

                    device = img.device
                    ref_3d = get_reference_points_3d(
                        cfg.bev_h, cfg.bev_w, cfg.num_points_in_pillar, device=device,
                        pc_range=cfg.pc_range, dtype=torch.float32
                    )
                    ref_2d = get_reference_points_2d(cfg.bev_h, cfg.bev_w, device=device, dtype=torch.float32)

                    print("=== Reference points ===")
                    print(f"  ref_3d (pillar anchors): {tuple(ref_3d.shape)}")
                    print(f"    = [B, D={cfg.num_points_in_pillar}, HW={cfg.bev_h*cfg.bev_w}, 3]")
                    print(f"  ref_2d (BEV 2D):         {tuple(ref_2d.shape)}")
                    print(f"    = [B, HW, 1, 2]")

                    lidar2img_tensors = []
                    for meta in img_metas:
                        mats = [torch.tensor(m, dtype=torch.float32) for m in meta["lidar2img"]]
                        lidar2img_tensors.append(torch.stack(mats))
                    lidar2img = torch.stack(lidar2img_tensors)
                    ref_pts_cam, bev_mask_cam = point_sampling(
                        ref_3d, lidar2img=lidar2img, pc_range=cfg.pc_range,
                        img_metas=img_metas, device=device
                    )
                    print(f"\\n  reference_points_cam: {tuple(ref_pts_cam.shape)}")
                    print(f"    = [Ncam, B, HW, D, 2]")
                    print(f"  bev_mask: {tuple(bev_mask_cam.shape)}")
                    print(f"    = [Ncam, B, HW, D]")
                    print(f"  fraction of valid projections: {bev_mask_cam.float().mean():.3f}")
                """))
            elif chunk_id == 4:
                nb.cells.append(code_cell("""\
                    # Chunk 4: Temporal self-attention (TSA)
                    # Source: pytorch_implementation/perception/bevformer/modules/temporal_self_attention.py
                    # Source: pytorch_implementation/perception/bevformer/modules/encoder.py

                    capture, handles = {}, []
                    for idx in range(cfg.num_encoder_layers):
                        _register_hook(model.head.transformer.encoder.layers[idx],
                                       f"encoder.layer{idx}", capture, handles)
                        _register_hook(model.head.transformer.encoder.layers[idx].temporal_attn,
                                       f"encoder.layer{idx}.temporal", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)

                    for h in handles:
                        h.remove()

                    print("=== Temporal self-attention outputs ===")
                    for idx in range(cfg.num_encoder_layers):
                        _print_shape(f"encoder.layer{idx}.temporal [B, HW, C]",
                                     capture[f"encoder.layer{idx}.temporal"])
                        _print_shape(f"encoder.layer{idx} [B, HW, C]",
                                     capture[f"encoder.layer{idx}"])
                """))
            elif chunk_id == 5:
                nb.cells.append(code_cell("""\
                    # Chunk 5: Spatial cross-attention (SCA)
                    # Source: pytorch_implementation/perception/bevformer/modules/spatial_cross_attention.py
                    # Source: pytorch_implementation/perception/bevformer/modules/deformable_attention.py

                    capture, handles = {}, []
                    for idx in range(cfg.num_encoder_layers):
                        _register_hook(model.head.transformer.encoder.layers[idx].spatial_attn,
                                       f"encoder.layer{idx}.spatial", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)

                    for h in handles:
                        h.remove()

                    print("=== Spatial cross-attention outputs ===")
                    for idx in range(cfg.num_encoder_layers):
                        _print_shape(f"encoder.layer{idx}.spatial [B, HW, C]",
                                     capture[f"encoder.layer{idx}.spatial"])
                """))
            elif chunk_id == 6:
                nb.cells.append(code_cell("""\
                    # Chunk 6: Encoder recurrence result
                    # Source: pytorch_implementation/perception/bevformer/modules/encoder.py

                    capture, handles = {}, []
                    for idx in range(cfg.num_encoder_layers):
                        _register_hook(model.head.transformer.encoder.layers[idx],
                                       f"encoder.layer{idx}", capture, handles)
                        _register_hook(model.head.transformer.encoder.layers[idx].ffn,
                                       f"encoder.layer{idx}.ffn", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)

                    for h in handles:
                        h.remove()

                    print("=== Encoder layer output shapes ===")
                    for idx in range(cfg.num_encoder_layers):
                        _print_shape(f"encoder.layer{idx}.ffn  [B, HW, C]",
                                     capture[f"encoder.layer{idx}.ffn"])
                        _print_shape(f"encoder.layer{idx} (full) [B, HW, C]",
                                     capture[f"encoder.layer{idx}"])

                    print(f"\\n=== Final BEV embed (encoder output) ===")
                    print(f"  bev_embed: {tuple(outputs['bev_embed'].shape)}")
                    print(f"    = [HW={cfg.bev_h*cfg.bev_w}, B, C]")
                """))
            elif chunk_id == 7:
                nb.cells.append(code_cell("""\
                    # Chunk 7: Decoder and iterative reference refinement
                    # Source: pytorch_implementation/perception/bevformer/modules/decoder.py
                    # Source: pytorch_implementation/perception/bevformer/head.py

                    capture, handles = {}, []
                    for idx in range(cfg.num_decoder_layers):
                        dec = model.head.transformer.decoder.layers[idx]
                        _register_hook(dec, f"decoder.layer{idx}", capture, handles)
                        _register_hook(dec.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
                        _register_hook(dec.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
                        _register_hook(dec.ffn, f"decoder.layer{idx}.ffn", capture, handles)
                    for idx, branch in enumerate(model.head.cls_branches):
                        _register_hook(branch, f"head.cls_branch{idx}", capture, handles)
                    for idx, branch in enumerate(model.head.reg_branches):
                        _register_hook(branch, f"head.reg_branch{idx}", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)

                    for h in handles:
                        h.remove()

                    print("=== Decoder layer shapes ===")
                    for idx in range(cfg.num_decoder_layers):
                        _print_shape(f"decoder.layer{idx} [Q, B, C]",
                                     capture[f"decoder.layer{idx}"])
                    print("\\n=== Head branch shapes ===")
                    for idx in range(cfg.num_decoder_layers):
                        _print_shape(f"cls_branch{idx} [B, Q, num_classes]",
                                     capture[f"head.cls_branch{idx}"])
                        _print_shape(f"reg_branch{idx} [B, Q, code_size]",
                                     capture[f"head.reg_branch{idx}"])

                    print(f"\\n=== Stacked outputs ===")
                    print(f"  all_cls_scores: {tuple(outputs['all_cls_scores'].shape)}")
                    print(f"  all_bbox_preds: {tuple(outputs['all_bbox_preds'].shape)}")
                """))
            elif chunk_id == 8:
                nb.cells.append(code_cell("""\
                    # Chunk 8: Box parameterization and decode
                    # Source: pytorch_implementation/perception/bevformer/postprocess/nms_free_coder.py
                    # Source: pytorch_implementation/perception/bevformer/utils/boxes.py

                    with torch.no_grad():
                        decoded_outputs = model(img, img_metas, decode=True)

                    preds = decoded_outputs["preds"]
                    decoded = decoded_outputs["decoded"]

                    print("=== Raw predictions (last decoder layer) ===")
                    print(f"  cls_scores[-1]: {tuple(preds['all_cls_scores'][-1].shape)}")
                    print(f"  bbox_preds[-1]: {tuple(preds['all_bbox_preds'][-1].shape)}")

                    print(f"\\n=== Decoded predictions (sample 0) ===")
                    for key, val in decoded[0].items():
                        print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
                """))
            elif chunk_id == 9:
                nb.cells.append(code_cell("""\
                    # Chunk 9: End-to-end trace validation
                    # Full pipeline: img -> backbone -> BEV encoder -> decoder -> decode

                    capture, handles = {}, []
                    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
                    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
                        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
                    for idx, conv in enumerate(model.backbone_neck.neck.output_convs):
                        _register_hook(conv, f"fpn.output{idx}", capture, handles)
                    for idx in range(cfg.num_encoder_layers):
                        enc = model.head.transformer.encoder.layers[idx]
                        _register_hook(enc, f"encoder.layer{idx}", capture, handles)
                        _register_hook(enc.temporal_attn, f"encoder.layer{idx}.temporal", capture, handles)
                        _register_hook(enc.spatial_attn, f"encoder.layer{idx}.spatial", capture, handles)
                        _register_hook(enc.ffn, f"encoder.layer{idx}.ffn", capture, handles)
                    for idx in range(cfg.num_decoder_layers):
                        dec = model.head.transformer.decoder.layers[idx]
                        _register_hook(dec, f"decoder.layer{idx}", capture, handles)
                    for idx, branch in enumerate(model.head.cls_branches):
                        _register_hook(branch, f"head.cls_branch{idx}", capture, handles)
                    for idx, branch in enumerate(model.head.reg_branches):
                        _register_hook(branch, f"head.reg_branch{idx}", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)

                    for h in handles:
                        h.remove()

                    _check_finite(capture, outputs)
                    print(f"\\nTotal hooks captured: {len(capture)}")
                    print(f"Captured names: {sorted(capture.keys())}")
                """))

    return nb


def _make_maptr_notebook():
    nb = _new_notebook()
    md = _read_markdown("perception/maptr")

    nb.cells.append(code_cell("""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.perception.maptr.config import debug_forward_config
        from pytorch_implementation.perception.maptr.model import MapTRLite

        cfg = debug_forward_config(num_vec=10, num_pts_per_vec=4, decoder_layers=2)
        model = MapTRLite(cfg).eval()

        batch_size = 1
        height, width = 96, 160
        img = torch.randn(batch_size, cfg.num_cams, 3, height, width)

        def _build_dummy_img_metas(batch_size, num_cams, height, width):
            metas = []
            for batch_idx in range(batch_size):
                metas.append({
                    "sample_idx": f"sample_{batch_idx}",
                    "img_shape": [(height, width, 3)] * num_cams,
                    "pad_shape": [(height, width, 3)] * num_cams,
                })
            return metas

        img_metas = _build_dummy_img_metas(batch_size, cfg.num_cams, height, width)

        print(f"Model: {type(model).__name__}")
        print(f"Config: {cfg.name}")
        print(f"Input: {tuple(img.shape)}")
        print(f"embed_dims={cfg.embed_dims}, num_vec={cfg.num_vec}, "
              f"num_pts_per_vec={cfg.num_pts_per_vec}, "
              f"decoder_layers={cfg.num_decoder_layers}")
    """))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    sections = _split_markdown_sections(md)
    chunk_ids_seen = set()
    i = 0
    while i < len(sections):
        heading, body = sections[i]
        full_md = (heading + "\n" + body).strip() if heading else body.strip()
        chunk_id = _chunk_id_from_heading(heading)
        if heading.startswith("## "):
            sub_parts = [full_md]
            j = i + 1
            while j < len(sections):
                h2, b2 = sections[j]
                if h2.startswith("## "):
                    break
                sub_parts.append((h2 + "\n" + b2).strip())
                j += 1
            nb.cells.append(md_cell("\n\n".join(sub_parts)))
            i = j
        else:
            if full_md:
                nb.cells.append(md_cell(full_md))
            i += 1

        if chunk_id is not None and chunk_id not in chunk_ids_seen:
            chunk_ids_seen.add(chunk_id)
            if chunk_id == 0:
                nb.cells.append(code_cell("""\
                    # Chunk 0: End-to-end forward
                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)
                    print("=== MapTR output shapes ===")
                    for key, val in outputs.items():
                        if val is not None and torch.is_tensor(val):
                            print(f"  {key}: {tuple(val.shape)}")
                """))
            elif chunk_id == 1:
                nb.cells.append(code_cell("""\
                    # Chunk 1: Hierarchical query embeddings
                    # Source: pytorch_implementation/perception/maptr/head.py

                    print("=== Query embedding shapes ===")
                    print(f"  instance_embedding.weight: {tuple(model.head.instance_embedding.weight.shape)}")
                    print(f"    = [num_vec={cfg.num_vec}, C={cfg.embed_dims}]")
                    print(f"  pts_embedding.weight: {tuple(model.head.pts_embedding.weight.shape)}")
                    print(f"    = [num_pts_per_vec={cfg.num_pts_per_vec}, C={cfg.embed_dims}]")

                    capture, handles = {}, []
                    _register_hook(model.head.instance_embedding, "head.instance_embedding", capture, handles)
                    _register_hook(model.head.pts_embedding, "head.pts_embedding", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)
                    for h in handles:
                        h.remove()

                    _print_shape("instance_embedding output", capture["head.instance_embedding"])
                    _print_shape("pts_embedding output", capture["head.pts_embedding"])
                """))
            elif chunk_id == 2:
                nb.cells.append(code_cell("""\
                    # Chunk 2: BEV token construction
                    # Source: pytorch_implementation/perception/maptr/transformer.py - MapTRBEVEncoderLite

                    capture, handles = {}, []
                    _register_hook(model.head.bev_embedding, "head.bev_embedding", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)
                    for h in handles:
                        h.remove()

                    print("=== BEV construction shapes ===")
                    _print_shape("bev_embedding", capture["head.bev_embedding"])
                    print(f"  bev_embed output: {tuple(outputs['bev_embed'].shape)}")
                    print(f"    = [B, bev_h*bev_w={cfg.bev_h*cfg.bev_w}, C={cfg.embed_dims}]")
                """))
            elif chunk_id == 3:
                nb.cells.append(code_cell("""\
                    # Chunk 3: Decoder and vectorized predictions
                    # Source: pytorch_implementation/perception/maptr/transformer.py, head.py

                    capture, handles = {}, []
                    for idx in range(cfg.num_decoder_layers):
                        dec = model.head.transformer.decoder.layers[idx]
                        _register_hook(dec, f"decoder.layer{idx}", capture, handles)
                        _register_hook(dec.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
                        _register_hook(dec.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
                        _register_hook(dec.ffn, f"decoder.layer{idx}.ffn", capture, handles)
                    for idx in range(cfg.num_decoder_layers):
                        _register_hook(model.head.cls_branches[idx], f"head.cls_branch{idx}", capture, handles)
                        _register_hook(model.head.reg_branches[idx], f"head.reg_branch{idx}", capture, handles)

                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)
                    for h in handles:
                        h.remove()

                    print("=== Decoder layer shapes ===")
                    for idx in range(cfg.num_decoder_layers):
                        _print_shape(f"decoder.layer{idx} [Q, B, C]", capture[f"decoder.layer{idx}"])

                    print("\\n=== Branch shapes ===")
                    for idx in range(cfg.num_decoder_layers):
                        _print_shape(f"cls_branch{idx}", capture[f"head.cls_branch{idx}"])
                        _print_shape(f"reg_branch{idx}", capture[f"head.reg_branch{idx}"])

                    print("\\n=== Stacked outputs ===")
                    print(f"  all_cls_scores: {tuple(outputs['all_cls_scores'].shape)}")
                    print(f"  all_bbox_preds: {tuple(outputs['all_bbox_preds'].shape)}")
                    if 'all_pts_preds' in outputs and outputs['all_pts_preds'] is not None:
                        print(f"  all_pts_preds:  {tuple(outputs['all_pts_preds'].shape)}")
                """))
            elif chunk_id == 4:
                nb.cells.append(code_cell("""\
                    # Chunk 4: Postprocess and metric-space decoding

                    with torch.no_grad():
                        decoded_out = model(img, img_metas, decode=True)
                    decoded = decoded_out["decoded"]
                    print("=== Decoded predictions (sample 0) ===")
                    for key, val in decoded[0].items():
                        if torch.is_tensor(val):
                            print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")

                    # Full finiteness check
                    capture, handles = {}, []
                    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
                    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
                        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)
                    for h in handles:
                        h.remove()
                    _check_finite(capture, outputs)
                """))
    return nb


def _make_simple_notebook(
    model_key: str,
    import_module: str,
    model_class: str,
    config_call: str,
    metas_fn: str,
    batch_size: int,
    extra_preamble: str = "",
    chunk_code=None,
) -> nbf.NotebookNode:
    """Generic notebook builder for models with a common structure."""
    nb = _new_notebook()
    md = _read_markdown(model_key)

    preamble = f"""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.{import_module}.config import debug_forward_config
        from pytorch_implementation.{import_module}.model import {model_class}

        cfg = {config_call}
        model = {model_class}(cfg).eval()

        batch_size = {batch_size}
        height, width = 96, 160
        img = torch.randn(batch_size, cfg.num_cams, 3, height, width)

        {metas_fn}
        {extra_preamble}
        print(f"Model: {{type(model).__name__}}")
        print(f"Config: {{cfg.name}}")
        print(f"Input: {{tuple(img.shape)}}")
    """
    nb.cells.append(code_cell(preamble))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    sections = _split_markdown_sections(md)
    chunk_ids_seen = set()
    i = 0
    while i < len(sections):
        heading, body = sections[i]
        full_md = (heading + "\n" + body).strip() if heading else body.strip()
        chunk_id = _chunk_id_from_heading(heading)
        if heading.startswith("## "):
            sub_parts = [full_md]
            j = i + 1
            while j < len(sections):
                h2, b2 = sections[j]
                if h2.startswith("## "):
                    break
                sub_parts.append((h2 + "\n" + b2).strip())
                j += 1
            nb.cells.append(md_cell("\n\n".join(sub_parts)))
            i = j
        else:
            if full_md:
                nb.cells.append(md_cell(full_md))
            i += 1

        if chunk_id is not None and chunk_id not in chunk_ids_seen:
            chunk_ids_seen.add(chunk_id)
            if chunk_code and chunk_id in chunk_code:
                nb.cells.append(code_cell(chunk_code[chunk_id]))
            else:
                # Generic end-to-end code for chunks without custom code
                nb.cells.append(code_cell(f"""\
                    # Chunk {chunk_id}: see markdown above for paper mapping
                    capture, handles = {{}}, []
                    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
                    with torch.no_grad():
                        outputs = model(img, img_metas, decode=False)
                    for h in handles:
                        h.remove()
                    print("=== Output shapes ===")
                    for key, val in outputs.items():
                        if val is not None and torch.is_tensor(val):
                            print(f"  {{key}}: {{tuple(val.shape)}}")
                """))

    # Final check
    nb.cells.append(code_cell("""\
        # Final finiteness validation
        capture, handles = {}, []
        _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
        for idx, stage in enumerate(model.backbone_neck.backbone.stages):
            _register_hook(stage, f"backbone.stage{idx}", capture, handles)
        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()
        _check_finite(capture, outputs)
    """))
    return nb


def _append_markdown_sections_with_chunks(
    nb: nbf.NotebookNode,
    md: str,
    chunk_code: dict[int, str],
) -> None:
    """Append grouped markdown sections and chunk-driven code cells."""
    sections = _split_markdown_sections(md)
    chunk_ids_seen = set()
    i = 0
    while i < len(sections):
        heading, body = sections[i]
        full_md = (heading + "\n" + body).strip() if heading else body.strip()
        chunk_id = _chunk_id_from_heading(heading)

        if heading.startswith("## "):
            sub_parts = [full_md]
            j = i + 1
            while j < len(sections):
                h2, b2 = sections[j]
                if h2.startswith("## "):
                    break
                sub_parts.append((h2 + "\n" + b2).strip())
                j += 1
            nb.cells.append(md_cell("\n\n".join(sub_parts)))
            i = j
        else:
            if full_md:
                nb.cells.append(md_cell(full_md))
            i += 1

        if chunk_id is not None and chunk_id not in chunk_ids_seen:
            chunk_ids_seen.add(chunk_id)
            if chunk_id in chunk_code:
                nb.cells.append(code_cell(chunk_code[chunk_id]))


def _build_final_finite_check_cell(
    *,
    hook_lines: list[str],
    forward_lines: list[str],
    title: str = "Final finite check with major hooks",
) -> str:
    hook_block = "\n".join(f"        {line}" for line in hook_lines)
    forward_block = "\n".join(f"            {line}" for line in forward_lines)
    return textwrap.dedent(
        f"""\
        # {title}
        capture, handles = {{}}, []
{hook_block}

        with torch.no_grad():
{forward_block}
        for h in handles:
            h.remove()

        _check_finite(capture, outputs)
    """
    ).strip()


# ---------------------------------------------------------------------------
# FB-BEV
# ---------------------------------------------------------------------------

FBBEV_METAS = '''\
def _build_dummy_img_metas(batch_size):
            metas = []
            for batch_idx in range(batch_size):
                curr_to_prev = torch.eye(4, dtype=torch.float32)
                curr_to_prev[0, 3] = 0.4 * (batch_idx + 1)
                curr_to_prev[1, 3] = -0.2 * (batch_idx + 1)
                metas.append({
                    "sample_idx": f"sample_{batch_idx}",
                    "sequence_group_idx": batch_idx,
                    "start_of_sequence": True,
                    "curr_to_prev_ego_rt": curr_to_prev,
                })
            return metas
        img_metas = _build_dummy_img_metas(batch_size)
'''

FBBEV_CHUNKS = {
    0: """\
        # Chunk 0: End-to-end forward
        # Source: pytorch_implementation/perception/fbbev/model.py - FBBEVLite.forward

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)

        print("=== FB-BEV output shapes ===")
        for key, val in outputs.items():
            if val is not None and torch.is_tensor(val):
                print(f"  {key}: {tuple(val.shape)}")

        assert outputs["all_cls_scores"].shape[0] == 1  # single head layer
        print("\\nShape assertions passed.")
    """,
    1: """\
        # Chunk 1: Camera projection and depth weighting
        # Source: pytorch_implementation/perception/fbbev/depth_net.py
        # Source: pytorch_implementation/perception/fbbev/forward_projection.py

        capture, handles = {}, []
        _register_hook(model.depth_net.context_proj, "depth_net.context_proj", capture, handles)
        _register_hook(model.depth_net.depth_logits, "depth_net.depth_logits", capture, handles)
        _register_hook(model.forward_projection, "forward_projection", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Depth net and forward projection ===")
        _print_shape("depth_net.context_proj", capture["depth_net.context_proj"])
        _print_shape("depth_net.depth_logits", capture["depth_net.depth_logits"])
        _print_shape("forward_projection (bev_volume)", capture["forward_projection"])
        print(f"\\n  context output: {tuple(outputs['context'].shape)}")
        print(f"  depth output:   {tuple(outputs['depth'].shape)}")
        print(f"  bev_volume:     {tuple(outputs['bev_volume'].shape)}")
    """,
    2: """\
        # Chunk 2: Depth-aware backward projection
        # Source: pytorch_implementation/perception/fbbev/backward_projection.py
        # Source: pytorch_implementation/perception/fbbev/depth_aware_attention.py

        capture, handles = {}, []
        _register_hook(model.backward_projection.depth_attention,
                       "backward.depth_attention", capture, handles)
        _register_hook(model.backward_projection.post_conv,
                       "backward.post", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Backward projection shapes ===")
        _print_shape("depth_attention", capture["backward.depth_attention"])
        _print_shape("post_conv (bev_refined)", capture["backward.post"])
        print(f"\\n  bev_refined: {tuple(outputs['bev_refined'].shape)}")
    """,
    3: """\
        # Chunk 3: Temporal fusion with ego-motion alignment
        # Source: pytorch_implementation/perception/fbbev/temporal_fusion.py

        capture, handles = {}, []
        _register_hook(model.temporal_fusion.time_conv, "temporal.time_conv", capture, handles)
        _register_hook(model.temporal_fusion.cat_conv, "temporal.cat_conv", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Temporal fusion shapes ===")
        _print_shape("time_conv", capture["temporal.time_conv"])
        _print_shape("cat_conv (fused)", capture["temporal.cat_conv"])
        print(f"\\n  bev_fused: {tuple(outputs['bev_fused'].shape)}")
    """,
    4: """\
        # Chunk 4: Detection outputs and decoded 3D boxes
        # Source: pytorch_implementation/perception/fbbev/detection_head.py
        # Source: pytorch_implementation/perception/fbbev/postprocess.py

        capture, handles = {}, []
        _register_hook(model.detection_head.shared_conv, "head.shared", capture, handles)
        _register_hook(model.detection_head.heatmap_head, "head.heatmap", capture, handles)
        _register_hook(model.detection_head.reg_head, "head.reg", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Detection head shapes ===")
        _print_shape("shared_conv", capture["head.shared"])
        _print_shape("heatmap_head", capture["head.heatmap"])
        _print_shape("reg_head", capture["head.reg"])

        print(f"\\n=== Final outputs ===")
        print(f"  all_cls_scores: {tuple(outputs['all_cls_scores'].shape)}")
        print(f"  all_bbox_preds: {tuple(outputs['all_bbox_preds'].shape)}")

        with torch.no_grad():
            decoded_out = model(img, img_metas, decode=True)
        decoded = decoded_out["decoded"]
        print(f"\\n=== Decoded (sample 0) ===")
        for key, val in decoded[0].items():
            if torch.is_tensor(val):
                print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
    """,
}


# ---------------------------------------------------------------------------
# PolarFormer
# ---------------------------------------------------------------------------

POLAR_METAS = '''\
def _build_dummy_img_metas(batch_size, num_cams, height, width):
            metas = []
            for batch_idx in range(batch_size):
                lidar2img, cam_intrinsic, cam2lidar = [], [], []
                for cam_idx in range(num_cams):
                    lidar2img.append([
                        [1.0, 0.0, 0.0, float(width) * (0.5 + cam_idx * 0.01)],
                        [0.0, 1.0, 0.0, float(height) * 0.5],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                    cam_intrinsic.append([
                        [float(width), 0.0, float(width) * 0.5],
                        [0.0, float(height), float(height) * 0.5],
                        [0.0, 0.0, 1.0],
                    ])
                    cam2lidar.append([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                metas.append({
                    "sample_idx": f"sample_{batch_idx}",
                    "lidar2img": lidar2img, "cam_intrinsic": cam_intrinsic,
                    "cam2lidar": cam2lidar,
                    "img_shape": [(height, width, 3)] * num_cams,
                    "pad_shape": [(height, width, 3)] * num_cams,
                })
            return metas
        img_metas = _build_dummy_img_metas(batch_size, cfg.num_cams, height, width)
'''

POLAR_CHUNKS = {
    0: """\
        # Chunk 0: End-to-end forward
        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        print("=== PolarFormer output shapes ===")
        for key, val in outputs.items():
            if val is not None and torch.is_tensor(val):
                print(f"  {key}: {tuple(val.shape)}")
    """,
    1: """\
        # Chunk 1: Camera features to polar rays
        # Source: pytorch_implementation/perception/polarformer/backbone_neck.py

        capture, handles = {}, []
        _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
        for idx, stage in enumerate(model.backbone_neck.backbone.stages):
            _register_hook(stage, f"backbone.stage{idx}", capture, handles)
        for idx, conv in enumerate(model.backbone_neck.neck.output_convs):
            _register_hook(conv, f"fpn.output{idx}", capture, handles)

        with torch.no_grad():
            mlvl_feats = model.extract_img_feat(img, img_metas)
        for h in handles:
            h.remove()

        print("=== Backbone/FPN ===")
        for key in sorted(capture.keys()):
            _print_shape(key, capture[key])
        print(f"\\n=== Polar features (after projection) ===")
        for lvl, feat in enumerate(mlvl_feats):
            print(f"  mlvl_feats[{lvl}]: {tuple(feat.shape)}")
    """,
    2: """\
        # Chunk 2: Multi-level polar memory + decoder
        # Source: pytorch_implementation/perception/polarformer/transformer.py

        capture, handles = {}, []
        for idx in range(cfg.num_decoder_layers):
            dec = model.head.transformer.decoder.layers[idx]
            _register_hook(dec, f"decoder.layer{idx}", capture, handles)
            _register_hook(dec.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
            _register_hook(dec.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
            _register_hook(dec.ffn, f"decoder.layer{idx}.ffn", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Decoder layer shapes ===")
        for idx in range(cfg.num_decoder_layers):
            _print_shape(f"decoder.layer{idx} [Q, B, C]", capture[f"decoder.layer{idx}"])
            _print_shape(f"  self_attn", capture[f"decoder.layer{idx}.self_attn"])
            _print_shape(f"  cross_attn", capture[f"decoder.layer{idx}.cross_attn"])
            _print_shape(f"  ffn", capture[f"decoder.layer{idx}.ffn"])
    """,
    3: """\
        # Chunk 3: Polar box parameters to Cartesian center
        # Source: pytorch_implementation/perception/polarformer/head.py, postprocess.py

        capture, handles = {}, []
        for idx in range(cfg.num_decoder_layers):
            _register_hook(model.head.cls_branches[idx], f"head.cls_branch{idx}", capture, handles)
            _register_hook(model.head.reg_branches[idx], f"head.reg_branch{idx}", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Head branch shapes ===")
        for idx in range(cfg.num_decoder_layers):
            _print_shape(f"cls_branch{idx}", capture[f"head.cls_branch{idx}"])
            _print_shape(f"reg_branch{idx}", capture[f"head.reg_branch{idx}"])

        print(f"\\n=== Stacked outputs ===")
        print(f"  all_cls_scores: {tuple(outputs['all_cls_scores'].shape)}")
        print(f"  all_bbox_preds: {tuple(outputs['all_bbox_preds'].shape)}")

        with torch.no_grad():
            decoded_out = model(img, img_metas, decode=True)
        decoded = decoded_out["decoded"]
        print(f"\\n=== Decoded (sample 0) ===")
        for key, val in decoded[0].items():
            if torch.is_tensor(val):
                print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
    """,
}


# ---------------------------------------------------------------------------
# Sparse4D
# ---------------------------------------------------------------------------

SPARSE4D_METAS = '''\
def _build_dummy_metas(batch_size, num_cams, height, width):
            projection_mat = torch.eye(4, dtype=torch.float32).view(1,1,4,4).repeat(batch_size, num_cams, 1, 1)
            for cam_idx in range(num_cams):
                projection_mat[:, cam_idx, 0, 3] = width * (0.45 + 0.02 * cam_idx)
                projection_mat[:, cam_idx, 1, 3] = height * 0.5
            image_wh = torch.tensor([width, height], dtype=torch.float32).view(1,1,2).repeat(batch_size, num_cams, 1)
            return {"projection_mat": projection_mat, "image_wh": image_wh}
        img_metas = _build_dummy_metas(batch_size, cfg.num_cams, height, width)
'''

SPARSE4D_CHUNKS = {
    0: """\
        # Chunk 0: End-to-end forward
        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        print("=== Sparse4D output shapes ===")
        for key, val in outputs.items():
            if val is not None and torch.is_tensor(val):
                print(f"  {key}: {tuple(val.shape)}")
    """,
    1: """\
        # Chunk 1: Image feature extraction
        # Source: pytorch_implementation/perception/sparse4d/backbone_neck.py, model.py

        capture, handles = {}, []
        _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
        for idx, stage in enumerate(model.backbone_neck.backbone.stages):
            _register_hook(stage, f"backbone.stage{idx}", capture, handles)
        for idx, conv in enumerate(model.backbone_neck.neck.output_convs):
            _register_hook(conv, f"neck.output{idx}", capture, handles)

        with torch.no_grad():
            img_feats = model.extract_img_feat(img)
        for h in handles:
            h.remove()

        print("=== Backbone/neck shapes ===")
        for key in sorted(capture.keys()):
            _print_shape(key, capture[key])
        print(f"\\n=== Multi-level features ===")
        for lvl, feat in enumerate(img_feats):
            print(f"  mlvl_feats[{lvl}]: {tuple(feat.shape)}")
    """,
    2: """\
        # Chunk 2: Instance bank and anchor encoding
        # Source: pytorch_implementation/perception/sparse4d/instance_bank.py, blocks.py

        capture, handles = {}, []
        _register_hook(model.head.instance_bank, "head.instance_bank", capture, handles)
        _register_hook(model.head.anchor_encoder, "head.anchor_encoder", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Instance bank and anchor encoder ===")
        _print_shape("instance_bank", capture["head.instance_bank"])
        _print_shape("anchor_encoder", capture["head.anchor_encoder"])
    """,
    3: """\
        # Chunk 3: Decoder updates with image aggregation
        # Source: pytorch_implementation/perception/sparse4d/decoder.py, blocks.py

        capture, handles = {}, []
        for idx in range(cfg.num_decoder_layers):
            dec = model.head.decoder.layers[idx]
            _register_hook(dec, f"decoder.layer{idx}", capture, handles)
            _register_hook(dec.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
            _register_hook(dec.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
            _register_hook(dec.ffn, f"decoder.layer{idx}.ffn", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Decoder layer shapes ===")
        for idx in range(cfg.num_decoder_layers):
            print(f"\\n  --- Layer {idx} ---")
            _print_shape(f"  self_attn", capture[f"decoder.layer{idx}.self_attn"])
            _print_shape(f"  cross_attn", capture[f"decoder.layer{idx}.cross_attn"])
            _print_shape(f"  ffn", capture[f"decoder.layer{idx}.ffn"])
            _print_shape(f"  layer_out", capture[f"decoder.layer{idx}"])
    """,
    4: """\
        # Chunk 4: Layer-wise class/box refinement and decode
        # Source: pytorch_implementation/perception/sparse4d/blocks.py, decoder.py

        capture, handles = {}, []
        for idx in range(cfg.num_decoder_layers):
            _register_hook(model.head.decoder.refine_layers[idx].cls_branch,
                           f"decoder.refine{idx}.cls_branch", capture, handles)
            _register_hook(model.head.decoder.refine_layers[idx].reg_branch,
                           f"decoder.refine{idx}.reg_branch", capture, handles)

        with torch.no_grad():
            outputs = model(img, img_metas, decode=False)
        for h in handles:
            h.remove()

        print("=== Refinement branch shapes ===")
        for idx in range(cfg.num_decoder_layers):
            _print_shape(f"refine{idx}.cls_branch", capture[f"decoder.refine{idx}.cls_branch"])
            _print_shape(f"refine{idx}.reg_branch", capture[f"decoder.refine{idx}.reg_branch"])

        print(f"\\n=== Stacked outputs ===")
        print(f"  all_cls_scores: {tuple(outputs['all_cls_scores'].shape)}")
        print(f"  all_bbox_preds: {tuple(outputs['all_bbox_preds'].shape)}")

        with torch.no_grad():
            decoded_out = model(img, img_metas, decode=True)
        decoded = decoded_out["decoded"]
        print(f"\\n=== Decoded (sample 0) ===")
        for key, val in decoded[0].items():
            if torch.is_tensor(val):
                print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
    """,
}


# ---------------------------------------------------------------------------
# StreamPETR
# ---------------------------------------------------------------------------

STREAMPETR_METAS = '''\
def _build_dummy_img_metas(batch_size, num_cams, height, width):
            metas = []
            for batch_idx in range(batch_size):
                lidar2img = []
                for cam_idx in range(num_cams):
                    lidar2img.append([
                        [1.0, 0.0, 0.0, float(width) * (0.5 + cam_idx * 0.01)],
                        [0.0, 1.0, 0.0, float(height) * 0.5],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                metas.append({
                    "sample_idx": f"sample_{batch_idx}",
                    "scene_token": f"scene_{batch_idx}",
                    "lidar2img": lidar2img,
                    "img_shape": [(height, width, 3)] * num_cams,
                    "pad_shape": [(height, width, 3)] * num_cams,
                })
            return metas
        img_metas = _build_dummy_img_metas(batch_size, cfg.num_cams, height, width)
'''

STREAMPETR_CHUNKS = {
    0: """\
        # Chunk 0: End-to-end StreamPETR contract
        # Source: pytorch_implementation/perception/streampetr/model.py

        prev_exists = torch.zeros(batch_size, dtype=torch.float32)
        with torch.no_grad():
            outputs_frame1 = model(img, img_metas, decode=False, prev_exists=prev_exists)

        print("=== Frame 1 (no history) output shapes ===")
        for key, val in outputs_frame1.items():
            if val is not None and torch.is_tensor(val):
                print(f"  {key}: {tuple(val.shape)}")

        # Frame 2 with temporal memory
        img2 = torch.randn_like(img)
        prev_exists_t2 = torch.ones(batch_size, dtype=torch.float32)
        with torch.no_grad():
            outputs_frame2 = model(img2, img_metas, decode=False, prev_exists=prev_exists_t2)

        print(f"\\n=== Frame 2 (with history) output shapes ===")
        for key, val in outputs_frame2.items():
            if val is not None and torch.is_tensor(val):
                print(f"  {key}: {tuple(val.shape)}")
    """,
    1: """\
        # Chunk 1: Multi-view feature extraction
        # Source: pytorch_implementation/perception/streampetr/backbone_neck.py

        capture, handles = {}, []
        _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
        for idx, stage in enumerate(model.backbone_neck.backbone.stages):
            _register_hook(stage, f"backbone.stage{idx}", capture, handles)
        _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)

        with torch.no_grad():
            img_feats = model.extract_img_feat(img)
        for h in handles:
            h.remove()

        print("=== Backbone/FPN shapes ===")
        for key in sorted(capture.keys()):
            _print_shape(key, capture[key])
        for lvl, feat in enumerate(img_feats):
            print(f"  mlvl_feats[{lvl}]: {tuple(feat.shape)}")
    """,
    2: """\
        # Chunk 2: 3D geometry-aware positional encoding
        # Source: pytorch_implementation/perception/streampetr/head.py - position_embeding
        # Source: pytorch_implementation/perception/streampetr/utils.py

        capture, handles = {}, []
        _register_hook(model.head.input_proj, "head.input_proj", capture, handles)
        _register_hook(model.head.position_encoder, "head.position_encoder", capture, handles)
        _register_hook(model.head.adapt_pos3d, "head.adapt_pos3d", capture, handles)

        prev_exists = torch.zeros(batch_size, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(img, img_metas, decode=False, prev_exists=prev_exists)
        for h in handles:
            h.remove()

        print("=== 3D position embedding shapes ===")
        _print_shape("input_proj", capture["head.input_proj"])
        _print_shape("position_encoder", capture["head.position_encoder"])
        _print_shape("adapt_pos3d", capture["head.adapt_pos3d"])
    """,
    3: """\
        # Chunk 3: Temporal alignment with object-centric memory
        # Source: pytorch_implementation/perception/streampetr/head.py - temporal_alignment

        capture, handles = {}, []
        _register_hook(model.head.reference_points, "head.reference_points", capture, handles)
        _register_hook(model.head.query_embedding, "head.query_embedding", capture, handles)
        _register_hook(model.head.time_embedding, "head.time_embedding", capture, handles)

        prev_exists = torch.zeros(batch_size, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(img, img_metas, decode=False, prev_exists=prev_exists)
        for h in handles:
            h.remove()

        print("=== Temporal alignment shapes ===")
        _print_shape("reference_points", capture["head.reference_points"])
        _print_shape("query_embedding", capture["head.query_embedding"])
        _print_shape("time_embedding", capture["head.time_embedding"])
    """,
    4: """\
        # Chunk 4: Temporal transformer decoding
        # Source: pytorch_implementation/perception/streampetr/transformer.py

        capture, handles = {}, []
        for idx in range(cfg.num_decoder_layers):
            dec = model.head.transformer.decoder.layers[idx]
            _register_hook(dec, f"decoder.layer{idx}", capture, handles)
            _register_hook(dec.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
            _register_hook(dec.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
            _register_hook(dec.ffn, f"decoder.layer{idx}.ffn", capture, handles)

        prev_exists = torch.zeros(batch_size, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(img, img_metas, decode=False, prev_exists=prev_exists)
        for h in handles:
            h.remove()

        print("=== Decoder layer shapes ===")
        for idx in range(cfg.num_decoder_layers):
            print(f"\\n  --- Layer {idx} ---")
            _print_shape(f"  self_attn", capture[f"decoder.layer{idx}.self_attn"])
            _print_shape(f"  cross_attn", capture[f"decoder.layer{idx}.cross_attn"])
            _print_shape(f"  ffn", capture[f"decoder.layer{idx}.ffn"])
            _print_shape(f"  layer_out", capture[f"decoder.layer{idx}"])
    """,
    5: """\
        # Chunk 5: Prediction heads and memory update
        # Source: pytorch_implementation/perception/streampetr/head.py

        capture, handles = {}, []
        for idx in range(cfg.num_decoder_layers):
            _register_hook(model.head.cls_branches[idx], f"head.cls_branch{idx}", capture, handles)
            _register_hook(model.head.reg_branches[idx], f"head.reg_branch{idx}", capture, handles)

        prev_exists = torch.zeros(batch_size, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(img, img_metas, decode=False, prev_exists=prev_exists)
        for h in handles:
            h.remove()

        print("=== Branch shapes ===")
        for idx in range(cfg.num_decoder_layers):
            _print_shape(f"cls_branch{idx}", capture[f"head.cls_branch{idx}"])
            _print_shape(f"reg_branch{idx}", capture[f"head.reg_branch{idx}"])

        print(f"\\n=== Stacked outputs ===")
        print(f"  all_cls_scores: {tuple(outputs['all_cls_scores'].shape)}")
        print(f"  all_bbox_preds: {tuple(outputs['all_bbox_preds'].shape)}")

        # Check memory update
        print(f"\\n=== Memory state ===")
        print(f"  memory_embedding: {tuple(model.head.memory_embedding.shape)}")
        print(f"  memory_reference_point: {tuple(model.head.memory_reference_point.shape)}")
        print(f"  memory_timestamp: {tuple(model.head.memory_timestamp.shape)}")
    """,
}


# ---------------------------------------------------------------------------
# prediction/surroundocc
# ---------------------------------------------------------------------------

def _make_prediction_surroundocc_notebook():
    nb = _new_notebook()
    md = _read_markdown("prediction/surroundocc")

    nb.cells.append(code_cell("""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.prediction.surroundocc.config import debug_forward_config
        from pytorch_implementation.prediction.surroundocc.model import SurroundOccPredictionLite
        from pytorch_implementation.prediction.surroundocc.postprocess import (
            occupancy_iou,
            trajectory_consistency_error,
            trajectory_metrics,
        )

        cfg = debug_forward_config(
            history_steps=4,
            future_steps=5,
            num_agents=6,
            in_channels=16,
            embed_dims=64,
            bev_hw=(24, 24),
            depth_bins=4,
        )
        model = SurroundOccPredictionLite(cfg).eval()

        batch_size = 2
        history_bev = torch.randn(
            batch_size,
            cfg.history_steps,
            cfg.in_channels,
            cfg.bev_hw[0],
            cfg.bev_hw[1],
        )
        agent_states = torch.randn(batch_size, cfg.num_agents, 4)

        print(f"Model: {type(model).__name__}")
        print(f"Config: {cfg.name}")
        print(f"history_bev: {tuple(history_bev.shape)}")
        print(f"agent_states: {tuple(agent_states.shape)}")
    """))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    sections = _split_markdown_sections(md)
    chunk_ids_seen = set()
    i = 0
    while i < len(sections):
        heading, body = sections[i]
        full_md = (heading + "\n" + body).strip() if heading else body.strip()
        chunk_id = _chunk_id_from_heading(heading)

        if heading.startswith("## "):
            sub_parts = [full_md]
            j = i + 1
            while j < len(sections):
                h2, b2 = sections[j]
                if h2.startswith("## "):
                    break
                sub_parts.append((h2 + "\n" + b2).strip())
                j += 1
            nb.cells.append(md_cell("\n\n".join(sub_parts)))
            i = j
        else:
            if full_md:
                nb.cells.append(md_cell(full_md))
            i += 1

        if chunk_id is not None and chunk_id not in chunk_ids_seen:
            chunk_ids_seen.add(chunk_id)
            if chunk_id == 0:
                nb.cells.append(code_cell("""\
                    # Chunk 0: end-to-end prediction + decode
                    with torch.no_grad():
                        outputs = model(history_bev, agent_states, decode=False)
                        decoded_outputs = model(history_bev, agent_states, decode=True)

                    print("=== Output shapes ===")
                    for key, val in outputs.items():
                        if torch.is_tensor(val):
                            print(f"  {key}: {tuple(val.shape)}")

                    decoded = decoded_outputs["decoded"]
                    print(f"\\nDecoded batch size: {len(decoded)}")
                    print(f"Decoded sample occupancy: {tuple(decoded[0]['occupancy'].shape)}")
                    print(f"Decoded sample trajectory: {tuple(decoded[0]['trajectory'].shape)}")
                """))
            elif chunk_id == 1:
                nb.cells.append(code_cell("""\
                    # Chunk 1: spatial + temporal encoding hooks
                    capture, handles = {}, []
                    _register_hook(model.spatial_encoder.stem, "spatial.stem", capture, handles)
                    _register_hook(model.spatial_encoder.block1, "spatial.block1", capture, handles)
                    _register_hook(model.spatial_encoder.block2, "spatial.block2", capture, handles)
                    _register_hook(model.spatial_encoder.out_proj, "spatial.out_proj", capture, handles)
                    _register_hook(model.temporal_encoder.gru, "temporal.gru", capture, handles)

                    with torch.no_grad():
                        outputs = model(history_bev, agent_states, decode=False)
                    for h in handles:
                        h.remove()

                    print("=== Spatial/Temporal shapes ===")
                    for key in sorted(capture.keys()):
                        _print_shape(key, capture[key])
                    _print_shape("temporal_sequence", outputs["temporal_sequence"])
                """))
            elif chunk_id == 2:
                nb.cells.append(code_cell("""\
                    # Chunk 2: horizon decoder + occupancy logits
                    capture, handles = {}, []
                    _register_hook(model.horizon_decoder.mlp, "horizon.mlp", capture, handles)
                    _register_hook(model.occupancy_head.refine, "occupancy.refine", capture, handles)
                    _register_hook(model.occupancy_head.classifier, "occupancy.classifier", capture, handles)

                    with torch.no_grad():
                        outputs = model(history_bev, agent_states, decode=False)
                    for h in handles:
                        h.remove()

                    print("=== Horizon/Occupancy shapes ===")
                    _print_shape("horizon.mlp", capture["horizon.mlp"])
                    _print_shape("occupancy.refine", capture["occupancy.refine"])
                    _print_shape("occupancy.classifier", capture["occupancy.classifier"])
                    print(f"occupancy_logits: {tuple(outputs['occupancy_logits'].shape)}")
                """))
            elif chunk_id == 3:
                nb.cells.append(code_cell("""\
                    # Chunk 3: trajectory forecasting + consistency
                    capture, handles = {}, []
                    _register_hook(model.trajectory_head.agent_proj, "trajectory.agent_proj", capture, handles)
                    _register_hook(model.trajectory_head.delta_mlp, "trajectory.delta_mlp", capture, handles)

                    with torch.no_grad():
                        outputs = model(history_bev, agent_states, decode=False)
                    for h in handles:
                        h.remove()

                    print("=== Trajectory head shapes ===")
                    _print_shape("trajectory.agent_proj", capture["trajectory.agent_proj"])
                    _print_shape("trajectory.delta_mlp", capture["trajectory.delta_mlp"])
                    print(f"trajectory: {tuple(outputs['trajectory'].shape)}")
                    print(f"velocity: {tuple(outputs['velocity'].shape)}")

                    consistency_error = trajectory_consistency_error(outputs["trajectory"], outputs["velocity"], dt=cfg.dt)
                    print(f"consistency_error: {float(consistency_error):.6f}")
                """))
            elif chunk_id == 4:
                nb.cells.append(code_cell("""\
                    # Chunk 4: metric smoke checks
                    with torch.no_grad():
                        outputs = model(history_bev, agent_states, decode=False)
                        decoded = model(history_bev, agent_states, decode=True)["decoded"]

                    pred_traj = outputs["trajectory"]
                    gt_traj = pred_traj + 0.05 * torch.randn_like(pred_traj)
                    metrics = trajectory_metrics(pred_traj, gt_traj)
                    print("trajectory metrics:", metrics)

                    pred_occ = torch.stack([sample["occupancy"] for sample in decoded], dim=0)
                    gt_occ = pred_occ.clone()
                    gt_occ[..., 0, 0, 0] = True
                    iou = occupancy_iou(pred_occ, gt_occ)
                    print("occupancy IoU smoke:", iou)
                """))

    nb.cells.append(
        code_cell(
            _build_final_finite_check_cell(
                hook_lines=[
                    '_register_hook(model.spatial_encoder.stem, "spatial.stem", capture, handles)',
                    '_register_hook(model.temporal_encoder.gru, "temporal.gru", capture, handles)',
                    '_register_hook(model.horizon_decoder.mlp, "horizon.mlp", capture, handles)',
                    '_register_hook(model.occupancy_head.classifier, "occupancy.classifier", capture, handles)',
                    '_register_hook(model.trajectory_head.delta_mlp, "trajectory.delta_mlp", capture, handles)',
                ],
                forward_lines=["outputs = model(history_bev, agent_states, decode=False)"],
            )
        )
    )
    return nb


def _make_prediction_beverse_notebook():
    nb = _new_notebook()
    md = _read_markdown("prediction/beverse")

    nb.cells.append(code_cell("""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.prediction.beverse.config import debug_forward_config
        from pytorch_implementation.prediction.beverse.metrics import compute_ade_fde, select_best_mode_by_ade
        from pytorch_implementation.prediction.beverse.model import BEVerseLite

        cfg = debug_forward_config()
        model = BEVerseLite(cfg).eval()

        batch_size = 2
        height, width = 96, 160
        img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
        img_metas = [{"sample_idx": f"sample_{batch_idx}"} for batch_idx in range(batch_size)]

        print(f"Model: {type(model).__name__}")
        print(f"Config: {cfg.name}")
        print(f"img: {tuple(img.shape)}")
    """))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    chunk_code = {
        0: """\
            # Chunk 0: End-to-end forward contract + decode
            with torch.no_grad():
                outputs = model(img, img_metas, decode=False)
                decoded_out = model(img, img_metas, decode=True)

            print("=== Output shapes ===")
            for key, val in outputs.items():
                if torch.is_tensor(val):
                    print(f"  {key}: {tuple(val.shape)}")

            decoded = decoded_out["decoded"]
            print("\\n=== Decoded contract ===")
            for key, val in decoded.items():
                if torch.is_tensor(val):
                    print(f"  {key}: {tuple(val.shape)}")
        """,
        1: """\
            # Chunk 1: Multi-camera image encoding and BEV seed
            capture, handles = {}, []
            _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
            for idx, stage in enumerate(model.backbone_neck.backbone.stages):
                _register_hook(stage, f"backbone.stage{idx}", capture, handles)
            _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)
            _register_hook(model.bev_encoder[0], "bev_encoder.conv0", capture, handles)
            _register_hook(model.bev_encoder[3], "bev_encoder.conv1", capture, handles)

            with torch.no_grad():
                outputs = model(img, img_metas, decode=False)
            for h in handles:
                h.remove()

            print("=== Backbone/BEV encoder shapes ===")
            for key in sorted(capture.keys()):
                _print_shape(key, capture[key])
            _print_shape("camera_feat", outputs["camera_feat"])
            _print_shape("bev_embed", outputs["bev_embed"])
        """,
        2: """\
            # Chunk 2: Temporal horizon decoding
            capture, handles = {}, []
            _register_hook(model.temporal_predictor.time_embedding, "temporal.time_embedding", capture, handles)
            _register_hook(model.temporal_predictor.gru, "temporal.gru", capture, handles)

            with torch.no_grad():
                outputs = model(img, img_metas, decode=False)
            for h in handles:
                h.remove()

            print("=== Temporal decoding ===")
            _print_shape("temporal.time_embedding", capture["temporal.time_embedding"])
            _print_shape("temporal.gru", capture["temporal.gru"])
            _print_shape("temporal_tokens", outputs["temporal_tokens"])
            print(f"time_stamps: {tuple(outputs['time_stamps'].shape)}")
            print(f"time_stamps values: {outputs['time_stamps']}")
        """,
        3: """\
            # Chunk 3: Multimodal trajectory and mode heads
            capture, handles = {}, []
            _register_hook(model.trajectory_head.shared[1], "head.shared_fc0", capture, handles)
            _register_hook(model.trajectory_head.delta_head, "head.delta", capture, handles)
            _register_hook(model.trajectory_head.mode_head, "head.mode", capture, handles)

            with torch.no_grad():
                outputs = model(img, img_metas, decode=False)
            for h in handles:
                h.remove()

            print("=== Trajectory head shapes ===")
            _print_shape("head.shared_fc0", capture["head.shared_fc0"])
            _print_shape("head.delta", capture["head.delta"])
            _print_shape("head.mode", capture["head.mode"])
            _print_shape("trajectory_deltas", outputs["trajectory_deltas"])
            _print_shape("trajectory_preds", outputs["trajectory_preds"])
            _print_shape("mode_logits", outputs["mode_logits"])
            _print_shape("mode_probs", outputs["mode_probs"])

            reconstructed = outputs["trajectory_deltas"].cumsum(dim=2)
            assert torch.allclose(outputs["trajectory_preds"], reconstructed, atol=1e-5, rtol=1e-5)
            print("cumsum consistency check passed.")
        """,
        4: """\
            # Chunk 4: Prediction metrics and horizon integrity
            with torch.no_grad():
                outputs = model(img, img_metas, decode=False)

            target = outputs["trajectory_preds"][:, 0] + 0.1
            valid_mask = torch.ones((outputs["trajectory_preds"].shape[0], outputs["trajectory_preds"].shape[2]), dtype=target.dtype, device=target.device)
            valid_mask[:, -1] = 0.0

            best_mode_idx, best_traj, best_ade = select_best_mode_by_ade(
                outputs["trajectory_preds"], target, valid_mask=valid_mask
            )
            ade, fde = compute_ade_fde(best_traj, target, valid_mask=valid_mask)

            print("=== Metric smoke ===")
            print(f"best_mode_idx shape: {tuple(best_mode_idx.shape)}")
            print(f"best_traj shape: {tuple(best_traj.shape)}")
            print(f"best_ade: {best_ade}")
            print(f"ADE: {ade}")
            print(f"FDE: {fde}")
        """,
    }
    _append_markdown_sections_with_chunks(nb, md, chunk_code)

    nb.cells.append(
        code_cell(
            _build_final_finite_check_cell(
                hook_lines=[
                    '_register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)',
                    "for idx, stage in enumerate(model.backbone_neck.backbone.stages):",
                    '    _register_hook(stage, f"backbone.stage{idx}", capture, handles)',
                    '_register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)',
                    '_register_hook(model.bev_encoder[0], "bev_encoder.conv0", capture, handles)',
                    '_register_hook(model.bev_encoder[3], "bev_encoder.conv1", capture, handles)',
                    '_register_hook(model.temporal_predictor.time_embedding, "temporal.time_embedding", capture, handles)',
                    '_register_hook(model.temporal_predictor.gru, "temporal.gru", capture, handles)',
                    '_register_hook(model.trajectory_head.shared[1], "head.shared_fc0", capture, handles)',
                    '_register_hook(model.trajectory_head.delta_head, "head.delta", capture, handles)',
                    '_register_hook(model.trajectory_head.mode_head, "head.mode", capture, handles)',
                ],
                forward_lines=["outputs = model(img, img_metas, decode=False)"],
            )
        )
    )
    return nb


def _make_prediction_flashocc_notebook():
    nb = _new_notebook()
    md = _read_markdown("prediction/flashocc")

    nb.cells.append(code_cell("""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.prediction.flashocc.config import debug_forward_config
        from pytorch_implementation.prediction.flashocc.metrics import (
            average_displacement_error,
            final_displacement_error,
            trajectory_smoothness_l2,
        )
        from pytorch_implementation.prediction.flashocc.model import FlashOccLite

        cfg = debug_forward_config(
            num_history=4,
            pred_horizon=8,
            bev_h=48,
            bev_w=64,
            num_queries=12,
            num_modes=3,
            topk=6,
        )
        model = FlashOccLite(cfg).eval()

        batch_size = 2
        occ_seq = torch.randn(
            batch_size,
            cfg.num_history,
            cfg.backbone.in_channels,
            cfg.bev_h,
            cfg.bev_w,
        )

        print(f"Model: {type(model).__name__}")
        print(f"Config: {cfg.name}")
        print(f"occ_seq: {tuple(occ_seq.shape)}")
    """))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    chunk_code = {
        0: """\
            # Chunk 0: End-to-end prediction contract
            with torch.no_grad():
                outputs = model(occ_seq, decode=False)
                decoded_out = model(occ_seq, decode=True)

            print("=== Output shapes ===")
            for key, val in outputs.items():
                if torch.is_tensor(val):
                    print(f"  {key}: {tuple(val.shape)}")

            decoded = decoded_out["decoded"]
            print(f"\\nDecoded batch size: {len(decoded)}")
            print("Decoded sample keys:", sorted(decoded[0].keys()))
            for key, val in decoded[0].items():
                if torch.is_tensor(val):
                    print(f"  {key}: {tuple(val.shape)}")
        """,
        1: """\
            # Chunk 1: Occupancy encoding and BEV lifting
            capture, handles = {}, []
            _register_hook(model.backbone.stem, "backbone.stem", capture, handles)
            for idx, block in enumerate(model.backbone.blocks):
                _register_hook(block, f"backbone.block{idx}", capture, handles)

            with torch.no_grad():
                outputs = model(occ_seq, decode=False)
            for h in handles:
                h.remove()

            print("=== Backbone shapes ===")
            for key in sorted(capture.keys()):
                _print_shape(key, capture[key])
            _print_shape("bev_sequence", outputs["bev_sequence"])
        """,
        2: """\
            # Chunk 2: Temporal fusion and time-axis integrity
            capture, handles = {}, []
            _register_hook(model.temporal_mixer.temporal_conv, "temporal.temporal_conv", capture, handles)
            _register_hook(model.temporal_mixer.proj, "temporal.proj", capture, handles)

            with torch.no_grad():
                outputs = model(occ_seq, decode=False)
            for h in handles:
                h.remove()

            print("=== Temporal mixer ===")
            _print_shape("temporal.temporal_conv", capture["temporal.temporal_conv"])
            _print_shape("temporal.proj", capture["temporal.proj"])
            _print_shape("temporal_tokens", outputs["temporal_tokens"])
            print(f"time_stamps: {outputs['time_stamps']}")
            assert torch.all(outputs["time_stamps"][1:] > outputs["time_stamps"][:-1])
            print("time-axis monotonicity check passed.")
        """,
        3: """\
            # Chunk 3: Query-conditioned multimodal trajectory head
            capture, handles = {}, []
            _register_hook(model.prediction_head.query_proj, "head.query_proj", capture, handles)
            _register_hook(model.prediction_head.cross_attn, "head.cross_attn", capture, handles)
            _register_hook(model.prediction_head.traj_head, "head.traj_head", capture, handles)
            _register_hook(model.prediction_head.mode_head, "head.mode_head", capture, handles)

            with torch.no_grad():
                outputs = model(occ_seq, decode=False)
            for h in handles:
                h.remove()

            print("=== Prediction head shapes ===")
            _print_shape("head.query_proj", capture["head.query_proj"])
            _print_shape("head.cross_attn", capture["head.cross_attn"])
            _print_shape("head.traj_head", capture["head.traj_head"])
            _print_shape("head.mode_head", capture["head.mode_head"])
            _print_shape("traj_positions", outputs["traj_positions"])
            _print_shape("traj_velocity", outputs["traj_velocity"])

            reconstructed = outputs["anchor_xy"].unsqueeze(2).unsqueeze(3) + torch.cumsum(outputs["traj_deltas"], dim=3)
            assert torch.allclose(outputs["traj_positions"], reconstructed, atol=1e-6)
            assert torch.allclose(outputs["traj_velocity"] * cfg.dt, outputs["traj_deltas"], atol=1e-6)
            print("trajectory consistency checks passed.")
        """,
        4: """\
            # Chunk 4: Decode and metric smoke checks
            with torch.no_grad():
                outputs = model(occ_seq, decode=False)
                decoded = model(occ_seq, decode=True)["decoded"]

            best_mode = outputs["mode_logits"].argmax(dim=-1)
            gather_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, outputs["traj_positions"].shape[-2], outputs["traj_positions"].shape[-1]
            )
            pred_best = torch.gather(outputs["traj_positions"], dim=2, index=gather_idx).squeeze(2)
            gt = pred_best + 0.05 * torch.randn_like(pred_best)

            ade = average_displacement_error(pred_best, gt)
            fde = final_displacement_error(pred_best, gt)
            smooth = trajectory_smoothness_l2(pred_best)
            print("metric smoke:", {"ade": float(ade), "fde": float(fde), "smoothness": float(smooth)})

            print("\\n=== Decoded contract ===")
            for sample_idx, sample in enumerate(decoded):
                print(f"sample {sample_idx}:")
                for key, val in sample.items():
                    if torch.is_tensor(val):
                        print(f"  {key}: {tuple(val.shape)}")
        """,
    }
    _append_markdown_sections_with_chunks(nb, md, chunk_code)

    nb.cells.append(
        code_cell(
            _build_final_finite_check_cell(
                hook_lines=[
                    '_register_hook(model.backbone.stem, "backbone.stem", capture, handles)',
                    "for idx, block in enumerate(model.backbone.blocks):",
                    '    _register_hook(block, f"backbone.block{idx}", capture, handles)',
                    '_register_hook(model.temporal_mixer.temporal_conv, "temporal.temporal_conv", capture, handles)',
                    '_register_hook(model.temporal_mixer.proj, "temporal.proj", capture, handles)',
                    '_register_hook(model.prediction_head.query_proj, "head.query_proj", capture, handles)',
                    '_register_hook(model.prediction_head.cross_attn, "head.cross_attn", capture, handles)',
                    '_register_hook(model.prediction_head.traj_head, "head.traj_head", capture, handles)',
                    '_register_hook(model.prediction_head.mode_head, "head.mode_head", capture, handles)',
                ],
                forward_lines=["outputs = model(occ_seq, decode=False)"],
            )
        )
    )
    return nb


def _make_prediction_vip3d_notebook():
    nb = _new_notebook()
    md = _read_markdown("prediction/vip3d")

    nb.cells.append(code_cell("""\
        import sys, os
        sys.path.insert(0, os.path.abspath("../.."))

        import torch
        from pytorch_implementation.prediction.vip3d.config import debug_forward_config
        from pytorch_implementation.prediction.vip3d.model import VIP3DLite, compute_ade_fde

        cfg = debug_forward_config()
        model = VIP3DLite(cfg).eval()

        def _build_debug_batch(cfg):
            torch.manual_seed(13)
            batch_size = 2
            num_agents = 5

            base_xy = torch.randn(batch_size, num_agents, 2)
            velocity = torch.randn(batch_size, num_agents, 2) * 0.2
            history_time = torch.arange(cfg.history_steps, dtype=torch.float32).view(1, 1, cfg.history_steps, 1)
            future_time = torch.arange(1, cfg.future_steps + 1, dtype=torch.float32).view(1, 1, cfg.future_steps, 1)

            history_xy = base_xy.unsqueeze(2) + history_time * velocity.unsqueeze(2)
            history_vel = velocity.unsqueeze(2).expand(-1, -1, cfg.history_steps, -1)
            agent_history = torch.cat([history_xy, history_vel], dim=-1)

            agent_valid = torch.ones(batch_size, num_agents, cfg.history_steps, dtype=torch.bool)
            agent_valid[0, 0, -1] = False
            agent_valid[1, 3, 0] = False

            map_polylines = torch.randn(
                batch_size,
                cfg.map_tokens,
                cfg.map_points_per_token,
                cfg.map_input_dim,
            )

            last_observed = history_xy[:, :, -1]
            gt_future = last_observed.unsqueeze(2) + future_time * velocity.unsqueeze(2)
            gt_valid = torch.ones(batch_size, num_agents, cfg.future_steps, dtype=torch.bool)
            gt_valid[0, 2, -2:] = False
            return agent_history, map_polylines, agent_valid, gt_future, gt_valid

        agent_history, map_polylines, agent_valid, gt_future, gt_valid = _build_debug_batch(cfg)

        print(f"Model: {type(model).__name__}")
        print(f"Config: {cfg.name}")
        print(f"agent_history: {tuple(agent_history.shape)}")
        print(f"map_polylines: {tuple(map_polylines.shape)}")
        print(f"agent_valid: {tuple(agent_valid.shape)}")
    """))
    nb.cells.append(code_cell_raw(COMMON_HELPERS))

    chunk_code = {
        0: """\
            # Chunk 0: End-to-end prediction contract
            with torch.no_grad():
                outputs = model(agent_history, map_polylines, agent_valid)

            print("=== Output shapes ===")
            for key, val in outputs.items():
                if torch.is_tensor(val):
                    print(f"  {key}: {tuple(val.shape)}")
        """,
        1: """\
            # Chunk 1: Agent temporal encoding
            capture, handles = {}, []
            _register_hook(model.history_input_proj, "history.input_proj", capture, handles)
            _register_hook(model.history_encoder.layers[0], "history.encoder.layer0", capture, handles)
            _register_hook(model.history_norm, "history.norm", capture, handles)

            with torch.no_grad():
                outputs = model(agent_history, map_polylines, agent_valid)
            for h in handles:
                h.remove()

            print("=== Agent temporal encoding ===")
            _print_shape("history.input_proj", capture["history.input_proj"])
            _print_shape("history.encoder.layer0", capture["history.encoder.layer0"])
            _print_shape("history.norm", capture["history.norm"])
            _print_shape("history_tokens", outputs["history_tokens"])
        """,
        2: """\
            # Chunk 2: Map tokenization
            capture, handles = {}, []
            _register_hook(model.map_point_mlp, "map.point_mlp", capture, handles)
            _register_hook(model.map_token_proj, "map.token_proj", capture, handles)

            with torch.no_grad():
                outputs = model(agent_history, map_polylines, agent_valid)
            for h in handles:
                h.remove()

            print("=== Map tokenization ===")
            _print_shape("map.point_mlp", capture["map.point_mlp"])
            _print_shape("map.token_proj", capture["map.token_proj"])
            _print_shape("map_tokens", outputs["map_tokens"])
        """,
        3: """\
            # Chunk 3: Agent-map fusion
            capture, handles = {}, []
            _register_hook(model.agent_map_attention, "fusion.cross_attention", capture, handles)
            _register_hook(model.fusion_norm, "fusion.norm", capture, handles)

            with torch.no_grad():
                outputs = model(agent_history, map_polylines, agent_valid)
            for h in handles:
                h.remove()

            print("=== Agent-map fusion ===")
            _print_shape("fusion.cross_attention", capture["fusion.cross_attention"])
            _print_shape("fusion.norm", capture["fusion.norm"])
            _print_shape("fused_tokens", outputs["fused_tokens"])
            _print_shape("attention_weights", outputs["attention_weights"])
        """,
        4: """\
            # Chunk 4: Multimodal trajectory decoding
            capture, handles = {}, []
            _register_hook(model.decoder.trunk, "decoder.trunk", capture, handles)
            _register_hook(model.decoder.mode_head, "decoder.mode_head", capture, handles)
            _register_hook(model.decoder.delta_head, "decoder.delta_head", capture, handles)

            with torch.no_grad():
                outputs = model(agent_history, map_polylines, agent_valid)
            for h in handles:
                h.remove()

            print("=== Decoder outputs ===")
            _print_shape("decoder.trunk", capture["decoder.trunk"])
            _print_shape("decoder.mode_head", capture["decoder.mode_head"])
            _print_shape("decoder.delta_head", capture["decoder.delta_head"])
            _print_shape("mode_probs", outputs["mode_probs"])
            _print_shape("trajectories", outputs["trajectories"])
            _print_shape("best_trajectory", outputs["best_trajectory"])

            assert torch.allclose(outputs["mode_probs"].sum(dim=-1), torch.ones_like(outputs["mode_probs"].sum(dim=-1)), atol=1e-5)
            batch_size, num_agents = outputs["best_mode"].shape
            gather_idx = outputs["best_mode"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, num_agents, 1, cfg.future_steps, 2
            )
            expected_best = outputs["trajectories"].gather(dim=2, index=gather_idx).squeeze(2)
            assert torch.allclose(outputs["best_trajectory"], expected_best, atol=1e-5)
            print("best-mode trajectory contract check passed.")
        """,
        5: """\
            # Chunk 5: Prediction metrics (ADE/FDE smoke)
            with torch.no_grad():
                outputs = model(agent_history, map_polylines, agent_valid)
            metrics = compute_ade_fde(
                trajectories=outputs["trajectories"],
                gt_future=gt_future,
                gt_valid=gt_valid,
            )
            print("metric smoke:", {"ade": float(metrics["ade"]), "fde": float(metrics["fde"])})
        """,
    }
    _append_markdown_sections_with_chunks(nb, md, chunk_code)

    nb.cells.append(
        code_cell(
            _build_final_finite_check_cell(
                hook_lines=[
                    '_register_hook(model.history_input_proj, "history.input_proj", capture, handles)',
                    '_register_hook(model.history_encoder.layers[0], "history.encoder.layer0", capture, handles)',
                    '_register_hook(model.history_norm, "history.norm", capture, handles)',
                    '_register_hook(model.map_point_mlp, "map.point_mlp", capture, handles)',
                    '_register_hook(model.map_token_proj, "map.token_proj", capture, handles)',
                    '_register_hook(model.agent_map_attention, "fusion.cross_attention", capture, handles)',
                    '_register_hook(model.fusion_norm, "fusion.norm", capture, handles)',
                    '_register_hook(model.decoder.trunk, "decoder.trunk", capture, handles)',
                    '_register_hook(model.decoder.mode_head, "decoder.mode_head", capture, handles)',
                    '_register_hook(model.decoder.delta_head, "decoder.delta_head", capture, handles)',
                ],
                forward_lines=["outputs = model(agent_history, map_polylines, agent_valid)"],
            )
        )
    )
    return nb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    generated = []

    # PETR
    nb = _make_petr_notebook()
    path = _write_notebook("perception/petr", nb)
    generated.append(path)
    print(f"[1/11] Generated: {path}  ({len(nb.cells)} cells)")

    # BEVFormer
    nb = _make_bevformer_notebook()
    path = _write_notebook("perception/bevformer", nb)
    generated.append(path)
    print(f"[2/11] Generated: {path}  ({len(nb.cells)} cells)")

    # MapTR
    nb = _make_maptr_notebook()
    path = _write_notebook("perception/maptr", nb)
    generated.append(path)
    print(f"[3/11] Generated: {path}  ({len(nb.cells)} cells)")

    # FB-BEV
    nb = _make_simple_notebook(
        model_key="perception/fbbev", import_module="perception.fbbev", model_class="FBBEVLite",
        config_call="debug_forward_config(max_num=48, depth_bins=6, bev_h=20, bev_w=20, bev_z=3)",
        metas_fn=FBBEV_METAS, batch_size=2, chunk_code=FBBEV_CHUNKS,
    )
    path = _write_notebook("perception/fbbev", nb)
    generated.append(path)
    print(f"[4/11] Generated: {path}  ({len(nb.cells)} cells)")

    # PolarFormer
    nb = _make_simple_notebook(
        model_key="perception/polarformer", import_module="perception.polarformer", model_class="PolarFormerLite",
        config_call="debug_forward_config(num_queries=48, decoder_layers=2, azimuth_bins=96, radius_bins=48)",
        metas_fn=POLAR_METAS, batch_size=1, chunk_code=POLAR_CHUNKS,
    )
    path = _write_notebook("perception/polarformer", nb)
    generated.append(path)
    print(f"[5/11] Generated: {path}  ({len(nb.cells)} cells)")

    # Sparse4D
    nb = _make_simple_notebook(
        model_key="perception/sparse4d", import_module="perception.sparse4d", model_class="Sparse4DLite",
        config_call="debug_forward_config(num_queries=48, decoder_layers=2)",
        metas_fn=SPARSE4D_METAS, batch_size=1, chunk_code=SPARSE4D_CHUNKS,
    )
    path = _write_notebook("perception/sparse4d", nb)
    generated.append(path)
    print(f"[6/11] Generated: {path}  ({len(nb.cells)} cells)")

    # StreamPETR
    nb = _make_simple_notebook(
        model_key="perception/streampetr", import_module="perception.streampetr", model_class="StreamPETRLite",
        config_call="debug_forward_config(num_queries=48, decoder_layers=2, depth_num=6, memory_len=40, topk_proposals=12, num_propagated=8)",
        metas_fn=STREAMPETR_METAS, batch_size=1, chunk_code=STREAMPETR_CHUNKS,
    )
    path = _write_notebook("perception/streampetr", nb)
    generated.append(path)
    print(f"[7/11] Generated: {path}  ({len(nb.cells)} cells)")

    # prediction/surroundocc
    nb = _make_prediction_surroundocc_notebook()
    path = _write_notebook("prediction/surroundocc", nb)
    generated.append(path)
    print(f"[8/11] Generated: {path}  ({len(nb.cells)} cells)")

    # prediction/beverse
    nb = _make_prediction_beverse_notebook()
    path = _write_notebook("prediction/beverse", nb)
    generated.append(path)
    print(f"[9/11] Generated: {path}  ({len(nb.cells)} cells)")

    # prediction/flashocc
    nb = _make_prediction_flashocc_notebook()
    path = _write_notebook("prediction/flashocc", nb)
    generated.append(path)
    print(f"[10/11] Generated: {path}  ({len(nb.cells)} cells)")

    # prediction/vip3d
    nb = _make_prediction_vip3d_notebook()
    path = _write_notebook("prediction/vip3d", nb)
    generated.append(path)
    print(f"[11/11] Generated: {path}  ({len(nb.cells)} cells)")

    print(f"\nAll {len(generated)} notebooks generated in {NB_DIR}")


if __name__ == "__main__":
    main()
