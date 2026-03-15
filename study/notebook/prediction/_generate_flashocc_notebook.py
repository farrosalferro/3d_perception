"""Generate prediction/flashocc notebook from markdown source."""

from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf

try:
    from .._builder_primitives import new_notebook as _shared_new_notebook
except ImportError:  # pragma: no cover - script execution fallback
    NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
    if str(NOTEBOOK_ROOT) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_ROOT))
    from _builder_primitives import new_notebook as _shared_new_notebook


def build_notebook(markdown_source: str) -> nbf.NotebookNode:
    nb = _shared_new_notebook()

    setup_code = """\
import os, sys
sys.path.insert(0, os.path.abspath("../.."))

import torch
from pytorch_implementation.prediction.flashocc.config import debug_forward_config
from pytorch_implementation.prediction.flashocc.model import FlashOccLite

cfg = debug_forward_config(num_history=4, pred_horizon=8, bev_h=48, bev_w=64, num_queries=12, num_modes=3, topk=6)
model = FlashOccLite(cfg).eval()
occ_seq = torch.randn(2, cfg.num_history, cfg.backbone.in_channels, cfg.bev_h, cfg.bev_w)

with torch.no_grad():
    outputs = model(occ_seq, decode=False)

print("traj_positions:", tuple(outputs["traj_positions"].shape))
print("mode_logits:", tuple(outputs["mode_logits"].shape))
print("time_stamps:", tuple(outputs["time_stamps"].shape))
"""

    metrics_code = """\
from pytorch_implementation.prediction.flashocc.metrics import average_displacement_error, final_displacement_error

traj = outputs["traj_positions"][:, :, 0]
gt = traj + 0.05 * torch.randn_like(traj)
ade = average_displacement_error(traj, gt)
fde = final_displacement_error(traj, gt)
print("ADE:", float(ade))
print("FDE:", float(fde))
"""

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# FlashOcc Prediction Paper-to-Code Notebook\n\n"
            "Generated from `study/markdown/prediction/flashocc_paper_to_code.md`."
        ),
        nbf.v4.new_code_cell(setup_code),
        nbf.v4.new_markdown_cell(markdown_source),
        nbf.v4.new_code_cell(metrics_code),
    ]
    return nb


def main() -> None:
    workspace = Path(__file__).resolve().parents[3]
    md_path = workspace / "study" / "markdown" / "prediction" / "flashocc_paper_to_code.md"
    nb_path = workspace / "study" / "notebook" / "prediction" / "flashocc_paper_to_code.ipynb"

    markdown_source = md_path.read_text(encoding="utf-8")
    notebook = build_notebook(markdown_source)
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, nb_path)
    print(f"Generated: {nb_path}")


if __name__ == "__main__":
    main()

