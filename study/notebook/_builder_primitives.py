"""Shared notebook generation primitives."""

from __future__ import annotations

import nbformat as nbf


def new_notebook() -> nbf.NotebookNode:
    """Create a notebook with the project kernel metadata."""

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (3d_perception)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    return nb

