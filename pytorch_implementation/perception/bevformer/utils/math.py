"""Math helpers."""

from __future__ import annotations

import torch

from ....common.utils.numerics import inverse_sigmoid as _shared_inverse_sigmoid


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid."""
    return _shared_inverse_sigmoid(x, eps=eps)
