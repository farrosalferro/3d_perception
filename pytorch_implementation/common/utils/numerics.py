"""Shared numerical helpers."""

from __future__ import annotations

import torch


def inverse_sigmoid(
    x: torch.Tensor,
    eps: float = 1e-5,
    *,
    clamp_input_to_unit: bool = True,
    strict_clamp: bool = False,
) -> torch.Tensor:
    """Numerically stable inverse sigmoid with compatibility modes.

    Args:
        x: Input tensor.
        eps: Numerical floor.
        clamp_input_to_unit: If true, clamp input to [0, 1] before inversion.
        strict_clamp: If true, clamp directly to [eps, 1-eps] then compute
            `log(x / (1 - x))`. This mirrors the stricter variant used in some
            model modules.
    """

    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if strict_clamp:
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

    if clamp_input_to_unit:
        x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)

