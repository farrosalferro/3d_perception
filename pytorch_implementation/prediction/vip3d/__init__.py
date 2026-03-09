"""Pure PyTorch VIP3D-style trajectory prediction module."""

from .config import VIP3DConfig, base_forward_config, debug_forward_config
from .model import VIP3DLite, compute_ade_fde

__all__ = [
    "VIP3DConfig",
    "base_forward_config",
    "debug_forward_config",
    "VIP3DLite",
    "compute_ade_fde",
]
