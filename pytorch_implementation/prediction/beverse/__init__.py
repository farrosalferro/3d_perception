"""Pure PyTorch forward-only BEVerse-style prediction implementation."""

from .config import (
    BEVerseForwardConfig,
    BackboneNeckConfig,
    beverse_base_forward_config,
    debug_forward_config,
    replace_backbone,
    replace_config,
)
from .metrics import compute_ade_fde, select_best_mode_by_ade
from .model import BEVerseLite

__all__ = [
    "BEVerseForwardConfig",
    "BackboneNeckConfig",
    "beverse_base_forward_config",
    "debug_forward_config",
    "replace_backbone",
    "replace_config",
    "compute_ade_fde",
    "select_best_mode_by_ade",
    "BEVerseLite",
]
