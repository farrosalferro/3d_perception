"""Pure PyTorch forward-only MapTR implementation."""

from .config import MapTRForwardConfig, debug_forward_config, maptr_tiny_forward_config
from .model import MapTRLite

__all__ = [
    "MapTRForwardConfig",
    "maptr_tiny_forward_config",
    "debug_forward_config",
    "MapTRLite",
]
