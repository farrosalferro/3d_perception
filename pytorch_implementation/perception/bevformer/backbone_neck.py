"""A lightweight backbone + FPN stack for standalone BEVFormer forward tests."""

from __future__ import annotations

from ...common.backbone.simple_backbone_fpn import BackboneNeck, SimpleBackbone, SimpleFPN, _ConvBlock

__all__ = ["_ConvBlock", "SimpleBackbone", "SimpleFPN", "BackboneNeck"]
