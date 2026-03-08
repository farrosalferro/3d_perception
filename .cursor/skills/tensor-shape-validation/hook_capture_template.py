"""Template for intermediate tensor capture tests."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _register_hook(module, name: str, capture: dict[str, Any], handles: list) -> None:
    def _hook(_module, _inputs, output):
        capture[name] = output

    handles.append(module.register_forward_hook(_hook))


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def test_intermediate_tensors_template():
    """
    Replace this template with model-specific setup:
    - build small debug config
    - create dummy inputs/metadata
    - register hooks for key modules
    - run no_grad forward
    - assert shapes + finite values
    """
    capture: dict[str, Any] = {}
    handles: list = []

    # Example:
    # _register_hook(model.encoder.layer0, "encoder.layer0", capture, handles)
    # with torch.no_grad():
    #     outputs = model(inputs)

    for handle in handles:
        handle.remove()

    # Example shape assertions:
    # assert capture["encoder.layer0"].shape == (batch_size, hw, embed_dims)

    # Example finite checks:
    # for name, value in capture.items():
    #     for tensor in _iter_tensors(value):
    #         assert torch.isfinite(tensor).all(), f"Non-finite values in {name}"
