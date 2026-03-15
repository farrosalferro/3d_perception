"""Reusable hook registration helpers for model instrumentation tests."""

from __future__ import annotations

from typing import Any


def register_hook_overwrite(module, name: str, capture: dict[str, Any], handles: list) -> None:
    def _hook(_module, _inputs, output):
        capture[name] = output

    handles.append(module.register_forward_hook(_hook))


def register_hook_append(module, name: str, capture: dict[str, list[Any]], handles: list) -> None:
    def _hook(_module, _inputs, output):
        capture.setdefault(name, []).append(output)

    handles.append(module.register_forward_hook(_hook))

