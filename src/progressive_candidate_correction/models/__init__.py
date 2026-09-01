"""Public model API."""

from __future__ import annotations

from importlib import import_module

_LAZY = {
    "ProgressiveCandidateCorrectionNetwork": (
        ".network",
        "ProgressiveCandidateCorrectionNetwork",
    ),
    "build_progressive_candidate_correction": (
        ".network",
        "build_progressive_candidate_correction",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(name)
    module_name, attribute = _LAZY[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY)
