"""Progressive Candidate Correction for visible symptoms in olive flounder."""

from __future__ import annotations

from importlib import import_module

from .config import load_config, save_resolved_config

__version__ = "0.1.0"

_LAZY = {
    "ProgressiveCandidateCorrectionNetwork": (
        ".models",
        "ProgressiveCandidateCorrectionNetwork",
    ),
    "ProgressiveCandidateCorrectionLoss": (
        ".losses",
        "ProgressiveCandidateCorrectionLoss",
    ),
    "build_progressive_candidate_correction": (
        ".models",
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


__all__ = [
    "ProgressiveCandidateCorrectionLoss",
    "ProgressiveCandidateCorrectionNetwork",
    "build_progressive_candidate_correction",
    "load_config",
    "save_resolved_config",
]
