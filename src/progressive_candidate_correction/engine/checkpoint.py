"""Checkpoint input and output."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    *,
    epoch: int = 0,
    stage: str = "joint_fine_tuning",
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    monitor_state: Mapping[str, Any] | None = None,
    checkpoint_role: str | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "olive-flounder-progressive-candidate-correction",
        "format_version": 1,
        "epoch": int(epoch),
        "stage": str(stage),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": (
            scheduler.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict")
            else None
        ),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "cfg": copy.deepcopy(dict(config or {})),
        "metrics": copy.deepcopy(dict(metrics or {})),
        "history": copy.deepcopy(history or []),
        "monitor_state": copy.deepcopy(dict(monitor_state or {})),
        "checkpoint_role": checkpoint_role,
    }
    torch.save(payload, str(output_path))
    return output_path


def extract_state_dict(checkpoint: Any, *, prefer_ema: bool = False) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint must be dict-like")
    ema_keys = ("ema_state", "ema_model_state", "ema_state_dict", "model_ema", "ema")
    raw_keys = ("model_state", "model_state_dict", "state_dict", "model", "net", "network")
    ordered_keys = ema_keys + raw_keys if prefer_ema else raw_keys + ema_keys
    for key in ordered_keys:
        state = checkpoint.get(key)
        if isinstance(state, Mapping) and any(torch.is_tensor(value) for value in state.values()):
            return dict(state)
    if any(torch.is_tensor(value) for value in checkpoint.values()):
        return dict(checkpoint)
    raise RuntimeError("Could not find model state_dict in checkpoint")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
    prefer_ema: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location=map_location)
    state_dict = extract_state_dict(checkpoint, prefer_ema=prefer_ema)
    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key).replace("module.", "", 1): value for key, value in state_dict.items()
        }
    result = model.load_state_dict(state_dict, strict=bool(strict))

    if isinstance(checkpoint, Mapping):
        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if (
            scheduler is not None
            and checkpoint.get("scheduler_state") is not None
            and hasattr(scheduler, "load_state_dict")
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

    return {
        "checkpoint": checkpoint,
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
    }
