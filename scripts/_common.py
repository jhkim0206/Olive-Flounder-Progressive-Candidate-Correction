"""Shared command-line adapters for experiment configuration."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from progressive_candidate_correction.config import (  # noqa: E402
    build_criterion,
    build_model,
)
from progressive_candidate_correction.config import (  # noqa: E402
    load_config as load_experiment_config,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Delegate all inheritance and preset resolution to the package API."""

    return load_experiment_config(path)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def data_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(config.get("data"))
    training = _mapping(config.get("training"))
    provider = str(data.get("provider", "olive_flounder_coco"))
    if provider == "olive_flounder_coco":
        root = data.get("root")
        if not root:
            raise ValueError("Set data.root in the config or pass --data-root")
        return {
            "root": root,
            "image_size": tuple(data.get("image_size", (384, 384))),
            "batch_size": int(training.get("batch_size", 4)),
            "seed": int(_mapping(config.get("experiment")).get("seed", 45)),
            "num_workers": int(data.get("num_workers", 2)),
        }
    if provider != "dummy":
        raise ValueError(f"Unknown data provider: {provider}")
    dummy = _mapping(data.get("dummy"))
    sample_count = int(dummy.get("num_samples", 4))
    return {
        "num_samples": sample_count,
        "image_size": tuple(data.get("image_size", (64, 64))),
        "batch_size": int(dummy.get("batch_size", 2)),
        "seed": int(dummy.get("deterministic_seed", 45)),
    }


def instantiate_model(
    config: Mapping[str, Any],
    *,
    pretrained_override: bool | None = None,
):
    overrides = {}
    if pretrained_override is not None:
        overrides["pretrained"] = bool(pretrained_override)
    return build_model(config, **overrides)


def instantiate_loss(
    config: Mapping[str, Any],
    **overrides: Any,
):
    return build_criterion(config, **overrides)


def instantiate_loaders(config: Mapping[str, Any]):
    from progressive_candidate_correction.data import (
        build_dummy_dataloaders,
        build_olive_flounder_dataloaders,
    )

    provider = str(_mapping(config.get("data")).get("provider", "olive_flounder_coco"))
    factory = build_dummy_dataloaders if provider == "dummy" else build_olive_flounder_dataloaders
    kwargs = data_kwargs(config)
    train_loader, validation_loader = factory(**kwargs)
    return {"train": train_loader, "val": validation_loader}


def instantiate_evaluation_loaders(config: Mapping[str, Any]):
    """Build non-shuffled loaders with evaluation preprocessing."""

    from progressive_candidate_correction.data import (
        build_dummy_evaluation_dataloaders,
        build_olive_flounder_evaluation_dataloaders,
    )

    provider = str(_mapping(config.get("data")).get("provider", "olive_flounder_coco"))
    factory = (
        build_dummy_evaluation_dataloaders
        if provider == "dummy"
        else build_olive_flounder_evaluation_dataloaders
    )
    train_loader, validation_loader = factory(**data_kwargs(config))
    return {"train": train_loader, "val": validation_loader}


def choose_device(value: str):
    import torch

    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def class_names(config: Mapping[str, Any]) -> list[str]:
    names = _mapping(config.get("data")).get("class_names", [])
    if not names:
        raise ValueError("data.class_names must be present in the configuration")
    return [str(name) for name in names]


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np

        if isinstance(value, np.integer | np.floating):
            scalar = value.item()
            return scalar if not isinstance(scalar, float) or math.isfinite(scalar) else None
    except ImportError:
        pass
    return value
