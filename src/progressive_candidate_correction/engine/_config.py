"""Configuration helpers used by the training engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def config_get(
    config: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Read one value from the resolved runtime configuration."""

    return config.get(key, default)


def config_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config_get(config, key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)
