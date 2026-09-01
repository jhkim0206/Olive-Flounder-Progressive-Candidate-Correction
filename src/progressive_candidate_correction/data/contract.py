"""Tensor contract shared by the dataset, model, and training loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import NUM_PART_CLASSES, NUM_SEMANTIC_CLASSES

BOUNDARY_SDF_CLIP_DISTANCE = 16.0
REQUIRED_SAMPLE_KEYS = (
    "image",
    "semantic_target",
    "fish_region_target",
    "part_map_target",
    "symptom_foreground_target",
    "unaffected_surface_target",
    "signed_distance_target",
    "head_to_tail_direction_target",
    "zone_map_target",
    "semantic_valid",
    "structure_valid",
    "positive_evidence_valid",
    "unaffected_surface_valid",
    "boundary_valid",
    "route_valid",
    "file_name",
    "fish_id",
    "image_id",
)

LOSS_TARGET_KEYS = (
    "semantic_target",
    "fish_region_target",
    "part_map_target",
    "symptom_foreground_target",
    "unaffected_surface_target",
    "signed_distance_target",
    "head_to_tail_direction_target",
    "zone_map_target",
    "semantic_valid",
    "structure_valid",
    "positive_evidence_valid",
    "unaffected_surface_valid",
    "boundary_valid",
    "route_valid",
)


class OliveFlounderDataContractError(ValueError):
    """Raised when a sample does not satisfy the dataset tensor interface."""


def _torch():
    try:
        import torch
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise ImportError("Data validation requires PyTorch") from exc
    return torch


def _check_tensor(sample: Mapping[str, Any], key: str, ndim: int) -> None:
    torch = _torch()
    value = sample[key]
    if not torch.is_tensor(value) or value.ndim != ndim:
        raise OliveFlounderDataContractError(f"{key} must be a {ndim}-D tensor")


def validate_olive_flounder_sample(sample: Mapping[str, Any]) -> None:
    """Check one item returned by ``OliveFlounderCocoDataset``."""

    missing = [key for key in REQUIRED_SAMPLE_KEYS if key not in sample]
    if missing:
        raise OliveFlounderDataContractError(f"missing sample keys: {missing}")

    _check_tensor(sample, "image", 3)
    _check_tensor(sample, "semantic_target", 2)
    _check_tensor(sample, "part_map_target", 2)
    _check_tensor(sample, "zone_map_target", 2)
    for key in (
        "fish_region_target",
        "symptom_foreground_target",
        "unaffected_surface_target",
        "signed_distance_target",
        "head_to_tail_direction_target",
        "semantic_valid",
        "structure_valid",
        "positive_evidence_valid",
        "unaffected_surface_valid",
        "boundary_valid",
        "route_valid",
    ):
        _check_tensor(sample, key, 3)

    height, width = sample["semantic_target"].shape
    for key in ("part_map_target", "zone_map_target"):
        if tuple(sample[key].shape) != (height, width):
            raise OliveFlounderDataContractError(f"{key} has a different spatial size")
    if tuple(sample["image"].shape[1:]) != (height, width):
        raise OliveFlounderDataContractError("image and labels have different spatial sizes")

    torch = _torch()
    semantic = sample["semantic_target"]
    part = sample["part_map_target"]
    if bool(((semantic < 0) | (semantic >= NUM_SEMANTIC_CLASSES)).any().item()):
        raise OliveFlounderDataContractError("semantic_target contains an unknown class ID")
    if bool(((part < 0) | (part >= NUM_PART_CLASSES)).any().item()):
        raise OliveFlounderDataContractError("part_map_target contains an unknown part ID")
    if not torch.equal(part > 0, sample["fish_region_target"].squeeze(0) > 0.5):
        raise OliveFlounderDataContractError("Part Map foreground must equal the Fish Region")


def validate_olive_flounder_batch(batch: Mapping[str, Any]) -> None:
    """Check the leading dimensions of a collated batch."""

    missing = [key for key in REQUIRED_SAMPLE_KEYS if key not in batch]
    if missing:
        raise OliveFlounderDataContractError(f"missing batch keys: {missing}")
    _check_tensor(batch, "image", 4)
    _check_tensor(batch, "semantic_target", 3)
    _check_tensor(batch, "part_map_target", 3)
    batch_size = int(batch["image"].shape[0])
    if int(batch["semantic_target"].shape[0]) != batch_size:
        raise OliveFlounderDataContractError("batch dimensions do not match")
