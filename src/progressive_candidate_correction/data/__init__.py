"""Dataset contract and optional data loaders."""

from __future__ import annotations

from importlib import import_module

from .contract import (
    BOUNDARY_SDF_CLIP_DISTANCE,
    NUM_PART_CLASSES,
    NUM_SEMANTIC_CLASSES,
    REQUIRED_SAMPLE_KEYS,
    OliveFlounderDataContractError,
    validate_olive_flounder_batch,
    validate_olive_flounder_sample,
)

_LAZY = {
    "OliveFlounderCocoDataset": (".dataset", "OliveFlounderCocoDataset"),
    "validate_dataset": (".dataset", "validate_dataset"),
    "build_olive_flounder_dataloaders": (
        ".dataset",
        "build_olive_flounder_dataloaders",
    ),
    "build_olive_flounder_evaluation_dataloaders": (
        ".dataset",
        "build_olive_flounder_evaluation_dataloaders",
    ),
    "DummyOliveFlounderDataset": (".dummy", "DummyOliveFlounderDataset"),
    "build_dummy_dataloaders": (".dummy", "build_dummy_dataloaders"),
    "build_dummy_evaluation_dataloaders": (
        ".dummy",
        "build_dummy_evaluation_dataloaders",
    ),
    "create_dummy_dataloader": (".dummy", "create_dummy_dataloader"),
    "build_image_only_transform": (".transforms", "build_image_only_transform"),
    "build_spatial_transform": (".transforms", "build_spatial_transform"),
    "normalize_image_to_tensor": (".transforms", "normalize_image_to_tensor"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(name)
    module_name, attribute = _LAZY[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BOUNDARY_SDF_CLIP_DISTANCE",
    "NUM_PART_CLASSES",
    "NUM_SEMANTIC_CLASSES",
    "REQUIRED_SAMPLE_KEYS",
    "OliveFlounderCocoDataset",
    "OliveFlounderDataContractError",
    "DummyOliveFlounderDataset",
    "validate_dataset",
    "build_dummy_dataloaders",
    "build_dummy_evaluation_dataloaders",
    "build_olive_flounder_dataloaders",
    "build_olive_flounder_evaluation_dataloaders",
    "build_image_only_transform",
    "build_spatial_transform",
    "create_dummy_dataloader",
    "normalize_image_to_tensor",
    "validate_olive_flounder_batch",
    "validate_olive_flounder_sample",
]
