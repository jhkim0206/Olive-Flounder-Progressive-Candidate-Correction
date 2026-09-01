from __future__ import annotations

import json

import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("PIL")

from progressive_candidate_correction.data.contract import REQUIRED_SAMPLE_KEYS
from progressive_candidate_correction.data.dataset import (
    OliveFlounderCocoDataset,
    build_olive_flounder_dataloaders,
    build_olive_flounder_evaluation_dataloaders,
    load_coco_records,
)
from progressive_candidate_correction.schema import (
    ROUTE_NAMES,
    SEMANTIC_CLASS_NAMES,
    SEMANTIC_TO_ROUTE,
    ZONE_LABEL_NAMES,
)


def test_dataset_interface_uses_paper_terms() -> None:
    assert OliveFlounderCocoDataset.__name__ == "OliveFlounderCocoDataset"
    assert "symptom_foreground_target" in REQUIRED_SAMPLE_KEYS
    assert "unaffected_surface_target" in REQUIRED_SAMPLE_KEYS
    assert "unaffected_surface_valid" in REQUIRED_SAMPLE_KEYS


def test_zone_and_route_order_matches_the_paper() -> None:
    assert ZONE_LABEL_NAMES == (
        "background",
        "body",
        "mouth",
        "fin_tip",
        "fin_middle",
        "fin_base",
        "caudal_fin_tip",
        "caudal_fin_middle",
        "caudal_fin_base",
    )
    assert ROUTE_NAMES == ("body", "mouth", "fin", "caudal_fin")
    assert SEMANTIC_TO_ROUTE == (0, 0, 1, 2, 2, 2, 3, 3, 3)


def test_evaluation_loader_separates_annotation_split_from_transform_mode(
    monkeypatch,
    tmp_path,
) -> None:
    from torch.utils.data import RandomSampler, SequentialSampler

    import progressive_candidate_correction.data.dataset as dataset_module

    transform_modes = []
    monkeypatch.setattr(
        dataset_module,
        "load_coco_records",
        lambda *_args, **_kwargs: ({}, [{}]),
    )
    monkeypatch.setattr(
        dataset_module,
        "build_semantic_mask",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        dataset_module,
        "build_spatial_transform",
        lambda training, _size: transform_modes.append(bool(training)) or object(),
    )
    monkeypatch.setattr(
        dataset_module,
        "build_image_only_transform",
        lambda training: object(),
    )

    training_loaders = build_olive_flounder_dataloaders(
        tmp_path,
        batch_size=1,
        num_workers=0,
    )
    evaluation_loaders = build_olive_flounder_evaluation_dataloaders(
        tmp_path,
        batch_size=1,
        num_workers=0,
    )

    train_loader, validation_loader = training_loaders
    train_evaluation_loader, validation_evaluation_loader = evaluation_loaders
    assert train_loader.dataset.split == "train"
    assert train_loader.dataset.training is True
    assert validation_loader.dataset.split == "val"
    assert validation_loader.dataset.training is False
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)

    assert train_evaluation_loader.dataset.split == "train"
    assert train_evaluation_loader.dataset.training is False
    assert validation_evaluation_loader.dataset.split == "val"
    assert validation_evaluation_loader.dataset.training is False
    assert isinstance(train_evaluation_loader.sampler, SequentialSampler)
    assert isinstance(validation_evaluation_loader.sampler, SequentialSampler)
    assert transform_modes == [True, False, False, False]


def test_fish_id_is_required(tmp_path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    document = {
        "images": [
            {
                "id": 1,
                "file_name": "sample.png",
                "width": 32,
                "height": 32,
            }
        ],
        "annotations": [],
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(SEMANTIC_CLASS_NAMES[1:], start=1)
        ],
    }
    (annotations / "train_annotations.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="missing a fish_id"):
        load_coco_records(tmp_path, "train")
