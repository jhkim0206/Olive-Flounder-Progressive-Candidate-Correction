#!/usr/bin/env python
"""Fit train component thresholds and score frozen validation candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from progressive_candidate_correction.evaluation.candidate_metrics import (  # noqa: E402
    apply_class_thresholds,
    calibrate_class_thresholds,
    candidate_location_metrics,
    evaluate_symptom_region_capture,
)


def _load(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _load_protocol(path: str | Path) -> dict:
    protocol_path = Path(path)
    document = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise TypeError("evaluation protocol must be a YAML mapping")
    return document


def _require(archive: dict[str, np.ndarray], names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in archive]
    if missing:
        raise KeyError(f"{label} archive is missing: {', '.join(missing)}")


def _scalar_text(archive: dict[str, np.ndarray], key: str, label: str) -> str:
    value = archive[key]
    if value.shape != ():
        raise ValueError(f"{label} {key} must be a scalar string")
    return str(value.item())


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        required=True,
        help="NPZ with candidate scores and semantic targets",
    )
    parser.add_argument(
        "--validation",
        required=True,
        help="NPZ with candidate scores, semantic targets, Part Maps, and Fish Regions",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol-config",
        default=str(ROOT / "configs" / "evaluation" / "candidate_evaluation.yaml"),
    )
    parser.add_argument(
        "--include-outside-fish-region-for-allowed-part-agreement",
        action="store_true",
        help="Include candidates outside the Fish Region in allowed-part agreement",
    )
    args = parser.parse_args()

    train = _load(args.train)
    validation = _load(args.validation)
    _require(
        train,
        (
            "candidate_scores",
            "candidate_source",
            "checkpoint_sha256",
            "semantic_targets",
        ),
        "training",
    )
    _require(
        validation,
        (
            "candidate_scores",
            "candidate_source",
            "checkpoint_sha256",
            "semantic_targets",
            "part_maps",
            "fish_regions",
        ),
        "validation",
    )
    train_source = _scalar_text(train, "candidate_source", "training")
    validation_source = _scalar_text(validation, "candidate_source", "validation")
    if train_source != validation_source:
        raise ValueError("training and validation archives must contain the same candidate source")
    train_checkpoint = _scalar_text(train, "checkpoint_sha256", "training")
    validation_checkpoint = _scalar_text(validation, "checkpoint_sha256", "validation")
    if train_checkpoint != validation_checkpoint:
        raise ValueError("training and validation archives must come from the same checkpoint")

    settings = _load_protocol(args.protocol_config)
    protocol = dict(settings.get("protocol", {}))
    reporting = dict(settings.get("reporting", {}))
    location_settings = dict(settings.get("candidate_location_metrics", {}))
    allowed_part_agreement = dict(location_settings.get("allowed_part_agreement", {}))
    allowed_parts = {
        int(class_id): tuple(int(part_id) for part_id in part_ids)
        for class_id, part_ids in dict(settings.get("allowed_parts", {})).items()
    }
    coverage = float(protocol.get("component_coverage_requirement", 0.50))
    target_capture_rate = float(protocol.get("target_train_symptom_region_capture_rate", 0.70))
    connectivity = int(protocol.get("component_connectivity", 8))
    minimum_area = int(protocol.get("minimum_component_area", 1))
    reporting_coverages = tuple(
        float(value) for value in reporting.get("component_coverage_thresholds", (0.50, 0.30))
    )
    include_outside = bool(
        allowed_part_agreement.get(
            "include_outside_fish_region_for_allowed_part_agreement",
            False,
        )
    ) or bool(args.include_outside_fish_region_for_allowed_part_agreement)

    thresholds = calibrate_class_thresholds(
        train["candidate_scores"],
        train["semantic_targets"],
        coverage_requirement=coverage,
        target_capture_rate=target_capture_rate,
        connectivity=connectivity,
        minimum_area=minimum_area,
    )
    capture_result = evaluate_symptom_region_capture(
        validation["candidate_scores"],
        validation["semantic_targets"],
        thresholds,
        coverage_requirements=reporting_coverages,
        connectivity=connectivity,
        minimum_area=minimum_area,
    )
    candidates = apply_class_thresholds(
        validation["candidate_scores"],
        thresholds,
    )
    location_metrics = candidate_location_metrics(
        candidates,
        validation["semantic_targets"],
        validation["part_maps"],
        validation["fish_regions"],
        allowed_parts,
        include_outside_fish_region_for_allowed_part_agreement=include_outside,
    )
    report = {
        "protocol": {
            "config": str(Path(args.protocol_config)),
            "candidate_source": train_source,
            "checkpoint_sha256": train_checkpoint,
            "threshold_split": str(protocol.get("calibration_split", "train")),
            "component_coverage": coverage,
            "target_train_symptom_region_capture_rate": target_capture_rate,
            "connectivity": connectivity,
            "minimum_component_area": minimum_area,
        },
        "thresholds": {
            str(class_id): asdict(calibration) for class_id, calibration in thresholds.items()
        },
        "validation": {
            "component_count": capture_result.component_count,
            "symptom_region_capture_rate_by_coverage": {
                str(key): value
                for key, value in capture_result.symptom_region_capture_rate_by_coverage.items()
            },
            "candidate_location_metrics": asdict(location_metrics),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
