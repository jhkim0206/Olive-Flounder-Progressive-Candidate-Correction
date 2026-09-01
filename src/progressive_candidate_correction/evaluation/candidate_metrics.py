"""Train-frozen symptom-region capture evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

COMPONENT_COVERAGE_REQUIREMENT = 0.50
TARGET_SYMPTOM_REGION_CAPTURE_RATE = 0.70
REPORTED_COMPONENT_COVERAGES = (0.50, 0.30)


@dataclass(frozen=True)
class ThresholdCalibration:
    """Frozen threshold and its training-set support."""

    class_id: int
    threshold: float
    achieved_capture_rate: float
    component_count: int
    coverage_requirement: float
    target_capture_rate: float


@dataclass(frozen=True)
class ComponentCoverage:
    """Matching-class coverage for one annotated component."""

    image_index: int
    class_id: int
    component_id: int
    area: int
    coverage: float


@dataclass(frozen=True)
class SymptomRegionCaptureResult:
    """Component coverages and capture rates at requested operating points."""

    component_count: int
    symptom_region_capture_rate_by_coverage: Mapping[float, float]
    components: tuple[ComponentCoverage, ...]


@dataclass(frozen=True)
class CandidateLocationMetrics:
    """Counts behind the two candidate-location measures."""

    allowed_part_candidate_pixels: int
    candidate_pixels_evaluated_for_allowed_part_agreement: int
    allowed_part_agreement_rate: float
    allowed_part_agreement_scope: str
    outside_annotation_candidate_within_fish_region_pixels: int
    outside_annotation_candidate_all_image_pixels: int
    outside_annotation_candidate_outside_fish_region_pixels: int
    fish_region_pixels: int
    outside_annotation_candidate_area_ratio: float


def _fraction(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return result


def _score_batch(
    candidate_scores: Any,
    semantic_targets: Any,
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(candidate_scores)
    target_array = np.asarray(semantic_targets)
    if score_array.ndim == 3 and target_array.ndim == 2:
        score_array = score_array[None, ...]
        target_array = target_array[None, ...]
    if score_array.ndim != 4 or target_array.ndim != 3:
        raise ValueError("expected candidate scores [N,C,H,W] and semantic targets [N,H,W]")
    if (
        score_array.shape[0] != target_array.shape[0]
        or score_array.shape[2:] != target_array.shape[1:]
    ):
        raise ValueError("score and target shapes do not match")
    if not np.isfinite(score_array).all():
        raise ValueError("candidate scores contain non-finite values")
    return score_array, target_array


def _class_ids(class_ids: Sequence[int] | None, channels: int) -> tuple[int, ...]:
    values = tuple(range(1, channels + 1)) if class_ids is None else tuple(map(int, class_ids))
    if len(values) != channels:
        raise ValueError("class_ids must match the number of score channels")
    if len(set(values)) != len(values) or any(class_id <= 0 for class_id in values):
        raise ValueError("class_ids must contain unique foreground IDs")
    return values


def connected_components(mask: Any, connectivity: int = 8) -> tuple[int, np.ndarray]:
    """Label a binary mask using the OpenCV count convention."""

    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError("component masks must be two-dimensional")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    try:
        import cv2

        return cv2.connectedComponents(
            np.ascontiguousarray(foreground, dtype=np.uint8),
            connectivity=connectivity,
        )
    except ImportError:
        pass

    labels = np.zeros(foreground.shape, dtype=np.int32)
    neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        neighbors += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    height, width = foreground.shape
    next_label = 0
    for row, column in np.argwhere(foreground):
        if labels[row, column]:
            continue
        next_label += 1
        labels[row, column] = next_label
        queue = deque([(int(row), int(column))])
        while queue:
            current_row, current_column = queue.popleft()
            for row_offset, column_offset in neighbors:
                neighbor_row = current_row + row_offset
                neighbor_column = current_column + column_offset
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                    and foreground[neighbor_row, neighbor_column]
                    and labels[neighbor_row, neighbor_column] == 0
                ):
                    labels[neighbor_row, neighbor_column] = next_label
                    queue.append((neighbor_row, neighbor_column))
    return next_label + 1, labels


def component_critical_score(
    candidate_scores: Any,
    coverage_requirement: float = COMPONENT_COVERAGE_REQUIREMENT,
) -> float:
    """Find the highest threshold that covers enough of one component."""

    coverage = _fraction(coverage_requirement, "coverage_requirement")
    values = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("nan")
    if not np.isfinite(values).all():
        raise ValueError("component candidate scores contain non-finite values")
    required_pixels = max(1, ceil(coverage * values.size))
    index = values.size - required_pixels
    return float(np.partition(values, index)[index])


def threshold_for_symptom_region_capture(
    critical_scores: Sequence[float],
    target_capture_rate: float = TARGET_SYMPTOM_REGION_CAPTURE_RATE,
) -> tuple[float, float, int]:
    """Choose the largest threshold that reaches the training target."""

    target = _fraction(target_capture_rate, "target_capture_rate")
    values = np.asarray(critical_scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    component_count = int(values.size)
    if component_count == 0:
        return float("nan"), float("nan"), 0
    required = max(1, ceil(target * component_count))
    ordered = np.sort(values)[::-1]
    threshold = float(ordered[required - 1])
    achieved = float(np.mean(values >= threshold))
    return threshold, achieved, component_count


def component_critical_scores(
    class_candidate_scores: Any,
    semantic_target: Any,
    class_id: int,
    *,
    coverage_requirement: float = COMPONENT_COVERAGE_REQUIREMENT,
    connectivity: int = 8,
    minimum_area: int = 1,
) -> list[float]:
    """Collect critical scores for one class in one image."""

    score_array = np.asarray(class_candidate_scores)
    target_array = np.asarray(semantic_target)
    if score_array.shape != target_array.shape or score_array.ndim != 2:
        raise ValueError("class candidate scores and semantic target must be matching 2-D arrays")
    if int(minimum_area) < 1:
        raise ValueError("minimum_area must be positive")
    count, labels = connected_components(target_array == int(class_id), connectivity)
    result = []
    for component_id in range(1, count):
        component = labels == component_id
        if int(component.sum()) < int(minimum_area):
            continue
        result.append(component_critical_score(score_array[component], coverage_requirement))
    return result


def calibrate_class_thresholds(
    train_candidate_scores: Any,
    train_semantic_targets: Any,
    *,
    class_ids: Sequence[int] | None = None,
    coverage_requirement: float = COMPONENT_COVERAGE_REQUIREMENT,
    target_capture_rate: float = TARGET_SYMPTOM_REGION_CAPTURE_RATE,
    connectivity: int = 8,
    minimum_area: int = 1,
) -> dict[int, ThresholdCalibration]:
    """Calibrate class thresholds from training components only."""

    scores, targets = _score_batch(train_candidate_scores, train_semantic_targets)
    ids = _class_ids(class_ids, scores.shape[1])
    calibrations: dict[int, ThresholdCalibration] = {}
    for channel, class_id in enumerate(ids):
        critical = []
        for image_index in range(scores.shape[0]):
            critical.extend(
                component_critical_scores(
                    scores[image_index, channel],
                    targets[image_index],
                    class_id,
                    coverage_requirement=coverage_requirement,
                    connectivity=connectivity,
                    minimum_area=minimum_area,
                )
            )
        threshold, achieved, count = threshold_for_symptom_region_capture(
            critical,
            target_capture_rate,
        )
        if count == 0:
            raise ValueError(f"class {class_id} has no training components")
        calibrations[class_id] = ThresholdCalibration(
            class_id=class_id,
            threshold=threshold,
            achieved_capture_rate=achieved,
            component_count=count,
            coverage_requirement=float(coverage_requirement),
            target_capture_rate=float(target_capture_rate),
        )
    return calibrations


def apply_class_thresholds(
    candidate_scores: Any,
    thresholds: Mapping[int, float | ThresholdCalibration],
    *,
    class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply frozen class thresholds to foreground score maps."""

    score_array = np.asarray(candidate_scores)
    squeeze = score_array.ndim == 3
    if squeeze:
        score_array = score_array[None, ...]
    if score_array.ndim != 4:
        raise ValueError("expected candidate scores [N,C,H,W] or [C,H,W]")
    ids = _class_ids(class_ids, score_array.shape[1])
    values = []
    for class_id in ids:
        if class_id not in thresholds:
            raise KeyError(f"missing threshold for class {class_id}")
        threshold = thresholds[class_id]
        values.append(
            float(threshold.threshold if isinstance(threshold, ThresholdCalibration) else threshold)
        )
    threshold_array = np.asarray(values, dtype=score_array.dtype)[None, :, None, None]
    candidates = score_array >= threshold_array
    return candidates[0] if squeeze else candidates


def evaluate_symptom_region_capture(
    candidate_scores: Any,
    semantic_targets: Any,
    thresholds: Mapping[int, float | ThresholdCalibration],
    *,
    class_ids: Sequence[int] | None = None,
    coverage_requirements: Sequence[float] = REPORTED_COMPONENT_COVERAGES,
    connectivity: int = 8,
    minimum_area: int = 1,
) -> SymptomRegionCaptureResult:
    """Evaluate frozen thresholds without validation retuning."""

    score_array, target_array = _score_batch(candidate_scores, semantic_targets)
    ids = _class_ids(class_ids, score_array.shape[1])
    requested = tuple(_fraction(value, "coverage requirement") for value in coverage_requirements)
    candidates = apply_class_thresholds(score_array, thresholds, class_ids=ids)
    rows = []
    for image_index, target in enumerate(target_array):
        for channel, class_id in enumerate(ids):
            count, labels = connected_components(target == class_id, connectivity)
            for component_id in range(1, count):
                component = labels == component_id
                area = int(component.sum())
                if area < int(minimum_area):
                    continue
                coverage = float(candidates[image_index, channel][component].mean())
                rows.append(
                    ComponentCoverage(
                        image_index=image_index,
                        class_id=class_id,
                        component_id=component_id,
                        area=area,
                        coverage=coverage,
                    )
                )
    symptom_region_capture_rate = {
        requirement: (
            float(np.mean([row.coverage >= requirement for row in rows])) if rows else float("nan")
        )
        for requirement in requested
    }
    return SymptomRegionCaptureResult(
        component_count=len(rows),
        symptom_region_capture_rate_by_coverage=symptom_region_capture_rate,
        components=tuple(rows),
    )


def candidate_location_metrics(
    candidates: Any,
    semantic_targets: Any,
    part_maps: Any,
    fish_regions: Any,
    allowed_parts: Mapping[int, Sequence[int]],
    *,
    class_ids: Sequence[int] | None = None,
    include_outside_fish_region_for_allowed_part_agreement: bool = False,
) -> CandidateLocationMetrics:
    """Measure allowed-part agreement and outside-annotation candidate area."""

    candidate_array = np.asarray(candidates, dtype=bool)
    target_array = np.asarray(semantic_targets)
    part_map_array = np.asarray(part_maps)
    fish_region_array = np.asarray(fish_regions, dtype=bool)
    if candidate_array.ndim == 3 and target_array.ndim == 2:
        candidate_array = candidate_array[None, ...]
        target_array = target_array[None, ...]
        part_map_array = part_map_array[None, ...]
        fish_region_array = fish_region_array[None, ...]
    if candidate_array.ndim != 4:
        raise ValueError("expected candidates [N,C,H,W] or [C,H,W]")
    expected_shape = (candidate_array.shape[0], *candidate_array.shape[2:])
    if any(
        array.shape != expected_shape for array in (target_array, part_map_array, fish_region_array)
    ):
        raise ValueError(
            "candidate, semantic target, Part Map, and Fish Region shapes do not match"
        )
    ids = _class_ids(class_ids, candidate_array.shape[1])

    annotated_symptom_foreground = target_array > 0
    outside_annotation_candidate_all = candidate_array.any(axis=1) & ~annotated_symptom_foreground
    outside_annotation_candidate_within_fish_region = (
        outside_annotation_candidate_all & fish_region_array
    )
    outside_annotation_candidate_outside_fish_region = (
        outside_annotation_candidate_all & ~fish_region_array
    )

    candidate_pixels_for_allowed_part_agreement = 0
    allowed_part_candidate_pixels = 0
    for channel, class_id in enumerate(ids):
        if class_id not in allowed_parts:
            raise KeyError(f"missing allowed parts for class {class_id}")
        outside_annotation_candidate = candidate_array[:, channel] & ~annotated_symptom_foreground
        if not include_outside_fish_region_for_allowed_part_agreement:
            outside_annotation_candidate &= fish_region_array
        allowed = np.isin(part_map_array, np.asarray(allowed_parts[class_id]))
        candidate_pixels_for_allowed_part_agreement += int(outside_annotation_candidate.sum())
        allowed_part_candidate_pixels += int((outside_annotation_candidate & allowed).sum())

    fish_region_pixels = int(fish_region_array.sum())
    allowed_part_agreement_rate = (
        float(allowed_part_candidate_pixels / candidate_pixels_for_allowed_part_agreement)
        if candidate_pixels_for_allowed_part_agreement
        else float("nan")
    )
    outside_annotation_candidate_area_ratio = (
        float(outside_annotation_candidate_within_fish_region.sum() / fish_region_pixels)
        if fish_region_pixels
        else float("nan")
    )
    return CandidateLocationMetrics(
        allowed_part_candidate_pixels=allowed_part_candidate_pixels,
        candidate_pixels_evaluated_for_allowed_part_agreement=(
            candidate_pixels_for_allowed_part_agreement
        ),
        allowed_part_agreement_rate=allowed_part_agreement_rate,
        allowed_part_agreement_scope=(
            "all_image"
            if include_outside_fish_region_for_allowed_part_agreement
            else "within_fish_region"
        ),
        outside_annotation_candidate_within_fish_region_pixels=int(
            outside_annotation_candidate_within_fish_region.sum()
        ),
        outside_annotation_candidate_all_image_pixels=int(outside_annotation_candidate_all.sum()),
        outside_annotation_candidate_outside_fish_region_pixels=int(
            outside_annotation_candidate_outside_fish_region.sum()
        ),
        fish_region_pixels=fish_region_pixels,
        outside_annotation_candidate_area_ratio=outside_annotation_candidate_area_ratio,
    )


__all__ = [
    "CandidateLocationMetrics",
    "ComponentCoverage",
    "SymptomRegionCaptureResult",
    "COMPONENT_COVERAGE_REQUIREMENT",
    "REPORTED_COMPONENT_COVERAGES",
    "TARGET_SYMPTOM_REGION_CAPTURE_RATE",
    "ThresholdCalibration",
    "candidate_location_metrics",
    "apply_class_thresholds",
    "calibrate_class_thresholds",
    "component_critical_score",
    "component_critical_scores",
    "connected_components",
    "evaluate_symptom_region_capture",
    "threshold_for_symptom_region_capture",
]
