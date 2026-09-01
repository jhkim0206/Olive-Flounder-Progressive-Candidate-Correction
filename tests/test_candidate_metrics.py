from __future__ import annotations

import math

import numpy as np

from progressive_candidate_correction.evaluation.candidate_metrics import (
    apply_class_thresholds,
    calibrate_class_thresholds,
    candidate_location_metrics,
    component_critical_score,
    evaluate_symptom_region_capture,
    threshold_for_symptom_region_capture,
)


def test_component_critical_score_is_highest_valid_threshold() -> None:
    assert component_critical_score([0.1, 0.2, 0.3, 0.4], 0.5) == 0.3
    assert component_critical_score([0.1, 0.2, 0.3], 2 / 3) == 0.2


def test_capture_threshold_uses_ceiling_and_preserves_ties() -> None:
    threshold, achieved, count = threshold_for_symptom_region_capture(
        [0.9, 0.8, 0.8, 0.1],
        target_capture_rate=0.5,
    )
    assert threshold == 0.8
    assert achieved == 0.75
    assert count == 4


def test_train_threshold_is_frozen_for_component_evaluation() -> None:
    train_scores = np.asarray(
        [[[[0.9, 0.1, 0.0], [0.8, 0.1, 0.0], [0.0, 0.0, 0.0], [0.5, 0.4, 0.0]]]],
        dtype=np.float32,
    )
    train_target = np.asarray(
        [[[1, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0]]],
        dtype=np.uint8,
    )
    calibrated = calibrate_class_thresholds(
        train_scores,
        train_target,
        coverage_requirement=0.5,
        target_capture_rate=0.5,
    )
    assert calibrated[1].threshold == np.float32(0.8)
    assert calibrated[1].achieved_capture_rate == 0.5

    validation_scores = np.asarray(
        [[[[0.81, 0.79, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 0.0], [0.7, 0.6, 0.0]]]],
        dtype=np.float32,
    )
    candidates = apply_class_thresholds(validation_scores, calibrated)
    assert candidates.sum() == 1
    result = evaluate_symptom_region_capture(
        validation_scores,
        train_target,
        calibrated,
        coverage_requirements=(0.25, 0.5),
    )
    assert result.component_count == 2
    assert result.symptom_region_capture_rate_by_coverage[0.25] == 0.5
    assert result.symptom_region_capture_rate_by_coverage[0.5] == 0.0


def test_allowed_part_agreement_defaults_to_the_fish_region() -> None:
    candidates = np.asarray([[[True, False, True], [False, False, False]]])
    target = np.zeros((2, 3), dtype=np.uint8)
    part_maps = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=np.uint8)
    fish_region = np.asarray([[True, True, False], [True, True, False]])
    allowed_parts = {1: [1]}

    fish_only = candidate_location_metrics(
        candidates,
        target,
        part_maps,
        fish_region,
        allowed_parts,
    )
    all_image = candidate_location_metrics(
        candidates,
        target,
        part_maps,
        fish_region,
        allowed_parts,
        include_outside_fish_region_for_allowed_part_agreement=True,
    )

    assert fish_only.allowed_part_agreement_scope == "within_fish_region"
    assert fish_only.candidate_pixels_evaluated_for_allowed_part_agreement == 1
    assert fish_only.allowed_part_agreement_rate == 1.0
    assert all_image.allowed_part_agreement_scope == "all_image"
    assert all_image.candidate_pixels_evaluated_for_allowed_part_agreement == 2
    assert all_image.allowed_part_agreement_rate == 0.5
    assert fish_only.outside_annotation_candidate_within_fish_region_pixels == 1
    assert fish_only.outside_annotation_candidate_outside_fish_region_pixels == 1
    assert fish_only.outside_annotation_candidate_area_ratio == 0.25


def test_annotated_pixels_are_excluded_from_candidate_location_metrics() -> None:
    candidates = np.ones((1, 2, 2), dtype=bool)
    target = np.asarray([[2, 0], [0, 0]], dtype=np.uint8)
    part_maps = np.ones((2, 2), dtype=np.uint8)
    fish_region = np.ones((2, 2), dtype=bool)
    stats = candidate_location_metrics(
        candidates,
        target,
        part_maps,
        fish_region,
        {1: [1]},
    )
    assert stats.candidate_pixels_evaluated_for_allowed_part_agreement == 3
    assert stats.outside_annotation_candidate_within_fish_region_pixels == 3
    assert math.isclose(stats.outside_annotation_candidate_area_ratio, 0.75)
