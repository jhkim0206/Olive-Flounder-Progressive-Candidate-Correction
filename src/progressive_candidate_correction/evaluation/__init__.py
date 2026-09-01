"""Candidate metrics plus optional torch-based evaluation."""

from importlib import import_module

from .candidate_metrics import (
    COMPONENT_COVERAGE_REQUIREMENT,
    REPORTED_COMPONENT_COVERAGES,
    TARGET_SYMPTOM_REGION_CAPTURE_RATE,
    CandidateLocationMetrics,
    ComponentCoverage,
    SymptomRegionCaptureResult,
    ThresholdCalibration,
    apply_class_thresholds,
    calibrate_class_thresholds,
    candidate_location_metrics,
    component_critical_score,
    component_critical_scores,
    connected_components,
    evaluate_symptom_region_capture,
    threshold_for_symptom_region_capture,
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
    "evaluate_model",
    "semantic_logits_to_labels",
]

_LAZY = {
    "evaluate_model": (".evaluator", "evaluate_model"),
    "semantic_logits_to_labels": (".evaluator", "semantic_logits_to_labels"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(name)
    module_name, attribute = _LAZY[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
