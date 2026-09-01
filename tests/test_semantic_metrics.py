from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from progressive_candidate_correction.evaluation.metrics import metrics_from_confusion  # noqa: E402


def test_symptom_macro_f1_includes_unpredicted_classes_as_zero() -> None:
    confusion = torch.zeros(9, 9, dtype=torch.float64)
    confusion[0, 0] = 10
    confusion[1, 0] = 5
    confusion[2, 2] = 4

    metrics, per_class = metrics_from_confusion(confusion)

    assert per_class[1]["f1"] == 0.0
    assert per_class[2]["f1"] == 1.0
    assert metrics["symptom_foreground_macro_f1"] == pytest.approx(1.0 / 8.0)
    assert metrics["symptom_foreground_mean_iou"] == pytest.approx(1.0 / 8.0)
