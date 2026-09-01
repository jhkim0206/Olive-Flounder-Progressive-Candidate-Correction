"""Semantic segmentation and allowed-part metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..schema import SEMANTIC_CLASS_NAMES


def confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a target-row/prediction-column confusion matrix."""

    predicted = prediction.detach().long().flatten().cpu()
    expected = target.detach().long().flatten().cpu()
    keep = (
        (expected >= 0)
        & (expected != 255)
        & (expected < int(num_classes))
        & (predicted >= 0)
        & (predicted < int(num_classes))
    )
    if valid is not None:
        valid_flat = valid.detach().flatten().cpu() > 0.5
        if valid_flat.numel() != keep.numel():
            raise ValueError("valid mask and target must have the same number of pixels")
        keep &= valid_flat
    if not bool(keep.any()):
        return torch.zeros(num_classes, num_classes, dtype=torch.float64)
    encoded = expected[keep] * int(num_classes) + predicted[keep]
    return (
        torch.bincount(
            encoded,
            minlength=int(num_classes) * int(num_classes),
        )
        .reshape(num_classes, num_classes)
        .double()
    )


def metrics_from_confusion(
    confusion: torch.Tensor | np.ndarray,
    class_names: Sequence[str] = SEMANTIC_CLASS_NAMES,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute per-class, macro, and binary-foreground metrics."""

    matrix = torch.as_tensor(confusion, dtype=torch.float64).cpu()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    num_classes = int(matrix.shape[0])
    names = list(class_names)
    if tuple(names) != SEMANTIC_CLASS_NAMES or num_classes != len(SEMANTIC_CLASS_NAMES):
        raise ValueError("class_names must match the method semantic class order")

    true_positive = matrix.diag()
    target_count = matrix.sum(dim=1)
    prediction_count = matrix.sum(dim=0)
    union = target_count + prediction_count - true_positive
    iou = torch.where(
        union > 0,
        true_positive / union.clamp_min(1e-7),
        torch.zeros_like(true_positive),
    )
    f1_denominator = prediction_count + target_count
    f1 = torch.where(
        f1_denominator > 0,
        2.0 * true_positive / f1_denominator.clamp_min(1e-7),
        torch.zeros_like(true_positive),
    )

    foreground_true_positive = float(matrix[1:, 1:].sum().item()) if num_classes > 1 else 0.0
    foreground_false_positive = float(matrix[0, 1:].sum().item()) if num_classes > 1 else 0.0
    foreground_false_negative = float(matrix[1:, 0].sum().item()) if num_classes > 1 else 0.0
    symptom_foreground_precision = foreground_true_positive / max(
        foreground_true_positive + foreground_false_positive, 1e-7
    )
    symptom_foreground_recall = foreground_true_positive / max(
        foreground_true_positive + foreground_false_negative, 1e-7
    )
    symptom_foreground_f1 = (
        2.0
        * symptom_foreground_precision
        * symptom_foreground_recall
        / max(symptom_foreground_precision + symptom_foreground_recall, 1e-7)
    )
    symptom_foreground_iou = foreground_true_positive / max(
        foreground_true_positive + foreground_false_positive + foreground_false_negative,
        1e-7,
    )

    metrics = {
        "pixel_accuracy": float(true_positive.sum().item() / max(matrix.sum().item(), 1.0)),
        "mean_iou": float(iou.mean().item()),
        "symptom_foreground_mean_iou": float(iou[1:].mean().item()),
        "macro_f1": float(f1.mean().item()),
        "symptom_foreground_macro_f1": float(f1[1:].mean().item()),
        "symptom_foreground_precision": float(symptom_foreground_precision),
        "symptom_foreground_recall": float(symptom_foreground_recall),
        "symptom_foreground_f1": float(symptom_foreground_f1),
        "symptom_foreground_iou": float(symptom_foreground_iou),
    }
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        class_iou = float(iou[index].item())
        class_f1 = float(f1[index].item())
        metrics[f"class_{index}_{name}_iou"] = class_iou
        metrics[f"class_{index}_{name}_f1"] = class_f1
        per_class.append(
            {
                "class_id": index,
                "class_name": name,
                "iou": class_iou,
                "f1": class_f1,
                "support_pixels": int(target_count[index].item()),
                "predicted_pixels": int(prediction_count[index].item()),
            }
        )
    return metrics, per_class
