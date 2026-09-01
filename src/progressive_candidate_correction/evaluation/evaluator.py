"""Semantic-segmentation evaluation and training-stage monitors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import SEMANTIC_CLASS_NAMES
from .metrics import confusion_matrix, metrics_from_confusion
from .stage_monitors import StageTrainingMonitorAccumulator


def _autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def resize_float_map(
    tensor: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    tensor = tensor.float()
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor, size=size, mode="nearest")
    return F.interpolate(tensor, size=size, mode=mode, align_corners=False)


def semantic_logits_to_labels(
    logits: torch.Tensor,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert nine-class Auxiliary semantic logits to class labels."""

    if size is not None:
        logits = resize_float_map(logits, size, mode="bilinear")
    if int(logits.shape[1]) != len(SEMANTIC_CLASS_NAMES):
        raise ValueError("semantic logits must contain background and eight symptom classes")
    return logits.argmax(dim=1).long()


def _image_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    image = batch.get("image")
    if not torch.is_tensor(image):
        raise KeyError("batch does not contain an image tensor")
    return image


def _target_from_batch(batch: Mapping[str, Any]) -> torch.Tensor | None:
    target = batch.get("semantic_target")
    if not torch.is_tensor(target):
        return None
    if target.ndim == 4:
        target = target[:, 0]
    return target.long()


class SemanticMetricsAccumulator:
    """Aggregate semantic segmentation metrics over a data loader."""

    def __init__(
        self,
        *,
        num_classes: int = 9,
        class_names: Sequence[str] = SEMANTIC_CLASS_NAMES,
    ) -> None:
        self.num_classes = int(num_classes)
        self.class_names = list(class_names)
        if tuple(self.class_names) != SEMANTIC_CLASS_NAMES:
            raise ValueError("class_names must match the method semantic class order")
        if self.num_classes != len(SEMANTIC_CLASS_NAMES):
            raise ValueError("num_classes must include background and eight symptom classes")
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)
        self.samples = 0
        self.batches = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> None:
        if valid is None:
            valid = (target >= 0) & (target != 255)
        self.confusion += confusion_matrix(prediction, target, self.num_classes, valid)
        self.samples += int(prediction.shape[0])
        self.batches += 1

    def finalize(self) -> dict[str, Any]:
        metrics, class_metrics = metrics_from_confusion(self.confusion, self.class_names)
        metrics["num_eval_samples"] = int(self.samples)
        metrics["num_eval_batches"] = int(self.batches)

        report: dict[str, Any] = dict(metrics)
        report.update(
            {
                "metrics": metrics,
                "per_class": class_metrics,
                "confusion_matrix": self.confusion.long().tolist(),
                "class_names": self.class_names,
                "metric_protocol": {
                    "foreground_macro_absent_class_policy": "include_zero",
                },
            }
        )
        return report


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str = "cuda",
    num_classes: int = 9,
    class_names: Sequence[str] = SEMANTIC_CLASS_NAMES,
    ignore_index: int = -1,
    max_batches: int | None = None,
    use_amp: bool = True,
    stage: str | int | None = "joint_fine_tuning",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a model over one data loader."""

    if config:
        evaluation_config = config.get("evaluation", config) if isinstance(config, Mapping) else {}
        ignore_index = int(evaluation_config.get("ignore_index", ignore_index))
        max_batches = evaluation_config.get("max_batches", max_batches)
        use_amp = bool(evaluation_config.get("use_amp", use_amp))

    device = torch.device(device)
    model.to(device)
    was_training = model.training
    model.eval()
    if stage is not None and hasattr(model, "set_training_stage"):
        model.set_training_stage(stage)
    accumulator = SemanticMetricsAccumulator(
        num_classes=num_classes,
        class_names=class_names,
    )
    # Import lazily so the evaluation package remains usable independently of
    # the training-loop facade during module initialization.
    from ..engine.stages import monitor_for_stage

    active_monitor = monitor_for_stage(stage or "joint_fine_tuning")
    stage_monitor_accumulator = StageTrainingMonitorAccumulator(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=ignore_index,
    )

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        image = _image_from_batch(batch)
        target = _target_from_batch(batch)
        if target is None:
            raise KeyError("evaluation batch does not contain a segmentation target")
        with _autocast(device, bool(use_amp) and device.type == "cuda"):
            outputs = model(image, return_aux=True)
        if torch.is_tensor(outputs):
            logits = outputs
            stage_outputs: Mapping[str, Any] = {"auxiliary_semantic_logits": outputs}
        elif isinstance(outputs, Mapping):
            stage_outputs = outputs
            logits = outputs.get("auxiliary_semantic_logits")
        else:
            logits = None
        if not torch.is_tensor(logits):
            raise KeyError("model output does not contain auxiliary_semantic_logits")

        target_size = tuple(int(value) for value in target.shape[-2:])
        prediction = semantic_logits_to_labels(logits, target_size)
        valid = (target >= 0) & (target != 255)

        accumulator.update(
            prediction,
            target,
            valid=valid,
        )
        stage_monitor_accumulator.update(
            stage_outputs,
            batch,
            target,
            valid=valid,
        )

    report = accumulator.finalize()
    training_monitor_metrics = stage_monitor_accumulator.finalize(active_monitor=active_monitor)
    # Keep training monitor values in their own namespace.
    report["training_monitor_metrics"] = training_monitor_metrics
    report.update(training_monitor_metrics)
    report["metric_protocol"]["semantic_output"] = "Auxiliary semantic mask Y"
    report["metric_protocol"]["training_monitor"] = {
        "active_key": active_monitor,
        "foreground_macro_absent_class_policy": "include_zero",
        "route_classification_domain": "ground_truth_foreground_pixels_only",
    }
    if was_training:
        model.train()
    return report
