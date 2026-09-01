"""Compact validation monitors for the six training stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from ..schema import SEMANTIC_CLASS_NAMES, SEMANTIC_TO_ROUTE, VISUAL_EVIDENCE_NAMES


def _resize_labels(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if value.ndim == 4:
        value = value[:, 0]
    if value.shape[-2:] == size:
        return value.long()
    return F.interpolate(value.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()


def _resize_mask(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.shape[-2:] == size:
        return value.float()
    return F.interpolate(value.float(), size=size, mode="nearest")


def _route_target(segmentation: torch.Tensor) -> torch.Tensor:
    lookup = segmentation.new_tensor(SEMANTIC_TO_ROUTE)
    return lookup[segmentation.clamp(0, 8)]


class StageTrainingMonitorAccumulator:
    """Accumulate normalized scores used for stage-local checkpoint selection."""

    def __init__(self, num_classes: int, class_names: list[str], ignore_index: int = 255):
        self.num_classes = int(num_classes)
        self.class_names = tuple(class_names)
        if self.num_classes != len(SEMANTIC_CLASS_NAMES):
            raise ValueError("num_classes must include background and eight symptom classes")
        if self.class_names != SEMANTIC_CLASS_NAMES:
            raise ValueError("class_names must match the method semantic class order")
        self.ignore_index = int(ignore_index)
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.samples = 0
        self.batches = 0

    def _add(self, name: str, value: torch.Tensor | float) -> None:
        scalar = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
        self.sums[name] = self.sums.get(name, 0.0) + scalar
        self.counts[name] = self.counts.get(name, 0) + 1

    def update(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        target: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> None:
        self.batches += 1
        self.samples += int(target.shape[0])

        auxiliary_logits = output.get("auxiliary_semantic_logits")
        if torch.is_tensor(auxiliary_logits):
            size = tuple(auxiliary_logits.shape[-2:])
            labels = _resize_labels(target, size)
            valid_pixels = labels != self.ignore_index
            if valid is not None:
                valid_pixels &= _resize_labels(valid.long(), size).bool()
            prediction = auxiliary_logits.argmax(dim=1)
            if bool(valid_pixels.any()):
                self._add(
                    "semantic_accuracy",
                    (prediction[valid_pixels] == labels[valid_pixels]).float().mean(),
                )

        fish_region = output.get("fish_region")
        fish_region_target = batch.get("fish_region_target")
        if torch.is_tensor(fish_region) and torch.is_tensor(fish_region_target):
            truth = (
                _resize_mask(
                    fish_region_target,
                    tuple(fish_region.shape[-2:]),
                )
                > 0.5
            )
            prediction = fish_region > 0.5
            intersection = (prediction & truth).sum().float()
            union = (prediction | truth).sum().float().clamp_min(1.0)
            self._add("fish_region_iou", intersection / union)

        part_map = output.get("part_map")
        part_map_target = batch.get("part_map_target")
        if torch.is_tensor(part_map) and torch.is_tensor(part_map_target):
            labels = _resize_labels(part_map_target, tuple(part_map.shape[-2:]))
            fish = labels > 0
            if bool(fish.any()):
                self._add(
                    "part_map_accuracy",
                    (part_map.argmax(dim=1)[fish] == labels[fish]).float().mean(),
                )

        visual_evidence = output.get("visual_evidence")
        symptom_foreground = batch.get("symptom_foreground_target")
        if torch.is_tensor(visual_evidence) and torch.is_tensor(symptom_foreground):
            if visual_evidence.shape[1] != len(VISUAL_EVIDENCE_NAMES):
                raise ValueError("visual_evidence must contain the four method channels")
            foreground = _resize_mask(symptom_foreground, tuple(visual_evidence.shape[-2:]))
            positive = visual_evidence[:, :3].amax(dim=1, keepdim=True)
            unaffected = visual_evidence[:, 3:4]
            alignment = (
                1.0 - ((positive - foreground).abs() + (unaffected * foreground).abs()).mean() / 2.0
            )
            self._add("visual_evidence_alignment", alignment.clamp(0.0, 1.0))

        corrected = output.get("corrected_symptom_candidate_response")
        if torch.is_tensor(corrected):
            labels = _resize_labels(target, tuple(corrected.shape[-2:]))
            foreground = labels > 0
            if bool(foreground.any()):
                self._add(
                    "candidate_class_accuracy",
                    (corrected.argmax(dim=1)[foreground] == (labels[foreground] - 1))
                    .float()
                    .mean(),
                )

        route_assignment = output.get("route_assignment")
        if torch.is_tensor(route_assignment):
            labels = _resize_labels(target, tuple(route_assignment.shape[-2:]))
            foreground = labels > 0
            if bool(foreground.any()):
                route_labels = _route_target(labels)
                self._add(
                    "route_assignment_accuracy",
                    (route_assignment.argmax(dim=1)[foreground] == route_labels[foreground])
                    .float()
                    .mean(),
                )

    def _mean(self, name: str) -> float:
        return self.sums.get(name, 0.0) / max(1, self.counts.get(name, 0))

    def finalize(self, *, active_monitor: str | None = None) -> dict[str, float]:
        structure = 0.55 * self._mean("fish_region_iou") + 0.45 * self._mean("part_map_accuracy")
        evidence = self._mean("visual_evidence_alignment")
        route_preparation = 0.50 * self._mean("candidate_class_accuracy") + 0.50 * self._mean(
            "route_assignment_accuracy"
        )
        candidate_route = 0.55 * self._mean("candidate_class_accuracy") + 0.45 * self._mean(
            "route_assignment_accuracy"
        )
        refinement = self._mean("semantic_accuracy")
        joint = 0.35 * candidate_route + 0.65 * refinement
        metrics = {
            "part_structure_information_formation_validation_score": structure,
            "visual_evidence_formation_validation_score": evidence,
            "part_route_preparation_validation_score": route_preparation,
            "candidate_and_route_interpretation_validation_score": candidate_route,
            "semantic_output_refinement_validation_score": refinement,
            "joint_fine_tuning_validation_score": joint,
            "stage_monitor_num_samples": float(self.samples),
            "stage_monitor_num_batches": float(self.batches),
        }
        if active_monitor is not None and active_monitor in metrics:
            metrics["active_stage_validation_score"] = metrics[active_monitor]
        return metrics
