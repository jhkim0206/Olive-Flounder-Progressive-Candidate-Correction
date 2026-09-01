"""Stage-weighted losses for the five supervision groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import (
    PART_MAP_CLASS_NAMES,
    PART_MAP_TO_ROUTE,
    SEMANTIC_TO_ROUTE,
    SYMPTOM_CLASS_NAMES,
    VISUAL_EVIDENCE_NAMES,
)
from ..training_stages import resolve_training_stage_id


def _as_mask(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 2:
        return value.unsqueeze(0).unsqueeze(0).float()
    if value.ndim == 3:
        return value.unsqueeze(1).float()
    return value.float()


def _resize_mask(value: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor | None:
    mask = _as_mask(value)
    if mask is None or mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask, size=size, mode="nearest")


def _resize_labels(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if value.ndim == 4:
        value = value[:, 0]
    if value.shape[-2:] == size:
        return value.long()
    return F.interpolate(value.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()


def _masked_mean(
    value: torch.Tensor, valid: torch.Tensor | None, epsilon: float = 1e-6
) -> torch.Tensor:
    if valid is None:
        return value.mean()
    mask = _resize_mask(valid, value.shape[-2:])
    if mask.shape[1] == 1 and value.ndim == 4 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(epsilon)


def _binary_dice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    target = _resize_mask(target, probability.shape[-2:])
    mask = (
        torch.ones_like(probability)
        if valid is None
        else _resize_mask(valid, probability.shape[-2:])
    )
    if mask.shape[1] == 1 and probability.shape[1] != 1:
        mask = mask.expand_as(probability)
    target = (
        target.expand_as(probability)
        if target.shape[1] == 1 and probability.shape[1] != 1
        else target
    )
    intersection = (probability * target * mask).sum(dim=(2, 3))
    denominator = ((probability + target) * mask).sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def _multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None,
    num_classes: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    labels = _resize_labels(target, logits.shape[-2:]).clamp(0, num_classes - 1)
    one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    probability = F.softmax(logits, dim=1)
    mask = (
        torch.ones_like(probability[:, :1])
        if valid is None
        else _resize_mask(valid, logits.shape[-2:])
    )
    intersection = (probability * one_hot * mask).sum(dim=(0, 2, 3))
    denominator = ((probability + one_hot) * mask).sum(dim=(0, 2, 3))
    present = (one_hot * mask).sum(dim=(0, 2, 3)) > 0
    score = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - score[present].mean() if bool(present.any()) else logits.sum() * 0.0


def _generalized_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None,
    q: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    labels = _resize_labels(target, logits.shape[-2:]).clamp(0, logits.shape[1] - 1)
    probability = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).clamp_min(1e-7)
    loss = -torch.log(probability) if abs(float(q)) < 1e-8 else (1.0 - probability.pow(q)) / q
    if class_weights is not None:
        weights = class_weights.to(logits.device)[labels].unsqueeze(1)
        loss = loss * weights
    return _masked_mean(loss, valid)


def _masked_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    resized_target = _resize_mask(target, logits.shape[-2:])
    if resized_target.shape[1] == 1 and logits.shape[1] != 1:
        resized_target = resized_target.expand_as(logits)
    return _masked_mean(
        F.binary_cross_entropy_with_logits(logits, resized_target, reduction="none"), valid
    )


def _total_variation(value: torch.Tensor) -> torch.Tensor:
    horizontal = (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean()
    vertical = (value[:, :, 1:, :] - value[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def _probability_to_logit(probability: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probability = probability.clamp(epsilon, 1.0 - epsilon)
    return probability.log() - (1.0 - probability).log()


def _soft_close(probability: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel <= 1:
        return probability.clamp(0.0, 1.0)
    dilated = F.max_pool2d(probability, kernel, 1, kernel // 2)
    return -F.max_pool2d(-dilated, kernel, 1, kernel // 2)


def _soft_boundary(probability: torch.Tensor) -> torch.Tensor:
    maximum = F.max_pool2d(probability, 3, 1, 1)
    minimum = -F.max_pool2d(-probability, 3, 1, 1)
    return (maximum - minimum).clamp(0.0, 1.0)


class ProgressiveCandidateCorrectionLoss(nn.Module):
    """Compute the five loss groups described in the training section."""

    def __init__(
        self,
        num_classes: int = 9,
        num_parts: int = 5,
        semantic_class_names: Sequence[str] = SYMPTOM_CLASS_NAMES,
        part_names: Sequence[str] = PART_MAP_CLASS_NAMES,
        class_weights: torch.Tensor | None = None,
        part_class_weights: torch.Tensor | None = None,
        generalized_cross_entropy_q: float = 0.70,
        fish_region_weight: float = 0.80,
        fish_region_dice_weight: float = 0.50,
        part_map_weight: float = 0.80,
        part_map_dice_weight: float = 0.35,
        head_to_tail_direction_weight: float = 0.0,
        zone_map_weight: float = 0.0,
        redness_weight: float = 0.25,
        shape_weight: float = 0.45,
        lesion_weight: float = 0.55,
        positive_evidence_dice_weight: float = 0.25,
        lesion_dice_weight: float = 0.35,
        evidence_union_weight: float = 0.12,
        unaffected_surface_weight: float = 0.55,
        unaffected_surface_dice_weight: float = 0.35,
        initial_candidate_weight: float = 0.15,
        signed_candidate_weight: float = 0.30,
        gated_candidate_weight: float = 0.25,
        corrected_candidate_weight: float = 0.90,
        prototype_classification_weight: float = 0.40,
        prototype_alignment_weight: float = 0.12,
        part_conditioned_weight: float = 0.45,
        candidate_foreground_bce_weight: float = 0.35,
        class_seed_positive_weight: float = 0.25,
        candidate_foreground_dice_weight: float = 0.25,
        unaffected_surface_inhibition_weight: float = 0.20,
        class_seed_unaffected_surface_weight: float = 0.12,
        body_lesion_unaffected_surface_weight: float = 0.20,
        body_lesion_budget: float = 0.12,
        body_lesion_budget_weight: float = 0.10,
        symptom_foreground_fish_region_budget: float = 0.18,
        symptom_foreground_fish_region_budget_weight: float = 0.08,
        caudal_necrosis_region_budget: float = 0.62,
        caudal_necrosis_region_budget_weight: float = 0.16,
        routed_semantic_weight: float = 0.75,
        route_assignment_weight: float = 0.35,
        route_part_consistency_weight: float = 0.25,
        spatial_support_weight: float = 0.55,
        spatial_support_dice_weight: float = 0.25,
        route_interpretation_weight: float = 0.65,
        route_preservation_weight: float = 0.025,
        route_delta_sparsity_weight: float = 0.010,
        route_consistency_weight: float = 0.030,
        spatial_support_closure_weight: float = 0.020,
        spatial_support_unaffected_surface_weight: float = 0.040,
        route_preservation_confidence_softness: float = 0.70,
        route_interpretation_unaffected_surface_weight: float = 0.10,
        spatial_support_closure_kernel: int = 5,
        auxiliary_semantic_weight: float = 0.45,
        auxiliary_semantic_dice_weight: float = 0.35,
        native_semantic_contribution: float = 0.35,
        final_refinement_weight: float = 0.40,
        signed_distance_weight: float = 0.35,
        refinement_foreground_dice_weight: float = 0.25,
        refinement_curvature_weight: float = 0.05,
        refinement_total_variation_weight: float = 0.025,
        refinement_fish_region_boundary_weight: float = 0.025,
        signed_distance_temperature: float = 4.0,
        stage_multipliers: Mapping[int | str, Mapping[str, float]] | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_symptom_classes = self.num_classes - 1
        self.num_parts = int(num_parts)
        if self.num_classes != 9 or self.num_parts != 5:
            raise ValueError("the objective expects 9 semantic and 5 part classes")
        self.semantic_class_names = tuple(semantic_class_names)
        self.part_names = tuple(part_names)
        if self.semantic_class_names != SYMPTOM_CLASS_NAMES:
            raise ValueError("semantic_class_names must match the method class order")
        if self.part_names != PART_MAP_CLASS_NAMES:
            raise ValueError("part_names must match the method Part Map order")
        self.q = float(generalized_cross_entropy_q)
        self.current_stage_id: int | None = None

        self.fish_region_weight = float(fish_region_weight)
        self.fish_region_dice_weight = float(fish_region_dice_weight)
        self.part_map_weight = float(part_map_weight)
        self.part_map_dice_weight = float(part_map_dice_weight)
        self.head_to_tail_direction_weight = float(head_to_tail_direction_weight)
        self.zone_map_weight = float(zone_map_weight)
        self.redness_weight = float(redness_weight)
        self.shape_weight = float(shape_weight)
        self.lesion_weight = float(lesion_weight)
        self.positive_evidence_dice_weight = float(positive_evidence_dice_weight)
        self.lesion_dice_weight = float(lesion_dice_weight)
        self.evidence_union_weight = float(evidence_union_weight)
        self.unaffected_surface_weight = float(unaffected_surface_weight)
        self.unaffected_surface_dice_weight = float(unaffected_surface_dice_weight)
        self.initial_candidate_weight = float(initial_candidate_weight)
        self.signed_candidate_weight = float(signed_candidate_weight)
        self.gated_candidate_weight = float(gated_candidate_weight)
        self.corrected_candidate_weight = float(corrected_candidate_weight)
        self.prototype_classification_weight = float(prototype_classification_weight)
        self.prototype_alignment_weight = float(prototype_alignment_weight)
        self.part_conditioned_weight = float(part_conditioned_weight)
        self.candidate_foreground_bce_weight = float(candidate_foreground_bce_weight)
        self.class_seed_positive_weight = float(class_seed_positive_weight)
        self.candidate_foreground_dice_weight = float(candidate_foreground_dice_weight)
        self.unaffected_surface_inhibition_weight = float(unaffected_surface_inhibition_weight)
        self.class_seed_unaffected_surface_weight = float(
            class_seed_unaffected_surface_weight
        )
        self.body_lesion_unaffected_surface_weight = float(
            body_lesion_unaffected_surface_weight
        )
        self.body_lesion_budget = float(body_lesion_budget)
        self.body_lesion_budget_weight = float(body_lesion_budget_weight)
        self.symptom_foreground_fish_region_budget = float(
            symptom_foreground_fish_region_budget
        )
        self.symptom_foreground_fish_region_budget_weight = float(
            symptom_foreground_fish_region_budget_weight
        )
        self.caudal_necrosis_region_budget = float(caudal_necrosis_region_budget)
        self.caudal_necrosis_region_budget_weight = float(
            caudal_necrosis_region_budget_weight
        )
        self.routed_semantic_weight = float(routed_semantic_weight)
        self.route_assignment_weight = float(route_assignment_weight)
        self.route_part_consistency_weight = float(route_part_consistency_weight)
        self.spatial_support_weight = float(spatial_support_weight)
        self.spatial_support_dice_weight = float(spatial_support_dice_weight)
        self.route_interpretation_weight = float(route_interpretation_weight)
        self.route_preservation_weight = float(route_preservation_weight)
        self.route_delta_sparsity_weight = float(route_delta_sparsity_weight)
        self.route_consistency_weight = float(route_consistency_weight)
        self.spatial_support_closure_weight = float(spatial_support_closure_weight)
        self.spatial_support_unaffected_surface_weight = float(
            spatial_support_unaffected_surface_weight
        )
        self.route_preservation_confidence_softness = float(
            route_preservation_confidence_softness
        )
        self.route_interpretation_unaffected_surface_weight = float(
            route_interpretation_unaffected_surface_weight
        )
        self.spatial_support_closure_kernel = int(spatial_support_closure_kernel)
        self.auxiliary_semantic_weight = float(auxiliary_semantic_weight)
        self.auxiliary_semantic_dice_weight = float(auxiliary_semantic_dice_weight)
        self.native_semantic_contribution = float(native_semantic_contribution)
        self.final_refinement_weight = float(final_refinement_weight)
        self.signed_distance_weight = float(signed_distance_weight)
        self.refinement_foreground_dice_weight = float(refinement_foreground_dice_weight)
        self.refinement_curvature_weight = float(refinement_curvature_weight)
        self.refinement_total_variation_weight = float(refinement_total_variation_weight)
        self.refinement_fish_region_boundary_weight = float(refinement_fish_region_boundary_weight)
        self.signed_distance_temperature = float(signed_distance_temperature)

        self.register_buffer(
            "class_weights",
            torch.ones(self.num_classes) if class_weights is None else class_weights.float(),
        )
        self.register_buffer(
            "part_class_weights",
            torch.ones(self.num_parts)
            if part_class_weights is None
            else part_class_weights.float(),
        )
        self.class_index = {
            name: self.semantic_class_names.index(name) for name in self.semantic_class_names
        }
        self.stage_multipliers = self._normalize_stage_multipliers(stage_multipliers)

    @staticmethod
    def _normalize_stage_multipliers(
        values: Mapping[int | str, Mapping[str, float]] | None,
    ) -> dict[int, dict[str, float]] | None:
        if values is None:
            return None
        normalized: dict[int, dict[str, float]] = {}
        for stage, group_values in values.items():
            stage_id = resolve_training_stage_id(stage)
            normalized[stage_id] = {str(key): float(value) for key, value in group_values.items()}
        return normalized

    def set_training_stage(self, stage: str | int) -> ProgressiveCandidateCorrectionLoss:
        self.current_stage_id = resolve_training_stage_id(stage)
        return self

    def _stage_id(self, model_output: Mapping[str, Any]) -> int:
        if self.current_stage_id is not None:
            return self.current_stage_id
        value = model_output.get("training_stage_id", 6)
        if torch.is_tensor(value):
            return int(value.detach().flatten()[0].item())
        return int(value)

    def _multipliers(self, stage_id: int) -> dict[str, float]:
        if self.stage_multipliers is not None and stage_id in self.stage_multipliers:
            return self.stage_multipliers[stage_id]
        default = {
            1: (1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
            2: (0.05, 0.05, 1.00, 1.00, 0.35, 0.00, 0.00, 0.00, 0.10, 0.00, 0.05),
            3: (0.00, 0.00, 0.00, 0.15, 0.35, 0.25, 1.00, 0.15, 0.20, 0.00, 0.10),
            4: (0.00, 0.00, 0.00, 0.10, 0.90, 1.00, 0.85, 0.45, 0.55, 0.10, 0.40),
            5: (0.00, 0.00, 0.00, 0.10, 0.45, 0.35, 0.35, 0.50, 0.35, 1.00, 0.30),
            6: (0.00, 0.00, 0.00, 0.15, 0.80, 0.85, 0.60, 0.65, 0.55, 0.45, 0.45),
        }
        try:
            values = default[stage_id]
        except KeyError as exc:
            raise ValueError(f"training stage ID must be in [1, 6], received {stage_id}") from exc
        return dict(zip(
            (
                "fish_region", "part_map", "visual_evidence", "unaffected_surface",
                "part_conditioned", "candidate", "prototype", "semantic", "foreground",
                "final_refinement", "suppression",
            ),
            values,
            strict=True,
        ))

    @staticmethod
    def _visual_evidence_targets(segmentation: torch.Tensor) -> torch.Tensor:
        table = segmentation.new_tensor(
            [
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 1.00],
                [0.35, 0.00, 0.95],
                [0.00, 1.00, 0.75],
                [0.15, 0.20, 1.00],
                [0.15, 0.20, 1.00],
                [0.00, 1.00, 0.75],
                [0.15, 0.20, 1.00],
                [0.15, 0.20, 1.00],
            ],
            dtype=torch.float32,
        )
        return table[segmentation.clamp(0, 8)].permute(0, 3, 1, 2)

    @staticmethod
    def _route_target(segmentation: torch.Tensor) -> torch.Tensor:
        # Body, Mouth, Fin, and Caudal-fin routes use zero-based channel IDs.
        lookup = segmentation.new_tensor(SEMANTIC_TO_ROUTE)
        return lookup[segmentation.clamp(0, 8)]

    @staticmethod
    def _part_route_target(part_map_target: torch.Tensor) -> torch.Tensor:
        # Part Map IDs are background, body, fin, caudal fin, and mouth.
        lookup = part_map_target.new_tensor(PART_MAP_TO_ROUTE)
        return lookup[part_map_target.clamp(0, 4)]

    def _part_structure_loss(
        self,
        output: Mapping[str, torch.Tensor],
        fish_region_target: torch.Tensor,
        part_map_target: torch.Tensor,
        head_to_tail_direction_target: torch.Tensor,
        zone_map_target: torch.Tensor,
        structure_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fish_logits = output["fish_region_logits"]
        fish_valid = torch.ones_like(fish_region_target)
        fish_bce = _masked_binary_loss(fish_logits, fish_region_target, fish_valid)
        fish_dice = _binary_dice_loss(torch.sigmoid(fish_logits), fish_region_target, fish_valid)
        part_logits = output["part_map_logits"]
        part_labels = _resize_labels(part_map_target, part_logits.shape[-2:])
        part_ce_map = F.cross_entropy(
            part_logits,
            part_labels,
            weight=self.part_class_weights.to(part_logits.device),
            reduction="none",
        ).unsqueeze(1)
        part_ce = _masked_mean(part_ce_map, structure_valid)
        part_dice = _multiclass_dice_loss(
            part_logits, part_map_target, structure_valid, self.num_parts
        )
        direction = output["head_to_tail_direction"]
        head_to_tail_direction_target_resized = _resize_mask(
            head_to_tail_direction_target, direction.shape[-2:]
        )
        direction_valid = _resize_mask(structure_valid, direction.shape[-2:]) * (
            head_to_tail_direction_target_resized >= 0.0
        )
        direction_loss = _masked_mean(
            F.smooth_l1_loss(
                direction,
                head_to_tail_direction_target_resized.clamp(0.0, 1.0),
                reduction="none",
            ),
            direction_valid,
        )
        zone_logits = output["zone_map_logits"]
        zone_labels = _resize_labels(zone_map_target, zone_logits.shape[-2:])
        zone_ce = _masked_mean(
            F.cross_entropy(zone_logits, zone_labels, reduction="none").unsqueeze(1),
            structure_valid,
        )
        weighted_fish_region = self.fish_region_weight * (
            fish_bce + self.fish_region_dice_weight * fish_dice
        )
        weighted_part_map = self.part_map_weight * (
            part_ce + self.part_map_dice_weight * part_dice
        )
        total = weighted_fish_region + weighted_part_map
        return total, {
            "loss_fish_region": fish_bce,
            "loss_fish_region_dice": fish_dice,
            "loss_part_map": part_ce,
            "loss_part_map_dice": part_dice,
            "loss_head_to_tail_direction": direction_loss,
            "loss_zone_map": zone_ce,
            "weighted_fish_region": weighted_fish_region,
            "weighted_part_map": weighted_part_map,
        }

    def _visual_evidence_loss(
        self,
        output: Mapping[str, torch.Tensor],
        segmentation: torch.Tensor,
        symptom_foreground: torch.Tensor,
        positive_valid: torch.Tensor,
        unaffected_surface_target: torch.Tensor,
        unaffected_surface_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self._visual_evidence_targets(segmentation)
        names = VISUAL_EVIDENCE_NAMES[:3]
        logits = [output[f"{name}_evidence_logits"] for name in names]
        losses: list[torch.Tensor] = []
        dice_losses: list[torch.Tensor] = []
        for channel, channel_logits in enumerate(logits):
            channel_target = target[:, channel : channel + 1]
            losses.append(_masked_binary_loss(channel_logits, channel_target, positive_valid))
            dice_losses.append(
                _binary_dice_loss(torch.sigmoid(channel_logits), channel_target, positive_valid)
            )
        evidence_union = torch.stack([torch.sigmoid(value) for value in logits], dim=0).amax(dim=0)
        union_loss = _masked_mean(
            F.binary_cross_entropy(
                evidence_union,
                _resize_mask(symptom_foreground, evidence_union.shape[-2:]),
                reduction="none",
            ),
            positive_valid,
        )
        unaffected_logits = output["unaffected_surface_evidence_logits"]
        combined_unaffected_valid = torch.maximum(
            _resize_mask(unaffected_surface_valid, unaffected_logits.shape[-2:]),
            _resize_mask(symptom_foreground, unaffected_logits.shape[-2:]),
        )
        unaffected_bce = _masked_binary_loss(
            unaffected_logits, unaffected_surface_target, combined_unaffected_valid
        )
        unaffected_dice = _binary_dice_loss(
            torch.sigmoid(unaffected_logits),
            unaffected_surface_target,
            combined_unaffected_valid,
        )
        weighted_positive = (
            self.redness_weight * (losses[0] + self.positive_evidence_dice_weight * dice_losses[0])
            + self.shape_weight * (losses[1] + self.positive_evidence_dice_weight * dice_losses[1])
            + self.lesion_weight * (losses[2] + self.lesion_dice_weight * dice_losses[2])
            + self.evidence_union_weight * union_loss
        )
        weighted_unaffected = self.unaffected_surface_weight * (
            unaffected_bce + self.unaffected_surface_dice_weight * unaffected_dice
        )
        total = weighted_positive + weighted_unaffected
        return total, {
            "loss_redness_evidence": losses[0],
            "loss_shape_evidence": losses[1],
            "loss_lesion_evidence": losses[2],
            "loss_visual_evidence_union": union_loss,
            "loss_unaffected_surface": unaffected_bce,
            "loss_unaffected_surface_dice": unaffected_dice,
            "weighted_positive_visual_evidence": weighted_positive,
            "weighted_unaffected_surface": weighted_unaffected,
        }

    def _candidate_loss(
        self,
        output: Mapping[str, torch.Tensor],
        segmentation: torch.Tensor,
        symptom_foreground: torch.Tensor,
        candidate_valid: torch.Tensor,
        unaffected_surface_target: torch.Tensor,
        unaffected_surface_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del unaffected_surface_target, unaffected_surface_valid
        symptom_labels = (segmentation - 1).clamp(0, self.num_symptom_classes - 1)
        positive_valid = _resize_mask(candidate_valid, segmentation.shape[-2:]) * _resize_mask(
            symptom_foreground, segmentation.shape[-2:]
        )
        candidate_names = (
            "initial_symptom_candidate_response",
            "signed_corrected_candidate_response",
            "gated_corrected_candidate_response_low",
            "corrected_symptom_candidate_response",
        )
        candidate_losses: list[torch.Tensor] = []
        for name in candidate_names:
            logits = output.get(name)
            if logits is None:
                candidate_losses.append(output["corrected_symptom_candidate_response"].sum() * 0.0)
            else:
                candidate_losses.append(
                _generalized_cross_entropy(
                    logits,
                    symptom_labels,
                    positive_valid,
                    self.q,
                    self.class_weights[1:],
                )
            )
        class_total = (
            self.initial_candidate_weight * candidate_losses[0]
            + self.signed_candidate_weight * candidate_losses[1]
            + self.gated_candidate_weight * candidate_losses[2]
            + self.corrected_candidate_weight * candidate_losses[3]
        )

        prototype_similarity = output.get("symptom_prototype_similarity_low")
        if prototype_similarity is None:
            prototype_classification = class_total * 0.0
        else:
            prototype_classification = _generalized_cross_entropy(
                prototype_similarity,
                symptom_labels,
                positive_valid,
                self.q,
                self.class_weights[1:],
            )
        embedding = output.get("latent_visual_embedding_low")
        prototypes = output.get("symptom_prototypes")
        prototype_alignment = class_total * 0.0
        if embedding is not None and prototypes is not None:
            labels = _resize_labels(segmentation, embedding.shape[-2:])
            valid = labels > 0
            if bool(valid.any()):
                class_ids = (labels - 1).clamp(0, prototypes.shape[0] - 1)
                selected_embedding = embedding.permute(0, 2, 3, 1)[valid]
                selected_prototype = F.normalize(prototypes.float(), dim=1)[class_ids[valid]]
                prototype_alignment = (
                    1.0
                    - (
                        F.normalize(selected_embedding.float(), dim=1)
                        * selected_prototype
                    ).sum(dim=1)
                ).mean()
        prototype_total = (
            self.prototype_classification_weight * prototype_classification
            + self.prototype_alignment_weight * prototype_alignment
        )

        part_conditioned_logits = output.get("part_conditioned_logits_low")
        part_conditioned = class_total * 0.0
        if part_conditioned_logits is not None:
            labels = _resize_labels(segmentation, part_conditioned_logits.shape[-2:])
            positive = labels > 0
            family_lookup = labels.new_tensor([0, 0, 3, 1, 1, 1, 2, 2, 2])
            family = family_lookup[labels.clamp(0, 8)]
            family_target = F.one_hot(family, num_classes=4).permute(0, 3, 1, 2).float()
            family_target = family_target * positive.unsqueeze(1)
            family_valid = family_target * _resize_mask(candidate_valid, labels.shape[-2:])
            part_conditioned = _masked_binary_loss(
                part_conditioned_logits, family_target, family_valid
            )
        part_conditioned_total = self.part_conditioned_weight * part_conditioned
        total = class_total + prototype_total + part_conditioned_total
        return total, {
            "loss_initial_candidate_response": candidate_losses[0],
            "loss_signed_corrected_response": candidate_losses[1],
            "loss_gated_candidate_response": candidate_losses[2],
            "loss_corrected_candidate_response": candidate_losses[3],
            "loss_symptom_prototype_classification": prototype_classification,
            "loss_symptom_prototype_alignment": prototype_alignment,
            "loss_part_conditioned_response": part_conditioned,
            "weighted_candidate_responses": class_total,
            "weighted_symptom_prototypes": prototype_total,
            "weighted_part_conditioned_response": part_conditioned_total,
        }

    def _route_loss(
        self,
        output: Mapping[str, torch.Tensor],
        segmentation: torch.Tensor,
        part_map_target: torch.Tensor,
        zone_map_target: torch.Tensor,
        route_valid: torch.Tensor,
        unaffected_surface_target: torch.Tensor,
        unaffected_surface_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del zone_map_target
        symptom_labels = (segmentation - 1).clamp(0, self.num_symptom_classes - 1)
        routed_loss = _generalized_cross_entropy(
            output["routed_semantic_logits"],
            symptom_labels,
            route_valid,
            self.q,
            self.class_weights[1:],
        )
        assignment_logits = output["route_assignment_with_background_logits_low"]
        assignment_target_lookup = segmentation.new_tensor([0, 1, 4, 2, 2, 2, 3, 3, 3])
        assignment_target = assignment_target_lookup[segmentation.clamp(0, 8)]
        route_ce = _masked_mean(
            F.cross_entropy(
                assignment_logits,
                _resize_labels(assignment_target, assignment_logits.shape[-2:]),
                reduction="none",
            ).unsqueeze(1),
            route_valid,
        )
        part_consistency = _masked_mean(
            F.cross_entropy(
                assignment_logits,
                _resize_labels(part_map_target, assignment_logits.shape[-2:]),
                reduction="none",
            ).unsqueeze(1),
            _as_mask(part_map_target > 0),
        )
        support_logits = output["spatial_support_internal_logits_low"]
        support_labels = _resize_labels(segmentation, support_logits.shape[-2:])
        support_lookup = support_labels.new_tensor([0, 0, 7, 3, 2, 1, 6, 5, 4])
        support_channels = support_lookup[support_labels.clamp(0, 8)]
        support_target = F.one_hot(support_channels, num_classes=8).permute(0, 3, 1, 2).float()
        support_target = support_target * (support_labels > 0).unsqueeze(1)
        support_positive_valid = support_target.amax(dim=1, keepdim=True) * _resize_mask(
            route_valid, support_logits.shape[-2:]
        )
        unaffected = _resize_mask(
            unaffected_surface_target, support_logits.shape[-2:]
        ) * _resize_mask(unaffected_surface_valid, support_logits.shape[-2:])
        support_valid = torch.maximum(support_positive_valid, unaffected).expand_as(
            support_target
        )
        support_bce = _masked_binary_loss(
            support_logits, support_target, support_valid
        )
        spatial_support = torch.sigmoid(support_logits).clamp(0.0, 1.0)
        support_dice = _binary_dice_loss(
            spatial_support,
            support_target,
            torch.maximum(support_positive_valid.expand_as(support_target), support_target),
        )
        closed_support = _soft_close(spatial_support, self.spatial_support_closure_kernel)
        support_holes = F.relu(closed_support.detach() - spatial_support)
        support_area = torch.maximum(
            support_positive_valid.expand_as(support_target),
            (spatial_support > 0.25).float(),
        )
        support_closure = (support_holes * support_area).sum() / support_area.sum().clamp_min(
            1e-6
        )
        support_unaffected = _masked_mean(spatial_support, unaffected)

        resized_segmentation = _resize_labels(segmentation, support_logits.shape[-2:])
        resized_route_valid = _resize_mask(route_valid, support_logits.shape[-2:])
        resized_unaffected = unaffected

        def route_interpretation_loss(
            key: str, class_names: Sequence[str]
        ) -> torch.Tensor:
            logits = output.get(key)
            if logits is None:
                return routed_loss * 0.0
            target = torch.zeros_like(resized_segmentation)
            valid = torch.zeros_like(resized_route_valid)
            for local_class, class_name in enumerate(class_names, start=1):
                class_id = self.class_index[class_name] + 1
                selected = resized_segmentation == class_id
                target = torch.where(selected, torch.full_like(target, local_class), target)
                valid = torch.maximum(valid, selected.unsqueeze(1).float() * resized_route_valid)
            valid = torch.maximum(
                valid,
                self.route_interpretation_unaffected_surface_weight * resized_unaffected,
            )
            return _generalized_cross_entropy(logits, target, valid, self.q)

        body_interpretation = route_interpretation_loss(
            "body_route_response_logits_low", ("body_lesion",)
        )
        fin_interpretation = route_interpretation_loss(
            "fin_route_response_logits_low",
            ("fin_deformity", "fin_necrosis", "fin_base_necrosis"),
        )
        caudal_interpretation = route_interpretation_loss(
            "caudal_fin_route_response_logits_low",
            ("caudal_deformity", "caudal_necrosis", "caudal_base_necrosis"),
        )
        mouth_interpretation = route_interpretation_loss(
            "mouth_route_response_logits_low", ("mouth_ulcer",)
        )
        interpretation = (
            0.25 * body_interpretation
            + 0.30 * fin_interpretation
            + 0.30 * caudal_interpretation
            + 0.15 * mouth_interpretation
        )

        corrected = output["corrected_symptom_candidate_response_low"]
        routed = output["routed_semantic_logits_low"]
        corrected_probability = torch.sigmoid(corrected)
        routed_probability = torch.sigmoid(routed)
        route_confidence = output["route_confidence_low"].clamp(0.0, 1.0)
        fish_region = output["fish_region_low"].clamp(0.0, 1.0)
        outside_route = (
            1.0 - self.route_preservation_confidence_softness * route_confidence
        ).clamp(0.0, 1.0) * fish_region
        preservation = (
            (routed_probability - corrected_probability).pow(2) * outside_route
        ).sum() / outside_route.sum().clamp_min(1e-6)
        route_delta = output["route_delta_logits_low"]
        delta_sparsity = (
            route_delta.abs() * (0.20 + 0.80 * route_confidence) * fish_region
        ).sum() / fish_region.sum().clamp_min(1e-6)

        assignment = output["route_assignment_with_background_prob_low"]
        body_assignment = assignment[:, 1:2]
        fin_assignment = assignment[:, 2:3]
        caudal_assignment = assignment[:, 3:4]
        mouth_assignment = assignment[:, 4:5]
        allowed_support = torch.cat(
            [
                body_assignment,
                torch.maximum(fin_assignment, 0.35 * body_assignment),
                fin_assignment,
                fin_assignment,
                torch.maximum(caudal_assignment, 0.35 * body_assignment),
                caudal_assignment,
                caudal_assignment,
                torch.maximum(mouth_assignment, 0.25 * body_assignment),
            ],
            dim=1,
        ).clamp(0.0, 1.0)
        route_consistency = (spatial_support * (1.0 - allowed_support)).mean()
        total = (
            self.routed_semantic_weight * routed_loss
            + self.route_assignment_weight * route_ce
            + self.route_part_consistency_weight * part_consistency
            + self.spatial_support_weight
            * (support_bce + self.spatial_support_dice_weight * support_dice)
            + self.route_interpretation_weight * interpretation
            + self.route_preservation_weight * preservation
            + self.route_delta_sparsity_weight * delta_sparsity
            + self.route_consistency_weight * route_consistency
            + self.spatial_support_closure_weight * support_closure
            + self.spatial_support_unaffected_surface_weight * support_unaffected
        )
        return total, {
            "loss_routed_semantic": routed_loss,
            "loss_route_assignment": route_ce,
            "loss_route_part_consistency": part_consistency,
            "loss_spatial_support": support_bce,
            "loss_spatial_support_dice": support_dice,
            "loss_body_route_interpretation": body_interpretation,
            "loss_fin_route_interpretation": fin_interpretation,
            "loss_caudal_fin_route_interpretation": caudal_interpretation,
            "loss_mouth_route_interpretation": mouth_interpretation,
            "loss_route_interpretation": interpretation,
            "loss_route_preservation": preservation,
            "loss_route_delta_sparsity": delta_sparsity,
            "loss_route_consistency": route_consistency,
            "loss_spatial_support_closure": support_closure,
            "loss_spatial_support_unaffected_surface": support_unaffected,
        }

    def _foreground_and_suppression_loss(
        self,
        output: Mapping[str, torch.Tensor],
        symptom_foreground: torch.Tensor,
        positive_valid: torch.Tensor,
        unaffected_surface_target: torch.Tensor,
        unaffected_surface_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        foreground = output["symptom_foreground_probability_low"].clamp(1e-6, 1.0 - 1e-6)
        positive = _resize_mask(symptom_foreground, foreground.shape[-2:])
        unaffected = _resize_mask(
            unaffected_surface_target, foreground.shape[-2:]
        ) * _resize_mask(unaffected_surface_valid, foreground.shape[-2:])
        foreground_valid = torch.maximum(
            _resize_mask(positive_valid, foreground.shape[-2:]), unaffected
        )
        foreground_bce = _masked_mean(
            F.binary_cross_entropy(foreground, positive, reduction="none"),
            foreground_valid,
        )
        foreground_dice = _binary_dice_loss(foreground, positive, foreground_valid)
        class_seed_mass = output["class_seed_mass_low"].clamp(1e-6, 1.0 - 1e-6)
        class_seed_positive = _masked_mean(
            F.binary_cross_entropy(
                class_seed_mass, torch.ones_like(class_seed_mass), reduction="none"
            ),
            positive,
        )
        foreground_total = (
            self.candidate_foreground_bce_weight
            * (foreground_bce + self.candidate_foreground_dice_weight * foreground_dice)
            + self.class_seed_positive_weight * class_seed_positive
        )

        class_evidence = output["class_evidence_low"]
        class_seed = output["class_seed_low"]
        body_lesion_seed = output["body_lesion_seed_low"]
        class_evidence_unaffected = _masked_mean(class_evidence, unaffected)
        class_seed_unaffected = _masked_mean(class_seed, unaffected)
        body_lesion_unaffected = _masked_mean(body_lesion_seed, unaffected)

        fish_region = output["fish_region_low"].clamp(0.0, 1.0)
        fish_denominator = fish_region.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        foreground_ratio = (
            (foreground * fish_region).sum(dim=(2, 3), keepdim=True) / fish_denominator
        )
        foreground_budget = F.relu(
            foreground_ratio - self.symptom_foreground_fish_region_budget
        ).pow(2).mean()
        body_lesion_ratio = (
            (body_lesion_seed * fish_region).sum(dim=(2, 3), keepdim=True)
            / fish_denominator
        )
        body_lesion_budget = F.relu(body_lesion_ratio - self.body_lesion_budget).pow(2).mean()
        caudal_necrosis = class_evidence[
            :, self.class_index["caudal_necrosis"] : self.class_index["caudal_necrosis"] + 1
        ]
        zone_map = output["zone_map_low"]
        caudal_region = torch.maximum(
            torch.maximum(zone_map[:, 5:6], zone_map[:, 6:7]), zone_map[:, 7:8]
        )
        caudal_denominator = caudal_region.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        caudal_ratio = (
            (caudal_necrosis * caudal_region).sum(dim=(2, 3), keepdim=True)
            / caudal_denominator
        )
        caudal_budget = F.relu(
            caudal_ratio - self.caudal_necrosis_region_budget
        ).pow(2).mean()
        suppression_total = (
            self.unaffected_surface_inhibition_weight * class_evidence_unaffected
            + self.class_seed_unaffected_surface_weight * class_seed_unaffected
            + self.body_lesion_unaffected_surface_weight * body_lesion_unaffected
            + self.body_lesion_budget_weight * body_lesion_budget
            + self.symptom_foreground_fish_region_budget_weight * foreground_budget
            + self.caudal_necrosis_region_budget_weight * caudal_budget
        )
        return foreground_total, suppression_total, {
            "loss_symptom_foreground": foreground_bce,
            "loss_symptom_foreground_dice": foreground_dice,
            "loss_class_seed_positive": class_seed_positive,
            "loss_class_evidence_unaffected_surface": class_evidence_unaffected,
            "loss_class_seed_unaffected_surface": class_seed_unaffected,
            "loss_body_lesion_unaffected_surface": body_lesion_unaffected,
            "loss_body_lesion_budget": body_lesion_budget,
            "loss_symptom_foreground_fish_region_budget": foreground_budget,
            "loss_caudal_necrosis_region_budget": caudal_budget,
        }

    @staticmethod
    def _foreground_probability(logits: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.softmax(logits, dim=1)[:, :1]

    def _edge_weight(
        self,
        output: Mapping[str, torch.Tensor],
        size: tuple[int, int],
        image: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = torch.sigmoid(
            F.interpolate(
                output["shape_evidence_logits_low"], size=size, mode="bilinear", align_corners=False
            )
        )
        lesion = F.interpolate(
            output.get("effective_visual_evidence_low", output["visual_evidence_low"])[
                :, 2:3
            ],
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        image_edge = torch.zeros_like(shape)
        if image is not None:
            resized = F.interpolate(image[:, :3], size=size, mode="bilinear", align_corners=False)
            image_edge = _soft_boundary(resized.mean(dim=1, keepdim=True))
        return ((1.0 - shape) * (1.0 - lesion) * (1.0 - image_edge)).clamp(0.05, 1.0)

    @staticmethod
    def _weighted_variation(
        probability: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        dx = (probability[:, :, :, 1:] - probability[:, :, :, :-1]).abs()
        dy = (probability[:, :, 1:, :] - probability[:, :, :-1, :]).abs()
        valid_x = valid[:, :, :, 1:] * valid[:, :, :, :-1]
        valid_y = valid[:, :, 1:, :] * valid[:, :, :-1, :]
        weight_x = 0.5 * (weight[:, :, :, 1:] + weight[:, :, :, :-1])
        weight_y = 0.5 * (weight[:, :, 1:, :] + weight[:, :, :-1, :])
        return (
            (dx * valid_x * weight_x).sum() + (dy * valid_y * weight_y).sum()
        ) / ((valid_x * weight_x).sum() + (valid_y * weight_y).sum()).clamp_min(1e-6)

    @staticmethod
    def _curvature(
        probability: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        dx = F.pad(probability[:, :, :, 1:] - probability[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(probability[:, :, 1:, :] - probability[:, :, :-1, :], (0, 0, 0, 1))
        norm = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6)
        nx = dx / norm
        ny = dy / norm
        divergence_x = F.pad(nx[:, :, :, 1:] - nx[:, :, :, :-1], (0, 1, 0, 0))
        divergence_y = F.pad(ny[:, :, 1:, :] - ny[:, :, :-1, :], (0, 0, 0, 1))
        curvature = divergence_x + divergence_y
        edge_magnitude = norm.detach().clamp(0.0, 1.0)
        combined = valid * weight * edge_magnitude
        return (combined * curvature.pow(2)).sum() / combined.sum().clamp_min(1e-6)

    @staticmethod
    def _bilateral(
        probability: torch.Tensor, image: torch.Tensor | None, valid: torch.Tensor
    ) -> torch.Tensor:
        gray = torch.zeros_like(probability)
        if image is not None:
            resized = F.interpolate(
                image[:, :3], size=probability.shape[-2:], mode="bilinear", align_corners=False
            )
            gray = resized.mean(dim=1, keepdim=True)
        probability_dx = (probability[:, :, :, 1:] - probability[:, :, :, :-1]).abs()
        probability_dy = (probability[:, :, 1:, :] - probability[:, :, :-1, :]).abs()
        image_dx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        image_dy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        weight_x = torch.exp(-6.0 * image_dx.pow(2))
        weight_y = torch.exp(-6.0 * image_dy.pow(2))
        valid_x = valid[:, :, :, 1:] * valid[:, :, :, :-1]
        valid_y = valid[:, :, 1:, :] * valid[:, :, :-1, :]
        return (
            (probability_dx * weight_x * valid_x).sum()
            + (probability_dy * weight_y * valid_y).sum()
        ) / ((weight_x * valid_x).sum() + (weight_y * valid_y).sum()).clamp_min(1e-6)

    def _auxiliary_loss(
        self,
        output: Mapping[str, torch.Tensor],
        segmentation: torch.Tensor,
        semantic_valid: torch.Tensor,
        symptom_foreground: torch.Tensor,
        signed_distance_target: torch.Tensor,
        boundary_valid: torch.Tensor,
        image: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        refined_logits = output["auxiliary_semantic_logits"]
        native_logits = output["native_semantic_logits"]

        def semantic_component(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            gce = _generalized_cross_entropy(
                logits, segmentation, semantic_valid, self.q, self.class_weights
            )
            dice = _binary_dice_loss(
                self._foreground_probability(logits), symptom_foreground, semantic_valid
            )
            return gce + self.auxiliary_semantic_dice_weight * dice, gce, dice

        refined_semantic, refined_gce, refined_dice = semantic_component(refined_logits)
        native_semantic, native_gce, native_dice = semantic_component(native_logits)
        semantic = refined_semantic + self.native_semantic_contribution * native_semantic
        weighted_semantic = self.auxiliary_semantic_weight * semantic

        foreground_probability = self._foreground_probability(refined_logits)
        boundary_mask = _resize_mask(boundary_valid, foreground_probability.shape[-2:])
        boundary_foreground_dice = _binary_dice_loss(
            foreground_probability, symptom_foreground, boundary_mask
        )
        signed_distance = output.get("final_refinement_signed_distance_low")
        if signed_distance is None:
            zero = refined_logits.sum() * 0.0
            signed_distance_loss = curvature = variation = bilateral = zero
        else:
            target = _resize_mask(signed_distance_target, signed_distance.shape[-2:])
            valid = _resize_mask(boundary_valid, signed_distance.shape[-2:])
            signed_distance_loss = _masked_mean(
                F.smooth_l1_loss(
                    torch.tanh(signed_distance / self.signed_distance_temperature),
                    torch.tanh(target / self.signed_distance_temperature),
                    reduction="none",
                ),
                valid,
            )
            refined_foreground_low = self._foreground_probability(
                output["auxiliary_semantic_logits_low"]
            )
            valid_low = _resize_mask(boundary_valid, refined_foreground_low.shape[-2:])
            edge_weight = self._edge_weight(
                output, refined_foreground_low.shape[-2:], image
            )
            curvature = self._curvature(refined_foreground_low, valid_low, edge_weight)
            variation = self._weighted_variation(
                refined_foreground_low, valid_low, edge_weight
            )
            bilateral = self._bilateral(refined_foreground_low, image, valid_low)
        refinement = (
            self.signed_distance_weight * signed_distance_loss
            + self.refinement_foreground_dice_weight * boundary_foreground_dice
            + self.refinement_curvature_weight * curvature
            + self.refinement_total_variation_weight * variation
            + self.refinement_fish_region_boundary_weight * bilateral
        )
        weighted_refinement = self.final_refinement_weight * refinement
        total = weighted_semantic + weighted_refinement
        return total, {
            "loss_auxiliary_semantic": refined_gce,
            "loss_auxiliary_semantic_dice": refined_dice,
            "loss_native_semantic": native_gce,
            "loss_native_semantic_dice": native_dice,
            "loss_refinement_foreground_dice": boundary_foreground_dice,
            "loss_signed_distance": signed_distance_loss,
            "loss_refinement_curvature": curvature,
            "loss_refinement_total_variation": variation,
            "loss_refinement_bilateral": bilateral,
            "weighted_semantic_output": weighted_semantic,
            "weighted_final_refinement": weighted_refinement,
        }

    def forward(
        self,
        model_output: Mapping[str, torch.Tensor],
        semantic_target: torch.Tensor,
        fish_region_target: torch.Tensor,
        part_map_target: torch.Tensor,
        symptom_foreground_target: torch.Tensor,
        unaffected_surface_target: torch.Tensor,
        signed_distance_target: torch.Tensor,
        head_to_tail_direction_target: torch.Tensor,
        zone_map_target: torch.Tensor,
        semantic_valid: torch.Tensor,
        structure_valid: torch.Tensor,
        positive_evidence_valid: torch.Tensor,
        unaffected_surface_valid: torch.Tensor,
        boundary_valid: torch.Tensor,
        route_valid: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        segmentation = semantic_target.long()
        part_structure, part_structure_terms = self._part_structure_loss(
            model_output,
            fish_region_target,
            part_map_target,
            head_to_tail_direction_target,
            zone_map_target,
            structure_valid,
        )
        visual_evidence, visual_evidence_terms = self._visual_evidence_loss(
            model_output,
            segmentation,
            symptom_foreground_target,
            positive_evidence_valid,
            unaffected_surface_target,
            unaffected_surface_valid,
        )
        candidate, candidate_terms = self._candidate_loss(
            model_output,
            segmentation,
            symptom_foreground_target,
            semantic_valid,
            unaffected_surface_target,
            unaffected_surface_valid,
        )
        route, route_terms = self._route_loss(
            model_output,
            segmentation,
            part_map_target,
            zone_map_target,
            route_valid,
            unaffected_surface_target,
            unaffected_surface_valid,
        )
        foreground, suppression, foreground_terms = self._foreground_and_suppression_loss(
            model_output,
            symptom_foreground_target,
            positive_evidence_valid,
            unaffected_surface_target,
            unaffected_surface_valid,
        )
        auxiliary, auxiliary_terms = self._auxiliary_loss(
            model_output,
            segmentation,
            semantic_valid,
            symptom_foreground_target,
            signed_distance_target,
            boundary_valid,
            image,
        )
        stage_id = self._stage_id(model_output)
        multipliers = self._multipliers(stage_id)
        total = (
            multipliers["fish_region"] * part_structure_terms["weighted_fish_region"]
            + multipliers["part_map"] * part_structure_terms["weighted_part_map"]
            + multipliers["visual_evidence"]
            * visual_evidence_terms["weighted_positive_visual_evidence"]
            + multipliers["unaffected_surface"]
            * visual_evidence_terms["weighted_unaffected_surface"]
            + multipliers["part_conditioned"]
            * candidate_terms["weighted_part_conditioned_response"]
            + multipliers["candidate"]
            * (candidate_terms["weighted_candidate_responses"] + route)
            + multipliers["prototype"] * candidate_terms["weighted_symptom_prototypes"]
            + multipliers["semantic"] * auxiliary_terms["weighted_semantic_output"]
            + multipliers["foreground"] * foreground
            + multipliers["final_refinement"]
            * auxiliary_terms["weighted_final_refinement"]
            + multipliers["suppression"] * suppression
        )
        return {
            "loss": total,
            "loss_part_structure": part_structure.detach(),
            "loss_visual_evidence": visual_evidence.detach(),
            "loss_candidate": candidate.detach(),
            "loss_route": route.detach(),
            "loss_auxiliary": auxiliary.detach(),
            "loss_foreground": foreground.detach(),
            "loss_suppression": suppression.detach(),
            **{key: value.detach() for key, value in part_structure_terms.items()},
            **{key: value.detach() for key, value in visual_evidence_terms.items()},
            **{key: value.detach() for key, value in candidate_terms.items()},
            **{key: value.detach() for key, value in route_terms.items()},
            **{key: value.detach() for key, value in auxiliary_terms.items()},
            **{key: value.detach() for key, value in foreground_terms.items()},
            **{
                f"stage_multiplier_{key}": total.new_tensor(value)
                for key, value in multipliers.items()
            },
        }


def safe_log_probability(probability: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probability = probability.clamp_min(epsilon)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(epsilon)
    return probability.log()
