"""Initial symptom candidate formation and progressive correction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import SYMPTOM_CLASS_NAMES
from .common import (
    ConvBNAct,
    DepthwiseSeparableConv,
    FeatureAdapter,
    LogitHead,
    RefineFuseBlock,
    ResidualDSConvBlock,
)


class CandidateCorrectionVariant(str, Enum):
    CONCAT = "concat"
    SIGNED = "signed"
    FULL = "full"

    @classmethod
    def parse(cls, value: str | CandidateCorrectionVariant) -> CandidateCorrectionVariant:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown correction variant {value!r}; choose {choices}") from exc


class InitialSymptomCandidateGenerator(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        num_parts: int,
        num_classes: int,
        zone_channels: int,
        head_channels: int,
    ):
        super().__init__()
        hidden = max(16, feature_channels // 2)
        self.detail_adapter = FeatureAdapter(feature_channels, hidden)
        self.part_adapter = FeatureAdapter(num_parts, hidden)
        self.zone_adapter = FeatureAdapter(zone_channels, hidden)
        self.redness_adapter = FeatureAdapter(1, hidden)
        self.shape_adapter = FeatureAdapter(1, hidden)
        self.lesion_adapter = FeatureAdapter(1, hidden)
        self.fuse = RefineFuseBlock(feature_channels + hidden * 6, feature_channels)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, feature_channels, k=3),
            ResidualDSConvBlock(feature_channels),
        )
        self.head = LogitHead(feature_channels, head_channels, num_classes)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        part_map: torch.Tensor,
        zone_map: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.part_adapter(part_map),
                self.zone_adapter(zone_map),
                self.redness_adapter(redness_prob),
                self.shape_adapter(shape_prob),
                self.lesion_adapter(lesion_prob),
            ],
            dim=1,
        )
        feature = self.context(self.fuse(x))
        return {
            "initial_symptom_candidate_feature_low": feature,
            "initial_symptom_candidate_response_low": self.head(feature),
        }


def _build_signed_cue_priors(
    semantic_class_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    cue_names = (
        "redness",
        "shape",
        "lesion",
        "body",
        "mouth",
        "fin_tip",
        "fin_middle",
        "fin_base",
        "caudal_fin_tip",
        "caudal_fin_middle",
        "caudal_fin_base",
        "redness_only",
        "shape_only",
        "lesion_only",
    )
    required = torch.zeros(len(semantic_class_names), len(cue_names), dtype=torch.float32)
    optional = torch.zeros_like(required)
    inhibitory = torch.zeros_like(required)
    cue_index = {name: index for index, name in enumerate(cue_names)}

    def assign(target: torch.Tensor, class_index: int, cue_name: str, value: float) -> None:
        target[class_index, cue_index[cue_name]] = float(value)

    for class_index, raw_name in enumerate(semantic_class_names):
        name = str(raw_name).lower()
        if "body" in name and "lesion" in name:
            assign(required, class_index, "lesion", 1.00)
            assign(required, class_index, "body", 1.00)
            assign(optional, class_index, "redness", 0.50)
            for cue_name in (
                "mouth",
                "fin_tip",
                "fin_middle",
                "fin_base",
                "caudal_fin_tip",
                "caudal_fin_middle",
                "caudal_fin_base",
            ):
                assign(
                    inhibitory,
                    class_index,
                    cue_name,
                    0.50 if "middle" in cue_name or "base" in cue_name else 0.70,
                )
        elif "mouth" in name or "ulcer" in name:
            assign(required, class_index, "lesion", 1.00)
            assign(required, class_index, "mouth", 1.00)
            assign(optional, class_index, "redness", 0.55)
            assign(inhibitory, class_index, "body", 0.55)
            for cue_name in (
                "fin_tip",
                "fin_middle",
                "fin_base",
                "caudal_fin_tip",
                "caudal_fin_middle",
                "caudal_fin_base",
            ):
                assign(inhibitory, class_index, cue_name, 0.70)
            assign(inhibitory, class_index, "redness_only", 1.00)
        elif "fin" in name and "deform" in name:
            assign(required, class_index, "shape", 1.00)
            assign(required, class_index, "fin_tip", 1.00)
            assign(optional, class_index, "lesion", 0.15)
            assign(inhibitory, class_index, "body", 0.60)
            assign(inhibitory, class_index, "fin_middle", 1.00)
            assign(inhibitory, class_index, "fin_base", 0.70)
            assign(inhibitory, class_index, "lesion_only", 0.90)
        elif "fin" in name and "base" in name:
            assign(required, class_index, "lesion", 1.00)
            assign(required, class_index, "fin_base", 1.00)
            assign(optional, class_index, "redness", 0.30)
            assign(optional, class_index, "shape", 0.10)
            assign(inhibitory, class_index, "body", 0.60)
            assign(inhibitory, class_index, "fin_tip", 0.70)
            assign(inhibitory, class_index, "shape_only", 0.70)
        elif "fin" in name and "necrosis" in name:
            assign(required, class_index, "lesion", 1.00)
            assign(required, class_index, "fin_middle", 1.00)
            assign(optional, class_index, "redness", 0.25)
            assign(optional, class_index, "shape", 0.10)
            assign(inhibitory, class_index, "body", 0.60)
            assign(inhibitory, class_index, "fin_tip", 0.85)
            assign(inhibitory, class_index, "shape_only", 0.80)
        elif ("caudal" in name or "tail" in name) and "deform" in name:
            assign(required, class_index, "shape", 1.00)
            assign(required, class_index, "caudal_fin_tip", 1.00)
            assign(optional, class_index, "lesion", 0.15)
            assign(inhibitory, class_index, "body", 0.60)
            assign(inhibitory, class_index, "caudal_fin_middle", 1.00)
            assign(inhibitory, class_index, "caudal_fin_base", 0.70)
            assign(inhibitory, class_index, "lesion_only", 0.90)
        elif ("caudal" in name or "tail" in name) and "base" in name:
            assign(required, class_index, "lesion", 1.00)
            assign(required, class_index, "caudal_fin_base", 1.00)
            assign(optional, class_index, "redness", 0.30)
            assign(optional, class_index, "shape", 0.10)
            assign(inhibitory, class_index, "body", 0.60)
            assign(inhibitory, class_index, "caudal_fin_tip", 0.70)
            assign(inhibitory, class_index, "shape_only", 0.70)
        elif ("caudal" in name or "tail" in name) and "necrosis" in name:
            assign(required, class_index, "lesion", 1.00)
            assign(required, class_index, "caudal_fin_middle", 1.00)
            assign(optional, class_index, "redness", 0.25)
            assign(optional, class_index, "shape", 0.10)
            assign(inhibitory, class_index, "body", 0.60)
            assign(inhibitory, class_index, "caudal_fin_tip", 0.85)
            assign(inhibitory, class_index, "shape_only", 0.80)
        else:
            assign(required, class_index, "lesion", 0.50)
            assign(optional, class_index, "redness", 0.25)
            assign(optional, class_index, "shape", 0.25)
            assign(optional, class_index, "body", 0.25)
    return required, optional, inhibitory, cue_names


def _build_competing_pair_mask(
    semantic_class_names: Sequence[str],
) -> torch.Tensor:
    part_group_ids: list[int] = []
    for raw_name in semantic_class_names:
        name = str(raw_name).lower()
        if "fin" in name:
            part_group_ids.append(1)
        elif "caudal" in name or "tail" in name:
            part_group_ids.append(2)
        else:
            part_group_ids.append(0)
    mask = torch.zeros(
        len(semantic_class_names), len(semantic_class_names), dtype=torch.float32
    )
    for i in range(len(semantic_class_names)):
        for j in range(len(semantic_class_names)):
            if i != j and part_group_ids[i] != 0 and part_group_ids[i] == part_group_ids[j]:
                mask[i, j] = 1.0
    return mask


class SignedCandidateCorrector(nn.Module):
    def __init__(
        self,
        semantic_class_names: Sequence[str],
        initial_zone_weight: float = 1.25,
        initial_temperature: float = 0.25,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.semantic_class_names = tuple(str(name) for name in semantic_class_names)
        if self.semantic_class_names != SYMPTOM_CLASS_NAMES:
            raise ValueError("semantic_class_names must match the method class order")
        self.num_classes = len(self.semantic_class_names)
        self.epsilon = float(epsilon)
        required, optional, inhibitory, cue_names = _build_signed_cue_priors(
            self.semantic_class_names
        )
        self.cue_names = cue_names
        self.num_cues = len(cue_names)
        competing_pair_mask = _build_competing_pair_mask(self.semantic_class_names)
        self.register_buffer("required_cue_mask", required)
        self.register_buffer("optional_cue_mask", optional)
        self.register_buffer("inhibitory_cue_mask", inhibitory)
        self.register_buffer("competing_pair_mask", competing_pair_mask)
        self.required_strength = nn.Parameter(required.clone())
        self.optional_strength = nn.Parameter(optional.clone())
        self.inhibitory_strength = nn.Parameter(inhibitory.clone())
        self.cue_threshold = nn.Parameter(
            torch.full((self.num_classes, self.num_cues), 0.35)
        )
        self.cue_temperature_unconstrained = nn.Parameter(
            torch.full(
                (self.num_classes, self.num_cues),
                math.log(math.exp(initial_temperature) - 1.0),
            )
        )
        self.zone_weight = nn.Parameter(
            torch.full((self.num_classes,), float(initial_zone_weight))
        )
        initial_scale = 0.35 * competing_pair_mask
        initial_margin = 0.15 * competing_pair_mask
        self.competition_scale_unconstrained = nn.Parameter(
            torch.log(torch.expm1(initial_scale + 1e-3))
        )
        self.competition_margin_unconstrained = nn.Parameter(
            torch.log(torch.expm1(initial_margin + 1e-3))
        )

    @staticmethod
    def _cue_tensor(
        zone_map: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                redness_prob,
                shape_prob,
                lesion_prob,
                zone_map,
                redness_prob * (1.0 - lesion_prob),
                shape_prob * (1.0 - lesion_prob),
                lesion_prob * (1.0 - shape_prob),
            ],
            dim=1,
        ).clamp(0.0, 1.0)

    def _cue_activation(self, cue_tensor: torch.Tensor) -> torch.Tensor:
        threshold = self.cue_threshold.view(1, self.num_classes, self.num_cues, 1, 1)
        temperature = F.softplus(self.cue_temperature_unconstrained).view(
            1, self.num_classes, self.num_cues, 1, 1
        ).clamp_min(0.05)
        return torch.sigmoid((cue_tensor.unsqueeze(1) - threshold) / temperature)

    def _competition(self, pre_response: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.competition_scale_unconstrained) * self.competing_pair_mask
        margin = F.softplus(self.competition_margin_unconstrained) * self.competing_pair_mask
        result = torch.zeros_like(pre_response)
        for i in range(self.num_classes):
            value = torch.zeros_like(pre_response[:, i : i + 1])
            for j in range(self.num_classes):
                if i == j or self.competing_pair_mask[i, j] <= 0:
                    continue
                value = value + scale[i, j] * F.softplus(
                    pre_response[:, j : j + 1]
                    - pre_response[:, i : i + 1]
                    + margin[i, j]
                )
            result[:, i : i + 1] = value
        return result

    def forward(
        self,
        initial_symptom_candidate_response: torch.Tensor,
        zone_map: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cue_tensor = self._cue_tensor(zone_map, redness_prob, shape_prob, lesion_prob)
        cue_activation = self._cue_activation(cue_tensor)
        required_weight = F.softplus(self.required_strength) * self.required_cue_mask
        optional_weight = F.softplus(self.optional_strength) * self.optional_cue_mask
        inhibitory_weight = F.softplus(self.inhibitory_strength) * self.inhibitory_cue_mask
        required_support = (
            required_weight.view(1, self.num_classes, self.num_cues, 1, 1)
            * torch.log(cue_activation.clamp_min(self.epsilon))
        ).sum(dim=2)
        optional_support = (
            optional_weight.view(1, self.num_classes, self.num_cues, 1, 1)
            * torch.log1p(cue_activation)
        ).sum(dim=2)
        inhibitory_term = (
            inhibitory_weight.view(1, self.num_classes, self.num_cues, 1, 1)
            * cue_activation
        ).sum(dim=2)
        zone_support = self.zone_weight.view(1, self.num_classes, 1, 1) * torch.log(
            zone_map.clamp_min(self.epsilon)
        )
        pre_response = (
            initial_symptom_candidate_response
            + zone_support
            + required_support
            + optional_support
            - inhibitory_term
        )
        competition = self._competition(pre_response)
        response = pre_response - competition
        return {
            "cue_activation_low": cue_activation,
            "required_support_term_low": required_support,
            "optional_support_term_low": optional_support,
            "zone_support_term_low": zone_support,
            "support_term_low": zone_support + required_support + optional_support,
            "inhibition_term_low": inhibitory_term + competition,
            "competing_class_inhibition_low": competition,
            "signed_corrected_candidate_response_low": response,
        }


class GatedResidualCorrector(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        num_classes: int,
        zone_channels: int,
        head_channels: int,
    ):
        super().__init__()
        hidden = max(16, feature_channels // 2)
        self.detail_adapter = FeatureAdapter(feature_channels, hidden)
        self.zone_adapter = FeatureAdapter(zone_channels, hidden)
        self.initial_adapter = FeatureAdapter(num_classes, hidden)
        self.signed_adapter = FeatureAdapter(num_classes, hidden)
        self.redness_adapter = FeatureAdapter(1, hidden)
        self.shape_adapter = FeatureAdapter(1, hidden)
        self.lesion_adapter = FeatureAdapter(1, hidden)
        self.fuse = RefineFuseBlock(feature_channels + hidden * 7, feature_channels)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, feature_channels, k=3),
            ResidualDSConvBlock(feature_channels),
        )
        self.gate_head = LogitHead(feature_channels, head_channels, num_classes)
        self.residual_head = LogitHead(feature_channels, head_channels, num_classes)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        zone_map: torch.Tensor,
        initial_symptom_candidate_response: torch.Tensor,
        signed_corrected_candidate_response: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.zone_adapter(zone_map),
                self.initial_adapter(torch.sigmoid(initial_symptom_candidate_response)),
                self.signed_adapter(torch.sigmoid(signed_corrected_candidate_response)),
                self.redness_adapter(redness_prob),
                self.shape_adapter(shape_prob),
                self.lesion_adapter(lesion_prob),
            ],
            dim=1,
        )
        feature = self.context(self.fuse(x))
        gate_logits = self.gate_head(feature)
        residual_delta = self.residual_head(feature)
        correction = torch.sigmoid(gate_logits) * residual_delta
        return {
            "gated_residual_feature_low": feature,
            "gate_logits_low": gate_logits,
            "residual_correction_delta_low": residual_delta,
            "gated_residual_correction_term_low": correction,
            "gated_corrected_candidate_response_low": (
                signed_corrected_candidate_response + correction
            ),
        }


class ConcatCandidateCorrector(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        part_structure_information_channels: int,
        num_classes: int,
        head_channels: int,
    ):
        super().__init__()
        hidden = max(16, feature_channels // 2)
        self.structure_adapter = FeatureAdapter(part_structure_information_channels, hidden)
        self.evidence_adapter = FeatureAdapter(4, hidden)
        self.initial_adapter = FeatureAdapter(num_classes, hidden)
        self.fuse = RefineFuseBlock(feature_channels + hidden * 3, feature_channels)
        self.context = ResidualDSConvBlock(feature_channels)
        self.head = LogitHead(feature_channels, head_channels, num_classes)

    def forward(
        self,
        common_feature: torch.Tensor,
        part_structure_information: torch.Tensor,
        visual_evidence: torch.Tensor,
        initial_symptom_candidate_response: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.context(
            self.fuse(
                torch.cat(
                    [
                        common_feature,
                        self.structure_adapter(part_structure_information),
                        self.evidence_adapter(visual_evidence),
                        self.initial_adapter(torch.sigmoid(initial_symptom_candidate_response)),
                    ],
                    dim=1,
                )
            )
        )
        return initial_symptom_candidate_response + self.head(feature)


class PartConditionedAbnormalityGate(nn.Module):
    def __init__(self, feature_channels: int, num_parts: int, zone_channels: int, head_channels: int):
        super().__init__()
        hidden = max(16, feature_channels // 2)
        self.detail_adapter = FeatureAdapter(feature_channels, hidden)
        self.part_adapter = FeatureAdapter(num_parts, hidden)
        self.zone_adapter = FeatureAdapter(zone_channels, hidden)
        self.evidence_adapter = FeatureAdapter(4, hidden)
        self.fuse = RefineFuseBlock(feature_channels + hidden * 4, feature_channels)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, feature_channels, k=3),
            ResidualDSConvBlock(feature_channels),
        )
        self.head = LogitHead(feature_channels, head_channels, 4)

    @staticmethod
    def _prior_logits(
        zone_map: torch.Tensor,
        redness: torch.Tensor,
        shape: torch.Tensor,
        lesion: torch.Tensor,
        unaffected_surface: torch.Tensor,
        epsilon: float = 1e-5,
    ) -> torch.Tensor:
        body = zone_map[:, 0:1]
        mouth = zone_map[:, 1:2]
        fin = torch.maximum(torch.maximum(zone_map[:, 2:3], zone_map[:, 3:4]), zone_map[:, 4:5])
        caudal_fin = torch.maximum(
            torch.maximum(zone_map[:, 5:6], zone_map[:, 6:7]), zone_map[:, 7:8]
        )
        unaffected_suppression = (1.0 - 0.75 * unaffected_surface).clamp(0.05, 1.0)
        probability = torch.cat(
            [
                body * (0.75 * lesion + 0.20 * redness + 0.05 * shape) * unaffected_suppression,
                fin * (0.70 * shape + 0.35 * lesion + 0.10 * redness) * unaffected_suppression,
                caudal_fin * (0.60 * lesion + 0.45 * shape + 0.10 * redness) * unaffected_suppression,
                mouth * (0.75 * lesion + 0.35 * redness + 0.05 * shape) * unaffected_suppression,
            ],
            dim=1,
        ).clamp(epsilon, 1.0 - epsilon)
        return torch.log(probability) - torch.log(1.0 - probability)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        part_map: torch.Tensor,
        zone_map: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
        unaffected_surface_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        evidence = torch.cat(
            [redness_prob, shape_prob, lesion_prob, unaffected_surface_prob], dim=1
        ).clamp(0.0, 1.0)
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.part_adapter(part_map),
                self.zone_adapter(zone_map),
                self.evidence_adapter(evidence),
            ],
            dim=1,
        )
        feature = self.context(self.fuse(x))
        raw_logits = self.head(feature)
        prior_logits = self._prior_logits(
            zone_map, redness_prob, shape_prob, lesion_prob, unaffected_surface_prob
        )
        logits = raw_logits + 0.65 * prior_logits
        probability = torch.sigmoid(logits).clamp(0.0, 1.0)
        return {
            "part_conditioned_feature_low": feature,
            "part_conditioned_logits_low": logits,
            "part_conditioned_prior_logits_low": prior_logits,
            "part_conditioned_probability_low": probability,
        }


class LatentVisualFeatureEncoder(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        num_parts: int,
        zone_channels: int,
        embedding_channels: int = 64,
    ):
        super().__init__()
        hidden = max(16, feature_channels // 2)
        self.detail_adapter = FeatureAdapter(feature_channels, hidden)
        self.part_adapter = FeatureAdapter(num_parts, hidden)
        self.zone_adapter = FeatureAdapter(zone_channels, hidden)
        self.evidence_adapter = FeatureAdapter(8, hidden)
        self.fuse = RefineFuseBlock(feature_channels + hidden * 4, feature_channels)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, feature_channels, k=3, dilation=1),
            DepthwiseSeparableConv(feature_channels, feature_channels, k=3, dilation=2),
            ResidualDSConvBlock(feature_channels),
        )
        self.projection = nn.Sequential(
            ConvBNAct(feature_channels, max(embedding_channels, 32), k=1, s=1, p=0),
            nn.Conv2d(max(embedding_channels, 32), embedding_channels, kernel_size=1, bias=True),
        )

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        part_map: torch.Tensor,
        zone_map: torch.Tensor,
        visual_evidence: torch.Tensor,
        part_conditioned_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        evidence = torch.cat([visual_evidence, part_conditioned_probability], dim=1).clamp(0.0, 1.0)
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.part_adapter(part_map),
                self.zone_adapter(zone_map),
                self.evidence_adapter(evidence),
            ],
            dim=1,
        )
        feature = self.context(self.fuse(x))
        embedding = F.normalize(
            torch.nan_to_num(self.projection(feature), nan=0.0, posinf=0.0, neginf=0.0),
            dim=1,
            eps=1e-6,
        )
        return {"latent_visual_feature_low": feature, "latent_visual_embedding_low": embedding}


class SymptomPrototypeBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_channels: int = 64,
        initialization_scale: float = 0.02,
        logit_scale: float = 2.5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_channels = int(embedding_channels)
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.embedding_channels))
        self.logit_scale = nn.Parameter(torch.tensor(float(math.log(max(logit_scale, 1e-3)))))
        nn.init.normal_(self.prototypes, mean=0.0, std=float(initialization_scale))

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=1, eps=1e-6)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        prototypes = self.normalized_prototypes()
        scale = self.logit_scale.exp().clamp(0.1, 20.0)
        logits = torch.einsum("bdhw,cd->bchw", embedding, prototypes) * scale
        return {
            "symptom_prototype_similarity_low": logits,
            "symptom_prototype_probability_low": torch.sigmoid(logits),
            "symptom_prototypes": prototypes,
            "symptom_prototype_logit_scale": scale.detach(),
        }

    @torch.no_grad()
    def momentum_update(
        self,
        class_means: torch.Tensor,
        class_mask: torch.Tensor | None = None,
        momentum: float = 0.95,
    ) -> None:
        if class_means.ndim != 2 or class_means.shape != self.prototypes.shape:
            raise ValueError(
                f"class_means must have shape {tuple(self.prototypes.shape)}, got {tuple(class_means.shape)}"
            )
        mask = (
            torch.ones(self.num_classes, device=self.prototypes.device, dtype=torch.bool)
            if class_mask is None
            else class_mask.to(device=self.prototypes.device).bool().view(-1)
        )
        updated = F.normalize(class_means.to(self.prototypes.device).float(), dim=1, eps=1e-6)
        current = F.normalize(self.prototypes.data, dim=1, eps=1e-6)
        current[mask] = F.normalize(
            momentum * current[mask] + (1.0 - momentum) * updated[mask], dim=1, eps=1e-6
        )
        self.prototypes.data.copy_(current)


class CandidateEvidenceIntegrator(nn.Module):
    def __init__(
        self,
        semantic_class_names: Sequence[str],
        initial_prototype_weight: float = 0.50,
        initial_part_weight: float = 1.10,
        initial_unaffected_weight: float = 0.60,
        initial_cue_weight: float = 0.25,
    ):
        super().__init__()
        self.semantic_class_names = tuple(semantic_class_names)
        self.num_classes = len(self.semantic_class_names)
        self.register_buffer("cue_rule", self._build_cue_rule(self.semantic_class_names))
        self.initial_weight = nn.Parameter(torch.tensor(0.20))
        self.signed_weight = nn.Parameter(torch.tensor(0.45))
        self.gated_weight = nn.Parameter(torch.tensor(0.75))
        self.prototype_weight = nn.Parameter(torch.tensor(float(initial_prototype_weight)))
        self.part_weight = nn.Parameter(torch.tensor(float(initial_part_weight)))
        self.unaffected_weight = nn.Parameter(torch.tensor(float(initial_unaffected_weight)))
        self.cue_weight = nn.Parameter(torch.tensor(float(initial_cue_weight)))
        self.residual_head = nn.Sequential(
            ConvBNAct(self.num_classes * 3, max(16, self.num_classes * 2), k=1, s=1, p=0),
            nn.Conv2d(max(16, self.num_classes * 2), self.num_classes, kernel_size=1, bias=True),
        )

    @staticmethod
    def _build_cue_rule(names: Sequence[str]) -> torch.Tensor:
        rule = torch.zeros(len(names), 6, dtype=torch.float32)
        for class_index, raw_name in enumerate(names):
            name = str(raw_name).lower()
            if "body" in name and "lesion" in name:
                values = [0.10, 0.00, 1.00, -1.10, -0.75, -0.10]
            elif "mouth" in name or "ulcer" in name:
                values = [0.35, 0.05, 0.95, -0.85, -0.35, -0.10]
            elif "fin" in name and "deform" in name:
                values = [0.05, 1.10, 0.15, -0.80, -0.10, 0.10]
            elif "fin" in name and "necrosis" in name:
                values = [0.15, 0.10, 1.00, -0.95, -0.25, -0.65]
            elif "fin" in name and "base" in name:
                values = [0.15, 0.15, 0.95, -0.95, -0.25, -0.35]
            elif ("caudal" in name or "tail" in name) and "deform" in name:
                values = [0.05, 1.05, 0.15, -0.80, -0.10, 0.15]
            elif ("caudal" in name or "tail" in name) and "necrosis" in name:
                values = [0.12, 0.08, 1.05, -1.00, -0.25, -0.75]
            elif ("caudal" in name or "tail" in name) and "base" in name:
                values = [0.15, 0.15, 0.95, -0.95, -0.25, -0.35]
            else:
                values = [0.10, 0.10, 0.45, -0.50, -0.20, -0.20]
            rule[class_index] = torch.tensor(values)
        return rule

    @staticmethod
    def _cue_tensor(visual_evidence: torch.Tensor) -> torch.Tensor:
        redness = visual_evidence[:, 0:1]
        shape = visual_evidence[:, 1:2]
        lesion = visual_evidence[:, 2:3]
        unaffected = visual_evidence[:, 3:4]
        return torch.cat(
            [redness, shape, lesion, unaffected, redness * (1.0 - lesion), shape * (1.0 - lesion)],
            dim=1,
        ).clamp(0.0, 1.0)

    def forward(
        self,
        initial_response: torch.Tensor,
        signed_response: torch.Tensor,
        gated_response: torch.Tensor,
        prototype_similarity: torch.Tensor,
        class_zone_map: torch.Tensor,
        part_conditioned_probability: torch.Tensor,
        visual_evidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cue_score = torch.einsum("cs,bshw->bchw", self.cue_rule, self._cue_tensor(visual_evidence))
        part_log = torch.log(part_conditioned_probability.clamp_min(1e-6))
        zone_log = torch.log(class_zone_map.clamp_min(1e-6))
        unaffected_penalty = visual_evidence[:, 3:4].expand_as(initial_response) * class_zone_map
        residual = self.residual_head(
            torch.cat(
                [torch.sigmoid(initial_response), torch.sigmoid(signed_response), torch.sigmoid(gated_response)],
                dim=1,
            )
        )
        response = (
            self.initial_weight * initial_response
            + self.signed_weight * signed_response
            + self.gated_weight * gated_response
            + self.prototype_weight * prototype_similarity
            + self.part_weight * part_log
            + 0.25 * zone_log
            + self.cue_weight * cue_score
            - self.unaffected_weight * unaffected_penalty
            + 0.25 * residual
        )
        return {
            "corrected_symptom_candidate_response_low": response,
            "candidate_cue_score_low": cue_score,
            "candidate_part_logit_bias_low": self.part_weight * part_log,
            "candidate_unaffected_surface_penalty_low": self.unaffected_weight * unaffected_penalty,
            "candidate_integration_residual_low": residual,
        }
