"""Route assignment, Spatial Support, route-specific responses, and Route-wise composition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import SYMPTOM_CLASS_NAMES, VISUAL_EVIDENCE_NAMES, ZONE_NAMES
from .common import (
    DepthwiseSeparableConv,
    FeatureAdapter,
    LogitHead,
    RefineFuseBlock,
    ResidualDSConvBlock,
    safe_log_prob,
    safe_multiclass_prob_from_logits,
    sanitize_logits_tensor,
)


class RouteAssignmentHead(nn.Module):
    """Assign each candidate pixel to a part route."""

    def __init__(
        self, dec_ch: int, num_parts: int, zone_ch: int, head_ch: int, prior_weight: float = 1.25
    ):
        super().__init__()
        self.num_parts = int(num_parts)
        self.prior_weight = float(prior_weight)
        m = max(16, dec_ch // 2)
        self.detail_adapter = FeatureAdapter(dec_ch, m)
        self.part_map_adapter = FeatureAdapter(num_parts, m)
        self.zone_adapter = FeatureAdapter(zone_ch, m)
        self.evidence_adapter = FeatureAdapter(len(VISUAL_EVIDENCE_NAMES), m)
        self.fuse = RefineFuseBlock(dec_ch + m + m + m + m, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3),
            ResidualDSConvBlock(dec_ch),
        )
        self.head = LogitHead(dec_ch, head_ch, num_parts)

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
                self.part_map_adapter(part_map),
                self.zone_adapter(zone_map),
                self.evidence_adapter(evidence),
            ],
            dim=1,
        )
        feat = self.context(self.fuse(x))
        raw = self.head(feat)
        prior = safe_log_prob(part_map.clamp_min(1e-6), eps=1e-6)
        logits = sanitize_logits_tensor(raw + float(self.prior_weight) * prior, clip=20.0)
        probability_with_background = safe_multiclass_prob_from_logits(
            logits, dim=1, clip=20.0
        )
        public_indices = (1, 4, 2, 3)
        route_logits = logits[:, public_indices]
        route_probability = probability_with_background[:, public_indices]
        return {
            "route_assignment_feature_low": feat,
            "route_assignment_raw_logits_low": raw,
            "route_assignment_prior_logits_low": prior,
            "route_assignment_with_background_logits_low": logits,
            "route_assignment_with_background_prob_low": probability_with_background,
            "route_assignment_logits_low": route_logits,
            "route_assignment_prob_low": route_probability,
        }


class SpatialSupportHead(nn.Module):
    """Estimate the eight Spatial Support responses used by the four routes."""

    SUPPORT_NAMES = tuple(f"{zone_name}_spatial_support" for zone_name in ZONE_NAMES)

    def __init__(
        self,
        dec_ch: int,
        num_routes: int,
        zone_ch: int,
        num_classes: int,
        semantic_class_names: Sequence[str],
        head_ch: int,
        prior_weight: float = 1.10,
        attach_kernel: int = 13,
        attach_body_weight: float = 0.65,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.semantic_class_names = list(semantic_class_names)
        self.prior_weight = float(prior_weight)
        self.attach_kernel = int(attach_kernel)
        self.attach_body_weight = float(attach_body_weight)
        m = max(16, dec_ch // 2)
        self.detail_adapter = FeatureAdapter(dec_ch, m)
        self.route_assignment_adapter = FeatureAdapter(num_routes + 1, m)
        self.zone_adapter = FeatureAdapter(zone_ch, m)
        self.evidence_adapter = FeatureAdapter(len(VISUAL_EVIDENCE_NAMES), m)
        self.candidate_adapter = FeatureAdapter(num_classes, m)
        self.fuse = RefineFuseBlock(dec_ch + m * 5, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3),
            ResidualDSConvBlock(dec_ch),
        )
        self.head = LogitHead(dec_ch, head_ch, len(self.SUPPORT_NAMES))
        self.class_groups = self._build_class_groups(self.semantic_class_names)

    def _build_class_groups(self, names: Sequence[str]) -> dict[str, list[int]]:
        if tuple(names) != SYMPTOM_CLASS_NAMES:
            raise ValueError("semantic_class_names must match the method class order")
        index = {name: position for position, name in enumerate(names)}
        return {
            "body": [index["body_lesion"]],
            "mouth": [index["mouth_ulcer"]],
            "fin_tip": [index["fin_deformity"]],
            "fin_middle": [index["fin_necrosis"]],
            "fin_base": [index["fin_base_necrosis"]],
            "caudal_fin_tip": [index["caudal_deformity"]],
            "caudal_fin_middle": [index["caudal_necrosis"]],
            "caudal_fin_base": [index["caudal_base_necrosis"]],
        }

    def _max_class(self, class_prob: torch.Tensor, key: str) -> torch.Tensor:
        ids = self.class_groups[key]
        return class_prob[:, ids].max(dim=1, keepdim=True)[0]

    def _dilate(self, x: torch.Tensor, k: int) -> torch.Tensor:
        k = max(1, int(k))
        if k % 2 == 0:
            k += 1
        if k <= 1:
            return x.clamp(0.0, 1.0)
        return F.max_pool2d(x.clamp(0.0, 1.0), kernel_size=k, stride=1, padding=k // 2)

    def _logit(self, p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return safe_log_prob(p, eps=eps) - safe_log_prob(1.0 - p, eps=eps)

    def _build_spatial_support_prior(
        self,
        route_assignment_with_background_prob: torch.Tensor,
        zone_map: torch.Tensor,
        corrected_symptom_candidate_response: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
        unaffected_surface_prob: torch.Tensor,
    ) -> torch.Tensor:
        body_assignment = route_assignment_with_background_prob[:, 1:2]
        fin_assignment = route_assignment_with_background_prob[:, 2:3]
        caudal_assignment = route_assignment_with_background_prob[:, 3:4]
        mouth_assignment = route_assignment_with_background_prob[:, 4:5]
        body_zone = zone_map[:, 0:1]
        mouth_zone = zone_map[:, 1:2]
        fin_tip = zone_map[:, 2:3]
        fin_middle = zone_map[:, 3:4]
        fin_base = zone_map[:, 4:5]
        caudal_fin_tip = zone_map[:, 5:6]
        caudal_fin_middle = zone_map[:, 6:7]
        caudal_fin_base = zone_map[:, 7:8]
        cp = torch.sigmoid(
            sanitize_logits_tensor(corrected_symptom_candidate_response, clip=20.0)
        ).clamp(0.0, 1.0)
        unaffected_support = (1.0 - 0.65 * unaffected_surface_prob).clamp(0.05, 1.0)
        fin_attach = torch.maximum(
            fin_base,
            float(self.attach_body_weight)
            * self._dilate(fin_base, self.attach_kernel)
            * body_assignment,
        ).clamp(0.0, 1.0)
        caudal_attach = torch.maximum(
            caudal_fin_base,
            float(self.attach_body_weight)
            * self._dilate(caudal_fin_base, self.attach_kernel)
            * body_assignment,
        ).clamp(0.0, 1.0)
        body = (
            body_assignment
            * body_zone
            * (0.50 * self._max_class(cp, "body") + 0.35 * lesion_prob + 0.15 * redness_prob)
            * unaffected_support
        ).clamp(0.0, 1.0)
        fin_base_spatial_support = (
            torch.maximum(fin_assignment, 0.35 * body_assignment)
            * fin_attach
            * (0.55 * self._max_class(cp, "fin_base") + 0.35 * lesion_prob + 0.10 * redness_prob)
            * unaffected_support
        ).clamp(0.0, 1.0)
        fin_middle_spatial_support = (
            fin_assignment
            * fin_middle
            * (0.55 * self._max_class(cp, "fin_middle") + 0.40 * lesion_prob + 0.05 * shape_prob)
            * unaffected_support
        ).clamp(0.0, 1.0)
        fin_tip_spatial_support = (
            fin_assignment
            * fin_tip
            * (0.55 * self._max_class(cp, "fin_tip") + 0.45 * shape_prob + 0.05 * lesion_prob)
            * unaffected_support
        ).clamp(0.0, 1.0)
        caudal_fin_base_spatial_support = (
            torch.maximum(caudal_assignment, 0.35 * body_assignment)
            * caudal_attach
            * (
                0.55 * self._max_class(cp, "caudal_fin_base")
                + 0.35 * lesion_prob
                + 0.10 * redness_prob
            )
            * unaffected_support
        ).clamp(0.0, 1.0)
        caudal_fin_middle_spatial_support = (
            caudal_assignment
            * caudal_fin_middle
            * (
                0.55 * self._max_class(cp, "caudal_fin_middle")
                + 0.40 * lesion_prob
                + 0.05 * shape_prob
            )
            * unaffected_support
        ).clamp(0.0, 1.0)
        caudal_fin_tip_spatial_support = (
            caudal_assignment
            * caudal_fin_tip
            * (
                0.55 * self._max_class(cp, "caudal_fin_tip")
                + 0.45 * shape_prob
                + 0.05 * lesion_prob
            )
            * unaffected_support
        ).clamp(0.0, 1.0)
        mouth_spatial_support = (
            mouth_assignment
            * mouth_zone
            * (0.60 * self._max_class(cp, "mouth") + 0.30 * lesion_prob + 0.10 * redness_prob)
            * unaffected_support
        ).clamp(0.0, 1.0)
        return torch.cat(
            [
                body,
                fin_base_spatial_support,
                fin_middle_spatial_support,
                fin_tip_spatial_support,
                caudal_fin_base_spatial_support,
                caudal_fin_middle_spatial_support,
                caudal_fin_tip_spatial_support,
                mouth_spatial_support,
            ],
            dim=1,
        ).clamp(0.0, 1.0)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        route_assignment_with_background_prob: torch.Tensor,
        zone_map: torch.Tensor,
        corrected_symptom_candidate_response: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
        unaffected_surface_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        evidence = torch.cat(
            [redness_prob, shape_prob, lesion_prob, unaffected_surface_prob], dim=1
        ).clamp(0.0, 1.0)
        candidate_prob = torch.sigmoid(
            sanitize_logits_tensor(corrected_symptom_candidate_response, clip=20.0)
        ).clamp(0.0, 1.0)
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.route_assignment_adapter(route_assignment_with_background_prob),
                self.zone_adapter(zone_map),
                self.evidence_adapter(evidence),
                self.candidate_adapter(candidate_prob),
            ],
            dim=1,
        )
        feat = self.context(self.fuse(x))
        raw = self.head(feat)
        prior = self._build_spatial_support_prior(
            route_assignment_with_background_prob,
            zone_map,
            corrected_symptom_candidate_response,
            redness_prob,
            shape_prob,
            lesion_prob,
            unaffected_surface_prob,
        )
        logits = sanitize_logits_tensor(
            raw + float(self.prior_weight) * self._logit(prior), clip=20.0
        )
        prob = torch.sigmoid(logits).clamp(0.0, 1.0)
        public_indices = (0, 7, 3, 2, 1, 6, 5, 4)
        public_prior = prior[:, public_indices]
        public_logits = logits[:, public_indices]
        public_prob = prob[:, public_indices]
        return {
            "spatial_support_feature_low": feat,
            "spatial_support_raw_logits_low": raw,
            "spatial_support_internal_prior_prob_low": prior,
            "spatial_support_internal_logits_low": logits,
            "spatial_support_internal_prob_low": prob,
            "spatial_support_prior_prob_low": public_prior,
            "spatial_support_logits_low": public_logits,
            "spatial_support_prob_low": public_prob,
            "body_spatial_support_prob_low": public_prob[:, 0:1],
            "mouth_spatial_support_prob_low": public_prob[:, 1:2],
            "fin_tip_spatial_support_prob_low": public_prob[:, 2:3],
            "fin_middle_spatial_support_prob_low": public_prob[:, 3:4],
            "fin_base_spatial_support_prob_low": public_prob[:, 4:5],
            "caudal_fin_tip_spatial_support_prob_low": public_prob[:, 5:6],
            "caudal_fin_middle_spatial_support_prob_low": public_prob[:, 6:7],
            "caudal_fin_base_spatial_support_prob_low": public_prob[:, 7:8],
        }


class RouteInterpretationHead(nn.Module):
    """Predict a route-specific class response from common feature F and candidate information."""

    def __init__(
        self, dec_ch: int, context_ch: int, out_ch: int, head_ch: int, prior_weight: float = 0.85
    ):
        super().__init__()
        self.out_ch = int(out_ch)
        self.prior_weight = float(prior_weight)
        m = max(16, dec_ch // 2)
        self.detail_adapter = FeatureAdapter(dec_ch, m)
        self.candidate_adapter = FeatureAdapter(context_ch, m)
        self.fuse = RefineFuseBlock(dec_ch + m + m, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3), ResidualDSConvBlock(dec_ch)
        )
        self.head = LogitHead(dec_ch, head_ch, out_ch)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        candidate_information: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.candidate_adapter(candidate_information),
            ],
            dim=1,
        )
        feat = self.context(self.fuse(x))
        raw = self.head(feat)
        logits = sanitize_logits_tensor(
            raw + float(self.prior_weight) * sanitize_logits_tensor(prior_logits, clip=20.0),
            clip=20.0,
        )
        prob = safe_multiclass_prob_from_logits(logits, dim=1, clip=20.0)
        return {
            "route_interpretation_feature_low": feat,
            "route_interpretation_raw_logits_low": raw,
            "route_response_logits_low": logits,
            "route_response_prob_low": prob,
        }


class RouteWiseComposer(nn.Module):
    """Compose body, mouth, fin, and caudal-fin responses in a common symptom-class space."""

    def __init__(
        self,
        semantic_class_names: Sequence[str],
        route_strength: float = 0.90,
        candidate_retention_weight: float = 0.12,
        overwrite_strength: float = 0.90,
        support_floor: float = 0.02,
    ):
        super().__init__()
        self.semantic_class_names = list(semantic_class_names)
        if tuple(self.semantic_class_names) != SYMPTOM_CLASS_NAMES:
            raise ValueError("semantic_class_names must match the method class order")
        self.num_classes = len(self.semantic_class_names)
        self.route_strength = float(route_strength)
        self.candidate_retention_weight = float(candidate_retention_weight)
        self.overwrite_strength = float(overwrite_strength)
        self.support_floor = float(support_floor)
        self.class_index = self._build_class_index(self.semantic_class_names)
        self.global_logit = nn.Parameter(torch.tensor(1.20))

    def _build_class_index(self, names: Sequence[str]) -> dict[str, int]:
        return {name: index for index, name in enumerate(names)}

    def _set_class(
        self, composed_response: torch.Tensor, key: str, value: torch.Tensor
    ) -> torch.Tensor:
        # Avoid in-place slices because the composed response participates in autograd.
        c = self.class_index[key]
        mask = composed_response.new_zeros((1, composed_response.shape[1], 1, 1))
        mask[:, c : c + 1] = 1.0
        v = value.clamp(0.0, 1.0)
        if v.shape[1] != 1:
            v = v[:, :1]
        v_full = v.expand(-1, composed_response.shape[1], -1, -1)
        candidate = torch.maximum(composed_response, v_full)
        return (composed_response * (1.0 - mask) + candidate * mask).clamp(0.0, 1.0)

    def _compose_group(
        self,
        composed_response: torch.Tensor,
        group_gate: torch.Tensor,
        updates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        gate = group_gate.clamp(0.0, 1.0)
        composed_response = composed_response * (1.0 - float(self.overwrite_strength) * gate)
        for key, val in updates.items():
            composed_response = self._set_class(composed_response, key, val)
        return composed_response.clamp(0.0, 1.0)

    def _logit(self, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return safe_log_prob(p, eps=eps) - safe_log_prob(1.0 - p, eps=eps)

    def forward(
        self,
        corrected_symptom_candidate_response: torch.Tensor,
        fish_region: torch.Tensor,
        route_assignment_prob: torch.Tensor,
        spatial_support_prob: torch.Tensor,
        body_route_response_prob: torch.Tensor,
        fin_route_response_prob: torch.Tensor,
        caudal_fin_route_response_prob: torch.Tensor,
        mouth_route_response_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        corrected_candidate_probability = torch.sigmoid(
            sanitize_logits_tensor(corrected_symptom_candidate_response, clip=20.0)
        ).clamp(0.0, 1.0)
        body_assignment = route_assignment_prob[:, 0:1]
        mouth_assignment = route_assignment_prob[:, 1:2]
        fin_assignment = route_assignment_prob[:, 2:3]
        caudal_assignment = route_assignment_prob[:, 3:4]
        body_spatial_support = spatial_support_prob[:, 0:1]
        mouth_spatial_support = spatial_support_prob[:, 1:2]
        fin_tip_spatial_support, fin_middle_spatial_support, fin_base_spatial_support = (
            spatial_support_prob[:, 2:3],
            spatial_support_prob[:, 3:4],
            spatial_support_prob[:, 4:5],
        )
        (
            caudal_fin_tip_spatial_support,
            caudal_fin_middle_spatial_support,
            caudal_fin_base_spatial_support,
        ) = (
            spatial_support_prob[:, 5:6],
            spatial_support_prob[:, 6:7],
            spatial_support_prob[:, 7:8],
        )

        # Route-response class orders:
        # body: [none, body_lesion]
        # fin: [none, fin_deformity, fin_necrosis, fin_base_necrosis]
        # caudal: [none, caudal_deformity, caudal_necrosis, caudal_base_necrosis]
        # mouth: [none, mouth_ulcer]
        composed_response = torch.zeros_like(corrected_candidate_probability)
        support = torch.full_like(
            corrected_candidate_probability, float(self.support_floor)
        ) * fish_region.clamp(0.0, 1.0)

        body_sig = (
            body_assignment * body_spatial_support * body_route_response_prob[:, 1:2]
        ).clamp(0.0, 1.0)
        composed_response = self._set_class(composed_response, "body_lesion", body_sig)
        support = self._set_class(
            support,
            "body_lesion",
            (body_assignment * body_spatial_support).clamp(0.0, 1.0),
        )

        fin_gate = (
            fin_assignment
            * torch.maximum(
                torch.maximum(fin_base_spatial_support, fin_middle_spatial_support),
                fin_tip_spatial_support,
            )
        ).clamp(0.0, 1.0)
        fin_updates = {
            "fin_deformity": (
                fin_assignment * fin_tip_spatial_support * fin_route_response_prob[:, 1:2]
            ).clamp(0.0, 1.0),
            "fin_necrosis": (
                fin_assignment * fin_middle_spatial_support * fin_route_response_prob[:, 2:3]
            ).clamp(0.0, 1.0),
            "fin_base_necrosis": (
                torch.maximum(fin_assignment, 0.35 * body_assignment)
                * fin_base_spatial_support
                * fin_route_response_prob[:, 3:4]
            ).clamp(0.0, 1.0),
        }
        composed_response = self._compose_group(composed_response, fin_gate, fin_updates)
        support = self._set_class(
            support,
            "fin_deformity",
            (fin_assignment * fin_tip_spatial_support).clamp(0, 1),
        )
        support = self._set_class(
            support,
            "fin_necrosis",
            (fin_assignment * fin_middle_spatial_support).clamp(0, 1),
        )
        support = self._set_class(
            support,
            "fin_base_necrosis",
            (
                torch.maximum(fin_assignment, 0.35 * body_assignment) * fin_base_spatial_support
            ).clamp(0, 1),
        )

        caudal_gate = (
            caudal_assignment
            * torch.maximum(
                torch.maximum(
                    caudal_fin_base_spatial_support,
                    caudal_fin_middle_spatial_support,
                ),
                caudal_fin_tip_spatial_support,
            )
        ).clamp(0.0, 1.0)
        caudal_updates = {
            "caudal_deformity": (
                caudal_assignment
                * caudal_fin_tip_spatial_support
                * caudal_fin_route_response_prob[:, 1:2]
            ).clamp(0.0, 1.0),
            "caudal_necrosis": (
                caudal_assignment
                * caudal_fin_middle_spatial_support
                * caudal_fin_route_response_prob[:, 2:3]
            ).clamp(0.0, 1.0),
            "caudal_base_necrosis": (
                torch.maximum(caudal_assignment, 0.35 * body_assignment)
                * caudal_fin_base_spatial_support
                * caudal_fin_route_response_prob[:, 3:4]
            ).clamp(0.0, 1.0),
        }
        composed_response = self._compose_group(composed_response, caudal_gate, caudal_updates)
        support = self._set_class(
            support,
            "caudal_deformity",
            (caudal_assignment * caudal_fin_tip_spatial_support).clamp(0, 1),
        )
        support = self._set_class(
            support,
            "caudal_necrosis",
            (caudal_assignment * caudal_fin_middle_spatial_support).clamp(0, 1),
        )
        support = self._set_class(
            support,
            "caudal_base_necrosis",
            (
                torch.maximum(caudal_assignment, 0.35 * body_assignment)
                * caudal_fin_base_spatial_support
            ).clamp(0, 1),
        )

        mouth_gate = (
            torch.maximum(mouth_assignment, 0.25 * body_assignment) * mouth_spatial_support
        ).clamp(0.0, 1.0)
        mouth_updates = {
            "mouth_ulcer": (mouth_gate * mouth_route_response_prob[:, 1:2]).clamp(0.0, 1.0)
        }
        composed_response = self._compose_group(composed_response, mouth_gate, mouth_updates)
        support = self._set_class(support, "mouth_ulcer", mouth_gate)

        route_confidence = composed_response.max(dim=1, keepdim=True)[0].clamp(0.0, 1.0)
        retained_candidate_probability = (
            float(self.candidate_retention_weight)
            * corrected_candidate_probability
            * (1.0 - route_confidence)
        )
        response_with_retained_candidate = torch.maximum(
            composed_response,
            retained_candidate_probability,
        ).clamp(1e-6, 1.0 - 1e-6)
        strength = float(self.route_strength) * torch.sigmoid(self.global_logit).clamp(0.0, 1.0)
        routed_semantic_probability = (
            (1.0 - strength) * corrected_candidate_probability
            + strength * response_with_retained_candidate
        ).clamp(1e-6, 1.0 - 1e-6)
        logits = sanitize_logits_tensor(self._logit(routed_semantic_probability), clip=20.0)
        route_class_support = torch.maximum(
            support,
            float(self.support_floor) * fish_region,
        ).clamp(0.0, 1.0)
        return {
            "routed_semantic_logits_low": logits,
            "routed_semantic_probability_low": routed_semantic_probability,
            "route_composed_probability_low": composed_response,
            "retained_candidate_probability_low": retained_candidate_probability,
            "route_class_support_low": route_class_support,
            "route_confidence_low": route_confidence,
            "route_delta_logits_low": sanitize_logits_tensor(
                logits - corrected_symptom_candidate_response, clip=20.0
            ),
            "route_global_strength": strength.detach(),
        }
