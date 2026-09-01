"""Final refinement of Routed semantic logits into the auxiliary output."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..schema import VISUAL_EVIDENCE_NAMES
from .common import (
    DepthwiseSeparableConv,
    FeatureAdapter,
    LogitHead,
    RefineFuseBlock,
    ResidualDSConvBlock,
    sanitize_logits_tensor,
)


class FinalRefinement(nn.Module):
    """Predict bounded corrections near semantic boundaries."""

    def __init__(
        self, dec_ch: int, num_classes: int, zone_ch: int, head_ch: int, strength: float = 0.35
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.strength = float(strength)
        m = max(16, dec_ch // 2)
        self.fine_grained_detail_adapter = FeatureAdapter(dec_ch, m)
        self.zone_adapter = FeatureAdapter(zone_ch, m)
        self.class_adapter = FeatureAdapter(num_classes * 2 + 1, m)
        self.evidence_adapter = FeatureAdapter(len(VISUAL_EVIDENCE_NAMES) + 2, m)
        self.fuse = RefineFuseBlock(dec_ch + m + m + m + m, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=1),
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=2),
            ResidualDSConvBlock(dec_ch),
        )
        self.sdf_head = LogitHead(dec_ch, head_ch, 1)
        self.fg_delta_head = LogitHead(dec_ch, head_ch, 1)
        self.class_delta_head = LogitHead(dec_ch, head_ch, num_classes)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        zone_map: torch.Tensor,
        native_semantic_logits: torch.Tensor,
        routed_class_probability: torch.Tensor,
        routed_class_distribution: torch.Tensor,
        symptom_foreground_probability: torch.Tensor,
        fish_region: torch.Tensor,
        fish_boundary: torch.Tensor,
        redness_prob: torch.Tensor,
        shape_prob: torch.Tensor,
        lesion_prob: torch.Tensor,
        unaffected_surface_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        refinement_evidence = torch.cat(
            [
                redness_prob,
                shape_prob,
                lesion_prob,
                unaffected_surface_prob,
                fish_boundary,
                fish_region,
            ],
            dim=1,
        ).clamp(0.0, 1.0)
        routed_class_context = torch.cat(
            [
                routed_class_probability,
                routed_class_distribution,
                symptom_foreground_probability,
            ],
            dim=1,
        ).clamp(0.0, 1.0)
        x = torch.cat(
            [
                common_feature,
                self.fine_grained_detail_adapter(fine_grained_detail_feature),
                self.zone_adapter(zone_map),
                self.class_adapter(routed_class_context),
                self.evidence_adapter(refinement_evidence),
            ],
            dim=1,
        )
        feature = self.context(self.fuse(x))
        signed_distance = self.sdf_head(feature)
        foreground_delta = torch.tanh(self.fg_delta_head(feature)) * self.strength
        class_delta = torch.tanh(self.class_delta_head(feature)) * (0.50 * self.strength)
        # Functional refinement; avoid in-place slice writes on tensors in the
        # computation graph.
        background_refined = native_semantic_logits[:, 0:1] - 0.25 * foreground_delta
        symptom_refined = native_semantic_logits[:, 1:] + foreground_delta + class_delta
        auxiliary_semantic_logits = sanitize_logits_tensor(
            torch.cat([background_refined, symptom_refined], dim=1), clip=20.0
        )
        return {
            "final_refinement_feature_low": feature,
            "final_refinement_signed_distance_low": signed_distance,
            "final_refinement_foreground_delta_low": foreground_delta,
            "final_refinement_class_delta_low": class_delta,
            "auxiliary_semantic_logits_low": auxiliary_semantic_logits,
        }
