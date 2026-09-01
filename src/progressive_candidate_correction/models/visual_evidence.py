"""Estimators for the four Visual evidence channels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import VISUAL_EVIDENCE_NAMES
from .common import (
    DepthwiseSeparableConv,
    FeatureAdapter,
    LogitHead,
    RefineFuseBlock,
    ResidualDSConvBlock,
    _imagenet_denorm,
    masked_average_pool,
    masked_weighted_mean,
    masked_weighted_std,
)


class RednessEvidenceEstimator(nn.Module):
    def __init__(
        self,
        dec_ch: int,
        num_parts: int,
        head_ch: int,
        local_kernel: int = 31,
        input_is_imagenet_normalized: bool = True,
    ):
        super().__init__()
        self.local_kernel = int(local_kernel)
        self.input_is_imagenet_normalized = bool(input_is_imagenet_normalized)

        m = max(16, dec_ch // 2)
        self.prior_adapter = FeatureAdapter(1, m)
        self.fish_region_adapter = FeatureAdapter(1, m)
        self.part_adapter = FeatureAdapter(num_parts, m)
        self.fuse = RefineFuseBlock(dec_ch + m + m + m, dec_ch)
        self.head = LogitHead(dec_ch, head_ch, 1)
        self.prior_gain = nn.Parameter(torch.tensor(0.65))

    def _build_prior(
        self, image: torch.Tensor, fish_region_prob_low: torch.Tensor, size_hw: tuple[int, int]
    ) -> torch.Tensor:
        rgb = (
            _imagenet_denorm(image) if self.input_is_imagenet_normalized else image.clamp(0.0, 1.0)
        )
        rgb = F.interpolate(rgb, size=size_hw, mode="bilinear", align_corners=False)
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        red_excess = (r - 0.5 * (g + b)).clamp_min(0.0)

        fish_region = fish_region_prob_low.clamp(0.0, 1.0)
        gmean = masked_weighted_mean(red_excess, fish_region)
        gstd = masked_weighted_std(red_excess, fish_region, gmean).clamp_min(0.03)
        global_rel = (red_excess - gmean) / gstd
        local_mean = masked_average_pool(red_excess, fish_region, kernel_size=self.local_kernel)
        local_rel = (red_excess - local_mean) / gstd

        prior = F.relu(0.55 * global_rel + 0.45 * local_rel)
        prior = torch.tanh(prior / 2.0).clamp_min(0.0)
        return prior

    def forward(
        self,
        image: torch.Tensor,
        common_feature: torch.Tensor,
        fish_region_prob_low: torch.Tensor,
        part_map_low: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prior = self._build_prior(image, fish_region_prob_low, common_feature.shape[-2:])
        feat = self.fuse(
            torch.cat(
                [
                    common_feature,
                    self.prior_adapter(prior),
                    self.fish_region_adapter(fish_region_prob_low),
                    self.part_adapter(part_map_low),
                ],
                dim=1,
            )
        )
        residual = self.head(feat)
        logits = residual + self.prior_gain * prior
        return {
            "redness_evidence_feature_low": feat,
            "redness_evidence_logits_low": logits,
            "redness_prior_low": prior,
            "redness_residual_logits_low": residual,
        }


class ShapeEvidenceEstimator(nn.Module):
    def __init__(self, dec_ch: int, head_ch: int):
        super().__init__()
        m = max(16, dec_ch // 2)
        self.edge_adapter = FeatureAdapter(3, m)
        self.zone_adapter = FeatureAdapter(4, m)
        self.detail_adapter = FeatureAdapter(dec_ch, m)
        self.fuse = RefineFuseBlock(dec_ch + m + m + m, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=1),
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=2),
            ResidualDSConvBlock(dec_ch),
        )
        self.head = LogitHead(dec_ch, head_ch, 1)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        fish_boundary_low: torch.Tensor,
        fin_tip_zone_low: torch.Tensor,
        fin_base_zone_low: torch.Tensor,
        caudal_fin_tip_zone_low: torch.Tensor,
        caudal_fin_base_zone_low: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        edge_stack = torch.cat(
            [fish_boundary_low, fin_tip_zone_low, caudal_fin_tip_zone_low], dim=1
        )
        zone_stack = torch.cat(
            [
                fin_tip_zone_low,
                fin_base_zone_low,
                caudal_fin_tip_zone_low,
                caudal_fin_base_zone_low,
            ],
            dim=1,
        )
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.edge_adapter(edge_stack),
                self.zone_adapter(zone_stack),
            ],
            dim=1,
        )
        feat = self.context(self.fuse(x))
        logits = self.head(feat)
        return {
            "shape_evidence_feature_low": feat,
            "shape_evidence_logits_low": logits,
            "shape_boundary_stack_low": edge_stack,
            "shape_zone_stack_low": zone_stack,
        }


class LesionEvidenceEstimator(nn.Module):
    def __init__(self, dec_ch: int, num_parts: int, head_ch: int):
        super().__init__()
        m = max(16, dec_ch // 2)
        self.fish_region_adapter = FeatureAdapter(1, m)
        self.part_adapter = FeatureAdapter(num_parts, m)
        self.detail_adapter = FeatureAdapter(dec_ch, m)
        self.contrast_adapter = FeatureAdapter(dec_ch, m)
        self.var_adapter = FeatureAdapter(1, m)

        self.fuse = RefineFuseBlock(dec_ch + m + m + m + m + m, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=1),
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=2),
            ResidualDSConvBlock(dec_ch),
        )
        self.head = LogitHead(dec_ch, head_ch, 1)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        fish_interior_low: torch.Tensor,
        part_map_low: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_mean = F.avg_pool2d(
            fine_grained_detail_feature, kernel_size=7, stride=1, padding=3
        )
        contrast = fine_grained_detail_feature - local_mean
        var_map = contrast.pow(2).mean(dim=1, keepdim=True)

        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.contrast_adapter(contrast),
                self.var_adapter(var_map),
                self.fish_region_adapter(fish_interior_low),
                self.part_adapter(part_map_low),
            ],
            dim=1,
        )
        feat = self.context(self.fuse(x))
        logits = self.head(feat)
        return {
            "lesion_evidence_feature_low": feat,
            "lesion_evidence_logits_low": logits,
            "lesion_local_mean_low": local_mean,
            "lesion_var_low": var_map,
        }


class UnaffectedSurfaceEvidenceEstimator(nn.Module):
    """Estimate conservative visual evidence for Unaffected Surface."""

    def __init__(self, dec_ch: int, num_parts: int, zone_ch: int, head_ch: int):
        super().__init__()
        m = max(16, dec_ch // 2)
        self.detail_adapter = FeatureAdapter(dec_ch, m)
        self.fish_region_adapter = FeatureAdapter(1, m)
        self.part_adapter = FeatureAdapter(num_parts, m)
        self.zone_adapter = FeatureAdapter(zone_ch, m)
        self.positive_visual_evidence_adapter = FeatureAdapter(len(VISUAL_EVIDENCE_NAMES) - 1, m)

        self.fuse = RefineFuseBlock(dec_ch + m + m + m + m + m, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=1),
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3, dilation=2),
            ResidualDSConvBlock(dec_ch),
        )
        self.head = LogitHead(dec_ch, head_ch, 1)

        # Conservative initialization limits early Unaffected Surface inhibition.
        last = self.head.block[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.constant_(last.bias, -2.0)

    def forward(
        self,
        common_feature: torch.Tensor,
        fine_grained_detail_feature: torch.Tensor,
        fish_region_prob_low: torch.Tensor,
        part_map_low: torch.Tensor,
        zone_map_low: torch.Tensor,
        positive_visual_evidence_prob_low: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        positive_visual_evidence = positive_visual_evidence_prob_low[:, :3].clamp(0.0, 1.0)
        x = torch.cat(
            [
                common_feature,
                self.detail_adapter(fine_grained_detail_feature),
                self.fish_region_adapter(fish_region_prob_low),
                self.part_adapter(part_map_low),
                self.zone_adapter(zone_map_low),
                self.positive_visual_evidence_adapter(positive_visual_evidence),
            ],
            dim=1,
        )
        feat = self.context(self.fuse(x))
        logits = self.head(feat)
        return {
            "unaffected_surface_evidence_feature_low": feat,
            "unaffected_surface_evidence_logits_low": logits,
        }
