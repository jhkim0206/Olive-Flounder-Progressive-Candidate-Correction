"""Fish Region, Part Map, Head-to-Tail Direction, and Zone Map estimators."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import ZONE_NAMES as CANONICAL_ZONE_NAMES
from .common import (
    DepthwiseSeparableConv,
    FeatureAdapter,
    LogitHead,
    MultiScaleContextLite,
    RefineFuseBlock,
    ResidualDSConvBlock,
    _imagenet_denorm,
    _soft_close,
    _soft_open,
    _weighted_pool,
    safe_binary_prob_from_logits,
    safe_log_prob,
    soft_boundary_map,
)


class FishRegionEstimator(nn.Module):
    """Estimate Fish Region from the common feature and an image prior."""

    def __init__(self, dec_ch: int, head_ch: int, prior_ch: int = 6):
        super().__init__()
        self.context = nn.Sequential(
            MultiScaleContextLite(dec_ch, dec_ch),
            ResidualDSConvBlock(dec_ch),
        )
        self.raw_head = LogitHead(dec_ch, head_ch, 1)

        m = max(16, dec_ch // 2)
        self.common_feature_adapter = FeatureAdapter(dec_ch, m)
        self.prior_adapter = FeatureAdapter(prior_ch, m)
        self.prior_fuse = RefineFuseBlock(m + m, dec_ch)
        self.prior_head = LogitHead(dec_ch, head_ch, 1)

        self.prior_scale = nn.Parameter(torch.tensor(0.85))
        self.heuristic_scale = nn.Parameter(torch.tensor(0.65))
        self.opening_refinement_mix = nn.Parameter(torch.tensor(0.20))
        self.closing_refinement_mix = nn.Parameter(torch.tensor(0.25))
        self.center_refinement_mix = nn.Parameter(torch.tensor(0.15))

    def _build_prior_maps(
        self, image: torch.Tensor, size_hw: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        rgb = _imagenet_denorm(image)
        rgb = F.interpolate(rgb, size=size_hw, mode="bilinear", align_corners=False)

        gray = rgb.mean(dim=1, keepdim=True)
        sat = rgb.max(dim=1, keepdim=True).values - rgb.min(dim=1, keepdim=True).values
        local_mean = F.avg_pool2d(gray, kernel_size=15, stride=1, padding=7)
        local_var = F.avg_pool2d((gray - local_mean) ** 2, kernel_size=15, stride=1, padding=7)

        B, _, H, W = gray.shape
        ys = (
            torch.linspace(-1.0, 1.0, H, device=gray.device, dtype=gray.dtype)
            .view(1, 1, H, 1)
            .expand(B, 1, H, W)
        )
        xs = (
            torch.linspace(-1.0, 1.0, W, device=gray.device, dtype=gray.dtype)
            .view(1, 1, 1, W)
            .expand(B, 1, H, W)
        )
        center_prior = torch.exp(-((xs / 0.70) ** 2 + (ys / 0.58) ** 2))
        edge_pen = torch.clamp(1.0 - center_prior, 0.0, 1.0)

        bright_flat = torch.sigmoid((gray - 0.78) / 0.06) * torch.sigmoid((0.07 - sat) / 0.03)
        surface_variation = torch.sigmoid((sat - 0.08) / 0.04) + torch.sigmoid(
            (local_var - 0.0025) / 0.0015
        )
        surface_variation = 0.5 * surface_variation
        fg_heur = (
            0.65 * center_prior + 0.45 * surface_variation - 0.70 * bright_flat - 0.20 * edge_pen
        )
        fg_heur = torch.sigmoid(fg_heur)
        heur_logit = safe_log_prob(fg_heur.clamp(1e-3, 1 - 1e-3), eps=1e-3) - safe_log_prob(
            (1 - fg_heur).clamp(1e-3, 1 - 1e-3), eps=1e-3
        )

        prior_maps = torch.cat([gray, sat, local_var, center_prior, bright_flat, fg_heur], dim=1)
        return {
            "prior_maps": prior_maps,
            "center_prior": center_prior,
            "bright_flat": bright_flat,
            "fg_heur": fg_heur,
            "heur_logit": heur_logit,
        }

    def _refine_fish_region(
        self, fish_region_prob: torch.Tensor, center_prior: torch.Tensor
    ) -> torch.Tensor:
        opened = _soft_open(fish_region_prob, k=3)
        closed = _soft_close(fish_region_prob, k=5)
        refined_fish_region = fish_region_prob
        refined_fish_region = refined_fish_region + torch.sigmoid(self.opening_refinement_mix) * (
            opened - fish_region_prob
        )
        refined_fish_region = refined_fish_region + torch.sigmoid(self.closing_refinement_mix) * (
            closed - refined_fish_region
        )
        refined_fish_region = refined_fish_region * (
            1.0
            - torch.sigmoid(self.center_refinement_mix)
            + torch.sigmoid(self.center_refinement_mix) * center_prior
        )
        return refined_fish_region.clamp(0.0, 1.0)

    def forward(self, common_feature: torch.Tensor, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.context(common_feature)
        raw_logits = self.raw_head(feat)

        prior_dict = self._build_prior_maps(image, common_feature.shape[-2:])
        prior_feat = self.prior_fuse(
            torch.cat(
                [
                    self.common_feature_adapter(common_feature),
                    self.prior_adapter(prior_dict["prior_maps"]),
                ],
                dim=1,
            )
        )
        prior_logits = self.prior_head(prior_feat)

        fused_logits = (
            raw_logits
            + self.prior_scale * prior_logits
            + self.heuristic_scale * prior_dict["heur_logit"]
        )
        fish_region_prob = safe_binary_prob_from_logits(fused_logits)
        refined_fish_region = self._refine_fish_region(fish_region_prob, prior_dict["center_prior"])

        return {
            "fish_region_feature_low": feat,
            "fish_region_raw_logits_low": raw_logits,
            "fish_region_prior_logits_low": prior_logits,
            "fish_prior_maps_low": prior_dict["prior_maps"],
            "fish_center_prior_low": prior_dict["center_prior"],
            "fish_bright_flat_low": prior_dict["bright_flat"],
            "fish_heur_prob_low": prior_dict["fg_heur"],
            "fish_heur_logit_low": prior_dict["heur_logit"],
            "fish_region_logits_low": fused_logits,
            "fish_region_prob_low": fish_region_prob,
            "refined_fish_region_low": refined_fish_region,
        }


class HeadToTailDirectionEstimator(nn.Module):
    """Build a fish-aligned coordinate system from second-order moments."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def _grid(self, fish_region_prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = fish_region_prob.shape
        device, dtype = fish_region_prob.device, fish_region_prob.dtype
        ys = (
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            .view(1, 1, H, 1)
            .expand(B, 1, H, W)
        )
        xs = (
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            .view(1, 1, 1, W)
            .expand(B, 1, H, W)
        )
        return xs, ys

    def forward(
        self, fish_region_prob: torch.Tensor, endpoint_role_prob: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        fish_region = fish_region_prob.clamp(0.0, 1.0)
        B, _, H, W = fish_region.shape
        xs, ys = self._grid(fish_region)
        w = fish_region + self.eps
        mass = w.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        cx = (xs * w).sum(dim=(2, 3), keepdim=True) / mass
        cy = (ys * w).sum(dim=(2, 3), keepdim=True) / mass
        dx = xs - cx
        dy = ys - cy
        cxx = (w * dx * dx).sum(dim=(2, 3), keepdim=True) / mass
        cyy = (w * dy * dy).sum(dim=(2, 3), keepdim=True) / mass
        cxy = (w * dx * dy).sum(dim=(2, 3), keepdim=True) / mass
        theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy + self.eps)
        ux, uy = torch.cos(theta), torch.sin(theta)
        vx, vy = -uy, ux
        s_raw = dx * ux + dy * uy
        t_raw = dx * vx + dy * vy
        s_std = torch.sqrt((w * s_raw.pow(2)).sum(dim=(2, 3), keepdim=True) / mass + self.eps)
        t_std = torch.sqrt((w * t_raw.pow(2)).sum(dim=(2, 3), keepdim=True) / mass + self.eps)
        s_norm = (s_raw / (2.25 * s_std + self.eps)).clamp(-1.5, 1.5)
        t_norm = (t_raw / (2.25 * t_std + self.eps)).clamp(-1.5, 1.5)

        endpoint_a_prior = (
            torch.sigmoid((-s_norm - 0.48) / 0.11) * (0.35 + 0.65 * fish_region)
        ).clamp(0.0, 1.0)
        endpoint_b_prior = (
            torch.sigmoid((s_norm - 0.48) / 0.11) * (0.35 + 0.65 * fish_region)
        ).clamp(0.0, 1.0)

        if endpoint_role_prob is None:
            role = fish_region.new_full((B, 1, 1, 1), 0.5)
        else:
            role = endpoint_role_prob.float()
            if role.ndim == 2:
                role = role.view(B, 1, 1, 1)
            elif role.ndim == 1:
                role = role.view(B, 1, 1, 1)
            elif role.ndim == 4:
                role = role[:, :1].mean(dim=(2, 3), keepdim=True)
            role = role.clamp(0.0, 1.0)

        # role = P(endpoint_a is head). signed_s is negative near head and positive near tail.
        signed_s = ((2.0 * role - 1.0) * s_norm).clamp(-1.5, 1.5)
        head_prior = (role * endpoint_a_prior + (1.0 - role) * endpoint_b_prior).clamp(0.0, 1.0)
        tail_prior = (role * endpoint_b_prior + (1.0 - role) * endpoint_a_prior).clamp(0.0, 1.0)
        abs_s = s_norm.abs().clamp(0.0, 1.5)
        t_lateral = t_norm.abs().clamp(0.0, 1.5)

        fish_boundary = soft_boundary_map(fish_region)
        fish_boundary = (
            fish_boundary + 0.35 * F.avg_pool2d(fish_boundary, kernel_size=5, stride=1, padding=2)
        ).clamp(0.0, 1.0)
        fish_contact = (
            fish_boundary + 0.65 * F.avg_pool2d(fish_boundary, kernel_size=9, stride=1, padding=4)
        ).clamp(0.0, 1.0)
        fish_interior = (fish_region * (1.0 - fish_boundary)).clamp(0.0, 1.0)

        midbody_prior = torch.exp(-((abs_s / 0.58).pow(2))).clamp(0.0, 1.0)
        lateral_prior = torch.sigmoid((t_lateral - 0.32) / 0.10).clamp(0.0, 1.0)
        center_dist = torch.sqrt(xs.pow(2) + ys.pow(2)).clamp(0.0, 2.0) / 2.0
        endpoint_role_map = role.expand(B, 1, H, W)

        return {
            "x_norm": xs,
            "y_norm": ys,
            "fish_center_x": cx,
            "fish_center_y": cy,
            "axis_ux": ux,
            "axis_uy": uy,
            "axis_vx": vx,
            "axis_vy": vy,
            "s_raw": s_raw,
            "t_raw": t_raw,
            "s_norm": s_norm,
            "t_norm": t_norm,
            "signed_s": signed_s,
            "abs_s": abs_s,
            "t_lateral": t_lateral,
            "center_dist": center_dist,
            "fish_boundary": fish_boundary,
            "fish_contact": fish_contact,
            "fish_interior": fish_interior,
            "endpoint_a_prior": endpoint_a_prior,
            "endpoint_b_prior": endpoint_b_prior,
            "endpoint_role_prob": role,
            "endpoint_role_map": endpoint_role_map,
            "head_prior": head_prior,
            "tail_prior": tail_prior,
            "midbody_prior": midbody_prior,
            "lateral_prior": lateral_prior,
        }


class HeadEndpointRoleHead(nn.Module):
    """Predicts P(endpoint_a is the head) from endpoint-local features."""

    def __init__(self, dec_ch: int, hidden_ch: int = 64):
        super().__init__()
        in_dim = dec_ch * 4 + 10
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_ch),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_ch, hidden_ch),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_ch, 1),
        )
        # Start neutral so the initial direction estimate is not biased to either endpoint.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        common_feature: torch.Tensor,
        fish_region_feature: torch.Tensor,
        refined_fish_region: torch.Tensor,
        base_coord: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ep_a = base_coord["endpoint_a_prior"] * refined_fish_region
        ep_b = base_coord["endpoint_b_prior"] * refined_fish_region
        fa = _weighted_pool(common_feature, ep_a)
        fb = _weighted_pool(common_feature, ep_b)
        ba = _weighted_pool(fish_region_feature, ep_a)
        bb = _weighted_pool(fish_region_feature, ep_b)
        # simple endpoint descriptors
        stats = torch.cat(
            [
                ep_a.mean(dim=(2, 3)),
                ep_b.mean(dim=(2, 3)),
                (ep_a * base_coord["fish_boundary"]).mean(dim=(2, 3)),
                (ep_b * base_coord["fish_boundary"]).mean(dim=(2, 3)),
                (ep_a * base_coord["x_norm"]).mean(dim=(2, 3)),
                (ep_b * base_coord["x_norm"]).mean(dim=(2, 3)),
                (ep_a * base_coord["y_norm"]).mean(dim=(2, 3)),
                (ep_b * base_coord["y_norm"]).mean(dim=(2, 3)),
                (ep_a * base_coord["t_lateral"]).mean(dim=(2, 3)),
                (ep_b * base_coord["t_lateral"]).mean(dim=(2, 3)),
            ],
            dim=1,
        )
        z = torch.cat([fa, fb, fa - fb, ba - bb, stats], dim=1)
        logit = self.mlp(z).view(-1, 1, 1, 1)
        prob = torch.sigmoid(logit)
        return {
            "endpoint_role_logit_low": logit,
            "endpoint_role_prob_low": prob,
        }


class PartMapPriorBuilder(nn.Module):
    """Build the direction-conditioned prior used by the Part Map estimator."""

    FOREGROUND_PART_NAMES = ("body", "fin", "caudal_fin", "mouth")

    def __init__(self):
        super().__init__()

    def forward(
        self, refined_fish_region: torch.Tensor, coord: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        fish_contact = coord["fish_contact"]
        fish_interior = coord["fish_interior"]
        head_prior = coord["head_prior"]
        tail_prior = coord["tail_prior"]
        midbody_prior = coord["midbody_prior"]
        lateral_prior = coord["lateral_prior"]
        signed_s = coord["signed_s"]

        head_gate = torch.sigmoid((-signed_s - 0.55) / 0.08).clamp(0.0, 1.0)
        tail_gate = torch.sigmoid((signed_s - 0.55) / 0.08).clamp(0.0, 1.0)
        mouth_part_prior = (
            refined_fish_region * head_prior.pow(1.8) * head_gate * (0.25 + 0.75 * fish_contact)
        ).clamp(0.0, 1.0)
        caudal_part_prior = (
            refined_fish_region * tail_prior.pow(1.35) * tail_gate * (0.40 + 0.60 * fish_contact)
        ).clamp(0.0, 1.0)
        fin_part_prior = (
            refined_fish_region * midbody_prior * lateral_prior * (0.55 + 0.45 * fish_contact)
        ).clamp(0.0, 1.0)
        specialized_part_support = torch.maximum(
            torch.maximum(mouth_part_prior, caudal_part_prior), 0.85 * fin_part_prior
        )
        body_part_prior = (fish_interior * (1.0 - specialized_part_support)).clamp(0.0, 1.0)
        body_part_prior = torch.maximum(
            body_part_prior,
            0.18 * refined_fish_region * midbody_prior * (1.0 - 0.55 * lateral_prior),
        ).clamp(0.0, 1.0)

        part_map_prior = torch.cat(
            [body_part_prior, fin_part_prior, caudal_part_prior, mouth_part_prior], dim=1
        ).clamp_min(1e-6)
        part_map_prior = part_map_prior / part_map_prior.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "part_map_prior_low": part_map_prior,
            "body_part_prior_low": part_map_prior[:, 0:1],
            "fin_part_prior_low": part_map_prior[:, 1:2],
            "caudal_part_prior_low": part_map_prior[:, 2:3],
            "mouth_part_prior_low": part_map_prior[:, 3:4],
            "foreground_part_names": self.FOREGROUND_PART_NAMES,
        }


class PartMapEstimator(nn.Module):
    def __init__(
        self,
        dec_ch: int,
        head_ch: int,
        num_foreground_parts: int,
        coord_ch: int = 12,
        part_map_prior_scale: float = 0.85,
    ):
        super().__init__()
        self.num_foreground_parts = int(num_foreground_parts)
        self.part_map_prior_scale = float(part_map_prior_scale)
        m = max(16, dec_ch // 2)
        self.fish_region_adapter = FeatureAdapter(1, m)
        self.coord_adapter = FeatureAdapter(coord_ch, m)
        self.part_map_prior_adapter = FeatureAdapter(num_foreground_parts, m)
        self.fuse = RefineFuseBlock(dec_ch + m + m + m, dec_ch)
        self.context = nn.Sequential(
            DepthwiseSeparableConv(dec_ch, dec_ch, k=3),
            ResidualDSConvBlock(dec_ch),
        )
        self.head = LogitHead(dec_ch, head_ch, self.num_foreground_parts)

    def forward(
        self,
        common_feature: torch.Tensor,
        fish_region: torch.Tensor,
        coord_maps: torch.Tensor,
        part_map_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                common_feature,
                self.fish_region_adapter(fish_region),
                self.coord_adapter(coord_maps),
                self.part_map_prior_adapter(part_map_prior),
            ],
            dim=1,
        )
        feat = self.context(self.fuse(x))
        raw_logits = self.head(feat)
        part_map_prior_logits = safe_log_prob(part_map_prior.clamp_min(1e-6), eps=1e-6)
        logits = raw_logits + self.part_map_prior_scale * part_map_prior_logits
        return {
            "part_map_feature_low": feat,
            "foreground_part_raw_logits_low": raw_logits,
            "foreground_part_logits_low": logits,
            "part_map_prior_logits_low": part_map_prior_logits,
        }


class ZoneMapEstimator(nn.Module):
    ZONE_NAMES = CANONICAL_ZONE_NAMES

    def __init__(
        self, fin_idx: int, caudal_idx: int, mouth_idx: int, body_idx: int, contact_kernel: int = 11
    ):
        super().__init__()
        self.fin_idx = int(fin_idx)
        self.caudal_idx = int(caudal_idx)
        self.mouth_idx = int(mouth_idx)
        self.body_idx = int(body_idx)
        self.contact_kernel = int(contact_kernel)

    def _avg_pool(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def forward(
        self, part_map: torch.Tensor, coord: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        body_part = part_map[:, self.body_idx : self.body_idx + 1]
        fin = part_map[:, self.fin_idx : self.fin_idx + 1]
        caudal = part_map[:, self.caudal_idx : self.caudal_idx + 1]
        mouth = part_map[:, self.mouth_idx : self.mouth_idx + 1]
        fish_boundary = coord["fish_boundary"]
        fish_contact = coord["fish_contact"]
        fish_interior = coord["fish_interior"]
        head_prior = coord["head_prior"]
        tail_prior = coord["tail_prior"]
        midbody_prior = coord["midbody_prior"]
        lateral_prior = coord["lateral_prior"]

        body_near = self._avg_pool(body_part, self.contact_kernel)
        fin_contact = (fin * body_near * midbody_prior * (0.50 + 0.50 * lateral_prior)).clamp(
            0.0, 1.0
        )
        caudal_contact = (caudal * body_near * tail_prior).clamp(0.0, 1.0)

        fin_tip = (fin * fish_boundary * midbody_prior * (0.45 + 0.55 * lateral_prior)).clamp(
            0.0, 1.0
        )
        caudal_fin_tip = (caudal * fish_boundary * tail_prior).clamp(0.0, 1.0)
        fin_base = torch.maximum(
            fin_contact, 0.35 * fin * self._avg_pool(body_part, 7) * midbody_prior
        ).clamp(0.0, 1.0)
        caudal_fin_base = torch.maximum(
            caudal_contact, 0.35 * caudal * self._avg_pool(body_part, 7) * tail_prior
        ).clamp(0.0, 1.0)
        fin_middle = (fin * midbody_prior * (1.0 - torch.maximum(fin_tip, fin_base))).clamp(
            0.0, 1.0
        )
        caudal_fin_middle = (
            caudal * tail_prior * (1.0 - torch.maximum(caudal_fin_tip, caudal_fin_base))
        ).clamp(0.0, 1.0)
        mouth_zone = (mouth * head_prior * (0.50 + 0.50 * fish_contact)).clamp(0.0, 1.0)
        body_zone = (
            body_part
            * fish_interior
            * (1.0 - 0.55 * fin_base)
            * (1.0 - 0.55 * caudal_fin_base)
            * (1.0 - 0.40 * mouth_zone)
        ).clamp(0.0, 1.0)
        zone_stack = torch.cat(
            [
                body_zone,
                mouth_zone,
                fin_tip,
                fin_middle,
                fin_base,
                caudal_fin_tip,
                caudal_fin_middle,
                caudal_fin_base,
            ],
            dim=1,
        )
        return {
            "zone_map_low": zone_stack,
            "zone_names": self.ZONE_NAMES,
            "body_zone_low": body_zone,
            "mouth_zone_low": mouth_zone,
            "fin_tip_zone_low": fin_tip,
            "fin_middle_zone_low": fin_middle,
            "fin_base_zone_low": fin_base,
            "caudal_fin_tip_zone_low": caudal_fin_tip,
            "caudal_fin_middle_zone_low": caudal_fin_middle,
            "caudal_fin_base_zone_low": caudal_fin_base,
            "head_prior_low": head_prior,
            "tail_prior_low": tail_prior,
            "fin_contact_low": fin_contact,
            "caudal_contact_low": caudal_contact,
        }
