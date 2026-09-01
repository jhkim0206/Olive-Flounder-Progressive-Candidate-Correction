"""Formation of the common feature, Part-structure information, and Visual evidence."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import PART_MAP_CLASS_NAMES
from .common import (
    DepthwiseSeparableConv,
    ResidualDSConvBlock,
    safe_log_prob,
)
from .part_structure_information import (
    FishRegionEstimator,
    HeadEndpointRoleHead,
    HeadToTailDirectionEstimator,
    PartMapEstimator,
    PartMapPriorBuilder,
    ZoneMapEstimator,
)
from .shared_encoder_decoder import SharedDecoder, SharedEncoder
from .visual_evidence import (
    LesionEvidenceEstimator,
    RednessEvidenceEstimator,
    ShapeEvidenceEstimator,
    UnaffectedSurfaceEvidenceEstimator,
)


class FeatureFormation(nn.Module):
    """Derive F, S, and E from an RGB image."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        input_channels: int,
        output_indices: tuple[int, ...],
        feature_channels: int,
        head_channels: int,
        part_names: Sequence[str] = PART_MAP_CLASS_NAMES,
        fish_region_prior_channels: int = 6,
        endpoint_hidden_channels: int = 64,
        direction_epsilon: float = 1e-6,
        part_coordinate_channels: int = 12,
        part_prior_scale: float = 0.90,
        zone_contact_kernel: int = 11,
        redness_local_kernel: int = 31,
        rgb_input_is_imagenet_normalized: bool = True,
    ):
        super().__init__()
        self.part_names = tuple(str(name) for name in part_names)
        if self.part_names != PART_MAP_CLASS_NAMES:
            raise ValueError("part_names must be background, body, fin, caudal_fin, and mouth")
        self.feature_channels = int(feature_channels)
        self.part_structure_information_channels = 1 + len(self.part_names) + 1 + 8

        self.encoder = SharedEncoder(
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=output_indices,
            in_chans=input_channels,
        )
        self.decoder = SharedDecoder(self.encoder.out_channels, out_ch=feature_channels)
        self.common_refinement = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, feature_channels, k=3),
            ResidualDSConvBlock(feature_channels),
        )

        self.fish_region_estimator = FishRegionEstimator(
            dec_ch=feature_channels,
            head_ch=head_channels,
            prior_ch=fish_region_prior_channels,
        )
        self.head_to_tail_direction_estimator = HeadToTailDirectionEstimator(eps=direction_epsilon)
        self.head_endpoint_role_head = HeadEndpointRoleHead(
            dec_ch=feature_channels, hidden_ch=endpoint_hidden_channels
        )
        self.part_map_prior_builder = PartMapPriorBuilder()
        self.part_map_estimator = PartMapEstimator(
            dec_ch=feature_channels,
            head_ch=head_channels,
            num_foreground_parts=4,
            coord_ch=part_coordinate_channels,
            part_map_prior_scale=part_prior_scale,
        )
        self.zone_map_estimator = ZoneMapEstimator(
            body_idx=1,
            fin_idx=2,
            caudal_idx=3,
            mouth_idx=4,
            contact_kernel=zone_contact_kernel,
        )

        self.redness_evidence_estimator = RednessEvidenceEstimator(
            dec_ch=feature_channels,
            num_parts=len(self.part_names),
            head_ch=head_channels,
            local_kernel=redness_local_kernel,
            input_is_imagenet_normalized=rgb_input_is_imagenet_normalized,
        )
        self.shape_evidence_estimator = ShapeEvidenceEstimator(
            dec_ch=feature_channels, head_ch=head_channels
        )
        self.lesion_evidence_estimator = LesionEvidenceEstimator(
            dec_ch=feature_channels,
            num_parts=len(self.part_names),
            head_ch=head_channels,
        )
        self.unaffected_surface_evidence_estimator = UnaffectedSurfaceEvidenceEstimator(
            dec_ch=feature_channels,
            num_parts=len(self.part_names),
            zone_ch=8,
            head_ch=head_channels,
        )
        self.reference_fish_region_estimator = copy.deepcopy(self.fish_region_estimator)
        self.reference_head_endpoint_role_head = copy.deepcopy(self.head_endpoint_role_head)
        self.reference_part_map_estimator = copy.deepcopy(self.part_map_estimator)
        self.reference_mode = "self_detach"
        self.reference_state_ready = False
        self.sync_part_structure_reference()

    @staticmethod
    def _coordinate_tensor(coordinates: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                coordinates["x_norm"],
                coordinates["y_norm"],
                coordinates["signed_s"],
                coordinates["abs_s"],
                coordinates["t_lateral"],
                coordinates["fish_boundary"],
                coordinates["fish_contact"],
                coordinates["head_prior"],
                coordinates["tail_prior"],
                coordinates["midbody_prior"],
                coordinates["lateral_prior"],
                coordinates["endpoint_role_map"],
            ],
            dim=1,
        )

    @staticmethod
    def _part_distribution(
        fish_region: torch.Tensor, foreground_part_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        foreground_part = F.softmax(foreground_part_logits, dim=1)
        part_map = torch.cat(
            [(1.0 - fish_region).clamp_min(0.0), fish_region * foreground_part], dim=1
        ).clamp_min(1e-6)
        part_map = part_map / part_map.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return safe_log_prob(part_map), part_map

    @staticmethod
    def _upsample(value: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _freeze_reference_modules(self) -> None:
        self._freeze(self.reference_fish_region_estimator)
        self._freeze(self.reference_head_endpoint_role_head)
        self._freeze(self.reference_part_map_estimator)

    @torch.no_grad()
    def sync_part_structure_reference(self) -> None:
        self.reference_fish_region_estimator.load_state_dict(
            self.fish_region_estimator.state_dict(), strict=True
        )
        self.reference_head_endpoint_role_head.load_state_dict(
            self.head_endpoint_role_head.state_dict(), strict=True
        )
        self.reference_part_map_estimator.load_state_dict(
            self.part_map_estimator.state_dict(), strict=True
        )
        self._freeze_reference_modules()
        self.reference_state_ready = True

    def set_training_stage(self, stage_id: int) -> None:
        self.reference_mode = "self_detach" if int(stage_id) <= 1 else "frozen_copy"
        self._freeze_reference_modules()

    def _form_part_structure(
        self,
        common_feature: torch.Tensor,
        image: torch.Tensor,
        fish_region_estimator: FishRegionEstimator,
        endpoint_role_head: HeadEndpointRoleHead,
        part_map_estimator: PartMapEstimator,
        detach_fish_for_part: bool,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        fish_output = fish_region_estimator(common_feature, image)
        fish_region = fish_output["refined_fish_region_low"].clamp(0.0, 1.0)
        base_coordinates = self.head_to_tail_direction_estimator(fish_region)
        endpoint_output = endpoint_role_head(
            common_feature=common_feature,
            fish_region_feature=fish_output["fish_region_feature_low"],
            refined_fish_region=fish_region,
            base_coord=base_coordinates,
        )
        coordinates = self.head_to_tail_direction_estimator(
            fish_region, endpoint_output["endpoint_role_prob_low"]
        )
        part_prior = self.part_map_prior_builder(fish_region, coordinates)
        part_output = part_map_estimator(
            common_feature=common_feature,
            fish_region=fish_region.detach() if detach_fish_for_part else fish_region,
            coord_maps=self._coordinate_tensor(coordinates),
            part_map_prior=part_prior["part_map_prior_low"],
        )
        part_map_logits, part_map = self._part_distribution(
            fish_region, part_output["foreground_part_logits_low"]
        )
        zone_output = self.zone_map_estimator(part_map, coordinates)
        head_to_tail_direction = (
            0.5 * (coordinates["signed_s"] + 1.0) * fish_region
        ).clamp(0.0, 1.0)
        return {
            "fish_output": fish_output,
            "fish_region": fish_region,
            "endpoint_output": endpoint_output,
            "coordinates": coordinates,
            "part_output": part_output,
            "part_map_logits": part_map_logits,
            "part_map": part_map,
            "zone_output": zone_output,
            "head_to_tail_direction": head_to_tail_direction,
        }

    def _reference_part_structure(
        self,
        common_feature: torch.Tensor,
        image: torch.Tensor,
        online: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.reference_mode == "self_detach" or not self.reference_state_ready:
            detached: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
            for key, value in online.items():
                if torch.is_tensor(value):
                    detached[key] = value.detach()
                elif isinstance(value, dict):
                    detached[key] = {
                        inner_key: inner_value.detach() if torch.is_tensor(inner_value) else inner_value
                        for inner_key, inner_value in value.items()
                    }
            return detached
        with torch.no_grad():
            return self._form_part_structure(
                common_feature.detach(),
                image,
                self.reference_fish_region_estimator,
                self.reference_head_endpoint_role_head,
                self.reference_part_map_estimator,
                True,
            )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(image)
        common_feature, pyramid = self.decoder(encoded)
        common_feature = self.common_refinement(common_feature)
        fine_grained_detail_feature = pyramid["p1"]

        online = self._form_part_structure(
            common_feature,
            image,
            self.fish_region_estimator,
            self.head_endpoint_role_head,
            self.part_map_estimator,
            True,
        )
        reference = self._reference_part_structure(common_feature, image, online)
        fish_output = online["fish_output"]
        fish_region_online = online["fish_region"]
        part_map_logits = online["part_map_logits"]
        part_map_online = online["part_map"]
        fish_region = reference["fish_region"]
        part_map = reference["part_map"]
        coordinates = reference["coordinates"]
        zone_output = reference["zone_output"]
        zone_map = zone_output["zone_map_low"].clamp(0.0, 1.0)
        head_to_tail_direction = reference["head_to_tail_direction"]

        redness_output = self.redness_evidence_estimator(
            image=image,
            common_feature=common_feature,
            fish_region_prob_low=fish_region,
            part_map_low=part_map,
        )
        redness_logits = redness_output["redness_evidence_logits_low"]
        shape_output = self.shape_evidence_estimator(
            common_feature=common_feature,
            fine_grained_detail_feature=fine_grained_detail_feature,
            fish_boundary_low=coordinates["fish_boundary"],
            fin_tip_zone_low=zone_output["fin_tip_zone_low"],
            fin_base_zone_low=zone_output["fin_base_zone_low"],
            caudal_fin_tip_zone_low=zone_output["caudal_fin_tip_zone_low"],
            caudal_fin_base_zone_low=zone_output["caudal_fin_base_zone_low"],
        )
        shape_logits = shape_output["shape_evidence_logits_low"]
        lesion_output = self.lesion_evidence_estimator(
            common_feature=common_feature,
            fine_grained_detail_feature=fine_grained_detail_feature,
            fish_interior_low=coordinates["fish_interior"],
            part_map_low=part_map,
        )
        lesion_logits = lesion_output["lesion_evidence_logits_low"]
        positive_visual_evidence = torch.sigmoid(
            torch.cat([redness_logits, shape_logits, lesion_logits], dim=1)
        )
        unaffected_output = self.unaffected_surface_evidence_estimator(
            common_feature=common_feature,
            fine_grained_detail_feature=fine_grained_detail_feature,
            fish_region_prob_low=fish_region,
            part_map_low=part_map,
            zone_map_low=zone_map,
            positive_visual_evidence_prob_low=positive_visual_evidence,
        )
        unaffected_surface_logits = unaffected_output["unaffected_surface_evidence_logits_low"]
        visual_evidence = torch.cat(
            [positive_visual_evidence, torch.sigmoid(unaffected_surface_logits)], dim=1
        )
        part_structure_information = torch.cat(
            [fish_region, part_map, head_to_tail_direction, zone_map], dim=1
        )

        zone_background = (1.0 - fish_region).clamp_min(1e-6)
        zone_distribution = torch.cat([zone_background, zone_map], dim=1).clamp_min(1e-6)
        zone_distribution = zone_distribution / zone_distribution.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)

        low_resolution = {
            "common_feature_low": common_feature,
            "fine_grained_detail_feature_low": fine_grained_detail_feature,
            "fish_region_logits_low": fish_output["fish_region_logits_low"],
            "fish_region_low": fish_region,
            "part_map_logits_low": part_map_logits,
            "part_map_low": part_map,
            "head_to_tail_direction_low": head_to_tail_direction,
            "zone_map_logits_low": safe_log_prob(zone_distribution),
            "zone_map_low": zone_map,
            "part_structure_information_low": part_structure_information,
            "redness_evidence_logits_low": redness_logits,
            "shape_evidence_logits_low": shape_logits,
            "lesion_evidence_logits_low": lesion_logits,
            "unaffected_surface_evidence_logits_low": unaffected_surface_logits,
            "visual_evidence_low": visual_evidence,
            "fish_boundary_low": coordinates["fish_boundary"],
            "online_fish_region_logits_low": fish_output["fish_region_logits_low"],
            "online_fish_region_low": fish_region_online,
            "online_part_map_logits_low": part_map_logits,
            "online_part_map_low": part_map_online,
        }
        full_resolution = {
            key.removesuffix("_low"): self._upsample(value, image)
            for key, value in low_resolution.items()
            if key.endswith("_low")
            and key not in {"common_feature_low", "fine_grained_detail_feature_low"}
        }
        return {
            **low_resolution,
            **full_resolution,
            "F": low_resolution["common_feature_low"],
            "S": low_resolution["part_structure_information_low"],
            "E": low_resolution["visual_evidence_low"],
        }
