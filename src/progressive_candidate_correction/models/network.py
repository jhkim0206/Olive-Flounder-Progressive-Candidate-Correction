"""Progressive Candidate Correction and Route-wise composition."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import PART_MAP_CLASS_NAMES, SYMPTOM_CLASS_NAMES
from ..training_stages import resolve_training_stage_id, resolve_training_stage_name
from .candidate_correction import (
    CandidateCorrectionVariant,
    CandidateEvidenceIntegrator,
    ConcatCandidateCorrector,
    GatedResidualCorrector,
    InitialSymptomCandidateGenerator,
    LatentVisualFeatureEncoder,
    PartConditionedAbnormalityGate,
    SignedCandidateCorrector,
    SymptomPrototypeBank,
)
from .common import (
    safe_binary_prob_tensor,
    safe_log_prob,
    safe_multiclass_prob_from_logits,
    sanitize_logits_tensor,
)
from .feature_formation import FeatureFormation
from .final_refinement import FinalRefinement
from .route_wise_composition import (
    RouteAssignmentHead,
    RouteInterpretationHead,
    RouteWiseComposer,
    SpatialSupportHead,
)


class ProgressiveCandidateCorrectionNetwork(nn.Module):
    """Network for visible-symptom recognition in olive flounder."""

    def __init__(
        self,
        num_classes: int = 9,
        backbone_name: str = "repvit_m1_1.dist_450e_in1k",
        pretrained: bool = True,
        in_ch: int = 3,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        dec_ch: int = 96,
        head_ch: int = 64,
        part_names: Sequence[str] = PART_MAP_CLASS_NAMES,
        semantic_class_names: Sequence[str] | None = None,
        candidate_correction_variant: CandidateCorrectionVariant
        | str = CandidateCorrectionVariant.FULL,
        use_progressive_candidate_correction: bool = True,
        use_route_wise_composition: bool = True,
        use_final_refinement: bool = True,
        visual_evidence_detach_stages: Sequence[int] = (3, 4, 5, 6),
        lesion_unaffected_surface_suppression: float = 0.75,
        latent_visual_channels: int = 64,
        symptom_prototype_logit_scale: float = 2.5,
        candidate_prototype_weight: float = 0.50,
        candidate_part_weight: float = 1.10,
        candidate_unaffected_surface_weight: float = 0.60,
        candidate_cue_weight: float = 0.25,
        final_refinement_strength: float = 0.35,
        route_assignment_prior_weight: float = 1.25,
        spatial_support_prior_weight: float = 1.10,
        spatial_support_attach_kernel: int = 13,
        spatial_support_attach_body_weight: float = 0.65,
        route_interpretation_prior_weight: float = 0.85,
        route_strength: float = 0.90,
        route_candidate_retention_weight: float = 0.12,
        route_overwrite_strength: float = 0.90,
        route_support_floor: float = 0.02,
        final_fish_region_support_kernel: int = 7,
        final_fish_region_support_threshold: float = 0.25,
        final_fish_region_support_temperature: float = 0.08,
        final_class_temperature: float = 1.0,
        final_discovery_weight: float = 0.10,
        final_visual_evidence_weight: float = 0.0,
        final_foreground_threshold: float = 0.28,
        class_seed_reduction: str = "max",
        class_seed_top_two_weight: float = 0.20,
        unaffected_surface_veto_strength: float = 0.25,
        unaffected_surface_veto_threshold: float = 0.65,
        unaffected_surface_veto_temperature: float = 0.12,
        unaffected_surface_protection_threshold: float = 0.20,
        unaffected_surface_protection_temperature: float = 0.08,
        detach_unaffected_surface_protection: bool = True,
    ):
        super().__init__()
        if int(num_classes) != 9:
            raise ValueError("the model uses background plus eight symptom classes")
        self.num_classes = int(num_classes)
        self.num_symptom_classes = self.num_classes - 1
        self.semantic_class_names = tuple(semantic_class_names or SYMPTOM_CLASS_NAMES)
        if self.semantic_class_names != SYMPTOM_CLASS_NAMES:
            raise ValueError("semantic_class_names must match the method class order")
        self.part_names = tuple(part_names)
        if self.part_names != PART_MAP_CLASS_NAMES:
            raise ValueError("part_names must match the method Part Map order")
        self.correction_variant = CandidateCorrectionVariant.parse(candidate_correction_variant)
        self.use_progressive_candidate_correction = bool(use_progressive_candidate_correction)
        self.use_route_wise_composition = bool(use_route_wise_composition)
        self.use_final_refinement = bool(use_final_refinement)
        self.visual_evidence_detach_stages = frozenset(
            int(stage_id) for stage_id in visual_evidence_detach_stages
        )
        self.current_stage_id = 6
        self.current_stage_name = "joint_fine_tuning"
        self.lesion_unaffected_surface_suppression = float(
            lesion_unaffected_surface_suppression
        )
        self.final_fish_region_support_kernel = int(final_fish_region_support_kernel)
        self.final_fish_region_support_threshold = float(final_fish_region_support_threshold)
        self.final_fish_region_support_temperature = float(final_fish_region_support_temperature)
        self.final_class_temperature = float(final_class_temperature)
        self.final_discovery_weight = float(final_discovery_weight)
        self.final_visual_evidence_weight = float(final_visual_evidence_weight)
        self.final_foreground_threshold = float(final_foreground_threshold)
        self.class_seed_reduction = str(class_seed_reduction).lower()
        self.class_seed_top_two_weight = float(class_seed_top_two_weight)
        self.unaffected_surface_veto_strength = float(unaffected_surface_veto_strength)
        self.unaffected_surface_veto_threshold = float(unaffected_surface_veto_threshold)
        self.unaffected_surface_veto_temperature = float(unaffected_surface_veto_temperature)
        self.unaffected_surface_protection_threshold = float(
            unaffected_surface_protection_threshold
        )
        self.unaffected_surface_protection_temperature = float(
            unaffected_surface_protection_temperature
        )
        self.detach_unaffected_surface_protection = bool(
            detach_unaffected_surface_protection
        )

        self.feature_formation = FeatureFormation(
            backbone_name=backbone_name,
            pretrained=pretrained,
            input_channels=in_ch,
            output_indices=out_indices,
            feature_channels=dec_ch,
            head_channels=head_ch,
            part_names=part_names,
        )
        part_structure_information_channels = (
            self.feature_formation.part_structure_information_channels
        )
        self.initial_symptom_candidate_generator = InitialSymptomCandidateGenerator(
            feature_channels=dec_ch,
            num_parts=len(part_names),
            num_classes=self.num_symptom_classes,
            zone_channels=8,
            head_channels=head_ch,
        )
        self.signed_candidate_corrector = SignedCandidateCorrector(self.semantic_class_names)
        self.gated_residual_corrector = GatedResidualCorrector(
            feature_channels=dec_ch,
            num_classes=self.num_symptom_classes,
            zone_channels=8,
            head_channels=head_ch,
        )
        self.concat_candidate_corrector = ConcatCandidateCorrector(
            feature_channels=dec_ch,
            part_structure_information_channels=part_structure_information_channels,
            num_classes=self.num_symptom_classes,
            head_channels=head_ch,
        )
        self.part_conditioned_abnormality_gate = PartConditionedAbnormalityGate(
            feature_channels=dec_ch,
            num_parts=len(part_names),
            zone_channels=8,
            head_channels=head_ch,
        )
        self.latent_visual_feature_encoder = LatentVisualFeatureEncoder(
            feature_channels=dec_ch,
            num_parts=len(part_names),
            zone_channels=8,
            embedding_channels=latent_visual_channels,
        )
        self.symptom_prototype_bank = SymptomPrototypeBank(
            num_classes=self.num_symptom_classes,
            embedding_channels=latent_visual_channels,
            logit_scale=symptom_prototype_logit_scale,
        )
        self.candidate_evidence_integrator = CandidateEvidenceIntegrator(
            semantic_class_names=self.semantic_class_names,
            initial_prototype_weight=candidate_prototype_weight,
            initial_part_weight=candidate_part_weight,
            initial_unaffected_weight=candidate_unaffected_surface_weight,
            initial_cue_weight=candidate_cue_weight,
        )

        self.route_assignment_head = RouteAssignmentHead(
            dec_ch=dec_ch,
            num_parts=len(part_names),
            zone_ch=8,
            head_ch=head_ch,
            prior_weight=route_assignment_prior_weight,
        )
        self.spatial_support_head = SpatialSupportHead(
            dec_ch=dec_ch,
            num_routes=4,
            zone_ch=8,
            num_classes=self.num_symptom_classes,
            semantic_class_names=self.semantic_class_names,
            head_ch=head_ch,
            prior_weight=spatial_support_prior_weight,
            attach_kernel=spatial_support_attach_kernel,
            attach_body_weight=spatial_support_attach_body_weight,
        )
        self.body_route_interpretation_head = RouteInterpretationHead(
            dec_ch=dec_ch,
            context_ch=7,
            out_ch=2,
            head_ch=head_ch,
            prior_weight=route_interpretation_prior_weight,
        )
        self.fin_route_interpretation_head = RouteInterpretationHead(
            dec_ch=dec_ch,
            context_ch=11,
            out_ch=4,
            head_ch=head_ch,
            prior_weight=route_interpretation_prior_weight,
        )
        self.caudal_fin_route_interpretation_head = RouteInterpretationHead(
            dec_ch=dec_ch,
            context_ch=11,
            out_ch=4,
            head_ch=head_ch,
            prior_weight=route_interpretation_prior_weight,
        )
        self.mouth_route_interpretation_head = RouteInterpretationHead(
            dec_ch=dec_ch,
            context_ch=7,
            out_ch=2,
            head_ch=head_ch,
            prior_weight=route_interpretation_prior_weight,
        )
        self.route_wise_composer = RouteWiseComposer(
            semantic_class_names=self.semantic_class_names,
            route_strength=route_strength,
            candidate_retention_weight=route_candidate_retention_weight,
            overwrite_strength=route_overwrite_strength,
            support_floor=route_support_floor,
        )
        self.final_refinement = FinalRefinement(
            dec_ch=dec_ch,
            num_classes=self.num_symptom_classes,
            zone_ch=8,
            head_ch=head_ch,
            strength=final_refinement_strength,
        )
        self.feature_formation.set_training_stage(self.current_stage_id)

    def set_training_stage(self, stage: str | int) -> ProgressiveCandidateCorrectionNetwork:
        stage_id = resolve_training_stage_id(stage)
        self.current_stage_id = stage_id
        self.current_stage_name = resolve_training_stage_name(stage_id)
        self.feature_formation.set_training_stage(stage_id)
        return self

    @torch.no_grad()
    def sync_part_structure_reference(self) -> None:
        self.feature_formation.sync_part_structure_reference()

    def _stage_conditioned_information(
        self, formed: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        part_structure_information = formed["part_structure_information_low"]
        visual_evidence = formed["visual_evidence_low"]
        if self.current_stage_id in self.visual_evidence_detach_stages:
            visual_evidence = visual_evidence.detach()
        return part_structure_information, visual_evidence

    def _effective_visual_evidence(self, visual_evidence: torch.Tensor) -> torch.Tensor:
        lesion = visual_evidence[:, 2:3]
        unaffected_surface = visual_evidence[:, 3:4]
        lesion = lesion * (
            1.0
            - self.lesion_unaffected_surface_suppression
            * unaffected_surface.detach()
        ).clamp(0.05, 1.0)
        return torch.cat(
            [visual_evidence[:, 0:2], lesion.clamp(0.0, 1.0), unaffected_surface], dim=1
        )

    def _part_probability_per_class(
        self, part_conditioned_probability: torch.Tensor
    ) -> torch.Tensor:
        selector = part_conditioned_probability.new_zeros(
            self.num_symptom_classes, 4
        )
        for class_index, raw_name in enumerate(self.semantic_class_names):
            name = str(raw_name).lower()
            if "fin" in name:
                selector[class_index, 1] = 1.0
            elif "caudal" in name or "tail" in name:
                selector[class_index, 2] = 1.0
            elif "mouth" in name or "ulcer" in name:
                selector[class_index, 3] = 1.0
            else:
                selector[class_index, 0] = 1.0
        return torch.einsum(
            "cf,bfhw->bchw", selector, part_conditioned_probability
        ).clamp(0.0, 1.0)

    def _apply_correction(
        self,
        formed: dict[str, torch.Tensor],
        initial_symptom_candidate_response: torch.Tensor,
        part_structure_information: torch.Tensor,
        visual_evidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros_like(initial_symptom_candidate_response)
        if not self.use_progressive_candidate_correction:
            return {
                "signed_corrected_candidate_response_low": initial_symptom_candidate_response,
                "gated_residual_correction_term_low": zero,
                "corrected_symptom_candidate_response_low": initial_symptom_candidate_response,
            }
        if self.correction_variant is CandidateCorrectionVariant.CONCAT:
            corrected = self.concat_candidate_corrector(
                common_feature=formed["common_feature_low"],
                part_structure_information=part_structure_information,
                visual_evidence=visual_evidence,
                initial_symptom_candidate_response=initial_symptom_candidate_response,
            )
            return {
                "signed_corrected_candidate_response_low": initial_symptom_candidate_response,
                "gated_residual_correction_term_low": (
                    corrected - initial_symptom_candidate_response
                ),
                "corrected_symptom_candidate_response_low": corrected,
            }

        signed = self.signed_candidate_corrector(
            initial_symptom_candidate_response=initial_symptom_candidate_response,
            zone_map=formed["zone_map_low"],
            redness_prob=visual_evidence[:, 0:1],
            shape_prob=visual_evidence[:, 1:2],
            lesion_prob=visual_evidence[:, 2:3],
        )
        signed_corrected_candidate_response = signed["signed_corrected_candidate_response_low"]
        if self.correction_variant is CandidateCorrectionVariant.SIGNED:
            return {
                **signed,
                "gated_residual_correction_term_low": zero,
                "corrected_symptom_candidate_response_low": signed_corrected_candidate_response,
            }
        residual = self.gated_residual_corrector(
            common_feature=formed["common_feature_low"],
            fine_grained_detail_feature=formed["fine_grained_detail_feature_low"],
            zone_map=formed["zone_map_low"],
            initial_symptom_candidate_response=initial_symptom_candidate_response,
            signed_corrected_candidate_response=signed_corrected_candidate_response,
            redness_prob=visual_evidence[:, 0:1],
            shape_prob=visual_evidence[:, 1:2],
            lesion_prob=visual_evidence[:, 2:3],
        )
        gated_response = residual["gated_corrected_candidate_response_low"]
        part_conditioned = self.part_conditioned_abnormality_gate(
            common_feature=formed["common_feature_low"],
            fine_grained_detail_feature=formed["fine_grained_detail_feature_low"],
            part_map=formed["part_map_low"],
            zone_map=formed["zone_map_low"],
            redness_prob=visual_evidence[:, 0:1],
            shape_prob=visual_evidence[:, 1:2],
            lesion_prob=visual_evidence[:, 2:3],
            unaffected_surface_prob=visual_evidence[:, 3:4],
        )
        latent_visual = self.latent_visual_feature_encoder(
            common_feature=formed["common_feature_low"],
            fine_grained_detail_feature=formed["fine_grained_detail_feature_low"],
            part_map=formed["part_map_low"],
            zone_map=formed["zone_map_low"],
            visual_evidence=visual_evidence,
            part_conditioned_probability=part_conditioned[
                "part_conditioned_probability_low"
            ],
        )
        prototypes = self.symptom_prototype_bank(
            latent_visual["latent_visual_embedding_low"]
        )
        per_class_part_probability = self._part_probability_per_class(
            part_conditioned["part_conditioned_probability_low"]
        )
        integrated = self.candidate_evidence_integrator(
            initial_response=initial_symptom_candidate_response,
            signed_response=signed_corrected_candidate_response,
            gated_response=gated_response,
            prototype_similarity=prototypes["symptom_prototype_similarity_low"],
            class_zone_map=formed["zone_map_low"],
            part_conditioned_probability=per_class_part_probability,
            visual_evidence=visual_evidence,
        )
        corrected_response = integrated["corrected_symptom_candidate_response_low"]
        return {
            **signed,
            **residual,
            **part_conditioned,
            **latent_visual,
            **prototypes,
            **integrated,
            "part_conditioned_probability_per_class_low": per_class_part_probability,
            "gated_residual_correction_term_low": (
                corrected_response - signed_corrected_candidate_response
            ),
        }

    def _class_probability(self, logits: torch.Tensor, name: str) -> torch.Tensor:
        index = self.semantic_class_names.index(name)
        return torch.sigmoid(logits[:, index : index + 1])

    @staticmethod
    def _route_response_prior(
        none_probability: torch.Tensor, class_probabilities: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        probabilities = torch.cat([none_probability, *class_probabilities], dim=1).clamp_min(1e-6)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return safe_log_prob(probabilities)

    def _route_responses(
        self,
        formed: dict[str, torch.Tensor],
        corrected_symptom_candidate_response: torch.Tensor,
        route_assignment_with_background: torch.Tensor,
        spatial_support_internal: torch.Tensor,
        visual_evidence: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        common = formed["common_feature_low"]
        detail = formed["fine_grained_detail_feature_low"]
        body_assignment = route_assignment_with_background[:, 1:2]
        fin_assignment = route_assignment_with_background[:, 2:3]
        caudal_assignment = route_assignment_with_background[:, 3:4]
        mouth_assignment = route_assignment_with_background[:, 4:5]
        body_support = spatial_support_internal[:, 0:1]
        fin_support = spatial_support_internal[:, 1:4]
        caudal_support = spatial_support_internal[:, 4:7]
        mouth_support = spatial_support_internal[:, 7:8]

        body_probability = self._class_probability(
            corrected_symptom_candidate_response, "body_lesion"
        )
        body_prior = self._route_response_prior(
            (1.0 - body_assignment * body_support).clamp(0.0, 1.0),
            [body_probability],
        )
        body_context = torch.cat(
            [body_assignment, body_support, visual_evidence, body_probability], dim=1
        )

        fin_probabilities = [
            self._class_probability(corrected_symptom_candidate_response, "fin_deformity"),
            self._class_probability(corrected_symptom_candidate_response, "fin_necrosis"),
            self._class_probability(corrected_symptom_candidate_response, "fin_base_necrosis"),
        ]
        fin_union = fin_assignment * fin_support.amax(dim=1, keepdim=True)
        fin_prior = self._route_response_prior((1.0 - fin_union).clamp(0.0, 1.0), fin_probabilities)
        fin_context = torch.cat(
            [fin_assignment, fin_support, visual_evidence, *fin_probabilities], dim=1
        )

        caudal_probabilities = [
            self._class_probability(corrected_symptom_candidate_response, "caudal_deformity"),
            self._class_probability(corrected_symptom_candidate_response, "caudal_necrosis"),
            self._class_probability(corrected_symptom_candidate_response, "caudal_base_necrosis"),
        ]
        caudal_union = caudal_assignment * caudal_support.amax(dim=1, keepdim=True)
        caudal_prior = self._route_response_prior(
            (1.0 - caudal_union).clamp(0.0, 1.0), caudal_probabilities
        )
        caudal_context = torch.cat(
            [caudal_assignment, caudal_support, visual_evidence, *caudal_probabilities], dim=1
        )

        mouth_probability = self._class_probability(
            corrected_symptom_candidate_response, "mouth_ulcer"
        )
        mouth_prior = self._route_response_prior(
            (1.0 - mouth_assignment * mouth_support).clamp(0.0, 1.0),
            [mouth_probability],
        )
        mouth_context = torch.cat(
            [mouth_assignment, mouth_support, visual_evidence, mouth_probability], dim=1
        )
        return {
            "body": self.body_route_interpretation_head(
                common, detail, body_context, body_prior
            ),
            "mouth": self.mouth_route_interpretation_head(
                common, detail, mouth_context, mouth_prior
            ),
            "fin": self.fin_route_interpretation_head(
                common, detail, fin_context, fin_prior
            ),
            "caudal_fin": self.caudal_fin_route_interpretation_head(
                common, detail, caudal_context, caudal_prior
            ),
        }

    def _soft_fish_region_support(self, fish_region: torch.Tensor) -> torch.Tensor:
        kernel = max(1, self.final_fish_region_support_kernel)
        if kernel % 2 == 0:
            kernel += 1
        dilated = (
            F.max_pool2d(fish_region.clamp(0.0, 1.0), kernel, 1, kernel // 2)
            if kernel > 1
            else fish_region.clamp(0.0, 1.0)
        )
        support = torch.sigmoid(
            (dilated - self.final_fish_region_support_threshold)
            / max(self.final_fish_region_support_temperature, 1e-4)
        )
        return safe_binary_prob_tensor(support)

    def _reduce_class_seed(
        self, class_seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values, _ = torch.sort(safe_binary_prob_tensor(class_seed), dim=1, descending=True)
        top_one = values[:, 0:1]
        top_two = values[:, 1:2] if values.shape[1] > 1 else torch.zeros_like(top_one)
        margin = (top_one - top_two).clamp(0.0, 1.0)
        if self.class_seed_reduction in {"soft_or", "softor", "or"}:
            mass = 1.0 - torch.prod(1.0 - values, dim=1, keepdim=True)
        elif self.class_seed_reduction in {"top2", "topk", "weighted_top2"}:
            weight = min(max(self.class_seed_top_two_weight, 0.0), 1.0)
            mass = (1.0 - weight) * top_one + weight * top_two
        elif self.class_seed_reduction == "mean_top2":
            mass = 0.5 * top_one + 0.5 * top_two
        else:
            mass = top_one
        return mass.clamp(0.0, 1.0), top_one, top_two, margin

    def _class_seed_for(self, class_seed: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        indices = [self.semantic_class_names.index(name) for name in names]
        return class_seed[:, indices].amax(dim=1, keepdim=True)

    def _compose_semantic_output(
        self,
        fish_region: torch.Tensor,
        routed_symptom_logits: torch.Tensor,
        class_zone_map: torch.Tensor,
        positive_visual_evidence: torch.Tensor,
        unaffected_surface: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        epsilon = 1e-6
        fish_support = self._soft_fish_region_support(fish_region)
        class_zone_map = safe_binary_prob_tensor(class_zone_map)
        class_probability = safe_binary_prob_tensor(torch.sigmoid(routed_symptom_logits))
        class_seed = (class_probability * class_zone_map * fish_support).clamp(0.0, 1.0)
        seed_mass, seed_top_one, seed_top_two, seed_margin = self._reduce_class_seed(class_seed)
        class_score = (
            routed_symptom_logits / max(self.final_class_temperature, 1e-4)
            + safe_log_prob(class_zone_map.clamp_min(epsilon), eps=epsilon)
        )
        class_distribution = safe_multiclass_prob_from_logits(class_score, dim=1)
        scout_raw = (
            torch.sigmoid(routed_symptom_logits.amax(dim=1, keepdim=True))
            * class_zone_map.amax(dim=1, keepdim=True)
            * fish_support
        ).clamp(0.0, 1.0)
        discovery_probability = (scout_raw * (1.0 - seed_mass)).clamp(0.0, 1.0)
        visual_union = (
            1.0
            - torch.prod(1.0 - safe_binary_prob_tensor(positive_visual_evidence[:, :3]), dim=1, keepdim=True)
        ).clamp(0.0, 1.0)
        visual_support = (visual_union * scout_raw * fish_support).clamp(0.0, 1.0)
        foreground_candidate = 1.0 - (
            (1.0 - seed_mass)
            * (1.0 - (self.final_discovery_weight * discovery_probability).clamp(0.0, 1.0))
            * (1.0 - (self.final_visual_evidence_weight * visual_support).clamp(0.0, 1.0))
        )
        foreground_candidate = torch.minimum(
            foreground_candidate.clamp(0.0, 1.0), fish_support
        )
        unaffected_gate = torch.sigmoid(
            (unaffected_surface - self.unaffected_surface_veto_threshold)
            / max(self.unaffected_surface_veto_temperature, 1e-4)
        )
        unaffected_raw = (unaffected_gate * fish_support).clamp(0.0, 1.0)
        protection = torch.sigmoid(
            (seed_mass - self.unaffected_surface_protection_threshold)
            / max(self.unaffected_surface_protection_temperature, 1e-4)
        ).clamp(0.0, 1.0)
        protection_for_veto = (
            protection.detach() if self.detach_unaffected_surface_protection else protection
        )
        unaffected_veto = (unaffected_raw * (1.0 - protection_for_veto)).clamp(0.0, 1.0)
        foreground_keep = foreground_candidate * (
            1.0 - self.unaffected_surface_veto_strength * unaffected_veto
        )
        foreground_keep = safe_binary_prob_tensor(
            torch.minimum(foreground_keep.clamp(0.0, 1.0), fish_support)
        )
        reject_probability = (1.0 - foreground_keep).clamp(0.0, 1.0)
        class_evidence = (foreground_keep * class_distribution).clamp(0.0, 1.0)
        background_logit = safe_log_prob(reject_probability.clamp_min(epsilon), eps=epsilon)
        foreground_logit = safe_log_prob(foreground_keep.clamp_min(epsilon), eps=epsilon)
        class_rank = class_score - class_score.amax(dim=1, keepdim=True)
        threshold = min(max(self.final_foreground_threshold, 1e-4), 1.0 - 1e-4)
        foreground_bias = math.log((1.0 - threshold) / threshold)
        semantic_logits = sanitize_logits_tensor(
            torch.cat(
                [background_logit, foreground_logit + foreground_bias + class_rank], dim=1
            ),
            clip=20.0,
        )
        return {
            "native_semantic_logits_low": semantic_logits,
            "fish_region_support_low": fish_support,
            "class_seed_low": class_seed,
            "class_seed_mass_low": seed_mass,
            "class_seed_top_one_low": seed_top_one,
            "class_seed_top_two_low": seed_top_two,
            "class_seed_margin_low": seed_margin,
            "class_distribution_low": class_distribution,
            "visual_evidence_union_low": visual_union,
            "visual_evidence_support_low": visual_support,
            "foreground_candidate_low": foreground_candidate,
            "unaffected_surface_veto_raw_low": unaffected_raw,
            "unaffected_surface_protection_low": protection,
            "unaffected_surface_veto_low": unaffected_veto,
            "symptom_foreground_probability_low": foreground_keep,
            "class_evidence_low": class_evidence,
            "semantic_mass_low": foreground_keep,
            "reject_probability_low": reject_probability,
            "discovery_probability_low": discovery_probability,
            "body_lesion_seed_low": self._class_seed_for(class_seed, ("body_lesion",)),
            "mouth_ulcer_seed_low": self._class_seed_for(class_seed, ("mouth_ulcer",)),
            "fin_symptom_seed_low": self._class_seed_for(
                class_seed,
                ("fin_deformity", "fin_necrosis", "fin_base_necrosis"),
            ),
            "caudal_fin_symptom_seed_low": self._class_seed_for(
                class_seed,
                ("caudal_deformity", "caudal_necrosis", "caudal_base_necrosis"),
            ),
        }

    @staticmethod
    def _upsample(value: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, image.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor, return_aux: bool = False) -> dict[str, torch.Tensor]:
        formed = self.feature_formation(image)
        part_structure_information, visual_evidence = self._stage_conditioned_information(formed)
        visual_evidence = self._effective_visual_evidence(visual_evidence)
        initial_output = self.initial_symptom_candidate_generator(
            common_feature=formed["common_feature_low"],
            fine_grained_detail_feature=formed["fine_grained_detail_feature_low"],
            part_map=formed["part_map_low"],
            zone_map=formed["zone_map_low"],
            redness_prob=visual_evidence[:, 0:1],
            shape_prob=visual_evidence[:, 1:2],
            lesion_prob=visual_evidence[:, 2:3],
        )
        initial_symptom_candidate_response = initial_output[
            "initial_symptom_candidate_response_low"
        ]
        corrected = self._apply_correction(
            formed,
            initial_symptom_candidate_response,
            part_structure_information,
            visual_evidence,
        )
        signed_corrected_candidate_response = corrected["signed_corrected_candidate_response_low"]
        gated_residual_correction_term = corrected["gated_residual_correction_term_low"]
        corrected_symptom_candidate_response = corrected["corrected_symptom_candidate_response_low"]

        route_output = self.route_assignment_head(
            common_feature=formed["common_feature_low"],
            fine_grained_detail_feature=formed["fine_grained_detail_feature_low"],
            part_map=formed["part_map_low"],
            zone_map=formed["zone_map_low"],
            redness_prob=visual_evidence[:, 0:1],
            shape_prob=visual_evidence[:, 1:2],
            lesion_prob=visual_evidence[:, 2:3],
            unaffected_surface_prob=visual_evidence[:, 3:4],
        )
        route_assignment = route_output["route_assignment_prob_low"]
        route_assignment_with_background = route_output[
            "route_assignment_with_background_prob_low"
        ]
        support_output = self.spatial_support_head(
            common_feature=formed["common_feature_low"],
            fine_grained_detail_feature=formed["fine_grained_detail_feature_low"],
            route_assignment_with_background_prob=route_assignment_with_background,
            zone_map=formed["zone_map_low"],
            corrected_symptom_candidate_response=corrected_symptom_candidate_response,
            redness_prob=visual_evidence[:, 0:1],
            shape_prob=visual_evidence[:, 1:2],
            lesion_prob=visual_evidence[:, 2:3],
            unaffected_surface_prob=visual_evidence[:, 3:4],
        )
        spatial_support = support_output["spatial_support_prob_low"]
        spatial_support_internal = support_output["spatial_support_internal_prob_low"]

        if self.use_route_wise_composition:
            route_responses = self._route_responses(
                formed,
                corrected_symptom_candidate_response,
                route_assignment_with_background,
                spatial_support_internal,
                visual_evidence,
            )
            composition = self.route_wise_composer(
                corrected_symptom_candidate_response=corrected_symptom_candidate_response,
                fish_region=formed["fish_region_low"],
                route_assignment_prob=route_assignment,
                spatial_support_prob=spatial_support,
                body_route_response_prob=route_responses["body"]["route_response_prob_low"],
                fin_route_response_prob=route_responses["fin"]["route_response_prob_low"],
                caudal_fin_route_response_prob=route_responses["caudal_fin"][
                    "route_response_prob_low"
                ],
                mouth_route_response_prob=route_responses["mouth"]["route_response_prob_low"],
            )
            routed_logits = composition["routed_semantic_logits_low"]
        else:
            route_responses = {}
            composition = {}
            routed_logits = corrected_symptom_candidate_response

        class_zone_map = composition.get("route_class_support_low", formed["zone_map_low"])
        semantic_output = self._compose_semantic_output(
            fish_region=formed["fish_region_low"],
            routed_symptom_logits=routed_logits,
            class_zone_map=class_zone_map,
            positive_visual_evidence=visual_evidence[:, :3],
            unaffected_surface=visual_evidence[:, 3:4],
        )
        semantic_logits = semantic_output["native_semantic_logits_low"]
        foreground = semantic_output["symptom_foreground_probability_low"]
        if self.use_final_refinement:
            refinement = self.final_refinement(
                common_feature=formed["common_feature_low"],
                fine_grained_detail_feature=formed["fine_grained_detail_feature_low"],
                zone_map=formed["zone_map_low"],
                native_semantic_logits=semantic_logits,
                routed_class_probability=semantic_output["class_evidence_low"],
                routed_class_distribution=semantic_output["class_distribution_low"],
                symptom_foreground_probability=foreground,
                fish_region=formed["fish_region_low"],
                fish_boundary=formed["fish_boundary_low"],
                redness_prob=visual_evidence[:, 0:1],
                shape_prob=visual_evidence[:, 1:2],
                lesion_prob=visual_evidence[:, 2:3],
                unaffected_surface_prob=visual_evidence[:, 3:4],
            )
            auxiliary_logits = refinement["auxiliary_semantic_logits_low"]
        else:
            refinement = {}
            auxiliary_logits = semantic_logits

        upsampled = {
            "initial_symptom_candidate_response": self._upsample(
                initial_symptom_candidate_response, image
            ),
            "signed_corrected_candidate_response": self._upsample(
                signed_corrected_candidate_response, image
            ),
            "gated_residual_correction_term": self._upsample(gated_residual_correction_term, image),
            "corrected_symptom_candidate_response": self._upsample(
                corrected_symptom_candidate_response, image
            ),
            "routed_semantic_logits": self._upsample(routed_logits, image),
            "native_semantic_logits": self._upsample(semantic_logits, image),
            "auxiliary_semantic_logits": self._upsample(auxiliary_logits, image),
            "route_assignment": self._upsample(route_assignment, image),
            "spatial_support": self._upsample(spatial_support, image),
        }
        auxiliary_semantic_mask = upsampled["auxiliary_semantic_logits"].argmax(dim=1)
        output = {
            "initial_symptom_candidate_response": upsampled["initial_symptom_candidate_response"],
            "signed_corrected_candidate_response": upsampled["signed_corrected_candidate_response"],
            "gated_residual_correction_term": upsampled["gated_residual_correction_term"],
            "corrected_symptom_candidate_response": upsampled[
                "corrected_symptom_candidate_response"
            ],
            "routed_semantic_logits": upsampled["routed_semantic_logits"],
            "native_semantic_logits": upsampled["native_semantic_logits"],
            "auxiliary_semantic_logits": upsampled["auxiliary_semantic_logits"],
            "R0": upsampled["initial_symptom_candidate_response"],
            "N": upsampled["signed_corrected_candidate_response"],
            "A": upsampled["gated_residual_correction_term"],
            "C": upsampled["corrected_symptom_candidate_response"],
            "Z": upsampled["routed_semantic_logits"],
            "Y": auxiliary_semantic_mask,
            "auxiliary_semantic_mask": auxiliary_semantic_mask,
        }
        if return_aux:
            output.update(formed)
            output["effective_visual_evidence_low"] = visual_evidence
            output.update(
                {
                    "route_assignment": upsampled["route_assignment"],
                    "spatial_support": upsampled["spatial_support"],
                    "initial_symptom_candidate_response_low": initial_symptom_candidate_response,
                    "signed_corrected_candidate_response_low": signed_corrected_candidate_response,
                    "gated_residual_correction_term_low": gated_residual_correction_term,
                    "corrected_symptom_candidate_response_low": (
                        corrected_symptom_candidate_response
                    ),
                    "routed_semantic_logits_low": routed_logits,
                    "native_semantic_logits_low": semantic_logits,
                    "auxiliary_semantic_logits_low": auxiliary_logits,
                    "route_assignment_logits_low": route_output["route_assignment_logits_low"],
                    "route_assignment_low": route_assignment,
                    "spatial_support_logits_low": support_output["spatial_support_logits_low"],
                    "spatial_support_low": spatial_support,
                    "training_stage_id": torch.tensor(
                        float(self.current_stage_id), device=image.device
                    ),
                    "training_stage_name": self.current_stage_name,
                    "correction_variant": self.correction_variant.value,
                }
            )
            output.update(initial_output)
            output.update(corrected)
            output.update(route_output)
            output.update(support_output)
            output.update(composition)
            output.update(semantic_output)
            output.update(refinement)
            for route_name, route_values in route_responses.items():
                output[f"{route_name}_route_response_logits_low"] = route_values[
                    "route_response_logits_low"
                ]
                output[f"{route_name}_route_response_prob_low"] = route_values[
                    "route_response_prob_low"
                ]
        return output


def build_progressive_candidate_correction(
    num_classes: int = 9,
    backbone_name: str = "repvit_m1_1.dist_450e_in1k",
    pretrained: bool = True,
    in_ch: int = 3,
    out_indices: tuple[int, ...] = (0, 1, 2, 3),
    dec_ch: int = 96,
    head_ch: int = 64,
    part_names: Sequence[str] = PART_MAP_CLASS_NAMES,
    semantic_class_names: Sequence[str] | None = None,
    **kwargs,
) -> ProgressiveCandidateCorrectionNetwork:
    """Build the proposed network."""

    return ProgressiveCandidateCorrectionNetwork(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained,
        in_ch=in_ch,
        out_indices=out_indices,
        dec_ch=dec_ch,
        head_ch=head_ch,
        part_names=part_names,
        semantic_class_names=semantic_class_names,
        **kwargs,
    )
