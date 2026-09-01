"""Stage-specific parameter freezing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from ..training_stages import resolve_training_stage_id


def module_by_path(root: nn.Module, path: str) -> nn.Module | None:
    current: Any = root
    for name in path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current if isinstance(current, nn.Module) else None


def set_requires_grad(module: nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters(recurse=True):
        parameter.requires_grad = bool(enabled)


def _set_all(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = bool(enabled)


def configure_trainability(
    model: nn.Module,
    stage: str | int,
    config: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Activate the module groups assigned to one training stage."""

    config = config or {}
    stage_id = resolve_training_stage_id(stage)
    if hasattr(model, "set_training_stage"):
        model.set_training_stage(stage)
    _set_all(model, False)

    def enable(*paths: str) -> None:
        for path in paths:
            set_requires_grad(module_by_path(model, path), True)

    structure_modules = (
        "feature_formation.decoder",
        "feature_formation.common_refinement",
        "feature_formation.fish_region_estimator",
        "feature_formation.head_to_tail_direction_estimator",
        "feature_formation.head_endpoint_role_head",
        "feature_formation.part_map_prior_builder",
        "feature_formation.part_map_estimator",
        "feature_formation.zone_map_estimator",
    )
    evidence_modules = (
        "feature_formation.redness_evidence_estimator",
        "feature_formation.shape_evidence_estimator",
        "feature_formation.lesion_evidence_estimator",
        "feature_formation.unaffected_surface_evidence_estimator",
    )
    candidate_modules = (
        "initial_symptom_candidate_generator",
        "signed_candidate_corrector",
        "gated_residual_corrector",
        "concat_candidate_corrector",
        "candidate_evidence_integrator",
    )
    candidate_context_modules = (
        "part_conditioned_abnormality_gate",
        "latent_visual_feature_encoder",
        "symptom_prototype_bank",
    )
    route_preparation_modules = (
        "route_assignment_head",
        "spatial_support_head",
    )
    route_interpretation_modules = (
        *route_preparation_modules,
        "body_route_interpretation_head",
        "mouth_route_interpretation_head",
        "fin_route_interpretation_head",
        "caudal_fin_route_interpretation_head",
        "route_wise_composer",
    )

    if stage_id == 1:
        enable(*structure_modules)
    elif stage_id == 2:
        enable(*evidence_modules, "part_conditioned_abnormality_gate")
    elif stage_id == 3:
        enable(*candidate_modules, *candidate_context_modules, *route_preparation_modules)
    elif stage_id == 4:
        enable(*candidate_modules, *candidate_context_modules, *route_interpretation_modules)
    elif stage_id == 5:
        enable(*route_interpretation_modules, "final_refinement")
    elif stage_id == 6:
        enable(
            *candidate_modules,
            *candidate_context_modules,
            *route_interpretation_modules,
            "final_refinement",
        )

    for path in (
        "feature_formation.reference_fish_region_estimator",
        "feature_formation.reference_head_endpoint_role_head",
        "feature_formation.reference_part_map_estimator",
    ):
        set_requires_grad(module_by_path(model, path), False)

    groups = {
        "encoder": ("feature_formation.encoder",),
        "decoder": ("feature_formation.decoder", "feature_formation.common_refinement"),
        "part_structure": structure_modules[2:],
        "visual_evidence": evidence_modules,
        "candidate_correction": (*candidate_modules, *candidate_context_modules),
        "candidate_context": candidate_context_modules,
        "route_assignment": ("route_assignment_head",),
        "spatial_support": ("spatial_support_head",),
        "route_interpretation_heads": route_interpretation_modules[2:-1],
        "route_wise_composer": ("route_wise_composer",),
        "final_refinement": ("final_refinement",),
    }
    counts: dict[str, int] = {}
    for group_name, paths in groups.items():
        counts[group_name] = sum(
            parameter.numel()
            for path in paths
            for module in (module_by_path(model, path),)
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    counts["total_trainable"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    counts["all_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return counts


def set_frozen_modules_to_eval(model: nn.Module) -> None:
    """Place fully frozen submodules in evaluation mode after ``model.train()``."""

    for module in model.modules():
        parameters = tuple(module.parameters(recurse=True))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()
