"""Load and translate experiment configurations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .schema import PART_MAP_CLASS_NAMES, SEMANTIC_CLASS_NAMES
from .training_stages import TRAINING_STAGE_IDS

ConfigLike = str | Path | Mapping[str, Any]


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_path(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(item.name for item in (*stack, path))
        raise ValueError(f"cyclic configuration inheritance: {chain}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise TypeError(f"configuration root must be a mapping: {path}")

    parents = document.get("extends", [])
    if isinstance(parents, str | Path):
        parents = [parents]
    resolved: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        resolved = _deep_merge(resolved, _load_path(parent_path, (*stack, path)))
    local = {key: value for key, value in document.items() if key != "extends"}
    return _deep_merge(resolved, local)


def _validate(config: Mapping[str, Any]) -> None:
    experiment = config.get("experiment", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    if int(experiment.get("epochs", 0)) <= 0:
        raise ValueError("experiment.epochs must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if int(model.get("num_classes", 0)) != len(data.get("class_names", [])):
        raise ValueError("model.num_classes must match data.class_names")
    if int(model.get("num_parts", 0)) != len(data.get("part_names", [])):
        raise ValueError("model.num_parts must match data.part_names")
    if tuple(str(name) for name in data.get("class_names", ())) != SEMANTIC_CLASS_NAMES:
        raise ValueError("data.class_names must match the method class order")
    if tuple(str(name) for name in data.get("part_names", ())) != PART_MAP_CLASS_NAMES:
        raise ValueError("data.part_names must match the method Part Map order")

    stages = training.get("stages", [])
    if not stages:
        raise ValueError("training.stages must not be empty")
    stage_names = [str(stage.get("name", "")) for stage in stages]
    if stage_names != list(TRAINING_STAGE_IDS):
        raise ValueError("training.stages must use the six method stage names in order")
    previous_end = 0
    for stage in stages:
        start, end = (int(value) for value in stage["epochs"])
        if start != previous_end + 1 or end < start:
            raise ValueError("training stages must be contiguous and ordered")
        previous_end = end
    if previous_end != int(experiment["epochs"]):
        raise ValueError("the final training stage must end at experiment.epochs")


def load_config(config: ConfigLike) -> dict[str, Any]:
    """Load a YAML file or copy an existing configuration mapping."""

    resolved = (
        _load_path(Path(config)) if isinstance(config, str | Path) else deepcopy(dict(config))
    )
    _validate(resolved)
    return resolved


def _section(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def model_kwargs_from_config(config: ConfigLike) -> dict[str, Any]:
    """Build model constructor arguments from a configuration."""

    cfg = load_config(config)
    data = _section(cfg["data"])
    model = _section(cfg["model"])
    backbone = _section(model.get("backbone"))
    correction = _section(model.get("candidate_correction"))
    route_wise_composition = _section(model.get("route_wise_composition"))
    refinement = _section(model.get("final_refinement"))
    evidence = _section(model.get("visual_evidence"))

    class_names = [str(name) for name in data["class_names"]]
    part_names = [str(name) for name in data["part_names"]]
    return {
        "num_classes": int(model["num_classes"]),
        "backbone_name": str(backbone["name"]),
        "pretrained": bool(backbone.get("pretrained", True)),
        "in_ch": int(backbone.get("input_channels", 3)),
        "out_indices": tuple(int(index) for index in backbone.get("output_indices", (0, 1, 2, 3))),
        "dec_ch": int(model.get("decoder_channels", 96)),
        "head_ch": int(model.get("head_channels", 64)),
        "semantic_class_names": class_names[1:],
        "part_names": part_names,
        "candidate_correction_variant": str(correction.get("variant", "full")),
        "visual_evidence_detach_stages": tuple(
            int(stage_id) for stage_id in evidence.get("detach_during_stages", (3, 4, 5, 6))
        ),
        "use_route_wise_composition": bool(route_wise_composition.get("enabled", True)),
        "route_strength": float(route_wise_composition.get("strength", 0.90)),
        "route_candidate_retention_weight": float(
            route_wise_composition.get("candidate_retention_weight", 0.12)
        ),
        "route_overwrite_strength": float(route_wise_composition.get("overwrite_strength", 0.90)),
        "use_final_refinement": bool(refinement.get("enabled", True)),
        "final_refinement_strength": float(refinement.get("strength", 0.35)),
    }


def loss_kwargs_from_config(config: ConfigLike) -> dict[str, Any]:
    """Build loss constructor arguments from a configuration."""

    cfg = load_config(config)
    data = _section(cfg["data"])
    model = _section(cfg["model"])
    loss = _section(cfg.get("loss"))
    candidate = _section(loss.get("candidate"))
    part_structure = _section(loss.get("part_structure"))
    evidence = _section(loss.get("visual_evidence"))
    route = _section(loss.get("route"))
    auxiliary = _section(loss.get("auxiliary"))

    return {
        "num_classes": int(model["num_classes"]),
        "num_parts": int(model["num_parts"]),
        "semantic_class_names": [str(name) for name in data["class_names"]][1:],
        "part_names": [str(name) for name in data["part_names"]],
        "generalized_cross_entropy_q": float(candidate.get("q", auxiliary.get("q", 0.7))),
        "fish_region_weight": float(part_structure.get("fish_region", 0.80)),
        "fish_region_dice_weight": float(part_structure.get("fish_region_dice", 0.50)),
        "part_map_weight": float(part_structure.get("part_map", 0.80)),
        "part_map_dice_weight": float(part_structure.get("part_map_dice", 0.35)),
        "head_to_tail_direction_weight": float(part_structure.get("head_to_tail_direction", 0.0)),
        "zone_map_weight": float(part_structure.get("zone_map", 0.0)),
        "redness_weight": float(evidence.get("redness", 0.25)),
        "shape_weight": float(evidence.get("shape", 0.45)),
        "lesion_weight": float(evidence.get("lesion", 0.55)),
        "positive_evidence_dice_weight": float(evidence.get("dice", 0.25)),
        "lesion_dice_weight": float(evidence.get("lesion_dice", 0.35)),
        "evidence_union_weight": float(evidence.get("union", 0.12)),
        "unaffected_surface_weight": float(evidence.get("unaffected_surface", 0.55)),
        "unaffected_surface_dice_weight": float(evidence.get("unaffected_surface_dice", 0.35)),
        "initial_candidate_weight": float(candidate.get("initial", 0.15)),
        "signed_candidate_weight": float(candidate.get("signed", 0.30)),
        "gated_candidate_weight": float(candidate.get("gated", 0.25)),
        "corrected_candidate_weight": float(candidate.get("corrected", 0.90)),
        "prototype_classification_weight": float(
            candidate.get("prototype_classification", 0.40)
        ),
        "prototype_alignment_weight": float(candidate.get("prototype_alignment", 0.12)),
        "part_conditioned_weight": float(candidate.get("part_conditioned", 0.45)),
        "candidate_foreground_bce_weight": float(candidate.get("foreground", 0.35)),
        "class_seed_positive_weight": float(candidate.get("class_seed_positive", 0.25)),
        "candidate_foreground_dice_weight": float(candidate.get("foreground_dice", 0.25)),
        "unaffected_surface_inhibition_weight": float(
            candidate.get("unaffected_surface_inhibition", 0.20)
        ),
        "class_seed_unaffected_surface_weight": float(
            candidate.get("class_seed_unaffected_surface", 0.12)
        ),
        "body_lesion_unaffected_surface_weight": float(
            candidate.get("body_lesion_unaffected_surface", 0.20)
        ),
        "body_lesion_budget": float(candidate.get("body_lesion_budget", 0.12)),
        "body_lesion_budget_weight": float(
            candidate.get("body_lesion_budget_weight", 0.10)
        ),
        "symptom_foreground_fish_region_budget": float(
            candidate.get("symptom_foreground_fish_region_budget", 0.18)
        ),
        "symptom_foreground_fish_region_budget_weight": float(
            candidate.get("symptom_foreground_fish_region_budget_weight", 0.08)
        ),
        "caudal_necrosis_region_budget": float(
            candidate.get("caudal_necrosis_region_budget", 0.62)
        ),
        "caudal_necrosis_region_budget_weight": float(
            candidate.get("caudal_necrosis_region_budget_weight", 0.16)
        ),
        "routed_semantic_weight": float(route.get("routed_semantic", 0.75)),
        "route_assignment_weight": float(route.get("assignment", 0.35)),
        "route_part_consistency_weight": float(route.get("part_consistency", 0.25)),
        "spatial_support_weight": float(route.get("spatial_support", 0.55)),
        "spatial_support_dice_weight": float(route.get("spatial_support_dice", 0.25)),
        "route_interpretation_weight": float(route.get("interpretation", 0.65)),
        "route_preservation_weight": float(route.get("preservation", 0.025)),
        "route_delta_sparsity_weight": float(route.get("delta_sparsity", 0.010)),
        "route_consistency_weight": float(route.get("consistency", 0.030)),
        "spatial_support_closure_weight": float(
            route.get("spatial_support_closure", 0.020)
        ),
        "spatial_support_unaffected_surface_weight": float(
            route.get("spatial_support_unaffected_surface", 0.040)
        ),
        "route_preservation_confidence_softness": float(
            route.get("preservation_confidence_softness", 0.70)
        ),
        "route_interpretation_unaffected_surface_weight": float(
            route.get("interpretation_unaffected_surface", 0.10)
        ),
        "spatial_support_closure_kernel": int(
            route.get("spatial_support_closure_kernel", 5)
        ),
        "auxiliary_semantic_weight": float(auxiliary.get("semantic_mask", 0.45)),
        "auxiliary_semantic_dice_weight": float(auxiliary.get("semantic_mask_dice", 0.35)),
        "native_semantic_contribution": float(
            auxiliary.get("native_semantic_contribution", 0.35)
        ),
        "final_refinement_weight": float(auxiliary.get("final_refinement", 0.40)),
        "signed_distance_weight": float(auxiliary.get("signed_distance", 0.35)),
        "signed_distance_temperature": float(
            auxiliary.get("signed_distance_temperature", 4.0)
        ),
        "refinement_foreground_dice_weight": float(auxiliary.get("foreground_dice", 0.25)),
        "refinement_curvature_weight": float(auxiliary.get("curvature", 0.05)),
        "refinement_total_variation_weight": float(auxiliary.get("total_variation", 0.025)),
        "refinement_fish_region_boundary_weight": float(auxiliary.get("bilateral", 0.025)),
        "stage_multipliers": deepcopy(loss.get("stage_multipliers")),
    }


def training_runtime_config(config: ConfigLike) -> dict[str, Any]:
    """Flatten training settings for the execution engine."""

    cfg = load_config(config)
    experiment = _section(cfg["experiment"])
    training = _section(cfg["training"])
    optimizer = _section(training.get("optimizer"))
    scheduler = _section(training.get("scheduler"))
    runtime = _section(training.get("runtime"))
    stages = list(training["stages"])

    stage_schedule = []
    stage_lr_config: dict[str, dict[str, float]] = {}
    for stage in stages:
        name = str(stage["name"])
        stage_schedule.append({"until": int(stage["epochs"][1]), "stage": name})
        stage_lr_config[name] = {
            "peak_lr": float(stage["peak_learning_rate"]),
            "warmup_epochs": float(stage.get("warmup_epochs", 0.0)),
            "min_lr_ratio": float(stage.get("minimum_learning_rate_ratio", 0.10)),
        }
        if "group_multipliers" in stage:
            stage_lr_config[name]["group_multipliers"] = deepcopy(stage["group_multipliers"])

    return {
        "epochs": int(experiment["epochs"]),
        "run_name": str(experiment.get("run_name", "progressive_candidate_correction")),
        "save_root": str(experiment.get("output_root", "outputs")),
        "stage_schedule": stage_schedule,
        "stage_lr_config": stage_lr_config,
        "optimizer": str(optimizer.get("type", "adamw")),
        "lr": float(optimizer.get("learning_rate", 3.0e-4)),
        "weight_decay": float(optimizer.get("weight_decay", 1.0e-4)),
        "betas": tuple(float(value) for value in optimizer.get("betas", (0.9, 0.999))),
        "adam_eps": float(optimizer.get("epsilon", 1.0e-8)),
        "optimizer_include_frozen": bool(optimizer.get("include_frozen_parameters", True)),
        "encoder_lr_scale": float(optimizer.get("encoder_lr_scale", 0.25)),
        "use_scheduler": bool(scheduler.get("enabled", True)),
        "scheduler_step_per_batch": bool(scheduler.get("step_per_batch", True)),
        "min_lr_ratio": float(scheduler.get("minimum_learning_rate_ratio", 0.10)),
        "amp": bool(runtime.get("automatic_mixed_precision", True)),
        "grad_clip_norm": float(runtime.get("gradient_clip_norm", 1.0)),
        "grad_accum_steps": int(runtime.get("gradient_accumulation_steps", 1)),
        "skip_nonfinite_loss": bool(runtime.get("skip_nonfinite_loss", True)),
        "strict_frozen_eval": bool(runtime.get("strict_frozen_eval", False)),
        "validate_every": int(runtime.get("validate_every_epochs", 1)),
        "save_best_checkpoints": bool(runtime.get("save_best_checkpoints", False)),
    }


def build_model(config: ConfigLike, **overrides: Any):
    """Construct the proposed network."""

    from .models import build_progressive_candidate_correction

    kwargs = model_kwargs_from_config(config)
    kwargs.update(overrides)
    return build_progressive_candidate_correction(**kwargs)


def build_criterion(config: ConfigLike, **overrides: Any):
    """Construct the stage-weighted training objective."""

    from .losses import ProgressiveCandidateCorrectionLoss

    kwargs = loss_kwargs_from_config(config)
    kwargs.update(overrides)
    return ProgressiveCandidateCorrectionLoss(**kwargs)


def save_resolved_config(config: ConfigLike, path: str | Path) -> None:
    """Write a resolved configuration to YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(load_config(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
