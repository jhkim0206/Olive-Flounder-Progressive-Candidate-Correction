"""Canonical names for the six training stages."""

from __future__ import annotations

TRAINING_STAGE_IDS = {
    "part_structure_information_formation": 1,
    "visual_evidence_formation": 2,
    "part_route_preparation": 3,
    "candidate_and_route_interpretation": 4,
    "semantic_output_refinement": 5,
    "joint_fine_tuning": 6,
}

TRAINING_STAGE_NAMES = {
    1: "part_structure_information_formation",
    2: "visual_evidence_formation",
    3: "part_route_preparation",
    4: "candidate_and_route_interpretation",
    5: "semantic_output_refinement",
    6: "joint_fine_tuning",
}


def resolve_training_stage_id(stage: str | int) -> int:
    """Return the numeric ID for a training-stage name."""

    if isinstance(stage, int):
        if stage in TRAINING_STAGE_NAMES:
            return stage
        raise ValueError(f"training stage ID must be in [1, 6], received {stage}")
    name = str(stage)
    try:
        return TRAINING_STAGE_IDS[name]
    except KeyError as exc:
        choices = ", ".join(TRAINING_STAGE_IDS)
        raise ValueError(f"unknown training stage {stage!r}; choose from {choices}") from exc


def resolve_training_stage_name(stage: str | int) -> str:
    """Return the canonical name for a training-stage ID or name."""

    stage_id = resolve_training_stage_id(stage)
    return TRAINING_STAGE_NAMES[stage_id]
