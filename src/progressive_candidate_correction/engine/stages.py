"""Training-stage boundaries and deterministic seed handling."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ..training_stages import resolve_training_stage_id

# Epoch boundaries are inclusive.
DEFAULT_STAGE_SCHEDULE = (
    {"until": 25, "stage": "part_structure_information_formation"},
    {"until": 50, "stage": "visual_evidence_formation"},
    {"until": 60, "stage": "part_route_preparation"},
    {"until": 85, "stage": "candidate_and_route_interpretation"},
    {"until": 100, "stage": "semantic_output_refinement"},
    {"until": 120, "stage": "joint_fine_tuning"},
)

STAGE_MONITOR = {
    1: "part_structure_information_formation_validation_score",
    2: "visual_evidence_formation_validation_score",
    3: "part_route_preparation_validation_score",
    4: "candidate_and_route_interpretation_validation_score",
    5: "semantic_output_refinement_validation_score",
    6: "joint_fine_tuning_validation_score",
}


def seed_everything(seed: int = 45, *, cudnn_benchmark: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch using the experiment policy."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def stage_for_epoch(
    epoch: int,
    stage_schedule: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    schedule = tuple(stage_schedule or DEFAULT_STAGE_SCHEDULE)
    if not schedule:
        raise ValueError("stage_schedule must contain at least one stage")
    for item in schedule:
        if int(epoch) <= int(item["until"]):
            return str(item["stage"])
    return str(schedule[-1]["stage"])


def monitor_for_stage(stage: str | int, monitor: str = "auto") -> str:
    if monitor and str(monitor).lower() != "auto":
        return str(monitor)
    return STAGE_MONITOR[resolve_training_stage_id(stage)]
