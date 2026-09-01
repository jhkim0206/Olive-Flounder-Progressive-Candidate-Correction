"""Training utilities for progressive candidate correction."""

from ..training_stages import (
    TRAINING_STAGE_IDS,
    TRAINING_STAGE_NAMES,
    resolve_training_stage_id,
    resolve_training_stage_name,
)
from .checkpoint import extract_state_dict, load_checkpoint, save_checkpoint
from .optim import StagewiseWarmupCosineScheduler, build_optimizer, build_scheduler
from .stages import (
    DEFAULT_STAGE_SCHEDULE,
    STAGE_MONITOR,
    monitor_for_stage,
    seed_everything,
    stage_for_epoch,
)
from .trainability import configure_trainability, set_frozen_modules_to_eval
from .trainer import fit, get_image, to_device, train_one_epoch

__all__ = [
    "DEFAULT_STAGE_SCHEDULE",
    "STAGE_MONITOR",
    "TRAINING_STAGE_IDS",
    "TRAINING_STAGE_NAMES",
    "StagewiseWarmupCosineScheduler",
    "build_optimizer",
    "build_scheduler",
    "configure_trainability",
    "extract_state_dict",
    "fit",
    "get_image",
    "load_checkpoint",
    "monitor_for_stage",
    "resolve_training_stage_id",
    "resolve_training_stage_name",
    "save_checkpoint",
    "seed_everything",
    "set_frozen_modules_to_eval",
    "stage_for_epoch",
    "to_device",
    "train_one_epoch",
]
