"""Optimizer groups and the stage-local learning-rate schedule."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ._config import config_bool, config_get


def build_optimizer(
    model: nn.Module,
    config: Mapping[str, Any] | None = None,
) -> torch.optim.Optimizer:
    config = config or {}
    learning_rate = float(config_get(config, "lr", 3e-4))
    weight_decay = float(config_get(config, "weight_decay", 1e-4))
    scales = {
        "other": 1.0,
        "encoder": float(config_get(config, "encoder_lr_scale", 0.25)),
        "final_refinement": float(config_get(config, "final_refinement_lr_scale", 1.0)),
        "route_assignment": float(config_get(config, "route_assignment_lr_scale", 1.0)),
        "spatial_support": float(config_get(config, "spatial_support_lr_scale", 1.0)),
        "route_interpretation_heads": float(
            config_get(config, "route_interpretation_head_lr_scale", 1.0)
        ),
        "route_wise_composer": float(config_get(config, "route_wise_composer_lr_scale", 0.75)),
    }
    buckets: dict[str, list[nn.Parameter]] = {name: [] for name in scales}
    include_frozen = config_bool(config, "optimizer_include_frozen", True)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and not include_frozen:
            continue
        if "feature_formation.encoder" in name:
            bucket = "encoder"
        elif "final_refinement" in name:
            bucket = "final_refinement"
        elif "route_assignment_head" in name:
            bucket = "route_assignment"
        elif "spatial_support_head" in name:
            bucket = "spatial_support"
        elif any(
            token in name
            for token in (
                "body_route_interpretation_head",
                "mouth_route_interpretation_head",
                "fin_route_interpretation_head",
                "caudal_fin_route_interpretation_head",
            )
        ):
            bucket = "route_interpretation_heads"
        elif "route_wise_composer" in name:
            bucket = "route_wise_composer"
        else:
            bucket = "other"
        buckets[bucket].append(parameter)

    groups = [
        {
            "params": buckets[name],
            "lr": learning_rate * scale,
            "weight_decay": weight_decay,
            "name": name,
            "group_lr_scale": scale,
        }
        for name, scale in scales.items()
        if buckets[name]
    ]
    if not groups:
        raise RuntimeError("No parameters found for optimizer")

    optimizer_name = str(config_get(config, "optimizer", "adamw")).lower()
    optimizer_type = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    return optimizer_type(
        groups,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=tuple(config_get(config, "betas", (0.9, 0.999))),
        eps=float(config_get(config, "adam_eps", 1e-8)),
    )


class StagewiseWarmupCosineScheduler:
    """Restart warmup and cosine decay at each stage without resetting Adam."""

    policy = "stage_local_warmup_cosine"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
        stage_schedule: Sequence[Mapping[str, Any]],
        stage_lr_config: Mapping[str, Mapping[str, Any]],
        *,
        default_peak_lr: float = 3e-4,
        default_min_lr_ratio: float = 0.10,
        grad_accum_steps: int = 1,
    ) -> None:
        self.optimizer = optimizer
        self.optimizer_steps_per_epoch = max(
            1, math.ceil(int(steps_per_epoch) / max(1, int(grad_accum_steps)))
        )
        self.stage_schedule = [dict(item) for item in stage_schedule]
        self.stage_lr_config = {
            str(name): dict(profile) for name, profile in stage_lr_config.items()
        }
        self.default_peak_lr = float(default_peak_lr)
        self.default_min_lr_ratio = float(default_min_lr_ratio)
        self.group_names = [
            str(group.get("name", f"group_{index}"))
            for index, group in enumerate(optimizer.param_groups)
        ]
        self.group_lr_scales = [
            float(
                group.get(
                    "group_lr_scale",
                    float(group.get("lr", 0.0)) / max(self.default_peak_lr, 1e-12),
                )
            )
            for group in optimizer.param_groups
        ]
        self.stage_name: str | None = None
        self.local_step = 0
        self.stage_total_steps = 1
        self.stage_warmup_steps = 0
        self.stage_min_lr_ratio = self.default_min_lr_ratio
        self.stage_peak_lr = self.default_peak_lr
        self.stage_group_peak_lrs = [self.default_peak_lr * scale for scale in self.group_lr_scales]
        self.transition_count = 0
        self._validate()

    def _validate(self) -> None:
        if not self.stage_schedule:
            raise ValueError("stage_schedule must not be empty")
        previous = 0
        names: list[str] = []
        for item in self.stage_schedule:
            until = int(item["until"])
            if until <= previous:
                raise ValueError("stage boundaries must increase")
            previous = until
            names.append(str(item["stage"]))
        missing = sorted(set(names) - set(self.stage_lr_config))
        if missing:
            raise KeyError(f"missing scheduler profiles: {missing}")

    def _stage_epochs(self, stage: str) -> int:
        previous = 0
        for item in self.stage_schedule:
            until = int(item["until"])
            if str(item["stage"]) == stage:
                return until - previous
            previous = until
        raise KeyError(f"unknown stage: {stage}")

    def start_stage(self, stage: str, *, force: bool = False) -> dict[str, Any]:
        stage = str(stage)
        if self.stage_name == stage and not force:
            return self.describe()
        profile = self.stage_lr_config[stage]
        self.stage_name = stage
        self.local_step = 0
        self.stage_total_steps = max(1, self._stage_epochs(stage) * self.optimizer_steps_per_epoch)
        warmup_epochs = float(profile.get("warmup_epochs", 0.0))
        self.stage_warmup_steps = min(
            self.stage_total_steps,
            max(0, int(round(warmup_epochs * self.optimizer_steps_per_epoch))),
        )
        self.stage_min_lr_ratio = float(profile.get("min_lr_ratio", self.default_min_lr_ratio))
        if not 0.0 <= self.stage_min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.stage_peak_lr = float(profile.get("peak_lr", self.default_peak_lr))
        multipliers = dict(profile.get("group_multipliers", {}))
        unknown = sorted(set(multipliers) - set(self.group_names))
        if unknown:
            raise KeyError(f"unknown optimizer groups for {stage}: {unknown}")
        self.stage_group_peak_lrs = [
            self.stage_peak_lr * scale * float(multipliers.get(name, 1.0))
            for name, scale in zip(self.group_names, self.group_lr_scales, strict=False)
        ]
        self.transition_count += 1
        self._set_next_lr()
        return self.describe()

    def _scale(self, update_index: int) -> float:
        index = min(max(1, int(update_index)), self.stage_total_steps)
        if self.stage_warmup_steps and index <= self.stage_warmup_steps:
            return index / self.stage_warmup_steps
        decay_steps = self.stage_total_steps - self.stage_warmup_steps
        if decay_steps <= 1:
            progress = 1.0 if self.local_step >= self.stage_total_steps else 0.0
        else:
            progress = (index - self.stage_warmup_steps - 1) / (decay_steps - 1)
            progress = min(max(progress, 0.0), 1.0)
        return self.stage_min_lr_ratio + (1.0 - self.stage_min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _set_next_lr(self) -> None:
        if self.stage_name is None:
            return
        scale = self._scale(self.local_step + 1)
        for group, peak_lr in zip(
            self.optimizer.param_groups, self.stage_group_peak_lrs, strict=False
        ):
            group["lr"] = float(peak_lr * scale)

    def step(self) -> None:
        if self.stage_name is None:
            raise RuntimeError("start_stage() must be called before step()")
        self.local_step = min(self.local_step + 1, self.stage_total_steps)
        self._set_next_lr()

    def get_last_lr(self) -> list[float]:
        return [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "stage": self.stage_name,
            "local_step": self.local_step,
            "total_steps": self.stage_total_steps,
            "warmup_steps": self.stage_warmup_steps,
            "peak_lr": self.stage_peak_lr,
            "min_lr_ratio": self.stage_min_lr_ratio,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "stage_name": self.stage_name,
            "local_step": self.local_step,
            "stage_total_steps": self.stage_total_steps,
            "stage_warmup_steps": self.stage_warmup_steps,
            "stage_min_lr_ratio": self.stage_min_lr_ratio,
            "stage_peak_lr": self.stage_peak_lr,
            "stage_group_peak_lrs": self.stage_group_peak_lrs,
            "group_names": self.group_names,
            "group_lr_scales": self.group_lr_scales,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "transition_count": self.transition_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if str(state.get("policy", "")) != self.policy:
            raise RuntimeError("scheduler checkpoint uses a different policy")
        if list(state.get("group_names", [])) != self.group_names:
            raise RuntimeError("scheduler optimizer groups do not match")
        if int(state.get("optimizer_steps_per_epoch", -1)) != self.optimizer_steps_per_epoch:
            raise RuntimeError("scheduler steps per epoch do not match")
        self.stage_name = state.get("stage_name")
        self.local_step = int(state.get("local_step", 0))
        self.stage_total_steps = int(state.get("stage_total_steps", 1))
        self.stage_warmup_steps = int(state.get("stage_warmup_steps", 0))
        self.stage_min_lr_ratio = float(state.get("stage_min_lr_ratio", self.default_min_lr_ratio))
        self.stage_peak_lr = float(state.get("stage_peak_lr", self.default_peak_lr))
        self.stage_group_peak_lrs = [
            float(value) for value in state.get("stage_group_peak_lrs", [])
        ]
        self.group_lr_scales = [
            float(value) for value in state.get("group_lr_scales", self.group_lr_scales)
        ]
        self.transition_count = int(state.get("transition_count", 0))
        if len(self.stage_group_peak_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError("scheduler learning-rate groups do not match")
        self._set_next_lr()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any] | None,
    steps_per_epoch: int,
) -> StagewiseWarmupCosineScheduler | None:
    config = config or {}
    if not config_bool(config, "use_scheduler", True):
        return None
    schedule = config_get(config, "stage_schedule", ())
    profiles = config_get(config, "stage_lr_config", {})
    if not schedule or not profiles:
        raise ValueError("stage_schedule and stage_lr_config are required")
    return StagewiseWarmupCosineScheduler(
        optimizer,
        steps_per_epoch,
        schedule,
        profiles,
        default_peak_lr=float(config_get(config, "lr", 3e-4)),
        default_min_lr_ratio=float(config_get(config, "min_lr_ratio", 0.10)),
        grad_accum_steps=int(config_get(config, "grad_accum_steps", 1)),
    )
