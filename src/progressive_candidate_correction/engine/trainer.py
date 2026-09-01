"""Dataset-agnostic staged training loop."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..data.contract import LOSS_TARGET_KEYS
from ..training_stages import resolve_training_stage_name
from ._config import config_bool, config_get
from .checkpoint import load_checkpoint, save_checkpoint
from .optim import build_optimizer, build_scheduler
from .stages import (
    DEFAULT_STAGE_SCHEDULE,
    monitor_for_stage,
    stage_for_epoch,
)
from .trainability import configure_trainability, set_frozen_modules_to_eval


def _autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _gradient_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # PyTorch versions with the transitional signature.
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def get_image(batch: Mapping[str, Any]) -> torch.Tensor:
    image = batch.get("image")
    if not torch.is_tensor(image):
        raise KeyError("batch does not contain an image tensor")
    return image


def _infinite_batches(loader: Iterable[Mapping[str, Any]]):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("Cannot train from an empty data loader")


def _criterion_call(
    criterion: nn.Module,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [key for key in LOSS_TARGET_KEYS if key not in batch]
    if missing:
        raise KeyError(f"training batch is missing loss targets: {', '.join(missing)}")
    targets = {key: batch[key] for key in LOSS_TARGET_KEYS}
    targets["image"] = batch["image"]
    result = criterion(outputs, **targets)
    if torch.is_tensor(result):
        return {"loss": result}
    if not isinstance(result, Mapping) or "loss" not in result:
        raise RuntimeError("criterion must return a Tensor or a mapping containing 'loss'")
    return dict(result)


def _mean_float(value: Any, default: float = float("nan")) -> float:
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            return float(value.detach().float().mean().cpu().item())
        if isinstance(value, float | int):
            return float(value)
    except (TypeError, ValueError, RuntimeError):
        pass
    return default


def _steps_per_epoch(loader: Any, config: Mapping[str, Any]) -> int:
    configured = config_get(config, "steps_per_epoch", None)
    maximum = config_get(config, "max_train_batches", None)
    if configured is None:
        try:
            configured = len(loader)
        except TypeError:
            configured = maximum or 100
    result = int(configured)
    if maximum is not None:
        result = min(result, int(maximum))
    if result <= 0:
        raise ValueError("steps_per_epoch must be positive")
    return result


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    epoch: int,
    config: Mapping[str, Any],
    stage: str,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> dict[str, float]:
    device = torch.device(device)
    model.train()
    if hasattr(model, "set_training_stage"):
        model.set_training_stage(stage)
    if hasattr(criterion, "set_training_stage"):
        criterion.set_training_stage(stage)
    if config_bool(config, "strict_frozen_eval", False):
        set_frozen_modules_to_eval(model)

    steps = _steps_per_epoch(train_loader, config)
    train_iterator = _infinite_batches(train_loader)
    use_amp = (
        config_bool(config, "amp", config_bool(config, "use_amp", True)) and device.type == "cuda"
    )
    gradient_clip = config_get(config, "grad_clip_norm", 1.0)
    accumulation_steps = max(1, int(config_get(config, "grad_accum_steps", 1)))
    skip_nonfinite = config_bool(config, "skip_nonfinite_loss", True)
    scheduler_per_batch = config_bool(config, "scheduler_step_per_batch", True)

    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    nonfinite_skips = 0
    optimizer_steps = 0

    for step in range(steps):
        raw_batch = next(train_iterator)
        batch = to_device(raw_batch, device)
        image = get_image(batch)

        with _autocast(device, use_amp):
            outputs = model(image, return_aux=True)
            loss_outputs = _criterion_call(criterion, outputs, batch)
            loss = loss_outputs["loss"] / accumulation_steps

        if not torch.isfinite(loss.detach()).all():
            nonfinite_skips += 1
            if skip_nonfinite:
                optimizer.zero_grad(set_to_none=True)
                continue
            raise FloatingPointError(
                f"non-finite loss at epoch={epoch}, step={step}: {loss.item()}"
            )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_update = (step + 1) % accumulation_steps == 0 or (step + 1) == steps
        if should_update:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            if gradient_clip is not None and float(gradient_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(gradient_clip),
                )
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if scheduler is not None and scheduler_per_batch:
                scheduler.step()

        for key, value in loss_outputs.items():
            if not torch.is_tensor(value):
                continue
            scalar = _mean_float(value)
            if math.isfinite(scalar):
                totals[key] = totals.get(key, 0.0) + scalar
                counts[key] = counts.get(key, 0) + 1

    if scheduler is not None and not scheduler_per_batch:
        scheduler.step()

    metrics = {f"train/{key}": total / max(1, counts.get(key, 1)) for key, total in totals.items()}
    metrics.update(
        {
            "train/amp_skip": float(nonfinite_skips),
            "train/optimizer_steps": float(optimizer_steps),
            "train/lr": float(optimizer.param_groups[0].get("lr", 0.0)),
        }
    )
    return metrics


def _call_validation(
    validation_fn: Callable[..., Mapping[str, Any]],
    *,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    stage: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result = validation_fn(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        epoch=epoch,
        stage=stage,
        config=config,
    )
    return dict(result or {})


def _initial_best_value(monitor_mode: str) -> float:
    return -float("inf") if monitor_mode == "max" else float("inf")


def _best_state_from_history(
    history: Iterable[Mapping[str, Any]],
    monitor_mode: str,
) -> tuple[float, int, str | None]:
    best_value = _initial_best_value(monitor_mode)
    best_epoch = -1
    best_stage: str | None = None
    for record in history:
        value = record.get("monitor_value")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        improved = numeric > best_value if monitor_mode == "max" else numeric < best_value
        if improved:
            best_value = numeric
            best_epoch = int(record.get("epoch", -1))
            stage = record.get("stage")
            best_stage = resolve_training_stage_name(stage) if stage is not None else None
    return best_value, best_epoch, best_stage


def _restore_best_state(
    checkpoint: Mapping[str, Any],
    history: list[dict[str, Any]],
    monitor_mode: str,
) -> tuple[float, int, str | None]:
    state = checkpoint.get("monitor_state", {})
    if isinstance(state, Mapping) and state:
        stored_mode = str(state.get("monitor_mode", monitor_mode)).lower()
        if stored_mode != monitor_mode:
            raise ValueError(
                "resume checkpoint monitor mode does not match the requested run: "
                f"{stored_mode!r} != {monitor_mode!r}"
            )
        try:
            best_value = float(state["best_value"])
            best_epoch = int(state.get("best_epoch", -1))
            best_stage_value = state.get("best_stage")
            best_stage = (
                resolve_training_stage_name(best_stage_value)
                if best_stage_value is not None
                else None
            )
            return best_value, best_epoch, best_stage
        except (KeyError, TypeError, ValueError):
            pass
    return _best_state_from_history(history, monitor_mode)


def _select_monitor_value(
    validation_metrics: Mapping[str, Any],
    requested_monitor: str,
    config: Mapping[str, Any],
    *,
    validation_performed: bool,
) -> tuple[Any | None, str | None, bool]:
    """Resolve one monitor without silently changing the metric definition."""

    if not validation_performed:
        return None, None, False
    if requested_monitor in validation_metrics:
        return validation_metrics[requested_monitor], requested_monitor, False

    configured_fallback = config_get(config, "fallback_monitor", None)
    if configured_fallback is None or str(configured_fallback).lower() in {
        "",
        "none",
        "null",
        "disabled",
    }:
        available = ", ".join(sorted(str(key) for key in validation_metrics))
        raise KeyError(
            f"validation did not return requested monitor {requested_monitor!r}; "
            f"available keys: {available or '<none>'}. Configure fallback_monitor "
            "explicitly to use another checkpoint-selection metric."
        )

    fallback_monitor = monitor_for_stage("joint_fine_tuning", str(configured_fallback))
    if fallback_monitor not in validation_metrics:
        available = ", ".join(sorted(str(key) for key in validation_metrics))
        raise KeyError(
            f"validation returned neither requested monitor {requested_monitor!r} nor "
            f"explicit fallback {fallback_monitor!r}; available keys: {available or '<none>'}"
        )
    warnings.warn(
        f"using explicit fallback monitor {fallback_monitor!r} instead of {requested_monitor!r}",
        RuntimeWarning,
        stacklevel=2,
    )
    return validation_metrics[fallback_monitor], fallback_monitor, True


def fit(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    *,
    val_loader: Iterable[Mapping[str, Any]] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    validation_fn: Callable[..., Mapping[str, Any]] | None = None,
    device: torch.device | str = "cuda",
    config: Mapping[str, Any] | None = None,
    save_dir: str | Path | None = None,
    start_epoch: int = 1,
    resume: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Train through the six configured stages."""

    config = dict(config or {})
    from ..config import training_runtime_config

    config.update(training_runtime_config(config))
    runtime_overrides = config.get("runtime_overrides", {})
    if isinstance(runtime_overrides, Mapping):
        config.update(runtime_overrides)
    device = torch.device(device)
    model.to(device)
    if optimizer is None:
        optimizer = build_optimizer(model, config)
    if scheduler is None:
        scheduler = build_scheduler(optimizer, config, _steps_per_epoch(train_loader, config))

    use_amp = (
        config_bool(config, "amp", config_bool(config, "use_amp", True)) and device.type == "cuda"
    )
    scaler = _gradient_scaler(use_amp)
    history: list[dict[str, Any]] = []
    resume_checkpoint: Mapping[str, Any] = {}
    if resume is not None:
        loaded = load_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            scaler,
            map_location=device,
            strict=config_bool(config, "strict_load", False),
            prefer_ema=config_bool(config, "prefer_ema", False),
        )
        checkpoint = loaded.get("checkpoint", {})
        if isinstance(checkpoint, Mapping):
            resume_checkpoint = checkpoint
            start_epoch = max(int(start_epoch), int(checkpoint.get("epoch", 0)) + 1)
            history = list(checkpoint.get("history", []))

    epochs = int(config_get(config, "epochs", 120))
    schedule = config_get(config, "stage_schedule", DEFAULT_STAGE_SCHEDULE)
    output_dir = Path(
        save_dir
        or config_get(config, "save_dir", None)
        or Path(config_get(config, "save_root", "checkpoints"))
        / str(config_get(config, "run_name", "progressive_candidate_correction"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_mode = str(config_get(config, "monitor_mode", "max")).lower()
    if monitor_mode not in {"max", "min"}:
        raise ValueError("monitor_mode must be 'max' or 'min'")
    if resume_checkpoint:
        best_value, best_epoch, best_stage = _restore_best_state(
            resume_checkpoint,
            history,
            monitor_mode,
        )
    else:
        best_value = _initial_best_value(monitor_mode)
        best_epoch = -1
        best_stage = None
    previous_stage: str | None = (
        resolve_training_stage_name(history[-1].get("stage"))
        if history and history[-1].get("stage") is not None
        else None
    )

    for epoch in range(int(start_epoch), epochs + 1):
        stage = resolve_training_stage_name(stage_for_epoch(epoch, schedule))
        if (
            stage != previous_stage
            and previous_stage is not None
            and hasattr(model, "sync_part_structure_reference")
        ):
            model.sync_part_structure_reference()
        if scheduler is not None and hasattr(scheduler, "start_stage"):
            scheduler.start_stage(stage)
        stage_monitor = monitor_for_stage(stage, str(config_get(config, "monitor", "auto")))
        if stage != previous_stage and config_bool(config, "reset_best_on_stage_change", False):
            best_value = _initial_best_value(monitor_mode)
        trainable_counts = configure_trainability(model, stage, config)
        previous_stage = stage

        train_metrics = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            config,
            stage,
            scheduler=scheduler,
            scaler=scaler,
        )

        validation_metrics: dict[str, Any] = {}
        validation_performed = False
        validate_every = max(1, int(config_get(config, "validate_every", 1)))
        if (
            val_loader is not None
            and validation_fn is not None
            and (epoch % validate_every == 0 or epoch == epochs)
        ):
            validation_performed = True
            validation_metrics = _call_validation(
                validation_fn,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                stage=stage,
                config=config,
            )

        monitor_value, monitor_key_used, monitor_fallback_used = _select_monitor_value(
            validation_metrics,
            stage_monitor,
            config,
            validation_performed=validation_performed,
        )
        improved = False
        if monitor_value is not None and math.isfinite(float(monitor_value)):
            numeric_monitor = float(monitor_value)
            improved = (
                numeric_monitor > best_value
                if monitor_mode == "max"
                else numeric_monitor < best_value
            )
            if improved:
                best_value = numeric_monitor
                best_epoch = epoch
                best_stage = stage

        record = {
            "epoch": epoch,
            "stage": stage,
            "monitor": stage_monitor,
            "requested_monitor": stage_monitor,
            "monitor_key_used": monitor_key_used,
            "monitor_fallback_used": monitor_fallback_used,
            "monitor_value": float(monitor_value) if monitor_value is not None else None,
            "best_epoch": best_epoch,
            "best_stage": best_stage,
            "best_value": best_value if math.isfinite(best_value) else None,
            "trainable": trainable_counts,
            "train": train_metrics,
            "val": validation_metrics,
        }
        history.append(record)
        monitor_state = {
            "monitor_mode": monitor_mode,
            "reset_best_on_stage_change": config_bool(config, "reset_best_on_stage_change", False),
            "best_value": best_value if math.isfinite(best_value) else None,
            "best_epoch": best_epoch,
            "best_stage": best_stage,
        }
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            stage=stage,
            config=config,
            metrics=validation_metrics,
            history=history,
            monitor_state=monitor_state,
            checkpoint_role="last_epoch",
        )
        if improved and config_bool(config, "save_best_checkpoints", False):
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                stage=stage,
                config=config,
                metrics=validation_metrics,
                history=history,
                monitor_state=monitor_state,
                checkpoint_role="best_validation_monitor",
            )
        loss_value = train_metrics.get("train/loss", float("nan"))
        message = (
            f"[epoch {epoch:03d}/{epochs:03d}] stage={stage} "
            f"train/loss={loss_value:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if monitor_value is not None:
            assert monitor_key_used is not None
            message += f" {monitor_key_used}={float(monitor_value):.6f}"
            if monitor_fallback_used:
                message += f" (fallback for {stage_monitor})"
        if improved:
            message += " best*"
        print(message)

    with (output_dir / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2, default=str)
    return history
