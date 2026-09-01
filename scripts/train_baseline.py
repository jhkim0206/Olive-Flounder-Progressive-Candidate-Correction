#!/usr/bin/env python
"""Train a comparison model to its configured final epoch."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _class_weights(dataset, classes: int = 9):
    import torch
    from PIL import Image

    counts = np.zeros(classes, dtype=np.float64)
    for record in dataset.records:
        with Image.open(record["semantic_cache_path"]) as image:
            target = np.asarray(image, dtype=np.int64)
        counts += np.bincount(target.reshape(-1), minlength=classes)[:classes]
    frequency = counts / max(float(counts.sum()), 1.0)
    weights = 1.0 / np.sqrt(frequency + 1e-8)
    weights /= max(float(weights.mean()), 1e-8)
    weights[0] *= 0.25
    return torch.from_numpy(weights.astype(np.float32))


def _foreground_dice_loss(logits, target, epsilon: float = 1e-6):
    import torch
    import torch.nn.functional as functional

    probabilities = torch.softmax(logits, dim=1)[:, 1:]
    one_hot = functional.one_hot(target.clamp(0, 8), 9).permute(0, 3, 1, 2).float()[:, 1:]
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def _optimizer(model_name: str, model, weight_decay: float):
    import torch

    if model_name == "unet_rgb":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=weight_decay)
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith("segformer.") else head).append(parameter)
    return torch.optim.AdamW(
        [{"params": encoder, "lr": 6e-5}, {"params": head, "lr": 6e-4}],
        weight_decay=weight_decay,
    )


def _scheduler(optimizer, epochs: int, warmup_epochs: int):
    import torch

    def factor(epoch_index: int) -> float:
        epoch = epoch_index + 1
        if epoch <= warmup_epochs:
            return max(epoch / max(warmup_epochs, 1), 1e-3)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _run_epoch(model_name, model, loader, class_weights, device, scaler, optimizer=None):
    import torch
    import torch.nn.functional as functional

    from progressive_candidate_correction.models.baselines import forward_baseline

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["semantic_target"].to(device, non_blocking=True)
        part_map = batch["part_map_target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = forward_baseline(model_name, model, image, part_map)
                loss = functional.cross_entropy(logits, target, weight=class_weights)
                loss = loss + 0.5 * _foreground_dice_loss(logits, target)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(image.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch

    from progressive_candidate_correction.data import build_olive_flounder_dataloaders
    from progressive_candidate_correction.models.baselines import build_baseline

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name = str(config["experiment"]["name"])
    training = config["training"]
    seed = int(training["seed"])
    epochs = int(training["epochs"])
    _seed_everything(seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    train_loader, val_loader = build_olive_flounder_dataloaders(
        args.data_root,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["workers"]),
        seed=seed,
        image_size=tuple(config["dataset"]["image_size"]),
    )
    model = build_baseline(name).to(device)
    weights = _class_weights(train_loader.dataset).to(device)
    optimizer = _optimizer(name, model, float(training["optimizer"]["weight_decay"]))
    scheduler = _scheduler(optimizer, epochs, int(training["scheduler"]["warmup_epochs"]))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    run_dir = Path(args.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint["model_name"] != name or int(checkpoint["training_endpoint_epoch"]) != epochs:
            raise ValueError("checkpoint does not match this baseline run")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, epochs + 1):
        train_loss = _run_epoch(name, model, train_loader, weights, device, scaler, optimizer)
        with torch.no_grad():
            validation_loss = _run_epoch(name, model, val_loader, weights, device, scaler)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "lr_group_0": float(optimizer.param_groups[0]["lr"]),
            "lr_group_1": float(optimizer.param_groups[-1]["lr"]),
        }
        history.append(row)
        payload = {
            "model_name": name,
            "epoch": epoch,
            "seed": seed,
            "training_endpoint_epoch": epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "resolved_config": config,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{name} epoch {epoch}/{epochs}: "
            f"train={train_loss:.5f} validation={validation_loss:.5f}"
        )


if __name__ == "__main__":
    main()
