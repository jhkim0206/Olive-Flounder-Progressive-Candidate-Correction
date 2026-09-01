#!/usr/bin/env python
"""Evaluate one semantic-segmentation comparison checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from _common import choose_device, json_ready

from progressive_candidate_correction.data import (
    build_olive_flounder_evaluation_dataloaders,
)
from progressive_candidate_correction.evaluation.evaluator import (
    SemanticMetricsAccumulator,
)
from progressive_candidate_correction.models.baselines import (
    build_baseline,
    forward_baseline,
)
from progressive_candidate_correction.schema import SEMANTIC_CLASS_NAMES


def baseline_outputs_from_logits(logits, size: tuple[int, int] | None = None):
    """Return semantic labels and foreground-class candidate scores."""

    import torch
    import torch.nn.functional as functional

    if logits.ndim != 4 or int(logits.shape[1]) != len(SEMANTIC_CLASS_NAMES):
        raise ValueError("baseline logits must contain background and eight symptom classes")
    if size is not None and tuple(logits.shape[-2:]) != tuple(size):
        logits = functional.interpolate(
            logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    probabilities = torch.softmax(logits.float(), dim=1)
    return logits.argmax(dim=1).long(), probabilities[:, 1:]


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_baseline_checkpoint(
    model,
    checkpoint_path: str | Path,
    *,
    expected_model_name: str,
    expected_epoch: int,
    device,
) -> None:
    """Load the configured final-epoch checkpoint with exact key matching."""

    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("baseline checkpoint must be a mapping")
    if checkpoint.get("model_name") != expected_model_name:
        raise ValueError("checkpoint model_name does not match the selected baseline")
    if int(checkpoint.get("training_endpoint_epoch", -1)) != int(expected_epoch):
        raise ValueError("checkpoint training endpoint does not match the baseline configuration")
    if int(checkpoint.get("epoch", -1)) != int(expected_epoch):
        raise ValueError("baseline evaluation requires the configured final-epoch checkpoint")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise KeyError("baseline checkpoint does not contain model_state")
    model.load_state_dict(state, strict=True)


def _save_candidate_archive(
    destination: str | Path,
    *,
    candidate_source: str,
    checkpoint_digest: str,
    score_batches: list[np.ndarray],
    target_batches: list[np.ndarray],
    part_map_batches: list[np.ndarray],
    fish_region_batches: list[np.ndarray],
) -> dict[str, Any]:
    archive_path = Path(destination)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_scores = np.concatenate(score_batches, axis=0)
    semantic_targets = np.concatenate(target_batches, axis=0)
    part_maps = np.concatenate(part_map_batches, axis=0)
    fish_regions = np.concatenate(fish_region_batches, axis=0)
    np.savez_compressed(
        archive_path,
        candidate_scores=candidate_scores,
        candidate_source=np.asarray(candidate_source),
        checkpoint_sha256=np.asarray(checkpoint_digest),
        semantic_targets=semantic_targets,
        part_maps=part_maps,
        fish_regions=fish_regions,
    )
    return {
        "path": str(archive_path),
        "samples": int(candidate_scores.shape[0]),
        "candidate_score_shape": list(candidate_scores.shape),
        "candidate_source": candidate_source,
        "checkpoint_sha256": checkpoint_digest,
    }


def evaluate_baseline_split(
    model_name: str,
    model,
    loader,
    *,
    device,
    candidate_archive: str | Path | None = None,
    checkpoint_digest: str | None = None,
) -> dict[str, Any]:
    """Evaluate a split and optionally retain its aligned candidate scores."""

    import torch

    if candidate_archive is not None and checkpoint_digest is None:
        raise ValueError("checkpoint_digest is required when exporting candidate scores")

    accumulator = SemanticMetricsAccumulator(
        num_classes=len(SEMANTIC_CLASS_NAMES),
        class_names=SEMANTIC_CLASS_NAMES,
    )
    score_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    part_map_batches: list[np.ndarray] = []
    fish_region_batches: list[np.ndarray] = []
    was_training = model.training
    model.to(device).eval()

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["semantic_target"].to(device, non_blocking=True)
            part_map = batch["part_map_target"].to(device, non_blocking=True)
            if target.ndim == 4:
                target = target[:, 0]
            logits = forward_baseline(model_name, model, image, part_map)
            size = tuple(int(value) for value in target.shape[-2:])
            prediction, candidate_scores = baseline_outputs_from_logits(logits, size)
            valid = (target >= 0) & (target != 255)
            accumulator.update(prediction, target.long(), valid=valid)

            if candidate_archive is not None:
                fish_region = batch["fish_region_target"]
                if fish_region.ndim == 4:
                    fish_region = fish_region[:, 0]
                score_batches.append(candidate_scores.cpu().numpy())
                target_batches.append(target.to(torch.uint8).cpu().numpy())
                part_map_batches.append(part_map.to(torch.uint8).cpu().numpy())
                fish_region_batches.append((fish_region > 0.5).to(torch.uint8).cpu().numpy())

    if was_training:
        model.train()
    if accumulator.samples == 0:
        raise ValueError("selected split contains no samples")

    report = accumulator.finalize()
    report["metric_protocol"].update(
        {
            "semantic_output": "9-class argmax",
            "foreground_macro_classes": list(SEMANTIC_CLASS_NAMES[1:]),
        }
    )
    if candidate_archive is not None:
        source = f"{model_name}:foreground_softmax"
        report["candidate_archive"] = _save_candidate_archive(
            candidate_archive,
            candidate_source=source,
            checkpoint_digest=str(checkpoint_digest),
            score_batches=score_batches,
            target_batches=target_batches,
            part_map_batches=part_map_batches,
            fish_region_batches=fish_region_batches,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument(
        "--candidate-archive",
        help="Write foreground softmax scores and aligned references to an NPZ archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_name = str(config["experiment"]["name"])
    training = config["training"]
    expected_epoch = int(training["checkpoint"]["training_endpoint_epoch"])
    device = choose_device(args.device)

    model_kwargs = {"pretrained": False} if model_name.startswith("segformer_") else {}
    model = build_baseline(model_name, **model_kwargs).to(device)
    load_baseline_checkpoint(
        model,
        args.checkpoint,
        expected_model_name=model_name,
        expected_epoch=expected_epoch,
        device=device,
    )
    digest = checkpoint_sha256(args.checkpoint)
    train_loader, validation_loader = build_olive_flounder_evaluation_dataloaders(
        args.data_root,
        batch_size=int(training.get("evaluation_batch_size", training["batch_size"])),
        num_workers=int(training["workers"]),
        seed=int(training["seed"]),
        image_size=tuple(config["dataset"]["image_size"]),
    )
    loader = train_loader if args.split == "train" else validation_loader
    report = evaluate_baseline_split(
        model_name,
        model,
        loader,
        device=device,
        candidate_archive=args.candidate_archive,
        checkpoint_digest=digest,
    )
    report["model_name"] = model_name
    report["split"] = args.split
    report["checkpoint_sha256"] = digest
    text = json.dumps(json_ready(report), indent=2, ensure_ascii=False, allow_nan=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
