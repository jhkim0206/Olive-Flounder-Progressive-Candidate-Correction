#!/usr/bin/env python
"""Evaluate semantic and candidate responses for one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _common import (
    REPOSITORY_ROOT,
    choose_device,
    class_names,
    instantiate_evaluation_loaders,
    instantiate_model,
    json_ready,
    load_config,
)

from progressive_candidate_correction.engine import load_checkpoint
from progressive_candidate_correction.evaluation import evaluate_model


def export_candidate_archive(
    model,
    loader,
    device,
    destination: str | Path,
    response_symbol: str,
    candidate_source: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Save one candidate response and its aligned reference masks."""

    import numpy as np
    import torch
    import torch.nn.functional as functional

    response_keys = {
        "R0": "initial_symptom_candidate_response",
        "C": "corrected_symptom_candidate_response",
    }
    response_key = response_keys[response_symbol]
    score_batches = []
    target_batches = []
    part_map_batches = []
    fish_region_batches = []
    was_training = model.training
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            output = model(image, return_aux=False)
            response = output.get(response_key)
            if not torch.is_tensor(response):
                raise KeyError(f"model output does not contain {response_key}")
            target = batch["semantic_target"]
            part_map = batch["part_map_target"]
            fish_region = batch["fish_region_target"]
            if fish_region.ndim == 4:
                fish_region = fish_region[:, 0]
            target_size = tuple(int(value) for value in target.shape[-2:])
            if tuple(response.shape[-2:]) != target_size:
                response = functional.interpolate(
                    response,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            score_batches.append(torch.sigmoid(response).float().cpu().numpy())
            target_batches.append(target.to(torch.uint8).cpu().numpy())
            part_map_batches.append(part_map.to(torch.uint8).cpu().numpy())
            fish_region_batches.append((fish_region > 0.5).to(torch.uint8).cpu().numpy())
    if was_training:
        model.train()
    if not score_batches:
        raise ValueError("selected split contains no samples")

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
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        semantic_targets=semantic_targets,
        part_maps=part_maps,
        fish_regions=fish_regions,
    )
    return {
        "path": str(archive_path),
        "samples": int(candidate_scores.shape[0]),
        "candidate_score_shape": list(candidate_scores.shape),
        "candidate_response": response_symbol,
        "candidate_source": candidate_source,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _checkpoint_sha256(path: str) -> str:
    if path.lower() == "none":
        return "none"
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs" / "progressive_candidate_correction.yaml"),
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Checkpoint path, or 'none' for current initialization."
    )
    parser.add_argument("--data-root")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument(
        "--candidate-archive",
        help="Write candidate scores and aligned references to an NPZ archive.",
    )
    parser.add_argument(
        "--candidate-response",
        choices=("R0", "C"),
        default="C",
        help="Candidate response exported with --candidate-archive (default: C).",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate one 64x64 dummy batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_batches is not None:
        config.setdefault("evaluation", {})["max_batches"] = int(args.max_batches)
    if args.smoke_test:
        config.setdefault("data", {})["image_size"] = [64, 64]
        config["data"].setdefault("dummy", {})["num_samples"] = 1
        config["data"]["dummy"]["batch_size"] = 1
        config["data"]["provider"] = "dummy"
        config.setdefault("evaluation", {})["max_batches"] = 1
    elif args.data_root:
        config.setdefault("data", {})["root"] = args.data_root
    device = choose_device(args.device)
    # A loaded checkpoint completely replaces initialization, avoiding an
    # unnecessary pretrained-weight download during evaluation.
    model = instantiate_model(config, pretrained_override=False)
    if args.checkpoint.lower() != "none":
        result = load_checkpoint(
            args.checkpoint,
            model,
            map_location=device,
            strict=args.strict,
        )
        if result["missing_keys"] or result["unexpected_keys"]:
            print(
                f"checkpoint keys: missing={len(result['missing_keys'])} "
                f"unexpected={len(result['unexpected_keys'])}"
            )

    loaders = instantiate_evaluation_loaders(config)
    loader = loaders.get(args.split)
    if loader is None:
        raise KeyError(f"unknown data split {args.split!r}; choose from {sorted(loaders)}")
    names = class_names(config)
    report = evaluate_model(
        model,
        loader,
        device=device,
        num_classes=len(names),
        class_names=names,
        config=config,
    )
    if args.candidate_archive:
        model_settings = config["model"]
        correction = model_settings["candidate_correction"]
        candidate_source = (
            f"{model_settings['name']}/{correction['variant']}:{args.candidate_response}"
        )
        report["candidate_archive"] = export_candidate_archive(
            model,
            loader,
            device,
            args.candidate_archive,
            args.candidate_response,
            candidate_source,
            _checkpoint_sha256(args.checkpoint),
        )
    serializable = json_ready(report)
    text = json.dumps(serializable, ensure_ascii=False, indent=2, allow_nan=True)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
