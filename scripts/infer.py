#!/usr/bin/env python
"""Run image inference and save integer symptom-label maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from _common import (
    REPOSITORY_ROOT,
    choose_device,
    class_names,
    instantiate_model,
    load_config,
)

from progressive_candidate_correction.engine import load_checkpoint
from progressive_candidate_correction.evaluation import semantic_logits_to_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs" / "progressive_candidate_correction.yaml"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", nargs="+", required=True, help="Image files or directories.")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def _input_files(values: list[str]) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            files.extend(
                child for child in sorted(path.iterdir()) if child.suffix.lower() in extensions
            )
        elif path.suffix.lower() in extensions:
            files.append(path)
        else:
            raise ValueError(f"Unsupported image input: {path}")
    if not files:
        raise ValueError("No input images found")
    return files


def _preprocess(path: Path, config: dict, device: torch.device) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency environment
        raise RuntimeError("Inference requires Pillow") from error
    data = config.get("data", {})
    height, width = [int(value) for value in data.get("image_size", (384, 384))]
    normalization = data.get("normalization", {})
    mean = torch.tensor(normalization.get("mean", (0.485, 0.456, 0.406))).view(3, 1, 1)
    std = torch.tensor(normalization.get("std", (0.229, 0.224, 0.225))).view(3, 1, 1)
    with Image.open(path) as image:
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((width, height), resample=resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    model = instantiate_model(config, pretrained_override=False)
    load_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
        strict=args.strict,
    )
    model.to(device).eval()
    if hasattr(model, "set_training_stage"):
        model.set_training_stage("joint_fine_tuning")

    names = class_names(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency environment
        raise RuntimeError("Inference requires Pillow") from error

    summaries = []
    for input_path in _input_files(args.input):
        image = _preprocess(input_path, config, device)
        autocast = (
            torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast(enabled=device.type == "cuda")
        )
        with torch.no_grad(), autocast:
            outputs = model(image)
        logits = outputs["auxiliary_semantic_logits"]
        labels = (
            semantic_logits_to_labels(
                logits,
                image.shape[-2:],
            )[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        destination = output_dir / f"{input_path.stem}_labels.png"
        Image.fromarray(labels, mode="L").save(destination)
        if args.save_logits:
            np.savez_compressed(
                output_dir / f"{input_path.stem}_logits.npz",
                logits=logits[0].detach().float().cpu().numpy(),
            )
        counts = np.bincount(labels.reshape(-1), minlength=len(names))
        summaries.append(
            {
                "input": str(input_path),
                "label_map": str(destination),
                "pixel_counts": {names[index]: int(counts[index]) for index in range(len(names))},
            }
        )
        print(destination)

    (output_dir / "inference_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
