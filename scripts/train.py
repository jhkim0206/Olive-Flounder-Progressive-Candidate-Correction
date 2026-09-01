#!/usr/bin/env python
"""Train the Progressive Candidate Correction network."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    REPOSITORY_ROOT,
    choose_device,
    class_names,
    instantiate_loaders,
    instantiate_loss,
    instantiate_model,
    load_config,
)

from progressive_candidate_correction import save_resolved_config
from progressive_candidate_correction.engine import fit, seed_everything
from progressive_candidate_correction.evaluation import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs" / "progressive_candidate_correction.yaml"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", help="Dataset root containing annotations/train/val")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a 64x64, one-epoch, one-step dummy run without pretrained downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    effective_epochs = int(args.epochs) if args.epochs is not None else None
    pretrained_override = None
    if args.smoke_test:
        effective_epochs = 1
        config.setdefault("data", {})["image_size"] = [64, 64]
        config["data"].setdefault("dummy", {})["num_samples"] = 2
        config["data"]["dummy"]["batch_size"] = 1
        config["model"]["backbone"]["pretrained"] = False
        config["data"]["provider"] = "dummy"
        pretrained_override = False
    elif args.data_root:
        config.setdefault("data", {})["root"] = args.data_root

    runtime_overrides = config.setdefault("runtime_overrides", {})
    if effective_epochs is not None:
        if effective_epochs <= 0:
            raise ValueError("--epochs must be positive")
        runtime_overrides["epochs"] = effective_epochs
    if args.smoke_test:
        runtime_overrides["smoke_test"] = True
        runtime_overrides["steps_per_epoch"] = 1

    seed = int(config.get("experiment", {}).get("seed", 45))
    runtime = config.get("training", {}).get("runtime", {})
    seed_everything(
        seed,
        cudnn_benchmark=bool(runtime.get("cudnn_benchmark", True)),
    )
    device = choose_device(args.device)
    model = instantiate_model(
        config,
        pretrained_override=pretrained_override,
    )
    loaders = instantiate_loaders(config)
    train_loader = loaders.get("train")
    if train_loader is None:
        raise KeyError("training data loader is missing")
    validation_loader = loaders.get("val")
    loss_overrides = {}
    weighting = config.get("loss", {}).get("class_weighting", {})
    train_dataset = getattr(train_loader, "dataset", None)
    if bool(weighting.get("enabled", False)) and hasattr(train_dataset, "pixel_class_weights"):
        semantic_weights, part_weights = train_dataset.pixel_class_weights(
            exponent=float(weighting.get("exponent", 0.5)),
            background_scale=float(weighting.get("background_scale", 0.25)),
        )
        loss_overrides.update(
            class_weights=semantic_weights,
            part_class_weights=part_weights,
        )
    criterion = instantiate_loss(config, **loss_overrides)
    names = class_names(config)

    def validation_fn(**kwargs):
        return evaluate_model(
            kwargs["model"],
            kwargs["loader"],
            device=kwargs["device"],
            num_classes=len(names),
            class_names=names,
            stage=kwargs.get("stage"),
            config=config,
        )

    output_dir = args.output_dir
    if output_dir is None:
        experiment = config.get("experiment", {})
        output_dir = str(
            Path(experiment.get("output_root", "outputs"))
            / str(experiment.get("run_name", "progressive_candidate_correction"))
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_path / "resolved_config.yaml")
    fit(
        model,
        criterion,
        train_loader,
        val_loader=validation_loader,
        validation_fn=validation_fn,
        device=device,
        config=config,
        save_dir=output_path,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
