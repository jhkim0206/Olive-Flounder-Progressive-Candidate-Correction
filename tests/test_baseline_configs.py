from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "configs" / "baselines"


def load_config(name: str) -> dict:
    return yaml.safe_load((BASELINE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def test_baseline_configs_match_the_training_settings() -> None:
    names = ("unet_rgb", "segformer_b0_rgb", "segformer_b0_part")
    configs = {name: load_config(name) for name in names}
    for name, config in configs.items():
        assert config["experiment"]["name"] == name
        assert config["dataset"]["image_size"] == [384, 384]
        assert config["dataset"]["train_images"] == 1169
        assert config["dataset"]["validation_images"] == 292
        assert config["training"]["seed"] == 45
        assert config["training"]["epochs"] == 120
        assert config["training"]["batch_size"] == 8
        assert config["training"]["optimizer"]["weight_decay"] == 0.0001
        assert config["training"]["checkpoint"] == {
            "file_name": "last.pt",
            "training_endpoint_epoch": 120,
            "selection": "final_epoch",
        }

    assert configs["unet_rgb"]["model"]["parameter_count"] == 7_763_305
    assert configs["unet_rgb"]["model"]["architecture"] == "unet"
    assert configs["unet_rgb"]["model"]["encoder_channels"] == [32, 64, 128, 256]
    assert configs["segformer_b0_rgb"]["model"]["parameter_count"] == 3_716_457
    assert configs["segformer_b0_part"]["model"]["parameter_count"] == 3_722_729
    assert configs["segformer_b0_rgb"]["model"]["architecture"] == "segformer"
    assert configs["segformer_b0_part"]["model"]["architecture"] == "segformer"


def test_part_baseline_uses_the_reviewed_part_map_input() -> None:
    config = load_config("segformer_b0_part")
    part = config["model"]["part_channels"]
    assert config["experiment"]["display_name"] == "Direct SegFormer-B0 + Part"
    assert config["model"]["input_channels"] == 7
    assert part["ids"] == [1, 4, 2, 3]
    assert (
        part["part_map_provenance"]
        == "part_segmentation_output_with_structural_refinement_and_review"
    )
    assert part["interpretation"] == "reference_input_comparison"
    assert config["model"]["first_patch_projection"]["added_channel_initialization"] == "zeros"


def test_candidate_evaluation_uses_the_method_part_scope() -> None:
    path = ROOT / "configs" / "evaluation" / "candidate_evaluation.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["protocol"]["calibration_split"] == "train"
    assert config["protocol"]["validation_split"] == "val"
    assert config["protocol"]["component_coverage_requirement"] == 0.5
    assert config["protocol"]["target_train_symptom_region_capture_rate"] == 0.7
    assert config["protocol"]["freeze_before_validation"] is True
    part = config["candidate_location_metrics"]["allowed_part_agreement"]
    assert part["include_outside_fish_region_for_allowed_part_agreement"] is False


def test_baseline_module_imports_without_training_dependencies() -> None:
    module = importlib.import_module("progressive_candidate_correction.models.baselines")
    assert module.BASELINE_MODEL_NAMES == (
        "unet_rgb",
        "segformer_b0_rgb",
        "segformer_b0_part",
    )
    if importlib.util.find_spec("torch") is None:
        assert module.TORCH_AVAILABLE is False
        with pytest.raises(RuntimeError, match="require PyTorch"):
            module.UNet()


def test_candidate_cli_exposes_the_part_scope_option() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_candidates.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--include-outside-fish-region-for-allowed-part-agreement" in result.stdout
