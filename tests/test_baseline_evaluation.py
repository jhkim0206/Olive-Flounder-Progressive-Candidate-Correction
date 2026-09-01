from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_command_module():
    scripts_path = str(ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "evaluate_baseline_command",
        ROOT / "scripts" / "evaluate_baseline.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_candidates_use_foreground_channels_of_nine_class_softmax() -> None:
    torch = pytest.importorskip("torch")
    command = _load_command_module()
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0]],
                [[1.0, 4.0]],
                [[0.0, 0.0]],
                [[-1.0, 0.5]],
                [[0.5, -0.5]],
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[0.0, 0.0]],
            ]
        ]
    )

    labels, candidate_scores = command.baseline_outputs_from_logits(logits)

    assert torch.equal(labels, logits.argmax(dim=1))
    assert torch.allclose(candidate_scores, torch.softmax(logits, dim=1)[:, 1:])
    assert candidate_scores.shape == (1, 8, 1, 2)
    assert not torch.allclose(candidate_scores, torch.sigmoid(logits[:, 1:]))


def test_baseline_split_uses_one_global_confusion_and_generic_archive(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    command = _load_command_module()

    class FixedLogits(torch.nn.Module):
        def forward(self, image):
            logits = torch.full(
                (image.shape[0], 9, image.shape[2], image.shape[3]),
                -4.0,
                device=image.device,
            )
            logits[:, 0] = 0.0
            logits[:, 1, 0, 0] = 5.0
            logits[:, 2, 0, 1] = 5.0
            return logits

    target = torch.tensor([[[1, 2], [0, 0]]], dtype=torch.long)
    loader = [
        {
            "image": torch.zeros(1, 3, 2, 2),
            "semantic_target": target,
            "part_map_target": torch.ones(1, 2, 2, dtype=torch.long),
            "fish_region_target": torch.ones(1, 2, 2),
        },
        {
            "image": torch.zeros(1, 3, 2, 2),
            "semantic_target": target,
            "part_map_target": torch.ones(1, 2, 2, dtype=torch.long),
            "fish_region_target": torch.ones(1, 2, 2),
        },
    ]
    archive_path = tmp_path / "candidates.npz"

    report = command.evaluate_baseline_split(
        "unet_rgb",
        FixedLogits(),
        loader,
        device=torch.device("cpu"),
        candidate_archive=archive_path,
        checkpoint_digest="abc123",
    )

    assert report["num_eval_samples"] == 2
    assert report["confusion_matrix"][1][1] == 2
    assert report["confusion_matrix"][2][2] == 2
    assert report["symptom_foreground_macro_f1"] == pytest.approx(0.25)
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "candidate_scores",
            "candidate_source",
            "checkpoint_sha256",
            "semantic_targets",
            "part_maps",
            "fish_regions",
        }
        assert archive["candidate_scores"].shape == (2, 8, 2, 2)
        assert archive["candidate_source"].item() == "unet_rgb:foreground_softmax"
        assert archive["checkpoint_sha256"].item() == "abc123"


def test_baseline_checkpoint_requires_exact_identity_and_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    command = _load_command_module()
    model = torch.nn.Conv2d(3, 9, 1)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_name": "unet_rgb",
            "epoch": 120,
            "training_endpoint_epoch": 120,
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    command.load_baseline_checkpoint(
        model,
        checkpoint_path,
        expected_model_name="unet_rgb",
        expected_epoch=120,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="model_name"):
        command.load_baseline_checkpoint(
            model,
            checkpoint_path,
            expected_model_name="segformer_b0_rgb",
            expected_epoch=120,
            device=torch.device("cpu"),
        )
