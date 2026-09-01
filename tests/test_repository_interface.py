from __future__ import annotations

import ast
from pathlib import Path

import progressive_candidate_correction
from progressive_candidate_correction.data.contract import REQUIRED_SAMPLE_KEYS

ROOT = Path(__file__).resolve().parents[1]


def test_package_configuration_entry_point_is_available() -> None:
    assert callable(progressive_candidate_correction.load_config)


def test_dataset_sample_fields_are_stable() -> None:
    assert set(REQUIRED_SAMPLE_KEYS) == {
        "image",
        "semantic_target",
        "fish_region_target",
        "part_map_target",
        "symptom_foreground_target",
        "unaffected_surface_target",
        "signed_distance_target",
        "head_to_tail_direction_target",
        "zone_map_target",
        "semantic_valid",
        "structure_valid",
        "positive_evidence_valid",
        "unaffected_surface_valid",
        "boundary_valid",
        "route_valid",
        "file_name",
        "fish_id",
        "image_id",
    }


def test_model_outputs_follow_the_paper_symbols() -> None:
    network_path = ROOT / "src/progressive_candidate_correction/models/network.py"
    network_source = network_path.read_text(encoding="utf-8")
    ast.parse(network_source)
    for key in (
        '"initial_symptom_candidate_response"',
        '"signed_corrected_candidate_response"',
        '"gated_residual_correction_term"',
        '"corrected_symptom_candidate_response"',
        '"routed_semantic_logits"',
        '"auxiliary_semantic_logits"',
        '"auxiliary_semantic_mask"',
    ):
        assert key in network_source


def test_candidate_archive_uses_the_evaluation_interface() -> None:
    source = (ROOT / "scripts/evaluate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for key in (
        '"corrected_symptom_candidate_response"',
        '"semantic_target"',
        '"part_map_target"',
        '"fish_region_target"',
    ):
        assert key in source
    archive_keys = {
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "savez_compressed"
        for keyword in node.keywords
    }
    assert archive_keys == {
        "candidate_scores",
        "candidate_source",
        "checkpoint_sha256",
        "semantic_targets",
        "part_maps",
        "fish_regions",
    }
    assert "--candidate-archive" in source
    assert "--candidate-response" in source
    assert "candidate_correction" in source
    assert "instantiate_evaluation_loaders" in source


def test_repository_entry_points_exist() -> None:
    expected = (
        "configs/progressive_candidate_correction.yaml",
        "configs/evaluation/candidate_evaluation.yaml",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/evaluate_baseline.py",
        "scripts/evaluate_candidates.py",
        "scripts/infer.py",
        "scripts/verify_dataset.py",
        "docs/DATASET.md",
        "docs/METHOD.md",
        "docs/SUPERVISION.md",
        "docs/LOSS_FUNCTIONS.md",
        "docs/TRAINING.md",
    )
    for relative in expected:
        assert (ROOT / relative).is_file()


def test_command_modules_are_valid_python() -> None:
    for path in sorted((ROOT / "scripts").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
