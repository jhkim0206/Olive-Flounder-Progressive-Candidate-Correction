from __future__ import annotations

from pathlib import Path

import pytest

from progressive_candidate_correction.config import (
    load_config,
    loss_kwargs_from_config,
    model_kwargs_from_config,
    training_runtime_config,
)
from progressive_candidate_correction.training_stages import TRAINING_STAGE_IDS

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


METHOD_CONFIG = CONFIGS / "progressive_candidate_correction.yaml"


def test_method_config_matches_the_paper_settings() -> None:
    config = load_config(METHOD_CONFIG)
    assert config["experiment"]["seed"] == 45
    assert config["experiment"]["epochs"] == 120
    assert config["training"]["batch_size"] == 8
    assert config["training"]["optimizer"]["weight_decay"] == 1e-4
    assert config["model"]["backbone"]["name"] == "repvit_m1_1.dist_450e_in1k"


def test_six_stage_schedule_and_learning_rate_profiles() -> None:
    runtime = training_runtime_config(METHOD_CONFIG)
    assert [item["until"] for item in runtime["stage_schedule"]] == [
        25,
        50,
        60,
        85,
        100,
        120,
    ]
    assert runtime["stage_lr_config"]["part_structure_information_formation"] == {
        "peak_lr": 3e-4,
        "warmup_epochs": 2.0,
        "min_lr_ratio": 0.1,
    }
    assert runtime["stage_lr_config"]["joint_fine_tuning"]["peak_lr"] == 7.5e-5


def test_candidate_correction_comparisons_use_inherited_configs() -> None:
    expected = {
        "progressive_candidate_correction.yaml": "full",
        "correction/concatenation.yaml": "concat",
        "correction/signed_correction.yaml": "signed",
    }
    for relative, variant in expected.items():
        kwargs = model_kwargs_from_config(CONFIGS / relative)
        assert kwargs["candidate_correction_variant"] == variant


def test_candidate_loss_weights_are_explicit() -> None:
    config = load_config(METHOD_CONFIG)
    configured_loss = config["loss"]
    assert set(configured_loss) == {
        "class_weighting",
        "part_structure",
        "visual_evidence",
        "candidate",
        "route",
        "auxiliary",
        "stage_multipliers",
    }
    assert configured_loss["class_weighting"] == {
        "enabled": True,
        "exponent": 0.5,
        "background_scale": 0.25,
    }
    assert configured_loss["visual_evidence"]["unaffected_surface"] == 0.55
    assert configured_loss["auxiliary"]["semantic_mask"] == 0.45
    assert all(
        set(weights)
        == {
            "fish_region",
            "part_map",
            "visual_evidence",
            "unaffected_surface",
            "part_conditioned",
            "candidate",
            "prototype",
            "semantic",
            "foreground",
            "final_refinement",
            "suppression",
        }
        for weights in configured_loss["stage_multipliers"].values()
    )
    loss = loss_kwargs_from_config(METHOD_CONFIG)
    assert loss["generalized_cross_entropy_q"] == 0.7
    assert loss["initial_candidate_weight"] == 0.15
    assert loss["signed_candidate_weight"] == 0.30
    assert loss["gated_candidate_weight"] == 0.25
    assert loss["corrected_candidate_weight"] == 0.90
    assert loss["prototype_classification_weight"] == 0.40
    assert loss["prototype_alignment_weight"] == 0.12
    assert loss["routed_semantic_weight"] == 0.75
    assert loss["route_assignment_weight"] == 0.35
    assert loss["spatial_support_weight"] == 0.55
    assert loss["auxiliary_semantic_weight"] == 0.45
    assert loss["auxiliary_semantic_dice_weight"] == 0.35
    assert loss["stage_multipliers"] == configured_loss["stage_multipliers"]


def test_stage_names_are_shared_by_engine_model_and_objective() -> None:
    config = load_config(METHOD_CONFIG)
    configured_names = []
    for stage_id, stage in enumerate(config["training"]["stages"], start=1):
        name = stage["name"]
        configured_names.append(name)
        assert TRAINING_STAGE_IDS[name] == stage_id
    assert configured_names == list(TRAINING_STAGE_IDS)


def test_trainability_and_optimizer_groups_follow_the_current_network() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    from progressive_candidate_correction.config import build_model
    from progressive_candidate_correction.engine.optim import build_optimizer, build_scheduler
    from progressive_candidate_correction.engine.trainability import configure_trainability

    model = build_model(METHOD_CONFIG, pretrained=False)

    stage_one = configure_trainability(model, "part_structure_information_formation")
    assert stage_one["part_structure"] > 0
    assert stage_one["encoder"] == 0

    stage_three = configure_trainability(model, "part_route_preparation")
    assert stage_three["candidate_correction"] > 0
    assert stage_three["route_assignment"] > 0
    assert stage_three["spatial_support"] > 0
    assert stage_three["route_interpretation_heads"] == 0

    stage_four = configure_trainability(model, "candidate_and_route_interpretation")
    assert stage_four["route_interpretation_heads"] > 0

    stage_five = configure_trainability(model, "semantic_output_refinement")
    assert stage_five["route_interpretation_heads"] > 0
    assert stage_five["final_refinement"] > 0

    stage_six = configure_trainability(model, "joint_fine_tuning")
    assert stage_six["encoder"] == 0
    assert stage_six["visual_evidence"] == 0
    assert stage_six["candidate_correction"] > 0
    assert stage_six["route_interpretation_heads"] > 0
    assert stage_six["final_refinement"] > 0
    assert stage_six["total_trainable"] < stage_six["all_parameters"]

    runtime = training_runtime_config(METHOD_CONFIG)
    optimizer = build_optimizer(model, runtime)
    optimizer_groups = {str(group["name"]) for group in optimizer.param_groups}
    assert "route_interpretation_heads" in optimizer_groups
    scheduler = build_scheduler(optimizer, runtime, steps_per_epoch=1)
    assert scheduler is not None
    scheduler.start_stage("semantic_output_refinement")

    del optimizer, scheduler, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_signed_cue_directions_match_the_paper() -> None:
    pytest.importorskip("torch")

    from progressive_candidate_correction.models.candidate_correction import (
        SignedCandidateCorrector,
    )
    from progressive_candidate_correction.schema import SYMPTOM_CLASS_NAMES

    corrector = SignedCandidateCorrector(SYMPTOM_CLASS_NAMES)
    cue_index = {name: index for index, name in enumerate(corrector.cue_names)}
    class_index = {name: index for index, name in enumerate(SYMPTOM_CLASS_NAMES)}
    assert (
        corrector.inhibitory_cue_mask[
            class_index["mouth_ulcer"], cue_index["redness_only"]
        ]
        > 0
    )
    assert (
        corrector.inhibitory_cue_mask[
            class_index["fin_deformity"], cue_index["lesion_only"]
        ]
        > 0
    )
    assert (
        corrector.inhibitory_cue_mask[
            class_index["fin_necrosis"], cue_index["shape_only"]
        ]
        > 0
    )


def test_full_forward_preserves_candidate_math_and_public_channel_orders() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    from progressive_candidate_correction.config import build_model

    model = build_model(METHOD_CONFIG, pretrained=False).eval()
    image = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        output = model(image, return_aux=True)

    assert output["R0"].shape == (1, 8, 96, 96)
    assert output["C"].shape == (1, 8, 96, 96)
    assert output["Z"].shape == (1, 8, 96, 96)
    assert output["Y"].shape == (1, 96, 96)
    assert torch.allclose(output["C"], output["N"] + output["A"], atol=1e-5, rtol=1e-5)

    gated_expected = output["signed_corrected_candidate_response_low"] + torch.sigmoid(
        output["gate_logits_low"]
    ) * output["residual_correction_delta_low"]
    assert torch.allclose(
        output["gated_corrected_candidate_response_low"],
        gated_expected,
        atol=1e-6,
        rtol=1e-5,
    )

    fusion = model.candidate_evidence_integrator
    corrected_expected = (
        fusion.initial_weight * output["initial_symptom_candidate_response_low"]
        + fusion.signed_weight * output["signed_corrected_candidate_response_low"]
        + fusion.gated_weight * output["gated_corrected_candidate_response_low"]
        + fusion.prototype_weight * output["symptom_prototype_similarity_low"]
        + output["candidate_part_logit_bias_low"]
        + 0.25 * torch.log(output["zone_map_low"].clamp_min(1e-6))
        + fusion.cue_weight * output["candidate_cue_score_low"]
        - output["candidate_unaffected_surface_penalty_low"]
        + 0.25 * output["candidate_integration_residual_low"]
    )
    assert torch.allclose(
        output["corrected_symptom_candidate_response_low"],
        corrected_expected,
        atol=1e-6,
        rtol=1e-5,
    )

    assert output["route_assignment_with_background_prob_low"].shape[1] == 5
    assert output["route_assignment_low"].shape[1] == 4
    internal_support = output["spatial_support_internal_prob_low"]
    expected_public_support = internal_support[:, [0, 7, 3, 2, 1, 6, 5, 4]]
    assert torch.allclose(output["spatial_support_low"], expected_public_support)
