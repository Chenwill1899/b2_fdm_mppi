"""Tests for the High-Level FDM subpackage.

These tests cover the public contract:

- schema tensor shapes and frame transforms
- synthetic generator determinism + label shape
- dataset builder round-trip
- model/trainer one-epoch overfit smoke
- rollout adapter + MPPI cost contract

If torch / numpy are missing the torch-dependent tests are skipped; the
schema constant/dataclass tests still run under pure stdlib so they are
useful in constrained CI environments.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

try:  # pragma: no cover - import guard for stdlib-only environments
    import pytest  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _PytestStub:
        class mark:
            @staticmethod
            def skipif(*_args, **_kwargs):
                def decorator(fn):
                    return fn
                return decorator

    pytest = _PytestStub()  # type: ignore


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _have_numpy() -> bool:
    return importlib.util.find_spec("numpy") is not None


def _have_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


needs_numpy = pytest.mark.skipif(
    not _have_numpy(), reason="numpy not available in this environment"
)
needs_torch = pytest.mark.skipif(
    not _have_torch(), reason="torch not available in this environment"
)


# ---------------------------------------------------------------------------
# Schema (stdlib-only checks first)
# ---------------------------------------------------------------------------

def test_schema_constants_have_expected_dims():
    from b2_fdm_mppi.high_level_fdm import schema as schema_mod

    assert schema_mod.STATE_DIM == 6
    assert schema_mod.CONTROL_DIM == 3
    assert schema_mod.POSE_PARAM_DIM == 4
    assert schema_mod.POSE_DELTA_DIM == 3
    assert schema_mod.NUM_RISK_CHANNELS == len(schema_mod.RISK_CHANNELS) == 4


def test_schema_serialization_round_trip():
    from b2_fdm_mppi.high_level_fdm.schema import HighLevelFdmSchema, RISK_CHANNELS

    original = HighLevelFdmSchema(
        history_length=8,
        horizon=12,
        map_channels=4,
        map_height=32,
        map_width=32,
        map_resolution=0.1,
    )
    payload = original.to_dict()
    round_tripped = HighLevelFdmSchema.from_dict(payload)
    assert round_tripped.history_length == 8
    assert round_tripped.horizon == 12
    assert round_tripped.map_channels == 4
    assert round_tripped.num_risk_channels() == len(RISK_CHANNELS)


@needs_numpy
def test_pose_delta_param_round_trip_preserves_angles():
    import numpy as np

    from b2_fdm_mppi.high_level_fdm.schema import (
        pose_delta_to_param,
        pose_param_to_delta,
    )

    deltas = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.2, -0.1, 0.5],
            [0.0, 0.0, -0.5],
            [0.5, 0.5, math.pi - 1e-3],
        ],
        dtype=np.float32,
    )
    param = pose_delta_to_param(deltas)
    recovered = pose_param_to_delta(param)
    assert param.shape == (4, 4)
    np.testing.assert_allclose(recovered[:, :2], deltas[:, :2], atol=1e-5)
    for idx in range(deltas.shape[0]):
        err = math.atan2(
            math.sin(float(recovered[idx, 2] - deltas[idx, 2])),
            math.cos(float(recovered[idx, 2] - deltas[idx, 2])),
        )
        assert abs(err) < 1e-4


@needs_numpy
def test_nominal_rollout_is_consistent_with_omni_b2():
    import numpy as np

    from b2_fdm_mppi.core.omni_b2 import OmniB2
    from b2_fdm_mppi.high_level_fdm.schema import nominal_rollout

    dt = 0.1
    horizon = 5
    controls = np.array(
        [
            [0.3, 0.1, 0.2],
            [0.3, 0.0, 0.0],
            [0.2, -0.1, -0.1],
            [0.0, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    robot = OmniB2(dt=dt, max_vx=1.0, max_vy=1.0, max_wz=2.0)
    init_state = np.zeros(6, dtype=np.float32)
    states = robot.rollout(init_state, controls)
    # robot.rollout returns the state *after* every step. The first element
    # is the initial state so the N deltas correspond to states[1:].
    expected = states[1:, :3]
    deltas = nominal_rollout(init_state[:3], controls, dt=dt)
    np.testing.assert_allclose(deltas[:, :2], expected[:, :2], atol=1e-5)
    np.testing.assert_allclose(deltas[:, 2], expected[:, 2], atol=1e-5)


# ---------------------------------------------------------------------------
# Synthetic generator + dataset builder
# ---------------------------------------------------------------------------

@needs_numpy
def test_synthetic_generator_is_deterministic_and_has_correct_shapes():
    from b2_fdm_mppi.high_level_fdm.schema import NUM_RISK_CHANNELS
    from b2_fdm_mppi.high_level_fdm.synthetic import (
        HighLevelFdmSyntheticGenerator,
        SyntheticConfig,
    )

    cfg = SyntheticConfig(
        history_length=4,
        horizon=6,
        map_height=16,
        map_width=16,
    )
    gen = HighLevelFdmSyntheticGenerator(cfg)

    sample_a = gen.generate_sample(42)
    sample_b = gen.generate_sample(42)
    sample_c = gen.generate_sample(43)

    assert sample_a.state_history.shape == (4, 6)
    assert sample_a.map_patch.shape == (len(cfg.map_channels), 16, 16)
    assert sample_a.control_sequence.shape == (6, 3)
    assert sample_a.pose_target.shape == (6, 4)
    assert sample_a.risk_target.shape == (6, NUM_RISK_CHANNELS)
    assert sample_a.risk_mask.shape == (6, NUM_RISK_CHANNELS)

    import numpy as np

    np.testing.assert_allclose(sample_a.state_history, sample_b.state_history)
    np.testing.assert_allclose(sample_a.pose_target, sample_b.pose_target)
    # Different seed should produce different trajectory data.
    assert not np.allclose(sample_a.pose_target, sample_c.pose_target)


@needs_numpy
def test_synthetic_dataset_builder_writes_splits(tmp_path):
    from b2_fdm_mppi.high_level_fdm.synthetic import SyntheticConfig
    from b2_fdm_mppi.high_level_fdm.synthetic_dataset_builder import (
        DatasetBuildSpec,
        build_synthetic_dataset,
    )

    summary = build_synthetic_dataset(
        output_dir=tmp_path,
        synthetic_config=SyntheticConfig(
            history_length=4, horizon=6, map_height=12, map_width=12
        ),
        spec=DatasetBuildSpec(
            train_samples=3, val_samples=2, test_samples=2, base_seed=7
        ),
    )
    assert summary["total_samples"] == 7
    for split in ("train", "val", "test"):
        assert (tmp_path / f"{split}.npz").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "build_summary.json").exists()


# ---------------------------------------------------------------------------
# Torch model, trainer, rollout
# ---------------------------------------------------------------------------

@needs_torch
def test_model_forward_output_shapes():
    import torch

    from b2_fdm_mppi.high_level_fdm.model import HighLevelFdm, HighLevelFdmModelConfig
    from b2_fdm_mppi.high_level_fdm.schema import (
        CONTROL_DIM,
        NUM_RISK_CHANNELS,
        POSE_PARAM_DIM,
        STATE_DIM,
    )

    cfg = HighLevelFdmModelConfig(
        history_length=4,
        horizon=6,
        map_channels=4,
        map_height=12,
        map_width=12,
        dt=0.1,
        num_risk_channels=NUM_RISK_CHANNELS,
    )
    model = HighLevelFdm(cfg)

    history = torch.randn(2, 4, STATE_DIM)
    map_patch = torch.randn(2, 4, 12, 12)
    controls = torch.randn(2, 6, CONTROL_DIM)

    pose_param, risk_logits = model(history, map_patch, controls)
    assert pose_param.shape == (2, 6, POSE_PARAM_DIM)
    assert risk_logits.shape == (2, 6, NUM_RISK_CHANNELS)
    assert torch.isfinite(pose_param).all()
    assert torch.isfinite(risk_logits).all()


@needs_torch
def test_train_overfit_beats_zero_residual_baseline(tmp_path):
    import torch

    from b2_fdm_mppi.high_level_fdm.losses import HighLevelFdmLossConfig
    from b2_fdm_mppi.high_level_fdm.synthetic import SyntheticConfig
    from b2_fdm_mppi.high_level_fdm.synthetic_dataset_builder import (
        DatasetBuildSpec,
        build_synthetic_dataset,
    )
    from b2_fdm_mppi.high_level_fdm.trainer import (
        HighLevelFdmTrainerConfig,
        train_high_level_fdm,
    )

    dataset_dir = tmp_path / "dataset"
    build_synthetic_dataset(
        output_dir=dataset_dir,
        synthetic_config=SyntheticConfig(
            history_length=4, horizon=6, map_height=12, map_width=12
        ),
        spec=DatasetBuildSpec(
            train_samples=16, val_samples=4, test_samples=4, base_seed=11
        ),
    )

    output_dir = tmp_path / "model"
    metrics = train_high_level_fdm(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        trainer_config=HighLevelFdmTrainerConfig(
            epochs=20,
            batch_size=8,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
        ),
        loss_config=HighLevelFdmLossConfig(
            lambda_pose=1.0, lambda_risk=0.2, step_weight_decay=None
        ),
        model_overrides={"d_hist": 32, "d_map": 32, "d_ctx": 64, "d_hidden": 64},
    )
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "schema.json").exists()

    val = metrics["val_metrics"]
    assert val["pose_xy_abs_err"] >= 0.0
    assert val["pose_xy_abs_err"] <= val["pose_zero_residual_baseline"] + 1e-6, (
        "trained model should at least match the zero-residual nominal baseline"
    )
    assert 0.0 <= val["risk_brier"] <= 1.0


@needs_torch
def test_rollout_batch_and_cost_contract(tmp_path):
    import numpy as np
    import torch

    from b2_fdm_mppi.high_level_fdm.model import HighLevelFdm, HighLevelFdmModelConfig
    from b2_fdm_mppi.high_level_fdm.rollout import (
        HighLevelFdmRollout,
        apply_pose_delta_to_world,
        high_level_fdm_cost,
    )
    from b2_fdm_mppi.high_level_fdm.schema import HighLevelFdmSchema

    schema = HighLevelFdmSchema(
        history_length=4,
        horizon=5,
        map_channels=4,
        map_height=12,
        map_width=12,
        map_resolution=0.1,
    )
    cfg = HighLevelFdmModelConfig.from_schema(schema, dt=0.1)
    model = HighLevelFdm(cfg)
    rollout = HighLevelFdmRollout(model=model, schema=schema, device="cpu")

    history = torch.zeros(schema.history_length, 6)
    map_patch = torch.zeros(schema.map_channels, schema.map_height, schema.map_width)
    controls = torch.zeros(8, schema.horizon, 3)
    controls[:, :, 0] = torch.linspace(-0.2, 0.4, 8).view(-1, 1)

    result = rollout.predict_batch(history, map_patch, controls)
    assert result.pose_param.shape == (8, schema.horizon, 4)
    assert result.pose_delta.shape == (8, schema.horizon, 3)
    assert result.risk_prob.shape == (8, schema.horizon, schema.num_risk_channels())
    assert torch.all(result.risk_prob >= 0.0) and torch.all(result.risk_prob <= 1.0)

    cost = high_level_fdm_cost(
        risk_prob=result.risk_prob,
        pose_delta=result.pose_delta,
        risk_channel_weights=[1.0, 1.0, 0.5, 2.0],
        goal_xy_body=[0.3, 0.0],
        goal_weight=1.0,
        step_weight_decay=0.95,
    )
    assert cost.shape == (8,)
    assert torch.isfinite(cost).all()

    # World composition should preserve shapes and finite values.
    current = np.zeros(6, dtype=np.float32)
    current[2] = 0.3
    world = apply_pose_delta_to_world(current, result.pose_delta.detach().numpy())
    assert world.shape == (8, schema.horizon, 3)
    assert np.all(np.isfinite(world))


# ---------------------------------------------------------------------------
# CLI parser still loads (no torch required)
# ---------------------------------------------------------------------------

def test_high_fdm_cli_subcommands_are_registered():
    from b2_fdm_mppi.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["high-fdm", "synth", "--output", "/tmp/out"])
    assert ns.handler == "high_fdm_synth"
    assert ns.output == "/tmp/out"

    ns = parser.parse_args(
        [
            "high-fdm",
            "train",
            "--dataset",
            "/tmp/ds",
            "--output",
            "/tmp/model",
        ]
    )
    assert ns.handler == "high_fdm_train"

    ns = parser.parse_args(
        [
            "high-fdm",
            "rollout-smoke",
            "--dataset",
            "/tmp/ds",
            "--checkpoint",
            "/tmp/ckpt.pt",
        ]
    )
    assert ns.handler == "high_fdm_rollout_smoke"
