import json

import numpy as np
import pandas as pd
import pytest

from b2_fdm_mppi.data.oracle_episode import build_episode_npz


def test_build_episode_npz_converts_runner_outputs_to_transitions(tmp_path):
    results_path = tmp_path / "results"
    results_path.mkdir()
    pd.DataFrame(
        [
            {"step": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0},
            {"step": 1, "x": 0.1, "y": 0.0, "theta": 0.0, "vx": 1.0, "vy": 0.0, "wz": 0.0},
            {"step": 2, "x": 0.2, "y": 0.1, "theta": 0.1, "vx": 1.0, "vy": 0.5, "wz": 0.1},
            {"step": 3, "x": 0.3, "y": 0.1, "theta": 0.1, "vx": 0.0, "vy": 0.0, "wz": 0.0},
        ]
    ).to_csv(results_path / "trajectory.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": 0,
                "cmd_vx": 1.0,
                "cmd_vy": 0.0,
                "cmd_wz": 0.0,
                "real_vx": 0.8,
                "real_vy": 0.1,
                "real_wz": 0.0,
                "exec_du_vx": -0.2,
                "exec_du_vy": 0.1,
                "exec_du_wz": 0.0,
                "oracle_du_vx": -0.1,
                "oracle_du_vy": 0.05,
                "oracle_du_wz": 0.0,
            },
            {
                "step": 1,
                "cmd_vx": 0.9,
                "cmd_vy": 0.2,
                "cmd_wz": 0.1,
                "real_vx": 0.7,
                "real_vy": 0.25,
                "real_wz": 0.05,
                "exec_du_vx": -0.2,
                "exec_du_vy": 0.05,
                "exec_du_wz": -0.05,
                "oracle_du_vx": -0.15,
                "oracle_du_vy": 0.03,
                "oracle_du_wz": -0.02,
            },
            {
                "step": 2,
                "cmd_vx": 0.0,
                "cmd_vy": 0.0,
                "cmd_wz": 0.0,
                "real_vx": 0.0,
                "real_vy": 0.0,
                "real_wz": 0.0,
                "exec_du_vx": 0.0,
                "exec_du_vy": 0.0,
                "exec_du_wz": 0.0,
                "oracle_du_vx": 0.0,
                "oracle_du_vy": 0.0,
                "oracle_du_wz": 0.0,
            },
        ]
    ).to_csv(results_path / "residuals.csv", index=False)
    pd.DataFrame(
        [
            {"step": 0, "slope_f": 0.1, "slope_l": 0.2, "roughness": 0.3, "friction": 0.8, "risk_cost": 0.5},
            {"step": 1, "slope_f": 0.2, "slope_l": 0.3, "roughness": 0.4, "friction": 0.7, "risk_cost": 0.6},
            {"step": 2, "slope_f": 0.3, "slope_l": 0.4, "roughness": 0.5, "friction": 0.6, "risk_cost": 0.7},
        ]
    ).to_csv(results_path / "terrain.csv", index=False)
    (results_path / "summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "failed": False,
                "start_goal_distance": 8.5,
                "final_distance": 0.4,
                "min_obstacle_clearance": 1.2,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "episode_000007.npz"

    metadata = build_episode_npz(results_path, episode_id=7, output_path=output_path)

    assert output_path.exists()
    data = np.load(output_path)
    transition_count = 3
    assert data["states"].shape == (transition_count, 6)
    assert data["next_states"].shape == (transition_count, 6)
    assert data["cmd_controls"].shape == (transition_count, 3)
    assert data["real_controls"].shape == (transition_count, 3)
    assert data["exec_residuals"].shape == (transition_count, 3)
    assert data["oracle_residuals"].shape == (transition_count, 3)
    assert data["terrain_features"].shape == (transition_count, 4)
    assert data["terrain_risk"].shape == (transition_count,)
    assert data["episode_ids"].tolist() == [7, 7, 7]
    assert data["steps"].tolist() == [0, 1, 2]
    assert data["success"].shape == ()
    assert bool(data["success"]) is True
    assert bool(data["failed"]) is False
    assert np.allclose(data["exec_residuals"], data["real_controls"] - data["cmd_controls"])
    assert float(data["start_goal_distance"]) == pytest.approx(8.5)
    assert float(data["final_distance"]) == pytest.approx(0.4)
    assert float(data["min_obstacle_clearance"]) == pytest.approx(1.2)
    assert metadata["episode_id"] == 7
    assert metadata["num_transitions"] == transition_count
    assert metadata["success"] is True
    assert metadata["failed"] is False
    assert metadata["results_path"] == str(results_path)
    assert metadata["output_path"] == str(output_path)


def test_build_episode_npz_uses_residual_rows_as_transition_count(tmp_path):
    results_path = tmp_path / "results"
    results_path.mkdir()
    pd.DataFrame(
        [
            {"step": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0},
            {"step": 1, "x": 0.1, "y": 0.0, "theta": 0.0, "vx": 1.0, "vy": 0.0, "wz": 0.0},
            {"step": 2, "x": 0.2, "y": 0.1, "theta": 0.1, "vx": 1.0, "vy": 0.5, "wz": 0.1},
            {"step": 3, "x": 0.3, "y": 0.2, "theta": 0.1, "vx": 1.0, "vy": 0.5, "wz": 0.1},
        ]
    ).to_csv(results_path / "trajectory.csv", index=False)
    residual_rows = [
        {
            "step": idx,
            "cmd_vx": 1.0,
            "cmd_vy": 0.0,
            "cmd_wz": 0.0,
            "real_vx": 0.8,
            "real_vy": 0.1,
            "real_wz": 0.0,
            "exec_du_vx": -0.2,
            "exec_du_vy": 0.1,
            "exec_du_wz": 0.0,
            "oracle_du_vx": -0.1,
            "oracle_du_vy": 0.05,
            "oracle_du_wz": 0.0,
        }
        for idx in range(2)
    ]
    pd.DataFrame(residual_rows).to_csv(results_path / "residuals.csv", index=False)
    pd.DataFrame(
        [
            {"step": idx, "slope_f": 0.1, "slope_l": 0.2, "roughness": 0.3, "friction": 0.8, "risk_cost": 0.5}
            for idx in range(2)
        ]
    ).to_csv(results_path / "terrain.csv", index=False)
    (results_path / "summary.json").write_text(
        json.dumps({"success": True, "failed": False}),
        encoding="utf-8",
    )

    metadata = build_episode_npz(results_path, episode_id=9, output_path=tmp_path / "episode_000009.npz")

    assert metadata["num_transitions"] == 2
