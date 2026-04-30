from pathlib import Path

import pandas as pd
import yaml

from b2_fdm_mppi.visualization.oracle_viewer import _resolve_goal_xy, plot_oracle_diagnostics


def test_plot_oracle_diagnostics_writes_nonempty_png(tmp_path: Path):
    pd.DataFrame(
        {
            "step": [0, 1, 2],
            "x": [0.0, 0.5, 1.0],
            "y": [0.0, 0.1, 0.2],
            "theta": [0.0, 0.0, 0.0],
            "vx": [0.0, 0.1, 0.2],
            "vy": [0.0, 0.0, 0.0],
            "wz": [0.0, 0.0, 0.0],
            "x_des": [1.0, 1.0, 1.0],
            "y_des": [0.0, 0.0, 0.0],
            "theta_des": [0.0, 0.0, 0.0],
        }
    ).to_csv(tmp_path / "trajectory.csv", index=False)
    pd.DataFrame(
        {
            "step": [0, 1, 2],
            "vx_cmd": [0.1, 0.2, 0.3],
            "vy_cmd": [0.0, 0.1, 0.0],
            "wz_cmd": [0.0, 0.0, 0.1],
        }
    ).to_csv(tmp_path / "controls.csv", index=False)
    pd.DataFrame(
        {
            "step": [0, 1, 2],
            "vx_cmd": [0.1, 0.2, 0.3],
            "vy_cmd": [0.0, 0.1, 0.0],
            "wz_cmd": [0.0, 0.0, 0.1],
        }
    ).to_csv(tmp_path / "raw_controls.csv", index=False)
    pd.DataFrame(
        {
            "step": [0, 1, 2],
            "cmd_vx": [0.1, 0.2, 0.3],
            "cmd_vy": [0.0, 0.1, 0.0],
            "cmd_wz": [0.0, 0.0, 0.1],
            "real_vx": [0.08, 0.15, 0.25],
            "real_vy": [0.0, 0.05, 0.0],
            "real_wz": [0.0, 0.0, 0.05],
            "du_vx": [-0.02, -0.05, -0.05],
            "du_vy": [0.0, -0.05, 0.0],
            "du_wz": [0.0, 0.0, -0.05],
            "du_norm": [0.02, 0.071, 0.071],
        }
    ).to_csv(tmp_path / "residuals.csv", index=False)
    pd.DataFrame(
        {
            "step": [0, 1, 2],
            "x": [0.0, 0.5, 1.0],
            "y": [0.0, 0.1, 0.2],
            "slope_f": [0.01, 0.02, 0.03],
            "slope_l": [0.02, 0.01, 0.0],
            "roughness": [0.1, 0.2, 0.15],
            "friction": [0.8, 0.75, 0.7],
            "risk_cost": [0.1, 0.2, 0.3],
        }
    ).to_csv(tmp_path / "terrain.csv", index=False)
    config = {
        "terrain": {"enabled": True},
        "obstacles": {"virtual": [[0.5, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0]]},
        "robot": {"radius": 0.4, "safety_dist": 0.2},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    plot_oracle_diagnostics(tmp_path, config)

    output = tmp_path / "oracle_diagnostics.png"
    assert output.exists()
    assert output.stat().st_size > 0


def test_oracle_viewer_resolves_config_goal_not_trajectory_end():
    trajectory = pd.DataFrame(
        {
            "x": [0.0, 18.4],
            "y": [0.0, -1.3],
            "x_des": [18.0, 18.0],
            "y_des": [0.0, 0.0],
        }
    )
    config = {"simulation": {"goal": [18.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}

    goal_x, goal_y = _resolve_goal_xy(trajectory, config)

    assert (goal_x, goal_y) == (18.0, 0.0)


def test_plot_oracle_diagnostics_supports_configured_map_size(tmp_path: Path):
    pd.DataFrame(
        {
            "step": [0, 1],
            "x": [5.0, 6.0],
            "y": [50.0, 50.2],
            "theta": [0.0, 0.0],
            "vx": [0.0, 0.1],
            "vy": [0.0, 0.0],
            "wz": [0.0, 0.0],
            "x_des": [95.0, 95.0],
            "y_des": [50.0, 50.0],
            "theta_des": [0.0, 0.0],
        }
    ).to_csv(tmp_path / "trajectory.csv", index=False)
    pd.DataFrame({"step": [0, 1], "vx_cmd": [0.1, 0.1], "vy_cmd": [0.0, 0.0], "wz_cmd": [0.0, 0.0]}).to_csv(
        tmp_path / "controls.csv", index=False
    )
    pd.DataFrame({"step": [0, 1], "vx_cmd": [0.1, 0.1], "vy_cmd": [0.0, 0.0], "wz_cmd": [0.0, 0.0]}).to_csv(
        tmp_path / "raw_controls.csv", index=False
    )
    pd.DataFrame(
        {
            "step": [0, 1],
            "cmd_vx": [0.1, 0.1],
            "cmd_vy": [0.0, 0.0],
            "cmd_wz": [0.0, 0.0],
            "real_vx": [0.08, 0.09],
            "real_vy": [0.0, 0.0],
            "real_wz": [0.0, 0.0],
            "du_vx": [-0.02, -0.01],
            "du_vy": [0.0, 0.0],
            "du_wz": [0.0, 0.0],
            "du_norm": [0.02, 0.01],
        }
    ).to_csv(tmp_path / "residuals.csv", index=False)
    pd.DataFrame(
        {
            "step": [0, 1],
            "x": [5.0, 6.0],
            "y": [50.0, 50.2],
            "slope_f": [0.0, 0.01],
            "slope_l": [0.0, 0.01],
            "roughness": [0.1, 0.1],
            "friction": [0.8, 0.8],
            "risk_cost": [0.1, 0.1],
        }
    ).to_csv(tmp_path / "terrain.csv", index=False)
    config = {
        "simulation": {"map_origin": [0.0, 0.0], "map_size": [100.0, 100.0], "goal": [95.0, 50.0, 0, 0, 0, 0]},
        "visualization": {"terrain_grid_resolution": 20},
        "terrain": {"enabled": True},
        "obstacles": {"virtual": [[40.0, 50.0, 1.0, 0.0, 0.0, 0.0, 0.0]]},
        "robot": {"radius": 0.6, "safety_dist": 0.5},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    plot_oracle_diagnostics(tmp_path, config)

    output = tmp_path / "oracle_diagnostics.png"
    assert output.exists()
    assert output.stat().st_size > 0
