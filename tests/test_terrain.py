import numpy as np

from b2_fdm_mppi.core.terrain import TerrainField


def test_terrain_feature_is_deterministic():
    terrain = TerrainField(enabled=True)

    first = terrain.feature(1.0, 2.0)
    second = terrain.feature(1.0, 2.0)
    risk = terrain.risk_cost(1.0, 2.0, features=first)

    assert np.allclose(first, second)
    assert risk >= 0.0


def test_terrain_noise_is_reproducible_for_same_seed():
    config = {
        "enabled": True,
        "noise_enabled": True,
        "noise_seed": 123,
        "noise_grid_size": [8, 8],
        "noise_scale": 0.35,
    }
    first = TerrainField.from_config(config)
    second = TerrainField.from_config(config)

    assert np.allclose(first.feature(4.2, 7.1), second.feature(4.2, 7.1))


def test_terrain_noise_seed_changes_features():
    config = {
        "enabled": True,
        "noise_enabled": True,
        "noise_grid_size": [8, 8],
        "noise_scale": 0.35,
    }
    first = TerrainField.from_config({**config, "noise_seed": 1})
    second = TerrainField.from_config({**config, "noise_seed": 2})

    assert not np.allclose(first.feature(4.2, 7.1), second.feature(4.2, 7.1))


def test_terrain_noise_disabled_preserves_analytic_features():
    config = {
        "enabled": True,
        "noise_enabled": False,
        "noise_seed": 123,
        "noise_grid_size": [8, 8],
        "noise_scale": 0.35,
    }
    terrain = TerrainField.from_config(config)
    legacy = TerrainField(enabled=True)

    assert np.allclose(terrain.feature(4.2, 7.1), legacy.feature(4.2, 7.1))


def test_terrain_disabled_returns_zero_features():
    terrain = TerrainField(enabled=False)

    features = terrain.feature(1.5, -0.3)
    risk = terrain.risk_cost(1.5, -0.3, features=features)

    assert features.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert risk == 0.0


def test_terrain_goal_relief_smoothly_reduces_goal_area_risk():
    base = TerrainField(enabled=True)
    terrain = TerrainField(
        enabled=True,
        goal_relief={
            "enabled": True,
            "center": [18.0, 0.0],
            "sigma": [2.0, 1.2],
            "strength": 0.75,
            "floor": 0.25,
        },
    )

    base_goal_features = base.feature(18.0, 0.0)
    goal_features = terrain.feature(18.0, 0.0)
    edge_features = terrain.feature(20.0, 0.0)
    far_features = terrain.feature(10.0, -3.0)
    base_goal_risk = base.risk_cost(18.0, 0.0, features=base_goal_features)
    goal_risk = terrain.risk_cost(18.0, 0.0, features=goal_features)
    edge_risk = terrain.risk_cost(20.0, 0.0, features=edge_features)
    far_risk = terrain.risk_cost(10.0, -3.0, features=far_features)

    assert goal_risk > 0.0
    assert goal_risk < base_goal_risk
    assert goal_risk < edge_risk
    assert far_risk > goal_risk


def test_terrain_goal_relief_reduces_patch_risk_at_goal():
    patch = {
        "name": "goal_risk_patch",
        "type": "ellipse",
        "center": [18.0, 0.0],
        "size": [4.0, 2.0],
        "edge_width": 0.5,
        "roughness_delta": 0.7,
        "friction_delta": -0.4,
    }
    base = TerrainField(
        enabled=True,
        slope_scale=0.0,
        roughness_scale=0.0,
        friction_base=0.8,
        friction_slope_scale=0.0,
        friction_roughness_scale=0.0,
        patches=[patch],
    )
    relieved = TerrainField(
        enabled=True,
        slope_scale=0.0,
        roughness_scale=0.0,
        friction_base=0.8,
        friction_slope_scale=0.0,
        friction_roughness_scale=0.0,
        goal_relief={
            "enabled": True,
            "center": [18.0, 0.0],
            "sigma": [2.0, 1.2],
            "strength": 0.75,
            "floor": 0.25,
        },
        patches=[patch],
    )

    base_goal_risk = base.risk_cost(18.0, 0.0)
    relieved_goal_risk = relieved.risk_cost(18.0, 0.0)
    relieved_edge_risk = relieved.risk_cost(20.0, 0.0)

    assert relieved_goal_risk < base_goal_risk
    assert relieved_goal_risk < relieved_edge_risk


def test_terrain_empty_patches_preserve_legacy_features():
    config = {
        "enabled": True,
        "friction_base": 0.72,
        "slope_scale": 0.10,
        "roughness_scale": 0.25,
        "friction_slope_scale": 0.18,
        "friction_roughness_scale": 0.10,
        "patches": [],
    }
    terrain = TerrainField.from_config(config)
    legacy = TerrainField.from_config({key: value for key, value in config.items() if key != "patches"})
    point = (7.2, -1.4)

    assert np.allclose(terrain.feature(*point), legacy.feature(*point))
    assert terrain.risk_cost(*point) == legacy.risk_cost(*point)


def test_terrain_ellipse_patch_changes_physical_features_with_smooth_edge():
    terrain = TerrainField(
        enabled=True,
        slope_scale=0.0,
        roughness_scale=0.0,
        friction_base=0.8,
        friction_slope_scale=0.0,
        friction_roughness_scale=0.0,
        patches=[
            {
                "name": "rough_low_friction_island",
                "type": "ellipse",
                "center": [10.0, 0.0],
                "size": [4.0, 2.0],
                "edge_width": 1.0,
                "roughness_delta": 0.5,
                "friction_delta": -0.3,
            }
        ],
    )

    center = terrain.feature(10.0, 0.0)
    edge = terrain.feature(12.4, 0.0)
    outside = terrain.feature(14.0, 0.0)

    assert center[2] > edge[2] > outside[2]
    assert center[3] < edge[3] < outside[3]
    assert terrain.risk_cost(10.0, 0.0, features=center) > terrain.risk_cost(14.0, 0.0, features=outside)


def test_terrain_finite_band_patch_does_not_extend_infinitely():
    terrain = TerrainField(
        enabled=True,
        slope_scale=0.0,
        roughness_scale=0.0,
        friction_base=0.8,
        friction_slope_scale=0.0,
        friction_roughness_scale=0.0,
        patches=[
            {
                "name": "finite_crossing_band",
                "type": "band",
                "center": [10.0, 0.0],
                "angle": 90.0,
                "size": [8.0, 2.0],
                "edge_width": 0.5,
                "roughness_delta": 0.6,
                "friction_delta": -0.25,
            }
        ],
    )

    inside = terrain.feature(10.0, 0.0)
    far_along_band = terrain.feature(10.0, 6.0)
    far_across_band = terrain.feature(13.0, 0.0)

    assert inside[2] > far_along_band[2]
    assert inside[2] > far_across_band[2]
    assert np.allclose(far_along_band, far_across_band)
