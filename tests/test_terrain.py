import numpy as np

from b2_fdm_mppi.core.terrain import TerrainField


def test_terrain_feature_is_deterministic():
    terrain = TerrainField(enabled=True)

    first = terrain.feature(1.0, 2.0)
    second = terrain.feature(1.0, 2.0)
    risk = terrain.risk_cost(1.0, 2.0, features=first)

    assert np.allclose(first, second)
    assert risk >= 0.0


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
