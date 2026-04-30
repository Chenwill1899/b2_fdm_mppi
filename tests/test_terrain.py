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


def test_terrain_safe_zone_reduces_goal_area_risk():
    terrain = TerrainField(
        enabled=True,
        safe_zones=[{"center": [18.0, 0.0], "radius": 1.5, "transition": 1.0}],
    )

    goal_features = terrain.feature(18.0, 0.0)
    far_features = terrain.feature(10.0, -3.0)
    goal_risk = terrain.risk_cost(18.0, 0.0, features=goal_features)
    far_risk = terrain.risk_cost(10.0, -3.0, features=far_features)

    assert goal_features.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert goal_risk == 0.0
    assert far_risk > goal_risk
