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
