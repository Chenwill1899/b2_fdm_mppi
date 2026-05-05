import numpy as np
import pytest
from b2_fdm_mppi.simulation.random_terrain import RandomTerrainGenerator
from b2_fdm_mppi.core.terrain import TerrainField


def test_generator_returns_terrain_field():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(2, 3))
    terrain = gen.generate(seed=42)
    assert isinstance(terrain, TerrainField)
    assert terrain.enabled


def test_generator_reproducible():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(2, 3))
    t1 = gen.generate(seed=123)
    t2 = gen.generate(seed=123)
    # Same seed should produce same patches
    assert len(t1.patches) == len(t2.patches)
    for p1, p2 in zip(t1.patches, t2.patches):
        assert p1["type"] == p2["type"]
        assert np.allclose(p1["center"], p2["center"])
        assert np.isclose(p1["size"][0], p2["size"][0])
        assert np.isclose(p1["size"][1], p2["size"][1])
        assert np.isclose(p1["angle"], p2["angle"])
        assert np.isclose(p1["edge_width"], p2["edge_width"])
        assert np.isclose(p1["slope_f_delta"], p2["slope_f_delta"])
        assert np.isclose(p1["slope_l_delta"], p2["slope_l_delta"])
        assert np.isclose(p1["roughness_delta"], p2["roughness_delta"])
        assert np.isclose(p1["friction_delta"], p2["friction_delta"])


def test_generator_different_seeds():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(3, 5))
    t1 = gen.generate(seed=1)
    t2 = gen.generate(seed=2)
    # Different seeds should likely produce different patch counts or positions
    any_diff = False
    if len(t1.patches) != len(t2.patches):
        any_diff = True
    else:
        for p1, p2 in zip(t1.patches, t2.patches):
            if not np.allclose(p1["center"], p2["center"]):
                any_diff = True
                break
            if not np.isclose(p1["size"][0], p2["size"][0]):
                any_diff = True
                break
            if not np.isclose(p1["size"][1], p2["size"][1]):
                any_diff = True
                break
            if not np.isclose(p1["angle"], p2["angle"]):
                any_diff = True
                break
            if not np.isclose(p1["slope_f_delta"], p2["slope_f_delta"]):
                any_diff = True
                break
            if not np.isclose(p1["slope_l_delta"], p2["slope_l_delta"]):
                any_diff = True
                break
            if not np.isclose(p1["roughness_delta"], p2["roughness_delta"]):
                any_diff = True
                break
            if not np.isclose(p1["friction_delta"], p2["friction_delta"]):
                any_diff = True
                break
    assert any_diff


def test_patches_within_bounds():
    bounds = (-20, 20, -15, 15)
    gen = RandomTerrainGenerator(map_bounds=bounds, num_patches_range=(5, 5))
    terrain = gen.generate(seed=999)
    for patch in terrain.patches:
        cx, cy = patch["center"]
        assert bounds[0] <= cx <= bounds[1]
        assert bounds[2] <= cy <= bounds[3]


def test_generator_creates_risk_variation():
    gen = RandomTerrainGenerator(map_bounds=(-10, 10, -10, 10), num_patches_range=(3, 5))
    terrain = gen.generate(seed=77)
    # Sample risk at a few points; should see variation
    risks = [terrain.risk_cost(x, y) for x in [-5, 0, 5] for y in [-5, 0, 5]]
    assert max(risks) > min(risks)
