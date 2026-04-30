import numpy as np

from b2_fdm_mppi.core.omni_b2 import OmniB2
from b2_fdm_mppi.core.residual_world import ResidualWorld
from b2_fdm_mppi.core.terrain import TerrainField


def test_residual_world_applies_delta_and_clips():
    model = OmniB2(dt=0.1, max_vx=1.0, max_vy=0.5, max_wz=0.5)
    terrain = TerrainField(enabled=True)
    world = ResidualWorld(
        model,
        terrain,
        enabled=True,
        alpha=1.0,
        residual_scale=1.0,
        noise_std=0.0,
        max_residual_ratio=0.2,
        seed=1,
    )
    state = np.zeros(6, dtype=np.float32)
    u_cmd = np.array([1.0, 0.2, 0.1], dtype=np.float32)

    next_state, u_real, delta, _features = world.update_state(state, u_cmd)

    max_delta = np.array([model.max_vx, model.max_vy, model.max_wz]) * 0.2
    assert np.any(np.abs(delta) > 0.0)
    assert np.all(np.abs(delta) <= max_delta + 1e-6)
    assert np.any(u_real != u_cmd)
    assert next_state.shape == (6,)


def test_residual_world_disabled_passes_through():
    model = OmniB2(dt=0.1, max_vx=1.0, max_vy=0.5, max_wz=0.5)
    terrain = TerrainField(enabled=True)
    world = ResidualWorld(
        model,
        terrain,
        enabled=False,
        alpha=1.0,
        residual_scale=1.0,
        noise_std=0.0,
        max_residual_ratio=0.2,
        seed=1,
    )
    state = np.zeros(6, dtype=np.float32)
    u_cmd = np.array([0.2, -0.1, 0.0], dtype=np.float32)

    next_state, u_real, delta, _features = world.update_state(state, u_cmd)

    assert delta.tolist() == [0.0, 0.0, 0.0]
    assert u_real.tolist() == u_cmd.tolist()
    assert next_state.shape == (6,)
