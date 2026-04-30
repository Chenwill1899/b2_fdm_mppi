import numpy as np
import pytest

from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.controllers.mppi_omni_numpy import MppiOmniNumpy
from tests.cuda_test_utils import require_cuda_device


pycuda = require_cuda_device()


def make_config():
    config = load_config("config/b2_omni_nominal.yaml")
    config["simulation"]["time_horizon"] = 0.3
    config["mppi"]["num_trajectories"] = 4
    config["mppi"]["draw_num_traj"] = 2
    config["mppi"]["cbf_weight"] = 0.0
    return config


def test_cuda_omni_cost_matches_numpy_without_cbf():
    from b2_fdm_mppi.controllers.mppi_omni_cuda import MppiOmniCuda

    config = make_config()
    numpy_controller = MppiOmniNumpy.from_config(config, seed=1)
    cuda_controller = MppiOmniCuda.from_config(config, seed=1)
    state = np.array([0.1, -0.1, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([1.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[0.7, 0.15, 0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    controls = np.array(
        [
            [[0.2, 0.0, 0.1], [0.3, 0.1, 0.0], [0.4, -0.1, -0.1]],
            [[0.0, 0.2, 0.0], [0.1, 0.1, 0.1], [0.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    numpy_costs = numpy_controller.trajectory_cost_batch(state, controls, goal, obstacles)
    cuda_costs = cuda_controller.trajectory_cost_batch(state, controls, goal, obstacles)

    assert cuda_costs == pytest.approx(numpy_costs, rel=1e-4, abs=1e-4)


def test_cuda_omni_cbf_cost_penalizes_decreasing_barrier():
    from b2_fdm_mppi.controllers.mppi_omni_cuda import MppiOmniCuda

    config = make_config()
    config["mppi"]["cbf_weight"] = 500.0
    config["cbf"]["dcbf_alpha"] = 0.1
    controller = MppiOmniCuda.from_config(config, seed=1)
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[0.45, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    controls = np.array(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0]],
        ],
        dtype=np.float32,
    )

    costs = controller.trajectory_cost_batch(state, controls, goal, obstacles)

    assert costs[0] > costs[1]


def test_cuda_omni_rcbf_penalizes_approaching_obstacle_more_than_static():
    from b2_fdm_mppi.controllers.mppi_omni_cuda import MppiOmniCuda

    config = make_config()
    config["mppi"]["cbf_weight"] = 500.0
    config["mppi"]["obstacle_weight"] = 0.0
    config["mppi"]["control_weight"] = 0.0
    config["mppi"]["smooth_weight"] = 0.0
    config["cbf"]["enabled"] = True
    config["cbf"]["type"] = 1
    config["cbf"]["atau"] = 0.2
    controller = MppiOmniCuda.from_config(config, seed=1)
    state = np.zeros(6, dtype=np.float32)
    goal = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    controls = np.array(
        [
            [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    static_obstacle = np.array([[1.4, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    approaching_obstacle = np.array([[1.4, 0.0, 0.05, 0.0, 0.0, -2.0, 0.0]], dtype=np.float32)

    static_cost = controller.trajectory_cost_batch(state, controls, goal, static_obstacle)[0]
    approaching_cost = controller.trajectory_cost_batch(state, controls, goal, approaching_obstacle)[0]

    assert approaching_cost > static_cost


def test_cuda_omni_from_config_disables_cbf_when_configured():
    from b2_fdm_mppi.controllers.mppi_omni_cuda import MppiOmniCuda

    config = make_config()
    config["mppi"]["cbf_weight"] = 500.0
    config["cbf"]["enabled"] = False

    controller = MppiOmniCuda.from_config(config, seed=1)

    assert controller.cbf_weight == 0.0
