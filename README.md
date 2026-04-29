# b2_fdm_mppi

ROS 2 Humble `ament_python` package for the MPPI/CBF internal simulation.

This refactor keeps the existing PyCUDA MPPI controller logic and exposes it
through a ROS 2 package layout. The v1 node runs the same internal simulation
loop; it does not subscribe to `/odom`, `/goal_pose`, or publish `/cmd_vel`.

## Build

From a ROS 2 Humble workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select b2_fdm_mppi
source install/setup.bash
```

Python dependencies used by the simulation include `numpy`, `scipy`, `pandas`,
`matplotlib`, `seaborn`, `PyYAML`, `jinja2`, `cvxpy`, `casadi`, and `pycuda`.
The real MPPI controller requires an NVIDIA CUDA/PyCUDA runtime. Unit tests use
a fake controller for non-GPU coverage.

## Run

```bash
ros2 launch b2_fdm_mppi fdm_mppi.launch.py
```

Or pass a specific config:

```bash
ros2 launch b2_fdm_mppi fdm_mppi.launch.py config_file:=/path/to/fdm_mppi.yaml
```

The default config is `config/fdm_mppi.yaml`. Results are written under:

```text
results/sim_results/<timestamp>/
```

## Test

```bash
python3 -m pytest -q
```

The package keeps top-level imports lightweight: importing `b2_fdm_mppi` does
not import PyCUDA. CUDA-backed controller modules import PyCUDA only when the
real controller is loaded.
