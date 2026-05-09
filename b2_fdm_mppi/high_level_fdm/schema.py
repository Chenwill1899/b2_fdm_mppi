"""Schema, dataclasses and frame transforms for the High-Level FDM.

All tensors here are numpy arrays. A torch-native view lives in
``b2_fdm_mppi.high_level_fdm.model`` and ``rollout``; those import this
module for constants and ordering.

Per-sample layout produced by the data pipeline:

    state_history     float32 (H, STATE_DIM)              body frame of t
    map_patch         float32 (MAP_CHANNELS, Gh, Gw)      aligned to t
    control_sequence  float32 (N, CONTROL_DIM)            body frame
    pose_target       float32 (N, POSE_PARAM_DIM)         [dx, dy, sin, cos]
    risk_target       float32 (N, NUM_RISK_CHANNELS)      in [0, 1]
    risk_mask         float32 (N, NUM_RISK_CHANNELS)      1.0 where valid

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # numpy is only an import-time dep for the typed helpers; tests that do
    # not load data never exercise it.
    import numpy as np  # type: ignore

    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - executed only when numpy missing
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

STATE_DIM = 6              # [x, y, theta, vx, vy, wz]
CONTROL_DIM = 3            # [vx, vy, wz]
POSE_DELTA_DIM = 3         # [dx, dy, dtheta]
POSE_PARAM_DIM = 4         # [dx, dy, sin_dth, cos_dth] - network target form

STATE_NAMES = ("x", "y", "theta", "vx", "vy", "wz")
CONTROL_NAMES = ("vx", "vy", "wz")
POSE_DELTA_NAMES = ("dx", "dy", "dtheta")
POSE_PARAM_NAMES = ("dx", "dy", "sin_dtheta", "cos_dtheta")

# Default risk channels. The model head width is len(RISK_CHANNELS).
RISK_CHANNELS: tuple[str, ...] = (
    "collision",
    "high_cost",
    "stuck",
    "untraversable",
)
NUM_RISK_CHANNELS = len(RISK_CHANNELS)

# Default map patch channels. Kept flexible; the trainer reads the actual
# channel count from the dataset manifest so real-robot topics with a
# different channel count drop in without code changes.
DEFAULT_MAP_CHANNELS: tuple[str, ...] = (
    "occupancy",
    "traversability_risk",
    "height_mean",
    "height_std",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HighLevelFdmSchema:
    """Runtime description of tensor shapes for a given dataset/model."""

    history_length: int
    horizon: int
    map_channels: int
    map_height: int
    map_width: int
    map_resolution: float
    state_dim: int = STATE_DIM
    control_dim: int = CONTROL_DIM
    pose_param_dim: int = POSE_PARAM_DIM
    risk_channels: tuple[str, ...] = RISK_CHANNELS

    def num_risk_channels(self) -> int:
        return len(self.risk_channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_length": int(self.history_length),
            "horizon": int(self.horizon),
            "map_channels": int(self.map_channels),
            "map_height": int(self.map_height),
            "map_width": int(self.map_width),
            "map_resolution": float(self.map_resolution),
            "state_dim": int(self.state_dim),
            "control_dim": int(self.control_dim),
            "pose_param_dim": int(self.pose_param_dim),
            "risk_channels": list(self.risk_channels),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HighLevelFdmSchema":
        return cls(
            history_length=int(payload["history_length"]),
            horizon=int(payload["horizon"]),
            map_channels=int(payload["map_channels"]),
            map_height=int(payload["map_height"]),
            map_width=int(payload["map_width"]),
            map_resolution=float(payload["map_resolution"]),
            state_dim=int(payload.get("state_dim", STATE_DIM)),
            control_dim=int(payload.get("control_dim", CONTROL_DIM)),
            pose_param_dim=int(payload.get("pose_param_dim", POSE_PARAM_DIM)),
            risk_channels=tuple(payload.get("risk_channels", RISK_CHANNELS)),
        )


@dataclass
class HighLevelFdmSample:
    """One sample handed to the trainer. All fields are numpy arrays."""

    state_history: Any          # (H, STATE_DIM)
    map_patch: Any              # (C, Gh, Gw)
    control_sequence: Any       # (N, CONTROL_DIM)
    pose_target: Any            # (N, POSE_PARAM_DIM)
    risk_target: Any            # (N, K)
    risk_mask: Any              # (N, K)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def require_numpy():
    if not _NUMPY_AVAILABLE:
        raise RuntimeError(
            "numpy is required for high_level_fdm.schema tensor helpers. "
            "Install numpy to use dataset / synthetic utilities."
        )


def world_to_body_xy(xy_world, origin_xy, theta) -> Any:
    """Rotate+translate a world-frame xy tensor into a body frame.

    Args:
        xy_world: array of shape (..., 2).
        origin_xy: array of shape (2,) - body frame origin in world coords.
        theta: scalar - body frame yaw in world coords.

    Returns:
        Body-frame array of the same leading shape as `xy_world`.
    """
    require_numpy()
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rot = np.asarray([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
    delta = np.asarray(xy_world, dtype=np.float32) - np.asarray(origin_xy, dtype=np.float32)
    return delta @ rot.T


def body_to_world_xy(xy_body, origin_xy, theta) -> Any:
    require_numpy()
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rot = np.asarray([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    rotated = np.asarray(xy_body, dtype=np.float32) @ rot.T
    return rotated + np.asarray(origin_xy, dtype=np.float32)


def wrap_angle(theta) -> Any:
    require_numpy()
    theta_np = np.asarray(theta, dtype=np.float32)
    return np.arctan2(np.sin(theta_np), np.cos(theta_np)).astype(np.float32)


def pose_delta_to_param(pose_delta) -> Any:
    """Convert (..., 3) [dx, dy, dtheta] into (..., 4) [dx, dy, sin, cos]."""
    require_numpy()
    pose_delta = np.asarray(pose_delta, dtype=np.float32)
    dx = pose_delta[..., 0]
    dy = pose_delta[..., 1]
    dth = wrap_angle(pose_delta[..., 2])
    sin_dth = np.sin(dth).astype(np.float32)
    cos_dth = np.cos(dth).astype(np.float32)
    return np.stack([dx, dy, sin_dth, cos_dth], axis=-1).astype(np.float32)


def pose_param_to_delta(pose_param) -> Any:
    """Convert (..., 4) [dx, dy, sin, cos] into (..., 3) [dx, dy, dtheta].

    The sin/cos pair is renormalized so that the returned dtheta remains
    well-defined even when the network output drifts off the unit circle.
    """
    require_numpy()
    pose_param = np.asarray(pose_param, dtype=np.float32)
    dx = pose_param[..., 0]
    dy = pose_param[..., 1]
    sin_dth = pose_param[..., 2]
    cos_dth = pose_param[..., 3]
    norm = np.sqrt(sin_dth * sin_dth + cos_dth * cos_dth) + 1e-8
    dth = np.arctan2(sin_dth / norm, cos_dth / norm).astype(np.float32)
    return np.stack([dx, dy, dth], axis=-1).astype(np.float32)


def nominal_rollout(initial_xy_theta, controls, dt) -> Any:
    """Integrate [vx, vy, wz] commands in the body frame of the initial pose.

    Returns body-frame pose deltas ``(N, 3)`` as [dx, dy, dtheta], i.e. the
    *relative* pose at each step with respect to the initial pose. This is
    the analytical base against which the network predicts a residual.
    """
    require_numpy()
    controls = np.asarray(controls, dtype=np.float32).reshape(-1, CONTROL_DIM)
    num_steps = int(controls.shape[0])
    deltas = np.zeros((num_steps, POSE_DELTA_DIM), dtype=np.float32)

    # Track pose in the *body frame of t = initial*. Initial body-frame pose
    # is the origin (0, 0, 0); we integrate forward from there.
    x = 0.0
    y = 0.0
    theta = 0.0
    for step in range(num_steps):
        vx, vy, wz = (float(v) for v in controls[step])
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        x += (vx * cos_t - vy * sin_t) * float(dt)
        y += (vx * sin_t + vy * cos_t) * float(dt)
        theta += wz * float(dt)
        deltas[step, 0] = x
        deltas[step, 1] = y
        deltas[step, 2] = float(wrap_angle(theta))
    # The `initial_xy_theta` argument is kept for API symmetry with the
    # world-frame variant but unused because the base pose is by definition
    # zero in the body frame of t.
    _ = initial_xy_theta
    return deltas


def compose_world_trajectory(origin_state, pose_deltas_body) -> Any:
    """Transform body-frame deltas back to world-frame absolute poses.

    Args:
        origin_state: (6,) full state of t.
        pose_deltas_body: (N, 3) [dx, dy, dtheta] expressed in body frame.

    Returns:
        world_poses: (N, 3) [x, y, theta] in world frame.
    """
    require_numpy()
    origin_state = np.asarray(origin_state, dtype=np.float32).reshape(STATE_DIM)
    pose_deltas_body = np.asarray(pose_deltas_body, dtype=np.float32)
    origin_xy = origin_state[:2]
    origin_theta = float(origin_state[2])

    xy_world = body_to_world_xy(pose_deltas_body[:, :2], origin_xy, origin_theta)
    theta_world = wrap_angle(origin_theta + pose_deltas_body[:, 2])
    return np.concatenate([xy_world, theta_world[:, None]], axis=1).astype(np.float32)
