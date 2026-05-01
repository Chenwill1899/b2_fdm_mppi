# FDM Design

Last updated: 2026-05-01

## Model Role

Residual velocity FDM is a planning-level black-box closed-loop forward dynamics model. It predicts how real B2-equivalent executed velocity differs from commanded velocity under terrain and history context.

## Nominal B2 SE(2) State And Control

State:

```text
[x, y, theta, vx_real, vy_real, wz_real]
```

Control:

```text
[vx_cmd, vy_cmd, wz_cmd]
```

Nominal integration:

```text
x_next     = x + (vx * cos(theta) - vy * sin(theta)) * dt
y_next     = y + (vx * sin(theta) + vy * cos(theta)) * dt
theta_next = theta + wz * dt
```

Velocity limits:

```text
max_vx = 1.5
max_vy = 0.5
max_wz = 1.0
```

Current implementation:

```text
b2_fdm_mppi/core/omni_b2.py
config/b2_omni_nominal.yaml
```

## First Learned FDM Interface

Inputs:

```text
history:  [B, L, Dh]
env:      [B, De]
future_u: [B, N, 3]
```

Outputs:

```text
du_hat:   [B, N, 3]
```

Risk prediction is deferred until residual velocity prediction is validated.

## Stage 4 Baseline Interface

The first training baseline is a one-step residual regressor over the Stage 3 oracle dataset.

Inputs:

```text
features = concat(states, cmd_controls, terrain_features, terrain_risk)
shape    = [B, 14]
```

Targets:

```text
exec_residuals = real_controls - cmd_controls
shape          = [B, 3]
```

Artifacts:

```text
model.pt
normalization.npz
metrics.json
```

This baseline validates dataset usability before adding history windows or multi-step rollout prediction.

## Planned Loss

```text
L = L_res + L_pose + 0.05 * L_smooth
```

## First Acceptance Criteria

- Training loss decreases.
- Validation loss does not diverge.
- Learned FDM ADE/FDE improves over nominal SE(2).
- Checkpoint can be saved and loaded.
