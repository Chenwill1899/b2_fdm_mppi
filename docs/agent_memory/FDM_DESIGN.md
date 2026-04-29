# FDM Design

Last updated: 2026-04-29

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

## Planned Loss

```text
L = L_res + L_pose + 0.05 * L_smooth
```

## First Acceptance Criteria

- Training loss decreases.
- Validation loss does not diverge.
- Learned FDM ADE/FDE improves over nominal SE(2).
- Checkpoint can be saved and loaded.

