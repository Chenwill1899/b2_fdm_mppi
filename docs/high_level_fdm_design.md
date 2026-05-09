# High-Level Learned Forward Dynamics Model (High-Level FDM)

> Branch: `feat/high-level-fdm`
> Scope: Design + minimal-but-self-proving implementation of a learned
> high-level forward dynamics model (High-Level FDM) that can act as
> (1) an MPPI rollout model and (2) an MPPI learned constraint.
>
> This document intentionally stays at the level of the *high-level
> velocity-command policy* (vx, vy, wz). It does not model the quadruped's
> joint-level dynamics. The target runtime is `ausim2` + `traversability_mapping`
> outside this repo, but all learning artifacts live here.

## 1. Why a High-Level FDM

The current `ausim2` MPPI cube demo keeps motion "reasonable" only because a
long list of handwritten cost/constraint terms patches the gap between the
toy SE(2) rollout model and the true quadruped's closed-loop high-level
behavior:

- `local costmap`, `goal progress`, `heading cost`
- `smoothness`, `acceleration`, `jerk`, `lateral velocity`, `yaw rate`
- `command filter`

Root cause: the rollout model inside MPPI is too simple. It predicts
"apply `[vx, vy, wz]` to a point mass" instead of "what does this sequence
of velocity commands actually do when fed into the quadruped's black-box
low-level policy, on this terrain, around these obstacles?"

**The High-Level FDM is the missing piece.** It learns, end-to-end, the
map `(history, local map, future velocity command sequence) -> (future
relative trajectory, per-step risk)`. Once it is good enough, MPPI can:

1. replace its analytical rollout with `HighLevelFdm.predict`, and
2. replace or dampen several handwritten constraints by using the predicted
   per-step risk `R_hat` directly as a constraint / cost term.

This project delivers the *model, data, and integration surface*. The
`ausim2` launch side is left untouched here.

## 2. Problem definition

At control time `t`:

- **Inputs**
  - `H_t`  history window of the last `H` states.
    Each row: `[x, y, theta, vx, vy, wz]` in a body-aligned frame where the
    current pose is the origin. Shape `(H, 6)`.
  - `E_t`  local environment patch. A 2D tensor (or stack of tensors) of
    shape `(C, Gh, Gw)` centered and aligned on the current pose. Channels
    encode e.g. occupancy / traversability / risk / height-statistics.
    The design is channel-agnostic; defaults are described below.
  - `U_{t:t+N-1}`  future high-level control sequence sampled by MPPI.
    Each row: `[vx, vy, wz]`. Shape `(N, 3)`.

- **Outputs**
  - `P_hat_{t+1:t+N}`  predicted future relative trajectory.
    Each row encodes `[dx, dy, sin(dtheta), cos(dtheta)]` in the body frame
    of state `t`. Shape `(N, 4)`. A helper returns the equivalent
    `[dx, dy, dtheta]` form with `atan2`.
  - `R_hat_{t+1:t+N}`  per-step risk probabilities. Shape `(N, K)` where `K`
    is the number of risk channels. Defaults:
    - `k=0` collision
    - `k=1` high-cost region (traversability exceeds a threshold)
    - `k=2` stuck / velocity-tracking collapse
    - `k=3` untraversable (catastrophic failure, e.g. flip/fall in the full
      quadruped; in the cube surrogate this is a hard obstacle hit)
    Each channel is a sigmoid output in `[0, 1]`. Channels can be disabled
    at training time.

## 3. Why this parameterization

- **Relative trajectory in body frame of `t`**  removes the global pose bias,
  lets the network generalize across positions, and makes the MPPI cost
  `sum_k w_k * f(dx_k, dy_k, dtheta_k)` invariant to world origin.
- **`sin/cos` instead of `dtheta`**  avoids `[-pi, pi]` wrap-around gradients,
  matches the inputs that attitude-aware heads expect, and makes the Huber
  loss well-defined at `+/- pi`.
- **Independent per-channel risk**  avoids softmax coupling (a step can be
  simultaneously in high-cost terrain and predicted to collide). BCE-with-logits
  is numerically stable.
- **Stepwise weighting**  lets us train the first few prediction steps harder
  since MPPI weighs early commands more.

## 4. Model

The model keeps three backbones and two heads:

```
history H_t ---- HistoryEncoder --------+
                                        |
local patch E_t -- MapEncoder ----------+--> ContextFused (D_ctx)
                                        |
                 StepDecoder (recurrent):
                   inputs  : (ContextFused, u_k, previous hidden)
                   outputs : pose_delta_head -> (dx, dy, sin_dth, cos_dth)
                             risk_head       -> logits (K)
                 repeated N times
```

- **HistoryEncoder**  a small 1D conv / MLP that takes the last `H` states
  (already rotated into the local frame). Default: two Conv1d layers + a
  linear bottleneck of size `d_hist`.
- **MapEncoder**  small 2D CNN. Two Conv2d + GAP + linear to `d_map`.
  Channel count is read from config, so it will later accept traversability
  patches published by `traversability_mapping` without code changes.
- **ContextFused**  concatenation of `d_hist + d_map` to `d_ctx`, projected
  by a linear layer. The map encoder runs once per MPPI optimization step
  (it does not depend on `u`); the decoder runs per `u` sample.
- **StepDecoder**  a GRUCell consuming the per-step command `u_k`, the shared
  context, and the previous hidden state. Two small MLPs emit:
  - `pose_delta_head`  `(dx, dy, sin_dth, cos_dth)` residual **relative to
    the analytical nominal rollout of `u_k` applied to the previous pose**.
    Training with a residual around a physics-consistent base improves
    generalization and keeps predictions stable at `N=0`.
  - `risk_head`  `K` logits.

This is the same conceptual shape as "Learned Perceptive Forward Dynamics
Model for Safe and Platform-aware Robotic Navigation" (Roth et al., RSS
2025), specialized for a repo that already has an SE(2) nominal model and
a terrain risk field.

## 5. Loss

```
L = lambda_pose * L_pose + lambda_risk * L_risk + lambda_smooth * L_smooth
```

- `L_pose`  Huber over `(dx, dy, sin_dth, cos_dth)` residuals, summed per
  step, then averaged over steps with configurable per-step weights.
- `L_risk`  per-channel BCE-with-logits, masked by `risk_mask` (so a
  channel without a label at a given step contributes zero).
- `L_smooth`  light penalty on consecutive pose-delta residuals to encourage
  temporally smooth predictions. Weight default `0.0`; present for ablation.

All losses are computed in standardized units (pose deltas normalized by a
data-driven std). Risk loss is computed in raw logit/probability space.

## 6. Data pipeline

### 6.1 Per-sample layout

```
state_history     float32 (H, 6)       body-frame, current pose = origin
map_patch         float32 (C, Gh, Gw)  traversability / occupancy / risk / ...
control_sequence  float32 (N, 3)       body-frame [vx, vy, wz]
pose_target       float32 (N, 4)       [dx, dy, sin_dth, cos_dth]
risk_target       float32 (N, K)       binary labels in [0, 1]
risk_mask         float32 (N, K)       1.0 where label is valid
meta              dict                 episode id, seed, scenario tags
```

### 6.2 Synthetic generator (this repo only)

To keep the High-Level FDM repo self-contained, a numpy-only synthetic
generator reuses existing pieces:

- `b2_fdm_mppi.core.omni_b2.OmniB2`       nominal high-level SE(2) model.
- `b2_fdm_mppi.core.residual_world.ResidualWorld`  oracle residual simulator
  over a terrain field.
- `b2_fdm_mppi.core.terrain.TerrainField` analytic terrain / traversability.
- `b2_fdm_mppi.simulation.random_obstacles` (for obstacle configs) or an
  inline obstacle generator that avoids coupling to the existing runner.

Each scenario samples:

1. a random terrain field and a small obstacle set,
2. a history of `H` past steps driven by band-limited random commands,
3. an MPPI-style command sequence of length `N`,
4. the ground-truth next-state trajectory by integrating `ResidualWorld`
   step-by-step under those commands,
5. the local map patch rasterized from `TerrainField.risk_cost` and the
   obstacle set,
6. risk labels:
   - collision from swept robot-disk vs obstacle-disk distance,
   - high-cost from the terrain risk threshold,
   - stuck from `||executed_velocity - command||` vs a threshold,
   - untraversable from the hardest collision or out-of-map event.

This generator is the stand-in until `ausim2` + `traversability_mapping`
provide real data. It is *not* MuJoCo. It is intentionally a surrogate so
we can validate the training code, the schema, and the MPPI adapter
end-to-end on CI-sized runs.

### 6.3 Real data hook

When real episodes become available, a thin reader can emit the same
per-sample layout by:

- reading `state_history` from the robot odometry,
- reading `map_patch` from the `traversability_mapping` message,
- reading `control_sequence` either from MPPI logs or from the
  actually-applied commands,
- reconstructing `pose_target` from the executed odometry,
- labeling `risk_target` from collision / cost-map / velocity-tracking
  logs.

The model is agnostic to which source produced the sample.

## 7. Training

```
fdm_mppi high-fdm synth      # build synthetic dataset shards
fdm_mppi high-fdm train      # train
fdm_mppi high-fdm eval       # offline dataset eval
fdm_mppi high-fdm rollout    # tiny rollout + MPPI-cost contract check
```

- AdamW, cosine LR decay, configurable epochs / batch size / seed.
- Deterministic init, seeded dataloaders.
- Saves `best_model.pt`, `model.pt`, `normalization.npz`, `schema.json`.
- TensorBoard logging reuses the residual FDM convention so both models
  can live side-by-side under `results/`.

## 8. MPPI integration contract

Two integration points are exposed:

1. `HighLevelFdmRollout.predict_batch(states, map_patch, controls_batch)`
   - `controls_batch` shape `(M, N, 3)` - M MPPI samples per step.
   - returns `(pose_deltas: (M, N, 4), risks: (M, N, K))`.
   - MPPI replaces its SE(2) rollout with the predicted pose deltas,
     reconstructing the absolute trajectory by composing them onto the
     current pose.

2. `high_level_fdm_cost(risks, weights)`
   - risks: `(M, N, K)`.
   - weights: `(K,)`.
   - returns `(M,)` cumulative cost over horizon.
   - MPPI adds this to its existing cost. Over time, handwritten
     obstacle / lateral / smoothness terms can be down-weighted as the
     risk head subsumes them.

The adapter keeps pose units identical to what `MppiOmniTorch`
already consumes, so drop-in replacement is mechanical.

## 9. Evaluation plan

Offline (no ROS required, repo-local):

- Dataset metrics on held-out split:
  - `pose_mse_xy`, `pose_mse_theta`, step-averaged.
  - Risk AUROC / Brier per channel.
  - Zero-residual baseline: predict the nominal SE(2) rollout with zero
    risk. Any model that cannot beat this is useless.
- Rollout contract test: feed `M=8`, `N=12` zero commands and check that
  the predicted deltas stay within a tolerance of zero motion. This is
  cheap and catches shape / frame bugs.

Closed-loop (future work, not in this PR):

- Drop-in replacement of the rollout model in `MppiOmniTorch` via a
  wrapper that uses `HighLevelFdmRollout` for prediction and `risk` for
  cost.
- Benchmark against the nominal MPPI on the existing `benchmark.yaml`
  scenarios. Reuse `b2_fdm_mppi.evaluation.benchmark`.

## 10. Directory layout

```
b2_fdm_mppi/high_level_fdm/
  __init__.py
  schema.py        # constants, dataclasses, frame transforms
  model.py         # HighLevelFdm nn.Module
  losses.py        # loss functions, stepwise weights
  trainer.py       # train/eval loops, checkpointing
  rollout.py       # HighLevelFdmRollout, MPPI cost adapter
  synthetic.py     # numpy-only synthetic generator
  dataset.py       # HighLevelFdmDataset, collate_fn, dataset builder
configs/high_level_fdm.yaml
tests/test_high_level_fdm.py
scripts/selfproof_high_level_fdm.sh
```

## 11. Self-proof plan

The sandbox this PR is prepared in does not have `numpy`/`torch`
available. The self-proof script `scripts/selfproof_high_level_fdm.sh`
is the single command a user runs on a machine with torch+numpy
installed:

1. build a tiny synthetic dataset (1-2 scenarios, small horizon),
2. train 3 epochs on CPU,
3. evaluate on val+test,
4. execute the MPPI rollout contract test,
5. emit `selfproof_summary.json` containing pass/fail per check.

The contract is: the trained model must (a) beat the zero-residual
baseline on pose MSE, (b) produce finite risk predictions in `[0, 1]`,
(c) preserve tensor shapes expected by MPPI. These are lightweight and
runnable in under a minute on CPU.

## 12. Out of scope (explicit)

- MuJoCo integration (the `ausim2` repo is not uploaded; we stub the
  real-data reader).
- `traversability_mapping` ROS integration.
- Full closed-loop MPPI benchmark with the high-level FDM.

Each of those will get its own PR once the learning side is stable.
