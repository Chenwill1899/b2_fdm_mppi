# Real Robot Readiness

Date: 2026-05-03

## Current Readiness Level

Current status: numerical simulation package only.

The current repository state is suitable for offline analysis, paper/report figures, and shadow-mode planning design. It is not suitable for direct real-robot closed-loop control.

## Explicit Non-Claims

Do not claim:

- direct deployment readiness on B2 hardware;
- real-time learned-FDM-MPPI equivalence to nominal MPPI;
- safety certification;
- hardware-validated terrain-risk avoidance;
- MuJoCo validated sim-to-real behavior;
- actuator, estimator, latency, contact, or communication robustness.

## What Is Ready

The following pieces are ready as software/research ingredients:

- learned residual FDM baseline and normalization path;
- same-backend Torch nominal/learned comparison path;
- explicit terrain-risk MPPI objective;
- map-level numerical risk-aware result package;
- fixed two-obstacle visualization package;
- forced-step runtime profiler and runtime figures;
- reproducibility command list.

## What Is Missing For Real Deployment

Before real robot closed-loop control, the project still needs:

- robot state-estimator interface definition;
- command transport adapter with rate limiting and watchdog;
- strict command saturation and emergency stop integration;
- latency budget measurement on target compute;
- terrain/risk input source on the robot;
- online model validity checks;
- shadow-mode log replay and comparison tooling;
- operator-visible safety dashboard;
- hardware-in-the-loop or MuJoCo validation, if that becomes the next stage.

## Minimum Shadow-Mode Entry Conditions

The next stage may start only as shadow mode. In shadow mode, learned-FDM-MPPI computes commands but does not publish them to actuators.

Minimum entry conditions:

1. The nominal production/safety controller remains the only command source.
2. Learned-FDM-MPPI receives a read-only copy of state, goal, terrain/risk features, and obstacle inputs.
3. Shadow commands are logged with timestamps and never sent to the robot.
4. A command comparator logs nominal command, learned command, deltas, saturation ratio, risk cost, and inferred residual.
5. A watchdog verifies the shadow loop rate and drops stale shadow outputs.
6. Runtime telemetry records per-cycle wall time, `compute_control()` time, and missed-deadline counts.
7. Safety monitors flag command deltas above configurable thresholds.
8. The run can be replayed offline against the same input log.

## Shadow-Mode Pass Criteria

A shadow-mode run may be considered stable enough for the next review only if:

- no command is published by the learned stack;
- no missed-deadline burst exceeds the agreed threshold;
- learned command deltas remain within a predeclared envelope;
- nominal controller safety behavior is unchanged;
- logs include enough metadata for deterministic replay;
- operator abort and watchdog paths are tested;
- the team reviews risk cases where learned and nominal diverge.

## Deployment Boundary

The current Stage 6 package can justify starting a shadow-mode adapter design. It cannot justify closed-loop hardware deployment.

The first real robot milestone should be:

```text
read-only learned-FDM-MPPI shadow mode with full telemetry and no actuator authority
```
