# B2-FDM-MPPI Project Context

Last updated: 2026-04-29

## Repository

- Path: `/home/mexxiie/prj/py-mppi`
- Current branch at startup: `fdm`
- Package: ROS 2 Humble `ament_python` package `b2_fdm_mppi`
- Current baseline structure: PyCUDA MPPI/CBF internal simulation with Jackal-style differential drive model.

## Research Goal

Build a reproducible numerical simulation and evaluation platform for B2 quadruped residual velocity FDM-MPPI.

Core chain:

1. Use an omnidirectional SE(2) model as the nominal dynamics.
2. Build an oracle residual world to emulate B2 execution bias.
3. Generate oracle residual datasets.
4. Train a residual velocity forward dynamics model (FDM).
5. Use the learned FDM as the MPPI rollout model.
6. Compare Nominal-MPPI, Oracle-MPPI, Learned-FDM-MPPI, and risk-aware variants.

## Important Distinction

- FDM is a planning-level black-box closed-loop forward dynamics model.
- MPPI is the sampling-based planning/control framework that uses the model.
- The project does not replace full quadruped analytical dynamics. It learns execution residuals on top of a nominal omnidirectional SE(2) model.

## Current Baseline Facts

- Default config: `config/fdm_mppi.yaml`
- Default target: `[10.0, 0.0, 0.0, 0.0, 0.0]`
- Default plots and animation are enabled.
- Results currently save CSV files and `test_summary.yaml` under `results/sim_results/<timestamp>/`.
- Existing tests use a fake controller for non-GPU runner coverage.

