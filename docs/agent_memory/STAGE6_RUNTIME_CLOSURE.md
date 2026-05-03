# Stage 6 Runtime Closure

Date: 2026-05-03

## Goal

Close Stage 6 runtime work as a reproducible profiling package with clear limitations. This does not attempt deeper runtime optimization or new model structure.

## Final Runtime Protocol

Official runtime closure uses:

- backend: `torch`
- device: `cuda` when available
- matrix: nominal/learned x risk off/on
- base seed: `123`
- episodes: `10`
- forced control calls per run: `10`
- config: `config/b2_omni_oracle_random100_dataset.yaml`

The profiler now defaults to fixed-step semantics:

```text
force_steps=true
simulation.disable_goal_termination=true
steps_semantics=forced_compute_control_calls
```

This means `--steps 10` is interpreted as exactly 10 `compute_control()` calls per run unless a run fails. This is different from ordinary benchmark semantics, where `simulation.max_steps` is only an upper bound and the runner may stop after reaching the goal.

Use `--allow-goal-termination` only when intentionally profiling ordinary max-step runner behavior.

## Closeout Command

```bash
python3 tools/profile_stage6_runtime_matrix.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --scenario-name id_random_fixed_seed_closeout \
  --output results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout \
  --episodes 10 \
  --steps 10 \
  --base-seed 123 \
  --backend torch \
  --device cuda \
  --risk-weight 3.0
```

Plot command:

```bash
python3 tools/plot_stage6_runtime_results.py \
  --summary \
    results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode/stage6_runtime_matrix_summary.json \
    results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout/stage6_runtime_matrix_summary.json \
  --output figures/stage6 \
  --tables-output tables/stage6
```

## Consistency Check

Closeout summary:

```text
profile_call_consistency.consistent = true
profile_call_consistency.expected_calls_per_run = 10
profile_call_consistency.unique_total_calls = [10]
```

Tracked table field:

```text
profile_total_calls_mean = 10
```

## Final Runtime Snapshot

| Case | calls/run | mean_mppi_time_ms | rollout_total_ms | terrain_features_ms | fdm_inference_ms | terrain_risk_cost_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal_risk_off | 10 | 13.90 | 8.67 | n/a | n/a | 0.042 |
| nominal_risk_on | 10 | 13.91 | 8.54 | n/a | n/a | 1.473 |
| learned_risk_off | 10 | 38.97 | 34.94 | 0.834 | 0.178 | 0.045 |
| learned_risk_on | 10 | 39.93 | 34.98 | 0.839 | 0.172 | 0.972 |

Paired deltas:

| Pair | mean_mppi_time_ms_delta | rollout_total_ms_delta | sample_candidates_delta | update_distribution_delta | terrain_risk_cost_delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_risk_on - nominal_risk_on | +26.02 | +26.44 | +0.003 | +0.001 | -0.501 |
| learned_risk_off - nominal_risk_off | +25.07 | +26.26 | -0.373 | -0.370 | +0.003 |
| learned_risk_on - learned_risk_off | +0.96 | +0.04 | +0.002 | -0.010 | +0.927 |

## Closure Interpretation

Runtime closure is complete for numerical reporting:

- same-backend Torch runtime comparison is reproducible;
- fixed control-call counts remove early-termination ambiguity;
- tracked figures and tables summarize the runtime result;
- learned runtime overhead is quantified and bounded.

Runtime closure is not a deployment claim:

- learned risk-on remains about `+26.02 ms` slower than nominal risk-on;
- rollout remains the dominant overhead;
- no real-time equivalence claim should be made;
- deeper optimization should be a separate follow-up, not part of the current final package.
