#!/usr/bin/env python3
"""Analyze Stage 5-E risk-aware sweeps with paired statistics and risk figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_DIRECTIONS = {
    "final_distance": "lower",
    "steps": "lower",
    "max_terrain_risk": "lower",
    "cumulative_terrain_risk": "lower",
    "terrain_risk_excess": "lower",
    "terrain_risk_excess_integral": "lower",
    "terrain_risk_exposure_ratio": "lower",
    "control_smoothness": "lower",
    "control_jerk": "lower",
    "mean_mppi_time_ms": "lower",
}
DEFAULT_METRICS = (
    "final_distance",
    "cumulative_terrain_risk",
    "max_terrain_risk",
    "terrain_risk_excess",
    "control_smoothness",
    "control_jerk",
    "mean_mppi_time_ms",
)


def analyze_sweep(
    *,
    sweep_summary_path: str | Path,
    output_dir: str | Path | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    bootstrap_samples: int = 2000,
    random_seed: int = 123,
) -> dict:
    sweep_summary_path = Path(sweep_summary_path)
    output_dir = Path(output_dir) if output_dir is not None else sweep_summary_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = json.loads(sweep_summary_path.read_text(encoding="utf-8"))
    cases = []
    plot_cases = []
    flat_rows = []

    for case in sweep.get("cases", []):
        case_dir = Path(case["output_dir"])
        benchmark_path = case_dir / "stage5_benchmark_summary.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        pairs = benchmark.get("paired_deltas", {}).get("pairs", [])
        metric_stats = {}
        metric_deltas = {}
        for metric in metrics:
            values = [pair.get(f"{metric}_delta") for pair in pairs]
            metric_deltas[metric] = [float(value) for value in values if _finite(value)]
            stats = metric_delta_stats(
                values,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
            )
            metric_stats[metric] = stats
            flat_rows.append(
                {
                    "scenario": case.get("scenario_name"),
                    "case_name": case.get("case_name"),
                    "terrain_risk_weight": case.get("terrain_risk_weight"),
                    "metric": metric,
                    **_flatten_metric_stats(stats),
                }
            )
        cases.append(
            {
                "case_name": case.get("case_name"),
                "scenario_name": case.get("scenario_name"),
                "terrain_risk_weight": case.get("terrain_risk_weight"),
                "output_dir": str(case_dir),
                "pair_count": len(pairs),
                "metrics": metric_stats,
            }
        )
        plot_cases.append(
            {
                "scenario_name": case.get("scenario_name"),
                "terrain_risk_weight": case.get("terrain_risk_weight"),
                "metric_deltas": metric_deltas,
            }
        )

    report = {
        "metadata": {
            "sweep_summary_path": str(sweep_summary_path),
            "output_dir": str(output_dir),
            "metrics": list(metrics),
            "bootstrap_samples": int(bootstrap_samples),
            "random_seed": int(random_seed),
        },
        "sweep_metadata": sweep.get("metadata", {}),
        "cases": cases,
    }
    (output_dir / "stage5_e_paired_stats.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "stage5_e_paired_stats.csv", flat_rows)
    _save_paired_delta_boxplots(plot_cases, output_dir / "stage5_e_paired_delta_boxplots.png")
    _save_risk_pareto(sweep, output_dir / "stage5_e_risk_pareto.png")
    _save_mean_risk_curves(sweep, output_dir)
    return report


def metric_delta_stats(
    deltas: Iterable[float | int | None],
    *,
    metric: str,
    bootstrap_samples: int = 2000,
    random_seed: int = 123,
) -> dict:
    values = np.asarray([float(v) for v in deltas if _finite(v)], dtype=float)
    direction = METRIC_DIRECTIONS.get(metric, "lower")
    if values.size == 0:
        return {
            "count": 0,
            "mean_delta": None,
            "median_delta": None,
            "std_delta": None,
            "bootstrap_ci95": [None, None],
            "pct_improved": None,
            "wilcoxon_p": None,
            "wilcoxon_method": None,
        }
    ci_low, ci_high = _bootstrap_mean_ci(values, bootstrap_samples, random_seed)
    return {
        "count": int(values.size),
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "std_delta": float(np.std(values)),
        "bootstrap_ci95": [ci_low, ci_high],
        "pct_improved": float(_improvement_mask(values, direction).mean()),
        **_wilcoxon(values),
    }


def _bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 1 or samples <= 0:
        mean = float(np.mean(values))
        return mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(samples), values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _wilcoxon(values: np.ndarray) -> dict:
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return {"wilcoxon_p": None, "wilcoxon_method": "all_zero"}
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", mode="auto")
        return {"wilcoxon_p": float(result.pvalue), "wilcoxon_method": "scipy"}
    except Exception:
        return {"wilcoxon_p": _normal_approx_wilcoxon_p(nonzero), "wilcoxon_method": "normal_approx"}


def _normal_approx_wilcoxon_p(values: np.ndarray) -> float:
    abs_values = np.abs(values)
    order = np.argsort(abs_values)
    ranks = np.empty_like(abs_values)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    w_plus = float(ranks[values > 0.0].sum())
    n = len(values)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0.0:
        return 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    cdf = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    return float(2.0 * (1.0 - cdf))


def _flatten_metric_stats(stats: dict) -> dict:
    low, high = stats.get("bootstrap_ci95", [None, None])
    flat = {key: value for key, value in stats.items() if key != "bootstrap_ci95"}
    flat["bootstrap_ci95_low"] = low
    flat["bootstrap_ci95_high"] = high
    return flat


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_paired_delta_boxplots(cases: list[dict], path: Path) -> None:
    metrics = ["final_distance", "cumulative_terrain_risk", "terrain_risk_excess", "control_jerk"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        labels = []
        values = []
        for case in cases:
            deltas = case["metric_deltas"].get(metric, [])
            if not deltas:
                continue
            labels.append(f"{case['scenario_name']}\nw={case['terrain_risk_weight']:g}")
            values.append(deltas)
        if values:
            ax.boxplot(values, labels=labels, showfliers=False)
            ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_title(f"Delta {metric}")
        ax.tick_params(axis="x", rotation=45)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_risk_pareto(sweep: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for case in sweep.get("cases", []):
        weight = float(case.get("terrain_risk_weight", 0.0))
        scenario = str(case.get("scenario_name", "scenario"))
        for controller, marker in (("nominal", "o"), ("learned", "s")):
            x = case.get(f"{controller}_final_distance_mean")
            y = case.get(f"{controller}_cumulative_terrain_risk_mean")
            if _finite(x) and _finite(y):
                ax.scatter(float(x), float(y), marker=marker, s=45, label=f"{controller}" if weight == 0 else None)
                ax.annotate(f"{scenario} w={weight:g}", (float(x), float(y)), fontsize=7, alpha=0.8)
    ax.set_xlabel("Final distance")
    ax.set_ylabel("Cumulative terrain risk")
    ax.set_title("Risk-aware Pareto scatter")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_mean_risk_curves(sweep: dict, output_dir: Path) -> None:
    records = []
    for case in sweep.get("cases", []):
        benchmark_path = Path(case["output_dir"]) / "stage5_benchmark_summary.json"
        if not benchmark_path.is_file():
            continue
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        records.append((case, benchmark))
    _plot_mean_series(records, output_dir / "stage5_e_mean_risk_timeseries.png", cumulative=False)
    _plot_mean_series(records, output_dir / "stage5_e_mean_cumulative_risk.png", cumulative=True)


def _plot_mean_series(records: list[tuple[dict, dict]], path: Path, *, cumulative: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for case, benchmark in records:
        for controller, linestyle in (("nominal", "-"), ("learned", "--")):
            curves = []
            for run in benchmark.get("runs", []):
                if run.get("controller") != controller:
                    continue
                terrain_path = Path(run.get("results_path", "")) / "terrain.csv"
                risks = _read_risk_series(terrain_path)
                if risks.size:
                    curves.append(np.cumsum(risks) if cumulative else risks)
            mean_curve = _mean_curve(curves)
            if mean_curve.size:
                label = f"{case.get('scenario_name')} w={case.get('terrain_risk_weight'):g} {controller}"
                ax.plot(np.arange(mean_curve.size), mean_curve, linestyle=linestyle, linewidth=1.3, alpha=0.8, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative risk" if cumulative else "Terrain risk")
    ax.set_title("Mean cumulative terrain risk" if cumulative else "Mean terrain risk")
    if ax.lines:
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _read_risk_series(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.asarray([], dtype=float)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        values = []
        for row in reader:
            value = row.get("risk_cost")
            if _finite(value):
                values.append(float(value))
    return np.asarray(values, dtype=float)


def _mean_curve(curves: list[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.asarray([], dtype=float)
    min_len = min(len(curve) for curve in curves)
    if min_len == 0:
        return np.asarray([], dtype=float)
    stacked = np.vstack([curve[:min_len] for curve in curves])
    return stacked.mean(axis=0)


def _improvement_mask(values: np.ndarray, direction: str) -> np.ndarray:
    if direction == "higher":
        return values > 0.0
    return values < 0.0


def _finite(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def parse_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    if not metrics:
        raise ValueError("At least one metric is required")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=123)
    args = parser.parse_args()

    report = analyze_sweep(
        sweep_summary_path=args.sweep_summary,
        output_dir=args.output,
        metrics=parse_metrics(args.metrics),
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    print(json.dumps({"output_dir": report["metadata"]["output_dir"], "cases": len(report["cases"])}, indent=2))


if __name__ == "__main__":
    main()
