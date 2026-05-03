#!/usr/bin/env python3
"""Generate Stage 6 runtime profiling tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ORDER = (
    "nominal_risk_off",
    "nominal_risk_on",
    "learned_risk_off",
    "learned_risk_on",
)
CASE_LABELS = {
    "nominal_risk_off": "Nominal\nrisk off",
    "nominal_risk_on": "Nominal\nrisk on",
    "learned_risk_off": "Learned\nrisk off",
    "learned_risk_on": "Learned\nrisk on",
}
CASE_COLORS = {
    "nominal_risk_off": "#7A869A",
    "nominal_risk_on": "#2F6DB3",
    "learned_risk_off": "#E68632",
    "learned_risk_on": "#B44CC2",
}
BREAKDOWN_BUCKETS = (
    ("profile_mean_sample_candidates_ms_mean", "sample", "#9CA3AF"),
    ("profile_mean_rollout_total_ms_mean", "rollout", "#B44CC2"),
    ("profile_mean_obstacle_cost_ms_mean", "obstacle", "#009E73"),
    ("profile_mean_terrain_risk_cost_ms_mean", "risk cost", "#2F6DB3"),
    ("profile_mean_update_distribution_ms_mean", "update", "#D55E00"),
    ("profile_mean_cpu_transfer_ms_mean", "transfer", "#6B7280"),
)
DELTA_BUCKETS = (
    ("mean_mppi_time_ms_delta_mean", "MPPI"),
    ("profile_mean_rollout_total_ms_delta_mean", "rollout"),
    ("profile_mean_sample_candidates_ms_delta_mean", "sample"),
    ("profile_mean_update_distribution_ms_delta_mean", "update"),
    ("profile_mean_terrain_features_ms_delta_mean", "terrain"),
    ("profile_mean_fdm_inference_ms_delta_mean", "FDM"),
    ("profile_mean_terrain_risk_cost_ms_delta_mean", "risk cost"),
)


def plot_stage6_runtime_results(
    *,
    summary_paths: Sequence[str | Path],
    output_dir: str | Path = "figures/stage6",
    tables_dir: str | Path = "tables/stage6",
) -> dict:
    summaries = [_load_summary(path) for path in summary_paths]
    if not summaries:
        raise ValueError("At least one Stage 6 runtime summary is required")
    output_dir = Path(output_dir)
    tables_dir = Path(tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    summary_rows = _summary_rows(summaries)
    paired_rows = _paired_delta_rows(summaries)
    _write_csv(tables_dir / "table_stage6_runtime_summary.csv", summary_rows)
    _write_csv(tables_dir / "table_stage6_runtime_paired_deltas.csv", paired_rows)

    _plot_runtime_summary(summaries, output_dir / "fig_stage6_runtime_summary")
    _plot_breakdown(summaries[-1], output_dir / "fig_stage6_runtime_breakdown")
    _plot_delta_buckets(summaries[-1], output_dir / "fig_stage6_runtime_delta_buckets")

    report = {
        "summary_count": len(summaries),
        "figure_dir": str(output_dir),
        "tables_dir": str(tables_dir),
        "summary_paths": [str(summary["path"]) for summary in summaries],
        "figures": [
            str(output_dir / "fig_stage6_runtime_summary.png"),
            str(output_dir / "fig_stage6_runtime_breakdown.png"),
            str(output_dir / "fig_stage6_runtime_delta_buckets.png"),
        ],
        "tables": [
            str(tables_dir / "table_stage6_runtime_summary.csv"),
            str(tables_dir / "table_stage6_runtime_paired_deltas.csv"),
        ],
    }
    (output_dir / "stage6_runtime_figure_manifest.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def _load_summary(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    scenario = str(metadata.get("scenario_name") or path.parent.name)
    label = scenario
    episodes = metadata.get("episodes")
    steps = metadata.get("steps")
    if episodes is not None and steps is not None:
        suffix = " closeout" if "closeout" in scenario else ""
        label = f"{episodes}ep x {steps}step{suffix}"
    return {
        "path": path,
        "label": label,
        "metadata": metadata,
        "aggregates": data.get("aggregates", {}),
        "paired_deltas": data.get("paired_deltas", {}),
    }


def _summary_rows(summaries: Sequence[dict]) -> list[dict]:
    rows = []
    metric_keys = (
        "success_rate",
        "mean_mppi_time_ms_mean",
        "max_mppi_time_ms_mean",
        "profile_total_calls_mean",
        "profile_mean_sample_candidates_ms_mean",
        "profile_mean_rollout_total_ms_mean",
        "profile_mean_terrain_features_ms_mean",
        "profile_mean_fdm_inference_ms_mean",
        "profile_mean_obstacle_cost_ms_mean",
        "profile_mean_terrain_risk_cost_ms_mean",
        "profile_mean_update_distribution_ms_mean",
        "profile_mean_cpu_transfer_ms_mean",
    )
    for summary in summaries:
        metadata = summary["metadata"]
        for case in CASE_ORDER:
            aggregate = summary["aggregates"].get(case, {})
            row = {
                "summary_label": summary["label"].replace("\n", " "),
                "summary_path": str(summary["path"]),
                "scenario_name": metadata.get("scenario_name"),
                "backend": metadata.get("backend"),
                "device": metadata.get("device"),
                "episodes": metadata.get("episodes"),
                "steps": metadata.get("steps"),
                "base_seed": metadata.get("base_seed"),
                "git_sha": metadata.get("git_sha"),
                "case": case,
            }
            for key in metric_keys:
                row[key] = _csv(aggregate.get(key))
            rows.append(row)
    return rows


def _paired_delta_rows(summaries: Sequence[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        metadata = summary["metadata"]
        for pair_name, pair_data in sorted(summary["paired_deltas"].items()):
            aggregate = pair_data.get("aggregate", {})
            row = {
                "summary_label": summary["label"].replace("\n", " "),
                "summary_path": str(summary["path"]),
                "scenario_name": metadata.get("scenario_name"),
                "episodes": metadata.get("episodes"),
                "steps": metadata.get("steps"),
                "pair": pair_name,
                "count": aggregate.get("count"),
            }
            for metric, _label in DELTA_BUCKETS:
                row[metric] = _csv(aggregate.get(metric))
            rows.append(row)
    return rows


def _plot_runtime_summary(summaries: Sequence[dict], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    width = 0.18 if len(summaries) > 1 else 0.45
    x = np.arange(len(CASE_ORDER))
    offsets = (np.arange(len(summaries)) - (len(summaries) - 1) / 2.0) * width
    for offset, summary in zip(offsets, summaries):
        values = [
            _finite(summary["aggregates"].get(case, {}).get("mean_mppi_time_ms_mean"))
            for case in CASE_ORDER
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=summary["label"].replace("\n", " "),
            color=[CASE_COLORS[case] for case in CASE_ORDER],
            alpha=0.85 if len(summaries) == 1 else 0.72,
            edgecolor="#333333",
            linewidth=0.5,
        )
    ax.set_ylabel("Mean MPPI time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in CASE_ORDER])
    ax.set_title("Stage 6 same-backend Torch runtime")
    if len(summaries) > 1:
        ax.legend(loc="upper left", fontsize=7)
    _panel(ax, "a")
    _save(fig, stem)


def _plot_breakdown(summary: dict, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(CASE_ORDER))
    bottoms = np.zeros(len(CASE_ORDER), dtype=float)
    for key, label, color in BREAKDOWN_BUCKETS:
        values = np.asarray(
            [_finite(summary["aggregates"].get(case, {}).get(key)) for case in CASE_ORDER],
            dtype=float,
        )
        ax.bar(x, values, bottom=bottoms, label=label, color=color, edgecolor="#333333", linewidth=0.35)
        bottoms += values
    ax.set_ylabel("Profile bucket mean (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in CASE_ORDER])
    ax.set_title(f"Runtime bucket breakdown: {summary['label'].replace(chr(10), ' ')}")
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    _panel(ax, "b")
    _save(fig, stem)


def _plot_delta_buckets(summary: dict, stem: Path) -> None:
    pair_name = "learned_risk_on_vs_nominal_risk_on"
    aggregate = summary["paired_deltas"].get(pair_name, {}).get("aggregate", {})
    labels = [label for metric, label in DELTA_BUCKETS if aggregate.get(metric) is not None]
    values = [_finite(aggregate.get(metric)) for metric, _label in DELTA_BUCKETS if aggregate.get(metric) is not None]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(values))
    colors = ["#B44CC2" if value >= 0.0 else "#2F6DB3" for value in values]
    ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Learned risk-on - nominal risk-on (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(f"Paired runtime deltas: {summary['label'].replace(chr(10), ' ')}")
    _panel(ax, "c")
    _save(fig, stem)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _finite(value) -> float:
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _csv(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True, help="Stage 6 runtime matrix summary JSON path(s).")
    parser.add_argument("--output", default="figures/stage6", help="Figure output directory.")
    parser.add_argument("--tables-output", default="tables/stage6", help="Table output directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = plot_stage6_runtime_results(
        summary_paths=args.summary,
        output_dir=args.output,
        tables_dir=args.tables_output,
    )
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
