"""Unified command line interface for the slim FDM-MPPI pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence


class PipelineCommands:
    """Command handlers kept thin so tests can inject a recorder."""

    def experiment(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.experiment import ExperimentConfigError, run_experiment_profile

        try:
            summary = run_experiment_profile(
                args.profile,
                controller_name=args.controller,
                seed=args.seed,
                backend=args.backend,
                output_root=args.output,
                results_dir=args.results_dir,
                model_dir=args.model_dir,
                checkpoint=args.checkpoint,
                normalization=args.normalization,
                device=args.device,
                residual_gain=args.residual_gain,
                enable_plots=args.plots,
                enable_animation=args.animation,
            )
        except ExperimentConfigError as exc:
            print(f"experiment error: {exc}", file=sys.stderr)
            return 2
        _print_json(summary)
        return 0

    def run(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.simulation.run_omni_mppi import print_run_summary, run_omni_mppi

        summary = run_omni_mppi(
            config_path=args.config,
            seed=args.seed,
            backend=args.backend,
            fdm_enabled=args.fdm_enabled,
            fdm_model_dir=args.fdm_model_dir,
            fdm_checkpoint=args.fdm_checkpoint,
            fdm_normalization=args.fdm_normalization,
            fdm_device=args.fdm_device,
            fdm_residual_gain=args.fdm_residual_gain,
        )
        print_run_summary(summary)
        return 0

    def dataset_collect(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.data.generate_oracle_episodes import generate_oracle_episodes

        summary = generate_oracle_episodes(
            config_path=args.config,
            episodes=args.episodes,
            base_seed=args.base_seed,
            output_dir=args.output,
            backend=args.backend,
            num_workers=args.num_workers,
        )
        _print_json(summary)
        return 0

    def dataset_build(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.data.oracle_dataset import build_oracle_dataset

        summary = build_oracle_dataset(
            input_dir=args.input,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.seed,
        )
        _print_json(summary)
        return 0

    def dataset_validate(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.data.oracle_dataset_validator import validate_oracle_dataset

        output = args.output or args.dataset
        quality = validate_oracle_dataset(Path(args.dataset), Path(output))
        _print_json(quality)
        return 0

    def train(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.training.residual_fdm import shell_join, train_residual_fdm

        argv = [sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]
        metrics = train_residual_fdm(
            dataset_dir=args.dataset,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            tensorboard_log_dir=args.tensorboard_log_dir,
            command=shell_join(argv),
            argv=argv,
        )
        _print_json(metrics)
        return 0

    def eval_dataset(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.evaluation.residual_fdm_dataset import evaluate_residual_fdm_dataset, shell_join

        metrics = evaluate_residual_fdm_dataset(
            dataset_dir=args.dataset,
            model_dir=args.model_dir,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            normalization=args.normalization,
            device=args.device,
            command=shell_join([sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]),
        )
        _print_json(metrics)
        return 0

    def eval_rollout(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.evaluation.residual_fdm_rollout import evaluate_residual_fdm_rollout, shell_join

        metrics = evaluate_residual_fdm_rollout(
            config_path=args.config,
            model_dir=args.model_dir,
            output_dir=args.output,
            seed=args.seed,
            backend=args.backend,
            device=args.device,
            checkpoint=args.checkpoint,
            normalization=args.normalization,
            generate_gif=not args.no_gif,
            gif_fps=args.gif_fps,
            gif_max_frames=args.gif_max_frames,
            command=shell_join([sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]),
        )
        _print_json(metrics)
        return 0

    def benchmark(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.evaluation.benchmark import (
            parse_controllers,
            parse_learned_mppi_overrides,
            parse_mppi_overrides,
            run_benchmark,
            shell_join,
        )

        summary = run_benchmark(
            config_path=args.config,
            scenario_name=args.scenario_name,
            output_dir=args.output,
            episodes=args.episodes,
            base_seed=args.base_seed,
            backend=args.backend,
            controllers=parse_controllers(args.controllers),
            fdm_model_dir=args.fdm_model_dir,
            fdm_checkpoint=args.fdm_checkpoint,
            fdm_normalization=args.fdm_normalization,
            fdm_device=args.fdm_device,
            fdm_residual_gain=args.fdm_residual_gain,
            mppi_overrides=parse_mppi_overrides(args.mppi_override),
            learned_mppi_overrides=parse_learned_mppi_overrides(args.learned_mppi_override),
            command=shell_join([sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]),
            argv=[sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]],
        )
        _print_json(summary)
        return 0

    def report(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.reporting.pipeline_report import write_pipeline_report

        report = write_pipeline_report(
            output_path=args.output,
            run_path=args.run_path,
            dataset_path=args.dataset,
            training_path=args.training,
            benchmark_path=args.benchmark,
        )
        _print_json(report)
        return 0

    # ------------------------------------------------------------------
    # High-Level FDM (traj + risk) subcommands
    # ------------------------------------------------------------------

    def high_fdm_synth(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.high_level_fdm.synthetic import SyntheticConfig, HighLevelFdmSyntheticGenerator
        from b2_fdm_mppi.high_level_fdm.synthetic_dataset_builder import (
            DatasetBuildSpec,
            build_synthetic_dataset,
        )

        config = _load_high_fdm_config(args.config)
        synth_cfg = _build_synthetic_config(config.get("synthetic", {}))
        generator = HighLevelFdmSyntheticGenerator(synth_cfg)
        spec = DatasetBuildSpec(
            train_samples=int(args.train_samples),
            val_samples=int(args.val_samples),
            test_samples=int(args.test_samples),
            base_seed=int(args.base_seed),
        )
        summary = build_synthetic_dataset(
            output_dir=args.output,
            generator=generator,
            spec=spec,
        )
        _print_json(summary)
        return 0

    def high_fdm_train(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.high_level_fdm.losses import HighLevelFdmLossConfig
        from b2_fdm_mppi.high_level_fdm.trainer import (
            HighLevelFdmTrainerConfig,
            train_high_level_fdm,
        )

        config = _load_high_fdm_config(args.config) if args.config else {}
        trainer_config = _build_trainer_config(
            config.get("trainer", {}),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
        )
        loss_config = _build_loss_config(config.get("loss", {}))
        model_overrides = _build_model_overrides(config.get("model", {}))
        metrics = train_high_level_fdm(
            dataset_dir=args.dataset,
            output_dir=args.output,
            trainer_config=trainer_config,
            loss_config=loss_config,
            model_overrides=model_overrides,
        )
        _print_json(metrics)
        return 0

    def high_fdm_eval(self, args: argparse.Namespace) -> int:
        from b2_fdm_mppi.high_level_fdm.trainer import evaluate_high_level_fdm

        metrics = evaluate_high_level_fdm(
            dataset_dir=args.dataset,
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            split=args.split,
            device=args.device,
        )
        _print_json(metrics)
        return 0

    def high_fdm_rollout_smoke(self, args: argparse.Namespace) -> int:
        import numpy as np
        import torch

        from b2_fdm_mppi.high_level_fdm.dataset import load_manifest
        from b2_fdm_mppi.high_level_fdm.rollout import (
            HighLevelFdmRollout,
            high_level_fdm_cost,
        )
        from b2_fdm_mppi.high_level_fdm.schema import HighLevelFdmSchema

        manifest = load_manifest(args.dataset)
        schema = HighLevelFdmSchema.from_dict(manifest["schema"])
        rollout = HighLevelFdmRollout.from_checkpoint(
            args.checkpoint, device=args.device, schema=schema
        )
        history = torch.zeros(schema.history_length, 6, dtype=torch.float32)
        map_patch = torch.zeros(
            schema.map_channels, schema.map_height, schema.map_width, dtype=torch.float32
        )
        controls = torch.zeros(
            int(args.num_samples), schema.horizon, 3, dtype=torch.float32
        )
        # Non-zero probe commands to exercise the decoder.
        controls[:, :, 0] = torch.linspace(0.0, 0.5, int(args.num_samples)).view(-1, 1)
        result = rollout.predict_batch(history, map_patch, controls)
        cost = high_level_fdm_cost(
            risk_prob=result.risk_prob,
            pose_delta=result.pose_delta,
            risk_channel_weights=[1.0] * schema.num_risk_channels(),
            goal_xy_body=[1.0, 0.0],
            goal_weight=1.0,
            step_weight_decay=0.95,
        )
        summary = {
            "dataset_dir": str(args.dataset),
            "checkpoint": str(args.checkpoint),
            "device": str(args.device),
            "num_samples": int(args.num_samples),
            "horizon": int(schema.horizon),
            "pose_param_shape": list(result.pose_param.shape),
            "pose_delta_shape": list(result.pose_delta.shape),
            "risk_prob_shape": list(result.risk_prob.shape),
            "cost_shape": list(cost.shape),
            "cost_min": float(cost.min()),
            "cost_max": float(cost.max()),
            "cost_mean": float(cost.mean()),
            "finite": bool(torch.isfinite(cost).all().item())
            and bool(torch.isfinite(result.pose_param).all().item())
            and bool(torch.isfinite(result.risk_prob).all().item()),
        }
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _print_json(summary)
        return 0


# ---------------------------------------------------------------------------
# High-Level FDM helpers
# ---------------------------------------------------------------------------

def _load_high_fdm_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"high-fdm config {path} must be a YAML mapping")
    return payload


def _build_synthetic_config(payload: dict):
    from b2_fdm_mppi.high_level_fdm.synthetic import synthetic_config_from_dict

    return synthetic_config_from_dict(payload or {})


def _build_trainer_config(
    payload: dict,
    *,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    seed: int | None,
    device: str | None,
    num_workers: int | None,
):
    from b2_fdm_mppi.high_level_fdm.trainer import HighLevelFdmTrainerConfig

    config = HighLevelFdmTrainerConfig(
        epochs=int(payload.get("epochs", 20)),
        batch_size=int(payload.get("batch_size", 64)),
        learning_rate=float(payload.get("learning_rate", 1e-3)),
        weight_decay=float(payload.get("weight_decay", 1e-5)),
        min_learning_rate=float(payload.get("min_learning_rate", 1e-5)),
        grad_clip=float(payload.get("grad_clip", 1.0)),
        num_workers=int(payload.get("num_workers", 0)),
        seed=int(payload.get("seed", 123)),
        device=str(payload.get("device", "cpu")),
        amp=bool(payload.get("amp", False)),
        log_every=int(payload.get("log_every", 50)),
    )
    if epochs is not None:
        config.epochs = int(epochs)
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if learning_rate is not None:
        config.learning_rate = float(learning_rate)
    if seed is not None:
        config.seed = int(seed)
    if device is not None:
        config.device = str(device)
    if num_workers is not None:
        config.num_workers = int(num_workers)
    return config


def _build_loss_config(payload: dict):
    from b2_fdm_mppi.high_level_fdm.losses import HighLevelFdmLossConfig

    return HighLevelFdmLossConfig(
        lambda_pose=float(payload.get("lambda_pose", 1.0)),
        lambda_risk=float(payload.get("lambda_risk", 1.0)),
        lambda_smooth=float(payload.get("lambda_smooth", 0.0)),
        huber_delta=float(payload.get("huber_delta", 1.0)),
        step_weight_decay=(
            None if payload.get("step_weight_decay") is None
            else float(payload.get("step_weight_decay", 0.98))
        ),
        risk_pos_weight=(
            None if payload.get("risk_pos_weight") is None
            else tuple(float(v) for v in payload.get("risk_pos_weight"))
        ),
    )


def _build_model_overrides(payload: dict) -> dict:
    allowed = {
        "d_hist",
        "d_map",
        "d_ctx",
        "d_hidden",
        "pose_head_hidden",
        "risk_head_hidden",
        "dropout",
    }
    overrides: dict = {}
    for key, value in (payload or {}).items():
        if key in allowed:
            overrides[key] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FDM-MPPI slim reproducible pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="Run a profile-defined MPPI experiment")
    experiment.add_argument("--profile", default="configs/experiment.yaml")
    experiment.add_argument("--controller", default=None)
    experiment.add_argument("--seed", type=int, default=None)
    experiment.add_argument("--backend", choices=["cuda", "numpy", "torch"], default=None)
    experiment.add_argument("--output", default=None)
    experiment.add_argument("--results-dir", default=None)
    experiment.add_argument("--model-dir", default=None)
    experiment.add_argument("--checkpoint", default=None)
    experiment.add_argument("--normalization", default=None)
    experiment.add_argument("--device", default=None)
    experiment.add_argument("--residual-gain", type=float, default=None)
    experiment.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    experiment.add_argument("--animation", action=argparse.BooleanOptionalAction, default=None)
    experiment.set_defaults(handler="experiment")

    run = subparsers.add_parser("run", help="Run one MPPI simulation")
    run.add_argument("--config", default="configs/smoke.yaml")
    run.add_argument("--seed", type=int, default=123)
    run.add_argument("--backend", choices=["cuda", "numpy", "torch"], default=None)
    run.add_argument("--fdm-enabled", action="store_true")
    run.add_argument("--fdm-model-dir", default=None)
    run.add_argument("--fdm-checkpoint", default=None)
    run.add_argument("--fdm-normalization", default=None)
    run.add_argument("--fdm-device", default=None)
    run.add_argument("--fdm-residual-gain", type=float, default=None)
    run.set_defaults(handler="run")

    dataset = subparsers.add_parser("dataset", help="Collect, build, or validate oracle datasets")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    collect = dataset_sub.add_parser("collect", help="Collect oracle episodes")
    collect.add_argument("--config", required=True)
    collect.add_argument("--episodes", type=int, required=True)
    collect.add_argument("--base-seed", type=int, required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    collect.add_argument("--num-workers", type=int, default=1)
    collect.set_defaults(handler="dataset_collect")

    build = dataset_sub.add_parser("build", help="Build train/val/test splits")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--train-ratio", type=float, default=0.7)
    build.add_argument("--val-ratio", type=float, default=0.15)
    build.add_argument("--test-ratio", type=float, default=0.15)
    build.add_argument("--seed", type=int, default=123)
    build.set_defaults(handler="dataset_build")

    validate = dataset_sub.add_parser("validate", help="Validate dataset splits")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--output", default=None)
    validate.set_defaults(handler="dataset_validate")

    train = subparsers.add_parser("train", help="Train residual FDM")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--device", default="cpu")
    train.add_argument("--tensorboard-log-dir", default=None)
    train.set_defaults(handler="train")

    eval_parser = subparsers.add_parser("eval", help="Evaluate trained residual FDM")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_dataset = eval_sub.add_parser("dataset", help="Evaluate checkpoint on dataset splits")
    eval_dataset.add_argument("--dataset", required=True)
    eval_dataset.add_argument("--model-dir", required=True)
    eval_dataset.add_argument("--output", required=True)
    eval_dataset.add_argument("--checkpoint", default="best_model.pt")
    eval_dataset.add_argument("--normalization", default="normalization.npz")
    eval_dataset.add_argument("--device", default="cpu")
    eval_dataset.set_defaults(handler="eval_dataset")

    eval_rollout = eval_sub.add_parser("rollout", help="Evaluate open-loop rollout replay")
    eval_rollout.add_argument("--config", required=True)
    eval_rollout.add_argument("--model-dir", required=True)
    eval_rollout.add_argument("--output", required=True)
    eval_rollout.add_argument("--seed", type=int, default=123)
    eval_rollout.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    eval_rollout.add_argument("--device", default="cpu")
    eval_rollout.add_argument("--checkpoint", default="model.pt")
    eval_rollout.add_argument("--normalization", default="normalization.npz")
    eval_rollout.add_argument("--no-gif", action="store_true")
    eval_rollout.add_argument("--gif-fps", type=int, default=8)
    eval_rollout.add_argument("--gif-max-frames", type=int, default=180)
    eval_rollout.set_defaults(handler="eval_rollout")

    benchmark = subparsers.add_parser("benchmark", help="Run nominal vs learned closed-loop benchmark")
    benchmark.add_argument("--config", default="configs/benchmark.yaml")
    benchmark.add_argument("--scenario-name", default="standard")
    benchmark.add_argument("--output", default="results/benchmark/standard_seed123")
    benchmark.add_argument("--episodes", type=int, default=1)
    benchmark.add_argument("--base-seed", type=int, default=123)
    benchmark.add_argument("--backend", choices=["numpy", "cuda", "torch"], default=None)
    benchmark.add_argument("--controllers", default="nominal,learned")
    benchmark.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    benchmark.add_argument("--fdm-checkpoint", default="best_model.pt")
    benchmark.add_argument("--fdm-normalization", default="normalization.npz")
    benchmark.add_argument("--fdm-device", default=None)
    benchmark.add_argument("--fdm-residual-gain", type=float, default=1.0)
    benchmark.add_argument("--mppi-override", action="append", default=[], metavar="KEY=VALUE")
    benchmark.add_argument("--learned-mppi-override", action="append", default=[], metavar="KEY=VALUE")
    benchmark.set_defaults(handler="benchmark")

    report = subparsers.add_parser("report", help="Write a compact pipeline report")
    report.add_argument("--output", default="results/reports/pipeline_report.json")
    report.add_argument("--run-path", default=None)
    report.add_argument("--dataset", default=None)
    report.add_argument("--training", default=None)
    report.add_argument("--benchmark", default=None)
    report.set_defaults(handler="report")

    high_fdm = subparsers.add_parser(
        "high-fdm",
        help="High-level learned forward dynamics model (trajectory + risk)",
    )
    high_fdm_sub = high_fdm.add_subparsers(dest="high_fdm_command", required=True)

    high_fdm_synth = high_fdm_sub.add_parser(
        "synth", help="Build a synthetic train/val/test dataset"
    )
    high_fdm_synth.add_argument("--config", default=None)
    high_fdm_synth.add_argument("--output", required=True)
    high_fdm_synth.add_argument("--train-samples", type=int, default=128)
    high_fdm_synth.add_argument("--val-samples", type=int, default=32)
    high_fdm_synth.add_argument("--test-samples", type=int, default=32)
    high_fdm_synth.add_argument("--base-seed", type=int, default=123)
    high_fdm_synth.set_defaults(handler="high_fdm_synth")

    high_fdm_train = high_fdm_sub.add_parser(
        "train", help="Train the high-level FDM"
    )
    high_fdm_train.add_argument("--dataset", required=True)
    high_fdm_train.add_argument("--output", required=True)
    high_fdm_train.add_argument("--config", default=None)
    high_fdm_train.add_argument("--epochs", type=int, default=None)
    high_fdm_train.add_argument("--batch-size", type=int, default=None)
    high_fdm_train.add_argument("--learning-rate", type=float, default=None)
    high_fdm_train.add_argument("--seed", type=int, default=None)
    high_fdm_train.add_argument("--device", default=None)
    high_fdm_train.add_argument("--num-workers", type=int, default=None)
    high_fdm_train.set_defaults(handler="high_fdm_train")

    high_fdm_eval = high_fdm_sub.add_parser(
        "eval", help="Evaluate a trained high-level FDM checkpoint"
    )
    high_fdm_eval.add_argument("--dataset", required=True)
    high_fdm_eval.add_argument("--checkpoint", required=True)
    high_fdm_eval.add_argument("--output", required=True)
    high_fdm_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    high_fdm_eval.add_argument("--device", default="cpu")
    high_fdm_eval.set_defaults(handler="high_fdm_eval")

    high_fdm_rollout = high_fdm_sub.add_parser(
        "rollout-smoke",
        help="Batched rollout + MPPI cost contract check",
    )
    high_fdm_rollout.add_argument("--dataset", required=True)
    high_fdm_rollout.add_argument("--checkpoint", required=True)
    high_fdm_rollout.add_argument("--output", default=None)
    high_fdm_rollout.add_argument("--device", default="cpu")
    high_fdm_rollout.add_argument("--num-samples", type=int, default=8)
    high_fdm_rollout.set_defaults(handler="high_fdm_rollout_smoke")

    return parser


def main(argv: Sequence[str] | None = None, *, commands: PipelineCommands | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command_handlers = commands or PipelineCommands()
    handler = getattr(command_handlers, args.handler)
    return int(handler(args) or 0)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2))


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


if __name__ == "__main__":
    raise SystemExit(main())
