from __future__ import annotations

from argparse import Namespace

from b2_fdm_mppi import cli


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Namespace]] = []

    def run(self, args: Namespace) -> int:
        self.calls.append(("run", args))
        return 17

    def dataset_collect(self, args: Namespace) -> int:
        self.calls.append(("dataset_collect", args))
        return 0


def test_main_dispatches_run_subcommand_to_pipeline_command():
    recorder = Recorder()

    status = cli.main(
        [
            "run",
            "--config",
            "configs/smoke.yaml",
            "--seed",
            "123",
            "--backend",
            "numpy",
        ],
        commands=recorder,
    )

    assert status == 17
    name, args = recorder.calls[0]
    assert name == "run"
    assert args.config == "configs/smoke.yaml"
    assert args.seed == 123
    assert args.backend == "numpy"


def test_main_dispatches_dataset_collect_subcommand():
    recorder = Recorder()

    status = cli.main(
        [
            "dataset",
            "collect",
            "--config",
            "configs/dataset.yaml",
            "--episodes",
            "2",
            "--base-seed",
            "123",
            "--output",
            "datasets/refactor_smoke",
        ],
        commands=recorder,
    )

    assert status == 0
    name, args = recorder.calls[0]
    assert name == "dataset_collect"
    assert args.config == "configs/dataset.yaml"
    assert args.episodes == 2
    assert args.base_seed == 123
    assert args.output == "datasets/refactor_smoke"
