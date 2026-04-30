from pathlib import Path

from b2_fdm_mppi.simulation import results_path


class FakeDatetime:
    @classmethod
    def now(cls):
        return cls()

    def strftime(self, fmt):
        assert fmt == "%Y-%m-%d_%H-%M-%S"
        return "2026-04-30_12-34-56"


def test_create_results_path_adds_timestamp_suffix_to_named_run(tmp_path, monkeypatch):
    monkeypatch.setattr(results_path._dt, "datetime", FakeDatetime)

    path = results_path.create_results_path(
        {
            "root": str(tmp_path),
            "run_name": "b2_omni_nominal",
            "timestamp_suffix": True,
        }
    )

    assert path == tmp_path / "b2_omni_nominal_2026-04-30_12-34-56"
    assert path.exists()


def test_create_results_path_rejects_overwrite_with_timestamp_suffix(tmp_path):
    try:
        results_path.create_results_path(
            {
                "root": str(tmp_path),
                "run_name": "latest",
                "timestamp_suffix": True,
                "overwrite": True,
            }
        )
    except ValueError as exc:
        assert "timestamp_suffix" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
