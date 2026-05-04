import subprocess
import sys


def test_train_cli_help():
    result = subprocess.run(
        [sys.executable, "tools/train_sequence_fdm_v2.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
