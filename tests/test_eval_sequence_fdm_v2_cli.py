import subprocess
import sys


def test_eval_cli_help():
    result = subprocess.run(
        [sys.executable, "tools/eval_sequence_fdm_v2.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
