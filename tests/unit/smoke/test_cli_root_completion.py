"""Exercise the installed console script's shell completion protocol."""

import subprocess
import sys
from pathlib import Path


def test_installed_cli_bash_completion(monkeypatch):
    monkeypatch.setenv("_MOBILERUN_COMPLETE", "bash_complete")
    monkeypatch.setenv("COMP_WORDS", "mobilerun ")
    monkeypatch.setenv("COMP_CWORD", "1")
    result = subprocess.run(
        [str(Path(sys.executable).with_name("mobilerun"))],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert {"plain,account", "plain,device", "plain,macro"} <= set(
        result.stdout.splitlines()
    )
