"""CLI entry points, completion and help are isolated from unrelated failures."""

import json
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli


def distribution(path: Path, name: str, entry: str) -> None:
    """Expose real metadata to importlib without installing a package."""
    info = path / f"{name}-1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {name}\nVersion: 1\n")
    (info / "entry_points.txt").write_text(f"[mobilerun.cli]\n{entry}\n")


def test_malformed_distribution_does_not_disable_healthy_plugins(tmp_path, monkeypatch):
    distribution(tmp_path, "invalidmetadata", "malformed-line")
    distribution(tmp_path, "validmetadata", "healthy = mobilerun.cli.doctor:doctor")
    monkeypatch.syspath_prepend(str(tmp_path))
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "healthy" in help_result.stdout and "device" in help_result.stdout
    assert "warning" in help_result.stderr.lower()
    assert "malformed-line" not in help_result.stderr
    assert runner.invoke(cli, ["healthy", "--help"]).exit_code == 0


def test_plugins_list_skips_malformed_distribution(tmp_path, monkeypatch):
    distribution(tmp_path, "invalidmetadata", "malformed-line")
    distribution(tmp_path, "validmetadata", "healthy = mobilerun.cli.doctor:doctor")
    monkeypatch.syspath_prepend(str(tmp_path))
    result = CliRunner().invoke(cli, ["plugins", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert "warning" in result.stderr.lower()
    assert "healthy" in {row["name"] for row in json.loads(result.stdout)}


def test_root_completion_ignores_broken_plugin_target(tmp_path, monkeypatch):
    distribution(tmp_path, "brokencompletion", "broken = nonexistent_target:command")
    monkeypatch.syspath_prepend(str(tmp_path))
    ctx = click.Context(cli)
    assert "device" in {item.value for item in cli.shell_complete(ctx, "")}
    assert "--debug" in {item.value for item in cli.shell_complete(ctx, "--d")}


def test_import_main_returns_without_running_cli():
    result = subprocess.run(
        [sys.executable, "-c", "import mobilerun.__main__; print('imported')"],
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "imported"


@pytest.mark.parametrize("command", [["account"], ["account", "logout"]])
def test_account_help_with_invalid_unrelated_setting(command, monkeypatch):
    monkeypatch.setenv("MOBILERUN_PORTAL_REMOTE_PORT", "invalid")
    result = subprocess.run(
        [sys.executable, "-m", "mobilerun", *command, "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
