"""Exercise failure reporting without devices or cloud resources."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.devices.base import selection
from mobilerun.clients.local import login_store
from mobilerun.core.state.ios import IOSStateProvider
from mobilerun.core.workspace import Workspace
from mobilerun.models import App


@pytest.fixture(autouse=True)
def default_config(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)


def test_workspace_id_forwards_cloud_install_options(monkeypatch):
    app = App("pkg", "1", "App", id="app-id")
    device = AsyncMock()
    device.install_app.return_value = app
    monkeypatch.setattr(Workspace, "get_app", AsyncMock(return_value=app))
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=device))
    result = CliRunner().invoke(
        cli,
        [
            "device",
            "app",
            "install",
            "--cloud",
            "phone",
            "--app-id",
            "app-id",
            "--no-grant",
            "--reinstall",
        ],
    )
    assert result.exit_code == 0, result.output
    device.install_app.assert_awaited_once_with(
        app, reinstall=True, grant_permissions=False
    )


def test_local_workspace_id_rejected_before_connection_or_lookup(monkeypatch):
    lookup, connect = AsyncMock(), AsyncMock()
    monkeypatch.setattr(Workspace, "get_app", lookup)
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(
        cli, ["device", "app", "install", "--device", "usb", "--app-id", "id"]
    )
    assert result.exit_code != 0
    assert "local APK" in result.stderr
    connect.assert_not_awaited()
    lookup.assert_not_awaited()


def test_ios_cli_distinguishes_timeout_from_empty_tree(monkeypatch):
    device = MagicMock()
    device.close = AsyncMock()
    device._get_ui_tree = AsyncMock(side_effect=[TimeoutError(), TimeoutError(), {}])
    provider = IOSStateProvider(device)
    device._create_state_provider.return_value = provider
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=device))
    assert (
        "No UI elements available" in asyncio.run(provider.get_state()).formatted_text
    )
    args = ["device", "screen", "ui", "--ios", "http://ios.test"]
    failed = CliRunner().invoke(cli, args)
    assert failed.exit_code != 0
    assert "timed out" in failed.stderr.lower()
    empty = CliRunner().invoke(cli, args)
    assert empty.exit_code == 0 and "No UI elements found" in empty.stdout


def test_env_status_ignores_broken_credentials(monkeypatch, tmp_path):
    path = tmp_path / "broken"
    path.write_text("{")
    monkeypatch.setattr(login_store, "credential_path", lambda: path)
    result = CliRunner().invoke(
        cli, ["account", "status", "--json"], env={"MOBILERUN_API_KEY": "test-only-key"}
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["source"] == "environment"
    assert "test-only-key" not in result.output


def test_partial_json_upload_reports_each_input(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("first", "bad", "last")]
    for path in paths:
        path.touch()
    monkeypatch.setattr(
        Workspace,
        "upload_file",
        AsyncMock(
            side_effect=[
                SimpleNamespace(id="one", name="first"),
                ValueError("private detail"),
                SimpleNamespace(id="three", name="last"),
            ]
        ),
    )
    result = CliRunner().invoke(
        cli, ["workspace", "files", "upload", "--json", *map(str, paths)]
    )
    assert result.exit_code != 0
    first, failed, last = json.loads(result.stdout)
    assert first["id"] == "one" and last["id"] == "three"
    assert failed["path"] == str(paths[1]) and failed["status"] == "failed"
    assert "private detail" not in result.output
