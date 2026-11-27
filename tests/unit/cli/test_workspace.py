"""Workspace resources and device file and app commands."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.base import menu, safety
from mobilerun.cli.devices import device
from mobilerun.cli.devices.base import selection, source
from mobilerun.cli.workspace import group
from mobilerun.clients.local import login_store
from mobilerun.core.state.ios import IOSStateProvider
from mobilerun.core.workspace import Workspace
from mobilerun.models import App, DeviceFile, WorkspaceFile


def test_workspace_cli_upload_list_and_delete_safety(monkeypatch, tmp_path):

    item = WorkspaceFile("f1", "a.png", "image/png", 3, datetime.now(timezone.utc))
    monkeypatch.setattr(group.Workspace, "upload_file", AsyncMock(return_value=item))
    monkeypatch.setattr(group.Workspace, "list_files", AsyncMock(return_value=[item]))
    monkeypatch.setattr(group.Workspace, "delete_file", AsyncMock())
    path = tmp_path / "a.png"
    path.write_bytes(b"png")
    runner = CliRunner()
    result = runner.invoke(cli, ["workspace", "files", "upload", str(path)])
    assert result.exit_code == 0, result.output
    assert "f1" in result.output
    result = runner.invoke(cli, ["workspace", "files", "list", "--json"])
    assert result.exit_code == 0 and '"f1"' in result.output
    result = runner.invoke(cli, ["workspace", "files", "remove", "f1"])
    assert result.exit_code != 0
    group.Workspace.delete_file.assert_not_awaited()
    result = runner.invoke(cli, ["workspace", "files", "remove", "f1", "--yes"])
    assert result.exit_code == 0, result.output
    group.Workspace.delete_file.assert_awaited_once()


def test_device_file_cli_push_list_pull_rm(monkeypatch, tmp_path):

    driver = AsyncMock()
    driver.push_file.return_value = DeviceFile(
        "/sdcard/DCIM/Camera/a.png", "a.png", 3, None
    )
    driver.list_files.return_value = [driver.push_file.return_value]
    driver.pull_file.return_value = tmp_path / "copy.png"
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    path = tmp_path / "a.png"
    path.write_bytes(b"png")
    runner = CliRunner()
    for args in (
        ["push", str(path), "serial", "--android"],
        ["list", "/sdcard/DCIM/Camera", "serial", "--android", "--json"],
        [
            "pull",
            "/sdcard/DCIM/Camera/a.png",
            str(tmp_path / "copy.png"),
            "serial",
            "--android",
        ],
        ["rm", "/sdcard/DCIM/Camera/a.png", "serial", "--android", "--yes"],
    ):
        result = runner.invoke(cli, ["device", "file", *args])
        assert result.exit_code == 0, result.output
    assert (
        driver.push_file.await_count
        == driver.list_files.await_count
        == driver.pull_file.await_count
        == driver.delete_file.await_count
        == 1
    )


def test_workspace_push_accepts_positional_device(monkeypatch):

    item = WorkspaceFile("f1", "a.png", "image/png", 3, datetime.now(timezone.utc))
    driver = AsyncMock()
    transfer = AsyncMock(return_value=DeviceFile("/sdcard/a.png", "a.png", 3, None))
    connect = AsyncMock(return_value=driver)
    monkeypatch.setattr(selection, "connect_target", connect)
    monkeypatch.setattr(Workspace, "list_files", AsyncMock(return_value=[item]))
    monkeypatch.setattr(Workspace, "push_file", transfer)
    result = CliRunner().invoke(
        cli, ["device", "file", "push", "--file-id", "f1", "serial", "--android"]
    )
    assert result.exit_code == 0, result.output
    assert connect.await_args.args[0] == "serial"
    transfer.assert_awaited_once_with(item, driver, None)


def test_workspace_groups_and_old_commands_removed():
    runner = CliRunner()
    for args in (
        ["workspace", "--help"],
        ["workspace", "files", "--help"],
        ["workspace", "apps", "--help"],
    ):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
    assert runner.invoke(cli, ["files", "--help"]).exit_code != 0
    assert runner.invoke(cli, ["device", "upload", "--help"]).exit_code != 0


def test_workspace_app_operations_and_removed_commands(monkeypatch, tmp_path):

    app = App("org.example", "1", "Example", id="app1")
    monkeypatch.setattr(group.Workspace, "list_apps", AsyncMock(return_value=[app]))
    monkeypatch.setattr(group.Workspace, "upload_app", AsyncMock(return_value=app))
    monkeypatch.setattr(group.Workspace, "delete_app", AsyncMock())
    path = tmp_path / "test.apk"
    path.write_bytes(b"apk")
    runner = CliRunner()
    for args in (["list", "--json"], ["upload", str(path)]):
        result = runner.invoke(cli, ["workspace", "apps", *args])
        assert result.exit_code == 0, result.output
    assert (
        '"app1"' in runner.invoke(cli, ["workspace", "apps", "list", "--json"]).output
    )
    assert runner.invoke(cli, ["workspace", "apps", "remove", "app1"]).exit_code != 0
    group.Workspace.delete_app.assert_not_awaited()
    assert (
        runner.invoke(cli, ["workspace", "apps", "remove", "app1", "--yes"]).exit_code
        == 0
    )
    group.Workspace.delete_app.assert_awaited_once_with("app1")
    for group_name, command in (
        ("files", "push"),
        ("files", "download"),
        ("apps", "install"),
        ("apps", "show"),
        ("apps", "delete"),
    ):
        assert runner.invoke(cli, ["workspace", group_name, command]).exit_code != 0


def test_install_workspace_app(monkeypatch):

    app = App("org.example", "1", "Example", id="app1")
    device = AsyncMock()
    device.install_app.return_value = app
    monkeypatch.setattr(Workspace, "get_app", AsyncMock(return_value=app))
    connect = AsyncMock(return_value=device)
    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(
        cli, ["device", "app", "install", "--app-id", "app1", "phone", "--cloud"]
    )
    assert result.exit_code == 0, result.output
    assert connect.await_args.args[0] == "phone"
    device.install_app.assert_awaited_once_with(
        app, reinstall=False, grant_permissions=True
    )


def test_device_source_exclusive(monkeypatch, tmp_path):
    apk = tmp_path / "file.apk"
    apk.write_bytes(b"data")
    runner = CliRunner()
    for command, flag in (("install", "--app-id"), ("push", "--file-id")):
        result = runner.invoke(
            cli,
            [
                "device",
                "app" if command == "install" else "file",
                command,
                str(apk),
                "serial",
                flag,
                "id",
                "--android",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "either a local path" in result.output


def test_workspace_picker_uses_free_lists(monkeypatch):

    monkeypatch.setattr(source, "is_interactive", lambda: True)
    monkeypatch.setattr(
        source.Workspace,
        "list_apps",
        AsyncMock(return_value=[App("pkg", "1", "Name", id="a")]),
    )
    answers = iter(["workspace", "a"])
    monkeypatch.setattr(source, "select_prompt", lambda *args: next(answers))
    assert source.select_source(None, None, apps=True) == (None, "a")
    source.Workspace.list_apps.assert_awaited_once()


def test_install_passes_options_and_closes_device(monkeypatch, tmp_path):

    apk = tmp_path / "example.apk"
    apk.write_bytes(b"apk")
    driver = AsyncMock()

    driver.install_apk.return_value = App("org.example", "1.0", "Example")
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    result = CliRunner().invoke(
        device,
        [
            "app",
            "install",
            str(apk),
            "serial",
            "--android",
            "--reinstall",
            "--no-grant",
        ],
    )
    assert result.exit_code == 0, result.output
    driver.install_apk.assert_awaited_once_with(
        str(apk), reinstall=True, grant_permissions=False
    )
    driver.close.assert_awaited_once_with()
    assert "org.example 1.0" in result.output


def test_uninstall_closes_device_even_on_failure(monkeypatch):

    driver = AsyncMock()
    driver.uninstall_app.side_effect = RuntimeError("failure")
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    result = CliRunner().invoke(
        device, ["app", "uninstall", "example.app", "serial", "--android", "--yes"]
    )
    assert result.exit_code == 1
    assert "--debug" in result.output
    driver.uninstall_app.assert_awaited_once_with("example.app")
    driver.close.assert_awaited_once_with()


def test_uninstall_refuses_noninteractive_without_yes(monkeypatch):

    connect = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", connect)
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    result = CliRunner().invoke(device, ["app", "uninstall", "example.app", "serial"])
    assert result.exit_code == 1
    assert (
        "Refusing to uninstall example.app without confirmation; pass --yes."
        in result.output
    )
    connect.assert_not_awaited()


def test_uninstall_interactive_cancel_before_connect(monkeypatch):

    connect = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", connect)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=False))
    result = CliRunner().invoke(
        device, ["app", "uninstall", "example.app", "serial"], input="n\n"
    )
    assert result.exit_code == 0
    assert "example.app" in result.output and "serial" in result.output
    connect.assert_not_awaited()


@pytest.mark.parametrize(
    "interactive,args", [(True, []), (False, ["-y"])], ids=["prompt", "flag"]
)
def test_confirmed_uninstall(monkeypatch, interactive, args):
    driver = AsyncMock()
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=driver))
    monkeypatch.setattr(menu, "is_interactive", lambda: interactive)
    prompt = AsyncMock(return_value=True)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    result = CliRunner().invoke(
        device, ["app", "uninstall", "example.app", "serial", *args]
    )
    assert result.exit_code == 0, result.output
    assert prompt.await_count == int(interactive)
    driver.uninstall_app.assert_awaited_once_with("example.app")


@pytest.fixture
def default_config(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)


@pytest.mark.usefixtures("default_config")
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


@pytest.mark.usefixtures("default_config")
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


@pytest.mark.usefixtures("default_config")
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


@pytest.mark.usefixtures("default_config")
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


@pytest.mark.usefixtures("default_config")
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
