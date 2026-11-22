from datetime import datetime, timezone
from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.models import WorkspaceFile


def test_workspace_cli_upload_list_and_delete_safety(monkeypatch, tmp_path):
    from mobilerun.cli.workspace import group

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
    from mobilerun.cli.devices.base import selection as files
    from mobilerun.models import DeviceFile

    driver = AsyncMock()
    driver.push_file.return_value = DeviceFile(
        "/sdcard/DCIM/Camera/a.png", "a.png", 3, None
    )
    driver.list_files.return_value = [driver.push_file.return_value]
    driver.pull_file.return_value = tmp_path / "copy.png"
    monkeypatch.setattr(files, "connect_target", AsyncMock(return_value=driver))
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
    from mobilerun.cli.devices.base import selection as files
    from mobilerun.core.workspace import Workspace
    from mobilerun.models import DeviceFile

    item = WorkspaceFile("f1", "a.png", "image/png", 3, datetime.now(timezone.utc))
    driver = AsyncMock()
    transfer = AsyncMock(return_value=DeviceFile("/sdcard/a.png", "a.png", 3, None))
    connect = AsyncMock(return_value=driver)
    monkeypatch.setattr(files, "connect_target", connect)
    monkeypatch.setattr(Workspace, "list_files", AsyncMock(return_value=[item]))
    monkeypatch.setattr(Workspace, "push_file", transfer)
    result = CliRunner().invoke(
        cli, ["device", "file", "push", "--file-id", "f1", "serial", "--android"]
    )
    assert result.exit_code == 0, result.output
    assert connect.await_args.args[0] == "serial"
    transfer.assert_awaited_once_with(item, driver, None)
