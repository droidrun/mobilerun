from click.testing import CliRunner

from mobilerun.cli.app import cli


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
    from unittest.mock import AsyncMock

    from mobilerun.cli.workspace import group
    from mobilerun.models import App

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
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices.base import selection as module
    from mobilerun.core.workspace import Workspace
    from mobilerun.models import App

    app = App("org.example", "1", "Example", id="app1")
    device = AsyncMock()
    device.install_app.return_value = app
    monkeypatch.setattr(Workspace, "get_app", AsyncMock(return_value=app))
    connect = AsyncMock(return_value=device)
    monkeypatch.setattr(module, "connect_target", connect)
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
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices.base import source
    from mobilerun.models import App

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
