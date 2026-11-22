from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.devices import device


def test_install_passes_options_and_closes_device(monkeypatch, tmp_path):
    from mobilerun.cli.devices.base import selection as install

    apk = tmp_path / "example.apk"
    apk.write_bytes(b"apk")
    driver = AsyncMock()
    from mobilerun.models.apps import App

    driver.install_apk.return_value = App("org.example", "1.0", "Example")
    monkeypatch.setattr(install, "connect_target", AsyncMock(return_value=driver))
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
    from mobilerun.cli.devices.base import selection as uninstall

    driver = AsyncMock()
    driver.uninstall_app.side_effect = RuntimeError("failure")
    monkeypatch.setattr(uninstall, "connect_target", AsyncMock(return_value=driver))
    result = CliRunner().invoke(
        device, ["app", "uninstall", "example.app", "serial", "--android", "--yes"]
    )
    assert result.exit_code == 1
    assert "--debug" in result.output
    driver.uninstall_app.assert_awaited_once_with("example.app")
    driver.close.assert_awaited_once_with()


def test_uninstall_refuses_noninteractive_without_yes(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.devices.base import selection as uninstall

    connect = AsyncMock()
    monkeypatch.setattr(uninstall, "connect_target", connect)
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    result = CliRunner().invoke(device, ["app", "uninstall", "example.app", "serial"])
    assert result.exit_code == 1
    assert (
        "Refusing to uninstall example.app without confirmation; pass --yes."
        in result.output
    )
    connect.assert_not_awaited()


def test_uninstall_interactive_cancel_before_connect(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.devices.base import selection as uninstall

    connect = AsyncMock()
    monkeypatch.setattr(uninstall, "connect_target", connect)
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=False))
    result = CliRunner().invoke(
        device, ["app", "uninstall", "example.app", "serial"], input="n\n"
    )
    assert result.exit_code == 0
    assert "example.app" in result.output and "serial" in result.output
    connect.assert_not_awaited()


def test_uninstall_interactive_yes(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.devices.base import selection as uninstall

    driver = AsyncMock()
    monkeypatch.setattr(uninstall, "connect_target", AsyncMock(return_value=driver))
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    from mobilerun.cli.base import safety

    monkeypatch.setattr(safety, "confirm_prompt_async", AsyncMock(return_value=True))
    result = CliRunner().invoke(
        device, ["app", "uninstall", "example.app", "serial"], input="y\n"
    )
    assert result.exit_code == 0, result.output
    driver.uninstall_app.assert_awaited_once_with("example.app")


def test_uninstall_yes_noninteractive(monkeypatch):
    from mobilerun.cli.base import menu
    from mobilerun.cli.devices.base import selection as uninstall

    driver = AsyncMock()
    monkeypatch.setattr(uninstall, "connect_target", AsyncMock(return_value=driver))
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    result = CliRunner().invoke(
        device, ["app", "uninstall", "example.app", "serial", "-y"]
    )
    assert result.exit_code == 0, result.output
    driver.uninstall_app.assert_awaited_once_with("example.app")
