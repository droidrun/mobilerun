"""Proxy CLI uses explicit resource-side binding, never device proxy setters."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.proxy.group import proxy


def test_bind_existing_proxy_passes_force_and_closes(monkeypatch):
    import importlib

    module = importlib.import_module("mobilerun.cli.proxy.group")
    phone = SimpleNamespace(close=AsyncMock())
    extension = SimpleNamespace(
        bind=AsyncMock(),
        status=AsyncMock(return_value=SimpleNamespace(connected=True)),
    )
    monkeypatch.setattr(module, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(module, "CloudFixedProxy", lambda id: extension)
    result = CliRunner().invoke(proxy, ["bind", "proxy-id", "phone-id", "--force"])
    assert result.exit_code == 0, result.output
    extension.bind.assert_awaited_once_with(phone, force=True)
    phone.close.assert_awaited_once()
