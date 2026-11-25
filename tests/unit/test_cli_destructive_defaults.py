from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from mobilerun.cli.credentials import group as credentials
from mobilerun.cli.proxy import group as proxy


@pytest.mark.parametrize("tty", [True, False])
def test_proxy_replacement_requires_opt_in(monkeypatch, tty):
    phone = SimpleNamespace(close=AsyncMock())
    extension = SimpleNamespace(
        bind=AsyncMock(
            side_effect=[proxy.UnmanagedProxyConflict("use force=True"), None]
        ),
        status=AsyncMock(return_value=SimpleNamespace(connected=True)),
    )
    prompt = AsyncMock(side_effect=lambda message, *, default: default)
    monkeypatch.setattr(proxy.menu, "is_interactive", lambda: tty)
    monkeypatch.setattr(proxy, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(proxy, "CloudFixedProxy", lambda id: extension)
    monkeypatch.setattr(proxy, "confirm_prompt_async", prompt)
    result = CliRunner().invoke(proxy.proxy, ["bind", "new", "phone"])
    assert result.exit_code == (0 if tty else 1), result.output
    assert ("Cancelled." if tty else "--force") in result.output
    assert prompt.await_count == int(tty)
    extension.bind.assert_awaited_once_with(phone, force=False)
    phone.close.assert_awaited_once()


@pytest.mark.parametrize("tty", [True, False])
def test_confirmed_credential_delete_skips_field_selection(monkeypatch, tty):
    store = SimpleNamespace(
        list=AsyncMock(return_value=[]), delete=AsyncMock(), close=AsyncMock()
    )
    monkeypatch.setattr(credentials, "CloudCredentials", lambda **kwargs: store)
    monkeypatch.setattr(credentials, "is_interactive", lambda: tty)
    prompt = Mock(return_value=None)
    monkeypatch.setattr(credentials, "select_prompt", prompt)
    result = CliRunner().invoke(
        credentials.credentials, ["delete", "app", "work", "--yes"]
    )
    assert result.exit_code == 0, result.output
    prompt.assert_not_called()
    store.delete.assert_awaited_once_with("app", "work", None)
