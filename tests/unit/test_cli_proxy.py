"""Proxy CLI displays resource metadata, never proxy credentials."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from click.testing import CliRunner

import mobilerun.cli.proxy.group as proxy
from mobilerun.cli.app import cli


def test_proxy_list_paginates_and_status_is_value_safe(monkeypatch):
    listing = AsyncMock(
        side_effect=[
            NS(
                items=[NS(id="one", country=None, status=None, type=None)],
                pagination=NS(has_next=True),
            ),
            NS(
                items=[NS(id="two", country=None, status=None, type=None)],
                pagination=NS(has_next=False),
            ),
        ]
    )
    monkeypatch.setattr(proxy, "list_proxies", listing)
    monkeypatch.setattr(
        proxy,
        "get_proxy",
        AsyncMock(
            return_value=NS(status="active", country=None, password="never-print")
        ),
    )
    result = CliRunner().invoke(cli, ["proxy", "list"])
    assert result.exit_code == 0
    assert [line.rstrip() for line in result.output.splitlines()] == [
        "Proxies · 2",
        "",
        "one        ○ unknown",
        "two        ○ unknown",
    ]
    listing.assert_awaited_with(page=2, page_size=100, billing="paid")
    result = CliRunner().invoke(cli, ["proxy", "status", "one"])
    assert result.exit_code == 0
    assert "one" in result.output and "● active" in result.output


def test_proxy_buy_passes_explicit_billing_and_redacts_credentials(monkeypatch):
    buy = AsyncMock(return_value=NS(id="new", password="never-print"))
    monkeypatch.setattr(proxy, "buy_proxy", buy)
    result = CliRunner().invoke(
        cli, ["proxy", "buy", "--country", "DE", "--billing", "paid", "--yes"]
    )
    assert result.exit_code == 0
    assert result.output == "✓ Bought proxy new (DE)\n"
    buy.assert_awaited_once_with(
        "DE",
        type="residential",
        billing="paid",
        checkout="browser",
        payment_timeout=600,
    )
