from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import click
from click.testing import CliRunner

from mobilerun.cli.base import choices, menu
from mobilerun.cli.proxy import group
from mobilerun.models.billing import Billing


def test_cloud_devices_order_labels_and_no_commerce(monkeypatch):
    listing = AsyncMock(
        return_value=[
            NS(id="d1", name="Sleeping", state="stopped"),
            NS(id="d2", name="Ready", state="running"),
        ]
    )
    monkeypatch.setattr(choices, "list_devices", listing, raising=False)
    result = choices.cloud_devices(click.Context(click.Command("test")))
    assert [item.value for item in result] == ["d2", "d1"]
    assert "Ready" in result[0].label and "running" in (
        result[0].hint or result[0].label
    )
    assert "stopped" in (result[1].hint or result[1].label)
    listing.assert_awaited_once_with(billing=Billing.NO_COMMERCE, base_url=None)


def test_resource_picker_status_and_order(monkeypatch):
    monkeypatch.setattr(
        choices,
        "list_mailboxes",
        AsyncMock(
            return_value=[
                NS(id="m1", address="a", status="failed"),
                NS(id="m2", address="b", status="active"),
            ]
        ),
    )
    monkeypatch.setattr(
        choices,
        "list_numbers",
        AsyncMock(
            return_value=[
                NS(id="n1", phone_number="1", state="failed"),
                NS(id="n2", phone_number="2", state="active"),
            ]
        ),
    )
    listing = AsyncMock(
        side_effect=[
            NS(
                items=[NS(id="p1", country="us", status="inactive", type="mobile")],
                pagination=NS(has_next=True),
            ),
            NS(
                items=[NS(id="p2", country="gb", status="active", type="residential")],
                pagination=NS(has_next=False),
            ),
        ]
    )
    monkeypatch.setattr(choices, "list_proxies", listing)
    ctx = click.Context(click.Command("test"))
    proxy_choices = choices.proxies(ctx)
    for result, ids, status in [
        (choices.mailboxes(ctx), ["m2", "m1"], "failed"),
        (choices.numbers(ctx), ["n2", "n1"], "failed"),
        (proxy_choices, ["p2", "p1"], "inactive"),
    ]:
        assert [item.value for item in result] == ids
        assert status in (result[1].hint or result[1].label)
    assert "residential" in proxy_choices[0].hint
    assert listing.await_count == 2


def test_proxy_bind_picks_device_and_reports_progress(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(
        choices,
        "list_devices",
        AsyncMock(return_value=[NS(id="dev", name="Phone", state="running")]),
        raising=False,
    )
    seen = []

    def select(label, options):
        seen.append((label, [item.value for item in options]))
        return options[0].value

    monkeypatch.setattr(menu, "select_prompt", select)
    phone = NS(close=AsyncMock())
    extension = NS(bind=AsyncMock(), status=AsyncMock(return_value=NS(connected=True)))
    monkeypatch.setattr(group, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(group, "CloudFixedProxy", lambda proxy_id: extension)
    result = CliRunner().invoke(group.proxy, ["bind", "proxy"])
    assert result.exit_code == 0, result.output
    assert seen == [("Cloud device ID", ["dev"])]
    assert (
        "Binding proxy" in result.output and "Waiting for connection" in result.output
    )
    assert "✓ Bound proxy proxy to dev" in result.output
    extension.bind.assert_awaited_once_with(phone, force=False)


def test_proxy_bind_conflict_asks_before_retry(monkeypatch):
    from mobilerun.core.proxy import UnmanagedProxyConflict

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    phone = NS(close=AsyncMock())
    extension = NS(
        bind=AsyncMock(
            side_effect=[
                UnmanagedProxyConflict(
                    "Device has an unmanaged active proxy; use force=True to take over"
                ),
                None,
            ]
        ),
        status=AsyncMock(return_value=NS(connected=True)),
    )
    monkeypatch.setattr(group, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(group, "CloudFixedProxy", lambda proxy_id: extension)
    monkeypatch.setattr(
        group, "confirm_prompt_async", AsyncMock(return_value=True), raising=False
    )
    result = CliRunner().invoke(group.proxy, ["bind", "proxy", "dev"])
    assert result.exit_code == 0, result.output
    assert [call.kwargs["force"] for call in extension.bind.await_args_list] == [
        False,
        True,
    ]


def test_proxy_bind_conflict_declined(monkeypatch):
    from mobilerun.core.proxy import UnmanagedProxyConflict

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    phone = NS(close=AsyncMock())
    extension = NS(
        bind=AsyncMock(
            side_effect=UnmanagedProxyConflict(
                "Device has an unmanaged active proxy; use force=True to take over"
            )
        )
    )
    monkeypatch.setattr(group, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(group, "CloudFixedProxy", lambda proxy_id: extension)
    monkeypatch.setattr(group, "confirm_prompt_async", AsyncMock(return_value=False))
    result = CliRunner().invoke(group.proxy, ["bind", "proxy", "dev"])
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    extension.bind.assert_awaited_once_with(phone, force=False)
    phone.close.assert_awaited_once()


def test_proxy_bind_noninteractive_conflict_recommends_force(monkeypatch):
    from mobilerun.core.proxy import UnmanagedProxyConflict

    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    phone = NS(close=AsyncMock())
    extension = NS(
        bind=AsyncMock(
            side_effect=UnmanagedProxyConflict(
                "Device has an unmanaged active proxy; use force=True to take over"
            )
        )
    )
    monkeypatch.setattr(group, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(group, "CloudFixedProxy", lambda proxy_id: extension)
    result = CliRunner().invoke(group.proxy, ["bind", "proxy", "dev"])
    assert result.exit_code != 0
    assert "--force" in result.output
    assert "force=True" not in result.output


def test_proxy_bind_force_skips_confirmation(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    phone = NS(close=AsyncMock())
    extension = NS(bind=AsyncMock(), status=AsyncMock(return_value=NS(connected=True)))
    prompt = AsyncMock()
    monkeypatch.setattr(group, "CloudDevice", lambda **kwargs: phone)
    monkeypatch.setattr(group, "CloudFixedProxy", lambda proxy_id: extension)
    monkeypatch.setattr(group, "confirm_prompt_async", prompt)
    result = CliRunner().invoke(group.proxy, ["bind", "proxy", "dev", "--force"])
    assert result.exit_code == 0
    extension.bind.assert_awaited_once_with(phone, force=True)
    prompt.assert_not_awaited()


def test_proxy_confirmation_default_and_escape(monkeypatch):
    import asyncio

    from mobilerun.cli.base import prompts

    select = AsyncMock(side_effect=["yes", "no", None])
    monkeypatch.setattr(prompts, "select_prompt_async", select)

    async def scenario():
        assert await prompts.confirm_prompt_async("Replace?", default=True)
        assert not await prompts.confirm_prompt_async("Replace?")
        assert not await prompts.confirm_prompt_async("Replace?", default=True)

    asyncio.run(scenario())
    assert [call.kwargs["default"] for call in select.await_args_list] == [
        "yes",
        "no",
        "yes",
    ]
