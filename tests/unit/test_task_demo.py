import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock


def test_demo_creates_minute_phone_and_closes(monkeypatch, capsys):
    from tests.demos import task_demo as demo

    proxy_status = AsyncMock(return_value=SimpleNamespace(connected=False))
    proxy = SimpleNamespace(country="US", status=proxy_status)
    monkeypatch.setattr(demo, "CloudRotatingProxy", lambda **kwargs: proxy)
    phone = SimpleNamespace(
        device_id="phone",
        proxy=SimpleNamespace(
            status=AsyncMock(side_effect=AssertionError("Use proxy handle"))
        ),
        info=AsyncMock(return_value=SimpleNamespace()),
        do=AsyncMock(
            side_effect=[
                SimpleNamespace(task_id="open"),
                SimpleNamespace(
                    task_id="version", output=demo.AndroidVersion(version="15")
                ),
                SimpleNamespace(task_id="timed"),
            ]
        ),
        expect=AsyncMock(
            return_value=SimpleNamespace(task=SimpleNamespace(task_id="expect"))
        ),
        press_button=AsyncMock(),
    )

    class Device:
        def __init__(self, **kwargs):
            proxy = kwargs.pop("proxy")
            assert proxy is not None and proxy.country == "US"
            assert kwargs == {
                "name": "cloud-phone-demo",
                "billing": "paid",
                "plan": None,
            }

        async def __aenter__(self):
            return phone

        async def __aexit__(self, *args):
            phone.closed = True

    monkeypatch.setattr(demo, "CloudDevice", Device)
    asyncio.run(demo.main(yes=True))
    assert phone.closed
    phone.info.assert_not_awaited()
    proxy_status.assert_awaited_once()
    phone.press_button.assert_awaited_once_with("home")
    assert phone.do.await_count == 3
    assert "15" in capsys.readouterr().out


def test_task_demo_decline_does_not_create(monkeypatch):
    from unittest.mock import Mock

    from tests.demos import task_demo as demo

    create = Mock()
    monkeypatch.setattr(demo, "CloudDevice", create)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    asyncio.run(demo.main(yes=False))
    create.assert_not_called()


def test_included_plan_passed_to_phone_and_warns(monkeypatch):
    import pytest

    from mobilerun.models import Billing
    from tests.demos import task_demo as demo

    calls = []
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text) or "y")

    def phone(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop before creation")

    monkeypatch.setattr(demo, "CloudDevice", phone)
    with pytest.raises(RuntimeError, match="stop before creation"):
        asyncio.run(demo.main(billing=Billing.INCLUDED, plan="subscription"))
    assert calls[0]["billing"] == "included"
    assert calls[0]["plan"] == "subscription"
    assert "included quota only (no charges)" in prompts[0]
    assert "paid" not in prompts[0].lower()


def test_included_minute_is_rejected_before_creation(monkeypatch):
    import pytest

    from mobilerun.models import Billing
    from tests.demos import task_demo as demo

    create = AsyncMock()
    monkeypatch.setattr(demo, "CloudDevice", create)
    with pytest.raises(ValueError, match="subscription"):
        asyncio.run(demo.main(yes=True, billing=Billing.INCLUDED, plan="minute"))
    create.assert_not_called()
