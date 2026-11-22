import asyncio
from unittest.mock import AsyncMock


def test_macro_records_replays_and_closes(monkeypatch, capsys):
    from mobilerun.core.devices.base import Device
    from tests.demos import macro_demo as demo

    device = Device()
    device.start_app = AsyncMock()
    device.tap = AsyncMock()
    device.press_button = AsyncMock()
    device.close = AsyncMock()
    monkeypatch.setattr(demo, "AndroidDevice", lambda *a, **kw: device)
    # Exercise the real recorder and player against a fake device.
    asyncio.run(demo.main())
    assert device.start_app.await_count == 2
    assert device.tap.await_count == 2
    assert device.press_button.await_count == 2
    device.close.assert_awaited_once()
    assert "start_app" in capsys.readouterr().out


def test_macro_closes_device_on_failure(monkeypatch):
    import pytest

    from mobilerun.core.devices.base import Device
    from tests.demos import macro_demo as demo

    device = Device()
    device.start_app = AsyncMock(side_effect=RuntimeError("failed"))
    device.close = AsyncMock()
    monkeypatch.setattr(demo, "AndroidDevice", lambda *a, **kw: device)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(demo.main())
    device.close.assert_awaited_once()
