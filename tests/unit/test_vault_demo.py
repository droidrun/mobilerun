import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def setup_demo(monkeypatch, *, fail=False):
    from tests.demos import vault_demo as demo

    calls = []
    snapshots = [
        SimpleNamespace(
            elements=[{"index": 7, "text": "Search settings", "resourceId": "search"}],
            focused_text="",
        ),
        SimpleNamespace(
            elements=[
                {
                    "index": 8,
                    "className": "EditText",
                    "resourceId": "search_input",
                    "text": "",
                }
            ],
            focused_text="",
        ),
        SimpleNamespace(elements=[], focused_text="demo-user"),
    ]
    device = SimpleNamespace(
        close=AsyncMock(side_effect=lambda: calls.append("device.close")),
        start_app=AsyncMock(),
        wait=AsyncMock(),
        tap_element=AsyncMock(),
        press_button=AsyncMock(),
        state=SimpleNamespace(refresh=AsyncMock(side_effect=snapshots)),
    )
    vault = SimpleNamespace()

    async def bind(target):
        assert target is device
        calls.append("bind")

    async def list_metadata(package):
        assert calls == ["bind"]
        assert package == demo.PACKAGE
        return [{"package": package, "name": "work", "fields": ["username"]}]

    async def type_field(package, field, *, name):
        assert calls[0] == "bind"
        calls.append("type")
        if fail:
            raise RuntimeError("failed")

    vault.bind = AsyncMock(side_effect=bind)
    vault.list = AsyncMock(side_effect=list_metadata)
    vault.type_field = AsyncMock(side_effect=type_field)
    vault.save = AsyncMock()
    vault.delete = AsyncMock()
    vault.unbind = AsyncMock(side_effect=lambda: calls.append("unbind"))
    vault.close = AsyncMock()
    monkeypatch.setattr(demo, "CloudCredentials", lambda: vault)
    monkeypatch.setattr(demo, "AndroidDevice", lambda *a, **kw: device)
    return demo, vault, device, calls


def test_vault_demo_saves_types_and_cleans_up(monkeypatch, capsys):
    demo, vault, device, calls = setup_demo(monkeypatch)
    asyncio.run(demo.main())
    name = vault.save.await_args.args[1]
    vault.type_field.assert_awaited_once_with(demo.PACKAGE, "username", name=name)
    vault.delete.assert_awaited_once_with(demo.PACKAGE, name)
    vault.unbind.assert_awaited_once()
    vault.close.assert_awaited_once()
    device.close.assert_awaited_once()
    assert calls == ["bind", "type", "unbind", "device.close"]
    output = capsys.readouterr().out
    assert "work" in output
    assert "demo-password" not in output
    assert "Field now contains: demo-user" in output
    device.start_app.assert_awaited_once_with("com.android.settings")
    assert [item.args for item in device.tap_element.await_args_list] == [(7,), (8,)]
    device.press_button.assert_awaited_once_with("home")


def test_vault_cleanup_on_typing_error(monkeypatch):
    demo, vault, device, calls = setup_demo(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(demo.main())
    vault.delete.assert_awaited_once()
    vault.unbind.assert_awaited_once()
    vault.close.assert_awaited_once()
    device.close.assert_awaited_once()
    assert calls == ["bind", "type", "unbind", "device.close"]


def test_vault_closes_after_bind_failure_without_deleting(monkeypatch):
    demo, vault, device, _ = setup_demo(monkeypatch)
    vault.bind.side_effect = RuntimeError("binding failed")
    with pytest.raises(RuntimeError, match="binding failed"):
        asyncio.run(demo.main())
    vault.save.assert_not_awaited()
    vault.delete.assert_not_awaited()
    vault.unbind.assert_not_awaited()
    vault.close.assert_awaited_once()
    device.close.assert_awaited_once()


def test_vault_closes_device_even_if_delete_fails(monkeypatch):
    demo, vault, device, calls = setup_demo(monkeypatch)
    vault.delete.side_effect = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        asyncio.run(demo.main())
    vault.unbind.assert_awaited_once()
    vault.close.assert_awaited_once()
    device.close.assert_awaited_once()
    assert calls == ["bind", "type", "unbind", "device.close"]


def test_existing_search_is_cleared_before_typing(monkeypatch):
    demo, vault, device, _ = setup_demo(monkeypatch)
    device.state.refresh.side_effect = [
        SimpleNamespace(
            elements=[
                {"index": 2, "className": "EditText"},
                {"index": 3, "resourceId": "search_clear_button"},
            ]
        ),
        SimpleNamespace(elements=[{"index": 4, "className": "EditText"}]),
        SimpleNamespace(elements=[], focused_text="demo-user"),
    ]
    asyncio.run(demo.main())
    assert [item.args for item in device.tap_element.await_args_list] == [(3,), (4,)]
    vault.type_field.assert_awaited_once()


def test_missing_search_still_cleans_up(monkeypatch):
    demo, vault, device, _ = setup_demo(monkeypatch)
    device.state.refresh.side_effect = [SimpleNamespace(elements=[])]
    with pytest.raises(RuntimeError, match="search entry not found"):
        asyncio.run(demo.main())
    vault.type_field.assert_not_awaited()
    vault.delete.assert_awaited_once()
    device.press_button.assert_awaited_once_with("home")
    device.close.assert_awaited_once()


def test_wrong_focused_text_is_not_printed(monkeypatch, capsys):
    demo, vault, device, _ = setup_demo(monkeypatch)
    snapshots = list(device.state.refresh.side_effect)
    snapshots[2].focused_text = "private-existing-text"
    device.state.refresh.side_effect = snapshots
    with pytest.raises(RuntimeError, match="not visible"):
        asyncio.run(demo.main())
    assert "private-existing-text" not in capsys.readouterr().out
    vault.delete.assert_awaited_once()
