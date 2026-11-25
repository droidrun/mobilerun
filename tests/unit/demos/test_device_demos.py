"""Offline contracts for the introduction, app, macro, and vault device demos."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mobilerun.core.devices.base import Device
from tests.demos import intro_demo
from tests.support.paths import REPO_ROOT


def test_import_does_not_load_agent_addon_or_deploy():
    code = """import builtins
import mobilerun.sandbox as sandbox

def blocked(*args, **kwargs):
    raise AssertionError('demo attempted deployment during import')

sandbox.deploy = blocked
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('mobilerun.agents', 'mobilerun_agents')):
        raise AssertionError('demo imported agents eagerly')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import tests.demos.intro_demo
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_sandbox_example_runs_locally():
    greeting = intro_demo.hello(intro_demo.GreetingInput(name="Ada"))
    assert greeting.message == "Hello, Ada!"


def test_agent_demo_uses_public_operator(monkeypatch):
    from types import ModuleType

    addon = ModuleType("mobilerun_agents")
    operator = Mock()
    monkeypatch.setattr(addon, "AgentOperator", operator, raising=False)
    monkeypatch.setitem(sys.modules, addon.__name__, addon)
    assert intro_demo.AgentOperator() is operator.return_value


def test_main_without_agent_uses_base_device(monkeypatch):
    device = Mock(_ensure_connected=AsyncMock())
    cloud = Mock(_ensure_connected=AsyncMock())
    android = Mock(return_value=device)
    monkeypatch.setattr(intro_demo, "AGENT", False)
    monkeypatch.setattr(intro_demo, "AndroidDevice", android)
    monkeypatch.setattr(intro_demo, "CloudDevice", Mock(return_value=cloud))
    monkeypatch.setattr(intro_demo, "sandbox_and_skills", Mock())
    for name in (
        "local_demos",
        "cloud_demos",
        "pool_demo",
        "new_cloud_device_demo",
        "emulator_demo",
    ):
        monkeypatch.setattr(intro_demo, name, AsyncMock())
    asyncio.run(intro_demo.main())
    android.assert_called_once_with(intro_demo.LOCAL_SERIAL)
    intro_demo.local_demos.assert_awaited_once_with(device)
    intro_demo.cloud_demos.assert_awaited_once_with(cloud)


def test_help_exits_without_running_demo(monkeypatch):
    main = AsyncMock()
    run = Mock()
    monkeypatch.setattr(intro_demo, "main", main)
    monkeypatch.setattr(intro_demo.asyncio, "run", run)
    with pytest.raises(SystemExit) as exc:
        intro_demo.cli(["--help"])
    assert exc.value.code == 0
    main.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("new_phone", [False, True])
def test_declining_does_not_run_demo(monkeypatch, capsys, new_phone):
    main = AsyncMock()
    monkeypatch.setattr(intro_demo, "main", main)
    monkeypatch.setattr(intro_demo, "NEW_CLOUD_DEVICE", new_phone)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "n")
    intro_demo.cli([])
    main.assert_not_called()
    assert "cloud task" in prompts[0].lower()
    assert "llm" in prompts[0].lower()
    assert ("new cloud phone" in prompts[0].lower()) is new_phone
    assert "Cancelled" in capsys.readouterr().out


def test_yes_runs_main(monkeypatch):
    main = AsyncMock()
    monkeypatch.setattr(intro_demo, "main", main)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unexpected prompt"))
    intro_demo.cli(["--yes"])
    main.assert_awaited_once_with()


def test_script_help_in_isolated_mode():
    root = REPO_ROOT
    script = root / "tests/demos/intro_demo.py"
    # Fail closed: if --help ever reaches asyncio.run, abort before the demo runs.
    launcher = (
        "import asyncio, runpy, sys\n"
        "def _blocked(coro, *a, **kw):\n"
        "    coro.close()\n"
        "    raise SystemExit('demo started during --help')\n"
        "asyncio.run = _blocked\n"
        f"sys.path[:] = [p for p in sys.path if p != {str(root)!r}]\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", launcher],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--yes" in result.stdout


@pytest.fixture
def app_demo():
    pytest.importorskip("mobilerun_agents", reason="demo requires the agents add-on")
    from tests.demos import app_demo

    return app_demo


def test_app_demo_runs_on_both_devices(app_demo, monkeypatch, capsys):

    devices = []

    class FakeDevice(Device):
        def __init__(self, identifier, **kwargs):
            super().__init__()
            self._identifier = identifier
            self._operator = None
            self.calls = []
            devices.append(self)

        @property
        def identifier(self):
            return self._identifier

        async def install_apk(self, path):
            self.calls.append(("install", path))
            return SimpleNamespace(package=self.identifier or "local")

        async def start_app(self, package, activity=None):
            self.calls.append(("start", package))

        async def uninstall_app(self, app):
            self.calls.append(("uninstall", app))

        async def wait(self, seconds):
            self.calls.append(("wait", seconds))

        async def do(self, instruction, **kwargs):
            self.calls.append(("do", instruction, kwargs["output"]))
            return SimpleNamespace(output=app_demo.CounterValue(value=10))

        async def expect(self, condition):
            self.calls.append(("expect", condition))
            return SimpleNamespace(passed=True)

        async def press_button(self, button):
            self.calls.append(("press", button))

    monkeypatch.setattr(app_demo, "AndroidDevice", FakeDevice)
    monkeypatch.setattr(app_demo, "CloudDevice", FakeDevice)
    monkeypatch.setattr(app_demo, "CloudTaskOperator", lambda **kwargs: object())
    asyncio.run(app_demo.main(yes=True))

    assert [device.identifier for device in devices] == [
        app_demo.LOCAL_SERIAL,
        app_demo.CLOUD_DEVICE_ID,
    ]
    for device in devices:
        assert device.calls[:3] == [
            ("install", app_demo.APK),
            ("start", device.identifier or "local"),
            ("wait", 2),
        ]


def test_app_demo_does_not_reuse_execution_results_when_expect_fails(
    app_demo, monkeypatch, capsys
):

    class Pool:
        def __init__(self, devices):
            pass

        async def do(self, *args, **kwargs):
            return [SimpleNamespace(output=None), SimpleNamespace(output=None)]

        async def expect(self, condition):
            raise RuntimeError("expectation unavailable")

        async def press_button(self, button):
            pass

        async def close(self):
            pass

    async def install(pool):
        pass

    monkeypatch.setattr(app_demo, "CloudDevice", lambda *a, **kw: object())
    monkeypatch.setattr(app_demo, "CloudTaskOperator", lambda: object())
    monkeypatch.setattr(app_demo, "DevicePool", Pool)
    monkeypatch.setattr(app_demo, "install_and_open", install)
    asyncio.run(app_demo.main(yes=True))
    assert "expect failed: expectation unavailable" in capsys.readouterr().out


def test_app_demo_zips_one_cloud_label_with_one_result(app_demo, monkeypatch, capsys):

    class Pool:
        def __init__(self, devices):
            pass

        async def do(self, *args, **kwargs):
            return [
                SimpleNamespace(output=app_demo.CounterValue(value=10)),
                SimpleNamespace(output=app_demo.CounterValue(value=10)),
            ]

        async def expect(self, condition):
            return [SimpleNamespace(passed=True), SimpleNamespace(passed=True)]

        async def press_button(self, button):
            pass

        async def close(self):
            pass

    async def install(pool):
        pass

    monkeypatch.setattr(app_demo, "AndroidDevice", lambda *a, **kw: object())
    monkeypatch.setattr(app_demo, "CloudDevice", lambda *a, **kw: object())
    monkeypatch.setattr(app_demo, "CloudTaskOperator", lambda: object())
    monkeypatch.setattr(app_demo, "DevicePool", Pool)
    monkeypatch.setattr(app_demo, "install_and_open", install)
    asyncio.run(app_demo.main(yes=True))
    output = capsys.readouterr().out
    assert "local counter: 10" in output
    assert "cloud counter: 10" in output
    assert "local expect: True" in output
    assert "cloud expect: True" in output


def test_app_demo_decline_does_not_create(app_demo, monkeypatch):

    create = Mock()
    monkeypatch.setattr(app_demo, "CloudDevice", create)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    asyncio.run(app_demo.main(yes=False))
    create.assert_not_called()


def test_macro_records_replays_and_closes(monkeypatch, capsys):
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
    from tests.demos import macro_demo as demo

    device = Device()
    device.start_app = AsyncMock(side_effect=RuntimeError("failed"))
    device.close = AsyncMock()
    monkeypatch.setattr(demo, "AndroidDevice", lambda *a, **kw: device)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(demo.main())
    device.close.assert_awaited_once()


def setup_vault_demo(monkeypatch, *, fail=False):
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
    demo, vault, device, calls = setup_vault_demo(monkeypatch)
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
    demo, vault, device, calls = setup_vault_demo(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(demo.main())
    vault.delete.assert_awaited_once()
    vault.unbind.assert_awaited_once()
    vault.close.assert_awaited_once()
    device.close.assert_awaited_once()
    assert calls == ["bind", "type", "unbind", "device.close"]


def test_vault_closes_after_bind_failure_without_deleting(monkeypatch):
    demo, vault, device, _ = setup_vault_demo(monkeypatch)
    vault.bind.side_effect = RuntimeError("binding failed")
    with pytest.raises(RuntimeError, match="binding failed"):
        asyncio.run(demo.main())
    vault.save.assert_not_awaited()
    vault.delete.assert_not_awaited()
    vault.unbind.assert_not_awaited()
    vault.close.assert_awaited_once()
    device.close.assert_awaited_once()


def test_vault_closes_device_even_if_delete_fails(monkeypatch):
    demo, vault, device, calls = setup_vault_demo(monkeypatch)
    vault.delete.side_effect = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        asyncio.run(demo.main())
    vault.unbind.assert_awaited_once()
    vault.close.assert_awaited_once()
    device.close.assert_awaited_once()
    assert calls == ["bind", "type", "unbind", "device.close"]


def test_existing_search_is_cleared_before_typing(monkeypatch):
    demo, vault, device, _ = setup_vault_demo(monkeypatch)
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
    demo, vault, device, _ = setup_vault_demo(monkeypatch)
    device.state.refresh.side_effect = [SimpleNamespace(elements=[])]
    with pytest.raises(RuntimeError, match="search entry not found"):
        asyncio.run(demo.main())
    vault.type_field.assert_not_awaited()
    vault.delete.assert_awaited_once()
    device.press_button.assert_awaited_once_with("home")
    device.close.assert_awaited_once()


def test_wrong_focused_text_is_not_printed(monkeypatch, capsys):
    demo, vault, device, _ = setup_vault_demo(monkeypatch)
    snapshots = list(device.state.refresh.side_effect)
    snapshots[2].focused_text = "private-existing-text"
    device.state.refresh.side_effect = snapshots
    with pytest.raises(RuntimeError, match="not visible"):
        asyncio.run(demo.main())
    assert "private-existing-text" not in capsys.readouterr().out
    vault.delete.assert_awaited_once()
