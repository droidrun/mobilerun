import asyncio
import importlib
from pathlib import Path

import pytest

from mobilerun.core.devices.base import Device as _FrameworkDevice

NAMES = (
    "android_basics",
    "cloud_device",
    "ios_device",
    "recording_wrapper",
    "custom_device",
    "visual_remote_device",
)


def test_examples_import():
    for name in NAMES:
        assert callable(importlib.import_module(f"assets.examples.devices.{name}").run)


def test_quickstart_runs_with_fake_devices(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    main = importlib.import_module("tests.demos.intro_demo")
    devices = []

    class Device(FakeDevice):
        def __init__(self, *args, **kwargs):
            super().__init__(with_state=True)
            self.state.current.elements = [{"index": 9, "text": "Search settings"}]
            devices.append(self)

        async def _input_secret(self, value):
            self.calls.append(("credential", value))
            return True

        async def do(self, *args, **kwargs):
            self.calls.append("do")
            if "operator" in kwargs:
                return await kwargs["operator"].execute(self, args[0])
            return SimpleNamespace(success=True, output=SimpleNamespace(version="15"))

        async def expect(self, *args, **kwargs):
            self.calls.append("expect")
            return SimpleNamespace(passed=True)

    class Pool:
        def __init__(self, members):
            self.devices = members

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def screenshot(self):
            return [b"local", b"cloud"]

        async def press_button(self, button):
            for device in self.devices:
                await device.press_button(button)

        async def do(self, instruction):
            return [await device.do(instruction) for device in self.devices]

        async def expect(self, condition):
            return [await device.expect(condition) for device in self.devices]

    monkeypatch.setattr(main, "AgentOperator", object)
    monkeypatch.setattr(main, "AndroidDevice", Device)
    monkeypatch.setattr(main, "CloudDevice", Device)
    monkeypatch.setattr(main, "DevicePool", Pool)
    monkeypatch.setattr(main, "vault_demo", _no_sleep)
    monkeypatch.setattr(main, "APK", str(tmp_path / "missing.apk"))
    monkeypatch.setattr(main, "NEW_CLOUD_DEVICE", False)
    monkeypatch.setattr(main, "ANDROID_AVD", None)
    assert callable(main.local_basics)
    assert callable(main.cloud_basics)
    assert callable(main.macro_demo)
    assert callable(main.agent_demo)
    assert callable(main.pool_demo)
    assert callable(main.sandbox_and_skills)
    monkeypatch.setattr(main, "AGENT", False)
    asyncio.run(main.main())
    assert Path("local-screenshot.png").read_bytes() == b"image"
    assert Path("cloud-screenshot.png").read_bytes() == b"image"
    assert all("connect" in device.calls for device in devices)
    assert devices[0].calls.count("do") == 1
    assert "do" not in devices[1].calls
    output = capsys.readouterr().out
    assert "macro replay:" in output
    assert "pool screenshots: 2" in output
    assert "sandbox function (local): Hello, world!" in output
    assert "skills:" in output

    if importlib.util.find_spec("mobilerun.agents") is not None:
        monkeypatch.setattr(main, "AGENT", True)
        asyncio.run(main.main())
        assert "15" in capsys.readouterr().out
        assert "do" in devices[-2].calls
        assert "expect" in devices[-1].calls


def test_macro_demo_replays_when_screen_changes(tmp_path, monkeypatch, capsys):
    from tests.demos import intro_demo as main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    class ChangedScreen:
        async def refresh(self):
            return {"phone_state": {"package": "home"}, "elements": [{"text": "Home"}]}

    class Device(FakeDevice):
        def __init__(self):
            super().__init__()
            self.state = ChangedScreen()

        async def _get_ui_tree(self):
            return {
                "phone_state": {"package": "previous"},
                "elements": [{"text": "Previous"}],
            }

        async def do(self, instruction, *, operator):
            return await operator.execute(self, instruction)

    device = Device()
    asyncio.run(main.macro_demo(device))
    assert device.calls == [
        "start_app",
        "swipe",
        "back",
        "home",
        "start_app",
        "swipe",
        "back",
    ]
    assert "macro replay: True" in capsys.readouterr().out


async def _no_sleep(*_args):
    return None


def test_agent_demo_handles_failed_typed_output(monkeypatch, capsys):
    from types import SimpleNamespace

    from tests.demos import intro_demo as main

    class Cloud:
        async def do(self, instruction, **kwargs):
            return SimpleNamespace(success=not kwargs, output=None, answer="no result")

        async def expect(self, instruction):
            return SimpleNamespace(passed=True)

    monkeypatch.setattr(main, "AGENT", True)
    asyncio.run(main.agent_demo(Cloud()))
    assert "typed output: failed (no result)" in capsys.readouterr().out


class FakeDevice(_FrameworkDevice):
    platform = "android"
    capabilities = {"tap", "screenshot"}
    supported_buttons = {"back", "home"}

    def __init__(self, with_state=False):
        super().__init__()
        self.calls = []
        self._operator = None
        self.state = FakeUIState() if with_state else None

    @property
    def state(self):
        return self._fake_state

    @state.setter
    def state(self, value):
        self._fake_state = value

    async def _ensure_connected(self):
        self.calls.append("connect")

    async def wait(self, seconds):
        await asyncio.sleep(seconds)

    async def screenshot(self):
        return b"image"

    async def start_app(self, *args):
        self.calls.append("start_app")

    async def stop_app(self, *args):
        self.calls.append("stop_app")

    async def tap(self, *args):
        self.calls.append("tap")

    async def swipe(self, *args):
        self.calls.append("swipe")

    async def input_text(self, *args, **kwargs):
        self.calls.append("input_text")
        return True

    async def press_button(self, button):
        self.calls.append(button)

    async def _get_ui_tree(self):
        return {"children": []}

    async def list_apps(self):
        return ["Settings"]

    async def tap_element(self, index):
        self.calls.append(f"element:{index}")


class FakeUIState:
    elements = [{"index": 9, "text": "Network & internet"}]

    @property
    def current(self):
        return self

    async def refresh(self):
        return object()


@pytest.mark.parametrize(
    "name", ["android_basics", "custom_device", "recording_wrapper"]
)
def test_examples_run_with_fake_device(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(f"assets.examples.devices.{name}")
    device = FakeDevice()
    asyncio.run(module.run(device=device))
    assert "connect" in device.calls
    if name == "recording_wrapper":
        assert "tap" in device.calls
    else:
        assert Path("screenshot.png").read_bytes() == b"image"


def test_credentials_demo_attaches_and_types_credential():
    from tests.demos import intro_demo as main

    class Device(FakeDevice):
        async def _input_secret(self, value):
            self.calls.append(("credential", value))
            return True

    device = Device(with_state=True)
    device.state.current.elements = [{"index": 9, "text": "Search settings"}]
    asyncio.run(main.credentials_demo(device))
    assert "element:9" in device.calls
    assert ("credential", "not-a-real-secret") in device.calls


def test_vault_demo_deletes_on_listing_failure(monkeypatch):
    from tests.demos import intro_demo as main

    calls = []

    class Vault:
        async def save(self, package, name, fields):
            calls.append(("save", package, name, fields))

        async def list(self, package):
            raise RuntimeError("list failed")

        async def delete(self, package, name):
            calls.append(("delete", package, name))

    monkeypatch.setattr(main, "CloudCredentials", Vault)
    with pytest.raises(RuntimeError, match="list failed"):
        asyncio.run(main.vault_demo())
    assert calls[1] == ("delete", calls[0][1], calls[0][2])


def test_vault_demo_lists_names_and_deletes(monkeypatch, capsys):

    from tests.demos import intro_demo as main

    calls = []

    class Vault:
        async def save(self, package, name, fields):
            calls.append(("save", package, name))

        async def list(self, package):
            return [{"package": package, "name": "demo-name", "fields": ["password"]}]

        async def delete(self, package, name):
            calls.append(("delete", package, name))

    monkeypatch.setattr(main, "CloudCredentials", Vault)
    asyncio.run(main.vault_demo())
    assert calls[1] == ("delete", calls[0][1], calls[0][2])
    assert "demo-name" in capsys.readouterr().out


def test_app_install_demos_and_missing_apk(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from tests.demos import intro_demo as main

    calls = []
    apk = tmp_path / "counter.apk"
    monkeypatch.setattr(main, "APK", str(apk))

    class Device:
        async def install_apk(self, path, **kwargs):
            calls.append(("install", path, kwargs))

        async def install_app(self, app, **kwargs):
            calls.append(("install", app, kwargs))

        async def uninstall_app(self, package):
            calls.append(("uninstall", package))

    app = SimpleNamespace(package="counter", version="1")

    async def upload(path, *, billing):
        calls.append(("upload", path, billing))
        return app

    monkeypatch.setattr(main, "upload_app", upload)
    asyncio.run(main.local_app_install_demo(Device()))
    asyncio.run(main.cloud_app_install_demo(Device()))
    assert not calls
    assert "MOBILERUN_SMOKE_APK" in capsys.readouterr().out
    apk.touch()
    asyncio.run(main.local_app_install_demo(Device()))
    asyncio.run(main.cloud_app_install_demo(Device()))
    assert calls == [
        ("install", str(apk), {"reinstall": True}),
        ("upload", str(apk), main.get_settings().MOBILERUN_BILLING),
        ("install", app, {}),
        ("uninstall", "counter"),
    ]


def test_optional_device_demos(monkeypatch, tmp_path, capsys):
    from tests.demos import intro_demo as main

    monkeypatch.chdir(tmp_path)
    calls = []

    class Cloud:
        device_id = "fresh-id"

        def __init__(self, *, billing):
            assert billing == "paid"

        async def __aenter__(self):
            calls.append("enter")
            return self

        async def __aexit__(self, *args):
            calls.append("exit")

        async def screenshot(self):
            return b"fresh"

    class Android:
        def __init__(self, *, avd):
            calls.append(("create", avd))

        def __await__(self):
            async def ready():
                return self

            return ready().__await__()

        async def screenshot(self):
            return b"emulator"

        async def close(self):
            calls.append("emulator-close")

    monkeypatch.setattr(main, "CloudDevice", Cloud)
    monkeypatch.setattr(main, "AndroidDevice", Android)
    monkeypatch.setattr(main, "NEW_CLOUD_DEVICE", False)
    monkeypatch.setattr(main, "ANDROID_AVD", None)
    asyncio.run(main.new_cloud_device_demo())
    asyncio.run(main.emulator_demo())
    assert not calls
    assert "skip" in capsys.readouterr().out
    monkeypatch.setattr(main, "NEW_CLOUD_DEVICE", True)
    monkeypatch.setattr(main, "ANDROID_AVD", "test-avd")
    asyncio.run(main.new_cloud_device_demo())
    asyncio.run(main.emulator_demo())
    assert calls == ["enter", "exit", ("create", "test-avd"), "emulator-close"]
