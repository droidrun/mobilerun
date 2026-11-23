import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("mobilerun.agents", reason="demo requires the agents add-on")

from mobilerun.core.devices.base import Device as _FrameworkDevice


def test_app_demo_runs_on_both_devices(monkeypatch, capsys):
    from tests.demos import app_demo

    devices = []

    class FakeDevice(_FrameworkDevice):
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
    monkeypatch, capsys
):
    from tests.demos import app_demo

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


def test_app_demo_zips_one_cloud_label_with_one_result(monkeypatch, capsys):
    from tests.demos import app_demo

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


def test_app_demo_decline_does_not_create(monkeypatch):
    from unittest.mock import Mock

    from tests.demos import app_demo as demo

    create = Mock()
    monkeypatch.setattr(demo, "CloudDevice", create)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    asyncio.run(demo.main(yes=False))
    create.assert_not_called()
