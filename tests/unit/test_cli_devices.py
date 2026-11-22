import pytest


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("serial", "android"),
        ("192.168.1.20:5555", "android"),
        ("emulator-5554", "android"),
        ("123e4567-e89b-12d3-a456-426614174000", "cloud"),
    ],
)
def test_detect_device_kind(target, expected):
    from mobilerun.cli.devices.base.selection import detect_device_kind

    assert detect_device_kind(target) == expected


def test_detect_device_kind_rejects_ambiguous_http_url():
    from mobilerun.cli.devices.base.selection import detect_device_kind

    with pytest.raises(ValueError, match="URL target: add -i/--ios"):
        detect_device_kind("http://127.0.0.1:8000")


def test_connect_target_builds_android_and_closes_android_only(monkeypatch):
    import asyncio

    from mobilerun.cli.devices.base import selection as factory

    made = []

    class FakeAndroid:
        device = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            made.append(self)

        async def connect(self):
            pass

    monkeypatch.setattr(factory, "AndroidDevice", FakeAndroid)
    device = asyncio.run(factory.connect_target("serial", tcp=True))
    assert isinstance(device, FakeAndroid)
    assert made[0].kwargs == {"serial": "serial", "use_tcp": True}


def test_create_android_url_uses_portal_http_driver(monkeypatch):
    import asyncio

    from mobilerun.cli.devices.base import selection as factory

    class FakeDriver:
        async def connect(self):
            pass

    driver = FakeDriver()

    def make_driver(url, token):
        return driver

    monkeypatch.setattr(factory, "create_android_portal_http_driver", make_driver)
    monkeypatch.setattr(
        factory.ConfigLoader,
        "load",
        lambda _: type(
            "C",
            (),
            {"device": type("D", (), {"resolve_auth_token": lambda self: "token"})()},
        )(),
    )
    monkeypatch.setattr(
        factory, "AndroidDevice", lambda **kwargs: pytest.fail("ADB used")
    )

    result = asyncio.run(factory.connect_target("http://portal.test", kind="android"))
    from typing import cast

    from mobilerun.core.devices.components.wrapper import DriverDevice

    assert cast(DriverDevice, result)._driver is driver
    from mobilerun.core.state.android import AndroidStateProvider
    from mobilerun.core.state.options import StateProviderOptions

    assert isinstance(
        result._create_state_provider(StateProviderOptions()), AndroidStateProvider
    )


def test_connect_target_passes_android_configuration(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices.base import selection as devices

    driver = SimpleNamespace(connect=AsyncMock())

    def constructor(**kwargs):
        seen.update(kwargs)
        return driver

    seen = {}
    monkeypatch.setattr(devices, "AndroidDevice", constructor)
    config = SimpleNamespace(
        device=SimpleNamespace(auto_setup=False, portal_mode="disabled")
    )
    assert (
        asyncio.run(devices.connect_target("serial", tcp=True, config=config)) is driver
    )
    assert seen == {"serial": "serial", "use_tcp": True, "portal_mode": "disabled"}
    driver.connect.assert_awaited_once()


def test_connect_target_constructs_cloud_and_visual_remote(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from mobilerun.cli.devices.base import selection as devices
    from mobilerun.core.devices import cloud

    driver = SimpleNamespace(connect=AsyncMock())
    constructor = Mock(return_value=driver)
    monkeypatch.setattr(cloud, "CloudDevice", constructor)
    assert (
        asyncio.run(
            devices.connect_target("id", kind="cloud", base_url="https://cloud")
        )
        is driver
    )
    constructor.assert_called_once_with(device_id="id", base_url="https://cloud")
    constructor = Mock(return_value=driver)
    monkeypatch.setattr(devices, "RemoteDevice", constructor)
    config = SimpleNamespace(device=SimpleNamespace(device_id="remote-id"))
    assert (
        asyncio.run(
            devices.connect_target("http://remote", kind="visual_remote", config=config)
        )
        is driver
    )
    constructor.assert_called_once_with(url="http://remote", device_id="remote-id")


def test_connect_target_constructs_ios_with_token(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from mobilerun.cli.devices.base import selection as devices

    driver = SimpleNamespace(connect=AsyncMock())
    create = Mock(return_value=driver)
    monkeypatch.setattr(devices, "IOSDevice", create)
    config = SimpleNamespace(device=SimpleNamespace(resolve_auth_token=lambda: "token"))
    result = asyncio.run(
        devices.connect_target("http://ios", kind="ios", config=config)
    )
    assert result is driver
    create.assert_called_once_with("http://ios", token="token")
    driver.connect.assert_awaited_once()
