import asyncio
from types import SimpleNamespace

from mobilerun.cli import selection
from mobilerun.config_manager.config_manager import DeviceConfig


def test_device_config_defaults_to_auto_portal_mode():
    config = DeviceConfig()

    assert config.portal_mode == "auto"


def test_device_config_accepts_disabled_portal_mode():
    config = DeviceConfig(portal_mode="disabled")

    assert config.portal_mode == "disabled"


def test_create_driver_passes_portal_mode(monkeypatch):
    android_calls = []

    class FakeAndroidDriver:
        def __init__(self, serial, use_tcp, portal_mode):
            android_calls.append(
                {
                    "serial": serial,
                    "use_tcp": use_tcp,
                    "portal_mode": portal_mode,
                }
            )

        async def connect(self):
            pass

    monkeypatch.setattr(
        selection.ConfigLoader,
        "load",
        lambda _path: SimpleNamespace(
            device=SimpleNamespace(
                serial="emulator-5556",
                use_tcp=True,
                platform="android",
                auto_setup=False,
                portal_mode="disabled",
            )
        ),
    )
    from mobilerun_core_local.driver import android
    monkeypatch.setattr(android, "AndroidDriver", FakeAndroidDriver)

    async def connect():
        selected = await selection.select_device()
        return await selection._native_driver(selected, selected.config, "https://example.test/v1")
    driver = asyncio.run(connect())

    assert isinstance(driver, FakeAndroidDriver)
    assert android_calls == [
        {
            "serial": "emulator-5556",
            "use_tcp": True,
            "portal_mode": "disabled",
        }
    ]


def test_create_driver_disabled_portal_mode_skips_auto_setup(monkeypatch):
    setup_calls = []

    class FakeAndroidDriver:
        def __init__(self, serial, use_tcp, portal_mode):
            pass

        async def connect(self):
            pass

    async def fake_ensure_portal_ready(*args, **kwargs):
        setup_calls.append((args, kwargs))

    monkeypatch.setattr(
        selection.ConfigLoader,
        "load",
        lambda _path: SimpleNamespace(
            device=SimpleNamespace(
                serial="emulator-5556",
                use_tcp=False,
                platform="android",
                auto_setup=True,
                portal_mode="disabled",
            )
        ),
    )
    from mobilerun_core_local.driver import android
    monkeypatch.setattr(android, "AndroidDriver", FakeAndroidDriver)
    monkeypatch.setattr(
        android.portal, "ensure_portal_ready", fake_ensure_portal_ready
    )

    async def connect():
        selected = await selection.select_device()
        return await selection._native_driver(selected, selected.config, "https://example.test/v1")
    asyncio.run(connect())

    assert setup_calls == []
