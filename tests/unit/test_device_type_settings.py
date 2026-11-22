"""Device type defaults can be configured without affecting other settings."""

from mobilerun.config.settings import get_settings


def test_device_type_defaults(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CLOUD_DEVICE_TYPE", raising=False)
    monkeypatch.delenv("MOBILERUN_ANDROID_DEVICE_TYPE", raising=False)
    settings = get_settings()
    assert settings.MOBILERUN_CLOUD_DEVICE_TYPE == "android_cloud_phone"
    assert settings.MOBILERUN_ANDROID_DEVICE_TYPE is None


def test_device_type_environment_overrides(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_DEVICE_TYPE", "custom-cloud")
    monkeypatch.setenv("MOBILERUN_ANDROID_DEVICE_TYPE", "Pixel_API")
    settings = get_settings()
    assert settings.MOBILERUN_CLOUD_DEVICE_TYPE == "custom-cloud"
    assert settings.MOBILERUN_ANDROID_DEVICE_TYPE == "Pixel_API"


def test_cloud_create_uses_setting_and_explicit_override(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.core.devices import cloud

    create = AsyncMock(return_value=SimpleNamespace(id="device", name="device"))
    devices = SimpleNamespace(
        create=create,
        wait_ready=AsyncMock(),
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(capabilities=SimpleNamespace(telephony=False))
        ),
    )

    class Client:
        def __init__(self, **kwargs):
            self.devices = devices

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setenv("MOBILERUN_CLOUD_DEVICE_TYPE", "ios_phone")
    from mobilerun_core_local.driver import cloud as driver

    client = Client()
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **kwargs: client)

    async def initialize():
        await cloud.CloudDevice(api_key="test", billing="paid")
        await cloud.CloudDevice(
            api_key="test", billing="paid", device_type="device_slot"
        )

    asyncio.run(initialize())
    assert [call.kwargs["device_type"] for call in create.await_args_list] == [
        "ios_phone",
        "device_slot",
    ]
