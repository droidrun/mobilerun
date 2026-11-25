import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.models.apps import App


def test_android_install_returns_app_and_uses_its_path(monkeypatch, tmp_path):
    import mobilerun.core.devices.android as android

    apk = tmp_path / "counter.apk"
    apk.write_bytes(b"apk")
    monkeypatch.setattr(
        android,
        "inspect_apk",
        lambda path: ("ai.droidrun.counter", 3, "1.2", "Counter"),
    )
    install = AsyncMock(return_value="success")
    monkeypatch.setattr(android.AndroidDriver, "install_app", install)
    device = android.AndroidDevice(serial="test")
    device._driver.ensure_connected = AsyncMock()
    device._adb_device = SimpleNamespace(
        shell=AsyncMock(return_value="package:/data/app/base.apk")
    )
    app = asyncio.run(device.install_apk(apk))
    assert app == App(
        package="ai.droidrun.counter",
        version="1.2",
        name="Counter",
        version_code=3,
        path=apk,
    )
    assert asyncio.run(device.install_app(app)) == app
    assert install.await_count == 2
    with pytest.raises(ValueError, match="This app has no local APK"):
        asyncio.run(device.install_app(App(package="p", version="1", name="N")))


def test_app_path_is_optional():
    assert App(package="p", version="1", name="N").path is None


def test_android_lifecycle_accepts_descriptor_and_string(monkeypatch):
    import mobilerun.core.devices.android as android

    app = App("org.example", "1", "Example")
    device = android.AndroidDevice(serial="test")
    device._driver.ensure_connected = AsyncMock()
    for method in ("start_app", "stop_app"):
        fake = AsyncMock(return_value="ok")
        monkeypatch.setattr(android.AndroidDriver, method, fake)
        assert asyncio.run(getattr(device, method)(app.package)) == "ok"
        assert fake.await_args.args[0] == "org.example"
        assert asyncio.run(getattr(device, method)("org.example")) == "ok"
        assert fake.await_args.args[0] == "org.example"

    fake = AsyncMock(return_value="ok")
    monkeypatch.setattr(android.AndroidDriver, "uninstall_app", fake)
    assert asyncio.run(device.uninstall_app(app.package)) == "ok"


def test_pool_pairs_installed_apps_with_devices():
    from mobilerun.core.devices.pool import DevicePool

    first, second = App("one", "1", "One"), App("two", "1", "Two")
    devices = [type("Fake", (), {})(), type("Fake", (), {})()]
    for device, app in zip(devices, [first, second], strict=True):
        device._operator = None
        device.install_apk = AsyncMock(return_value=app)
        device.start_app = AsyncMock()
        device.uninstall_app = AsyncMock()
    pool = DevicePool(devices)
    apps = asyncio.run(pool.install_apk("app.apk"))

    async def run():
        for device, app in zip(pool.devices, apps, strict=True):
            await device.start_app(app.package, activity=None)
            await device.uninstall_app(app.package)

    asyncio.run(run())
    devices[0].start_app.assert_awaited_once_with(first.package, activity=None)
    devices[1].start_app.assert_awaited_once_with(second.package, activity=None)
    devices[0].uninstall_app.assert_awaited_once_with(first.package)
    devices[1].uninstall_app.assert_awaited_once_with(second.package)


def test_cloud_install_uploads_without_billing_guard(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mobilerun.clients.cloud import apps
    from mobilerun.core.devices.cloud import CloudDevice

    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    monkeypatch.setattr(
        apps, "inspect_apk", lambda path: ("org.example", 3, "1", "Example")
    )
    signed = SimpleNamespace(
        app_id="id",
        version_id="version",
        r2_upload_urls=[
            SimpleNamespace(file_name="app.apk", r2_upload_url="https://upload")
        ],
    )
    sdk = SimpleNamespace(
        apps=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    items=[], pagination=SimpleNamespace(has_next=False)
                )
            ),
            create_signed_upload_url=AsyncMock(return_value=signed),
            confirm_upload=AsyncMock(),
        )
    )
    context = AsyncMock()
    context.__aenter__.return_value = sdk
    monkeypatch.setattr(apps, "cloud_client", lambda *args, **kwargs: context)
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.put.return_value = SimpleNamespace(raise_for_status=lambda: None)
    device = CloudDevice(device_id="device", api_key="test")
    monkeypatch.setattr(apps.httpx, "AsyncClient", lambda **kwargs: http)
    device._ensure_connected = AsyncMock()
    request = AsyncMock(
        side_effect=[
            SimpleNamespace(installs=[]),
            SimpleNamespace(
                installs=[
                    SimpleNamespace(
                        app_id="org.example", status="succeeded", started_at=1
                    )
                ]
            ),
        ]
    )
    device.list_apps = AsyncMock(
        return_value=[{"package_name": "org.example", "version_code": 3}]
    )
    monkeypatch.setattr(apps, "list_installs", request)
    install = AsyncMock()
    monkeypatch.setattr(apps, "install", install)
    app = asyncio.run(device.install_apk(apk))
    assert app.package == "org.example"
    install.assert_awaited_once_with(
        device._client,
        "device",
        package_name="org.example",
        background=True,
        version_code=3,
        billing="paid",
    )


def test_ios_lifecycle_accepts_app(monkeypatch):
    import mobilerun.core.devices.ios as ios

    app = App("org.example", "1", "Example")
    fake = AsyncMock(return_value="ok")
    device = ios.IOSDevice("http://localhost")
    device._driver = ios.IOSPortalDriver("http://localhost")
    device._driver.ensure_connected = AsyncMock()
    monkeypatch.setattr(device._driver, "start_app", fake)
    assert asyncio.run(device.start_app(app.package)) == "ok"
    fake.assert_awaited_once_with("org.example", None)


def test_cloud_install_app_with_local_path_uploads(monkeypatch, tmp_path):
    import mobilerun.core.devices.components.cloud_apps as cloud
    from mobilerun.core.devices.cloud import CloudDevice

    path = tmp_path / "app.apk"
    local = App("example", "1", "Example", path=path)
    uploaded = App("example", "1", "Example", id="id", path=path)
    device = CloudDevice(device_id="device", api_key="test")
    device._ensure_connected = AsyncMock()
    request = AsyncMock(
        side_effect=[
            SimpleNamespace(installs=[]),
            SimpleNamespace(
                installs=[
                    SimpleNamespace(app_id="example", status="succeeded", started_at=1)
                ]
            ),
        ]
    )
    device.list_apps = AsyncMock(
        return_value=[{"package_name": "example", "version_name": "1"}]
    )
    monkeypatch.setattr(cloud.apps, "list_installs", request)
    install = AsyncMock()
    monkeypatch.setattr(cloud.apps, "install", install)
    upload = AsyncMock(return_value=uploaded)
    monkeypatch.setattr(cloud, "upload_app", upload)
    assert asyncio.run(device.install_app(local)) == uploaded
    assert upload.await_args.args == (path,)
    install.assert_awaited_once_with(
        device._client,
        "device",
        package_name="example",
        background=True,
        billing="paid",
    )
    with pytest.raises(ValueError, match="This app has no local APK"):
        asyncio.run(device.install_app(App("example", "1", "Example")))


@pytest.mark.parametrize(
    "status, expected", [("failed", "failed"), ("running", "timed out")]
)
def test_cloud_install_failure_and_timeout(monkeypatch, status, expected):
    from types import SimpleNamespace

    import mobilerun.core.devices.components.cloud_apps as cloud
    from mobilerun.core.devices.cloud import CloudDevice

    device = CloudDevice(device_id="device", api_key="test")
    device._ensure_connected = AsyncMock()
    install = SimpleNamespace(
        app_id="example", status=status, error_class="failed", started_at=1
    )
    calls = 0

    async def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(installs=[] if calls == 1 else [install])

    monkeypatch.setattr(cloud.apps, "list_installs", AsyncMock(side_effect=request))
    monkeypatch.setattr(cloud.apps, "install", AsyncMock())
    with pytest.raises((RuntimeError, TimeoutError), match=expected):
        asyncio.run(
            device.install_app(App("example", "1", "Example", id="id"), timeout=0.05)
        )


def test_cloud_install_waits_until_succeeded_and_installed(monkeypatch):
    from types import SimpleNamespace

    import mobilerun.core.devices.components.cloud_apps as cloud
    from mobilerun.core.devices.cloud import CloudDevice

    device = CloudDevice(device_id="device", api_key="test")
    device._ensure_connected = AsyncMock()
    attempts = iter(["running", "succeeded"])
    seen = []
    calls = 0

    async def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(installs=[])
        status = next(attempts)
        seen.append(status)
        return SimpleNamespace(
            installs=[SimpleNamespace(app_id="example", status=status, started_at=1)]
        )

    monkeypatch.setattr(cloud.apps, "list_installs", AsyncMock(side_effect=request))
    monkeypatch.setattr(cloud.apps, "install", AsyncMock())
    device.list_apps = AsyncMock(
        return_value=[{"package_name": "example", "version_code": 3}]
    )
    app = App("example", "1", "Example", id="id", version_code=3)
    assert asyncio.run(device.install_app(app, timeout=2)) == app
    assert seen == ["running", "succeeded"]
    device.list_apps.assert_awaited_once()


def test_android_install_checks_package_presence(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import mobilerun.core.devices.android as android

    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    monkeypatch.setattr(
        android, "inspect_apk", lambda path: ("example", 1, "1", "Example")
    )
    monkeypatch.setattr(android.AndroidDriver, "install_app", AsyncMock())
    device = android.AndroidDevice(serial="test")
    device._driver.ensure_connected = AsyncMock()
    device._adb_device = SimpleNamespace(shell=AsyncMock(return_value=""))
    with pytest.raises(RuntimeError, match="not installed"):
        asyncio.run(device.install_apk(apk))


def test_driver_device_passes_package_to_portal():
    from mobilerun.core.devices.components.wrapper import DriverDevice

    driver = SimpleNamespace(
        platform="android",
        credentials=[],
        start_app=AsyncMock(return_value="ok"),
        stop_app=AsyncMock(return_value="ok"),
        uninstall_app=AsyncMock(return_value="ok"),
    )
    from mobilerun.core.state.options import android_state_provider

    device = DriverDevice(driver, state_provider_factory=android_state_provider)
    app = App("org.example", "1", "Example")
    assert asyncio.run(device.start_app(app.package)) == "ok"
    assert asyncio.run(device.stop_app(app.package)) == "ok"
    assert asyncio.run(device.uninstall_app(app.package)) == "ok"
    driver.start_app.assert_awaited_once_with("org.example", None)
    driver.stop_app.assert_awaited_once_with("org.example")
    driver.uninstall_app.assert_awaited_once_with("org.example")


def test_missing_aapt_explains_build_tools(monkeypatch, tmp_path):
    from mobilerun.clients.local import apk as apps

    monkeypatch.setattr(apps.shutil, "which", lambda name: None)
    monkeypatch.setattr(apps.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="Android build-tools"):
        apps.inspect_apk(tmp_path / "app.apk")


def test_recording_device_records_package_string():
    from mobilerun.core.devices.base import Device
    from mobilerun.core.devices.recording import RecordingDevice

    inner = Device()
    inner.start_app = AsyncMock(return_value="ok")
    device = RecordingDevice(inner)
    app = App("org.example", "1", "Example")
    assert asyncio.run(device.start_app(app.package)) == "ok"
    assert device.recorded_actions[-1]["package"] == "org.example"


def test_cloud_install_rejects_stale_installed_version(monkeypatch):
    import mobilerun.core.devices.components.cloud_apps as cloud
    from mobilerun.core.devices.cloud import CloudDevice

    device = CloudDevice(device_id="device", api_key="test")
    device._ensure_connected = AsyncMock()
    request = AsyncMock(
        side_effect=[
            SimpleNamespace(installs=[]),
            SimpleNamespace(
                installs=[
                    SimpleNamespace(app_id="example", status="succeeded", started_at=1)
                ]
            ),
        ]
    )
    monkeypatch.setattr(cloud.apps, "list_installs", request)
    monkeypatch.setattr(cloud.apps, "install", AsyncMock())
    device.list_apps = AsyncMock(
        return_value=[{"package_name": "example", "version_code": 2}]
    )
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(
            device.install_app(
                App("example", "1", "Example", id="id", version_code=3), timeout=0.05
            )
        )
