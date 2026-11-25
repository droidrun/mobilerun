"""Device environment controls, app installation, and APK inspection."""

import asyncio
import json
import subprocess
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner
from mobilerun_core_local.driver.android import AndroidPortalHttpDriver
from mobilerun_sdk.types.shared.location import Location

from mobilerun.cli.devices.base import selection
from mobilerun.cli.devices.group import device as device_command
from mobilerun.cli.devices.settings import location as location_command
from mobilerun.clients.cloud import apps, devices
from mobilerun.clients.local import apk
from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.components.wrapper import DeviceWrapper, DriverDevice
from mobilerun.core.devices.ios import IOSDevice
from mobilerun.core.devices.pool import DevicePool
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.apps import App
from mobilerun.models.billing import Billing
from mobilerun.models.location import Location as DeviceLocation


def test_cloud_location_helper_and_device():
    async def run():
        value = Location(latitude=52.52, longitude=13.405)
        get = AsyncMock(return_value=value)
        client = SimpleNamespace(
            devices=SimpleNamespace(location=SimpleNamespace(get=get))
        )
        assert (
            await devices.get_location(
                "abc", billing=Billing.NO_COMMERCE, client=client
            )
            == value
        )
        phone = CloudDevice("abc", api_key="test")
        phone._client = client
        assert await phone.get_location() == DeviceLocation(
            latitude=52.52, longitude=13.405
        )
        assert get.await_count == 2
        get.assert_awaited_with("abc")

    asyncio.run(run())


def test_unsupported_and_wrapped_pool():
    async def run():
        with pytest.raises(DeviceOperationNotSupported, match="get_location"):
            await AndroidDevice(serial="emulator-5554").get_location()
        inner = Device()
        inner.get_location = AsyncMock(return_value=Location(latitude=1, longitude=2))
        assert (await DevicePool([DeviceWrapper(inner)]).get_location())[
            0
        ].latitude == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "value,args,expected",
    [
        (Location(latitude=52.52, longitude=13.405), [], "Location: 52.5200, 13.4050"),
        (None, [], "Location: Not available"),
        (
            Location(latitude=52.52, longitude=13.405),
            ["--json"],
            {"latitude": 52.52, "longitude": 13.405},
        ),
        (None, ["--json"], None),
    ],
)
def test_cli_location_get(monkeypatch, value, args, expected):
    phone = SimpleNamespace(
        get_location=AsyncMock(return_value=value), close=AsyncMock()
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=phone))
    result = CliRunner().invoke(
        device_command, ["settings", "location", "get", "--device", "abc", *args]
    )
    assert result.exit_code == 0, result.output
    assert (json.loads(result.output) if args else result.output.strip()) == expected
    phone.close.assert_awaited_once()


def test_location_menu_lists_get():
    result = CliRunner().invoke(device_command, ["settings", "location", "--help"])
    assert "get" in result.output


def test_interactive_location_menu_lists_get(monkeypatch):
    from mobilerun.cli.base import menu

    seen = []
    monkeypatch.setattr(menu, "is_interactive", lambda: True)

    def choose(label, choices):
        seen.extend(choice.value for choice in choices)
        return None

    monkeypatch.setattr(menu, "select_prompt", choose)
    CliRunner().invoke(location_command, [])
    assert {"get", "set", "reset"} <= set(seen)


def test_default_environment_is_unsupported():
    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(Device().open_url("https://example.com"))


def test_cloud_environment_calls_sdk():
    async def run():
        device = CloudDevice("abc", api_key="test")
        resources = SimpleNamespace(
            location=SimpleNamespace(set=AsyncMock(), reset=AsyncMock()),
            language=SimpleNamespace(set=AsyncMock()),
            timezone=SimpleNamespace(set=AsyncMock()),
            clipboard=SimpleNamespace(
                set=AsyncMock(),
                get=AsyncMock(return_value=SimpleNamespace(text="hello")),
            ),
            deep_link=SimpleNamespace(execute_deep_link=AsyncMock()),
        )
        device._write_client = cast(Any, SimpleNamespace(devices=resources))
        device._client = device._write_client
        await device.set_location(1.0, 2.0)
        await device.reset_location()
        await device.set_language("de-DE")
        await device.set_timezone("Europe/Berlin")
        await device.set_clipboard("hello")
        assert await device.get_clipboard() == "hello"
        await device.open_url("https://example.com")
        resources.location.set.assert_awaited_once_with(
            "abc", latitude=1.0, longitude=2.0
        )
        resources.location.reset.assert_awaited_once_with("abc")
        resources.language.set.assert_awaited_once_with("abc", locale="de-DE")
        resources.timezone.set.assert_awaited_once_with("abc", timezone="Europe/Berlin")
        resources.clipboard.set.assert_awaited_once_with("abc", text="hello")
        resources.deep_link.execute_deep_link.assert_awaited_once_with(
            "abc", deep_link="https://example.com"
        )

    asyncio.run(run())


def test_wrapper_and_pool_forward_open():
    async def run():
        inner = Device()
        inner.open_url = AsyncMock()
        wrapper = DeviceWrapper(inner)
        await wrapper.open_url("test://url")
        await DevicePool([wrapper]).open_url("test://url")
        assert inner.open_url.await_count == 2

    asyncio.run(run())


def test_android_open_quotes_and_rejects_invalid_settings():

    async def run():
        device = AndroidDevice(serial="serial")
        device._ensure_connected = AsyncMock()
        shell = AsyncMock(return_value=SimpleNamespace(returncode=0, output=""))
        device._adb_device = cast(Any, SimpleNamespace(shell2=shell, serial="serial"))
        await device.open_url("https://example.com/?a=1&b='two'")
        assert shell.await_args is not None
        assert (
            shell.await_args.args[0]
            == "am start -a android.intent.action.VIEW -d 'https://example.com/?a=1&b='\"'\"'two'\"'\"''"
        )
        with pytest.raises(ValueError):
            await device.set_timezone("../bad")
        with pytest.raises(DeviceOperationNotSupported, match="physical Android"):
            await device.set_location(1, 2)

    asyncio.run(run())


def test_emulator_reset_explains_location_persists():

    with pytest.raises(
        DeviceOperationNotSupported,
        match="emulators keep the last fixed location; set a new one instead",
    ):
        asyncio.run(AndroidDevice(serial="emulator-5554").reset_location())


def test_emulator_geo_fix_and_timezone():

    async def run():
        device = AndroidDevice(serial="emulator-5554")
        device._ensure_connected = AsyncMock()
        adb_output = Mock(return_value="OK")
        shell = AsyncMock(return_value=SimpleNamespace(returncode=0, output=""))
        device._adb_device = cast(
            Any,
            SimpleNamespace(
                serial="emulator-5554", adb_output=adb_output, shell2=shell
            ),
        )
        await device.set_location(52.52, 13.4)
        adb_output.assert_called_once_with("emu", "geo", "fix", "13.4", "52.52")
        await device.set_timezone("Europe/Berlin")
        shell.assert_awaited_once_with("cmd alarm set-timezone Europe/Berlin")
        with pytest.raises(ValueError):
            await device.set_timezone("Europe/Berlin;reboot")
        shell.return_value = SimpleNamespace(returncode=1, output="Permission denial")
        with pytest.raises(DeviceOperationNotSupported, match="Permission denial"):
            await device.set_timezone("Europe/Berlin")
        adb_output.return_value = "KO: denied"
        with pytest.raises(DeviceOperationNotSupported, match="rejected"):
            await device.set_location(0, 0)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("set_location", (1.0, 2.0)),
        ("reset_location", ()),
        ("set_timezone", ("Europe/Berlin",)),
        ("open_url", ("https://example.com",)),
        ("set_language", ("de-DE",)),
        ("set_clipboard", ("hello",)),
        ("get_clipboard", ()),
    ],
)
def test_wrapper_pool_delegate_each_environment_method(method, args):
    async def run():
        inner = Device()
        mock = AsyncMock(return_value="value" if method == "get_clipboard" else None)
        setattr(inner, method, mock)
        wrapper = DeviceWrapper(inner)
        assert await getattr(wrapper, method)(*args) == (
            "value" if method == "get_clipboard" else None
        )
        assert await getattr(DevicePool([wrapper]), method)(*args) == (
            ["value"] if method == "get_clipboard" else [None]
        )
        assert mock.await_count == 2
        mock.assert_awaited_with(*args)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("set_location", (1.0, 2.0)),
        ("reset_location", ()),
        ("set_timezone", ("Europe/Berlin",)),
        ("open_url", ("https://example.com",)),
    ],
)
@pytest.mark.parametrize("platform", ["ios", "remote"])
def test_unsupported_ios_remote(method, args, platform):
    from mobilerun.core.devices.remote import RemoteDevice

    device = (
        IOSDevice("http://localhost:1")
        if platform == "ios"
        else RemoteDevice("http://localhost:1")
    )
    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(device, method)(*args))


@pytest.mark.parametrize("serial", ["emulator-5554", "physical-serial"])
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("reset_location", ()),
    ],
)
def test_android_unsupported_environment_operations_on_both_kinds(serial, method, args):

    with pytest.raises(DeviceOperationNotSupported):
        asyncio.run(getattr(AndroidDevice(serial=serial), method)(*args))


def test_device_protocol_declares_app_installation():
    assert callable(Device.__dict__.get("install_app"))
    assert callable(Device.__dict__.get("uninstall_app"))


def test_android_portal_app_operations_raise_domain_error():
    from mobilerun.core.state.options import android_state_provider

    device = DriverDevice(
        AndroidPortalHttpDriver("http://localhost:8080", "test"),
        state_provider_factory=android_state_provider,
    )
    with pytest.raises(DeviceOperationNotSupported, match="cannot install"):
        asyncio.run(device.install_apk("app.apk"))
    with pytest.raises(DeviceOperationNotSupported, match="cannot uninstall"):
        asyncio.run(device.uninstall_app("org.example"))


def test_ios_uninstall_raises_domain_error():

    device = IOSDevice("http://localhost")
    from mobilerun_core_local.driver.ios import IOSPortalDriver

    device._driver = IOSPortalDriver("http://localhost")
    with pytest.raises(DeviceOperationNotSupported, match="uninstall"):
        asyncio.run(device.uninstall_app("org.example"))


def test_android_local_install_delegates(monkeypatch):

    from mobilerun_core_local.driver.android import AndroidDriver

    install = AsyncMock(return_value="installed")
    monkeypatch.setattr(AndroidDriver, "install_app", install)
    monkeypatch.setattr(
        "mobilerun.core.devices.android.inspect_apk",
        lambda path: ("org.example", 1, "1", "Example"),
    )

    device = AndroidDevice("fake")
    device._driver.ensure_connected = AsyncMock()
    device._adb_device = SimpleNamespace(
        shell=AsyncMock(return_value="package:/data/app/base.apk")
    )
    assert (
        asyncio.run(device.install_apk("app.apk", reinstall=True)).package
        == "org.example"
    )
    install.assert_awaited_once_with("app.apk", reinstall=True)


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

    import mobilerun.core.devices.components.cloud_apps as cloud

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

    import mobilerun.core.devices.components.cloud_apps as cloud

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


@pytest.fixture
def fake_aapt(monkeypatch, tmp_path):
    path = tmp_path / "app.apk"
    path.write_bytes(b"test apk")
    monkeypatch.setattr(apk.shutil, "which", lambda name: str(tmp_path / "aapt"))
    run = Mock()
    monkeypatch.setattr(apk.subprocess, "run", run)
    return path, run


@pytest.mark.parametrize(
    "fields",
    ["versionCode='3'", "name='org.example'"],
    ids=["no-name", "no-version-code"],
)
@pytest.mark.parametrize("operation", ["install", "upload"])
def test_incomplete_apk_metadata_stops_operation(
    monkeypatch, fake_aapt, fields, operation
):
    path, run = fake_aapt
    run.return_value.stdout = f"package: {fields}\napplication-label-de:'Beispiel'\n"
    if operation == "install":
        device = AndroidDevice(serial="test")
        downstream = AsyncMock()
        monkeypatch.setattr(device, "install_app", downstream)
        result = device.install_apk(path)
    else:
        downstream = Mock(side_effect=AssertionError("unexpected cloud client"))
        monkeypatch.setattr(apps, "cloud_client", downstream)
        result = apps.upload_app(path, billing=Billing.NO_COMMERCE)
    with pytest.raises(ValueError, match="missing package or version"):
        asyncio.run(result)
    downstream.assert_not_called()


def test_apk_inspection_timeout(fake_aapt):
    path, run = fake_aapt
    run.side_effect = subprocess.TimeoutExpired("aapt", 30)
    with pytest.raises(RuntimeError, match="APK inspection timed out"):
        apk.inspect_apk(path)


@pytest.mark.parametrize(
    "default_label, expected",
    [("", "org.example"), ("application-label:'Example'", "Example")],
)
def test_apk_label_uses_default_or_package(fake_aapt, default_label, expected):
    path, run = fake_aapt
    run.return_value.stdout = (
        "package: name='org.example' versionCode='3' versionName='1.2'\n"
        f"application-label-de:'Beispiel'\n{default_label}\n"
    )
    assert apk.inspect_apk(path) == ("org.example", 3, "1.2", expected)


def test_missing_aapt_explains_build_tools(monkeypatch, tmp_path):

    monkeypatch.setattr(apk.shutil, "which", lambda name: None)
    monkeypatch.setattr(apk.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="Android build-tools"):
        apk.inspect_apk(tmp_path / "app.apk")


def test_recording_device_records_package_string():
    from mobilerun.core.devices.recording import RecordingDevice

    inner = Device()
    inner.start_app = AsyncMock(return_value="ok")
    device = RecordingDevice(inner)
    app = App("org.example", "1", "Example")
    assert asyncio.run(device.start_app(app.package)) == "ok"
    assert device.recorded_actions[-1]["package"] == "org.example"


def test_cloud_install_rejects_stale_installed_version(monkeypatch):
    import mobilerun.core.devices.components.cloud_apps as cloud

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
