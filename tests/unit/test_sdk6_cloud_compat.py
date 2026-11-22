"""Exercise cloud adapters through the real SDK serializer and HTTP transport.

No SDK resources or response models are mocked: changes to their signatures,
request routing, raw-response API, or parsed models must reach these tests.
"""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from mobilerun_core_local.driver import cloud as cloud_module
from mobilerun_core_local.driver.base import DeviceDisconnectedError
from mobilerun_sdk import AsyncMobilerun

from mobilerun.clients.cloud import proxies
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.errors import CloudConflict, CloudConnectionError, CloudTimeout


@pytest.fixture(
    params=[(cloud_module.CloudDriver, False), (CloudDevice, True)],
    ids=["driver", "device"],
)
def cloud_factory(request, monkeypatch):
    clients = []
    factory_class, translates_errors = request.param

    @asynccontextmanager
    async def proxy_client(*args, **kwargs):
        yield clients[-1]

    monkeypatch.setattr(proxies, "cloud_client", proxy_client)

    def factory(handler, **kwargs):
        def sdk_client(**options):
            client = AsyncMobilerun(
                **options,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            clients.append(client)
            return client

        monkeypatch.setattr(cloud_module, "AsyncMobilerun", sdk_client)
        return factory_class(
            "device-123",
            api_key="test-key",
            base_url="https://sdk.test/v1",
            display_id=2,
            **kwargs,
        )

    factory.translates_errors = translates_errors
    yield factory
    for client in clients:
        asyncio.run(client.close())


def test_connect_screenshot_tap_and_apps_use_real_sdk(cloud_factory):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(
                200,
                json={
                    "deviceType": "android_cloud_phone",
                    "capabilities": {"telephony": True},
                },
            )
        if request.url.path.endswith("/number"):
            return httpx.Response(200, json={"data": None})
        if request.url.path.endswith("/proxy"):
            assert request.method == "GET"
            return httpx.Response(
                200,
                json={
                    "connected": True,
                    "name": "mrc:known-proxy",
                    "protocol": "socks5",
                },
            )
        assert request.headers["x-device-display-id"] == "2"
        if request.url.path.endswith("/screenshot"):
            return httpx.Response(
                200,
                content=b"\x89PNG\r\n\x1a\nfixture",
                headers={"content-type": "image/png"},
            )
        if request.url.path.endswith("/apps"):
            assert request.url.params["includeSystemApps"] == "false"
            return httpx.Response(
                200,
                json=[
                    {
                        "isSystemApp": False,
                        "label": "Example",
                        "packageName": "com.example",
                        "versionCode": 1,
                        "versionName": "1.0",
                    }
                ],
            )
        assert request.method == "POST"
        assert request.url.path == "/v1/devices/device-123/tap"
        assert json.loads(request.content) == {"x": 12, "y": 34, "stealth": True}
        return httpx.Response(204)

    driver = cloud_factory(handler, stealth=True)

    async def exercise():
        await driver.connect()
        if cloud_factory.translates_errors:
            await driver._ensure_connected()
            assert len(seen) == 3  # Capabilities, number binding and proxy adoption.
            assert driver.proxy.id == "known-proxy"
        else:
            await driver.ensure_connected()
            assert not seen
        assert await driver.screenshot() == b"\x89PNG\r\n\x1a\nfixture"
        await driver.tap(12, 34)
        apps = (
            await driver.list_apps(include_system=False)
            if cloud_factory.translates_errors
            else await driver.get_apps(include_system=False)
        )
        assert len(apps) == 1
        assert apps[0]["package_name"] == "com.example"
        assert apps[0]["label"] == "Example"
        assert apps[0]["is_system_app"] is False

    asyncio.run(exercise())
    assert len(seen) == (6 if cloud_factory.translates_errors else 3)
    assert driver._client.max_retries == 4
    assert driver._write_client.max_retries == 0


def test_remaining_driver_calls_preserve_wire_contract(cloud_factory):
    seen = []
    tree = {
        "a11y_tree": {
            "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 200}
        }
    }

    def handler(request):
        assert request.headers["x-device-display-id"] == "2"
        seen.append(
            (
                request.method,
                request.url.path,
                json.loads(request.content) if request.content else None,
            )
        )
        if request.url.path.endswith("/ui-state"):
            return httpx.Response(200, json=tree)
        if request.url.path.endswith("/time"):
            return httpx.Response(
                200, text="2026-11-17", headers={"content-type": "text/plain"}
            )
        if request.url.path.endswith("/packages"):
            assert request.url.params["includeSystemPackages"] == "false"
            return httpx.Response(200, json=["com.example"])
        return httpx.Response(204)

    driver = cloud_factory(handler)

    async def exercise():
        await driver.swipe(1, 2, 3, 4, duration_ms=500)
        assert await driver.input_text("hello", clear=True, stealth=True, wpm=70)
        await driver.press_button("enter")
        await driver.press_button("back")
        await driver.press_button("home")
        assert await driver.start_app("com.example") == "App started: com.example"
        await driver.start_app("com.example", activity="MainActivity")
        if cloud_factory.translates_errors:
            assert await driver._get_date() == "2026-11-17"
            result = await driver._get_ui_tree()
        else:
            assert await driver.list_packages() == ["com.example"]
            assert await driver.get_date() == "2026-11-17"
            result = await driver.get_ui_tree()
        assert result["a11y_tree"] == tree["a11y_tree"]
        assert result["device_context"]["screen_bounds"] == {
            "width": 100,
            "height": 200,
        }

    asyncio.run(exercise())
    prefix = "/v1/devices/device-123"
    assert seen[:7] == [
        (
            "POST",
            prefix + "/swipe",
            {"startX": 1, "startY": 2, "endX": 3, "endY": 4, "duration": 500},
        ),
        (
            "POST",
            prefix + "/keyboard",
            {"text": "hello", "clear": True, "stealth": True, "wpm": 70},
        ),
        ("PUT", prefix + "/keyboard", {"key": 66}),
        ("POST", prefix + "/global", {"action": 1}),
        ("POST", prefix + "/global", {"action": 2}),
        ("PUT", prefix + "/apps/com.example", {}),
        ("PUT", prefix + "/apps/com.example", {"activity": "MainActivity"}),
    ]


@pytest.mark.parametrize("failure", ["conflict", "connection", "timeout"])
def test_sdk_disconnect_errors_are_translated_without_retry(cloud_factory, failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "connection":
            raise httpx.ConnectError("offline", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(409, json={"detail": "device disconnected"})

    driver = cloud_factory(handler)
    expected = (
        {
            "conflict": CloudConflict,
            "connection": CloudConnectionError,
            "timeout": CloudTimeout,
        }[failure]
        if cloud_factory.translates_errors
        else DeviceDisconnectedError
    )
    with pytest.raises(expected):
        asyncio.run(driver.tap(1, 2))
    assert len(calls) == 1


def test_internal_auth_headers_survive_sdk_client_copy(cloud_factory):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer x"
        assert request.headers["x-user-id"] == "test-user"
        assert request.headers["x-owner-id"] == "test-owner"
        return httpx.Response(204)

    driver = cloud_factory(handler, user_id="test-user", owner_id="test-owner")
    asyncio.run(driver.tap(1, 2))
    assert len(seen) == 1


def test_cli_device_listing_uses_sdk_page_and_models(monkeypatch):
    import importlib
    from io import StringIO

    from rich.console import Console

    listing = importlib.import_module("mobilerun.cli.devices.list")
    clients = []
    output = StringIO()

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/v1/devices"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "device-123",
                        "name": "Example",
                        "state": "ready",
                        "type": "android_cloud_phone",
                        "platform": "android",
                        "activeTaskId": "",
                        "assignedAt": None,
                        "createdAt": "2026-11-17T00:00:00Z",
                        "updatedAt": "2026-11-17T00:00:00Z",
                        "stateMessage": "",
                        "streamUrl": "",
                        "taskCount": 0,
                        "terminatesAt": None,
                    }
                ],
                "pagination": {
                    "page": 1,
                    "pageSize": 20,
                    "total": 1,
                    "pages": 1,
                    "hasNext": False,
                    "hasPrev": False,
                },
            },
        )

    def sdk_client(**options):
        client = AsyncMobilerun(
            **options,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    from mobilerun.clients.cloud import client as transport

    monkeypatch.setattr(transport, "AsyncMobilerun", sdk_client)
    from mobilerun.cli.devices.base import discovery

    monkeypatch.setattr(discovery, "resolve_cloud_api_key", lambda: "test-key")
    monkeypatch.setattr(
        listing.render, "stdout", Console(file=output, force_terminal=False)
    )

    async def exercise():
        try:
            items = await discovery.discover_devices(
                local=False, base_url="https://sdk.test/v1"
            )
            assert len(items) == 1
            listing._render_cloud(items)
        finally:
            for client in clients:
                await client.close()

    asyncio.run(exercise())
    assert "device-123" in output.getvalue()
    assert "Example" in output.getvalue()
