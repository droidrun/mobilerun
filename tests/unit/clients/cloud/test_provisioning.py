"""Cloud provisioning, ownership, billing guardrails and SDK wire contracts."""

import ast
import asyncio
import inspect
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_core_local.driver import cloud as cloud_module
from mobilerun_core_local.driver.base import DeviceDisconnectedError
from mobilerun_sdk import APIStatusError, AsyncMobilerun

from mobilerun.clients.cloud import devices, numbers, payments, proxies
from mobilerun.core.devices import AndroidDevice, CloudDevice, DevicePool
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components import environment
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.errors import (
    BillingError,
    CloudConflict,
    CloudConnectionError,
    CloudError,
    CloudPermissionDenied,
    CloudTimeout,
    PurchaseInProgress,
)
from mobilerun.models import Billing
from mobilerun.models.checkout import PaymentSnapshot
from tests.support.paths import REPO_ROOT


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    from mobilerun_core_local.driver import cloud as driver

    devices = NS(
        retrieve_capabilities=AsyncMock(
            return_value=NS(capabilities=NS(telephony=False, proxy=True))
        ),
        create=AsyncMock(return_value=NS(id="created", name="created")),
        wait_ready=AsyncMock(),
        terminate=AsyncMock(),
        number=NS(retrieve=AsyncMock(return_value=NS(binding=None))),
        actions=NS(tap=AsyncMock()),
    )
    client = NS(devices=devices, with_options=lambda **kw: client, close=AsyncMock())
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **kw: client)
    return devices


def test_create_on_connect_and_cleanup(cloud, monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_DEVICE_TYPE", "ios_phone")

    async def run():
        device = CloudDevice(api_key="test", name="example")
        assert device.device_id is None
        await asyncio.gather(device.connect(), device.connect())
        assert device.device_id == "created"
        cloud.create.assert_awaited_once_with(
            device_type="ios_phone",
            name="example",
            billing="auto",
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        await device.close()
        await device.close()
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


@pytest.mark.parametrize("keep", [False, True])
def test_context_and_keep(cloud, keep):
    async def run():
        async with CloudDevice(
            api_key="test", keep=keep, device_type="device_slot"
        ) as device:
            assert device.identifier == "created"
            assert device._keep is keep
        cloud.create.assert_awaited_once_with(
            device_type="device_slot",
            billing="auto",
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        assert cloud.terminate.await_count == (0 if keep else 1)

    asyncio.run(run())


def test_existing_id_is_never_owned(cloud):
    async def run():
        phone = CloudDevice("existing", api_key="test")
        from mobilerun.core.proxy import ProxyStatus

        phone._proxy_status = AsyncMock(return_value=ProxyStatus(False))
        async with phone:
            pass
        phone._proxy_status.assert_awaited_once()
        cloud.create.assert_not_awaited()
        cloud.terminate.assert_not_awaited()

    asyncio.run(run())


def test_pool_lifecycle(cloud):
    async def run():
        phone = CloudDevice(api_key="test")
        async with DevicePool([phone]):
            assert phone.device_id == "created"
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_action_requires_awaited_construction(cloud):
    async def run():
        device = CloudDevice(api_key="test")
        with pytest.raises(RuntimeError, match="Await"):
            await device.tap(1, 2)
        cloud.create.assert_not_awaited()
        await device
        await device.tap(1, 2)
        assert cloud.actions.tap.await_args.args == ("created",)
        await device.close()

    asyncio.run(run())


def test_creation_failure_cleans_up(cloud):
    cloud.wait_ready.side_effect = RuntimeError("not ready")

    async def run():
        device = CloudDevice(api_key="test")
        with pytest.raises(RuntimeError, match="not ready"):
            await device
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_auto_create_accepts_device_options(cloud):
    from mobilerun.core.operator import Operator

    operator = Operator()
    device = CloudDevice(api_key="test", operator=operator, name="test")
    assert device._operator is operator
    asyncio.run(device.connect())
    cloud.create.assert_awaited_once_with(
        device_type="android_cloud_phone",
        name="test",
        billing="auto",
        proxy={"connect": {"country": "us"}, "smart_ip": True},
    )


def test_documented_creation_options_reach_sdk_and_name_comes_from_server(cloud):
    cloud.create.return_value = NS(id="created", name="server-name")

    async def run():
        phone = await CloudDevice(
            api_key="test",
            name="requested",
            country="DE",
            locale="de-DE",
            timezone="Europe/Berlin",
            android_version=15,
            apps=["com.android.chrome"],
        )
        assert phone.name == "server-name"
        cloud.create.assert_awaited_once_with(
            device_type="android_cloud_phone",
            billing="auto",
            name="requested",
            body_country="DE",
            locale="de-DE",
            timezone="Europe/Berlin",
            android_version=15,
            apps=["com.android.chrome"],
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        await phone.close()

    asyncio.run(run())


def test_template_is_forwarded_without_request_apps_or_version(cloud):
    phone = CloudDevice(api_key="test", name="prepared", template="onboarding")
    asyncio.run(phone.connect())
    assert cloud.create.await_args.kwargs["template"] == "onboarding"
    assert "apps" not in cloud.create.await_args.kwargs
    assert "android_version" not in cloud.create.await_args.kwargs


def test_pool_explicit_close(cloud):
    async def run():
        pool = DevicePool([CloudDevice(api_key="test")])
        await pool.__aenter__()
        await pool.close()
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_pool_context_cleans_up_if_another_device_cannot_connect(cloud):
    class Broken(Device):
        @property
        def needs_initialization(self):
            return False

        async def _ensure_connected(self):
            raise RuntimeError("connect failed")

        async def close(self):
            pass

    async def run():
        with pytest.raises(ExceptionGroup):
            async with DevicePool([CloudDevice(api_key="test"), Broken()]):
                pass
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


def test_owned_cloud_phone_is_terminated_through_wrapper(cloud):
    from mobilerun.core.devices.recording import RecordingDevice

    async def run():
        async with RecordingDevice(
            CloudDevice(api_key="test", billing="paid", plan="minute")
        ) as device:
            assert device.identifier == "created"
        cloud.create.assert_awaited_once_with(
            device_type="android_cloud_phone",
            billing="minute",
            proxy={"connect": {"country": "us"}, "smart_ip": True},
        )
        cloud.terminate.assert_awaited_once_with("created")

    asyncio.run(run())


OPTIONS = dict(
    name="named",
    country="DE",
    locale="de_DE",
    timezone="Europe/Berlin",
    android_version=15,
    apps=["app"],
    template="onboarding",
)


@pytest.mark.parametrize("key,value", OPTIONS.items())
def test_creation_options_are_explicit_and_forbidden_for_existing_phone(key, value):
    assert key in inspect.signature(CloudDevice).parameters
    with pytest.raises(TypeError, match="only used when creating a phone"):
        CloudDevice("existing", api_key="fake", **{key: value})


def test_unknown_options_fail_before_transport():
    with pytest.raises(TypeError, match="unexpected"):
        CloudDevice(api_key="fake", naem="typo")


def test_name_is_read_only_and_refreshed_from_info_and_rename(monkeypatch):
    phone = CloudDevice("existing", api_key="fake")
    monkeypatch.setattr(devices, "retrieve", AsyncMock(return_value=NS(name="old")))
    rename = AsyncMock(return_value=NS(name="server normalized"))
    monkeypatch.setattr(devices, "set_name", rename)

    async def run():
        assert phone.name is None
        await phone.info()
        assert phone.name == "old"
        await phone.rename("new")
        assert phone.name == "server normalized"
        rename.assert_awaited_once_with(
            "existing", "new", billing=phone._billing, client=phone._cloud_client
        )

    asyncio.run(run())
    with pytest.raises(AttributeError):
        phone.name = "local"


def test_set_name_boundary_uses_sdk_and_free_billing():
    endpoint = AsyncMock(return_value=NS(name="new"))
    result = asyncio.run(
        devices.set_name(
            "id", "new", billing="no_commerce", client=NS(devices=NS(set_name=endpoint))
        )
    )
    assert result.name == "new"
    endpoint.assert_awaited_once_with("id", name="new")


def test_number_waits_for_active_with_configured_timeout(monkeypatch, caplog):
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    monkeypatch.setenv("MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS", "5")
    retrieve = AsyncMock(
        side_effect=[
            NS(data=NS(state="provisioning", checkout_url=None)),
            NS(data=NS(state="active", phone_number="example", checkout_url=None)),
        ]
    )
    monkeypatch.setattr(numbers, "retrieve", retrieve)
    monkeypatch.setattr("mobilerun.core.phone_number.cloud.asyncio.sleep", AsyncMock())
    number = CloudPhoneNumber("id", billing="no_commerce")
    with caplog.at_level("DEBUG", logger="mobilerun.core.phone_number.cloud"):
        assert asyncio.run(number.wait_ready()) is number
    assert number.phone_number == "example"
    assert "provisioning" in caplog.text and "active" in caplog.text


@pytest.mark.parametrize("state", ["failed", "expired", "cancel_scheduled"])
def test_number_wait_rejects_terminal_states(monkeypatch, state):
    monkeypatch.setattr(
        numbers,
        "retrieve",
        AsyncMock(return_value=NS(data=NS(state=state, checkout_url=None))),
    )
    with pytest.raises(CloudError) as caught:
        asyncio.run(CloudPhoneNumber("id").wait_ready())
    assert caught.value.status == state


def test_rename_maps_sdk_errors_without_changing_cached_name():
    phone = CloudDevice("id", api_key="fake")
    phone._name = "unchanged"
    error = APIStatusError(
        "private",
        response=httpx.Response(
            403, request=httpx.Request("PUT", "https://example.test")
        ),
        body={},
    )
    phone._client = NS(devices=NS(set_name=AsyncMock(side_effect=error)))
    with pytest.raises(CloudPermissionDenied):
        asyncio.run(phone.rename("new"))
    assert phone.name == "unchanged"


@pytest.mark.parametrize("timeout", [0, -1])
def test_number_wait_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="positive"):
        asyncio.run(CloudPhoneNumber("id").wait_ready(timeout=timeout))


def test_number_wait_explicit_timeout_overrides_setting(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS", "600")
    monkeypatch.setattr(
        numbers,
        "retrieve",
        AsyncMock(return_value=NS(data=NS(state="provisioning", checkout_url=None))),
    )
    with pytest.raises(TimeoutError, match="0.001s.*provisioning"):
        asyncio.run(CloudPhoneNumber("id").wait_ready(timeout=0.001))


def test_number_wait_timeout_reports_last_state(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CLOUD_NUMBER_READY_TIMEOUT_SECONDS", "0.001")
    monkeypatch.setattr(
        numbers,
        "retrieve",
        AsyncMock(return_value=NS(data=NS(state="provisioning", checkout_url=None))),
    )
    with pytest.raises(TimeoutError, match="provisioning"):
        asyncio.run(CloudPhoneNumber("id").wait_ready())


@pytest.mark.parametrize("resource", [CloudPhoneNumber, CloudMailbox])
@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
def test_existing_checkout_failure_keeps_resource(resource, failure):
    data = NS(
        status="awaiting_payment",
        state="awaiting_payment",
        address=None,
        phone_number=None,
        device_binding=None,
        checkout_expires_at=None,
        checkout_url="https://checkout.test",
    )
    endpoint = NS(retrieve=AsyncMock(return_value=NS(data=data)), delete=AsyncMock())
    sdk = NS(numbers=endpoint, mailboxes=endpoint)

    async def run():
        with pytest.raises(failure):
            await resource(
                "existing", client=sdk, on_checkout=AsyncMock(side_effect=failure)
            )
        endpoint.delete.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"label": "work"},
        {"label": "personal", "idempotency_key": "new-request"},
        {"label": "personal", "extra_body": {"label": "work"}},
        {"label": "personal", "extra_query": {"country": "GB"}},
        {"label": "personal", "extra_headers": {"Idempotency-Key": "new-request"}},
    ],
)
def test_pending_purchase_requires_verifiable_identity(options):
    conflict = APIStatusError(
        "conflict",
        response=httpx.Response(
            409, request=httpx.Request("POST", "https://example.test")
        ),
        body={"code": "purchase_in_progress"},
    )
    pending = NS(
        id="pending",
        state="awaiting_payment",
        country_code="US",
        label="personal",
        checkout_url="https://checkout.test",
    )
    sdk = NS(
        numbers=NS(
            create=AsyncMock(side_effect=conflict),
            list=AsyncMock(
                return_value=NS(items=[pending], pagination=NS(has_next=False))
            ),
        )
    )
    with pytest.raises(CloudConflict) as caught:
        asyncio.run(
            numbers.create(client=sdk, billing=Billing.PAID, country="US", **options)
        )
    assert not isinstance(caught.value, PurchaseInProgress)


@pytest.mark.parametrize("intermediate", [[], ["provisioning"]])
def test_number_wait_ready_uses_activation_seen_during_checkout_poll(
    monkeypatch, intermediate
):
    retrieve = AsyncMock(
        side_effect=[
            NS(data=NS(state="awaiting_payment", checkout_url=None, phone_number=None)),
            *(
                NS(data=NS(state=state, checkout_url=None, phone_number=None))
                for state in intermediate
            ),
            NS(data=NS(state="active", checkout_url=None, phone_number="+123")),
        ]
    )
    number = CloudPhoneNumber("existing", client=NS(numbers=NS(retrieve=retrieve)))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    assert asyncio.run(number.wait_ready()) is number
    assert number.phone_number == "+123"


def test_paid_binding_smoke_calls_match_device_client_signatures():
    # Find the workflow by test name, not filename: smoke modules can be regrouped.
    name = "test_paid_number_bind_unbind_and_release"
    workflows = []
    for path in (REPO_ROOT / "tests/unit/smoke").glob("test_*.py"):
        source = path.read_text()
        if name in source:
            workflows.extend(
                node
                for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            )
    assert len(workflows) == 1, f"Expected one smoke workflow named {name}"
    # The workflow uses `phone` for the owned device and `observer` for the borrowed one.
    attributes = {
        node.attr
        for node in ast.walk(workflows[0])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"phone", "observer"}
    }
    assert {"number", "actions", "close"} <= attributes
    unknown = {attr for attr in attributes if not hasattr(CloudDevice, attr)}
    assert not unknown, f"Unknown CloudDevice attributes: {sorted(unknown)}"
    operations = set()
    for call in ast.walk(workflows[0]):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "devices"
        ):
            operations.add(call.func.attr)
            inspect.signature(getattr(devices, call.func.attr)).bind(
                *[None for _ in call.args], **{kw.arg: None for kw in call.keywords}
            )
    assert {"number", "list_devices"} <= operations, operations


@pytest.mark.parametrize(
    "billing,plan,country",
    [("included", "auto", "US"), ("paid", "included", "US"), ("paid", "rent", "GB")],
)
def test_incompatible_pending_purchase_does_not_open_checkout(billing, plan, country):
    response = httpx.Response(
        409, request=httpx.Request("POST", "https://example.test")
    )
    conflict = APIStatusError(
        "conflict", response=response, body={"code": "purchase_in_progress"}
    )
    pending = NS(
        id="pending",
        state="awaiting_payment",
        country_code="US",
        label=None,
        checkout_url="https://checkout.test",
    )
    sdk = NS(
        numbers=NS(
            create=AsyncMock(side_effect=conflict),
            list=AsyncMock(
                return_value=NS(items=[pending], pagination=NS(has_next=False))
            ),
        )
    )
    callback = AsyncMock(side_effect=AssertionError("unexpected checkout"))

    async def run():
        with pytest.raises(CloudConflict) as caught:
            await CloudPhoneNumber(
                client=sdk,
                country=country,
                billing=billing,
                plan=plan,
                on_checkout=callback,
            )
        assert (
            not isinstance(caught.value, PurchaseInProgress)
            or caught.value.checkout is None
        )
        callback.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("billing", [Billing.NO_COMMERCE, Billing.INCLUDED])
def test_sms_requires_paid_before_transport(billing):
    send = AsyncMock()
    sdk = NS(numbers=NS(messages=NS(send=send)))
    with pytest.raises(BillingError):
        asyncio.run(
            numbers.send("existing", "+123", "test", billing=billing, client=sdk)
        )
    send.assert_not_awaited()


def test_checking_proxy_returns_at_provisioning_without_checkout_or_cancel(monkeypatch):
    provisioning = PaymentSnapshot("provisioning")
    retrieve = AsyncMock(side_effect=[provisioning, TimeoutError()])
    callback, cancel = AsyncMock(), AsyncMock()
    adapter = payments.PaymentAdapter(
        "Connect proxy", "proxy", retrieve, cancel, preparing=frozenset({"checking"})
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    result = asyncio.run(
        payments.wait_for_payment(
            adapter,
            PaymentSnapshot("checking"),
            resume=AsyncMock(),
            on_checkout=callback,
        )
    )
    assert result == provisioning
    retrieve.assert_awaited_once()
    callback.assert_not_awaited()
    cancel.assert_not_awaited()


@pytest.mark.parametrize("states", [["active"], ["provisioning", "active"]])
def test_activation_without_checkout_url(monkeypatch, states):
    retrieve = AsyncMock(side_effect=[PaymentSnapshot(state) for state in states])
    adapter = payments.PaymentAdapter("number", "existing", retrieve, AsyncMock())
    callback = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    result = asyncio.run(
        payments.wait_for_payment(
            adapter,
            PaymentSnapshot("awaiting_payment"),
            resume=AsyncMock(),
            on_checkout=callback,
        )
    )
    assert result.status == "active"
    callback.assert_not_awaited()


@pytest.mark.parametrize("resource", [CloudPhoneNumber, CloudMailbox])
def test_create_without_checkout_url_returns_usable_resource(monkeypatch, resource):
    def response(state):
        if resource is CloudPhoneNumber:
            return NS(
                data=NS(
                    number_id="n",
                    state=state,
                    checkout_url=None,
                    phone_number="+123" if state == "active" else None,
                )
            )
        return NS(
            data=NS(
                id="m",
                status=state,
                checkout_url=None,
                checkout_expires_at=None,
                address="a@example.test" if state == "active" else None,
            )
        )

    states = iter(["provisioning", "active"])
    endpoint = NS(
        create=AsyncMock(return_value=response("awaiting_payment")),
        retrieve=AsyncMock(side_effect=lambda _: response(next(states, "active"))),
        delete=AsyncMock(),
    )
    sdk = NS(numbers=endpoint, mailboxes=endpoint)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    async def run():
        async with resource(client=sdk, billing=Billing.PAID) as handle:
            if resource is CloudPhoneNumber:
                assert (await handle.wait_ready()).phone_number == "+123"
            else:
                assert handle.address == "a@example.test"
        endpoint.delete.assert_awaited_once_with(handle.id)

    asyncio.run(run())


@pytest.mark.parametrize("cloud", [False, True])
def test_device_catalog_dispatches_open_url(monkeypatch, cloud):
    device = (
        CloudDevice("existing", api_key="test") if cloud else AndroidDevice("local")
    )
    backend = AsyncMock()
    monkeypatch.setattr(
        environment, "cloud_open_url" if cloud else "android_shell", backend
    )
    url = "https://example.test"
    asyncio.run(device.actions.call("device.open_url", {"url": url}))
    backend.assert_awaited_once_with(
        device, url if cloud else f"am start -a android.intent.action.VIEW -d {url}"
    )


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
