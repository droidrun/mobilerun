"""Phone number and mailbox policy, ownership, polling, and SDK contracts."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_sdk import APIStatusError

from mobilerun.cli.phone_number import group
from mobilerun.clients.cloud import numbers
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.config.settings import get_settings
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.mailbox import CloudMailbox as Mailbox
from mobilerun.core.phone_number import CloudPhoneNumber as PhoneNumber
from mobilerun.errors import CloudConflict, DeviceOperationNotSupported
from mobilerun.errors.base import is_value_safe
from mobilerun.errors.cloud import CloudAPIError
from mobilerun.models.billing import Billing
from mobilerun.utils import codes
from mobilerun.utils.codes import extract_code
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


def run(coro):
    async def execute():
        return await coro

    return asyncio.run(execute())


def test_shared_code_poller_returns_code_for_sender():

    async def fetch(sender):
        return "123456" if sender == "example" else None

    assert run(codes.wait_for_code(fetch, sender="example")) == "123456"


def page(items, has_next=False):
    return NS(items=items, pagination=NS(has_next=has_next))


@pytest.fixture
def polling_clock(monkeypatch):
    # Advance only the poller's clock, not the event loop's monotonic clock.
    clock = NS(now=0.0)

    async def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(codes, "time", NS(monotonic=lambda: clock.now))
    monkeypatch.setattr(codes, "asyncio", NS(sleep=sleep, wait_for=asyncio.wait_for))
    return clock


def test_delayed_empty_reply_spends_original_deadline(polling_clock):
    async def delayed_reply(sender):
        assert polling_clock.now == 0, "fetch retried after the original deadline"
        polling_clock.now += 2.5
        return ""

    fetch = AsyncMock(side_effect=delayed_reply)
    with pytest.raises(TimeoutError, match="verification code"):
        run(codes.wait_for_code(fetch, timeout=3, sender="bank"))
    fetch.assert_awaited_once_with("bank")
    assert polling_clock.now == 3


def test_retry_preserves_sender_after_empty_reply(polling_clock):
    fetch = AsyncMock(side_effect=["", "001234"])
    assert run(codes.wait_for_code(fetch, timeout=3, sender="bank")) == "001234"
    assert [call.args for call in fetch.await_args_list] == [("bank",), ("bank",)]


def test_cancelling_poller_cancels_inflight_fetch_without_retry():
    async def exercise():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending(sender):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        fetch = AsyncMock(side_effect=pending)
        task = asyncio.create_task(codes.wait_for_code(fetch, timeout=1, sender="bank"))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        fetch.assert_awaited_once_with("bank")

    run(exercise())


@pytest.fixture
def sdk():
    numbers = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(number_id="number-id", state="provisioning", checkout_url=None)
            )
        ),
        retrieve=AsyncMock(
            return_value=NS(
                data=NS(
                    device_binding=None,
                    phone_number=None,
                    state="active",
                    checkout_url=None,
                )
            )
        ),
        list=AsyncMock(
            return_value=page(
                [NS(id="existing-number", phone_number=None, device_binding=None)]
            )
        ),
        delete=AsyncMock(),
        update=AsyncMock(),
        countries=AsyncMock(),
        capacity=AsyncMock(),
        messages=NS(list=AsyncMock(return_value=page([])), send=AsyncMock()),
    )
    mailboxes = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(
                    id="mailbox-id",
                    address="test@example.com",
                    status="active",
                    checkout_url=None,
                )
            )
        ),
        retrieve=AsyncMock(
            return_value=NS(
                data=NS(address="test@example.com", status="active", checkout_url=None)
            )
        ),
        list=AsyncMock(return_value=page([NS(id="existing-mailbox", address=None)])),
        delete=AsyncMock(),
        capacity=AsyncMock(),
        restart=AsyncMock(),
        otp=AsyncMock(return_value=NS(data=None)),
        messages=NS(list=AsyncMock(return_value=NS(items=[], next_cursor=None))),
    )
    binding = NS(
        retrieve=AsyncMock(return_value=NS(binding=None)),
        bind=AsyncMock(),
        unbind=AsyncMock(),
    )
    return NS(
        numbers=numbers,
        mailboxes=mailboxes,
        devices=NS(
            number=binding,
            retrieve_capabilities=AsyncMock(
                return_value=NS(capabilities=NS(telephony=True, proxy=True))
            ),
        ),
    )


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
@pytest.mark.parametrize(
    "billing,enabled,expected",
    [
        (None, "included", "included_only"),
        (None, "paid", "included"),
        ("paid", "no_commerce", "included"),
        ("included", "no_commerce", "included_only"),
        ("included", "paid", "included_only"),
    ],
)
def test_create_billing(sdk, monkeypatch, resource, billing, enabled, expected):
    monkeypatch.setenv("MOBILERUN_BILLING", enabled)
    assert get_settings().MOBILERUN_BILLING == enabled
    item = run(resource(client=sdk, billing=billing))
    adapter = sdk.numbers if resource is PhoneNumber else sdk.mailboxes
    adapter.create.assert_awaited_once_with(billing_preference=expected)
    assert item.id


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
def test_no_paid_fallback_and_actionable_error(sdk, resource):
    response = httpx.Response(
        402, request=httpx.Request("POST", "https://example.invalid")
    )
    adapter = sdk.numbers if resource is PhoneNumber else sdk.mailboxes
    adapter.create.side_effect = APIStatusError(
        "capacity unavailable", response=response, body={}
    )
    with pytest.raises(
        CloudAPIError,
    ):
        run(resource(client=sdk, billing="included"))
    assert adapter.create.await_count == 1


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
def test_resource_requires_id_get_verifies_and_list_is_full(sdk, resource):
    with pytest.raises(RuntimeError, match="Await"):
        _ = resource().id
    with pytest.raises(ValueError):
        resource("")
    assert resource("existing").id == "existing"
    item = run(resource("existing", client=sdk))
    adapter = sdk.numbers if resource is PhoneNumber else sdk.mailboxes
    adapter.retrieve.assert_awaited_once_with("existing")
    assert item.id == "existing"
    assert run(resource.list(client=sdk))[0].id.startswith("existing-")
    run(item.release())
    adapter.delete.assert_awaited_once_with("existing")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Code: 0123", "0123"),
        ("code 12345678!", "12345678"),
        ("123456789", None),
        ("123", None),
        ("no code", None),
    ],
)
def test_extract_code(text, expected):
    assert extract_code(text) == expected


def test_sms_wait_filters_sender_and_after_and_send(sdk):
    after = datetime.now(timezone.utc)
    message = {
        "id": "id",
        "createdAt": after,
        "occurredAt": after,
        "direction": "inbound",
        "status": "received",
    }
    sdk.numbers.messages.list.return_value = page(
        [
            numbers.MessageItem.model_validate(
                {**message, "detectedSender": "Other", "body": "Code 1111"}
            ),
            numbers.MessageItem.model_validate(
                {**message, "detectedSender": "Example", "body": "Code 012345"}
            ),
        ]
    )
    sdk.numbers.messages.send.return_value = numbers.MessageSendResponse.model_validate(
        {"data": message}
    )
    number = PhoneNumber("number-id", client=sdk)
    device = MemoryDevice(
        cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
    )
    run(number.bind(device))
    assert run(number.wait_for_code(sender="example", after=after)) == "012345"
    sdk.numbers.messages.list.assert_awaited_once_with(
        "number-id",
        page=1,
        page_size=100,
        direction="inbound",
        since=after.isoformat().replace("+00:00", "Z"),
    )
    run(number.send("destination", "body"))
    sdk.numbers.messages.send.assert_awaited_once_with(
        "number-id", to="destination", body="body"
    )


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
def test_wait_timeout_is_clear(sdk, resource):
    from mobilerun.errors import ExtensionActionError

    item = resource("id", client=sdk)
    device = MemoryDevice(
        cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
    )
    run(item.bind(device))
    with pytest.raises(ExtensionActionError, match="failed") as caught:
        run(item.wait_for_code(timeout=0.001))
    assert isinstance(caught.value.__context__, TimeoutError)


def test_mailbox_otp_filters_and_cursor_messages(sdk):
    sdk.mailboxes.otp.return_value = NS(data=NS(code="012345"))
    mailbox = Mailbox("id", client=sdk)
    run(mailbox.bind(MemoryDevice()))
    assert run(mailbox.wait_for_code(sender="example")) == "012345"
    assert sdk.mailboxes.otp.call_args.kwargs["sender"] == "example"
    assert sdk.mailboxes.otp.call_args.kwargs["min_length"] == 4


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_device_ownership_existing_never_released(sdk, monkeypatch, keep, owned):
    from mobilerun.clients.cloud import devices

    monkeypatch.setattr(
        devices,
        "capabilities",
        AsyncMock(return_value=NS(capabilities=NS(telephony=True, proxy=True))),
    )
    sdk.numbers.retrieve.return_value = NS(
        data=NS(
            device_binding=None, phone_number=None, state="active", checkout_url=None
        )
    )
    phone = CloudDevice("phone", api_key="fake", billing="paid", keep=keep)
    phone._client = sdk
    phone._owned = True
    phone._cloud_telephony = True
    sdk.devices.terminate = AsyncMock()

    async def exercise():
        number = await PhoneNumber(
            None if owned else "existing", client=sdk, billing="paid"
        )
        mailbox = await Mailbox(
            None if owned else "existing", client=sdk, billing="paid"
        )
        await number.bind(phone, close_on_device_close=owned and not keep)
        await mailbox.bind(phone, close_on_device_close=owned and not keep)
        assert phone.number is number
        assert phone.mailbox is mailbox
        await phone.close()
        await phone.close()

    run(exercise())
    assert sdk.numbers.delete.await_count == (1 if owned and not keep else 0)
    assert sdk.mailboxes.delete.await_count == (1 if owned and not keep else 0)
    assert sdk.devices.terminate.await_count == (0 if keep else 1)
    assert sdk.devices.number.unbind.await_count == 1


def test_thin_adapter_pagination_and_metadata(sdk):
    sdk.numbers.list.side_effect = [page([NS(id="one")], True), page([NS(id="two")])]
    assert [
        item.id
        for item in run(numbers.list_numbers(billing=Billing.INCLUDED, client=sdk))
    ] == [
        "one",
        "two",
    ]
    run(numbers.countries(billing=Billing.INCLUDED, client=sdk))
    run(numbers.capacity("US", billing=Billing.INCLUDED, client=sdk))
    sdk.numbers.capacity.assert_awaited_once_with(country="US")


def test_binding_local_device_rejects_before_requests(sdk):
    from mobilerun.core.devices import AndroidDevice
    from mobilerun.errors import ExtensionNotSupported

    with pytest.raises(ExtensionNotSupported):
        run(PhoneNumber("number", client=sdk).bind(AndroidDevice("local")))
    sdk.numbers.retrieve.assert_not_awaited()


@pytest.mark.parametrize(
    "telephony,bound,error",
    [(False, None, "telephony"), (True, "elsewhere", "already bound")],
)
def test_binding_capability_and_ownership_errors(
    sdk, monkeypatch, telephony, bound, error
):
    from mobilerun.clients.cloud import devices
    from mobilerun.core.devices import CloudDevice
    from mobilerun.errors import DeviceError

    monkeypatch.setattr(
        devices,
        "capabilities",
        AsyncMock(return_value=NS(capabilities=NS(telephony=telephony, proxy=True))),
    )
    sdk.numbers.retrieve.return_value = NS(
        data=NS(
            device_binding=NS(device_id=bound) if bound else None, phone_number=None
        )
    )
    phone = CloudDevice("phone", api_key="fake")
    phone._client = sdk
    phone._cloud_telephony = telephony
    if bound:
        sdk.devices.number.bind.side_effect = DeviceError("already bound")
    with pytest.raises(DeviceError, match=error):
        run(PhoneNumber("number", client=sdk).bind(phone))
    assert sdk.devices.number.bind.await_count == int(telephony)


def test_attachments_are_read_only_and_old_phone_helpers_are_gone():
    from mobilerun.core.devices import AndroidDevice, CloudDevice

    for phone in (AndroidDevice("local"), CloudDevice("phone", api_key="fake")):
        for name in ("number", "mailbox"):
            with pytest.raises(AttributeError):
                setattr(phone, name, object())
            with pytest.raises(DeviceOperationNotSupported):
                getattr(phone, name)
        assert not hasattr(phone, "set_number")
    with pytest.raises(TypeError, match="number.bind"):
        CloudDevice(api_key="fake", number="id")


def test_number_unbind_and_mailbox_detach(sdk, monkeypatch):
    from mobilerun.clients.cloud import devices
    from mobilerun.core.devices import CloudDevice

    monkeypatch.setattr(
        devices,
        "capabilities",
        AsyncMock(return_value=NS(capabilities=NS(telephony=True, proxy=True))),
    )
    sdk.numbers.retrieve.return_value = NS(
        data=NS(device_binding=None, phone_number=None)
    )
    phone = CloudDevice("phone", api_key="fake")
    phone._client = sdk
    phone._cloud_telephony = True

    async def exercise():
        number = PhoneNumber("number", client=sdk)
        mailbox = Mailbox("mailbox", client=sdk)
        await number.bind(phone)
        assert number.device_id == "phone"
        await number.unbind()
        assert number.device_id is None
        with pytest.raises(DeviceOperationNotSupported, match="No sms extension"):
            _ = phone.number
        await mailbox.bind(phone)
        assert phone.mailbox is mailbox
        await mailbox.unbind()
        with pytest.raises(DeviceOperationNotSupported, match="No email extension"):
            _ = phone.mailbox

    run(exercise())
    sdk.devices.number.unbind.assert_awaited_once_with("phone")


def test_existing_phone_fetches_and_caches_number_from_device_api(sdk):
    from mobilerun.core.devices import CloudDevice
    from mobilerun.core.proxy import ProxyStatus

    sdk.devices.number.retrieve.return_value = NS(
        binding=NS(number_id="bound", phone_number="+15555550123")
    )
    phone = CloudDevice("phone", api_key="fake")
    phone._client = sdk
    phone._proxy_status = AsyncMock(return_value=ProxyStatus(False))

    async def exercise():
        await phone
        assert phone.number.id == "bound"
        assert phone.number.device_id == "phone"
        assert phone.number.phone_number == "+15555550123"
        await phone

    run(exercise())
    sdk.devices.number.retrieve.assert_awaited_once_with("phone")


@pytest.mark.parametrize("error", [RuntimeError("ambiguous"), asyncio.CancelledError()])
def test_number_ambiguous_bind_reconciles_remote_binding(error):
    async def run():
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        number = PhoneNumber("number")
        response = NS(binding=NS(number_id="number", phone_number="+123"))
        with (
            patch(
                "mobilerun.core.phone_number.cloud.devices.bind_number",
                AsyncMock(side_effect=error),
            ),
            patch(
                "mobilerun.core.phone_number.cloud.devices.number",
                AsyncMock(return_value=response),
            ),
        ):
            with pytest.raises(type(error)):
                await number.bind(device)
            assert device.number.id == "number"
            assert device.number.device_id == "device"

    asyncio.run(run())


def test_number_failed_readback_marks_binding_uncertain():
    async def run():
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        number = PhoneNumber("number")
        with (
            patch(
                "mobilerun.core.phone_number.cloud.devices.bind_number",
                AsyncMock(side_effect=RuntimeError("failed")),
            ),
            patch(
                "mobilerun.core.phone_number.cloud.devices.number",
                AsyncMock(side_effect=RuntimeError("offline")),
            ),
        ):
            with pytest.raises(RuntimeError):
                await number.bind(device)
            assert device._extension_state.bindings["sms"].uncertain

    asyncio.run(run())


def test_number_unbind_failure_adopts_remote_absence():
    async def run():
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        number = PhoneNumber("number")
        with (
            patch("mobilerun.core.phone_number.cloud.devices.bind_number", AsyncMock()),
            patch(
                "mobilerun.core.phone_number.cloud.devices.unbind_number",
                AsyncMock(side_effect=RuntimeError("lost response")),
            ),
            patch(
                "mobilerun.core.phone_number.cloud.devices.number",
                AsyncMock(return_value=NS(binding=None)),
            ),
        ):
            await number.bind(device)
            with pytest.raises(RuntimeError):
                await number.unbind()
            assert not device.extensions
            assert number.device is None

    asyncio.run(run())


def test_number_force_rollback_failure_reads_back_observed_binding():
    async def run():
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        old = PhoneNumber("old")
        new = PhoneNumber("new")
        bind = AsyncMock(
            side_effect=[
                None,
                RuntimeError("new response lost"),
                RuntimeError("restore failed"),
            ]
        )
        response = NS(binding=NS(number_id="new", phone_number="+123"))
        with (
            patch("mobilerun.core.phone_number.cloud.devices.bind_number", bind),
            patch(
                "mobilerun.core.phone_number.cloud.devices.unbind_number", AsyncMock()
            ),
            patch(
                "mobilerun.core.phone_number.cloud.devices.number",
                AsyncMock(return_value=response),
            ),
        ):
            await old.bind(device)
            with pytest.raises(RuntimeError, match="new response lost"):
                await new.bind(device, force=True)
            assert device.number.id == "new"
            assert old.device is None

    asyncio.run(run())


def test_number_force_successful_rollback_also_reads_back():
    async def run():
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        old = PhoneNumber("old")
        new = PhoneNumber("new")
        bind = AsyncMock(side_effect=[None, RuntimeError("new response lost"), None])
        readback = AsyncMock(
            return_value=NS(binding=NS(number_id="old", phone_number="+123"))
        )
        with (
            patch("mobilerun.core.phone_number.cloud.devices.bind_number", bind),
            patch(
                "mobilerun.core.phone_number.cloud.devices.unbind_number", AsyncMock()
            ),
            patch("mobilerun.core.phone_number.cloud.devices.number", readback),
        ):
            await old.bind(device)
            with pytest.raises(RuntimeError):
                await new.bind(device, force=True)
            readback.assert_awaited_once()

    asyncio.run(run())


def test_number_extension_contract():
    async def device():
        return MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )

    async def incompatible():
        return MemoryDevice()

    with (
        patch("mobilerun.core.phone_number.cloud.devices.bind_number", AsyncMock()),
        patch("mobilerun.core.phone_number.cloud.devices.unbind_number", AsyncMock()),
    ):
        asyncio.run(
            check_extension_contract(
                lambda name: PhoneNumber("number", name=name),
                device,
                incompatible_device=incompatible,
            )
        )


def test_number_bind_and_unbind_call_backend():
    async def run():
        client = object()
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        number = PhoneNumber("number", client=client)
        with (
            patch(
                "mobilerun.core.phone_number.cloud.devices.bind_number", AsyncMock()
            ) as bind,
            patch(
                "mobilerun.core.phone_number.cloud.devices.unbind_number", AsyncMock()
            ) as unbind,
        ):
            await number.bind(device)
            await number.unbind()
            assert bind.await_args.args == ("device", "number")
            assert unbind.await_args.args == ("device",)
            assert bind.await_args.kwargs["client"] is client
            assert unbind.await_args.kwargs["client"] is client

    asyncio.run(run())


def test_connect_adopts_number_without_mutation():
    async def run():
        phone = CloudDevice("device", api_key="test")
        response = NS(binding=NS(number_id="number", phone_number="+123"))
        capabilities = NS(capabilities=NS(telephony=True, proxy=True))
        with (
            patch(
                "mobilerun.core.devices.components.cloud_lifecycle.devices.number",
                AsyncMock(return_value=response),
            ),
            patch(
                "mobilerun.core.devices.components.cloud_lifecycle.devices.capabilities",
                AsyncMock(return_value=capabilities),
            ),
            patch(
                "mobilerun.core.phone_number.cloud.devices.bind_number", AsyncMock()
            ) as bind,
            patch(
                "mobilerun.core.phone_number.cloud.devices.unbind_number", AsyncMock()
            ) as unbind,
            patch("mobilerun.core.proxy.adoption._adopt_remote_proxy", AsyncMock()),
        ):
            await phone.connect()
            assert phone.number.id == "number"
            assert phone.number.phone_number == "+123"
            assert "cloud_telephony" in phone.supported_operations
            bind.assert_not_awaited()
            number = phone.number
            await phone.close()
            assert not phone.extensions
            assert number.device is None
            assert number.device_id == "device"
            unbind.assert_not_awaited()

    asyncio.run(run())


def test_sms_code_and_typed_send():

    async def run():
        now = datetime.now(timezone.utc)
        item = {
            "id": "one",
            "createdAt": now,
            "occurredAt": now,
            "direction": "inbound",
            "status": "received",
            "detectedSender": "Example",
            "body": "Code 012345",
        }
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        number = PhoneNumber("number")
        with (
            patch("mobilerun.core.phone_number.cloud.devices.bind_number", AsyncMock()),
            patch(
                "mobilerun.core.phone_number.cloud.numbers.messages",
                AsyncMock(return_value=[numbers.MessageItem.model_validate(item)]),
            ),
            patch(
                "mobilerun.core.phone_number.cloud.numbers.send",
                AsyncMock(
                    return_value=numbers.MessageSendResponse.model_validate(
                        {"data": item}
                    )
                ),
            ),
        ):
            await number.bind(device)
            assert await number.wait_for_code(sender="example") == "012345"
            assert (await number.send("to", "body")).data.id == "one"

    asyncio.run(run())


def sdk_conflict(body):
    response = httpx.Response(
        409, request=httpx.Request("DELETE", "https://example.test")
    )
    return APIStatusError("server secret", response=response, body=body)


def test_not_cancellable_number_release_has_specific_safe_message():
    error = cloud_error(
        sdk_conflict({"code": "not_cancellable", "message": "server secret"}),
        "Release number",
    )
    assert isinstance(error, CloudConflict)
    assert str(error) == (
        "Release number failed (HTTP 409). This number can't be released: "
        "only unpaid pending purchases and active rented numbers can be cancelled."
    )
    assert is_value_safe(error)


@pytest.mark.parametrize("body", [None, "invalid json", [], {}, {"code": "other"}])
def test_other_conflicts_keep_generic_message(body):
    error = cloud_error(sdk_conflict(body), "Release number")
    assert type(error) is CloudConflict
    assert str(error) == (
        "Release number failed (HTTP 409). "
        "Wait for the current operation to finish and retry."
    )


def test_not_cancellable_code_for_other_operation_stays_generic():
    error = cloud_error(sdk_conflict({"code": "not_cancellable"}), "Release mailbox")
    assert type(error) is CloudConflict


def test_number_create_without_country_reaches_api():
    endpoint = NS(
        create=AsyncMock(
            return_value=NS(data=NS(number_id="n1", state="active", checkout_url=None))
        )
    )
    asyncio.run(numbers.create(billing=Billing.PAID, client=NS(numbers=endpoint)))
    endpoint.create.assert_awaited_once_with(billing_preference="included")


def test_known_validation_code_is_safe_and_actionable():

    exc = NS(
        status_code=422,
        response=None,
        request_id=None,
        body={"code": "unsupported_country", "message": "SECRET"},
    )
    error = cloud_error(exc, "Create number")
    assert "mobilerun number countries" in str(error)
    assert "SECRET" not in str(error)


def test_unknown_validation_code_does_not_leak():

    exc = NS(
        status_code=422,
        response=None,
        request_id=None,
        body={"code": "SECRET", "message": "SECRET"},
    )
    assert "SECRET" not in str(cloud_error(exc, "Create number"))


def test_known_device_validation_code_is_safe():

    exc = NS(
        status_code=422,
        response=None,
        request_id=None,
        body={"code": "DEVICE_TYPE_UNAVAILABLE", "message": "SECRET"},
    )
    result = str(cloud_error(exc, "Create cloud phone"))
    assert "cloud phone type is unavailable" in result
    assert "SECRET" not in result


def test_noninteractive_create_forwards_missing_country(monkeypatch):
    monkeypatch.setattr(group, "is_interactive", lambda: False)
    monkeypatch.setattr(group, "confirm_action_async", AsyncMock())
    created = []

    async def phone(**kwargs):
        created.append(kwargs)
        return NS(id="n1")

    monkeypatch.setattr(group, "PhoneNumber", phone)
    monkeypatch.setattr(
        group.cloud_numbers,
        "retrieve",
        AsyncMock(return_value=NS(data=NS(state="active", phone_number="+1"))),
    )
    result = CliRunner().invoke(group.number, ["create", "--billing", "paid", "--yes"])
    assert result.exit_code == 0, result.output
    assert created[0]["country"] is None


def test_number_create_prompts_for_supported_country(monkeypatch):
    @asynccontextmanager
    async def client(**kwargs):
        yield object()

    monkeypatch.setattr(group, "cloud_client", client)
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(
        group.cloud_numbers,
        "countries",
        AsyncMock(
            return_value=NS(
                data=NS(items=[NS(country="US", name="United States", in_stock=True)])
            )
        ),
    )
    monkeypatch.setattr(group, "select_prompt", AsyncMock(return_value="US"))
    monkeypatch.setattr(group, "confirm_action_async", AsyncMock())
    created = []

    async def phone(**kwargs):
        created.append(kwargs)
        return NS(id="n1")

    phone.included_available = AsyncMock(return_value=1)
    monkeypatch.setattr(group, "PhoneNumber", phone)
    monkeypatch.setattr(
        group.cloud_numbers,
        "retrieve",
        AsyncMock(return_value=NS(data=NS(state="active", phone_number="+1"))),
    )
    result = CliRunner().invoke(group.number, ["create", "--billing", "paid", "--yes"])
    assert result.exit_code == 0, result.output
    assert created[0]["country"] == "US"
    assert created[0]["checkout"] == "browser"


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_context_ownership_and_idempotent_close(resource, existing, keep):
    adapter = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(
                    number_id="new",
                    id="new",
                    address="demo@example.test",
                    state="provisioning",
                    status="active",
                    checkout_url=None,
                )
            )
        ),
        retrieve=AsyncMock(
            return_value=NS(
                data=NS(
                    device_binding=None,
                    phone_number=None,
                    address="demo@example.test",
                    state="active",
                    status="active",
                    checkout_url=None,
                )
            )
        ),
        delete=AsyncMock(),
    )

    async def exercise():
        instance = resource(
            "existing" if existing else None,
            client=NS(
                **{"numbers" if resource is PhoneNumber else "mailboxes": adapter}
            ),
            billing="paid",
            keep=keep,
        )
        async with instance as ready:
            assert ready is instance
            assert instance.id == ("existing" if existing else "new")
        await instance.close()

    asyncio.run(exercise())
    assert adapter.create.await_count == (0 if existing else 1)
    assert adapter.delete.await_count == (1 if not existing and not keep else 0)


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
def test_unawaited_creation_cannot_be_used(resource):
    instance = resource()
    with pytest.raises(RuntimeError, match="Await"):
        _ = instance.id
    with pytest.raises(RuntimeError, match="Await"):
        asyncio.run(instance.release())
    assert not hasattr(resource, "create")


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
def test_creation_failure_does_not_claim_ownership(resource):
    adapter = NS(
        create=AsyncMock(side_effect=ValueError("blocked")), delete=AsyncMock()
    )

    async def exercise():
        instance = resource(
            client=NS(
                **{"numbers" if resource is PhoneNumber else "mailboxes": adapter}
            ),
            billing="paid",
        )
        with pytest.raises(ValueError, match="blocked"):
            await instance
        await instance.close()

    asyncio.run(exercise())
    adapter.delete.assert_not_awaited()
