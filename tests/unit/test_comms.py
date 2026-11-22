"""Communications policy, ownership and SDK contracts without external calls."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import numbers
from mobilerun.config.settings import get_settings
from mobilerun.core.mailbox import CloudMailbox as Mailbox
from mobilerun.core.phone_number import CloudPhoneNumber as PhoneNumber
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.errors.cloud import CloudAPIError
from mobilerun.models.billing import Billing
from mobilerun.utils.codes import extract_code
from tests.support.devices import MemoryDevice


def run(coro):
    async def execute():
        return await coro

    return asyncio.run(execute())


def page(items, has_next=False):
    return NS(items=items, pagination=NS(has_next=has_next))


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
                return_value=NS(capabilities=NS(telephony=True))
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
    from mobilerun.core.devices.cloud import CloudDevice

    monkeypatch.setattr(
        devices,
        "capabilities",
        AsyncMock(return_value=NS(capabilities=NS(telephony=True))),
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
        AsyncMock(return_value=NS(capabilities=NS(telephony=telephony))),
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
        AsyncMock(return_value=NS(capabilities=NS(telephony=True))),
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
