"""Mailbox history, extension behavior, and readiness of purchased resources."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from click.testing import CliRunner
from mobilerun_sdk import APIStatusError
from mobilerun_sdk.types.mailboxes.message_list_response import (
    Item,
    MessageListResponse,
)

from mobilerun.cli.app import cli
from mobilerun.clients.cloud import mailboxes, numbers
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.errors import PaymentRequired, PurchaseInProgress
from mobilerun.models.billing import Billing
from mobilerun.models.checkout import Checkout
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


def message(id="msg-1"):
    return Item(
        id=id,
        mailboxId="box-1",
        fromAddress="a@example.com",
        hasOtp=True,
        receivedAt="2025-01-01T00:00:00Z",
        subject="Your code",
        preview="Hello",
    )


def test_mailbox_messages_paginates_and_uses_existing_client():
    listing = AsyncMock(
        side_effect=[
            MessageListResponse(items=[message()], nextCursor="next"),
            MessageListResponse(items=[message("msg-2")]),
        ]
    )
    client = SimpleNamespace(
        mailboxes=SimpleNamespace(messages=SimpleNamespace(list=listing))
    )
    result = asyncio.run(
        mailboxes.messages("box-1", billing=Billing.NO_COMMERCE, client=client)
    )
    assert [item.id for item in result] == ["msg-1", "msg-2"]
    assert listing.await_args_list[1].kwargs["cursor"] == "next"


@pytest.mark.parametrize("populated", [True, False], ids=["populated", "empty"])
def test_mailbox_messages_cli_json_and_text(monkeypatch, populated):
    monkeypatch.setattr(
        mailboxes, "messages", AsyncMock(return_value=[message()] if populated else [])
    )
    runner = CliRunner()
    args = ["mailbox", "messages", "box-1"]
    result = runner.invoke(cli, [*args, "--json"])
    assert result.exit_code == 0, result.output
    entries = json.loads(result.stdout)
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    if populated:
        (entry,) = entries
        assert entry["from_address"] == "a@example.com"
        assert entry["has_otp"] is True
        assert entry["preview"] == "Hello"
        assert "a@example.com" in result.output
        assert "Your code" in result.output
        assert "Hello" in result.output
    else:
        assert entries == []
        assert "messages" in result.output.lower()


def test_mailbox_extension_contract():
    async def device():
        return MemoryDevice()

    asyncio.run(
        check_extension_contract(lambda name: CloudMailbox("id", name=name), device)
    )


def test_mailbox_code_and_context():
    async def run():
        mailbox = CloudMailbox("id")
        mailbox._address = "test@example.com"
        await mailbox.bind(MemoryDevice())
        with patch(
            "mobilerun.core.mailbox.cloud.mailboxes.otp",
            AsyncMock(
                return_value=SimpleNamespace(data=SimpleNamespace(code="123456"))
            ),
        ):
            assert await mailbox.wait_for_code() == "123456"
        assert "test@example.com" in repr(await mailbox.context())

    asyncio.run(run())


def _data(status, address=None):
    return SimpleNamespace(
        data=SimpleNamespace(
            status=status, address=address, checkout_url=None, checkout_expires_at=None
        )
    )


async def _payment_error(monkeypatch):
    checkout = Checkout("https://checkout.test", None, "mailbox", "mailbox-id")
    monkeypatch.setattr(
        mailboxes,
        "create",
        AsyncMock(side_effect=PaymentRequired(checkout, AsyncMock())),
    )
    with pytest.raises(PaymentRequired) as caught:
        await CloudMailbox(billing="paid")
    return caught.value


def test_resume_waits_for_active_address(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        retrieve = AsyncMock(
            side_effect=[_data("active"), _data("active", "ready@example.test")]
        )
        monkeypatch.setattr(mailboxes, "retrieve", retrieve)
        result = await error.wait_until_paid(timeout=2)
        assert result.address == "ready@example.test"
        assert retrieve.await_count == 2

    asyncio.run(run())


def test_resume_timeout_keeps_mailbox(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        monkeypatch.setattr(
            mailboxes, "retrieve", AsyncMock(return_value=_data("active"))
        )
        delete = AsyncMock()
        monkeypatch.setattr(mailboxes, "delete", delete)
        with pytest.raises(TimeoutError, match="mailbox-id"):
            await error.wait_until_paid(timeout=0.01)
        delete.assert_not_awaited()

    asyncio.run(run())


def test_resume_terminal_state_is_clear(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        monkeypatch.setattr(
            mailboxes, "retrieve", AsyncMock(return_value=_data("archived"))
        )
        with pytest.raises(RuntimeError, match="archived"):
            await error.wait_until_paid(timeout=1)

    asyncio.run(run())


def test_resume_payment_and_address_share_one_timeout(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        retrieve = AsyncMock(side_effect=[_data("awaiting_payment"), _data("active")])
        monkeypatch.setattr(mailboxes, "retrieve", retrieve)
        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError, match="mailbox-id"):
            await error.wait_until_paid(timeout=1.1)
        assert retrieve.await_count == 2
        assert asyncio.get_running_loop().time() - started < 1.8

    asyncio.run(run())


def test_conflicting_number_purchase_reports_pending_checkout():
    async def run():
        request = httpx.Request("POST", "https://example.test/numbers")
        response = httpx.Response(409, request=request)
        error = APIStatusError(
            "conflict", response=response, body={"code": "purchase_in_progress"}
        )
        pending = SimpleNamespace(
            id="pending-id",
            country_code="US",
            label="work",
            state="awaiting_payment",
            checkout_url="https://checkout.test/pay",
            created_at=None,
        )
        sdk = SimpleNamespace(
            numbers=SimpleNamespace(
                create=AsyncMock(side_effect=error),
                list=AsyncMock(
                    return_value=SimpleNamespace(
                        items=[pending], pagination=SimpleNamespace(has_next=False)
                    )
                ),
            )
        )
        with pytest.raises(PurchaseInProgress) as caught:
            await numbers.create(
                billing=Billing.PAID, client=sdk, country="US", label="work"
            )
        assert caught.value.resource_id == "pending-id"
        assert caught.value.checkout.url == "https://checkout.test/pay"

    asyncio.run(run())


def test_new_mailbox_waits_for_active_address(monkeypatch):
    async def run():
        monkeypatch.setattr(
            mailboxes,
            "create",
            AsyncMock(
                return_value=SimpleNamespace(
                    data=SimpleNamespace(
                        id="mailbox",
                        status="provisioning",
                        address=None,
                        checkout_url=None,
                    )
                )
            ),
        )
        monkeypatch.setattr(
            mailboxes,
            "retrieve",
            AsyncMock(
                return_value=SimpleNamespace(
                    data=SimpleNamespace(
                        status="active",
                        address="test@example.com",
                        checkout_url=None,
                        checkout_expires_at=None,
                    )
                )
            ),
        )
        box = await CloudMailbox(billing="paid")
        assert box.address == "test@example.com"

    asyncio.run(run())


def test_number_resumes_existing_purchase_without_deleting(monkeypatch):
    async def run():
        checkout = Checkout(
            "https://checkout.test/pay", None, "phone number", "pending-id"
        )
        monkeypatch.setattr(
            numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending-id", checkout)),
        )
        data = SimpleNamespace(
            state="active",
            checkout_url=None,
            phone_number="+15550000000",
            device_binding=None,
        )
        monkeypatch.setattr(
            numbers, "retrieve", AsyncMock(return_value=SimpleNamespace(data=data))
        )
        delete = AsyncMock()
        monkeypatch.setattr(numbers, "delete", delete)
        callback = AsyncMock()
        number = await CloudPhoneNumber(billing="paid", on_checkout=callback)
        assert number.id == "pending-id"
        assert number.resumed_existing
        callback.assert_awaited_once_with(checkout)
        await number.close()
        delete.assert_not_awaited()

    asyncio.run(run())


def test_resumed_number_timeout_does_not_delete(monkeypatch):
    async def run():
        checkout = Checkout(
            "https://checkout.test/pay", None, "phone number", "pending-id"
        )
        monkeypatch.setattr(
            numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending-id", checkout)),
        )
        delete = AsyncMock()
        monkeypatch.setattr(numbers, "delete", delete)

        async def slow(_checkout):
            await asyncio.sleep(1)

        with pytest.raises(TimeoutError):
            await CloudPhoneNumber(
                billing="paid", on_checkout=slow, payment_timeout=0.001
            )
        delete.assert_not_awaited()

    asyncio.run(run())


def test_mailbox_readiness_timeout_preserves_resource(monkeypatch):
    async def run():
        data = SimpleNamespace(
            id="mailbox", status="provisioning", address=None, checkout_url=None
        )
        monkeypatch.setattr(
            mailboxes, "create", AsyncMock(return_value=SimpleNamespace(data=data))
        )
        monkeypatch.setattr(
            mailboxes, "retrieve", AsyncMock(return_value=SimpleNamespace(data=data))
        )
        delete = AsyncMock()
        monkeypatch.setattr(mailboxes, "delete", delete)
        with pytest.raises(TimeoutError, match="mailbox"):
            await CloudMailbox(billing="paid", ready_timeout=0.001)
        delete.assert_not_awaited()

    asyncio.run(run())


def test_sdk_shaped_mailbox_waits_then_reads_otp():
    from mobilerun_sdk.types.mailbox_create_response import MailboxCreateResponse
    from mobilerun_sdk.types.mailbox_otp_response import MailboxOtpResponse
    from mobilerun_sdk.types.mailbox_retrieve_response import MailboxRetrieveResponse

    async def run():
        data = {
            "id": "box",
            "address": None,
            "billingMode": "rent",
            "cancelAtPeriodEnd": False,
            "createdAt": "2026-01-01T00:00:00Z",
            "inboundMessages": {"exhausted": False, "included": 10, "used": 0},
            "provider": "matix",
            "status": "provisioning",
        }
        created = MailboxCreateResponse.model_validate({"data": data})
        ready = MailboxRetrieveResponse.model_validate(
            {"data": {**data, "status": "active", "address": "test@example.com"}}
        )
        otp = MailboxOtpResponse.model_validate(
            {
                "data": {
                    "code": "012345",
                    "confidence": "high",
                    "messageId": "msg",
                    "receivedAt": "2026-01-01T00:00:01Z",
                }
            }
        )
        sdk = SimpleNamespace(
            mailboxes=SimpleNamespace(
                create=AsyncMock(return_value=created),
                retrieve=AsyncMock(return_value=ready),
                otp=AsyncMock(return_value=otp),
            )
        )
        mailbox = await CloudMailbox(client=sdk, billing="paid", ready_timeout=1)
        await mailbox.bind(MemoryDevice())
        assert await mailbox.wait_for_code(timeout=1) == "012345"

    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "archived"])
def test_terminal_mailbox_does_not_wait(monkeypatch, status):
    async def run():
        data = SimpleNamespace(
            id="mailbox", status=status, address=None, checkout_url=None
        )
        monkeypatch.setattr(
            mailboxes, "create", AsyncMock(return_value=SimpleNamespace(data=data))
        )
        monkeypatch.setattr(
            mailboxes, "retrieve", AsyncMock(return_value=SimpleNamespace(data=data))
        )
        with pytest.raises(RuntimeError, match=status):
            await CloudMailbox(billing="paid", ready_timeout=1)

    asyncio.run(run())


def test_resuming_without_checkout_raises_typed_conflict(monkeypatch):
    async def run():
        monkeypatch.setattr(
            numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending", None)),
        )
        with pytest.raises(PurchaseInProgress):
            await CloudPhoneNumber(billing="paid", checkout="print")

    asyncio.run(run())


@pytest.mark.parametrize("ambiguous", [False, True])
def test_pending_lookup_paginates_and_rejects_ambiguity(ambiguous):
    from mobilerun.errors import CloudConflict

    async def run():
        response = httpx.Response(
            409, request=httpx.Request("POST", "https://example.test")
        )
        error = APIStatusError(
            "conflict", response=response, body={"code": "purchase_in_progress"}
        )
        pending = SimpleNamespace(
            id="pending",
            state="awaiting_payment",
            label=None,
            checkout_url="https://checkout.test",
        )
        first = [pending] if ambiguous else []
        sdk = SimpleNamespace(
            numbers=SimpleNamespace(
                create=AsyncMock(side_effect=error),
                list=AsyncMock(
                    side_effect=[
                        SimpleNamespace(
                            items=first, pagination=SimpleNamespace(has_next=True)
                        ),
                        SimpleNamespace(
                            items=[pending], pagination=SimpleNamespace(has_next=False)
                        ),
                    ]
                ),
            )
        )
        with pytest.raises(
            CloudConflict if ambiguous else PurchaseInProgress
        ) as caught:
            await numbers.create(billing=Billing.PAID, client=sdk)
        if ambiguous:
            assert not isinstance(caught.value, PurchaseInProgress)
        else:
            assert caught.value.resource_id == "pending"
        assert sdk.numbers.list.await_count == 2

    asyncio.run(run())


def test_cancelled_resume_does_not_delete(monkeypatch):
    async def run():
        checkout = Checkout("https://checkout.test", None, "phone number", "pending")
        monkeypatch.setattr(
            numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending", checkout)),
        )
        delete = AsyncMock()
        monkeypatch.setattr(numbers, "delete", delete)
        callback = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await CloudPhoneNumber(billing="paid", on_checkout=callback)
        delete.assert_not_awaited()

    asyncio.run(run())


def test_payment_failure_does_not_disclose_server_text():
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        wait_until_active,
    )
    from mobilerun.errors import CloudError
    from mobilerun.models.checkout import PaymentSnapshot

    async def run():
        adapter = PaymentAdapter(
            "phone",
            "id",
            AsyncMock(return_value=PaymentSnapshot("server-secret-canary")),
            AsyncMock(),
        )
        with pytest.raises(CloudError) as caught:
            await wait_until_active(adapter)
        assert "server-secret-canary" not in str(caught.value)
        assert caught.value.status == "server-secret-canary"
        assert caught.value.value_safe

    asyncio.run(run())


def test_cloud_error_allows_support_request_id_but_not_response_body():
    from mobilerun.clients.cloud.errors import cloud_error

    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://example.test"),
        headers={"x-request-id": "support-request-id"},
    )
    sdk_error = APIStatusError(
        "server-secret-canary",
        response=response,
        body={"message": "server-secret-canary"},
    )
    error = cloud_error(sdk_error, "Create number")
    assert "support-request-id" in str(error)
    assert "server-secret-canary" not in str(error)
    assert error.value_safe


def test_cloud_error_action_passthrough_does_not_expose_response_body():
    from mobilerun.clients.cloud.errors import cloud_error
    from mobilerun.core.extensions import Extension, action
    from mobilerun.errors import CloudValidationError

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Report a sanitized cloud error."""
            response = httpx.Response(
                400, request=httpx.Request("POST", "https://example.test")
            )
            raise cloud_error(
                APIStatusError(
                    "private-canary",
                    response=response,
                    body={"detail": "private-canary"},
                ),
                "Create number",
            )

    with pytest.raises(CloudValidationError) as caught:
        asyncio.run(Resource().run())
    assert "private-canary" not in str(caught.value)
