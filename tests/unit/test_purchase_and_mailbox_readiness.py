import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import numbers
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.models.billing import Billing


def test_conflicting_number_purchase_reports_pending_checkout():
    from mobilerun.errors import PurchaseInProgress

    async def run():
        request = httpx.Request("POST", "https://example.test/numbers")
        response = httpx.Response(409, request=request)
        error = APIStatusError(
            "conflict", response=response, body={"code": "purchase_in_progress"}
        )
        pending = NS(
            id="pending-id",
            state="awaiting_payment",
            checkout_url="https://checkout.test/pay",
            created_at=None,
        )
        sdk = NS(
            numbers=NS(
                create=AsyncMock(side_effect=error),
                list=AsyncMock(
                    return_value=NS(items=[pending], pagination=NS(has_next=False))
                ),
            )
        )
        with pytest.raises(PurchaseInProgress) as caught:
            await numbers.create(billing=Billing.PAID, client=sdk, country="US")
        assert caught.value.resource_id == "pending-id"
        assert caught.value.checkout.url == "https://checkout.test/pay"

    asyncio.run(run())


def test_new_mailbox_waits_for_active_address(monkeypatch):
    async def run():
        from mobilerun.core.mailbox import cloud

        monkeypatch.setattr(
            cloud.mailboxes,
            "create",
            AsyncMock(
                return_value=NS(
                    data=NS(
                        id="mailbox",
                        status="provisioning",
                        address=None,
                        checkout_url=None,
                    )
                )
            ),
        )
        monkeypatch.setattr(
            cloud.mailboxes,
            "retrieve",
            AsyncMock(
                return_value=NS(
                    data=NS(
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
    from mobilerun.core.phone_number import CloudPhoneNumber, cloud
    from mobilerun.errors import PurchaseInProgress
    from mobilerun.models.checkout import Checkout

    async def run():
        checkout = Checkout(
            "https://checkout.test/pay", None, "phone number", "pending-id"
        )
        monkeypatch.setattr(
            cloud.numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending-id", checkout)),
        )
        data = NS(
            state="active",
            checkout_url=None,
            phone_number="+15550000000",
            device_binding=None,
        )
        monkeypatch.setattr(
            cloud.numbers, "retrieve", AsyncMock(return_value=NS(data=data))
        )
        delete = AsyncMock()
        monkeypatch.setattr(cloud.numbers, "delete", delete)
        callback = AsyncMock()
        number = await CloudPhoneNumber(billing="paid", on_checkout=callback)
        assert number.id == "pending-id"
        assert number.resumed_existing
        callback.assert_awaited_once_with(checkout)
        await number.close()
        delete.assert_not_awaited()

    asyncio.run(run())


def test_resumed_number_timeout_does_not_delete(monkeypatch):
    from mobilerun.core.phone_number import CloudPhoneNumber, cloud
    from mobilerun.errors import PurchaseInProgress
    from mobilerun.models.checkout import Checkout

    async def run():
        checkout = Checkout(
            "https://checkout.test/pay", None, "phone number", "pending-id"
        )
        monkeypatch.setattr(
            cloud.numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending-id", checkout)),
        )
        delete = AsyncMock()
        monkeypatch.setattr(cloud.numbers, "delete", delete)

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
        from mobilerun.core.mailbox import cloud

        data = NS(id="mailbox", status="provisioning", address=None, checkout_url=None)
        monkeypatch.setattr(
            cloud.mailboxes, "create", AsyncMock(return_value=NS(data=data))
        )
        monkeypatch.setattr(
            cloud.mailboxes, "retrieve", AsyncMock(return_value=NS(data=data))
        )
        delete = AsyncMock()
        monkeypatch.setattr(cloud.mailboxes, "delete", delete)
        with pytest.raises(TimeoutError, match="mailbox"):
            await CloudMailbox(billing="paid", ready_timeout=0.001)
        delete.assert_not_awaited()

    asyncio.run(run())


def test_sdk_shaped_mailbox_waits_then_reads_otp():
    from mobilerun_sdk.types.mailbox_create_response import MailboxCreateResponse
    from mobilerun_sdk.types.mailbox_otp_response import MailboxOtpResponse
    from mobilerun_sdk.types.mailbox_retrieve_response import MailboxRetrieveResponse

    from tests.support.devices import MemoryDevice

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
        sdk = NS(
            mailboxes=NS(
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
    from mobilerun.core.mailbox import cloud

    async def run():
        data = NS(id="mailbox", status=status, address=None, checkout_url=None)
        monkeypatch.setattr(
            cloud.mailboxes, "create", AsyncMock(return_value=NS(data=data))
        )
        monkeypatch.setattr(
            cloud.mailboxes, "retrieve", AsyncMock(return_value=NS(data=data))
        )
        with pytest.raises(RuntimeError, match=status):
            await CloudMailbox(billing="paid", ready_timeout=1)

    asyncio.run(run())


def test_resuming_without_checkout_raises_typed_conflict(monkeypatch):
    from mobilerun.core.phone_number import CloudPhoneNumber, cloud
    from mobilerun.errors import PurchaseInProgress

    async def run():
        monkeypatch.setattr(
            cloud.numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending", None)),
        )
        with pytest.raises(PurchaseInProgress):
            await CloudPhoneNumber(billing="paid", checkout="print")

    asyncio.run(run())


@pytest.mark.parametrize("ambiguous", [False, True])
def test_pending_lookup_paginates_and_rejects_ambiguity(ambiguous):
    from mobilerun.errors import CloudConflict, PurchaseInProgress

    async def run():
        response = httpx.Response(
            409, request=httpx.Request("POST", "https://example.test")
        )
        error = APIStatusError(
            "conflict", response=response, body={"code": "purchase_in_progress"}
        )
        pending = NS(
            id="pending", state="awaiting_payment", checkout_url="https://checkout.test"
        )
        first = [pending] if ambiguous else []
        sdk = NS(
            numbers=NS(
                create=AsyncMock(side_effect=error),
                list=AsyncMock(
                    side_effect=[
                        NS(items=first, pagination=NS(has_next=True)),
                        NS(items=[pending], pagination=NS(has_next=False)),
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
    from mobilerun.core.phone_number import CloudPhoneNumber, cloud
    from mobilerun.errors import PurchaseInProgress
    from mobilerun.models.checkout import Checkout

    async def run():
        checkout = Checkout("https://checkout.test", None, "phone number", "pending")
        monkeypatch.setattr(
            cloud.numbers,
            "create",
            AsyncMock(side_effect=PurchaseInProgress("pending", checkout)),
        )
        delete = AsyncMock()
        monkeypatch.setattr(cloud.numbers, "delete", delete)
        callback = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await CloudPhoneNumber(billing="paid", on_checkout=callback)
        delete.assert_not_awaited()

    asyncio.run(run())


def test_payment_failure_does_not_disclose_server_text():
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        PaymentSnapshot,
        wait_until_active,
    )
    from mobilerun.errors import CloudError

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
