import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.phone_number import CloudPhoneNumber
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


@pytest.mark.parametrize("error", [RuntimeError("ambiguous"), asyncio.CancelledError()])
def test_number_ambiguous_bind_reconciles_remote_binding(error):
    async def run():
        device = MemoryDevice(
            cloud_id="device", supported_operations=frozenset({"cloud_telephony"})
        )
        number = CloudPhoneNumber("number")
        response = SimpleNamespace(
            binding=SimpleNamespace(number_id="number", phone_number="+123")
        )
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
        number = CloudPhoneNumber("number")
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
        number = CloudPhoneNumber("number")
        with (
            patch("mobilerun.core.phone_number.cloud.devices.bind_number", AsyncMock()),
            patch(
                "mobilerun.core.phone_number.cloud.devices.unbind_number",
                AsyncMock(side_effect=RuntimeError("lost response")),
            ),
            patch(
                "mobilerun.core.phone_number.cloud.devices.number",
                AsyncMock(return_value=SimpleNamespace(binding=None)),
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
        old = CloudPhoneNumber("old")
        new = CloudPhoneNumber("new")
        bind = AsyncMock(
            side_effect=[
                None,
                RuntimeError("new response lost"),
                RuntimeError("restore failed"),
            ]
        )
        response = SimpleNamespace(
            binding=SimpleNamespace(number_id="new", phone_number="+123")
        )
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
        old = CloudPhoneNumber("old")
        new = CloudPhoneNumber("new")
        bind = AsyncMock(side_effect=[None, RuntimeError("new response lost"), None])
        readback = AsyncMock(
            return_value=SimpleNamespace(
                binding=SimpleNamespace(number_id="old", phone_number="+123")
            )
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
                lambda name: CloudPhoneNumber("number", name=name),
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
        number = CloudPhoneNumber("number", client=client)
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
        response = SimpleNamespace(
            binding=SimpleNamespace(number_id="number", phone_number="+123")
        )
        capabilities = SimpleNamespace(
            capabilities=SimpleNamespace(telephony=True, proxy=True)
        )
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
    from datetime import datetime, timezone

    from mobilerun.clients.cloud import numbers

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
        number = CloudPhoneNumber("number")
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
