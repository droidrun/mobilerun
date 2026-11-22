import asyncio
import inspect
import logging
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import devices, numbers
from mobilerun.core.devices import CloudDevice
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.errors import CloudError, CloudPermissionDenied

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
