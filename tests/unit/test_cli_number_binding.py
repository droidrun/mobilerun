import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.cli.phone_number import group
from mobilerun.models.billing import Billing


@pytest.fixture
def cloud(monkeypatch):
    from mobilerun.core.devices.cloud import CloudDevice

    def no_device(*args, **kwargs):
        pytest.fail("binding commands must not construct or close a CloudDevice")

    monkeypatch.setattr(CloudDevice, "__init__", no_device)
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(group, "cloud_client", lambda **kwargs: client, raising=False)
    from mobilerun.clients.cloud import devices, numbers

    capabilities = AsyncMock(return_value=NS(capabilities=NS(telephony=True)))
    bind = AsyncMock()
    current = AsyncMock(return_value=NS(binding=None))
    unbind = AsyncMock()
    terminate = AsyncMock()
    retrieve = AsyncMock(return_value=NS(data=NS(device_binding=NS(device_id="dev-1"))))
    monkeypatch.setattr(devices, "capabilities", capabilities)
    monkeypatch.setattr(devices, "bind_number", bind)
    monkeypatch.setattr(devices, "number", current)
    monkeypatch.setattr(devices, "unbind_number", unbind)
    monkeypatch.setattr(devices, "terminate", terminate)
    monkeypatch.setattr(numbers, "retrieve", retrieve)
    return NS(
        client=client,
        capabilities=capabilities,
        bind=bind,
        current=current,
        unbind=unbind,
        retrieve=retrieve,
        terminate=terminate,
    )


def test_bind_passes_ids_billing_and_outputs_json(cloud):
    result = CliRunner().invoke(
        cli, ["number", "bind", "num-1", "dev-1", "--billing", "included", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "id": "num-1",
        "status": "bound",
        "number_id": "num-1",
        "device_id": "dev-1",
    }
    cloud.capabilities.assert_awaited_once_with(
        "dev-1", billing=Billing.INCLUDED, client=cloud.client
    )
    cloud.bind.assert_awaited_once_with(
        "dev-1", "num-1", billing=Billing.INCLUDED, client=cloud.client
    )
    cloud.unbind.assert_not_awaited()
    cloud.terminate.assert_not_awaited()


def test_unsupported_device_fails_cleanly(cloud):
    cloud.capabilities.return_value.capabilities.telephony = False
    result = CliRunner().invoke(cli, ["number", "bind", "num-1", "dev-1"])
    assert result.exit_code == 1
    assert "does not support cloud telephony" in result.output
    assert "Traceback" not in result.output
    cloud.bind.assert_not_awaited()


def test_bind_rejects_existing_device_binding(cloud):
    cloud.current.return_value.binding = NS(number_id="other")
    result = CliRunner().invoke(cli, ["number", "bind", "num-1", "dev-1"])
    assert result.exit_code == 1
    assert "already has" in result.output
    cloud.bind.assert_not_awaited()


def test_unbind_resolves_number_and_outputs_json(cloud):
    result = CliRunner().invoke(
        cli, ["number", "unbind", "num-1", "--billing", "no_commerce", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "id": "num-1",
        "status": "unbound",
        "number_id": "num-1",
        "device_id": "dev-1",
    }
    cloud.retrieve.assert_awaited_once_with(
        "num-1", billing=Billing.NO_COMMERCE, client=cloud.client
    )
    cloud.unbind.assert_awaited_once_with(
        "dev-1", billing=Billing.NO_COMMERCE, client=cloud.client
    )
    cloud.terminate.assert_not_awaited()


def test_unbind_unbound_number_reports_error(cloud):
    cloud.retrieve.return_value.data.device_binding = None
    result = CliRunner().invoke(cli, ["number", "unbind", "num-1"])
    assert result.exit_code == 1
    assert "not bound" in result.output
    cloud.unbind.assert_not_awaited()


def test_commands_are_in_number_menu():
    with click.Context(group.number) as ctx:
        names = {choice.value for choice in group.number.menu_choices(ctx)}
    assert {"bind", "unbind"} <= names
