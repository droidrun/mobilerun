"""Scheduled flows are read/toggle targets, not invented custom triggers."""

import pytest

from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox.errors import SandboxRunError
from tests.unit.test_sandbox_execution import remote as remote_fixture

remote = remote_fixture


@pytest.mark.anyio
async def test_scheduled_flow_rejects_fire_without_mutation(remote):
    remote.document["on"] = {"schedule": {"cron": "* * * * *", "timezone": "UTC"}}
    with pytest.raises(SandboxRunError, match="schedule"):
        await deploy.fire_remote("demo", {})
    remote.fake.fire_trigger.assert_not_awaited()
    assert not (await deploy.describe_remote("demo")).get("endpoint")


@pytest.mark.anyio
async def test_legacy_custom_trigger_without_endpoint_schema_cannot_fire(remote):
    remote.document["on"] = {"custom": {}}
    remote.fake.get_version.return_value.pop("inputSchema")
    with pytest.raises(SandboxRunError, match="deploy as an endpoint"):
        await deploy.fire_remote("demo", {})
    remote.fake.fire_trigger.assert_not_awaited()
    assert not (await deploy.describe_remote("demo")).get("endpoint")
