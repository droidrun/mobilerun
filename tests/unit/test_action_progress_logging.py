import asyncio
import logging
from types import SimpleNamespace

from mobilerun.clients.cloud.tasks import _progress
from mobilerun.logs.actions import agent_line


def test_action_logging_accepts_context_without_optional_state_or_ui():
    ctx = SimpleNamespace(driver=SimpleNamespace(identifier="device"))
    assert agent_line(ctx, "tap", {"x": 1, "y": 2}) == ("device · step 1  tap (1, 2)")


def test_cloud_progress_summarizes_action_without_sensitive_values(caplog):
    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def iter_lines(self):
            yield 'data: {"type":"tool_execution", "tool_name":"click_at", "tool_args":{"x":540,"y":1200,"text":"sensitive"}}'
            yield 'data: {"type":"unknown", "token":"sensitive"}'
            yield "data: not-json"

    client = SimpleNamespace(
        tasks=SimpleNamespace(
            with_streaming_response=SimpleNamespace(attach=lambda *a, **k: Stream())
        )
    )
    with caplog.at_level(logging.INFO, logger="mobilerun.actions"):
        asyncio.run(_progress(client, "task", billing="no_commerce"))
    assert [r.message for r in caplog.records] == [
        "unknown       · task  · step 1  tap (540, 1200) <redacted>",
    ]


def test_agent_action_uses_device_identifier_not_legacy_attributes():
    from mobilerun.core.devices.base import Device

    class ExampleDevice(Device):
        @property
        def identifier(self):
            return "typed-id"

        serial = "stale-secret"
        device_id = "old-id"

    ctx = SimpleNamespace(
        driver=ExampleDevice(), shared_state=SimpleNamespace(step_number=1), ui=None
    )
    assert agent_line(ctx, "tap", {"x": 1, "y": 2}) == "typed-id · step 1  tap (1, 2)"


def test_agent_action_with_no_identifier_has_safe_unknown_label():
    from mobilerun.core.devices.base import Device

    ctx = SimpleNamespace(
        driver=Device(), shared_state=SimpleNamespace(step_number=1), ui=None
    )
    assert agent_line(ctx, "tap", {"x": 1, "y": 2}) == "unknown · step 1  tap (1, 2)"
