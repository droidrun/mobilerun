"""Paid replay consent through the real CLI and an offline device."""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from click import Context
from click.testing import CliRunner

from mobilerun.cli.base import billing, safety
from mobilerun.cli.devices.base import options, selection
from mobilerun.core.extensions import Extension, action
from mobilerun.macro import MacroOperator
from mobilerun.macro.cli import macro
from mobilerun.macro.cli.replay import run_replay
from tests.support.devices import MemoryDevice


class Outbox(Extension):
    """Offline provider with the same action shape as a bound phone number."""

    role = "phone_number"

    def __init__(self, name):
        super().__init__(name=name)
        self.messages = []

    @action()
    async def send(self, to: str, text: str) -> None:
        """Record the message without sending it."""
        self.messages.append((to, text))


@pytest.fixture
def replay_sms(tmp_path, monkeypatch, request):
    name = getattr(request, "param", "outbox")
    device = MemoryDevice(cloud_id="offline-cloud")
    device.tap = AsyncMock()
    outbox = Outbox(name)
    if name != "outbox":
        outbox.role = None
    asyncio.run(outbox.bind(device))
    path = tmp_path / "macro.json"
    messages = ["first", "second"]
    path.write_text(
        json.dumps(
            {
                "macro_schema_version": "2.0",
                "actions": [
                    {"action_type": "tap", "x": 10, "y": 20},
                    *[
                        {
                            "type": "extension_call",
                            "name": f"{name}.send",
                            "arguments": {"to": "+15550100000", "text": text},
                        }
                        for text in messages
                    ],
                ],
            }
        )
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=device))
    monkeypatch.setattr(billing, "billing_prompt_is_interactive", lambda: True)
    monkeypatch.setenv("MOBILERUN_BILLING", "paid")
    confirm = AsyncMock(return_value=False)
    monkeypatch.setattr(safety, "confirm_prompt_async", confirm)

    def invoke(*flags, parent=None, explicit_target=True):
        return CliRunner().invoke(
            macro,
            [
                "replay",
                str(path),
                *(["offline-cloud", "--cloud"] if explicit_target else []),
                "--delay",
                "0",
                *flags,
            ],
            parent=parent,
        )

    return invoke, device, outbox, confirm


@pytest.mark.parametrize("replay_sms", ["outbox", "sms", "other"], indirect=True)
def test_declining_paid_sms_replays_nothing(replay_sms):
    invoke, device, outbox, confirm = replay_sms
    result = invoke()

    assert result.exit_code == 1, result.output
    assert "Cancelled." in result.output
    confirm.assert_awaited_once_with(
        "Send 2 SMS with paid billing (charges may apply)?", default=False
    )
    device.tap.assert_not_awaited()
    assert outbox.messages == []


def test_accepting_paid_sms_replays_with_one_prompt(replay_sms):
    invoke, device, outbox, confirm = replay_sms

    async def accept(message, *, default):
        device.tap.assert_not_awaited()
        assert outbox.messages == []
        return True

    confirm.side_effect = accept
    result = invoke()

    assert result.exit_code == 0, result.output
    confirm.assert_awaited_once_with(
        "Send 2 SMS with paid billing (charges may apply)?", default=False
    )
    device.tap.assert_awaited_once_with(10, 20)
    assert outbox.messages == [("+15550100000", "first"), ("+15550100000", "second")]


def test_yes_replays_paid_sms_without_a_terminal(replay_sms, monkeypatch):
    invoke, device, outbox, confirm = replay_sms
    monkeypatch.setattr(billing, "billing_prompt_is_interactive", lambda: False)
    result = invoke("--yes")

    assert result.exit_code == 0, result.output
    confirm.assert_not_awaited()
    device.tap.assert_awaited_once_with(10, 20)
    assert len(outbox.messages) == 2


def test_noninteractive_paid_sms_requires_yes_before_any_step(replay_sms, monkeypatch):
    invoke, device, outbox, confirm = replay_sms
    monkeypatch.setattr(billing, "billing_prompt_is_interactive", lambda: False)
    result = invoke()

    assert result.exit_code == 1, result.output
    assert "pass --yes" in result.output
    confirm.assert_not_awaited()
    device.tap.assert_not_awaited()
    assert outbox.messages == []


def test_custom_sms_in_included_mode_still_requires_consent(replay_sms, monkeypatch):
    invoke, device, outbox, confirm = replay_sms
    monkeypatch.setenv("MOBILERUN_BILLING", "included")
    monkeypatch.setattr(billing, "billing_prompt_is_interactive", lambda: False)
    result = invoke()

    assert result.exit_code == 1, result.output
    assert "pass --yes" in result.output
    confirm.assert_not_awaited()
    device.tap.assert_not_awaited()
    assert outbox.messages == []


@pytest.mark.parametrize("interactive", [True, False])
def test_sms_refusal_never_discovers_or_opens_device(
    replay_sms, monkeypatch, interactive
):
    invoke, device, _, _ = replay_sms
    discoveries = AsyncMock(
        return_value=[
            options.DiscoveredDevice("offline-cloud", "cloud", None, "ready", "android")
        ]
    )
    opened = []

    @asynccontextmanager
    async def fake_open(target):
        opened.append(target)
        yield device

    monkeypatch.setattr(options, "discover_devices", discoveries)
    monkeypatch.setattr(options.DeviceTarget, "open", fake_open)
    monkeypatch.setattr(billing, "billing_prompt_is_interactive", lambda: interactive)
    for explicit_target in (True, False):
        result = invoke(explicit_target=explicit_target)
        assert result.exit_code == 1, result.output
        assert ("Cancelled." if interactive else "pass --yes") in result.output
    assert opened == []
    discoveries.assert_not_awaited()


def test_agent_handoff_replay_keeps_the_sms_preflight(replay_sms):
    invoke, device, outbox, confirm = replay_sms
    parent = Context(macro)

    handed_off = []

    def handoff(*, provider, model, **options):
        handed_off.append(True)
        run_replay(MacroOperator, **options)

    parent.meta["macro_agent_handoff"] = handoff
    result = invoke("--on-mismatch", "agent", parent=parent)

    assert result.exit_code == 1, result.output
    assert "Cancelled." in result.output
    confirm.assert_awaited_once()
    assert handed_off == []
    device.tap.assert_not_awaited()
    assert outbox.messages == []


def test_replay_uses_the_recording_checked_before_open(
    replay_sms, monkeypatch, tmp_path
):
    invoke, device, outbox, confirm = replay_sms
    path = tmp_path / "macro.json"

    @asynccontextmanager
    async def fake_open(target):
        data = json.loads(path.read_text())
        data["actions"] = data["actions"][1:]
        path.write_text(json.dumps(data))
        yield device

    monkeypatch.setattr(options.DeviceTarget, "open", fake_open)
    result = invoke("--max-steps", "1")

    assert result.exit_code == 0, result.output
    confirm.assert_not_awaited()
    device.tap.assert_awaited_once_with(10, 20)
    assert outbox.messages == []


def test_replay_without_sms_in_selected_steps_does_not_prompt(replay_sms):
    invoke, device, outbox, confirm = replay_sms
    result = invoke("--max-steps", "1")

    assert result.exit_code == 0, result.output
    confirm.assert_not_awaited()
    device.tap.assert_awaited_once_with(10, 20)
    assert outbox.messages == []


def test_paid_sms_confirmation_counts_only_selected_steps(replay_sms):
    invoke, device, outbox, confirm = replay_sms
    confirm.return_value = True
    result = invoke("--start-from", "3", "--max-steps", "1")

    assert result.exit_code == 0, result.output
    confirm.assert_awaited_once_with(
        "Send 1 SMS with paid billing (charges may apply)?", default=False
    )
    device.tap.assert_not_awaited()
    assert outbox.messages == [("+15550100000", "second")]


def test_dry_run_with_paid_sms_does_not_prompt_or_replay(replay_sms):
    invoke, device, outbox, confirm = replay_sms
    result = invoke("--dry-run")

    assert result.exit_code == 0, result.output
    assert "extension_call [redacted]" in result.output
    confirm.assert_not_awaited()
    device.tap.assert_not_awaited()
    assert outbox.messages == []
