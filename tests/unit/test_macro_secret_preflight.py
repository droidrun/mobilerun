import asyncio
import json
import traceback
from enum import Enum
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import Device, RecordingDevice
from mobilerun.core.extensions import Extension, device_action
from mobilerun.macro.errors import MacroError
from mobilerun.macro.player import MacroPlayer
from tests.support.devices import MemoryDevice


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize(
    "action",
    [
        {"type": "type_secret", "secret_id": "login", "protected": True},
        {"action_type": "type_secret", "name": "  "},
        {
            "action_type": "extension_call",
            "name": "vault.type_field",
            "arguments": {"package": None, "field": "password"},
        },
    ],
)
def test_credential_gate_precedes_all_device_actions(action, direct):
    async def run():
        device = MemoryDevice()
        device.tap = AsyncMock()
        device._input_secret = AsyncMock(return_value=True)
        await DictCredentials({"login": "dummy"}).bind(device)
        player = MacroPlayer(device, delay_between_actions=0)
        tap = {"action_type": "tap", "x": 1, "y": 2}
        macro = {"macro_schema_version": "2.0", "actions": [tap, action, tap]}
        with pytest.raises(MacroError, match="approval|package or name"):
            if direct:
                await player.replay_action(action)
            else:
                await player.replay_macro(macro)
        device.tap.assert_not_awaited()
        device._input_secret.assert_not_awaited()
        assert await player.replay_macro(macro, max_steps=1)
        assert await player.replay_macro(macro, start_from_step=2)

    asyncio.run(run())


class Mode(Enum):
    FIRST = "first"


def test_unreadable_start_mask_aborts_without_observation_or_provider_error():
    from mobilerun.core.credentials import CredentialPolicy

    async def run():
        device = MemoryDevice()
        vault = await DictCredentials({}).bind(device)
        vault.policies = (CredentialPolicy("missing", mask_from_start=True),)
        vault._list_credentials = AsyncMock(side_effect=ValueError("private-example"))
        player = MacroPlayer(device)
        player.state_provider = AsyncMock()
        device.tap = AsyncMock()
        macro = {
            "macro_schema_version": "2.0",
            "actions": [{"type": "tap", "pre_state": {"nodes": []}}],
        }
        with pytest.raises(MacroError, match="prepare credential masking") as exc:
            await player.replay_macro(macro)
        assert exc.value.__context__ is None
        player.state_provider.get_state.assert_not_awaited()
        device.tap.assert_not_awaited()

    asyncio.run(run())


def test_resolver_error_in_debug_traceback_never_contains_value(caplog):
    from mobilerun.errors import ExtensionActionError

    async def run():
        device = MemoryDevice()
        vault = await DictCredentials({}).bind(device)
        vault._find = AsyncMock(side_effect=ValueError("private-canary"))
        with pytest.raises(ExtensionActionError) as exc:
            await MacroPlayer(device).replay_action(
                {"type": "type_secret", "name": "login"}
            )
        assert "private-canary" not in "".join(traceback.format_exception(exc.value))

    with caplog.at_level("DEBUG", logger="mobilerun"):
        asyncio.run(run())


def test_provider_policy_blocks_package_only_and_named_extension_replay():
    from mobilerun.core.credentials import CredentialPolicy

    async def run():
        device = MemoryDevice()
        device.tap = AsyncMock()
        vault = await DictCredentials(
            {"pkg": {"card": {"password": "dummy"}}}, name="cards"
        ).bind(device)
        vault.policies = (
            CredentialPolicy("card", package="pkg", requires_approval=True),
        )
        action = {
            "type": "extension_call",
            "name": "cards.type_field",
            "arguments": {"package": "pkg", "field": "password"},
        }
        with pytest.raises(MacroError, match="approval"):
            await MacroPlayer(device).replay_macro(
                {"macro_schema_version": "2.0", "actions": [{"type": "tap"}, action]}
            )
        device.tap.assert_not_awaited()

    asyncio.run(run())


def test_card_rule_masks_split_valid_numbers_before_any_field_resolution():
    from mobilerun.core.credentials import CredentialPolicy
    from mobilerun.macro.models import normalize_ui_state

    async def run():
        device = MemoryDevice()
        vault = await DictCredentials({}).bind(device)
        vault.policies = (CredentialPolicy("login", mask_card_numbers=True),)
        vault._find = AsyncMock(
            side_effect=AssertionError("ordinary values must not be resolved")
        )
        player = MacroPlayer(device, state_timeout=0)
        player.state_provider = AsyncMock()
        texts = ["prefix", "4111", "1111", "1111", "1111", "invalid 4111111111111112"]
        player.state_provider.get_state.return_value = {
            "elements": [{"text": v} for v in texts]
        }
        masked = [texts[0], *(["[secret:card_number]"] * 4), texts[-1]]
        pre = normalize_ui_state({"elements": [{"text": v} for v in masked]})
        assert await player.replay_macro(
            {
                "macro_schema_version": "2.0",
                "actions": [{"type": "wait", "duration": 0, "pre_state": pre}],
            }
        )
        vault._find.assert_not_awaited()

    asyncio.run(run())


def test_prefilled_and_typed_values_masked_for_guarded_replay(caplog):
    from mobilerun.core.credentials import CredentialPolicy
    from mobilerun.macro.models import normalize_ui_state

    async def run():
        device = MemoryDevice()
        vault = await DictCredentials(
            {"card": "4111111111111111", "login": "private-login"}
        ).bind(device)
        vault.policies = (CredentialPolicy("card", mask_from_start=True),)
        player = MacroPlayer(device, state_timeout=0, delay_between_actions=0)
        first = {"nodes": [{"text": "4111 1111 1111 1111"}]}
        later = {"nodes": [{"text": "private-login"}]}
        player.state_provider = AsyncMock()
        player.state_provider.get_state.side_effect = [first, later]
        device._input_secret = AsyncMock(return_value=True)

        def masked(name):
            return normalize_ui_state({"nodes": [{"text": f"[secret:{name}]"}]})

        macro = {
            "macro_schema_version": "2.0",
            "description": "4111111111111111",
            "actions": [
                {"type": "type_secret", "name": "login", "pre_state": masked("card")},
                {
                    "type": "wait",
                    "duration": 0,
                    "description": "private-login",
                    "pre_state": masked("login"),
                },
            ],
        }
        assert await player.replay_macro(macro)
        assert first["nodes"][0]["text"] == "4111 1111 1111 1111"
        assert later["nodes"][0]["text"] == "private-login"
        device._input_secret.assert_awaited_once_with("private-login")

    with caplog.at_level("INFO"):
        asyncio.run(run())
    assert "4111111111111111" not in caplog.text
    assert "private-login" not in caplog.text


class Payload(BaseModel):
    n: int


class InternalPayload(Payload):
    private_note: str


class Mutator(Extension):
    @device_action(sensitive_parameters=frozenset({"private"}))
    async def mutate(
        self,
        device: Device,
        values: list[int],
        mode: Mode,
        data: Payload,
        private: str,
        fail: bool = False,
    ) -> None:
        """Mutate arguments and optionally fail."""
        values.append(2)
        data.n += 1
        if fail:
            raise ValueError("failed")


def test_recorded_arguments_are_redacted_json_at_call_time_only_on_success():
    async def run():
        from mobilerun.errors import ExtensionActionError

        recorder = RecordingDevice(MemoryDevice())
        extension = await Mutator(name="test").bind(recorder)
        args = (
            [1],
            Mode.FIRST,
            InternalPayload(n=1, private_note="not in action schema"),
            "private-example",
        )
        await extension.mutate(*args)
        with pytest.raises(ExtensionActionError):
            await extension.mutate(*args, fail=True)
        (record,) = recorder.recorded_actions
        assert json.loads(json.dumps(record["arguments"])) == {
            "values": [1],
            "mode": "first",
            "data": {"n": 1},
            "private": "<redacted>",
            "fail": False,
        }

    asyncio.run(run())
