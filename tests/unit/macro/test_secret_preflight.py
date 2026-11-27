import asyncio
import traceback
from enum import Enum
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import Device, RecordingDevice
from mobilerun.core.extensions import Extension, device_action
from mobilerun.errors import ExtensionActionError
from mobilerun.macro import MacroStep
from mobilerun.macro.errors import MacroError, MacroMismatch
from mobilerun.macro.player import MacroPlayer
from mobilerun.models import Fingerprint
from tests.support.devices import MemoryDevice
from tests.support.macro import read_run_steps


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize(
    "action,args",
    [
        ("type_secret", {"name": "login", "protected": True}),
        ("type_secret", {"name": "  "}),
        ("input_text", {"text": "[secret:login]"}),
        (
            "extension_call",
            {
                "name": "vault.type_field",
                "arguments": {"package": None, "field": "password"},
            },
        ),
    ],
)
def test_credential_gate_precedes_all_device_actions(action, args, direct):
    async def run():
        device = MemoryDevice()
        device.tap = AsyncMock()
        device._input_secret = AsyncMock(return_value=True)
        await DictCredentials({"login": "dummy"}).bind(device)
        player = MacroPlayer(device, delay_between_actions=0)
        step = MacroStep(i=2, action=action, args=args)
        macro = [
            MacroStep(i=1, action="tap", args={"x": 1, "y": 2}),
            step,
            MacroStep(i=3, action="tap", args={"x": 1, "y": 2}),
        ]
        with pytest.raises(MacroError, match="approval|package or name|redacted"):
            if direct:
                await player.play_step(step)
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
        macro = [
            MacroStep(
                i=1,
                action="tap",
                args={"x": 1, "y": 2},
                before={"app": "settings", "activity": None, "elements": []},
            )
        ]
        with pytest.raises(MacroError, match="prepare credential masking") as exc:
            await player.replay_macro(macro)
        assert exc.value.__context__ is None
        player.state_provider.get_state.assert_not_awaited()
        device.tap.assert_not_awaited()

    asyncio.run(run())


def test_resolver_error_in_debug_traceback_never_contains_value(caplog):
    async def run():
        device = MemoryDevice()
        vault = await DictCredentials({}).bind(device)
        vault._find = AsyncMock(side_effect=ValueError("private-canary"))
        with pytest.raises(MacroMismatch) as exc:
            await MacroPlayer(device).play_step(
                MacroStep(i=1, action="type_secret", args={"name": "login"})
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
        step = MacroStep(
            i=2,
            action="extension_call",
            args={
                "name": "cards.type_field",
                "arguments": {"package": "pkg", "field": "password"},
            },
        )
        with pytest.raises(MacroError, match="approval"):
            await MacroPlayer(device).replay_macro(
                [MacroStep(i=1, action="tap", args={"x": 1, "y": 2}), step]
            )
        device.tap.assert_not_awaited()

    asyncio.run(run())


def test_structural_guard_ignores_card_text_without_resolving_fields():
    from mobilerun.core.credentials import CredentialPolicy

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
            "phone_state": {"packageName": "settings"},
            "elements": [{"text": v} for v in texts],
        }
        pre = Fingerprint(app="settings", activity=None, elements=())
        assert await player.replay_macro(
            [MacroStep(i=1, action="wait", args={"duration": 0}, before=pre)]
        )
        vault._find.assert_not_awaited()

    asyncio.run(run())


def test_prefilled_and_typed_values_ignored_by_structural_guards(caplog):
    from mobilerun.core.credentials import CredentialPolicy

    async def run():
        device = MemoryDevice()
        vault = await DictCredentials(
            {"card": "4111111111111111", "login": "private-login"}
        ).bind(device)
        vault.policies = (CredentialPolicy("card", mask_from_start=True),)
        player = MacroPlayer(device, state_timeout=0, delay_between_actions=0)
        first = {
            "phone_state": {"packageName": "settings"},
            "nodes": [{"text": "4111 1111 1111 1111"}],
        }
        later = {
            "phone_state": {"packageName": "settings"},
            "nodes": [{"text": "private-login"}],
        }
        player.state_provider = AsyncMock()
        player.state_provider.get_state.side_effect = [first, later]
        device._input_secret = AsyncMock(return_value=True)

        guard = Fingerprint(app="settings", activity=None, elements=())
        macro = [
            MacroStep(i=1, action="type_secret", args={"name": "login"}, before=guard),
            MacroStep(i=2, action="wait", args={"duration": 0}, before=guard),
        ]
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


def test_recorded_arguments_are_redacted_before_mutation_even_on_failure(tmp_path):
    async def run():
        recorder = RecordingDevice(MemoryDevice(), runs_dir=tmp_path, capture="actions")
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
        succeeded, failed = read_run_steps(recorder.run_dir)
        assert succeeded.ok and not failed.ok
        assert succeeded.args["arguments"] == {
            "values": [1],
            "mode": "first",
            "data": {"n": 1},
            "private": "***",
            "fail": False,
        }
        assert failed.args["arguments"] == {
            "values": [1],
            "mode": "first",
            "data": {"n": 2},
            "private": "***",
            "fail": True,
        }

    asyncio.run(run())
