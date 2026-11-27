"""Guarded playback uses original evidence indices and fails closed."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.credentials import CredentialPolicy, DictCredentials
from mobilerun.errors.base import DeviceError
from mobilerun.macro.errors import MacroError, MacroMismatch
from mobilerun.macro.models import MacroStep
from mobilerun.macro.operator import MacroOperator
from mobilerun.macro.player import MacroPlayer
from mobilerun.models import Fingerprint
from tests.support.macro import FakeDevice, write_macro

READY = Fingerprint(app="settings", activity=".Ready", elements=("id:continue",))
DONE = Fingerprint(app="settings", activity=".Done", elements=())


def test_stepwise_replay_refuses_handoff_after_a_vault_row():
    from mobilerun.core.credentials.masking import current_masks

    async def run():
        device = FakeDevice(_screen(".Wrong"))
        device._input_secret = AsyncMock(return_value=True)
        await DictCredentials({"login": "example-secret"}).bind(device)
        player = MacroPlayer(device, state_timeout=0)
        with player.replay_scope():
            await player.play_step(_step(action="type_secret", args={"name": "login"}))
            assert player.tainted
            assert current_masks.get().active
            with pytest.raises(MacroError, match="handoff refused"):
                await player.play_step(_step(before=READY))
        assert current_masks.get() is None
        assert player.tainted

    asyncio.run(run())


def test_masks_prepare_once_per_replay_scope_and_lazily_outside_it():
    class Vault(DictCredentials):
        preparations = 0

        async def prepare_masks(self, masks):
            self.preparations += 1
            await super().prepare_masks(masks)

    async def run():
        device = FakeDevice()
        vault = await Vault({"login": "example-secret"}).bind(device)
        player = MacroPlayer(device)
        with player.replay_scope():
            assert vault.preparations == 0
            await player.play_step(_step())
            with player.replay_scope():
                await player.play_step(_step())
        assert vault.preparations == 1
        await player.play_step(_step())
        assert vault.preparations == 2

    asyncio.run(run())


def _screen(activity, element=None):
    return {
        "phone_state": {"packageName": "settings", "activityName": activity},
        "elements": [element] if element is not None else [],
    }


def _step(**fields):
    return MacroStep(**{"i": 7, "action": "tap", "args": {"x": 60, "y": 40}, **fields})


def test_player_polls_guards_and_uses_saved_coordinates():
    device = FakeDevice(
        _screen(".Loading"),
        _screen(".Ready", {"resourceId": "continue", "bounds": "100,100,120,120"}),
        _screen(".Done"),
    )
    player = MacroPlayer(device, state_timeout=0.5, state_poll_interval=0)

    asyncio.run(player.play_step(_step(before=READY, after=DONE, target="continue")))

    assert device.taps == [(60, 40)]
    assert device.observations == 3


@pytest.mark.parametrize("phase", ["before", "action", "after"])
def test_divergence_reports_original_index_and_phase(phase):
    class RejectTap(FakeDevice):
        async def tap(self, x, y):
            if phase == "action":
                raise DeviceError("not accepted")
            await super().tap(x, y)

    ready = _screen(".Ready", {"resourceId": "continue"})
    device = RejectTap(_screen(".Wrong") if phase == "before" else ready, ready)
    player = MacroPlayer(device, state_timeout=0)
    step = _step(before=READY, after=DONE)

    with pytest.raises(MacroMismatch, match=rf"step 7.*{phase}") as caught:
        asyncio.run(player.replay_macro([_step(i=2), step], start_from_step=1))

    assert caught.value.step is step
    assert caught.value.phase == phase
    assert caught.value.observed == (
        Fingerprint(app="settings", activity=".Wrong", elements=())
        if phase == "before"
        else READY
    )
    assert device.taps == ([(60, 40)] if phase == "after" else [])


def test_false_input_text_stops_at_action_without_checking_after():
    class RejectText(FakeDevice):
        async def input_text(self, text, clear=False):
            self.inputs.append((text, clear))
            return False

    device = RejectText()
    player = MacroPlayer(device, state_timeout=0)
    step = _step(
        action="input_text", args={"text": "offline", "clear": True}, after=DONE
    )
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(player.play_step(step))
    assert caught.value.phase == "action"
    assert caught.value.observed is None
    assert device.inputs == [("offline", True)]
    assert device.observations == 0


@pytest.mark.parametrize("guard", ["before", "after"])
def test_missing_observed_evidence_cannot_pass_present_guard(guard):
    device = FakeDevice({})
    player = MacroPlayer(device, state_timeout=0, state_threshold=0)

    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(player.play_step(_step(**{guard: READY})))

    assert caught.value.phase == guard
    assert caught.value.observed is None
    assert device.taps == ([(60, 40)] if guard == "after" else [])


def test_missing_guards_skip_observation():
    device = FakeDevice({})
    asyncio.run(MacroPlayer(device).play_step(_step()))
    assert device.taps == [(60, 40)]
    assert device.observations == 0


def test_unavailable_state_samples_are_not_observations():
    device = FakeDevice(None)
    player = MacroPlayer(device, state_timeout=0)
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(player.play_step(_step(before=READY)))
    assert caught.value.observed is None
    assert device.taps == []


def test_active_credential_masks_match_recorded_structural_keys():
    async def run():
        device = FakeDevice(_screen(".Ready", {"resourceId": "private-canary"}))
        vault = await DictCredentials({"login": "private-canary"}).bind(device)
        vault.policies = (CredentialPolicy("login", mask_from_start=True),)
        player = MacroPlayer(device, state_timeout=0)
        expected = READY.model_copy(update={"elements": ("id:[secret:login]",)})
        await player.play_step(_step(before=expected))
        assert device.taps == [(60, 40)]

    asyncio.run(run())


def test_credential_reference_types_field_without_persisting_value(caplog):
    async def run():
        device = FakeDevice()
        device._input_secret = AsyncMock(return_value=True)
        await DictCredentials({"login": "private-canary"}).bind(device)
        await MacroPlayer(device).play_step(
            _step(action="type_secret", args={"name": "login"})
        )
        device._input_secret.assert_awaited_once_with("private-canary")

    with caplog.at_level("DEBUG"):
        asyncio.run(run())
    assert "private-canary" not in caplog.text


def test_credential_preflight_blocks_earlier_actions_and_observation():
    async def run():
        device = FakeDevice()
        vault = await DictCredentials({"login": "example-only"}).bind(device)
        vault.policies = (CredentialPolicy("login", requires_approval=True),)
        player = MacroPlayer(device)
        protected = _step(
            i=8,
            action="extension_call",
            args={
                "name": "vault.type_field",
                "arguments": {"name": "login", "field": "password", "package": None},
            },
        )
        with pytest.raises(MacroError, match="approval"):
            await player.replay_macro([_step(before=READY), protected])
        assert device.taps == []
        assert device.observations == 0

    asyncio.run(run())


def test_masked_macro_argument_is_not_sent_to_device():
    device = FakeDevice()
    step = _step(action="input_text", args={"text": "prefix [secret:login]"})
    with pytest.raises(MacroError, match="redacted"):
        asyncio.run(MacroPlayer(device).play_step(step))
    assert device.inputs == []


def test_operator_consumes_preflight_snapshot_even_if_file_changes(tmp_path):
    from mobilerun.macro.builder import _replay_snapshot, load_macro

    path = write_macro(tmp_path)
    snapshot = load_macro(path)
    token = _replay_snapshot.set((path, snapshot))
    try:
        write_macro(
            tmp_path, rows=[{"i": 2, "action": "tap", "args": {"x": 1, "y": 2}}]
        )
        device = FakeDevice()
        result = asyncio.run(MacroOperator().execute(device, str(path)))
    finally:
        _replay_snapshot.reset(token)

    assert result.success and result.steps == 1
    assert device.taps == [(10, 20)]
