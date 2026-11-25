import asyncio
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.credentials import CredentialPolicy, DictCredentials
from mobilerun.core.credentials.masking import CredentialMasks
from mobilerun.macro.errors import MacroError
from mobilerun.macro.models import normalize_ui_state
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.utils import compare_states
from tests.support.devices import MemoryDevice


def test_reordered_split_echo_masks_only_bounded_fragments():
    masks = CredentialMasks()
    masks.register("pin", "123456")
    state = normalize_ui_state(
        [
            {"text": text, "contentDescription": f"Order 1234; group {echo}"}
            for text, echo in [("123", "456"), ("456", "123")]
        ]
    )
    assert [node["content_description"] for node in masks.snapshot(state)["nodes"]] == [
        "Order 1234; group [secret:pin]"
    ] * 2


def test_split_pin_masks_echoes_without_erasing_description_text():
    masks = CredentialMasks()
    masks.register("pin", "123456")
    state = normalize_ui_state(
        [
            {"text": p, "contentDescription": f"Digit group: {p}; done"}
            for p in ("123", "456")
        ]
    )
    for node in masks.snapshot(state)["nodes"]:
        assert node["text"] == "[secret:pin]"
        assert node["content_description"] == "Digit group: [secret:pin]; done"


def snapshot(masks, *texts):
    return masks.snapshot(normalize_ui_state([{"text": t} for t in texts]))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_overlapping_pin_and_card_mask_outer_span(monkeypatch, reverse, split):
    masks = CredentialMasks()
    masks.register("pin", "4111")
    masks.mask_card_numbers = True
    if reverse:
        rules = tuple(masks._rules())
        monkeypatch.setattr(masks, "_rules", lambda: reversed(rules))
    card = "4111 1111 1111 1111"
    texts = card.split() if split else [card]
    assert [node["text"] for node in snapshot(masks, *texts)["nodes"]] == [
        "[secret:card_number]"
    ] * len(texts)


def test_package_independent_mask_registers_every_matching_field():
    masks = CredentialMasks()
    vault = DictCredentials(
        {p: {"login": {"password": v}} for p, v in [("a", "first"), ("b", "second")]}
    )
    vault.policies = (CredentialPolicy("login", mask_from_start=True),)
    asyncio.run(vault.prepare_masks(masks))
    assert [n["text"] for n in snapshot(masks, "first", "second")["nodes"]] == [
        "[secret:login]"
    ] * 2


@pytest.mark.parametrize("protected", [True, False])
def test_missing_named_credential_binding_fails_before_tap(protected):
    async def run():
        device = MemoryDevice()
        device.tap = AsyncMock()
        await DictCredentials({"login": "test"}).bind(device)
        macro = {
            "macro_schema_version": "2.0",
            "actions": [
                {"type": "tap"},
                {
                    "type": "extension_call",
                    "name": "other_vault.type_field",
                    "arguments": {"name": "login"},
                    "protected": protected,
                },
            ],
        }
        with pytest.raises(MacroError):
            await MacroPlayer(device, delay_between_actions=0).replay_macro(macro)
        device.tap.assert_not_awaited()

    asyncio.run(run())


def test_masks_preserve_digit_word_boundaries_and_unrelated_label_text():
    masks = CredentialMasks()
    masks.register("pin", "123")
    masks.register("name", "Ann")
    assert not compare_states(
        snapshot(masks, "Order 1234"), snapshot(masks, "Order 9123"), 1.0
    ).matches
    text = "Ann versus Annie: 123 or 1234"
    assert (
        snapshot(masks, text)["nodes"][0]["text"]
        == "[secret:name] versus Annie: [secret:pin] or 1234"
    )
