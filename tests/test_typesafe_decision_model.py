import asyncio
import random
import sys
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from mobilerun.agent.decision_models.typesafe import (
    TypeSafeDecisionModel,
    _prepare_screenshot,
)
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.agent.selectors import ActionCandidate, DecisionSelector, SelectionRequest


@pytest.fixture
def sdk(monkeypatch):
    """Replace only the external SDK; exercise the real adapter and lifecycle."""
    client = SimpleNamespace(system_one=AsyncMock(return_value=SimpleNamespace(
        choices={"action": SimpleNamespace(choice="action_0", confidence=0.9)},
    )))
    exit_client = AsyncMock(return_value=False)

    class ScopedClient:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            return await exit_client(*args)

    factory = Mock(return_value=ScopedClient())
    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(
        AsyncTypeSafeClient=factory,
        Choice=lambda **kwargs: kwargs,
        Image=lambda data: SimpleNamespace(data=data),
    ))
    return SimpleNamespace(client=client, factory=factory, exit_client=exit_client)


def _request(**kwargs):
    values = {
        "goal": "Open Battery",
        "tools": {"click": {}},
        "options": (ActionCandidate(ToolCall("click", {"index": 7}), "Click Battery"),),
        "instructions": "Choose the action that advances the goal.",
    }
    values.update(kwargs)
    return SelectionRequest(**values)


def _image(size, mode="RGB", format="PNG"):
    output = BytesIO()
    Image.new(mode, size, "white").save(output, format=format)
    return output.getvalue()


@pytest.mark.parametrize("size,expected", [
    ((1080, 2400), (706, 1568)),
    ((2400, 1080), (1568, 706)),
    ((1568, 1568), (1568, 1568)),
    ((280, 280), (280, 280)),
    ((100, 200), (280, 560)),
    ((100, 2400), (280, 1568)),
    ((2400, 100), (1568, 280)),
])
def test_screenshot_fits_typesafe_limits_without_cropping(size, expected):
    prepared = _prepare_screenshot(_image(size))
    with Image.open(BytesIO(prepared)) as image:
        assert image.size == expected
        assert image.format == "JPEG"
        assert image.mode == "RGB"
    assert len(prepared) < 16 * 1024 * 1024


@pytest.mark.parametrize("mode,format", [("RGBA", "PNG"), ("RGB", "JPEG")])
def test_screenshot_normalizes_format_and_color_mode(mode, format):
    prepared = _prepare_screenshot(_image((400, 600), mode, format))
    with Image.open(BytesIO(prepared)) as image:
        assert image.size == (400, 600)
        assert image.format == "JPEG"
        assert image.mode == "RGB"


def test_screenshot_rotates_exif_orientation():
    output = BytesIO()
    image = Image.new("RGB", (400, 600))
    exif = image.getexif()
    exif[274] = 6
    image.save(output, format="JPEG", exif=exif)
    with Image.open(BytesIO(_prepare_screenshot(output.getvalue()))) as prepared:
        assert prepared.size == (600, 400)
        assert not prepared.getexif()


def test_screenshot_keeps_full_image_when_padding():
    original = Image.new("RGB", (100, 2400), "red")
    output = BytesIO()
    original.save(output, format="PNG")
    with Image.open(BytesIO(_prepare_screenshot(output.getvalue()))) as prepared:
        assert prepared.getpixel((140, 0))[0] >= 250
        assert prepared.getpixel((140, 1567))[0] >= 250
        assert prepared.getpixel((0, 784)) == (255, 255, 255)


def test_screenshot_rejects_invalid_image():
    with pytest.raises(ValueError, match="valid screenshot"):
        _prepare_screenshot(b"not an image")


def test_screenshot_rejects_animation():
    output = BytesIO()
    Image.new("RGB", (400, 400), "red").save(
        output, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (400, 400), "blue")],
    )
    with pytest.raises(ValueError, match="still images"):
        _prepare_screenshot(output.getvalue())


def test_photo_like_screenshot_upload_is_smaller_and_baseline_jpeg():
    output = BytesIO()
    data = random.Random(42).randbytes(400 * 800 * 3)
    Image.frombytes("RGB", (400, 800), data).save(output, format="PNG")
    prepared = _prepare_screenshot(output.getvalue())
    assert len(prepared) < len(output.getvalue()) / 2
    with Image.open(BytesIO(prepared)) as image:
        assert image.size == (400, 800)
        assert image.format == "JPEG"
        assert not image.info.get("progressive", False)


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("instructions", ["", "Custom decision instructions"])
def test_decision_prepares_vision_and_opts_in_only_for_image_requests(sdk, vision, instructions):
    headers = {"X-Custom": "preserve"}
    model = TypeSafeDecisionModel(api_key="test-only", headers=headers)
    request = SelectionRequest(
        "Open Battery", memory="Do not change settings",
        history=[{"action": "click", "result": "Opened Settings"}],
        tools={"click": {}},
        options=(ActionCandidate(ToolCall("click", {"index": 7}), "Click Battery"),),
        screenshot=_image((1080, 2400)) if vision else None,
        instructions=instructions,
    )
    decision = asyncio.run(model.decide(request))

    assert decision.option_index == 0
    sdk.factory.assert_called_once_with(api_key="test-only", headers=headers)
    kwargs = sdk.client.system_one.call_args.kwargs
    assert kwargs["state"]["memory"] == request.memory
    assert kwargs["state"]["history"] == request.history
    sent_instructions = kwargs["questions"]["action"]["instructions"]
    if instructions:
        assert sent_instructions == instructions
    else:
        assert "You cannot write arbitrary text" in sent_instructions
    if vision:
        assert kwargs["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
        with Image.open(BytesIO(kwargs["state"]["screenshot"].data)) as screenshot:
            assert screenshot.size == (706, 1568)
    else:
        assert "extra_headers" not in kwargs
        assert "screenshot" not in kwargs["state"]
    assert headers == {"X-Custom": "preserve"}
    sdk.exit_client.assert_awaited_once_with(None, None, None)


@pytest.mark.parametrize("choice,confidence,expected_index", [
    ("action_0", 0.49, None),
    ("action_0", 0.5, 0),
    ("abstain", 0.9, None),
])
def test_confidence_threshold_and_explicit_abstention(sdk, choice, confidence, expected_index):
    sdk.client.system_one.return_value.choices["action"] = SimpleNamespace(
        choice=choice, confidence=confidence,
    )

    decision = asyncio.run(TypeSafeDecisionModel(min_confidence=0.5).decide(_request()))

    if expected_index is None:
        assert decision is None
    else:
        assert decision.option_index == expected_index


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_provider_confidence_is_rejected(sdk, confidence):
    sdk.client.system_one.return_value.choices["action"].confidence = confidence

    with pytest.raises(ValueError, match="invalid action confidence"):
        asyncio.run(TypeSafeDecisionModel().decide(_request()))

    sdk.exit_client.assert_awaited_once()


@pytest.mark.parametrize("choice", [None, "unknown", "complete_success"])
def test_missing_unknown_and_unavailable_completion_choices_are_rejected(sdk, choice):
    sdk.client.system_one.return_value.choices = (
        {} if choice is None else {"action": SimpleNamespace(choice=choice, confidence=0.9)}
    )

    with pytest.raises(ValueError, match="unknown or missing action choice"):
        asyncio.run(TypeSafeDecisionModel().decide(_request()))

    criteria = sdk.client.system_one.call_args.kwargs["questions"]["action"]["criteria"]
    assert "complete_success" not in criteria
    assert "complete_failure" not in criteria


@pytest.mark.parametrize("success", [False, True])
def test_completion_choices_require_the_completion_tool(sdk, success):
    sdk.client.system_one.return_value.choices["action"].choice = (
        "complete_success" if success else "complete_failure"
    )

    decision = asyncio.run(TypeSafeDecisionModel().decide(
        _request(tools={"click": {}, "complete": {}}),
    ))

    assert decision.success is success
    assert decision.option_index is None
    assert decision.message


def test_opaque_choice_labels_preserve_actions_with_duplicate_descriptions(sdk):
    options = tuple(
        ActionCandidate(ToolCall("click", {"index": index}), "Click item")
        for index in (7, 12)
    )
    sdk.client.system_one.return_value.choices["action"].choice = "action_1"

    decision = asyncio.run(TypeSafeDecisionModel().decide(_request(options=options)))

    assert decision.option_index == 1
    criteria = sdk.client.system_one.call_args.kwargs["questions"]["action"]["criteria"]
    assert criteria["action_0"]["parameters"] == {"index": 7}
    assert criteria["action_1"]["parameters"] == {"index": 12}
    assert criteria["action_0"]["description"] == criteria["action_1"]["description"]


def test_empty_options_fail_before_opening_a_client(sdk):
    with pytest.raises(ValueError, match="non-empty action options"):
        asyncio.run(TypeSafeDecisionModel().decide(_request(options=())))
    sdk.factory.assert_not_called()


def test_provider_failure_closes_client_and_stops_progress_task(sdk):
    error = RuntimeError("Provider unavailable")
    sdk.client.system_one.side_effect = error

    async def run():
        before = asyncio.all_tasks()
        with pytest.raises(RuntimeError) as raised:
            await TypeSafeDecisionModel().decide(_request())
        assert raised.value is error
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    sdk.exit_client.assert_awaited_once()
    assert sdk.exit_client.call_args.args[:2] == (RuntimeError, error)


@pytest.mark.parametrize("interrupt", ["cancel", "timeout"])
def test_interrupted_requests_close_client_and_leave_no_background_tasks(sdk, interrupt):
    async def run():
        started = asyncio.Event()

        async def blocked_request(**kwargs):
            started.set()
            await asyncio.Event().wait()

        sdk.client.system_one.side_effect = blocked_request
        before = asyncio.all_tasks()
        model = TypeSafeDecisionModel()
        if interrupt == "timeout":
            # Exercise cancellation through the actual selector/provider boundary.
            with pytest.raises(TimeoutError):
                await DecisionSelector(model, timeout=0.05).select(_request())
        else:
            task = asyncio.create_task(model.decide(_request()))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert started.is_set()
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    sdk.exit_client.assert_awaited_once()
    assert sdk.exit_client.call_args.args[0] is asyncio.CancelledError
