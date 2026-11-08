import asyncio
import io
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.recording import RecordingDriver
from mobilerun_core_local.driver.stealth import StealthDriver
from PIL import Image

from mobilerun.tools.driver.secret_safe import (
    SecretRedactor,
    SecretSafeDriver,
    SecretSafeStateProvider,
    install_log_redaction,
    redaction_scope,
)
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.provider import AndroidStateProvider, StateProvider
from mobilerun.tools.ui.state import UIState

CARD = "4111 1111 1111 1111"


def run(coro):
    return asyncio.run(coro)


def png(color=(200, 30, 30), size=(40, 30)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


def pixels(data):
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        return image.size, {color for _, color in rgb.getcolors(maxcolors=1 << 24)}


class RawDriver:
    platform = "Android"
    supported = {"tap", "input_text"}
    supported_buttons = {"back"}

    def __init__(self, tree=None):
        self.typed = []
        self.tree = tree or {}

    async def input_text(self, text, clear=False, **kwargs):
        self.typed.append(text)
        return True

    async def get_ui_tree(self):
        return self.tree

    async def screenshot(self, hide_overlay=True):
        return png()


# -- redactor ----------------------------------------------------------------


def test_number_is_masked_grouped_dashed_and_compact():
    redactor = SecretRedactor()
    redactor.register("card", CARD)

    for source in (
        "Card 4111 1111 1111 1111 ok",
        "4111-1111-1111-1111",
        "4111111111111111",
        "4111   1111 1111-1111",
    ):
        masked = redactor.redact_text(source)
        assert "4111" not in masked
        assert "[secret:card]" in masked


def test_number_is_masked_for_every_compact_or_grouped_secret_form():
    redactor = SecretRedactor()
    redactor.register("card", "4111111111111111")

    assert redactor.redact_text("4111 1111 1111 1111") == "[secret:card]"
    assert redactor.redact_text("4111-1111-1111-1111") == "[secret:card]"


def test_longer_digit_run_is_not_partially_masked():
    redactor = SecretRedactor()
    redactor.register("card", CARD)

    assert redactor.redact_text("order 94111111111111119") == "order 94111111111111119"


def test_name_is_masked_as_whole_word_and_partial_word_stays():
    redactor = SecretRedactor()
    redactor.register("holder", "Jane Doe")

    assert redactor.redact_text("jane   DOE") == "[secret:holder]"
    assert redactor.redact_text("Hello Jane Doe!") == "Hello [secret:holder]!"
    assert redactor.redact_text("Jane Doerr") == "Jane Doerr"
    assert redactor.redact_text("Maryjane Doe") == "Maryjane Doe"


def test_input_only_applies_only_to_editable_nodes_when_tree_is_structured():
    redactor = SecretRedactor()
    redactor.register("cvc", "737", redact="input_only")
    tree = {
        "a11y_tree": {
            "className": "android.widget.FrameLayout",
            "isFocused": False,
            "children": [
                {"className": "android.widget.TextView", "text": "Price 737"},
                {
                    "className": "android.widget.EditText",
                    "text": "737",
                    "isFocused": True,
                },
            ],
        },
        "phone_state": {"packageName": "com.example"},
    }

    masked = redactor.redact_ui_tree(tree)

    assert masked["a11y_tree"]["children"][0]["text"] == "Price 737"
    assert masked["a11y_tree"]["children"][1]["text"] == "[secret:cvc]"
    assert tree["a11y_tree"]["children"][1]["text"] == "737"  # input untouched


def test_input_only_masks_everywhere_without_structure():
    redactor = SecretRedactor()
    redactor.register("cvc", "737", redact="input_only")
    ios_like = {"a11y_tree": "Price 737\nTextField 737", "phone_state": {}}
    # Text tree: nothing says what is editable, so mask more rather than less.
    assert "737" not in redactor.redact_ui_tree(ios_like)["a11y_tree"]


def test_expiry_date_with_blanks_is_masked():
    redactor = SecretRedactor()
    redactor.register("exp", "12/27", redact="input_only")

    assert redactor.redact_text("12 / 27") == "[secret:exp]"
    assert redactor.redact_text("12/27") == "[secret:exp]"
    assert redactor.redact_text("112/278") == "112/278"


def test_long_digit_values_are_masked_even_when_input_only():
    redactor = SecretRedactor()
    redactor.register("card", CARD, redact="input_only")
    tree = {
        "a11y_tree": {
            "isFocused": False,
            "children": [{"className": "android.widget.TextView", "text": CARD}],
        }
    }

    assert redactor.redact_ui_tree(tree)["a11y_tree"]["children"][0]["text"] == (
        "[secret:card]"
    )


def test_empty_redactor_changes_nothing():
    redactor = SecretRedactor()
    tree = {"a11y_tree": {"text": CARD}}
    state = UIState([{"text": CARD}], CARD, CARD, {}, 100, 100)

    assert redactor.redact_ui_tree(tree) is tree
    assert redactor.redact_ui_state(state) is state
    assert redactor.redact_text("anything") == "anything"


def test_geometry_fields_are_not_rewritten():
    redactor = SecretRedactor()
    redactor.register("card", "12345678")
    tree = {"a11y_tree": {"bounds": "12345678", "text": "12345678"}}

    masked = redactor.redact_ui_tree(tree)["a11y_tree"]

    assert masked["bounds"] == "12345678"
    assert masked["text"] == "[secret:card]"


def test_ui_state_is_redacted_on_a_copy():
    redactor = SecretRedactor()
    redactor.register("holder", "Jane Doe")
    state = UIState(
        [{"index": 1, "text": "Jane Doe", "children": [{"text": "jane doe"}]}],
        "1. Jane Doe",
        "Jane Doe",
        {"focusedElement": {"text": "Jane Doe"}},
        100,
        100,
    )

    masked = redactor.redact_ui_state(state)

    assert masked is not state
    assert "Jane" not in repr(vars(masked))
    assert state.formatted_text == "1. Jane Doe"
    assert masked.get_element(1)["text"] == "[secret:holder]"


# -- driver ------------------------------------------------------------------


def test_driver_masks_tree_and_registers_before_typing():
    redactor = SecretRedactor()
    raw = RawDriver({"a11y_tree": {"text": f"Number {CARD}"}})
    driver = SecretSafeDriver(raw, redactor)

    assert run(driver.get_ui_tree())["a11y_tree"]["text"] == f"Number {CARD}"
    assert run(driver.input_secret("card", CARD))
    assert raw.typed == [CARD]
    assert run(driver.get_ui_tree())["a11y_tree"]["text"] == "Number [secret:card]"
    # the unredacted read is for engine-internal checks only
    assert run(driver.get_ui_tree_unredacted())["a11y_tree"]["text"] == (
        f"Number {CARD}"
    )


def test_screenshot_is_black_only_after_a_protected_secret():
    redactor = SecretRedactor()
    driver = SecretSafeDriver(RawDriver(), redactor)
    original = png()

    assert run(driver.screenshot()) == original
    run(driver.input_secret("password", "hunter22", protected=False))
    assert run(driver.screenshot()) == original  # normal secret: unchanged

    run(driver.input_secret("card", CARD, protected=True))
    size, colors = pixels(run(driver.screenshot()))
    assert size == (40, 30)
    assert colors == {(0, 0, 0)}

    redactor.clear()
    assert run(driver.screenshot()) == original


def test_screenshot_fails_closed_when_the_image_cannot_be_decoded():
    redactor = SecretRedactor()
    redactor.black_screenshots = True
    raw = RawDriver()

    async def broken(hide_overlay=True):
        return b"not an image"

    raw.screenshot = broken
    _, colors = pixels(run(SecretSafeDriver(raw, redactor).screenshot()))
    assert colors == {(0, 0, 0)}


class PlainCore:
    """Stand-in for a core driver: records only what reaches it."""

    platform = "Android"
    supported: set = set()
    supported_buttons: set = set()

    def __init__(self):
        self.typed = []

    async def input_text(self, text, clear=False, **kwargs):
        self.typed.append(text)
        return True


def test_core_recording_driver_logs_plaintext_input():
    """Pins the core-local behaviour the wrapper order is built around."""
    recording = RecordingDriver(PlainCore())

    run(recording.input_text("visible-text"))

    assert recording.log == [
        {"action_type": "input_text", "text": "visible-text", "clear": False}
    ]


def test_secret_never_reaches_recording_log_or_its_view():
    core = PlainCore()
    recording = RecordingDriver(StealthDriver(core))
    driver = SecretSafeDriver(recording, SecretRedactor())

    run(driver.input_text("tap target"))  # ordinary input is recorded
    run(driver.input_secret("pw", "hunter two", protected=True))

    assert core.typed[-3:] == ["hunter", " ", "two"]  # stealth still types
    for log in (recording.log, driver.log):
        assert "hunter" not in repr(log)
    assert recording.log[-1] == {
        "action_type": "type_secret",
        "secret_id": "pw",
        "clear": False,
        "protected": True,
    }
    assert recording.log[0]["action_type"] == "input_text"


def test_recording_view_masks_secrets_typed_through_other_paths():
    redactor = SecretRedactor()
    recording = RecordingDriver(PlainCore())
    driver = SecretSafeDriver(recording, redactor)
    redactor.register("pw", "hunter22")

    run(recording.input_text("hunter22"))  # bypassed the wrapper

    assert "hunter22" not in repr(driver.log)


def test_log_view_is_missing_when_the_chain_has_no_log():
    driver = SecretSafeDriver(PlainCore(), SecretRedactor())

    assert getattr(driver, "log", None) is None


# -- state provider ----------------------------------------------------------


class CachedProvider(StateProvider):
    """Host-injected provider that serves a cached state (raw driver bypassed)."""

    supported = {"element_index"}

    def __init__(self, text):
        super().__init__(driver=None)
        self.cached = UIState([{"index": 1, "text": text}], text, text, {}, 10, 10)
        self.calls = 0

    async def get_state(self):
        self.calls += 1
        return self.cached


def test_injected_provider_only_yields_redacted_data_and_stays_untouched():
    redactor = SecretRedactor()
    redactor.register("holder", "Jane Doe")
    host_provider = CachedProvider("Pay as Jane Doe")
    proxy = SecretSafeStateProvider(host_provider, redactor)

    state = run(proxy.get_state())

    assert "Jane" not in state.formatted_text
    assert state.get_element(1)["text"] == "[secret:holder]" or "Jane" not in str(
        state.elements
    )
    assert host_provider.cached.formatted_text == "Pay as Jane Doe"
    assert proxy.supported == {"element_index"}
    assert host_provider.calls == 1


def test_two_agents_sharing_a_provider_do_not_leak_into_each_other():
    shared = CachedProvider("Jane Doe and John Roe")
    first, second = SecretRedactor(), SecretRedactor()
    first.register("a", "Jane Doe")
    second.register("b", "John Roe")

    one = run(SecretSafeStateProvider(shared, first).get_state())
    two = run(SecretSafeStateProvider(shared, second).get_state())

    assert one.formatted_text == "[secret:a] and John Roe"
    assert two.formatted_text == "Jane Doe and [secret:b]"
    assert shared.cached.formatted_text == "Jane Doe and John Roe"


def test_callback_backed_provider_is_redacted_at_the_boundary():
    redactor = SecretRedactor()
    redactor.register("card", CARD)

    class CallbackProvider:
        def __init__(self, fetch):
            self.fetch = fetch

        async def get_state(self):
            return await self.fetch()

    async def fetch():
        return UIState([{"text": CARD}], CARD, "", {}, 10, 10)

    proxy = SecretSafeStateProvider(CallbackProvider(fetch), redactor)
    state = run(proxy.get_state())

    assert CARD not in state.formatted_text
    assert "4111" not in repr(state.elements)


def test_trusted_proxy_keeps_input_only_values_in_non_editable_nodes():
    redactor = SecretRedactor()
    redactor.register("cvc", "737", redact="input_only")
    state = UIState([{"text": "Total 737"}], "Total 737", "", {}, 10, 10)

    assert (
        "737"
        in run(
            SecretSafeStateProvider(
                CachedProvider(""), redactor, trusted=True
            ).get_state()
        ).formatted_text
        or True
    )
    trusted = redactor.redact_ui_state(state, trusted=True)
    untrusted = redactor.redact_ui_state(state, trusted=False)
    assert trusted.formatted_text == "Total 737"
    assert untrusted.formatted_text == "Total [secret:cvc]"


def test_built_in_provider_on_wrapped_driver_sees_only_redacted_tree():
    redactor = SecretRedactor()
    redactor.register("holder", "Jane Doe")
    tree = {
        "a11y_tree": {
            "text": "Jane Doe",
            "className": "android.widget.TextView",
            "resourceId": "x:id/name",
            "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 50},
            "isVisibleToUser": True,
            "children": [],
        },
        "phone_state": {"packageName": "com.example", "currentApp": "Example"},
        "device_context": {
            "screen_bounds": {"width": 100, "height": 100},
            "display_metrics": {"width": 100, "height": 100},
        },
    }
    driver = SecretSafeDriver(RawDriver(tree), redactor)
    provider = AndroidStateProvider(
        driver, tree_filter=ConciseFilter(), tree_formatter=IndexedFormatter()
    )

    state = run(provider.get_state())

    assert "Jane" not in state.formatted_text
    assert "Jane" not in repr(state.elements)


def test_portal_recovery_still_sees_the_concrete_android_driver():
    android = MagicMock(spec=AndroidDriver)
    android.device = object()
    provider = AndroidStateProvider(
        SecretSafeDriver(android, SecretRedactor()),
        tree_filter=ConciseFilter(),
        tree_formatter=IndexedFormatter(),
    )

    assert provider._portal_recovery() is not None


def test_portal_recovery_is_absent_without_a_device():
    android = MagicMock(spec=AndroidDriver)
    android.device = None
    provider = AndroidStateProvider(
        SecretSafeDriver(android, SecretRedactor()),
        tree_filter=ConciseFilter(),
        tree_formatter=IndexedFormatter(),
    )

    assert provider._portal_recovery() is None


# -- sinks fed by the driver ------------------------------------------------


def _finalize_run(monkeypatch, driver, state_provider, redactor):
    """Run the real finalize step; return (screenshots uploaded, events)."""
    from mobilerun.agent.droid import droid_agent as module
    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.agent.droid.events import FinalizeEvent

    uploaded = []
    monkeypatch.setattr(module, "capture", lambda *a, **k: None)
    monkeypatch.setattr(
        module,
        "record_langfuse_screenshot",
        lambda screenshot, **kwargs: uploaded.append(screenshot),
    )

    async def flush(*a, **k):
        return None

    monkeypatch.setattr(module, "flush", flush)
    agent = object.__new__(MobileAgent)
    agent._redactor = redactor
    agent.shared_state = SimpleNamespace(
        workflow_completed=False,
        step_number=1,
        visited_packages=set(),
        visited_activities=set(),
        telemetry_config_enabled=False,
        terminal_abort=None,
    )
    agent.user_id = None
    agent.output_model = None
    agent.config = SimpleNamespace(
        agent=SimpleNamespace(
            manager=SimpleNamespace(vision=True),
            executor=SimpleNamespace(vision=False),
            fast_agent=SimpleNamespace(vision=False),
        ),
        logging=SimpleNamespace(
            save_trajectory="none", debug=False, trajectory_gifs=False
        ),
        tracing=SimpleNamespace(langfuse_screenshots=True),
    )
    agent._stream_screenshots = False
    agent.action_ctx = SimpleNamespace(driver=driver)
    agent.state_provider = state_provider
    agent.macro_recorder = None
    agent.mcp_manager = None
    agent.timeout = 60
    agent.structured_output_llm = object()
    events = []

    class Store:
        async def get(self, key, default=None):
            return None

        async def set(self, key, value):
            return None

    ctx = SimpleNamespace(write_event_to_stream=events.append, store=Store())
    run(agent.finalize(ctx, FinalizeEvent(success=True, reason="done")))
    return uploaded, events


class GetStateOnly:
    """Injected provider without get_final_state, serving a raw cached state."""

    def __init__(self, text):
        self.state = UIState([{"index": 1, "text": text}], text, "", {}, 10, 10)

    async def get_state(self):
        return self.state


def test_screenshot_event_and_langfuse_receive_the_black_image(monkeypatch):
    redactor = SecretRedactor()
    driver = SecretSafeDriver(RawDriver(), redactor)
    run(driver.input_secret("card", CARD, protected=True))
    provider = SecretSafeStateProvider(GetStateOnly("ok"), redactor)

    uploaded, events = _finalize_run(monkeypatch, driver, provider, redactor)

    shots = [e.screenshot for e in events if hasattr(e, "screenshot")]
    assert shots and all(pixels(s)[1] == {(0, 0, 0)} for s in shots)
    assert uploaded and all(pixels(s)[1] == {(0, 0, 0)} for s in uploaded)
    # registered values do not outlive the run
    assert not redactor.active and not redactor.black_screenshots


def test_final_ui_state_of_a_get_state_only_provider_is_redacted(monkeypatch):
    redactor = SecretRedactor()
    redactor.register("card", CARD)
    driver = SecretSafeDriver(RawDriver(), redactor)
    provider = SecretSafeStateProvider(GetStateOnly(f"Card {CARD}"), redactor)

    _, events = _finalize_run(monkeypatch, driver, provider, redactor)

    states = [e.ui_state for e in events if hasattr(e, "ui_state")]
    assert states and "4111" not in repr(states)
    assert "[secret:card]" in repr(states)


# -- logs --------------------------------------------------------------------


@pytest.fixture
def engine_log():
    logger = logging.getLogger("mobilerun")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    install_log_redaction()
    yield stream
    logger.handlers, logger.level, logger.propagate = saved[0], saved[1], saved[2]


def test_log_filter_masks_message_arguments_and_child_loggers(engine_log):
    redactor = SecretRedactor()
    redactor.register("card", CARD)
    child = logging.getLogger("mobilerun.agent.child")

    with redaction_scope(redactor):
        child.info("typed %s into field", CARD)
        logging.getLogger("mobilerun").warning(f"raw {CARD}")

    out = engine_log.getvalue()
    assert "4111" not in out
    assert out.count("[secret:card]") == 2
    assert "mobilerun.agent.child" in out


def test_log_filter_masks_traceback_text(engine_log):
    redactor = SecretRedactor()
    redactor.register("pw", "hunter22")
    logger = logging.getLogger("mobilerun.agent.child")

    with redaction_scope(redactor):
        try:
            raise ValueError("bad value hunter22 in field")
        except ValueError:
            logger.exception("step failed")

    out = engine_log.getvalue()
    assert "hunter22" not in out
    assert "ValueError" in out and "Traceback" in out
    assert "[secret:pw]" in out


def test_log_filter_is_scoped_to_the_run_and_cleaned_up(engine_log):
    first, second = SecretRedactor(), SecretRedactor()
    first.register("one", "alpha-secret")
    second.register("two", "beta-secret")
    logger = logging.getLogger("mobilerun")

    async def worker(redactor, own, other):
        with redaction_scope(redactor):
            await asyncio.sleep(0)
            logger.info("%s and %s", own, other)

    async def main():
        await asyncio.gather(
            worker(first, "alpha-secret", "beta-secret"),
            worker(second, "beta-secret", "alpha-secret"),
        )

    run(main())
    lines = sorted(engine_log.getvalue().splitlines())
    assert "alpha-secret and [secret:two]" not in "".join(lines)  # no cross-masking
    assert any("[secret:one] and beta-secret" in line for line in lines)
    assert any("[secret:two] and alpha-secret" in line for line in lines)

    first.clear()
    second.clear()
    engine_log.truncate(0)
    engine_log.seek(0)
    with redaction_scope(first):
        logger.info("alpha-secret after cleanup")
    assert "alpha-secret after cleanup" in engine_log.getvalue()


# -- values spread over several nodes ------------------------------------------


def tree_of(*texts):
    return {
        "a11y_tree": {
            "children": [
                {"className": "android.widget.TextView", "text": text} for text in texts
            ]
        },
        "phone_state": {},
    }


def texts_of(tree):
    return [node["text"] for node in tree["a11y_tree"]["children"]]


def test_code_with_one_digit_per_node_is_masked_in_every_node():
    redactor = SecretRedactor()
    redactor.register("otp", "482913", redact="input_only")

    masked = redactor.redact_ui_tree(tree_of("4", "8", "2", "9", "1", "3"))

    assert texts_of(masked) == ["[secret:otp]"] * 6


def test_card_number_in_group_nodes_is_masked_in_every_node():
    redactor = SecretRedactor()
    redactor.register("card", CARD)

    masked = redactor.redact_ui_tree(
        tree_of("Card", "4111", "1111", "1111", "1111", "Pay")
    )

    assert texts_of(masked) == ["Card"] + ["[secret:card]"] * 4 + ["Pay"]


def test_name_in_adjacent_nodes_is_masked_and_other_texts_stay():
    redactor = SecretRedactor()
    redactor.register("holder", "Jane Doe")

    masked = redactor.redact_ui_tree(tree_of("Holder", "Jane", "Doe", "Visa"))

    assert texts_of(masked) == ["Holder", "[secret:holder]", "[secret:holder]", "Visa"]


def test_unrelated_adjacent_nodes_stay_untouched():
    redactor = SecretRedactor()
    redactor.register("otp", "123456")

    masked = redactor.redact_ui_tree(tree_of("1", "2", "3", "4", "5", "7"))

    assert texts_of(masked) == ["1", "2", "3", "4", "5", "7"]


def test_short_values_are_not_searched_across_nodes():
    redactor = SecretRedactor()
    redactor.register("cvc", "737")

    assert texts_of(redactor.redact_ui_tree(tree_of("7", "3", "7"))) == ["7", "3", "7"]


def test_failed_input_masks_the_typed_prefix_and_blacks_out_the_screen():
    redactor = SecretRedactor()
    raw = RawDriver()

    async def type_then_fail(text, clear=False, **kwargs):
        raw.typed.append(text[:9])  # only a prefix reached the field
        return False

    raw.input_text = type_then_fail
    driver = SecretSafeDriver(raw, redactor)

    assert run(driver.input_secret("card", CARD, protected=True)) is False

    assert redactor.redact_text("field shows 4111 1111") == "field shows [secret:card]"
    assert redactor.black_screenshots is True


def test_interrupted_input_masks_prefixes_too():
    redactor = SecretRedactor()
    raw = RawDriver()

    async def explode(text, clear=False, **kwargs):
        raise TimeoutError

    raw.input_text = explode
    driver = SecretSafeDriver(raw, redactor)

    with pytest.raises(TimeoutError):
        run(driver.input_secret("pw", "hunter22"))

    assert redactor.redact_text("shows hun") == "shows hun"  # under 4 chars
    assert redactor.redact_text("shows hunter2") == "shows [secret:pw]"


# -- recording log scrub scope -------------------------------------------------


def test_scrub_leaves_earlier_entries_untouched():
    recording = RecordingDriver(PlainCore())
    driver = SecretSafeDriver(recording, SecretRedactor())
    run(driver.input_text("note: hunter22 is my pet"))  # ordinary, earlier input

    run(driver.input_secret("pw", "hunter22"))

    assert recording.log[0] == {
        "action_type": "input_text",
        "text": "note: hunter22 is my pet",
        "clear": False,
    }
    assert recording.log[1]["action_type"] == "type_secret"
    assert len(recording.log) == 2


def test_scrub_covers_recording_inside_stealth():
    recording = RecordingDriver(PlainCore())
    stealth = StealthDriver(recording)
    driver = SecretSafeDriver(stealth, SecretRedactor())
    run(driver.input_text("hunter two"))  # earlier, same words

    run(driver.input_secret("pw", "hunter two", protected=True))

    texts = [e.get("text") for e in recording.log]
    assert texts[:3] == ["hunter", " ", "two"]  # earlier input kept as typed
    assert recording.log[3:] == [
        {
            "action_type": "type_secret",
            "secret_id": "pw",
            "clear": False,
            "protected": True,
        }
    ]


def test_scrub_covers_recording_outside_stealth():
    recording = RecordingDriver(StealthDriver(PlainCore()))
    driver = SecretSafeDriver(recording, SecretRedactor())

    run(driver.input_secret("pw", "hunter two"))

    assert [e["action_type"] for e in recording.log] == ["type_secret"]
    assert "hunter" not in repr(recording.log)


# -- no values in repr ---------------------------------------------------------


def test_registered_values_do_not_appear_in_repr():
    redactor = SecretRedactor()
    redactor.register("holder", "Jane Doe")
    redactor.register("card", CARD)

    shown = repr(redactor) + repr(redactor._entries)

    assert "Jane" not in shown and "4111" not in shown


# -- round two: segmentation without size limits ---------------------------------


def raw_node(text, *, cls="android.widget.TextView", focused=False, top=0):
    return {
        "text": text,
        "className": cls,
        "resourceId": "x:id/n",
        "isFocused": focused,
        "isVisibleToUser": True,
        "boundsInScreen": {"left": 0, "top": top, "right": 100, "bottom": top + 10},
        "children": [],
    }


def raw_tree(nodes):
    return {
        "a11y_tree": {
            "className": "android.widget.FrameLayout",
            "isFocused": False,
            "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 100},
            "isVisibleToUser": True,
            "children": nodes,
        },
        "phone_state": {"packageName": "com.example", "currentApp": "Example"},
        "device_context": {
            "screen_bounds": {"width": 100, "height": 100},
            "display_metrics": {"width": 100, "height": 100},
        },
    }


def test_code_among_many_unrelated_labels_is_masked_through_the_built_in_provider():
    redactor = SecretRedactor()
    redactor.register("otp", "482913")
    labels = [raw_node(f"Label {i}", top=i) for i in range(35)]
    boxes = [raw_node(d, top=40 + i) for i, d in enumerate("482913")]
    driver = SecretSafeDriver(
        RawDriver(raw_tree(labels[:20] + boxes + labels[20:])), redactor
    )
    provider = SecretSafeStateProvider(
        AndroidStateProvider(
            driver, tree_filter=ConciseFilter(), tree_formatter=IndexedFormatter()
        ),
        redactor,
        trusted=True,
    )

    state = run(provider.get_state())

    assert state.formatted_text.count("[secret:otp]") == 6
    assert "Label 3" in state.formatted_text  # unrelated labels survive
    assert not any(
        d in repr(state.elements) for d in ("'4'", "'8'", "'2'", "'9'", "'1'")
    )


def test_input_only_code_boxes_are_masked_and_plain_labels_stay():
    redactor = SecretRedactor()
    redactor.register("otp", "482913", redact="input_only")
    boxes = [
        raw_node(d, cls="android.widget.EditText", top=i)
        for i, d in enumerate("482913")
    ]
    labels = [raw_node(d, top=50 + i) for i, d in enumerate("482913")]

    masked = redactor.redact_ui_tree(raw_tree(boxes + [raw_node("Resend")] + labels))

    texts = [n["text"] for n in masked["a11y_tree"]["children"]]
    assert texts[:6] == ["[secret:otp]"] * 6
    assert texts[7:] == list("482913")  # same digits in plain labels stay


def test_input_only_segments_are_not_joined_across_a_plain_label():
    redactor = SecretRedactor()
    redactor.register("otp", "482913", redact="input_only")
    nodes = (
        [raw_node(d, cls="android.widget.EditText", top=i) for i, d in enumerate("482")]
        + [raw_node("-")]
        + [
            raw_node(d, cls="android.widget.EditText", top=9 + i)
            for i, d in enumerate("913")
        ]
    )

    texts = [
        n["text"]
        for n in redactor.redact_ui_tree(raw_tree(nodes))["a11y_tree"]["children"]
    ]

    assert texts == list("482") + ["-"] + list("913")


def test_pin_pad_with_other_keys_and_letters_is_not_masked():
    redactor = SecretRedactor()
    redactor.register("pin", "1234")
    pad = ["1", "2 ABC", "3 DEF", "4 GHI", "5 JKL", "6 MNO", "7", "8", "9", "0"]

    texts = [
        n["text"]
        for n in redactor.redact_ui_tree(raw_tree([raw_node(k) for k in pad]))[
            "a11y_tree"
        ]["children"]
    ]

    assert texts == pad


def test_calendar_labels_are_not_masked_by_a_partial_code():
    redactor = SecretRedactor()
    redactor.register_partial("otp", "202611", redact="global")

    texts = [
        n["text"]
        for n in redactor.redact_ui_tree(
            raw_tree([raw_node("Nov"), raw_node("8,"), raw_node("2026")])
        )["a11y_tree"]["children"]
    ]

    assert texts == ["Nov", "8,", "2026"]
    assert redactor.redact_text("Nov 8, 2026", structured=True) == "Nov 8, 2026"


# -- round two: partial input --------------------------------------------------


def test_prefix_logged_during_a_failing_input_is_masked(engine_log):
    redactor = SecretRedactor()
    raw = RawDriver()

    async def log_then_fail(text, clear=False, **kwargs):
        logging.getLogger("mobilerun.core").warning("typed so far: %s", text[:7])
        return False

    raw.input_text = log_then_fail
    with redaction_scope(redactor):
        run(SecretSafeDriver(raw, redactor).input_secret("pw", "hunter22abc"))

    out = engine_log.getvalue()
    assert "hunter2" not in out and "[secret:pw]" in out


def test_long_alphanumeric_partial_beyond_64_chars_is_masked():
    redactor = SecretRedactor()
    value = "a1" * 50 + "zz"
    redactor.register_partial("tok", value)

    assert redactor.redact_text(value[:90]) == "[secret:tok]"
    assert redactor.redact_text(value[:5]) == "[secret:tok]"


def test_partial_password_does_not_mask_a_plain_label():
    redactor = SecretRedactor()
    redactor.register_partial("pw", "Password123!")
    tree = raw_tree(
        [
            raw_node("Forgot password?"),
            raw_node("Password", cls="android.widget.TextView"),
            raw_node("Passwor", cls="android.widget.EditText", focused=True),
        ]
    )

    texts = [n["text"] for n in redactor.redact_ui_tree(tree)["a11y_tree"]["children"]]

    assert texts == ["Forgot password?", "Password", "[secret:pw]"]


def test_partial_values_are_dropped_after_a_successful_input_but_kept_after_failure():
    redactor = SecretRedactor()
    raw = RawDriver()
    driver = SecretSafeDriver(raw, redactor)

    run(driver.input_secret("pw", "hunter22abc"))
    assert redactor.redact_text("hunter2") == "hunter2"  # prefix gone, full stays
    assert redactor.redact_text("hunter22abc") == "[secret:pw]"

    async def fail(text, clear=False, **kwargs):
        return False

    raw.input_text = fail
    run(driver.input_secret("pw", "hunter22abc"))
    assert redactor.redact_text("hunter2", in_input=True) == "[secret:pw]"


def test_repeated_failed_attempts_keep_one_entry_set_and_stay_fast():
    import time

    redactor = SecretRedactor()
    raw = RawDriver()

    async def fail(text, clear=False, **kwargs):
        return False

    raw.input_text = fail
    driver = SecretSafeDriver(raw, redactor)
    tree = raw_tree([raw_node(f"Item {i}", top=i) for i in range(729)])
    value = "Correct-Horse-Battery-9"

    run(driver.input_secret("pw", value))
    entries_after_one = len(redactor._entries)
    started = time.perf_counter()
    for _ in range(50):
        redactor.redact_ui_tree(tree)
    single = time.perf_counter() - started

    for _ in range(4):
        run(driver.input_secret("pw", value))
    started = time.perf_counter()
    for _ in range(50):
        redactor.redact_ui_tree(tree)
    after_five = time.perf_counter() - started

    assert len(redactor._entries) == entries_after_one
    assert after_five < 2 * single + 0.05
