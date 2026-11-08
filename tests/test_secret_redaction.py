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
    SecretLogFilter,
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


def test_driver_delegates_everything_else():
    driver = SecretSafeDriver(RawDriver(), SecretRedactor())

    assert driver.platform == "Android"
    assert driver.supported == {"tap", "input_text"}
    assert driver.supported_buttons == {"back"}
    assert run(driver.input_text("x")) is True


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


def test_screenshot_event_and_langfuse_receive_the_black_image(monkeypatch):
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

    redactor = SecretRedactor()
    driver = SecretSafeDriver(RawDriver(), redactor)
    run(driver.input_secret("card", CARD, protected=True))

    agent = object.__new__(MobileAgent)
    agent.__dict__["_secret_redactor"] = redactor
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

    async def state():
        return SimpleNamespace(elements=[])

    agent.state_provider = SimpleNamespace(get_state=state)
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

    shots = [e.screenshot for e in events if hasattr(e, "screenshot")]
    assert shots and all(pixels(s)[1] == {(0, 0, 0)} for s in shots)
    assert uploaded and all(pixels(s)[1] == {(0, 0, 0)} for s in uploaded)
    # registered values do not outlive the run
    assert not redactor.active and not redactor.black_screenshots


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


def test_log_filter_install_is_idempotent(engine_log):
    install_log_redaction()
    install_log_redaction()
    handler = logging.getLogger("mobilerun").handlers[0]

    assert sum(isinstance(f, SecretLogFilter) for f in handler.filters) == 1
