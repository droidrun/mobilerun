import asyncio
import copy
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from llama_index.core.workflow import StartEvent

import mobilerun.agent.droid.droid_agent as droid_module
import mobilerun.agent.jev.agent as agent_module
from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.common.events import (
    RecordUIStateEvent,
    ScreenshotEvent,
    ToolExecutionEvent,
)
from mobilerun.agent.droid.events import (
    ExternalUserMessageDroppedEvent,
    FastAgentExecuteEvent,
    ResultEvent,
)
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.jev.agent import JevAgent, wait_duration
from mobilerun.agent.jev.client import MISSING_KEY_MESSAGE, JevResponse
from mobilerun.agent.jev.errors import JevAuthError, JevConfigError, JevServiceError
from mobilerun.agent.jev.events import JevDecisionEvent, JevEndEvent, JevStepEvent
from mobilerun.agent.jev.observation import JevIndexedFormatter
from mobilerun.agent.jev.policy import NONE_OPTION, Capabilities
from mobilerun.agent.utils import llm_loader
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.config_manager import env_keys
from mobilerun.config_manager.config_manager import (
    DEFAULT_DISABLED_TOOLS,
    AgentConfig,
    DeviceConfig,
    JevConfig,
    LoggingConfig,
    MobileConfig,
    TelemetryConfig,
)
from mobilerun.macro.recorder import MacroRecorder
from mobilerun.tools.filters import DetailedFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.provider import AndroidStateProvider

KEY = "tsk_test_jevagent_5Hq8Wd3Kp7Rt1Zx9"
GOAL = "Open Maps and search for 'Golden Gate Bridge'"
QUERY = "Golden Gate Bridge"
LAUNCHER = "com.android.launcher3"
MAPS = "com.google.android.apps.maps"
SETTINGS = "com.android.settings"
WIDTH, HEIGHT = 1080, 2400
PNG = b"\x89PNG\r\n\x1a\nfake"
APPS = [
    {"package": MAPS, "label": "Maps"},
    {"package": SETTINGS, "label": "Settings"},
    {"package": LAUNCHER, "label": "Launcher"},
]
TARGET_QUESTIONS = {
    "OPEN_APP": "app_target",
    "TAP": "tap_target",
    "TYPE_TEXT": "text_value",
}


# ── Portal-shaped screens ──────────────────────────────────────────────


def _node(class_name, bounds, children=(), package=MAPS, **flags):
    left, top, right, bottom = bounds
    node = {
        "className": class_name,
        "packageName": package,
        "boundsInScreen": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        },
        "isEnabled": True,
        "isVisibleToUser": True,
        "children": list(children),
    }
    node.update(flags)
    return node


def _phone(app, package, *, editable=False, text=None):
    phone = {
        "currentApp": app,
        "packageName": package,
        "keyboardVisible": editable,
        "isEditable": editable,
    }
    if editable:
        phone["focusedElement"] = {
            "text": text or "",
            "className": "android.widget.EditText",
        }
    return phone


def launcher_screen():
    tree = _node(
        "android.widget.FrameLayout",
        (0, 0, WIDTH, HEIGHT),
        [
            _node(
                "android.widget.TextView",
                (100, 1800, 300, 2000),
                package=LAUNCHER,
                text="Phone",
                isClickable=True,
            )
        ],
        package=LAUNCHER,
    )
    return tree, _phone("Launcher", LAUNCHER)


def maps_screen(*, focused=False, text=""):
    search = _node(
        "android.widget.EditText",
        (40, 150, 1040, 280),
        resourceId=f"{MAPS}:id/search_omnibox_text_box",
        text=text or "Search here",
        hint="Search here",
        isShowingHintText=not text,
        isEditable=True,
        isFocused=focused,
        isClickable=True,
    )
    explore = _node(
        "android.widget.Button",
        (40, 2200, 400, 2350),
        text="Explore",
        isClickable=True,
    )
    tree = _node("android.widget.FrameLayout", (0, 0, WIDTH, HEIGHT), [search, explore])
    return tree, _phone("Maps", MAPS, editable=focused, text=text)


def results_screen():
    row = _node(
        "android.widget.LinearLayout",
        (0, 400, WIDTH, 600),
        [
            _node("android.widget.TextView", (20, 410, 1000, 500), text=QUERY),
            _node(
                "android.widget.TextView", (20, 500, 1000, 590), text="San Francisco"
            ),
        ],
        isClickable=True,
    )
    results = _node(
        "androidx.recyclerview.widget.RecyclerView",
        (0, 300, WIDTH, 2200),
        [row],
        resourceId=f"{MAPS}:id/results",
        isScrollable=True,
    )
    tree = _node("android.widget.FrameLayout", (0, 0, WIDTH, HEIGHT), [results])
    return tree, _phone("Maps", MAPS)


SCREENS = {
    "launcher": launcher_screen,
    "maps": maps_screen,
    "maps_focused": lambda: maps_screen(focused=True),
    "maps_typed": lambda: maps_screen(focused=True, text=QUERY),
    "results": results_screen,
}
SEARCH_FLOW = {
    ("launcher", "start_app"): "maps",
    ("maps", "tap"): "maps_focused",
    ("maps_focused", "input_text"): "maps_typed",
    ("maps_typed", "press_button"): "results",
}


class FakeDriver:
    platform = "Android"
    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_button",
        "start_app",
        "screenshot",
        "get_ui_tree",
        "get_date",
        "get_apps",
    }
    supported_buttons = {"back", "home", "enter"}

    def __init__(self, start="launcher", transitions=None, fail_start_app=False):
        self.screen = start
        self.transitions = SEARCH_FLOW if transitions is None else transitions
        self.fail_start_app = fail_start_app
        self.calls = []
        self.get_apps_calls = 0

    def _advance(self, verb):
        self.screen = self.transitions.get((self.screen, verb), self.screen)

    async def tap(self, x, y):
        self.calls.append(("tap", x, y))
        self._advance("tap")

    async def swipe(self, x1, y1, x2, y2, duration_ms=1000):
        self.calls.append(("swipe", x1, y1, x2, y2, duration_ms))
        self._advance("swipe")

    async def input_text(self, text, clear=False, **kwargs):
        self.calls.append(("input_text", text, clear))
        self._advance("input_text")
        return True

    async def press_button(self, button):
        self.calls.append(("press_button", button))
        self._advance("press_button")

    async def start_app(self, package, activity=None):
        self.calls.append(("start_app", package))
        if self.fail_start_app:
            raise RuntimeError("activity not found")
        self._advance("start_app")
        return f"App started: {package}"

    async def get_apps(self, include_system=True):
        self.get_apps_calls += 1
        return copy.deepcopy(APPS)

    async def get_ui_tree(self):
        tree, phone = SCREENS[self.screen]()
        return {
            "a11y_tree": tree,
            "phone_state": phone,
            "device_context": {"screen_bounds": {"width": WIDTH, "height": HEIGHT}},
        }

    async def get_date(self):
        return "2026-10-15"

    async def screenshot(self, hide_overlay=True):
        return PNG


# ── Scripted Jev answers ───────────────────────────────────────────────


def say(operation, target=None, *, confidence=0.9, target_confidence=0.9):
    return (operation, target, confidence, target_confidence)


def _find_key(criteria, target):
    if target in criteria:
        return target
    for key, label in criteria.items():
        if label == target:
            return key
    for key, label in criteria.items():
        if target in label:
            return key
    raise AssertionError(f"{target!r} was not offered: {criteria}")


def _choice(criteria, key, confidence):
    keys = list(criteria)
    if len(keys) == 1:
        probabilities = {key: 1.0}
    else:
        rest = 0.1 / (len(keys) - 1)
        probabilities = {k: 0.9 if k == key else rest for k in keys}
    return {
        "type": "choice",
        "choice": key,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def answers_for(questions, step):
    operation, target, confidence, target_confidence = step
    criteria = questions["operation"]["criteria"]
    answers = {
        "operation": _choice(criteria, _find_key(criteria, operation), confidence)
    }
    question_id = TARGET_QUESTIONS.get(operation) or (
        "scroll_target" if operation.startswith("SCROLL_") else None
    )
    if question_id:
        target_criteria = questions[question_id]["criteria"]
        answers[question_id] = _choice(
            target_criteria, _find_key(target_criteria, target), target_confidence
        )
    return answers


class FakeJev:
    model = "jev-latest"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.list_models_calls = 0
        self.models_error = None
        self.closed = False

    async def systemone(self, state, questions):
        self.requests.append((copy.deepcopy(state), copy.deepcopy(questions)))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return JevResponse(
            answers=answers_for(questions, step),
            model="jev-1.13.0",
            usage={"input_tokens": 120, "output_tokens": 4},
            request_id=f"req_{len(self.requests)}",
            latency_ms=15,
        )

    async def list_models(self):
        self.list_models_calls += 1
        if self.models_error is not None:
            raise self.models_error
        return ["jev-latest", "jev-preview"]

    async def aclose(self):
        self.closed = True


HAPPY_SCRIPT = [
    say("OPEN_APP", "Maps"),
    say("TAP", "Search here"),
    say("TYPE_TEXT", QUERY),
    say("ENTER"),
    say("DONE"),
]


# ── Fixtures and helpers ───────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    for name in ("TYPESAFE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "false")
    monkeypatch.delenv("MOBILERUN_STREAM_SCREENSHOTS", raising=False)
    monkeypatch.delenv("DROIDRUN_STREAM_SCREENSHOTS", raising=False)


@pytest.fixture
def clock(monkeypatch):
    """Instant asyncio.sleep that advances a fake monotonic clock."""
    real_sleep = asyncio.sleep
    state = SimpleNamespace(now=1000.0, sleeps=[])

    async def fake_sleep(delay, result=None):
        if delay:
            state.sleeps.append(delay)
            state.now += delay
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        agent_module, "time", SimpleNamespace(monotonic=lambda: state.now)
    )
    return state


@pytest.fixture
def mobilerun_logs(caplog):
    mobilerun_logger = logging.getLogger("mobilerun")
    previous = mobilerun_logger.propagate
    mobilerun_logger.propagate = True
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    yield caplog
    mobilerun_logger.propagate = previous


async def _run_jev(
    driver,
    client,
    *,
    goal=GOAL,
    max_steps=15,
    min_confidence=0.3,
    after_sleep_action=0.0,
    input_texts=(),
    save_trajectory="none",
    platform="android",
):
    shared_state = MobileAgentState(instruction=goal, telemetry_config_enabled=False)
    recorder = MacroRecorder()
    provider = AndroidStateProvider(
        driver, tree_filter=DetailedFilter(), tree_formatter=JevIndexedFormatter()
    )
    registry, _ = await build_tool_registry(
        supported_buttons=driver.supported_buttons,
        platform=platform,
        exact_app_launch=True,
    )
    registry.disable_unsupported(driver.supported | provider.supported)
    registry.disable(list(DEFAULT_DISABLED_TOOLS))
    action_ctx = ActionContext(
        driver=driver,
        ui=None,
        shared_state=shared_state,
        state_provider=provider,
        macro_recorder=recorder,
    )
    agent_config = AgentConfig(
        max_steps=max_steps,
        after_sleep_action=after_sleep_action,
        jev=JevConfig(min_confidence=min_confidence),
    )
    agent = JevAgent(
        client,
        agent_config,
        registry,
        action_ctx,
        provider,
        shared_state,
        platform=platform,
        save_trajectory=save_trajectory,
        input_texts=input_texts,
        timeout=60,
    )
    handler = agent.run(input=goal)
    events = [event async for event in handler.stream_events()]
    result = await handler
    return SimpleNamespace(
        result=result, events=events, recorder=recorder, shared_state=shared_state
    )


def _of(events, event_type):
    return [event for event in events if isinstance(event, event_type)]


# ── JevAgent loop ──────────────────────────────────────────────────────


def test_happy_path_opens_app_taps_types_and_submits(clock):
    driver = FakeDriver()
    client = FakeJev(HAPPY_SCRIPT)

    run = asyncio.run(_run_jev(driver, client, after_sleep_action=0.25))

    assert run.result["success"] is True
    assert run.result["reason"].startswith("Jev reports the goal is complete")
    assert run.result["steps"] == 4
    assert run.shared_state.step_number == 4
    assert driver.calls == [
        ("start_app", MAPS),
        ("tap", 540, 215),
        ("input_text", QUERY, True),
        ("press_button", "enter"),
    ]
    assert driver.get_apps_calls == 1

    actions = run.recorder.actions
    assert [a["action_type"] for a in actions] == [
        "start_app",
        "tap",
        "input_text",
        "button_press",
    ]
    assert actions[0]["package"] == MAPS
    assert (actions[1]["x"], actions[1]["y"]) == (540, 215)
    assert (actions[2]["text"], actions[2]["clear"]) == (QUERY, True)
    assert actions[3]["button"] == "enter"
    assert all(action["pre_state"]["nodes"] for action in actions)
    # Macro snapshots never carry Jev's namespaced flags.
    assert "jev" not in json.dumps([a["pre_state"] for a in actions])

    decisions = _of(run.events, JevDecisionEvent)
    assert [d.status for d in decisions] == ["action"] * 4 + ["done"]
    assert [d.operation for d in decisions] == [
        "OPEN_APP",
        "TAP",
        "TYPE_TEXT",
        "ENTER",
        "DONE",
    ]
    assert [d.step for d in decisions] == [1, 2, 3, 4, 5]
    assert decisions[0].label == f"Maps ({MAPS})"
    assert decisions[0].input_tokens == 120
    steps = _of(run.events, JevStepEvent)
    assert [(s.step, s.success) for s in steps] == [(n, True) for n in (1, 2, 3, 4)]
    assert [e.tool_name for e in _of(run.events, ToolExecutionEvent)] == [
        "open_app",
        "click",
        "type",
        "system_button",
    ]
    end = _of(run.events, JevEndEvent)
    assert len(end) == 1 and end[0].success and end[0].steps == 4
    assert len(_of(run.events, RecordUIStateEvent)) == 5
    assert not _of(run.events, ScreenshotEvent)
    assert clock.sleeps.count(0.25) == 4

    # Apps are narrowed to the one the goal names; the foreground app is excluded.
    first_state, first_questions = client.requests[0]
    assert first_state["availableApps"] == [
        {"index": "1", "label": "Maps", "package": MAPS}
    ]
    assert "ENTER" not in first_questions["operation"]["criteria"]
    second_state = client.requests[1][0]
    assert second_state["recentActions"] == [
        {
            "operation": "OPEN_APP",
            "label": f"Maps ({MAPS})",
            "screenChanged": True,
            "succeeded": True,
        }
    ]
    typing_state, typing_questions = client.requests[2]
    # Focusing the field changes the screen even though no text moved.
    assert typing_state["recentActions"][1]["screenChanged"] is True
    assert typing_state["focusedField"]["label"] == "Search here"
    assert {"TYPE_TEXT", "ENTER"} <= set(typing_questions["operation"]["criteria"])
    last_actions = client.requests[4][0]["recentActions"]
    assert last_actions[2] == {
        "operation": "TYPE_TEXT",
        "label": last_actions[2]["label"],
        "text": QUERY,
        "screenChanged": True,
        "succeeded": True,
    }
    assert last_actions[2]["label"].endswith("Search here")
    assert run.shared_state.current_package_name == MAPS


def test_screenshots_are_streamed_when_saving_trajectories(clock):
    run = asyncio.run(
        _run_jev(FakeDriver(), FakeJev([say("DONE")]), save_trajectory="action")
    )

    assert [e.screenshot for e in _of(run.events, ScreenshotEvent)] == [PNG]


def test_done_after_the_last_allowed_action_succeeds(clock):
    driver = FakeDriver()
    client = FakeJev([say("OPEN_APP", "Maps"), say("TAP", "Search here"), say("DONE")])

    run = asyncio.run(_run_jev(driver, client, max_steps=2))

    assert run.result["success"] is True
    assert run.result["steps"] == 2
    assert len(driver.calls) == 2


def test_step_limit_fails_when_another_action_is_requested(clock):
    driver = FakeDriver()
    client = FakeJev(
        [say("OPEN_APP", "Maps"), say("TAP", "Search here"), say("TYPE_TEXT", QUERY)]
    )

    run = asyncio.run(_run_jev(driver, client, max_steps=2))

    assert run.result == {
        "success": False,
        "reason": "Reached max step count of 2 steps",
        "steps": 2,
    }
    assert [call[0] for call in driver.calls] == ["start_app", "tap"]
    assert len(client.requests) == 3


@pytest.mark.parametrize(
    "step",
    [
        say("OPEN_APP", "Maps", confidence=0.1),
        say("OPEN_APP", "Maps", target_confidence=0.1),
    ],
)
def test_low_confidence_stops_before_any_action(clock, step):
    driver = FakeDriver()

    run = asyncio.run(_run_jev(driver, FakeJev([step]), min_confidence=0.3))

    assert run.result["success"] is False
    assert run.result["reason"].startswith("Stopped: Jev's confidence in OPEN_APP")
    assert "below the 0.30 threshold" in run.result["reason"]
    assert run.result["steps"] == 0
    assert driver.calls == []
    assert run.recorder.actions == []
    assert _of(run.events, JevDecisionEvent)[0].status == "uncertain"


def test_zero_min_confidence_disables_the_threshold(clock):
    driver = FakeDriver(transitions={})
    client = FakeJev([say("OPEN_APP", "Maps", confidence=0.01), say("DONE")])

    run = asyncio.run(_run_jev(driver, client, min_confidence=0))

    assert run.result["success"] is True
    assert driver.calls == [("start_app", MAPS)]


def test_blocked_stops_without_acting(clock):
    driver = FakeDriver()

    run = asyncio.run(_run_jev(driver, FakeJev([say("BLOCKED")])))

    assert run.result["success"] is False
    assert run.result["reason"].startswith("Jev found no operation")
    assert driver.calls == []


def test_needs_input_when_no_text_fits(clock):
    driver = FakeDriver(start="maps_focused")
    client = FakeJev([say("TYPE_TEXT", NONE_OPTION)])

    run = asyncio.run(_run_jev(driver, client, goal="Search Maps for a place"))

    assert run.result["success"] is False
    assert "--var NAME=VALUE" in run.result["reason"]
    assert _of(run.events, JevDecisionEvent)[0].status == "needs_input"
    assert driver.calls == []


def test_input_texts_replace_goal_spans(clock):
    driver = FakeDriver(start="maps_focused")
    client = FakeJev([say("TYPE_TEXT", QUERY), say("DONE")])

    run = asyncio.run(
        _run_jev(driver, client, goal="Search Maps for the bridge", input_texts=[QUERY])
    )

    assert run.result["success"] is True
    state, questions = client.requests[0]
    assert state["textSource"] == "supplied"
    assert questions["text_value"]["criteria"]["1"] == QUERY
    assert set(questions["text_value"]["criteria"]) == {"1", NONE_OPTION}
    assert driver.calls == [("input_text", QUERY, True)]


def test_repeating_an_action_on_an_unchanged_screen_stops(clock):
    driver = FakeDriver(start="maps", transitions={})
    client = FakeJev([say("TAP", "Explore")] * 3)

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is False
    assert run.result["reason"].startswith("Stopped: Jev repeated TAP [")
    assert run.result["reason"].endswith("Explore on an unchanged screen.")
    # One repeat is allowed; the third identical attempt stops the run.
    assert run.result["steps"] == 2
    assert [call[0] for call in driver.calls] == ["tap", "tap"]
    assert client.requests[1][0]["recentActions"][0]["screenChanged"] is False
    # The unchanged screen is read again before Jev is asked.
    assert clock.sleeps.count(2.0) == 2


def test_focus_tap_counts_as_a_screen_change(clock):
    driver = FakeDriver(start="maps", transitions={("maps", "tap"): "maps_focused"})
    client = FakeJev([say("TAP", "Search here"), say("DONE")])

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is True
    assert client.requests[1][0]["recentActions"][0]["screenChanged"] is True


def test_back_that_only_hides_the_keyboard_is_not_a_repeat(clock, monkeypatch):
    def keyboard(visible):
        def make():
            tree, phone = maps_screen(focused=True)
            # The Android tree never includes the IME window itself.
            phone["keyboardVisible"] = visible
            return tree, phone

        return make

    monkeypatch.setitem(SCREENS, "keyboard_open", keyboard(True))
    monkeypatch.setitem(SCREENS, "keyboard_hidden", keyboard(False))
    driver = FakeDriver(
        start="keyboard_open",
        transitions={
            ("keyboard_open", "press_button"): "keyboard_hidden",
            ("keyboard_hidden", "press_button"): "launcher",
        },
    )
    client = FakeJev([say("BACK"), say("BACK"), say("DONE")])

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is True
    assert driver.calls == [("press_button", "back")] * 2
    assert client.requests[1][0]["recentActions"][0]["screenChanged"] is True


def test_repeated_type_text_reason_names_the_field_not_the_value(clock):
    canary = "OTP-918273-CANARY"
    driver = FakeDriver(start="maps_focused", transitions={})
    client = FakeJev([say("TYPE_TEXT", canary)] * 3)

    run = asyncio.run(
        _run_jev(driver, client, goal="Enter the code", input_texts=[canary])
    )

    assert run.result["success"] is False
    assert run.result["reason"] == (
        "Stopped: Jev repeated TYPE_TEXT [2] Search here on an unchanged screen."
    )
    assert canary not in _of(run.events, JevEndEvent)[0].reason
    assert driver.calls == [("input_text", canary, True)] * 2


def _wda_node(class_name, bounds, children=(), **fields):
    # iOS --local over the WDA fork: prefixed classes, no input focus anywhere.
    return _node(
        f"XCUIElementType{class_name}",
        bounds,
        children,
        package="com.example.shop",
        isFocused=False,
        **fields,
    )


def _login_screen(*, keyboard, email=""):
    fields = [
        _wda_node(
            "TextField",
            (40, 400, 1040, 520),
            resourceId="email",
            text=email,
            contentDescription="Email",
        ),
        _wda_node(
            "SecureTextField",
            (40, 560, 1040, 680),
            resourceId="password",
            contentDescription="Password",
        ),
        _wda_node("Button", (40, 720, 1040, 840), text="Log In", isClickable=True),
    ]
    if keyboard:
        fields.append(
            _wda_node(
                "Keyboard",
                (0, 1600, WIDTH, HEIGHT),
                [_wda_node("Key", (0, 1700, 100, 1800), contentDescription="q")],
            )
        )
    tree = _wda_node("Application", (0, 0, WIDTH, HEIGHT), fields)
    phone = {
        "currentApp": "Shop",
        "packageName": "com.example.shop",
        "keyboardVisible": keyboard,
        "isEditable": False,
    }
    return tree, phone


@pytest.fixture
def login_screens(monkeypatch):
    screens = {
        "login": lambda: _login_screen(keyboard=False),
        "login_keyboard": lambda: _login_screen(keyboard=True),
        "login_typed": lambda: _login_screen(keyboard=True, email="me@example.com"),
    }
    for name, make in screens.items():
        monkeypatch.setitem(SCREENS, name, make)
    return {
        ("login", "tap"): "login_keyboard",
        ("login_keyboard", "input_text"): "login_typed",
    }


def test_tapped_field_is_typed_into_when_focus_is_not_reported(clock, login_screens):
    driver = FakeDriver(start="login", transitions=login_screens)
    client = FakeJev(
        [say("TAP", "Email"), say("TYPE_TEXT", "me@example.com"), say("DONE")]
    )

    run = asyncio.run(
        _run_jev(driver, client, goal="Log in as 'me@example.com'", platform="ios")
    )

    assert run.result["success"] is True
    assert [call[0] for call in driver.calls] == ["tap", "input_text"]
    assert driver.calls[1] == ("input_text", "me@example.com", True)
    assert client.requests[0][0]["focusedField"] is None
    typing_state, typing_questions = client.requests[1]
    assert typing_state["focusedField"]["label"] == "Email"
    assert "TYPE_TEXT" in typing_questions["operation"]["criteria"]
    # Typing keeps the tapped field focused.
    assert client.requests[2][0]["focusedField"]["index"] == (
        typing_state["focusedField"]["index"]
    )


def test_tapped_field_is_forgotten_after_another_action(clock, login_screens):
    driver = FakeDriver(start="login", transitions=login_screens)
    client = FakeJev([say("TAP", "Email"), say("ENTER"), say("DONE")])

    run = asyncio.run(
        _run_jev(driver, client, goal="Log in as 'me@example.com'", platform="ios")
    )

    assert run.result["success"] is True
    assert client.requests[1][0]["focusedField"]["label"] == "Email"
    after_enter, questions = client.requests[2]
    assert after_enter["focusedField"] is None
    assert "TYPE_TEXT" not in questions["operation"]["criteria"]


def _signup_screen(*, first=""):
    fields = [
        _wda_node(
            "TextField",
            (40, 400, 1040, 520),
            resourceId="first",
            text=first,
            contentDescription="First name",
        ),
        _wda_node(
            "TextField",
            (40, 560, 1040, 680),
            resourceId="last",
            contentDescription="Last name",
        ),
        _wda_node(
            "Keyboard",
            (0, 1600, WIDTH, HEIGHT),
            [_wda_node("Key", (0, 1700, 100, 1800), contentDescription="q")],
        ),
    ]
    tree = _wda_node("Application", (0, 0, WIDTH, HEIGHT), fields)
    phone = {
        "currentApp": "Shop",
        "packageName": "com.example.shop",
        "keyboardVisible": True,
        "isEditable": False,
    }
    return tree, phone


def test_moving_focus_to_another_field_counts_as_a_screen_change(clock, monkeypatch):
    monkeypatch.setitem(SCREENS, "signup", _signup_screen)
    monkeypatch.setitem(SCREENS, "signup_typed", lambda: _signup_screen(first="Ann"))
    driver = FakeDriver(
        start="signup", transitions={("signup", "input_text"): "signup_typed"}
    )
    client = FakeJev(
        [
            say("TAP", "First name"),
            say("TYPE_TEXT", "Ann"),
            say("TAP", "Last name"),
            say("DONE"),
        ]
    )

    run = asyncio.run(
        _run_jev(driver, client, goal="Enter 'Ann' and 'Lee'", platform="ios")
    )

    assert run.result["success"] is True
    # The keyboard stays up and no element reports focus, yet no tap looks a no-op.
    assert clock.sleeps == []
    last_state = client.requests[3][0]
    assert last_state["focusedField"]["label"] == "Last name"
    assert [a["screenChanged"] for a in last_state["recentActions"]] == [True] * 3


def test_failed_action_stops_without_retry(clock):
    driver = FakeDriver(fail_start_app=True)
    client = FakeJev([say("OPEN_APP", "Maps"), say("OPEN_APP", "Maps")])

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is False
    assert run.result["reason"].startswith("Action OPEN_APP failed: Failed to open app")
    assert run.result["steps"] == 1
    assert driver.calls == [("start_app", MAPS)]
    assert len(client.requests) == 1
    steps = _of(run.events, JevStepEvent)
    assert [(s.step, s.success) for s in steps] == [(1, False)]


def test_wait_backs_off_and_times_out_while_loading(clock):
    driver = FakeDriver(transitions={})
    client = FakeJev([say("WAIT")] * 12)

    run = asyncio.run(_run_jev(driver, client, max_steps=30, after_sleep_action=5))

    expected = [0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5]
    assert run.result["success"] is False
    assert run.result["reason"] == "Stopped: the screen kept loading for 15s."
    assert run.result["steps"] == len(expected)
    assert [a["duration"] for a in run.recorder.actions] == expected
    # WAIT sleeps only its own duration, not after_sleep_action.
    assert clock.sleeps == expected
    assert len(client.requests) == len(expected) + 1


def test_wait_backoff_resets_after_another_action(clock):
    driver = FakeDriver(start="maps", transitions={("maps", "tap"): "maps_focused"})
    client = FakeJev(
        [
            say("WAIT"),
            say("WAIT"),
            say("TAP", "Search here"),
            say("WAIT"),
            say("DONE"),
        ]
    )

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is True
    waits = [a["duration"] for a in run.recorder.actions if a["action_type"] == "wait"]
    assert waits == [0.5, 1.0, 0.5]
    assert [wait_duration(n) for n in range(5)] == [0.5, 1.0, 2.0, 2.0, 2.0]


def test_jev_error_from_the_client_propagates(clock):
    driver = FakeDriver()
    error = JevServiceError("TypeSafe is overloaded (HTTP 529) after 3 attempts.")

    with pytest.raises(JevServiceError):
        asyncio.run(_run_jev(driver, FakeJev([error])))

    assert driver.calls == []


def test_external_user_messages_are_dropped(clock):
    async def run():
        driver = FakeDriver()
        client = FakeJev([say("DONE")])
        shared_state = MobileAgentState(
            instruction=GOAL, telemetry_config_enabled=False
        )
        shared_state.queue_user_message("also open settings")
        provider = AndroidStateProvider(
            driver, tree_filter=DetailedFilter(), tree_formatter=JevIndexedFormatter()
        )
        registry, _ = await build_tool_registry(
            supported_buttons=driver.supported_buttons, exact_app_launch=True
        )
        agent = JevAgent(
            client,
            AgentConfig(after_sleep_action=0),
            registry,
            ActionContext(driver, None, shared_state, provider),
            provider,
            shared_state,
            platform="android",
        )
        handler = agent.run(input=GOAL)
        events = [event async for event in handler.stream_events()]
        await handler
        return events, shared_state

    events, shared_state = asyncio.run(run())

    dropped = _of(events, ExternalUserMessageDroppedEvent)
    assert len(dropped) == 1 and dropped[0].reason == "jev_agent"
    assert shared_state.pending_user_messages == []


@pytest.mark.parametrize(
    ("platform", "buttons", "exact", "expected"),
    [
        (
            "android",
            {"back", "home", "enter"},
            True,
            Capabilities(buttons=frozenset({"back", "home", "enter"})),
        ),
        ("ios", {"home"}, True, Capabilities(buttons=frozenset({"home"}))),
        (
            "android",
            {"back", "home", "enter"},
            False,
            Capabilities(open_app=False, buttons=frozenset({"back", "home", "enter"})),
        ),
    ],
)
def test_capabilities_follow_the_registry_and_driver(
    platform, buttons, exact, expected
):
    registry, _ = asyncio.run(
        build_tool_registry(
            supported_buttons=buttons, platform=platform, exact_app_launch=exact
        )
    )
    driver = SimpleNamespace(supported_buttons=buttons | {"app_switch"})
    agent = JevAgent(
        FakeJev([]),
        AgentConfig(),
        registry,
        SimpleNamespace(driver=driver),
        None,
        MobileAgentState(),
        platform=platform,
    )

    assert agent._capabilities() == expected


# ── MobileAgent integration ────────────────────────────────────────────


class FakeContext:
    def __init__(self):
        self.events = []
        self.store = self

    def write_event_to_stream(self, event):
        self.events.append(event)

    async def set(self, key, value):
        setattr(self, key, value)

    async def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture
def mobile_env(monkeypatch):
    """Quiet MobileAgent setup; any LLM load fails the test."""

    def no_llm(*args, **kwargs):
        raise AssertionError("Jev must not load an LLM")

    async def no_flush(**kwargs):
        return None

    monkeypatch.setattr(droid_module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(
        droid_module.MobileAgent, "_configure_default_logging", lambda *a, **kw: None
    )
    monkeypatch.setattr(droid_module, "capture", lambda *a, **kw: None)
    monkeypatch.setattr(droid_module, "flush", no_flush)
    monkeypatch.setattr(llm_loader, "load_llm", no_llm)


def _install_fake_jev(monkeypatch, client):
    created = []

    def make_client(api_key, **kwargs):
        created.append(kwargs)
        return client

    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    monkeypatch.setattr(droid_module, "JevClient", make_client)
    return created


def _install_typesafe(monkeypatch, script, *, systemone_status=200, models_status=200):
    """Real JevClient against a mock TypeSafe server."""
    server = SimpleNamespace(requests=[], auth=[], paths=[], clients=[])
    script = list(script)

    def handler(request):
        server.auth.append(request.headers.get("authorization"))
        server.paths.append(request.url.path)
        headers = {"x-typesafe-request-id": f"req_{len(server.paths)}"}
        if request.url.path == "/v1/models":
            if models_status != 200:
                return httpx.Response(
                    models_status, json={"error": "invalid key"}, headers=headers
                )
            return httpx.Response(
                200, json={"models": [{"name": "jev-latest"}]}, headers=headers
            )
        if systemone_status != 200:
            return httpx.Response(
                systemone_status,
                json={"error": {"message": f"invalid api key {KEY}"}},
                headers=headers,
            )
        body = json.loads(request.content)
        server.requests.append(body)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers_for(body["questions"], script.pop(0)),
                "usage": {"input_tokens": 300, "output_tokens": 5},
            },
            headers=headers,
        )

    real_client = droid_module.JevClient

    def make_client(api_key, **kwargs):
        transport = httpx.MockTransport(handler)
        client = real_client(
            api_key, http_client=httpx.AsyncClient(transport=transport), **kwargs
        )
        server.clients.append(client)
        return client

    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    monkeypatch.setattr(droid_module, "JevClient", make_client)
    return server


def _jev_config(tmp_path=None, **agent_options):
    logging_config = (
        LoggingConfig(
            save_trajectory="action",
            trajectory_path=str(tmp_path / "trajectories"),
            trajectory_gifs=False,
        )
        if tmp_path is not None
        else LoggingConfig()
    )
    agent_options.setdefault("after_sleep_action", 0.0)
    return MobileConfig(
        agent=AgentConfig(name="jev", **agent_options),
        logging=logging_config,
        telemetry=TelemetryConfig(enabled=False),
    )


def test_missing_key_is_a_setup_error(mobile_env):
    with pytest.raises(JevConfigError) as error:
        droid_module.MobileAgent(GOAL, config=_jev_config())

    assert str(error.value) == MISSING_KEY_MESSAGE


def test_jev_loads_no_llm_profiles(mobile_env, monkeypatch):
    created = _install_fake_jev(monkeypatch, FakeJev([]))
    config = _jev_config()
    config.agent.jev = JevConfig(model="jev-1.13.0", timeout=12.0, max_retries=1)

    agent = droid_module.MobileAgent(GOAL, config=config)

    assert agent.fast_agent_llm is None
    assert agent.app_opener_llm is None
    assert agent.manager_agent is None and agent.executor_agent is None
    assert agent._using_jev and not agent._using_external_agent
    assert created == [
        {
            "model": "jev-1.13.0",
            "base_url": "https://api.typesafe.ai",
            "timeout": 12.0,
            "max_retries": 1,
        }
    ]
    assert llm_loader._get_required_profiles(config) == []
    assert llm_loader._get_required_profiles(config, output_model=object) == []
    assert llm_loader._get_required_profiles(MobileConfig()) == [
        "fast_agent",
        "app_opener",
    ]


def test_reasoning_is_ignored_with_a_warning(mobile_env, monkeypatch, mobilerun_logs):
    _install_fake_jev(monkeypatch, FakeJev([]))

    agent = droid_module.MobileAgent(GOAL, config=_jev_config(reasoning=True))

    assert agent.config.agent.reasoning is False
    assert agent.manager_agent is None
    assert "reasoning is ignored with the Jev agent" in mobilerun_logs.text


@pytest.mark.parametrize(
    "config",
    [
        MobileConfig(agent=AgentConfig(name="jev", vision_only=True)),
        MobileConfig(
            agent=AgentConfig(name="jev"),
            device=DeviceConfig(control_backend="visual-remote"),
        ),
    ],
)
def test_screenshot_only_modes_are_rejected(mobile_env, config):
    with pytest.raises(ValueError, match="Jev needs the accessibility tree"):
        droid_module.MobileAgent(GOAL, config=config)


def test_variables_become_typeable_values(mobile_env, monkeypatch):
    _install_fake_jev(monkeypatch, FakeJev([]))
    variables = {
        "place": QUERY,
        "count": 3,
        "ratio": 1.5,
        "flag": True,
        "blank": " ",
        "nested": {"a": 1},
    }

    agent = droid_module.MobileAgent(GOAL, config=_jev_config(), variables=variables)

    assert agent._jev_input_texts() == [QUERY, "3", "1.5"]


def test_start_handler_wires_the_tree_pipeline(mobile_env, monkeypatch):
    client = FakeJev([])
    _install_fake_jev(monkeypatch, client)
    config = _jev_config(use_normalized_coordinates=True)
    config.agent.fast_agent.vision = True
    agent = droid_module.MobileAgent(GOAL, config=config, driver=FakeDriver())

    event = asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

    assert isinstance(event, FastAgentExecuteEvent)
    assert client.list_models_calls == 1
    provider = agent.state_provider
    assert isinstance(provider, AndroidStateProvider)
    assert isinstance(provider.tree_formatter, JevIndexedFormatter)
    assert isinstance(provider.tree_filter, DetailedFilter)
    assert provider.vision_enabled is False
    assert provider.use_normalized is False
    assert "app_id" in agent.registry.tools["open_app"].params
    assert agent.action_ctx.app_opener_llm is None


@pytest.mark.parametrize(
    ("formatter", "warned"),
    [(JevIndexedFormatter, False), (IndexedFormatter, True)],
)
def test_injected_provider_without_jev_formatter_warns(
    mobile_env, monkeypatch, mobilerun_logs, formatter, warned
):
    _install_fake_jev(monkeypatch, FakeJev([]))
    driver = FakeDriver()
    provider = AndroidStateProvider(
        driver, tree_filter=DetailedFilter(), tree_formatter=formatter()
    )
    agent = droid_module.MobileAgent(
        GOAL, config=_jev_config(), driver=driver, state_provider=provider
    )

    asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

    assert agent.state_provider is provider
    assert ("JevIndexedFormatter" in mobilerun_logs.text) is warned


def test_invalid_key_fails_before_device_work(mobile_env, monkeypatch, tmp_path):
    client = FakeJev([])
    client.models_error = JevAuthError(
        "TypeSafe rejected the API key (HTTP 401). Check TYPESAFE_API_KEY.",
        status=401,
    )
    _install_fake_jev(monkeypatch, client)
    device_calls = []

    async def resolve_serial():
        device_calls.append("serial")
        raise AssertionError("device work must not start")

    monkeypatch.setattr(droid_module, "resolve_android_serial", resolve_serial)
    agent = droid_module.MobileAgent(GOAL, config=_jev_config(tmp_path))

    async def run():
        return await agent.run()

    with pytest.raises(JevAuthError):
        asyncio.run(run())

    assert device_calls == []
    assert client.closed is True
    assert agent.jev_client is None
    assert not (tmp_path / "trajectories").exists()


def test_full_run_records_a_replayable_macro(
    mobile_env, monkeypatch, tmp_path, clock, mobilerun_logs
):
    server = _install_typesafe(monkeypatch, HAPPY_SCRIPT)
    driver = FakeDriver()
    agent = droid_module.MobileAgent(GOAL, config=_jev_config(tmp_path), driver=driver)

    async def run():
        handler = agent.run()
        events = [event async for event in handler.stream_events()]
        return await handler, events

    result, events = asyncio.run(run())

    assert isinstance(result, ResultEvent)
    assert result.success is True
    assert result.steps == 4
    assert result.reason.startswith("Jev reports the goal is complete")
    assert server.paths == ["/v1/models"] + ["/v1/systemone"] * 5
    assert set(server.auth) == {f"Bearer {KEY}"}
    assert [call[0] for call in driver.calls] == [
        "start_app",
        "tap",
        "input_text",
        "press_button",
    ]
    assert agent.jev_client is None

    [folder] = list((tmp_path / "trajectories").iterdir())
    macro = json.loads((folder / "macro.json").read_text())
    assert macro["macro_schema_version"] == "2.0"
    assert macro["description"] == GOAL
    assert macro["total_actions"] == 4
    assert [a["action_type"] for a in macro["actions"]] == [
        "start_app",
        "tap",
        "input_text",
        "button_press",
    ]
    assert all(a["pre_state"]["nodes"] for a in macro["actions"])
    trajectory = json.loads((folder / "trajectory.json").read_text())
    types = [entry["type"] for entry in trajectory]
    assert types.count("JevDecisionEvent") == 5
    assert types.count("JevStepEvent") == 4
    assert len(_of(events, JevDecisionEvent)) == 5

    # The key never reaches files, logs or events.
    for path in (tmp_path / "trajectories").rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes()
    assert KEY not in mobilerun_logs.text
    assert KEY not in repr(events)
    assert KEY not in repr(agent.__dict__)


def test_jev_error_during_the_run_becomes_a_failed_result(
    mobile_env, monkeypatch, clock, mobilerun_logs
):
    server = _install_typesafe(monkeypatch, [], systemone_status=401)
    driver = FakeDriver()
    agent = droid_module.MobileAgent(GOAL, config=_jev_config(), driver=driver)

    async def run():
        return await agent.run()

    result = asyncio.run(run())

    assert result.success is False
    assert result.reason.startswith("Jev error: TypeSafe rejected the API key")
    assert result.steps == 0
    assert driver.calls == []
    assert server.paths == ["/v1/models", "/v1/systemone"]
    assert agent.jev_client is None
    assert KEY not in result.reason
    assert KEY not in mobilerun_logs.text


class BlankThenDriver(FakeDriver):
    """Returns an unreadable (empty) screen for the first ``blank_reads`` reads."""

    def __init__(self, blank_reads, **kwargs):
        super().__init__(**kwargs)
        self.blank_reads = blank_reads

    async def get_ui_tree(self):
        if self.blank_reads:
            self.blank_reads -= 1
            _, phone = SCREENS[self.screen]()
            return {
                "a11y_tree": _node("FrameLayout", (0, 0, 0, 0)),
                "phone_state": phone,
                "device_context": {"screen_bounds": {"width": WIDTH, "height": HEIGHT}},
            }
        return await super().get_ui_tree()


def test_unreadable_screen_is_read_again_without_asking_jev(clock):
    driver = BlankThenDriver(2, start="maps", transitions={})
    client = FakeJev([say("DONE")])

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is True
    assert run.result["steps"] == 0
    assert len(client.requests) == 1
    assert clock.sleeps == [0.5, 1.0]
    assert run.recorder.actions == []


def test_screen_that_stays_unreadable_stops_the_run(clock):
    driver = BlankThenDriver(100, start="maps", transitions={})
    client = FakeJev([])

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is False
    assert run.result["reason"] == (
        "Stopped: the device screen could not be read for 15s."
    )
    assert client.requests == []


def test_trajectory_records_one_frame_per_jev_decision(clock):
    # The tap changes nothing, so the screen is read again before Jev is asked.
    driver = FakeDriver(start="maps", transitions={})
    client = FakeJev([say("TAP", "Explore"), say("DONE")])

    run = asyncio.run(_run_jev(driver, client, save_trajectory="action"))

    assert run.result["success"] is True
    assert clock.sleeps == [0.5, 1.0, 2.0]
    assert len(_of(run.events, JevDecisionEvent)) == 2
    assert len(_of(run.events, ScreenshotEvent)) == 2
    assert len(_of(run.events, RecordUIStateEvent)) == 2


def test_unreadable_reads_record_no_frames(clock):
    driver = BlankThenDriver(2, start="maps", transitions={})

    run = asyncio.run(
        _run_jev(driver, FakeJev([say("DONE")]), save_trajectory="action")
    )

    assert run.result["success"] is True
    assert len(_of(run.events, ScreenshotEvent)) == 1
    [recorded] = _of(run.events, RecordUIStateEvent)
    assert recorded.ui_state == run.shared_state.a11y_tree
    assert recorded.ui_state


class LaggingDriver(FakeDriver):
    """Reports the previous screen for ``lag`` reads after each transition."""

    def __init__(self, lag, **kwargs):
        super().__init__(**kwargs)
        self.lag = lag
        self.stale = None
        self.stale_reads = 0

    def _advance(self, verb):
        before = self.screen
        super()._advance(verb)
        if self.screen != before:
            self.stale, self.stale_reads = before, self.lag

    async def get_ui_tree(self):
        if self.stale_reads:
            self.stale_reads -= 1
            current, self.screen = self.screen, self.stale
            try:
                return await super().get_ui_tree()
            finally:
                self.screen = current
        return await super().get_ui_tree()


def test_lagging_tree_after_an_action_is_read_again(clock):
    driver = LaggingDriver(2, transitions={("launcher", "start_app"): "maps"})
    client = FakeJev([say("OPEN_APP", "Maps"), say("DONE")])

    run = asyncio.run(_run_jev(driver, client))

    assert run.result["success"] is True
    assert len(client.requests) == 2
    assert client.requests[1][0]["app"]["package"] == MAPS
    assert client.requests[1][0]["recentActions"][0]["screenChanged"] is True


def _wda_form(*, first="", last=""):
    nodes = [
        _wda_node(
            "TextField",
            (40, 400, 1040, 520),
            resourceId="first",
            text=first,
            contentDescription="First name",
        ),
        _wda_node(
            "TextField",
            (40, 560, 1040, 680),
            resourceId="last",
            text=last,
            contentDescription="Last name",
        ),
        _wda_node(
            "Button", (40, 720, 1040, 840), contentDescription="Send", isClickable=True
        ),
        _wda_node(
            "Keyboard",
            (0, 1600, WIDTH, HEIGHT),
            [_wda_node("Key", (0, 1700, 100, 1800), contentDescription="q")],
        ),
    ]
    tree = _wda_node("Application", (0, 0, WIDTH, HEIGHT), nodes)
    phone = {
        "currentApp": "Shop",
        "packageName": "com.example.shop",
        "keyboardVisible": True,
        "isEditable": False,
    }
    return tree, phone


def test_button_tap_keeps_the_typed_field_so_unchanged_screens_settle(
    clock, monkeypatch
):
    monkeypatch.setitem(SCREENS, "f0", lambda: _wda_form())
    monkeypatch.setitem(SCREENS, "f1", lambda: _wda_form(first="Ann"))
    driver = FakeDriver(start="f0", transitions={("f0", "input_text"): "f1"})
    client = FakeJev(
        [
            say("TAP", "First name"),
            say("TYPE_TEXT", "Ann"),
            say("TAP", "Send"),
            say("TAP", "Send"),
            say("TAP", "Send"),
        ]
    )

    run = asyncio.run(
        _run_jev(
            driver, client, goal="Send the form", platform="ios", input_texts=["Ann"]
        )
    )

    # Send changes nothing: each tap is followed by settle reads, and the
    # third identical tap stops the run.
    assert run.result["reason"].startswith("Stopped: Jev repeated TAP")
    assert [c[0] for c in driver.calls].count("tap") == 3
    assert client.requests[3][0]["recentActions"][-1]["screenChanged"] is False
