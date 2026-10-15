import math

import pytest

from mobilerun.agent.jev.client import JevResponse
from mobilerun.agent.jev.errors import JevResponseError
from mobilerun.agent.jev.observation import JevElement, JevScreen
from mobilerun.agent.jev.policy import (
    OPERATIONS,
    RULES,
    Capabilities,
    HistoryEntry,
    build_plan,
    decide,
    describe_field,
    narrow_apps,
    text_candidates,
    validate_choice,
)

GOAL = "Open Maps and search for 'Golden Gate Bridge'"
APPS = [
    {"package": "com.google.android.apps.maps", "label": "Maps"},
    {"package": "com.android.settings", "label": "Settings"},
    {"package": "com.google.android.apps.messaging", "label": "Messages"},
]
LEGACY_IOS = Capabilities(buttons=frozenset({"home"}))
DEFAULT_CAPABILITIES = Capabilities()


def el(index, label="", *, bounds=(0, 0, 100, 100), **flags):
    values = {
        "clickable": False,
        "editable": False,
        "scrollable": False,
        "enabled": True,
        "focused": False,
        "password": False,
        "checkable": False,
        "checked": False,
        "selected": False,
        "value": "",
        "keyboard": False,
    }
    values.update(flags)
    return JevElement(index=index, label=label, bounds=bounds, **values)


def screen(elements, *, focused=None, keyboard_visible=False, package="com.example"):
    return JevScreen(
        platform="android",
        package=package,
        app_name="Example",
        width=1000,
        height=2000,
        elements=elements,
        focused=focused,
        keyboard_visible=keyboard_visible,
        visible_text=[e.label for e in elements if e.label],
        fingerprint="fp",
    )


def maps_screen():
    search = el(
        2,
        "Search here",
        bounds=(40, 150, 960, 280),
        clickable=True,
        editable=True,
        focused=True,
        hint="Search here",
    )
    elements = [
        el(1, "root", bounds=(0, 0, 1000, 2000), descendants=8),
        search,
        el(3, "Results", bounds=(0, 300, 1000, 1800), scrollable=True, descendants=2),
        el(
            4,
            "Golden Gate Bridge / San Francisco",
            bounds=(0, 300, 1000, 500),
            clickable=True,
        ),
        el(5, "Golden Gate Bridge", bounds=(20, 310, 900, 400)),
        el(
            6,
            "Wi-Fi",
            bounds=(800, 1850, 950, 1950),
            clickable=True,
            checkable=True,
            checked=True,
        ),
        el(
            7, "", bounds=(0, 1900, 100, 2000), clickable=True, class_name="ImageButton"
        ),
        el(8, "Disabled", bounds=(200, 1900, 300, 2000), clickable=True, enabled=False),
        el(9, "q", bounds=(0, 1500, 100, 1600), clickable=True, keyboard=True),
    ]
    return screen(
        elements, focused=search, keyboard_visible=True, package="com.android.settings"
    )


def plan_for(
    scr=None,
    *,
    texts=(),
    apps=APPS,
    capabilities=DEFAULT_CAPABILITIES,
    history=(),
    goal=GOAL,
):
    return build_plan(
        goal,
        scr or maps_screen(),
        history=list(history),
        texts=texts,
        apps=apps,
        capabilities=capabilities,
    )


def choice(key, keys, confidence=0.9):
    keys = list(keys)
    others = [k for k in keys if k != key]
    top = 0.9 if others else 1.0
    probabilities = {k: top if k == key else (1 - top) / len(others) for k in keys}
    return {
        "type": "choice",
        "choice": key,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def target_question(operation):
    if operation.startswith("SCROLL_"):
        return "scroll_target"
    return {
        "TAP": "tap_target",
        "OPEN_APP": "app_target",
        "TYPE_TEXT": "text_value",
    }.get(operation)


def respond(
    plan, operation, target=None, *, confidence=0.9, target_confidence=0.9, **extra
):
    answers = {
        "operation": choice(
            operation, plan.questions["operation"]["criteria"], confidence
        )
    }
    if target is not None:
        qid = target_question(operation)
        answers[qid] = choice(
            target, plan.questions[qid]["criteria"], target_confidence
        )
    answers.update(extra)
    return JevResponse(
        answers=answers,
        model="jev-1.13.0",
        usage={"input_tokens": 120, "output_tokens": 3},
        request_id="req_abc",
        latency_ms=42,
    )


def key_for(plan, qid, label_part):
    return next(
        k for k, v in plan.questions[qid]["criteria"].items() if label_part in v
    )


# -- text candidates ----------------------------------------------------------


def test_text_candidates_from_goal_strip_quotes():
    values, overflow = text_candidates(GOAL)

    assert not overflow
    assert "Golden Gate Bridge" in values
    assert "'Golden Gate Bridge'" not in values
    assert "Golden Gate Bridge'" not in values
    # Quoted values come first, then the word spans.
    assert values[:4] == ["Golden Gate Bridge", "Open", "Maps", "and"]
    assert len(values) == len(set(values))


LONG_MESSAGE = "Thanks, I will review the document and get back to you tomorrow"


@pytest.mark.parametrize(
    ("goal", "value"),
    [
        ('Reply with "See you at 5pm!"', "See you at 5pm!"),
        ('Rename the file to "report (final)"', "report (final)"),
        ("Type '(555) 123-4567' in the dialer", "(555) 123-4567"),
        ("Save the note 'Buy milk.'", "Buy milk."),
        ("Send “Are you coming?” to Bob", "Are you coming?"),
        ("Say ‘it’s fine’ to Ann", "it’s fine"),
        ("Type 'it's late' now", "it's late"),
        (f'Reply "{LONG_MESSAGE}" to Sam', LONG_MESSAGE),
    ],
)
def test_text_candidates_offer_quoted_values_verbatim(goal, value):
    values, overflow = text_candidates(goal)
    assert not overflow
    assert values[0] == value


def test_text_candidates_apostrophes_do_not_pair_as_quotes():
    values, _ = text_candidates("Don't tell John's sister")
    assert values[:4] == ["Don't", "tell", "John's", "sister"]
    assert not any(value.startswith("t tell") for value in values)


def test_text_candidates_quoted_values_count_toward_the_cap():
    def goal(quoted_groups):
        # 35 distinct words give 252 spans; each quoted 9-word group adds one value.
        words = [f"w{i}" for i in range(35)]
        for group in range(quoted_groups):
            words[9 * group] = '"' + words[9 * group]
            words[9 * group + 8] += '"'
        return " ".join(words)

    values, overflow = text_candidates(goal(2))
    assert len(values) == 254 and not overflow
    assert text_candidates(goal(3)) == ([], True)


def test_text_candidates_explicit_values_replace_spans():
    values, overflow = text_candidates(GOAL, ["94016", "Golden Gate", "94016", ""])
    assert values == ["94016", "Golden Gate"]
    assert not overflow


def test_text_candidates_overflow_returns_nothing():
    values, overflow = text_candidates(" ".join(f"word{i}" for i in range(40)))
    assert values == []
    assert overflow

    values, overflow = text_candidates(" ".join(f"word{i}" for i in range(30)))
    assert 0 < len(values) <= 254
    assert not overflow


# -- apps ---------------------------------------------------------------------


def test_narrow_apps_prefers_named_apps_and_excludes_foreground():
    assert narrow_apps("open settings and wifi", APPS, "com.example") == [
        {"package": "com.android.settings", "label": "Settings"}
    ]
    # No whole-word match: every app except the foreground one.
    assert [
        a["package"] for a in narrow_apps("open Mapsy", APPS, "com.android.settings")
    ] == [
        "com.google.android.apps.maps",
        "com.google.android.apps.messaging",
    ]
    # A goal that only names the foreground app offers nothing to switch to.
    assert narrow_apps(GOAL, APPS, "com.google.android.apps.maps") == []
    assert narrow_apps("open my_maps", APPS, "")[0]["label"] == "Maps"


def test_narrow_apps_unicode_word_boundaries_and_cap():
    apps = [
        {"package": "a.cafe", "label": "Café"},
        {"package": "b.cpp", "label": "C++"},
    ]
    assert narrow_apps("open Café now", apps, "") == [apps[0]]
    assert narrow_apps("open Cafés", apps, "") == apps
    assert narrow_apps("use C++ today", apps, "") == [apps[1]]

    many = [{"package": f"p{i}", "label": f"App {i}"} for i in range(250)] + [
        {"label": "x"}
    ]
    assert len(narrow_apps("do something", many, "")) == 200


# -- plan ---------------------------------------------------------------------


def test_build_plan_android_questions_and_state():
    history = [
        HistoryEntry("TAP", "[2] Focus text input: Search here", None, True)
    ] * 10
    plan = plan_for(history=history)
    operation = plan.questions["operation"]

    assert operation["type"] == "choice"
    assert operation["instructions"] == {"goal": GOAL, "rules": RULES}
    assert list(operation["criteria"]) == [
        "OPEN_APP",
        "TAP",
        "TYPE_TEXT",
        "SCROLL_DOWN",
        "SCROLL_UP",
        "SCROLL_LEFT",
        "SCROLL_RIGHT",
        "BACK",
        "HOME",
        "ENTER",
        "WAIT",
        "DONE",
        "BLOCKED",
    ]
    assert set(operation["criteria"]) == set(OPERATIONS)
    assert operation["criteria"]["HOME"] == "Go to the device home screen."
    assert plan.operations == list(operation["criteria"])

    tap = plan.questions["tap_target"]
    assert tap["instructions"]["goal"] == GOAL
    assert tap["instructions"]["rules"].startswith(
        "Assuming the next operation is TAP,"
    )
    assert tap["criteria"] == {
        "2": "[2] Focus text input: Search here",
        "4": "[4] Golden Gate Bridge / San Francisco",
        "6": "[6] Wi-Fi (on)",
        "7": "[7] unlabeled ImageButton",
    }

    scroll = plan.questions["scroll_target"]
    assert "any SCROLL direction" in scroll["instructions"]["rules"]
    assert scroll["criteria"] == {
        "3": "Scrollable region [3] Results",
        "screen": "The whole screen",
    }

    apps = plan.questions["app_target"]["criteria"]
    assert apps == {
        "1": "Maps (com.google.android.apps.maps)",
    }

    text = plan.questions["text_value"]
    assert "Golden Gate Bridge" in text["criteria"].values()
    assert text["criteria"]["NONE"] == (
        "None of the supplied text spans is an appropriate complete value for this field."
    )
    assert text["instructions"]["rules"].endswith(
        "Do not type the entire goal. If the desired value is missing, select NONE."
    )

    state = plan.state
    assert state["goal"] == GOAL
    assert state["platform"] == "android"
    assert state["app"] == {"package": "com.android.settings", "name": "Example"}
    assert state["keyboardVisible"] is True
    assert state["focusedField"] == {"index": "2", "label": "Search here", "value": ""}
    assert state["textSource"] == "goal"
    assert state["textEntryAvailableAfterFocus"] is True
    assert state["availableApps"] == [
        {"index": "1", "label": "Maps", "package": "com.google.android.apps.maps"}
    ]
    assert [e["index"] for e in state["elements"]] == ["2", "3", "4", "6", "7"]
    results = next(e for e in state["elements"] if e["index"] == "3")
    assert results["scrollable"] is True
    assert results["operations"] == [
        "SCROLL_DOWN",
        "SCROLL_UP",
        "SCROLL_LEFT",
        "SCROLL_RIGHT",
    ]
    wifi = next(e for e in state["elements"] if e["index"] == "6")
    assert wifi == {
        "index": "6",
        "label": "Wi-Fi",
        "checked": True,
        "operations": ["TAP"],
    }
    assert len(state["recentActions"]) == 8
    assert state["recentActions"][0] == {
        "operation": "TAP",
        "label": "[2] Focus text input: Search here",
        "screenChanged": True,
        "succeeded": True,
    }

    assert plan.targets["tap_target"]["6"] == {"index": 6, "label": "[6] Wi-Fi (on)"}
    assert plan.targets["app_target"]["1"]["package"] == "com.google.android.apps.maps"


def test_build_plan_supplied_texts():
    plan = plan_for(texts=["94016"])
    assert plan.state["textSource"] == "supplied"
    assert list(plan.questions["text_value"]["criteria"]) == ["1", "NONE"]
    assert plan.targets["text_value"]["1"]["text"] == "94016"


def test_build_plan_legacy_ios_capabilities():
    plan = plan_for(capabilities=LEGACY_IOS)
    ops = plan.questions["operation"]["criteria"]
    assert "HOME" in ops
    assert "BACK" not in ops
    assert "ENTER" not in ops


@pytest.mark.parametrize(
    ("capabilities", "missing", "question"),
    [
        (Capabilities(tap=False), {"TAP"}, "tap_target"),
        (
            Capabilities(swipe=False),
            {"SCROLL_DOWN", "SCROLL_UP", "SCROLL_LEFT", "SCROLL_RIGHT"},
            "scroll_target",
        ),
        (Capabilities(open_app=False), {"OPEN_APP"}, "app_target"),
        (Capabilities(type_text=False), {"TYPE_TEXT"}, "text_value"),
        (Capabilities(wait=False), {"WAIT"}, None),
        (Capabilities(buttons=frozenset()), {"BACK", "HOME", "ENTER"}, None),
    ],
)
def test_build_plan_operations_follow_capabilities(capabilities, missing, question):
    plan = plan_for(capabilities=capabilities)
    ops = set(plan.questions["operation"]["criteria"])
    assert not ops & missing
    assert {"DONE", "BLOCKED"} <= ops
    if question:
        assert question not in plan.questions


def test_type_text_requires_focused_non_password_field_and_values():
    base = maps_screen()
    unfocused = screen(base.elements, focused=None, keyboard_visible=False)
    plan = plan_for(unfocused)
    assert "TYPE_TEXT" not in plan.questions["operation"]["criteria"]
    assert "ENTER" not in plan.questions["operation"]["criteria"]
    assert "text_value" not in plan.questions

    password = el(
        2,
        "[password field]",
        clickable=True,
        editable=True,
        focused=True,
        password=True,
    )
    plan = plan_for(screen([password], focused=password, keyboard_visible=True))
    assert "TYPE_TEXT" not in plan.questions["operation"]["criteria"]
    assert "ENTER" in plan.questions["operation"]["criteria"]
    assert plan.state["focusedField"] == {
        "index": "2",
        "label": "[password field]",
        "value": "",
    }

    plan = plan_for(goal=" ".join(f"word{i}" for i in range(40)))
    assert "TYPE_TEXT" not in plan.questions["operation"]["criteria"]
    assert plan.text_overflow
    assert plan.state["textEntryAvailableAfterFocus"] is False


def test_no_apps_no_open_app():
    plan = plan_for(apps=[])
    assert "OPEN_APP" not in plan.questions["operation"]["criteria"]
    assert "availableApps" not in plan.state


def test_tap_targets_are_capped_and_prefer_labeled_elements():
    elements = [el(i, "", clickable=True) for i in range(1, 101)]
    elements += [el(i, f"Item {i}", clickable=True) for i in range(101, 401)]
    plan = plan_for(screen(elements), apps=[])
    criteria = plan.questions["tap_target"]["criteria"]

    assert len(criteria) == 254
    assert all(int(k) > 100 for k in criteria)
    assert list(criteria) == sorted(criteria, key=int)
    for question in plan.questions.values():
        assert len(question["criteria"]) <= 255


def test_text_value_and_scroll_targets_fit_choice_limit():
    texts = [f"value {i}" for i in range(300)]
    plan = plan_for(texts=texts)
    assert len(plan.questions["text_value"]["criteria"]) == 255

    regions = [
        el(i, f"Pane {i}", bounds=(i, 0, i + 50, 50), scrollable=True)
        for i in range(1, 301)
    ]
    plan = plan_for(screen(regions), apps=[])
    criteria = plan.questions["scroll_target"]["criteria"]
    assert len(criteria) == 255
    assert list(criteria)[-1] == "screen"


def test_scroll_regions_prefer_nested_lists_and_skip_duplicates():
    elements = [
        # Container whose nested list covers most of it: container skipped.
        el(1, "Container", bounds=(0, 0, 1000, 1000), scrollable=True, descendants=2),
        el(2, "List", bounds=(0, 100, 1000, 1000), scrollable=True),
        # Container with a small nested pane: both kept.
        el(3, "Page", bounds=(0, 1000, 1000, 2000), scrollable=True, descendants=1),
        el(4, "Carousel", bounds=(0, 1000, 1000, 1200), scrollable=True),
        # Same bounds as the list.
        el(5, "Mirror", bounds=(0, 100, 1000, 1000), scrollable=True),
        el(6, "Keys", bounds=(0, 1500, 1000, 2000), scrollable=True, keyboard=True),
        el(7, "Off", bounds=(0, 0, 10, 10), scrollable=True, enabled=False),
    ]
    plan = plan_for(screen(elements), apps=[])
    assert list(plan.questions["scroll_target"]["criteria"]) == [
        "2",
        "3",
        "4",
        "screen",
    ]


# -- decide -------------------------------------------------------------------


def test_decide_tap():
    plan = plan_for()
    decision = decide(plan, respond(plan, "TAP", "4"), min_confidence=0.3)

    assert decision.status == "action"
    assert decision.operation == "TAP"
    assert decision.tool == "click"
    assert decision.args == {"index": 4}
    assert isinstance(decision.args["index"], int)
    assert decision.label == "[4] Golden Gate Bridge / San Francisco"
    assert decision.target_key == "4"
    assert decision.confidence == 0.9
    assert decision.target_confidence == 0.9
    assert decision.model == "jev-1.13.0"
    assert decision.latency_ms == 42
    assert decision.usage == {"input_tokens": 120, "output_tokens": 3}
    assert math.isclose(sum(decision.probabilities.values()), 1)
    assert decision.reason is None


def test_decide_type_text():
    plan = plan_for()
    key = next(
        k
        for k, v in plan.questions["text_value"]["criteria"].items()
        if v == "Golden Gate Bridge"
    )
    decision = decide(plan, respond(plan, "TYPE_TEXT", key), min_confidence=0.3)

    assert decision.status == "action"
    assert decision.tool == "type"
    assert decision.args == {"text": "Golden Gate Bridge", "index": -1, "clear": True}
    assert decision.text == "Golden Gate Bridge"
    assert decision.label == "Golden Gate Bridge"


@pytest.mark.parametrize(
    ("operation", "start", "end"),
    [
        ("SCROLL_DOWN", [500, 1500], [500, 600]),
        ("SCROLL_UP", [500, 600], [500, 1500]),
        ("SCROLL_RIGHT", [800, 1050], [200, 1050]),
        ("SCROLL_LEFT", [200, 1050], [800, 1050]),
    ],
)
def test_decide_scroll_region_geometry(operation, start, end):
    plan = plan_for()
    decision = decide(plan, respond(plan, operation, "3"), min_confidence=0.3)

    assert decision.tool == "swipe"
    assert decision.args == {"coordinate": start, "coordinate2": end, "duration": 0.4}
    assert all(
        isinstance(v, int)
        for v in decision.args["coordinate"] + decision.args["coordinate2"]
    )
    assert decision.label == "Scrollable region [3] Results"


def test_decide_scroll_whole_screen():
    plan = plan_for()
    decision = decide(plan, respond(plan, "SCROLL_DOWN", "screen"), min_confidence=0.3)
    # Screen inset by 10% (100,200)-(900,1800), then 20%/80% of that region.
    assert decision.args == {
        "coordinate": [500, 1480],
        "coordinate2": [500, 520],
        "duration": 0.4,
    }
    assert decision.label == "The whole screen"


@pytest.mark.parametrize(
    ("operation", "button"), [("BACK", "back"), ("HOME", "home"), ("ENTER", "enter")]
)
def test_decide_system_buttons(operation, button):
    plan = plan_for()
    decision = decide(plan, respond(plan, operation), min_confidence=0.3)
    assert decision.status == "action"
    assert (decision.tool, decision.args) == ("system_button", {"button": button})
    assert decision.target_key is None and decision.label is None


def test_decide_open_app_and_wait():
    plan = plan_for()
    decision = decide(plan, respond(plan, "OPEN_APP", "1"), min_confidence=0.3)
    assert (decision.tool, decision.args) == (
        "open_app",
        {"app_id": "com.google.android.apps.maps"},
    )
    assert decision.label == "Maps (com.google.android.apps.maps)"

    decision = decide(plan, respond(plan, "WAIT"), min_confidence=0.3)
    assert (decision.status, decision.tool, decision.args) == (
        "action",
        "wait",
        {"duration": 1.0},
    )


def test_decide_done_and_blocked():
    plan = plan_for()
    done = decide(plan, respond(plan, "DONE", confidence=0.81), min_confidence=0.3)
    assert done.status == "done"
    assert done.tool is None and done.args is None
    assert done.reason == "Jev reports the goal is complete (confidence 0.81)."

    blocked = decide(plan, respond(plan, "BLOCKED", confidence=0.5), min_confidence=0.3)
    assert blocked.status == "blocked"
    assert (
        blocked.reason
        == "Jev found no operation that advances the goal (confidence 0.50)."
    )


def test_blocked_hints_at_var_when_a_field_has_no_values():
    plan = plan_for(goal=" ".join(f"word{i}" for i in range(40)))
    blocked = decide(plan, respond(plan, "BLOCKED"), min_confidence=0.3)
    assert "--var NAME=VALUE" in blocked.reason
    assert "too many text spans" in blocked.reason

    unfocused = screen(maps_screen().elements[3:6])
    plan = plan_for(unfocused, goal="Open the settings page now")
    blocked = decide(plan, respond(plan, "BLOCKED"), min_confidence=0.3)
    assert "--var" not in blocked.reason


def test_low_confidence_is_uncertain():
    plan = plan_for()
    low_op = decide(plan, respond(plan, "TAP", "4", confidence=0.1), min_confidence=0.3)
    assert low_op.status == "uncertain"
    assert low_op.tool is None and low_op.args is None
    assert low_op.reason == (
        "Stopped: Jev's confidence in TAP [4] Golden Gate Bridge / San Francisco is 0.10, "
        "below the 0.30 threshold."
    )

    low_target = decide(
        plan, respond(plan, "TAP", "4", target_confidence=0.2), min_confidence=0.3
    )
    assert low_target.status == "uncertain"
    assert "is 0.20" in low_target.reason

    low_done = decide(plan, respond(plan, "DONE", confidence=0.05), min_confidence=0.3)
    assert low_done.status == "uncertain"
    assert low_done.reason.startswith("Stopped: Jev's confidence in DONE is 0.05")

    disabled = decide(
        plan,
        respond(plan, "TAP", "4", confidence=0.0, target_confidence=0.0),
        min_confidence=0,
    )
    assert disabled.status == "action"


@pytest.mark.parametrize("scores", [{"confidence": 0.1}, {"target_confidence": 0.1}])
def test_uncertain_type_text_reason_names_the_field_not_the_value(scores):
    canary = "OTP-918273-CANARY"
    plan = plan_for(texts=[canary])
    decision = decide(
        plan, respond(plan, "TYPE_TEXT", "1", **scores), min_confidence=0.3
    )

    assert decision.status == "uncertain"
    assert canary not in decision.reason
    assert decision.reason == (
        "Stopped: Jev's confidence in TYPE_TEXT [2] Search here is 0.10, "
        "below the 0.30 threshold."
    )
    # The local decision label still shows what would have been typed.
    assert decision.label == canary


def test_describe_field_never_includes_field_contents():
    assert describe_field(el(3, "Email", editable=True)) == "[3] Email"
    assert describe_field(el(3, "Email", editable=True, hint="Work email")) == (
        "[3] Work email"
    )
    # A filled field's label carries its text.
    filled = el(3, "4111-2222 / Card", editable=True, value="4111-2222")
    assert describe_field(filled) == "[3] text field"
    filled.hint = "Card number"
    assert describe_field(filled) == "[3] Card number"


def test_none_text_needs_input():
    plan = plan_for()
    decision = decide(
        plan,
        respond(plan, "TYPE_TEXT", "NONE", target_confidence=0.0),
        min_confidence=0.3,
    )
    assert decision.status == "needs_input"
    assert decision.tool is None
    assert decision.text is None
    assert "--var NAME=VALUE" in decision.reason


def _bad(answer):
    plan = plan_for()
    response = respond(plan, "TAP", "4")
    response.answers["operation"] = answer
    with pytest.raises(JevResponseError) as info:
        decide(plan, response, min_confidence=0.3)
    return info.value


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(type="noul"),
        lambda a: a.update(choice="SWIPE"),
        lambda a: a.update(choice=None),
        lambda a: a["probabilities"].pop("WAIT"),
        lambda a: a["probabilities"].update(EXTRA=0.0),
        lambda a: a["probabilities"].update(WAIT=0.5),
        lambda a: a["probabilities"].update(TAP=0.0, WAIT=0.9),
        lambda a: a["probabilities"].update(WAIT=float("nan")),
        lambda a: a["probabilities"].update(WAIT=-0.01),
        lambda a: a["probabilities"].update(WAIT=True),
        lambda a: a.update(confidence=1.5),
        lambda a: a.update(confidence="0.9"),
        lambda a: a.update(probabilities=[0.9, 0.1]),
    ],
)
def test_invalid_operation_distribution_raises(mutate):
    plan = plan_for()
    answer = choice("TAP", plan.questions["operation"]["criteria"])
    mutate(answer)
    error = _bad(answer)
    assert "invalid answer for 'operation'" in str(error)
    assert "req_abc" in str(error)
    assert error.request_id == "req_abc"


def test_missing_answers_raise():
    plan = plan_for()
    with pytest.raises(JevResponseError):
        decide(plan, JevResponse(answers={}), min_confidence=0.3)

    response = respond(plan, "TAP")
    with pytest.raises(JevResponseError, match="tap_target"):
        decide(plan, response, min_confidence=0.3)

    response = respond(plan, "TAP", "4")
    response.answers["tap_target"]["choice"] = "999"
    with pytest.raises(JevResponseError, match="tap_target"):
        decide(plan, response, min_confidence=0.3)


def test_unused_malformed_speculative_answers_are_ignored():
    plan = plan_for()
    response = respond(
        plan,
        "TAP",
        "4",
        scroll_target={"type": "choice", "choice": "nope"},
        app_target="garbage",
        text_value={"probabilities": None},
    )
    decision = decide(plan, response, min_confidence=0.3)
    assert decision.status == "action"
    assert decision.args == {"index": 4}


def test_validate_choice_tolerates_rounding():
    keys = ["A", "B", "C"]
    answer = {
        "type": "choice",
        "choice": "A",
        "confidence": 0.5,
        "probabilities": {"A": 0.67, "B": 0.17, "C": 0.17},
    }
    assert validate_choice(answer, keys)["choice"] == "A"
    answer["probabilities"] = {"A": 0.5, "B": 0.5, "C": 0.0}
    assert validate_choice(answer, keys)["choice"] == "A"
    answer["probabilities"] = {"A": 0.6, "B": 0.3, "C": 0.2}
    with pytest.raises(JevResponseError, match="'operation'"):
        validate_choice(answer, keys)


def test_validate_choice_tolerates_rounding_across_many_options():
    # 2-decimal rounding: 199 options at ~0.0035 each come back as 0.0.
    keys = [str(i) for i in range(1, 201)]
    probabilities = {key: 0.0 for key in keys}
    probabilities["7"] = 0.3
    answer = {
        "type": "choice",
        "choice": "7",
        "confidence": 0.3,
        "probabilities": probabilities,
    }
    assert validate_choice(answer, keys, "tap_target")["choice"] == "7"


def test_decide_enter_taps_the_return_key_without_an_enter_button():
    search = el(3, "Search Maps", editable=True, focused=True)
    key = el(9, "search", clickable=True, keyboard=True, bounds=(300, 752, 399, 806))
    scr = screen([search, key], focused=search, keyboard_visible=True)
    scr.return_key = key
    plan = plan_for(scr, capabilities=Capabilities(buttons=frozenset({"home"})))

    decision = decide(plan, respond(plan, "ENTER"), min_confidence=0.3)

    assert decision.status == "action"
    assert (decision.tool, decision.args) == ("click", {"index": 9})


def test_covered_elements_are_not_tap_or_scroll_targets():
    code = el(1, "Code", bounds=(0, 1700, 1000, 1800), editable=True, covered=True)
    elements = [
        code,
        el(2, "Visible", clickable=True),
        el(3, "Hidden", bounds=(0, 1600, 1000, 1700), clickable=True, covered=True),
        el(4, "List", bounds=(0, 1200, 1000, 2000), scrollable=True, covered=True),
    ]
    scr = screen(elements, focused=code, keyboard_visible=True)
    plan = plan_for(scr, apps=[], goal="Enter the code '123456'")

    assert list(plan.questions["tap_target"]["criteria"]) == ["2"]
    assert list(plan.questions["scroll_target"]["criteria"]) == ["screen"]
    # A covered field can still hold input focus.
    assert plan.state["focusedField"]["index"] == "1"
    assert "TYPE_TEXT" in plan.operations
