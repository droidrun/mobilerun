import asyncio
import copy

from mobilerun.agent.jev.observation import (
    JevIndexedFormatter,
    build_screen,
)
from mobilerun.agent.jev.policy import Capabilities, build_plan
from mobilerun.macro.matcher import compare_states
from mobilerun.macro.state import normalize_ui_state
from mobilerun.tools.filters import DetailedFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.state import UIState

MAPS = "com.google.android.apps.maps"
IME = "com.google.android.inputmethod.latin"
CONTEXT = {"screen_bounds": {"width": 1080, "height": 2400}}


def _b(left, top, right, bottom):
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _node(class_name, bounds, children=(), package=MAPS, **flags):
    node = {
        "className": class_name,
        "packageName": package,
        "boundsInScreen": _b(*bounds),
        "isEnabled": True,
        "isVisibleToUser": True,
        "children": list(children),
    }
    node.update(flags)
    return node


def android_tree():
    return _node(
        "android.widget.FrameLayout",
        (0, 0, 1080, 2400),
        [
            _node(
                "android.widget.EditText",
                (40, 150, 1040, 280),
                resourceId=f"{MAPS}:id/search_omnibox_text_box",
                text="Search here",
                hint="Search here",
                isShowingHintText=True,
                isEditable=True,
                isFocused=True,
                isClickable=True,
            ),
            _node(
                "androidx.recyclerview.widget.RecyclerView",
                (0, 300, 1080, 2200),
                [
                    _node(
                        "android.widget.LinearLayout",
                        (0, 300, 1080, 500),
                        [
                            _node(
                                "android.widget.TextView",
                                (20, 310, 1000, 400),
                                text="Golden Gate Bridge",
                            ),
                            _node(
                                "android.widget.TextView",
                                (20, 400, 1000, 490),
                                text="San Francisco",
                            ),
                        ],
                        isClickable=True,
                    )
                ],
                resourceId=f"{MAPS}:id/list",
                isScrollable=True,
            ),
            _node(
                "android.widget.Switch",
                (900, 2200, 1040, 2300),
                text="Wi-Fi",
                isCheckable=True,
                isChecked=True,
                isClickable=True,
            ),
            _node(
                "android.widget.EditText",
                (40, 2300, 800, 2390),
                text="hunter2",
                contentDescription="Password",
                isPassword=True,
                isEditable=True,
                isClickable=True,
            ),
            _node(
                "android.widget.FrameLayout",
                (0, 1600, 1080, 2400),
                [
                    _node(
                        "android.widget.Button",
                        (0, 1700, 100, 1800),
                        package=IME,
                        text="q",
                        isClickable=True,
                    )
                ],
                package=IME,
            ),
        ],
    )


def _format(formatter, tree, phone_state):
    filtered = DetailedFilter().filter(copy.deepcopy(tree), CONTEXT)
    return formatter.format(filtered, phone_state)


def _ui_state(elements, phone_state, width=1080, height=2400):
    return UIState(
        elements=elements,
        formatted_text="",
        focused_text="",
        phone_state=phone_state,
        screen_width=width,
        screen_height=height,
    )


def _android_state(tree=None, phone_state=None):
    phone_state = phone_state or {
        "currentApp": "Maps",
        "packageName": MAPS,
        "keyboardVisible": True,
        "isEditable": True,
    }
    _, _, elements, _ = _format(
        JevIndexedFormatter(), tree or android_tree(), phone_state
    )
    return _ui_state(elements, phone_state)


def _by_class(elements, class_name):
    return [e for e in elements if e["className"] == class_name]


def test_formatter_adds_only_the_namespaced_jev_key():
    phone_state = {"currentApp": "Maps", "packageName": MAPS}
    plain = _format(IndexedFormatter(), android_tree(), phone_state)
    jev = _format(JevIndexedFormatter(), android_tree(), phone_state)

    assert jev[0] == plain[0]
    assert jev[1] == plain[1]
    assert len(jev[2]) == len(plain[2])
    for with_jev, without in zip(jev[2], plain[2], strict=True):
        assert set(with_jev) - set(without) == {"jev"}
        assert {k: v for k, v in with_jev.items() if k != "jev"} == without


def test_formatter_flags_labels_and_keyboard():
    _, _, elements, _ = _format(JevIndexedFormatter(), android_tree(), {})
    search, password = _by_class(elements, "EditText")
    recycler = _by_class(elements, "RecyclerView")[0]
    row = _by_class(elements, "LinearLayout")[0]
    switch = _by_class(elements, "Switch")[0]
    key = _by_class(elements, "Button")[0]
    ime_container = elements[key["index"] - 2]

    assert search["jev"]["editable"] and search["jev"]["focused"]
    assert search["jev"]["clickable"] and search["jev"]["enabled"]
    # Hint text shown in an empty field is not its value.
    assert search["jev"]["text"] == ""
    assert search["jev"]["hint"] == "Search here"

    assert row["jev"]["label"] == "Golden Gate Bridge / San Francisco"
    assert recycler["jev"]["scrollable"]
    assert recycler["jev"]["descendants"] == 3
    assert switch["jev"]["checkable"] and switch["jev"]["checked"]

    assert password["jev"]["password"]
    assert password["jev"]["text"] == ""
    assert "hunter2" not in password["jev"]["label"]

    assert key["jev"]["keyboard"]
    assert ime_container["jev"]["keyboard"]
    assert not any(
        e["jev"]["keyboard"] for e in elements if e["index"] < ime_container["index"]
    )
    # Keyboard keys never leak into the labels of app containers.
    assert "q" not in elements[0]["jev"]["label"].split(" / ")


def test_formatter_resets_keyboard_tracking_between_states():
    formatter = JevIndexedFormatter()
    _format(formatter, android_tree(), {})
    tree = _node(
        "android.widget.FrameLayout",
        (0, 0, 1080, 2400),
        [_node("android.widget.Button", (0, 0, 100, 100), text="OK", isClickable=True)],
    )
    _, _, elements, _ = _format(formatter, tree, {})
    assert not any(e["jev"]["keyboard"] for e in elements)


def test_formatter_label_is_capped():
    tree = _node(
        "android.widget.LinearLayout",
        (0, 0, 1080, 2400),
        [
            _node(
                "android.widget.TextView",
                (0, i * 10, 1080, i * 10 + 10),
                text=f"item {i} " * 5,
            )
            for i in range(40)
        ],
        isClickable=True,
    )
    _, _, elements, _ = _format(JevIndexedFormatter(), tree, {})
    assert len(elements[0]["jev"]["label"]) == 160
    assert elements[0]["jev"]["label"].endswith("…")


def test_jev_formatter_keeps_macro_snapshots_identical():
    phone_state = {"currentApp": "Maps", "packageName": MAPS, "isEditable": True}
    _, _, plain, _ = _format(IndexedFormatter(), android_tree(), phone_state)
    _, _, jev, _ = _format(JevIndexedFormatter(), android_tree(), phone_state)

    plain_snapshot = normalize_ui_state(_ui_state(plain, phone_state))
    jev_snapshot = normalize_ui_state(_ui_state(jev, phone_state))

    assert jev_snapshot == plain_snapshot
    result = compare_states(jev_snapshot, plain_snapshot, threshold=0.85)
    assert result.matches
    assert result.score == 1.0


def test_build_screen_android():
    screen = build_screen(_android_state(), platform="android")

    assert screen.platform == "android"
    assert screen.package == MAPS
    assert screen.app_name == "Maps"
    assert (screen.width, screen.height) == (1080, 2400)
    assert screen.keyboard_visible

    assert screen.focused is not None
    assert screen.focused.editable and screen.focused.focused
    assert screen.focused.hint == "Search here"
    assert screen.focused.value == ""

    password = next(e for e in screen.elements if e.password)
    assert password.label == "[password field]"
    assert password.value == ""

    assert "Golden Gate Bridge" in screen.visible_text
    assert "Wi-Fi" in screen.visible_text
    assert "hunter2" not in screen.visible_text
    assert "q" not in screen.visible_text
    assert any(e.keyboard for e in screen.elements)
    assert len(screen.fingerprint) == 40


def test_build_screen_clamps_bounds_and_drops_empty_elements():
    tree = _node(
        "android.widget.FrameLayout",
        (0, 0, 1080, 2400),
        [
            _node("android.widget.ScrollView", (0, 200, 1080, 3000), isScrollable=True),
            _node(
                "android.widget.Button", (10, 10, 10, 50), text="zero", isClickable=True
            ),
        ],
    )
    _, _, elements, _ = JevIndexedFormatter().format(tree, {})
    screen = build_screen(_ui_state(elements, {}), platform="android")

    scroll = next(e for e in screen.elements if e.scrollable)
    assert scroll.bounds == (0, 200, 1080, 2400)
    assert all(e.class_name != "Button" for e in screen.elements)


def test_build_screen_ios_local_short_class_names():
    tree = _node(
        "Application",
        (0, 0, 440, 956),
        [
            _node(
                "TextField",
                (16, 100, 420, 140),
                package="com.apple.Maps",
                isFocused=True,
            ),
            _node(
                "SearchField",
                (16, 160, 420, 200),
                package="com.apple.Maps",
                text="Search Maps",
            ),
            _node(
                "SecureTextField",
                (16, 220, 420, 260),
                package="com.apple.Maps",
                text="••••",
            ),
            _node("TextView", (16, 280, 420, 400), package="com.apple.Maps"),
            _node(
                "Button",
                (300, 40, 420, 80),
                package="com.apple.Maps",
                text="Done",
                isClickable=True,
            ),
            _node(
                "Keyboard",
                (0, 600, 440, 956),
                [
                    _node(
                        "Key",
                        (0, 600, 44, 650),
                        package="com.apple.Maps",
                        text="q",
                        isClickable=True,
                    )
                ],
                package="com.apple.Maps",
            ),
        ],
        package="com.apple.Maps",
    )
    phone_state = {
        "currentApp": "Maps",
        "packageName": "com.apple.Maps",
        "keyboardVisible": True,
    }
    _, _, elements, _ = JevIndexedFormatter().format(tree, phone_state)
    screen = build_screen(_ui_state(elements, phone_state, 440, 956), platform="ios")

    by_class = {e.class_name: e for e in screen.elements}
    assert by_class["TextField"].editable
    assert by_class["SearchField"].editable
    assert by_class["TextView"].editable
    assert by_class["SecureTextField"].editable and by_class["SecureTextField"].password
    assert by_class["SecureTextField"].value == ""
    assert not by_class["Button"].editable and by_class["Button"].clickable
    assert by_class["Key"].keyboard and by_class["Keyboard"].keyboard
    assert not by_class["Button"].keyboard
    assert screen.focused is by_class["TextField"]
    assert screen.keyboard_visible


def test_build_screen_ios_local_without_focus_flags_uses_single_input():
    tree = _node(
        "Application",
        (0, 0, 440, 956),
        [
            _node("SearchField", (16, 160, 420, 200), package="com.apple.Maps"),
            _node("Key", (0, 600, 44, 650), package="com.apple.Maps", text="q"),
        ],
    )
    phone_state = {"packageName": "com.apple.Maps", "keyboardVisible": True}
    _, _, elements, _ = JevIndexedFormatter().format(tree, phone_state)
    screen = build_screen(_ui_state(elements, phone_state, 440, 956), platform="ios")

    assert screen.focused is not None and screen.focused.class_name == "SearchField"


IOS_CONTEXT = {"screen_bounds": {"width": 440, "height": 956}}


def _ios_node(class_name, bounds, children=(), **fields):
    """A node shaped like the iOS --local portal's a11y_tree JSON."""
    node = {
        "className": class_name,
        "resourceId": "",
        "text": "",
        "contentDescription": "",
        "packageName": "com.apple.Maps",
        "boundsInScreen": _b(*bounds),
        "isCheckable": False,
        "isChecked": False,
        "isClickable": False,
        "isEnabled": True,
        "isFocusable": False,
        # The WDA fork never reports input focus.
        "isFocused": False,
        "isScrollable": False,
        "isLongClickable": False,
        "isPassword": False,
        "isSelected": False,
        "children": list(children),
    }
    node.update(fields)
    return node


def _ios_local_screen(tree, phone_state, **kwargs):
    filtered = DetailedFilter().filter(copy.deepcopy(tree), IOS_CONTEXT)
    _, _, elements, _ = JevIndexedFormatter().format(filtered, phone_state)
    state = _ui_state(elements, phone_state, 440, 956)
    return build_screen(state, platform="ios", **kwargs)


def _ios_plan(screen, goal='Search for "coffee"'):
    return build_plan(
        goal,
        screen,
        history=[],
        texts=[],
        apps=[],
        capabilities=Capabilities(buttons=frozenset({"home", "enter"})),
    )


def test_build_screen_ios_wda_fork_prefixed_class_names():
    tree = _ios_node(
        "XCUIElementTypeApplication",
        (0, 0, 440, 956),
        [
            _ios_node(
                "XCUIElementTypeSearchField",
                (16, 160, 420, 200),
                text="Search Maps",
                contentDescription="Search Maps",
                isFocusable=True,
            ),
            _ios_node(
                "XCUIElementTypeButton",
                (340, 60, 420, 100),
                text="Cancel",
                contentDescription="Cancel",
                isClickable=True,
            ),
            _ios_node(
                "XCUIElementTypeKeyboard",
                (0, 600, 440, 956),
                [
                    _ios_node(
                        "XCUIElementTypeKey", (0, 600, 44, 650), contentDescription="q"
                    ),
                    _ios_node(
                        "XCUIElementTypeButton",
                        (0, 850, 80, 900),
                        contentDescription="shift",
                        isClickable=True,
                    ),
                    _ios_node(
                        "XCUIElementTypeButton",
                        (340, 900, 440, 956),
                        contentDescription="search",
                        isClickable=True,
                    ),
                ],
            ),
        ],
    )
    phone_state = {
        "currentApp": "Maps",
        "packageName": "com.apple.Maps",
        "keyboardVisible": True,
        "isEditable": False,
    }
    screen = _ios_local_screen(tree, phone_state)

    search = next(e for e in screen.elements if e.class_name.endswith("SearchField"))
    cancel = next(e for e in screen.elements if e.label == "Cancel")
    keyboard = [e for e in screen.elements if e.index > cancel.index]
    assert search.editable and not search.keyboard
    assert len(keyboard) == 4 and all(e.keyboard for e in keyboard)
    assert screen.focused is search
    assert screen.keyboard_visible
    assert {"Search Maps", "Cancel"} <= set(screen.visible_text)
    assert not {"q", "shift", "search"} & set(screen.visible_text)

    plan = _ios_plan(screen)
    assert "TYPE_TEXT" in plan.operations
    assert plan.questions["tap_target"]["criteria"] == {
        str(search.index): f"[{search.index}] Focus text input: Search Maps",
        str(cancel.index): f"[{cancel.index}] Cancel",
    }

    secure = _ios_local_screen(
        _ios_node(
            "XCUIElementTypeApplication",
            (0, 0, 440, 956),
            [
                _ios_node(
                    "XCUIElementTypeSecureTextField",
                    (16, 160, 420, 200),
                    text="••••",
                )
            ],
        ),
        phone_state,
    )
    password = next(e for e in secure.elements if e.editable)
    assert password.password and password.label == "[password field]"
    assert password.value == ""


def test_build_screen_legacy_shape_with_prefixed_types():
    elements = [
        {"index": 0, "type": "XCUIElementTypeTextField", "label": "Email"},
        {"index": 1, "type": "XCUIElementTypeButton", "label": "Go"},
        {"index": 2, "type": "XCUIElementTypeKey", "label": "q"},
    ]
    for i, element in enumerate(elements):
        element["bounds"] = f"0,{i * 50},100,{i * 50 + 40}"
    screen = build_screen(_ui_state(elements, {}, 440, 956), platform="ios")
    field, button, key = screen.elements

    assert field.editable and field.class_name == "TextField"
    assert button.clickable and not button.keyboard
    assert key.keyboard
    assert "q" not in screen.visible_text


def _login_tree(email_text=""):
    return _ios_node(
        "XCUIElementTypeApplication",
        (0, 0, 440, 956),
        [
            _ios_node(
                "XCUIElementTypeTextField",
                (16, 200, 420, 244),
                resourceId="email",
                text=email_text,
                contentDescription="Email",
                isFocusable=True,
            ),
            _ios_node(
                "XCUIElementTypeSecureTextField",
                (16, 260, 420, 304),
                resourceId="password",
                contentDescription="Password",
                isFocusable=True,
            ),
            _ios_node(
                "XCUIElementTypeButton",
                (16, 330, 420, 374),
                text="Log In",
                isClickable=True,
            ),
        ],
    )


def _login_phone(keyboard):
    return {
        "packageName": "com.example.shop",
        "keyboardVisible": keyboard,
        "isEditable": False,
    }


def test_ios_local_tapped_field_is_focused_without_focus_flags():
    before = _ios_local_screen(_login_tree(), _login_phone(False))
    email = next(e for e in before.elements if e.resource_id == "email")
    assert before.focused is None

    after_tap = _ios_local_screen(_login_tree(), _login_phone(True), tapped_input=email)
    assert after_tap.focused is not None
    assert after_tap.focused.resource_id == "email"
    assert "TYPE_TEXT" in _ios_plan(after_tap, "Log in as 'me@example.com'").operations

    # Typing changes the field's label, not its identity.
    typed = _ios_local_screen(
        _login_tree("me@example.com"), _login_phone(True), tapped_input=email
    )
    assert typed.focused is not None and typed.focused.resource_id == "email"


def test_ios_local_never_guesses_between_inputs():
    email = next(
        e
        for e in _ios_local_screen(_login_tree(), _login_phone(False)).elements
        if e.resource_id == "email"
    )
    no_tap = _ios_local_screen(_login_tree(), _login_phone(True))
    assert no_tap.focused is None
    assert "TYPE_TEXT" not in _ios_plan(no_tap).operations

    keyboard_hidden = _ios_local_screen(
        _login_tree(), _login_phone(False), tapped_input=email
    )
    assert keyboard_hidden.focused is None

    moved = copy.copy(email)
    moved.bounds = (16, 400, 420, 444)
    assert (
        _ios_local_screen(_login_tree(), _login_phone(True), tapped_input=moved).focused
        is None
    )

    # The --local focusedElement is a guess by field metadata, not input focus.
    guessed = dict(
        _login_phone(True),
        isEditable=True,
        focusedElement={
            "text": "",
            "className": "XCUIElementTypeTextField",
            "resourceId": "email",
        },
    )
    assert _ios_local_screen(_login_tree(), guessed).focused is None


def test_ios_local_reported_is_editable_is_not_input_focus():
    # Stock-WDA /source state: isEditable and focusedElement name the best
    # editable candidate whether or not it has focus.
    tree = _ios_node(
        "Application",
        (0, 0, 440, 956),
        [
            _ios_node(
                "SearchField",
                (16, 100, 420, 136),
                text="Search",
                contentDescription="Search",
                isFocusable=True,
            ),
            _ios_node("Cell", (16, 180, 420, 224), text="General", isClickable=True),
        ],
    )
    phone_state = {
        "packageName": "com.apple.Preferences",
        "currentApp": "Settings",
        "keyboardVisible": False,
        "isEditable": True,
        "focusedElement": {"text": "Search", "className": "SearchField"},
    }
    screen = _ios_local_screen(tree, phone_state)

    assert screen.focused is None
    assert not screen.keyboard_visible
    plan = _ios_plan(screen, "Type 'hello' into notes")
    assert "TYPE_TEXT" not in plan.operations
    assert "ENTER" not in plan.operations
    assert plan.state["focusedField"] is None


def test_ios_local_placeholder_text_is_a_hint_not_a_value():
    # Contacts' New Contact form over WDA: an empty field reads back as its
    # placeholder, which is also its label.
    def field(name, top, text):
        return _ios_node(
            "XCUIElementTypeTextField",
            (40, top, 400, top + 22),
            resourceId=name,
            text=text,
            contentDescription=name,
            isFocusable=True,
        )

    tree = _ios_node(
        "XCUIElementTypeApplication",
        (0, 0, 440, 956),
        [field("First name", 410, "Ada"), field("Last name", 463, "Last name")],
    )
    before = _ios_local_screen(tree, _login_phone(False))
    last = next(e for e in before.elements if e.resource_id == "Last name")
    screen = _ios_local_screen(tree, _login_phone(True), tapped_input=last)
    fields = {e.resource_id: e for e in screen.elements if e.editable}

    assert (fields["First name"].value, fields["First name"].hint) == ("Ada", "")
    assert (fields["Last name"].value, fields["Last name"].hint) == ("", "Last name")
    plan = _ios_plan(screen, "Type 'Lovelace' as the last name")
    assert plan.state["focusedField"] == {
        "index": str(last.index),
        "label": "Last name",
        "value": "",
    }
    entry = next(e for e in plan.state["elements"] if e["index"] == str(last.index))
    assert entry["label"] == "Focus text input: Last name" and "value" not in entry


LEGACY_LOGIN = """Attributes: Application, pid: 1, label: 'Shop'
Element subtree:
 ->Application, pid: 1, label: 'Shop'
    Window (Main), {{0.0, 0.0}, {393.0, 852.0}}
      TextField, {{16.0, 200.0}, {361.0, 44.0}}, identifier: 'email', placeholderValue: 'Email'
      SecureTextField, {{16.0, 260.0}, {361.0, 44.0}}, identifier: 'password', placeholderValue: 'Password'
      Button, {{16.0, 330.0}, {361.0, 44.0}}, label: 'Log In'
      Key, {{0.0, 600.0}, {39.0, 50.0}}, label: 'q'
"""


def _legacy_focus(focused_element, a11y=LEGACY_LOGIN, **phone):
    phone_state = {
        "packageName": "com.example.shop",
        "keyboardVisible": True,
        "isEditable": True,
        "focusedElement": focused_element,
        **phone,
    }
    return _legacy_screen(phone_state, a11y)[1]


def test_legacy_ios_focused_element_picks_one_of_several_inputs():
    screen = _legacy_focus(
        {"text": "", "className": "TextField", "resourceId": "email"}
    )
    assert screen.focused is not None and screen.focused.resource_id == "email"
    plan = _ios_plan(screen, "Log in with email 'me@example.com'")
    assert "TYPE_TEXT" in plan.operations
    assert plan.state["focusedField"]["index"] == str(screen.focused.index)

    password = _legacy_focus(
        {"text": "", "className": "SecureTextField", "resourceId": "password"}
    )
    assert password.focused is not None and password.focused.password
    assert "TYPE_TEXT" not in _ios_plan(password).operations

    # Focus is reported even when a hardware keyboard hides the software one.
    no_keyboard = _legacy_focus(
        {"text": "", "className": "TextField", "resourceId": "email"},
        a11y=LEGACY_LOGIN.replace("      Key,", "      StaticText,"),
        keyboardVisible=False,
    )
    assert no_keyboard.focused is not None
    assert no_keyboard.focused.resource_id == "email"


def test_legacy_ios_focused_element_never_guesses():
    two_fields = """Attributes: Application, pid: 1, label: 'Shop'
Element subtree:
 ->Application, pid: 1, label: 'Shop'
    Window (Main), {{0.0, 0.0}, {393.0, 852.0}}
      TextField, {{16.0, 200.0}, {361.0, 44.0}}, placeholderValue: 'First name', value: Ada
      TextField, {{16.0, 260.0}, {361.0, 44.0}}, placeholderValue: 'Last name'
"""
    ambiguous = _legacy_focus({"text": "", "className": "TextField"}, two_fields)
    assert ambiguous.focused is None

    by_text = _legacy_focus({"text": "Ada", "className": "TextField"}, two_fields)
    assert by_text.focused is not None and by_text.focused.value == "Ada"

    unknown_id = _legacy_focus(
        {"text": "", "className": "TextField", "resourceId": "nickname"}
    )
    assert unknown_id.focused is None


LEGACY_A11Y = """Attributes: Application, pid: 123, label: 'Settings'
Element subtree:
 ->Application, pid: 123, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {440.0, 956.0}}
      Button, {{340.0, 60.0}, {44.0, 44.0}}, label: 'Edit'
      SearchField, {{17.0, 583.0}, {346.0, 38.0}}, placeholderValue: 'Search'
      Switch, {{300.0, 200.0}, {51.0, 31.0}}, label: 'Airplane Mode', value: 1
      Switch, {{300.0, 260.0}, {51.0, 31.0}}, label: 'Wi-Fi', value: 0
      StaticText, {{20.0, 119.7}, {133.0, 40.7}}, label: 'Settings'
      Key, {{0.0, 700.0}, {40.0, 50.0}}, label: 'q'
"""


class FakeLegacyIOSDriver:
    def __init__(self, phone_state, a11y=LEGACY_A11Y):
        self.phone_state = phone_state
        self.a11y = a11y

    async def get_ui_tree(self):
        return {
            "phone_state": dict(self.phone_state),
            "device_context": {"screen_bounds": {"width": 440, "height": 956}},
            "a11y_tree": [],
            "raw_a11y_tree": self.a11y,
        }


def _legacy_screen(phone_state, a11y=LEGACY_A11Y):
    state = asyncio.run(
        IOSStateProvider(FakeLegacyIOSDriver(phone_state, a11y)).get_state()
    )
    assert all("jev" not in element for element in state.elements)
    return state, build_screen(state, platform="ios")


def test_build_screen_legacy_ios_elements():
    state, screen = _legacy_screen(
        {
            "currentApp": "Settings",
            "packageName": "com.apple.Preferences",
            "keyboardVisible": True,
        }
    )
    by_label = {e.label: e for e in screen.elements}

    assert screen.package == "com.apple.Preferences"
    assert by_label["Edit"].clickable
    assert by_label["Airplane Mode"].checkable and by_label["Airplane Mode"].checked
    assert by_label["Wi-Fi"].checkable and not by_label["Wi-Fi"].checked
    assert not by_label["Settings"].clickable
    assert by_label["q"].keyboard

    search = next(e for e in screen.elements if e.editable)
    assert search.hint == "Search"
    # Indices are the provider's own (0-based, re-sorted) indices.
    assert state.get_element(search.index)["className"] == "SearchField"
    assert screen.focused is search
    assert "Airplane Mode" in screen.visible_text
    assert "1" not in screen.visible_text
    assert "q" not in screen.visible_text


def test_build_screen_legacy_ios_password_field():
    a11y = LEGACY_A11Y.replace(
        "      Key,",
        "      SecureTextField, {{17.0, 300.0}, {346.0, 38.0}}, placeholderValue: 'Password', value: abc\n      Key,",
    )
    _, screen = _legacy_screen({"packageName": "com.apple.Preferences"}, a11y)
    secure = next(e for e in screen.elements if e.password)

    assert secure.label == "[password field]"
    assert secure.value == ""
    assert secure.hint == "Password"
    assert "abc" not in screen.visible_text
    assert screen.focused is None
    # A visible keyboard key implies a keyboard even without phone-state flags.
    assert screen.keyboard_visible


def _inputs_tree(*fields):
    return _node(
        "android.widget.FrameLayout",
        (0, 0, 1080, 2400),
        [
            _node(
                "android.widget.EditText",
                (0, 100 * i, 1080, 100 * i + 90),
                text=f"f{i}",
                **flags,
            )
            for i, flags in enumerate(fields, start=1)
        ],
    )


def _focus(fields, **phone_state):
    phone_state = {"packageName": MAPS, **phone_state}
    _, _, elements, _ = JevIndexedFormatter().format(_inputs_tree(*fields), phone_state)
    return build_screen(_ui_state(elements, phone_state), platform="android")


def test_focus_rule():
    single_focused = _focus([{"isFocused": True}, {}])
    assert single_focused.focused is not None and single_focused.focused.value == "f1"

    two_focused = _focus([{"isFocused": True}, {"isFocused": True}], isEditable=True)
    assert two_focused.focused is None

    single_with_keyboard = _focus([{}], keyboardVisible=True)
    assert single_with_keyboard.focused is not None

    single_reported_editable = _focus([{}], isEditable=True)
    assert single_reported_editable.focused is not None

    single_without_keyboard = _focus([{}])
    assert single_without_keyboard.focused is None
    assert not single_without_keyboard.keyboard_visible

    ambiguous = _focus([{}, {}], keyboardVisible=True)
    assert ambiguous.focused is None
    assert ambiguous.keyboard_visible

    disabled_other = _focus([{}, {"isEnabled": False}], keyboardVisible=True)
    assert disabled_other.focused is not None and disabled_other.focused.value == "f1"

    disabled_focused = _focus([{"isFocused": True, "isEnabled": False}])
    assert disabled_focused.focused is None


def test_fingerprint_tracks_focus_and_input_flags():
    plain = _focus([{}, {}]).fingerprint

    assert _focus([{"isFocused": True}, {}]).fingerprint != plain
    assert _focus([{"isEnabled": False}, {}]).fingerprint != plain
    assert _focus([{}, {}], keyboardVisible=True).fingerprint != plain
    assert _focus([{}, {}], isEditable=True).fingerprint != plain
    assert _focus([{}, {}]).fingerprint == plain


def test_fingerprint_tracks_the_resolved_focus():
    fields = _ios_local_screen(_login_tree(), _login_phone(False)).elements
    email, password = (e for e in fields if e.editable)
    screens = [
        _ios_local_screen(_login_tree(), _login_phone(True), tapped_input=field)
        for field in (None, email, password)
    ]

    assert [scr.focused and scr.focused.resource_id for scr in screens] == [
        None,
        "email",
        "password",
    ]
    assert len({scr.fingerprint for scr in screens}) == 3


def test_fingerprint_is_order_independent_and_tracks_content():
    state = _android_state()
    screen = build_screen(state, platform="android")

    reversed_state = _ui_state(list(reversed(state.elements)), state.phone_state)
    assert (
        build_screen(reversed_state, platform="android").fingerprint
        == screen.fingerprint
    )

    tree = android_tree()
    tree["children"][2]["isChecked"] = False
    changed = build_screen(_android_state(tree), platform="android")
    assert changed.fingerprint != screen.fingerprint

    other_app = build_screen(
        _android_state(phone_state={"packageName": "com.android.settings"}),
        platform="android",
    )
    assert other_app.fingerprint != screen.fingerprint


def test_build_screen_tolerates_empty_state():
    screen = build_screen(_ui_state([], {}, 390, 844), platform="ios")
    assert screen.elements == []
    assert screen.focused is None
    assert screen.package == ""
    assert screen.visible_text == []


LEGACY_MAPS_SEARCH = """Attributes: Application, pid: 1, label: 'Maps'
Element subtree:
 ->Application, pid: 1, label: 'Maps'
    Window (Main), {{0.0, 0.0}, {402.0, 874.0}}
      SearchField, {{16.0, 126.0}, {315.0, 44.0}}, label: 'Apple Maps', value: Golden Gate Bridge
      Image, {{28.0, 138.0}, {20.0, 19.0}}, label: 'Search'
      Button, {{296.0, 137.0}, {20.0, 20.0}}, label: 'Clear text'
      Button, {{76.0, 199.0}, {239.0, 42.0}}, label: 'Golden Gate Bridge, San Francisco'
      Keyboard, {{0.0, 590.0}, {402.0, 284.0}}
        Key, {{4.0, 600.0}, {36.0, 46.0}}, label: 'q'
        Button, {{4.0, 698.0}, {52.0, 54.0}}, label: 'shift'
        Button, {{300.0, 752.0}, {99.0, 54.0}}, label: 'search'
"""


def _legacy_maps(**phone):
    phone_state = {
        "packageName": "com.apple.Maps",
        "currentApp": "Maps",
        "keyboardVisible": True,
        "focusedElement": {"className": "SearchField"},
        **phone,
    }
    return _legacy_screen(phone_state, LEGACY_MAPS_SEARCH)[1]


def test_legacy_ios_keyboard_buttons_are_keyboard_keys():
    screen = _legacy_maps()
    by_label = {e.label: e for e in screen.elements}

    assert by_label["shift"].keyboard and by_label["search"].keyboard
    assert not by_label["Golden Gate Bridge, San Francisco"].keyboard
    assert "shift" not in screen.visible_text and "search" not in screen.visible_text
    assert screen.return_key is by_label["search"]
    assert not by_label["Search"].clickable
    assert by_label["Clear text"].clickable


def test_legacy_ios_enter_taps_the_keyboard_submit_key():
    screen = _legacy_maps()
    plan = build_plan(
        "Open Maps and search for 'Golden Gate Bridge'",
        screen,
        history=[],
        texts=(),
        apps=[],
        capabilities=Capabilities(buttons=frozenset({"home"})),
    )
    taps = plan.questions["tap_target"]["criteria"]

    assert "ENTER" in plan.operations
    assert not any(
        "shift" in label or "search" == label[-6:] for label in taps.values()
    )
    assert plan.targets["enter_key"]["index"] == screen.return_key.index


def test_enter_key_is_not_used_when_the_device_has_an_enter_button():
    screen = _legacy_maps()
    plan = build_plan(
        "Search for 'Golden Gate Bridge'",
        screen,
        history=[],
        texts=(),
        apps=[],
        capabilities=Capabilities(),
    )
    assert "ENTER" in plan.operations
    assert "enter_key" not in plan.targets


def test_no_return_key_without_a_keyboard():
    no_keyboard = LEGACY_MAPS_SEARCH.split("      Keyboard,")[0]
    phone_state = {"packageName": "com.apple.Maps", "keyboardVisible": False}
    screen = _legacy_screen(phone_state, no_keyboard)[1]

    assert screen.return_key is None
    assert not any(e.keyboard for e in screen.elements)


def test_legacy_ios_static_text_views_are_not_inputs():
    a11y = """Attributes: Application, pid: 1, label: 'Settings'
Element subtree:
 ->Application, pid: 1, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {402.0, 874.0}}
      TextView, {{36.0, 268.0}, {307.0, 48.0}}, label: 'Allow content from Apple to be shown when searching.'
      TextView, {{16.0, 400.0}, {370.0, 200.0}}, value: Shopping list
      TextView, {{16.0, 620.0}, {370.0, 100.0}}, placeholderValue: 'Note'
"""
    screen = _legacy_screen({"packageName": "com.apple.Preferences"}, a11y)[1]
    editable = {e.label or e.value or e.hint: e.editable for e in screen.elements}

    assert editable == {
        "Allow content from Apple to be shown when searching.": False,
        "Shopping list": True,
        "Note": True,
    }


LEGACY_HEAD = """Attributes: Application, pid: 1, label: 'Example'
Element subtree:
 ->Application, pid: 1, label: 'Example'
    Window (Main), {{0.0, 0.0}, {402.0, 874.0}}
"""
LEGACY_KEYBOARD = """      Keyboard, {{0.0, 590.0}, {402.0, 284.0}}
        Key, {{4.0, 650.0}, {36.0, 46.0}}, label: 'q'
        Button, {{300.0, 812.0}, {99.0, 54.0}}, label: 'return'
"""


def _legacy_input(a11y, focused_element=None, **phone):
    phone_state = {
        "packageName": "com.example",
        "keyboardVisible": True,
        "isEditable": True,
        "focusedElement": focused_element,
        **phone,
    }
    return _legacy_screen(phone_state, LEGACY_HEAD + a11y)


def _legacy_plan(screen, goal):
    return build_plan(
        goal,
        screen,
        history=[],
        texts=(),
        apps=[],
        capabilities=Capabilities(buttons=frozenset({"home"})),
    )


def _tap_labels(plan):
    return list(plan.questions["tap_target"]["criteria"].values())


def test_legacy_ios_covered_app_button_is_not_the_return_key():
    # A number pad has no return key; the app's Next button sits under it.
    a11y = """      TextField, {{16.0, 200.0}, {370.0, 50.0}}, placeholderValue: 'Phone number', value: 4155550123
      Button, {{16.0, 790.0}, {370.0, 50.0}}, label: 'Next'
      Keyboard, {{0.0, 598.0}, {402.0, 276.0}}
        Key, {{6.0, 604.0}, {126.0, 54.0}}, label: '1'
        Key, {{138.0, 784.0}, {126.0, 54.0}}, label: '0'
"""
    _, screen = _legacy_input(
        a11y, {"text": "4155550123", "className": "TextField", "resourceId": ""}
    )
    next_button = next(e for e in screen.elements if e.label == "Next")

    assert next_button.covered and not next_button.keyboard
    assert screen.return_key is None
    assert "Next" not in screen.visible_text
    plan = _legacy_plan(screen, "Sign up with phone number '4155550123'")
    assert "ENTER" not in plan.operations
    assert not any("Next" in label for label in _tap_labels(plan))


def test_legacy_ios_return_key_ignores_covered_controls_with_its_label():
    # App Store: the tab bar's Search tab sits behind the keyboard's search key.
    a11y = """      SearchField, {{16.0, 110.0}, {370.0, 36.0}}, label: 'Games, Apps and More', value: chess
      TabBar, {{0.0, 791.0}, {402.0, 83.0}}, identifier: 'Tab Bar'
        Button, {{242.0, 792.0}, {78.0, 48.0}}, label: 'Arcade'
        Button, {{322.0, 792.0}, {78.0, 48.0}}, label: 'Search'
      Keyboard, {{0.0, 590.0}, {402.0, 284.0}}
        Key, {{4.0, 650.0}, {36.0, 46.0}}, label: 'q'
        Button, {{300.0, 812.0}, {99.0, 54.0}}, label: 'search'
"""
    _, screen = _legacy_input(
        a11y, {"text": "chess", "className": "SearchField", "resourceId": ""}
    )
    tab = next(e for e in screen.elements if e.label == "Search")
    key = next(e for e in screen.elements if e.label == "search")

    assert tab.covered and not tab.keyboard
    assert key.keyboard and not key.covered
    assert screen.return_key is key
    plan = _legacy_plan(screen, "Search the App Store for 'chess'")
    assert "ENTER" in plan.operations
    assert plan.targets["enter_key"]["index"] == key.index
    assert not any("earch" in label for label in _tap_labels(plan))

    # Among several keyboard members, the key named in lowercase wins.
    accessory = a11y.replace(
        "        Key, {{4.0, 650.0}",
        "        Button, {{300.0, 596.0}, {90.0, 40.0}}, label: 'Search'\n"
        "        Key, {{4.0, 650.0}",
    )
    _, screen = _legacy_input(accessory, {"className": "SearchField"})
    assert screen.return_key is not None and screen.return_key.label == "search"


def test_legacy_ios_row_whose_tap_point_is_under_the_keyboard_is_not_tappable():
    a11y = (
        """      SearchField, {{16.0, 126.0}, {315.0, 44.0}}, label: 'Search', value: coffee
      Cell, {{0.0, 500.0}, {402.0, 60.0}}, label: 'Blue Bottle Coffee'
      Cell, {{0.0, 560.0}, {402.0, 70.0}}, label: 'Philz Coffee'
"""
        + LEGACY_KEYBOARD
    )
    state, screen = _legacy_input(a11y, {"className": "SearchField"})
    by_label = {e.label: e for e in screen.elements}

    assert by_label["Philz Coffee"].covered
    assert not by_label["Blue Bottle Coffee"].covered
    # Partly visible rows are still read.
    assert "Philz Coffee" in screen.visible_text
    plan = _legacy_plan(screen, "Open Philz Coffee")
    targets = plan.targets["tap_target"].values()
    assert by_label["Philz Coffee"].index not in {t["index"] for t in targets}
    assert all(state.get_element_coords(t["index"])[1] < 590 for t in targets)


def test_legacy_ios_covered_field_can_be_the_focused_field():
    form = """      TextField, {{16.0, 200.0}, {370.0, 44.0}}, placeholderValue: 'Email'
      TextField, {{16.0, 700.0}, {370.0, 44.0}}, placeholderValue: 'Referral code'
"""
    hint = {"text": "", "className": "TextField", "resourceId": ""}
    _, before = _legacy_screen({"packageName": "com.example"}, LEGACY_HEAD + form)
    referral = next(e for e in before.elements if e.hint == "Referral code")
    state, _ = _legacy_input(form + LEGACY_KEYBOARD, hint)
    screen = build_screen(state, platform="ios", tapped_input=referral)

    assert screen.focused is not None and screen.focused.hint == "Referral code"
    assert screen.focused.covered
    plan = _legacy_plan(screen, "Use the referral code 'FRIEND10'")
    assert plan.state["focusedField"]["label"] == "Referral code"
    assert "TYPE_TEXT" in plan.operations
    assert not any("Referral" in label for label in _tap_labels(plan))

    # The only field, covered: the hint still finds it.
    code = "      TextField, {{16.0, 700.0}, {370.0, 44.0}}, placeholderValue: 'Code'\n"
    _, screen = _legacy_input(code + LEGACY_KEYBOARD, hint)
    assert screen.focused is not None and screen.focused.hint == "Code"
    assert "TYPE_TEXT" in _legacy_plan(screen, "Enter the code '123456'").operations


def test_legacy_ios_focus_hint_for_another_kind_of_field_is_not_the_single_input():
    title = "      TextField, {{16.0, 130.0}, {370.0, 44.0}}, label: 'Title'\n"
    _, screen = _legacy_input(
        title + LEGACY_KEYBOARD,
        {"text": "", "className": "SecureTextField", "resourceId": ""},
    )
    assert screen.focused is None

    _, screen = _legacy_input(title + LEGACY_KEYBOARD)
    assert screen.focused is not None and screen.focused.label == "Title"


def test_legacy_ios_labeled_empty_text_view_can_be_tapped_and_focused():
    details = """      TextField, {{16.0, 130.0}, {370.0, 44.0}}, label: 'Title', value: Buy milk
      TextView, {{16.0, 200.0}, {370.0, 120.0}}, label: 'Notes'
"""
    _, screen = _legacy_screen({"packageName": "com.example"}, LEGACY_HEAD + details)
    notes = next(e for e in screen.elements if e.label == "Notes")
    assert not notes.editable and notes.clickable
    assert f"[{notes.index}] Notes" in _tap_labels(_legacy_plan(screen, "Add a note"))

    view_focus = {"text": "", "className": "TextView", "resourceId": ""}
    _, screen = _legacy_input(details + LEGACY_KEYBOARD, view_focus)
    assert screen.focused is not None and screen.focused.label == "Notes"
    assert screen.focused.editable
    plan = _legacy_plan(screen, "Add the note 'Use oat milk'")
    assert plan.state["focusedField"]["label"] == "Notes"
    assert "TYPE_TEXT" in plan.operations

    # The only input on a Notes-style screen.
    note = "      TextView, {{0.0, 120.0}, {402.0, 460.0}}, label: 'Note'\n"
    _, screen = _legacy_input(note + LEGACY_KEYBOARD, view_focus)
    assert screen.focused is not None and screen.focused.label == "Note"
    assert "TYPE_TEXT" in _legacy_plan(screen, "Create a note 'Buy milk'").operations

    # Among several text views, the reported identifier picks the focused one.
    two_views = (
        details.replace("label: 'Notes'", "identifier: 'notes', label: 'Notes'")
        + "      TextView, {{16.0, 400.0}, {370.0, 40.0}}, label: 'Synced to iCloud.'\n"
    )
    _, screen = _legacy_input(two_views + LEGACY_KEYBOARD, view_focus)
    assert screen.focused is None
    _, screen = _legacy_input(
        two_views + LEGACY_KEYBOARD, dict(view_focus, resourceId="notes")
    )
    assert screen.focused is not None and screen.focused.label == "Notes"


def test_legacy_ios_identifier_only_text_view_is_an_input():
    a11y = """      TextView, {{16.0, 200.0}, {370.0, 120.0}}, identifier: 'commentInput'
      Button, {{16.0, 340.0}, {370.0, 44.0}}, label: 'Post'
"""
    _, screen = _legacy_screen({"packageName": "com.example"}, LEGACY_HEAD + a11y)
    comment = next(e for e in screen.elements if e.resource_id == "commentInput")

    assert comment.editable
    assert f"[{comment.index}] Focus text input: commentInput" in _tap_labels(
        _legacy_plan(screen, "Comment 'Nice'")
    )


def test_legacy_ios_image_controls_inside_a_field_stay_tappable():
    a11y = """      TextField, {{16.0, 200.0}, {361.0, 44.0}}, placeholderValue: 'Email', value: me@example.com
      Image, {{24.0, 210.0}, {24.0, 24.0}}, label: 'Email'
      SecureTextField, {{16.0, 260.0}, {361.0, 44.0}}, placeholderValue: 'Password', value: abc
      Image, {{340.0, 270.0}, {24.0, 24.0}}, label: 'Show password'
      Button, {{16.0, 330.0}, {361.0, 44.0}}, label: 'Log In'
"""
    _, screen = _legacy_screen({"packageName": "com.example"}, LEGACY_HEAD + a11y)
    eye = next(e for e in screen.elements if e.label == "Show password")
    icon = next(e for e in screen.elements if e.class_name == "Image" and e is not eye)

    assert eye.clickable and not icon.clickable
    assert f"[{eye.index}] Show password" in _tap_labels(
        _legacy_plan(screen, "Show the password")
    )


def test_ios_local_elements_under_the_keyboard_are_covered():
    tree = _ios_node(
        "XCUIElementTypeApplication",
        (0, 0, 440, 956),
        [
            _ios_node(
                "XCUIElementTypeTextField",
                (16, 160, 420, 200),
                contentDescription="Code",
                isFocusable=True,
            ),
            _ios_node(
                "XCUIElementTypeCell",
                (0, 560, 440, 660),
                text="Resend code",
                isClickable=True,
            ),
            _ios_node(
                "XCUIElementTypeButton",
                (16, 860, 420, 910),
                text="Done",
                isClickable=True,
            ),
            _ios_node(
                "XCUIElementTypeKeyboard",
                (0, 600, 440, 956),
                [_ios_node("XCUIElementTypeKey", (0, 610, 44, 660), text="1")],
            ),
        ],
    )
    phone_state = {
        "packageName": "com.example",
        "keyboardVisible": True,
        "isEditable": False,
    }
    screen = _ios_local_screen(tree, phone_state)
    by_label = {e.label: e for e in screen.elements}

    assert by_label["Done"].covered and by_label["Resend code"].covered
    assert not by_label["Code"].covered
    assert screen.return_key is None
    assert screen.focused is by_label["Code"]
    assert "Resend code" in screen.visible_text
    assert "Done" not in screen.visible_text
    assert _ios_plan(screen).questions["tap_target"]["criteria"] == {
        str(
            by_label["Code"].index
        ): f"[{by_label['Code'].index}] Focus text input: Code"
    }


def test_android_single_input_with_keyboard_ignores_the_focus_report_class():
    # Android Portal always sends focusedElement, often with no className.
    for hint in ({"resourceId": ""}, {"className": "android.view.View"}):
        screen = _focus([{}], keyboardVisible=True, focusedElement=hint)
        assert screen.focused is not None
