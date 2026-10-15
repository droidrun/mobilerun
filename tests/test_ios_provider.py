import asyncio

from mobilerun.tools.ui.ios_provider import IOSStateProvider

RAW_SETTINGS_A11Y = """Attributes: Application, pid: 123, label: 'Settings'
Element subtree:
 ->Application, pid: 123, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {440.0, 956.0}}
      NavigationBar, {{0.0, 62.0}, {440.0, 106.0}}, identifier: 'Settings'
        StaticText, {{20.0, 119.7}, {133.0, 40.7}}, label: 'Settings'
      SearchField, {{17.0, 583.0}, {346.0, 38.0}}, placeholderValue: 'Search'
"""


class FakeIOSDriver:
    async def get_ui_tree(self):
        return {
            "phone_state": {
                "currentApp": "Settings",
                "packageName": "com.apple.Preferences",
                "keyboardVisible": False,
            },
            "device_context": {
                "screen_bounds": {
                    "width": 440,
                    "height": 956,
                },
            },
            "a11y_tree": [
                {
                    "type": "Application",
                    "text": "Settings",
                    "children": [],
                }
            ],
            "raw_a11y_tree": RAW_SETTINGS_A11Y,
        }


def test_ios_state_provider_reads_raw_a11y_tree_from_normalized_driver_state():
    state = asyncio.run(IOSStateProvider(FakeIOSDriver()).get_state())

    assert state.screen_width == 440
    assert state.screen_height == 956
    assert state.phone_state["currentApp"] == "Settings"
    assert state.phone_state["packageName"] == "com.apple.Preferences"
    assert any(element["className"] == "SearchField" for element in state.elements)
    assert "SearchField" in state.formatted_text


RAW_KEYBOARD_A11Y = """Attributes: Application, pid: 123, label: 'Maps'
Element subtree:
 →Application, pid: 123, label: 'Maps'
    Window (Main), {{0.0, 0.0}, {402.0, 874.0}}
      SearchField, {{16.0, 126.0}, {315.0, 44.0}}, label: 'Search', value: coffee
      Keyboard, {{0.0, 590.0}, {402.0, 284.0}}
        Other, {{0.0, 590.0}, {402.0, 284.0}}, identifier: 'keys'
          Key, {{4.0, 650.0}, {36.0, 46.0}}, label: 'q'
          Button, {{300.0, 812.0}, {99.0, 54.0}}, label: 'search'
      Button, {{16.0, 800.0}, {370.0, 50.0}}, label: 'Search'
    Window, {{0.0, 0.0}, {402.0, 874.0}}
      Button, {{16.0, 60.0}, {40.0, 40.0}}, label: 'Back'
"""


def test_ios_state_provider_marks_elements_nested_under_the_keyboard():
    class KeyboardDriver(FakeIOSDriver):
        async def get_ui_tree(self):
            return dict(await super().get_ui_tree(), raw_a11y_tree=RAW_KEYBOARD_A11Y)

    state = asyncio.run(IOSStateProvider(KeyboardDriver()).get_state())
    by_key = {(e["className"], e["text"]): e for e in state.elements}

    assert {key for key, e in by_key.items() if e.get("in_keyboard")} == {
        ("Other", "keys"),
        ("Key", "q"),
        ("Button", "search"),
    }
    # Only members carry the key; the Keyboard itself and later siblings do not.
    for key in [
        ("SearchField", "Search"),
        ("Keyboard", ""),
        ("Button", "Search"),
        ("Button", "Back"),
    ]:
        assert "in_keyboard" not in by_key[key]
    assert [e["index"] for e in state.elements] == list(range(len(state.elements)))
