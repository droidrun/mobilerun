from mobilerun.macro.models import normalize_ui_state
from mobilerun.macro.utils import compare_states


def _state(text="Continue", bounds="10,20,110,60", extra=None):
    element = {
        "index": 4,
        "resourceId": "com.example:id/continue",
        "className": "android.widget.Button",
        "text": text,
        "contentDescription": "",
        "clickable": True,
        "enabled": True,
        "focused": False,
        "bounds": bounds,
    }
    if extra:
        element.update(extra)
    return {
        "elements": [element],
        "phone_state": {
            "package": "com.example",
            "activity": ".MainActivity",
            "battery": 64,
        },
        "screen_width": 400,
        "screen_height": 800,
    }


def test_normalized_matching_ignores_volatile_fields_but_detects_meaningful_changes():
    saved = normalize_ui_state(_state(extra={"index": 99, "visible": True}))
    same_screen = normalize_ui_state(_state(extra={"index": 1, "visible": False}))
    different_screen = normalize_ui_state(_state(text="Delete"))

    assert compare_states(saved, same_screen, threshold=0.85).matches

    result = compare_states(saved, different_screen, threshold=0.85)
    assert not result.matches
    assert "below threshold" in result.reason


def test_normalized_matching_ignores_bounds_movement():
    saved = normalize_ui_state(_state(bounds="10,20,110,60"))
    moved = normalize_ui_state(_state(bounds="50,100,250,180"))

    assert compare_states(saved, moved, threshold=0.85).matches
