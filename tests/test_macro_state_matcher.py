from mobilerun.macro.matcher import compare_states
from mobilerun.macro.state import normalize_ui_state


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


def _portal_state(package, activity, texts, resource_ids=True):
    # Shape of a real Portal state: the app is reported as packageName/activityName.
    return {
        "elements": [
            {
                "resourceId": f"{package}:id/title" if resource_ids else "",
                "className": "android.widget.TextView",
                "text": text,
                "contentDescription": "",
                "clickable": True,
                "enabled": True,
                "focused": False,
                "bounds": "0,0,100,50",
            }
            for text in texts
        ],
        "phone_state": {
            "currentApp": "Settings",
            "packageName": package,
            "activityName": activity,
            "keyboardVisible": False,
        },
        "screen_width": 720,
        "screen_height": 1600,
    }


_ROWS = [f"Row {i}" for i in range(20)]
# Same screen with one row changed: high node overlap, but not identical.
_ROWS_ONE_CHANGED = _ROWS[:-1] + ["Row changed"]
_SETTINGS = ("com.android.settings", "com.android.settings.SubSettings")


def test_portal_package_and_activity_are_normalized():
    snapshot = normalize_ui_state(_portal_state(*_SETTINGS, _ROWS))

    assert snapshot["phone_state"] == {
        "package": "com.android.settings",
        "activity": "com.android.settings.SubSettings",
    }


def test_similar_screen_in_a_different_app_does_not_match():
    # Without resource ids, the elements alone can't tell the two apps apart.
    saved = normalize_ui_state(_portal_state(*_SETTINGS, _ROWS, resource_ids=False))
    other_app = normalize_ui_state(
        _portal_state(
            "com.example.other",
            "com.example.other.Main",
            _ROWS_ONE_CHANGED,
            resource_ids=False,
        )
    )

    assert not compare_states(saved, other_app, threshold=0.85).matches


def test_similar_screen_in_the_same_app_still_matches():
    saved = normalize_ui_state(_portal_state(*_SETTINGS, _ROWS))
    current = normalize_ui_state(_portal_state(*_SETTINGS, _ROWS_ONE_CHANGED))

    assert compare_states(saved, current, threshold=0.85).matches


def test_macro_recorded_without_app_still_matches_a_live_state():
    # Macros recorded before Portal's keys were read have no package/activity.
    saved = normalize_ui_state(_portal_state(*_SETTINGS, _ROWS))
    saved["phone_state"] = {"package": None, "activity": None}
    current = normalize_ui_state(_portal_state(*_SETTINGS, _ROWS_ONE_CHANGED))

    assert compare_states(saved, current, threshold=0.85).matches
