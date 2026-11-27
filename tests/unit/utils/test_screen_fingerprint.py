import pytest


def test_fingerprint_normalizes_camelcase_and_legacy_snapshots():
    from mobilerun.utils.screen_fingerprint import screen_fingerprint

    raw = {
        "phone_state": {"packageName": "example", "activityName": "main"},
        "a11y_tree": {
            "children": [
                {
                    "resourceId": "example:id/edit",
                    "className": "android.widget.EditText",
                    "text": "secret",
                    "isEditable": True,
                },
                {"className": "android.widget.Button", "text": "Save"},
                {"className": "android.widget.Button", "text": "Discard"},
            ]
        },
    }
    older = {
        "phone_state": {"package": "example", "activity": "main"},
        "nodes": [
            {"class": "Button", "text": "Save now"},
            {
                "resource_id": "example:id/edit",
                "class": "EditText",
                "value": "another-secret",
            },
        ],
    }
    observed = screen_fingerprint(raw)
    assert observed == screen_fingerprint(older)
    assert observed.model_dump() == {
        "app": "example",
        "activity": "main",
        "elements": ("class:Button", "id:example:id/edit"),
    }


def test_fingerprint_ignores_text_only_nodes_and_keeps_app_with_empty_tree():
    from mobilerun.models import Fingerprint
    from mobilerun.utils.screen_fingerprint import screen_fingerprint

    assert screen_fingerprint({}) == Fingerprint(app=None, activity=None, elements=())
    assert screen_fingerprint(
        {"packageName": "example", "children": [{"text": "secret"}]}
    ) == Fingerprint(
        app="example",
        activity=None,
        elements=(),
    )


@pytest.mark.parametrize(
    "app,activity,elements,threshold,matched",
    [
        ("app", "main", ("a", "b", "c"), 2 / 3, True),
        ("app", "main", ("a", "b", "c"), 0.7, False),
        ("app", "other", ("a", "b"), 0.85, False),
        ("other", "main", ("a", "b"), 0.85, False),
    ],
)
def test_fingerprint_match_requires_same_app_and_activity(
    app, activity, elements, threshold, matched
):
    from mobilerun.models import Fingerprint
    from mobilerun.utils.screen_fingerprint import fingerprint_matches

    expected = Fingerprint(app="app", activity="main", elements=("a", "b"))
    observed = Fingerprint(app=app, activity=activity, elements=elements)
    assert fingerprint_matches(expected, observed, threshold) is matched


def test_fingerprint_match_rejects_missing_observation_and_matches_app_only():
    from mobilerun.models import Fingerprint
    from mobilerun.utils.screen_fingerprint import fingerprint_matches

    empty = Fingerprint(app=None, activity=None, elements=())
    app_only = Fingerprint(app="example", activity=None, elements=())
    assert not fingerprint_matches(empty, empty, 0.85)
    assert not fingerprint_matches(app_only, None, 0.85)
    assert fingerprint_matches(app_only, app_only, 0.85)
