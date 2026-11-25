"""Offline macro replay device and recording builder."""

import json

__all__ = ["FakeDevice", "write_macro"]


class FakeDevice:
    state = None

    def __init__(self):
        self.taps = []

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def _get_ui_tree(self):
        return {"phone_state": {"package": "other"}, "a11y_tree": []}


def write_macro(tmp_path, guarded=False):
    action = {"action_type": "tap", "x": 10, "y": 20}
    if guarded:
        action["pre_state"] = {
            "phone_state": {"package": "expected"},
            "nodes": [{"text": "Continue"}],
        }
    path = tmp_path / "macro.json"
    path.write_text(
        json.dumps(
            {
                "macro_schema_version": "2.0",
                "description": "finish task",
                "actions": [action],
            }
        )
    )
    return path
