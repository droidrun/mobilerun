import asyncio
import json

import pytest


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


def test_macro_operator_replays_on_supplied_device(tmp_path):
    from mobilerun.macro import MacroOperator

    device = FakeDevice()
    result = asyncio.run(
        MacroOperator(delay_between_actions=0).execute(
            device, str(write_macro(tmp_path))
        )
    )
    assert result.success
    assert result.steps == 1
    assert device.taps == [(10, 20)]


def test_macro_operator_mismatch_carries_unexecuted_steps(tmp_path):
    from mobilerun.macro import MacroMismatch, MacroOperator

    device = FakeDevice()
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(
            MacroOperator(state_timeout=0).execute(
                device, str(write_macro(tmp_path, True))
            )
        )
    assert caught.value.step_index == 0
    assert caught.value.remaining_actions[0]["action_type"] == "tap"
    assert caught.value.current_state["phone_state"]["package"] == "other"
    assert device.taps == []


def test_mismatch_index_and_remaining_steps_exclude_completed_prefix(tmp_path):
    from mobilerun.macro import MacroMismatch, MacroOperator

    path = write_macro(tmp_path, True)
    data = json.loads(path.read_text())
    data["actions"].insert(0, {"action_type": "tap", "x": 1, "y": 2})
    data["actions"].append({"action_type": "input_text", "text": "never typed"})
    path.write_text(json.dumps(data))
    device = FakeDevice()
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(
            MacroOperator(state_timeout=0, delay_between_actions=0).execute(
                device, str(path)
            )
        )
    assert caught.value.step_index == 1
    assert len(caught.value.remaining_actions) == 2
    assert caught.value.remaining_actions[0]["x"] == 10
    assert device.taps == [(1, 2)]


def test_step_slice_preserves_absolute_mismatch_index(tmp_path):
    from mobilerun.macro import MacroMismatch, MacroOperator

    path = write_macro(tmp_path, True)
    data = json.loads(path.read_text())
    data["actions"].insert(0, {"action_type": "tap", "x": 1, "y": 2})
    path.write_text(json.dumps(data))
    device = FakeDevice()
    with pytest.raises(MacroMismatch) as caught:
        asyncio.run(
            MacroOperator(state_timeout=0, start_from_step=1).execute(device, str(path))
        )
    assert caught.value.step_index == 1
    assert device.taps == []


def test_zero_step_limit_executes_nothing(tmp_path):
    from mobilerun.macro import MacroOperator

    device = FakeDevice()
    result = asyncio.run(
        MacroOperator().execute(device, str(write_macro(tmp_path)), max_steps=0)
    )
    assert result.success
    assert result.steps == 0
    assert device.taps == []
