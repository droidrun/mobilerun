import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.agent.utils import actions
from mobilerun.agent.utils.actions import click, open_app, type_secret, type_text, wait
from mobilerun.credential_manager.secret_gate import SecretRunState
from mobilerun.macro.recorder import MacroRecorder


class FakeDriver:
    def __init__(self):
        self.taps = []
        self.inputs = []
        self.log = []

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text, clear=False):
        self.inputs.append((text, clear))
        return True

    async def start_app(self, package, activity=None):
        self.log.append(
            {"action_type": "start_app", "package": package, "activity": activity}
        )
        return f"App started: {package}"


class FakeUI:
    elements = [
        {
            "index": 7,
            "resourceId": "com.example:id/name",
            "className": "android.widget.EditText",
            "text": "",
            "bounds": "10,20,110,60",
        }
    ]
    phone_state = {"package": "com.example", "activity": ".MainActivity"}
    screen_width = 400
    screen_height = 800

    def get_element_coords(self, index):
        return (60, 40)

    def get_element_info(self, index):
        return {
            "text": "",
            "className": "android.widget.EditText",
            "type": "input",
        }


class SequencedStateProvider:
    def __init__(self, states):
        self.states = list(states)

    async def get_state(self):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


class FakeCredentialManager:
    async def resolve_key(self, secret_id):
        assert secret_id == "login_password"
        return "super-secret-password"


def test_wait_records_first_class_macro_action():
    recorder = MacroRecorder()
    ctx = SimpleNamespace(macro_recorder=recorder, ui=FakeUI())

    result = asyncio.run(wait(1.25, ctx=ctx))

    assert result.success
    assert recorder.actions[0]["action_type"] == "wait"
    assert recorder.actions[0]["duration"] == 1.25
    assert recorder.actions[0]["pre_state"]["nodes"]


def test_element_click_records_pre_state_without_target_hint():
    recorder = MacroRecorder()
    ctx = SimpleNamespace(driver=FakeDriver(), ui=FakeUI(), macro_recorder=recorder)

    result = asyncio.run(click(7, ctx=ctx))

    assert result.success
    action = recorder.actions[0]
    assert action["action_type"] == "tap"
    assert action["x"] == 60
    assert action["y"] == 40
    assert action["pre_state"]["nodes"]
    assert "target_hint" not in action


def test_type_text_with_index_records_input_with_refreshed_pre_state():
    first_ui = FakeUI()
    focused_ui = FakeUI()
    focused_ui.elements = [
        {
            "index": 7,
            "resourceId": "com.example:id/name",
            "className": "android.widget.EditText",
            "text": "",
            "focused": True,
            "bounds": "10,20,110,60",
        }
    ]

    recorder = MacroRecorder()
    driver = FakeDriver()
    ctx = SimpleNamespace(
        driver=driver,
        ui=first_ui,
        macro_recorder=recorder,
        state_provider=SequencedStateProvider([first_ui, focused_ui]),
    )

    result = asyncio.run(type_text("hello", index=7, clear=True, ctx=ctx))

    assert result.success
    assert [action["action_type"] for action in recorder.actions] == [
        "tap",
        "input_text",
    ]
    assert recorder.actions[0]["pre_state"]["nodes"][0]["focused"] is None
    assert "target_hint" not in recorder.actions[0]
    assert recorder.actions[1]["pre_state"]["nodes"][0]["focused"] is True
    assert "target_hint" not in recorder.actions[1]


def test_type_secret_records_replayable_secret_action_not_placeholder():
    first_ui = FakeUI()
    focused_ui = FakeUI()
    focused_ui.elements = [
        {
            "index": 7,
            "resourceId": "com.example:id/name",
            "className": "android.widget.EditText",
            "text": "",
            "focused": True,
            "bounds": "10,20,110,60",
        }
    ]

    recorder = MacroRecorder()
    driver = FakeDriver()
    ctx = SimpleNamespace(
        driver=driver,
        ui=first_ui,
        macro_recorder=recorder,
        credential_manager=FakeCredentialManager(),
        state_provider=SequencedStateProvider([first_ui, focused_ui]),
        secret_run=SecretRunState(),
    )

    result = asyncio.run(type_secret("login_password", 7, ctx=ctx))

    assert result.success
    assert driver.inputs == [("super-secret-password", False)]
    assert [action["action_type"] for action in recorder.actions] == [
        "tap",
        "type_secret",
    ]
    secret_action = recorder.actions[1]
    assert secret_action["secret_id"] == "login_password"
    assert secret_action["clear"] is False
    assert "text" not in secret_action
    assert "target_hint" not in recorder.actions[0]
    assert "target_hint" not in secret_action
    assert secret_action["pre_state"]["nodes"][0]["focused"] is True
    serialized = str(recorder.actions)
    assert "super-secret-password" not in serialized


def test_open_app_promotes_recording_driver_start_app_log(monkeypatch):
    class FakeAppStarter:
        def __init__(self, driver, llm, timeout, stream, verbose):
            self.driver = driver

        async def run(self, app_description):
            assert app_description == "Settings"
            return await self.driver.start_app("com.android.settings")

    monkeypatch.setattr(actions, "AppStarter", FakeAppStarter)

    recorder = MacroRecorder()
    driver = FakeDriver()
    ctx = SimpleNamespace(
        driver=driver,
        ui=FakeUI(),
        macro_recorder=recorder,
        app_opener_llm=object(),
        streaming=False,
    )

    result = asyncio.run(open_app("Settings", ctx=ctx))

    assert result.success
    assert recorder.actions[0]["action_type"] == "start_app"
    assert recorder.actions[0]["package"] == "com.android.settings"
    assert recorder.actions[0]["pre_state"]["nodes"]


def test_sequential_actions_record_refreshed_current_pre_state():
    first_ui = FakeUI()
    second_ui = FakeUI()
    second_ui.elements = [
        {
            "index": 1,
            "resourceId": "com.example:id/after_click",
            "className": "android.widget.TextView",
            "text": "After click",
            "bounds": "0,0,100,100",
        }
    ]

    recorder = MacroRecorder()
    ctx = SimpleNamespace(
        driver=FakeDriver(),
        ui=first_ui,
        macro_recorder=recorder,
        state_provider=SequencedStateProvider([first_ui, second_ui]),
    )

    async def run():
        click_result = await click(7, ctx=ctx)
        wait_result = await wait(0, ctx=ctx)
        return click_result, wait_result

    click_result, wait_result = asyncio.run(run())

    assert click_result.success
    assert wait_result.success
    assert recorder.actions[0]["pre_state"]["nodes"][0]["resource_id"] == (
        "com.example:id/name"
    )
    assert recorder.actions[1]["pre_state"]["nodes"][0]["resource_id"] == (
        "com.example:id/after_click"
    )


def _png_header(width, height):
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


class ScreenshotDriver(FakeDriver):
    def __init__(self, size, input_size=None):
        super().__init__()
        self.size = size
        if input_size is not None:
            self.input_coordinate_size = self._input_size(input_size)

    @staticmethod
    def _input_size(input_size):
        async def size(width, height):
            return input_size

        return size

    async def screenshot(self):
        return _png_header(*self.size)


@pytest.mark.parametrize(
    ("use_normalized", "input_size", "point", "screen", "tap"),
    [
        (False, None, (954, 2047), (1080, 2316), (1079, 2315)),
        (True, None, (1000, 1000), (1080, 2316), (1079, 2315)),
        (False, (440, 956), (942, 2047), (440, 956), (440, 956)),
    ],
    ids=["pixels", "normalized", "ios-points"],
)
def test_screenshot_only_taps_record_the_device_screen_size(
    use_normalized, input_size, point, screen, tap
):
    from mobilerun.tools.ui.screenshot_provider import ScreenshotOnlyStateProvider

    size = (1320, 2868) if input_size else (1080, 2316)
    driver = ScreenshotDriver(size, input_size)
    provider = ScreenshotOnlyStateProvider(driver, use_normalized=use_normalized)
    recorder = MacroRecorder()
    ctx = SimpleNamespace(
        driver=driver,
        ui=asyncio.run(provider.get_state()),
        state_provider=provider,
        macro_recorder=recorder,
    )

    result = asyncio.run(actions.click_at(*point, ctx=ctx))

    assert result.success
    action = recorder.actions[0]
    assert (action["x"], action["y"]) == tap
    assert action["screen"] == {"width": screen[0], "height": screen[1]}
    assert action["pre_state"]["screen"] == action["screen"]
