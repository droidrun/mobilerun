"""Tests for the iOS a11y+vision coordinate contract.

iOS screenshots are physical pixels while taps and a11y bounds are points,
so the display space is derived from the actual screenshot bytes and
``convert_point`` maps the model's display-space output back to POINTS.
Mirrors tests/unit/test_android_vision_coordinate_contract.py.
"""

import asyncio
from io import BytesIO

from PIL import Image

from mobilerun.core.state.base import _should_resize_model_screenshot
from mobilerun.core.state.ios import IOSStateProvider

POINTS_W, POINTS_H = 440, 956
PIXELS_W, PIXELS_H = 1320, 2868  # 3x physical screenshot

RAW_A11Y = """Attributes: Application, pid: 123, label: 'Settings'
Element subtree:
 ->Application, pid: 123, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {440.0, 956.0}}
      Cell, {{20.0, 200.0}, {400.0, 44.0}}, label: 'General'
      StaticText, {{20.0, 119.7}, {133.0, 40.7}}, label: 'Settings'
"""


def _png(width, height):
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class FakeIOSDriver:
    def __init__(
        self, screenshot_bytes=None, screenshot_error=None, screenshot_plan=None
    ):
        # screenshot_plan: per-call behaviors (bytes or Exception); the last
        # entry repeats. Overrides screenshot_bytes/screenshot_error.
        self._plan = screenshot_plan or (
            [screenshot_error] if screenshot_error else [screenshot_bytes]
        )
        self.screenshot_calls = 0
        self.ui_tree_error = None

    async def _get_ui_tree(self):
        if self.ui_tree_error:
            raise self.ui_tree_error
        return {
            "phone_state": {
                "currentApp": "Settings",
                "packageName": "com.apple.Preferences",
            },
            "device_context": {
                "screen_bounds": {"width": POINTS_W, "height": POINTS_H}
            },
            "a11y_tree": RAW_A11Y,
        }

    async def screenshot(self):
        step = self._plan[min(self.screenshot_calls, len(self._plan) - 1)]
        self.screenshot_calls += 1
        if isinstance(step, Exception):
            raise step
        return step


def _state(provider):
    return asyncio.run(provider.get_state())


def _general_element(state):
    return next(e for e in state.elements if e.get("text") == "General")


def test_resize_flag_pairs_with_agent_gating():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    assert _should_resize_model_screenshot(
        IOSStateProvider(driver, vision_enabled=True)
    )
    assert not _should_resize_model_screenshot(IOSStateProvider(driver))
    assert not _should_resize_model_screenshot(
        IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    )


def test_failed_probe_recovers_and_snapshots_keep_their_contract():
    driver = FakeIOSDriver(
        screenshot_plan=[RuntimeError("probe failed"), _png(PIXELS_W, PIXELS_H)]
    )
    provider = IOSStateProvider(driver, vision_enabled=True)

    fallback = _state(provider)
    assert provider.resize_model_screenshot is False
    assert fallback.coordinate_contract_active is False

    good = _state(provider)
    assert provider.resize_model_screenshot is True
    assert good.coordinate_contract_active is True
    assert good.convert_point(471, 1024) == (220, 478)
    assert fallback.coordinate_contract_active is False

    provider.resize_model_screenshot = False
    assert good.coordinate_contract_active is True
    assert good.convert_point(471, 1024) == (220, 478)
    provider.resize_model_screenshot = True
    assert fallback.coordinate_contract_active is False


def test_tree_failure_returns_inactive_contract():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    driver.ui_tree_error = RuntimeError("tree fetch failed")

    state = _state(provider)
    assert state.coordinate_contract_active is False


def test_non_vision_ios_does_not_require_contract():
    provider = IOSStateProvider(FakeIOSDriver(screenshot_bytes=_png(8, 8)))
    state = _state(provider)

    assert provider.requires_active_contract_for_coords is False
    assert state.convert_point(100, 200) == (100, 200)


def test_normalized_mode_shows_normalized_bounds_and_taps_points():
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    provider = IOSStateProvider(driver, use_normalized=True)
    state = _state(provider)

    element = _general_element(state)
    assert element["bounds"] == "20,200,420,244"
    assert element["displayBounds"] == "45,209,954,255"
    assert "(normalized [0-1000])" in state.formatted_text
    assert "(45,209,954,255)" in state.formatted_text
    assert "(20,200,420,244)" not in state.formatted_text
    # element taps keep points
    assert state.get_element_coords(element["index"]) == (220, 222)
    assert driver.screenshot_calls == 0
