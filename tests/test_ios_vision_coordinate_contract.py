"""Tests for the iOS a11y+vision coordinate contract.

iOS screenshots are physical pixels while taps and a11y bounds are points,
so the display space is derived from the actual screenshot bytes and
``convert_point`` maps the model's display-space output back to POINTS.
Mirrors tests/test_android_vision_coordinate_contract.py.
"""

import asyncio
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from mobilerun.agent.utils.actions import _convert_action_point
from mobilerun.tools.helpers.images import fit_dimensions_to_max_side
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import should_resize_model_screenshot

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
    def __init__(self, screenshot_bytes=None, screenshot_error=None, screenshot_plan=None):
        # screenshot_plan: per-call behaviors (bytes or Exception); the last
        # entry repeats. Overrides screenshot_bytes/screenshot_error.
        self._plan = screenshot_plan or (
            [screenshot_error] if screenshot_error else [screenshot_bytes]
        )
        self.screenshot_calls = 0
        self.ui_tree_error = None

    async def get_ui_tree(self):
        if self.ui_tree_error:
            raise self.ui_tree_error
        return {
            "phone_state": {"currentApp": "Settings", "packageName": "com.apple.Preferences"},
            "device_context": {"screen_bounds": {"width": POINTS_W, "height": POINTS_H}},
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


def test_vision_disabled_keeps_legacy_behavior():
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    state = _state(IOSStateProvider(driver))

    assert state.coordinate_scale_x == 1.0
    assert state.coordinate_scale_y == 1.0
    assert "coordinate space" not in state.formatted_text
    assert "displayBounds" not in str(state.elements)
    assert state.convert_point(220, 478) == (220, 478)
    # no extra screenshot traffic in non-vision mode
    assert driver.screenshot_calls == 0


def test_vision_enabled_declares_display_space_and_scales_to_points():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)

    display_w, display_h = fit_dimensions_to_max_side(PIXELS_W, PIXELS_H)
    assert (display_w, display_h) == (943, 2048)
    assert state.coordinate_scale_x == POINTS_W / display_w
    assert state.coordinate_scale_y == POINTS_H / display_h

    # Contract declared in the display space
    assert f"{display_w}x{display_h} coordinate space" in state.formatted_text

    # Model-facing bounds in display space; tap bounds stay points
    element = _general_element(state)
    assert element["bounds"] == "20,200,420,244"
    scale_x, scale_y = state.coordinate_scale_x, state.coordinate_scale_y
    expected_display = ",".join(
        str(round(v / s))
        for v, s in zip((20, 200, 420, 244), (scale_x, scale_y, scale_x, scale_y))
    )
    assert element["displayBounds"] == expected_display
    assert expected_display in state.formatted_text
    assert "(20,200,420,244)" not in state.formatted_text

    # convert_point maps display-space output back to POINTS (hard anchors)
    assert state.convert_point(471, 1024) == (220, 478)
    # element tap path keeps points
    assert state.get_element_coords(element["index"]) == (220, 222)
    assert driver.screenshot_calls == 1


def test_normalized_mode_opts_out_of_the_contract():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    state = _state(provider)

    assert provider.resize_model_screenshot is False
    assert state.coordinate_scale_x == 1.0
    assert "coordinate space" not in state.formatted_text
    assert driver.screenshot_calls == 0


def test_screenshot_failure_falls_back_to_legacy_behavior():
    driver = FakeIOSDriver(screenshot_error=RuntimeError("portal hiccup"))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)

    assert state.coordinate_scale_x == 1.0
    assert state.coordinate_scale_y == 1.0
    assert "coordinate space" not in state.formatted_text
    assert "displayBounds" not in str(state.elements)
    assert state.convert_point(220, 478) == (220, 478)
    # The resize gate must fall back WITH the state: agents capture their
    # screenshot before get_state, and resizing it while this state converts
    # 1:1 would desynchronize the model's space from convert_point.
    assert should_resize_model_screenshot(provider) is False


def test_probe_failure_disables_resize_for_agent_captured_screenshot():
    """The review scenario, in the agents' real order: the agent's own
    capture succeeds, the provider's probe fails — that step must not be
    resized."""
    driver = FakeIOSDriver(screenshot_plan=[
        _png(PIXELS_W, PIXELS_H),          # 1: agent's capture (succeeds)
        RuntimeError("transient hiccup"),  # 2: provider probe (fails)
        _png(PIXELS_W, PIXELS_H),          # 3+: recovery
    ])
    provider = IOSStateProvider(driver, vision_enabled=True)

    agent_screenshot = asyncio.run(driver.screenshot())  # agents capture first
    state = _state(provider)                             # probe fails inside

    assert agent_screenshot is not None
    assert state.coordinate_scale_x == 1.0
    assert should_resize_model_screenshot(provider) is False  # raw image sent

    # Next step recovers: probe succeeds, contract and gate return together
    state2 = _state(provider)
    assert should_resize_model_screenshot(provider) is True
    assert state2.coordinate_scale_x != 1.0
    assert "coordinate space" in state2.formatted_text


def test_ui_tree_failure_also_disables_resize():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    driver.ui_tree_error = RuntimeError("tree fetch failed")

    state = _state(provider)

    assert state.coordinate_scale_x == 1.0
    assert should_resize_model_screenshot(provider) is False


def test_resize_flag_pairs_with_agent_gating():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    assert should_resize_model_screenshot(IOSStateProvider(driver, vision_enabled=True))
    assert not should_resize_model_screenshot(IOSStateProvider(driver))
    assert not should_resize_model_screenshot(
        IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    )


def test_convert_action_point_validates_ios_model_space():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    # display-space coordinates convert to points
    assert _convert_action_point(471, 1024, ctx=ctx) == (220, 478)

    # raw physical-pixel coordinates (the pre-contract failure mode) are
    # outside the 943x2048 display space and get rejected
    try:
        _convert_action_point(1200, 2500, ctx=ctx)
    except ValueError as e:
        assert "943x2048" in str(e)
    else:
        raise AssertionError("expected ValueError for image-space point")


def test_click_at_auto_unmasks_with_vision_contract():
    from mobilerun.agent.droid.droid_agent import _effective_disabled_tools
    from mobilerun.config_manager.config_manager import DEFAULT_DISABLED_TOOLS

    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))

    # vision + contract -> click_at auto-enabled
    provider = IOSStateProvider(driver, vision_enabled=True)
    effective = _effective_disabled_tools(
        list(DEFAULT_DISABLED_TOOLS), provider, vision_enabled=True, explicit=False
    )
    assert "click_at" not in effective
    assert "click_area" in effective  # only click_at auto-unmasks

    # no vision -> stays masked
    provider = IOSStateProvider(driver)
    effective = _effective_disabled_tools(
        list(DEFAULT_DISABLED_TOOLS), provider, vision_enabled=False, explicit=False
    )
    assert "click_at" in effective

    # normalized mode -> stays masked even with vision
    provider = IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    effective = _effective_disabled_tools(
        list(DEFAULT_DISABLED_TOOLS), provider, vision_enabled=True, explicit=False
    )
    assert "click_at" in effective

    # explicit user list is honored verbatim
    provider = IOSStateProvider(driver, vision_enabled=True)
    effective = _effective_disabled_tools(
        ["click_at"], provider, vision_enabled=True, explicit=True
    )
    assert "click_at" in effective


def test_coordinate_actions_refused_when_contract_drops_for_a_step():
    """Registry exposes click_at under vision, but a step whose contract
    dropped (probe failure) must refuse coordinate actions rather than tap
    raw pixels at 1:1 into point space."""
    driver = FakeIOSDriver(screenshot_plan=[
        RuntimeError("probe failed this step"),  # contract drops
        _png(PIXELS_W, PIXELS_H),                # recovers next step
    ])
    provider = IOSStateProvider(driver, vision_enabled=True)

    state = _state(provider)  # fallback state, contract inactive
    assert provider.resize_model_screenshot is False
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    # An in-points coordinate that would otherwise tap fine is still refused,
    # because this step cannot guarantee the model's coordinate space.
    import pytest

    with pytest.raises(ValueError, match="unavailable for this step"):
        _convert_action_point(100, 200, ctx=ctx)

    # Next step recovers the contract -> coordinate actions work again.
    state2 = _state(provider)
    assert provider.resize_model_screenshot is True
    ctx2 = SimpleNamespace(ui=state2, state_provider=provider)
    assert _convert_action_point(471, 1024, ctx=ctx2) == (220, 478)


def test_active_contract_state_allows_coordinate_actions():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    # contract active -> converts, no refusal
    assert _convert_action_point(471, 1024, ctx=ctx) == (220, 478)


def test_non_vision_ios_does_not_require_contract():
    # Without vision the provider never intends the contract, so the guard is
    # inert and (masked-by-default) coordinate paths behave as before.
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    provider = IOSStateProvider(driver)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    assert provider.requires_active_contract_for_coords is False
    assert _convert_action_point(100, 200, ctx=ctx) == (100, 200)


def test_guard_reads_snapshot_not_mutable_provider_flag():
    """Regression: a mid-action get_state re-probe (macro pre-state) can flip
    provider.resize_model_screenshot without updating ctx.ui. The guard and
    model-space validator must follow ctx.ui (the snapshot the tap uses), so a
    provider flip cannot cause a wrong tap or a spurious refusal."""
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    good = _state(provider)  # contract active snapshot
    assert good.coordinate_contract_active is True
    ctx = SimpleNamespace(ui=good, state_provider=provider)

    # provider flag flipped False by a hypothetical macro re-probe -> the tap
    # still uses `good`, so it must NOT be spuriously refused.
    provider.resize_model_screenshot = False
    assert _convert_action_point(471, 1024, ctx=ctx) == (220, 478)

    # Inverse: a fallback snapshot must be refused even if the provider flag
    # currently reads True (re-probe succeeded after the step's snapshot).
    drop_driver = FakeIOSDriver(screenshot_error=RuntimeError("probe failed"))
    drop_provider = IOSStateProvider(drop_driver, vision_enabled=True)
    fallback = _state(drop_provider)
    assert fallback.coordinate_contract_active is False
    drop_provider.resize_model_screenshot = True  # mimic later re-probe success
    import pytest

    with pytest.raises(ValueError, match="unavailable for this step"):
        _convert_action_point(
            100, 200, ctx=SimpleNamespace(ui=fallback, state_provider=drop_provider)
        )


def test_swipe_refused_on_dropped_contract_step():
    """swipe (default-enabled) routes through the same guard."""
    import pytest

    from mobilerun.agent.utils.actions import swipe

    driver = FakeIOSDriver(screenshot_error=RuntimeError("probe failed"))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider, driver=driver, macro_recorder=None)

    result = asyncio.run(swipe([100, 200], [100, 600], ctx=ctx))
    assert result.success is False
    assert "unavailable for this step" in result.summary


def test_click_at_end_to_end_refused_then_works(monkeypatch):
    """End-to-end through the async click_at wrapper: refused on a dropped
    contract, taps on an active one. Taps are stubbed to avoid a real driver."""
    taps = []

    class _Driver(FakeIOSDriver):
        async def tap(self, x, y):
            taps.append((x, y))

    # active contract -> taps in points
    driver = _Driver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    from mobilerun.agent.utils.actions import click_at

    ctx = SimpleNamespace(ui=state, state_provider=provider, driver=driver, macro_recorder=None)
    ok = asyncio.run(click_at(471, 1024, ctx=ctx))
    assert ok.success and taps == [(220, 478)]

    # dropped contract -> refused, no tap
    taps.clear()
    drop = _Driver(screenshot_error=RuntimeError("probe failed"))
    drop_provider = IOSStateProvider(drop, vision_enabled=True)
    drop_state = _state(drop_provider)
    ctx2 = SimpleNamespace(ui=drop_state, state_provider=drop_provider, driver=drop, macro_recorder=None)
    refused = asyncio.run(click_at(100, 200, ctx=ctx2))
    assert refused.success is False and taps == []


def test_empty_state_from_ui_tree_failure_refuses_coordinate_actions():
    """ui_tree fetch failure returns the empty state (contract inactive);
    coordinate actions must be refused there too."""
    import pytest

    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    driver.ui_tree_error = RuntimeError("tree fetch failed")
    state = _state(provider)

    assert state.coordinate_contract_active is False
    with pytest.raises(ValueError, match="unavailable for this step"):
        _convert_action_point(
            10, 20, ctx=SimpleNamespace(ui=state, state_provider=provider)
        )
