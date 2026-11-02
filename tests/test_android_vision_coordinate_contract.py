"""Tests for the a11y+vision coordinate contract (issue #350).

When vision is enabled on AndroidStateProvider, the screenshot the agents
attach is resized (with a labeled grid) into a deterministic display space;
the provider declares that space in formatted_text, emits model-facing
bounds in it, and convert_point scales the model's coordinates back to
native device pixels. Element-index taps keep using native bounds.
"""

import asyncio
import inspect
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from mobilerun.agent.utils import actions as actions_module
from mobilerun.agent.utils.actions import _convert_action_point
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.helpers.images import (
    fit_dimensions_to_max_side,
    image_dimensions,
    resize_image_to_max_side_with_grid,
)
from mobilerun.tools.ui.provider import (
    AndroidStateProvider,
    StateProvider,
    should_resize_model_screenshot,
)
from mobilerun.tools.ui.screenshot_provider import ScreenshotOnlyStateProvider

NATIVE_W, NATIVE_H = 1080, 2400


def _combined_data(width=NATIVE_W, height=NATIVE_H):
    return {
        "a11y_tree": {
            "boundsInScreen": {"left": 0, "top": 0, "right": width, "bottom": height},
            "className": "android.widget.FrameLayout",
            "text": "root",
            "children": [
                {
                    "boundsInScreen": {
                        "left": 221,
                        "top": 409,
                        "right": 600,
                        "bottom": 460,
                    },
                    "className": "android.widget.TextView",
                    "text": "Services & preferences",
                    "children": [],
                }
            ],
        },
        "device_context": {"screen_bounds": {"width": width, "height": height}},
        "phone_state": {
            "currentApp": "Settings",
            "packageName": "com.android.settings",
        },
    }


def _provider(width=NATIVE_W, height=NATIVE_H, **kwargs):
    driver = SimpleNamespace(
        get_ui_tree=AsyncMock(return_value=_combined_data(width, height))
    )
    return AndroidStateProvider(
        driver,
        tree_filter=SimpleNamespace(filter=lambda tree, ctx: tree),
        tree_formatter=IndexedFormatter(),
        **kwargs,
    )


def _get_state(provider):
    return asyncio.run(provider.get_state())


def test_vision_disabled_keeps_legacy_behavior():
    state = _get_state(_provider())

    assert state.coordinate_scale_x == 1.0
    assert state.coordinate_scale_y == 1.0
    assert "coordinate space" not in state.formatted_text
    assert "displayBounds" not in str(state.elements)
    assert "221,409,600,460" in state.formatted_text
    assert state.convert_point(540, 1200) == (540, 1200)


def test_vision_enabled_declares_display_space_and_scales_back():
    state = _get_state(_provider(vision_enabled=True))

    display_w, display_h = fit_dimensions_to_max_side(NATIVE_W, NATIVE_H)
    assert (display_w, display_h) == (922, 2048)
    assert state.coordinate_scale_x == NATIVE_W / display_w
    assert state.coordinate_scale_y == NATIVE_H / display_h

    # Contract is declared, in display space
    assert f"{display_w}x{display_h} coordinate space" in state.formatted_text
    # Model-facing bounds are display-space; native bounds no longer leak there
    assert "189,349,512,393" in state.formatted_text
    assert "221,409,600,460" not in state.formatted_text

    # convert_point maps display-space model output back to native pixels
    x, y = state.convert_point(461, 371)  # display center of the element
    assert abs(x - 540) <= 2 and abs(y - 434) <= 2

    # Element-index taps keep native coordinates (the real-tap path)
    element = state.get_element(2)
    assert element["bounds"] == "221,409,600,460"
    assert element["displayBounds"] == "189,349,512,393"
    assert state.get_element_coords(2) == (410, 434)


def test_policy_declares_model_effective_dims_for_anthropic_standard():
    # issue #365: a standard-budget Anthropic model grounds at 706x1568, so the
    # contract must declare 706x1568 (not the 2048 default) and scale back from it.
    from mobilerun.agent.utils.vision_sizing import VisionResizePolicy

    provider = _provider(
        vision_enabled=True,
        vision_resize_policy=VisionResizePolicy(["claude-sonnet-4-6"]),
    )
    state = _get_state(provider)

    assert (state.model_screenshot_width, state.model_screenshot_height) == (706, 1568)
    assert provider.model_screenshot_width == 706
    assert state.coordinate_scale_x == NATIVE_W / 706
    assert state.coordinate_scale_y == NATIVE_H / 1568
    assert "706x1568 coordinate space" in state.formatted_text
    # ~center of the 706x1568 image maps back to the native screen center.
    x, y = state.convert_point(353, 784)
    assert abs(x - 540) <= 4 and abs(y - 1200) <= 6


def test_no_policy_keeps_2048_default_dims():
    state = _get_state(_provider(vision_enabled=True))
    assert (state.model_screenshot_width, state.model_screenshot_height) == (922, 2048)


def test_resize_helper_resizes_to_provider_dims():
    from mobilerun.agent.utils.vision_sizing import VisionResizePolicy
    from mobilerun.tools.ui.provider import resize_model_screenshot_with_grid

    provider = _provider(
        vision_enabled=True,
        vision_resize_policy=VisionResizePolicy(["claude-sonnet-4-6"]),
    )
    _get_state(provider)
    buf = BytesIO()
    Image.new("RGB", (NATIVE_W, NATIVE_H), (10, 20, 30)).save(buf, "PNG")
    out = resize_model_screenshot_with_grid(provider, buf.getvalue())
    assert image_dimensions(out) == (706, 1568)


def test_small_screens_keep_scale_one_but_still_declare_space():
    state = _get_state(_provider(height=1920, vision_enabled=True))

    assert state.coordinate_scale_x == 1.0
    assert state.coordinate_scale_y == 1.0
    assert "1080x1920 coordinate space" in state.formatted_text
    assert "scaled down" not in state.formatted_text
    assert "displayBounds" not in str(state.elements)


def test_normalized_mode_opts_out_of_the_contract():
    provider = _provider(vision_enabled=True, use_normalized=True)
    state = _get_state(provider)

    assert provider.resize_model_screenshot is False
    assert state.coordinate_scale_x == 1.0
    assert "coordinate space" not in state.formatted_text


def test_resize_flags_pair_providers_with_agent_gating():
    # The providers declare the flag…
    assert StateProvider.resize_model_screenshot is False
    assert ScreenshotOnlyStateProvider.resize_model_screenshot is True
    assert _provider(vision_enabled=True).resize_model_screenshot is True
    assert _provider().resize_model_screenshot is False

    # …and every agent that attaches screenshots gates its resize on the
    # shared helper. (This pairing is what makes coordinate_scale valid.)
    import mobilerun.agent.executor.executor_agent as executor_agent
    import mobilerun.agent.fast_agent.fast_agent as fast_agent
    import mobilerun.agent.manager.manager_agent as manager_agent
    import mobilerun.agent.manager.stateless_manager_agent as stateless_manager_agent

    for module in (fast_agent, executor_agent, manager_agent, stateless_manager_agent):
        source = inspect.getsource(module)
        assert "resize_model_screenshot_async" in source, module.__name__


def test_missing_screen_bounds_disables_resize_for_that_state():
    """When a state arrives without screen bounds, no contract can be
    declared — the agents must not resize that step's screenshot either,
    or the model's space desynchronizes from convert_point."""
    provider = _provider(vision_enabled=True)
    no_bounds = _combined_data()
    no_bounds["device_context"]["screen_bounds"] = {}
    provider.driver.get_ui_tree.return_value = no_bounds

    state = _get_state(provider)

    assert state.coordinate_scale_x == 1.0
    assert "coordinate space" not in state.formatted_text
    assert should_resize_model_screenshot(provider) is False

    # Bounds return → contract and resize gate come back together
    provider.driver.get_ui_tree.return_value = _combined_data()
    state2 = _get_state(provider)
    assert state2.coordinate_scale_x != 1.0
    assert should_resize_model_screenshot(provider) is True


def test_hybrid_vision_zero_bounds_refuses_click_at():
    from mobilerun.agent.utils.actions import click_at

    provider = _provider(width=0, height=0, vision_enabled=True)
    provider.driver.tap = AsyncMock()
    state = _get_state(provider)
    ctx = SimpleNamespace(
        ui=state,
        state_provider=provider,
        driver=provider.driver,
        macro_recorder=None,
    )

    assert state.coordinate_contract_active is False
    result = asyncio.run(click_at(10, 20, ctx=ctx))

    assert result.success is False
    assert "unavailable for this step" in result.summary
    provider.driver.tap.assert_not_awaited()


def test_legacy_injected_screenshot_providers_keep_getting_resized():
    """Injected providers that only implement the pre-existing
    ``requires_coordinate_tools`` contract must still get resized screenshots:
    the tool registry, prompts, and validation all still treat them as
    screenshot-coordinate mode."""

    class LegacyScreenshotProvider(StateProvider):
        requires_coordinate_tools = True

    assert should_resize_model_screenshot(LegacyScreenshotProvider(SimpleNamespace()))

    class DuckTypedLegacyProvider:
        requires_coordinate_tools = True

    assert should_resize_model_screenshot(DuckTypedLegacyProvider())

    # Non-coordinate providers are unaffected
    assert not should_resize_model_screenshot(StateProvider(SimpleNamespace()))
    assert not should_resize_model_screenshot(_provider())
    assert not should_resize_model_screenshot(SimpleNamespace())

    # And the new contract still activates it
    assert should_resize_model_screenshot(_provider(vision_enabled=True))
    assert should_resize_model_screenshot(
        ScreenshotOnlyStateProvider(SimpleNamespace())
    )


def test_provider_scale_matches_what_agents_actually_send():
    """The invariant that PR #355 lacked: the provider's scale must describe
    the image the agents really attach."""
    buf = BytesIO()
    Image.new("RGB", (NATIVE_W, NATIVE_H)).save(buf, format="PNG")
    sent = resize_image_to_max_side_with_grid(buf.getvalue())
    sent_w, sent_h = image_dimensions(sent)

    state = _get_state(_provider(vision_enabled=True))
    assert sent_w == round(NATIVE_W / state.coordinate_scale_x)
    assert sent_h == round(NATIVE_H / state.coordinate_scale_y)


def _action_ctx(state, provider):
    return SimpleNamespace(ui=state, state_provider=provider)


def test_convert_action_point_validates_model_space():
    provider = _provider(vision_enabled=True)
    state = _get_state(provider)
    ctx = _action_ctx(state, provider)

    # In-space coordinates convert to native pixels
    assert _convert_action_point(461, 1024, ctx=ctx) == (540, 1200)

    # Out-of-space coordinates (native-space habits) are rejected with a
    # message instead of tapping off-screen
    try:
        _convert_action_point(1000, 2300, ctx=ctx)
    except ValueError as e:
        assert "922x2048" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-space point")


def test_convert_action_point_unchanged_without_contract():
    provider = _provider()
    state = _get_state(provider)
    ctx = _action_ctx(state, provider)

    assert _convert_action_point(1000, 2300, ctx=ctx) == (1000, 2300)


def test_validation_helper_is_wired_into_convert():
    source = inspect.getsource(actions_module._convert_action_point)
    assert "_validate_model_space_point" in source
