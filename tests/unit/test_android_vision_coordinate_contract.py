"""Tests for the a11y+vision coordinate contract (issue #350).

When vision is enabled on AndroidStateProvider, the screenshot the agents
attach is resized (with a labeled grid) into a deterministic display space;
the provider declares that space in formatted_text, emits model-facing
bounds in it, and convert_point scales the model's coordinates back to
native device pixels. Element-index taps keep using native bounds.
"""

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from mobilerun.core.state.android import AndroidStateProvider
from mobilerun.core.state.base import (
    StateProvider,
    _resize_model_screenshot_with_grid,
    _should_resize_model_screenshot,
)
from mobilerun.core.state.formatters import IndexedFormatter
from mobilerun.core.state.screenshot import (
    ScreenshotOnlyStateProvider,
)
from mobilerun.utils.images import (
    image_dimensions,
    resize_image_to_max_side_with_grid,
)

NATIVE_W, NATIVE_H = 1080, 2400


def test_screenshot_fallback_uses_configured_max_side(monkeypatch):
    monkeypatch.setenv("MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE", "500")
    image = BytesIO()
    Image.new("RGB", (1000, 2000)).save(image, format="PNG")
    geometry = SimpleNamespace(
        model_screenshot_width=None,
        model_screenshot_height=None,
        model_screenshot_grid=False,
    )
    assert image_dimensions(
        _resize_model_screenshot_with_grid(geometry, image.getvalue())
    ) == (
        250,
        500,
    )


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
        _state_recovery=None,
        _get_ui_tree=AsyncMock(return_value=_combined_data(width, height)),
    )
    return AndroidStateProvider(
        driver,
        tree_filter=SimpleNamespace(filter=lambda tree, ctx: tree),
        tree_formatter=IndexedFormatter(),
        **kwargs,
    )


def _get_state(provider):
    return asyncio.run(provider.get_state())


def test_resize_flags_pair_providers_with_agent_gating():
    # The providers declare the flag…
    assert StateProvider.resize_model_screenshot is False
    assert ScreenshotOnlyStateProvider.resize_model_screenshot is True
    assert _provider(vision_enabled=True).resize_model_screenshot is True
    assert _provider().resize_model_screenshot is False


def test_legacy_injected_screenshot_providers_keep_getting_resized():
    """Injected providers that only implement the pre-existing
    ``requires_coordinate_tools`` contract must still get resized screenshots:
    the tool registry, prompts, and validation all still treat them as
    screenshot-coordinate mode."""

    class LegacyScreenshotProvider(StateProvider):
        requires_coordinate_tools = True

    assert _should_resize_model_screenshot(LegacyScreenshotProvider(SimpleNamespace()))

    class DuckTypedLegacyProvider(StateProvider):
        requires_coordinate_tools = True

    assert _should_resize_model_screenshot(DuckTypedLegacyProvider(SimpleNamespace()))

    # Non-coordinate providers are unaffected
    assert not _should_resize_model_screenshot(StateProvider(SimpleNamespace()))
    assert not _should_resize_model_screenshot(_provider())
    assert not _should_resize_model_screenshot(StateProvider(SimpleNamespace()))

    # And the new contract still activates it
    assert _should_resize_model_screenshot(_provider(vision_enabled=True))
    assert _should_resize_model_screenshot(
        ScreenshotOnlyStateProvider(SimpleNamespace())
    )


def test_provider_scale_matches_what_agents_actually_send():
    """The invariant that PR #355 lacked: the provider's scale must describe
    the image the agents really attach."""
    buf = BytesIO()
    Image.new("RGB", (NATIVE_W, NATIVE_H)).save(buf, format="PNG")
    sent = resize_image_to_max_side_with_grid(buf.getvalue(), 2048)
    sent_w, sent_h = image_dimensions(sent)

    state = _get_state(_provider(vision_enabled=True))
    assert sent_w == round(NATIVE_W / state.coordinate_scale_x)
    assert sent_h == round(NATIVE_H / state.coordinate_scale_y)
