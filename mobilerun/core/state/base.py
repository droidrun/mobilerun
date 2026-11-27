"""StateProvider — orchestrates fetching and parsing device state.

Fetches raw data from a ``DeviceDriver``, applies tree filters/formatters,
and produces a ``UIState`` snapshot.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mobilerun.config.settings import get_settings
from mobilerun.core.state.ui_state import UIState
from mobilerun.utils.images import (
    resize_image_to_dimensions_with_grid,
    resize_image_to_max_side_with_grid,
)

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


class StateProvider:
    """Base class — subclass to support different platforms."""

    capabilities: set[str] = set()
    use_normalized: bool = False
    requires_active_contract_for_coords: bool = False
    # True when coordinates a model derives from the screenshot map to the
    # driver's tap input space — either directly (Android: screenshot and
    # input coords are both device pixels) or via ``convert_point`` scaling
    # (the vision coordinate contract; iOS sets this only when the contract
    # is active, since its screenshots are physical pixels while taps use
    # XCTest points). Gates the click_at auto-unmask with vision.
    screenshot_matches_input_coords: bool = False
    # True when the screenshot attached to LLM messages must first be resized
    # (with a labeled coordinate grid) into the deterministic coordinate space
    # the provider declared in ``formatted_text`` and compensates for via
    # ``coordinate_scale_x/y``. Vision-model output spaces are otherwise
    # model-dependent: a raw native screenshot is downscaled inside the LLM
    # provider's API and different models answer in different spaces.
    resize_model_screenshot: bool = False
    requires_coordinate_tools: bool = False
    model_screenshot_width: int | None = None
    model_screenshot_height: int | None = None
    model_screenshot_grid: bool = True

    def __init__(self, driver: "Device") -> None:
        self._device = driver

    async def get_state(self) -> UIState:
        raise NotImplementedError

    async def get_final_state(self) -> UIState:
        """Read the final observation; subclasses may make this narrower."""
        return await self.get_state()


def _should_resize_model_screenshot(state_provider: StateProvider) -> bool:
    """True when screenshots attached to LLM messages must be resized (with
    the labeled grid) into the coordinate space the provider declared.

    Falls back to the legacy ``requires_coordinate_tools`` contract: injected
    screenshot-only providers that predate ``resize_model_screenshot`` are
    still treated as screenshot-coordinate mode by the tool registry, prompts,
    and validation, so they must keep receiving resized screenshots too.
    """
    return bool(
        state_provider.resize_model_screenshot
        or state_provider.requires_coordinate_tools
    )


def _resize_model_screenshot_with_grid(
    state_provider: StateProvider | _ScreenshotGeometry, screenshot: bytes
) -> bytes:
    """Resize a screenshot to the exact model-facing dims the provider declared.

    The provider stores ``model_screenshot_width/height`` (resolved from the
    active vision model's effective image size) on each ``get_state``. Resizing
    to those exact dims keeps the image the model grounds on equal to the
    declared coordinate space. Falls back to the legacy max-side resize when no
    dims are available (e.g. injected screenshot-only providers)."""
    width = state_provider.model_screenshot_width
    height = state_provider.model_screenshot_height
    # A pixel grid contradicts a normalized coordinate contract.
    divisions = 10 if state_provider.model_screenshot_grid else 0
    if width and height:
        return resize_image_to_dimensions_with_grid(
            screenshot, width, height, divisions=divisions
        )
    return resize_image_to_max_side_with_grid(
        screenshot,
        get_settings().MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE,
        divisions=divisions,
    )


async def _resize_model_screenshot_async(
    state_provider: StateProvider, screenshot: bytes
) -> bytes:
    """Resize when required, snapshotting model geometry before offloading."""
    if not _should_resize_model_screenshot(state_provider):
        return screenshot

    geometry = _ScreenshotGeometry(
        model_screenshot_width=state_provider.model_screenshot_width,
        model_screenshot_height=state_provider.model_screenshot_height,
        model_screenshot_grid=state_provider.model_screenshot_grid,
    )
    return await asyncio.to_thread(
        _resize_model_screenshot_with_grid, geometry, screenshot
    )


@dataclass(frozen=True)
class _ScreenshotGeometry:
    model_screenshot_width: int | None
    model_screenshot_height: int | None
    model_screenshot_grid: bool = True
