"""StateProvider — orchestrates fetching and parsing device state.

Fetches raw data from a ``DeviceDriver``, applies tree filters/formatters,
and produces a ``UIState`` snapshot.
"""

from __future__ import annotations

import asyncio
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Optional

from mobilerun.clients.portal.state_fetch import fetch_state_with_retry
from mobilerun.config.settings import get_settings
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.stealth import StealthUIState
from mobilerun.core.state.ui_state import UIState
from mobilerun.utils.images import (
    fit_dimensions_to_max_side,
)
from mobilerun.utils.prompts import load_prompt_text

NORMALIZED_STATE_NOTE = load_prompt_text("state/normalized_state_note.md")

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.state.filters import TreeFilter
    from mobilerun.core.state.formatters import TreeFormatter
    from mobilerun.core.state.options import VisionResizePolicy


class AndroidStateProvider(StateProvider):
    """Fetches state from an Android device via ``driver._get_ui_tree()``.

    Includes retry logic with exponential backoff and mid-retry recovery
    (accessibility service restart) for robustness against intermittent
    Portal/a11y failures.
    """

    capabilities = {"element_index", "convert_point"}

    def __init__(
        self,
        driver: "Device",
        tree_filter: "TreeFilter",
        tree_formatter: "TreeFormatter",
        use_normalized: bool = False,
        stealth: bool = False,
        ui_cls: "type[UIState] | None" = None,
        vision_enabled: bool = False,
        vision_resize_policy: VisionResizePolicy | None = None,
    ) -> None:
        super().__init__(driver)
        self.tree_filter = tree_filter
        self.tree_formatter = tree_formatter
        self.use_normalized = use_normalized
        self._ui_cls = ui_cls or (StealthUIState if stealth else UIState)
        self.vision_enabled = vision_enabled
        # Resolves the exact screenshot dims the active vision model grounds on
        # (duck-typed: needs ``effective_dims(w, h)``). None → legacy 2048 cap.
        self.vision_resize_policy = vision_resize_policy
        # Exact model-facing screenshot dims for the current state (set per
        # get_state); read by _resize_model_screenshot_with_grid.
        self.model_screenshot_width: Optional[int] = None
        self.model_screenshot_height: Optional[int] = None
        # Model coordinates reach device pixels either 1:1, through the vision
        # contract's scaling, or through normalized [0-1000] conversion.
        self.screenshot_matches_input_coords = True
        # Normalized [0-1000] coordinates are scale-invariant, so the
        # deterministic-space contract is only needed for pixel coordinates.
        # Re-evaluated per get_state: the flag must describe the CURRENT
        # state's contract — if screen bounds are unavailable for a state,
        # resizing that step's screenshot would desynchronize the model's
        # space from convert_point.
        self._vision_contract_intent = vision_enabled and not use_normalized
        self.resize_model_screenshot = self._vision_contract_intent
        # Hybrid vision exposes click_at when a coordinate contract is
        # expected, but individual states can still arrive with missing or
        # zero screen bounds. Refuse coordinate actions for those snapshots
        # instead of mapping model coordinates against a wrong screen size.
        self.requires_active_contract_for_coords = vision_enabled

    async def get_state(self) -> UIState:
        combined_data = await fetch_state_with_retry(
            fetch=self._device._get_ui_tree,
            recovery=self._device._state_recovery,
        )
        return await self._state_from_data_async(combined_data)

    async def get_final_state(self) -> UIState:
        """Finalization must not restart accessibility after a failed read."""
        combined_data = await self._device._get_ui_tree()
        return await self._state_from_data_async(combined_data)

    async def _state_from_data_async(self, combined_data: dict) -> UIState:
        # Workers may outlive cancellation. Only their per-call copies may change.
        processor = copy(self)
        processor.tree_filter = deepcopy(self.tree_filter)
        processor.tree_formatter = deepcopy(self.tree_formatter)
        processor.vision_resize_policy = deepcopy(self.vision_resize_policy)

        def process() -> tuple[UIState, bool]:
            state = processor._state_from_data(combined_data)
            return state, processor.resize_model_screenshot

        state, resize = await asyncio.to_thread(process)
        # Publish together on the loop only when processing completed successfully.
        self.resize_model_screenshot = resize
        self.model_screenshot_width = state.model_screenshot_width
        self.model_screenshot_height = state.model_screenshot_height
        self.tree_formatter.screen_width = state.screen_width
        self.tree_formatter.screen_height = state.screen_height
        self.tree_formatter.use_normalized = state.use_normalized
        self.tree_formatter.display_scale_x = state.coordinate_scale_x
        self.tree_formatter.display_scale_y = state.coordinate_scale_y
        return state

    def _state_from_data(self, combined_data: dict) -> UIState:

        device_context = combined_data["device_context"]
        screen_bounds = device_context.get("screen_bounds", {})
        screen_width = screen_bounds.get("width")
        screen_height = screen_bounds.get("height")

        filtered = self.tree_filter.filter(combined_data["a11y_tree"], device_context)

        coordinate_scale_x = 1.0
        coordinate_scale_y = 1.0
        display_width = None
        display_height = None
        if self._vision_contract_intent and screen_width and screen_height:
            if self.vision_resize_policy is not None:
                display_width, display_height = (
                    self.vision_resize_policy.effective_dims(
                        screen_width, screen_height
                    )
                )
            else:
                display_width, display_height = fit_dimensions_to_max_side(
                    screen_width,
                    screen_height,
                    get_settings().MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE,
                )
            coordinate_scale_x = screen_width / display_width
            coordinate_scale_y = screen_height / display_height
        if self._vision_contract_intent:
            # Keep agent-side resizing paired with the contract this state
            # actually declares (screen bounds can be missing for a state).
            self.resize_model_screenshot = bool(display_width and display_height)
        # Exact dims the model-facing screenshot must match for this state.
        self.model_screenshot_width = display_width
        self.model_screenshot_height = display_height

        self.tree_formatter.screen_width = screen_width
        self.tree_formatter.screen_height = screen_height
        self.tree_formatter.use_normalized = self.use_normalized
        # Model-facing bounds text is emitted in the display space the model
        # actually sees; tap bounds in ``elements`` stay native device pixels.
        self.tree_formatter.display_scale_x = coordinate_scale_x
        self.tree_formatter.display_scale_y = coordinate_scale_y

        formatted_text, focused_text, elements, phone_state = (
            self.tree_formatter.format(filtered, combined_data["phone_state"])
        )

        if (
            self.use_normalized
            and self.vision_enabled
            and screen_width
            and screen_height
        ):
            formatted_text += f"\n\n{NORMALIZED_STATE_NOTE}"
        if display_width and display_height:
            scaled_note = (
                f" — the {screen_width}x{screen_height} device screen scaled down"
                if (display_width, display_height) != (screen_width, screen_height)
                else ""
            )
            formatted_text += (
                f"\n\nAll coordinates in this device state (element bounds above, "
                f"and any x/y you provide to coordinate actions) are in a "
                f"{display_width}x{display_height} coordinate space{scaled_note}. "
                f"When a screenshot is provided it matches this "
                f"{display_width}x{display_height} space and carries a labeled "
                f"coordinate grid for reference."
            )

        return self._ui_cls(
            elements=elements,
            formatted_text=formatted_text,
            focused_text=focused_text,
            phone_state=phone_state,
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=self.use_normalized,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
            coordinate_contract_active=bool(display_width and display_height)
            or bool(self.use_normalized and screen_width and screen_height),
            model_screenshot_width=display_width,
            model_screenshot_height=display_height,
        )
