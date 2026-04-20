"""Screenshot-only state provider for visual control backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mobilerun.tools.helpers.images import fit_dimensions_to_max_side, image_dimensions
from mobilerun.tools.ui.provider import StateProvider
from mobilerun.tools.ui.state import UIState

if TYPE_CHECKING:
    from mobilerun.tools.driver.base import DeviceDriver


class ScreenshotOnlyStateProvider(StateProvider):
    """Build UI state from screenshots without reading an accessibility tree."""

    supported = {"convert_point", "direct_text_input"}
    requires_coordinate_tools = True

    def __init__(self, driver: "DeviceDriver") -> None:
        super().__init__(driver)

    async def get_state(self) -> UIState:
        screenshot = await self.driver.screenshot()
        native_width, native_height = image_dimensions(screenshot)
        screen_width, screen_height = fit_dimensions_to_max_side(
            native_width,
            native_height,
        )
        max_x = max(screen_width - 1, 0)
        max_y = max(screen_height - 1, 0)

        return UIState(
            elements=[],
            formatted_text=(
                "Screenshot-only mode is active. There is no accessibility tree "
                "or element index list. Inspect the screenshot and use coordinate "
                "actions in screenshot pixel coordinates. The screenshot shown "
                f"to the model is {screen_width}x{screen_height}; (0,0) is top-left and "
                f"({max_x},{max_y}) is bottom-right. Do not tap toggles or "
                "destructive controls unless the user explicitly asks. Tap the "
                "actual target text or control, not section headers or category "
                "labels. If a tap does not change the screen, do not repeat the "
                "same coordinate; choose a better point on the intended target or "
                "use navigation. For text entry, focus a field with a coordinate "
                "action first, then use direct text typing."
            ),
            focused_text="",
            phone_state={
                "observationMode": "screenshot_only",
                "accessibilityTree": False,
            },
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=False,
            coordinate_scale_x=native_width / screen_width,
            coordinate_scale_y=native_height / screen_height,
        )
