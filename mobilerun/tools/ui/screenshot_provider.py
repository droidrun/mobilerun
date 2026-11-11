"""Screenshot-only state provider for visual control backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from mobilerun.tools.helpers.images import fit_dimensions_to_max_side, image_dimensions
from mobilerun.tools.ui.provider import StateProvider
from mobilerun.tools.ui.state import UIState

if TYPE_CHECKING:
    from mobilerun_core_local.driver.base import DeviceDriver


_SCREENSHOT_ONLY_GUIDANCE = (
    "Prefer click_at on the center of visible text or controls, especially in "
    "dense lists, adjacent rows, and compact menus. Use click_area only for "
    "large, unambiguous targets. If a tap does not change the screen, do not "
    "repeat the same coordinate; choose a better point on the intended target "
    "or use navigation. For text entry, focus a field with a coordinate action "
    "first, then use direct text typing."
)


class ScreenshotOnlyStateProvider(StateProvider):
    """Build UI state from screenshots without reading an accessibility tree."""

    supported = {"convert_point", "direct_text_input"}
    requires_coordinate_tools = True
    resize_model_screenshot = True

    def __init__(
        self,
        driver: "DeviceDriver",
        vision_resize_policy: Any = None,
        use_normalized: bool = False,
    ) -> None:
        super().__init__(driver)
        # Resolves the exact screenshot dims the active vision model grounds on
        # (duck-typed: needs ``effective_dims(w, h)``). None → legacy 2048 cap.
        self.vision_resize_policy = vision_resize_policy
        # Normalized mode declares a 0-1000 grid instead of screenshot pixels.
        self.use_normalized = use_normalized
        self.model_screenshot_grid = not use_normalized
        self.model_screenshot_width: Optional[int] = None
        self.model_screenshot_height: Optional[int] = None

    async def get_state(self) -> UIState:
        screenshot = await self.driver.screenshot()
        native_width, native_height = image_dimensions(screenshot)
        input_size = getattr(self.driver, "input_coordinate_size", None)
        if input_size is None:
            input_width, input_height = native_width, native_height
        else:
            input_width, input_height = await input_size(native_width, native_height)
        if self.vision_resize_policy is not None:
            screen_width, screen_height = self.vision_resize_policy.effective_dims(
                native_width, native_height
            )
        else:
            screen_width, screen_height = fit_dimensions_to_max_side(
                native_width,
                native_height,
            )
        self.model_screenshot_width = screen_width
        self.model_screenshot_height = screen_height
        phone_state = {
            "observationMode": "screenshot_only",
            "accessibilityTree": False,
        }

        if self.use_normalized:
            return UIState(
                elements=[],
                formatted_text=(
                    "Screenshot-only mode is active. There is no accessibility "
                    "tree or element index list. Inspect the screenshot and use "
                    "coordinate actions in normalized coordinates: both axes run "
                    "from 0 to 1000 across the screenshot, whatever the image "
                    "size, so (0,0) is top-left, (500,500) is the center and "
                    f"(1000,1000) is bottom-right. {_SCREENSHOT_ONLY_GUIDANCE}"
                ),
                focused_text="",
                phone_state=phone_state,
                screen_width=input_width,
                screen_height=input_height,
                use_normalized=True,
                model_screenshot_width=screen_width,
                model_screenshot_height=screen_height,
            )

        max_x = max(screen_width - 1, 0)
        max_y = max(screen_height - 1, 0)
        return UIState(
            elements=[],
            formatted_text=(
                "Screenshot-only mode is active. There is no accessibility tree "
                "or element index list. Inspect the screenshot and use coordinate "
                "actions in the screenshot pixel coordinates shown to the model. "
                "The screenshot includes a coordinate grid for visual reference; "
                "use the underlying screenshot pixel coordinates, not grid-cell "
                "numbers. "
                f"The screenshot shown to the model is {screen_width}x{screen_height}; "
                "(0,0) is top-left and "
                f"({max_x},{max_y}) is bottom-right. {_SCREENSHOT_ONLY_GUIDANCE}"
            ),
            focused_text="",
            phone_state=phone_state,
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=False,
            coordinate_scale_x=input_width / screen_width,
            coordinate_scale_y=input_height / screen_height,
            model_screenshot_width=screen_width,
            model_screenshot_height=screen_height,
        )
