"""Action functions for device interaction.

Each function receives ``ctx: ActionContext`` as a keyword argument and
interacts with the device via ``ctx.driver``, resolves UI elements via
``ctx.ui``, and accesses shared state via ``ctx.shared_state``.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from mobilerun.agent.action_context import ActionContext

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.oneflows.app_starter_workflow import AppStarter

logger = logging.getLogger("mobilerun")


# ---------------------------------------------------------------------------
# Core UI actions
# ---------------------------------------------------------------------------


def _uses_screenshot_only_coordinates(ctx: "ActionContext") -> bool:
    return bool(getattr(ctx.state_provider, "requires_coordinate_tools", False))


def _screenshot_only_coordinate_error(ctx: "ActionContext") -> str:
    width = getattr(ctx.ui, "screen_width", None)
    height = getattr(ctx.ui, "screen_height", None)
    if width and height:
        return (
            f"Coordinates must be inside the screenshot size {width}x{height} "
            "in screenshot-only mode. Observe the screenshot and retry with "
            "pixel coordinates inside the image."
        )
    return (
        "Coordinates must be inside the screenshot bounds in screenshot-only mode. "
        "Observe the screenshot and retry with pixel coordinates inside the image."
    )


def _validate_screenshot_only_point(
    x: int | float, y: int | float, *, ctx: "ActionContext"
) -> None:
    if not _uses_screenshot_only_coordinates(ctx):
        return
    try:
        width = float(ctx.ui.screen_width)
        height = float(ctx.ui.screen_height)
        px = float(x)
        py = float(y)
    except TypeError as exc:
        raise ValueError(_screenshot_only_coordinate_error(ctx)) from exc
    except ValueError as exc:
        raise ValueError(_screenshot_only_coordinate_error(ctx)) from exc

    out_of_range = (
        width <= 0
        or height <= 0
        or px < 0
        or px >= width
        or py < 0
        or py >= height
    )
    if out_of_range:
        raise ValueError(_screenshot_only_coordinate_error(ctx))


def _convert_action_point(
    x: int | float, y: int | float, *, ctx: "ActionContext"
) -> tuple[int, int]:
    _validate_screenshot_only_point(x, y, ctx=ctx)
    abs_x, abs_y = ctx.ui.convert_point(x, y)
    return int(round(abs_x)), int(round(abs_y))


async def click(index: int, *, ctx: "ActionContext") -> ActionResult:
    """Click the element with the given index."""
    try:
        x, y = ctx.ui.get_element_coords(index)
        await ctx.driver.tap(x, y)

        info = ctx.ui.get_element_info(index)
        detail_parts = [
            f"Text: '{info.get('text', 'No text')}'",
            f"Class: {info.get('className', 'Unknown class')}",
            f"Type: {info.get('type', 'unknown')}",
        ]
        if info.get("child_texts"):
            detail_parts.append(f"Contains text: {' | '.join(info['child_texts'])}")
        detail_parts.append(f"Coordinates: ({x}, {y})")

        return ActionResult(
            success=True, summary=f"Clicked on {' | '.join(detail_parts)}"
        )
    except ValueError as e:
        return ActionResult(
            success=False, summary=f"Failed to click element at index {index}: {e}"
        )


async def long_press(index: int, *, ctx: "ActionContext") -> ActionResult:
    """Long press the element with the given index."""
    try:
        x, y = ctx.ui.get_element_coords(index)
        await ctx.driver.swipe(x, y, x, y, 1000)
        return ActionResult(
            success=True, summary=f"Long pressed element at index {index} at ({x}, {y})"
        )
    except ValueError as e:
        return ActionResult(
            success=False, summary=f"Failed to long press element at index {index}: {e}"
        )


async def long_press_at(x: int, y: int, *, ctx: "ActionContext") -> ActionResult:
    """Long press at screen coordinates."""
    try:
        abs_x, abs_y = _convert_action_point(x, y, ctx=ctx)
        await ctx.driver.swipe(abs_x, abs_y, abs_x, abs_y, 1000)
        return ActionResult(success=True, summary=f"Long pressed at ({abs_x}, {abs_y})")
    except Exception as e:
        return ActionResult(
            success=False, summary=f"Failed to long press at ({x}, {y}): {e}"
        )


async def click_at(x: int, y: int, *, ctx: "ActionContext") -> ActionResult:
    """Click at screen coordinates."""
    try:
        abs_x, abs_y = _convert_action_point(x, y, ctx=ctx)
        await ctx.driver.tap(abs_x, abs_y)
        return ActionResult(success=True, summary=f"Tapped at ({abs_x}, {abs_y})")
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to tap at ({x}, {y}): {e}")


async def click_area(
    x1: int, y1: int, x2: int, y2: int, *, ctx: "ActionContext"
) -> ActionResult:
    """Click center of area."""
    try:
        _validate_screenshot_only_point(x1, y1, ctx=ctx)
        _validate_screenshot_only_point(x2, y2, ctx=ctx)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        abs_x, abs_y = _convert_action_point(cx, cy, ctx=ctx)
        await ctx.driver.tap(abs_x, abs_y)
        return ActionResult(
            success=True, summary=f"Tapped center of area at ({abs_x}, {abs_y})"
        )
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to tap area center: {e}")


async def type_text(
    text: str, index: int, clear: bool = False, *, ctx: "ActionContext"
) -> ActionResult:
    """Type text into the element with the given index."""
    try:
        # Tap the element first if a specific index is given
        if index != -1:
            x, y = ctx.ui.get_element_coords(index)
            await ctx.driver.tap(x, y)

        success = await ctx.driver.input_text(text, clear)
        if success:
            return ActionResult(
                success=True, summary=f"Text typed successfully (clear={clear})"
            )
        else:
            return ActionResult(
                success=False, summary="Failed to type text: input failed"
            )
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to type text: {e}")


async def type_text_direct(
    text: str, clear: bool = False, *, ctx: "ActionContext"
) -> ActionResult:
    """Type text into the currently focused input."""
    try:
        success = await ctx.driver.input_text(text, clear)
        if success:
            return ActionResult(
                success=True, summary=f"Text typed successfully (clear={clear})"
            )
        return ActionResult(success=False, summary="Failed to type text: input failed")
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to type text: {e}")


async def system_button(button: str, *, ctx: "ActionContext") -> ActionResult:
    """Press a system button (back, home, or enter)."""
    try:
        await ctx.driver.press_button(button)
        return ActionResult(success=True, summary=f"Pressed {button.upper()} button")
    except ValueError as e:
        return ActionResult(success=False, summary=str(e))
    except Exception as e:
        return ActionResult(
            success=False,
            summary=f"Failed to press {button} button: {e.__class__.__name__}: {e}",
        )


async def swipe(
    coordinate: List[int],
    coordinate2: List[int],
    duration: float = 1.0,
    *,
    ctx: "ActionContext",
) -> ActionResult:
    """Swipe from one coordinate to another."""
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        return ActionResult(
            success=False,
            summary=f"Failed: coordinate must be a list of 2 integers, got: {coordinate}",
        )
    if not isinstance(coordinate2, list) or len(coordinate2) != 2:
        return ActionResult(
            success=False,
            summary=f"Failed: coordinate2 must be a list of 2 integers, got: {coordinate2}",
        )

    try:
        start_x, start_y = _convert_action_point(*coordinate, ctx=ctx)
        end_x, end_y = _convert_action_point(*coordinate2, ctx=ctx)
        duration_ms = int(duration * 1000)
        await ctx.driver.swipe(start_x, start_y, end_x, end_y, duration_ms=duration_ms)
        return ActionResult(
            success=True,
            summary=f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y})",
        )
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to swipe: {e}")


async def open_app(text: str, *, ctx: "ActionContext") -> ActionResult:
    """Open an app by its name."""
    if ctx.app_opener_llm is None:
        return ActionResult(
            success=False,
            summary="Failed: app_opener_llm not configured.",
        )

    workflow = AppStarter(
        driver=ctx.driver,
        llm=ctx.app_opener_llm,
        timeout=60,
        stream=ctx.streaming,
        verbose=False,
    )

    result = await workflow.run(app_description=text)
    await asyncio.sleep(1)

    if isinstance(result, str) and "could not open app" in result.lower():
        return ActionResult(success=False, summary=result)
    return ActionResult(success=True, summary=str(result))


async def open_bundle_id(bundle_id: str, *, ctx: "ActionContext") -> ActionResult:
    """Open an app by exact package name or iOS bundle identifier."""
    hint = (
        "Maybe you got the wrong bundle ID. You could try using swipes and search "
        "to find the app."
    )
    try:
        result = await ctx.driver.start_app(bundle_id)
        await asyncio.sleep(1)
        if isinstance(result, str) and result.lower().startswith("failed"):
            return ActionResult(
                success=False,
                summary=f"Failed to open app '{bundle_id}': {result}\n{hint}",
            )
        return ActionResult(success=True, summary=str(result))
    except Exception as e:
        return ActionResult(
            success=False,
            summary=f"Failed to open app '{bundle_id}': {e.__class__.__name__}: {e}\n{hint}",
        )


async def wait(duration: float = 1.0, *, ctx: "ActionContext") -> ActionResult:
    """Wait for a specified duration in seconds."""
    await asyncio.sleep(duration)
    return ActionResult(success=True, summary=f"Waited for {duration} seconds")


# ---------------------------------------------------------------------------
# State / memory actions
# ---------------------------------------------------------------------------


async def remember(information: str, *, ctx: "ActionContext") -> ActionResult:
    """Remember important information for later use."""
    result = await ctx.shared_state.remember(information)
    success = not result.startswith("Failed")
    return ActionResult(success=success, summary=result)


async def complete(
    success: bool, reason: str = "", message: str = "", *, ctx: "ActionContext"
) -> ActionResult:
    """Mark the task as complete.

    Accepts both ``reason`` and ``message`` — FastAgent XML prompt uses
    ``message``, action signature uses ``reason``.
    """
    await ctx.shared_state.complete(success, reason=reason, message=message)
    return ActionResult(success=True, summary=ctx.shared_state.answer)


async def type_secret(
    secret_id: str, index: int, *, ctx: "ActionContext"
) -> ActionResult:
    """Type a secret credential into an input field without exposing the value."""
    if ctx.credential_manager is None:
        return ActionResult(
            success=False,
            summary="Failed to type secret: Credential manager not initialized. Enable credentials in config.yaml",
        )

    try:
        secret_value = await ctx.credential_manager.resolve_key(secret_id)

        # Tap the element first if a specific index is given
        if index != -1:
            x, y = ctx.ui.get_element_coords(index)
            await ctx.driver.tap(x, y)

        ok = await ctx.driver.input_text(secret_value)
        if ok:
            return ActionResult(
                success=True,
                summary=f"Successfully typed secret '{secret_id}' into element {index}",
            )
        else:
            return ActionResult(
                success=False,
                summary=f"Failed to type secret '{secret_id}': input failed",
            )
    except Exception as e:
        logger.error(f"Failed to type secret '{secret_id}': {e}")
        available = (
            await ctx.credential_manager.get_keys() if ctx.credential_manager else []
        )
        return ActionResult(
            success=False,
            summary=f"Failed to type secret '{secret_id}': not found. Available: {available}",
        )
