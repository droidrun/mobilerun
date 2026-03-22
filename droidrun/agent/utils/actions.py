"""Action functions for device interaction.

Each function receives ``ctx: ActionContext`` as a keyword argument and
interacts with the device via ``ctx.driver``, resolves UI elements via
``ctx.ui``, and accesses shared state via ``ctx.shared_state``.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from droidrun.agent.action_context import ActionContext

from droidrun.agent.action_result import ActionResult
from droidrun.agent.oneflows.app_starter_workflow import AppStarter

logger = logging.getLogger("droidrun")


# ---------------------------------------------------------------------------
# Core UI actions
# ---------------------------------------------------------------------------


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
        abs_x, abs_y = ctx.ui.convert_point(x, y)
        await ctx.driver.swipe(abs_x, abs_y, abs_x, abs_y, 1000)
        return ActionResult(success=True, summary=f"Long pressed at ({abs_x}, {abs_y})")
    except Exception as e:
        return ActionResult(
            success=False, summary=f"Failed to long press at ({x}, {y}): {e}"
        )


async def click_at(x: int, y: int, *, ctx: "ActionContext") -> ActionResult:
    """Click at screen coordinates."""
    try:
        abs_x, abs_y = ctx.ui.convert_point(x, y)
        await ctx.driver.tap(abs_x, abs_y)
        return ActionResult(success=True, summary=f"Tapped at ({abs_x}, {abs_y})")
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to tap at ({x}, {y}): {e}")


async def click_area(
    x1: int, y1: int, x2: int, y2: int, *, ctx: "ActionContext"
) -> ActionResult:
    """Click center of area."""
    try:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        abs_x, abs_y = ctx.ui.convert_point(cx, cy)
        await ctx.driver.tap(abs_x, abs_y)
        return ActionResult(
            success=True, summary=f"Tapped center of area at ({abs_x}, {abs_y})"
        )
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to tap area center: {e}")


async def type(
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


async def system_button(button: str, *, ctx: "ActionContext") -> ActionResult:
    """Press a supported semantic system button on the active device."""
    button_lower = button.lower()
    supported_buttons = sorted(ctx.driver.supported_system_buttons)

    if button_lower not in supported_buttons:
        valid = ", ".join(supported_buttons) if supported_buttons else "none"
        return ActionResult(
            success=False,
            summary=(
                f"Failed to press {button} button: unsupported on this device. "
                f"Valid options: {valid}"
            ),
        )

    button_name = button_lower.upper()

    try:
        await ctx.driver.press_system_button(button_lower)
        return ActionResult(success=True, summary=f"Pressed {button_name} button")
    except Exception as e:
        return ActionResult(
            success=False, summary=f"Failed to press {button_name} button: {e}"
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
        start_x, start_y = ctx.ui.convert_point(*coordinate)
        end_x, end_y = ctx.ui.convert_point(*coordinate2)
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
        tools=ctx.driver,
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
