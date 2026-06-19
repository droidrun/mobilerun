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
from mobilerun.credential_manager.placeholders import resolve_credential_placeholders

logger = logging.getLogger("mobilerun")

_MACRO_FOCUS_SETTLE_SECONDS = 0.5


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
        width <= 0 or height <= 0 or px < 0 or px >= width or py < 0 or py >= height
    )
    if out_of_range:
        raise ValueError(_screenshot_only_coordinate_error(ctx))


def _model_space_dimensions(ctx: "ActionContext") -> tuple[float, float] | None:
    """Return the (width, height) coordinate space the model was shown.

    Only applies when the provider declared a scaled coordinate contract
    (``resize_model_screenshot``) outside screenshot-only mode: the UIState
    keeps native screen dimensions while the model sees and answers in the
    ``native / coordinate_scale`` display space.
    """
    if _uses_screenshot_only_coordinates(ctx):
        return None
    # Read the contract state from the UIState snapshot the tap uses.
    if not getattr(ctx.ui, "coordinate_contract_active", False):
        return None
    if getattr(ctx.ui, "use_normalized", False):
        return None
    try:
        scale_x = float(ctx.ui.coordinate_scale_x)
        scale_y = float(ctx.ui.coordinate_scale_y)
        width = float(ctx.ui.screen_width) / scale_x
        height = float(ctx.ui.screen_height) / scale_y
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _validate_model_space_point(
    x: int | float, y: int | float, *, ctx: "ActionContext"
) -> None:
    dims = _model_space_dimensions(ctx)
    if dims is None:
        return
    width, height = dims
    try:
        px = float(x)
        py = float(y)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Coordinates must be numbers inside the {round(width)}x{round(height)} "
            "coordinate space of the provided device state."
        ) from exc
    if px < 0 or px >= width or py < 0 or py >= height:
        raise ValueError(
            f"Coordinates ({x}, {y}) are outside the {round(width)}x{round(height)} "
            "coordinate space of the provided device state. Use the element "
            "bounds and screenshot shown to you and retry with coordinates "
            "inside that space."
        )


def _require_active_coordinate_contract(ctx: "ActionContext") -> None:
    """Reject coordinate actions when the provider requires the vision
    coordinate contract but it is not active for the current state.

    Some providers (e.g. iOS) can only map model coordinates to the tap input
    space while the contract is active. Without it, refuse the action so the
    model retries or uses an element-index ``click`` instead of tapping the
    wrong location."""
    provider = ctx.state_provider
    if not getattr(provider, "requires_active_contract_for_coords", False):
        return
    # Read the contract state from the UIState snapshot the tap uses, not a
    # mutable provider flag.
    if getattr(ctx.ui, "coordinate_contract_active", False):
        return
    raise ValueError(
        "Coordinate actions are unavailable for this step — the screen state "
        "could not be captured fully. Use an element index with `click`, or "
        "retry to get a fresh screen state."
    )


def _convert_action_point(
    x: int | float, y: int | float, *, ctx: "ActionContext"
) -> tuple[int, int]:
    _validate_screenshot_only_point(x, y, ctx=ctx)
    _require_active_coordinate_contract(ctx)
    _validate_model_space_point(x, y, ctx=ctx)
    abs_x, abs_y = ctx.ui.convert_point(x, y)
    return int(round(abs_x)), int(round(abs_y))


def _macro_recorder(ctx: "ActionContext"):
    return getattr(ctx, "macro_recorder", None)


def _record_macro_action(
    ctx: "ActionContext",
    action: dict,
    *,
    pre_ui=None,
) -> None:
    recorder = _macro_recorder(ctx)
    if recorder is None:
        return
    recorder.record_action(
        action,
        pre_ui=pre_ui if pre_ui is not None else getattr(ctx, "ui", None),
    )


async def _macro_pre_ui(ctx: "ActionContext"):
    if _macro_recorder(ctx) is None:
        return getattr(ctx, "ui", None)
    state_provider = getattr(ctx, "state_provider", None)
    if state_provider is None or not hasattr(state_provider, "get_state"):
        return getattr(ctx, "ui", None)
    try:
        return await state_provider.get_state()
    except Exception as e:
        logger.debug(f"Failed to refresh macro pre-state: {e}")
        return getattr(ctx, "ui", None)


async def _macro_pre_ui_after_focus_tap(ctx: "ActionContext"):
    if _macro_recorder(ctx) is not None:
        await asyncio.sleep(_MACRO_FOCUS_SETTLE_SECONDS)
    return await _macro_pre_ui(ctx)


def _driver_log_length(ctx: "ActionContext") -> int | None:
    log = getattr(getattr(ctx, "driver", None), "log", None)
    if isinstance(log, list):
        return len(log)
    return None


def _record_driver_log_delta(
    ctx: "ActionContext", before: int | None, *, pre_ui=None
) -> None:
    if before is None:
        return
    log = getattr(getattr(ctx, "driver", None), "log", None)
    if not isinstance(log, list):
        return
    for raw_action in log[before:]:
        _record_macro_action(ctx, dict(raw_action), pre_ui=pre_ui)


async def click(index: int, *, ctx: "ActionContext") -> ActionResult:
    """Click the element with the given index."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        x, y = ctx.ui.get_element_coords(index)
        await ctx.driver.tap(x, y)
        _record_macro_action(
            ctx,
            {"action_type": "tap", "x": x, "y": y},
            pre_ui=pre_ui,
        )

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
        pre_ui = await _macro_pre_ui(ctx)
        x, y = ctx.ui.get_element_coords(index)
        await ctx.driver.swipe(x, y, x, y, 1000)
        _record_macro_action(
            ctx,
            {
                "action_type": "swipe",
                "start_x": x,
                "start_y": y,
                "end_x": x,
                "end_y": y,
                "duration_ms": 1000,
            },
            pre_ui=pre_ui,
        )
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
        pre_ui = await _macro_pre_ui(ctx)
        abs_x, abs_y = _convert_action_point(x, y, ctx=ctx)
        await ctx.driver.swipe(abs_x, abs_y, abs_x, abs_y, 1000)
        _record_macro_action(
            ctx,
            {
                "action_type": "swipe",
                "start_x": abs_x,
                "start_y": abs_y,
                "end_x": abs_x,
                "end_y": abs_y,
                "duration_ms": 1000,
            },
            pre_ui=pre_ui,
        )
        return ActionResult(success=True, summary=f"Long pressed at ({abs_x}, {abs_y})")
    except Exception as e:
        return ActionResult(
            success=False, summary=f"Failed to long press at ({x}, {y}): {e}"
        )


async def click_at(x: int, y: int, *, ctx: "ActionContext") -> ActionResult:
    """Click at screen coordinates."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        abs_x, abs_y = _convert_action_point(x, y, ctx=ctx)
        await ctx.driver.tap(abs_x, abs_y)
        _record_macro_action(
            ctx,
            {"action_type": "tap", "x": abs_x, "y": abs_y},
            pre_ui=pre_ui,
        )
        return ActionResult(success=True, summary=f"Tapped at ({abs_x}, {abs_y})")
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to tap at ({x}, {y}): {e}")


async def click_area(
    x1: int, y1: int, x2: int, y2: int, *, ctx: "ActionContext"
) -> ActionResult:
    """Click center of area."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        _validate_screenshot_only_point(x1, y1, ctx=ctx)
        _validate_screenshot_only_point(x2, y2, ctx=ctx)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        abs_x, abs_y = _convert_action_point(cx, cy, ctx=ctx)
        await ctx.driver.tap(abs_x, abs_y)
        _record_macro_action(
            ctx,
            {"action_type": "tap", "x": abs_x, "y": abs_y},
            pre_ui=pre_ui,
        )
        return ActionResult(
            success=True, summary=f"Tapped center of area at ({abs_x}, {abs_y})"
        )
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to tap area center: {e}")


async def type_text(
    text: str, index: int | None = None, clear: bool = False, *, ctx: "ActionContext"
) -> ActionResult:
    """Type text into an indexed element or the currently focused input."""
    try:
        original_text = text
        pre_ui = await _macro_pre_ui(ctx)
        text = await resolve_credential_placeholders(
            text, getattr(ctx, "credential_manager", None)
        )
        if index is not None and index != -1:
            x, y = ctx.ui.get_element_coords(index)
            await ctx.driver.tap(x, y)
            _record_macro_action(
                ctx,
                {"action_type": "tap", "x": x, "y": y},
                pre_ui=pre_ui,
            )
            pre_ui = await _macro_pre_ui_after_focus_tap(ctx)

        success = await ctx.driver.input_text(text, clear)
        if success:
            _record_macro_action(
                ctx,
                {"action_type": "input_text", "text": original_text, "clear": clear},
                pre_ui=pre_ui,
            )
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
        original_text = text
        pre_ui = await _macro_pre_ui(ctx)
        text = await resolve_credential_placeholders(
            text, getattr(ctx, "credential_manager", None)
        )
        success = await ctx.driver.input_text(text, clear)
        if success:
            _record_macro_action(
                ctx,
                {"action_type": "input_text", "text": original_text, "clear": clear},
                pre_ui=pre_ui,
            )
            return ActionResult(
                success=True, summary=f"Text typed successfully (clear={clear})"
            )
        return ActionResult(success=False, summary="Failed to type text: input failed")
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to type text: {e}")


async def system_button(button: str, *, ctx: "ActionContext") -> ActionResult:
    """Press a system button (back, home, or enter)."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        await ctx.driver.press_button(button)
        _record_macro_action(
            ctx,
            {"action_type": "button_press", "button": button},
            pre_ui=pre_ui,
        )
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
        pre_ui = await _macro_pre_ui(ctx)
        start_x, start_y = _convert_action_point(*coordinate, ctx=ctx)
        end_x, end_y = _convert_action_point(*coordinate2, ctx=ctx)
        duration_ms = int(duration * 1000)
        await ctx.driver.swipe(start_x, start_y, end_x, end_y, duration_ms=duration_ms)
        _record_macro_action(
            ctx,
            {
                "action_type": "swipe",
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "duration_ms": duration_ms,
            },
            pre_ui=pre_ui,
        )
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

    pre_ui = await _macro_pre_ui(ctx)
    driver_log_before = _driver_log_length(ctx)
    result = await workflow.run(app_description=text)
    await asyncio.sleep(1)

    if isinstance(result, str) and "could not open app" in result.lower():
        return ActionResult(success=False, summary=result)
    _record_driver_log_delta(ctx, driver_log_before, pre_ui=pre_ui)
    return ActionResult(success=True, summary=str(result))


async def open_bundle_id(
    bundle_id: str | None = None,
    app_id: str | None = None,
    *,
    ctx: "ActionContext",
) -> ActionResult:
    """Open an app by exact package name, app id, or iOS bundle identifier."""
    identifier = app_id or bundle_id
    if not identifier:
        return ActionResult(
            success=False,
            summary="Failed to open app: exact app identifier is required.",
        )

    hint = (
        "Maybe you got the wrong app identifier. You could try using swipes and "
        "search to find the app."
    )
    try:
        pre_ui = await _macro_pre_ui(ctx)
        result = await ctx.driver.start_app(identifier)
        await asyncio.sleep(1)
        if isinstance(result, str) and result.lower().startswith("failed"):
            return ActionResult(
                success=False,
                summary=f"Failed to open app '{identifier}': {result}\n{hint}",
            )
        _record_macro_action(
            ctx,
            {"action_type": "start_app", "package": identifier, "activity": None},
            pre_ui=pre_ui,
        )
        return ActionResult(success=True, summary=str(result))
    except Exception as e:
        return ActionResult(
            success=False,
            summary=f"Failed to open app '{identifier}': {e.__class__.__name__}: {e}\n{hint}",
        )


async def wait(duration: float = 1.0, *, ctx: "ActionContext") -> ActionResult:
    """Wait for a specified duration in seconds."""
    pre_ui = await _macro_pre_ui(ctx)
    await asyncio.sleep(duration)
    recorder = _macro_recorder(ctx)
    if recorder is not None:
        recorder.record_wait(duration, pre_ui=pre_ui)
    return ActionResult(success=True, summary=f"Waited for {duration} seconds")


# ---------------------------------------------------------------------------
# State / memory actions
# ---------------------------------------------------------------------------


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
        pre_ui = await _macro_pre_ui(ctx)

        # Tap the element first if a specific index is given
        if index != -1:
            x, y = ctx.ui.get_element_coords(index)
            await ctx.driver.tap(x, y)
            _record_macro_action(
                ctx,
                {"action_type": "tap", "x": x, "y": y},
                pre_ui=pre_ui,
            )
            pre_ui = await _macro_pre_ui_after_focus_tap(ctx)

        ok = await ctx.driver.input_text(secret_value)
        if ok:
            _record_macro_action(
                ctx,
                {
                    "action_type": "type_secret",
                    "secret_id": secret_id,
                    "clear": False,
                },
                pre_ui=pre_ui,
            )
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
