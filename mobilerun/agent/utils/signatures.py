"""Tool registry builder — single place for all standard mobilerun tools."""

import logging

from mobilerun.agent.tool_registry import ToolRegistry
from mobilerun.agent.utils.actions import (
    click,
    click_area,
    click_at,
    complete,
    long_press,
    long_press_at,
    open_app,
    open_bundle_id,
    remember,
    swipe,
    system_button,
    type_secret,
    type_text,
    type_text_direct,
    wait,
)

logger = logging.getLogger("mobilerun")


async def build_tool_registry(
    supported_buttons: set[str] | None = None,
    credential_manager=None,
    platform: str = "android",
    exact_app_launch: bool = False,
    screenshot_only: bool = False,
) -> tuple[ToolRegistry, set[str]]:
    """Build a ToolRegistry with all standard mobilerun tools.

    Args:
        supported_buttons: Buttons available for system_button description.
            Defaults to ``{"back", "home", "enter"}`` if *None*.
        credential_manager: If provided and has keys, ``type_secret`` is registered.
        platform: ``"android"`` or ``"ios"``. Controls which ``open_app``
            implementation is registered unless exact_app_launch is enabled.
        exact_app_launch: Register ``open_app`` as an exact app identifier
            launcher that only depends on ``start_app``.
        screenshot_only: When true, coordinate tool descriptions refer to the
            screenshot shown to the model. Normal mode keeps generic wording.

    Returns:
        ``(registry, standard_tool_names)`` where *standard_tool_names* is the
        set of tool names registered here.  The ManagerAgent uses this to
        exclude already-described tools from its ``<custom_actions>`` prompt
        section.  User/MCP tools added later by MobileAgent will NOT be in
        this set, so they correctly appear in ``<custom_actions>``.
    """
    registry = ToolRegistry()

    if screenshot_only:
        click_at_description = (
            "Click at screenshot position (x, y). Use screenshot pixel "
            "coordinates shown to the model. The coordinate grid is only a "
            "reference; do not use grid-cell numbers. Prefer click_at for "
            "dense lists, adjacent rows, compact menus, visible text, and "
            "small controls. "
            'Usage: {"action": "click_at", "x": 500, "y": 300}'
        )
        click_area_description = (
            "Click the center of a screenshot area (x1, y1, x2, y2). Use "
            "screenshot pixel coordinates shown to the model. The coordinate "
            "grid is only a reference; do not use grid-cell numbers. Use "
            "click_area only for large, unambiguous targets; prefer click_at "
            "for dense rows or text labels. "
            'Usage: {"action": "click_area", "x1": 100, "y1": 200, "x2": 300, "y2": 400}'
        )
        long_press_at_description = (
            "Long press at screenshot position (x, y). Use screenshot pixel "
            "coordinates shown to the model. The coordinate grid is only a "
            "reference; do not use grid-cell numbers. "
            'Usage: {"action": "long_press_at", "x": 500, "y": 300}'
        )
        swipe_description = (
            "Swipe from screenshot coordinate to coordinate2. Use screenshot "
            "pixel coordinates shown to the model. The coordinate grid is only "
            "a reference; do not use grid-cell numbers. Duration is in seconds "
            "(default: 1.0). "
            'Usage Example: {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2], "duration": 1.5}'
        )
    else:
        click_at_description = (
            "Click at screen position (x, y). "
            'Usage: {"action": "click_at", "x": 500, "y": 300}'
        )
        click_area_description = (
            "Click the center of area (x1, y1, x2, y2). Use click_area only "
            "for large, unambiguous targets. "
            'Usage: {"action": "click_area", "x1": 100, "y1": 200, "x2": 300, "y2": 400}'
        )
        long_press_at_description = (
            "Long press at screen position (x, y). "
            'Usage: {"action": "long_press_at", "x": 500, "y": 300}'
        )
        swipe_description = (
            "Swipe from the position with coordinate to the position with "
            "coordinate2. Duration is in seconds (default: 1.0). "
            'Usage Example: {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2], "duration": 1.5}'
        )

    # -- Core UI actions -----------------------------------------------------

    registry.register(
        "click",
        fn=click,
        params={"index": {"type": "number", "required": True}},
        description=(
            "Click the point on the screen with specified index. "
            'Usage Example: {"action": "click", "index": element_index}'
        ),
        deps={"tap", "element_index"},
    )

    registry.register(
        "long_press",
        fn=long_press,
        params={"index": {"type": "number", "required": True}},
        description=(
            "Long press on the position with specified index. "
            'Usage Example: {"action": "long_press", "index": element_index}'
        ),
        deps={"swipe", "element_index"},
    )

    registry.register(
        "click_at",
        fn=click_at,
        params={
            "x": {"type": "number", "required": True},
            "y": {"type": "number", "required": True},
        },
        description=click_at_description,
        deps={"tap", "convert_point"},
    )

    registry.register(
        "click_area",
        fn=click_area,
        params={
            "x1": {"type": "number", "required": True},
            "y1": {"type": "number", "required": True},
            "x2": {"type": "number", "required": True},
            "y2": {"type": "number", "required": True},
        },
        description=click_area_description,
        deps={"tap", "convert_point"},
    )

    registry.register(
        "long_press_at",
        fn=long_press_at,
        params={
            "x": {"type": "number", "required": True},
            "y": {"type": "number", "required": True},
        },
        description=long_press_at_description,
        deps={"swipe", "convert_point"},
    )

    registry.register(
        "type",
        fn=type_text,
        params={
            "text": {"type": "string", "required": True},
            "index": {"type": "number", "required": True},
            "clear": {"type": "boolean", "required": False, "default": False},
        },
        description=(
            "Type text into an input box or text field. Specify the element with "
            "index to focus the input field before typing. By default, text is "
            "APPENDED to existing content. Set clear=True to clear the field first "
            "(recommended for URL bars, search fields, or when replacing text). "
            'Usage Example: {"action": "type", "text": "example.com", "index": element_index, "clear": true}'
        ),
        deps={"tap", "input_text", "element_index"},
    )

    registry.register(
        "type_text",
        fn=type_text_direct,
        params={
            "text": {"type": "string", "required": True},
            "clear": {"type": "boolean", "required": False, "default": False},
        },
        description=(
            "Type text into the currently focused input field. Use a coordinate "
            "click first if the field is not focused. By default, text is "
            "APPENDED to existing content. Set clear=True to clear the field first. "
            'Usage Example: {"action": "type_text", "text": "example.com", "clear": true}'
        ),
        deps={"input_text", "direct_text_input"},
    )

    # -- system_button (dynamic description) ---------------------------------

    buttons = ", ".join(sorted(supported_buttons or set()))
    buttons_desc = f"Available buttons: {buttons}. " if buttons else ""
    registry.register(
        "system_button",
        fn=system_button,
        params={"button": {"type": "string", "required": True}},
        description=(
            f"Press a system button. {buttons_desc}"
            'Usage example: {"action": "system_button", "button": "home"}'
        ),
        deps={"press_button"},
    )

    # -- Navigation / timing -------------------------------------------------

    registry.register(
        "swipe",
        fn=swipe,
        params={
            "coordinate": {"type": "list", "required": True},
            "coordinate2": {"type": "list", "required": True},
            "duration": {"type": "number", "required": False, "default": 1.0},
        },
        description=swipe_description,
        deps={"swipe", "convert_point"},
    )

    registry.register(
        "wait",
        fn=wait,
        params={
            "duration": {"type": "number", "required": False, "default": 1.0},
        },
        description=(
            "Wait for a specified duration in seconds. Useful for waiting for "
            "animations, page loads, or other time-based operations. "
            'Usage Example: {"action": "wait", "duration": 2.0}'
        ),
    )

    # -- App / state / flow control ------------------------------------------

    if exact_app_launch:
        registry.register(
            "open_app",
            fn=open_bundle_id,
            params={"app_id": {"type": "string", "required": True}},
            description=(
                "Open an app by exact app identifier. Use the package name or "
                "bundle identifier required by the current device backend. "
                'Usage: {"action": "open_app", "app_id": "com.example.app"}'
            ),
            deps={"start_app"},
        )
    elif platform.lower() == "ios":
        registry.register(
            "open_app",
            fn=open_bundle_id,
            params={"bundle_id": {"type": "string", "required": True}},
            description=(
                "Open an app by its exact bundle identifier. "
                'Usage: {"action": "open_app", "bundle_id": "com.apple.Preferences"}'
            ),
            deps={"start_app"},
        )
    else:
        registry.register(
            "open_app",
            fn=open_app,
            params={"text": {"type": "string", "required": True}},
            description=(
                "Open an app by name or description. "
                'Usage: {"action": "open_app", "text": "Gmail"}'
            ),
            deps={"start_app", "get_apps"},
        )

    registry.register(
        "remember",
        fn=remember,
        params={"information": {"type": "string", "required": True}},
        description="Remember information for later use",
    )

    registry.register(
        "complete",
        fn=complete,
        params={
            "success": {"type": "boolean", "required": True},
            "message": {"type": "string", "required": True},
        },
        description=(
            "Mark task as complete. "
            "success=true if task succeeded, false if failed. "
            "message contains the result, answer, or explanation."
        ),
    )

    standard_tool_names = set(registry.tools.keys())

    # -- Credential tools (conditional) --------------------------------------

    if credential_manager is not None:
        available_secrets = await credential_manager.get_keys()
        if available_secrets:
            logger.debug(
                f"Building credential tools with {len(available_secrets)} secrets"
            )
            registry.register(
                "type_secret",
                fn=type_secret,
                params={
                    "secret_id": {"type": "string", "required": True},
                    "index": {"type": "number", "required": True},
                },
                description=(
                    "Type a secret credential from the credential store into an "
                    "input field. The agent never sees the actual secret value, "
                    "only the secret_id. "
                    'Usage: {"action": "type_secret", "secret_id": "MY_PASSWORD", "index": 5}'
                ),
                deps={"tap", "input_text", "element_index"},
            )
            standard_tool_names.add("type_secret")

    return registry, standard_tool_names
