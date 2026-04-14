"""Tool registry builder — single place for all standard droidrun tools."""

import logging

from droidrun.agent.tool_registry import ToolRegistry
from droidrun.agent.utils.actions import (
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
    type_text,
    type_secret,
    wait,
)

logger = logging.getLogger("droidrun")


async def build_tool_registry(
    supported_buttons: set[str] | None = None,
    credential_manager=None,
    platform: str = "android",
) -> tuple[ToolRegistry, set[str]]:
    """Build a ToolRegistry with all standard droidrun tools.

    Args:
        supported_buttons: Buttons available for system_button description.
            Defaults to ``{"back", "home", "enter"}`` if *None*.
        credential_manager: If provided and has keys, ``type_secret`` is registered.
        platform: ``"android"`` or ``"ios"``. Controls which ``open_app``
            implementation is registered.

    Returns:
        ``(registry, standard_tool_names)`` where *standard_tool_names* is the
        set of tool names registered here.  The ManagerAgent uses this to
        exclude already-described tools from its ``<custom_actions>`` prompt
        section.  User/MCP tools added later by DroidAgent will NOT be in
        this set, so they correctly appear in ``<custom_actions>``.
    """
    registry = ToolRegistry()

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
        description=(
            "Click at screen position (x, y). Use element bounds as reference "
            'to determine where to click. Usage: {"action": "click_at", "x": 500, "y": 300}'
        ),
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
        description=(
            "Click center of area (x1, y1, x2, y2). Useful when you want to click "
            'a specific region. Usage: {"action": "click_area", "x1": 100, "y1": 200, "x2": 300, "y2": 400}'
        ),
        deps={"tap", "convert_point"},
    )

    registry.register(
        "long_press_at",
        fn=long_press_at,
        params={
            "x": {"type": "number", "required": True},
            "y": {"type": "number", "required": True},
        },
        description=(
            "Long press at screen position (x, y). Use element bounds as reference. "
            'Usage: {"action": "long_press_at", "x": 500, "y": 300}'
        ),
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
        description=(
            "Scroll from the position with coordinate to the position with "
            "coordinate2. Duration is in seconds (default: 1.0). "
            'Usage Example: {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2], "duration": 1.5}'
        ),
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

    if platform.lower() == "ios":
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
