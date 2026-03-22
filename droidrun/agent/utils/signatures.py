"""Action signatures and credential tool builder."""

import logging
from typing import Iterable

from droidrun.agent.utils.actions import (
    click,
    click_at,
    click_area,
    long_press,
    long_press_at,
    swipe,
    system_button,
    type,
    type_secret,
    wait,
)


def build_atomic_action_signatures(
    supported_system_buttons: Iterable[str] | None = None,
) -> dict:
    """Build atomic action signatures for the active device.

    Only ``system_button`` varies by platform today; all other action
    signatures are stable across drivers.`
    """
    buttons = sorted(
        button.lower() for button in (supported_system_buttons or ("back", "home", "enter"))
    )
    button_list = ", ".join(buttons)
    example_button = buttons[0] if buttons else "back"

    return {
        "click": {
            "parameters": {
                "index": {"type": "number", "required": True},
            },
            "description": 'Click the point on the screen with specified index. Usage Example: {"action": "click", "index": element_index}',
            "function": click,
            "deps": {"tap", "element_index"},
        },
        "long_press": {
            "parameters": {
                "index": {"type": "number", "required": True},
            },
            "description": 'Long press on the position with specified index. Usage Example: {"action": "long_press", "index": element_index}',
            "function": long_press,
            "deps": {"swipe", "element_index"},
        },
        "click_at": {
            "parameters": {
                "x": {"type": "number", "required": True},
                "y": {"type": "number", "required": True},
            },
            "description": 'Click at screen position (x, y). Use element bounds as reference to determine where to click. Usage: {"action": "click_at", "x": 500, "y": 300}',
            "function": click_at,
            "deps": {"tap", "convert_point"},
        },
        "click_area": {
            "parameters": {
                "x1": {"type": "number", "required": True},
                "y1": {"type": "number", "required": True},
                "x2": {"type": "number", "required": True},
                "y2": {"type": "number", "required": True},
            },
            "description": 'Click center of area (x1, y1, x2, y2). Useful when you want to click a specific region. Usage: {"action": "click_area", "x1": 100, "y1": 200, "x2": 300, "y2": 400}',
            "function": click_area,
            "deps": {"tap", "convert_point"},
        },
        "long_press_at": {
            "parameters": {
                "x": {"type": "number", "required": True},
                "y": {"type": "number", "required": True},
            },
            "description": 'Long press at screen position (x, y). Use element bounds as reference. Usage: {"action": "long_press_at", "x": 500, "y": 300}',
            "function": long_press_at,
            "deps": {"swipe", "convert_point"},
        },
        "type": {
            "parameters": {
                "text": {"type": "string", "required": True},
                "index": {"type": "number", "required": True},
                "clear": {"type": "boolean", "required": False, "default": False},
            },
            "description": 'Type text into an input box or text field. Specify the element with index to focus the input field before typing. By default, text is APPENDED to existing content. Set clear=True to clear the field first (recommended for URL bars, search fields, or when replacing text). Usage Example: {"action": "type", "text": "example.com", "index": element_index, "clear": true}',
            "function": type,
            "deps": {"tap", "input_text", "element_index"},
        },
        "system_button": {
            "parameters": {
                "button": {"type": "string", "required": True},
            },
            "description": f'Press a system button. Available buttons on this device: {button_list}. Usage example: {{"action": "system_button", "button": "{example_button}"}}',
            "function": system_button,
            "deps": {"press_key"},
        },
        "swipe": {
            "parameters": {
                "coordinate": {"type": "list", "required": True},
                "coordinate2": {"type": "list", "required": True},
                "duration": {"type": "number", "required": False, "default": 1.0},
            },
            "description": 'Scroll from the position with coordinate to the position with coordinate2. Duration is in seconds (default: 1.0). Usage Example: {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2], "duration": 1.5}',
            "function": swipe,
            "deps": {"swipe", "convert_point"},
        },
        "wait": {
            "parameters": {
                "duration": {"type": "number", "required": False, "default": 1.0},
            },
            "description": 'Wait for a specified duration in seconds. Useful for waiting for animations, page loads, or other time-based operations. Usage Example: {"action": "wait", "duration": 2.0}',
            "function": wait,
        },
    }


ATOMIC_ACTION_SIGNATURES = build_atomic_action_signatures()


async def build_credential_tools(credential_manager) -> dict:
    """Build credential-related custom tools if credential manager is available."""
    logger = logging.getLogger("droidrun")

    if credential_manager is None:
        return {}

    available_secrets = await credential_manager.get_keys()
    if not available_secrets:
        logger.debug("No enabled secrets found, credential tools disabled")
        return {}

    logger.debug(f"Building credential tools with {len(available_secrets)} secrets")

    return {
        "type_secret": {
            "parameters": {
                "secret_id": {"type": "string", "required": True},
                "index": {"type": "number", "required": True},
            },
            "description": 'Type a secret credential from the credential store into an input field. The agent never sees the actual secret value, only the secret_id. Usage: {"action": "type_secret", "secret_id": "MY_PASSWORD", "index": 5}',
            "function": type_secret,
            "deps": {"tap", "input_text", "element_index"},
        },
    }
