"""IOSStateProvider — builds UIState from iOS portal accessibility data.

Parses the raw text-based accessibility tree returned by the iOS portal
into structured elements compatible with UIState.

Known limitations:
- Normalized coordinates untested on iOS
- No filter/formatter pipeline (iOS a11y tree is raw text, not structured JSON)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from droidrun.tools.driver.base import DeviceDisconnectedError, DeviceDriver
from droidrun.tools.ui.provider import StateProvider
from droidrun.tools.ui.state import UIState

logger = logging.getLogger("droidrun")

# Element types to skip — layout containers that add noise without useful info.
# Everything else is kept (buttons, cells, text, icons, images, etc.)
_SKIP_TYPES = {
    "Window",
    "Window (Main)",
    "ScrollView",
    "CollectionView",
    "Table",
    "Toolbar",
    "TabBar",
    "StatusBar",
    "PageIndicator",
}

_COORD_RE = re.compile(r"\{\{([0-9.]+),\s*([0-9.]+)\},\s*\{([0-9.]+),\s*([0-9.]+)\}\}")
_ELEMENT_TYPE_RE = re.compile(r"\s*(.+?),")
_LABEL_RE = re.compile(r"label:\s*'([^']*)'")
_IDENTIFIER_RE = re.compile(r"identifier:\s*'([^']*)'")
_PLACEHOLDER_RE = re.compile(r"placeholderValue:\s*'([^']*)'")
_VALUE_RE = re.compile(r"value:\s*([^,}]+)")
_CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}$")


class IOSStateProvider(StateProvider):
    """Produces ``UIState`` from an iOS device's accessibility tree."""

    supported = {"element_index", "convert_point"}

    def __init__(self, driver: DeviceDriver, use_normalized: bool = False) -> None:
        super().__init__(driver)
        self.use_normalized = use_normalized

    async def get_state(self) -> UIState:
        try:
            raw = await self.driver.get_ui_tree()
        except DeviceDisconnectedError:
            raise
        except Exception as e:
            logger.warning(f"iOS state retrieval failed, returning empty state: {e}")
            return UIState(
                elements=[],
                formatted_text="No UI elements available — device may be loading.",
                focused_text="",
                phone_state={},
                screen_width=390,
                screen_height=844,
                use_normalized=self.use_normalized,
            )

        a11y_text = raw.get("a11y_tree", "")
        phone_state = dict(raw.get("phone_state", {}) or {})
        device_context = raw.get("device_context", {})

        elements = _parse_a11y_tree(a11y_text)
        phone_state = _normalize_phone_state(phone_state, a11y_text)

        # Screen size from device_context
        screen_bounds = device_context.get("screen_bounds", {})
        screen_width = int(screen_bounds.get("width", 390))
        screen_height = int(screen_bounds.get("height", 844))

        formatted_text = _format_elements(elements, screen_width, screen_height)

        # Extract focused text from phone_state
        focused_element = phone_state.get("focusedElement")
        focused_text = ""
        if focused_element and isinstance(focused_element, dict):
            focused_text = focused_element.get("text", "")

        return UIState(
            elements=elements,
            formatted_text=formatted_text,
            focused_text=focused_text,
            phone_state=phone_state,
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=self.use_normalized,
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_a11y_tree(a11y_text: str) -> List[Dict[str, Any]]:
    """Parse iOS accessibility tree text into structured elements.

    Moved verbatim from ``IOSTools._parse_ios_accessibility_tree``.
    """
    elements: List[Dict[str, Any]] = []
    element_index = 0

    seen_signatures: set[tuple[str, str, str]] = set()

    for line in a11y_text.strip().split("\n"):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("Attributes:")
            or stripped.startswith("Element subtree:")
            or stripped.startswith("Path to element:")
            or stripped.startswith("Query chain:")
        ):
            continue

        coord_match = _COORD_RE.search(line)
        if not coord_match:
            continue

        x, y, width, height = map(float, coord_match.groups())

        # Skip elements with no tappable area
        if width == 0 or height == 0:
            continue

        # Element type
        type_match = _ELEMENT_TYPE_RE.match(line)
        element_type = type_match.group(1).strip() if type_match else "Unknown"
        element_type = re.sub(r"^[→\s]+", "", element_type)

        # Skip layout containers that add noise without useful info
        if element_type in _SKIP_TYPES:
            continue

        # Extract properties
        label_m = _LABEL_RE.search(line)
        label = label_m.group(1) if label_m else ""
        ident_m = _IDENTIFIER_RE.search(line)
        identifier = ident_m.group(1) if ident_m else ""
        ph_m = _PLACEHOLDER_RE.search(line)
        placeholder = ph_m.group(1) if ph_m else ""
        val_m = _VALUE_RE.search(line)
        value = val_m.group(1).strip() if val_m else ""

        text = label or identifier or placeholder or ""

        # Bounds in "left,top,right,bottom" format — compatible with UIState
        bounds_str = f"{int(x)},{int(y)},{int(x + width)},{int(y + height)}"

        # Filter noisy wrapper nodes that duplicate a more useful action target.
        signature = (element_type, text, bounds_str)
        if element_type == "Other" and not (
            label or identifier or placeholder or value
        ):
            continue
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        elements.append(
            {
                "index": element_index,
                "type": element_type,
                "className": element_type,
                "text": text,
                "label": label,
                "identifier": identifier,
                "placeholder": placeholder,
                "value": value,
                "bounds": bounds_str,
                "rect": f"{x},{y},{width},{height}",
                "children": [],
            }
        )
        element_index += 1

    return _prioritize_actionable_elements(elements)


def _normalize_phone_state(
    phone_state: Dict[str, Any], a11y_text: str
) -> Dict[str, Any]:
    package_name = phone_state.get("packageName", "") or ""
    current_app = phone_state.get("currentApp", "") or ""

    is_home = (
        package_name == "com.apple.springboard" or "Home screen icons" in a11y_text
    )
    if is_home:
        phone_state["packageName"] = "com.apple.springboard"
        if not current_app or _CLOCK_RE.match(current_app):
            phone_state["currentApp"] = "Home Screen"
    elif current_app and _CLOCK_RE.match(current_app):
        phone_state["currentApp"] = ""

    return phone_state


def _prioritize_actionable_elements(
    elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    actionable_types = {
        "Icon",
        "Button",
        "SearchField",
        "TextField",
        "SecureTextField",
        "TextView",
        "Cell",
        "StaticText",
        "Image",
        "Switch",
    }

    def sort_key(el: Dict[str, Any]) -> tuple[int, int]:
        class_name = el.get("className", "")
        text = el.get("text", "")
        actionable_rank = 0 if class_name in actionable_types and text else 1
        return (actionable_rank, el.get("index", 0))

    ordered = sorted(elements, key=sort_key)
    for i, el in enumerate(ordered):
        el["index"] = i
    return ordered


# ---------------------------------------------------------------------------
# Formatting for agent prompt
# ---------------------------------------------------------------------------


def _format_elements(
    elements: List[Dict[str, Any]],
    screen_width: int,
    screen_height: int,
) -> str:
    """Build the text representation shown to the agent."""
    schema = "'index. className: text - bounds(x1,y1,x2,y2)'"
    if not elements:
        return f"Current UI elements:\n{schema}:\nNo UI elements found"

    lines = [f"Current UI elements:\n{schema}:"]
    for el in elements:
        idx = el.get("index", 0)
        cls = el.get("className", "Unknown")
        text = el.get("text", "")
        bounds = el.get("bounds", "")

        parts = [f"{idx}. {cls}:"]
        if text:
            parts.append(text)
        if bounds:
            parts.append(f"- ({bounds})")
        lines.append(" ".join(parts))

    return "\n".join(lines)
