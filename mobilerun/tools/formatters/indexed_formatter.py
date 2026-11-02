"""Indexed formatter - Standard Mobilerun format."""

from typing import Any, Dict, List, Optional, Tuple

from ..helpers.coordinate import bounds_to_normalized
from ..helpers.geometry import rects_overlap
from .base import TreeFormatter


class IndexedFormatter(TreeFormatter):
    """Format tree in the standard Mobilerun format."""

    def __init__(self):
        self.screen_width: Optional[int] = None
        self.screen_height: Optional[int] = None
        self.use_normalized: bool = False
        # Native-pixels-per-displayed-pixel of the screenshot shown to the
        # model. When != 1.0, the bounds in the model-facing text are emitted
        # in display space while ``bounds`` (used for real taps) stay native.
        self.display_scale_x: float = 1.0
        self.display_scale_y: float = 1.0

    def format(
        self, filtered_tree: Optional[Dict[str, Any]], phone_state: Dict[str, Any]
    ) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        """Format device state with indices and hierarchy."""
        focused_text = self._get_focused_text(phone_state)

        if filtered_tree is None:
            a11y_tree = []
        else:
            a11y_tree = self._flatten_with_index(filtered_tree, [1])

        phone_state_text = self._format_phone_state(phone_state)
        ui_elements_text = self._format_ui_elements_text(a11y_tree)

        formatted_text = f"{phone_state_text}\n\n{ui_elements_text}"

        return (formatted_text, focused_text, a11y_tree, phone_state)

    @staticmethod
    def _get_focused_text(phone_state: Dict[str, Any]) -> str:
        """Extract focused element text."""
        focused_element = phone_state.get("focusedElement")
        if focused_element:
            return focused_element.get("text", "")
        return ""

    @staticmethod
    def _format_phone_state(phone_state: Dict[str, Any]) -> str:
        """Format phone state."""
        if isinstance(phone_state, dict) and "error" not in phone_state:
            current_app = phone_state.get("currentApp", "")
            package_name = phone_state.get("packageName", "")
            focused_element = phone_state.get("focusedElement")
            is_editable = phone_state.get("isEditable", False)

            if focused_element and focused_element.get("text"):
                focused_desc = f"'{focused_element.get('text', '')}'"
            else:
                focused_desc = "''"

            # Build app line — skip package if empty, skip whole line if both empty
            if current_app and package_name:
                app_line = f"• **App:** {current_app} ({package_name})"
            elif current_app:
                app_line = f"• **App:** {current_app}"
            elif package_name:
                app_line = f"• **App:** {package_name}"
            else:
                app_line = ""

            lines = ["**Current Phone State:**"]
            if app_line:
                lines.append(app_line)
            lines.append(f"• **Keyboard:** {'Visible' if is_editable else 'Hidden'}")
            lines.append(f"• **Focused Element:** {focused_desc}")

            phone_state_text = "\n".join(lines)
        else:
            if isinstance(phone_state, dict) and "error" in phone_state:
                phone_state_text = f"📱 **Phone State Error:** {phone_state.get('message', 'Unknown error')}"
            else:
                phone_state_text = f"📱 **Phone State:** {phone_state}"

        return phone_state_text

    def _format_ui_elements_text(self, a11y_tree: List[Dict[str, Any]]) -> str:
        """Format UI elements text."""
        # Label only bounds that were actually normalized.
        normalized = self.use_normalized and self.screen_width and self.screen_height
        coord_note = " (normalized [0-1000])" if normalized else ""
        schema = (
            "'index. className: resourceId; checkedState, text - bounds(x1,y1,x2,y2)'"
        )
        if a11y_tree:
            formatted_ui = IndexedFormatter._format_ui_elements(a11y_tree)
            ui_elements_text = (
                f"Current Clickable UI elements{coord_note}:\n{schema}:\n{formatted_ui}"
            )
        else:
            ui_elements_text = f"Current Clickable UI elements{coord_note}:\n{schema}:\nNo UI elements found"
        return ui_elements_text

    @staticmethod
    def _format_ui_elements(ui_data: List[Dict[str, Any]], level: int = 0) -> str:
        """Format UI elements."""
        if not ui_data:
            return ""

        formatted_lines = []
        indent = "  " * level

        elements = ui_data if isinstance(ui_data, list) else [ui_data]

        for element in elements:
            if not isinstance(element, dict):
                continue

            index = element.get("index", "")
            class_name = element.get("className", "")
            resource_id = element.get("resourceId", "")
            text = element.get("text", "")
            # Model-facing text shows display-space or normalized bounds;
            # "bounds" (native pixels) drive real taps.
            bounds = element.get("displayBounds") or element.get("bounds", "")
            checkedState = element.get("checkedState", "")
            children = element.get("children") or []

            line_parts = []
            if index != "":
                line_parts.append(f"{index}.")
            if class_name:
                line_parts.append(class_name + ":")

            details = []
            if resource_id:
                details.append(f'"{resource_id}"')
            if text:
                details.append(f'"{text}"')

            if details:
                line_parts.append(", ".join(details))

            if checkedState:
                line_parts.append(f"; {checkedState}")

            if bounds:
                line_parts.append(f"- ({bounds})")

            formatted_line = f"{indent}{' '.join(line_parts)}"
            formatted_lines.append(formatted_line)

            if children:
                child_formatted = IndexedFormatter._format_ui_elements(
                    children, level + 1
                )
                if child_formatted:
                    formatted_lines.append(child_formatted)

        return "\n".join(formatted_lines)

    def _flatten_with_index(
        self, node: Dict[str, Any], counter: List[int]
    ) -> List[Dict[str, Any]]:
        """Recursively flatten tree with index assignment."""
        results = []

        formatted = self._format_node(node, counter[0])
        results.append(formatted)
        counter[0] += 1

        siblings = []
        for child in node.get("children") or []:
            descendants = self._flatten_with_index(child, counter)
            siblings.append((child, descendants[0]))
            results.extend(descendants)

        self._add_tap_blockers(siblings)

        return results

    @staticmethod
    def _add_tap_blockers(
        siblings: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> None:
        """Retain proven sibling touch obstructions before discarding hierarchy.

        Android drawingOrder is relative to siblings, not a global z-index.
        Portal can append other windows as children, so windowId must match.
        Missing or tied order is insufficient evidence to change existing taps.
        """
        for raw, formatted in siblings:
            order = raw.get("drawingOrder")
            window = raw.get("windowId")
            if not isinstance(order, int) or not isinstance(window, int) or window < 0:
                continue
            bounds = tuple(map(int, formatted["bounds"].split(",")))
            blockers = []
            for other, candidate in siblings:
                other_order = other.get("drawingOrder")
                if (
                    other.get("windowId") != window
                    or not isinstance(other_order, int)
                    or other_order <= order
                    or other.get("isVisibleToUser") is False
                    or not (other.get("isClickable") or other.get("isLongClickable"))
                ):
                    continue
                candidate_bounds = tuple(map(int, candidate["bounds"].split(",")))
                if rects_overlap(bounds, candidate_bounds):
                    blockers.append(candidate["bounds"])
            if blockers:
                formatted["tapBlockers"] = blockers

    def _format_node(self, node: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format single node to Mobilerun format."""
        bounds = node.get("boundsInScreen", {})
        bounds_str = f"{bounds.get('left', 0)},{bounds.get('top', 0)},{bounds.get('right', 0)},{bounds.get('bottom', 0)}"

        display_bounds_str = None
        if self.use_normalized and self.screen_width and self.screen_height:
            display_bounds_str = bounds_to_normalized(
                bounds_str, self.screen_width, self.screen_height
            )
        elif self.display_scale_x != 1.0 or self.display_scale_y != 1.0:
            display_bounds_str = ",".join(
                str(round(value / scale))
                for value, scale in zip(
                    (int(part) for part in bounds_str.split(",")),
                    (
                        self.display_scale_x,
                        self.display_scale_y,
                        self.display_scale_x,
                        self.display_scale_y,
                    ),
                    strict=True,
                )
            )

        text = (
            node.get("text")
            or node.get("contentDescription")
            or node.get("resourceId")
            or node.get("className", "")
        )

        class_name = node.get("className", "")
        short_class = class_name.split(".")[-1] if class_name else ""

        checked_state = ""
        if node.get("isCheckable"):
            checked_state = (
                "isChecked=True" if node.get("isChecked") else "isChecked=False"
            )

        formatted = {
            "index": index,
            "resourceId": node.get("resourceId", ""),
            "className": short_class,
            "checkedState": checked_state,
            "text": text,
            "bounds": bounds_str,
            "children": [],
        }
        if display_bounds_str is not None:
            formatted["displayBounds"] = display_bounds_str
        return formatted
