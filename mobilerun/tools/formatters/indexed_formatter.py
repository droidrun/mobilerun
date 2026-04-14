"""Indexed formatter - Standard Droidrun format."""

from typing import Dict, Any, List, Optional, Tuple
from .base import TreeFormatter
from ..helpers.coordinate import bounds_to_normalized


class IndexedFormatter(TreeFormatter):
    """Format tree in the standard Droidrun format."""

    def __init__(self):
        self.screen_width: Optional[int] = None
        self.screen_height: Optional[int] = None
        self.use_normalized: bool = False

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
        coord_note = " (normalized [0-1000])" if self.use_normalized else ""
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
            bounds = element.get("bounds", "")
            checkedState = element.get("checkedState", "")
            children = element.get("children", [])

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

        for child in node.get("children", []):
            results.extend(self._flatten_with_index(child, counter))

        return results

    def _format_node(self, node: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format single node to Droidrun format."""
        bounds = node.get("boundsInScreen", {})
        bounds_str = f"{bounds.get('left', 0)},{bounds.get('top', 0)},{bounds.get('right', 0)},{bounds.get('bottom', 0)}"

        if self.use_normalized and self.screen_width and self.screen_height:
            bounds_str = bounds_to_normalized(
                bounds_str, self.screen_width, self.screen_height
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

        return {
            "index": index,
            "resourceId": node.get("resourceId", ""),
            "className": short_class,
            "checkedState": checked_state,
            "text": text,
            "bounds": bounds_str,
            "children": [],
        }
