"""UIState — parsed UI elements with element resolution and coordinate conversion.

Replaces ``clickable_elements_cache``, ``_extract_element_coordinates_by_index``,
and the scattered ``find_element_by_index`` local functions from ``adb.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from droidrun.tools.helpers.coordinate import to_absolute
from droidrun.tools.helpers.geometry import find_clear_point, rects_overlap


class UIState:
    """Holds parsed UI elements for a single device state snapshot."""

    def __init__(
        self,
        elements: List[Dict[str, Any]],
        formatted_text: str,
        focused_text: str,
        phone_state: Dict[str, Any],
        screen_width: int,
        screen_height: int,
        use_normalized: bool = False,
    ) -> None:
        self.elements = elements
        self.formatted_text = formatted_text
        self.focused_text = focused_text
        self.phone_state = phone_state
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.use_normalized = use_normalized

    # -- element lookup ------------------------------------------------------

    def get_element(self, index: int) -> Optional[Dict[str, Any]]:
        """Recursively find an element by its index."""
        return self._find_by_index(self.elements, index)

    def get_element_coords(self, index: int) -> Tuple[int, int]:
        """Return the centre (x, y) of element *index*.

        Raises ``ValueError`` when the element is missing or has no bounds.
        """
        element = self._find_by_index(self.elements, index)

        if element is None:
            indices = sorted(self._collect_indices(self.elements))
            indices_str = ", ".join(str(i) for i in indices[:20])
            if len(indices) > 20:
                indices_str += f"... and {len(indices) - 20} more"
            raise ValueError(
                f"No element found with index {index}. "
                f"Available indices: {indices_str}"
            )

        bounds_str = element.get("bounds")
        if not bounds_str:
            text = element.get("text", "No text")
            cls = element.get("className", "Unknown class")
            etype = element.get("type", "unknown")
            raise ValueError(
                f"Element with index {index} ('{text}', {cls}, type: {etype}) "
                f"has no bounds and cannot be tapped"
            )

        try:
            left, top, right, bottom = map(int, bounds_str.split(","))
        except ValueError as e:
            raise ValueError(
                f"Invalid bounds format for element with index {index}: "
                f"{bounds_str}"
            ) from e

        return (left + right) // 2, (top + bottom) // 2

    def get_element_info(self, index: int) -> Dict[str, Any]:
        """Return a dict with common element fields for display."""
        element = self.get_element(index)
        if element is None:
            return {}

        info: Dict[str, Any] = {
            "text": element.get("text", "No text"),
            "className": element.get("className", "Unknown class"),
            "type": element.get("type", "unknown"),
        }

        children = element.get("children", [])
        if children:
            child_texts = [c.get("text") for c in children if c.get("text")]
            if child_texts:
                info["child_texts"] = child_texts

        return info

    def get_clear_point(self, index: int) -> Tuple[int, int]:
        """Find a tap point for *index* that avoids overlapping elements.

        Falls back to the centre if no clear point exists.
        """
        element = self._find_by_index(self.elements, index)
        if element is None:
            raise ValueError(f"No element found with index {index}")

        bounds_str = element.get("bounds")
        if not bounds_str:
            raise ValueError(f"Element {index} has no bounds")

        target_bounds = tuple(map(int, bounds_str.split(",")))

        all_elements = self._collect_all(self.elements)
        blockers = []
        for el in all_elements:
            el_idx = el.get("index")
            el_bounds_str = el.get("bounds")
            if el_idx is not None and el_idx > index and el_bounds_str:
                el_bounds = tuple(map(int, el_bounds_str.split(",")))
                if rects_overlap(target_bounds, el_bounds):
                    blockers.append(el_bounds)

        point = find_clear_point(target_bounds, blockers)
        if point is None:
            raise ValueError(
                f"Element {index} is fully obscured by overlapping elements"
            )
        return point

    def convert_point(self, x: int, y: int) -> Tuple[int, int]:
        """Convert point to absolute pixels if normalized mode is active."""
        if self.use_normalized:
            return to_absolute(x, y, self.screen_width, self.screen_height)
        return x, y

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _find_by_index(
        elements: List[Dict[str, Any]], target: int
    ) -> Optional[Dict[str, Any]]:
        for item in elements:
            if item.get("index") == target:
                return item
            child = UIState._find_by_index(item.get("children", []), target)
            if child is not None:
                return child
        return None

    @staticmethod
    def _collect_indices(elements: List[Dict[str, Any]]) -> List[int]:
        indices: List[int] = []
        for item in elements:
            if item.get("index") is not None:
                indices.append(item["index"])
            indices.extend(UIState._collect_indices(item.get("children", [])))
        return indices

    @staticmethod
    def _collect_all(
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for item in elements:
            result.append(item)
            result.extend(UIState._collect_all(item.get("children", [])))
        return result
