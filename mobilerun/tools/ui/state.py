"""UIState — parsed UI elements with element resolution and coordinate conversion.

Replaces ``clickable_elements_cache``, ``_extract_element_coordinates_by_index``,
and the scattered ``find_element_by_index`` local functions from ``adb.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mobilerun.tools.helpers.coordinate import NORMALIZED_MAX, to_absolute
from mobilerun.tools.helpers.geometry import (
    find_clear_point,
    find_uncovered_point,
    rects_overlap,
)


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
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
        coordinate_contract_active: bool = False,
        model_screenshot_width: Optional[int] = None,
        model_screenshot_height: Optional[int] = None,
    ) -> None:
        self.elements = elements
        self.formatted_text = formatted_text
        self.focused_text = focused_text
        self.phone_state = phone_state
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.use_normalized = use_normalized
        self.coordinate_scale_x = coordinate_scale_x
        self.coordinate_scale_y = coordinate_scale_y
        # Exact pixel size the model-facing screenshot must be resized to so the
        # image the model grounds on matches this snapshot's declared coordinate
        # space (== convert_point basis). None means "no contract / native".
        self.model_screenshot_width = model_screenshot_width
        self.model_screenshot_height = model_screenshot_height
        # Whether a coordinate contract (resized+grid screenshot with declared
        # display space, or normalized 0-1000 with a real screen size) is
        # active for this snapshot. Action-time
        # coordinate guards read it from here so they match the space this
        # snapshot's convert_point uses.
        self.coordinate_contract_active = coordinate_contract_active

    # -- element lookup ------------------------------------------------------

    def get_element(self, index: int) -> Optional[Dict[str, Any]]:
        """Recursively find an element by its index."""
        return self._find_by_index(self.elements, index)

    def get_element_coords(self, index: int) -> Tuple[int, int]:
        """Return the centre, avoiding known touchable sibling obstructions.

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

        return self._avoid_tap_blockers(
            element, ((left + right) // 2, (top + bottom) // 2)
        )

    def _avoid_tap_blockers(
        self, element: Dict[str, Any], point: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Keep a preferred point unless the formatter identified an obstruction."""
        blockers = [
            tuple(map(int, bounds.split(",")))
            for bounds in element.get("tapBlockers", [])
        ]
        x, y = point
        if not any(
            left <= x < right and top <= y < bottom
            for left, top, right, bottom in blockers
        ):
            return point
        left, top, right, bottom = map(int, element["bounds"].split(","))
        if self.screen_width and self.screen_height:
            left, top = max(0, left), max(0, top)
            right = min(self.screen_width, right)
            bottom = min(self.screen_height, bottom)
        clear = find_uncovered_point((left, top, right, bottom), blockers)
        if clear is None:
            raise ValueError(f"No clear tap point for element {element.get('index')}")
        return clear

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

        children = element.get("children") or []
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
            abs_x, abs_y = to_absolute(x, y, self.screen_width, self.screen_height)
            # 1000 maps to one past the last pixel; keep it on screen.
            if x == NORMALIZED_MAX:
                abs_x = max(self.screen_width - 1, 0)
            if y == NORMALIZED_MAX:
                abs_y = max(self.screen_height - 1, 0)
            return abs_x, abs_y
        return (
            int(round(x * self.coordinate_scale_x)),
            int(round(y * self.coordinate_scale_y)),
        )

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _find_by_index(
        elements: List[Dict[str, Any]], target: int
    ) -> Optional[Dict[str, Any]]:
        for item in elements:
            if item.get("index") == target:
                return item
            child = UIState._find_by_index(item.get("children") or [], target)
            if child is not None:
                return child
        return None

    @staticmethod
    def _collect_indices(elements: List[Dict[str, Any]]) -> List[int]:
        indices: List[int] = []
        for item in elements:
            if item.get("index") is not None:
                indices.append(item["index"])
            indices.extend(UIState._collect_indices(item.get("children") or []))
        return indices

    @staticmethod
    def _collect_all(
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for item in elements:
            result.append(item)
            result.extend(UIState._collect_all(item.get("children") or []))
        return result
