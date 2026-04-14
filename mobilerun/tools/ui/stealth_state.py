"""StealthUIState — randomized coordinate resolution for human-like taps.

Overrides ``UIState.get_element_coords()`` and ``get_clear_point()``
to return randomized positions within element bounds, avoiding the
predictable dead-center taps that betray automation.
"""

from __future__ import annotations

import random
from typing import Tuple

from droidrun.tools.ui.state import UIState


class StealthUIState(UIState):
    """UIState variant that randomizes tap coordinates within element bounds."""

    def get_element_coords(self, index: int) -> Tuple[int, int]:
        """Return a randomized point within the safe zone of element *index*.

        The safe zone is the inner 40% of the element (20% inset from each
        edge), ensuring taps land clearly inside the element.

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

        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        width = right - left
        height = bottom - top

        # Safe zone: inner 40% (20% inset from each edge)
        safe_zone = 0.4
        x_range = max(5, int(width * safe_zone))
        y_range = max(5, int(height * safe_zone))

        x = center_x + random.randint(-x_range // 2, x_range // 2)
        y = center_y + random.randint(-y_range // 2, y_range // 2)

        # Clamp to stay within bounds
        x = max(left + 2, min(x, right - 2))
        y = max(top + 2, min(y, bottom - 2))

        return x, y

    def get_clear_point(self, index: int) -> Tuple[int, int]:
        """Find a clear tap point for *index*, then randomize around it.

        Uses the parent overlap-avoidance logic, then adds jitter within
        a 20% radius of the element dimensions.
        """
        # Get the overlap-free center from parent
        cx, cy = super().get_clear_point(index)

        element = self._find_by_index(self.elements, index)
        bounds_str = element.get("bounds", "")
        if not bounds_str:
            return cx, cy

        left, top, right, bottom = map(int, bounds_str.split(","))
        width = right - left
        height = bottom - top

        x_range = max(5, int(width * 0.2))
        y_range = max(5, int(height * 0.2))

        x = cx + random.randint(-x_range // 2, x_range // 2)
        y = cy + random.randint(-y_range // 2, y_range // 2)

        x = max(left + 2, min(x, right - 2))
        y = max(top + 2, min(y, bottom - 2))

        return x, y
