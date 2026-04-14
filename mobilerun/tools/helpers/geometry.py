"""Geometry utilities for UI element bounds and tap point calculation."""

from typing import List, Tuple, Optional

Bounds = Tuple[int, int, int, int]


def rects_overlap(a: Bounds, b: Bounds) -> bool:
    """Check if two rectangles overlap."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def find_clear_point(
    bounds: Bounds,
    blockers: List[Bounds],
    depth: int = 0,
) -> Optional[Tuple[int, int]]:
    """Find a clear point in bounds using quadrant subdivision."""
    left, top, right, bottom = bounds
    cx, cy = (left + right) // 2, (top + bottom) // 2

    blocked = any(b[0] <= cx < b[2] and b[1] <= cy < b[3] for b in blockers)

    if not blocked:
        return cx, cy

    if depth > 4 or (right - left) * (bottom - top) < 100:
        return None

    quadrants = [
        (left, top, cx, cy),
        (cx, top, right, cy),
        (left, cy, cx, bottom),
        (cx, cy, right, bottom),
    ]

    best_point = None
    best_area = 0

    for q in quadrants:
        q_area = (q[2] - q[0]) * (q[3] - q[1])
        if q_area <= 0:
            continue
        point = find_clear_point(q, blockers, depth + 1)
        if point and q_area > best_area:
            best_point = point
            best_area = q_area

    return best_point
