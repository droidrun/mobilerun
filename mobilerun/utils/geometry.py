"""Geometry utilities for UI element bounds and tap point calculation."""

from typing import List, Optional, Tuple

Bounds = Tuple[int, int, int, int]


def rects_overlap(a: Bounds, b: Bounds) -> bool:
    """Check if two rectangles overlap."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def find_uncovered_point(
    bounds: Bounds, blockers: List[Bounds]
) -> Optional[Tuple[int, int]]:
    """Return the center of the largest remaining rectangle, including narrow gaps."""
    regions = [bounds] if bounds[0] < bounds[2] and bounds[1] < bounds[3] else []
    for blocker in blockers:
        remaining = []
        for left, top, right, bottom in regions:
            bl = max(left, blocker[0])
            bt = max(top, blocker[1])
            br = min(right, blocker[2])
            bb = min(bottom, blocker[3])
            if bl >= br or bt >= bb:
                remaining.append((left, top, right, bottom))
                continue
            for region in (
                (left, top, right, bt),
                (left, bb, right, bottom),
                (left, bt, bl, bb),
                (br, bt, right, bb),
            ):
                if region[0] < region[2] and region[1] < region[3]:
                    remaining.append(region)
        regions = remaining
    if not regions:
        return None
    left, top, right, bottom = max(regions, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    return (left + right) // 2, (top + bottom) // 2


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
