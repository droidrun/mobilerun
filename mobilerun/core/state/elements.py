"""Typed representation of elements emitted by the UI tree formatters."""

from mobilerun.utils.geometry import Bounds


def _parse_bounds(value: str) -> Bounds:
    left, top, right, bottom = map(int, value.split(","))
    return left, top, right, bottom
