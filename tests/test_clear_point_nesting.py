"""Tests for UIState.get_clear_point and its handling of nested elements.

Regression cover for #199. Element indices are assigned by a pre-order walk
(IndexedFormatter._flatten_with_index), so a descendant always has a higher
index than its ancestor. get_clear_point treats higher-indexed overlapping
elements as blockers, which meant a container was blocked by its own children.
"""

import pytest

from mobilerun.tools.ui.state import UIState


def _state(elements):
    return UIState(
        elements=elements,
        formatted_text="",
        focused_text="",
        phone_state={},
        screen_width=1000,
        screen_height=2000,
    )


def _row_with_children():
    """A clickable row whose icon + label together cover it completely."""
    return [
        {
            "index": 5,
            "className": "LinearLayout",
            "text": "row",
            "bounds": "0,100,1000,300",
            "children": [
                {
                    "index": 6,
                    "className": "ImageView",
                    "text": "icon",
                    "bounds": "0,100,200,300",
                    "children": [],
                },
                {
                    "index": 7,
                    "className": "TextView",
                    "text": "Settings",
                    "bounds": "200,100,1000,300",
                    "children": [],
                },
            ],
        },
    ]


def test_container_is_tappable_despite_being_filled_by_its_children():
    """#199: a row covered by its own icon and label must still be tappable."""
    state = _state(_row_with_children())
    assert state.get_clear_point(5) == (500, 200)


def test_container_point_falls_inside_its_own_bounds():
    state = _state(_row_with_children())
    x, y = state.get_clear_point(5)
    assert 0 <= x < 1000
    assert 100 <= y < 300


@pytest.mark.parametrize("index,expected", [(6, (100, 200)), (7, (600, 200))])
def test_leaf_children_still_resolve_to_their_own_centre(index, expected):
    state = _state(_row_with_children())
    assert state.get_clear_point(index) == expected


def test_deeply_nested_descendants_are_not_blockers():
    """Grandchildren are descendants too, not overlays."""
    elements = [
        {
            "index": 1,
            "className": "FrameLayout",
            "text": "outer",
            "bounds": "0,0,1000,400",
            "children": [
                {
                    "index": 2,
                    "className": "LinearLayout",
                    "text": "inner",
                    "bounds": "0,0,1000,400",
                    "children": [
                        {
                            "index": 3,
                            "className": "TextView",
                            "text": "label",
                            "bounds": "0,0,1000,400",
                            "children": [],
                        },
                    ],
                },
            ],
        },
    ]
    state = _state(elements)
    assert state.get_clear_point(1) == (500, 200)


def test_later_sibling_overlay_still_blocks():
    """A real overlay drawn after the target must still be avoided."""
    elements = [
        {
            "index": 1,
            "className": "LinearLayout",
            "text": "row",
            "bounds": "0,0,1000,400",
            "children": [],
        },
        {
            "index": 2,
            "className": "FrameLayout",
            "text": "overlay",
            # Covers the left half of the row.
            "bounds": "0,0,500,400",
            "children": [],
        },
    ]
    state = _state(elements)
    x, _ = state.get_clear_point(1)
    assert x >= 500, "tap point should avoid the overlay covering the left half"


def test_fully_covering_overlay_still_raises():
    """A modal covering the target completely is a genuine failure."""
    elements = [
        {
            "index": 1,
            "className": "LinearLayout",
            "text": "row",
            "bounds": "0,0,1000,400",
            "children": [],
        },
        {
            "index": 2,
            "className": "FrameLayout",
            "text": "modal",
            "bounds": "0,0,1000,400",
            "children": [],
        },
    ]
    state = _state(elements)
    with pytest.raises(ValueError, match="fully obscured"):
        state.get_clear_point(1)


def test_overlay_blocks_container_even_when_it_has_children():
    """Descendants are exempt; a genuine later overlay is not."""
    elements = _row_with_children() + [
        {
            "index": 20,
            "className": "FrameLayout",
            "text": "snackbar",
            "bounds": "0,100,1000,300",
            "children": [],
        },
    ]
    state = _state(elements)
    with pytest.raises(ValueError, match="fully obscured"):
        state.get_clear_point(5)


def test_ancestor_never_blocks_its_descendant():
    """An ancestor is painted behind, and is excluded by the index rule."""
    state = _state(_row_with_children())
    # Child 6 sits inside parent 5; the parent must not push the point around.
    assert state.get_clear_point(6) == (100, 200)


def test_missing_index_or_bounds_is_skipped_not_crashed():
    elements = [
        {
            "index": 1,
            "className": "LinearLayout",
            "text": "row",
            "bounds": "0,0,1000,400",
            "children": [],
        },
        # No index.
        {"className": "View", "text": "ghost", "bounds": "0,0,500,400"},
        # No bounds.
        {"index": 3, "className": "View", "text": "boundless", "children": []},
    ]
    state = _state(elements)
    assert state.get_clear_point(1) == (500, 200)
