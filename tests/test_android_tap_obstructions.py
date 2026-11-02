"""Regression coverage through raw Portal data, formatting, and real actions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mobilerun.agent.utils.actions import click, long_press, type_secret, type_text
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.provider import AndroidStateProvider


def node(text, bounds=(0, 0, 1000, 400), children=(), **properties):
    return {
        "text": text,
        "className": "android.widget.Button",
        "boundsInScreen": dict(
            zip(("left", "top", "right", "bottom"), bounds, strict=True)
        ),
        "windowId": 7,
        "drawingOrder": 1,
        "isClickable": True,
        "isVisibleToUser": True,
        "children": list(children),
        **properties,
    }


def context(children, screen=(1000, 2000), **provider_kwargs):
    width, height = screen
    raw = {
        "a11y_tree": node("root", children=children, isClickable=False),
        "phone_state": {},
        "device_context": {"screen_bounds": {"width": width, "height": height}},
    }
    driver = SimpleNamespace(
        get_ui_tree=AsyncMock(return_value=raw),
        tap=AsyncMock(),
        swipe=AsyncMock(),
        input_text=AsyncMock(return_value=True),
    )
    provider = AndroidStateProvider(
        driver, ConciseFilter(), IndexedFormatter(), **provider_kwargs
    )
    ui = asyncio.run(provider.get_state())
    return SimpleNamespace(ui=ui, driver=driver, state_provider=provider)


def element(ctx, text):
    return next(e for e in ctx.ui.elements if e["text"] == text)


@pytest.mark.parametrize("action", ["click", "long_press", "type"])
def test_indexed_actions_avoid_later_touchable_sibling(action):
    ctx = context([node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)])
    index = element(ctx, "target")["index"]
    if action == "click":
        result = asyncio.run(click(index, ctx=ctx))
    elif action == "long_press":
        result = asyncio.run(long_press(index, ctx=ctx))
    else:
        result = asyncio.run(type_text("test", index=index, ctx=ctx))
        ctx.driver.input_text.assert_awaited_once_with("test", False)
    assert result.success
    operation = ctx.driver.swipe if action == "long_press" else ctx.driver.tap
    x, y = operation.await_args.args[:2]
    assert 0 <= x < 1000 and 0 <= y < 400
    assert not 300 <= x < 700


def test_drawing_order_can_block_an_element_with_a_higher_index():
    ctx = context([node("overlay", (300, 0, 700, 400), drawingOrder=8), node("target")])
    target = element(ctx, "target")
    assert target["index"] > element(ctx, "overlay")["index"]
    result = asyncio.run(click(target["index"], ctx=ctx))
    assert result.success
    assert not 300 <= ctx.driver.tap.await_args.args[0] < 700


def test_fully_covered_target_fails_without_dispatching_a_tap():
    ctx = context([node("target"), node("overlay", drawingOrder=2)])
    result = asyncio.run(click(element(ctx, "target")["index"], ctx=ctx))
    assert not result.success
    assert "No clear tap point" in result.summary
    ctx.driver.tap.assert_not_awaited()


def test_child_filled_container_remains_clickable_and_children_keep_indices():
    ctx = context(
        [
            node(
                "row",
                children=[
                    node("icon", (0, 0, 200, 400), isClickable=False),
                    node(
                        "label", (200, 0, 1000, 400), isClickable=False, drawingOrder=2
                    ),
                ],
            )
        ]
    )
    result = asyncio.run(click(element(ctx, "row")["index"], ctx=ctx))
    assert result.success
    ctx.driver.tap.assert_awaited_once_with(500, 200)
    assert [e["index"] for e in ctx.ui.elements] == [1, 2, 3, 4]
    assert all(e["children"] == [] for e in ctx.ui.elements)


def test_independent_child_and_parent_keep_their_own_tap_targets():
    ctx = context([node("row", children=[node("child", (800, 0, 1000, 400))])])
    for text, expected in [("row", (500, 200)), ("child", (900, 200))]:
        result = asyncio.run(click(element(ctx, text)["index"], ctx=ctx))
        assert result.success
        assert ctx.driver.tap.await_args.args == expected


@pytest.mark.parametrize(
    "properties",
    [
        {"drawingOrder": 1},
        {"drawingOrder": 0},
        {"drawingOrder": None},
        {"windowId": 8},
        {"windowId": None},
        {"isClickable": False},
        {"isVisibleToUser": False},
    ],
)
def test_unknown_order_other_windows_and_nonblocking_siblings_do_not_move_taps(
    properties,
):
    overlay = node("overlay", (300, 0, 700, 400), drawingOrder=2)
    overlay.update(properties)
    ctx = context([node("target"), overlay])
    assert ctx.ui.get_element_coords(element(ctx, "target")["index"]) == (500, 200)


def test_drawing_order_is_not_compared_across_different_parents():
    ctx = context(
        [
            node("group1", children=[node("target")], isClickable=False),
            node(
                "group2", children=[node("other", drawingOrder=20)], isClickable=False
            ),
        ]
    )
    assert ctx.ui.get_element_coords(element(ctx, "target")["index"]) == (500, 200)


def test_stealth_jitter_cannot_reenter_known_obstructions(monkeypatch):
    monkeypatch.setattr(
        "mobilerun.tools.ui.stealth_state.random.randint", lambda a, b: 0
    )
    ctx = context(
        [node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)],
        stealth=True,
    )
    result = asyncio.run(click(element(ctx, "target")["index"], ctx=ctx))
    assert result.success
    assert not 300 <= ctx.driver.tap.await_args.args[0] < 700


def test_native_tap_bounds_are_used_when_model_screenshot_is_resized():
    policy = SimpleNamespace(effective_dims=lambda w, h: (500, 1000))
    ctx = context(
        [node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)],
        vision_enabled=True,
        vision_resize_policy=policy,
    )
    target = element(ctx, "target")
    assert target["displayBounds"] == "0,0,500,200"
    assert target["tapBlockers"] == ["300,0,700,400"]
    x, y = ctx.ui.get_element_coords(target["index"])
    assert 0 <= x < 1000 and 0 <= y < 400
    assert not 300 <= x < 700


def test_unobstructed_preferred_center_is_unchanged():
    ctx = context([node("target"), node("overlay", (0, 0, 100, 400), drawingOrder=2)])
    assert ctx.ui.get_element_coords(element(ctx, "target")["index"]) == (500, 200)


def test_narrow_uncovered_strip_is_still_tappable():
    ctx = context([node("target"), node("overlay", (1, 0, 1000, 400), drawingOrder=2)])
    assert ctx.ui.get_element_coords(element(ctx, "target")["index"]) == (0, 200)


def test_relocated_point_stays_on_screen_for_partly_offscreen_target():
    ctx = context(
        [
            node("target", (-600, 0, 1000, 400)),
            node("overlay", (0, 0, 500, 400), drawingOrder=2),
        ]
    )
    assert ctx.ui.get_element_coords(element(ctx, "target")["index"]) == (750, 200)


def test_multiple_siblings_can_collectively_cover_target():
    ctx = context(
        [
            node("target"),
            node("left", (0, 0, 500, 400), drawingOrder=2),
            node("right", (500, 0, 1000, 400), drawingOrder=3),
        ],
        stealth=True,
    )
    result = asyncio.run(click(element(ctx, "target")["index"], ctx=ctx))
    assert not result.success
    ctx.driver.tap.assert_not_awaited()


def test_normalized_mode_shows_normalized_bounds_but_taps_device_pixels():
    ctx = context(
        [node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)],
        use_normalized=True,
    )
    target = element(ctx, "target")
    assert target["bounds"] == "0,0,1000,400"
    assert target["displayBounds"] == "0,0,1000,200"
    assert "(0,0,1000,200)" in ctx.ui.formatted_text
    assert target["tapBlockers"] == ["300,0,700,400"]
    x, y = ctx.ui.get_element_coords(target["index"])
    assert y == 200
    assert not 300 <= x < 700


@pytest.mark.parametrize("stealth", [False, True])
@pytest.mark.parametrize("action", ["click", "long_press", "type", "type_secret"])
def test_normalized_indexed_actions_tap_device_pixels(action, stealth):
    ctx = context(
        [node("send", (930, 2150, 1050, 2270))],
        screen=(1080, 2400),
        use_normalized=True,
        stealth=stealth,
    )
    assert "(861,895,972,945)" in ctx.ui.formatted_text
    index = element(ctx, "send")["index"]
    if action == "click":
        result = asyncio.run(click(index, ctx=ctx))
    elif action == "long_press":
        result = asyncio.run(long_press(index, ctx=ctx))
    elif action == "type":
        result = asyncio.run(type_text("test", index=index, ctx=ctx))
    else:
        ctx.credential_manager = SimpleNamespace(
            resolve_key=AsyncMock(return_value="secret"),
            get_keys=AsyncMock(return_value=[]),
        )
        result = asyncio.run(type_secret("pw", index, ctx=ctx))
    assert result.success
    operation = ctx.driver.swipe if action == "long_press" else ctx.driver.tap
    x, y = operation.await_args.args[:2]
    assert 930 <= x < 1050 and 2150 <= y < 2270
    if not stealth:
        assert (x, y) == (990, 2210)
        if action == "click":
            assert result.summary.endswith("Coordinates: (916, 920)")
        if action == "long_press":
            assert result.summary.endswith("at (916, 920)")


@pytest.mark.parametrize("stealth", [False, True])
def test_normalized_blocker_avoidance_uses_device_pixels(stealth):
    ctx = context(
        [
            node("row", (0, 1800, 1080, 2000)),
            node("undo", (300, 1800, 800, 2000), drawingOrder=2),
            node("wide", (0, 300, 1080, 500)),
            node("wide overlay", (0, 300, 1000, 500), drawingOrder=2),
        ],
        screen=(1080, 2400),
        use_normalized=True,
        stealth=stealth,
    )
    x, y = ctx.ui.get_element_coords(element(ctx, "row")["index"])
    assert 1800 <= y < 2000 and not 300 <= x < 800
    x, y = ctx.ui.get_element_coords(element(ctx, "wide")["index"])
    assert 1000 <= x < 1080 and 300 <= y < 500


def test_normalized_macro_records_device_pixels():
    ctx = context(
        [node("send", (930, 2150, 1050, 2270))],
        screen=(1080, 2400),
        use_normalized=True,
    )
    ctx.macro_recorder = SimpleNamespace(record_action=Mock())
    asyncio.run(click(element(ctx, "send")["index"], ctx=ctx))
    action = ctx.macro_recorder.record_action.call_args.args[0]
    assert action == {"action_type": "tap", "x": 990, "y": 2210}


@pytest.mark.parametrize(
    "properties",
    [{"isEnabled": False}, {"isClickable": False, "isLongClickable": True}],
)
def test_disabled_or_long_clickable_siblings_still_consume_touches(properties):
    ctx = context(
        [
            node("target"),
            node("overlay", (300, 0, 700, 400), drawingOrder=2, **properties),
        ]
    )
    assert (
        not 300 <= ctx.ui.get_element_coords(element(ctx, "target")["index"])[0] < 700
    )


def test_undefined_window_ids_are_not_evidence_of_shared_window():
    ctx = context(
        [node("target", windowId=-1), node("overlay", drawingOrder=2, windowId=-1)]
    )
    assert ctx.ui.get_element_coords(element(ctx, "target")["index"]) == (500, 200)
