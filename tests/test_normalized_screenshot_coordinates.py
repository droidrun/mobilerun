import asyncio
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from mobilerun.agent.utils.actions import click_area, click_at, long_press_at, swipe
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.agent.utils.vision_sizing import model_uses_normalized_coordinates
from mobilerun.tools.ui.provider import resize_model_screenshot_with_grid
from mobilerun.tools.ui.screenshot_provider import ScreenshotOnlyStateProvider
from mobilerun.tools.ui.state import UIState


def _png(width: int = 1080, height: int = 2400) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=(16, 20, 24)).save(output, format="PNG")
    return output.getvalue()


class _Driver:
    platform = "Android"
    supported = set()
    supported_buttons = set()

    def __init__(self) -> None:
        self.taps = []
        self.swipes = []

    async def screenshot(self) -> bytes:
        return _png()

    async def tap(self, x, y) -> None:
        self.taps.append((x, y))

    async def swipe(self, x1, y1, x2, y2, duration_ms=1000) -> None:
        self.swipes.append((x1, y1, x2, y2))

    async def get_date(self) -> str:
        return "2026-09-29"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemma4", True),
        ("gemma4:latest", True),
        ("google/gemma-3-27b-it", True),
        ("Gemma-4-E4B-it", True),
        ("gemini-3.8-flash", True),
        ("gemini-3.8-flash-tiered", True),
        ("models/Gemini-3.8-Flash-High", True),
        ("gemini-3.7-flash", False),
        ("gemini-3.1-pro-preview", False),
        ("claude-sonnet-5", False),
        ("gpt-6-astra", False),
        ("", False),
    ],
)
def test_gemma_models_use_normalized_coordinates(model: str, expected: bool) -> None:
    assert model_uses_normalized_coordinates(model) is expected


def _normalized_state():
    driver = _Driver()
    provider = ScreenshotOnlyStateProvider(driver, use_normalized=True)
    return driver, provider, asyncio.run(provider.get_state())


def test_normalized_state_declares_0_1000_and_maps_to_device_pixels() -> None:
    _driver, provider, state = _normalized_state()

    assert state.use_normalized is True
    assert (state.screen_width, state.screen_height) == (1080, 2400)
    assert (state.model_screenshot_width, state.model_screenshot_height) == (
        922,
        2048,
    )
    assert "from 0 to 1000" in state.formatted_text
    assert "(1000,1000) is bottom-right" in state.formatted_text
    assert "pixel" not in state.formatted_text
    assert "grid" not in state.formatted_text
    assert state.convert_point(900, 920) == (972, 2208)
    assert state.convert_point(1000, 1000) == (1079, 2399)
    assert provider.model_screenshot_grid is False


def test_normalized_model_screenshot_has_no_grid() -> None:
    _driver, provider, _state = _normalized_state()

    sent = resize_model_screenshot_with_grid(provider, _png())

    with Image.open(BytesIO(sent)) as image:
        assert image.size == (922, 2048)
        assert image.convert("RGB").getcolors() == [(922 * 2048, (16, 20, 24))]


def test_pixel_mode_keeps_the_grid() -> None:
    provider = ScreenshotOnlyStateProvider(_Driver())
    asyncio.run(provider.get_state())

    sent = resize_model_screenshot_with_grid(provider, _png())

    with Image.open(BytesIO(sent)) as image:
        assert len(image.convert("RGB").getcolors(maxcolors=4096)) > 1


def _ctx():
    driver, provider, state = _normalized_state()
    return SimpleNamespace(driver=driver, ui=state, state_provider=provider), driver


def test_normalized_actions_accept_0_1000_and_tap_device_pixels() -> None:
    ctx, driver = _ctx()

    tap = asyncio.run(click_at(900, 920, ctx=ctx))
    area = asyncio.run(click_area(0, 0, 1000, 1000, ctx=ctx))
    press = asyncio.run(long_press_at(250, 750, ctx=ctx))
    drag = asyncio.run(swipe([500, 800], [500, 200], ctx=ctx))

    assert driver.taps == [(972, 2208), (540, 1200)]
    assert driver.swipes == [(270, 1800, 270, 1800), (540, 1920, 540, 480)]
    assert tap.summary == "Tapped at (900, 920)"
    assert area.summary == "Tapped center of area at (500, 500)"
    assert press.summary == "Long pressed at (250, 750)"
    assert drag.summary == "Swiped from (500, 800) to (500, 200)"


def test_pixel_action_summaries_report_device_pixels() -> None:
    driver = _Driver()
    provider = ScreenshotOnlyStateProvider(driver)
    state = asyncio.run(provider.get_state())
    ctx = SimpleNamespace(driver=driver, ui=state, state_provider=provider)

    result = asyncio.run(click_at(461, 1024, ctx=ctx))

    assert result.summary == f"Tapped at {driver.taps[0]}"
    assert driver.taps[0] != (461, 1024)


def test_normalized_clamp_only_moves_the_1000_endpoint() -> None:
    state = UIState(
        elements=[],
        formatted_text="",
        focused_text="",
        phone_state={},
        screen_width=1080,
        screen_height=2400,
        use_normalized=True,
    )

    assert state.convert_point(1000, 999) == (1079, 2397)
    assert state.convert_point(1200, 1100) == (1296, 2640)


@pytest.mark.parametrize("point", [(1001, 500), (500, -1)])
def test_normalized_actions_reject_points_outside_0_1000(point) -> None:
    ctx, driver = _ctx()

    result = asyncio.run(click_at(*point, ctx=ctx))

    assert result.success is False
    assert "normalized values from 0 to 1000" in result.summary
    assert driver.taps == []


def test_normalized_screenshot_tools_describe_0_1000() -> None:
    registry, _ = asyncio.run(
        build_tool_registry(screenshot_only=True, normalized_coordinates=True)
    )

    for name in ("click_at", "click_area", "long_press_at", "swipe"):
        description = registry.tools[name].description
        assert "0-1000 on both axes" in description
        assert "pixel" not in description
        assert "grid" not in description


def _start_agent(monkeypatch, *, model: str, reasoning: bool = False, config=None):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    from mobilerun.agent.droid import droid_agent as agent_module
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        MobileConfig,
        TelemetryConfig,
    )

    registry_kwargs = {}

    class FakeRegistry:
        def disable_unsupported(self, capabilities) -> None:
            return None

        def disable(self, names) -> None:
            return None

        def register_from_dict(self, tools) -> None:
            return None

    async def fake_build_tool_registry(**kwargs):
        registry_kwargs.update(kwargs)
        return FakeRegistry(), {"click_at"}

    class FakeContext:
        store = None

        def __init__(self):
            self.store = self

        def write_event_to_stream(self, event) -> None:
            return None

        async def set(self, key, value) -> None:
            setattr(self, key, value)

        async def get(self, key, default=None):
            return getattr(self, key, default)

    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(
        agent_module.MobileAgent, "_configure_default_logging", lambda *a, **kw: None
    )
    monkeypatch.setattr(agent_module, "build_tool_registry", fake_build_tool_registry)
    monkeypatch.setattr(agent_module, "capture", lambda *a, **kw: None)

    llm = MockLLM()
    object.__setattr__(llm, "model", model)
    agent_config = config or AgentConfig(vision_only=True, reasoning=reasoning)
    agent = agent_module.MobileAgent(
        "Tap send",
        config=MobileConfig(
            agent=agent_config, telemetry=TelemetryConfig(enabled=False)
        ),
        llms=llm,
        driver=_Driver(),
    )
    asyncio.run(agent.start_handler(FakeContext(), StartEvent()))
    return agent, registry_kwargs


@pytest.mark.parametrize("reasoning", [False, True])
def test_vision_only_gemma_agent_uses_normalized_coordinates(
    monkeypatch, reasoning: bool
) -> None:
    agent, registry_kwargs = _start_agent(
        monkeypatch, model="gemma4:latest", reasoning=reasoning
    )

    assert isinstance(agent.state_provider, ScreenshotOnlyStateProvider)
    assert agent.state_provider.use_normalized is True
    assert registry_kwargs["screenshot_only"] is True
    assert registry_kwargs["normalized_coordinates"] is True


def test_vision_only_non_gemma_agent_keeps_pixel_coordinates(monkeypatch) -> None:
    agent, registry_kwargs = _start_agent(monkeypatch, model="gpt-6-astra")

    assert agent.state_provider.use_normalized is False
    assert registry_kwargs["normalized_coordinates"] is False


def test_normalized_config_applies_to_vision_only(monkeypatch) -> None:
    from mobilerun.config_manager.config_manager import AgentConfig

    agent, _ = _start_agent(
        monkeypatch,
        model="gpt-6-astra",
        config=AgentConfig(vision_only=True, use_normalized_coordinates=True),
    )

    assert agent.state_provider.use_normalized is True


def _a11y_config(*, reasoning: bool = False, vision: bool = True):
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        ExecutorConfig,
        FastAgentConfig,
        ManagerConfig,
    )

    return AgentConfig(
        reasoning=reasoning,
        fast_agent=FastAgentConfig(vision=vision),
        manager=ManagerConfig(vision=vision),
        executor=ExecutorConfig(vision=vision),
    )


@pytest.mark.parametrize("reasoning", [False, True])
def test_a11y_vision_gemma_agent_uses_normalized_coordinates(
    monkeypatch, reasoning: bool
) -> None:
    from mobilerun.tools.ui.provider import AndroidStateProvider

    agent, registry_kwargs = _start_agent(
        monkeypatch,
        model="gemma4:latest",
        config=_a11y_config(reasoning=reasoning),
    )

    assert isinstance(agent.state_provider, AndroidStateProvider)
    assert agent.state_provider.use_normalized is True
    assert registry_kwargs["screenshot_only"] is False
    assert registry_kwargs["normalized_coordinates"] is True


@pytest.mark.parametrize(
    ("model", "vision"), [("gemma4:latest", False), ("gpt-6-astra", True)]
)
def test_a11y_agent_keeps_pixel_coordinates_without_gemma_vision(
    monkeypatch, model: str, vision: bool
) -> None:
    agent, registry_kwargs = _start_agent(
        monkeypatch, model=model, config=_a11y_config(vision=vision)
    )

    assert agent.state_provider.use_normalized is False
    assert registry_kwargs["normalized_coordinates"] is False


def _a11y_context(*, vision: bool, screen=(1080, 2400)):
    from unittest.mock import AsyncMock

    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    def node(text, bounds, children=()):
        return {
            "text": text,
            "className": "android.widget.Button",
            "boundsInScreen": dict(
                zip(("left", "top", "right", "bottom"), bounds, strict=True)
            ),
            "isClickable": True,
            "isVisibleToUser": True,
            "children": list(children),
        }

    raw = {
        "a11y_tree": node(
            "root", (0, 0, 1080, 2400), [node("Send", (930, 2150, 1050, 2270))]
        ),
        "phone_state": {},
        "device_context": {
            "screen_bounds": dict(zip(("width", "height"), screen, strict=True))
        },
    }
    driver = SimpleNamespace(
        get_ui_tree=AsyncMock(return_value=raw), tap=AsyncMock(), swipe=AsyncMock()
    )
    provider = AndroidStateProvider(
        driver,
        ConciseFilter(),
        IndexedFormatter(),
        use_normalized=True,
        vision_enabled=vision,
    )
    ui = asyncio.run(provider.get_state())
    return SimpleNamespace(ui=ui, driver=driver, state_provider=provider)


@pytest.mark.parametrize("vision", [True, False])
def test_normalized_a11y_vision_declares_0_1000_and_enables_click_at(
    vision: bool,
) -> None:
    from mobilerun.agent.droid.droid_agent import _effective_disabled_tools
    from mobilerun.config_manager.config_manager import DEFAULT_DISABLED_TOOLS

    ctx = _a11y_context(vision=vision)
    effective = _effective_disabled_tools(
        list(DEFAULT_DISABLED_TOOLS), ctx.state_provider, vision_enabled=vision
    )

    assert "(861,895,972,945)" in ctx.ui.formatted_text
    assert ("normalized 0-1000 on both axes" in ctx.ui.formatted_text) is vision
    assert ("click_at" in effective) is not vision
    assert ctx.state_provider.resize_model_screenshot is False


def test_normalized_a11y_click_at_rejects_points_outside_0_1000() -> None:
    ctx = _a11y_context(vision=True)

    inside = asyncio.run(click_at(1000, 1000, ctx=ctx))
    outside = asyncio.run(click_at(1001, 5, ctx=ctx))
    drag = asyncio.run(swipe([500, 800], [500, -1], ctx=ctx))

    assert inside.success
    ctx.driver.tap.assert_awaited_once_with(1079, 2399)
    assert outside.success is False
    assert "outside the normalized 0-1000 range" in outside.summary
    assert drag.success is False
    ctx.driver.swipe.assert_not_awaited()


def test_normalized_a11y_refuses_coordinates_without_a_screen_size() -> None:
    ctx = _a11y_context(vision=True, screen=(0, 0))
    assert "(normalized [0-1000])" not in ctx.ui.formatted_text

    tap = asyncio.run(click_at(500, 500, ctx=ctx))
    drag = asyncio.run(swipe([500, 800], [500, 200], ctx=ctx))

    assert tap.success is False and drag.success is False
    assert "unavailable for this step" in tap.summary
    ctx.driver.tap.assert_not_awaited()
    ctx.driver.swipe.assert_not_awaited()


def test_shared_gemma_llm_without_executor_vision_keeps_pixel_coordinates(
    monkeypatch,
) -> None:
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        ExecutorConfig,
        ManagerConfig,
    )

    agent, _ = _start_agent(
        monkeypatch,
        model="gemma4:latest",
        config=AgentConfig(
            reasoning=True,
            manager=ManagerConfig(vision=True),
            executor=ExecutorConfig(vision=False),
        ),
    )

    assert agent.manager_llm is agent.executor_llm
    assert agent.state_provider.use_normalized is False


def test_gemma_executor_with_its_own_vision_uses_normalized_coordinates(
    monkeypatch,
) -> None:
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        ExecutorConfig,
        ManagerConfig,
    )

    agent, _ = _start_agent(
        monkeypatch,
        model="gemma4:latest",
        config=AgentConfig(
            reasoning=True,
            manager=ManagerConfig(vision=False),
            executor=ExecutorConfig(vision=True),
        ),
    )

    assert agent.state_provider.use_normalized is True


@pytest.mark.parametrize("area", [(-100, 400, 1100, 600), (200, 900, 400, 1100)])
def test_normalized_a11y_click_area_rejects_corners_outside_0_1000(area) -> None:
    ctx = _a11y_context(vision=True)

    result = asyncio.run(click_area(*area, ctx=ctx))

    assert result.success is False
    assert "outside the normalized 0-1000 range" in result.summary
    ctx.driver.tap.assert_not_awaited()


def test_normalized_a11y_click_area_taps_the_center() -> None:
    ctx = _a11y_context(vision=True)

    assert asyncio.run(click_area(100, 100, 300, 300, ctx=ctx)).success
    ctx.driver.tap.assert_awaited_once_with(216, 480)
