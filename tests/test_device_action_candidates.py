import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core.llms.mock import MockLLM
from llama_index.core.workflow import Context, StartEvent
from mobilerun_core_local.driver.base import DeviceDisconnectedError
from selector_helpers import StubSelector

from mobilerun import MobileAgent, MobileConfig
from mobilerun.agent.action_context import ActionCandidateApp, ActionContext
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.agent.selectors import (
    ActionCandidate,
    Decision,
    DecisionSelector,
    FallbackSelector,
    SelectionRequest,
)
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.tools.ui.provider import StateProvider
from mobilerun.tools.ui.state import UIState


def _context(**state_kwargs):
    driver = SimpleNamespace(
        supported={"get_apps", "start_app", "press_button", "swipe", "tap"},
        supported_buttons={"back", "home", "enter", "power"},
        platform="Android",
        get_apps=AsyncMock(return_value=[
            {"package": "com.android.settings", "label": "Settings"},
            {"package_name": "com.google.android.calculator", "label": "Calculator"},
            {"package": "com.android.chrome", "label": "Chrome"},
        ]),
        start_app=AsyncMock(return_value="Started Calculator"),
        press_button=AsyncMock(), swipe=AsyncMock(), tap=AsyncMock(),
    )
    values = dict(
        elements=[{"index": 1, "text": "Search", "bounds": "10,20,50,60"}],
        formatted_text="Chrome", focused_text="Search",
        phone_state={"packageName": "com.android.chrome"},
        screen_width=1080, screen_height=2400,
    )
    values.update(state_kwargs)
    state = UIState(**values)
    provider = StateProvider(driver)
    return ActionContext(driver, state, MobileAgentState(instruction="Calculate"), provider)


def _tools():
    return {
        "click": {}, "long_press": {}, "system_button": {}, "swipe": {},
        "wait": {}, "open_app": {"parameters": {"app_id": {}}}, "complete": {},
    }


def _build(ctx, tools=None, enabled=True):
    return asyncio.run(ctx.state_provider.build_action_candidates(
        ctx, _tools() if tools is None else tools, include_device_actions=enabled,
    ))


def test_builds_navigation_swipes_wait_and_installed_apps_without_execution():
    ctx = _context()
    options = _build(ctx)
    calls = [option.action for option in options]
    assert ToolCall("click", {"index": 1}) in calls
    for button in ("back", "home", "enter"):
        assert ToolCall("system_button", {"button": button}) in calls
    assert ToolCall("system_button", {"button": "power"}) not in calls
    assert len([call for call in calls if call.name == "swipe"]) == 4
    assert ToolCall("wait", {"duration": 1.0}) in calls
    assert ToolCall("open_app", {"app_id": "com.google.android.calculator"}) in calls
    assert ToolCall("open_app", {"app_id": "com.android.settings"}) in calls
    assert not any(call.name == "open_app" and "com.android.chrome" in call.parameters.values() for call in calls)
    ctx.driver.get_apps.assert_awaited_once_with(include_system=True)
    for method in (ctx.driver.start_app, ctx.driver.press_button, ctx.driver.swipe, ctx.driver.tap):
        method.assert_not_called()


def test_llm_path_does_not_enumerate_apps_or_add_device_actions():
    ctx = _context()
    assert [option.action for option in _build(ctx, enabled=False)] == [ToolCall("click", {"index": 1})]
    ctx.driver.get_apps.assert_not_called()


def test_disabled_tools_are_never_offered_or_queried():
    ctx = _context()
    assert [option.action.name for option in _build(ctx, {"click": {}})] == ["click"]
    ctx.driver.get_apps.assert_not_called()
    assert _build(ctx, {}) == ()


def test_app_list_is_cached_per_run_but_current_app_filter_is_refreshed():
    ctx = _context()
    first = _build(ctx)
    assert ctx.action_candidate_apps == (
        ActionCandidateApp(identifier="com.google.android.calculator", label="Calculator"),
        ActionCandidateApp(identifier="com.android.chrome", label="Chrome"),
        ActionCandidateApp(identifier="com.android.settings", label="Settings"),
    )
    assert first == _build(ctx)
    ctx.ui.phone_state["packageName"] = "com.google.android.calculator"
    calls = [option.action for option in _build(ctx)]
    assert ToolCall("open_app", {"app_id": "com.android.chrome"}) in calls
    assert ToolCall("open_app", {"app_id": "com.google.android.calculator"}) not in calls
    ctx.driver.get_apps.assert_awaited_once()


@pytest.mark.parametrize("parameter", ["app_id", "bundle_id", "text"])
def test_app_candidates_match_the_registered_tool_schema(parameter):
    ctx = _context()
    options = _build(ctx, {"open_app": {"parameters": {parameter: {}}}})
    assert options[0].action == ToolCall("open_app", {parameter: "com.google.android.calculator"})
    assert "Calculator" in options[0].description


def test_unsupported_app_listing_and_unknown_schema_are_not_guessed():
    ctx = _context()
    ctx.driver.supported.remove("get_apps")
    assert _build(ctx, {"open_app": {"parameters": {"app_id": {}}}}) == ()
    ctx.driver.get_apps.assert_not_called()
    ctx.driver.supported.add("get_apps")
    assert _build(ctx, {"open_app": {"parameters": {"unknown": {}}}}) == ()
    ctx.driver.get_apps.assert_not_called()


def test_app_list_failure_preserves_navigation_and_can_be_retried():
    ctx = _context()
    ctx.driver.get_apps.side_effect = [RuntimeError("temporarily unavailable"), []]
    assert any(option.action.name == "system_button" for option in _build(ctx))
    assert ctx.action_candidate_apps is None
    _build(ctx)
    assert ctx.action_candidate_apps == ()
    assert ctx.driver.get_apps.await_count == 2


def test_device_disconnect_is_not_swallowed():
    ctx = _context()
    ctx.driver.get_apps.side_effect = DeviceDisconnectedError("Disconnected")
    with pytest.raises(DeviceDisconnectedError):
        _build(ctx)


def test_app_candidates_are_deduplicated_and_skip_invalid_records():
    ctx = _context()
    ctx.driver.get_apps.return_value = [
        None, {}, {"package": ""}, {"package": 12},
        {"package": "hidden", "label": "Hidden", "launchable": False},
        {"bundle_id": "calc", "label": "Calculator"},
        {"app_id": "calc", "label": "Calculator"},
    ]
    apps = [option for option in _build(ctx) if option.action.name == "open_app"]
    assert [option.action for option in apps] == [ToolCall("open_app", {"app_id": "calc"})]


def test_global_actions_work_without_ui_elements_or_an_observation():
    ctx = _context(elements=[])
    assert any(option.action.name == "open_app" for option in _build(ctx))
    ctx.ui = None
    calls = [option.action for option in _build(ctx)]
    assert ToolCall("system_button", {"button": "home"}) in calls
    assert ToolCall("wait", {"duration": 1.0}) in calls
    assert not any(call.name == "swipe" for call in calls)


def test_provider_overrides_are_preserved_without_duplicate_actions():
    ctx = _context()
    custom = ActionCandidate(ToolCall("system_button", {"button": "back"}), "Custom Back")
    ctx.state_provider.get_action_candidates = Mock(return_value=(custom,))
    options = _build(ctx)
    assert options[0] is custom
    assert sum(option.action == custom.action for option in options) == 1


@pytest.mark.parametrize("normalized,screenshot_only,scaled", [
    (False, False, False), (False, False, True),
    (True, False, False), (False, True, True),
])
def test_swipes_use_the_action_coordinate_space(normalized, screenshot_only, scaled):
    ctx = _context(
        use_normalized=normalized, coordinate_contract_active=scaled,
        coordinate_scale_x=2.0 if scaled else 1.0,
        coordinate_scale_y=2.0 if scaled else 1.0,
    )
    ctx.state_provider.requires_coordinate_tools = screenshot_only
    options = _build(ctx, {"swipe": {}})
    assert len(options) == 4
    width, height = (1000, 1000) if normalized else (
        (1080, 2400) if screenshot_only or not scaled else (540, 1200)
    )
    up = options[0].action.parameters
    assert up["coordinate"] == [round((width - 1) * 0.5), round((height - 1) * 0.8)]
    assert up["coordinate2"] == [round((width - 1) * 0.5), round((height - 1) * 0.2)]


def test_swipes_are_not_offered_for_missing_or_inactive_coordinate_contracts():
    ctx = _context()
    ctx.state_provider.requires_active_contract_for_coords = True
    assert _build(ctx, {"swipe": {}}) == ()
    ctx.state_provider.requires_active_contract_for_coords = False
    ctx.ui.screen_width = 0
    assert _build(ctx, {"swipe": {}}) == ()


def test_generated_actions_execute_via_the_existing_registry():
    async def run():
        ctx = _context()
        registry, _ = await build_tool_registry(supported_buttons=ctx.driver.supported_buttons, exact_app_launch=True)
        registry.disable_unsupported(ctx.driver.supported | {"element_index", "convert_point"})
        tools = registry.get_signatures()
        options = await ctx.state_provider.build_action_candidates(ctx, tools, include_device_actions=True)
        for name in ("open_app", "system_button", "swipe", "wait"):
            index = next(i for i, option in enumerate(options) if option.action.name == name)
            model = SimpleNamespace(decide=AsyncMock(return_value=Decision(index)))
            selection = await DecisionSelector(model).select(SelectionRequest("Calculate", tools=tools, options=options))
            action = selection.actions[0]
            with patch("mobilerun.agent.utils.actions.asyncio.sleep", new_callable=AsyncMock):
                result = await registry.execute(action.name, action.parameters, ctx)
            assert result.success, result.summary
        ctx.driver.start_app.assert_awaited_once_with("com.google.android.calculator")
        ctx.driver.press_button.assert_awaited_once_with("back")
        ctx.driver.swipe.assert_awaited_once()
    asyncio.run(run())


def test_fallbacks_propagate_candidate_and_screenshot_requirements():
    selector = DecisionSelector(SimpleNamespace(screenshot_max_side=1568, decide=AsyncMock()))
    chain = FallbackSelector(StubSelector(), [FallbackSelector(selector, [])])
    assert chain.requires_action_candidates
    assert chain.screenshot_max_side == 1568
    assert not FallbackSelector(StubSelector(), []).requires_action_candidates
    assert FallbackSelector(StubSelector(), []).screenshot_max_side is None


def test_native_decision_run_uses_exact_app_launch_and_matching_image_dimensions(monkeypatch):
    from mobilerun.agent.droid import droid_agent

    async def run():
        config = MobileConfig()
        config.agent.fast_agent.vision = True
        config.telemetry.enabled = False
        config.agent.model_screenshot_max_side = 2048
        selector = DecisionSelector(SimpleNamespace(screenshot_max_side=1568, decide=AsyncMock()))
        ctx = _context()
        ctx.driver.get_date = AsyncMock(return_value="2026-10-23")
        monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value={
            "fast_agent": None, "app_opener": MockLLM(),
        }))
        agent = MobileAgent("Calculate", config=config, action_selectors=selector, driver=ctx.driver)
        with patch.object(Context, "write_event_to_stream"):
            await agent.start_handler(Context(agent), StartEvent())
        assert "app_id" in agent.registry.get_signatures()["open_app"]["parameters"]
        assert agent.state_provider.vision_resize_policy.effective_dims(1080, 2400) == (706, 1568)
    asyncio.run(run())
