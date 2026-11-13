import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image
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
from mobilerun.agent.utils import actions
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.tools.ui.provider import StateProvider
from mobilerun.tools.ui.screenshot_provider import ScreenshotOnlyStateProvider
from mobilerun.tools.ui.state import UIState


def _context(**state_kwargs):
    driver = SimpleNamespace(
        supported={"get_apps", "start_app", "press_button", "swipe", "tap"},
        supported_buttons={"back", "home", "enter", "power"},
        platform="Android",
        get_apps=AsyncMock(
            return_value=[
                {"package": "com.android.settings", "label": "Settings"},
                {
                    "package_name": "com.google.android.calculator",
                    "label": "Calculator",
                },
                {"package": "com.android.chrome", "label": "Chrome"},
            ]
        ),
        start_app=AsyncMock(return_value="Started Calculator"),
        press_button=AsyncMock(),
        swipe=AsyncMock(),
        tap=AsyncMock(),
    )
    values = dict(
        elements=[{"index": 1, "text": "Search", "bounds": "10,20,50,60"}],
        formatted_text="Chrome",
        focused_text="Search",
        phone_state={"packageName": "com.android.chrome"},
        screen_width=1080,
        screen_height=2400,
    )
    values.update(state_kwargs)
    state = UIState(**values)
    provider = StateProvider(driver)
    return ActionContext(
        driver, state, MobileAgentState(instruction="Calculate"), provider
    )


def _tools():
    return {
        "click": {},
        "long_press": {},
        "system_button": {},
        "swipe": {},
        "wait": {},
        "open_app": {"parameters": {"app_id": {}}},
        "complete": {},
    }


def _build(ctx, tools=None, enabled=True):
    return asyncio.run(
        ctx.state_provider.build_action_candidates(
            ctx,
            _tools() if tools is None else tools,
            include_device_actions=enabled,
        )
    )


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
    assert not any(
        call.name == "open_app" and "com.android.chrome" in call.parameters.values()
        for call in calls
    )
    ctx.driver.get_apps.assert_awaited_once_with(include_system=True)
    for method in (
        ctx.driver.start_app,
        ctx.driver.press_button,
        ctx.driver.swipe,
        ctx.driver.tap,
    ):
        method.assert_not_called()


def test_llm_path_does_not_enumerate_apps_or_add_device_actions():
    ctx = _context()
    assert [option.action for option in _build(ctx, enabled=False)] == [
        ToolCall("click", {"index": 1})
    ]
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
        ActionCandidateApp(
            identifier="com.google.android.calculator", label="Calculator"
        ),
        ActionCandidateApp(identifier="com.android.chrome", label="Chrome"),
        ActionCandidateApp(identifier="com.android.settings", label="Settings"),
    )
    assert first == _build(ctx)
    ctx.ui.phone_state["packageName"] = "com.google.android.calculator"
    calls = [option.action for option in _build(ctx)]
    assert ToolCall("open_app", {"app_id": "com.android.chrome"}) in calls
    assert (
        ToolCall("open_app", {"app_id": "com.google.android.calculator"}) not in calls
    )
    ctx.driver.get_apps.assert_awaited_once()


@pytest.mark.parametrize("parameter", ["app_id", "bundle_id", "text"])
def test_app_candidates_match_the_registered_tool_schema(parameter):
    ctx = _context()
    options = _build(ctx, {"open_app": {"parameters": {parameter: {}}}})
    assert options[0].action == ToolCall(
        "open_app", {parameter: "com.google.android.calculator"}
    )
    assert "Calculator" in options[0].description


def test_apps_named_in_the_goal_are_listed_first():
    ctx = _context()
    ctx.ui.phone_state["packageName"] = "other"
    ctx.shared_state.instruction = "Open settings and check the battery"
    apps = [
        option.action.parameters["app_id"]
        for option in _build(ctx)
        if option.action.name == "open_app"
    ]
    assert apps == [
        "com.android.settings",
        "com.google.android.calculator",
        "com.android.chrome",
    ]


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
        None,
        {},
        {"package": ""},
        {"package": 12},
        {"package": "hidden", "label": "Hidden", "launchable": False},
        {"bundle_id": "calc", "label": "Calculator"},
        {"app_id": "calc", "label": "Calculator"},
    ]
    apps = [option for option in _build(ctx) if option.action.name == "open_app"]
    assert [option.action for option in apps] == [
        ToolCall("open_app", {"app_id": "calc"})
    ]


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
    custom = ActionCandidate(
        ToolCall("system_button", {"button": "back"}), "Custom Back"
    )
    ctx.state_provider.get_action_candidates = Mock(return_value=(custom,))
    options = _build(ctx)
    assert options[0] is custom
    assert sum(option.action == custom.action for option in options) == 1


@pytest.mark.parametrize(
    "normalized,screenshot_only,scaled",
    [
        (False, False, False),
        (False, False, True),
        (True, False, False),
        (False, True, True),
    ],
)
def test_swipes_use_the_action_coordinate_space(normalized, screenshot_only, scaled):
    ctx = _context(
        use_normalized=normalized,
        coordinate_contract_active=scaled,
        coordinate_scale_x=2.0 if scaled else 1.0,
        coordinate_scale_y=2.0 if scaled else 1.0,
    )
    ctx.state_provider.requires_coordinate_tools = screenshot_only
    options = _build(ctx, {"swipe": {}})
    assert len(options) == 4
    width, height = (
        (1000, 1000)
        if normalized
        else ((1080, 2400) if screenshot_only or not scaled else (540, 1200))
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
        registry, _ = await build_tool_registry(
            supported_buttons=ctx.driver.supported_buttons, exact_app_launch=True
        )
        registry.disable_unsupported(
            ctx.driver.supported | {"element_index", "convert_point"}
        )
        tools = registry.get_signatures()
        options = await ctx.state_provider.build_action_candidates(
            ctx, tools, include_device_actions=True
        )
        for name in ("open_app", "system_button", "swipe", "wait"):
            index = next(
                i for i, option in enumerate(options) if option.action.name == name
            )
            model = SimpleNamespace(decide=AsyncMock(return_value=Decision(index)))
            selection = await DecisionSelector(model).select(
                SelectionRequest("Calculate", tools=tools, options=options)
            )
            action = selection.actions[0]
            with patch(
                "mobilerun.agent.utils.actions.asyncio.sleep", new_callable=AsyncMock
            ):
                result = await registry.execute(action.name, action.parameters, ctx)
            assert result.success, result.summary
        ctx.driver.start_app.assert_awaited_once_with("com.google.android.calculator")
        ctx.driver.press_button.assert_awaited_once_with("back")
        ctx.driver.swipe.assert_awaited_once()

    asyncio.run(run())


def test_fallbacks_propagate_candidate_and_screenshot_requirements():
    selector = DecisionSelector(
        SimpleNamespace(screenshot_max_side=1568, decide=AsyncMock())
    )
    chain = FallbackSelector(StubSelector(), [FallbackSelector(selector, [])])
    assert chain.requires_action_candidates
    assert chain.screenshot_max_side == 1568
    assert not FallbackSelector(StubSelector(), []).requires_action_candidates
    assert FallbackSelector(StubSelector(), []).screenshot_max_side is None


def test_native_decision_run_uses_exact_app_launch_and_matching_image_dimensions(
    monkeypatch,
):
    from mobilerun.agent.droid import droid_agent

    async def run():
        config = MobileConfig()
        config.agent.fast_agent.vision = True
        config.telemetry.enabled = False
        config.agent.model_screenshot_max_side = 2048
        selector = DecisionSelector(
            SimpleNamespace(screenshot_max_side=1568, decide=AsyncMock())
        )
        ctx = _context()
        ctx.driver.get_date = AsyncMock(return_value="2026-10-23")
        monkeypatch.setattr(
            droid_agent,
            "load_agent_llms",
            Mock(
                return_value={
                    "fast_agent": None,
                    "app_opener": MockLLM(),
                }
            ),
        )
        agent = MobileAgent(
            "Calculate", config=config, action_selectors=selector, driver=ctx.driver
        )
        with patch.object(Context, "write_event_to_stream"):
            await agent.start_handler(Context(agent), StartEvent())
        assert "app_id" in agent.registry.get_signatures()["open_app"]["parameters"]
        assert agent.state_provider.vision_resize_policy.effective_dims(1080, 2400) == (
            706,
            1568,
        )

    asyncio.run(run())


def test_elements_with_off_screen_centres_are_not_clickable_candidates():
    ctx = _context(
        elements=[
            {"index": 1, "text": "Visible", "bounds": "0,2300,1080,2460"},
            {"index": 2, "text": "Scrolled away", "bounds": "0,-300,1080,100"},
        ]
    )

    options = ctx.state_provider.get_action_candidates(ctx.ui)

    assert [option.description for option in options] == [
        "Click Visible (bottom of screen)"
    ]


def _node(index, text, bounds, clickable=False, resource_id="", class_name="X"):
    return {
        "index": index,
        "text": text,
        "resourceId": resource_id,
        "className": class_name,
        "bounds": bounds,
        "isClickable": clickable,
    }


def _labels(elements):
    ctx = _context(elements=elements)
    return [
        option.description
        for option in ctx.state_provider.get_action_candidates(ctx.ui)
    ]


def test_text_less_rows_are_named_by_the_text_inside_them():
    search_id = "com.android.settings:id/search_action_bar"
    row = "android.widget.LinearLayout"
    assert _labels(
        [
            _node(1, row, "0,942,1080,1129", True, class_name="LinearLayout"),
            _node(2, "Adaptive Battery", "84,974,400,1038"),
            _node(3, "Adaptive Battery", "84,1038,400,1097"),
            _node(4, "Battery saver", "84,1060,400,1100"),
            _node(5, search_id, "0,100,1080,256", True, search_id),
            _node(6, "Search Settings", "120,140,600,220"),
            _node(7, "android:id/icon", "0,1500,100,1600", True, "android:id/icon"),
            _node(8, "OK", "500,1500,700,1600", clickable=True),
        ]
    ) == [
        "Click Adaptive Battery, Battery saver",
        "Click Search Settings (top of screen)",
        "Click icon",
        "Click OK",
    ]


def test_rows_do_not_borrow_text_owned_by_a_smaller_clickable():
    scrim = "com.google.android.material:id/touch_outside"
    assert _labels(
        [
            _node(1, scrim, "0,0,1080,2400", True, scrim),
            _node(2, "Delete photo?", "100,1500,980,1600"),
            _node(3, "Delete", "100,1700,500,1800", clickable=True),
            _node(
                4,
                "android.widget.LinearLayout",
                "0,300,1080,500",
                True,
                "",
                "LinearLayout",
            ),
            _node(5, "T-Mobile", "84,320,600,380"),
            _node(6, "Settings", "900,320,1000,420", clickable=True),
        ]
    ) == [
        "Click touch_outside",
        "Click Delete",
        "Click T-Mobile",
        "Click Settings",
    ]


def test_screen_roots_do_not_borrow_page_text():
    root = "android.widget.FrameLayout"
    email = "com.app:id/email"
    assert _labels(
        [
            _node(1, root, "0,0,1080,2400", True, class_name="FrameLayout"),
            _node(2, "Welcome back", "84,300,900,400"),
            _node(3, "Email", "84,600,400,660"),
            _node(4, email, "84,680,996,800", True, email, "EditText"),
        ]
    ) == ["Click unnamed FrameLayout", 'Click text field "email"']


def test_labels_name_the_widget_kind_and_its_state():
    def node(index, text, bounds, clickable, class_name, checked="", rid=""):
        return {
            **_node(index, text, bounds, clickable, rid, class_name),
            "checkedState": checked,
        }

    switch_id = "android:id/switch_widget"
    ctx = _context(
        elements=[
            # A row whose title and switch are separate, non-clickable elements.
            node(
                1, "android.widget.FrameLayout", "0,384,1080,657", True, "FrameLayout"
            ),
            node(2, "Use Battery Saver", "126,492,454,549", False, "TextView"),
            node(
                3,
                switch_id,
                "848,457,985,583",
                False,
                "Switch",
                "isChecked=True",
                switch_id,
            ),
            node(4, "M", "200,1000,300,1100", True, "CheckBox", "isChecked=False"),
            node(5, "Vibrate", "0,1200,1080,1300", True, "Switch", "isChecked=False"),
            node(
                6,
                "Search settings",
                "100,700,980,800",
                True,
                "EditText",
                rid="app:id/q",
            ),
            node(7, "Save", "800,1950,1000,2090", True, "Button"),
        ],
        phone_state={
            "packageName": "com.android.settings",
            "isEditable": True,
            "focusedElement": {"resourceId": "app:id/q"},
        },
    )
    assert [
        option.description
        for option in ctx.state_provider.get_action_candidates(ctx.ui)
    ] == [
        'Click switch "Use Battery Saver" (on)',
        'Click checkbox "M" (unchecked)',
        'Click switch "Vibrate" (off)',
        'Click text field "Search settings" (focused)',
        'Click button "Save"',
    ]


@pytest.mark.parametrize(
    ("focused_text", "expected"), [("", []), ("2", ["2"])], ids=["ambiguous", "by-text"]
)
def test_focus_is_marked_only_on_the_one_focused_field(focused_text, expected):
    fields = [
        _node(
            i,
            str(i),
            f"0,{300 * i},1080,{300 * i + 200}",
            True,
            "shop:id/qty",
            "EditText",
        )
        for i in (1, 2, 3)
    ]
    ctx = _context(
        elements=fields,
        phone_state={
            "isEditable": True,
            "focusedElement": {"resourceId": "shop:id/qty", "text": focused_text},
        },
    )
    focused = [
        option.description.split('"')[1]
        for option in ctx.state_provider.get_action_candidates(ctx.ui)
        if "focused" in option.description
    ]
    assert focused == expected


def test_single_choice_rows_are_not_called_checkboxes():
    assert _labels(
        [
            {
                **_node(
                    1, "1 minute", "0,600,1080,700", True, class_name="CheckedTextView"
                ),
                "checkedState": "isChecked=False",
            },
        ]
    ) == ['Click option "1 minute" (unchecked)']


def test_ios_switch_state_comes_from_its_value():
    assert _labels(
        [
            {
                "index": 1,
                "type": "Switch",
                "className": "Switch",
                "text": "Wi-Fi",
                "value": "1",
                "bounds": "0,600,1080,700",
            },
            {
                "index": 2,
                "type": "TextField",
                "className": "TextField",
                "text": "Search",
                "bounds": "0,800,1080,900",
            },
        ]
    ) == ['Click switch "Wi-Fi" (on)', 'Click text field "Search"']


def test_ios_switch_knob_is_not_offered_twice():
    def switch(index, text, bounds, value="0"):
        return {
            "index": index,
            "type": "Switch",
            "className": "Switch",
            "text": text,
            "value": value,
            "bounds": bounds,
        }

    assert _labels(
        [
            switch(3, "Bold Text", "0,620,402,670"),
            switch(33, "", "305,628,368,662"),
            switch(34, "", "305,828,368,862"),  # unlabelled and alone: still offered
        ]
    ) == ['Click switch "Bold Text" (off)', 'Click switch "unnamed" (off)']


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Gym", 'Click text field "Title" (contains "Gym")'),
        ("Title", 'Click text field "Title"'),
        ("", 'Click text field "Title"'),
    ],
)
def test_ios_text_field_shows_what_it_contains(value, expected):
    assert _labels(
        [
            {
                "index": 1,
                "type": "TextField",
                "className": "TextField",
                "text": "Title",
                "value": value,
                "bounds": "0,300,402,350",
            }
        ]
    ) == [expected]


def test_borrowed_text_keeps_the_resource_name():
    cart = "com.shop:id/menu_cart"
    assert _labels(
        [
            _node(1, cart, "900,100,1040,240", True, cart, "FrameLayout"),
            _node(2, "3", "980,110,1030,150"),
        ]
    ) == ["Click menu_cart: 3 (top of screen)"]


def test_llm_in_the_chain_keeps_name_based_app_launching(monkeypatch):
    from mobilerun.agent.droid import droid_agent
    from mobilerun.agent.selectors import DelegatingSelector, LlamaIndexLLMSelector

    async def run():
        config = MobileConfig()
        config.telemetry.enabled = False
        selector = DelegatingSelector(
            DecisionSelector(SimpleNamespace(decide=AsyncMock())),
            LlamaIndexLLMSelector(MockLLM()),
        )
        ctx = _context()
        ctx.driver.get_date = AsyncMock(return_value="2026-10-23")
        monkeypatch.setattr(
            droid_agent,
            "load_agent_llms",
            Mock(return_value={"fast_agent": None, "app_opener": MockLLM()}),
        )
        agent = MobileAgent(
            "Calculate", config=config, action_selectors=selector, driver=ctx.driver
        )
        with patch.object(Context, "write_event_to_stream"):
            await agent.start_handler(Context(agent), StartEvent())
        assert "text" in agent.registry.get_signatures()["open_app"]["parameters"]
        options = await agent.state_provider.build_action_candidates(
            ctx, agent.registry.get_signatures(), include_device_actions=True
        )
        assert ToolCall("open_app", {"text": "com.android.settings"}) in [
            option.action for option in options
        ]

        def no_llm(**kwargs):
            raise AssertionError("A chosen app id must not ask the app opener LLM")

        monkeypatch.setattr(actions, "AppStarter", no_llm)
        monkeypatch.setattr(actions.asyncio, "sleep", AsyncMock())
        result = await actions.open_app("com.android.settings", ctx=ctx)
        assert result.success
        ctx.driver.start_app.assert_awaited_once_with("com.android.settings")

    asyncio.run(run())


def _png(size=(1080, 2400)):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def _screenshot_driver():
    driver = SimpleNamespace(
        supported=set(), supported_buttons=set(), screenshot=AsyncMock()
    )
    driver.screenshot.return_value = _png()
    return driver


@pytest.mark.parametrize("use_normalized", [False, True])
def test_screenshot_only_offers_one_located_tap(use_normalized):
    async def run():
        driver = _screenshot_driver()
        provider = ScreenshotOnlyStateProvider(driver, use_normalized=use_normalized)
        state = await provider.get_state()
        ctx = ActionContext(driver, state, MobileAgentState(), provider)

        assert not await provider.build_action_candidates(ctx, {"click_at": {}})
        options = await provider.build_action_candidates(
            ctx, {"click_at": {}}, include_device_actions=True, locate_taps=True
        )

        space = (
            (1000, 1000)
            if use_normalized
            else (state.screen_width, state.screen_height)
        )
        assert [(o.action, o.point_space) for o in options] == [
            (ToolCall("click_at"), space)
        ]

    asyncio.run(run())


def test_accessibility_providers_offer_no_located_tap():
    async def run():
        ctx = _context()
        options = await ctx.state_provider.build_action_candidates(
            ctx,
            {"click": {}, "click_at": {}},
            include_device_actions=True,
            locate_taps=True,
        )
        assert options
        assert not any(option.point_space for option in options)

    asyncio.run(run())


@pytest.mark.parametrize("injected", [False, True])
def test_vision_only_decision_agent_offers_located_taps(monkeypatch, injected):
    from mobilerun.agent.droid import droid_agent

    async def run():
        monkeypatch.setattr(
            droid_agent, "load_agent_llms", Mock(return_value={"app_opener": MockLLM()})
        )
        ctx = _context()
        ctx.driver.get_date = AsyncMock(return_value="2026-10-23")
        ctx.driver.screenshot = AsyncMock(return_value=_png())
        config = MobileConfig()
        config.agent.vision_only = not injected
        config.telemetry.enabled = False
        agent = MobileAgent(
            "Calculate",
            config=config,
            action_selectors=DecisionSelector(
                SimpleNamespace(decide=AsyncMock(), locates_points=True)
            ),
            driver=ctx.driver,
            state_provider=(
                ScreenshotOnlyStateProvider(ctx.driver) if injected else None
            ),
        )
        with patch.object(Context, "write_event_to_stream"):
            await agent.start_handler(Context(agent), StartEvent())

        provider = agent.state_provider
        assert isinstance(provider, ScreenshotOnlyStateProvider)
        assert agent.action_selector.locates_points
        state = await provider.get_state()
        options = await provider.build_action_candidates(
            ActionContext(ctx.driver, state, MobileAgentState(), provider),
            agent.registry.get_signatures(),
            include_device_actions=True,
            locate_taps=True,
        )
        assert [o.action.name for o in options if o.point_space] == ["click_at"]

    asyncio.run(run())


def test_composite_without_app_opener_keeps_exact_app_launching():
    from mobilerun.agent.selectors import DelegatingSelector, LlamaIndexLLMSelector

    async def run():
        selector = DelegatingSelector(
            DecisionSelector(SimpleNamespace(decide=AsyncMock())),
            LlamaIndexLLMSelector(MockLLM()),
        )
        ctx = _context()
        ctx.driver.get_date = AsyncMock(return_value="2026-10-23")
        agent = MobileAgent("Calculate", action_selectors=selector, driver=ctx.driver)
        agent.config.telemetry.enabled = False
        assert agent.app_opener_llm is None
        with patch.object(Context, "write_event_to_stream"):
            await agent.start_handler(Context(agent), StartEvent())
        assert "app_id" in agent.registry.get_signatures()["open_app"]["parameters"]

    asyncio.run(run())


def _formatted(*nodes):
    from mobilerun.tools.formatters.indexed_formatter import IndexedFormatter

    formatter = IndexedFormatter()
    return [formatter._format_node(node, index) for index, node in enumerate(nodes, 1)]


def _raw(class_name, text, top, **extra):
    return {
        "className": class_name,
        "text": text,
        "resourceId": "",
        "boundsInScreen": {"left": 0, "top": top, "right": 1080, "bottom": top + 100},
        "isClickable": True,
        "isEnabled": True,
        "isVisibleToUser": True,
        **extra,
    }


def test_labels_use_node_facts_the_screen_text_drops():
    elements = _formatted(
        # mobilerun-ios on a real iPhone: typed text, field name as description.
        _raw("TextField", "Gym", 600, contentDescription="Title", isEditable=True),
        # Android field showing its hint is empty.
        _raw(
            "android.widget.EditText",
            "Search",
            800,
            hint="Search",
            isShowingHintText=True,
        ),
        _raw("android.widget.EditText", "••••", 1000, hint="Password", isPassword=True),
        _raw(
            "android.widget.CheckBox",
            "T",
            1200,
            contentDescription="Tuesday",
            isCheckable=True,
        ),
        _raw("android.widget.FrameLayout", "Timers", 2250, isSelected=True),
    )
    assert "details" not in _formatted(_raw("android.widget.Button", "OK", 600))[0]
    assert _labels(elements) == [
        'Click text field "Title" (contains "Gym")',
        'Click text field "Search"',
        'Click text field "Password" (password)',
        'Click checkbox "T, Tuesday" (unchecked)',
        "Click Timers (selected, bottom of screen)",
    ]


def test_ios_value_keeps_commas_and_skips_the_placeholder():
    from mobilerun.tools.ui.ios_provider import _parse_a11y_tree

    elements = _parse_a11y_tree(
        "Application, 0x1, {{0.0, 0.0}, {402.0, 874.0}}, label: 'Reminders'\n"
        "  TextField, 0x2, {{16.0, 300.0}, {343.0, 34.0}}, label: 'Address', "
        "value: Main St, Apt 4\n"
        "  TextField, 0x3, {{16.0, 400.0}, {343.0, 34.0}}, label: 'Title', "
        "placeholderValue: 'Enter a title', value: Enter a title\n"
    )
    fields = [e for e in elements if e["type"] == "TextField"]
    assert fields[0]["value"] == "Main St, Apt 4"
    assert _labels(fields) == [
        'Click text field "Address" (contains "Main St, Apt 4")',
        'Click text field "Title"',
    ]


def _ios_local(class_name, text, top, height=60, left=0, right=402, **extra):
    """A node as mobilerun-ios --local sends it."""
    return {
        "className": f"XCUIElementType{class_name}",
        "text": text,
        "resourceId": "",
        "boundsInScreen": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": top + height,
        },
        "isClickable": True,
        "isEnabled": True,
        "isVisibleToUser": True,
        **extra,
    }


def test_real_iphone_switch_is_one_named_option_on_the_knob():
    elements = _formatted(
        _ios_local(
            "Switch", "0", 600, contentDescription="Bold Text", isCheckable=True
        ),
        _ios_local("Switch", "0", 606, 48, 305, 368, isCheckable=True),
    )
    ctx = _context(elements=elements)
    options = ctx.state_provider.get_action_candidates(ctx.ui)
    assert [(o.description, o.action.parameters["index"]) for o in options] == [
        ('Click switch "Bold Text" (off)', 2)
    ]


def test_real_iphone_fields_show_name_and_contents():
    assert _labels(
        _formatted(
            _ios_local(
                "TextField", "Gym", 600, contentDescription="Title", isClickable=False
            ),
            _ios_local(
                "TextField",
                "Title",
                700,
                contentDescription="Title",
                hint="Enter a title",
                isClickable=False,
            ),
            _ios_local(
                "SearchField",
                "Bold Text",
                800,
                contentDescription="Search",
                isClickable=False,
            ),
            _ios_local("Cell", "XCUIElementTypeCell", 900),
        )
    ) == [
        'Click text field "Title" (contains "Gym")',
        'Click text field "Title"',
        'Click search field "Search" (contains "Bold Text")',
        "Click unnamed Cell",
    ]


def test_cell_and_its_button_with_one_label_are_one_option():
    assert _labels(
        _formatted(
            _ios_local("Other", "Notifications", 600),
            _ios_local("Other", "Notifications", 600),
        )
    ) == ["Click Notifications"]


def test_samsung_switch_widget_takes_its_row_name_and_tells_the_row_its_state():
    switch_id = "android:id/switch_widget"
    elements = _formatted(
        {
            **_raw("android.widget.LinearLayout", "", 600),
            "boundsInScreen": {"left": 0, "top": 600, "right": 1080, "bottom": 800},
        },
        _raw("android.widget.TextView", "Extra brightness", 620, isClickable=False),
        _raw(
            "android.widget.TextView",
            "Hidden summary",
            640,
            isClickable=False,
            isVisibleToUser=False,
        ),
        {
            **_raw(
                "android.widget.Switch", "", 610, resourceId=switch_id, isCheckable=True
            ),
            "boundsInScreen": {"left": 900, "top": 610, "right": 1000, "bottom": 690},
        },
    )
    # The row may open a page of its own, so both stay; only the switch has a state.
    assert _labels(elements) == [
        "Click Extra brightness",
        'Click switch "Extra brightness" (off)',
    ]


def test_masked_secret_pieces_do_not_reach_labels_through_descriptions():
    boxes = [
        _raw(
            "android.widget.EditText",
            "[secret:otp]",
            600 + 120 * i,
            contentDescription=d,
            isEditable=True,
        )
        for i, d in enumerate("482913")
    ]
    labels = _labels(_formatted(*boxes))
    assert len(labels) == 6 and not any(d in "".join(labels) for d in "482913")


def test_platform_state_words_do_not_repeat_the_state():
    assert _labels(
        _formatted(
            _raw(
                "android.widget.CheckBox",
                "Sunday",
                600,
                isCheckable=True,
                stateDescription="Not selected",
            ),
            _raw("android.widget.SeekBar", "Brightness", 800, stateDescription="83%"),
        )
    ) == ['Click checkbox "Sunday" (unchecked)', 'Click slider "Brightness" (83%)']


def test_icon_buttons_inside_a_named_button_stay_offered():
    assert _labels(
        _formatted(
            _raw("android.widget.Button", "Profile", 600),
            {
                **_raw("android.widget.ImageButton", "", 610),
                "boundsInScreen": {
                    "left": 900,
                    "top": 610,
                    "right": 1000,
                    "bottom": 690,
                },
            },
        )
    ) == ['Click button "Profile"', 'Click button "unnamed"']


def test_wrapper_without_options_does_not_hide_its_clickable_twin():
    assert _labels(
        _formatted(
            _raw("android.view.View", "Sign in", 600, isClickable=False),
            _raw("android.view.View", "Sign in", 600),
        )
    ) == ["Click Sign in"]


def test_one_option_per_tap_target_keeps_the_named_one():
    assert _labels(
        _formatted(
            _ios_local(
                "Cell", "XCUIElementTypeCell", 600, resourceId="ENHANCE_TEXT_LEGIBILITY"
            ),
            _ios_local("Button", "Bold Text", 600),
        )
    ) == ['Click button "Bold Text"']


@pytest.mark.parametrize(
    "knob", [{"isEnabled": False}, {"isClickable": False}, {"isVisibleToUser": False}]
)
def test_row_switch_stays_when_its_knob_cannot_be_offered(knob):
    assert _labels(
        _formatted(
            _ios_local(
                "Switch", "0", 600, contentDescription="Bold Text", isCheckable=True
            ),
            _ios_local("Switch", "0", 606, 48, 305, 368, isCheckable=True, **knob),
        )
    ) == ['Click switch "Bold Text" (off)']


def test_knob_sticking_out_of_its_row_still_takes_the_row_name():
    assert _labels(
        _formatted(
            _ios_local(
                "Switch",
                "1",
                646,
                29,
                40,
                400,
                contentDescription="Bold Text",
                isCheckable=True,
                isChecked=True,
            ),
            _ios_local(
                "Switch", "1", 646, 29, 339, 402, isCheckable=True, isChecked=True
            ),
        )
    ) == ['Click switch "Bold Text" (on)']


def test_android_options_named_zero_and_one_keep_their_names():
    assert _labels(
        _formatted(
            *(
                _raw(
                    "android.widget.RadioButton",
                    str(n),
                    600 + 120 * n,
                    isCheckable=True,
                )
                for n in range(3)
            )
        )
    ) == [f'Click radio button "{n}" (unchecked)' for n in range(3)]


def test_text_ending_in_a_dot_is_kept_without_a_class_name():
    assert _labels([{"index": 1, "text": "Done.", "bounds": "0,600,1080,700"}]) == [
        "Click Done."
    ]


def test_masking_a_secret_piece_also_masks_its_description():
    from mobilerun.tools.driver.secret_safe import SecretRedactor

    redactor = SecretRedactor()
    redactor.register("otp", "482913")
    tree = {
        "children": [
            {
                "className": "android.widget.EditText",
                "text": d,
                "contentDescription": d,
                "isEditable": True,
                "children": [],
            }
            for d in "482913"
        ]
    }
    masked = redactor.redact_ui_tree(tree)
    assert all(
        child["text"] == child["contentDescription"] == "[secret:otp]"
        for child in masked["children"]
    )


def _tree(*nodes):
    """Format a nested raw tree the way the provider does (keeps nesting)."""
    from mobilerun.tools.formatters.indexed_formatter import IndexedFormatter

    formatter = IndexedFormatter()
    counter = [1]
    return [e for node in nodes for e in formatter._flatten_with_index(node, counter)]


def test_text_is_only_borrowed_from_nested_elements():
    row = {
        **_raw("android.widget.LinearLayout", "", 1884),
        "boundsInScreen": {"left": 0, "top": 1884, "right": 1080, "bottom": 2205},
        "children": [_raw("android.widget.TextView", "09:30", 1900, isClickable=False)],
    }
    # The tab bar is drawn over the row's lower part but is not inside it.
    tab = {
        **_raw("android.widget.LinearLayout", "", 2011, isClickable=False),
        "boundsInScreen": {"left": 235, "top": 2011, "right": 370, "bottom": 2147},
        "children": [_raw("android.widget.TextView", "Alarm", 2020, isClickable=False)],
    }
    tab["children"][0]["boundsInScreen"] = {
        "left": 240,
        "top": 2020,
        "right": 360,
        "bottom": 2100,
    }
    assert _labels(_tree(row, tab)) == ["Click 09:30"]


def test_ios_cell_wrapping_its_own_controls_is_not_an_option():
    cell = {
        **_ios_local(
            "Cell",
            "XCUIElementTypeCell",
            146,
            29,
            20,
            420,
            resourceId="ENHANCE_TEXT_LEGIBILITY",
        ),
        "children": [
            _ios_local("Button", "Bold Text", 146, 29, 40, 300),
            {
                **_ios_local(
                    "Switch",
                    "0",
                    146,
                    29,
                    40,
                    400,
                    contentDescription="Bold Text",
                    isCheckable=True,
                ),
                "children": [
                    _ios_local("Switch", "0", 146, 29, 339, 402, isCheckable=True)
                ],
            },
        ],
    }
    for node in (cell, *cell["children"], cell["children"][1]["children"][0]):
        node["boundsInScreen"]["top"] += 500
        node["boundsInScreen"]["bottom"] += 500
    # The title button lies under the full-row switch drawn after it.
    assert _labels(_tree(cell)) == ['Click switch "Bold Text" (off)']


def test_ios_cell_holding_a_switch_is_not_named_by_its_value():
    cell = {
        **_ios_local("Cell", "1", 600, 60, 20, 420),
        "children": [
            _ios_local("StaticText", "Snooze", 610, 40, 40, 200, isClickable=False),
            _ios_local(
                "Switch",
                "1",
                610,
                40,
                339,
                402,
                contentDescription="Snooze",
                isCheckable=True,
                isChecked=True,
            ),
        ],
    }
    assert _labels(_tree(cell)) == ["Click Snooze", 'Click switch "Snooze" (on)']


def test_field_keeps_its_label_over_a_cell_with_the_same_bounds():
    cell = {
        **_ios_local("Cell", "XCUIElementTypeCell", 600),
        "children": [
            _ios_local(
                "TextField", "Gym", 600, contentDescription="Title", isClickable=False
            )
        ],
    }
    assert _labels(_tree(cell)) == ['Click text field "Title" (contains "Gym")']


def test_legacy_ios_split_secret_in_values_is_masked():
    from mobilerun.tools.driver.secret_safe import SecretRedactor

    redactor = SecretRedactor()
    redactor.register("otp", "482915")
    elements = [
        {
            "index": i,
            "type": "TextField",
            "className": "TextField",
            "text": "",
            "value": d,
            "bounds": f"{20 + 60 * i},600,{70 + 60 * i},660",
            "children": [],
        }
        for i, d in enumerate("482915", 1)
    ]
    state = UIState(
        elements=elements,
        formatted_text="",
        focused_text="",
        phone_state={},
        screen_width=402,
        screen_height=874,
    )
    masked = redactor.redact_ui_state(state)
    labels = [
        o.description
        for o in StateProvider.get_action_candidates(
            StateProvider.__new__(StateProvider), masked
        )
    ]
    assert labels and not any(d in "".join(labels) for d in "482915")


def test_unlabelled_ios_switch_takes_the_label_beside_it():
    row = {
        **_ios_local(
            "Other", "XCUIElementTypeOther", 600, 40, 0, 402, isClickable=False
        ),
        "children": [
            _ios_local("StaticText", "Dark mode", 605, 30, 16, 200),
            _ios_local("Switch", "0", 604, 31, 330, 381, isCheckable=True),
        ],
    }
    assert _labels(_tree(row)) == ["Click Dark mode", 'Click switch "Dark mode" (off)']


def test_knob_of_an_off_screen_ios_row_is_not_offered():
    row = {
        **_ios_local(
            "Switch",
            "0",
            0,
            0,
            0,
            0,
            contentDescription="Reduce Bright Effects",
            isCheckable=True,
        ),
        "children": [_ios_local("Switch", "0", 616, 28, 0, 63, isCheckable=True)],
    }
    row["boundsInScreen"] = {"left": 0, "top": 0, "right": 0, "bottom": 0}
    assert _labels(_tree(row)) == []


def test_same_bounds_twin_keeps_its_long_press():
    row = {
        **_raw("android.widget.LinearLayout", "", 600, contentDescription="Alice"),
        "children": [
            _raw("android.view.View", "", 600, isClickable=False, isLongClickable=True)
        ],
    }
    ctx = _context(elements=_tree(row))
    options = ctx.state_provider.get_action_candidates(ctx.ui)
    assert [(o.action.name, o.action.parameters["index"]) for o in options] == [
        ("click", 1),
        ("long_press", 2),
    ]


def test_ios_tab_value_does_not_leak_into_its_name():
    assert _labels(
        _formatted(
            _ios_local(
                "Button", "1", 2250, contentDescription="Alarms", isSelected=True
            ),
            _ios_local("Button", "1", 600, contentDescription="1"),  # a key "1"
        )
    ) == ['Click button "Alarms" (selected, bottom of screen)', 'Click button "1"']


def test_ios_row_title_button_is_not_offered_beside_the_knob():
    row = {
        **_ios_local(
            "Switch",
            "0",
            646,
            29,
            40,
            400,
            contentDescription="Bold Text",
            isCheckable=True,
        ),
        "children": [
            _ios_local("Button", "Bold Text", 649, 22, 40, 111),
            _ios_local("Switch", "0", 646, 29, 339, 402, isCheckable=True),
        ],
    }
    assert _labels(_tree(row)) == ['Click switch "Bold Text" (off)']


def test_legacy_ios_cells_labelled_zero_and_one_keep_their_names():
    from mobilerun.tools.ui.ios_provider import _parse_a11y_tree

    elements = _parse_a11y_tree(
        "Application, 0x1, {{0.0, 0.0}, {402.0, 874.0}}, label: 'Picker'\n"
        + "".join(
            f"  Cell, 0x{n + 2}, {{{{0.0, {300 + 60 * n}.0}}, {{402.0, 50.0}}}}, label: '{n}'\n"
            for n in range(3)
        )
    )
    cells = [e for e in elements if e["type"] == "Cell"]
    assert _labels(cells) == ["Click 0", "Click 1", "Click 2"]


def _ios_fixture(name, change=None):
    """Option labels for a captured mobilerun-ios --local screen, optionally
    after ``change`` edits the raw payload."""
    import json
    from pathlib import Path

    from mobilerun.tools.filters import DetailedFilter
    from mobilerun.tools.formatters.indexed_formatter import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    path = Path(__file__).parent / "fixtures" / "ios_local" / f"{name}.json"
    data = json.loads(path.read_text())["result"]
    if change is not None:
        change(data)
    provider = AndroidStateProvider(
        None, tree_filter=DetailedFilter(), tree_formatter=IndexedFormatter()
    )
    state = provider._state_from_data(data)
    return [option.description for option in provider.get_action_candidates(state)]


def _walk(node):
    yield node
    for child in node.get("children") or []:
        yield from _walk(child)


def test_real_iphone_sheet_hides_the_list_behind_it():
    labels = _ios_fixture("clock_add_alarm_sheet")
    assert not {'Click button "Add" (top of screen)', "Click No Alarm"} & set(labels)
    assert not any("02:37" in label or "Alarms" in label for label in labels)
    assert 'Click text field "Label" (contains "Alarm")' in labels
    assert "Click Label" not in labels
    assert ["Click Snooze", 'Click switch "Snooze" (on)'] == [
        label for label in labels if "Snooze" in label and "Duration" not in label
    ]


def test_real_iphone_floating_search_bar_hides_the_rows_under_it():
    labels = _ios_fixture("settings_root")
    assert not any(
        "Action Button" in label or "Appearance" in label for label in labels
    )
    assert 'Click search field "Search" (bottom of screen)' in labels
    assert [label for label in labels if "Airplane" in label] == [
        "Click Airplane Mode",
        'Click switch "Airplane Mode" (off)',
    ]


def test_real_iphone_keyboard_and_the_page_under_suggestions_are_not_options():
    labels = _ios_fixture("settings_search_keyboard")
    assert not any(
        key in label.lower()
        for label in labels
        for key in ("shift", "emoji", "next keyboard", "delete", "space")
    )
    assert not any("Bluetooth" in label or "Battery" in label for label in labels)
    assert 'Click button "Software Update, General"' in labels
    assert 'Click search field "Search"' in labels


def test_real_iphone_alarm_row_does_not_repeat_its_switch_state():
    labels = _ios_fixture("clock_alarm_list")
    assert "Click 02:37, Alarm" in labels
    assert 'Click switch "02:37, Alarm" (on)' in labels


def test_nav_bar_buttons_stay_when_a_row_scrolls_under_the_bar():
    def scroll_row_under_bar(data):
        cell = next(
            n
            for n in _walk(data["a11y_tree"])
            if n.get("className") == "XCUIElementTypeCell"
            and "02:37" in (n.get("contentDescription") or "")
        )
        shift = cell["boundsInScreen"]["top"] - 50
        for node in _walk(cell):
            node["boundsInScreen"]["top"] -= shift
            node["boundsInScreen"]["bottom"] -= shift

    labels = _ios_fixture("clock_alarm_list", scroll_row_under_bar)
    assert 'Click button "Edit" (top of screen)' in labels
    assert 'Click button "Add" (top of screen)' in labels


def test_keyboard_accessory_bar_stays_offered():
    def add_composer(data):
        keyboard_window = next(
            n
            for n in _walk(data["a11y_tree"])
            if n.get("className") == "XCUIElementTypeWindow"
            and any(c.get("className") == "XCUIElementTypeKeyboard" for c in _walk(n))
        )

        def node(cls, text, left, right, **extra):
            return {
                "className": f"XCUIElementType{cls}",
                "text": text,
                "resourceId": "",
                "contentDescription": "",
                "children": [],
                "isEnabled": True,
                "isClickable": True,
                "isVisibleToUser": True,
                "boundsInScreen": {
                    "left": left,
                    "top": 590,
                    "right": right,
                    "bottom": 630,
                },
                **extra,
            }

        keyboard_window["children"].insert(
            0,
            {
                **node("Other", "", 0, 440, isClickable=False),
                "children": [
                    node("Button", "", 8, 48, contentDescription="Add attachment"),
                    node(
                        "TextView",
                        "",
                        56,
                        380,
                        contentDescription="Message",
                        isClickable=False,
                    ),
                    node("Button", "", 388, 432, contentDescription="Send"),
                ],
            },
        )

    labels = _ios_fixture("settings_search_keyboard", add_composer)
    assert 'Click button "Add attachment"' in labels
    assert 'Click text field "Message"' in labels
    assert 'Click button "Send"' in labels


def test_half_height_sheet_leaves_the_page_above_it():
    def node(cls, top, bottom, children=(), **extra):
        return {
            "className": f"XCUIElementType{cls}",
            "text": "",
            "resourceId": "",
            "contentDescription": "",
            "isEnabled": True,
            "isClickable": False,
            "isVisibleToUser": True,
            "children": list(children),
            "boundsInScreen": {"left": 0, "top": top, "right": 440, "bottom": bottom},
            **extra,
        }

    page = node(
        "Other",
        0,
        956,
        [
            node("Button", 220, 270, contentDescription="Directions", isClickable=True),
            node("Button", 700, 750, contentDescription="Hidden row", isClickable=True),
        ],
    )
    sheet = node(
        "Other",
        0,
        956,
        [
            node(
                "Other",
                520,
                956,
                [
                    node(
                        "NavigationBar",
                        530,
                        580,
                        [
                            node(
                                "Button",
                                535,
                                575,
                                contentDescription="Close",
                                isClickable=True,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    window = node("Window", 0, 956, [page, sheet])
    labels = _labels(_tree(node("Application", 0, 956, [window])))
    assert 'Click button "Directions"' in labels
    assert 'Click button "Hidden row"' not in labels
    assert 'Click button "Close"' in labels


def test_legacy_ios_placeholder_is_not_reported_as_contents():
    from mobilerun.tools.ui.ios_provider import _parse_a11y_tree

    elements = _parse_a11y_tree(
        "Application, 0x1, {{0.0, 0.0}, {402.0, 874.0}}, label: 'Form'\n"
        "  StaticText, 0x2, {{16.0, 300.0}, {60.0, 34.0}}, label: 'Email'\n"
        "  TextField, 0x3, {{90.0, 300.0}, {290.0, 34.0}}, "
        "placeholderValue: 'name@example.com', value: name@example.com\n"
        "  StaticText, 0x4, {{16.0, 400.0}, {60.0, 34.0}}, label: 'Name'\n"
        "  TextField, 0x5, {{90.0, 400.0}, {290.0, 34.0}}, "
        "placeholderValue: 'Your name', value: Bob\n"
    )
    fields = [e for e in elements if e["type"] == "TextField"]
    assert _labels(fields) == [
        'Click text field "name@example.com"',
        'Click text field "Your name" (contains "Bob")',
    ]
    labels = _labels(elements)
    assert 'Click text field "Email"' in labels
    assert 'Click text field "Name" (contains "Bob")' in labels


def test_full_screen_modal_hides_the_page_under_it():
    def make_full_screen(data):
        for node in _walk(data["a11y_tree"]):
            box = node.get("boundsInScreen") or {}
            if (
                box.get("left"),
                box.get("top"),
                box.get("right"),
                box.get("bottom"),
            ) == (0, 62, 440, 956):
                box["top"] = 0

    labels = _ios_fixture("clock_add_alarm_sheet", make_full_screen)
    assert not any("Timers" in label or "World Clock" in label for label in labels)
    assert 'Click button "Cancel"' in labels


def test_keyboard_hides_the_page_rows_behind_its_keys():
    def drop_suggestions(data):
        for node in _walk(data["a11y_tree"]):
            node["children"] = [
                child
                for child in node.get("children") or []
                if not (
                    child.get("className") == "XCUIElementTypeCollectionView"
                    and not child.get("resourceId")
                )
            ]

    labels = _ios_fixture("settings_search_keyboard", drop_suggestions)
    assert not any(
        name in label
        for label in labels
        for name in ("Action Button", "Appearance", "Display & Brightness")
    )
    assert 'Click search field "Search"' in labels


def test_action_sheet_hides_the_tab_bar_under_it():
    def add_action_sheet(data):
        window = next(
            n
            for n in _walk(data["a11y_tree"])
            if n.get("className") == "XCUIElementTypeWindow"
        )

        def button(label, top, bottom):
            return {
                "className": "XCUIElementTypeButton",
                "text": label,
                "resourceId": "",
                "contentDescription": label,
                "children": [],
                "isEnabled": True,
                "isClickable": True,
                "isVisibleToUser": True,
                "boundsInScreen": {
                    "left": 8,
                    "top": top,
                    "right": 432,
                    "bottom": bottom,
                },
            }

        window["children"].append(
            {
                "className": "XCUIElementTypeSheet",
                "text": "",
                "resourceId": "",
                "contentDescription": "",
                "isEnabled": True,
                "isClickable": False,
                "isVisibleToUser": True,
                "boundsInScreen": {"left": 8, "top": 700, "right": 432, "bottom": 936},
                "children": [
                    button("Delete Alarm", 740, 800),
                    button("Cancel", 872, 936),
                ],
            }
        )

    labels = _ios_fixture("clock_alarm_list", add_action_sheet)
    assert not any(
        tab in label
        for label in labels
        for tab in ("World Clock", "Stopwatch", "Timers")
    )
    assert 'Click button "Cancel" (bottom of screen)' in labels
    assert 'Click button "Edit" (top of screen)' in labels


def test_empty_field_takes_the_label_beside_it():
    assert _labels(
        _formatted(
            {
                **_raw("android.widget.TextView", "Name", 600, isClickable=False),
                "boundsInScreen": {"left": 40, "top": 610, "right": 280, "bottom": 690},
            },
            {
                **_raw("android.widget.EditText", "", 600, isEditable=True),
                "boundsInScreen": {
                    "left": 300,
                    "top": 600,
                    "right": 1000,
                    "bottom": 700,
                },
            },
        )
    ) == ['Click text field "Name"']


def test_legacy_ios_field_is_named_by_its_placeholder_not_its_identifier():
    from mobilerun.tools.ui.ios_provider import _parse_a11y_tree

    elements = _parse_a11y_tree(
        "Application, 0x1, {{0.0, 0.0}, {402.0, 874.0}}, label: 'Form'\n"
        "  TextField, 0x2, {{16.0, 300.0}, {370.0, 34.0}}, "
        "identifier: 'email-input', placeholderValue: 'Email'\n"
    )
    assert _labels([e for e in elements if e["type"] == "TextField"]) == [
        'Click text field "Email"'
    ]
