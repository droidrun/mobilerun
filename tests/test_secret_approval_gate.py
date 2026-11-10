import inspect

import pytest
from conftest import (
    CARD_CVC,
    CARD_NUMBER,
    CHROME,
    CHROME_BAR,
    SHOP,
    Approver,
    Device,
    Provider,
    Secrets,
    browser,
    card_specs,
    field,
    make_ctx,
    run,
)

from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.utils.actions import type_secret
from mobilerun.credential_manager import (
    ApprovalProvider,
    CredentialManager,
    FileCredentialManager,
    SecretSpec,
)
from mobilerun.credential_manager.secret_gate import observe_target
from mobilerun.tools.driver.secret_safe import SecretSafeDriver
from mobilerun.tools.ui.state import UIState

pytestmark = pytest.mark.usefixtures("no_settle")


# -- public contract ---------------------------------------------------------


def test_approval_provider_is_keyword_only_and_last_on_the_context():
    param = inspect.signature(MobileAgent.__init__).parameters["approval_provider"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None
    assert list(inspect.signature(ActionContext.__init__).parameters)[-1] == (
        "approval_provider"
    )


def test_default_specs_come_from_keys_and_manager_stays_concrete():
    class OnlyOld(CredentialManager):
        async def resolve_key(self, key):
            return "v"

        async def get_keys(self):
            return ["a", "b"]

    specs = run(OnlyOld().get_secret_specs())

    assert specs == [SecretSpec("a"), SecretSpec("b")]
    assert all(not s.requires_approval and s.redact == "global" for s in specs)
    assert all(not s.mask_from_start and not s.mask_card_numbers for s in specs)
    assert not getattr(CredentialManager.get_secret_specs, "__isabstractmethod__", 0)


def test_file_credential_manager_runs_unchanged_without_provider_or_specs():
    manager = FileCredentialManager({"login": "hunter22"})
    device = Device()
    ctx = make_ctx(device, manager)

    result = run(type_secret("login", 1, ctx=ctx))

    assert result.success
    assert device.typed == ["hunter22"]
    assert device.taps  # tap-then-type as before
    assert ctx.shared_state.terminal_abort is None
    assert "hunter22" not in result.summary


def test_ordinary_secret_never_asks_even_with_a_provider_and_is_registered():
    device = Device()
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    result = run(type_secret("login", 3, ctx=ctx))

    assert result.success and approver.requests == []
    assert device.typed == ["hunter22"]
    assert ctx.driver.redactor.redact_text("pw hunter22") == "pw [secret:login]"
    assert ctx.driver.redactor.black_screenshots is False


def test_unknown_secret_reports_available_key_names_only():
    ctx = make_ctx(Device(), Secrets(card_specs()))

    result = run(type_secret("nope", 1, ctx=ctx))

    assert not result.success and "not found" in result.summary
    assert "Available" in result.summary and "card_number" in result.summary
    assert CARD_NUMBER not in result.summary


# -- approval ----------------------------------------------------------------


def test_approved_secret_types_into_verified_field_and_goes_dark():
    device = Device()
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert result.success, result.summary
    assert device.typed == [CARD_NUMBER]
    assert approver.requests == [
        (
            "payment_card",
            {"label": "Test card", "secretGroup": "Test card", "targetPackage": SHOP},
        )
    ]
    assert ctx.driver.redactor.black_screenshots is True
    assert CARD_NUMBER not in result.summary


def test_approval_is_requested_once_per_run_for_the_same_card_and_target():
    device = Device()
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    first = run(type_secret("card_number", 1, ctx=ctx))
    second = run(type_secret("card_cvc", 2, ctx=ctx))
    third = run(type_secret("card_number", 1, ctx=ctx))

    assert first.success and second.success and third.success
    assert len(approver.requests) == 1
    assert device.typed == [CARD_NUMBER, CARD_CVC, CARD_NUMBER]


def test_another_card_or_target_needs_its_own_approval():
    device = Device()
    approver = Approver()
    specs = card_specs() + [
        SecretSpec(
            "card_note",
            requires_approval=True,
            approval_kind="payment_card",
            approval_label="Other card",
        )
    ]
    ctx = make_ctx(device, Secrets(specs), approver)

    run(type_secret("card_number", 1, ctx=ctx))
    run(type_secret("card_note", 3, ctx=ctx))  # different label
    assert [r[1]["label"] for r in approver.requests] == ["Test card", "Other card"]

    device.package = "com.example.other"  # different target
    for node in device.nodes:
        node["resourceId"] = node["resourceId"].replace(SHOP, "com.example.other")
    ctx.ui = device.ui_state()
    run(type_secret("card_number", 1, ctx=ctx))
    assert approver.requests[-1][1]["targetPackage"] == "com.example.other"
    assert len(approver.requests) == 3


def test_a_reused_agent_starts_every_run_without_approvals():
    device = Device()
    approver = Approver()
    manager = Secrets(card_specs())
    state = MobileAgentState()

    run(
        type_secret(
            "card_number", 1, ctx=make_ctx(device, manager, approver, state=state)
        )
    )
    run(
        type_secret(
            "card_number", 1, ctx=make_ctx(device, manager, approver, state=state)
        )
    )

    assert len(approver.requests) == 2  # new ActionContext per run


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        ("denied", "approval_denied"),
        ("expired", "approval_expired"),
        ("maybe", "approval_unavailable"),  # anything else is not an approval
        (None, "approval_unavailable"),
    ],
)
def test_refused_approval_types_nothing_and_aborts_the_run(decision, reason):
    device = Device()
    ctx = make_ctx(device, Secrets(card_specs()), Approver(decision))

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and reason in result.summary
    assert device.typed == [] and device.taps == []
    assert ctx.shared_state.terminal_abort == reason
    assert ctx.driver.redactor.active is False
    assert CARD_NUMBER not in result.summary


def test_missing_provider_types_nothing_and_aborts_the_run():
    device = Device()
    ctx = make_ctx(device, Secrets(card_specs()), None)

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success
    assert device.typed == [] and device.taps == []
    assert ctx.shared_state.terminal_abort == "approval_unavailable"


def test_a_failing_provider_fails_closed_without_leaking_its_error():
    class Broken(ApprovalProvider):
        async def request(self, kind, details):
            raise RuntimeError(f"host exploded near {CARD_NUMBER}")

    device = Device()
    ctx = make_ctx(device, Secrets(card_specs()), Broken())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert device.typed == []
    assert ctx.shared_state.terminal_abort == "approval_unavailable"
    assert CARD_NUMBER not in result.summary


def test_abort_is_sticky_and_first_reason_wins():
    state = MobileAgentState()
    state.abort_run("approval_denied")
    state.abort_run("approval_expired")

    assert state.terminal_abort == "approval_denied"


# -- guards: nothing may be typed -------------------------------------------


def test_app_switch_between_approval_and_input_types_nothing():
    device = Device()

    def switch():
        device.package = "com.evil.app"

    ctx = make_ctx(device, Secrets(card_specs()), Approver(on_request=switch))

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "changed after approval" in result.summary
    assert device.typed == []


def test_app_switch_while_the_value_is_resolved_types_nothing():
    device = Device()

    def switch():
        device.package = "com.evil.app"

    ctx = make_ctx(device, Secrets(card_specs(), on_resolve=switch), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success
    assert device.typed == []


def test_screen_changed_since_the_model_chose_the_element():
    device = Device()
    ctx = make_ctx(device, Secrets(card_specs()), Approver())
    device.nodes[0]["bounds"] = "100,250,900,350"  # field moved after observation

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "changed" in result.summary
    assert device.typed == [] and device.taps == []


def test_tap_that_focuses_nothing_types_nothing():
    device = Device()
    device.focus_override = lambda node: False
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "focused text field" in result.summary
    assert device.typed == []


def test_another_field_keeping_the_focus_types_nothing():
    device = Device()
    device.nodes[2]["_focused"] = True  # note field already has the focus
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    async def tap_without_moving_focus(x, y):
        device.taps.append((x, y))

    device.tap = tap_without_moving_focus

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success
    assert device.typed == []


def test_focus_that_is_not_the_selected_element_types_nothing():
    device = Device()

    async def focus_elsewhere(x, y):
        device.taps.append((x, y))
        for node in device.nodes:
            node["_focused"] = node["resourceId"].endswith("note")

    device.tap = focus_elsewhere
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "not the one that was selected" in result.summary
    assert device.typed == []


def test_index_minus_one_requires_a_focused_editable_field():
    device = Device()
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    nothing = run(type_secret("card_number", -1, ctx=ctx))
    assert not nothing.success and device.typed == []

    device.nodes[0]["_focused"] = True
    focused = run(type_secret("card_number", -1, ctx=ctx))
    assert focused.success and device.typed == [CARD_NUMBER]


def test_two_focused_fields_are_ambiguous():
    device = Device()
    device.nodes[0]["_focused"] = True
    device.nodes[1]["_focused"] = True
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", -1, ctx=ctx))

    assert not result.success and device.typed == []


def test_browser_approval_is_bound_to_package_and_host():
    device = browser("https://Shop.Example.com/checkout?x=1")
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert result.success, result.summary
    assert approver.requests[0][1]["targetPackage"] == CHROME
    assert approver.requests[0][1]["targetHost"] == "shop.example.com"


def test_host_change_after_approval_types_nothing():
    device = browser("shop.example.com")

    def navigate():
        device.url = (CHROME_BAR, "evil.example.net")

    ctx = make_ctx(device, Secrets(card_specs()), Approver(on_request=navigate))

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and device.typed == []


def test_host_change_for_an_approved_run_asks_again():
    device = browser("shop.example.com")
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)
    run(type_secret("card_number", 1, ctx=ctx))

    device.url = (CHROME_BAR, "other.example.org")
    run(type_secret("card_number", 1, ctx=ctx))

    assert [r[1]["targetHost"] for r in approver.requests] == [
        "shop.example.com",
        "other.example.org",
    ]


@pytest.mark.parametrize(
    "text", ["", "Search or type web address", "localhost-less", "a b.com"]
)
def test_unreadable_host_types_nothing_and_asks_nobody(text):
    device = browser(text)
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and device.typed == []
    assert approver.requests == []
    assert ctx.shared_state.terminal_abort is None  # a retry may still work


def test_missing_address_bar_in_a_known_browser_types_nothing():
    device = browser("shop.example.com")
    device.url = None
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    assert not run(type_secret("card_number", 1, ctx=ctx)).success
    assert device.typed == []


def test_unknown_browser_with_an_address_bar_is_refused():
    device = browser(
        "shop.example.com",
        package="com.unknown.browser",
        bar="com.unknown.browser:id/url_bar",
    )
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "not supported" in result.summary
    assert device.typed == [] and approver.requests == []


def test_page_content_cannot_spoof_the_address_bar():
    device = browser("good.example.com")
    device.nodes.append(
        field(2, "url_bar", (100, 800, 900, 900), text="evil.example.net")
    )
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    run(type_secret("card_number", 1, ctx=ctx))

    assert approver.requests[0][1]["targetHost"] == "good.example.com"


def test_ambiguous_hosts_in_browser_chrome_type_nothing():
    device = browser("good.example.com")
    original = device.get_ui_tree

    async def two_bars():
        data = await original()
        data["a11y_tree"]["children"].append(
            {
                "resourceId": CHROME_BAR,
                "text": "other.example.com",
                "className": "android.widget.EditText",
            }
        )
        return data

    device.get_ui_tree = two_bars
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    assert not run(type_secret("card_number", 1, ctx=ctx)).success
    assert device.typed == []


def test_secret_is_never_typed_into_the_address_bar():
    device = browser("good.example.com")
    original = device.get_ui_tree

    async def bar_focused():
        data = await original()
        for node in data["a11y_tree"]["children"]:
            if node["resourceId"] == CHROME_BAR:
                node["isFocused"] = True
        for node in data["a11y_tree"]["children"]:
            if node["resourceId"] != CHROME_BAR:
                node["isFocused"] = False
        return data

    device.get_ui_tree = bar_focused
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", -1, ctx=ctx))

    assert not result.success and "address bar" in result.summary
    assert device.typed == []


def test_safari_address_field_is_a_supported_adapter():
    data = {
        "a11y_tree": {
            "children": [
                {"identifier": "URL", "text": "shop.example.com"},
                {"className": "XCUIElementTypeTextField", "isFocused": True},
            ]
        },
        "phone_state": {"packageName": "com.apple.mobilesafari"},
    }

    observation = observe_target(data)

    assert observation.target.host == "shop.example.com"
    assert len(observation.focused_editable) == 1


def test_host_normalization_rejects_userinfo_tricks():
    from mobilerun.credential_manager.secret_gate import host_from_text

    assert host_from_text("https://good.example.com@evil.example.net/x") == (
        "evil.example.net"
    )
    assert host_from_text("EXAMPLE.com:8443/path") == "example.com"
    assert host_from_text("not a url") is None


# -- destination identity ------------------------------------------------------


def _focus(device, resource_suffix, *, mutate=None):
    """Tap handler that moves the focus to the node with this id suffix."""

    async def tap(x, y):
        device.taps.append((x, y))
        for node in device.nodes:
            node["_focused"] = node["resourceId"].endswith(resource_suffix)
            if node["_focused"] and mutate:
                mutate(node)

    return tap


def test_focused_enclosing_container_is_not_the_selected_field():
    device = Device()
    device.nodes.append(
        field(4, f"{SHOP}:id/container", (50, 150, 950, 350))  # wraps field 1
    )
    device.tap = _focus(device, "container")
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "not the one that was selected" in result.summary
    assert device.typed == []


def test_same_bounds_replacement_with_another_id_is_refused():
    device = Device()

    def replace(node):
        node["resourceId"] = f"{SHOP}:id/lookalike"

    async def tap(x, y):
        device.taps.append((x, y))
        for node in device.nodes:
            node["_focused"] = node["resourceId"].endswith("number")
            if node["_focused"]:
                replace(node)

    device.tap = tap
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and device.typed == []


def test_selected_element_without_text_field_evidence_is_refused():
    device = Device()
    device.nodes.append(
        field(4, f"{SHOP}:id/pay", (100, 800, 900, 900), cls="android.widget.Button")
    )
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", 4, ctx=ctx))

    assert not result.success and "not a text field" in result.summary
    assert device.typed == [] and device.taps == []


def test_visible_keyboard_alone_is_not_text_field_evidence():
    device = Device()

    def strip_editability(node):
        node["className"] = "android.view.View"

    device.tap = _focus(device, "number", mutate=strip_editability)
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    assert not run(type_secret("card_number", 1, ctx=ctx)).success
    assert device.typed == []

    observation = observe_target(
        {
            "a11y_tree": {
                "children": [{"className": "android.view.View", "isFocused": True}]
            },
            "phone_state": {"packageName": SHOP, "isEditable": True},
        }
    )
    assert observation.focused_editable == []


def test_two_nodes_with_the_same_signature_are_ambiguous():
    device = Device()
    device.nodes.append(dict(device.nodes[0], index=9))  # identical twin
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "uniquely" in result.summary
    assert device.typed == []


def test_class_replaced_with_a_lookalike_during_the_tap_is_refused():
    device = Device()

    async def tap(x, y):
        device.taps.append((x, y))
        for node in device.nodes:
            node["_focused"] = node["resourceId"].endswith("number")
            if node["_focused"]:
                node["className"] = "com.example.replacement.EditText"

    device.tap = tap
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and "not the one that was selected" in result.summary
    assert device.typed == []


def test_protected_secret_needs_a_driver_that_can_mask_afterwards():
    device = Device()
    ctx = make_ctx(device, Secrets(card_specs()), Approver(), wrapped=False)

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and device.typed == []


# -- address bar identity ------------------------------------------------------


def _web_spoof_device():
    """A fully qualified address-bar id that only exists inside web content."""
    device = Device(package=CHROME, nodes=[], url=None)
    original = device.get_ui_tree

    async def tree():
        data = await original()
        data["a11y_tree"]["children"] = [
            {
                "className": "android.webkit.WebView",
                "children": [
                    {
                        "resourceId": CHROME_BAR,
                        "className": "android.widget.EditText",
                        "text": "good.example.com",
                    },
                    {
                        "className": "android.widget.EditText",
                        "isFocused": True,
                        "text": "",
                    },
                ],
            }
        ]
        return data

    device.get_ui_tree = tree
    return device


def test_address_bar_id_inside_web_content_is_not_chrome():
    device = _web_spoof_device()
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    result = run(type_secret("card_number", -1, ctx=ctx))

    assert not result.success
    assert approver.requests == [] and device.typed == []


def test_address_bar_must_be_a_text_node():
    device = browser("good.example.com")
    device.url = (CHROME_BAR, "good.example.com")
    original = device.get_ui_tree

    async def tree():
        data = await original()
        for node in data["a11y_tree"]["children"]:
            if node["resourceId"] == CHROME_BAR:
                node["className"] = "android.view.View"
        return data

    device.get_ui_tree = tree
    approver = Approver()
    ctx = make_ctx(device, Secrets(card_specs()), approver)

    assert not run(type_secret("card_number", 1, ctx=ctx)).success
    assert approver.requests == []


def test_duckduckgo_address_bar_is_a_supported_adapter():
    observation = observe_target(
        {
            "a11y_tree": {
                "children": [
                    {
                        "resourceId": "com.duckduckgo.mobile.android:id/omnibarTextInput",
                        "className": "android.widget.EditText",
                        "text": "duckduckgo.com",
                    }
                ]
            },
            "phone_state": {"packageName": "com.duckduckgo.mobile.android"},
        }
    )

    assert observation.target.host == "duckduckgo.com"


# -- values never leak -------------------------------------------------------


def test_exceptions_from_the_driver_never_surface_the_value(caplog):
    device = Device()

    async def explode(text, clear=False, **kwargs):
        raise ValueError(f"cannot type {text}")

    device.input_text = explode
    ctx = make_ctx(device, Secrets(card_specs()), Approver())

    with caplog.at_level("DEBUG"):
        result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success
    assert CARD_NUMBER not in result.summary
    assert CARD_NUMBER not in caplog.text
    assert "ValueError" in result.summary


def test_resolver_errors_never_surface_the_value():
    class Leaky(Secrets):
        async def resolve_key(self, key):
            raise RuntimeError(f"vault said {CARD_NUMBER}")

    ctx = make_ctx(Device(), Leaky(card_specs()), Approver())

    result = run(type_secret("card_number", 1, ctx=ctx))

    assert not result.success and CARD_NUMBER not in result.summary


def test_macro_records_protected_marker_without_the_value():
    from mobilerun.macro.recorder import MacroRecorder

    device = Device()
    ctx = make_ctx(device, Secrets(card_specs()), Approver())
    ctx.macro_recorder = MacroRecorder()

    run(type_secret("card_number", 1, ctx=ctx))
    run(type_secret("login", 3, ctx=ctx))

    secret_steps = [
        a for a in ctx.macro_recorder.actions if a["action_type"] == "type_secret"
    ]
    assert secret_steps[0]["protected"] is True
    assert "protected" not in secret_steps[1]
    assert CARD_NUMBER not in str(ctx.macro_recorder.actions)
    assert "hunter22" not in str(ctx.macro_recorder.actions)


# -- agent wiring ------------------------------------------------------------


def _agent_module(monkeypatch):
    from mobilerun.agent.droid import droid_agent as agent_module

    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_module.MobileAgent, "_configure_default_logging", lambda *a, **k: None
    )
    monkeypatch.setattr(agent_module, "capture", lambda *a, **k: None)
    return agent_module


class FakeFlowContext:
    def __init__(self):
        self.store = self

    def write_event_to_stream(self, event):
        return None

    async def set(self, key, value):
        setattr(self, key, value)

    async def get(self, key, default=None):
        return getattr(self, key, default)


class InjectableDriver:
    platform = "Android"
    supported = {"tap", "input_text", "screenshot", "get_ui_tree"}
    supported_buttons = {"back"}

    def __init__(self):
        self.typed = []

    async def connect(self):
        return None

    async def get_date(self):
        raise NotImplementedError

    async def get_ui_tree(self):
        return {}

    async def input_text(self, text, clear=False, **kwargs):
        self.typed.append(text)
        return True


def _started_agent(monkeypatch, **kwargs):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    from mobilerun.config_manager.config_manager import MobileConfig, TelemetryConfig

    agent_module = _agent_module(monkeypatch)
    config = MobileConfig(telemetry=TelemetryConfig(enabled=False))
    config.logging.save_trajectory = "none"
    agent = agent_module.MobileAgent(
        "Pay", config=config, llms=MockLLM(), driver=InjectableDriver(), **kwargs
    )
    run(agent.start_handler(FakeFlowContext(), StartEvent()))
    return agent


def test_agent_wraps_driver_and_provider_when_credentials_exist(monkeypatch):
    approver = Approver()
    host_provider = Provider(Device())
    agent = _started_agent(
        monkeypatch,
        credentials=Secrets(card_specs()),
        approval_provider=approver,
        state_provider=host_provider,
    )

    assert isinstance(agent.driver, SecretSafeDriver)
    assert agent.action_ctx.approval_provider is approver
    assert agent.action_ctx.driver is agent.driver
    # the host-owned provider is untouched; the engine talks to the proxy
    assert agent.state_provider.inner is host_provider
    assert host_provider.device is not None
    assert agent.state_provider.trusted is False


def test_agent_without_credentials_keeps_the_plain_driver(monkeypatch):
    agent = _started_agent(monkeypatch)

    assert not isinstance(agent.driver, SecretSafeDriver)
    assert agent.action_ctx.approval_provider is None


def test_injected_provider_sees_only_redacted_data_after_typing(monkeypatch):
    class Cached:
        supported = {"element_index"}
        requires_coordinate_tools = False

        def __init__(self):
            self.cached = UIState(
                [{"index": 1, "text": f"Card {CARD_NUMBER}"}],
                f"Card {CARD_NUMBER}",
                "",
                {},
                10,
                10,
            )

        async def get_state(self):
            return self.cached

    host = Cached()
    agent = _started_agent(
        monkeypatch,
        credentials=Secrets(card_specs()),
        state_provider=host,
        approval_provider=Approver(),
    )
    run(agent.driver.input_secret("card_number", CARD_NUMBER, protected=True))

    state = run(agent.state_provider.get_state())

    assert "4111" not in state.formatted_text
    assert "4111" not in repr(state.elements)


def test_run_start_resets_registrations_and_abort(monkeypatch):
    agent = _started_agent(monkeypatch, credentials=Secrets(card_specs()))
    agent._redactor.register("x", "leftover-value")
    agent.shared_state.abort_run("approval_denied")

    from llama_index.core.workflow import StartEvent

    run(agent.start_handler(FakeFlowContext(), StartEvent()))

    assert agent._redactor.active is False
    assert agent.shared_state.terminal_abort is None


def test_first_observation_masks_prefilled_card_and_input_only_fields(monkeypatch):
    from test_secret_redaction import pixels, png, raw_node, raw_tree

    class FilledDriver(InjectableDriver):
        async def get_ui_tree(self):
            return raw_tree(
                [
                    raw_node(CARD_NUMBER, cls="android.widget.TextView"),
                    raw_node("12 / 27", cls="android.widget.EditText"),
                    raw_node(CARD_CVC, cls="android.widget.EditText"),
                    raw_node("12/27"),
                    raw_node(f"Price {CARD_CVC}"),
                ]
            )

        async def screenshot(self, **kwargs):
            return png()

    specs = [
        SecretSpec("number", mask_from_start=True, redact="input_only"),
        SecretSpec("expiry", mask_from_start=True, redact="input_only"),
        SecretSpec("cvc", mask_from_start=True, redact="input_only"),
    ]
    # Even without approval gating, masking from start blacks out vision.
    manager = Secrets(
        specs, values={"number": CARD_NUMBER, "expiry": "12/27", "cvc": CARD_CVC}
    )
    agent = _started_agent(monkeypatch, credentials=manager)
    agent.driver.inner = FilledDriver()

    tree = run(agent.driver.get_ui_tree())
    assert [n["text"] for n in tree["a11y_tree"]["children"]] == [
        "[secret:number]",
        "[secret:expiry]",
        "[secret:cvc]",
        "12/27",
        f"Price {CARD_CVC}",
    ]
    state = run(agent.state_provider.get_state())
    assert CARD_NUMBER not in state.formatted_text
    assert "[secret:number]" in state.formatted_text
    assert "[secret:expiry]" in state.formatted_text
    assert "[secret:cvc]" in state.formatted_text
    assert pixels(run(agent.driver.screenshot()))[1] == {(0, 0, 0)}
    assert agent.driver.inner.typed == []


def test_start_masking_covers_host_provider_and_preserves_approval(monkeypatch):
    approver = Approver("denied")
    device = Device(
        nodes=[field(1, f"{SHOP}:id/card", (0, 0, 500, 100), text=CARD_NUMBER)]
    )
    manager = Secrets(
        [SecretSpec("card", mask_from_start=True, requires_approval=True)],
        values={"card": CARD_NUMBER},
    )
    agent = _started_agent(
        monkeypatch,
        credentials=manager,
        approval_provider=approver,
        state_provider=Provider(device),
    )
    agent.driver.inner = device
    state = run(agent.state_provider.get_state())
    assert state.elements[0]["text"] == "[secret:card]"
    assert approver.requests == []
    agent.action_ctx.ui = state
    result = run(type_secret("card", 1, ctx=agent.action_ctx))
    assert not result.success
    assert agent.shared_state.terminal_abort == "approval_denied"
    assert device.typed == []


def test_run_start_enables_global_card_rule_without_resolving_ordinary_secrets(
    monkeypatch,
):
    def no_resolve():
        pytest.fail("ordinary secrets must still be resolved only when typed")

    manager = Secrets(
        [SecretSpec("login", mask_card_numbers=True)], on_resolve=no_resolve
    )
    agent = _started_agent(monkeypatch, credentials=manager)

    assert agent._redactor.redact_text(CARD_NUMBER) == "[secret:card_number]"
    assert not agent._redactor.black_screenshots


@pytest.mark.parametrize("failure", ["specs", "resolve"])
def test_start_masking_failure_ends_real_workflow_without_observations_or_llm(
    monkeypatch, caplog, failure
):
    from unittest.mock import Mock

    from llama_index.core.llms.mock import MockLLM

    from mobilerun.config_manager.config_manager import MobileConfig, TelemetryConfig

    class Unavailable(Secrets):
        async def get_secret_specs(self):
            if failure == "specs":
                raise RuntimeError("host error with unregistered-value")
            return await super().get_secret_specs()

        async def resolve_key(self, key):
            raise RuntimeError("host error with unregistered-value")

    module = _agent_module(monkeypatch)
    fast = Mock(side_effect=AssertionError("LLM loop must not start"))
    extract = Mock(side_effect=AssertionError("structured extraction must not start"))
    monkeypatch.setattr(module, "FastAgent", fast)
    monkeypatch.setattr(module, "StructuredOutputAgent", extract)
    driver = InjectableDriver()
    driver.get_ui_tree = Mock(
        side_effect=AssertionError("must not observe unresolved secrets")
    )
    config = MobileConfig(telemetry=TelemetryConfig(enabled=False))
    agent = module.MobileAgent(
        "Pay",
        config=config,
        llms=MockLLM(),
        driver=driver,
        credentials=Unavailable([SecretSpec("card", mask_from_start=True)]),
    )

    async def go():
        return await agent.run()

    result = run(go())
    assert result.success is False and result.reason == "secret_unavailable"
    assert "unregistered-value" not in repr(result)
    assert "unregistered-value" not in caplog.text
    assert agent.shared_state.terminal_abort == "secret_unavailable"
    assert agent.shared_state.workflow_completed
    assert not agent._redactor.active
    fast.assert_not_called()
    extract.assert_not_called()
    driver.get_ui_tree.assert_not_called()
