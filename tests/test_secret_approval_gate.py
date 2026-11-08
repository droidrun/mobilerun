import asyncio
import inspect
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from mobilerun.agent import action_context as action_context_module
from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.utils import actions
from mobilerun.agent.utils.actions import type_secret
from mobilerun.credential_manager import (
    ApprovalDecision,
    ApprovalProvider,
    CredentialManager,
    FileCredentialManager,
    SecretSpec,
)
from mobilerun.credential_manager.secret_gate import (
    observe_target,
)
from mobilerun.tools.driver.secret_safe import SecretRedactor, SecretSafeDriver
from mobilerun.tools.ui.state import UIState

CARD_NUMBER = "4111 1111 1111 1111"
CARD_CVC = "737"
SHOP = "com.example.shop"
CHROME = "com.android.chrome"
CHROME_BAR = "com.android.chrome:id/url_bar"


@pytest.fixture(autouse=True)
def no_settle(monkeypatch):
    monkeypatch.setattr(actions, "_SECRET_FOCUS_SETTLE_SECONDS", 0)


def run(coro):
    return asyncio.run(coro)


# -- fakes -------------------------------------------------------------------


def field(index, res, bounds, *, text="", cls="android.widget.EditText", focused=False):
    left, top, right, bottom = bounds
    return {
        "index": index,
        "resourceId": res,
        "className": cls,
        "text": text,
        "bounds": f"{left},{top},{right},{bottom}",
        "_focused": focused,
    }


class Device:
    """One screen: nodes, foreground package, taps move the focus."""

    def __init__(self, package=SHOP, nodes=None, url=None):
        self.package = package
        self.url = url
        self.nodes = nodes or [
            field(1, f"{SHOP}:id/number", (100, 200, 900, 300)),
            field(2, f"{SHOP}:id/cvc", (100, 400, 500, 500)),
            field(3, f"{SHOP}:id/note", (100, 600, 900, 700)),
        ]
        self.taps = []
        self.typed = []
        self.focus_override = None  # callable(node) -> bool, to break focus

    # raw tree
    async def get_ui_tree(self):
        children = []
        for node in self.nodes:
            left, top, right, bottom = (int(p) for p in node["bounds"].split(","))
            children.append(
                {
                    "resourceId": node["resourceId"],
                    "className": node["className"],
                    "text": node["text"],
                    "isFocused": node["_focused"],
                    "boundsInScreen": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    },
                }
            )
        if self.url is not None:
            children.append(
                {
                    "resourceId": self.url[0],
                    "className": "android.widget.EditText",
                    "text": self.url[1],
                    "isFocused": False,
                    "boundsInScreen": {
                        "left": 0,
                        "top": 0,
                        "right": 1080,
                        "bottom": 100,
                    },
                }
            )
        return {
            "a11y_tree": {"packageName": self.package, "children": children},
            "phone_state": {
                "packageName": self.package,
                "isEditable": any(n["_focused"] for n in self.nodes),
            },
            "device_context": {},
        }

    async def tap(self, x, y):
        self.taps.append((x, y))
        for node in self.nodes:
            left, top, right, bottom = (int(p) for p in node["bounds"].split(","))
            node["_focused"] = left <= x < right and top <= y < bottom
            if self.focus_override is not None and node["_focused"]:
                node["_focused"] = self.focus_override(node)

    async def input_text(self, text, clear=False, **kwargs):
        self.typed.append(text)
        return True

    async def screenshot(self, hide_overlay=True):
        return b"png"

    def ui_state(self):
        return UIState(
            [{k: v for k, v in n.items() if k != "_focused"} for n in self.nodes],
            "screen",
            "",
            {"packageName": self.package},
            1080,
            2400,
        )


class Provider:
    supported = {"element_index"}
    requires_coordinate_tools = False

    def __init__(self, device):
        self.device = device

    async def get_state(self):
        return self.device.ui_state()


class Approver(ApprovalProvider):
    def __init__(self, decision="approved", on_request=None):
        self.decision = decision
        self.requests = []
        self.on_request = on_request

    async def request(self, kind, details):
        self.requests.append((kind, dict(details)))
        if self.on_request:
            self.on_request()
        return self.decision


class Secrets(CredentialManager):
    def __init__(self, specs, values=None, on_resolve=None):
        self.specs = specs
        self.values = values or {
            "card_number": CARD_NUMBER,
            "card_cvc": CARD_CVC,
            "card_note": "Jane Doe",
            "login": "hunter22",
        }
        self.on_resolve = on_resolve

    async def resolve_key(self, key):
        if self.on_resolve:
            self.on_resolve()
        return self.values[key]

    async def get_keys(self):
        return list(self.values)

    async def get_secret_specs(self):
        return self.specs


def card_specs(label="Test card", kind="payment_card"):
    return [
        SecretSpec(
            "card_number",
            requires_approval=True,
            approval_kind=kind,
            approval_label=label,
        ),
        SecretSpec(
            "card_cvc",
            requires_approval=True,
            approval_kind=kind,
            approval_label=label,
            redact="input_only",
        ),
        SecretSpec("login"),
    ]


def make_ctx(device, manager, approver=None, *, wrapped=True, state=None):
    driver = SecretSafeDriver(device, SecretRedactor()) if wrapped else device
    state = state or MobileAgentState()
    ctx = ActionContext(
        driver=driver,
        ui=device.ui_state(),
        shared_state=state,
        state_provider=Provider(device),
        credential_manager=manager,
        approval_provider=approver,
    )
    return ctx


# -- public contract ---------------------------------------------------------


def test_frozen_public_contract():
    assert inspect.isabstract(ApprovalProvider)
    assert ApprovalProvider.__abstractmethods__ == {"request"}
    assert ApprovalDecision.__args__ == ("approved", "denied", "expired")
    assert SecretSpec("k") == SecretSpec(
        "k", False, None, None, "global"
    )  # field order and defaults
    with pytest.raises(FrozenInstanceError):
        SecretSpec("k").key = "x"

    signature = inspect.signature(MobileAgent.__init__)
    param = signature.parameters["approval_provider"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None
    names = list(signature.parameters)
    assert names.index("args") < names.index("approval_provider")

    ctx_params = list(inspect.signature(ActionContext.__init__).parameters)
    assert ctx_params[-1] == "approval_provider"
    assert ctx_params[1:-1] == [
        "driver",
        "ui",
        "shared_state",
        "state_provider",
        "app_opener_llm",
        "credential_manager",
        "streaming",
        "macro_recorder",
    ]
    assert action_context_module.ActionContext is ActionContext


def test_default_specs_come_from_keys_and_manager_stays_concrete():
    class OnlyOld(CredentialManager):
        async def resolve_key(self, key):
            return "v"

        async def get_keys(self):
            return ["a", "b"]

    specs = run(OnlyOld().get_secret_specs())

    assert specs == [SecretSpec("a"), SecretSpec("b")]
    assert all(not s.requires_approval and s.redact == "global" for s in specs)
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


def browser(url, package=CHROME, bar=CHROME_BAR):
    return Device(
        package=package,
        url=(bar, url),
        nodes=[field(1, "web:id/card", (100, 200, 900, 300))],
    )


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
