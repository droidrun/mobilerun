"""Shared fakes for the secret approval tests."""

import asyncio

import pytest

from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.utils import actions
from mobilerun.credential_manager import (
    ApprovalProvider,
    CredentialManager,
    SecretSpec,
)
from mobilerun.tools.driver.secret_safe import SecretRedactor, SecretSafeDriver
from mobilerun.tools.ui.state import UIState

CARD_NUMBER = "4111 1111 1111 1111"
CARD_CVC = "737"
SHOP = "com.example.shop"
CHROME = "com.android.chrome"
CHROME_BAR = "com.android.chrome:id/url_bar"


@pytest.fixture
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


def browser(url, package=CHROME, bar=CHROME_BAR):
    return Device(
        package=package,
        url=(bar, url),
        nodes=[field(1, "web:id/card", (100, 200, 900, 300))],
    )
