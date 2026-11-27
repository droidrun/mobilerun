"""Safe, concise summaries of agent device actions."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

from mobilerun.logs.handlers import VERBOSE


class ActionUI(Protocol):
    def get_element(self, index: int) -> dict[str, Any] | None: ...


class ActionState(Protocol):
    @property
    def step_number(self) -> int: ...


class ActionDevice(Protocol):
    @property
    def identifier(self) -> str | None: ...


class ActionContext(Protocol):
    @property
    def driver(self) -> ActionDevice: ...

    @property
    def ui(self) -> ActionUI | None: ...

    @property
    def shared_state(self) -> ActionState: ...


logger = logging.getLogger("mobilerun.actions")


@dataclass
class ActionCounter:
    count: int = 0


_action_counter: ContextVar[ActionCounter | None] = ContextVar(
    "action_counter", default=None
)


@contextmanager
def action_counting() -> Iterator[ActionCounter]:
    """Track action numbering independently for each asynchronous agent run."""
    counter = ActionCounter()
    token = _action_counter.set(counter)
    try:
        yield counter
    finally:
        _action_counter.reset(token)


_ACTIONS = {
    "click": "tap",
    "click_at": "tap",
    "click_area": "tap",
    "tap": "tap",
    "long_press": "long press",
    "long_press_at": "long press",
    "swipe": "swipe",
    "system_button": "button",
    "press_button": "button",
    "type_text": "type",
    "type_text_direct": "type",
    "type_secret": "type",
    "input_text": "type",
    "open_app": "open app",
    "open_bundle_id": "open app",
    "start_app": "open app",
    "wait": "wait",
    "complete": "done",
}


def format_action(name: str, args: object, ui: ActionUI | None = None) -> str | None:
    verb = _ACTIONS.get(name)
    if verb is None:
        return None
    args = args if isinstance(args, dict) else {}
    parts = [verb]
    index = args.get("index")
    if type(index) is int:
        element = ui.get_element(index) if ui is not None else None
        if isinstance(element, dict):
            class_name = element.get("className")
            resource_id = element.get("resourceId")
            class_name = (
                class_name.rsplit(".", 1)[-1] if isinstance(class_name, str) else ""
            )
            resource_id = (
                resource_id.rsplit("/", 1)[-1] if isinstance(resource_id, str) else ""
            )
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", class_name):
                class_name = ""
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", resource_id):
                resource_id = ""
            target = class_name + (f"[{resource_id}]" if resource_id else "")
            if target:
                parts.append(target)
        parts.append(f"#{index}")
    elif all(type(args.get(k)) in (int, float) for k in ("x", "y")):
        parts.append(f"({args['x']}, {args['y']})")
    elif (
        isinstance(args.get("coordinate"), list)
        and len(args["coordinate"]) == 2
        and all(type(v) in (int, float) for v in args["coordinate"])
    ):
        parts.append(f"({args['coordinate'][0]}, {args['coordinate'][1]})")
    if verb == "type" or any(
        k in args for k in ("text", "password", "secret", "token", "credential")
    ):
        parts.append("<redacted>")
    if verb == "button" and args.get("button") in ("home", "back", "enter"):
        parts.append(args["button"])
    return " ".join(parts)


def action_line(device: object, source: str, step: int, action: str) -> str:
    label = str(device or "unknown")
    if source == "agent" and not logger.isEnabledFor(VERBOSE):
        return f"{label} · step {step}  {action}"
    return f"{label:<13} · {source:<5} · step {step}  {action}"


def agent_line(ctx: ActionContext | None, name: str, args: object) -> str | None:
    # Older callers may provide a partial context instead of an ActionContext.
    try:
        driver = ctx.driver if ctx is not None else None
    except AttributeError:
        driver = None
    label = driver.identifier if driver is not None else None
    try:
        step = (ctx.shared_state.step_number if ctx is not None else 0) or 1
    except AttributeError:
        step = 1
    try:
        ui = ctx.ui if ctx is not None else None
    except AttributeError:
        ui = None
    action = format_action(name, args, ui)
    counter = _action_counter.get()
    if action and name != "complete" and counter is not None:
        counter.count += 1
        step = counter.count
    return action_line(label, "agent", step, action) if action else None


def cloud_progress(data: str | bytes | bytearray) -> tuple[str, int | None] | None:
    """Return a safe action summary and step, or None for non-action events."""
    try:
        payload = json.loads(data)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    action = payload.get("action", payload.get("tool_name", payload.get("action_name")))
    args = payload.get("tool_args", payload.get("args", payload.get("parameters", {})))
    if isinstance(action, dict):
        args = action.get("args", action.get("parameters", args))
        action = action.get("name", action.get("type"))
    summary = format_action(action, args) if isinstance(action, str) else None
    step = payload.get("step_number", payload.get("step"))
    return (
        (summary, step if type(step) is int and step > 0 else None) if summary else None
    )
