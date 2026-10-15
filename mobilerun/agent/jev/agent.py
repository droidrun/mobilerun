"""JevAgent: observe the device, ask Jev for one action, run it, repeat."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from mobilerun_core_local.driver.base import DeviceDisconnectedError

from mobilerun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agent.droid.events import ExternalUserMessageDroppedEvent
from mobilerun.agent.jev.events import JevDecisionEvent, JevEndEvent, JevStepEvent
from mobilerun.agent.jev.observation import JevElement, JevScreen, build_screen
from mobilerun.agent.jev.policy import (
    Capabilities,
    HistoryEntry,
    JevDecision,
    build_plan,
    decide,
    describe_field,
)
from mobilerun.config_manager.config_manager import AgentConfig

if TYPE_CHECKING:
    from mobilerun.agent.action_context import ActionContext
    from mobilerun.agent.droid.state import MobileAgentState
    from mobilerun.agent.jev.client import JevClient
    from mobilerun.agent.tool_registry import ToolRegistry
    from mobilerun.tools.ui.provider import StateProvider
    from mobilerun.tools.ui.state import UIState

logger = logging.getLogger("mobilerun")

LOADING_TIMEOUT = 15.0
# Extra reads while an action's effect has not reached the accessibility tree.
SETTLE_READS = 3
# Identical action on an unchanged screen: one repeat is allowed, then the run stops.
MAX_REPEATS = 1
FIRST_WAIT = 0.5
MAX_WAIT = 2.0
_BUTTONS = frozenset({"back", "home", "enter"})
_OPERATION_LABELS = {
    "BACK": "Navigate back",
    "HOME": "Go to the home screen",
    "ENTER": "Press Enter",
    "WAIT": "Wait for the screen to update",
}


def wait_duration(consecutive_waits: int) -> float:
    """Backoff for consecutive WAITs: 0.5, 1, 2, 2, ... seconds."""
    return min(FIRST_WAIT * 2**consecutive_waits, MAX_WAIT)


class JevAgent(Workflow):
    """Runs a goal with TypeSafe Jev choosing every device action.

    Each step observes the device, asks Jev one question set and executes the
    chosen action through the tool registry, so actions record into macros
    like any other agent's.
    """

    def __init__(
        self,
        client: JevClient,
        agent_config: AgentConfig,
        registry: ToolRegistry,
        action_ctx: ActionContext,
        state_provider: StateProvider,
        shared_state: MobileAgentState,
        *,
        platform: str,
        save_trajectory: str = "none",
        input_texts: Sequence[str] = (),
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.client = client
        self.agent_config = agent_config
        self.max_steps = agent_config.max_steps
        self.min_confidence = agent_config.jev.min_confidence
        self.registry = registry
        self.action_ctx = action_ctx
        self.state_provider = state_provider
        self.shared_state = shared_state
        self.platform = platform
        self.save_trajectory = save_trajectory
        self.input_texts = [str(text) for text in input_texts]
        self.debug = debug
        self._stream_screenshots = (
            os.environ.get("MOBILERUN_STREAM_SCREENSHOTS")
            or os.environ.get("DROIDRUN_STREAM_SCREENSHOTS")
            or ""
        ).lower() in ("1", "true")

    @step
    async def run_loop(self, ctx: Context, ev: StartEvent) -> StopEvent:
        goal = ev.get("input", default=None)
        if not goal:
            raise ValueError("Jev needs a goal to run.")

        capabilities = self._capabilities()
        apps = await self._installed_apps() if capabilities.open_app else []
        history: list[HistoryEntry] = []
        executed: dict[str, int] = {}
        previous_fingerprint: str | None = None
        waiting_since: float | None = None
        consecutive_waits = 0
        # Some backends never report input focus: remember the field Jev tapped.
        tapped_input: JevElement | None = None
        unreadable_since: float | None = None
        unreadable_reads = 0
        settle_reads = SETTLE_READS

        while True:
            self._drop_user_messages(ctx)
            ui = await self._read_state()
            screen = build_screen(ui, platform=self.platform, tapped_input=tapped_input)
            if not screen.elements:
                # An empty tree means the screen could not be read (e.g. an app is
                # launching); read it again instead of asking Jev about nothing.
                now = time.monotonic()
                unreadable_since = unreadable_since or now
                if now - unreadable_since >= LOADING_TIMEOUT:
                    return self._finish(
                        ctx,
                        False,
                        "Stopped: the device screen could not be read for "
                        f"{LOADING_TIMEOUT:g}s.",
                    )
                logger.debug("Screen not readable yet; reading it again")
                await asyncio.sleep(wait_duration(unreadable_reads))
                unreadable_reads += 1
                continue
            unreadable_since = None
            unreadable_reads = 0
            if (
                settle_reads < SETTLE_READS
                and screen.fingerprint == previous_fingerprint
            ):
                # The tree can lag behind the screen, e.g. while an app launches.
                await asyncio.sleep(wait_duration(settle_reads))
                settle_reads += 1
                continue
            settle_reads = SETTLE_READS
            if history and previous_fingerprint is not None:
                history[-1].screen_changed = previous_fingerprint != screen.fingerprint

            # One recorded frame per Jev decision; settle and unreadable reads are not.
            await self._record_observation(ctx, ui)
            plan = build_plan(
                goal,
                screen,
                history=history,
                texts=self.input_texts,
                apps=apps,
                capabilities=capabilities,
            )
            response = await self.client.systemone(plan.state, plan.questions)
            decision = decide(plan, response, min_confidence=self.min_confidence)
            self._emit_decision(ctx, decision)

            if decision.status != "action":
                reason = decision.reason or f"Jev stopped with {decision.operation}."
                return self._finish(ctx, decision.status == "done", reason)
            # A DONE after the last allowed action is still honored above.
            if self.shared_state.step_number >= self.max_steps:
                return self._finish(
                    ctx, False, f"Reached max step count of {self.max_steps} steps"
                )
            if not decision.tool:
                return self._finish(
                    ctx, False, f"Jev chose {decision.operation}, which cannot run."
                )

            operation = decision.operation
            args = dict(decision.args or {})
            is_wait = operation == "WAIT"
            signature = ""
            if is_wait:
                now = time.monotonic()
                if waiting_since is None:
                    waiting_since = now
                elapsed = now - waiting_since
                if elapsed >= LOADING_TIMEOUT:
                    return self._finish(
                        ctx,
                        False,
                        f"Stopped: the screen kept loading for {LOADING_TIMEOUT:g}s.",
                    )
                args["duration"] = min(
                    wait_duration(consecutive_waits), LOADING_TIMEOUT - elapsed
                )
            else:
                waiting_since = None
                signature = (
                    f"{screen.fingerprint}:{decision.tool}:"
                    f"{json.dumps(args, sort_keys=True, default=str)}"
                )
                if executed.get(signature, 0) > MAX_REPEATS:
                    subject = _stop_subject(decision, screen)
                    return self._finish(
                        ctx,
                        False,
                        f"Stopped: Jev repeated {subject} on an unchanged screen.",
                    )

            self.shared_state.step_number += 1
            step_number = self.shared_state.step_number
            result = await self.registry.execute(
                decision.tool, args, self.action_ctx, workflow_ctx=ctx
            )
            history.append(
                HistoryEntry(
                    operation=operation,
                    label=_history_label(decision, screen),
                    text=decision.text,
                    screen_changed=None,
                    succeeded=result.success,
                )
            )
            previous_fingerprint = screen.fingerprint

            if not result.success:
                # Never retry: the action may have partly happened.
                ctx.write_event_to_stream(
                    JevStepEvent(
                        step=step_number,
                        operation=operation,
                        success=False,
                        summary=result.summary,
                    )
                )
                return self._finish(
                    ctx, False, f"Action {operation} failed: {result.summary}"
                )

            if operation == "TAP":
                # Tapping a button, e.g. to submit, keeps the field Jev typed into.
                tapped_input = _tapped_input(screen, args.get("index")) or tapped_input
            elif operation not in ("TYPE_TEXT", "WAIT"):
                tapped_input = None

            if is_wait:
                consecutive_waits += 1
            else:
                executed[signature] = executed.get(signature, 0) + 1
                consecutive_waits = 0
                settle_reads = 0
                await asyncio.sleep(self.agent_config.after_sleep_action)
            ctx.write_event_to_stream(
                JevStepEvent(
                    step=step_number,
                    operation=operation,
                    success=True,
                    summary=result.summary,
                )
            )

    def _capabilities(self) -> Capabilities:
        tools = self.registry.tools
        open_app = tools.get("open_app")
        buttons: frozenset[str] = frozenset()
        if "system_button" in tools:
            supported = getattr(self.action_ctx.driver, "supported_buttons", None)
            buttons = frozenset(str(b).lower() for b in supported or ()) & _BUTTONS
        return Capabilities(
            tap="click" in tools,
            type_text="type" in tools,
            swipe="swipe" in tools,
            # Jev opens apps by exact identifier only.
            open_app=open_app is not None
            and "app_id" in (getattr(open_app, "params", None) or {}),
            wait="wait" in tools,
            buttons=buttons,
        )

    async def _installed_apps(self) -> list[dict]:
        try:
            apps = await self.action_ctx.driver.get_apps(include_system=True)
        except DeviceDisconnectedError:
            raise
        except Exception as e:
            logger.debug(f"Could not list installed apps for Jev: {e}")
            return []
        return [
            app for app in apps or [] if isinstance(app, dict) and app.get("package")
        ]

    async def _read_state(self) -> UIState:
        ui = await self.state_provider.get_state()
        self.action_ctx.ui = ui
        phone_state = ui.phone_state if isinstance(ui.phone_state, dict) else {}
        state = self.shared_state
        state.previous_formatted_device_state = state.formatted_device_state
        state.formatted_device_state = ui.formatted_text
        state.focused_text = ui.focused_text
        state.a11y_tree = ui.elements
        state.phone_state = phone_state
        state.update_current_app(
            package_name=phone_state.get("packageName", "Unknown"),
            activity_name=phone_state.get("currentApp", "Unknown"),
        )
        return ui

    async def _record_observation(self, ctx: Context, ui: UIState) -> None:
        if self.save_trajectory != "none" or self._stream_screenshots:
            try:
                screenshot = await self.action_ctx.driver.screenshot()
                if screenshot:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
            except DeviceDisconnectedError:
                raise
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")
        ctx.write_event_to_stream(RecordUIStateEvent(ui_state=ui.elements))

    def _emit_decision(self, ctx: Context, decision: JevDecision) -> None:
        usage = decision.usage or {}
        ctx.write_event_to_stream(
            JevDecisionEvent(
                step=self.shared_state.step_number + 1,
                status=decision.status,
                operation=decision.operation,
                confidence=decision.confidence,
                label=decision.label,
                target_confidence=decision.target_confidence,
                model=decision.model,
                latency_ms=decision.latency_ms,
                input_tokens=usage.get("input_tokens"),
            )
        )
        if decision.probabilities:
            top = sorted(
                decision.probabilities.items(), key=lambda item: item[1], reverse=True
            )[:4]
            logger.debug(
                "Jev operation probabilities: "
                + ", ".join(f"{name} {value:.2f}" for name, value in top)
            )

    def _drop_user_messages(self, ctx: Context) -> None:
        pending = self.shared_state.drain_user_messages()
        if not pending:
            return
        logger.warning(
            f"⚠️ Dropping {len(pending)} external user message(s): "
            "the Jev agent does not read messages"
        )
        ctx.write_event_to_stream(
            ExternalUserMessageDroppedEvent(
                message_ids=[message.id for message in pending],
                reason="jev_agent",
                step_number=self.shared_state.step_number,
            )
        )

    def _finish(self, ctx: Context, success: bool, reason: str) -> StopEvent:
        self._drop_user_messages(ctx)
        steps = self.shared_state.step_number
        ctx.write_event_to_stream(
            JevEndEvent(success=success, reason=reason, steps=steps)
        )
        return StopEvent(result={"success": success, "reason": reason, "steps": steps})


def _stop_subject(decision: JevDecision, screen: JevScreen) -> str:
    # Stop reasons leave the device; the value to type stays out of them.
    if decision.operation == "TYPE_TEXT":
        field = screen.focused
        return f"TYPE_TEXT {describe_field(field)}" if field else "TYPE_TEXT"
    if decision.label:
        return f"{decision.operation} {decision.label}"
    return decision.operation


def _tapped_input(screen: JevScreen, index: Any) -> JevElement | None:
    return next((e for e in screen.elements if e.index == index and e.editable), None)


def _history_label(decision: JevDecision, screen: JevScreen) -> str:
    field = screen.focused
    if decision.operation == "TYPE_TEXT" and field is not None:
        return f"[{field.index}] {field.hint or field.label or 'text field'}"
    if decision.label:
        return decision.label
    return _OPERATION_LABELS.get(decision.operation, decision.operation)
