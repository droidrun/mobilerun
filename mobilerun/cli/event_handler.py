"""
Shared event handler for CLI and SDK.

Translates workflow events into ``logging`` calls with ``extra`` params
(color, step_increment, etc.).  The actual rendering is handled by
whichever ``logging.Handler`` is attached (CLILogHandler, TUILogHandler, …).
"""

import logging

from mobilerun.agent.common.events import (
    RecordUIStateEvent,
    ScreenshotEvent,
)
from mobilerun.agent.droid.events import (
    ExecutorResultEvent,
    FastAgentExecuteEvent,
    FastAgentResultEvent,
    FinalizeEvent,
)
from mobilerun.agent.executor.events import (
    ExecutorActionEvent,
    ExecutorActionResultEvent,
)
from mobilerun.agent.fast_agent.events import (
    FastAgentEndEvent,
    FastAgentInputEvent,
    FastAgentOutputEvent,
    FastAgentResponseEvent,
    FastAgentToolCallEvent,
)
from mobilerun.agent.jev.events import JevDecisionEvent, JevEndEvent, JevStepEvent
from mobilerun.agent.manager.events import (
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)

logger = logging.getLogger("mobilerun")

_JEV_STATUS_COLORS = {"action": "cyan", "done": "green"}


def _preview(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def format_jev_decision(event: JevDecisionEvent) -> str:
    """One-line summary of a Jev decision."""
    target = f" {_preview(event.label, 80)}" if event.label else ""
    scores = f"confidence {event.confidence:.2f}"
    if event.target_confidence is not None:
        scores += f", target {event.target_confidence:.2f}"
    summary = f"{event.operation}{target} ({scores})"
    if event.status != "action":
        summary = f"{event.status}: {summary}"
    return f"🧭 Step {event.step}: {summary}"


class EventHandler:
    """Translates workflow events into logger calls.

    No UI state tracking — purely converts events into log records with
    ``extra`` params so that any attached handler can render them.
    """

    def handle(self, event) -> None:  # noqa: C901
        # ── Screenshots / UI state ──────────────────────────────────
        if isinstance(event, ScreenshotEvent):
            logger.debug("📸 Taking screenshot...")

        elif isinstance(event, RecordUIStateEvent):
            logger.debug("✏️ Recording UI state")

        # ── Manager events (reasoning mode) ─────────────────────────
        elif isinstance(event, ManagerContextEvent):
            logger.debug("🧠 Manager preparing context...")

        elif isinstance(event, ManagerResponseEvent):
            logger.debug("📥 Manager received LLM response")

        elif isinstance(event, ManagerPlanDetailsEvent):
            if event.thought:
                preview = (
                    event.thought[:120] + "..."
                    if len(event.thought) > 120
                    else event.thought
                )
                logger.debug(f"💭 Thought: {preview}", extra={"color": "cyan"})
            if event.subgoal:
                preview = (
                    event.subgoal[:150] + "..."
                    if len(event.subgoal) > 150
                    else event.subgoal
                )
                logger.debug(f"📋 Next step: {preview}", extra={"color": "yellow"})
            if event.answer:
                preview = (
                    event.answer[:200] + "..."
                    if len(event.answer) > 200
                    else event.answer
                )
                logger.debug(f"💬 Answer: {preview}", extra={"color": "green"})
            if event.plan:
                logger.debug(f"▸ {event.plan}", extra={"color": "yellow"})
            if event.memory_update:
                logger.debug(
                    f"🧠 Memory: {event.memory_update[:100]}...",
                    extra={"color": "cyan"},
                )

        # ── Executor events (reasoning mode) ────────────────────────
        elif isinstance(event, ExecutorActionEvent):
            if event.description:
                logger.debug(
                    f"🎯 Action: {event.description}", extra={"color": "yellow"}
                )
            if event.thought:
                preview = (
                    event.thought[:120] + "..."
                    if len(event.thought) > 120
                    else event.thought
                )
                logger.debug(f"💭 Reasoning: {preview}", extra={"color": "cyan"})

        elif isinstance(event, ExecutorActionResultEvent):
            if event.success:
                logger.debug(f"✅ {event.summary}", extra={"color": "green"})
            else:
                error_msg = event.error or "Unknown error"
                logger.debug(
                    f"❌ {event.summary} ({error_msg})", extra={"color": "red"}
                )

        elif isinstance(event, ExecutorResultEvent):
            logger.debug("Step complete", extra={"color": "magenta"})

        # ── FastAgent events (direct mode, XML tool-calling) ────────
        elif isinstance(event, FastAgentInputEvent):
            logger.debug("💬 Task input received...")

        elif isinstance(event, FastAgentResponseEvent):
            logger.debug("FastAgent response", extra={"color": "magenta"})
            if event.thought:
                preview = (
                    event.thought[:150] + "..."
                    if len(event.thought) > 150
                    else event.thought
                )
                logger.debug(f"🧠 Thinking: {preview}", extra={"color": "cyan"})
            if event.code:
                logger.debug("💻 Executing action code", extra={"color": "yellow"})
                logger.debug(f"{event.code}", extra={"color": "blue"})

        elif isinstance(event, FastAgentToolCallEvent):
            logger.debug("⚡ Executing tool calls...", extra={"color": "yellow"})

        elif isinstance(event, FastAgentOutputEvent):
            if event.output:
                output = str(event.output)
                preview = output[:100] + "..." if len(output) > 100 else output
                if "Error" in output or "Exception" in output:
                    logger.debug(f"❌ Action error: {preview}", extra={"color": "red"})
                else:
                    logger.debug(
                        f"⚡ Action result: {preview}", extra={"color": "green"}
                    )

        elif isinstance(event, FastAgentEndEvent):
            status = "done" if event.success else "failed"
            color = "green" if event.success else "red"
            logger.debug(
                f"■ {status}: {event.reason} ({event.tool_call_count} runs)",
                extra={"color": color},
            )

        # ── Jev events ──────────────────────────────────────────────
        elif isinstance(event, JevDecisionEvent):
            color = _JEV_STATUS_COLORS.get(event.status, "yellow")
            logger.info(format_jev_decision(event), extra={"color": color})
            logger.debug(
                f"Jev model {event.model or 'unknown'}, {event.latency_ms} ms"
                + (
                    f", {event.input_tokens} input tokens"
                    if event.input_tokens is not None
                    else ""
                )
            )

        elif isinstance(event, JevStepEvent):
            if event.success:
                logger.debug(f"✅ {event.summary}", extra={"color": "green"})
            else:
                logger.info(
                    f"❌ {event.operation} failed: {event.summary}",
                    extra={"color": "red"},
                )

        elif isinstance(event, JevEndEvent):
            status = "done" if event.success else "failed"
            color = "green" if event.success else "red"
            logger.debug(
                f"■ {status}: {event.reason} ({event.steps} steps)",
                extra={"color": color},
            )

        # ── Droid coordination events ───────────────────────────────
        elif isinstance(event, FastAgentExecuteEvent):
            logger.debug("🔧 Starting task execution...", extra={"color": "magenta"})

        elif isinstance(event, FastAgentResultEvent):
            if event.success:
                logger.debug(f"Task result: {event.reason}", extra={"color": "green"})
            else:
                logger.debug(f"Task failed: {event.reason}", extra={"color": "red"})

        elif isinstance(event, FinalizeEvent):
            if event.success:
                logger.info(
                    f"🎉 Goal achieved: {event.reason}", extra={"color": "green"}
                )
            else:
                logger.info(f"❌ Goal failed: {event.reason}", extra={"color": "red"})

        # ── Fallback ────────────────────────────────────────────────
        else:
            logger.debug(f"🔄 {event.__class__.__name__}")
