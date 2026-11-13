"""Event types and logging bridge for the orchestration runner.

This module intentionally does not import MobileAgent. The invoker is the
only orchestration module that should touch the real MobileAgent runtime.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from mobilerun.orchestration.models import TaskRequest, TaskResult

logger = logging.getLogger("mobilerun.orchestration")


class OrchestrationEventType(str, Enum):
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    AGENT_PROGRESS = "agent_progress"
    AGENT_ACTION = "agent_action"
    AGENT_RESULT = "agent_result"
    AGENT_ERROR = "agent_error"
    AGENT_EVENT = "agent_event"


@dataclass(frozen=True)
class OrchestrationEvent:
    """Normalized event emitted by the orchestration layer."""

    type: OrchestrationEventType
    task_id: str | None = None
    message: str | None = None
    raw_event_type: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


EventCallback = Callable[[OrchestrationEvent], None]
RawAgentEventCallback = Callable[[Any], None]


def task_started_event(request: TaskRequest) -> OrchestrationEvent:
    return OrchestrationEvent(
        type=OrchestrationEventType.TASK_STARTED,
        task_id=request.id,
        message=f"Task started: {request.goal}",
        payload={
            "goal": request.goal,
            "priority": request.priority,
            "device_serial": request.device_serial,
            "llm_provider": request.llm_provider,
            "llm_model": request.llm_model,
            "timeout": request.timeout,
            "metadata": request.metadata,
        },
    )


def task_completed_event(result: TaskResult) -> OrchestrationEvent:
    return OrchestrationEvent(
        type=OrchestrationEventType.TASK_COMPLETED,
        task_id=result.task_id,
        message=result.reason,
        payload={
            "success": result.success,
            "reason": result.reason,
            "steps": result.steps,
            "structured_output": _safe_value(result.structured_output),
            "started_at": result.started_at,
            "finished_at": result.finished_at,
        },
    )


def task_failed_event(result: TaskResult) -> OrchestrationEvent:
    return OrchestrationEvent(
        type=OrchestrationEventType.TASK_FAILED,
        task_id=result.task_id,
        message=result.error or result.reason,
        payload={
            "success": result.success,
            "reason": result.reason,
            "steps": result.steps,
            "error": result.error,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
        },
    )


def raw_agent_event_to_orchestration_event(
    raw_event: Any,
    task_id: str | None = None,
) -> OrchestrationEvent:
    """Convert a MobileRun runtime event into a normalized orchestration event."""

    raw_event_type = raw_event.__class__.__name__
    payload = _payload_from_raw_event(raw_event)

    event_type = _classify_raw_event(raw_event_type, payload)
    message = _message_from_payload(raw_event_type, payload)

    return OrchestrationEvent(
        type=event_type,
        task_id=task_id,
        message=message,
        raw_event_type=raw_event_type,
        payload=payload,
    )


def emit_event(
    event_callback: EventCallback | None,
    event: OrchestrationEvent,
    *,
    log: bool = True,
) -> None:
    """Log and forward an orchestration event without crashing the runner."""

    if log:
        log_event(event)

    if event_callback is None:
        return

    try:
        event_callback(event)
    except Exception:
        logger.exception(
            "orchestration event callback failed for task %s",
            event.task_id,
        )


def log_event(event: OrchestrationEvent) -> None:
    """Default logging bridge for orchestration-level events."""

    extra = {
        "task_id": event.task_id,
        "event_type": event.type.value,
        "raw_event_type": event.raw_event_type,
    }

    message = event.message or event.type.value

    if event.type in {
        OrchestrationEventType.TASK_FAILED,
        OrchestrationEventType.AGENT_ERROR,
    }:
        logger.warning(message, extra=extra)
    elif event.type == OrchestrationEventType.TASK_COMPLETED:
        logger.info(message, extra=extra)
    else:
        logger.debug(message, extra=extra)


class LoggingEventBridge:
    """Bridge raw MobileRun agent events to orchestration events and logging."""

    def __init__(
        self,
        task_id: str | None = None,
        event_callback: EventCallback | None = None,
        *,
        use_cli_handler: bool = True,
    ) -> None:
        self._task_id = task_id
        self._event_callback = event_callback
        self._cli_handler = _load_cli_event_handler() if use_cli_handler else None

    def handle_agent_event(self, raw_event: Any) -> None:
        """Handle one raw event from MobileAgent.handler.stream_events()."""

        if self._cli_handler is not None:
            try:
                self._cli_handler.handle(raw_event)
            except Exception:
                logger.exception(
                    "CLI event handler failed for event %s",
                    raw_event.__class__.__name__,
                )

        event = raw_agent_event_to_orchestration_event(
            raw_event,
            task_id=self._task_id,
        )
        emit_event(self._event_callback, event)


def make_agent_event_callback(
    task_id: str | None = None,
    event_callback: EventCallback | None = None,
    *,
    use_cli_handler: bool = True,
) -> RawAgentEventCallback:
    """Create the callback that TaskRunner passes into AgentInvoker.run_task()."""

    bridge = LoggingEventBridge(
        task_id=task_id,
        event_callback=event_callback,
        use_cli_handler=use_cli_handler,
    )
    return bridge.handle_agent_event


def _load_cli_event_handler() -> Any | None:
    """Load the existing CLI event handler lazily so this module stays optional."""

    try:
        from mobilerun.cli.event_handler import EventHandler
    except Exception:
        logger.debug("CLI EventHandler is unavailable", exc_info=True)
        return None

    return EventHandler()


def _classify_raw_event(
    raw_event_type: str,
    payload: Mapping[str, Any],
) -> OrchestrationEventType:
    success = payload.get("success")
    error = payload.get("error")

    if error or success is False:
        return OrchestrationEventType.AGENT_ERROR

    if any(token in raw_event_type for token in ("Action", "ToolCall", "Execute")):
        return OrchestrationEventType.AGENT_ACTION

    if any(token in raw_event_type for token in ("Result", "Output", "End", "Finalize")):
        return OrchestrationEventType.AGENT_RESULT

    if any(token in raw_event_type for token in ("Plan", "Response", "Context", "Input")):
        return OrchestrationEventType.AGENT_PROGRESS

    return OrchestrationEventType.AGENT_EVENT


def _message_from_payload(
    raw_event_type: str,
    payload: Mapping[str, Any],
) -> str:
    for key in (
        "summary",
        "reason",
        "description",
        "subgoal",
        "answer",
        "plan",
        "thought",
        "error",
    ):
        value = payload.get(key)
        if value:
            return _truncate(str(value), 200)

    return raw_event_type


def _payload_from_raw_event(raw_event: Any) -> dict[str, Any]:
    if raw_event is None:
        return {}

    if isinstance(raw_event, Mapping):
        return {
            str(key): _safe_value(value)
            for key, value in raw_event.items()
            if not str(key).startswith("_")
        }

    if is_dataclass(raw_event):
        return {
            str(key): _safe_value(value)
            for key, value in asdict(raw_event).items()
            if not str(key).startswith("_")
        }

    if hasattr(raw_event, "model_dump"):
        try:
            dumped = raw_event.model_dump()
            if isinstance(dumped, Mapping):
                return {
                    str(key): _safe_value(value)
                    for key, value in dumped.items()
                    if not str(key).startswith("_")
                }
        except Exception:
            logger.debug("model_dump failed for %s", raw_event.__class__.__name__)

    if hasattr(raw_event, "__dict__"):
        return {
            str(key): _safe_value(value)
            for key, value in vars(raw_event).items()
            if not str(key).startswith("_")
        }

    return {"repr": _truncate(repr(raw_event), 500)}


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return _truncate(value, 500) if isinstance(value, str) else value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }

    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]

    return _truncate(repr(value), 500)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."