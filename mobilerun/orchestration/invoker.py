"""Agent invocation seam — the only orchestration module that touches MobileAgent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Protocol

from mobilerun import MobileAgent, ResultEvent
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.config_manager.loader import ConfigLoader

from mobilerun.orchestration.models import TaskRequest, TaskResult


class AgentInvoker(Protocol):
    async def run_task(
        self,
        request: TaskRequest,
        event_callback: Callable[[Any], None] | None = None,
    ) -> TaskResult: ...


class MobileAgentInvoker:
    """Default AgentInvoker wrapping the existing MobileAgent runtime."""

    def __init__(self, config: MobileConfig | None = None) -> None:
        self._config = config

    async def run_task(
        self,
        request: TaskRequest,
        event_callback: Callable[[Any], None] | None = None,
    ) -> TaskResult:
        started_at = datetime.now()
        try:
            config = self._config or ConfigLoader.load(request.config_path)
            if request.device_serial:
                config.device.serial = request.device_serial

            llms = None
            if request.llm_provider:
                llms = load_llm(request.llm_provider, model=request.llm_model)

            agent = MobileAgent(
                goal=request.goal,
                config=config,
                llms=llms,
                timeout=request.timeout,
            )
            handler = agent.run()

            async for event in handler.stream_events():
                if event_callback:
                    event_callback(event)

            result: ResultEvent = await handler
            return TaskResult(
                task_id=request.id,
                success=result.success,
                reason=result.reason,
                steps=result.steps,
                structured_output=result.structured_output,
                started_at=started_at,
                finished_at=datetime.now(),
            )
        except Exception as exc:
            return TaskResult(
                task_id=request.id,
                success=False,
                reason="agent raised",
                steps=0,
                error=str(exc),
                started_at=started_at,
                finished_at=datetime.now(),
            )
