import logging
from typing import Any, Callable

from mobilerun.orchestration.invoker import AgentInvoker
from mobilerun.orchestration.models import TaskRequest, TaskResult
from mobilerun.orchestration.queue import TaskQueue

logger = logging.getLogger("mobilerun.orchestration")

EventCallback = Callable[[Any], None]


class TaskRunner:
    """Drives TaskQueue -> AgentInvoker sequentially, one task at a time."""

    def __init__(
        self,
        queue: TaskQueue,
        invoker: AgentInvoker,
        event_callback: EventCallback | None = None,
    ) -> None:
        self._queue = queue
        self._invoker = invoker
        self._event_callback = event_callback
        self._stop_requested = False

    async def run_once(self) -> TaskResult:
        request: TaskRequest = await self._queue.dequeue()
        self._queue.mark_running(request.id)
        try:
            result = await self._invoker.run_task(request, event_callback=self._event_callback)
        except Exception as exc:
            # Invoker exceptions must never kill the loop. Note this except only wraps _invoker.run_task
            logger.exception("invoker.run_task raised for task %s", request.id)
            result = TaskResult(
                task_id=request.id,
                success=False,
                reason=f"invoker raised {type(exc).__name__}: {exc}",
                steps=0,
                error=str(exc) or type(exc).__name__,
            )


        if result.success:
            self._queue.mark_completed(request.id, result)
        else:
            self._queue.mark_failed(request.id, result)
        return result

    async def run_forever(self) -> None:
        """Runs run_once in a loop until stop() is called between tasks. stop() only takes effect between tasks
        (graceful drain of whatever is currently running); it will not interrupt an idle dequeue() that is blocked
        waiting for the next ready task.
        """
        self._stop_requested = False
        while not self._stop_requested:
            await self.run_once()

    def stop(self) -> None:
        self._stop_requested = True
