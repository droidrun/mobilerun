"""Functional cloud task SDK boundaries, including streaming and polling."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Literal, Required, TypedDict, TypeVar, Unpack

import httpx
from mobilerun_sdk import APIError, AsyncMobilerun
from mobilerun_sdk._types import Body, Headers, NotGiven, Query, SequenceNotStr
from mobilerun_sdk.types import (
    TaskGetStatusResponse,
    TaskGetTrajectoryResponse,
    TaskRunResponse,
    TaskStopResponse,
)
from mobilerun_sdk.types.task_run_params import Credential
from mobilerun_sdk.types.tasks import ScreenshotListResponse
from pydantic import BaseModel, ValidationError

from mobilerun.clients.cloud.billing import requires
from mobilerun.config.settings import get_settings
from mobilerun.errors import CloudTaskTimeout, CloudTimeout
from mobilerun.logs.actions import action_line, cloud_progress
from mobilerun.logs.actions import logger as action_logger
from mobilerun.models.billing import Billing
from mobilerun.models.results import ExecutionResult
from mobilerun.clients.cloud.errors import cloud_error

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)
TaskRunOptions = TypedDict(
    "TaskRunOptions",
    {
        "device_id": Required[str],
        "task": Required[str],
        "accessibility": bool,
        "apps": SequenceNotStr[str],
        "continue_on_failure": bool,
        "credentials": Iterable[Credential],
        "display_id": int,
        "execution_timeout": int,
        "files": SequenceNotStr[str],
        "humanize": bool,
        "internal": bool,
        "llm_model": str,
        "max_steps": int,
        "output_schema": dict[str, object] | None,
        "reasoning": bool,
        "recording_enabled": bool,
        "stealth": bool,
        "subagent_model": str,
        "system_prompt": str | None,
        "temperature": float,
        "vision": bool,
        "vpn_country": Literal["US", "BR", "FR", "DE", "IN", "JP", "KR", "ZA"] | None,
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
        "idempotency_key": str | None,
    },
    total=False,
)
RequestOptions = TypedDict(
    "RequestOptions",
    {
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
    },
    total=False,
)


@requires(Billing.INCLUDED)
async def run(
    client: AsyncMobilerun, *, billing: Billing, **params: Unpack[TaskRunOptions]
) -> TaskRunResponse:
    """Run a task on an existing phone."""
    try:
        return await client.tasks.run(**params)
    except APIError as exc:
        body = exc.body
        detail = body.get("detail", "") if isinstance(body, dict) else ""
        device = (
            isinstance(detail, str)
            and "device" in detail.lower()
            and "not found" in detail.lower()
        )
        raise cloud_error(exc, "Run cloud task", device=device) from exc


@requires(Billing.NO_COMMERCE)
async def progress(
    client: AsyncMobilerun, task_id: str, *, billing: Billing
) -> AsyncIterator[str]:
    """Yield SSE lines with transport failures mapped."""
    try:
        async with client.tasks.with_streaming_response.attach(
            task_id, timeout=None
        ) as stream:
            async for line in stream.iter_lines():
                yield line
    except (APIError, httpx.HTTPError) as exc:
        raise cloud_error(exc, "Stream cloud task progress") from exc


@requires(Billing.NO_COMMERCE)
async def status(
    client: AsyncMobilerun, task_id: str, *, billing: Billing
) -> TaskGetStatusResponse:
    """Poll a cloud task's status."""
    try:
        return await client.tasks.get_status(task_id)
    except APIError as exc:
        raise cloud_error(exc, "Poll cloud task") from exc


@requires(Billing.NO_COMMERCE)
async def stop(
    client: AsyncMobilerun, task_id: str, *, billing: Billing
) -> TaskStopResponse:
    """Cancel a running cloud task."""
    try:
        return await client.tasks.stop(task_id)
    except APIError as exc:
        raise cloud_error(exc, "Stop cloud task") from exc


@requires(Billing.NO_COMMERCE)
async def trajectory(
    client: AsyncMobilerun, task_id: str, *, billing: Billing
) -> TaskGetTrajectoryResponse:
    """Retrieve a task trajectory."""
    try:
        return await client.tasks.get_trajectory(task_id)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve cloud task trajectory") from exc


@requires(Billing.NO_COMMERCE)
async def screenshots(
    client: AsyncMobilerun,
    task_id: str,
    *,
    billing: Billing,
    **params: Unpack[RequestOptions],
) -> ScreenshotListResponse:
    """Retrieve task screenshots."""
    try:
        return await client.tasks.screenshots.list(task_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve cloud task screenshots") from exc


async def _progress(
    client: AsyncMobilerun,
    task_id: str,
    device_id: str | None = None,
    *,
    billing: Billing,
) -> None:
    step_count = 0
    # SDK 6 attach() parses to None; its streaming response exposes SSE lines.
    async for line in progress(client, task_id, billing=billing):
        if line.startswith("data:"):
            parsed = cloud_progress(line[5:].strip())
            if parsed:
                action, step = parsed
                step_count += 1
                action_logger.info(
                    "%s",
                    action_line(
                        str(device_id)[:8] if device_id else "unknown",
                        "task",
                        step or step_count,
                        action,
                    ),
                )


async def _stop(client: AsyncMobilerun, task_id: str, *, billing: Billing) -> None:
    if task_id == "unknown":
        return
    try:
        async with asyncio.timeout(5):
            await stop(client, task_id, billing=billing)
    except Exception:
        # Best effort: cleanup must not mask the original timeout/cancellation.
        logger.warning("Could not stop cloud task %s", task_id)


@requires(Billing.INCLUDED)
async def run_task(
    client: AsyncMobilerun,
    *,
    device_id: str,
    task: str,
    billing: Billing,
    output: type[T] | None = None,
    max_steps: int | None = None,
    timeout: float | None = None,
    poll_interval: float | None = None,
    on_task_started: Callable[[str], None] | None = None,
    credentials: Iterable[Credential] | None = None,
) -> ExecutionResult:
    """Run a cloud task and return pure result data, including its task_id."""
    settings = get_settings()
    timeout = (
        settings.MOBILERUN_CLOUD_TASK_TIMEOUT_SECONDS if timeout is None else timeout
    )
    poll_interval = (
        settings.MOBILERUN_CLOUD_TASK_POLL_INTERVAL_SECONDS
        if poll_interval is None
        else poll_interval
    )
    params: TaskRunOptions = {"device_id": device_id, "task": task}
    if output is not None:
        params["output_schema"] = output.model_json_schema()
    if max_steps is not None:
        params["max_steps"] = max_steps
    if credentials is not None:
        params["credentials"] = credentials
    task_id = "unknown"
    try:
        async with asyncio.timeout(timeout):
            response = await run(client, billing=billing, **params)
            task_id = response.id
            if on_task_started is not None:
                on_task_started(task_id)
            await _progress(client, task_id, device_id, billing=billing)
            while True:
                state = await status(client, task_id, billing=billing)
                if state.status in {"completed", "failed", "cancelled"}:
                    success = bool(state.succeeded)
                    answer = state.message
                    parsed_output = None
                    if output is not None and state.output is not None:
                        try:
                            parsed_output = output.model_validate(state.output)
                        except ValidationError as exc:
                            success = False
                            answer = str(exc)
                    result_type = (
                        ExecutionResult[output]
                        if output is not None
                        else ExecutionResult
                    )
                    return result_type(
                        success=success,
                        answer=answer,
                        output=parsed_output,
                        steps=state.steps,
                        task_id=task_id,
                        device="CloudDevice",
                    )
                await asyncio.sleep(poll_interval)
    except CloudTimeout:
        await _stop(client, task_id, billing=billing)
        raise
    except TimeoutError as exc:
        await _stop(client, task_id, billing=billing)
        raise CloudTaskTimeout(
            operation="Run cloud task",
            code="task_timeout",
            resource_id=task_id,
            timeout=timeout,
        ) from exc
    except asyncio.CancelledError:
        await _stop(client, task_id, billing=billing)
        raise
    except Exception:
        await _stop(client, task_id, billing=billing)
        raise
