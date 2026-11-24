"""Cloud declarations and lazy synchronous handles for deployed functions."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable, Coroutine, Mapping
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from pydantic import AliasChoices, AliasPath, BaseModel, TypeAdapter

from mobilerun.models.billing import Billing

from . import deployment as _operations
from .errors import SandboxDeclarationError, SandboxDeploymentError, SandboxRunError
from .models import (
    CloudDeclaration,
    DeploymentPreview,
    DeploymentReceipt,
    Endpoint,
    resolve_billing,
)
from .settings import get_sandbox_settings

JSON = dict[str, Any]
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


def _sync(operation: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Reject a running loop before constructing the coroutine."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "The synchronous sandbox API cannot run inside an active event loop"
        )
    return asyncio.run(operation())


def _write_alias(container: Any, path: list[str | int], value: Any) -> None:
    key, *rest = path
    if isinstance(key, str) and isinstance(container, dict):
        pass
    elif isinstance(key, int) and key >= 0 and isinstance(container, list):
        if key >= len(container):
            container.extend([None] * (key + 1 - len(container)))
    else:
        raise ValueError("Unsupported or conflicting alias path")
    if not rest:
        container[key] = value
        return
    if isinstance(container, dict):
        child = container.get(key)
    else:
        child = container[key]
    if child is None:
        child = {} if isinstance(rest[0], str) else []
        container[key] = child
    _write_alias(child, rest, value)


def _validation_keys(value: Any, dumped: Any) -> Any:
    """Shape dumped model fields according to their validation aliases."""
    if isinstance(value, BaseModel) and isinstance(dumped, dict):
        fields = type(value).model_fields
        result = {key: entry for key, entry in dumped.items() if key not in fields}
        for name, field in fields.items():
            if name not in dumped:
                continue
            alias = field.validation_alias
            if isinstance(alias, AliasChoices):
                alias = alias.choices[0]
            path = (
                alias.path
                if isinstance(alias, AliasPath)
                else [name if alias is None else alias]
            )
            _write_alias(
                result, path, _validation_keys(getattr(value, name), dumped[name])
            )
        return result
    if (
        isinstance(value, (list, tuple))
        and isinstance(dumped, list)
        and len(value) == len(dumped)
    ):
        return [
            _validation_keys(child, entry)
            for child, entry in zip(value, dumped, strict=True)
        ]
    if isinstance(value, dict) and isinstance(dumped, dict):
        return {
            key: _validation_keys(value.get(key), entry)
            for key, entry in dumped.items()
        }
    return dumped


def _validation_payload(value: BaseModel) -> JSON:
    message = "Sandbox input could not round-trip through its model's JSON validation"
    try:
        payload = _validation_keys(
            value, value.model_dump(mode="json", by_alias=False, round_trip=True)
        )
        if isinstance(payload, dict):
            # Match the worker's JSON validation, including strict model types.
            rendered = json.dumps(payload, allow_nan=False)
            if type(value).model_validate_json(rendered) == value:
                return payload
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise SandboxDeclarationError(message) from None
    raise SandboxDeclarationError(message)


class Remote:
    """A named function handle; lookup itself does not contact the cloud."""

    def __init__(
        self,
        name: str,
        input_type: type[BaseModel] | None = None,
        output_type: type[BaseModel] | None = None,
        *,
        billing: Billing | str | None = None,
    ) -> None:
        self.name = name
        self.input_type = input_type
        self.output_type = output_type
        self._billing = resolve_billing(billing)

    def _mode(self, billing: Billing | str | None) -> Billing:
        return self._billing if billing is None else resolve_billing(billing)

    @property
    def _description(self) -> JSON:
        return _sync(
            partial(_operations.describe_remote, self.name, billing=self._billing)
        )

    @property
    def function_id(self) -> str:
        return self._description["function_id"]

    @property
    def version(self) -> int:
        return self._description["version"]

    @property
    def flow_id(self) -> str:
        flow_id = self._description.get("flow_id")
        if flow_id is None:
            raise SandboxDeploymentError(
                "Sandbox flow not found", missing_deployment=True
            )
        return flow_id

    @property
    def endpoint(self) -> Endpoint:
        endpoint = self._description.get("endpoint")
        if endpoint is None:
            raise SandboxDeploymentError(
                "Sandbox endpoint not found", missing_deployment=True
            )
        return Endpoint(**endpoint)

    def fire(
        self,
        input: Mapping[str, Any] | BaseModel | None = None,
        *,
        billing: Billing | str | None = None,
        **values: Any,
    ) -> Run:
        """Enqueue a run without waiting for execution to start."""
        if input is not None and values:
            raise TypeError("Pass either input= or keyword payload values, not both")
        if input is None:
            data: Mapping[str, Any] | BaseModel = values
        elif not isinstance(input, (BaseModel, Mapping)):
            raise TypeError("input must be a mapping or a pydantic BaseModel")
        else:
            data = input
        if self.input_type is not None and not isinstance(data, self.input_type):
            if isinstance(data, BaseModel):
                data = _validation_payload(data)
            data = self.input_type.model_validate(data)
        payload = (
            _validation_payload(data) if isinstance(data, BaseModel) else dict(data)
        )
        mode = self._mode(billing)
        fired = _sync(
            partial(_operations.fire_remote, self.name, payload, billing=mode)
        )
        run = Run(self, invocation_id=fired["invocation_id"])
        run._billing = mode
        return run

    def __call__(
        self,
        input: Mapping[str, Any] | BaseModel | None = None,
        *,
        billing: Billing | str | None = None,
        **values: Any,
    ) -> Any:
        return self.fire(input=input, billing=billing, **values).result()

    def runs(
        self, limit: int = 10, *, billing: Billing | str | None = None
    ) -> list[Run]:
        """Read recent executions, newest first."""
        if limit <= 0:
            return []
        mode = self._mode(billing)
        rows = _sync(partial(_operations.recent_runs, self.name, limit, billing=mode))
        runs = []
        for row in rows:
            run = Run(self, execution_id=row["id"], snapshot=row)
            run._billing = mode
            runs.append(run)
        return runs

    def enable(self, *, billing: Billing | str | None = None) -> JSON:
        """Enable this owned flow."""
        return _sync(
            partial(
                _operations.set_enabled, self.name, True, billing=self._mode(billing)
            )
        )

    def disable(self, *, billing: Billing | str | None = None) -> JSON:
        """Pause this owned flow until enabled again."""
        return _sync(
            partial(
                _operations.set_enabled, self.name, False, billing=self._mode(billing)
            )
        )


class Run:
    """A run keyed by invocation or execution ID."""

    def __init__(
        self,
        remote: Remote,
        *,
        invocation_id: str | None = None,
        execution_id: str | None = None,
        snapshot: JSON | None = None,
    ) -> None:
        self.remote = remote
        self._snapshot = deepcopy(snapshot) if snapshot is not None else {}
        self.invocation_id = invocation_id or self._snapshot.get("invocationId")
        self._id = execution_id or self._snapshot.get("id")
        self._billing = remote._billing

    @property
    def id(self) -> str:
        if self._id is None:
            if self.invocation_id is None:
                raise SandboxRunError("Provide an invocation or execution ID")
            self._id = _sync(
                partial(
                    _operations.resolve_execution,
                    self.remote.name,
                    self.invocation_id,
                    deadline=time.monotonic()
                    + get_sandbox_settings().MOBILERUN_SANDBOX_RUN_TIMEOUT,
                    billing=self._billing,
                )
            )
        return self._id

    @property
    def status(self) -> str:
        self._snapshot = _sync(
            partial(_operations.execution_snapshot, self.id, billing=self._billing)
        )
        return self._snapshot["status"]

    @property
    def started_at(self) -> str | None:
        return self._snapshot.get("startedAt")

    @property
    def ended_at(self) -> str | None:
        return self._snapshot.get("finishedAt")

    @property
    def duration(self) -> float | None:
        start, end = self.started_at, self.ended_at
        if not start or not end:
            return None
        return (
            datetime.fromisoformat(end) - datetime.fromisoformat(start)
        ).total_seconds()

    @property
    def snapshot(self) -> JSON:
        return deepcopy(self._snapshot)

    def result(self, timeout: float | None = None) -> Any:
        """Wait for completion and validate the function result when typed."""
        if timeout is None:
            timeout = get_sandbox_settings().MOBILERUN_SANDBOX_RUN_TIMEOUT
        self._snapshot = _sync(
            partial(
                _operations.wait_result,
                self.remote.name,
                invocation_id=self.invocation_id,
                execution_id=self._id,
                timeout=timeout,
                billing=self._billing,
            )
        )
        self._id = self._snapshot["id"]
        # The orchestration helper validates the execution and function steps.
        step = next(
            step
            for step in self._snapshot["result"]["steps"]
            if isinstance(step, dict) and step.get("uses") == "functions.run"
        )
        value = step["body"]["result"]
        output_type = self.remote.output_type
        return (
            TypeAdapter(output_type).validate_python(value)
            if output_type is not None
            else deepcopy(value)
        )

    def logs(self) -> str:
        """Return execution logs through the cloud boundary."""
        return _sync(
            partial(_operations.execution_logs, self.id, billing=self._billing)
        )


class Cloud:
    """Register local scripts or look up an untyped remote by deployment name."""

    def __call__(
        self,
        name: str,
        schedule: str | None = None,
        device: str | None = None,
        endpoint: bool | None = None,
    ) -> Callable[[F], F]:
        CloudDeclaration(
            name, schedule=schedule, device=device, endpoint=endpoint
        ).validate()

        def decorate(func: F) -> F:
            from ._runtime import EntryPointError, script

            try:
                script(func)
            except EntryPointError as exc:
                raise SandboxDeclarationError(str(exc)) from exc
            hints = get_type_hints(func)
            first = next(iter(inspect.signature(func).parameters))
            func.remote = Remote(name, hints[first], hints.get("return"))  # type: ignore[attr-defined]
            return func

        return decorate

    def __getitem__(self, name: str) -> Remote:
        return Remote(name)


cloud = Cloud()


def deploy(
    path: str | Path,
    *,
    device: str | None = None,
    device_free: bool = False,
    billing: Billing | str | None = None,
    yes: bool = False,
    confirm: Callable[[DeploymentPreview], bool] | None = None,
) -> DeploymentReceipt:
    """Explicitly deploy a script after confirmation."""
    return _sync(
        partial(
            _operations.deploy_file,
            Path(path),
            device=device,
            device_free=device_free,
            billing=resolve_billing(billing),
            yes=yes,
            confirm=confirm,
        )
    )
