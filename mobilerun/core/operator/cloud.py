"""Cloud task operator and device pairing validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from mobilerun.models.results import ExecutionResult

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.devices.cloud import CloudDevice


from mobilerun.clients.cloud.tasks import run_task
from mobilerun.config.settings import get_settings
from mobilerun.core.operator import Operator
from mobilerun.errors import OperatorNotSupported
from mobilerun.models.billing import Billing

T = TypeVar("T", bound=BaseModel)


class CloudTaskOperator(Operator):
    def __init__(self, *, billing: Billing | str | None = None) -> None:
        self._billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )

    def check(self, device: Device) -> None:
        if not device._is_cloud:
            kind = type(device).__name__
            inner = device._unwrap()
            if inner is not device:
                kind = f"{kind}({type(inner).__name__})"
            raise OperatorNotSupported(
                f"{kind} ({device.identifier or 'unidentified'}) is not a cloud device. Use an operator from mobilerun-agents for local LLM tasks."
            )
        local = [
            extension.name
            for extension in device.extensions.values()
            if not extension.cloud_native
        ]
        if local:
            raise OperatorNotSupported(
                f"This device has local extensions ({', '.join(local)}) that cloud tasks can't use; use an operator from mobilerun-agents or unbind them"
            )

    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        self.check(device)
        references: dict[str, set[str]] = {}
        extensions = tuple(device.extensions.values())
        for extension in extensions:
            if extension.cloud_native:
                inputs = await extension.cloud_task_inputs()
                for reference in inputs.credential_references:
                    references.setdefault(reference.package, set()).add(reference.name)
        await device._ensure_connected()
        cloud = cast("CloudDevice", device._unwrap())
        params: dict[str, Any] = {
            "device_id": cloud.device_id,
            "task": instruction,
            "output": output,
            "max_steps": max_steps,
        }
        if timeout is not None:
            params["timeout"] = timeout
        if references:
            params["credentials"] = [
                {"package_name": package, "credential_names": sorted(names)}
                for package, names in references.items()
            ]
        if cloud._task_started_callback is not None:
            params["on_task_started"] = cloud._task_started_callback
        outcome = await run_task(cloud._cloud_client, billing=self._billing, **params)
        outcome.device = type(device).__name__
        return outcome
