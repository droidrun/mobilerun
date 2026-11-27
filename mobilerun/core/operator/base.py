"""Base interface for device operators."""

from abc import ABC
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from mobilerun.errors import OperatorNotAvailable
from mobilerun.models.results import ExecutionResult

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device

T = TypeVar("T", bound=BaseModel)


class Operator(ABC):  # noqa: B024 - the base implementation deliberately raises
    def check(self, device: "Device") -> None:  # noqa: B027
        """Validate a pairing before use."""

    async def execute(
        self,
        device: "Device",
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        raise OperatorNotAvailable(
            f"No operator for {type(device).__name__}. Pass operator=..., "
            "e.g. AgentOperator() from the mobilerun-agents add-on "
            "(from mobilerun_agents import AgentOperator)."
        )
