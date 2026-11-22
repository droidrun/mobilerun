"""Strict macro execution on caller-owned devices."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar, Unpack

from pydantic import BaseModel

from mobilerun.core.operator import Operator
from mobilerun.macro.player import MacroPlayer
from mobilerun.models.results import ExecutionResult

if TYPE_CHECKING:
    from mobilerun.core.devices import Device

T = TypeVar("T", bound=BaseModel)


class PlayerOptions(TypedDict, total=False):
    delay_between_actions: float
    state_timeout: float
    state_threshold: float
    state_poll_interval: float


class MacroOperator(Operator):
    def __init__(
        self,
        macro_file: str | Path | None = None,
        *,
        start_from_step: int = 0,
        **player_options: Unpack[PlayerOptions],
    ) -> None:
        self.macro_file = macro_file
        self.start_from_step = start_from_step
        self.player_options = player_options

    async def execute(
        self,
        device: "Device",
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        if output is not None:
            raise ValueError("Strict macro replay does not produce structured output")
        player = MacroPlayer(device, **self.player_options)
        data = player.load_macro_from_file(self.macro_file or instruction)
        async with asyncio.timeout(timeout):
            return await self._execute_macro(player, device, data, max_steps=max_steps)

    async def _execute_macro(
        self,
        player: MacroPlayer,
        device: "Device",
        data: dict[str, Any],
        *,
        max_steps: int | None = None,
    ) -> ExecutionResult[Any]:
        success = await player.replay_macro(
            data, start_from_step=self.start_from_step, max_steps=max_steps
        )
        return ExecutionResult(
            success=success,
            answer="Macro replay completed" if success else "Macro action failed",
            steps=player.executed_steps,
            device=type(device).__name__,
        )
