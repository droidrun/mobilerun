from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from mobilerun.agent.droid.events import ResultEvent
from mobilerun.agent.selectors import ActionSelection, BaseActionSelector, SelectionRequest

if TYPE_CHECKING:
    from llama_index.core.llms.llm import LLM


class CompletedRun:
    """Successful workflow handle for CLI tests that must not contact a device."""

    async def stream_events(self):
        for event in ():
            yield event

    def __await__(self):
        async def result():
            return ResultEvent(success=True, reason="Test completed", steps=0)

        return result().__await__()


class StubSelector(BaseActionSelector):
    """Configurable selector double implementing the explicit capability contract."""

    def __init__(
        self,
        *,
        select: AsyncMock | None = None,
        requires_action_llm: bool = True,
        requires_action_candidates: bool = False,
        shared_llm: LLM | None = None,
        llms: tuple[LLM, ...] = (),
        screenshot_max_side: int | None = None,
    ) -> None:
        self.select = select if select is not None else AsyncMock()
        self.requires_action_llm = requires_action_llm
        self.requires_action_candidates = requires_action_candidates
        self._shared_llm = shared_llm
        self._llms = llms
        self._screenshot_max_side = screenshot_max_side

    @property
    def shared_llm(self) -> LLM | None:
        return self._shared_llm

    @property
    def llms(self) -> tuple[LLM, ...]:
        return self._llms

    @property
    def screenshot_max_side(self) -> int | None:
        return self._screenshot_max_side

    async def select(self, request: SelectionRequest) -> ActionSelection | None:
        raise AssertionError("The instance's select mock should be used")
