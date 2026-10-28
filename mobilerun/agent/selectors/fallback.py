from collections.abc import Sequence
from typing import TYPE_CHECKING

from mobilerun.agent.selectors.base import (
    ActionSelection,
    ActionSelector,
    BaseActionSelector,
    SelectionRequest,
)

if TYPE_CHECKING:
    from llama_index.core.llms.llm import LLM


class FallbackSelector(BaseActionSelector):
    def __init__(
        self,
        primary: ActionSelector,
        fallbacks: Sequence[ActionSelector],
        errors: tuple[type[Exception], ...] = (Exception,),
    ):
        self.primary = primary
        self.fallbacks = tuple(fallbacks)
        self.errors = errors

    @property
    def requires_action_candidates(self) -> bool:
        return any(selector.requires_action_candidates
                   for selector in (self.primary, *self.fallbacks))

    @property
    def screenshot_max_side(self) -> int | None:
        limits = [selector.screenshot_max_side
                  for selector in (self.primary, *self.fallbacks)]
        return min(limit for limit in limits if limit is not None) if any(
            limit is not None for limit in limits
        ) else None

    @property
    def llms(self) -> tuple["LLM", ...]:
        return tuple(
            llm
            for selector in (self.primary, *self.fallbacks)
            for llm in selector.llms
        )

    async def select(self, request: SelectionRequest) -> ActionSelection | None:
        selectors = (self.primary, *self.fallbacks)
        for selector in selectors[:-1]:
            try:
                selection = await selector.select(request)
                if selection is not None and selection.actions:
                    return selection
            except self.errors:
                pass
        return await selectors[-1].select(request)
