import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from mobilerun.agent.selectors.base import (
    ActionSelection,
    ActionSelector,
    BaseActionSelector,
    SelectionRequest,
    all_locate_points,
    sum_usage,
    supports_native_tool_calls,
)

if TYPE_CHECKING:
    from llama_index.core.llms.llm import LLM

logger = logging.getLogger("mobilerun")


def _usable(selection: ActionSelection | None) -> bool:
    """An error placeholder (executor 'invalid', failure, parse error) is not usable."""
    return (
        selection is not None
        and bool(selection.actions)
        and selection.failure is None
        and selection.parse_error is None
        and any(action.name != "invalid" for action in selection.actions)
    )


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
    def supports_native_tool_calls(self) -> bool:
        return supports_native_tool_calls((self.primary, *self.fallbacks))

    @property
    def requires_action_candidates(self) -> bool:
        return any(
            selector.requires_action_candidates
            for selector in (self.primary, *self.fallbacks)
        )

    @property
    def locates_points(self) -> bool:
        return all_locate_points((self.primary, *self.fallbacks))

    @property
    def screenshot_max_side(self) -> int | None:
        limits = [
            selector.screenshot_max_side for selector in (self.primary, *self.fallbacks)
        ]
        return (
            min(limit for limit in limits if limit is not None)
            if any(limit is not None for limit in limits)
            else None
        )

    @property
    def llms(self) -> tuple["LLM", ...]:
        return tuple(
            llm for selector in (self.primary, *self.fallbacks) for llm in selector.llms
        )

    async def select(self, request: SelectionRequest) -> ActionSelection | None:
        selectors = (self.primary, *self.fallbacks)
        # Unusable attempts still made model calls; report their usage too.
        spent = []
        for selector in selectors[:-1]:
            try:
                selection = await selector.select(request)
                if _usable(selection):
                    return _with_usage(selection, spent)
                if selection is not None:
                    spent.append(selection.usage)
            except self.errors as e:
                logger.warning(
                    "%s failed, falling back: %s", type(selector).__name__, e
                )
        return _with_usage(await selectors[-1].select(request), spent)


def _with_usage(
    selection: ActionSelection | None, spent: list
) -> ActionSelection | None:
    if selection is None or not any(usage is not None for usage in spent):
        return selection
    return replace(selection, usage=sum_usage(*spent, selection.usage))
