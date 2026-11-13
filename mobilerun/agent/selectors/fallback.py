import json
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


def _runnable(name: object, request: SelectionRequest) -> bool:
    return (
        isinstance(name, str)
        and name != "invalid"
        and (not request.tools or name in request.tools)
    )


def _usable(selection: ActionSelection | None, request: SelectionRequest) -> bool:
    """Only a selection with a call the registry can run is usable."""
    if (
        selection is None
        or selection.failure is not None
        or selection.parse_error is not None
    ):
        return False
    if selection.actions:
        return any(
            action.error is None and _runnable(action.name, request)
            for action in selection.actions
        )
    # The executor also runs a selection given only as action JSON.
    if request.protocol != "executor" or not selection.action_json:
        return False
    try:
        action = json.loads(selection.action_json)
    except (TypeError, ValueError):
        return False
    name = action.get("action") if isinstance(action, dict) else None
    return _runnable(name, request)


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
                if _usable(selection, request):
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
