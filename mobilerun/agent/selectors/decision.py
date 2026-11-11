import asyncio
import logging
import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Protocol

from mobilerun.agent.selectors.base import (
    ActionCandidate,
    ActionSelection,
    BaseActionSelector,
    SelectionRequest,
)
from mobilerun.config_manager.config_manager import AgentConfig
from mobilerun.config_manager.prompt_loader import PromptLoader

logger = logging.getLogger("mobilerun")


@dataclass
class Decision:
    option_index: int | None = None
    thought: str = ""
    success: bool | None = None
    message: str = ""
    # Located tap as (x, y) fractions of the screenshot, for options that
    # leave their point to the model.
    point: tuple[float, float] | None = None


class DecisionModel(Protocol):
    async def decide(self, request: SelectionRequest) -> Decision | None: ...


# Lower ranks are kept first when a model limits its option count.
_OPTION_RANKS = {"click": 1, "click_at": 1, "open_app": 2, "long_press": 3}


def _cap_options(
    options: tuple[ActionCandidate, ...], limit: int | None
) -> tuple[ActionCandidate, ...]:
    if limit is None or len(options) <= limit:
        return options
    ranked = sorted(
        range(len(options)),
        key=lambda i: (_OPTION_RANKS.get(options[i].action.name, 0), i),
    )
    kept = sorted(ranked[:limit])
    dropped = Counter(options[i].action.name for i in ranked[limit:])
    logger.warning(
        "Dropped %d decision options to fit the model limit of %d: %s",
        len(options) - limit,
        limit,
        ", ".join(f"{count} {name}" for name, count in sorted(dropped.items())),
    )
    return tuple(options[i] for i in kept)


def _place(
    point: tuple[float, float] | None, space: tuple[float, float]
) -> tuple[int, int]:
    """Map a located (x, y) screenshot fraction into the action's space."""
    if (
        point is None
        or len(point) != 2
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in point)
    ):
        raise ValueError(f"Invalid located point: {point}")
    x, y = (
        min(math.floor(value * size), math.ceil(size) - 1)
        for value, size in zip(point, space)
    )
    return x, y


class DecisionSelector(BaseActionSelector):
    requires_action_candidates = True

    def __init__(
        self,
        model: DecisionModel,
        timeout: float = 500,
    ):
        if not callable(getattr(model, "decide", None)):
            raise TypeError("Decision model must implement decide(request)")
        self.model = model
        if timeout <= 0:
            raise ValueError("Decision timeout must be positive")
        self.timeout = timeout

    @property
    def screenshot_max_side(self) -> int | None:
        return getattr(self.model, "screenshot_max_side", None)

    async def select(self, request: SelectionRequest) -> ActionSelection | None:
        from mobilerun.agent.fast_agent.xml_parser import ToolCall

        if not request.options:
            raise ValueError(
                "Decision selection requires non-empty options in the request"
            )
        options = tuple(o for o in request.options if o.action.name in request.tools)
        if request.screenshot is None or not getattr(
            self.model, "locates_points", False
        ):
            options = tuple(o for o in options if o.point_space is None)
        if not options:
            raise ValueError("No decision options match the available tools")
        options = _cap_options(options, getattr(self.model, "max_options", None))
        model_request = replace(request, options=options)
        if not model_request.instructions:
            model_request.instructions = await PromptLoader.load_prompt(
                AgentConfig().get_decision_system_prompt_path(),
                model_request.get_decision_prompt_variables(),
            )
        decision = await asyncio.wait_for(
            self.model.decide(model_request), self.timeout
        )
        if decision is None:
            return None
        if not isinstance(decision, Decision):
            raise TypeError("DecisionModel.decide() must return Decision or None")
        if decision.success is not None:
            if type(decision.success) is not bool:
                raise ValueError("Completion success must be a boolean")
            if decision.option_index is not None or "complete" not in request.tools:
                raise ValueError("Completion is not available for this selection")
            return ActionSelection(
                actions=[
                    ToolCall(
                        "complete",
                        {
                            "success": decision.success,
                            "message": decision.message,
                        },
                    )
                ],
                thought=decision.thought,
                description=decision.message,
            )
        index = decision.option_index
        if index is None:
            return None
        if type(index) is not int or not 0 <= index < len(options):
            raise ValueError(f"Invalid decision option index: {index}")
        option = options[index]
        action, description = option.action, option.description
        if option.point_space is not None:
            x, y = _place(decision.point, option.point_space)
            action = ToolCall(action.name, {**action.parameters, "x": x, "y": y})
            description = f"Tap at ({x}, {y})"
        return ActionSelection(
            actions=[action],
            thought=decision.thought,
            description=description,
        )
