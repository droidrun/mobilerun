import asyncio
from dataclasses import dataclass, replace
from typing import Protocol

from mobilerun.agent.selectors.base import (
    ActionSelection,
    BaseActionSelector,
    SelectionRequest,
)
from mobilerun.config_manager.config_manager import AgentConfig
from mobilerun.config_manager.prompt_loader import PromptLoader


@dataclass
class Decision:
    option_index: int | None = None
    thought: str = ""
    success: bool | None = None
    message: str = ""


class DecisionModel(Protocol):
    async def decide(self, request: SelectionRequest) -> Decision | None: ...


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
            raise ValueError("Decision selection requires non-empty options in the request")
        options = tuple(o for o in request.options if o.action.name in request.tools)
        if not options:
            raise ValueError("No decision options match the available tools")
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
                    ToolCall("complete", {
                        "success": decision.success,
                        "message": decision.message,
                    })
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
        return ActionSelection(
            actions=[option.action],
            thought=decision.thought,
            description=option.description,
        )
