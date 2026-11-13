from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from mobilerun.agent.usage import UsageResult

if TYPE_CHECKING:
    from llama_index.core.llms.llm import LLM

    from mobilerun.agent.fast_agent.xml_parser import ToolCall, ToolCallParseResult
    from mobilerun.tools.ui.state import UIState


@dataclass
class ActionCandidate:
    action: ToolCall
    description: str
    # Set when the decision model locates the action's x/y on the screenshot:
    # the (width, height) of the coordinate space that point is written in.
    point_space: tuple[float, float] | None = None


@dataclass
class SelectionRequest:
    goal: str
    state: UIState | None = None
    memory: str = ""
    history: list[Any] = field(default_factory=list)
    screenshot: bytes | None = None
    tools: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    options: tuple[ActionCandidate, ...] | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    protocol: Literal["executor", "xml", "native"] = "xml"
    param_types: dict[str, str] = field(default_factory=dict)
    stream: bool = False
    instructions: str = ""
    tool_specs: list[dict[str, Any]] | None = None
    parallel_tools: bool = True

    def get_decision_prompt_variables(
        self, delegate_can_type: bool = False
    ) -> dict[str, Any]:
        """Describe selectable capabilities for decision prompt templates."""
        available_tools = {
            option.action.name
            for option in self.options or ()
            if option.action.name in self.tools
        }
        can_complete = "complete" in self.tools and self.protocol != "executor"
        if can_complete:
            available_tools.add("complete")
        return {
            "goal": self.goal,
            "protocol": self.protocol,
            "available_tools": available_tools,
            "registered_tools": set(self.tools),
            "can_type": bool(available_tools & {"type", "type_text"}),
            "delegate_can_type": delegate_can_type,
            "can_scroll": "swipe" in available_tools,
            "can_locate": any(option.point_space for option in self.options or ()),
            "can_complete": can_complete,
            "vision": self.screenshot is not None,
            "variables": self.variables,
        }


def all_locate_points(selectors) -> bool:
    """Whether every selector that chooses among options can locate taps."""
    choosers = [s for s in selectors if s.requires_action_candidates]
    return bool(choosers) and all(s.locates_points for s in choosers)


def sum_usage(*usages: UsageResult | None) -> UsageResult | None:
    """Add the usage of every model call made for one step."""
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    return UsageResult(
        **{
            name: sum(getattr(usage, name) for usage in present)
            for name in UsageResult.model_fields
        }
    )


@dataclass
class ActionSelection:
    actions: list[ToolCall] = field(default_factory=list)
    thought: str = ""
    description: str = ""
    memory_update: str = ""
    response: ChatResponse | None = None
    response_text: str = ""
    usage: UsageResult | None = None
    parse_result: ToolCallParseResult | None = None
    parse_error: str | None = None
    action_json: str | None = None
    # Why this selection cannot run; reported to the model instead of executing.
    failure: str | None = None
    # The selector's confidence in this choice, from 0 to 1, when it reports one.
    confidence: float | None = None


class ActionSelector(ABC):
    """Explicit action-selection and capability contract."""

    @property
    def supports_native_tool_calls(self) -> bool:
        """Whether native tool-call messages can be used throughout this selector."""
        return False

    @property
    def locates_points(self) -> bool:
        """Whether selection can locate a tap on the screenshot itself."""
        return False

    @property
    def delegate_can_type(self) -> bool:
        """Whether a delegate can type free text for the candidate chooser."""
        return False

    @property
    @abstractmethod
    def requires_action_candidates(self) -> bool:
        """Whether selection needs concrete actions enumerated by the provider."""
        ...

    @property
    @abstractmethod
    def requires_action_llm(self) -> bool:
        """Whether a separate action-role LLM must be loaded from config."""
        ...

    @property
    @abstractmethod
    def shared_llm(self) -> LLM | None:
        """An explicitly shared LLM for planning/helpers, if any."""
        ...

    @property
    @abstractmethod
    def abstention_reason(self) -> str:
        """Message to report when selection returns no action."""
        ...

    @property
    @abstractmethod
    def screenshot_max_side(self) -> int | None:
        """Maximum supported screenshot side length, if constrained."""
        ...

    @property
    @abstractmethod
    def llms(self) -> tuple[LLM, ...]:
        """Native LLMs used for action selection, for vision sizing and telemetry."""
        ...

    @abstractmethod
    async def select(self, request: SelectionRequest) -> ActionSelection | None:
        """Select actions, or return None to abstain. Never execute tools here."""
        ...


def supports_native_tool_calls(selectors: tuple[ActionSelector, ...]) -> bool:
    """Native when every LLM-backed selector supports it; others ignore the protocol."""
    llm_selectors = [selector for selector in selectors if selector.llms]
    # One history is shared, so every LLM must speak the same native API.
    apis = {
        getattr(llm, "native_tool_api", "responses")
        for selector in llm_selectors
        for llm in selector.llms
    }
    return (
        bool(llm_selectors)
        and len(apis) == 1
        and all(selector.supports_native_tool_calls for selector in llm_selectors)
    )


class BaseActionSelector(ActionSelector):
    """Capability defaults for selectors that implement their own select()."""

    requires_action_candidates: bool = False
    requires_action_llm: bool = False
    abstention_reason: str = "Action selector abstained without a fallback."

    @property
    def shared_llm(self) -> LLM | None:
        """Only an explicitly shared LLM should also supply planning/helpers."""
        return None

    @property
    def screenshot_max_side(self) -> int | None:
        return None

    @property
    def llms(self) -> tuple[LLM, ...]:
        return ()
