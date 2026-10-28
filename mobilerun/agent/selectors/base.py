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

    def get_decision_prompt_variables(self) -> dict[str, Any]:
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
            "can_scroll": "swipe" in available_tools,
            "can_complete": can_complete,
            "vision": self.screenshot is not None,
            "variables": self.variables,
        }


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


class ActionSelector(ABC):
    """Explicit action-selection and capability contract."""

    @property
    def supports_native_tool_calls(self) -> bool:
        """Whether native tool-call messages can be used throughout this selector."""
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
