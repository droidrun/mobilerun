"""Let a selector explicitly hand off the current action to another selector."""

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from llama_index.core.base.llms.types import TextBlock

from mobilerun.agent.selectors.base import (
    ActionCandidate,
    ActionSelection,
    ActionSelector,
    BaseActionSelector,
    SelectionRequest,
    all_locate_points,
    sum_usage,
    supports_native_tool_calls,
)
from mobilerun.config_manager.config_manager import AgentConfig
from mobilerun.config_manager.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from llama_index.core.llms.llm import LLM

logger = logging.getLogger("mobilerun")

DEFAULT_DESCRIPTION = (
    "Ask the delegate model to choose the next action when you need its capabilities."
)
# Decision models only see this text, so name what an LLM delegate adds.
LLM_DELEGATE_DESCRIPTION = (
    "Hand this step to a general model that can type text into fields "
    "(for example a search box) and handle steps that need free text."
)


class DelegatingSelector(BaseActionSelector):
    """Offer the primary a virtual ``delegate`` action, not an error fallback.

    The delegate receives the original request without the virtual action. Only
    its final selection is returned for execution; the next step starts with
    the primary again. Neither selector executes tools here.
    """

    def __init__(
        self,
        primary: ActionSelector,
        delegate: ActionSelector,
        description: str | None = None,
    ):
        self.primary = primary
        self.delegate = delegate
        self.description = description or (
            LLM_DELEGATE_DESCRIPTION if delegate.llms else DEFAULT_DESCRIPTION
        )

    @property
    def supports_native_tool_calls(self) -> bool:
        return supports_native_tool_calls((self.primary, self.delegate))

    @property
    def requires_action_candidates(self) -> bool:
        return any(
            selector.requires_action_candidates
            for selector in (self.primary, self.delegate)
        )

    @property
    def locates_points(self) -> bool:
        return all_locate_points((self.primary, self.delegate))

    @property
    def screenshot_max_side(self) -> int | None:
        limits = [
            limit
            for selector in (self.primary, self.delegate)
            if (limit := selector.screenshot_max_side) is not None
        ]
        return min(limits) if limits else None

    @property
    def llms(self) -> tuple["LLM", ...]:
        return tuple(
            llm for selector in (self.primary, self.delegate) for llm in selector.llms
        )

    async def select(self, request: SelectionRequest) -> ActionSelection | None:
        from mobilerun.agent.fast_agent.xml_parser import ToolCall, extract_add_memory

        if "delegate" in request.tools or any(
            option.action.name == "delegate" for option in request.options or ()
        ):
            raise ValueError("DelegatingSelector reserves the tool name 'delegate'")

        guidance = (
            f"Additional action: delegate() — {self.description}\n"
            "Choose it explicitly to let the delegate select this step's action "
            "using the same goal, observation, tools, memory, and history. "
            "If its capabilities are needed, delegate instead of declaring the goal impossible. "
            "Delegation must be the only action in your response and takes no parameters."
        )
        messages = [message.model_copy(deep=True) for message in request.messages]
        if messages:
            example = {
                "executor": '{"action":"delegate"}',
                "native": "call the delegate function with no arguments",
            }.get(
                request.protocol,
                '<function_calls><invoke name="delegate"></invoke></function_calls>',
            )
            target = next(
                (message for message in messages if message.role == "system"),
                messages[-1],
            )
            target.blocks.append(
                TextBlock(text=f"\n\n{guidance}\nCall format: {example}")
            )

        primary_request = replace(
            request,
            tools={
                **request.tools,
                "delegate": {"parameters": {}, "description": self.description},
            },
            options=(
                *(request.options or ()),
                ActionCandidate(ToolCall("delegate"), self.description),
            ),
            messages=messages,
        )
        if request.tool_specs is not None:
            primary_request.tool_specs = [
                *request.tool_specs,
                {
                    "type": "function",
                    "name": "delegate",
                    "description": self.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "strict": False,
                },
            ]
        instructions = request.instructions
        if not instructions and self.primary.requires_action_candidates:
            instructions = await PromptLoader.load_prompt(
                AgentConfig().get_decision_system_prompt_path(),
                primary_request.get_decision_prompt_variables(),
            )
        primary_request.instructions = f"{instructions}\n\n{guidance}".strip()

        selection = await self.primary.select(primary_request)
        if selection is None or not self._calls_delegate(selection):
            return selection
        actions = selection.actions
        if not actions:
            action = json.loads(selection.action_json)
            actions = [
                ToolCall(
                    action["action"],
                    {key: value for key, value in action.items() if key != "action"},
                )
            ]
        if len(actions) > 1 or any(
            action.name != "delegate" or action.parameters for action in actions
        ):
            return self._invalid(
                selection,
                request,
                "Delegation must be the only selected action and takes no parameters",
            )

        logger.info("Delegating action selection to %s", type(self.delegate).__name__)
        delegated = await self.delegate.select(request)
        if self._calls_delegate(delegated):
            return self._invalid(
                delegated,
                request,
                "The delegate must select a real action, not delegate again",
            )
        if delegated is None:
            return None

        usage = sum_usage(selection.usage, delegated.usage)
        memory_update = "\n".join(
            filter(
                None,
                (
                    extract_add_memory(selection.thought),
                    selection.memory_update,
                    delegated.memory_update,
                ),
            )
        )
        return replace(delegated, usage=usage, memory_update=memory_update)

    @staticmethod
    def _invalid(
        selection: ActionSelection, request: SelectionRequest, error: str
    ) -> ActionSelection:
        """Report an unusable delegation like malformed output, not a crash."""
        from mobilerun.agent.fast_agent.xml_parser import (
            ToolCall,
            ToolCallParseResult,
            ToolCallParseStatus,
        )

        logger.warning(error)
        if request.protocol == "executor":
            return replace(
                selection,
                actions=[ToolCall("invalid")],
                action_json=json.dumps({"action": "invalid"}),
                failure=error,
            )
        end = selection.parse_result.end if selection.parse_result else None
        return replace(
            selection,
            actions=[],
            failure=error,
            # Native tool calls without tool results would break the next request.
            response=None if request.protocol == "native" else selection.response,
            parse_result=ToolCallParseResult(
                thought=selection.thought,
                calls=[],
                status=ToolCallParseStatus.MALFORMED,
                end=end,
            ),
        )

    @staticmethod
    def _calls_delegate(selection: ActionSelection | None) -> bool:
        if selection is None:
            return False
        if any(action.name == "delegate" for action in selection.actions):
            return True
        if selection.action_json is not None:
            try:
                action = json.loads(selection.action_json)
            except (TypeError, ValueError):
                return False
            return isinstance(action, dict) and action.get("action") == "delegate"
        return False
