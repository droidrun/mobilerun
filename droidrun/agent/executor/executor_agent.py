"""
ExecutorAgent - Action execution workflow.

This agent is responsible for:
- Taking a specific subgoal from the Manager
- Analyzing the current UI state
- Selecting and executing appropriate actions using XML tool-calling protocol
- Supports multiple tool calls per response (up to 4)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from droidrun.agent.action_result import ActionRecord
from droidrun.agent.common.xml_parser import (
    CLOSE_TAG,
    OPEN_TAG,
    parse_tool_calls,
)
from droidrun.agent.executor.events import (
    ExecutorActionResultEvent,
    ExecutorContextEvent,
    ExecutorResponseEvent,
    ExecutorToolCallEvent,
)
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.config_manager.config_manager import AgentConfig
from droidrun.config_manager.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from droidrun.agent.action_context import ActionContext
    from droidrun.agent.droid import DroidAgentState
    from droidrun.agent.tool_registry import ToolRegistry

logger = logging.getLogger("droidrun")

# Maximum number of tool calls per executor response
MAX_TOOL_CALLS = 4


class ExecutorAgent(Workflow):
    """
    Action execution agent that performs specific actions.

    Single-turn agent: receives subgoal, selects action(s), executes them.
    Uses XML tool-calling protocol (<function_calls>) for structured output.
    Supports up to MAX_TOOL_CALLS tool invocations per response.
    """

    # Flow-control tools hidden from executor's LLM prompt
    _EXCLUDE_TOOLS = {"remember", "complete", "read", "grep"}

    def __init__(
        self,
        llm: LLM,
        registry: "ToolRegistry | None",
        action_ctx: "ActionContext | None",
        shared_state: "DroidAgentState",
        agent_config: AgentConfig,
        prompt_resolver: PromptResolver | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.agent_config = agent_config
        self.config = agent_config.executor
        self.vision = agent_config.executor.vision
        self.registry = registry
        self.action_ctx = action_ctx
        self.shared_state = shared_state
        self.prompt_resolver = prompt_resolver or PromptResolver()

        # Lazily built from registry on first use
        self._tool_descriptions: str | None = None
        self._param_types: dict | None = None

        logger.debug("ExecutorAgent initialized.")

    @property
    def tool_descriptions(self) -> str:
        if self._tool_descriptions is None:
            self._tool_descriptions = self.registry.get_tool_descriptions_xml(
                exclude=self._EXCLUDE_TOOLS
            )
        return self._tool_descriptions

    @property
    def param_types(self) -> dict:
        if self._param_types is None:
            self._param_types = self.registry.get_param_types(
                exclude=self._EXCLUDE_TOOLS
            )
        return self._param_types

    @step
    async def prepare_context(
        self, ctx: Context, ev: StartEvent
    ) -> ExecutorContextEvent:
        """Prepare executor context and prompt."""
        subgoal = ev.get("subgoal", "")
        logger.debug(f"🧠 Executor thinking about action for: {subgoal}")

        # Build action history (last 5)
        action_history = []
        if self.shared_state.action_history:
            n = min(5, len(self.shared_state.action_history))
            action_history = [
                {"action": act, "summary": summ, "outcome": outcome, "error": err}
                for act, summ, outcome, err in zip(
                    self.shared_state.action_history[-n:],
                    self.shared_state.summary_history[-n:],
                    self.shared_state.action_outcomes[-n:],
                    self.shared_state.error_descriptions[-n:],
                    strict=True,
                )
            ]

        # Get available secrets
        available_secrets = []
        if self.action_ctx and self.action_ctx.credential_manager:
            available_secrets = await self.action_ctx.credential_manager.get_keys()

        # Build prompt variables
        variables = {
            "instruction": self.shared_state.instruction,
            "app_card": "",
            "device_state": self.shared_state.formatted_device_state,
            "plan": self.shared_state.plan,
            "subgoal": subgoal,
            "progress_status": self.shared_state.progress_summary,
            "tool_descriptions": self.tool_descriptions,
            "action_history": action_history,
            "available_secrets": available_secrets,
            "variables": self.shared_state.custom_variables,
        }

        custom_prompt = self.prompt_resolver.get_prompt("executor_system")
        if custom_prompt:
            prompt_text = PromptLoader.render_template(custom_prompt, variables)
        else:
            prompt_text = await PromptLoader.load_prompt(
                self.agent_config.get_executor_system_prompt_path(),
                variables,
            )

        # Build message
        messages = [ChatMessage(role="user", blocks=[TextBlock(text=prompt_text)])]

        # Add screenshot if vision enabled
        if self.vision:
            screenshot = self.shared_state.screenshot
            if screenshot is not None:
                messages[0].blocks.append(ImageBlock(image=screenshot))
                logger.debug("📸 Using screenshot for Executor")
            else:
                logger.warning("⚠️ Vision enabled but no screenshot available")
        await ctx.store.set("executor_messages", messages)
        await ctx.store.set("current_subgoal", subgoal)
        event = ExecutorContextEvent(subgoal=subgoal)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ExecutorContextEvent
    ) -> ExecutorResponseEvent:
        """Get LLM response."""
        logger.debug("Executor getting LLM response...")

        messages = await ctx.store.get("executor_messages")

        try:
            logger.info("Executor response:", extra={"color": "green"})
            response = await acall_with_retries(
                self.llm, messages, stream=self.agent_config.streaming
            )
            response_text = str(response)
        except ValueError as e:
            logger.warning(f"Executor LLM returned empty response: {e}")
            event = ExecutorResponseEvent(response="", usage=None)
            ctx.write_event_to_stream(event)
            return event
        except Exception as e:
            raise RuntimeError(f"Error calling LLM in executor: {e}") from e

        # Extract usage
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        event = ExecutorResponseEvent(response=response_text, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ExecutorResponseEvent
    ) -> ExecutorToolCallEvent:
        """Parse LLM response and extract tool calls."""
        logger.debug("⚙️ Processing executor response...")

        response_text = ev.response

        # Parse XML tool calls
        thought, tool_calls = parse_tool_calls(response_text, self.param_types)

        if not tool_calls:
            # No tool calls — pass the executor's text through to the Manager
            logger.debug("Executor returned text without tool calls")
            self.shared_state.last_thought = thought
            event = ExecutorToolCallEvent(
                tool_calls_repr=None,
                thought=thought,
                full_response=response_text,
            )
            ctx.write_event_to_stream(event)
            return event

        # Cap at MAX_TOOL_CALLS
        if len(tool_calls) > MAX_TOOL_CALLS:
            logger.warning(
                f"Executor produced {len(tool_calls)} tool calls, capping at {MAX_TOOL_CALLS}"
            )
            tool_calls = tool_calls[:MAX_TOOL_CALLS]

        await ctx.store.set("pending_tool_calls", tool_calls)

        # Extract tool calls XML for event (only for capped calls)
        blocks = []
        for i, part in enumerate(response_text.split(OPEN_TAG)[1:]):
            if i >= len(tool_calls):
                break
            close_idx = part.find(CLOSE_TAG)
            if close_idx != -1:
                blocks.append(OPEN_TAG + part[: close_idx + len(CLOSE_TAG)])
        tool_calls_xml = "\n".join(blocks) if blocks else None

        # Update shared state
        self.shared_state.last_thought = thought

        event = ExecutorToolCallEvent(
            tool_calls_repr=tool_calls_xml,
            thought=thought,
            full_response=response_text,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def execute(
        self, ctx: Context, ev: ExecutorToolCallEvent
    ) -> ExecutorActionResultEvent:
        """Execute all parsed tool calls."""
        tool_calls = await ctx.store.get("pending_tool_calls", [])

        if not tool_calls:
            # No tool calls — executor returned text only, pass through
            event = ExecutorActionResultEvent(
                actions=[],
                thought=ev.thought,
                full_response=ev.full_response,
            )
            ctx.write_event_to_stream(event)
            return event

        actions: list[ActionRecord] = []
        for call in tool_calls:
            logger.debug(f"Executing: {call.name}({call.parameters})")

            if call.error:
                error_msg = f"Invalid arguments for {call.name}: {call.error}"
                actions.append(
                    ActionRecord(
                        action=call.name,
                        args=call.parameters,
                        outcome=False,
                        error=error_msg,
                        summary=error_msg,
                    )
                )
                # Stop on error
                break

            action_result = await self.registry.execute(
                call.name, call.parameters, self.action_ctx, workflow_ctx=ctx
            )

            actions.append(
                ActionRecord(
                    action=call.name,
                    args=call.parameters,
                    outcome=action_result.success,
                    error="" if action_result.success else action_result.summary,
                    summary=action_result.summary,
                )
            )

            if not action_result.success:
                # Stop on failure
                break

        await asyncio.sleep(self.agent_config.after_sleep_action)

        logger.debug(f"Executor executed {len(actions)} tool call(s)")

        event = ExecutorActionResultEvent(
            actions=actions,
            thought=ev.thought,
            full_response=ev.full_response,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def finalize(self, ctx: Context, ev: ExecutorActionResultEvent) -> StopEvent:
        """Return executor results to parent workflow."""
        logger.debug("✅ Executor execution complete")

        return StopEvent(
            result={
                "actions": ev.actions,
                "thought": ev.thought,
                "full_response": ev.full_response,
            }
        )
