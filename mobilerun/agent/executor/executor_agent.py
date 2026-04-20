"""
ExecutorAgent - Action execution workflow.

This agent is responsible for:
- Taking a specific subgoal from the Manager
- Analyzing the current UI state
- Selecting and executing appropriate actions
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from mobilerun.agent.executor.events import (
    ExecutorActionEvent,
    ExecutorActionResultEvent,
    ExecutorContextEvent,
    ExecutorResponseEvent,
)
from mobilerun.agent.executor.prompts import parse_executor_response
from mobilerun.agent.usage import get_usage_from_response
from mobilerun.agent.utils.inference import acall_with_retries
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.config_manager.config_manager import AgentConfig
from mobilerun.config_manager.prompt_loader import PromptLoader
from mobilerun.tools.helpers.images import resize_image_to_max_side

if TYPE_CHECKING:
    from mobilerun.agent.action_context import ActionContext
    from mobilerun.agent.droid import MobileAgentState
    from mobilerun.agent.tool_registry import ToolRegistry

logger = logging.getLogger("mobilerun")


class ExecutorAgent(Workflow):
    """
    Action execution agent that performs specific actions.

    Single-turn agent: receives subgoal, selects action, executes it.
    Uses ChatMessage objects directly for LLM calls.
    """

    # Flow-control tools hidden from executor's LLM prompt
    _EXCLUDE_TOOLS = {"remember", "complete"}

    def __init__(
        self,
        llm: LLM,
        registry: "ToolRegistry | None",
        action_ctx: "ActionContext | None",
        shared_state: "MobileAgentState",
        agent_config: AgentConfig,
        prompt_resolver: Optional[PromptResolver] = None,
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

        logger.debug("ExecutorAgent initialized.")

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

        # Get available secrets (only if type_secret is actually in the registry)
        available_secrets = []
        if (
            self.registry
            and "type_secret" in self.registry.tools
            and self.action_ctx
            and self.action_ctx.credential_manager
        ):
            available_secrets = await self.action_ctx.credential_manager.get_keys()

        # Build prompt variables
        variables = {
            "instruction": self.shared_state.instruction,
            "app_card": "",
            "device_state": self.shared_state.formatted_device_state,
            "plan": self.shared_state.plan,
            "subgoal": subgoal,
            "progress_status": self.shared_state.progress_summary,
            "atomic_actions": self.registry.get_signatures(exclude=self._EXCLUDE_TOOLS),
            "action_history": action_history,
            "available_secrets": available_secrets,
            "variables": self.shared_state.custom_variables,
            "platform": self.shared_state.platform,
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
                if getattr(
                    self.action_ctx.state_provider,
                    "requires_coordinate_tools",
                    False,
                ):
                    screenshot = resize_image_to_max_side(screenshot)
                messages[0].blocks.append(ImageBlock(image=screenshot))
                logger.debug("📸 Using screenshot for Executor")
            else:
                logger.warning("⚠️ Vision enabled but no screenshot available")
        await ctx.store.set("executor_messages", messages)
        event = ExecutorContextEvent(subgoal=subgoal)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ExecutorContextEvent
    ) -> ExecutorResponseEvent:
        """Get LLM response."""
        logger.debug("Executor getting LLM response...")

        # Get messages from context
        messages = await ctx.store.get("executor_messages")

        try:
            logger.info("Executor response:", extra={"color": "green"})
            response = await acall_with_retries(
                self.llm, messages, stream=self.agent_config.streaming
            )
            response_text = str(response)
        except ValueError as e:
            logger.warning(f"Executor LLM returned empty response: {e}")
            error_response = (
                "### Thought\nExecutor failed to respond, try again\n"
                '### Action\n{"action": "invalid"}\n'
                "### Description\nExecutor failed to respond, try again"
            )
            event = ExecutorResponseEvent(response=error_response, usage=None)
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
    ) -> ExecutorActionEvent:
        """Parse LLM response and extract action."""
        logger.debug("⚙️ Processing executor response...")

        response_text = ev.response

        try:
            parsed = parse_executor_response(response_text)
        except Exception as e:
            logger.error(f"❌ Failed to parse executor response: {e}")
            return ExecutorActionEvent(
                action_json=json.dumps({"action": "invalid"}),
                thought=f"Failed to parse response: {str(e)}",
                description="Invalid response format from LLM",
                full_response=response_text,
            )

        # Update unified state
        self.shared_state.last_thought = parsed["thought"]

        event = ExecutorActionEvent(
            action_json=parsed["action"],
            thought=parsed["thought"],
            description=parsed["description"],
            full_response=response_text,
        )

        ctx.write_event_to_stream(event)
        return event

    @step
    async def execute(
        self, ctx: Context, ev: ExecutorActionEvent
    ) -> ExecutorActionResultEvent:
        """Execute the action."""
        logger.debug(f"⚡ Executing action: {ev.description}")

        try:
            action_dict = json.loads(ev.action_json)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse action JSON: {e}")
            event = ExecutorActionResultEvent(
                action={"action": "invalid"},
                success=False,
                error=f"Invalid action JSON: {str(e)}",
                summary="Failed to parse action",
                thought=ev.thought,
                full_response=ev.full_response,
            )
            ctx.write_event_to_stream(event)
            return event

        # Extract action type and args, dispatch via registry
        action_type = action_dict.get("action", "unknown")
        action_args = {k: v for k, v in action_dict.items() if k != "action"}

        result = await self.registry.execute(
            action_type, action_args, self.action_ctx, workflow_ctx=ctx
        )

        await asyncio.sleep(self.agent_config.after_sleep_action)

        logger.debug(
            f"{'✅' if result.success else '❌'} Execution complete: {result.summary}"
        )

        event = ExecutorActionResultEvent(
            action=action_dict,
            success=result.success,
            error="" if result.success else result.summary,
            summary=result.summary,
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
                "action": ev.action,
                "outcome": ev.success,
                "error": ev.error,
                "summary": ev.summary,
                "thought": ev.thought,
            }
        )
