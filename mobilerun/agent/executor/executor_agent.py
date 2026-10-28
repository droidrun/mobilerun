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
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from mobilerun.agent.executor.events import (
    ExecutorActionEvent,
    ExecutorActionResultEvent,
    ExecutorContextEvent,
    ExecutorResponseEvent,
)
from mobilerun.agent.selectors import ActionSelector, LlamaIndexLLMSelector, SelectionRequest
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.config_manager.config_manager import AgentConfig
from mobilerun.config_manager.prompt_loader import PromptLoader
from mobilerun.tools.ui.provider import (
    resize_model_screenshot_async,
)

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
    _EXCLUDE_TOOLS = {"complete"}

    def __init__(
        self,
        action_selector: ActionSelector | None = None,
        registry: "ToolRegistry | None" = None,
        action_ctx: "ActionContext | None" = None,
        shared_state: "MobileAgentState | None" = None,
        agent_config: AgentConfig | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        **kwargs,
    ):
        if "llm" in kwargs:
            logger.debug(
                "ExecutorAgent(llm=...) is deprecated. Use action_selector=LlamaIndexLLMSelector(llm)."
            )
        llm = kwargs.pop("llm", None)
        if action_selector is None and llm is not None:
            action_selector = LlamaIndexLLMSelector(llm)
        if action_selector is None:
            raise ValueError("action_selector or llm is required")
        if not isinstance(action_selector, ActionSelector):
            raise TypeError("action_selector must implement ActionSelector")
        if agent_config is None:
            raise TypeError("agent_config is required")
        if shared_state is None:
            raise TypeError("shared_state is required")
        super().__init__(**kwargs)
        self.action_selector = action_selector
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
            "app_card": self.shared_state.app_card,
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
                self.agent_config.get_executor_system_prompt_path(), variables,
            )

        # Build message
        messages = [ChatMessage(role="user", blocks=[TextBlock(text=prompt_text)])]

        # Add screenshot if vision enabled
        screenshot = None
        if self.vision:
            screenshot = self.shared_state.screenshot
            if screenshot is not None:
                screenshot = await resize_model_screenshot_async(
                    self.action_ctx.state_provider, screenshot
                )
                messages[0].blocks.append(ImageBlock(image=screenshot))
                logger.debug("📸 Using screenshot for Executor")
            else:
                logger.warning("⚠️ Vision enabled but no screenshot available")
        await ctx.store.set("executor_messages", messages)
        await ctx.store.set("executor_history", action_history)
        await ctx.store.set("executor_screenshot", screenshot)
        event = ExecutorContextEvent(subgoal=subgoal)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ExecutorContextEvent
    ) -> ExecutorResponseEvent:
        """Select an action using the current observation and subgoal."""
        logger.debug("Executor getting LLM response...")

        # Get messages from context
        messages = await ctx.store.get("executor_messages")
        logger.info("Executor response:", extra={"color": "green"})
        tools = self.registry.get_signatures(exclude=self._EXCLUDE_TOOLS)
        if self.action_selector.requires_action_candidates:
            options = await self.action_ctx.state_provider.build_action_candidates(
                self.action_ctx, tools, include_device_actions=True,
            )
        else:
            options = self.action_ctx.state_provider.get_action_candidates(self.action_ctx.ui) if self.action_ctx.ui is not None else ()
        request = SelectionRequest(
            goal=ev.subgoal,
            state=self.action_ctx.ui,
            memory=self.shared_state.agent_memory,
            history=await ctx.store.get("executor_history", default=[]),
            screenshot=await ctx.store.get("executor_screenshot", default=self.shared_state.screenshot)
            if self.vision else None,
            tools=tools,
            variables=self.shared_state.custom_variables,
            messages=messages,
            protocol="executor",
            stream=self.agent_config.streaming,
            options=options,
        )
        if self.action_selector.requires_action_candidates:
            decision_variables = request.get_decision_prompt_variables()
            custom_prompt = self.prompt_resolver.get_prompt("decision_system")
            if custom_prompt:
                request.instructions = PromptLoader.render_template(custom_prompt, decision_variables)
            else:
                request.instructions = await PromptLoader.load_prompt(
                    self.agent_config.get_decision_system_prompt_path(), decision_variables,
                )
        selection = await self.action_selector.select(request)
        if selection is None:
            raise ValueError("Action selector abstained without a fallback")
        if len(selection.actions) > 1 or (not selection.actions and selection.action_json is None):
            raise ValueError("Executor requires exactly one selected action")
        await ctx.store.set("executor_selection", selection)
        response_text = selection.response_text
        event = ExecutorResponseEvent(response=response_text, usage=selection.usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ExecutorResponseEvent
    ) -> ExecutorActionEvent:
        """Translate the selection to the existing execution event."""
        logger.debug("⚙️ Processing executor response...")
        selection = await ctx.store.get("executor_selection")
        if selection.parse_error is not None:
            logger.error(f"❌ Failed to parse executor response: {selection.parse_error}")
            return ExecutorActionEvent(
                action_json=json.dumps({"action": "invalid"}),
                thought=selection.thought,
                description=selection.description,
                full_response=ev.response,
            )
        action_json = selection.action_json
        if action_json is None:
            action = selection.actions[0]
            action_json = json.dumps({"action": action.name, **action.parameters})
        self.shared_state.last_thought = selection.thought
        if selection.memory_update:
            self.shared_state.append_memory(selection.memory_update)
        event = ExecutorActionEvent(
            action_json=action_json,
            thought=selection.thought,
            description=selection.description,
            full_response=ev.response,
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
