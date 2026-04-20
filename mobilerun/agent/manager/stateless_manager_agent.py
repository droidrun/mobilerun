"""
StatelessManagerAgent - Stateless planning agent that rebuilds context each turn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from opentelemetry import trace
from pydantic import BaseModel

from mobilerun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agent.manager.events import (
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)
from mobilerun.agent.manager.prompts import parse_manager_response
from mobilerun.agent.usage import get_usage_from_response
from mobilerun.agent.utils.chat_utils import to_chat_messages
from mobilerun.agent.utils.inference import acall_with_retries
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.agent.utils.tracing_setup import record_langfuse_screenshot
from mobilerun.config_manager.prompt_loader import PromptLoader
from mobilerun.tools.driver.base import DeviceDisconnectedError
from mobilerun.tools.helpers.images import resize_image_to_max_side

if TYPE_CHECKING:
    from mobilerun.agent.action_context import ActionContext
    from mobilerun.agent.droid import MobileAgentState
    from mobilerun.agent.tool_registry import ToolRegistry
    from mobilerun.config_manager.config_manager import AgentConfig, TracingConfig
    from mobilerun.tools.ui.provider import StateProvider


logger = logging.getLogger("mobilerun")


class StatelessManagerAgent(Workflow):
    def __init__(
        self,
        llm: LLM,
        action_ctx: "ActionContext | None",
        state_provider: "StateProvider | None",
        save_trajectory: str = "none",
        shared_state: "MobileAgentState" = None,
        agent_config: "AgentConfig" = None,
        registry: "ToolRegistry | None" = None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        tracing_config: "TracingConfig | None" = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = agent_config.manager
        self.vision = self.config.vision
        self.action_ctx = action_ctx
        self.state_provider = state_provider
        self.save_trajectory = save_trajectory
        self.shared_state = shared_state
        self.registry = registry
        self.output_model = output_model
        self.agent_config = agent_config
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config

    def _build_action_history(self) -> list[dict]:
        if not self.shared_state.action_history:
            return []

        n = min(5, len(self.shared_state.action_history))
        return [
            {
                "action": act,
                "summary": summ,
                "outcome": outcome,
                "error": err,
            }
            for act, summ, outcome, err in zip(
                self.shared_state.action_history[-n:],
                self.shared_state.summary_history[-n:],
                self.shared_state.action_outcomes[-n:],
                self.shared_state.error_descriptions[-n:],
                strict=True,
            )
        ]

    async def _build_prompt(self) -> str:
        variables = {
            "instruction": self.shared_state.instruction,
            "device_date": self.shared_state.device_date,
            "previous_plan": self.shared_state.previous_plan,
            "previous_state": self.shared_state.previous_formatted_device_state,
            "memory": self.shared_state.manager_memory,
            "last_thought": self.shared_state.last_thought,
            "progress_summary": self.shared_state.progress_summary,
            "action_history": self._build_action_history(),
            "current_state": self.shared_state.formatted_device_state,
        }

        custom_prompt = self.prompt_resolver.get_prompt("manager_system")
        if custom_prompt:
            return PromptLoader.render_template(custom_prompt, variables)

        return await PromptLoader.load_prompt(
            self.agent_config.get_manager_system_prompt_path(),
            variables,
        )

    async def _validate_and_retry(
        self, messages: list[dict], initial_response: str
    ) -> str:
        output = initial_response
        parsed = parse_manager_response(output)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            error_message = None

            if parsed["answer"] and not parsed["plan"]:
                if parsed["success"] is None:
                    error_message = (
                        'You must include success="true" or success="false" attribute '
                        "in the <answer> or <request_accomplished> tag.\n"
                        'Example: <answer success="true">Task completed</answer>\n'
                        "Retry again."
                    )
                else:
                    break
            elif parsed["plan"] and parsed["answer"]:
                error_message = (
                    "You cannot include both <plan> and <answer> tags. "
                    "Use <answer> only when the task is complete.\n"
                    "Retry again."
                )
            elif not parsed["plan"] and not parsed["answer"]:
                error_message = (
                    "You must provide either a <plan> or an <answer>. "
                    "Please provide a plan with numbered steps."
                )
            else:
                break

            if error_message:
                retry_count += 1
                logger.warning(
                    f"Manager response invalid (retry {retry_count}/{max_retries}): {error_message}"
                )

                retry_messages = messages + [
                    {"role": "assistant", "content": [{"text": output}]},
                    {"role": "user", "content": [{"text": error_message}]},
                ]

                chat_messages = to_chat_messages(retry_messages)

                try:
                    response = await acall_with_retries(self.llm, chat_messages)
                    output = response.message.content
                    parsed = parse_manager_response(output)
                except Exception as e:
                    logger.error(f"LLM retry failed: {e}")
                    break

        return output

    @step
    async def prepare_context(
        self, ctx: Context, ev: StartEvent
    ) -> ManagerContextEvent:
        screenshot = None
        if self.vision or self.save_trajectory != "none":
            try:
                screenshot = await self.action_ctx.driver.screenshot()

                if screenshot:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    parent_span = trace.get_current_span()
                    record_langfuse_screenshot(
                        screenshot,
                        parent_span=parent_span,
                        screenshots_enabled=bool(
                            self.tracing_config
                            and self.tracing_config.langfuse_screenshots
                        ),
                        vision_enabled=self.vision,
                    )
            except DeviceDisconnectedError:
                raise
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")

        ui_state = await self.state_provider.get_state()
        self.action_ctx.ui = ui_state

        self.shared_state.previous_formatted_device_state = (
            self.shared_state.formatted_device_state
        )
        self.shared_state.formatted_device_state = ui_state.formatted_text
        self.shared_state.focused_text = ui_state.focused_text
        self.shared_state.a11y_tree = ui_state.elements
        self.shared_state.phone_state = ui_state.phone_state

        self.shared_state.update_current_app(
            package_name=ui_state.phone_state.get("packageName", "Unknown"),
            activity_name=ui_state.phone_state.get("currentApp", "Unknown"),
        )

        ctx.write_event_to_stream(RecordUIStateEvent(ui_state=ui_state.elements))

        self.shared_state.screenshot = screenshot

        event = ManagerContextEvent()
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(
        self, ctx: Context, ev: ManagerContextEvent
    ) -> ManagerResponseEvent:
        screenshot = self.shared_state.screenshot

        prompt_text = await self._build_prompt()
        messages = [{"role": "user", "content": [{"text": prompt_text}]}]

        if self.vision and screenshot:
            if getattr(self.state_provider, "requires_coordinate_tools", False):
                screenshot = resize_image_to_max_side(screenshot)
            messages[0]["content"].append({"image": screenshot})

        chat_messages = to_chat_messages(messages)

        try:
            logger.info("Manager response:", extra={"color": "cyan"})
            response = await acall_with_retries(
                self.llm, chat_messages, stream=self.agent_config.streaming
            )
            output = response.message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Error calling LLM in stateless manager: {e}") from e

        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        output = await self._validate_and_retry(messages, output)

        event = ManagerResponseEvent(response=output, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ManagerResponseEvent
    ) -> ManagerPlanDetailsEvent:
        output = ev.response
        parsed = parse_manager_response(output)

        self.shared_state.previous_plan = parsed["plan"]
        self.shared_state.last_thought = parsed["thought"]

        if parsed.get("progress_summary"):
            self.shared_state.progress_summary = parsed["progress_summary"]

        memory_update = parsed.get("memory", "").strip()
        if memory_update:
            if self.shared_state.manager_memory:
                self.shared_state.manager_memory += "\n" + memory_update
            else:
                self.shared_state.manager_memory = memory_update

        self.shared_state.plan = parsed["plan"]
        self.shared_state.current_subgoal = parsed["current_subgoal"]
        self.shared_state.answer = parsed["answer"]

        event = ManagerPlanDetailsEvent(
            plan=parsed["plan"],
            subgoal=parsed["current_subgoal"],
            thought=parsed["thought"],
            answer=parsed["answer"],
            memory_update=memory_update,
            progress_summary=parsed.get("progress_summary", ""),
            success=parsed["success"],
            full_response=output,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def finalize(self, ctx: Context, ev: ManagerPlanDetailsEvent) -> StopEvent:
        return StopEvent(
            result={
                "plan": ev.plan,
                "current_subgoal": ev.subgoal,
                "thought": ev.thought,
                "answer": ev.answer,
                "memory_update": ev.memory_update,
                "success": ev.success,
            }
        )
