"""ToolsAgent — XML tool-calling agent for device interaction.

Replaces CodeActAgent's Python code generation + exec() with a structured
XML tool-calling protocol. The LLM emits <function_calls> blocks, the agent
parses them, executes the tools, and feeds <function_results> back.

Uses the same event system and workflow structure as CodeActAgent for
compatibility with DroidAgent's execute_task() method.
"""

import asyncio
import copy
import logging
import os
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from opentelemetry import trace
from pydantic import BaseModel

from droidrun.agent.action_result import ActionResult
from droidrun.agent.codeact.events import (
    FastAgentEndEvent,
    FastAgentInputEvent,
    FastAgentOutputEvent,
    FastAgentResponseEvent,
    FastAgentToolCallEvent,
)
from droidrun.agent.codeact.xml_parser import (
    CLOSE_TAG,
    OPEN_TAG,
    ToolResult,
    format_tool_results,
    parse_tool_calls,
)
from droidrun.agent.common.constants import LLM_HISTORY_LIMIT
from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.droid.events import (
    ExternalUserMessageAppliedEvent,
    ExternalUserMessageDroppedEvent,
)
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.chat_utils import limit_history
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tracing_setup import record_langfuse_screenshot
from droidrun.config_manager.config_manager import AgentConfig, TracingConfig
from droidrun.config_manager.prompt_loader import PromptLoader
from droidrun.tools.driver.base import DeviceDisconnectedError

if TYPE_CHECKING:
    from droidrun.agent.action_context import ActionContext
    from droidrun.agent.droid import DroidAgentState
    from droidrun.agent.tool_registry import ToolRegistry
    from droidrun.tools.ui.provider import StateProvider

logger = logging.getLogger("droidrun")


class FastAgent(Workflow):
    """Agent that uses XML tool-calling instead of code generation.

    Uses ReAct cycle: Thought -> Tool Call -> Observation -> repeat until complete().
    Messages stored as list[ChatMessage] to preserve thinking tokens across turns.
    """

    def __init__(
        self,
        llm: LLM,
        agent_config: AgentConfig,
        registry: "ToolRegistry",
        action_ctx: "ActionContext",
        state_provider: "StateProvider",
        save_trajectory: str = "none",
        debug: bool = False,
        shared_state: Optional["DroidAgentState"] = None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        tracing_config: TracingConfig | None = None,
        *args,
        **kwargs,
    ):
        assert llm, "llm must be provided."
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.agent_config = agent_config
        self.config = agent_config.fast_agent
        self.max_steps = agent_config.max_steps
        self.vision = agent_config.fast_agent.vision
        self.debug = debug
        self.registry = registry
        self.action_ctx = action_ctx
        self.state_provider = state_provider
        self.save_trajectory = save_trajectory
        self._stream_screenshots = os.environ.get(
            "DROIDRUN_STREAM_SCREENSHOTS", ""
        ).lower() in ("1", "true")
        self.shared_state = shared_state
        self.output_model = output_model
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config

        self.system_prompt: ChatMessage | None = None
        self.tool_call_counter = 0
        self.remembered_info: list[str] | None = None

        # Build tool descriptions and param types from registry
        self.tool_descriptions = self.registry.get_tool_descriptions_xml()
        self.param_types = self.registry.get_param_types()

        self._available_secrets = []
        self._output_schema = None
        if self.output_model is not None:
            self._output_schema = self.output_model.model_json_schema()

        logger.debug("FastAgent initialized.")

    async def _build_system_prompt(self) -> ChatMessage:
        """Build system prompt message."""
        template_context = {
            "tool_descriptions": self.tool_descriptions,
            "available_secrets": self._available_secrets,
            "available_tools": set(self.registry.tools.keys()),
            "device_prompt_name": self.action_ctx.driver.device_prompt_name,
            "variables": (
                self.shared_state.custom_variables if self.shared_state else {}
            ),
            "output_schema": self._output_schema,
            "parallel_tools": self.config.parallel_tools,
            "vision": self.vision,
        }

        custom_system_prompt = self.prompt_resolver.get_prompt("fast_agent_system")
        if custom_system_prompt:
            system_text = PromptLoader.render_template(
                custom_system_prompt,
                template_context,
            )
        else:
            system_text = await PromptLoader.load_prompt(
                self.agent_config.get_fast_agent_system_prompt_path(),
                template_context,
            )
        return ChatMessage(role="system", content=system_text)

    async def _build_user_prompt(self, goal: str) -> ChatMessage:
        """Build initial user prompt message."""
        custom_user_prompt = self.prompt_resolver.get_prompt("fast_agent_user")
        if custom_user_prompt:
            user_text = PromptLoader.render_template(
                custom_user_prompt,
                {
                    "goal": goal,
                    "variables": (
                        self.shared_state.custom_variables if self.shared_state else {}
                    ),
                },
            )
        else:
            user_text = await PromptLoader.load_prompt(
                self.agent_config.get_fast_agent_user_prompt_path(),
                {
                    "goal": goal,
                    "variables": (
                        self.shared_state.custom_variables if self.shared_state else {}
                    ),
                },
            )
        return ChatMessage(role="user", content=user_text)

    @step
    async def prepare_chat(self, ctx: Context, ev: StartEvent) -> FastAgentInputEvent:
        """Initialize message history with goal."""
        logger.debug("Preparing chat for task execution...")

        # Get available secrets
        if self.action_ctx and self.action_ctx.credential_manager:
            self._available_secrets = (
                await self.action_ctx.credential_manager.get_keys()
            )

        # Build system prompt (lazy load)
        if self.system_prompt is None:
            self.system_prompt = await self._build_system_prompt()

        # Get goal and build user message
        user_input = ev.get("input", default=None)
        assert user_input, "User input cannot be empty."

        user_message = await self._build_user_prompt(user_input)
        self.shared_state.message_history.clear()
        self.shared_state.message_history.append(user_message)

        # Store remembered info if provided
        remembered_info = ev.get("remembered_info", default=None)
        if remembered_info:
            self.remembered_info = remembered_info
            memory_text = "\n### Remembered Information:\n"
            for idx, item in enumerate(remembered_info, 1):
                memory_text += f"{idx}. {item}\n"
            self.shared_state.message_history[0].blocks.append(
                TextBlock(text=memory_text)
            )

        return FastAgentInputEvent()

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: FastAgentInputEvent
    ) -> FastAgentResponseEvent | FastAgentEndEvent:
        """Get device state, call LLM, return response."""
        ctx.write_event_to_stream(ev)

        # Check then bump step counter
        if self.shared_state.step_number >= self.max_steps:
            pending = self.shared_state.drain_user_messages()
            if pending:
                logger.warning(
                    f"⚠️ Dropping {len(pending)} external user message(s) at max steps"
                )
                ctx.write_event_to_stream(
                    ExternalUserMessageDroppedEvent(
                        message_ids=[m.id for m in pending],
                        reason="max_steps_reached",
                        step_number=self.shared_state.step_number,
                    )
                )
            event = FastAgentEndEvent(
                success=False,
                reason=f"Reached max step count of {self.max_steps} steps",
                tool_call_count=self.tool_call_counter,
            )
            ctx.write_event_to_stream(event)
            return event

        self.shared_state.step_number += 1
        logger.info(f"🔄 Step {self.shared_state.step_number}/{self.max_steps}")

        # Capture screenshot if needed
        screenshot = None
        if self.vision or self._stream_screenshots or self.save_trajectory != "none":
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
                    await ctx.store.set("screenshot", screenshot)
                    logger.debug("📸 Screenshot captured for FastAgent")
            except DeviceDisconnectedError:
                raise
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")

        # Get device state
        try:
            ui_state = await self.state_provider.get_state()
            self.action_ctx.ui = ui_state

            # Update shared state (previous ← current, current ← new)
            self.shared_state.previous_formatted_device_state = (
                self.shared_state.formatted_device_state
            )
            self.shared_state.formatted_device_state = ui_state.formatted_text
            self.shared_state.focused_text = ui_state.focused_text
            self.shared_state.a11y_tree = ui_state.elements
            self.shared_state.phone_state = ui_state.phone_state

            # Extract and store package/app name
            self.shared_state.update_current_app(
                package_name=ui_state.phone_state.get("packageName", "Unknown"),
                activity_name=ui_state.phone_state.get("currentApp", "Unknown"),
            )

            # Stream formatted state for trajectory
            ctx.write_event_to_stream(RecordUIStateEvent(ui_state=ui_state.elements))

        except DeviceDisconnectedError:
            raise
        except Exception as e:
            logger.warning(f"⚠️ Error retrieving state from the connected device: {e}")
            if self.debug:
                logger.error("State retrieval error details:", exc_info=True)

        # Limit history and build ephemeral copy for LLM
        limited_history = limit_history(
            self.shared_state.message_history,
            LLM_HISTORY_LIMIT * 2,
            preserve_first=True,
        )
        messages_to_send = [self.system_prompt] + copy.deepcopy(limited_history)

        # Inject device state and screenshot into the copy (not the original)
        user_indices = [
            i for i, msg in enumerate(messages_to_send) if msg.role == "user"
        ]
        if user_indices:
            last_user_idx = user_indices[-1]

            # Current device state → last user message
            current_state = self.shared_state.formatted_device_state.strip()
            if current_state:
                messages_to_send[last_user_idx].blocks.append(
                    TextBlock(
                        text=f"\n<device_state>\n{current_state}\n</device_state>\n"
                    )
                )

            # Screenshot → last user message
            if self.vision and screenshot:
                messages_to_send[last_user_idx].blocks.append(
                    ImageBlock(image=screenshot)
                )

            # Previous device state → second-to-last user message
            if len(user_indices) >= 2:
                second_last_idx = user_indices[-2]
                prev_state = self.shared_state.previous_formatted_device_state.strip()
                if prev_state:
                    messages_to_send[second_last_idx].blocks.append(
                        TextBlock(
                            text=f"\n<previous_device_state>\n{prev_state}\n</previous_device_state>\n"
                        )
                    )

        # Call LLM
        logger.info("FastAgent response:", extra={"color": "yellow"})
        response = await acall_with_retries(
            self.llm, messages_to_send, stream=self.agent_config.streaming
        )

        if response is None:
            return FastAgentEndEvent(
                success=False,
                reason="LLM response is None. This is a critical error.",
                tool_call_count=self.tool_call_counter,
            )

        # Extract usage
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        # Store assistant response (preserves ThinkingBlock, additional_kwargs, etc.)
        self.shared_state.message_history.append(response.message)
        response_text = response.message.content

        # Parse tool calls from response
        thought, tool_calls = parse_tool_calls(response_text, self.param_types)

        # Extract just the <function_calls> blocks for the event
        tool_calls_xml = None
        if tool_calls:
            blocks = []
            for part in response_text.split(OPEN_TAG)[1:]:
                close_idx = part.find(CLOSE_TAG)
                if close_idx != -1:
                    blocks.append(OPEN_TAG + part[: close_idx + len(CLOSE_TAG)])
            tool_calls_xml = "\n".join(blocks) if blocks else None

        # Store tool calls in context for execute step (avoid re-parsing)
        if tool_calls:
            await ctx.store.set("pending_tool_calls", tool_calls)

        # Update unified state
        self.shared_state.last_thought = thought

        event = FastAgentResponseEvent(
            thought=thought,
            code=tool_calls_xml,
            usage=usage,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_llm_output(
        self, ctx: Context, ev: FastAgentResponseEvent
    ) -> FastAgentToolCallEvent | FastAgentInputEvent:
        """Route to execution or request tool call if missing."""
        has_tool_calls = ev.code is not None

        if not ev.thought:
            logger.warning("LLM provided tool calls without reasoning.")
            no_thoughts_text = (
                "Your previous response called tools without explaining your reasoning first. "
                "Remember to always describe your thought process and plan *before* calling tools.\n\n"
                "The tool calls you made will be executed below.\n\n"
                "Now, describe the next step you will take to address the original goal."
            )
            self.shared_state.message_history.append(
                ChatMessage(role="user", content=no_thoughts_text)
            )
        else:
            logger.debug(f"Reasoning: {ev.thought}")

        if has_tool_calls:
            event = FastAgentToolCallEvent(tool_calls_repr=ev.code)
            ctx.write_event_to_stream(event)
            return event
        else:
            # No tool calls — ask for them
            no_tools_text = (
                "No tool calls were provided. If you want to mark the task as complete "
                "(whether it failed or succeeded), use the `complete` tool:\n\n"
                "<function_calls>\n"
                '<invoke name="complete">\n'
                '<parameter name="success">true</parameter>\n'
                '<parameter name="message">Explanation here</parameter>\n'
                "</invoke>\n"
                "</function_calls>"
            )
            self.shared_state.message_history.append(
                ChatMessage(role="user", content=no_tools_text)
            )
            return FastAgentInputEvent()

    @step
    async def execute_code(
        self, ctx: Context, ev: FastAgentToolCallEvent
    ) -> FastAgentOutputEvent | FastAgentEndEvent:
        """Execute parsed tool calls and return results."""
        tool_calls = await ctx.store.get("pending_tool_calls", [])

        if not tool_calls:
            event = FastAgentOutputEvent(output="No tool calls to execute.")
            ctx.write_event_to_stream(event)
            return event

        results: list[ToolResult] = []

        for call in tool_calls:
            logger.debug(f"Executing: {call.name}({call.parameters})")
            self.tool_call_counter += 1

            # Skip execution if parsing failed
            if call.error:
                action_result = ActionResult(
                    success=False,
                    summary=f"Invalid arguments for {call.name}: {call.error}",
                )
            else:
                # Dispatch via registry
                action_result = await self.registry.execute(
                    call.name, call.parameters, self.action_ctx, workflow_ctx=ctx
                )
            results.append(
                ToolResult(
                    name=call.name,
                    output=action_result.summary,
                    is_error=not action_result.success,
                )
            )

            # Check if complete() was called successfully
            if self.shared_state.finished:
                if self.shared_state.pending_user_messages:
                    logger.info(
                        "⏸️ complete() called but external messages pending, continuing",
                        extra={"color": "cyan"},
                    )
                    self.shared_state.finished = False
                    self.shared_state.success = None
                    self.shared_state.answer = ""
                    results_xml = format_tool_results(results)
                    event = FastAgentOutputEvent(output=results_xml)
                    ctx.write_event_to_stream(event)
                    return event

                logger.debug("✅ Task marked as complete via complete() tool")

                success = (
                    self.shared_state.success
                    if self.shared_state.success is not None
                    else False
                )
                reason = (
                    self.shared_state.answer
                    if self.shared_state.answer
                    else "Task completed without reason"
                )
                self.shared_state.finished = False

                event = FastAgentEndEvent(
                    success=success,
                    reason=reason,
                    tool_call_count=self.tool_call_counter,
                )
                ctx.write_event_to_stream(event)
                return event

        # Format results
        results_xml = format_tool_results(results)
        logger.info("💡 Tool results:", extra={"color": "dim"})
        logger.info(f"{results_xml}")
        await asyncio.sleep(self.agent_config.after_sleep_action)

        # Update remembered info
        self.remembered_info = self.shared_state.fast_memory

        event = FastAgentOutputEvent(output=results_xml)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_execution_result(
        self, ctx: Context, ev: FastAgentOutputEvent
    ) -> FastAgentInputEvent:
        """Add execution result to history and loop back."""
        output = ev.output or "Tool executed, but produced no output."

        drained = self.shared_state.drain_user_messages()
        if drained:
            external_block = "\n".join(
                f"<external_user_message>\n{m.message}\n</external_user_message>"
                for m in drained
            )
            output += "\n" + external_block
            logger.info(
                f"📩 Applied {len(drained)} external user message(s)",
                extra={"color": "cyan"},
            )
            ctx.write_event_to_stream(
                ExternalUserMessageAppliedEvent(
                    message_ids=[m.id for m in drained],
                    consumer="fast_agent",
                    step_number=self.shared_state.step_number,
                )
            )

        # Add results (+ any external messages) as a single user message
        self.shared_state.message_history.append(
            ChatMessage(role="user", content=output)
        )

        return FastAgentInputEvent()

    @step
    async def finalize(self, ev: FastAgentEndEvent, ctx: Context) -> StopEvent:
        self.shared_state.finished = False
        ctx.write_event_to_stream(ev)

        return StopEvent(
            result={
                "success": ev.success,
                "reason": ev.reason,
                "tool_call_count": ev.tool_call_count,
            }
        )
