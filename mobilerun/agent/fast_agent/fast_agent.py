"""FastAgent — tool-calling agent for device interaction.

Uses a structured XML tool-calling protocol by default. The LLM emits
<function_calls> blocks, the agent parses them, executes the tools via
ToolRegistry, and feeds <function_results> back as user messages. LLMs that
support native function calling receive the tools as API tool definitions
instead and get tool-result messages back.
"""

import asyncio
import copy
import logging
import os
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from mobilerun_core_local.driver.base import DeviceDisconnectedError
from opentelemetry import trace
from pydantic import BaseModel

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.common.constants import LLM_HISTORY_LIMIT
from mobilerun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agent.droid.events import (
    ExternalUserMessageAppliedEvent,
    ExternalUserMessageDroppedEvent,
)
from mobilerun.agent.fast_agent.events import (
    FastAgentEndEvent,
    FastAgentInputEvent,
    FastAgentOutputEvent,
    FastAgentResponseEvent,
    FastAgentToolCallEvent,
)
from mobilerun.agent.fast_agent.native_tools import (
    tool_result_messages,
)
from mobilerun.agent.fast_agent.xml_parser import (
    ToolCallParseResult,
    ToolCallParseStatus,
    ToolResult,
    extract_add_memory,
    format_tool_calls,
    format_tool_results,
)
from mobilerun.agent.selectors import ActionSelector, LlamaIndexLLMSelector, SelectionRequest
from mobilerun.agent.utils.chat_utils import limit_history
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.agent.utils.tracing_setup import record_langfuse_screenshot
from mobilerun.config_manager.config_manager import AgentConfig, TracingConfig
from mobilerun.config_manager.prompt_loader import PromptLoader
from mobilerun.tools.ui.provider import (
    resize_model_screenshot_async,
)

if TYPE_CHECKING:
    from mobilerun.agent.action_context import ActionContext
    from mobilerun.agent.droid import MobileAgentState
    from mobilerun.agent.tool_registry import ToolRegistry
    from mobilerun.tools.ui.provider import StateProvider

logger = logging.getLogger("mobilerun")

_MALFORMED_TOOL_CALL_LIMIT = 3
_TOOL_CALLING_MODES = ("auto", "xml")
_NATIVE_NO_TOOL_CALL = (
    "No tool was called. Call one of the provided tools to continue. To mark the "
    "task as complete (whether it failed or succeeded), call the `complete` tool."
)
_NATIVE_CONTINUE = "The tool results are above. Continue with the next step."


def _malformed_tool_call_correction(attempt: int) -> str:
    """Build a focused retry instruction without echoing malformed model output."""
    return (
        "Your previous response contained tool-call markup that could not be parsed "
        f"(attempt {attempt}/{_MALFORMED_TOOL_CALL_LIMIT}). No tool was executed.\n\n"
        "Repeat the intended call using only the ASCII XML tags shown below. Replace "
        "the placeholder names and value with the intended tool and arguments. Do not "
        "use provider-specific markers, full-width punctuation, or alternate tag names, "
        "and write tool-call tags only inside the call, never in your "
        "explanation. Write <add_memory> before the <function_calls> block, never "
        "inside it; memory written inside the block was not saved.\n\n"
        "<function_calls>\n"
        '<invoke name="tool_name">\n'
        '<parameter name="parameter_name">value</parameter>\n'
        "</invoke>\n"
        "</function_calls>"
    )


def _truncate_message_text(message: ChatMessage, end: int) -> ChatMessage:
    """Cut text at offset ``end`` of ``message.content``, keeping non-text blocks."""
    blocks = []
    start = 0
    for block in message.blocks:
        if not isinstance(block, TextBlock):
            blocks.append(block)
            continue
        text = block.text
        if start + len(text) <= end:
            blocks.append(block)
        elif start < end:
            blocks.append(block.model_copy(update={"text": text[: end - start]}))
        # ChatMessage.content joins text blocks with "\n".
        start += len(text) + 1
    return message.model_copy(update={"blocks": blocks})


def _format_results(results: list[ToolResult], ignored_blocks: int) -> str:
    output = format_tool_results(results)
    if ignored_blocks:
        output += (
            "\nOnly the first <function_calls> block of your response was executed; "
            f"{ignored_blocks} later block(s) were ignored. Emit exactly one "
            "<function_calls> block per response."
        )
    return output


class FastAgent(Workflow):
    """Agent that uses XML tool-calling instead of code generation.

    Uses ReAct cycle: Thought -> Tool Call -> Observation -> repeat until complete().
    Messages stored as list[ChatMessage] to preserve thinking tokens across turns.
    """

    def __init__(
        self,
        action_selector: ActionSelector | None = None,
        agent_config: AgentConfig | None = None,
        registry: "ToolRegistry | None" = None,
        action_ctx: "ActionContext | None" = None,
        state_provider: "StateProvider | None" = None,
        save_trajectory: str = "none",
        debug: bool = False,
        shared_state: Optional["MobileAgentState"] = None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        tracing_config: TracingConfig | None = None,
        *args,
        **kwargs,
    ):
        if "llm" in kwargs:
            logger.debug(
                "FastAgent(llm=...) is deprecated. Use action_selector=LlamaIndexLLMSelector(llm)."
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
        if registry is None:
            raise TypeError("registry is required")
        super().__init__(*args, **kwargs)

        self.action_selector = action_selector
        self.agent_config = agent_config
        self.config = agent_config.fast_agent
        self.max_steps = agent_config.max_steps
        self.vision = agent_config.fast_agent.vision
        self.debug = debug
        self.registry = registry
        self.action_ctx = action_ctx
        self.state_provider = state_provider
        self.save_trajectory = save_trajectory
        self._stream_screenshots = (
            os.environ.get("MOBILERUN_STREAM_SCREENSHOTS")
            or os.environ.get("DROIDRUN_STREAM_SCREENSHOTS")
            or ""
        ).lower() in ("1", "true")
        self.shared_state = shared_state
        self.output_model = output_model
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config

        self.system_prompt: ChatMessage | None = None
        self.tool_call_counter = 0
        self._consecutive_malformed_tool_calls = 0

        tool_calling = getattr(self.config, "tool_calling", "auto")
        if tool_calling not in _TOOL_CALLING_MODES:
            raise ValueError(
                f"fast_agent.tool_calling must be one of {_TOOL_CALLING_MODES}, "
                f"got {tool_calling!r}"
            )
        self.native_tools = (
            tool_calling == "auto" and self.action_selector.supports_native_tool_calls
        )

        # Build tool descriptions and param types from registry
        self.tool_descriptions = self.registry.get_tool_descriptions_xml()
        self.param_types = self.registry.get_param_types()
        self.tool_specs = (
            self.registry.get_function_tool_specs() if self.native_tools else None
        )

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
            "variables": (
                self.shared_state.custom_variables if self.shared_state else {}
            ),
            "output_schema": self._output_schema,
            "parallel_tools": self.config.parallel_tools,
            "native_tools": self.native_tools,
            "vision": self.vision,
            "platform": self.shared_state.platform,
            "screenshot_only": bool(
                getattr(self.state_provider, "requires_coordinate_tools", False)
            ),
        }

        custom_prompt = self.prompt_resolver.get_prompt("fast_agent_system")
        if custom_prompt:
            system_text = PromptLoader.render_template(custom_prompt, template_context)
        else:
            system_text = await PromptLoader.load_prompt(
                self.agent_config.get_fast_agent_system_prompt_path(), template_context,
            )
        return ChatMessage(role="system", content=system_text)

    async def _build_user_prompt(self, goal: str) -> ChatMessage:
        """Build initial user prompt message."""
        template_context = {
            "goal": goal,
            "variables": (
                self.shared_state.custom_variables if self.shared_state else {}
            ),
        }
        custom_prompt = self.prompt_resolver.get_prompt("fast_agent_user")
        if custom_prompt:
            user_text = PromptLoader.render_template(custom_prompt, template_context)
        else:
            user_text = await PromptLoader.load_prompt(
                self.agent_config.get_fast_agent_user_prompt_path(), template_context,
            )
        return ChatMessage(role="user", content=user_text)

    @step
    async def prepare_chat(self, ctx: Context, ev: StartEvent) -> FastAgentInputEvent:
        """Initialize message history with goal."""
        logger.debug("Preparing chat for task execution...")
        self._consecutive_malformed_tool_calls = 0

        # Get available secrets (only if type_secret is actually in the registry)
        if (
            self.registry
            and "type_secret" in self.registry.tools
            and self.action_ctx
            and self.action_ctx.credential_manager
        ):
            self._available_secrets = (
                await self.action_ctx.credential_manager.get_keys()
            )

        # Build system prompt (lazy load)
        if self.system_prompt is None:
            self.system_prompt = await self._build_system_prompt()

        # Get goal and build user message
        user_input = ev.get("input", default=None)
        assert user_input, "User input cannot be empty."
        await ctx.store.set("selection_goal", user_input)

        user_message = await self._build_user_prompt(user_input)
        self.shared_state.message_history.clear()
        self.shared_state.message_history.append(user_message)

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
        ui_state = None
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
            err_desc = str(e) or type(e).__name__
            logger.warning(
                f"⚠️ Error retrieving state from the connected device: {err_desc}"
            )
            if self.debug:
                logger.error("State retrieval error details:", exc_info=True)

        # Limit history and build ephemeral copy for LLM. A native step adds
        # a tool-result message per call besides the assistant and user turns.
        limited_history = limit_history(
            self.shared_state.message_history,
            LLM_HISTORY_LIMIT * (3 if self.native_tools else 2),
            preserve_first=True,
        )
        messages_to_send = [self.system_prompt] + copy.deepcopy(limited_history)

        # Inject device state and screenshot into the copy (not the original)
        user_indices = [
            i for i, msg in enumerate(messages_to_send) if msg.role == "user"
        ]
        if user_indices:
            last_user_idx = user_indices[-1]

            # Accumulated agent memory → last user message
            current_memory = (self.shared_state.agent_memory or "").strip()
            if current_memory:
                messages_to_send[last_user_idx].blocks.append(
                    TextBlock(text=f"\n<memory>\n{current_memory}\n</memory>\n")
                )

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
                screenshot = await resize_model_screenshot_async(
                    self.state_provider, screenshot
                )
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

        # Select actions
        logger.info("FastAgent response:", extra={"color": "yellow"})
        tools = self.registry.get_signatures()
        if self.action_selector.requires_action_candidates:
            options = await self.state_provider.build_action_candidates(
                self.action_ctx, tools, include_device_actions=True,
            )
        else:
            options = self.state_provider.get_action_candidates(ui_state) if ui_state is not None else ()
        request = SelectionRequest(
            goal=await ctx.store.get("selection_goal", default=self.shared_state.instruction),
            state=ui_state,
            memory=self.shared_state.agent_memory,
            history=limited_history,
            screenshot=screenshot if self.vision else None,
            tools=tools,
            variables=self.shared_state.custom_variables,
            messages=messages_to_send,
            param_types=self.param_types,
            stream=self.agent_config.streaming,
            options=options,
            protocol="native" if self.native_tools else "xml",
            tool_specs=self.tool_specs,
            parallel_tools=self.config.parallel_tools,
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
            return FastAgentEndEvent(
                success=False,
                reason=self.action_selector.abstention_reason,
                tool_call_count=self.tool_call_counter,
            )

        usage = selection.usage
        parse_result = selection.parse_result or ToolCallParseResult(
            thought=selection.thought,
            calls=selection.actions,
            status=ToolCallParseStatus.VALID
            if selection.actions
            else ToolCallParseStatus.NO_MARKUP,
        )
        thought = parse_result.thought
        tool_calls = parse_result.calls

        # Store assistant response (preserves ThinkingBlock, additional_kwargs, etc.)
        # up to the end of its first block, so later text is never replayed.
        message = (
            selection.response.message
            if selection.response is not None
            else ChatMessage(
                role="assistant", content=thought + "\n" + format_tool_calls(tool_calls)
            )
        )
        response_text = message.content or ""
        if parse_result.end is not None and response_text[parse_result.end :].strip():
            message = _truncate_message_text(message, parse_result.end)
        if parse_result.ignored_blocks:
            logger.warning(
                "Ignoring %d <function_calls> block(s) after the first one",
                parse_result.ignored_blocks,
            )
        self.shared_state.message_history.append(message)

        # Extract <add_memory> from thought text and append to unified memory
        memory_update = extract_add_memory(thought)
        if memory_update:
            self.shared_state.append_memory(memory_update)
        if selection.memory_update:
            self.shared_state.append_memory(selection.memory_update)

        # Store parsed calls for logging/trajectory output. This uses the
        # executable representation so ignored XML blocks are not shown as
        # pending work.
        tool_calls_xml = format_tool_calls(tool_calls) if tool_calls else None
        if self.native_tools and tool_calls_xml:
            # Native calls are not part of the streamed text.
            logger.info(tool_calls_xml)

        # Store tool calls in context for execute step (avoid re-parsing)
        if tool_calls:
            await ctx.store.set("pending_tool_calls", tool_calls)
            await ctx.store.set("ignored_tool_call_blocks", parse_result.ignored_blocks)

        # Update unified state
        self.shared_state.last_thought = thought

        event = FastAgentResponseEvent(
            thought=thought,
            code=tool_calls_xml,
            usage=usage,
            tool_call_status=parse_result.status,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_llm_output(
        self, ctx: Context, ev: FastAgentResponseEvent
    ) -> FastAgentToolCallEvent | FastAgentInputEvent | FastAgentEndEvent:
        """Route to execution or request tool call if missing."""
        has_tool_calls = ev.code is not None

        if ev.tool_call_status == ToolCallParseStatus.MALFORMED and not has_tool_calls:
            self._consecutive_malformed_tool_calls += 1
            attempt = self._consecutive_malformed_tool_calls
            logger.warning(
                "Malformed tool-call markup detected (%d/%d)",
                attempt,
                _MALFORMED_TOOL_CALL_LIMIT,
            )

            if attempt >= _MALFORMED_TOOL_CALL_LIMIT:
                pending = self.shared_state.drain_user_messages()
                if pending:
                    logger.warning(
                        "⚠️ Dropping %d external user message(s) at malformed tool-call limit",
                        len(pending),
                    )
                    ctx.write_event_to_stream(
                        ExternalUserMessageDroppedEvent(
                            message_ids=[message.id for message in pending],
                            reason="malformed_tool_call_limit_reached",
                            step_number=self.shared_state.step_number,
                        )
                    )
                event = FastAgentEndEvent(
                    success=False,
                    reason=(
                        "Model produced malformed tool-call markup "
                        f"{_MALFORMED_TOOL_CALL_LIMIT} consecutive times; stopped to prevent "
                        "a retry loop. Switch models or verify tool-call protocol compatibility."
                    ),
                    tool_call_count=self.tool_call_counter,
                )
                return event

            self.shared_state.message_history.append(
                ChatMessage(
                    role="user",
                    content=_malformed_tool_call_correction(attempt),
                )
            )
            return FastAgentInputEvent()

        self._consecutive_malformed_tool_calls = 0

        if not ev.thought and self.native_tools:
            # Native calls must be answered by tool results before any user
            # message, and reasoning models may reason without visible text.
            logger.debug("LLM provided native tool calls without visible text.")
        elif not ev.thought:
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
        elif self.native_tools:
            self.shared_state.message_history.append(
                ChatMessage(role="user", content=_NATIVE_NO_TOOL_CALL)
            )
            return FastAgentInputEvent()
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
        ignored_blocks = await ctx.store.get("ignored_tool_call_blocks", 0)

        if not tool_calls:
            event = FastAgentOutputEvent(output="No tool calls to execute.")
            ctx.write_event_to_stream(event)
            return event

        results: list[ToolResult] = []

        for index, call in enumerate(tool_calls):
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
                    await self._store_native_results(ctx, tool_calls, results)
                    results_xml = _format_results(results, ignored_blocks)
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
                return event

            # Later calls were planned assuming this one succeeded.
            if not action_result.success:
                skipped = tool_calls[index + 1 :]
                if skipped:
                    logger.info(
                        "Skipping %d tool call(s) after %s failed",
                        len(skipped),
                        call.name,
                    )
                    results.extend(
                        ToolResult(
                            name=later.name,
                            output=f"Not executed because {call.name} failed.",
                            is_error=True,
                        )
                        for later in skipped
                    )
                break

        # Format results
        await self._store_native_results(ctx, tool_calls, results)
        results_xml = _format_results(results, ignored_blocks)
        logger.info("💡 Tool results:", extra={"color": "dim"})
        logger.info(f"{results_xml}")
        await asyncio.sleep(self.agent_config.after_sleep_action)

        event = FastAgentOutputEvent(output=results_xml)
        ctx.write_event_to_stream(event)
        return event

    async def _store_native_results(
        self, ctx: Context, tool_calls: list, results: list[ToolResult]
    ) -> None:
        if self.native_tools:
            await ctx.store.set(
                "native_tool_results", tool_result_messages(tool_calls, results)
            )

    @step
    async def handle_execution_result(
        self, ctx: Context, ev: FastAgentOutputEvent
    ) -> FastAgentInputEvent:
        """Add execution result to history and loop back."""
        output = ev.output or "Tool executed, but produced no output."
        if self.native_tools:
            # Results go back as tool messages; the user turn carries state.
            self.shared_state.message_history.extend(
                await ctx.store.get("native_tool_results", [])
            )
            await ctx.store.set("native_tool_results", [])
            output = _NATIVE_CONTINUE

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
        # All exit paths publish their terminal event exactly once, here.
        ctx.write_event_to_stream(ev)

        return StopEvent(
            result={
                "success": ev.success,
                "reason": ev.reason,
                "tool_call_count": ev.tool_call_count,
            }
        )
