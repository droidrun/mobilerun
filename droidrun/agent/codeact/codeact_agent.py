import asyncio
import copy
import inspect
import logging
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from opentelemetry import trace
from pydantic import BaseModel

from droidrun.agent.codeact.events import (
    CodeActCodeEvent,
    CodeActEndEvent,
    CodeActInputEvent,
    CodeActOutputEvent,
    CodeActResponseEvent,
)
from droidrun.agent.common.constants import LLM_HISTORY_LIMIT
from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.chat_utils import (
    extract_code_and_thought,
    limit_history,
)
from droidrun.agent.utils.executer import ExecuterState, SimpleCodeExecutor
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tracing_setup import record_langfuse_screenshot
from droidrun.config_manager.config_manager import AgentConfig, TracingConfig
from droidrun.config_manager.path_resolver import PathResolver
from droidrun.config_manager.prompt_loader import PromptLoader
from droidrun.tools.driver.base import DeviceDisconnectedError

# Legacy codeact prompt paths (used when code_exec=true but config points to tools defaults)
_LEGACY_SYSTEM_PROMPT = "config/prompts/codeact/system.jinja2"
_LEGACY_USER_PROMPT = "config/prompts/codeact/user.jinja2"
_TOOLS_SYSTEM_PROMPT = "config/prompts/codeact/tools_system.jinja2"
_TOOLS_USER_PROMPT = "config/prompts/codeact/tools_user.jinja2"

if TYPE_CHECKING:
    from droidrun.agent.action_context import ActionContext
    from droidrun.agent.droid import DroidAgentState
    from droidrun.agent.tool_registry import ToolRegistry
    from droidrun.tools.ui.provider import StateProvider

logger = logging.getLogger("droidrun")


class CodeActAgent(Workflow):
    """
    Agent that generates and executes Python code using atomic actions.

    Uses ReAct cycle: Thought -> Code -> Observation -> repeat until complete().
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
        safe_execution_config=None,
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
        self.shared_state = shared_state
        self.output_model = output_model
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config

        self.system_prompt: ChatMessage | None = None
        self.code_exec_counter = 0
        self.remembered_info: list[str] | None = None

        # Build tool_list for code executor from registry
        # Each tool is wrapped to pass ctx automatically
        self.tool_list = {}
        for tool_name, entry in self.registry.tools.items():
            func = entry.fn
            if inspect.iscoroutinefunction(func):

                async def async_wrapper(*a, f=func, ac=action_ctx, **kw):
                    return await f(*a, ctx=ac, **kw)

                self.tool_list[tool_name] = async_wrapper
            else:

                def sync_wrapper(*a, f=func, ac=action_ctx, **kw):
                    return f(*a, ctx=ac, **kw)

                self.tool_list[tool_name] = sync_wrapper

        # Build tool descriptions
        self.tool_descriptions = self.registry.get_tool_descriptions_text()

        self._available_secrets = []
        self._output_schema = None
        if self.output_model is not None:
            self._output_schema = self.output_model.model_json_schema()

        # Initialize code executor
        safe_mode = self.config.safe_execution
        safe_config = safe_execution_config

        self.executor = SimpleCodeExecutor(
            locals={},
            tools=self.tool_list,
            globals={"__builtins__": __builtins__},
            safe_mode=safe_mode,
            allowed_modules=(
                safe_config.get_allowed_modules() if safe_config and safe_mode else None
            ),
            blocked_modules=(
                safe_config.get_blocked_modules() if safe_config and safe_mode else None
            ),
            allowed_builtins=(
                safe_config.get_allowed_builtins()
                if safe_config and safe_mode
                else None
            ),
            blocked_builtins=(
                safe_config.get_blocked_builtins()
                if safe_config and safe_mode
                else None
            ),
            event_loop=None,
        )

        logger.debug("CodeActAgent initialized.")

    async def _build_system_prompt(self) -> ChatMessage:
        """Build system prompt message."""
        # Build template context with available tools for conditional examples
        template_context = {
            "tool_descriptions": self.tool_descriptions,
            "available_secrets": self._available_secrets,
            "available_tools": set(self.registry.tools.keys()),
            "device_prompt_name": self.action_ctx.driver.device_prompt_name,
            "variables": (
                self.shared_state.custom_variables if self.shared_state else {}
            ),
            "output_schema": self._output_schema,
        }

        custom_system_prompt = self.prompt_resolver.get_prompt("fast_agent_system")
        if custom_system_prompt:
            system_text = PromptLoader.render_template(
                custom_system_prompt,
                template_context,
            )
        else:
            # If config still points to tools template, use legacy codeact template
            prompt_path = self.agent_config.fast_agent.system_prompt
            if prompt_path == _TOOLS_SYSTEM_PROMPT:
                prompt_path = _LEGACY_SYSTEM_PROMPT
            system_text = await PromptLoader.load_prompt(
                str(PathResolver.resolve(prompt_path, must_exist=True)),
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
            # If config still points to tools template, use legacy codeact template
            prompt_path = self.agent_config.fast_agent.user_prompt
            if prompt_path == _TOOLS_USER_PROMPT:
                prompt_path = _LEGACY_USER_PROMPT
            user_text = await PromptLoader.load_prompt(
                str(PathResolver.resolve(prompt_path, must_exist=True)),
                {
                    "goal": goal,
                    "variables": (
                        self.shared_state.custom_variables if self.shared_state else {}
                    ),
                },
            )
        return ChatMessage(role="user", content=user_text)

    @step
    async def prepare_chat(self, ctx: Context, ev: StartEvent) -> CodeActInputEvent:
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
            # Append to first user message
            self.shared_state.message_history[0].blocks.append(
                TextBlock(text=memory_text)
            )

        return CodeActInputEvent()

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: CodeActInputEvent
    ) -> CodeActResponseEvent | CodeActEndEvent:
        """Get device state, call LLM, return response."""
        ctx.write_event_to_stream(ev)

        # Check then bump step counter
        if self.shared_state.step_number >= self.max_steps:
            event = CodeActEndEvent(
                success=False,
                reason=f"Reached max step count of {self.max_steps} steps",
                code_executions=self.code_exec_counter,
            )
            ctx.write_event_to_stream(event)
            return event

        self.shared_state.step_number += 1
        logger.info(f"🔄 Step {self.shared_state.step_number}/{self.max_steps}")

        # Capture screenshot if needed
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
                    await ctx.store.set("screenshot", screenshot)
                    logger.debug("📸 Screenshot captured for CodeAct")
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

            # Extract and store package/app name (using unified update method)
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
        logger.info("CodeAct response:", extra={"color": "yellow"})
        response = await acall_with_retries(
            self.llm, messages_to_send, stream=self.agent_config.streaming
        )

        if response is None:
            return CodeActEndEvent(
                success=False,
                reason="LLM response is None. This is a critical error.",
                code_executions=self.code_exec_counter,
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

        # Extract thought and code
        code, thought = extract_code_and_thought(response_text)

        # Update unified state
        self.shared_state.last_thought = thought

        event = CodeActResponseEvent(thought=thought, code=code, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_llm_output(
        self, ctx: Context, ev: CodeActResponseEvent
    ) -> CodeActCodeEvent | CodeActInputEvent:
        """Route to execution or request code if missing."""
        if not ev.thought:
            logger.warning("LLM provided code without thoughts.")
            # Add reminder to get thoughts
            no_thoughts_text = (
                "Your previous response provided code without explaining your reasoning first. "
                "Remember to always describe your thought process and plan *before* providing the code block.\n\n"
                "The code you provided will be executed below.\n\n"
                "Now, describe the next step you will take to address the original goal."
            )
            self.shared_state.message_history.append(
                ChatMessage(role="user", content=no_thoughts_text)
            )
        else:
            logger.debug(f"Reasoning: {ev.thought}")

        if ev.code:
            event = CodeActCodeEvent(code=ev.code)
            ctx.write_event_to_stream(event)
            return event
        else:
            # No code - ask for it
            no_code_text = (
                "No code was provided. If you want to mark task as complete "
                "(whether it failed or succeeded), use complete(success: bool, reason: str) "
                "function within a <python></python> code block."
            )
            self.shared_state.message_history.append(
                ChatMessage(role="user", content=no_code_text)
            )
            return CodeActInputEvent()

    @step
    async def execute_code(
        self, ctx: Context, ev: CodeActCodeEvent
    ) -> CodeActOutputEvent | CodeActEndEvent:
        """Execute the code and return result."""
        code = ev.code
        logger.debug(f"Executing:\n<python>\n{code}\n</python>")

        try:
            self.code_exec_counter += 1
            result = await self.executor.execute(
                ExecuterState(ui_state=await ctx.store.get("ui_state", None)),
                code,
                timeout=self.config.execution_timeout,
            )
            logger.info("💡 Execution result:", extra={"color": "dim"})
            logger.info(f"{result}")
            await asyncio.sleep(self.agent_config.after_sleep_action)

            # Check if complete() was called
            if self.shared_state.finished:
                logger.debug("✅ Task marked as complete via complete() function")

                # Validate completion state
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

                event = CodeActEndEvent(
                    success=success,
                    reason=reason,
                    code_executions=self.code_exec_counter,
                )
                ctx.write_event_to_stream(event)
                return event

            # Update remembered info
            self.remembered_info = self.shared_state.fast_memory

            event = CodeActOutputEvent(output=str(result))
            ctx.write_event_to_stream(event)
            return event

        except Exception as e:
            logger.error(f"💥 Action failed: {e}")
            if self.debug:
                logger.error("Exception details:", exc_info=True)

            event = CodeActOutputEvent(output=f"Error during execution: {e}")
            ctx.write_event_to_stream(event)
            return event

    @step
    async def handle_execution_result(
        self, ctx: Context, ev: CodeActOutputEvent
    ) -> CodeActInputEvent:
        """Add execution result to history and loop back."""
        output = ev.output or "Code executed, but produced no output."

        # Add execution output as user message
        observation_text = f"Execution Result:\n<result>\n{output}\n</result>"
        self.shared_state.message_history.append(
            ChatMessage(role="user", content=observation_text)
        )

        return CodeActInputEvent()

    @step
    async def finalize(self, ev: CodeActEndEvent, ctx: Context) -> StopEvent:
        self.shared_state.finished = False
        ctx.write_event_to_stream(ev)

        return StopEvent(
            result={
                "success": ev.success,
                "reason": ev.reason,
                "code_executions": ev.code_executions,
            }
        )
