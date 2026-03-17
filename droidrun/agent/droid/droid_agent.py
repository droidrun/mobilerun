"""
DroidAgent - A wrapper class that coordinates the planning and execution of tasks
to achieve a user's goal on an Android device.

Architecture:
- When reasoning=False: Uses CodeActAgent directly
- When reasoning=True: Uses Manager (planning) + Executor (action) workflows
"""

import logging
import os
import traceback
from typing import TYPE_CHECKING, Awaitable, Type, Union

from async_adbutils import adb
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from opentelemetry import trace
from pydantic import BaseModel
from workflows.events import Event
from workflows.handler import WorkflowHandler

from droidrun.agent.action_context import ActionContext
from droidrun.agent.codeact import CodeActAgent, FastAgent
from droidrun.agent.codeact.events import CodeActOutputEvent, FastAgentOutputEvent
from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.droid.events import (
    ExecutorInputEvent,
    ExecutorResultEvent,
    ExternalUserMessageDroppedEvent,
    FastAgentExecuteEvent,
    FastAgentResultEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
    ResultEvent,
    ScripterExecutorInputEvent,
    ScripterExecutorResultEvent,
    TextManipulatorInputEvent,
    TextManipulatorResultEvent,
)
from droidrun.agent.droid.state import DroidAgentState, QueuedUserMessage
from droidrun.agent.executor import ExecutorAgent
from droidrun.agent.external import load_agent
from droidrun.agent.manager import ManagerAgent, StatelessManagerAgent
from droidrun.agent.oneflows.structured_output_agent import StructuredOutputAgent
from droidrun.agent.oneflows.text_manipulator import run_text_manipulation_agent
from droidrun.agent.scripter import ScripterAgent
from droidrun.agent.step_store import StepStore, grep_steps, read_step_file
from droidrun.agent.tool_registry import ToolRegistry
from droidrun.agent.trajectory import TrajectoryWriter
from droidrun.agent.utils.actions import complete, open_app
from droidrun.agent.utils.llm_loader import (
    load_agent_llms,
    merge_llms_with_config,
)
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.signatures import (
    ATOMIC_ACTION_SIGNATURES,
    build_credential_tools,
)
from droidrun.agent.utils.tracing_setup import (
    apply_session_context,
    record_langfuse_screenshot,
    setup_tracing,
)
from droidrun.agent.utils.trajectory import Trajectory
from droidrun.config_manager.config_manager import (
    AgentConfig,
    CredentialsConfig,
    DeviceConfig,
    DroidrunConfig,
    LoggingConfig,
    TelemetryConfig,
    ToolsConfig,
    TracingConfig,
)
from droidrun.config_manager.safe_execution import SafeExecutionConfig
from droidrun.credential_manager import CredentialManager, FileCredentialManager
from droidrun.log_handlers import CLILogHandler, configure_logging
from droidrun.mcp.adapter import mcp_to_droidrun_tools
from droidrun.mcp.client import MCPClientManager
from droidrun.mcp.config import MCPConfig
from droidrun.portal import ensure_portal_ready
from droidrun.telemetry import (
    DroidAgentFinalizeEvent,
    DroidAgentInitEvent,
    capture,
    flush,
)
from droidrun.tools.driver.android import AndroidDriver
from droidrun.tools.driver.base import DeviceDisconnectedError
from droidrun.tools.driver.ios import IOSDriver
from droidrun.tools.driver.recording import RecordingDriver
from droidrun.tools.driver.stealth import StealthDriver
from droidrun.tools.filters import ConciseFilter, DetailedFilter
from droidrun.tools.formatters import IndexedFormatter
from droidrun.tools.ui.ios_provider import IOSStateProvider
from droidrun.tools.ui.provider import AndroidStateProvider

if TYPE_CHECKING:
    from droidrun.tools.driver.base import DeviceDriver
    from droidrun.tools.ui.provider import StateProvider

logger = logging.getLogger("droidrun")


class DroidAgent(Workflow):
    """
    A wrapper class that coordinates between agents to achieve a user's goal.

    Reasoning modes:
    - reasoning=False: Uses CodeActAgent directly for immediate execution
    - reasoning=True: Uses ManagerAgent (planning) + ExecutorAgent (actions)
    """

    @staticmethod
    def _configure_default_logging(debug: bool = False):
        """
        Configure default logging for DroidAgent if no real handler is present.
        """
        has_real_handler = any(
            not isinstance(h, logging.NullHandler) for h in logger.handlers
        )
        if not has_real_handler:
            handler = CLILogHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
                if debug
                else logging.Formatter("%(message)s")
            )
            configure_logging(debug=debug, handler=handler)

    def __init__(
        self,
        goal: str,
        config: DroidrunConfig | None = None,
        llms: dict[str, LLM] | LLM | None = None,
        custom_tools: dict = None,
        credentials: Union[dict, "CredentialManager", None] = None,
        variables: dict | None = None,
        output_model: Type[BaseModel] | None = None,
        prompts: dict[str, str] | None = None,
        driver: "DeviceDriver | None" = None,
        state_provider: "StateProvider | None" = None,
        timeout: int = 1000,
        *args,
        **kwargs,
    ):
        self.user_id = kwargs.pop("user_id", None)
        self.runtype = kwargs.pop("runtype", "developer")
        self.shared_state = DroidAgentState(
            instruction=goal,
            err_to_manager_thresh=2,
            user_id=self.user_id,
            runtype=self.runtype,
        )
        self.output_model = output_model

        # Initialize prompt resolver for custom prompts
        self.prompt_resolver = PromptResolver(custom_prompts=prompts)

        # Store custom variables in shared state
        if variables:
            self.shared_state.custom_variables = variables

        # Load credential manager (supports both config and direct dict)
        credentials_source = (
            credentials
            if credentials is not None
            else (config.credentials if config else None)
        )

        if isinstance(credentials_source, CredentialManager):
            self.credential_manager = credentials_source
        elif credentials_source is not None:
            cm = FileCredentialManager(credentials_source)
            self.credential_manager = cm if cm.secrets else None
        else:
            self.credential_manager = None

        self.resolved_device_config = config.device if config else DeviceConfig()

        self.config = DroidrunConfig(
            agent=config.agent if config else AgentConfig(),
            device=self.resolved_device_config,
            tools=config.tools if config else ToolsConfig(),
            logging=config.logging if config else LoggingConfig(),
            tracing=config.tracing if config else TracingConfig(),
            telemetry=config.telemetry if config else TelemetryConfig(),
            llm_profiles=config.llm_profiles if config else {},
            credentials=config.credentials if config else CredentialsConfig(),
            safe_execution=config.safe_execution if config else SafeExecutionConfig(),
            external_agents=config.external_agents if config else {},
            mcp=config.mcp if config else MCPConfig(),
        )

        # These are populated in start_handler (unless injected via __init__)
        self._injected_driver = driver
        self._injected_state_provider = state_provider
        self.driver = None
        self.registry = None
        self.action_ctx = None
        self.state_provider = None

        super().__init__(*args, timeout=timeout, **kwargs)

        self._configure_default_logging(debug=self.config.logging.debug)

        setup_tracing(self.config.tracing, agent=self)

        # Check if using external agent - skip LLM loading
        self._using_external_agent = self.config.agent.name != "droidrun"

        self._stream_screenshots = os.environ.get(
            "DROIDRUN_STREAM_SCREENSHOTS", ""
        ).lower() in ("1", "true")

        self.timeout = timeout

        # Store user custom tools
        self.user_custom_tools = custom_tools or {}

        # Initialize MCP manager (connections made lazily in start_handler)
        self.mcp_manager = None

        # Only load LLMs for native DroidRun agents
        if not self._using_external_agent:
            if llms is None:
                if config is None:
                    raise ValueError(
                        "Either 'llms' or 'config' must be provided. "
                        "If llms is not provided, config is required to load LLMs from profiles."
                    )

                logger.debug("🔄 Loading LLMs from config (llms not provided)...")

                llms = load_agent_llms(
                    config=self.config, output_model=output_model, **kwargs
                )
            if isinstance(llms, dict):
                llms = merge_llms_with_config(
                    self.config, llms, output_model=output_model, **kwargs
                )
            elif isinstance(llms, LLM):
                pass
            else:
                raise ValueError(f"Invalid LLM type: {type(llms)}")

            if isinstance(llms, dict):
                self.manager_llm = llms.get("manager")
                self.executor_llm = llms.get("executor")
                self.fast_agent_llm = llms.get("fast_agent")
                self.text_manipulator_llm = llms.get("text_manipulator")
                self.app_opener_llm = llms.get("app_opener")
                self.scripter_llm = llms.get("scripter", self.fast_agent_llm)
                self.structured_output_llm = llms.get(
                    "structured_output", self.fast_agent_llm
                )
            else:
                self.manager_llm = llms
                self.executor_llm = llms
                self.fast_agent_llm = llms
                self.text_manipulator_llm = llms
                self.app_opener_llm = llms
                self.scripter_llm = llms
                self.structured_output_llm = llms
        else:
            logger.debug(f"🔄 Using external agent: {self.config.agent.name}")
            self.manager_llm = None
            self.executor_llm = None
            self.fast_agent_llm = None
            self.text_manipulator_llm = None
            self.app_opener_llm = None
            self.scripter_llm = None
            self.structured_output_llm = None

        if self.config.logging.save_trajectory != "none":
            self.trajectory = Trajectory(
                goal=self.shared_state.instruction,
                base_path=self.config.logging.trajectory_path,
            )
            self.trajectory_writer = TrajectoryWriter(queue_size=300)
        else:
            self.trajectory = None
            self.trajectory_writer = None

        # Sub-agents are created in __init__ but wired up in start_handler
        if self._using_external_agent:
            self.manager_agent = None
            self.executor_agent = None
        elif self.config.agent.reasoning:
            if self.config.agent.manager.stateless:
                ManagerClass = StatelessManagerAgent
            else:
                ManagerClass = ManagerAgent

            # Pass None for tools-related params — wired up in start_handler
            self.manager_agent = ManagerClass(
                llm=self.manager_llm,
                action_ctx=None,
                state_provider=None,
                save_trajectory=self.config.logging.save_trajectory,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                registry=None,
                output_model=self.output_model,
                prompt_resolver=self.prompt_resolver,
                tracing_config=self.config.tracing,
                timeout=self.timeout,
            )
            self.executor_agent = ExecutorAgent(
                llm=self.executor_llm,
                registry=None,
                action_ctx=None,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                prompt_resolver=self.prompt_resolver,
                timeout=self.timeout,
            )
        else:
            self.manager_agent = None
            self.executor_agent = None

        atomic_tools = list(ATOMIC_ACTION_SIGNATURES.keys())

        capture(
            DroidAgentInitEvent(
                goal=self.shared_state.instruction,
                llms={
                    "manager": (
                        self.manager_llm.class_name() if self.manager_llm else "None"
                    ),
                    "executor": (
                        self.executor_llm.class_name() if self.executor_llm else "None"
                    ),
                    "fast_agent": (
                        self.fast_agent_llm.class_name()
                        if self.fast_agent_llm
                        else "None"
                    ),
                    "text_manipulator": (
                        self.text_manipulator_llm.class_name()
                        if self.text_manipulator_llm
                        else "None"
                    ),
                    "app_opener": (
                        self.app_opener_llm.class_name()
                        if self.app_opener_llm
                        else "None"
                    ),
                },
                tools=",".join(atomic_tools + ["read", "grep", "complete"]),
                max_steps=self.config.agent.max_steps,
                timeout=timeout,
                vision={
                    "manager": self.config.agent.manager.vision,
                    "executor": self.config.agent.executor.vision,
                    "fast_agent": self.config.agent.fast_agent.vision,
                },
                reasoning=self.config.agent.reasoning,
                enable_tracing=self.config.tracing.enabled,
                debug=self.config.logging.debug,
                save_trajectories=self.config.logging.save_trajectory,
                runtype=self.runtype,
                custom_prompts=prompts,
            ),
            self.user_id,
        )

        logger.debug("✅ DroidAgent initialized successfully.")

    def run(self, *args, **kwargs) -> Awaitable[ResultEvent] | WorkflowHandler:
        apply_session_context()
        handler = super().run(*args, **kwargs)  # type: ignore[assignment]
        return handler

    # ========================================================================
    # start_handler — creates driver, registry, action_ctx
    # ========================================================================

    @step
    async def start_handler(
        self, ctx: Context, ev: StartEvent
    ) -> FastAgentExecuteEvent | ManagerInputEvent:
        logger.info(
            f"🚀 Running DroidAgent to achieve goal: {self.shared_state.instruction}"
        )
        ctx.write_event_to_stream(ev)

        if self.trajectory_writer:
            await self.trajectory_writer.start()

        # ── 1. Create driver ──────────────────────────────────────────
        if self.config.agent.reasoning:
            vision_enabled = self.config.agent.manager.vision
        else:
            vision_enabled = self.config.agent.fast_agent.vision

        is_ios = self.resolved_device_config.platform.lower() == "ios"

        if self._injected_driver is not None:
            driver = self._injected_driver
        elif is_ios:
            ios_url = self.resolved_device_config.serial
            if not ios_url:
                raise ValueError("iOS device URL required in config.device.serial")
            # TODO: bundle_identifiers not configurable yet
            driver = IOSDriver(url=ios_url)
            await driver.connect()
        else:
            device_serial = self.resolved_device_config.serial
            if device_serial is None:
                devices = await adb.list()
                if not devices:
                    raise ValueError("No connected Android devices found.")
                device_serial = devices[0].serial

            # Auto-setup portal if enabled
            if self.config.device.auto_setup:
                device_obj = await adb.device(serial=device_serial)
                await ensure_portal_ready(device_obj, debug=self.config.logging.debug)

            driver = AndroidDriver(
                serial=device_serial,
                use_tcp=self.resolved_device_config.use_tcp,
            )
            await driver.connect()

        # Wrap with StealthDriver if stealth mode enabled
        stealth_enabled = self.config.tools and self.config.tools.stealth
        if stealth_enabled and not is_ios:
            driver = StealthDriver(driver)

        # Wrap with RecordingDriver if trajectory saving enabled
        if self.config.logging.save_trajectory != "none":
            if not isinstance(driver, RecordingDriver):
                driver = RecordingDriver(driver)

        self.driver = driver

        # ── 2. Create state provider ──────────────────────────────────
        if self._injected_state_provider is not None:
            self.state_provider = self._injected_state_provider
        elif is_ios:
            self.state_provider = IOSStateProvider(
                driver,
                use_normalized=self.config.agent.use_normalized_coordinates,
            )
        else:
            tree_filter = ConciseFilter() if vision_enabled else DetailedFilter()
            tree_formatter = IndexedFormatter()
            self.state_provider = AndroidStateProvider(
                driver,
                tree_filter=tree_filter,
                tree_formatter=tree_formatter,
                use_normalized=self.config.agent.use_normalized_coordinates,
                stealth=stealth_enabled,
            )

        # ── 3. Build tool registry ────────────────────────────────────
        registry = ToolRegistry()

        # 3a. Atomic tools (click, long_press, type, system_button, swipe, etc.)
        registry.register_from_dict(ATOMIC_ACTION_SIGNATURES)

        # 3b. open_app (always registered)
        registry.register(
            "open_app",
            fn=open_app,
            params={"text": {"type": "string", "required": True}},
            description='Open an app by name or description. Usage: {"action": "open_app", "text": "Gmail"}',
            deps={"start_app", "get_apps"},
        )

        # 3c. Step history tools (read + grep)
        registry.register(
            "read",
            fn=read_step_file,
            params={
                "path": {
                    "type": "string",
                    "description": "Step file path: /steps/{n}/ui_state.txt or /steps/{n}/screenshot.png",
                    "required": True,
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from. Ignored for images.",
                    "required": False,
                    "default": 0,
                },
                "limit": {
                    "type": "number",
                    "description": "Number of lines to read. Ignored for images.",
                    "required": False,
                    "default": 0,
                },
            },
            description="Read a file from step history. UI state files return text, screenshot files return the image.",
        )
        registry.register(
            "grep",
            fn=grep_steps,
            params={
                "pattern": {
                    "type": "string",
                    "description": "Python regex pattern (case-insensitive). Special characters like . ( ) [ ] * + must be escaped with \\\\ to match literally.",
                    "required": True,
                },
                "steps": {
                    "type": "list",
                    "description": "List of step numbers to search, e.g. [3, 5, 7]",
                    "required": True,
                },
            },
            description="Search UI state history using Python regex. Searches line-by-line across the specified steps. Returns matching lines with file path and line number.",
        )

        # 3d. complete (always registered)
        registry.register(
            "complete",
            fn=complete,
            params={
                "success": {"type": "boolean", "required": True},
                "message": {"type": "string", "required": True},
            },
            description=(
                "Mark task as complete. "
                "success=true if task succeeded, false if failed. "
                "message contains the result, answer, or explanation."
            ),
        )

        # 3d. type_secret (conditional)
        if self.credential_manager:
            credential_tools = await build_credential_tools(self.credential_manager)
            if credential_tools:
                registry.register_from_dict(credential_tools)

        # 3e. User custom tools
        if self.user_custom_tools:
            registry.register_from_dict(self.user_custom_tools)

        # 3f. MCP tools
        if self.config.mcp and self.config.mcp.enabled:
            self.mcp_manager = MCPClientManager(self.config.mcp)
            await self.mcp_manager.discover_tools()
            mcp_tools = mcp_to_droidrun_tools(self.mcp_manager)
            if mcp_tools:
                registry.register_from_dict(mcp_tools)

        # 3g. Disable unsupported tools based on driver + state provider capabilities
        capabilities = driver.supported | self.state_provider.supported
        registry.disable_unsupported(capabilities)

        # 3h. Disable tools from config
        disabled_tools = (
            self.config.tools.disabled_tools
            if self.config.tools and self.config.tools.disabled_tools
            else []
        )
        if disabled_tools:
            registry.disable(disabled_tools)

        self.registry = registry

        # ── 4. Create ActionContext ────────────────────────────────────
        self.step_store = StepStore()
        self.action_ctx = ActionContext(
            driver=driver,
            ui=None,  # populated each step by state_provider
            shared_state=self.shared_state,
            state_provider=self.state_provider,
            app_opener_llm=self.app_opener_llm,
            credential_manager=self.credential_manager,
            streaming=self.config.agent.streaming,
            step_store=self.step_store,
        )

        # ── 5. Wire up sub-agents ─────────────────────────────────────
        if self.config.agent.reasoning and self.executor_agent:
            self.manager_agent.action_ctx = self.action_ctx
            self.manager_agent.state_provider = self.state_provider
            self.manager_agent.registry = self.registry
            self.manager_agent.save_trajectory = self.config.logging.save_trajectory
            self.executor_agent.registry = self.registry
            self.executor_agent.action_ctx = self.action_ctx

        # ── 6. Fetch device date once ─────────────────────────────────
        self.shared_state.device_date = await driver.get_date()

        # ── 7. External agent mode ────────────────────────────────────
        if self._using_external_agent:
            agent_name = self.config.agent.name
            agent_module = load_agent(agent_name)
            if not agent_module:
                raise ValueError(f"Failed to load external agent: {agent_name}")

            agent_config = self.config.external_agents.get(agent_name)
            if not agent_config:
                raise ValueError(
                    f"No config found for agent '{agent_name}' in external_agents section"
                )

            final_config = {**agent_module["config"], **agent_config}

            logger.info(f"🤖 Using external agent: {agent_name}")

            result = await agent_module["run"](
                driver=self.driver,
                action_ctx=self.action_ctx,
                instruction=self.shared_state.instruction,
                config=final_config,
                max_steps=self.config.agent.max_steps,
            )

            return FinalizeEvent(success=result["success"], reason=result["reason"])

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(self.trajectory, stage="init")

        if not self.config.agent.reasoning:
            logger.debug(
                f"🔄 Direct execution mode - executing goal: {self.shared_state.instruction}"
            )
            event = FastAgentExecuteEvent(instruction=self.shared_state.instruction)
            ctx.write_event_to_stream(event)
            return event

        logger.debug("🧠 Reasoning mode - initializing Manager/Executor workflow")
        event = ManagerInputEvent()
        ctx.write_event_to_stream(event)
        return event

    # ========================================================================
    # External user message injection
    # ========================================================================

    def send_user_message(self, message: str) -> QueuedUserMessage:
        queued = self.shared_state.queue_user_message(message)
        logger.info(
            f"📩 External user message queued [id={queued.id}] "
            f"(queue length: {len(self.shared_state.pending_user_messages)})",
            extra={"color": "cyan"},
        )
        return queued

    # ========================================================================
    # execute_task — FastAgent / CodeActAgent
    # ========================================================================

    @step
    async def execute_task(
        self, ctx: Context, ev: FastAgentExecuteEvent
    ) -> FastAgentResultEvent:
        """Execute a single task using CodeActAgent or FastAgent."""

        logger.debug(f"🔧 Executing task: {ev.instruction}")

        try:
            if self.config.agent.fast_agent.codeact:
                agent = CodeActAgent(
                    llm=self.fast_agent_llm,
                    agent_config=self.config.agent,
                    registry=self.registry,
                    action_ctx=self.action_ctx,
                    state_provider=self.state_provider,
                    save_trajectory=self.config.logging.save_trajectory,
                    debug=self.config.logging.debug,
                    shared_state=self.shared_state,
                    safe_execution_config=self.config.safe_execution,
                    output_model=self.output_model,
                    prompt_resolver=self.prompt_resolver,
                    timeout=self.timeout,
                    tracing_config=self.config.tracing,
                )
            else:
                agent = FastAgent(
                    llm=self.fast_agent_llm,
                    agent_config=self.config.agent,
                    registry=self.registry,
                    action_ctx=self.action_ctx,
                    state_provider=self.state_provider,
                    save_trajectory=self.config.logging.save_trajectory,
                    debug=self.config.logging.debug,
                    shared_state=self.shared_state,
                    output_model=self.output_model,
                    prompt_resolver=self.prompt_resolver,
                    timeout=self.timeout,
                    tracing_config=self.config.tracing,
                )

            handler = agent.run(
                input=ev.instruction,
            )

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

                if isinstance(nested_ev, (CodeActOutputEvent, FastAgentOutputEvent)):
                    if self.config.logging.save_trajectory != "none":
                        self.trajectory_writer.write(
                            self.trajectory,
                            stage=f"fast_agent_step_{self.shared_state.step_number}",
                        )

            result = await handler

            return FastAgentResultEvent(
                success=result.get("success", False),
                reason=result["reason"],
                instruction=ev.instruction,
            )

        except DeviceDisconnectedError as e:
            logger.error(f"Device disconnected: {e}")
            return FastAgentResultEvent(
                success=False,
                reason=f"Device disconnected: {e}",
                instruction=ev.instruction,
            )

        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            if self.config.logging.debug:
                logger.error(traceback.format_exc())
            return FastAgentResultEvent(
                success=False, reason=f"Error: {str(e)}", instruction=ev.instruction
            )

    @step
    async def handle_fast_agent_result(
        self, ctx: Context, ev: FastAgentResultEvent
    ) -> FinalizeEvent:
        try:
            return FinalizeEvent(success=ev.success, reason=ev.reason)

        except Exception as e:
            logger.error(f"❌ Error during DroidAgent execution: {e}")
            if self.config.logging.debug:
                logger.error(traceback.format_exc())
            return FinalizeEvent(
                success=False,
                reason=str(e),
            )

    # ========================================================================
    # Manager/Executor Workflow Steps
    # ========================================================================

    @step
    async def run_manager(
        self, ctx: Context, ev: ManagerInputEvent
    ) -> ManagerPlanEvent | FinalizeEvent:
        """Run Manager planning phase."""
        if self.shared_state.step_number >= self.config.agent.max_steps:
            logger.warning(f"⚠️ Reached maximum steps ({self.config.agent.max_steps})")
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
            return FinalizeEvent(
                success=False,
                reason=f"Reached maximum steps ({self.config.agent.max_steps})",
            )

        self.shared_state.step_number += 1
        logger.info(
            f"🔄 Step {self.shared_state.step_number}/{self.config.agent.max_steps}"
        )

        try:
            handler = self.manager_agent.run()

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

            result = await handler
        except DeviceDisconnectedError as e:
            logger.error(f"Device disconnected: {e}")
            return FinalizeEvent(success=False, reason=f"Device disconnected: {e}")

        event = ManagerPlanEvent(
            plan=result["plan"],
            current_subgoal=result["current_subgoal"],
            thought=result["thought"],
            answer=result.get("answer", ""),
            success=result.get("success"),
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_manager_plan(
        self, ctx: Context, ev: ManagerPlanEvent
    ) -> (
        ExecutorInputEvent
        | ScripterExecutorInputEvent
        | FinalizeEvent
        | TextManipulatorInputEvent
        | ManagerInputEvent
    ):
        """Process Manager output and decide next step."""
        # Check for answer-type termination
        if ev.answer.strip():
            if self.shared_state.pending_user_messages:
                logger.info(
                    "⏸️ Manager tried to finish but external messages pending, "
                    "looping back to Manager",
                    extra={"color": "cyan"},
                )
                return ManagerInputEvent()
            success = ev.success if ev.success is not None else True
            self.shared_state.progress_summary = f"Answer: {ev.answer}"
            return FinalizeEvent(success=success, reason=ev.answer)

        # Check for <script> tag
        if "<script>" in ev.current_subgoal:
            start_idx = ev.plan.find("<script>")
            end_idx = ev.plan.find("</script>")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                task = ev.plan[start_idx + len("<script>") : end_idx].strip()
                logger.debug(f"🐍 Routing to ScripterAgent: {task[:80]}...")
                event = ScripterExecutorInputEvent(task=task)
                ctx.write_event_to_stream(event)
                return event
            else:
                logger.warning(
                    "⚠️ Found <script> in subgoal but not properly closed in plan, treating as regular subgoal"
                )
        if "TEXT_TASK" in ev.current_subgoal:
            if self.config.agent.fast_agent.codeact:
                return TextManipulatorInputEvent(
                    task=ev.current_subgoal.replace("TEXT_TASK:", "")
                    .replace("TEXT_TASK", "")
                    .strip()
                )
            else:
                logger.debug(
                    "⚠️ TEXT_TASK in tools mode — routing to Executor instead of TextManipulator"
                )
                subgoal = (
                    ev.current_subgoal.replace("TEXT_TASK:", "")
                    .replace("TEXT_TASK", "")
                    .strip()
                )
                return ExecutorInputEvent(current_subgoal=subgoal)

        logger.debug(f"▶️  Proceeding to Executor with subgoal: {ev.current_subgoal}")
        return ExecutorInputEvent(current_subgoal=ev.current_subgoal)

    @step
    async def run_text_manipulator(
        self, ctx: Context, ev: TextManipulatorInputEvent
    ) -> TextManipulatorResultEvent:
        logger.debug(f"🔍 Running TextManipulatorAgent for task: {ev.task}")

        if not self.shared_state.focused_text:
            logger.warning("⚠️ No focused text available, using empty string")
            current_text = ""
        else:
            current_text = self.shared_state.focused_text

        try:
            text_to_type, code_ran = await run_text_manipulation_agent(
                instruction=self.shared_state.instruction,
                current_subgoal=ev.task,
                current_text=current_text,
                overall_plan=self.shared_state.plan,
                llm=self.text_manipulator_llm,
                stream=self.config.agent.streaming,
            )

            return TextManipulatorResultEvent(
                task=ev.task, text_to_type=text_to_type, code_ran=code_ran
            )

        except Exception as e:
            logger.error(f"❌ TextManipulator agent failed: {e}")
            if self.config.logging.debug:
                logger.error(traceback.format_exc())

            return TextManipulatorResultEvent(
                task=ev.task, text_to_type="", code_ran=""
            )

    @step
    async def handle_text_manipulator_result(
        self, ctx: Context, ev: TextManipulatorResultEvent
    ) -> ManagerInputEvent:
        if not ev.text_to_type or not ev.text_to_type.strip():
            logger.warning("⚠️ TextManipulator returned empty text, treating as no-op")
            self.shared_state.last_executor_full_response = (
                "Text manipulation returned empty result"
            )
            self.shared_state.action_outcomes.append(False)
        else:
            try:
                success = await self.action_ctx.driver.input_text(
                    ev.text_to_type, clear=True
                )

                if not success:
                    logger.warning("⚠️ Text input may have failed")
                    self.shared_state.last_executor_full_response = (
                        "Text manipulation attempted but may have failed"
                    )
                    self.shared_state.action_outcomes.append(False)
                else:
                    logger.debug(
                        f"✅ Text manipulator successfully typed {len(ev.text_to_type)} characters"
                    )
                    self.shared_state.last_executor_full_response = f"Text manipulation successful: typed {len(ev.text_to_type)} characters"
                    self.shared_state.action_outcomes.append(True)
            except Exception as e:
                logger.error(f"❌ Error during text input: {e}")
                self.shared_state.last_executor_full_response = (
                    f"Text manipulation error: {str(e)}"
                )
                self.shared_state.action_outcomes.append(False)

        text_manipulation_record = {
            "task": ev.task,
            "code_ran": ev.code_ran,
            "text_length": len(ev.text_to_type) if ev.text_to_type else 0,
            "success": (
                self.shared_state.action_outcomes[-1]
                if self.shared_state.action_outcomes
                else False
            ),
        }

        self.shared_state.text_manipulation_history.append(text_manipulation_record)
        self.shared_state.last_text_manipulation_success = text_manipulation_record[
            "success"
        ]

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(
                self.trajectory, stage=f"step_{self.shared_state.step_number}"
            )

        return ManagerInputEvent()

    @step
    async def run_executor(
        self, ctx: Context, ev: ExecutorInputEvent
    ) -> ExecutorResultEvent:
        """Run Executor action phase."""
        logger.debug("⚡ Running Executor for action...")

        handler = self.executor_agent.run(subgoal=ev.current_subgoal)

        async for nested_ev in handler.stream_events():
            self.handle_stream_event(nested_ev, ctx)

        result = await handler

        # Update coordination state after execution
        actions = result["actions"]
        self.shared_state.last_executor_action_count = len(actions)

        for action_record in actions:
            action_dict = {
                "action": action_record["action"],
                **action_record.get("args", {}),
            }
            self.shared_state.action_history.append(action_dict)
            self.shared_state.summary_history.append(action_record["summary"])
            self.shared_state.action_outcomes.append(action_record["outcome"])
            self.shared_state.error_descriptions.append(action_record["error"])

        if not actions:
            # Executor returned text without tool calls — pass output to Manager
            self.shared_state.last_executor_full_response = result.get(
                "full_response", ""
            )

        return ExecutorResultEvent(
            actions=actions,
            thought=result.get("thought", ""),
        )

    @step
    async def handle_executor_result(
        self, ctx: Context, ev: ExecutorResultEvent
    ) -> ManagerInputEvent:
        """Process Executor result and continue."""
        err_thresh = self.shared_state.err_to_manager_thresh

        if len(self.shared_state.action_outcomes) >= err_thresh:
            latest = self.shared_state.action_outcomes[-err_thresh:]
            error_count = sum(1 for o in latest if not o)
            if error_count == err_thresh:
                logger.warning(f"⚠️ Error escalation: {err_thresh} consecutive errors")
                self.shared_state.error_flag_plan = True
            else:
                if self.shared_state.error_flag_plan:
                    logger.debug("✅ Error resolved - resetting error flag")
                self.shared_state.error_flag_plan = False

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(
                self.trajectory, stage=f"step_{self.shared_state.step_number}"
            )

        return ManagerInputEvent()

    # ========================================================================
    # Script Executor Workflow Steps
    # ========================================================================

    @step
    async def run_scripter(
        self, ctx: Context, ev: ScripterExecutorInputEvent
    ) -> ScripterExecutorResultEvent:
        """Instantiate and run ScripterAgent for off-device operations."""
        logger.debug(f"🐍 Starting ScripterAgent for task: {ev.task[:2000]}...")

        scripter_agent = ScripterAgent(
            llm=self.scripter_llm,
            agent_config=self.config.agent,
            shared_state=self.shared_state,
            task=ev.task,
            safe_execution_config=self.config.safe_execution,
            timeout=self.timeout,
        )

        handler = scripter_agent.run()

        async for nested_ev in handler.stream_events():
            self.handle_stream_event(nested_ev, ctx)

        result = await handler

        script_record = {
            "task": ev.task,
            "message": result["message"],
            "success": result["success"],
            "code_executions": result.get("code_executions", 0),
        }
        self.shared_state.scripter_history.append(script_record)
        self.shared_state.last_scripter_message = result["message"]
        self.shared_state.last_scripter_success = result["success"]

        return ScripterExecutorResultEvent(
            task=ev.task,
            message=result["message"],
            success=result["success"],
            code_executions=result.get("code_executions", 0),
        )

    @step
    async def handle_scripter_result(
        self, ctx: Context, ev: ScripterExecutorResultEvent
    ) -> ManagerInputEvent:
        """Process ScripterAgent result and loop back to Manager."""
        if ev.success:
            logger.debug(
                f"✅ Script completed successfully in {ev.code_executions} steps"
            )
        else:
            logger.warning(f"⚠️ Script failed or reached max steps: {ev.message}")

        if self.config.logging.save_trajectory != "none":
            self.trajectory_writer.write(
                self.trajectory, stage=f"step_{self.shared_state.step_number}"
            )

        return ManagerInputEvent()

    # ========================================================================
    # Finalize
    # ========================================================================

    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> ResultEvent:
        self.shared_state.workflow_completed = True
        ctx.write_event_to_stream(ev)
        capture(
            DroidAgentFinalizeEvent(
                success=ev.success,
                reason=ev.reason,
                steps=self.shared_state.step_number,
                unique_packages_count=len(self.shared_state.visited_packages),
                unique_activities_count=len(self.shared_state.visited_activities),
            ),
            self.user_id,
        )
        await flush()

        # Base result with answer
        result = ResultEvent(
            success=ev.success,
            reason=ev.reason,
            steps=self.shared_state.step_number,
            structured_output=None,
        )

        # Extract structured output if model was provided
        if self.output_model is not None and ev.reason:
            logger.debug("🔄 Running structured output extraction...")

            try:
                structured_agent = StructuredOutputAgent(
                    llm=self.structured_output_llm,
                    pydantic_model=self.output_model,
                    answer_text=ev.reason,
                    timeout=self.timeout,
                )

                handler = structured_agent.run()

                async for nested_ev in handler.stream_events():
                    self.handle_stream_event(nested_ev, ctx)

                extraction_result = await handler

                if extraction_result["success"]:
                    result.structured_output = extraction_result["structured_output"]
                    logger.debug("✅ Structured output added to final result")
                else:
                    logger.warning(
                        f"⚠️  Structured extraction failed: {extraction_result['error_message']}"
                    )

            except Exception as e:
                logger.error(f"❌ Error during structured extraction: {e}")
                if self.config.logging.debug:
                    logger.error(traceback.format_exc())

        # Capture final screenshot and UI state (independent of trajectory persistence)
        vision_any = (
            self.config.agent.manager.vision
            or self.config.agent.executor.vision
            or self.config.agent.fast_agent.vision
        )
        if (
            vision_any
            or self._stream_screenshots
            or self.config.logging.save_trajectory != "none"
        ):
            try:
                screenshot = await self.action_ctx.driver.screenshot()
                if screenshot:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    parent_span = trace.get_current_span()
                    record_langfuse_screenshot(
                        screenshot,
                        parent_span=parent_span,
                        screenshots_enabled=self.config.tracing.langfuse_screenshots,
                        vision_enabled=vision_any,
                    )
                    logger.debug("📸 Final screenshot captured")
            except Exception as e:
                logger.warning(f"Failed to capture final screenshot: {e}")

            try:
                ui_state = await self.state_provider.get_state()
                ctx.write_event_to_stream(
                    RecordUIStateEvent(ui_state=ui_state.elements)
                )
                logger.debug("📋 Final UI state captured")
            except Exception as e:
                logger.warning(f"Failed to capture final UI state: {e}")

        # Save trajectory to disk
        if self.config.logging.save_trajectory != "none":
            # Populate macro data from RecordingDriver log
            if isinstance(self.driver, RecordingDriver):
                self.trajectory.macro = list(self.driver.log)

            self.trajectory_writer.write_final(
                self.trajectory, self.config.logging.trajectory_gifs
            )
            await self.trajectory_writer.stop()
            logger.info(f"📁 Trajectory saved: {self.trajectory.trajectory_folder}")

        # Cleanup MCP connections
        if self.mcp_manager:
            try:
                await self.mcp_manager.disconnect_all()
            except Exception as e:
                logger.warning(f"MCP cleanup error: {e}")

        return result

    # ========================================================================
    # Event streaming
    # ========================================================================

    def handle_stream_event(self, ev: Event, ctx: Context):
        if not isinstance(ev, StopEvent):
            ctx.write_event_to_stream(ev)

            if self.trajectory:
                if isinstance(ev, ScreenshotEvent):
                    self.trajectory.screenshot_queue.append(ev.screenshot)
                    self.trajectory.screenshot_count += 1
                elif isinstance(ev, RecordUIStateEvent):
                    self.trajectory.ui_states.append(ev.ui_state)
                else:
                    self.trajectory.events.append(ev)
