"""
MobileAgent - A wrapper class that coordinates the planning and execution of tasks
to achieve a user's goal on a mobile device.

Architecture:
- When reasoning=False: Uses FastAgent directly
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

from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agent.droid.events import (
    ExecutorInputEvent,
    ExecutorResultEvent,
    ExternalUserMessageDroppedEvent,
    FastAgentExecuteEvent,
    FastAgentResultEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
    ResultEvent,
)
from mobilerun.agent.droid.state import MobileAgentState, QueuedUserMessage
from mobilerun.agent.executor import ExecutorAgent
from mobilerun.agent.external import load_agent
from mobilerun.agent.fast_agent import FastAgent
from mobilerun.agent.fast_agent.events import FastAgentOutputEvent
from mobilerun.agent.manager import ManagerAgent, StatelessManagerAgent
from mobilerun.agent.oneflows.structured_output_agent import StructuredOutputAgent
from mobilerun.agent.trajectory import TrajectoryWriter
from mobilerun.agent.utils.llm_loader import (
    load_agent_llms,
    merge_llms_with_config,
)
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.agent.utils.tracing_setup import (
    apply_session_context,
    record_langfuse_screenshot,
    setup_tracing,
)
from mobilerun.agent.utils.trajectory import Trajectory
from mobilerun.config_manager.config_manager import (
    AgentConfig,
    CredentialsConfig,
    DeviceConfig,
    LoggingConfig,
    MobileConfig,
    TelemetryConfig,
    ToolsConfig,
    TracingConfig,
)
from mobilerun.credential_manager import CredentialManager, FileCredentialManager
from mobilerun.log_handlers import CLILogHandler, configure_logging
from mobilerun.mcp.adapter import mcp_to_mobilerun_tools
from mobilerun.mcp.client import MCPClientManager
from mobilerun.mcp.config import MCPConfig
from mobilerun.portal import ensure_portal_ready
from mobilerun.telemetry import (
    MobileAgentFinalizeEvent,
    MobileAgentInitEvent,
    capture,
    flush,
)
from mobilerun.tools.driver.android import AndroidDriver
from mobilerun.tools.driver.base import DeviceDisconnectedError
from mobilerun.tools.driver.ios import IOSDriver, discover_ios_portal
from mobilerun.tools.driver.recording import RecordingDriver
from mobilerun.tools.driver.stealth import StealthDriver
from mobilerun.tools.driver.visual_remote import (
    VISUAL_REMOTE_CONNECTION,
    VISUAL_REMOTE_DEFAULT_URL,
    VisualRemoteDriver,
)
from mobilerun.tools.filters import ConciseFilter, DetailedFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import AndroidStateProvider
from mobilerun.tools.ui.screenshot_provider import ScreenshotOnlyStateProvider

if TYPE_CHECKING:
    from mobilerun.tools.driver.base import DeviceDriver
    from mobilerun.tools.ui.provider import StateProvider

logger = logging.getLogger("mobilerun")

_COORDINATE_TOOL_NAMES = {"click_at", "click_area", "long_press_at"}


def _normalize_control_backend(control_backend: str | None) -> str | None:
    if control_backend is None:
        return None
    normalized = control_backend.strip().lower()
    return normalized or None


def _force_screenshot_only_vision(agent_config: AgentConfig) -> None:
    agent_config.vision_only = True
    agent_config.manager.vision = True
    agent_config.executor.vision = True
    agent_config.fast_agent.vision = True


def _effective_disabled_tools(disabled_tools: list[str], state_provider) -> list[str]:
    if getattr(state_provider, "requires_coordinate_tools", False):
        return [name for name in disabled_tools if name not in _COORDINATE_TOOL_NAMES]
    return disabled_tools


class MobileAgent(Workflow):
    """
    A wrapper class that coordinates between agents to achieve a user's goal.

    Reasoning modes:
    - reasoning=False: Uses FastAgent directly for immediate execution
    - reasoning=True: Uses ManagerAgent (planning) + ExecutorAgent (actions)
    """

    @staticmethod
    def _configure_default_logging(debug: bool = False):
        """
        Configure default logging for MobileAgent if no real handler is present.
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
        config: MobileConfig | None = None,
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
        self.shared_state = MobileAgentState(
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

        self.config = MobileConfig(
            agent=config.agent if config else AgentConfig(),
            device=self.resolved_device_config,
            tools=config.tools if config else ToolsConfig(),
            logging=config.logging if config else LoggingConfig(),
            tracing=config.tracing if config else TracingConfig(),
            telemetry=config.telemetry if config else TelemetryConfig(),
            llm_profiles=config.llm_profiles if config else {},
            credentials=config.credentials if config else CredentialsConfig(),
            external_agents=config.external_agents if config else {},
            mcp=config.mcp if config else MCPConfig(),
        )
        control_backend = _normalize_control_backend(
            self.resolved_device_config.control_backend
        )
        if (
            self.config.agent.vision_only
            or control_backend == VISUAL_REMOTE_CONNECTION
            or getattr(state_provider, "requires_coordinate_tools", False)
        ):
            _force_screenshot_only_vision(self.config.agent)

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
        _BUILTIN_AGENT_NAMES = {"mobilerun", "droidrun"}
        self._using_external_agent = self.config.agent.name not in _BUILTIN_AGENT_NAMES

        self._stream_screenshots = (
            os.environ.get("MOBILERUN_STREAM_SCREENSHOTS")
            or os.environ.get("DROIDRUN_STREAM_SCREENSHOTS")
            or ""
        ).lower() in ("1", "true")

        self.timeout = timeout

        # Store user custom tools
        self.user_custom_tools = custom_tools or {}

        # Initialize MCP manager (connections made lazily in start_handler)
        self.mcp_manager = None

        # Only load LLMs for native Mobilerun agents
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
                self.app_opener_llm = llms.get("app_opener")
                self.structured_output_llm = llms.get(
                    "structured_output", self.fast_agent_llm
                )
            else:
                self.manager_llm = llms
                self.executor_llm = llms
                self.fast_agent_llm = llms
                self.app_opener_llm = llms
                self.structured_output_llm = llms
        else:
            logger.debug(f"🔄 Using external agent: {self.config.agent.name}")
            self.manager_llm = None
            self.executor_llm = None
            self.fast_agent_llm = None
            self.app_opener_llm = None
            self.structured_output_llm = None

        if (
            not self._using_external_agent
            and self.config.logging.save_trajectory != "none"
        ):
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

        # Telemetry init event is fired in start_handler after registry is built.
        self._init_prompts = prompts  # stash for telemetry
        self._init_timeout = timeout

        logger.debug("✅ MobileAgent initialized successfully.")

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
            f"🚀 Running MobileAgent to achieve goal: {self.shared_state.instruction}"
        )
        ctx.write_event_to_stream(ev)

        if self.trajectory_writer:
            await self.trajectory_writer.start()

        # ── 0. External agent — early exit ────────────────────────────
        if self._using_external_agent:
            agent_name = self.config.agent.name

            # Load the agent module
            agent_module = load_agent(agent_name)
            if not agent_module:
                from mobilerun.agent.external import list_agents

                available = list_agents()
                if available:
                    agents_str = ", ".join(available)
                    raise ValueError(
                        f"Failed to load external agent '{agent_name}'.\n"
                        f"Available agents: {agents_str}"
                    )
                raise ValueError(
                    f"External agent '{agent_name}' not found.\n"
                    "No external agents are currently installed.\n"
                    "Run: mobilerun run --help  to see available agents."
                )

            # Resolve config — missing section is fine, agent may use DEFAULT_CONFIG or env vars
            agent_config = self.config.external_agents.get(agent_name) or {}
            final_config = {**agent_module["config"], **agent_config}

            # Resolve device serial and get raw AdbDevice
            device_serial = self.resolved_device_config.serial
            if device_serial is None:
                devices = await adb.list()
                if not devices:
                    raise ValueError("No connected Android devices found.")
                device_serial = devices[0].serial

            adb_device = await adb.device(serial=device_serial)

            logger.info(f"🤖 Using external agent: {agent_name}")

            result = await agent_module["run"](
                device=adb_device,
                instruction=self.shared_state.instruction,
                config=final_config,
                max_steps=self.config.agent.max_steps,
            )

            return FinalizeEvent(success=result["success"], reason=result["reason"])

        # ── 1. Create driver ──────────────────────────────────────────
        if self.config.agent.reasoning:
            vision_enabled = (
                self.config.agent.vision_only or self.config.agent.manager.vision
            )
        else:
            vision_enabled = (
                self.config.agent.vision_only or self.config.agent.fast_agent.vision
            )

        is_ios = self.resolved_device_config.platform.lower() == "ios"
        control_backend = _normalize_control_backend(
            self.resolved_device_config.control_backend
        )
        if control_backend and control_backend != VISUAL_REMOTE_CONNECTION:
            raise ValueError(
                "Unsupported device control backend "
                f"'{self.resolved_device_config.control_backend}'. Supported: "
                f"{VISUAL_REMOTE_CONNECTION}. Omit control backend for the "
                "platform default."
            )
        is_visual_remote = control_backend == VISUAL_REMOTE_CONNECTION

        if self._injected_driver is not None:
            driver = self._injected_driver
        elif is_visual_remote:
            visual_remote_url = (
                self.resolved_device_config.serial or VISUAL_REMOTE_DEFAULT_URL
            )
            driver = VisualRemoteDriver(
                url=visual_remote_url,
                device_id=self.resolved_device_config.device_id,
            )
            await driver.connect()
        elif is_ios:
            ios_url = self.resolved_device_config.serial
            if not ios_url:
                ios_url = await discover_ios_portal()
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
        if stealth_enabled and not is_ios and not is_visual_remote:
            driver = StealthDriver(driver)

        # Wrap with RecordingDriver if trajectory saving enabled
        if self.config.logging.save_trajectory != "none":
            if not isinstance(driver, RecordingDriver):
                driver = RecordingDriver(driver)

        self.driver = driver
        self.shared_state.platform = driver.platform

        # ── 2. Create state provider ──────────────────────────────────
        if self._injected_state_provider is not None:
            self.state_provider = self._injected_state_provider
        elif self.config.agent.vision_only or is_visual_remote:
            self.state_provider = ScreenshotOnlyStateProvider(driver)
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
        registry, standard_tool_names = await build_tool_registry(
            supported_buttons=driver.supported_buttons,
            credential_manager=self.credential_manager,
            platform="ios" if driver.platform.lower() == "ios" else "android",
            exact_app_launch=is_visual_remote,
            screenshot_only=getattr(
                self.state_provider,
                "requires_coordinate_tools",
                False,
            ),
        )

        # User custom tools
        if self.user_custom_tools:
            registry.register_from_dict(self.user_custom_tools)

        # MCP tools
        if self.config.mcp and self.config.mcp.enabled:
            self.mcp_manager = MCPClientManager(self.config.mcp)
            await self.mcp_manager.discover_tools()
            mcp_tools = mcp_to_mobilerun_tools(self.mcp_manager)
            if mcp_tools:
                registry.register_from_dict(mcp_tools)

        # Capability-based filtering (deps vs driver+provider supported)
        capabilities = driver.supported | self.state_provider.supported
        registry.disable_unsupported(capabilities)

        # Config-level filtering
        disabled_tools = (
            self.config.tools.disabled_tools
            if self.config.tools and self.config.tools.disabled_tools
            else []
        )
        disabled_tools = _effective_disabled_tools(disabled_tools, self.state_provider)
        if disabled_tools:
            registry.disable(disabled_tools)

        self.registry = registry
        self.standard_tool_names = standard_tool_names

        # ── 4. Create ActionContext ────────────────────────────────────
        self.action_ctx = ActionContext(
            driver=driver,
            ui=None,  # populated each step by state_provider
            shared_state=self.shared_state,
            state_provider=self.state_provider,
            app_opener_llm=self.app_opener_llm,
            credential_manager=self.credential_manager,
            streaming=self.config.agent.streaming,
        )

        # ── 5. Wire up sub-agents ─────────────────────────────────────
        if self.config.agent.reasoning and self.executor_agent:
            self.manager_agent.action_ctx = self.action_ctx
            self.manager_agent.state_provider = self.state_provider
            self.manager_agent.registry = self.registry
            self.manager_agent.save_trajectory = self.config.logging.save_trajectory
            self.manager_agent.standard_tool_names = self.standard_tool_names
            self.executor_agent.registry = self.registry
            self.executor_agent.action_ctx = self.action_ctx

        # ── 6. Fetch device date once ─────────────────────────────────
        self.shared_state.device_date = await driver.get_date()

        # ── 7. Telemetry init event ───────────────────────────────────
        capture(
            MobileAgentInitEvent(
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
                    "app_opener": (
                        self.app_opener_llm.class_name()
                        if self.app_opener_llm
                        else "None"
                    ),
                },
                tools=",".join(sorted(standard_tool_names)),
                max_steps=self.config.agent.max_steps,
                timeout=self._init_timeout,
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
                custom_prompts=self._init_prompts,
            ),
            self.user_id,
        )

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
    # execute_task — FastAgent
    # ========================================================================

    @step
    async def execute_task(
        self, ctx: Context, ev: FastAgentExecuteEvent
    ) -> FastAgentResultEvent:
        """Execute a single task using FastAgent."""

        logger.debug(f"🔧 Executing task: {ev.instruction}")

        try:
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
                remembered_info=self.shared_state.fast_memory,
            )

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

                if isinstance(nested_ev, FastAgentOutputEvent):
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
            logger.error(f"❌ Error during MobileAgent execution: {e}")
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
    ) -> ExecutorInputEvent | FinalizeEvent | ManagerInputEvent:
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

        logger.debug(f"▶️  Proceeding to Executor with subgoal: {ev.current_subgoal}")
        return ExecutorInputEvent(current_subgoal=ev.current_subgoal)

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
        self.shared_state.action_history.append(result["action"])
        self.shared_state.summary_history.append(result["summary"])
        self.shared_state.action_outcomes.append(result["outcome"])
        self.shared_state.error_descriptions.append(result["error"])
        self.shared_state.last_action = result["action"]
        self.shared_state.last_summary = result["summary"]

        return ExecutorResultEvent(
            action=result["action"],
            outcome=result["outcome"],
            error=result["error"],
            summary=result["summary"],
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
    # Finalize
    # ========================================================================

    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> ResultEvent:
        self.shared_state.workflow_completed = True
        ctx.write_event_to_stream(ev)
        capture(
            MobileAgentFinalizeEvent(
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


# Legacy alias — deprecated, will be removed in v0.8.0
DroidAgent = MobileAgent
