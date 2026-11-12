"""Run tasks with the framework agent and native async drivers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click
from async_adbutils import adb
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mobilerun.agent.external import list_agents
from mobilerun.cli.options import (
    DEFAULT_API_URL,
    Command,
    coro,
    device_options,
    emit,
    examples,
    json_option,
    llm_url_option,
    max_steps_option,
    model_option,
    tcp_option,
)
from mobilerun.cli.providers import PROVIDERS, llm_url_kwargs, provider_option
from mobilerun.cli.selection import _native_driver, select_device
from mobilerun.log_handlers import CLILogHandler, configure_logging

console = Console(stderr=True)


if TYPE_CHECKING:
    from mobilerun import ResultEvent
    from mobilerun.config_manager import MobileConfig


def _force_screenshot_only_vision(config: MobileConfig) -> None:
    config.agent.vision_only = True
    config.agent.manager.vision = True
    config.agent.executor.vision = True
    config.agent.fast_agent.vision = True


def _setup_cli_logging(debug: bool) -> None:
    """Configure the mobilerun logger with a CLILogHandler."""
    handler = CLILogHandler()
    handler.console = console
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s", "%H:%M:%S")
        if debug
        else logging.Formatter("%(message)s")
    )
    configure_logging(debug=debug, handler=handler)


async def _run_task(
    command: str,
    config_path: str | None = None,
    device: str | None = None,
    agent: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    steps: int | None = None,
    vision: bool | None = None,
    vision_only: bool | None = None,
    reasoning: bool | None = None,
    stream: bool | None = None,
    tracing: bool | None = None,
    debug: bool | None = None,
    tcp: bool | None = None,
    control_backend: str | None = None,
    device_id: str | None = None,
    save_trajectory: str | None = None,
    platform: str | None = None,
    temperature: float | None = None,
    llm_url: str | None = None,
    target: str = "local",
    api_url: str = DEFAULT_API_URL,
) -> tuple[bool, str | None, int | None]:
    """Execute a task and return the result-event fields."""
    from mobilerun_core_local.driver.visual_remote import VISUAL_REMOTE_CONNECTION

    from mobilerun import MobileAgent
    from mobilerun.agent.utils.llm_picker import load_llm
    from mobilerun.cli.event_handler import EventHandler
    from mobilerun.config_manager import ConfigLoader
    from mobilerun.telemetry import print_telemetry_message

    config = ConfigLoader.load(config_path)

    # Print cloud link in a box
    if config.logging.rich_text:
        cloud_text = Text()
        cloud_text.append("✨ Try Mobilerun Cloud: ", style="bold cyan")
        cloud_text.append("https://cloud.mobilerun.ai/sign-in", style="bold blue underline")
        cloud_panel = Panel(
            cloud_text,
            border_style="cyan",
            padding=(0, 1),
        )
        console.print(cloud_panel)
    else:
        console.print("\n✨ Try Mobilerun Cloud: https://cloud.mobilerun.ai/sign-in\n")

    # Initialize logging
    debug_mode = debug if debug is not None else config.logging.debug
    _setup_cli_logging(debug_mode)
    logger = logging.getLogger("mobilerun")
    droid_agent = None
    injected_driver = None
    is_cloud = False

    try:
        logger.info(f"🚀 Starting: {command}")
        print_telemetry_message(config_enabled=config.telemetry.enabled)

        # ================================================================
        # STEP 1: Apply CLI overrides via direct mutation
        # ================================================================

        # Vision overrides
        if vision_only is not None:
            config.agent.vision_only = vision_only

        if config.agent.vision_only:
            _force_screenshot_only_vision(config)
            logger.debug("CLI override: vision_only=True")
        elif vision is not None:
            # --vision flag overrides all agents
            config.agent.manager.vision = vision
            config.agent.executor.vision = vision
            config.agent.fast_agent.vision = vision
            logger.debug(f"CLI override: vision={vision} (all agents)")
        # Agent overrides
        if agent is not None:
            config.agent.name = agent
        if steps is not None:
            config.agent.max_steps = steps
        if reasoning is not None:
            config.agent.reasoning = reasoning
        if stream is not None:
            config.agent.streaming = stream

        # Device overrides
        if platform is not None:
            # Resolve against the loaded platform before changing token ownership.
            config.device.auth_token = config.device.resolve_auth_token(platform)
            config.device.platform = platform
        if device is not None:
            config.device.serial = device
        if tcp is not None:
            config.device.use_tcp = tcp
        if control_backend is not None:
            config.device.control_backend = control_backend
        if device_id is not None:
            config.device.device_id = device_id

        if (config.device.control_backend or "").strip().lower() == VISUAL_REMOTE_CONNECTION:
            _force_screenshot_only_vision(config)
            logger.debug("Control backend visual-remote forces screenshot-only vision")

        # Logging overrides
        if debug is not None:
            config.logging.debug = debug
        if save_trajectory is not None:
            config.logging.save_trajectory = save_trajectory

        # Tracing overrides
        if tracing is not None:
            config.tracing.enabled = tracing

        # The framework keeps its native async drivers. Core decides the backend.
        if not config.device.control_backend:
            selection = await select_device(
                config.device.serial,
                config_path,
                tcp,
                config.device.platform,
                target,
                discover=False,
                config=config,
            )
            config.device.serial = selection.device
            config.device.platform = selection.platform
            if selection.backend == "cloud":
                is_cloud = True
                if tcp:
                    raise click.UsageError("--tcp is only supported for local Android devices")
                injected_driver = await _native_driver(selection, config, api_url)
                # CloudDriver supplies Android-shaped trees for every platform.
                config.device.platform = "android"
                config.device.serial = None
                config.device.device_id = selection.device
                config.device.use_tcp = False
                config.device.auto_setup = False

        # ================================================================
        # STEP 2: Initialize MobileAgent with config
        # ================================================================

        mode = "planning with reasoning" if config.agent.reasoning else "direct execution"
        logger.info(f"🤖 Agent mode: {mode}")
        logger.info(
            f"👁️  Vision settings: Manager={config.agent.manager.vision}, "
            f"Executor={config.agent.executor.vision}, FastAgent={config.agent.fast_agent.vision}"
        )

        if config.tracing.enabled:
            logger.info("🔍 Tracing enabled")

        # Build MobileAgent kwargs for LLM loading
        droid_agent_kwargs = {"runtype": "cli"}
        llm = None

        if provider or model:
            if not (provider and model):
                raise ValueError("--provider and --model must be given together")
            llm_kwargs = {}
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            llm_kwargs.update(llm_url_kwargs(provider, llm_url))
            llm = load_llm(provider, model=model, **llm_kwargs)
        else:
            if llm_url is not None:
                for profile in config.llm_profiles.values():
                    profile.kwargs.update(llm_url_kwargs(profile.provider, llm_url))
            if temperature is not None:
                droid_agent_kwargs["temperature"] = temperature

        droid_agent = MobileAgent(
            goal=command,
            llms=llm,
            config=config,
            driver=injected_driver,
            timeout=1000,
            **droid_agent_kwargs,
        )

        # ================================================================
        # STEP 3: Run agent
        # ================================================================

        logger.debug("▶️  Starting agent execution...")
        logger.debug("Press Ctrl+C to stop")

        event_handler = EventHandler()

        handler = droid_agent.run()
        async for event in handler.stream_events():
            event_handler.handle(event)
        result: ResultEvent = await handler
        return result.success, result.reason, result.steps

    finally:
        if not is_cloud:
            await _cleanup_android_keyboard(config, getattr(droid_agent, "android_serial", None))


async def run_command(*args, **kwargs) -> bool:
    """Keep the boolean interface used by macro handoff and Python callers."""
    try:
        success, _, _ = await _run_task(*args, **kwargs)
        return success
    except Exception:
        return False


async def _cleanup_android_keyboard(config: MobileConfig, serial: str | None = None) -> None:
    from mobilerun_core_local.driver.visual_remote import VISUAL_REMOTE_CONNECTION

    from mobilerun.agent.utils.android_device import is_portal_http_url

    platform = (config.device.platform or "").lower()
    control_backend = (config.device.control_backend or "").lower()
    if platform == "ios" or control_backend == VISUAL_REMOTE_CONNECTION:
        return

    # Without a selected device the keyboard was never enabled.
    serial = serial or config.device.serial
    if not serial or is_portal_http_url(serial):
        return

    try:
        device_obj = await adb.device(serial)
        if device_obj:
            from mobilerun_core_local.driver.android.portal import (
                PORTAL_PACKAGE_NAME,
                portal_ime_id,
            )

            ime = portal_ime_id(PORTAL_PACKAGE_NAME)
            await device_obj.shell(f"ime disable {ime}")
    except Exception:
        click.echo("Failed to disable Mobilerun keyboard", err=True)


class AgentOption(click.Option):
    def installed_type(self):
        self.type = click.Choice(list_agents())
        return self.type

    def get_help_record(self, ctx):
        if not self.installed_type().choices:
            return None
        return super().get_help_record(ctx)

    def type_cast_value(self, ctx, value):
        if value is not None and not self.installed_type().choices:
            raise click.BadParameter(
                "No external agents are installed; install one before using --agent.",
                ctx=ctx, param=self,
            )
        return super().type_cast_value(ctx, value)

    def shell_complete(self, ctx, incomplete):
        return self.installed_type().shell_complete(ctx, self, incomplete)


@click.command(
    cls=Command,
    epilog=examples(
        'mobilerun run "Open Settings"',
        'mobilerun run "Open Settings" -d DEVICE -p openai -m MODEL',
    ),
)
@click.argument("task")
@device_options
@tcp_option
@json_option
@provider_option
@model_option
@llm_url_option
@max_steps_option
@click.option("-T", "--temperature", type=float, help="LLM sampling temperature.")
@click.option("-a", "--agent", cls=AgentOption, help="External agent.")
@click.option("-r", "--reasoning/--no-reasoning", default=None, help="Enable agent reasoning.")
@click.option("--vision/--no-vision", default=None, help="Enable vision for all agents.")
@click.option("--vision-only", is_flag=True, default=None, help="Use screenshots without the UI tree.")
@click.option("--stream/--no-stream", default=None, help="Stream agent responses.")
@click.option("--tracing/--no-tracing", default=None, help="Enable tracing.")
@click.option("--backend", type=click.Choice(["visual-remote"]), help="Device control backend.")
@click.option("--backend-device", help="Device id for a visual remote backend.")
@click.option(
    "--save-trajectory", type=click.Choice(["none", "step", "action"]),
    help="Choose which trajectory data to save.",
)
@coro
async def run(
    task,
    config_path,
    device,
    target,
    platform,
    api_url,
    tcp,
    verbose,
    json_output,
    provider,
    model,
    llm_url,
    max_steps,
    temperature,
    agent,
    reasoning,
    vision,
    vision_only,
    stream,
    tracing,
    backend,
    backend_device,
    save_trajectory,
):
    """Run a task on a mobile device using natural language."""
    if (provider is None) != (model is None):
        raise click.UsageError("--provider and --model must be given together")
    success, reason, steps = await _run_task(
        command=task,
        config_path=config_path,
        device=device,
        target=target,
        platform=platform,
        api_url=api_url,
        tcp=tcp,
        debug=verbose,
        provider=PROVIDERS[provider].runtime if provider else None,
        model=model,
        llm_url=llm_url,
        steps=max_steps,
        temperature=temperature,
        agent=agent,
        reasoning=reasoning,
        vision=vision,
        vision_only=vision_only,
        stream=stream,
        tracing=tracing,
        control_backend=backend,
        device_id=backend_device,
        save_trajectory=save_trajectory,
    )
    if json_output:
        emit({"success": success, "reason": reason, "steps": steps}, json_output=True)
    if not success:
        raise click.ClickException(reason or "Task failed")
