"""
Droidrun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import asyncio
import logging
import os
import sys
import warnings
from contextlib import nullcontext
from functools import wraps

import click
import importlib.metadata
import tomllib
from pathlib import Path
from async_adbutils import adb
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from droidrun import ResultEvent, DroidAgent
from droidrun.log_handlers import CLILogHandler, configure_logging
from droidrun.cli.configure_wizard import (
    ConfigureWizardCallbacks,
    run_configure_wizard,
)
from droidrun.cli.event_handler import EventHandler
from droidrun.config_manager import ConfigLoader
from droidrun.cli.device_commands import device_cli
from droidrun.macro.cli import macro_cli
from droidrun.portal import (
    PORTAL_PACKAGE_NAME,
    download_portal_apk,
    download_versioned_portal_apk,
    enable_portal_accessibility,
    ping_portal,
    ping_portal_content,
    ping_portal_tcp,
    setup_portal,
)
from droidrun.agent.external import list_agents
from droidrun.agent.utils.llm_picker import load_llm
from droidrun.agent.utils.oauth.openai_oauth_llm import (
    DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH,
)
from droidrun.cli.oauth_actions import (
    run_openai_oauth_login,
    run_gemini_oauth_login,
    run_anthropic_setup_token_oauth,
    save_anthropic_setup_token,
)
from droidrun.config_manager.credential_paths import (
    ANTHROPIC_OAUTH_CREDENTIAL_PATH,
    GEMINI_OAUTH_CREDENTIAL_PATH,
)
from droidrun.telemetry import print_telemetry_message
from droidrun.tools.driver.ios import discover_ios_portal, validate_ios_portal_url

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

console = Console()


def _setup_cli_logging(debug: bool) -> None:
    """Configure the droidrun logger with a CLILogHandler."""
    handler = CLILogHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s", "%H:%M:%S")
        if debug
        else logging.Formatter("%(message)s")
    )
    configure_logging(debug=debug, handler=handler)


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper




async def run_command(
    command: str,
    config_path: str | None = None,
    device: str | None = None,
    agent: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    steps: int | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    vision: bool | None = None,
    manager_vision: bool | None = None,
    executor_vision: bool | None = None,
    fast_agent_vision: bool | None = None,
    reasoning: bool | None = None,
    stream: bool | None = None,
    tracing: bool | None = None,
    debug: bool | None = None,
    tcp: bool | None = None,
    save_trajectory: str | None = None,
    ios: bool = False,
    temperature: float | None = None,
    **kwargs,
) -> bool:
    """Run a command on your Android device using natural language.

    Returns:
        bool: True if the task completed successfully, False otherwise.
    """
    config = ConfigLoader.load(config_path)

    # Print cloud link in a box
    if config.logging.rich_text:
        cloud_text = Text()
        cloud_text.append("✨ Try Droidrun Cloud: ", style="bold cyan")
        cloud_text.append(
            "https://cloud.droidrun.ai/sign-in", style="bold blue underline"
        )
        cloud_panel = Panel(
            cloud_text,
            border_style="cyan",
            padding=(0, 1),
        )
        console.print(cloud_panel)
    else:
        console.print("\n✨ Try Droidrun Cloud: https://cloud.droidrun.ai/sign-in\n")

    # Initialize logging
    debug_mode = debug if debug is not None else config.logging.debug
    _setup_cli_logging(debug_mode)
    logger = logging.getLogger("droidrun")

    try:
        logger.info(f"🚀 Starting: {command}")
        print_telemetry_message()

        # ================================================================
        # STEP 1: Apply CLI overrides via direct mutation
        # ================================================================

        # Vision overrides
        if vision is not None:
            # --vision flag overrides all agents
            config.agent.manager.vision = vision
            config.agent.executor.vision = vision
            config.agent.fast_agent.vision = vision
            logger.debug(f"CLI override: vision={vision} (all agents)")
        else:
            # Apply individual agent vision overrides
            if manager_vision is not None:
                config.agent.manager.vision = manager_vision
            if executor_vision is not None:
                config.agent.executor.vision = executor_vision
            if fast_agent_vision is not None:
                config.agent.fast_agent.vision = fast_agent_vision

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
        if device is not None:
            config.device.serial = device
        if tcp is not None:
            config.device.use_tcp = tcp

        # Logging overrides
        if debug is not None:
            config.logging.debug = debug
        if save_trajectory is not None:
            config.logging.save_trajectory = save_trajectory

        # Tracing overrides
        if tracing is not None:
            config.tracing.enabled = tracing

        # Platform overrides
        if ios:
            config.device.platform = "ios"
            if config.device.serial:
                config.device.serial = validate_ios_portal_url(config.device.serial)
            else:
                logger.info("🔍 Searching for iOS portal...")
                config.device.serial = await discover_ios_portal()

        # ================================================================
        # STEP 2: Initialize DroidAgent with config
        # ================================================================

        mode = (
            "planning with reasoning" if config.agent.reasoning else "direct execution"
        )
        logger.info(f"🤖 Agent mode: {mode}")
        logger.info(
            f"👁️  Vision settings: Manager={config.agent.manager.vision}, "
            f"Executor={config.agent.executor.vision}, FastAgent={config.agent.fast_agent.vision}"
        )

        if config.tracing.enabled:
            logger.info("🔍 Tracing enabled")

        # Build DroidAgent kwargs for LLM loading
        droid_agent_kwargs = {"runtype": "cli"}
        llm = None

        if provider or model:
            assert (
                provider and model
            ), "Either both provider and model must be provided or none of them"
            llm_kwargs = {}
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            if base_url is not None:
                llm_kwargs["base_url"] = base_url
            if api_base is not None:
                llm_kwargs["api_base"] = api_base
            llm = load_llm(provider, model=model, **llm_kwargs, **kwargs)
        else:
            if temperature is not None:
                droid_agent_kwargs["temperature"] = temperature
            if base_url is not None:
                droid_agent_kwargs["base_url"] = base_url
            if api_base is not None:
                droid_agent_kwargs["api_base"] = api_base

        droid_agent = DroidAgent(
            goal=command,
            llms=llm,
            config=config,
            timeout=1000,
            **droid_agent_kwargs,
        )

        # ================================================================
        # STEP 3: Run agent
        # ================================================================

        logger.debug("▶️  Starting agent execution...")
        logger.debug("Press Ctrl+C to stop")

        event_handler = EventHandler()

        try:
            handler = droid_agent.run()

            async for event in handler.stream_events():
                event_handler.handle(event)
            result: ResultEvent = await handler
            return result.success

        except KeyboardInterrupt:
            logger.info("⏹️ Stopped by user")
            return False

        except Exception as e:
            err_desc = str(e) or type(e).__name__
            logger.error(f"💥 Error: {err_desc}")
            if config.logging.debug:
                import traceback

                logger.debug(traceback.format_exc())
            return False

    except Exception as e:
        err_desc = str(e) or type(e).__name__
        logger.error(f"💥 Setup error: {err_desc}")
        if debug_mode:
            import traceback

            logger.debug(traceback.format_exc())
        return False


def _print_version(ctx, param, value):
    """Click callback to print version and exit early when --version is passed."""
    if not value or ctx.resilient_parsing:
        return
    version = None
    try:
        version = importlib.metadata.version("droidrun")
        # print("debug: step 1")
    except Exception:
        pass

    if not version:
        try:
            from droidrun import __version__ as pkg_version

            version = pkg_version
            # print("debug: step 2")
        except Exception:
            pass

    if not version:
        try:
            repo_root = Path(__file__).resolve().parents[2]
            pyproject = repo_root / "pyproject.toml"
            if pyproject.exists():
                with pyproject.open("rb") as f:
                    data = tomllib.load(f)
                    version = data.get("project", {}).get("version")
            # print("debug: step 3")
        except Exception:
            version = None

    if not version:
        version = "unknown"
    click.echo(f"v{version}")
    ctx.exit()


@click.group()
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show droidrun version and exit",
)
def cli():
    """Droidrun - Control your Android device through LLM agents."""
    pass


def _print_oauth_login_success(provider_label: str, credential_path: str) -> None:
    console.print(f"[green]{provider_label} login succeeded.[/]")
    console.print(f"[blue]Credentials saved to:[/] {Path(credential_path).expanduser()}")


def _run_openai_oauth_login(credential_path: str, model: str | None, **kwargs) -> None:
    run_openai_oauth_login(credential_path=credential_path, model=model, **kwargs)
    _print_oauth_login_success("OpenAI", credential_path)


def _run_gemini_oauth_login(credential_path: str, model: str | None, **kwargs) -> None:
    run_gemini_oauth_login(credential_path=credential_path, model=model, **kwargs)
    _print_oauth_login_success("Gemini", credential_path)


def _run_anthropic_oauth_login(credential_path: str, **kwargs) -> None:
    """Run the full Anthropic OAuth flow inline and save the token."""
    console.print("[blue]Opening browser for Anthropic login...[/]")
    token = run_anthropic_setup_token_oauth(**kwargs)
    save_anthropic_setup_token(credential_path, token)
    _print_oauth_login_success("Anthropic", credential_path)


def _prompt_anthropic_setup_token(token: str | None) -> str:
    if token:
        return token
    return click.prompt("Paste your Anthropic setup token", hide_input=True)


try:
    _available_agents = list_agents()
except Exception:
    _available_agents = []


@cli.command()
@click.argument("command", type=str)
@click.option("--config", "-c", help="Path to custom config file", default=None)
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--agent",
    "-a",
    type=click.Choice(_available_agents) if _available_agents else None,
    help="External agent to use"
    + (
        f" [{', '.join(_available_agents)}]"
        if _available_agents
        else " (none available)"
    ),
    default=None,
)
@click.option(
    "--provider",
    "-p",
    help="LLM provider (OpenAI, openai_oauth, Ollama, Anthropic, anthropic_oauth, GoogleGenAI, gemini_oauth_code_assist, DeepSeek)",
    default=None,
)
@click.option(
    "--model",
    "-m",
    help="LLM model name",
    default=None,
)
@click.option("--temperature", type=float, help="Temperature for LLM", default=None)
@click.option("--steps", type=int, help="Maximum number of steps", default=None)
@click.option(
    "--base_url",
    "-u",
    help="Base URL for API (e.g., OpenRouter or Ollama)",
    default=None,
)
@click.option(
    "--api_base",
    help="Base URL for API (e.g., OpenAI or OpenAI-Like)",
    default=None,
)
@click.option(
    "--vision/--no-vision",
    default=None,
    help="Enable vision capabilites by using screenshots for all agents.",
)
@click.option(
    "--reasoning/--no-reasoning", default=None, help="Enable planning with reasoning"
)
@click.option(
    "--stream/--no-stream",
    default=None,
    help="Stream LLM responses to console in real-time",
)
@click.option(
    "--tracing/--no-tracing", default=None, help="Enable Arize Phoenix tracing"
)
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug logging")
@click.option(
    "--tcp/--no-tcp",
    default=None,
    help="Use TCP communication for device control",
)
@click.option(
    "--save-trajectory",
    type=click.Choice(["none", "step", "action"]),
    help="Trajectory saving level: none (no saving), step (save per step), action (save per action)",
    default=None,
)
@click.option("--ios", is_flag=True, default=False, help="Run on iOS device")
@coro
async def run(
    command: str,
    config: str | None,
    device: str | None,
    agent: str | None,
    provider: str | None,
    model: str | None,
    steps: int | None,
    base_url: str | None,
    api_base: str | None,
    temperature: float | None,
    vision: bool | None,
    reasoning: bool | None,
    stream: bool | None,
    tracing: bool | None,
    debug: bool | None,
    tcp: bool | None,
    save_trajectory: str | None,
    ios: bool,
):
    """Run a command on your Android device using natural language."""

    try:
        success = await run_command(
            command=command,
            config_path=config,
            device=device,
            agent=agent,
            provider=provider,
            model=model,
            steps=steps,
            base_url=base_url,
            api_base=api_base,
            vision=vision,
            reasoning=reasoning,
            stream=stream,
            tracing=tracing,
            debug=debug,
            tcp=tcp,
            temperature=temperature,
            save_trajectory=save_trajectory,
            ios=ios,
        )
    finally:
        # Disable Droidrun keyboard after execution
        # Note: Port forwards are managed automatically and persist until device disconnect
        try:
            if not ios:
                device_obj = await adb.device(device)
                if device_obj:
                    await device_obj.shell(
                        "ime disable com.droidrun.portal/.input.DroidrunKeyboardIME"
                    )
        except Exception:
            click.echo("Failed to disable Droidrun keyboard")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


@cli.command()
@coro
async def devices():
    """List connected Android devices."""
    try:
        devices = await adb.list()
        if not devices:
            console.print("[yellow]No devices connected.[/]")
            return

        console.print(f"[green]Found {len(devices)} connected device(s):[/]")
        for device in devices:
            console.print(f"  • [bold]{device.serial}[/]")
    except Exception as e:
        console.print(f"[red]Error listing devices: {e}[/]")


@cli.command()
@click.argument("serial")
@coro
async def connect(serial: str):
    """Connect to a device over TCP/IP."""
    try:
        device = await adb.connect(serial)
        if device.count("already connected"):
            console.print(f"[green]Successfully connected to {serial}[/]")
        else:
            console.print(f"[red]Failed to connect to {serial}: {device}[/]")
    except Exception as e:
        console.print(f"[red]Error connecting to device: {e}[/]")


@cli.command()
@click.argument("serial")
@coro
async def disconnect(serial: str):
    """Disconnect from a device."""
    try:
        success = await adb.disconnect(serial, raise_error=True)
        if success:
            console.print(f"[green]Successfully disconnected from {serial}[/]")
        else:
            console.print(f"[yellow]Device {serial} was not connected[/]")
    except Exception as e:
        console.print(f"[red]Error disconnecting from device: {e}[/]")


async def _setup_portal(
    path: str | None,
    device: str | None,
    debug: bool,
    latest: bool = False,
    specific_version: str | None = None,
):
    """Internal async function to install and enable the Droidrun Portal on a device."""
    try:
        if not device:
            devices = await adb.list()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return

            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")

        device_obj = await adb.device(device)
        if not device_obj:
            console.print(
                f"[bold red]Error:[/] Could not get device object for {device}"
            )
            return

        # CLI-specific options: path, specific_version, latest
        if path:
            console.print(f"[bold blue]Using provided APK:[/] {path}")
            apk_context = nullcontext(path)
        elif specific_version:
            version = specific_version.lstrip("v")
            version = f"v{version}"
            download_base = (
                "https://github.com/droidrun/droidrun-portal/releases/download"
            )
            apk_context = download_versioned_portal_apk(version, download_base, debug)
        elif latest:
            console.print("[bold blue]Downloading latest Portal APK...[/]")
            apk_context = download_portal_apk(debug)
        else:
            # Default: delegate to shared setup_portal()
            success = await setup_portal(device_obj, debug)
            if success:
                console.print(
                    "\n[bold green]Setup complete![/] The Droidrun Portal is now installed and ready to use."
                )
            else:
                console.print(
                    "[bold red]Setup failed.[/] Run 'droidrun doctor' for diagnostics."
                )
            return

        # Install from explicit path/version/latest
        with apk_context as apk_path:
            if not os.path.exists(apk_path):
                console.print(f"[bold red]Error:[/] APK file not found at {apk_path}")
                return

            console.print(f"[bold blue]Step 1/2: Installing APK:[/] {apk_path}")
            try:
                await device_obj.install(
                    apk_path, uninstall=True, flags=["-g"], silent=not debug
                )
            except Exception as e:
                console.print(f"[bold red]Installation failed:[/] {e}")
                return

            console.print("[bold green]Installation successful![/]")

            console.print("[bold blue]Step 2/2: Enabling accessibility service[/]")

            try:
                await enable_portal_accessibility(device_obj)

                console.print("[green]Accessibility service enabled successfully![/]")
                console.print(
                    "\n[bold green]Setup complete![/] The Droidrun Portal is now installed and ready to use."
                )

            except Exception as e:
                console.print(
                    f"[yellow]Could not automatically enable accessibility service: {e}[/]"
                )
                console.print(
                    "[yellow]Opening accessibility settings for manual configuration...[/]"
                )

                await device_obj.shell(
                    "am start -a android.settings.ACCESSIBILITY_SETTINGS"
                )

                console.print(
                    "\n[yellow]Please complete the following steps on your device:[/]"
                )
                console.print(
                    f"1. Find [bold]{PORTAL_PACKAGE_NAME}[/] in the accessibility services list"
                )
                console.print("2. Tap on the service name")
                console.print(
                    "3. Toggle the switch to [bold]ON[/] to enable the service"
                )
                console.print("4. Accept any permission dialogs that appear")

                console.print(
                    "\n[bold green]APK installation complete![/] Please manually enable the accessibility service using the steps above."
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")

        if debug:
            import traceback

            traceback.print_exc()


@cli.command()
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--path",
    help="Path to the Droidrun Portal APK to install on the device. If not provided, the latest portal apk version will be downloaded and installed.",
    default=None,
)
@click.option(
    "--portal-version",
    "-pv",
    help="Specific Portal version to install (e.g., 0.4.7)",
    default=None,
)
@click.option(
    "--latest",
    is_flag=True,
    help="Install latest Portal instead of compatible version",
    default=False,
)
@click.option(
    "--debug", is_flag=True, help="Enable verbose debug logging", default=False
)
@coro
async def setup(
    path: str | None,
    device: str | None,
    portal_version: str | None,
    latest: bool,
    debug: bool,
):
    """Install and enable the Droidrun Portal on a device."""
    await _setup_portal(path, device, debug, latest, portal_version)


@cli.command()
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--tcp/--no-tcp",
    default=None,
    help="Use TCP communication for device control",
)
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug logging")
@coro
async def ping(device: str | None, tcp: bool | None, debug: bool | None):
    """Ping a device to check if it is ready and accessible."""
    # Handle None defaults
    debug_mode = debug if debug is not None else False
    use_tcp_mode = tcp if tcp is not None else False

    try:
        device_obj = await adb.device(device)
        if not device_obj:
            console.print(f"[bold red]Error:[/] Could not find device {device}")
            return

        await ping_portal(device_obj, debug_mode)

        if use_tcp_mode:
            await ping_portal_tcp(device_obj, debug_mode)
        else:
            await ping_portal_content(device_obj, debug_mode)

        console.print(
            "[bold green]Portal is installed and accessible. You're good to go![/]"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if debug_mode:
            import traceback

            traceback.print_exc()


# Add macro commands as a subgroup
cli.add_command(macro_cli, name="macro")

# Add device action commands as a subgroup
cli.add_command(device_cli, name="device")


@cli.command()
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug output")
@coro
async def doctor(device: str | None, debug: bool | None):
    """Check system health and diagnose issues."""
    from droidrun.cli.doctor import run_doctor

    await run_doctor(device, debug if debug is not None else False)


@cli.command()
def tui():
    """Launch the Droidrun Terminal User Interface."""
    from droidrun.cli.tui import run_tui

    run_tui()


@cli.command(name="setup-token")
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Max seconds to wait for the browser callback.",
)
@click.option(
    "--callback-host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind the local OAuth callback server.",
)
@click.option(
    "--callback-port",
    type=int,
    default=0,
    show_default=True,
    help="Port to bind the local OAuth callback server. Use 0 for auto.",
)
@click.option(
    "--callback-path",
    default="/callback",
    show_default=True,
    help="Callback path for the local OAuth server.",
)
@click.option(
    "--open-browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the authorization URL automatically.",
)
def setup_token(
    timeout: float,
    callback_host: str,
    callback_port: int,
    callback_path: str,
    open_browser: bool,
):
    """Create a long-lived Anthropic setup token using Droidrun's native OAuth flow."""
    console.print(
        "This will guide you through long-lived (1-year) auth token setup for your Claude account."
    )
    token = run_anthropic_setup_token_oauth(
        timeout=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        open_browser=open_browser,
    )
    console.print("\n[green]Setup token created.[/]")
    console.print("Paste this token into `droidrun configure` or `droidrun anthropic login`.")
    click.echo(token)


@cli.command(name="configure")
@click.option(
    "--provider",
    type=str,
    default=None,
    help="Provider family (gemini, openai, anthropic, ollama, openai_like, minimax, zai).",
)
@click.option(
    "--auth-mode",
    type=str,
    default=None,
    help="Auth mode for the selected provider family.",
)
@click.option("--model", type=str, default=None, help="Model to configure.")
@click.option("--api-key", type=str, default=None, help="API key for API-key providers.")
@click.option("--base-url", type=str, default=None, help="Base URL override for compatible providers.")
def configure(
    provider: str | None,
    auth_mode: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
):
    """Configure LLM provider, auth mode, and model."""
    run_configure_wizard(
        console,
        ConfigureWizardCallbacks(
            run_openai_oauth_login=_run_openai_oauth_login,
            run_anthropic_oauth_login=_run_anthropic_oauth_login,
            run_gemini_oauth_login=_run_gemini_oauth_login,
        ),
        provider=provider,
        auth_mode=auth_mode,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


@cli.group()
def openai():
    """OpenAI OAuth commands."""
    pass


@openai.command("login")
@click.option(
    "--credential-path",
    default=str(DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH),
    show_default=True,
    help="Where to store OpenAI OAuth credentials.",
)
@click.option("--model", default=None, help="Optional model override for later API calls.")
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Max seconds to wait for the browser callback.",
)
@click.option(
    "--callback-host",
    default=DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    show_default=True,
    help="Host to bind the local OAuth callback server.",
)
@click.option(
    "--callback-port",
    type=int,
    default=DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    show_default=True,
    help="Port to bind the local OAuth callback server.",
)
@click.option(
    "--callback-path",
    default=DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    show_default=True,
    help="Callback path for the local OAuth server.",
)
@click.option(
    "--open-browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the authorization URL automatically.",
)
def openai_login(
    credential_path: str,
    model: str | None,
    timeout: float,
    callback_host: str,
    callback_port: int,
    callback_path: str,
    open_browser: bool,
):
    """Login with ChatGPT/OpenAI OAuth and save credentials locally."""
    _run_openai_oauth_login(
        credential_path=credential_path,
        model=model,
        timeout=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        open_browser=open_browser,
    )


@cli.group()
def anthropic():
    """Anthropic authentication commands."""
    pass


@anthropic.command("login")
@click.option(
    "--credential-path",
    default=str(ANTHROPIC_OAUTH_CREDENTIAL_PATH),
    show_default=True,
    help="Where to store the Anthropic setup-token.",
)
@click.option(
    "--token",
    default=None,
    help="Anthropic setup-token value. If omitted, you will be prompted.",
)
def anthropic_login(credential_path: str, token: str | None):
    """Save an Anthropic setup-token. This is the only supported Anthropic auth flow."""
    token_value = _prompt_anthropic_setup_token(token)
    save_anthropic_setup_token(credential_path, token_value)
    _print_oauth_login_success("Anthropic setup-token", credential_path)


@anthropic.command("setup-token")
@click.option(
    "--credential-path",
    default=str(ANTHROPIC_OAUTH_CREDENTIAL_PATH),
    show_default=True,
    help="Where to store the Anthropic setup-token.",
)
@click.option(
    "--token",
    default=None,
    help="Setup-token value. If omitted, you will be prompted.",
)
def anthropic_setup_token(credential_path: str, token: str | None):
    """Paste and save an Anthropic setup-token."""
    save_anthropic_setup_token(credential_path, _prompt_anthropic_setup_token(token))
    _print_oauth_login_success("Anthropic setup-token", credential_path)


@cli.group(name="gemini")
def gemini_group():
    """Gemini OAuth commands."""
    pass


@gemini_group.command("login")
@click.option(
    "--credential-path",
    default=str(GEMINI_OAUTH_CREDENTIAL_PATH),
    show_default=True,
    help="Where to store Gemini OAuth credentials.",
)
@click.option("--model", default=None, help="Optional model override for later API calls.")
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Max seconds to wait for the browser callback.",
)
@click.option(
    "--callback-host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind the local OAuth callback server.",
)
@click.option(
    "--callback-port",
    type=int,
    default=0,
    show_default=True,
    help="Port to bind the local OAuth callback server. Use 0 for auto.",
)
@click.option(
    "--callback-path",
    default="/oauth2callback",
    show_default=True,
    help="Callback path for the local OAuth server.",
)
@click.option(
    "--open-browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the authorization URL automatically.",
)
def gemini_login(
    credential_path: str,
    model: str | None,
    timeout: float,
    callback_host: str,
    callback_port: int,
    callback_path: str,
    open_browser: bool,
):
    """Login with Gemini Code Assist OAuth and save credentials locally."""
    _run_gemini_oauth_login(
        credential_path=credential_path,
        model=model,
        timeout=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        open_browser=open_browser,
    )


async def test(
    command: str,
    config_path: str | None = None,
    device: str | None = None,
    steps: int | None = None,
    vision: bool | None = None,
    reasoning: bool | None = None,
    tracing: bool | None = None,
    debug: bool | None = None,
    use_tcp: bool | None = None,
    save_trajectory: str | None = None,
    temperature: float | None = None,
    ios: bool = False,
):
    config = ConfigLoader.load(config_path)

    # Initialize logging
    debug_mode = debug if debug is not None else config.logging.debug
    _setup_cli_logging(debug_mode)
    logger = logging.getLogger("droidrun")

    try:
        logger.info(f"🚀 Starting: {command}")
        print_telemetry_message()

        # ================================================================
        # STEP 1: Apply CLI overrides via direct mutation
        # ================================================================

        # Vision overrides
        if vision is not None:
            # --vision flag overrides all agents
            config.agent.manager.vision = vision
            config.agent.executor.vision = vision
            config.agent.fast_agent.vision = vision
            logger.debug(f"CLI override: vision={vision} (all agents)")

        # Agent overrides
        if steps is not None:
            config.agent.max_steps = steps
        if reasoning is not None:
            config.agent.reasoning = reasoning

        # Device overrides
        if device is not None:
            config.device.serial = device
        if use_tcp is not None:
            config.device.use_tcp = use_tcp

        # Logging overrides
        if debug is not None:
            config.logging.debug = debug
        if save_trajectory is not None:
            config.logging.save_trajectory = save_trajectory

        # Tracing overrides
        if tracing is not None:
            config.tracing.enabled = tracing

        # Platform overrides
        if ios:
            config.device.platform = "ios"

        # ================================================================
        # STEP 2: Initialize DroidAgent with config
        # ================================================================

        mode = (
            "planning with reasoning" if config.agent.reasoning else "direct execution"
        )
        logger.info(f"🤖 Agent mode: {mode}")
        logger.info(
            f"👁️  Vision settings: Manager={config.agent.manager.vision}, "
            f"Executor={config.agent.executor.vision}, FastAgent={config.agent.fast_agent.vision}"
        )

        if config.tracing.enabled:
            logger.info("🔍 Tracing enabled")

        # Build DroidAgent kwargs for LLM loading
        droid_agent_kwargs = {}
        if temperature is not None:
            droid_agent_kwargs["temperature"] = temperature

        droid_agent = DroidAgent(
            goal=command,
            config=config,
            timeout=1000,
            **droid_agent_kwargs,
        )

        # ================================================================
        # STEP 3: Run agent
        # ================================================================

        logger.debug("▶️  Starting agent execution...")
        logger.debug("Press Ctrl+C to stop")

        event_handler = EventHandler()

        try:
            handler = droid_agent.run()

            async for event in handler.stream_events():
                event_handler.handle(event)
            result = await handler  # noqa: F841

        except KeyboardInterrupt:
            logger.info("⏹️ Stopped by user")

        except Exception as e:
            logger.error(f"💥 Error: {e}")
            if config.logging.debug:
                import traceback

                logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error(f"💥 Setup error: {e}")
        if debug_mode:
            import traceback

            logger.debug(traceback.format_exc())


if __name__ == "__main__":
    command = "open youtube and play a song by shakira"
    command = "use open_app to open the settings and search for the battery and enter the first result"
    device = None
    provider = "OpenAIResponses"
    model = "gpt-5.4-pro"
    temperature = 1
    api_key = os.getenv("OPENAI_API_KEY")
    steps = 15
    vision = True
    reasoning = False
    tracing = True
    debug = True
    use_tcp = False
    base_url = None
    api_base = None
    ios = False
    save_trajectory = "none"
    asyncio.run(run_command(command, device="emulator-5556", reasoning=False))
