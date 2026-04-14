"""Droidrun TUI - Main application."""

from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Static
from textual.worker import Worker, WorkerState
from textual import events

from droidrun.cli.tui.commands import match_commands, resolve_command
from droidrun.cli.event_handler import EventHandler
from droidrun.log_handlers import TUILogHandler, configure_logging
from droidrun.cli.tui.settings import SettingsData, SettingsScreen
from droidrun.cli.tui.widgets import (
    InputBar,
    CommandDropdown,
    DevicePicker,
    LogView,
    StatusBar,
)

# Map basic color names (from EventHandler) to TUI hex palette
COLOR_HEX = {
    "blue": "#b7bdf8",
    "cyan": "#CAD3F6",
    "green": "#a6da95",
    "red": "#ed8796",
    "yellow": "#f5a97f",
    "magenta": "#838BBC",
    "white": "#a1a1aa",
    "dim": "#71717a",
}
DEFAULT_LOG_STYLE = "#a1a1aa"


BANNER = """[#CAD3F6]
\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557   \u2588\u2588\u2557\u2588\u2588\u2588\u2557   \u2588\u2588\u2557
\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2551
\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2554\u2588\u2588\u2557 \u2588\u2588\u2551
\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2551\u255a\u2588\u2588\u2557\u2588\u2588\u2551
\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551  \u2588\u2588\u2551\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551  \u2588\u2588\u2551\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551 \u255a\u2588\u2588\u2588\u2588\u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u255d  \u255a\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u255d  \u255a\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u255d  \u255a\u2550\u2550\u2550\u255d
[/#CAD3F6]
[#838BBC]Type a command or [bold]/[/bold] for options[/#838BBC]"""


class DroidTUI(App):

    ALLOW_SELECT = False
    CSS_PATH = "css/app.tcss"

    BINDINGS = [
        Binding("ctrl+l", "clear_logs", "Clear Logs", show=False),
        Binding("ctrl+shift+c", "copy_logs", "Copy Logs", show=False),
        Binding("escape", "handle_esc", "Esc", show=False),
        Binding("ctrl+c", "handle_ctrl_c", "Quit", show=False),
        Binding("ctrl+z", "quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self._cancel_requested = False
        self._logs_visible = False
        self._dropdown_visible = False
        self._device_pick_visible = False
        self._esc_last: float = 0.0
        self._ctrl_c_last: float = 0.0

        self.reasoning: bool = False
        self.device_serial: str = ""
        self._device_connected: bool = False  # tracks portal health
        self._pending_input: str | None = (
            None  # e.g. "steps" — next submit goes to handler
        )

        # Load settings from user config
        try:
            from droidrun.config_manager import ConfigLoader

            config = ConfigLoader.load()
            self.settings = SettingsData.from_config(config)
            self._config_serial = config.device.serial or ""
            self.reasoning = config.agent.reasoning
        except Exception:
            self.settings = SettingsData()
            self._config_serial = ""
            self.reasoning = self.settings.reasoning

    def compose(self) -> ComposeResult:
        yield Static(BANNER, id="banner")

        with Container(id="log-container", classes="hidden"):
            yield LogView(id="log-display")

        with Vertical(id="bottom-area"):
            yield CommandDropdown(id="command-dropdown", classes="hidden")
            yield DevicePicker(id="device-picker", classes="hidden")
            yield InputBar(
                placeholder="Type a command or / for options",
                id="input-bar",
            )
            yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        self.query_one("#input-bar", InputBar).focus()
        self._sync_status_bar()
        self._update_hint()
        self.run_worker(self._autoconnect(), group="autoconnect", exclusive=True)
        self.set_interval(5, self._health_check_tick, name="health-check")

    def on_key(self, event: events.Key) -> None:
        if self._device_pick_visible or self._dropdown_visible:
            return
        input_bar = self.query_one("#input-bar", InputBar)
        if not input_bar.has_focus and event.is_printable and event.character:
            input_bar.focus()
            input_bar.value += event.character
            input_bar.cursor_position = len(input_bar.value)
            event.prevent_default()
            event.stop()

    # ── Status bar ──

    def _sync_status_bar(self) -> None:
        status = self.query_one("#status-bar", StatusBar)
        status.device_serial = self.device_serial
        status.device_connected = self._device_connected
        # Show model name from first profile in status bar
        first_profile = next(iter(self.settings.profiles.values()), None)
        if first_profile:
            model_display = first_profile.model
            if "/" in model_display:
                model_display = model_display.rsplit("/", 1)[-1]
        else:
            model_display = "no model"
        status.device_name = model_display
        status.mode = "reasoning" if self.reasoning else "fast"

    def _update_hint(self) -> None:
        status = self.query_one("#status-bar", StatusBar)
        if self.running:
            status.hint = "esc to stop"
        else:
            status.hint = ""

    # ── Autoconnect ──

    async def _autoconnect(self) -> None:
        """Silently find and verify a device on startup."""
        from async_adbutils import adb

        status = self.query_one("#status-bar", StatusBar)
        status.hint = "connecting..."

        try:
            devices = await adb.list()
        except Exception:
            status.hint = ""
            return

        # Filter to online devices
        available: list[str] = []
        for d in devices:
            if getattr(d, "state", None) == "device":
                available.append(d.serial)

        if not available:
            status.hint = ""
            return

        # Config serial gets priority, then the rest
        candidates = []
        if self._config_serial and self._config_serial in available:
            candidates.append(self._config_serial)
        for s in available:
            if s not in candidates:
                candidates.append(s)

        # Try each until portal responds
        for serial in candidates:
            try:
                await self._verify_portal(serial)
                self.device_serial = serial
                self._device_connected = True
                self._sync_status_bar()
                status.hint = ""
                return
            except Exception:
                continue

        # No portal found — use config serial if it's online (device exists but no portal)
        if self._config_serial and self._config_serial in available:
            self.device_serial = self._config_serial
            self._sync_status_bar()

        status.hint = ""

    # ── Health check ──

    def _health_check_tick(self) -> None:
        """Called every 5s by set_interval. Kicks off async health check."""
        if not self.device_serial:
            return
        self.run_worker(self._health_check(), group="health", exclusive=True)

    async def _health_check(self) -> None:
        """Ping the current device. Handle disconnect/reconnect."""
        import logging

        logger = logging.getLogger("droidrun")
        serial = self.device_serial
        if not serial:
            return

        try:
            await self._verify_portal(serial)

            if not self._device_connected:
                # Was disconnected, now back
                self._device_connected = True
                self._sync_status_bar()
                logger.info(f"Device reconnected: {serial}", extra={"color": "green"})

        except Exception:
            if self._device_connected:
                # Was connected, now lost
                self._device_connected = False
                self._sync_status_bar()
                logger.warning(
                    f"Device disconnected: {serial}", extra={"color": "yellow"}
                )

            # Try to reconnect
            try:
                await self._verify_portal(serial)
                self._device_connected = True
                self._sync_status_bar()
                logger.info(f"Device reconnected: {serial}", extra={"color": "green"})
            except Exception:
                pass

    # ── Worker crash recovery ──

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR:
            # Recover UI state so it's not stuck
            self._device_pick_visible = False
            self.query_one("#device-picker").add_class("hidden")
            self.query_one("#status-bar").remove_class("hidden")
            input_bar = self.query_one("#input-bar", InputBar)
            input_bar.disabled = False
            input_bar.focus()
            # Show the error
            self._show_logs()
            log = self.query_one("#log-display", LogView)
            log.append(f"  worker error: {event.worker.error}")

    # ── Debug ──

    def _dbg(self, msg: str) -> None:
        if not self.settings.debug:
            return
        ts = time.strftime("%H:%M:%S", time.localtime())
        ms = f"{time.time() % 1:.3f}"[1:]  # .XXX
        self._show_logs()
        log = self.query_one("#log-display", LogView)
        log.append(f"  [{ts}{ms}] {msg}")

    # ── Input messages ──

    def on_input_bar_submitted(self, message: InputBar.Submitted) -> None:
        text = message.value.strip()
        if not text:
            if self._pending_input:
                self._cancel_pending_input()
            return

        self._hide_dropdown()

        if self._pending_input:
            self._handle_pending_input(text)
        elif text.startswith("/"):
            self._handle_slash_command(text[1:])
        else:
            self.run_worker(self._execute_command(text), exclusive=True)

    def on_input_bar_slash_changed(self, message: InputBar.SlashChanged) -> None:
        commands = match_commands(message.query)
        if not commands:
            self._hide_dropdown()
            return
        dropdown = self.query_one("#command-dropdown", CommandDropdown)
        dropdown.update_commands(commands)
        self._show_dropdown()

    def on_input_bar_slash_exited(self, message: InputBar.SlashExited) -> None:
        self._hide_dropdown()

    def on_input_bar_slash_select(self, message: InputBar.SlashSelect) -> None:
        dropdown = self.query_one("#command-dropdown", CommandDropdown)
        if dropdown.has_commands:
            dropdown.select_highlighted()

    def on_input_bar_slash_navigate(self, message: InputBar.SlashNavigate) -> None:
        dropdown = self.query_one("#command-dropdown", CommandDropdown)
        dropdown.move_highlight(message.direction)

    def on_input_bar_tab_pressed(self, message: InputBar.TabPressed) -> None:
        if self._dropdown_visible:
            dropdown = self.query_one("#command-dropdown", CommandDropdown)
            if dropdown.has_commands:
                cmd = dropdown._commands[dropdown.highlighted]
                input_bar = self.query_one("#input-bar", InputBar)
                input_bar.value = f"/{cmd.name}"
                input_bar.cursor_position = len(input_bar.value)
        else:
            self.reasoning = not self.reasoning
            self._sync_status_bar()

    def on_command_dropdown_selected(self, message: CommandDropdown.Selected) -> None:
        input_bar = self.query_one("#input-bar", InputBar)
        input_bar.value = ""
        self._hide_dropdown()
        self._handle_slash_command(message.command_name)

    # ── Dropdown visibility ──

    def _show_dropdown(self) -> None:
        self.query_one("#command-dropdown").remove_class("hidden")
        self.query_one("#status-bar").add_class("hidden")
        self._dropdown_visible = True
        self.query_one("#input-bar", InputBar).slash_mode = True

    def _hide_dropdown(self) -> None:
        self.query_one("#command-dropdown").add_class("hidden")
        self.query_one("#status-bar").remove_class("hidden")
        self._dropdown_visible = False
        self.query_one("#input-bar", InputBar).slash_mode = False

    # ── Slash commands ──

    def _handle_slash_command(self, text: str) -> None:
        parts = text.strip().split()
        if not parts:
            return

        cmd = resolve_command(parts[0])

        if cmd is None:
            self._show_logs()
            log = self.query_one("#log-display", LogView)
            log.append(f"  unknown command: /{parts[0]}")
            return

        handler = getattr(self, cmd.handler, None)
        if handler:
            args = parts[1:]
            handler(*args) if args else handler()

    # ── Pending input ──

    def _start_pending_input(
        self, kind: str, prompt: str, placeholder: str = ""
    ) -> None:
        self._pending_input = kind
        picker = self.query_one("#device-picker", DevicePicker)
        picker.set_status(prompt)
        self.query_one("#device-picker").remove_class("hidden")
        self.query_one("#status-bar").add_class("hidden")
        input_bar = self.query_one("#input-bar", InputBar)
        input_bar.placeholder = placeholder
        input_bar.focus()

    def _cancel_pending_input(self) -> None:
        self._pending_input = None
        self.query_one("#device-picker").add_class("hidden")
        self.query_one("#status-bar").remove_class("hidden")
        input_bar = self.query_one("#input-bar", InputBar)
        input_bar.placeholder = "Type a command or / for options"
        input_bar.focus()

    def _handle_pending_input(self, text: str) -> None:
        kind = self._pending_input
        self._cancel_pending_input()

        if kind == "steps":
            self._apply_steps(text)

    # ── Command handlers ──

    def action_toggle_debug(self) -> None:
        self.settings.debug = not self.settings.debug
        self._show_logs()
        log = self.query_one("#log-display", LogView)
        state = "on" if self.settings.debug else "off"
        log.append(f"  debug logging {state}")

    def action_show_help(self) -> None:
        self._show_logs()
        log = self.query_one("#log-display", LogView)
        dim = "#838BBC"
        heading = "#CAD3F6"
        cmd_style = "#b7bdf8"

        log.append("")
        log.append("  ── commands ──", style=heading)
        log.append("")
        log.append("    /config       ", style=cmd_style)
        log.append("      open settings modal", style=dim)
        log.append("    /devices      ", style=cmd_style)
        log.append("      select connected device", style=dim)
        log.append("    /steps [n]    ", style=cmd_style)
        log.append("      set max agent steps (prompts if no number given)", style=dim)
        log.append("    /debug        ", style=cmd_style)
        log.append("      toggle debug logging on/off", style=dim)
        log.append("    /copy         ", style=cmd_style)
        log.append("      copy log output to clipboard", style=dim)
        log.append("    /clear        ", style=cmd_style)
        log.append("      clear log and show banner", style=dim)
        log.append("    /help         ", style=cmd_style)
        log.append("      show this help", style=dim)
        log.append("")
        log.append("  ── keys ──", style=heading)
        log.append("")
        log.append("    tab           ", style=cmd_style)
        log.append("      toggle reasoning / fast mode", style=dim)
        log.append("    up / down     ", style=cmd_style)
        log.append("      navigate command history", style=dim)
        log.append("    ctrl+l        ", style=cmd_style)
        log.append("      clear logs", style=dim)
        log.append("    ctrl+shift+c  ", style=cmd_style)
        log.append("      copy logs to clipboard", style=dim)
        log.append("    esc           ", style=cmd_style)
        log.append("      stop running agent", style=dim)
        log.append("    esc ×2        ", style=cmd_style)
        log.append("      clear input", style=dim)
        log.append("    ctrl+c ×2     ", style=cmd_style)
        log.append("      quit", style=dim)
        log.append("")
        log.append("  ── status bar ──", style=heading)
        log.append("")
        log.append("    green dot     ", style="#a6da95")
        log.append("      device connected, portal verified", style=dim)
        log.append("    yellow dot    ", style="#f5a97f")
        log.append("      device set but portal unreachable", style=dim)
        log.append("    red dot       ", style="#ed8796")
        log.append("      no device", style=dim)
        log.append("")

    def action_set_steps(self, *args: str) -> None:
        if args:
            self._apply_steps(args[0])
        else:
            self._start_pending_input(
                "steps",
                f"How many steps should the agent take? (current: {self.settings.max_steps})",
                placeholder="Enter number of steps",
            )

    def _apply_steps(self, value: str) -> None:
        try:
            n = int(value)
            if n < 1:
                raise ValueError
        except (ValueError, TypeError):
            self._show_logs()
            log = self.query_one("#log-display", LogView)
            log.append(f"  invalid steps: {value}")
            return

        self.settings.max_steps = n
        self._show_logs()
        log = self.query_one("#log-display", LogView)
        log.append(f"  max steps set to {n}")

    def action_open_config(self) -> None:
        modal = SettingsScreen(self.settings)
        self.push_screen(modal, callback=self._on_settings_dismissed)

    def _on_settings_dismissed(self, result: SettingsData | None) -> None:
        if result is None:
            return

        self.settings = result
        self.reasoning = result.reasoning
        self.settings.save()
        self._sync_status_bar()

        if self._logs_visible:
            log = self.query_one("#log-display", LogView)
            log.append("  settings updated")

    def action_open_device(self) -> None:
        self.run_worker(self._scan_devices(), exclusive=True)

    async def _scan_devices(self) -> None:
        from async_adbutils import adb

        picker = self.query_one("#device-picker", DevicePicker)
        picker.set_status("scanning...")
        self._show_device_picker()

        try:
            devices = await adb.list()
        except Exception:
            devices = []

        if not devices:
            picker.set_status("no devices found")
            return

        entries = []
        for d in devices:
            state = getattr(d, "state", "unknown")
            entries.append((d.serial, state))

        picker.set_devices(entries)
        picker.focus()

    def _show_device_picker(self) -> None:
        self.query_one("#device-picker").remove_class("hidden")
        self.query_one("#status-bar").add_class("hidden")
        self.query_one("#input-bar", InputBar).disabled = True
        self._device_pick_visible = True

    def _hide_device_picker(self) -> None:
        self.query_one("#device-picker").add_class("hidden")
        self.query_one("#status-bar").remove_class("hidden")
        self._device_pick_visible = False
        input_bar = self.query_one("#input-bar", InputBar)
        input_bar.disabled = False
        input_bar.focus()

    def on_device_picker_device_selected(
        self, message: DevicePicker.DeviceSelected
    ) -> None:
        self._dbg(f"device_selected serial={message.serial}")
        picker = self.query_one("#device-picker", DevicePicker)
        picker.set_status("checking portal...")
        self.run_worker(self._check_device(message.serial), exclusive=True)

    def on_device_picker_option_selected(
        self, message: DevicePicker.OptionSelected
    ) -> None:
        self._dbg(f"option_selected id={message.option_id} serial={message.serial}")
        picker = self.query_one("#device-picker", DevicePicker)
        if message.option_id == "setup":
            # Hide picker but keep input disabled during setup
            self.query_one("#device-picker").add_class("hidden")
            self.query_one("#status-bar").remove_class("hidden")
            self._device_pick_visible = False
            self._dbg("starting _run_device_setup worker")
            self.run_worker(self._run_device_setup(message.serial), exclusive=True)
        elif message.option_id == "back":
            picker.set_devices(picker._devices)
            picker.focus()

    def on_device_picker_cancelled(self, message: DevicePicker.Cancelled) -> None:
        self._hide_device_picker()

    async def _verify_portal(self, serial: str) -> None:
        """Check portal connectivity. Raises on failure."""
        from async_adbutils import adb
        from droidrun.tools.android.portal_client import PortalClient

        device_obj = await adb.device(serial)
        portal = PortalClient(device_obj, prefer_tcp=self.settings.use_tcp)
        await portal.connect()
        result = await portal.ping()

        status = result.get("status")
        self._dbg(
            f"verify_portal serial={serial} tcp={portal.tcp_available} status={status}"
        )

        if status != "success":
            raise Exception(result.get("message", "portal not responding"))

    async def _check_device(self, serial: str) -> None:
        """Check device from the picker flow — shows options on failure."""
        self._dbg(f"_check_device start serial={serial}")
        picker = self.query_one("#device-picker", DevicePicker)
        picker.set_status("checking portal...")

        try:
            await self._verify_portal(serial)

            # All good
            self.device_serial = serial
            self._device_connected = True
            self._hide_device_picker()
            self._sync_status_bar()
            self._show_logs()
            log = self.query_one("#log-display", LogView)
            log.append(f"  device ready: {serial}")
            self._dbg("_check_device ok")

        except Exception as e:
            self._dbg(f"_check_device failed: {e}")
            picker.set_options(
                serial,
                "Droidrun Portal is not set up on this device",
                [
                    ("setup", "Auto-install and set up Droidrun Portal"),
                    ("back", "Back to devices"),
                ],
            )

    async def _run_device_setup(self, serial: str) -> None:
        self._dbg(f"_run_device_setup ENTER serial={serial}")

        self._show_logs()
        log = self.query_one("#log-display", LogView)
        status = self.query_one("#status-bar", StatusBar)
        status.hint = "installing portal..."

        log.append(f"\n\u2500\u2500 setup {serial}")
        log.append("")

        apk_tmp = None
        try:
            import asyncio
            import os
            import tempfile

            import requests
            from async_adbutils import adb
            from droidrun.portal import (
                get_compatible_portal_version,
                enable_portal_accessibility,
                ASSET_NAME,
            )

            self._dbg("_run_device_setup imports done")

            # Resolve device
            device_obj = await adb.device(serial)
            self._dbg("_run_device_setup got device")

            # Determine APK version (blocking HTTP call)
            log.append("  checking compatible version...")
            from droidrun import __version__

            portal_version, download_base, mapping_fetched = await asyncio.to_thread(
                get_compatible_portal_version, __version__
            )
            self._dbg(
                f"_run_device_setup version={portal_version} fetched={mapping_fetched}"
            )

            # Build download URL
            if portal_version:
                log.append(f"  downloading portal {portal_version}...")
                url = f"{download_base}/{portal_version}/{ASSET_NAME}-{portal_version}.apk"
            else:
                if not mapping_fetched:
                    log.append("  version map unavailable, using latest")
                log.append("  downloading latest portal...")

                from droidrun.portal import get_latest_release_assets

                assets = await asyncio.to_thread(get_latest_release_assets)
                url = None
                for asset in assets:
                    if "browser_download_url" in asset and asset.get(
                        "name", ""
                    ).startswith(ASSET_NAME):
                        url = asset["browser_download_url"]
                        break
                    elif "downloadUrl" in asset and os.path.basename(
                        asset["downloadUrl"]
                    ).startswith(ASSET_NAME):
                        url = asset["downloadUrl"]
                        break
                if not url:
                    raise Exception("portal APK not found in latest release")

            # Download APK in thread
            def _download(download_url: str) -> str:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".apk")
                r = requests.get(download_url, stream=True)
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp.close()
                return tmp.name

            apk_tmp = await asyncio.to_thread(_download, url)
            log.append("  downloaded")
            self._dbg("_run_device_setup downloaded")

            # Install
            log.append("  installing APK...")
            await device_obj.install(apk_tmp, uninstall=True, flags=["-g"], silent=True)
            log.append("  APK installed")
            self._dbg("_run_device_setup installed")

            # Enable accessibility
            log.append("  enabling accessibility service...")
            await enable_portal_accessibility(device_obj)
            log.append("  accessibility enabled")
            self._dbg("_run_device_setup accessibility enabled, waiting for portal...")

            # Give the Portal service time to initialize
            log.append("  waiting for portal to start...")
            await asyncio.sleep(3)

            # Verify
            log.append("  verifying portal...")
            await self._verify_portal(serial)

            # Success — link device
            self.device_serial = serial
            self._device_connected = True
            self._sync_status_bar()
            log.append(f"  device ready: {serial}")
            log.append("")
            self._dbg("_run_device_setup SUCCESS")

        except Exception as e:
            self._dbg(f"_run_device_setup FAILED: {type(e).__name__}: {e}")
            import traceback

            for line in traceback.format_exc().splitlines():
                if line.strip():
                    self._dbg(f"  {line}")
            log.append(f"  setup failed: {e}")
            log.append("")

            # Re-show picker with retry options
            picker = self.query_one("#device-picker", DevicePicker)
            self._show_device_picker()
            picker.set_options(
                serial,
                f"setup failed: {e}",
                [
                    ("setup", "Retry setup"),
                    ("back", "Back to devices"),
                ],
            )

        else:
            # Re-enable input only on success
            input_bar = self.query_one("#input-bar", InputBar)
            input_bar.disabled = False
            input_bar.focus()

        finally:
            if apk_tmp:
                import os

                if os.path.exists(apk_tmp):
                    os.unlink(apk_tmp)
            status.hint = ""

    def action_clear_logs(self) -> None:
        log = self.query_one("#log-display", LogView)
        log.clear_log()
        self.query_one("#log-container").add_class("hidden")
        self.query_one("#banner").remove_class("hidden")
        self._logs_visible = False

    def action_copy_logs(self) -> None:
        log = self.query_one("#log-display", LogView)
        text = log.get_plain_text()
        if text:
            self.copy_to_clipboard(text)
            self.notify("Copied", timeout=1.5)

    # ── Esc handling ──

    def action_handle_esc(self) -> None:
        if self._pending_input:
            self._cancel_pending_input()
            return

        if self._device_pick_visible:
            self._hide_device_picker()
            return

        now = time.monotonic()
        double_esc = (now - self._esc_last) < 0.3
        self._esc_last = now

        if self.running and not double_esc:
            self._cancel_requested = True
            log = self.query_one("#log-display", LogView)
            log.append("  stopping...")
        elif double_esc:
            self.query_one("#input-bar", InputBar).clear_input()

    def action_handle_ctrl_c(self) -> None:
        now = time.monotonic()
        if (now - self._ctrl_c_last) < 1.5:
            self.exit()
        else:
            self._ctrl_c_last = now
            status = self.query_one("#status-bar", StatusBar)
            status.hint = "ctrl+c again to quit"
            self.set_timer(1.5, self._reset_ctrl_c_hint)

    def _reset_ctrl_c_hint(self) -> None:
        if (time.monotonic() - self._ctrl_c_last) >= 1.0:
            self._update_hint()

    # ── Log visibility ──

    def _show_logs(self) -> None:
        if not self._logs_visible:
            self.query_one("#log-container").remove_class("hidden")
            self.query_one("#banner").add_class("hidden")
            self._logs_visible = True

    # ── Agent execution ──

    async def _execute_command(self, command: str) -> None:
        if self.running:
            log = self.query_one("#log-display", LogView)
            log.append("  already running")
            return

        self.running = True
        self._cancel_requested = False
        input_bar = self.query_one("#input-bar", InputBar)
        input_bar.disabled = True
        self._update_hint()

        self._show_logs()

        log = self.query_one("#log-display", LogView)
        status = self.query_one("#status-bar", StatusBar)

        log.append(f"\n\u2500\u2500 {command}", style="#CAD3F6")
        log.append("")

        status.is_running = True

        # Set up TUI logging handler
        _stream_buf: list[str] = []

        def _append_indented(msg: str, style: str) -> None:
            for line in msg.split("\n"):
                log.append(f"  {line}", style=style)

        def _on_record(rec: dict) -> None:
            style = COLOR_HEX.get(rec.get("color")) or DEFAULT_LOG_STYLE

            if rec["stream"]:
                _stream_buf.append(rec["msg"])
            elif rec["stream_end"]:
                if _stream_buf:
                    _append_indented("".join(_stream_buf), style)
                    _stream_buf.clear()
            else:
                if _stream_buf:
                    _append_indented("".join(_stream_buf), style)
                    _stream_buf.clear()
                _append_indented(rec["msg"], style)

        tui_handler = TUILogHandler(on_record=_on_record)
        configure_logging(debug=self.settings.debug, handler=tui_handler)
        event_handler = EventHandler()
        success = False

        try:
            from droidrun import DroidAgent, ResultEvent
            from droidrun.config_manager import ConfigLoader

            config = ConfigLoader.load()

            # Apply settings to config
            self.settings.apply_to_config(config)

            config.agent.reasoning = self.reasoning
            config.device.serial = self.device_serial or None

            log.append("  initializing...", style="#838BBC")
            first_profile = next(iter(self.settings.profiles.values()), None)
            if first_profile:
                log.append(
                    f"  {first_profile.provider} \u2022 {first_profile.model}",
                    style="#838BBC",
                )

            # DroidAgent loads LLMs from config.llm_profiles via load_agent_llms
            droid_agent = DroidAgent(
                goal=command,
                config=config,
                timeout=1000,
                runtype="tui",
            )

            log.append("")

            handler = droid_agent.run()

            async for event in handler.stream_events():
                if self._cancel_requested:
                    log.append("\n  stopped by user")
                    break
                event_handler.handle(event)

            if not self._cancel_requested:
                result: ResultEvent = await handler
                success = result.success
                log.append(f"\n  {result.steps} steps", style="#838BBC")

        except Exception as e:
            log.append(f"\n  error: {e}", style="#ed8796")
            import traceback

            for line in traceback.format_exc().split("\n"):
                if line.strip():
                    log.append(f"  {line}", style="#ed8796")

        finally:
            if _stream_buf:
                log.append("  " + "".join(_stream_buf))
                _stream_buf.clear()

            if success:
                log.append("  done", style="#a6da95")
            else:
                log.append("  failed", style="#ed8796")
            log.append("")

            self.running = False
            self._cancel_requested = False
            status.is_running = False
            input_bar.disabled = False
            input_bar.focus()
            self._update_hint()
