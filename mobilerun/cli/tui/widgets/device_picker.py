"""Inline device picker widget — keyboard-driven, shown above input bar."""

from __future__ import annotations

from textual.widget import Widget
from textual.message import Message
from textual.app import RenderResult
from textual.reactive import reactive
from textual import events
from rich.text import Text


class DevicePicker(Widget):
    """Inline picker for devices and setup prompts.

    Modes:
        pick    — arrow keys navigate device list, enter selects
        options — arrow keys navigate action options, enter selects
        status  — non-interactive text (e.g. "scanning...")
    """

    can_focus = True

    class DeviceSelected(Message):
        def __init__(self, serial: str) -> None:
            super().__init__()
            self.serial = serial

    class OptionSelected(Message):
        def __init__(self, serial: str, option_id: str) -> None:
            super().__init__()
            self.serial = serial
            self.option_id = option_id

    class Cancelled(Message):
        pass

    highlighted: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._devices: list[tuple[str, str]] = []  # (serial, state)
        self._mode: str = "pick"  # pick | options | status
        self._status_text: str = ""
        self._options_serial: str = ""
        self._options_header: str = ""
        self._options: list[tuple[str, str]] = []  # (option_id, label)
        self._options_highlighted: int = 0

    # ── Public API ──

    def set_devices(self, devices: list[tuple[str, str]]) -> None:
        self._devices = devices
        self._mode = "pick"
        self.highlighted = 0
        self.refresh(layout=True)

    def set_status(self, text: str) -> None:
        self._mode = "status"
        self._status_text = text
        self.refresh(layout=True)

    def set_options(
        self,
        serial: str,
        header: str,
        options: list[tuple[str, str]],
    ) -> None:
        """Show a list of selectable options.

        Args:
            serial: Device serial this relates to.
            header: Text shown above the options (e.g. error message).
            options: List of (option_id, label) tuples.
        """
        self._mode = "options"
        self._options_serial = serial
        self._options_header = header
        self._options = options
        self._options_highlighted = 0
        self.refresh(layout=True)
        self.set_timer(0.05, self.focus)

    @property
    def has_devices(self) -> bool:
        return bool(self._devices)

    # ── Keyboard ──

    def on_key(self, event: events.Key) -> None:
        if self._mode == "pick":
            self._handle_pick_key(event)
        elif self._mode == "options":
            self._handle_options_key(event)
        elif self._mode == "status":
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self.post_message(self.Cancelled())

    def _handle_pick_key(self, event: events.Key) -> None:
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self.highlighted = max(0, self.highlighted - 1)
        elif event.key == "down":
            event.stop()
            event.prevent_default()
            self.highlighted = min(len(self._devices) - 1, self.highlighted + 1)
        elif event.key == "enter":
            event.stop()
            event.prevent_default()
            if self._devices:
                serial = self._devices[self.highlighted][0]
                self.post_message(self.DeviceSelected(serial))
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.Cancelled())

    def _handle_options_key(self, event: events.Key) -> None:
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self._options_highlighted = max(0, self._options_highlighted - 1)
            self.refresh(layout=True)
        elif event.key == "down":
            event.stop()
            event.prevent_default()
            self._options_highlighted = min(
                len(self._options) - 1, self._options_highlighted + 1
            )
            self.refresh(layout=True)
        elif event.key == "enter":
            event.stop()
            event.prevent_default()
            if self._options:
                option_id, _ = self._options[self._options_highlighted]
                self.post_message(self.OptionSelected(self._options_serial, option_id))
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            # Back to pick mode if we have devices, otherwise cancel
            if self._devices:
                self._mode = "pick"
                self.refresh(layout=True)
            else:
                self.post_message(self.Cancelled())

    # ── Render ──

    def render(self) -> RenderResult:
        if self._mode == "status":
            return self._render_status()
        if self._mode == "options":
            return self._render_options()
        return self._render_pick()

    def _render_status(self) -> Text:
        return Text(f"  {self._status_text}", style="#52525b")

    def _render_pick(self) -> Text:
        if not self._devices:
            return Text("  no devices found", style="#52525b")

        try:
            total_width = self.size.width - 4
        except Exception:
            total_width = 60

        out = Text()
        for i, (serial, state) in enumerate(self._devices):
            if i > 0:
                out.append("\n")

            hl = i == self.highlighted
            bg = " on #2e2e4a" if hl else ""
            serial_style = f"bold #CAD3F6{bg}"
            state_style = f"#838BBC{bg}" if not hl else f"#a6da95{bg}"

            label = f"  {serial}  ({state})"
            pad = max(0, total_width - len(label))
            out.append(f"  {serial}", style=serial_style)
            out.append(f"  ({state})", style=state_style)
            if hl:
                out.append(" " * pad, style=bg)

        return out

    def _render_options(self) -> Text:
        try:
            total_width = self.size.width - 4
        except Exception:
            total_width = 60

        out = Text()

        # Header (error / info text)
        if self._options_header:
            out.append(f"  {self._options_header}\n", style="#ed8796")
            out.append("\n")

        # Option list
        for i, (_, label) in enumerate(self._options):
            if i > 0:
                out.append("\n")

            hl = i == self._options_highlighted
            bg = " on #2e2e4a" if hl else ""
            label_style = f"bold #CAD3F6{bg}" if hl else f"#838BBC{bg}"

            display = f"  {label}"
            pad = max(0, total_width - len(display))
            out.append(display, style=label_style)
            if hl:
                out.append(" " * pad, style=bg)

        return out
