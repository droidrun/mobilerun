"""Shared, stream-aware rendering for CLI output."""

import os  # NO_COLOR is a terminal-wide convention, not a Mobilerun setting.
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.text import Text

stdout = Console()
stderr = Console(stderr=True)


def _print(console: Console, value: object, *, style: str | None = None) -> None:
    console.no_color = "NO_COLOR" in os.environ
    if console.no_color:
        value = value.plain if isinstance(value, Text) else value
        style = None
    console.print(value, style=style, highlight=False, markup=False)


def heading(count: int, noun: str) -> None:
    if noun.endswith("y"):
        plural = noun[:-1] + "ies"
    elif noun.endswith(("s", "x", "ch", "sh")):
        plural = noun + "es"
    else:
        plural = noun + "s"
    _print(stdout, f"{plural.capitalize()} · {count}", style="bold")
    _print(stdout, "")


def table(rows: Iterable[Iterable[object]]) -> None:
    rows = [list(row) for row in rows]
    if not rows:
        return
    grid = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    for _ in range(max(map(len, rows))):
        grid.add_column()
    for row in rows:
        grid.add_row(
            *(
                (
                    value
                    if isinstance(value, Text)
                    else str(value if value is not None else "")
                )
                for value in row
            )
        )
    _print(stdout, grid)


_STATUS_STYLES = {
    **dict.fromkeys(
        ("active", "ready", "running", "connected", "completed"), ("●", "green")
    ),
    **dict.fromkeys(
        (
            "pending",
            "awaiting_payment",
            "provisioning",
            "starting",
            "prepared",
            "queued",
            "created",
            "cancelling",
        ),
        ("●", "yellow"),
    ),
    **dict.fromkeys(
        ("inactive", "stopped", "released", "expired", "cancelled"), ("○", "dim")
    ),
    **dict.fromkeys(("error", "failed"), ("●", "red")),
}


def status(value: object) -> Text:
    label = str(value or "unknown")
    dot, style = _STATUS_STYLES.get(label.lower(), ("○", "dim"))
    return Text(f"{dot} {label}", style=style)


def secondary(value: object) -> Text:
    return Text(str(value or ""), style="dim")


def relative_time(value: object) -> Text:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return secondary(value)
        seconds = (datetime.now(timezone.utc) - parsed).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return secondary(value)
    delta = abs(seconds)
    if delta < 60:
        return secondary("just now")
    size, unit = next(
        (size, unit)
        for size, unit in ((86400, "day"), (3600, "hr"), (60, "min"))
        if delta >= size
    )
    count = int(delta // size)
    label = f"{count} {unit}{'s' if unit == 'day' and count != 1 else ''}"
    return secondary(f"{label} ago" if seconds >= 0 else f"in {label}")


def empty(noun: str, hint: str | None = None) -> None:
    message(f"No {noun} yet.")
    if hint:
        _print(stdout, hint, style="dim")


def bullets(rows: Iterable[object]) -> None:
    for row in rows:
        line = Text("  • ")
        line.append_text(row if isinstance(row, Text) else Text(str(row)))
        _print(stdout, line)


def kv(pairs: Iterable[tuple[object, object]]) -> None:
    for key, value in pairs:
        line = Text(f"{key}: ")
        line.append_text(value if isinstance(value, Text) else Text(str(value)))
        _print(stdout, line)


def message(value: object) -> None:
    _print(stdout, value)


def success(message: object) -> None:
    _print(stdout, f"✓ {message}", style="green")


def warn(message: object) -> None:
    _print(stdout, f"! {message}", style="yellow")


def error(message: object) -> None:
    _print(stderr, f"✗ {message}", style="red")


def spinner(message: str) -> AbstractContextManager[object]:
    if not stderr.is_terminal:
        return nullcontext()
    stderr.no_color = "NO_COLOR" in os.environ
    return stderr.status(message)
