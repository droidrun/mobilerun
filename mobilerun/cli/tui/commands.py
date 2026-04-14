"""Slash command registry with alias support."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Command:
    """A slash command with optional hidden aliases."""

    name: str
    description: str
    handler: str  # Method name on the App
    aliases: list[str] = field(default_factory=list)


COMMANDS: list[Command] = [
    Command(
        "config", "Configure agent settings", "action_open_config", aliases=["settings"]
    ),
    Command("copy", "Copy log to clipboard", "action_copy_logs"),
    Command("debug", "Toggle debug logging", "action_toggle_debug"),
    Command("devices", "Select connected device", "action_open_device"),
    Command("help", "Show keybindings and commands", "action_show_help"),
    Command("steps", "Set max agent steps", "action_set_steps"),
    Command("clear", "Clear log output", "action_clear_logs"),
]


def match_commands(query: str) -> list[Command]:
    """
    Match query against canonical names and aliases.

    Returns deduplicated Commands sorted by canonical name.
    Typing "set" matches alias "settings" and returns the "config" Command.
    """
    query = query.lower().strip()
    if not query:
        return list(COMMANDS)

    seen: set[str] = set()
    results: list[Command] = []

    for cmd in COMMANDS:
        if cmd.name in seen:
            continue
        # Check canonical name
        if cmd.name.startswith(query):
            seen.add(cmd.name)
            results.append(cmd)
            continue
        # Check aliases
        for alias in cmd.aliases:
            if alias.startswith(query):
                seen.add(cmd.name)
                results.append(cmd)
                break

    return sorted(results, key=lambda c: c.name)


def resolve_command(text: str) -> Command | None:
    """Resolve exact command name or alias to a Command."""
    text = text.lower().strip()
    for cmd in COMMANDS:
        if cmd.name == text:
            return cmd
        if text in cmd.aliases:
            return cmd
    return None
