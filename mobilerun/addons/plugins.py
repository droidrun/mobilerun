"""Discover optional plugins without importing their implementations."""

import logging
from importlib import metadata
from typing import Any

import click

logger = logging.getLogger(__name__)


class MissingPlugin(click.ClickException, ModuleNotFoundError):
    """An optional plugin is not installed."""


def discover(
    group: str, *, entries: list[metadata.EntryPoint] | None = None
) -> dict[str, metadata.EntryPoint]:
    """Return entry points in deterministic order; ignore ambiguous duplicates."""
    entries = sorted(
        metadata.entry_points(group=group) if entries is None else entries,
        key=lambda ep: (ep.name, str(ep.value)),
    )
    grouped: dict[str, list[metadata.EntryPoint]] = {}
    for entry in entries:
        grouped.setdefault(entry.name, []).append(entry)
    result = {}
    for name, candidates in grouped.items():
        first_party = [
            entry
            for entry in candidates
            if distribution_name(entry).lower() == "mobilerun"
        ]
        if len(candidates) == 1:
            result[name] = candidates[0]
        elif len(first_party) == 1:
            result[name] = first_party[0]
            logger.warning("Skipping duplicate plugin %s: first-party wins", name)
        else:
            logger.warning("Skipping duplicate plugin %s", name)
    return result


def distribution_name(entry: metadata.EntryPoint) -> str:
    """Return the owning distribution name, or unknown for synthetic entries."""
    dist = getattr(entry, "dist", None)
    return dist.metadata.get("Name", "unknown") if dist else "unknown"


def load(group: str, name: str) -> Any:
    """Load one plugin, reporting missing and broken installations safely."""
    entry = discover(group).get(name)
    if entry is None:
        message = f"Command '{name}' is not installed."
        if name == "agent" or (
            group == "mobilerun.operators" and name == "agent_macro"
        ):
            message += (
                " Install the mobilerun-agents add-on (see README: Agent add-on)."
            )
        raise MissingPlugin(message)
    try:
        return entry.load()
    except Exception as error:
        logger.debug("Plugin %s failed to load (%s)", name, type(error).__name__)
        distribution = distribution_name(entry)
        raise click.ClickException(
            f"Plugin '{name}' from '{distribution}' failed to load; "
            f"reinstall '{distribution}'."
        ) from None
