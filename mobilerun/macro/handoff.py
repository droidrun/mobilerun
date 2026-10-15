"""Agent handoff for guarded macro replay divergence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


async def run_agent_handoff(
    *,
    goal: str,
    device_serial: Optional[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    divergence: Dict[str, Any],
    current_state: Dict[str, Any],
    remaining_actions: List[Dict[str, Any]],
    agent: Optional[str] = None,
    ios: bool = False,
) -> bool:
    """Invoke the normal Mobilerun agent after macro replay diverges.

    ``agent`` selects the agent backend (e.g. ``"jev"``); ``None`` keeps the
    configured one. With ``ios``, ``device_serial`` is the iOS portal URL.
    """
    from mobilerun.cli.main import run_command

    _ = divergence, current_state, remaining_actions
    return await run_command(
        goal,
        config_path=config_path,
        device=device_serial,
        agent=agent,
        provider=provider,
        model=model,
        save_trajectory="none",
        ios=ios,
    )
