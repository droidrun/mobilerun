"""
Telemetry event models for Droidrun analytics.

This module defines Pydantic models for telemetry events captured during
agent execution. All events inherit from TelemetryEvent base class.
"""

from typing import Dict, Optional

from pydantic import BaseModel


class TelemetryEvent(BaseModel):
    """Base class for all telemetry events."""

    pass


class DroidAgentInitEvent(TelemetryEvent):
    """Event captured when DroidAgent is initialized."""

    goal: str
    llms: Dict[str, str]
    tools: str
    max_steps: int
    timeout: int
    vision: Dict[str, bool]
    reasoning: bool
    enable_tracing: bool
    debug: bool
    save_trajectories: str = "none"
    runtype: str = "developer"  # "cli" | "developer" | "web"
    custom_prompts: Optional[Dict[str, str]] = (
        None  # Keys: prompt names, Values: "custom" or None
    )


class PackageVisitEvent(TelemetryEvent):
    """Event captured when agent visits a new app package."""

    package_name: str
    activity_name: str
    step_number: int


class DroidAgentFinalizeEvent(TelemetryEvent):
    """Event captured when DroidAgent execution completes."""

    success: bool
    reason: str
    steps: int
    unique_packages_count: int
    unique_activities_count: int
