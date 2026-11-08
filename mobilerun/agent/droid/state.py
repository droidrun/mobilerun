from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel, ConfigDict, Field

from mobilerun.telemetry import PackageVisitEvent, capture


class QueuedUserMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    queued_at_step: int = 0


class MobileAgentState(BaseModel):
    """
    State model for MobileAgent workflow - shared across parent and child workflows.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Task context
    instruction: str = ""
    step_number: int = 0
    runtype: str = "developer"
    user_id: str | None = None
    telemetry_config_enabled: bool = Field(default=True, exclude=True)
    platform: str = "Android"

    # ========================================================================
    # Device State (current)
    # ========================================================================
    device_date: str = ""  # Fetched once at startup
    formatted_device_state: str = ""  # Text description for prompts
    focused_text: str = ""  # Text in focused input field
    a11y_tree: List[Dict] = Field(default_factory=list)  # Raw accessibility tree
    phone_state: Dict = Field(default_factory=dict)  # Package, activity, etc.
    screenshot: str | bytes | None = None  # Current screenshot
    width: int = 0
    height: int = 0

    # ========================================================================
    # Device State (previous - for before/after comparison)
    # ========================================================================
    previous_formatted_device_state: str = ""

    # ========================================================================
    # App Tracking
    # ========================================================================
    app_card: str = ""
    current_package_name: str = ""
    current_activity_name: str = ""
    visited_packages: set = Field(default_factory=set)
    visited_activities: set = Field(default_factory=set)

    # ========================================================================
    # Unified Thought/Plan Tracking (used by all agents)
    # ========================================================================
    last_thought: str = ""  # Most recent thought from any agent
    previous_plan: str = ""  # Plan from previous iteration
    progress_summary: str = ""  # Cumulative progress (replaces each turn)

    # ========================================================================
    # Planning State (Manager sets these)
    # ========================================================================
    plan: str = ""  # Current plan
    current_subgoal: str = ""  # Current subgoal for Executor
    answer: str = (
        ""  # Final answer (used by both manager completion and complete() tool)
    )

    # ========================================================================
    # Action Tracking
    # ========================================================================
    action_history: List[Dict] = Field(default_factory=list)
    summary_history: List[str] = Field(default_factory=list)
    action_outcomes: List[bool] = Field(default_factory=list)
    error_descriptions: List[str] = Field(default_factory=list)
    last_action: Dict = Field(default_factory=dict)
    last_summary: str = ""

    # ========================================================================
    # Memory (unified — used by both FastAgent and Manager via <add_memory> tags)
    # ========================================================================
    agent_memory: str = ""

    # ========================================================================
    # Completion State (set by complete() tool, checked by FastAgent)
    # ========================================================================
    finished: bool = False
    success: Optional[bool] = None
    # Sticky terminal failure (e.g. a denied approval). Unlike ``finished`` it
    # is never reset by queued messages and ends the run before any new turn.
    terminal_abort: Optional[str] = None

    # ========================================================================
    # Message History (for stateful agents - preserves ChatMessage blocks)
    # ========================================================================
    message_history: List[ChatMessage] = Field(default_factory=list)

    # ========================================================================
    # Error Handling
    # ========================================================================
    error_flag_plan: bool = False
    err_to_manager_thresh: int = 2

    # ========================================================================
    # External User Messages (mid-run injection queue)
    # ========================================================================
    pending_user_messages: List[QueuedUserMessage] = Field(default_factory=list)
    workflow_completed: bool = False

    # ========================================================================
    # Custom Variables (user-defined)
    # ========================================================================
    custom_variables: Dict = Field(default_factory=dict)
    output_dir: str = ""

    # ========================================================================
    # Methods for action functions
    # ========================================================================

    def append_memory(self, text: str) -> None:
        """Append text to agent_memory (shared by FastAgent and Manager)."""
        text = text.strip()
        if not text:
            return
        if self.agent_memory:
            self.agent_memory += "\n" + text
        else:
            self.agent_memory = text

    async def complete(
        self, success: bool, reason: str = "", message: str = ""
    ) -> None:
        """Mark task as finished.

        Accepts both ``reason`` and ``message`` params — FastAgent XML
        prompt uses ``message``, action signature uses ``reason``.
        """
        answer = reason or message
        if not success and not answer:
            raise ValueError("Reason for failure is required if success is False.")
        self.finished = True
        self.success = success
        self.answer = answer or "Task completed successfully."

    def abort_run(self, reason: str) -> None:
        """End the run as failed with ``reason``. Cannot be undone within a run."""
        if self.terminal_abort is None:
            self.terminal_abort = reason

    def queue_user_message(self, message: str) -> QueuedUserMessage:
        if not message or not message.strip():
            raise ValueError("Cannot queue an empty or whitespace-only message.")
        if self.workflow_completed:
            raise RuntimeError("Cannot queue messages: agent has already finished.")
        queued = QueuedUserMessage(message=message, queued_at_step=self.step_number)
        self.pending_user_messages.append(queued)
        return queued

    def drain_user_messages(self) -> list[QueuedUserMessage]:
        if not self.pending_user_messages:
            return []
        messages = list(self.pending_user_messages)
        self.pending_user_messages.clear()
        return messages

    def update_current_app(self, package_name: str, activity_name: str):
        """
        Update package and activity together, capturing telemetry event only once.
        Skips empty values — won't overwrite a known package/activity with "".
        """
        package_name = package_name.strip() if package_name else ""
        activity_name = activity_name.strip() if activity_name else ""

        # Don't overwrite known values with empty strings
        effective_package = package_name or self.current_package_name
        effective_activity = activity_name or self.current_activity_name

        package_changed = effective_package != self.current_package_name
        activity_changed = effective_activity != self.current_activity_name

        if not (package_changed or activity_changed):
            return

        if package_changed and effective_package:
            self.visited_packages.add(effective_package)
        if activity_changed and effective_activity:
            self.visited_activities.add(effective_activity)

        self.current_package_name = effective_package
        self.current_activity_name = effective_activity

        capture(
            PackageVisitEvent(
                package_name=effective_package or "Unknown",
                activity_name=effective_activity or "Unknown",
                step_number=self.step_number,
            ),
            user_id=self.user_id,
            config_enabled=self.telemetry_config_enabled,
        )


# Legacy alias — deprecated, will be removed in v0.8.0
DroidAgentState = MobileAgentState
