from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel, ConfigDict, Field

from droidrun.telemetry import PackageVisitEvent, capture


class QueuedUserMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    queued_at_step: int = 0


class DroidAgentState(BaseModel):
    """
    State model for DroidAgent workflow - shared across parent and child workflows.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Task context
    instruction: str = ""
    step_number: int = 0
    runtype: str = "developer"
    user_id: str | None = None

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
    last_executor_full_response: str = ""
    last_executor_action_count: int = 0  # Number of actions from last executor turn

    # ========================================================================
    # Memory
    # ========================================================================
    manager_memory: str = ""  # Manager's planning notes (append-only string)

    # ========================================================================
    # Completion State (set by complete() tool, checked by FastAgent/CodeAct)
    # ========================================================================
    finished: bool = False
    success: Optional[bool] = None

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
    # Script Execution Tracking
    # ========================================================================
    scripter_history: List[Dict] = Field(default_factory=list)
    last_scripter_message: str = ""
    last_scripter_success: bool = True

    # ========================================================================
    # Text Manipulation Tracking
    # ========================================================================
    has_text_to_modify: bool = False
    text_manipulation_history: List[Dict] = Field(default_factory=list)
    last_text_manipulation_success: bool = False

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
        """
        package_changed = package_name != self.current_package_name
        activity_changed = activity_name != self.current_activity_name

        if not (package_changed or activity_changed):
            return

        if package_changed and package_name:
            self.visited_packages.add(package_name)
        if activity_changed and activity_name:
            self.visited_activities.add(activity_name)

        self.current_package_name = package_name
        self.current_activity_name = activity_name

        capture(
            PackageVisitEvent(
                package_name=package_name or "Unknown",
                activity_name=activity_name or "Unknown",
                step_number=self.step_number,
            ),
            user_id=self.user_id,
        )
