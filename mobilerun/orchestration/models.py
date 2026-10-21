"""Shared orchestration data model (contract §0.1)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InvalidTransitionError(Exception):
    """Raised when a task status transition is not allowed."""


@dataclass
class TaskRequest:
    goal: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: datetime | None = None
    repeat_every: timedelta | None = None
    priority: int = 0
    device_serial: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    config_path: str | None = None
    timeout: int = 1000
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    success: bool
    reason: str
    steps: int
    structured_output: Any | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class TaskRecord:
    request: TaskRequest
    status: TaskStatus = TaskStatus.WAITING
    result: TaskResult | None = None
