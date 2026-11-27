"""Shared metadata, action outcomes, and screen evidence for recorded runs."""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
)

RunKind = Literal["recording", "macro_replay", "agent"]
RunStatus = Literal["running", "succeeded", "failed", "cancelled"]
Actor = Literal["human", "script", "macro", "agent"]


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


_UtcDatetime = Annotated[AwareDatetime, AfterValidator(_utc)]
_StepIndex = Annotated[int, Field(ge=1, strict=True)]


class _RunValue(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )


class DeviceInfo(_RunValue):
    """Identity of the device being recorded."""

    platform: str
    identifier: str


class Fingerprint(_RunValue):
    """Stable screen identity without bounds or visible text."""

    app: str | None
    activity: str | None
    elements: tuple[str, ...]


class RunRecord(_RunValue):
    """Metadata and lifecycle status for one recording directory."""

    run_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    kind: RunKind
    source_macro: str | None = None
    device: DeviceInfo
    started_at: _UtcDatetime
    finished_at: _UtcDatetime | None = None
    status: RunStatus = "running"
    failed_step: _StepIndex | None = None


class RunStep(_RunValue):
    """One dispatched action, including unsuccessful outcomes."""

    i: _StepIndex
    started_at: _UtcDatetime
    finished_at: _UtcDatetime
    action: str = Field(min_length=1)
    args: dict[str, JsonValue]
    ok: bool = Field(strict=True)
    error: str | None = None
    before: Fingerprint | None = None
    after: Fingerprint | None = None
    note: str | None = None
    target: str | None = None
    by: Actor
    screens: (
        dict[Literal["before", "after"], dict[Literal["json", "png"], str]] | None
    ) = None
