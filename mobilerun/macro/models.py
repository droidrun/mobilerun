"""Persisted replay steps derived from run evidence."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mobilerun.models import Fingerprint


class MacroStep(BaseModel):
    """One deterministic action with its original index and optional guards."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    i: int = Field(ge=1, strict=True)
    action: str = Field(min_length=1)
    args: dict[str, JsonValue] = Field(repr=False)
    before: Fingerprint | None = None
    after: Fingerprint | None = None
    note: str | None = Field(default=None, repr=False)
    target: str | None = Field(default=None, repr=False)
